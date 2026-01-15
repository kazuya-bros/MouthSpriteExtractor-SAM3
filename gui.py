#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui.py

SAM3を使用した口スプライト抽出GUI。

License: AGPL-3.0 (Ultralyticsライセンスに準拠)
Note: SAM3モデルの使用にはMeta SAM Licenseが適用されます。
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Tkinter imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Optional: drag & drop support
_HAS_TK_DND = False
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_TK_DND = True
except Exception:
    pass

# PIL for image display in Tkinter
try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    print("[warn] PIL not installed. Preview will be limited.")

# Core modules
from sam3_detector import is_sam3_available
from mouth_sprite_extractor import (
    MouthSpriteExtractor,
    MouthFrameInfo,
    get_unique_output_dir,
    quad_wh,
    warp_frame_to_norm,
    make_ellipse_mask,
    feather_mask,
    ensure_even_ge2,
    extract_mouth_sprite,
    adjust_quad,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE = "Mouth Sprite Extractor (SAM3)"
CANDIDATE_COUNT = 20
CANDIDATE_ROWS = 2
CANDIDATE_PER_ROW = (CANDIDATE_COUNT + CANDIDATE_ROWS - 1) // CANDIDATE_ROWS
THUMB_SIZE = 70
PREVIEW_SIZE = 100

DEFAULT_FEATHER = 15
MAX_FEATHER = 50
DEFAULT_PADDING = 0.3

# Fine-tuning defaults
DEFAULT_OFFSET_X = 0
DEFAULT_OFFSET_Y = 0
DEFAULT_SCALE = 1.0
MAX_OFFSET = 100
MIN_SCALE = 0.5
MAX_SCALE = 2.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def numpy_to_photoimage(
    bgra: np.ndarray,
    max_size: int,
    keep_aspect: bool = True,
) -> Optional["ImageTk.PhotoImage"]:
    """numpy配列をPhotoImageに変換

    Args:
        bgra: 入力画像（BGR/BGRA）
        max_size: 最大サイズ（幅または高さの大きい方）
        keep_aspect: アスペクト比を維持するか（デフォルト: True）

    Returns:
        PhotoImage または None
    """
    if not _HAS_PIL:
        return None
    try:
        if bgra.shape[2] == 4:
            img = Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA))
        else:
            img = Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGR2RGB))

        if keep_aspect:
            # アスペクト比を維持してリサイズ
            w, h = img.size
            if w >= h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            img = img.resize((max_size, max_size), Image.Resampling.LANCZOS)

        return ImageTk.PhotoImage(img)
    except Exception:
        return None


def composite_on_checkerboard(bgra: np.ndarray, checker_size: int = 8) -> np.ndarray:
    """チェッカーボード背景に合成"""
    h, w = bgra.shape[:2]
    checker = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(0, h, checker_size):
        for x in range(0, w, checker_size):
            if ((x // checker_size) + (y // checker_size)) % 2 == 0:
                checker[y:y+checker_size, x:x+checker_size] = [200, 200, 200]
            else:
                checker[y:y+checker_size, x:x+checker_size] = [255, 255, 255]

    alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
    rgb = bgra[:, :, :3].astype(np.float32)
    result = (rgb * alpha + checker.astype(np.float32) * (1 - alpha)).astype(np.uint8)

    return result


# ---------------------------------------------------------------------------
# Main GUI Application
# ---------------------------------------------------------------------------

class MouthSpriteExtractorApp(TkinterDnD.Tk if _HAS_TK_DND else tk.Tk):
    """口スプライト抽出GUIアプリケーション"""

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("900x700")
        self.minsize(800, 600)

        # State
        self.video_path: str = ""
        self.extractor: Optional[MouthSpriteExtractor] = None
        self.candidate_frames: List[MouthFrameInfo] = []
        self.candidate_images: List[Optional["ImageTk.PhotoImage"]] = []
        self.assignments: Dict[int, int] = {}
        self.preview_sprites: Dict[str, np.ndarray] = {}
        self.preview_images: Dict[str, "ImageTk.PhotoImage"] = {}
        self.unified_size: Optional[Tuple[int, int]] = None
        self.is_analyzing = False

        # Video capture cache
        self._cached_cap: Optional[cv2.VideoCapture] = None

        # Log queue
        self.log_queue: queue.Queue[str] = queue.Queue()

        # Build UI
        self._build_ui()

        # Check SAM3 availability
        self._check_sam3()

        # Start log polling
        self._poll_logs()

    def _check_sam3(self):
        """SAM3の利用可否をチェック"""
        available, error = is_sam3_available()
        if not available:
            self.log(f"警告: SAM3が利用できません: {error}")
            self.log("セットアップ手順:")
            self.log("  1. pip install sam3")
            self.log("  2. HuggingFaceでアクセス申請")
            self.log("  3. huggingface-cli login")
            self.analyze_btn.configure(state=tk.DISABLED)
        else:
            self.log("SAM3: 利用可能")

    def _build_ui(self):
        """UIを構築"""
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Video selection ---
        video_frame = ttk.LabelFrame(main_frame, text="動画ファイル", padding=5)
        video_frame.pack(fill=tk.X, pady=(0, 10))

        self.video_label = ttk.Label(video_frame, text="ファイルを選択してください")
        self.video_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.select_btn = ttk.Button(
            video_frame, text="選択...", command=self._on_select_video
        )
        self.select_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Drag & drop support
        if _HAS_TK_DND:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)

        # --- Settings ---
        settings_frame = ttk.LabelFrame(main_frame, text="設定", padding=5)
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        # Padding
        padding_frame = ttk.Frame(settings_frame)
        padding_frame.pack(fill=tk.X, pady=2)

        ttk.Label(padding_frame, text="パディング:").pack(side=tk.LEFT)
        self.padding_var = tk.DoubleVar(value=DEFAULT_PADDING)
        self.padding_slider = ttk.Scale(
            padding_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
            variable=self.padding_var,
        )
        self.padding_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.padding_label = ttk.Label(padding_frame, text="30%", width=5)
        self.padding_label.pack(side=tk.LEFT)

        # Feather
        feather_frame = ttk.Frame(settings_frame)
        feather_frame.pack(fill=tk.X, pady=2)

        ttk.Label(feather_frame, text="フェザー:").pack(side=tk.LEFT)
        self.feather_var = tk.IntVar(value=DEFAULT_FEATHER)
        self.feather_slider = ttk.Scale(
            feather_frame, from_=0, to=MAX_FEATHER, orient=tk.HORIZONTAL,
            variable=self.feather_var,
        )
        self.feather_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.feather_label = ttk.Label(feather_frame, text=f"{DEFAULT_FEATHER}px", width=6)
        self.feather_label.pack(side=tk.LEFT)

        # --- Fine-tuning frame ---
        tuning_frame = ttk.LabelFrame(main_frame, text="口位置の微調整", padding=5)
        tuning_frame.pack(fill=tk.X, pady=(0, 10))

        # Offset X
        offset_x_frame = ttk.Frame(tuning_frame)
        offset_x_frame.pack(fill=tk.X, pady=2)

        ttk.Label(offset_x_frame, text="オフセットX:").pack(side=tk.LEFT)
        self.offset_x_var = tk.IntVar(value=DEFAULT_OFFSET_X)
        self.offset_x_slider = ttk.Scale(
            offset_x_frame, from_=-MAX_OFFSET, to=MAX_OFFSET, orient=tk.HORIZONTAL,
            variable=self.offset_x_var,
        )
        self.offset_x_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.offset_x_label = ttk.Label(offset_x_frame, text="0px", width=6)
        self.offset_x_label.pack(side=tk.LEFT)

        # Offset Y
        offset_y_frame = ttk.Frame(tuning_frame)
        offset_y_frame.pack(fill=tk.X, pady=2)

        ttk.Label(offset_y_frame, text="オフセットY:").pack(side=tk.LEFT)
        self.offset_y_var = tk.IntVar(value=DEFAULT_OFFSET_Y)
        self.offset_y_slider = ttk.Scale(
            offset_y_frame, from_=-MAX_OFFSET, to=MAX_OFFSET, orient=tk.HORIZONTAL,
            variable=self.offset_y_var,
        )
        self.offset_y_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.offset_y_label = ttk.Label(offset_y_frame, text="0px", width=6)
        self.offset_y_label.pack(side=tk.LEFT)

        # Scale
        scale_frame = ttk.Frame(tuning_frame)
        scale_frame.pack(fill=tk.X, pady=2)

        ttk.Label(scale_frame, text="スケール:").pack(side=tk.LEFT)
        self.scale_var = tk.DoubleVar(value=DEFAULT_SCALE)
        self.scale_slider = ttk.Scale(
            scale_frame, from_=MIN_SCALE, to=MAX_SCALE, orient=tk.HORIZONTAL,
            variable=self.scale_var,
        )
        self.scale_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.scale_label = ttk.Label(scale_frame, text="100%", width=6)
        self.scale_label.pack(side=tk.LEFT)

        # Reset button
        reset_btn = ttk.Button(
            tuning_frame, text="リセット", command=self._reset_fine_tuning
        )
        reset_btn.pack(anchor=tk.E, pady=(5, 0))

        # --- Analyze button ---
        self.analyze_btn = ttk.Button(
            main_frame, text="解析開始", command=self._on_analyze, state=tk.DISABLED
        )
        self.analyze_btn.pack(fill=tk.X, pady=(0, 10))

        # --- Candidates area ---
        candidates_frame = ttk.LabelFrame(main_frame, text="候補フレーム（1-5を入力して割り当て）", padding=5)
        candidates_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Scrollable canvas
        self.candidates_canvas = tk.Canvas(candidates_frame, height=200)
        self.candidates_scroll = ttk.Scrollbar(
            candidates_frame, orient=tk.HORIZONTAL, command=self.candidates_canvas.xview
        )
        self.candidates_canvas.configure(xscrollcommand=self.candidates_scroll.set)

        self.candidates_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.candidates_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.candidates_inner = ttk.Frame(self.candidates_canvas)
        self.candidates_canvas.create_window((0, 0), window=self.candidates_inner, anchor=tk.NW)

        self.candidate_widgets: List[Dict] = []
        self._create_candidate_widgets()

        self.candidates_inner.bind("<Configure>", lambda e: self.candidates_canvas.configure(
            scrollregion=self.candidates_canvas.bbox("all")
        ))

        # --- Preview area ---
        preview_frame = ttk.LabelFrame(main_frame, text="出力プレビュー", padding=5)
        preview_frame.pack(fill=tk.X, pady=(0, 10))

        preview_inner = ttk.Frame(preview_frame)
        preview_inner.pack()

        mouth_names = ["open", "closed", "half", "e", "u"]
        self.preview_labels: Dict[str, ttk.Label] = {}
        self.out_frame_labels: Dict[str, ttk.Label] = {}

        for name in mouth_names:
            col = ttk.Frame(preview_inner)
            col.pack(side=tk.LEFT, padx=5)

            lbl = ttk.Label(col, text=f"[{name}]", width=PREVIEW_SIZE // 8)
            lbl.pack()

            img_lbl = ttk.Label(col, text="---", width=PREVIEW_SIZE // 8)
            img_lbl.pack()
            self.preview_labels[name] = img_lbl

            frame_lbl = ttk.Label(col, text="", font=("", 8))
            frame_lbl.pack()
            self.out_frame_labels[name] = frame_lbl

        # --- Buttons ---
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        self.update_btn = ttk.Button(
            btn_frame, text="プレビュー更新", command=self._on_update_preview, state=tk.DISABLED
        )
        self.update_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        self.output_btn = ttk.Button(
            btn_frame, text="PNG出力", command=self._on_output, state=tk.DISABLED
        )
        self.output_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

        # --- Log area ---
        log_frame = ttk.LabelFrame(main_frame, text="ログ", padding=5)
        log_frame.pack(fill=tk.X)

        self.log_text = tk.Text(log_frame, height=5, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.X)

    def _create_candidate_widgets(self):
        """候補ウィジェットを作成"""
        for row in range(CANDIDATE_ROWS):
            row_frame = ttk.Frame(self.candidates_inner)
            row_frame.pack(fill=tk.X, pady=2)

            for col in range(CANDIDATE_PER_ROW):
                idx = row * CANDIDATE_PER_ROW + col
                if idx >= CANDIDATE_COUNT:
                    break

                cell = ttk.Frame(row_frame)
                cell.pack(side=tk.LEFT, padx=2)

                img_lbl = ttk.Label(cell, text=f"[{idx}]", width=THUMB_SIZE // 8)
                img_lbl.pack()

                entry_var = tk.StringVar()
                entry = ttk.Entry(cell, width=3, textvariable=entry_var)
                entry.pack()

                self.candidate_widgets.append({
                    "frame": cell,
                    "label": img_lbl,
                    "entry": entry,
                    "var": entry_var,
                })

    def _poll_logs(self):
        """ログキューをポーリング"""
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                self._append_log(msg)
            except queue.Empty:
                break

        # Update labels
        padding_pct = int(self.padding_var.get() * 100)
        self.padding_label.configure(text=f"{padding_pct}%")
        self.feather_label.configure(text=f"{self.feather_var.get()}px")

        # Update fine-tuning labels
        self.offset_x_label.configure(text=f"{self.offset_x_var.get()}px")
        self.offset_y_label.configure(text=f"{self.offset_y_var.get()}px")
        scale_pct = int(self.scale_var.get() * 100)
        self.scale_label.configure(text=f"{scale_pct}%")

        self.after(100, self._poll_logs)

    def _append_log(self, msg: str):
        """ログを追加"""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def log(self, msg: str):
        """スレッドセーフなログ"""
        self.log_queue.put(msg)

    def _reset_fine_tuning(self):
        """微調整をリセット"""
        self.offset_x_var.set(DEFAULT_OFFSET_X)
        self.offset_y_var.set(DEFAULT_OFFSET_Y)
        self.scale_var.set(DEFAULT_SCALE)
        self.log("微調整をリセットしました")

    def _on_select_video(self):
        """動画ファイルを選択"""
        path = filedialog.askopenfilename(
            title="動画ファイルを選択",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self._set_video(path)

    def _on_drop(self, event):
        """ドラッグ&ドロップ"""
        path = event.data.strip()
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]
        if os.path.isfile(path):
            self._set_video(path)

    def _set_video(self, path: str):
        """動画を設定"""
        self.video_path = path
        self.video_label.configure(text=os.path.basename(path))
        self.analyze_btn.configure(state=tk.NORMAL)
        self.log(f"動画を選択: {path}")

    def _on_analyze(self):
        """解析開始"""
        if not self.video_path or self.is_analyzing:
            return

        self.is_analyzing = True
        self.analyze_btn.configure(state=tk.DISABLED)

        thread = threading.Thread(target=self._analyze_worker, daemon=True)
        thread.start()

    def _analyze_worker(self):
        """解析ワーカースレッド"""
        try:
            self.log("解析を開始...")

            padding_ratio = self.padding_var.get()

            self.extractor = MouthSpriteExtractor(
                self.video_path,
                padding_ratio=padding_ratio,
            )
            self.extractor.analyze(callback=self.log)

            # 候補フレームを選択
            valid_frames = [mf for mf in self.extractor.mouth_frames if mf.valid]

            if len(valid_frames) == 0:
                self.log("エラー: 有効なフレームがありません")
                return

            # バリエーションのある候補を選択
            heights = np.array([mf.height for mf in valid_frames])
            widths = np.array([mf.width for mf in valid_frames])
            aspect_ratios = widths / np.maximum(heights, 1e-6)

            selected_indices = set()
            candidates = []

            def pick_by_score(scores, count, maximize=True, label=""):
                sorted_idx = np.argsort(scores)
                if maximize:
                    sorted_idx = sorted_idx[::-1]

                picked = 0
                for idx in sorted_idx:
                    if idx not in selected_indices and picked < count:
                        selected_indices.add(idx)
                        candidates.append((valid_frames[idx], label))
                        picked += 1
                    if picked >= count:
                        break

            pick_by_score(heights, 4, maximize=True, label="open")
            pick_by_score(heights, 4, maximize=False, label="closed")

            median_h = np.median(heights)
            half_scores = -np.abs(heights - median_h)
            pick_by_score(half_scores, 4, label="half")

            pick_by_score(aspect_ratios, 4, maximize=True, label="e")
            pick_by_score(-widths, 4, maximize=True, label="u")

            self.candidate_frames = [c[0] for c in candidates[:CANDIDATE_COUNT]]

            # 統一サイズを計算
            max_w = max(mf.width for mf in self.candidate_frames)
            max_h = max(mf.height for mf in self.candidate_frames)
            self.unified_size = (
                ensure_even_ge2(int(max_w * 1.2)),
                ensure_even_ge2(int(max_h * 1.2))
            )

            self.log(f"候補フレーム: {len(self.candidate_frames)}件")

            # サムネイル生成
            self.after(0, self._generate_thumbnails)
            self.after(0, self._update_candidates_ui)

            self.log("解析完了")

        except Exception as e:
            self.log(f"エラー: {e}")
            traceback.print_exc()
        finally:
            self.is_analyzing = False
            self.after(0, lambda: self.analyze_btn.configure(state=tk.NORMAL))
            self.after(0, lambda: self.update_btn.configure(state=tk.NORMAL))

    def _get_video_capture(self) -> cv2.VideoCapture:
        """キャッシュされたVideoCaptureを取得"""
        if self._cached_cap is None or not self._cached_cap.isOpened():
            self._cached_cap = cv2.VideoCapture(self.video_path)
        return self._cached_cap

    def _generate_thumbnails(self):
        """サムネイル生成"""
        if not self.candidate_frames or not self.unified_size:
            return

        cap = self._get_video_capture()
        unified_w, unified_h = self.unified_size

        self.candidate_images = []
        for mf in self.candidate_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                self.candidate_images.append(None)
                continue

            patch = warp_frame_to_norm(frame, mf.quad, unified_w, unified_h)
            photo = numpy_to_photoimage(patch, THUMB_SIZE)
            self.candidate_images.append(photo)

    def _update_candidates_ui(self):
        """候補UIを更新"""
        for i, mf in enumerate(self.candidate_frames):
            if i >= len(self.candidate_widgets):
                break

            widget = self.candidate_widgets[i]

            if i < len(self.candidate_images) and self.candidate_images[i]:
                widget["label"].configure(image=self.candidate_images[i], text="")
            else:
                widget["label"].configure(image="", text=f"F:{mf.frame_idx}")

    def _on_update_preview(self):
        """プレビュー更新"""
        if not self.candidate_frames or not self.unified_size:
            messagebox.showwarning("警告", "先に解析を実行してください")
            return

        # 割り当てを取得
        assignments = {}
        mouth_names = {1: "open", 2: "closed", 3: "half", 4: "e", 5: "u"}

        for i, widget in enumerate(self.candidate_widgets):
            if i >= len(self.candidate_frames):
                break

            val = widget["var"].get().strip()
            if val and val.isdigit():
                num = int(val)
                if 1 <= num <= 5:
                    assignments[num] = i

        if len(assignments) < 5:
            messagebox.showwarning("警告", "1-5の数字を候補に入力してください")
            return

        self.assignments = assignments
        feather_px = self.feather_var.get()
        unified_w, unified_h = self.unified_size
        cap = self._get_video_capture()

        # 微調整パラメータを取得
        offset_x = self.offset_x_var.get()
        offset_y = self.offset_y_var.get()
        scale = self.scale_var.get()

        self.preview_sprites = {}

        for num, cand_idx in assignments.items():
            name = mouth_names[num]
            mf = self.candidate_frames[cand_idx]

            cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # 微調整を適用したquadを使用
            adjusted_quad = adjust_quad(mf.quad, offset_x, offset_y, scale)

            bgra = extract_mouth_sprite(
                frame, adjusted_quad, unified_w, unified_h,
                feather_px=feather_px
            )

            self.preview_sprites[name] = bgra

            composited = composite_on_checkerboard(bgra)
            photo = numpy_to_photoimage(composited, PREVIEW_SIZE)
            if photo:
                self.preview_images[name] = photo
                self.preview_labels[name].configure(image=photo, text="")
                self.out_frame_labels[name].configure(text=f"F:{mf.frame_idx}")

        self.output_btn.configure(state=tk.NORMAL)
        self.log(f"プレビュー更新完了 ({len(self.preview_sprites)}枚)")

    def _on_output(self):
        """PNG出力"""
        if not self.preview_sprites:
            messagebox.showwarning("警告", "先にプレビューを更新してください")
            return

        video_dir = os.path.dirname(self.video_path)
        output_dir = get_unique_output_dir(os.path.join(video_dir, "mouth"))

        os.makedirs(output_dir, exist_ok=True)

        mouth_names = {
            "open": "mouth_open",
            "closed": "mouth_closed",
            "half": "mouth_half",
            "e": "mouth_e",
            "u": "mouth_u",
        }

        for name, bgra in self.preview_sprites.items():
            out_path = os.path.join(output_dir, f"{mouth_names[name]}.png")
            cv2.imwrite(out_path, bgra)
            self.log(f"出力: {out_path}")

        self.log(f"完了: {output_dir}")
        messagebox.showinfo("完了", f"出力先: {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    app = MouthSpriteExtractorApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
