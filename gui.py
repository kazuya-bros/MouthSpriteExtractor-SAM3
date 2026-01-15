#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui.py

SAM3を使用した口スプライト抽出GUI（3ステップ版）。

STEP1: 動画から口を抽出・自動分類
STEP2: 5種類の口を選択
STEP3: マスク調整・出力

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
    classify_mouth_frames,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE = "Mouth Sprite Extractor (SAM3)"
CANDIDATES_PER_CATEGORY = 5
THUMB_SIZE = 80
PREVIEW_SIZE = 120

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

# Mask settings
DEFAULT_MASK_DILATE = 0
MAX_MASK_DILATE = 30

MOUTH_CATEGORIES = ["open", "closed", "half", "e", "u"]
CATEGORY_LABELS = {
    "open": "1: Open (大きく開いた口)",
    "closed": "2: Closed (閉じた口)",
    "half": "3: Half (半開き)",
    "e": "4: E (横長「え」)",
    "u": "5: U (すぼめ「う」)",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def numpy_to_photoimage(
    bgra: np.ndarray,
    max_size: int,
    keep_aspect: bool = True,
) -> Optional["ImageTk.PhotoImage"]:
    """numpy配列をPhotoImageに変換"""
    if not _HAS_PIL:
        return None
    try:
        if len(bgra.shape) == 2:
            # グレースケール
            img = Image.fromarray(bgra)
        elif bgra.shape[2] == 4:
            img = Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA))
        else:
            img = Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGR2RGB))

        if keep_aspect:
            w, h = img.size
            if w >= h:
                new_w = max_size
                new_h = max(1, int(h * max_size / w))
            else:
                new_h = max_size
                new_w = max(1, int(w * max_size / h))
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
    """口スプライト抽出GUIアプリケーション（3ステップ版）"""

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1000x800")
        self.minsize(900, 700)

        # State
        self.video_path: str = ""
        self.extractor: Optional[MouthSpriteExtractor] = None
        self.classified_frames: Dict[str, List[MouthFrameInfo]] = {}
        self.selected_frames: Dict[str, Optional[MouthFrameInfo]] = {
            cat: None for cat in MOUTH_CATEGORIES
        }
        self.preview_sprites: Dict[str, np.ndarray] = {}
        self.preview_images: Dict[str, "ImageTk.PhotoImage"] = {}
        self.unified_size: Optional[Tuple[int, int]] = None
        self.is_analyzing = False
        self.current_step = 1

        # Video capture cache
        self._cached_cap: Optional[cv2.VideoCapture] = None

        # Log queue
        self.log_queue: queue.Queue[str] = queue.Queue()

        # Image references (prevent GC)
        self._thumb_images: List["ImageTk.PhotoImage"] = []

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
            self.step1_btn.configure(state=tk.DISABLED)
        else:
            self.log("SAM3: 利用可能")

    def _build_ui(self):
        """UIを構築"""
        # Main notebook for steps
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create step frames
        self.step1_frame = ttk.Frame(self.notebook)
        self.step2_frame = ttk.Frame(self.notebook)
        self.step3_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.step1_frame, text="STEP1: 抽出")
        self.notebook.add(self.step2_frame, text="STEP2: 選択")
        self.notebook.add(self.step3_frame, text="STEP3: 出力")

        self._build_step1_ui()
        self._build_step2_ui()
        self._build_step3_ui()

        # Log area (common)
        log_frame = ttk.LabelFrame(self, text="ログ", padding=5)
        log_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.log_text = tk.Text(log_frame, height=4, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.X)

    def _build_step1_ui(self):
        """STEP1: 抽出UIを構築"""
        frame = self.step1_frame

        # Video selection
        video_frame = ttk.LabelFrame(frame, text="動画ファイル", padding=10)
        video_frame.pack(fill=tk.X, pady=(0, 10))

        self.video_label = ttk.Label(video_frame, text="ファイルを選択してください（またはドラッグ&ドロップ）")
        self.video_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.select_btn = ttk.Button(
            video_frame, text="選択...", command=self._on_select_video
        )
        self.select_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Drag & drop support
        if _HAS_TK_DND:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)

        # Settings
        settings_frame = ttk.LabelFrame(frame, text="検出設定", padding=10)
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

        # Analyze button
        self.step1_btn = ttk.Button(
            frame, text="解析開始（SAM3で口を検出）", command=self._on_analyze, state=tk.DISABLED
        )
        self.step1_btn.pack(fill=tk.X, pady=10)

        # Progress
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(fill=tk.X)

        # Info
        info_text = """
【STEP1: 抽出】
1. 動画ファイルを選択
2. 「解析開始」をクリック
3. SAM3が自動的に口を検出し、5種類に分類します

検出完了後、自動的にSTEP2に進みます。
        """
        info_label = ttk.Label(frame, text=info_text, justify=tk.LEFT)
        info_label.pack(fill=tk.X, pady=10)

    def _build_step2_ui(self):
        """STEP2: 選択UIを構築"""
        frame = self.step2_frame

        # Instructions
        ttk.Label(
            frame,
            text="各カテゴリから最適な口を1つ選択してください（クリックで選択）",
            font=("", 10, "bold")
        ).pack(fill=tk.X, pady=(0, 10))

        # Scrollable area for categories
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        self.step2_inner = ttk.Frame(canvas)

        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.create_window((0, 0), window=self.step2_inner, anchor=tk.NW)

        self.step2_inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        ))

        # Category frames (will be populated after analysis)
        self.category_frames: Dict[str, ttk.LabelFrame] = {}
        self.candidate_buttons: Dict[str, List[ttk.Button]] = {}
        self.selection_labels: Dict[str, ttk.Label] = {}

        for cat in MOUTH_CATEGORIES:
            cat_frame = ttk.LabelFrame(
                self.step2_inner, text=CATEGORY_LABELS[cat], padding=5
            )
            cat_frame.pack(fill=tk.X, pady=5, padx=5)
            self.category_frames[cat] = cat_frame

            # Candidates row
            cand_frame = ttk.Frame(cat_frame)
            cand_frame.pack(fill=tk.X)

            self.candidate_buttons[cat] = []
            for i in range(CANDIDATES_PER_CATEGORY):
                btn = ttk.Button(cand_frame, text=f"[{i+1}]", width=10)
                btn.pack(side=tk.LEFT, padx=2, pady=2)
                self.candidate_buttons[cat].append(btn)

            # Selection indicator
            sel_label = ttk.Label(cat_frame, text="未選択", foreground="gray")
            sel_label.pack(anchor=tk.W, pady=(5, 0))
            self.selection_labels[cat] = sel_label

        # Next button
        self.step2_btn = ttk.Button(
            frame, text="STEP3へ進む", command=self._goto_step3, state=tk.DISABLED
        )
        self.step2_btn.pack(fill=tk.X, pady=10, side=tk.BOTTOM)

    def _build_step3_ui(self):
        """STEP3: 出力UIを構築"""
        frame = self.step3_frame

        # Settings frame
        settings_frame = ttk.LabelFrame(frame, text="マスク設定", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        # Mask type
        mask_type_frame = ttk.Frame(settings_frame)
        mask_type_frame.pack(fill=tk.X, pady=2)

        ttk.Label(mask_type_frame, text="マスク種類:").pack(side=tk.LEFT)
        self.mask_type_var = tk.StringVar(value="ellipse")
        ttk.Radiobutton(
            mask_type_frame, text="楕円マスク", variable=self.mask_type_var,
            value="ellipse", command=self._on_mask_type_changed
        ).pack(side=tk.LEFT, padx=(10, 5))
        ttk.Radiobutton(
            mask_type_frame, text="SAM3マスク", variable=self.mask_type_var,
            value="sam3", command=self._on_mask_type_changed
        ).pack(side=tk.LEFT, padx=5)

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

        # Mask dilate (for SAM3 mask)
        dilate_frame = ttk.Frame(settings_frame)
        dilate_frame.pack(fill=tk.X, pady=2)

        ttk.Label(dilate_frame, text="マスク膨張:").pack(side=tk.LEFT)
        self.dilate_var = tk.IntVar(value=DEFAULT_MASK_DILATE)
        self.dilate_slider = ttk.Scale(
            dilate_frame, from_=-MAX_MASK_DILATE, to=MAX_MASK_DILATE, orient=tk.HORIZONTAL,
            variable=self.dilate_var,
        )
        self.dilate_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.dilate_label = ttk.Label(dilate_frame, text="0px", width=6)
        self.dilate_label.pack(side=tk.LEFT)

        # Fine-tuning
        tuning_frame = ttk.LabelFrame(frame, text="位置の微調整", padding=10)
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

        # Preview button
        preview_btn_frame = ttk.Frame(frame)
        preview_btn_frame.pack(fill=tk.X, pady=(0, 10))

        self.preview_btn = ttk.Button(
            preview_btn_frame, text="プレビュー更新", command=self._on_update_preview
        )
        self.preview_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        reset_btn = ttk.Button(
            preview_btn_frame, text="リセット", command=self._reset_settings
        )
        reset_btn.pack(side=tk.LEFT, padx=(5, 0))

        # Preview area
        preview_frame = ttk.LabelFrame(frame, text="出力プレビュー", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        preview_inner = ttk.Frame(preview_frame)
        preview_inner.pack()

        self.preview_labels: Dict[str, ttk.Label] = {}
        self.preview_name_labels: Dict[str, ttk.Label] = {}

        for cat in MOUTH_CATEGORIES:
            col = ttk.Frame(preview_inner)
            col.pack(side=tk.LEFT, padx=10)

            name_lbl = ttk.Label(col, text=f"[{cat}]", font=("", 9, "bold"))
            name_lbl.pack()
            self.preview_name_labels[cat] = name_lbl

            img_lbl = ttk.Label(col, text="---")
            img_lbl.pack(pady=5)
            self.preview_labels[cat] = img_lbl

        # Output button
        self.output_btn = ttk.Button(
            frame, text="PNG出力", command=self._on_output, state=tk.DISABLED
        )
        self.output_btn.pack(fill=tk.X, pady=10)

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
        self.dilate_label.configure(text=f"{self.dilate_var.get()}px")
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
        self.step1_btn.configure(state=tk.NORMAL)
        self.log(f"動画を選択: {path}")

    def _on_analyze(self):
        """解析開始"""
        if not self.video_path or self.is_analyzing:
            return

        self.is_analyzing = True
        self.step1_btn.configure(state=tk.DISABLED)
        self.progress_var.set(0)

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

            def progress_callback(msg: str):
                self.log(msg)
                # プログレスバーを更新
                if "処理中" in msg:
                    try:
                        parts = msg.split("/")
                        if len(parts) >= 2:
                            current = int(parts[0].split()[-1])
                            total = int(parts[1].split()[0])
                            pct = (current / total) * 100
                            self.progress_var.set(pct)
                    except Exception:
                        pass

            self.extractor.analyze(callback=progress_callback)

            if len(self.extractor.mouth_frames) == 0:
                self.log("エラー: 口が検出されませんでした")
                return

            # 5カテゴリに分類
            self.classified_frames = classify_mouth_frames(
                self.extractor.mouth_frames,
                self.extractor.cluster_mask,
                candidates_per_category=CANDIDATES_PER_CATEGORY,
            )

            self.log(f"分類完了: 各カテゴリ{CANDIDATES_PER_CATEGORY}件ずつ")

            # 統一サイズを計算
            all_frames = []
            for frames in self.classified_frames.values():
                all_frames.extend(frames)

            if all_frames:
                max_w = max(mf.width for mf in all_frames)
                max_h = max(mf.height for mf in all_frames)
                self.unified_size = (
                    ensure_even_ge2(int(max_w * 1.2)),
                    ensure_even_ge2(int(max_h * 1.2))
                )
                self.log(f"統一サイズ: {self.unified_size}")

            # UIを更新
            self.after(0, self._populate_step2)
            self.after(0, lambda: self.notebook.select(1))  # STEP2へ

            self.progress_var.set(100)
            self.log("解析完了 - STEP2で口を選択してください")

        except Exception as e:
            self.log(f"エラー: {e}")
            traceback.print_exc()
        finally:
            self.is_analyzing = False
            self.after(0, lambda: self.step1_btn.configure(state=tk.NORMAL))

    def _populate_step2(self):
        """STEP2のUIを候補で埋める"""
        if not self.classified_frames or not self.unified_size:
            return

        cap = self._get_video_capture()
        unified_w, unified_h = self.unified_size

        self._thumb_images = []  # Reset image references

        for cat in MOUTH_CATEGORIES:
            frames = self.classified_frames.get(cat, [])
            buttons = self.candidate_buttons[cat]

            for i, btn in enumerate(buttons):
                if i < len(frames):
                    mf = frames[i]

                    # サムネイル生成
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        patch = warp_frame_to_norm(frame, mf.quad, unified_w, unified_h)
                        photo = numpy_to_photoimage(patch, THUMB_SIZE)
                        if photo:
                            self._thumb_images.append(photo)
                            btn.configure(image=photo, text="", compound=tk.TOP)

                    # クリックイベント
                    btn.configure(
                        command=lambda c=cat, idx=i: self._on_candidate_click(c, idx)
                    )
                    btn.configure(state=tk.NORMAL)
                else:
                    btn.configure(state=tk.DISABLED, text="---", image="")

    def _on_candidate_click(self, category: str, index: int):
        """候補がクリックされた"""
        frames = self.classified_frames.get(category, [])
        if index < len(frames):
            self.selected_frames[category] = frames[index]
            self.selection_labels[category].configure(
                text=f"選択中: フレーム {frames[index].frame_idx}",
                foreground="green"
            )
            self.log(f"{category}: フレーム {frames[index].frame_idx} を選択")

            # 全カテゴリ選択済みかチェック
            if all(self.selected_frames.values()):
                self.step2_btn.configure(state=tk.NORMAL)

    def _goto_step3(self):
        """STEP3へ進む"""
        if not all(self.selected_frames.values()):
            messagebox.showwarning("警告", "すべてのカテゴリで口を選択してください")
            return

        self.notebook.select(2)  # STEP3へ
        self._on_update_preview()  # プレビューを自動更新

    def _on_mask_type_changed(self):
        """マスク種類が変更された"""
        mask_type = self.mask_type_var.get()
        if mask_type == "sam3":
            self.dilate_slider.configure(state=tk.NORMAL)
        else:
            self.dilate_slider.configure(state=tk.DISABLED)

    def _reset_settings(self):
        """設定をリセット"""
        self.feather_var.set(DEFAULT_FEATHER)
        self.dilate_var.set(DEFAULT_MASK_DILATE)
        self.offset_x_var.set(DEFAULT_OFFSET_X)
        self.offset_y_var.set(DEFAULT_OFFSET_Y)
        self.scale_var.set(DEFAULT_SCALE)
        self.mask_type_var.set("ellipse")
        self.log("設定をリセットしました")

    def _get_video_capture(self) -> cv2.VideoCapture:
        """キャッシュされたVideoCaptureを取得"""
        if self._cached_cap is None or not self._cached_cap.isOpened():
            self._cached_cap = cv2.VideoCapture(self.video_path)
        return self._cached_cap

    def _on_update_preview(self):
        """プレビュー更新"""
        if not all(self.selected_frames.values()) or not self.unified_size:
            return

        cap = self._get_video_capture()
        unified_w, unified_h = self.unified_size

        feather_px = self.feather_var.get()
        offset_x = self.offset_x_var.get()
        offset_y = self.offset_y_var.get()
        scale = self.scale_var.get()
        use_sam_mask = self.mask_type_var.get() == "sam3"
        mask_dilate = self.dilate_var.get()

        self.preview_sprites = {}

        for cat, mf in self.selected_frames.items():
            if mf is None:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            adjusted_quad = adjust_quad(mf.quad, offset_x, offset_y, scale)

            bgra = extract_mouth_sprite(
                frame, adjusted_quad, unified_w, unified_h,
                feather_px=feather_px,
                sam_mask=mf.mask,
                use_sam_mask=use_sam_mask,
                mask_dilate=mask_dilate,
            )

            self.preview_sprites[cat] = bgra

            composited = composite_on_checkerboard(bgra)
            photo = numpy_to_photoimage(composited, PREVIEW_SIZE)
            if photo:
                self.preview_images[cat] = photo
                self.preview_labels[cat].configure(image=photo, text="")

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

        for cat, bgra in self.preview_sprites.items():
            out_path = os.path.join(output_dir, f"{mouth_names[cat]}.png")
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
