#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui.py

SAM3を使用した口スプライト抽出GUI（2ステップ版・左右分割レイアウト）。

STEP1-2: 動画選択・抽出・口選択（左右分割）
STEP3: マスク調整・出力

License: AGPL-3.0 (Ultralyticsライセンスに準拠)
Note: SAM3モデルの使用にはMeta SAM Licenseが適用されます。
"""

from __future__ import annotations

import os
import queue
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
    warp_frame_to_norm,
    ensure_even_ge2,
    extract_mouth_sprite,
    adjust_quad,
    classify_mouth_frames,
    center_to_quad,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE = "Mouth Sprite Extractor (SAM3)"
INITIAL_CANDIDATES_PER_CATEGORY = 5
MAX_CANDIDATES_PER_CATEGORY = 20
THUMB_HEIGHT = 40  # サムネイルの高さ（アスペクト比維持）
PREVIEW_MAX_SIZE = 400  # 右側プレビューの最大サイズ

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
REQUIRED_CATEGORIES = ["open", "closed", "half"]
OPTIONAL_CATEGORIES = ["e", "u"]

CATEGORY_LABELS_SHORT = {
    "open": "Open [必須]",
    "closed": "Closed [必須]",
    "half": "Half [必須]",
    "e": "E [任意]",
    "u": "U [任意]",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def numpy_to_photoimage(
    img: np.ndarray,
    max_size: Optional[int] = None,
    target_height: Optional[int] = None,
) -> Optional["ImageTk.PhotoImage"]:
    """numpy配列をPhotoImageに変換"""
    if not _HAS_PIL:
        return None
    try:
        if len(img.shape) == 2:
            pil_img = Image.fromarray(img)
        elif img.shape[2] == 4:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        w, h = pil_img.size

        if target_height is not None:
            # 高さ固定でアスペクト比維持
            new_h = target_height
            new_w = max(1, int(w * target_height / h))
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        elif max_size is not None:
            # 最大サイズでアスペクト比維持
            scale = min(max_size / w, max_size / h, 1.0)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return ImageTk.PhotoImage(pil_img)
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
    """口スプライト抽出GUIアプリケーション（左右分割版）"""

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1100x750")
        self.minsize(1000, 650)

        # State
        self.video_path: str = ""
        self.extractor: Optional[MouthSpriteExtractor] = None
        self.classified_frames: Dict[str, List[MouthFrameInfo]] = {}
        self.all_candidates: List[MouthFrameInfo] = []
        self.selected_frames: Dict[str, Optional[MouthFrameInfo]] = {
            cat: None for cat in MOUTH_CATEGORIES
        }
        self.preview_sprites: Dict[str, np.ndarray] = {}
        self.preview_images: Dict[str, "ImageTk.PhotoImage"] = {}
        self.unified_size: Optional[Tuple[int, int]] = None
        self.is_analyzing = False

        # 現在表示中の候補数
        self.shown_candidates: Dict[str, int] = {
            cat: INITIAL_CANDIDATES_PER_CATEGORY for cat in MOUTH_CATEGORIES
        }

        # 現在プレビュー中のフレーム
        self.current_preview_mf: Optional[MouthFrameInfo] = None

        # Video capture cache
        self._cached_cap: Optional[cv2.VideoCapture] = None

        # Log queue
        self.log_queue: queue.Queue[str] = queue.Queue()

        # Image references (prevent GC)
        self._thumb_images: List["ImageTk.PhotoImage"] = []
        self._preview_photo: Optional["ImageTk.PhotoImage"] = None

        # Build UI
        self._build_ui()

        # Check SAM3 availability
        self._check_sam3()

        # Start log polling
        self._poll_logs()

    def _reset_state(self):
        """状態をリセット"""
        self.classified_frames = {}
        self.all_candidates = []
        self.selected_frames = {cat: None for cat in MOUTH_CATEGORIES}
        self.preview_sprites = {}
        self.preview_images = {}
        self.unified_size = None
        self._thumb_images = []
        self.current_preview_mf = None
        self.shown_candidates = {
            cat: INITIAL_CANDIDATES_PER_CATEGORY for cat in MOUTH_CATEGORIES
        }

        if self._cached_cap is not None:
            self._cached_cap.release()
            self._cached_cap = None

        # UIリセット
        self._clear_candidates_ui()
        self._clear_preview_panel()
        self._update_selection_status()
        self.goto_step3_btn.configure(state=tk.DISABLED)

    def _check_sam3(self):
        """SAM3の利用可否をチェック"""
        available, error = is_sam3_available()
        if not available:
            self.log(f"警告: SAM3が利用できません: {error}")
            self.analyze_btn.configure(state=tk.DISABLED)
        else:
            self.log("SAM3: 利用可能")

    def _build_ui(self):
        """UIを構築"""
        # Main notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # STEP1-2: 抽出・選択（左右分割）
        self.step12_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.step12_frame, text="STEP1-2: 抽出・選択")

        # STEP3: 出力設定
        self.step3_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.step3_frame, text="STEP3: 出力")

        self._build_step12_ui()
        self._build_step3_ui()

        # Log area
        log_frame = ttk.LabelFrame(self, text="ログ", padding=3)
        log_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        self.log_text = tk.Text(log_frame, height=3, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.X)

    def _build_step12_ui(self):
        """STEP1-2: 左右分割UIを構築"""
        frame = self.step12_frame

        # 上部: 動画選択 & 解析
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill=tk.X, pady=(0, 5))

        # 動画選択
        ttk.Label(top_frame, text="動画:").pack(side=tk.LEFT)
        self.video_label = ttk.Label(top_frame, text="ファイルを選択...", width=40)
        self.video_label.pack(side=tk.LEFT, padx=5)

        self.select_btn = ttk.Button(top_frame, text="選択", command=self._on_select_video)
        self.select_btn.pack(side=tk.LEFT)

        # パディング
        ttk.Label(top_frame, text="  パディング:").pack(side=tk.LEFT)
        self.padding_var = tk.DoubleVar(value=DEFAULT_PADDING)
        self.padding_slider = ttk.Scale(
            top_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
            variable=self.padding_var, length=100
        )
        self.padding_slider.pack(side=tk.LEFT, padx=2)
        self.padding_label = ttk.Label(top_frame, text="30%", width=4)
        self.padding_label.pack(side=tk.LEFT)

        # 解析ボタン
        self.analyze_btn = ttk.Button(
            top_frame, text="解析開始", command=self._on_analyze, state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=10)

        # プログレスバー
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            top_frame, variable=self.progress_var, maximum=100, length=150
        )
        self.progress_bar.pack(side=tk.LEFT, padx=5)

        # Drag & drop
        if _HAS_TK_DND:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)

        # 左右分割メインエリア
        main_paned = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # 左側: 候補一覧
        left_frame = ttk.LabelFrame(main_paned, text="候補一覧", padding=5)
        main_paned.add(left_frame, weight=1)

        # 候補一覧用スクロール
        self.candidates_canvas = tk.Canvas(left_frame, width=450)
        candidates_scrollbar = ttk.Scrollbar(
            left_frame, orient=tk.VERTICAL, command=self.candidates_canvas.yview
        )
        self.candidates_inner = ttk.Frame(self.candidates_canvas)

        self.candidates_canvas.configure(yscrollcommand=candidates_scrollbar.set)
        candidates_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.candidates_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.candidates_canvas.create_window(
            (0, 0), window=self.candidates_inner, anchor=tk.NW
        )
        self.candidates_inner.bind("<Configure>", lambda e: self.candidates_canvas.configure(
            scrollregion=self.candidates_canvas.bbox("all")
        ))

        # カテゴリごとのUI
        self.category_frames: Dict[str, ttk.Frame] = {}
        self.candidate_buttons: Dict[str, List[ttk.Button]] = {}
        self.more_buttons: Dict[str, ttk.Button] = {}

        for cat in MOUTH_CATEGORIES:
            self._build_category_row(cat)

        # 右側: プレビュー & 選択状況
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        # プレビューパネル
        preview_panel = ttk.LabelFrame(right_frame, text="プレビュー", padding=5)
        preview_panel.pack(fill=tk.BOTH, expand=True)

        self.preview_label = ttk.Label(preview_panel, text="候補をクリックしてプレビュー")
        self.preview_label.pack(pady=10)

        self.preview_info_label = ttk.Label(preview_panel, text="", foreground="gray")
        self.preview_info_label.pack()

        # カテゴリ選択ボタン
        self.cat_btn_frame = ttk.Frame(preview_panel)
        self.cat_btn_frame.pack(pady=10)

        ttk.Label(self.cat_btn_frame, text="割り当て先:").pack(side=tk.LEFT, padx=(0, 5))

        self.assign_buttons: Dict[str, ttk.Button] = {}
        for cat in MOUTH_CATEGORIES:
            btn = ttk.Button(
                self.cat_btn_frame, text=cat.upper(), width=8,
                command=lambda c=cat: self._assign_current_to_category(c),
                state=tk.DISABLED
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.assign_buttons[cat] = btn

        # 選択状況
        status_frame = ttk.LabelFrame(right_frame, text="選択状況", padding=5)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        self.status_labels: Dict[str, ttk.Label] = {}
        for cat in MOUTH_CATEGORIES:
            row = ttk.Frame(status_frame)
            row.pack(fill=tk.X, pady=1)

            is_required = cat in REQUIRED_CATEGORIES
            label_text = f"{cat.upper()}" + (" [必須]" if is_required else " [任意]")
            ttk.Label(row, text=label_text, width=15).pack(side=tk.LEFT)

            status_lbl = ttk.Label(row, text="未選択", foreground="gray")
            status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.status_labels[cat] = status_lbl

            if not is_required:
                clear_btn = ttk.Button(
                    row, text="解除", width=4,
                    command=lambda c=cat: self._clear_selection(c)
                )
                clear_btn.pack(side=tk.RIGHT)

        # STEP3へボタン
        self.goto_step3_btn = ttk.Button(
            right_frame, text="STEP3へ進む（必須項目を選択してください）",
            command=self._goto_step3, state=tk.DISABLED
        )
        self.goto_step3_btn.pack(fill=tk.X, pady=(10, 0))

    def _build_category_row(self, cat: str):
        """カテゴリ行を構築"""
        cat_frame = ttk.Frame(self.candidates_inner)
        cat_frame.pack(fill=tk.X, pady=3)
        self.category_frames[cat] = cat_frame

        # カテゴリラベル
        is_required = cat in REQUIRED_CATEGORIES
        label_text = f"▼ {CATEGORY_LABELS_SHORT[cat]}"
        ttk.Label(cat_frame, text=label_text, font=("", 9, "bold")).pack(anchor=tk.W)

        # サムネイル行
        thumb_frame = ttk.Frame(cat_frame)
        thumb_frame.pack(fill=tk.X)

        self.candidate_buttons[cat] = []

        # もっと見るボタン
        more_btn = ttk.Button(
            thumb_frame, text="+", width=3,
            command=lambda c=cat: self._load_more_candidates(c),
            state=tk.DISABLED
        )
        more_btn.pack(side=tk.RIGHT, padx=2)
        self.more_buttons[cat] = more_btn

    def _build_step3_ui(self):
        """STEP3: 出力UIを構築"""
        frame = self.step3_frame

        # 上下分割
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # マスク設定
        settings_frame = ttk.LabelFrame(top_frame, text="マスク設定", padding=5)
        settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Mask type
        mask_row = ttk.Frame(settings_frame)
        mask_row.pack(fill=tk.X, pady=2)
        ttk.Label(mask_row, text="種類:").pack(side=tk.LEFT)
        self.mask_type_var = tk.StringVar(value="ellipse")
        ttk.Radiobutton(
            mask_row, text="楕円", variable=self.mask_type_var,
            value="ellipse", command=self._on_mask_type_changed
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            mask_row, text="SAM3", variable=self.mask_type_var,
            value="sam3", command=self._on_mask_type_changed
        ).pack(side=tk.LEFT)

        # Feather
        feather_row = ttk.Frame(settings_frame)
        feather_row.pack(fill=tk.X, pady=2)
        ttk.Label(feather_row, text="フェザー:").pack(side=tk.LEFT)
        self.feather_var = tk.IntVar(value=DEFAULT_FEATHER)
        ttk.Scale(
            feather_row, from_=0, to=MAX_FEATHER, orient=tk.HORIZONTAL,
            variable=self.feather_var, length=100
        ).pack(side=tk.LEFT, padx=5)
        self.feather_label = ttk.Label(feather_row, text=f"{DEFAULT_FEATHER}px", width=5)
        self.feather_label.pack(side=tk.LEFT)

        # Dilate
        dilate_row = ttk.Frame(settings_frame)
        dilate_row.pack(fill=tk.X, pady=2)
        ttk.Label(dilate_row, text="膨張:").pack(side=tk.LEFT)
        self.dilate_var = tk.IntVar(value=DEFAULT_MASK_DILATE)
        self.dilate_slider = ttk.Scale(
            dilate_row, from_=-MAX_MASK_DILATE, to=MAX_MASK_DILATE, orient=tk.HORIZONTAL,
            variable=self.dilate_var, length=100
        )
        self.dilate_slider.pack(side=tk.LEFT, padx=5)
        self.dilate_label = ttk.Label(dilate_row, text="0px", width=5)
        self.dilate_label.pack(side=tk.LEFT)
        self.dilate_slider.configure(state=tk.DISABLED)

        # 位置調整
        tuning_frame = ttk.LabelFrame(top_frame, text="位置調整", padding=5)
        tuning_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Offset X
        ox_row = ttk.Frame(tuning_frame)
        ox_row.pack(fill=tk.X, pady=1)
        ttk.Label(ox_row, text="X:").pack(side=tk.LEFT)
        self.offset_x_var = tk.IntVar(value=DEFAULT_OFFSET_X)
        ttk.Scale(
            ox_row, from_=-MAX_OFFSET, to=MAX_OFFSET, orient=tk.HORIZONTAL,
            variable=self.offset_x_var, length=100
        ).pack(side=tk.LEFT, padx=5)
        self.offset_x_label = ttk.Label(ox_row, text="0px", width=5)
        self.offset_x_label.pack(side=tk.LEFT)

        # Offset Y
        oy_row = ttk.Frame(tuning_frame)
        oy_row.pack(fill=tk.X, pady=1)
        ttk.Label(oy_row, text="Y:").pack(side=tk.LEFT)
        self.offset_y_var = tk.IntVar(value=DEFAULT_OFFSET_Y)
        ttk.Scale(
            oy_row, from_=-MAX_OFFSET, to=MAX_OFFSET, orient=tk.HORIZONTAL,
            variable=self.offset_y_var, length=100
        ).pack(side=tk.LEFT, padx=5)
        self.offset_y_label = ttk.Label(oy_row, text="0px", width=5)
        self.offset_y_label.pack(side=tk.LEFT)

        # Scale
        scale_row = ttk.Frame(tuning_frame)
        scale_row.pack(fill=tk.X, pady=1)
        ttk.Label(scale_row, text="倍率:").pack(side=tk.LEFT)
        self.scale_var = tk.DoubleVar(value=DEFAULT_SCALE)
        ttk.Scale(
            scale_row, from_=MIN_SCALE, to=MAX_SCALE, orient=tk.HORIZONTAL,
            variable=self.scale_var, length=100
        ).pack(side=tk.LEFT, padx=5)
        self.scale_label = ttk.Label(scale_row, text="100%", width=5)
        self.scale_label.pack(side=tk.LEFT)

        # ボタン
        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(side=tk.LEFT, padx=10)

        ttk.Button(btn_frame, text="プレビュー更新", command=self._on_update_preview).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="リセット", command=self._reset_settings).pack(fill=tk.X, pady=2)

        # プレビューエリア
        preview_frame = ttk.LabelFrame(frame, text="出力プレビュー", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True)

        preview_inner = ttk.Frame(preview_frame)
        preview_inner.pack()

        self.output_preview_labels: Dict[str, ttk.Label] = {}
        self.output_preview_name_labels: Dict[str, ttk.Label] = {}

        for cat in MOUTH_CATEGORIES:
            col = ttk.Frame(preview_inner)
            col.pack(side=tk.LEFT, padx=15, pady=5)

            name_lbl = ttk.Label(col, text=f"[{cat}]", font=("", 9, "bold"))
            name_lbl.pack()
            self.output_preview_name_labels[cat] = name_lbl

            img_lbl = ttk.Label(col, text="---")
            img_lbl.pack(pady=5)
            self.output_preview_labels[cat] = img_lbl

        # 出力ボタン
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

        # ラベル更新
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
        self.analyze_btn.configure(state=tk.NORMAL)
        self.log(f"動画を選択: {path}")

    def _on_analyze(self):
        """解析開始"""
        if not self.video_path or self.is_analyzing:
            return

        self._reset_state()
        self.is_analyzing = True
        self.analyze_btn.configure(state=tk.DISABLED)
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

            # 多めに分類
            self.classified_frames = classify_mouth_frames(
                self.extractor.mouth_frames,
                self.extractor.cluster_mask,
                candidates_per_category=MAX_CANDIDATES_PER_CATEGORY,
            )

            # 全候補リスト
            self.all_candidates = []
            seen_frames = set()
            for frames in self.classified_frames.values():
                for mf in frames:
                    if mf.frame_idx not in seen_frames:
                        self.all_candidates.append(mf)
                        seen_frames.add(mf.frame_idx)

            self.log(f"分類完了: {len(self.all_candidates)}件の候補")

            # 統一サイズ計算
            if self.all_candidates:
                max_w = max(mf.width for mf in self.all_candidates)
                max_h = max(mf.height for mf in self.all_candidates)
                self.unified_size = (
                    ensure_even_ge2(int(max_w * 1.2)),
                    ensure_even_ge2(int(max_h * 1.2))
                )
                self.log(f"統一サイズ: {self.unified_size}")

            # UIを更新
            self.after(0, self._populate_candidates)

            self.progress_var.set(100)
            self.log("解析完了 - 候補を選択してください")

        except Exception as e:
            self.log(f"エラー: {e}")
            traceback.print_exc()
        finally:
            self.is_analyzing = False
            self.after(0, lambda: self.analyze_btn.configure(state=tk.NORMAL))

    def _clear_candidates_ui(self):
        """候補UIをクリア"""
        for cat in MOUTH_CATEGORIES:
            # ボタンを削除
            for btn in self.candidate_buttons[cat]:
                btn.destroy()
            self.candidate_buttons[cat] = []
            self.more_buttons[cat].configure(state=tk.DISABLED)

    def _populate_candidates(self):
        """候補をUIに表示"""
        if not self.classified_frames or not self.unified_size:
            return

        cap = self._get_video_capture()
        unified_w, unified_h = self.unified_size

        self._thumb_images = []

        for cat in MOUTH_CATEGORIES:
            self._populate_category_candidates(cat, cap, unified_w, unified_h)

    def _populate_category_candidates(
        self, cat: str, cap: cv2.VideoCapture, unified_w: int, unified_h: int
    ):
        """カテゴリの候補を表示"""
        frames = self.classified_frames.get(cat, [])
        shown_count = self.shown_candidates[cat]

        # 既存ボタンを削除
        for btn in self.candidate_buttons[cat]:
            btn.destroy()
        self.candidate_buttons[cat] = []

        # サムネイルフレーム取得
        cat_frame = self.category_frames[cat]
        thumb_frame = None
        for child in cat_frame.winfo_children():
            if isinstance(child, ttk.Frame):
                thumb_frame = child
                break

        if thumb_frame is None:
            return

        # ボタン作成
        for i in range(min(shown_count, len(frames))):
            mf = frames[i]

            # 統一サイズのquadで正規化
            unified_quad = center_to_quad(mf.center, unified_w, unified_h)

            cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
            ok, frame = cap.read()

            btn = ttk.Button(thumb_frame, text="", width=8)
            btn.pack(side=tk.LEFT, padx=1, before=self.more_buttons[cat])

            if ok and frame is not None:
                patch = warp_frame_to_norm(frame, unified_quad, unified_w, unified_h)
                photo = numpy_to_photoimage(patch, target_height=THUMB_HEIGHT)
                if photo:
                    self._thumb_images.append(photo)
                    btn.configure(image=photo, compound=tk.TOP)

            btn.configure(command=lambda c=cat, idx=i: self._on_candidate_click(c, idx))
            self.candidate_buttons[cat].append(btn)

        # もっと見るボタン
        if len(frames) > shown_count:
            self.more_buttons[cat].configure(state=tk.NORMAL)
        else:
            self.more_buttons[cat].configure(state=tk.DISABLED)

        # 割り当て済み表示を更新
        self._update_candidate_buttons_appearance()

    def _load_more_candidates(self, cat: str):
        """候補を追加ロード"""
        self.shown_candidates[cat] = min(
            self.shown_candidates[cat] + 5,
            MAX_CANDIDATES_PER_CATEGORY
        )

        cap = self._get_video_capture()
        unified_w, unified_h = self.unified_size
        self._populate_category_candidates(cat, cap, unified_w, unified_h)
        self.log(f"{cat}: {self.shown_candidates[cat]}件表示")

    def _on_candidate_click(self, cat: str, index: int):
        """候補クリック - プレビュー表示"""
        frames = self.classified_frames.get(cat, [])
        if index >= len(frames):
            return

        mf = frames[index]
        self.current_preview_mf = mf
        self._update_preview_panel(mf)

        # 割り当てボタンを有効化
        for btn in self.assign_buttons.values():
            btn.configure(state=tk.NORMAL)

    def _update_preview_panel(self, mf: MouthFrameInfo):
        """プレビューパネルを更新"""
        cap = self._get_video_capture()
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
        ok, frame = cap.read()

        if not ok or frame is None:
            return

        unified_w, unified_h = self.unified_size
        unified_quad = center_to_quad(mf.center, unified_w, unified_h)

        # 口の位置を矩形で描画
        frame_draw = frame.copy()
        quad_int = unified_quad.astype(np.int32)
        cv2.polylines(frame_draw, [quad_int], isClosed=True, color=(0, 255, 0), thickness=2)

        photo = numpy_to_photoimage(frame_draw, max_size=PREVIEW_MAX_SIZE)
        if photo:
            self._preview_photo = photo
            self.preview_label.configure(image=photo, text="")

        self.preview_info_label.configure(
            text=f"フレーム {mf.frame_idx} | 元カテゴリ: {mf.category} | サイズ: {unified_w}x{unified_h}"
        )

    def _clear_preview_panel(self):
        """プレビューパネルをクリア"""
        self.preview_label.configure(image="", text="候補をクリックしてプレビュー")
        self.preview_info_label.configure(text="")
        self.current_preview_mf = None
        self._preview_photo = None

        for btn in self.assign_buttons.values():
            btn.configure(state=tk.DISABLED)

    def _assign_current_to_category(self, category: str):
        """現在プレビュー中のフレームをカテゴリに割り当て"""
        if self.current_preview_mf is None:
            return

        mf = self.current_preview_mf
        self.selected_frames[category] = mf

        source_cat = mf.category if mf.category else "?"
        self.log(f"{category}: フレーム {mf.frame_idx} を選択 (元: {source_cat})")

        self._update_candidate_buttons_appearance()
        self._update_selection_status()

    def _update_candidate_buttons_appearance(self):
        """候補ボタンの見た目を更新"""
        frame_to_assigned_cat: Dict[int, str] = {}
        for cat, mf in self.selected_frames.items():
            if mf is not None:
                frame_to_assigned_cat[mf.frame_idx] = cat

        for cat in MOUTH_CATEGORIES:
            frames = self.classified_frames.get(cat, [])
            buttons = self.candidate_buttons[cat]

            for i, btn in enumerate(buttons):
                if i < len(frames):
                    mf = frames[i]
                    assigned_cat = frame_to_assigned_cat.get(mf.frame_idx)

                    if assigned_cat:
                        btn.configure(text=f"→{assigned_cat.upper()}", compound=tk.BOTTOM)
                    else:
                        btn.configure(text="", compound=tk.TOP)

    def _update_selection_status(self):
        """選択状況を更新"""
        for cat in MOUTH_CATEGORIES:
            mf = self.selected_frames.get(cat)
            if mf:
                source_cat = mf.category if mf.category else "?"
                self.status_labels[cat].configure(
                    text=f"フレーム {mf.frame_idx} (元: {source_cat})",
                    foreground="green"
                )
            else:
                self.status_labels[cat].configure(text="未選択", foreground="gray")

        # STEP3ボタンの状態
        required_complete = all(
            self.selected_frames[cat] is not None
            for cat in REQUIRED_CATEGORIES
        )

        if required_complete:
            self.goto_step3_btn.configure(state=tk.NORMAL, text="STEP3へ進む")
        else:
            missing = [cat for cat in REQUIRED_CATEGORIES if self.selected_frames[cat] is None]
            self.goto_step3_btn.configure(
                state=tk.DISABLED,
                text=f"STEP3へ進む（{', '.join(missing)} を選択）"
            )

    def _clear_selection(self, category: str):
        """カテゴリの選択を解除"""
        self.selected_frames[category] = None
        self.log(f"{category}: 選択を解除")
        self._update_candidate_buttons_appearance()
        self._update_selection_status()

    def _goto_step3(self):
        """STEP3へ進む"""
        required_complete = all(
            self.selected_frames[cat] is not None
            for cat in REQUIRED_CATEGORIES
        )

        if not required_complete:
            missing = [cat for cat in REQUIRED_CATEGORIES if self.selected_frames[cat] is None]
            messagebox.showwarning("警告", f"必須項目を選択してください: {', '.join(missing)}")
            return

        self.notebook.select(1)
        self._on_update_preview()

    def _on_mask_type_changed(self):
        """マスク種類が変更された"""
        if self.mask_type_var.get() == "sam3":
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
        self._on_mask_type_changed()
        self.log("設定をリセット")

    def _get_video_capture(self) -> cv2.VideoCapture:
        """キャッシュされたVideoCaptureを取得"""
        if self._cached_cap is None or not self._cached_cap.isOpened():
            self._cached_cap = cv2.VideoCapture(self.video_path)
        return self._cached_cap

    def _on_update_preview(self):
        """出力プレビュー更新"""
        if not self.unified_size:
            return

        required_complete = all(
            self.selected_frames[cat] is not None
            for cat in REQUIRED_CATEGORIES
        )
        if not required_complete:
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

        for cat in MOUTH_CATEGORIES:
            mf = self.selected_frames.get(cat)
            if mf is None:
                self.output_preview_labels[cat].configure(image="", text="---")
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            unified_quad = center_to_quad(mf.center, unified_w, unified_h)
            adjusted_quad = adjust_quad(unified_quad, offset_x, offset_y, scale)

            bgra = extract_mouth_sprite(
                frame, adjusted_quad, unified_w, unified_h,
                feather_px=feather_px,
                sam_mask=mf.mask,
                use_sam_mask=use_sam_mask,
                mask_dilate=mask_dilate,
            )

            self.preview_sprites[cat] = bgra

            composited = composite_on_checkerboard(bgra)
            photo = numpy_to_photoimage(composited, target_height=80)
            if photo:
                self.preview_images[cat] = photo
                self.output_preview_labels[cat].configure(image=photo, text="")

        self.output_btn.configure(state=tk.NORMAL)
        self.log(f"プレビュー更新 ({len(self.preview_sprites)}枚)")

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
