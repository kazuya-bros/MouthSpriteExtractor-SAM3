#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mouth_sprite_extractor.py

SAM3を使用して動画から口スプライト（5種類のPNG）を自動抽出するコアモジュール。

機能:
1. SAM3のテキストプロンプト"mouth"で口を自動検出
2. 5種類の口形状を自動選別（open, closed, half, e, u）
3. 楕円マスク＋フェザーで透過PNG出力

License: AGPL-3.0 (Ultralyticsライセンスに準拠)
Note: SAM3モデルの使用にはMeta SAM Licenseが適用されます。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from sam3_detector import (
    SAM3MouthDetector,
    bbox_to_quad,
    is_sam3_available,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MouthFrameInfo:
    """1フレームの口情報"""
    frame_idx: int
    quad: np.ndarray          # (4, 2) float32
    center: np.ndarray        # (2,) float32 - 口の中心座標
    width: float              # 口quadの幅
    height: float             # 口quadの高さ
    confidence: float         # 検出信頼度
    valid: bool               # 検出が有効か


@dataclass
class MouthTypeSelection:
    """5種類の口の選択結果"""
    open_idx: int             # 口を大きく開けたフレーム
    closed_idx: int           # 口を閉じたフレーム
    half_idx: int             # 半開きフレーム
    e_idx: int                # 横長の口フレーム
    u_idx: int                # すぼめた口フレーム

    def as_dict(self) -> Dict[str, int]:
        return {
            "open": self.open_idx,
            "closed": self.closed_idx,
            "half": self.half_idx,
            "e": self.e_idx,
            "u": self.u_idx,
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def quad_center(quad: np.ndarray) -> np.ndarray:
    """quadの中心座標を計算"""
    return quad.mean(axis=0).astype(np.float32)


def quad_wh(quad: np.ndarray) -> Tuple[float, float]:
    """quadの幅と高さを計算"""
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    w = float(np.linalg.norm(quad[1] - quad[0]))
    h = float(np.linalg.norm(quad[3] - quad[0]))
    return w, h


def ensure_even_ge2(n: int) -> int:
    """偶数に丸める（最小2）"""
    n = int(n)
    if n < 2:
        return 2
    return n if (n % 2 == 0) else (n - 1)


def adjust_quad(
    quad: np.ndarray,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    quadに微調整を適用する。

    Args:
        quad: (4, 2) float32 配列
        offset_x: X方向のオフセット（ピクセル）
        offset_y: Y方向のオフセット（ピクセル）
        scale: スケール係数（1.0 = 元のサイズ）

    Returns:
        調整後のquad
    """
    quad = quad.copy()

    # オフセットを適用
    quad[:, 0] += offset_x
    quad[:, 1] += offset_y

    # スケールを適用（中心を基準に拡大/縮小）
    if scale != 1.0:
        center = quad.mean(axis=0)
        quad = center + (quad - center) * scale

    return quad


# ---------------------------------------------------------------------------
# Position-aware clustering
# ---------------------------------------------------------------------------

def find_stable_position_cluster(
    centers: np.ndarray,
    valid: np.ndarray,
    distance_threshold: float = 50.0,
) -> np.ndarray:
    """
    口の中心座標が安定しているフレーム群を特定する。
    """
    N = len(centers)
    valid_indices = np.where(valid)[0]

    if len(valid_indices) < 5:
        return valid.copy()

    valid_centers = centers[valid_indices]

    counts = np.zeros(len(valid_indices), dtype=np.int32)
    for i, c in enumerate(valid_centers):
        dists = np.linalg.norm(valid_centers - c, axis=1)
        counts[i] = np.sum(dists <= distance_threshold)

    best_idx = np.argmax(counts)
    best_center = valid_centers[best_idx]

    dists_from_best = np.linalg.norm(valid_centers - best_center, axis=1)
    cluster_valid_mask = dists_from_best <= distance_threshold

    cluster_mask = np.zeros(N, dtype=bool)
    for i, orig_idx in enumerate(valid_indices):
        if cluster_valid_mask[i]:
            cluster_mask[orig_idx] = True

    return cluster_mask


# ---------------------------------------------------------------------------
# Mouth type selection
# ---------------------------------------------------------------------------

def select_5_mouth_types(
    mouth_frames: List[MouthFrameInfo],
    cluster_mask: np.ndarray,
) -> MouthTypeSelection:
    """
    5種類の口タイプを自動選別する。
    """
    candidates = [
        mf for mf in mouth_frames
        if mf.valid and cluster_mask[mf.frame_idx]
    ]

    if len(candidates) < 5:
        candidates = [mf for mf in mouth_frames if mf.valid]

    if len(candidates) == 0:
        raise ValueError("No valid mouth frames found")

    heights = np.array([mf.height for mf in candidates])
    widths = np.array([mf.width for mf in candidates])
    aspect_ratios = widths / np.maximum(heights, 1e-6)

    used_indices = set()

    def pick_best(scores: np.ndarray, maximize: bool = True) -> int:
        sorted_indices = np.argsort(scores)
        if maximize:
            sorted_indices = sorted_indices[::-1]

        for idx in sorted_indices:
            frame_idx = candidates[idx].frame_idx
            if frame_idx not in used_indices:
                used_indices.add(frame_idx)
                return frame_idx

        return candidates[sorted_indices[0]].frame_idx

    open_idx = pick_best(heights, maximize=True)
    closed_idx = pick_best(heights, maximize=False)

    median_height = np.median(heights)
    half_scores = -np.abs(heights - median_height)
    half_idx = pick_best(half_scores, maximize=True)

    e_idx = pick_best(aspect_ratios, maximize=True)

    height_median_dist = np.abs(heights - median_height)
    u_scores = -widths - 0.5 * height_median_dist
    u_idx = pick_best(u_scores, maximize=True)

    return MouthTypeSelection(
        open_idx=open_idx,
        closed_idx=closed_idx,
        half_idx=half_idx,
        e_idx=e_idx,
        u_idx=u_idx,
    )


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def make_ellipse_mask(w: int, h: int, rx: int, ry: int) -> np.ndarray:
    """楕円マスクを生成（0/255）"""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    rx = int(max(1, min(rx, w // 2 - 1)))
    ry = int(max(1, min(ry, h // 2 - 1)))
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0.0, 0.0, 360.0, 255, -1)
    return mask


def feather_mask(mask_u8: np.ndarray, feather_px: int) -> np.ndarray:
    """マスクにフェザー（グラデーション）を適用"""
    if feather_px <= 0:
        return (mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)

    k = 2 * int(feather_px) + 1
    m = cv2.GaussianBlur(mask_u8, (k, k), sigmaX=0)
    return (m.astype(np.float32) / 255.0).clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Sprite extraction
# ---------------------------------------------------------------------------

def warp_frame_to_norm(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    norm_w: int,
    norm_h: int,
) -> np.ndarray:
    """フレームから口パッチを正規化空間に変換"""
    src = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    dst = np.array([
        [0, 0],
        [norm_w - 1, 0],
        [norm_w - 1, norm_h - 1],
        [0, norm_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    patch = cv2.warpPerspective(
        frame_bgr,
        M,
        (int(norm_w), int(norm_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return patch


def extract_mouth_sprite(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    unified_w: int,
    unified_h: int,
    feather_px: int = 15,
    mask_scale: float = 0.85,
) -> np.ndarray:
    """
    フレームから口スプライトを抽出する。

    Returns:
        bgra: (H, W, 4) uint8 - 透過PNG用
    """
    patch = warp_frame_to_norm(frame_bgr, quad, unified_w, unified_h)

    rx = int((unified_w * mask_scale) * 0.5)
    ry = int((unified_h * mask_scale) * 0.5)
    mask_u8 = make_ellipse_mask(unified_w, unified_h, rx, ry)

    mask_f = feather_mask(mask_u8, feather_px)

    bgra = np.zeros((unified_h, unified_w, 4), dtype=np.uint8)
    bgra[:, :, :3] = patch
    bgra[:, :, 3] = (mask_f * 255).astype(np.uint8)

    return bgra


def compute_unified_size(
    mouth_frames: List[MouthFrameInfo],
    selected_indices: List[int],
    padding: float = 1.1,
) -> Tuple[int, int]:
    """選択されたフレームの口がすべて収まるサイズを計算"""
    idx_to_mf = {mf.frame_idx: mf for mf in mouth_frames}

    max_w = 0.0
    max_h = 0.0
    for idx in selected_indices:
        if idx in idx_to_mf:
            mf = idx_to_mf[idx]
            max_w = max(max_w, mf.width)
            max_h = max(max_h, mf.height)

    w = ensure_even_ge2(int(max_w * padding))
    h = ensure_even_ge2(int(max_h * padding))

    return w, h


def get_unique_output_dir(base_name: str = "mouth") -> str:
    """ユニークな出力ディレクトリ名を生成"""
    if not os.path.exists(base_name):
        return base_name

    i = 1
    while os.path.exists(f"{base_name}_{i:02d}"):
        i += 1
    return f"{base_name}_{i:02d}"


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class MouthSpriteExtractor:
    """SAM3を使用した口スプライト抽出器"""

    def __init__(
        self,
        video_path: str,
        device: str = "auto",
        padding_ratio: float = 0.3,
    ):
        """
        Args:
            video_path: 入力動画のパス
            device: SAM3のデバイス ("cuda", "cpu", "auto")
            padding_ratio: 口検出時のパディング比率
        """
        self.video_path = video_path
        self.device = device
        self.padding_ratio = padding_ratio

        # 動画情報
        self.vid_w = 0
        self.vid_h = 0
        self.fps = 0.0
        self.n_frames = 0

        # 解析結果
        self.mouth_frames: List[MouthFrameInfo] = []
        self.cluster_mask: Optional[np.ndarray] = None
        self.selection: Optional[MouthTypeSelection] = None
        self.unified_size: Optional[Tuple[int, int]] = None

        # SAM3検出器（遅延初期化）
        self._detector: Optional[SAM3MouthDetector] = None

        self._load_video_info()

    def _load_video_info(self):
        """動画の情報を取得"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        self.vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

    def _ensure_detector(self):
        """SAM3検出器を初期化"""
        if self._detector is None:
            self._detector = SAM3MouthDetector(device=self.device)

    def analyze(
        self,
        max_frames: int = 100,
        callback: Optional[Callable[[str], None]] = None,
    ):
        """
        動画を解析して口情報を取得。

        Args:
            max_frames: 処理する最大フレーム数
            callback: ログコールバック
        """
        def log(msg: str):
            if callback:
                callback(msg)
            else:
                print(msg)

        log(f"動画: {self.vid_w}x{self.vid_h}, {self.n_frames}フレーム, {self.fps:.1f}fps")

        # SAM3検出器を初期化
        log("SAM3を初期化中...")
        self._ensure_detector()
        log("SAM3の初期化完了")

        # フレーム間隔を計算
        stride = max(1, self.n_frames // max_frames)
        log(f"フレーム間隔: {stride}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        self.mouth_frames = []

        for frame_idx in range(0, self.n_frames, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # SAM3で口を検出
            result = self._detector.detect_mouth(
                frame,
                prompt="mouth",
                padding_ratio=self.padding_ratio,
            )

            if result is not None:
                mask, bbox, center = result
                quad = bbox_to_quad(bbox)  # 元のアスペクト比を維持

                mf = MouthFrameInfo(
                    frame_idx=frame_idx,
                    quad=quad,
                    center=np.array([center[0], center[1]], dtype=np.float32),
                    width=float(bbox[2] - bbox[0]),
                    height=float(bbox[3] - bbox[1]),
                    confidence=1.0,
                    valid=True,
                )
                self.mouth_frames.append(mf)

            if (frame_idx // stride) % 10 == 0:
                log(f"処理中... {frame_idx}/{self.n_frames} ({len(self.mouth_frames)}件検出)")

        cap.release()

        log(f"口検出完了: {len(self.mouth_frames)}件")

        if len(self.mouth_frames) == 0:
            log("警告: 口が検出されませんでした")
            return

        # クラスタリング
        centers = np.array([mf.center for mf in self.mouth_frames])
        valid = np.array([mf.valid for mf in self.mouth_frames])

        # フレームインデックスでマッピング
        max_idx = max(mf.frame_idx for mf in self.mouth_frames) + 1
        centers_full = np.zeros((max_idx, 2), dtype=np.float32)
        valid_full = np.zeros(max_idx, dtype=bool)

        for mf in self.mouth_frames:
            centers_full[mf.frame_idx] = mf.center
            valid_full[mf.frame_idx] = mf.valid

        self.cluster_mask = find_stable_position_cluster(centers_full, valid_full)

        # 5種類の口を選択
        self.selection = select_5_mouth_types(self.mouth_frames, self.cluster_mask)

        # 統一サイズを計算
        selected_indices = list(self.selection.as_dict().values())
        self.unified_size = compute_unified_size(
            self.mouth_frames, selected_indices, padding=1.2
        )

        log(f"5種類の口を選択: {self.selection.as_dict()}")
        log(f"統一サイズ: {self.unified_size}")

    def extract_sprites(
        self,
        output_dir: str,
        feather_px: int = 15,
        mask_scale: float = 0.85,
        callback: Optional[Callable[[str], None]] = None,
    ):
        """
        選択された5種類の口スプライトを出力。

        Args:
            output_dir: 出力ディレクトリ
            feather_px: フェザー幅
            mask_scale: マスクスケール
            callback: ログコールバック
        """
        def log(msg: str):
            if callback:
                callback(msg)
            else:
                print(msg)

        if self.selection is None or self.unified_size is None:
            raise RuntimeError("analyze() を先に実行してください")

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        idx_to_mf = {mf.frame_idx: mf for mf in self.mouth_frames}
        unified_w, unified_h = self.unified_size

        mouth_names = {
            "open": "mouth_open",
            "closed": "mouth_closed",
            "half": "mouth_half",
            "e": "mouth_e",
            "u": "mouth_u",
        }

        for mouth_type, frame_idx in self.selection.as_dict().items():
            if frame_idx not in idx_to_mf:
                log(f"警告: {mouth_type}のフレーム{frame_idx}が見つかりません")
                continue

            mf = idx_to_mf[frame_idx]

            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                log(f"警告: フレーム{frame_idx}の読み込みに失敗")
                continue

            bgra = extract_mouth_sprite(
                frame, mf.quad, unified_w, unified_h,
                feather_px=feather_px, mask_scale=mask_scale
            )

            out_path = os.path.join(output_dir, f"{mouth_names[mouth_type]}.png")
            cv2.imwrite(out_path, bgra)
            log(f"出力: {out_path}")

        cap.release()
        log(f"完了: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SAM3を使用して動画から口スプライトを抽出"
    )
    parser.add_argument("--video", "-v", required=True, help="入力動画ファイル")
    parser.add_argument("--out", "-o", default="", help="出力ディレクトリ")
    parser.add_argument("--feather", type=int, default=15, help="フェザー幅 (px)")
    parser.add_argument("--padding", type=float, default=0.3, help="口検出パディング比率")
    parser.add_argument("--device", default="auto", help="デバイス (cuda/cpu/auto)")

    args = parser.parse_args()

    # SAM3の利用可否をチェック
    available, error = is_sam3_available()
    if not available:
        print(f"エラー: SAM3が利用できません: {error}")
        print("\nセットアップ手順:")
        print("  1. pip install sam3 (git+https://github.com/facebookresearch/sam3.git)")
        print("  2. HuggingFaceでモデルへのアクセスを申請")
        print("  3. huggingface-cli login でログイン")
        return 1

    output_dir = args.out or get_unique_output_dir("mouth")

    extractor = MouthSpriteExtractor(
        args.video,
        device=args.device,
        padding_ratio=args.padding,
    )
    extractor.analyze()
    extractor.extract_sprites(output_dir, feather_px=args.feather)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
