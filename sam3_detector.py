#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sam3_detector.py

SAM3 (Segment Anything Model 3) を使用した口検出モジュール。
Ultralytics経由でSAM3を使用。

License: AGPL-3.0 (Ultralyticsライセンスに準拠)
Note: SAM3モデルの使用にはHuggingFaceでのアクセス承認が必要です。
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, List

import cv2
import numpy as np

# SAM3の利用可否
_SAM3_AVAILABLE = False
_SAM3_ERROR = ""

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError as e:
    _TORCH_AVAILABLE = False
    _SAM3_ERROR = f"PyTorch not available: {e}"

if _TORCH_AVAILABLE:
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
        _SAM3_AVAILABLE = True
    except ImportError as e:
        _SAM3_ERROR = f"ultralytics SAM3 not available: {e}"
    except Exception as e:
        _SAM3_ERROR = str(e)


def _test_cuda_actually_works() -> bool:
    """CUDAが実際に動作するかテストする（互換性のないGPUを検出）"""
    if not _TORCH_AVAILABLE:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        # 小さなテンソルでCUDA演算をテスト
        test_tensor = torch.zeros(1, device="cuda")
        _ = test_tensor + 1
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"[SAM3] CUDA test failed: {e}")
        return False


def is_sam3_available() -> Tuple[bool, str]:
    """SAM3が利用可能かチェック

    Returns:
        (available, error_message)
    """
    return _SAM3_AVAILABLE, _SAM3_ERROR


class SAM3MouthDetector:
    """SAM3を使用した口検出クラス（Ultralytics版）"""

    def __init__(self, device: str = "auto", confidence_threshold: float = 0.25):
        """
        Args:
            device: "cuda", "cpu", or "auto"
            confidence_threshold: 検出の信頼度閾値
        """
        if not _SAM3_AVAILABLE:
            raise RuntimeError(f"SAM3 is not available: {_SAM3_ERROR}")

        if device == "auto":
            # CUDAが利用可能かつ実際に動作するかテスト
            if torch.cuda.is_available() and _test_cuda_actually_works():
                device = "cuda"
            else:
                device = "cpu"
                if torch.cuda.is_available():
                    print("[SAM3] CUDA is available but not compatible with this GPU, using CPU")

        self.device = device
        self.confidence_threshold = confidence_threshold
        self._fallback_to_cpu = False

        print(f"[SAM3] Loading model on {device}...")
        print("[SAM3] Note: First run will download model from HuggingFace (requires access approval)")

        # Ultralytics SAM3 Predictorを初期化
        self.predictor = self._create_predictor(device)
        self._current_image = None

        print("[SAM3] Model loaded successfully")

    def _create_predictor(self, device: str):
        """Predictorを作成"""
        overrides = dict(
            conf=self.confidence_threshold,
            task="segment",
            mode="predict",
            model="sam3.pt",
            device=device,
            verbose=False,
        )
        return SAM3SemanticPredictor(overrides=overrides)

    def _reinit_on_cpu(self):
        """CPUでPredictorを再初期化"""
        if self.device == "cpu":
            return False
        print("[SAM3] CUDA error detected, falling back to CPU...")
        self.device = "cpu"
        self._fallback_to_cpu = True
        self.predictor = self._create_predictor("cpu")
        print("[SAM3] Model reloaded on CPU")
        return True

    def detect_mouth(
        self,
        image: np.ndarray,
        prompt: str = "mouth",
        padding_ratio: float = 0.0,
        padding_px: int = 0,
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[float, float]]]:
        """
        テキストプロンプトで口を検出する。

        Args:
            image: BGR画像（OpenCV形式）
            prompt: テキストプロンプト（デフォルト: "mouth"）
            padding_ratio: bboxを広げる比率（0.3 = 30%広げる）
            padding_px: bboxを広げるピクセル数

        Returns:
            (mask, bbox, center) または検出失敗時はNone
            - mask: (H, W) uint8 マスク（0/255）
            - bbox: (x_min, y_min, x_max, y_max)
            - center: (cx, cy) 口の中心座標
        """
        try:
            h, w = image.shape[:2]

            # 画像をセット（BGR -> RGB）
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image_rgb)

            # テキストプロンプトで検出
            results = self.predictor(text=[prompt])

            if results is None or len(results) == 0:
                return None

            result = results[0]

            # マスクを取得
            if result.masks is None or len(result.masks) == 0:
                return None

            masks = result.masks.data
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()

            # ボックスを取得
            if result.boxes is None or len(result.boxes) == 0:
                return None

            boxes = result.boxes.xyxy
            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu().numpy()

            # 信頼度スコアを取得
            scores = result.boxes.conf
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()

            # 最も信頼度の高いものを選択
            if len(scores) > 0:
                best_idx = int(np.argmax(scores))
            else:
                best_idx = 0

            # マスクを取得
            mask = masks[best_idx]
            if mask.ndim == 3:
                mask = mask.squeeze()

            # 元画像サイズにリサイズ
            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_LINEAR
                )

            mask_u8 = (mask > 0.5).astype(np.uint8) * 255

            # bboxを取得
            bbox = boxes[best_idx]
            x_min, y_min, x_max, y_max = map(int, bbox[:4])

            # 中心座標（パディング前）
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0

            # パディングを適用
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min

            pad_w = bbox_w * padding_ratio + padding_px
            pad_h = bbox_h * padding_ratio + padding_px

            x_min = max(0, int(x_min - pad_w))
            y_min = max(0, int(y_min - pad_h))
            x_max = min(w, int(x_max + pad_w))
            y_max = min(h, int(y_max + pad_h))

            return mask_u8, (x_min, y_min, x_max, y_max), (cx, cy)

        except (RuntimeError, Exception) as e:
            error_str = str(e)
            # CUDAエラーの場合、CPUにフォールバックしてリトライ
            if "CUDA" in error_str or "cuda" in error_str:
                if self._reinit_on_cpu():
                    # CPUで再試行
                    return self.detect_mouth(image, prompt, padding_ratio, padding_px)
            print(f"[SAM3] Detection error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def reset(self):
        """内部状態をリセット"""
        self._current_image = None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """マスクからバウンディングボックスを取得"""
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    return (int(x_min), int(y_min), int(x_max), int(y_max))


def bbox_to_quad(
    bbox: Tuple[int, int, int, int],
    sprite_aspect: float = 0.0,
) -> np.ndarray:
    """
    bboxからquad（4点）を生成する。

    Args:
        bbox: (x_min, y_min, x_max, y_max)
        sprite_aspect: スプライトのアスペクト比 (w/h)。0以下の場合は元のアスペクト比を維持。

    Returns:
        quad: (4, 2) float32 配列 [TL, TR, BR, BL]
    """
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    # アスペクト比を調整（sprite_aspect > 0 の場合のみ）
    if sprite_aspect > 0:
        current_aspect = w / max(1, h)
        if current_aspect > sprite_aspect:
            h = w / sprite_aspect
        else:
            w = h * sprite_aspect

    hw, hh = w / 2.0, h / 2.0

    quad = np.array([
        [cx - hw, cy - hh],  # TL
        [cx + hw, cy - hh],  # TR
        [cx + hw, cy + hh],  # BR
        [cx - hw, cy + hh],  # BL
    ], dtype=np.float32)

    return quad


def refine_mask_morphology(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """モルフォロジー処理でマスクを整える"""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def feather_mask(mask_u8: np.ndarray, feather_px: int) -> np.ndarray:
    """マスクにフェザー（グラデーション）を適用"""
    if feather_px <= 0:
        return (mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)

    k = 2 * int(feather_px) + 1
    m = cv2.GaussianBlur(mask_u8, (k, k), sigmaX=0)
    return (m.astype(np.float32) / 255.0).clip(0.0, 1.0)
