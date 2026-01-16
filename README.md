# MouthSpriteExtractor-SAM3

SAM3 (Segment Anything Model 3) を使用して、動画から口スプライト（5種類のPNG）を自動抽出するツールです。

[MotionPNGTuber](https://github.com/rotejin/MotionPNGTuber) 向けの口パク素材作成に使用できます。

## このプロジェクトについて

本プロジェクトは、[MotionPNGTuber](https://github.com/rotejin/MotionPNGTuber)（作者: ろてじん[@rotejin](https://github.com/rotejin)さん）の口スプライト抽出機能を、SAM3を使用して再実装したものです。

**なぜ別リポジトリなのか:**

本来であればMotionPNGTuberにプルリクエストを送るべきところですが、以下の理由から別リポジトリとして公開しています：

1. **Windows対応の問題**: Meta公式のSAM3はTriton依存でWindowsで動作しないため、Ultralytics経由での利用が必要。しかしUltralyticsはPython 3.12以降を推奨しており、MotionPNGTuber（Python 3.10）と互換性がない
2. **ライセンスの問題**: UltralyticsはAGPL-3.0ライセンスであり、MotionPNGTuber本体（MITライセンス）に影響を与える可能性がある

## 特徴

- **SAM3のテキストプロンプト**で口を自動検出（"mouth"プロンプト使用）
- anime-face-detector等の顔検出ライブラリ不要
- **Python 3.12以降対応**
- 5種類の口形状を自動分類（open, closed, half, e, u）
- ユーザーが候補から選択可能（自動分類の候補を手動で修正できる）
- 楕円マスク または SAM3マスク＋フェザーで自然な透過PNG出力
- **MotionPNGTuber互換**の正方形スプライト出力
- GUIで直感的に操作可能

## 必要環境

- Python 3.12以上
- PyTorch 2.5以上（CUDA 12.6推奨）
- CUDA対応GPU（推奨）

## セットアップ

### クイックスタート（uv使用・推奨）

```bash
# 1. リポジトリをクローン
git clone https://github.com/kazuya-bros/MouthSpriteExtractor-SAM3.git
cd MouthSpriteExtractor-SAM3

# 2. 依存パッケージを一括インストール（CUDA 12.6版PyTorch含む）
uv sync

# 3. sam3.ptをダウンロード（HuggingFaceアクセス承認後）
uv run huggingface-cli download facebook/sam3 sam3.pt --local-dir .

# 4. GUIを起動
uv run python gui.py
```

### 手動セットアップ

#### 1. 仮想環境を作成

```bash
uv venv .venv --python 3.12
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

#### 2. PyTorchをインストール

```bash
# CUDA 12.6の場合
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CPUのみの場合
uv pip install torch torchvision
```

#### 3. 依存パッケージをインストール

```bash
uv pip install -r requirements.txt
```

### SAM3モデルの取得

1. [HuggingFace SAM3ページ](https://huggingface.co/facebook/sam3) にアクセス
2. 「Request access」をクリックしてアクセス申請
3. 承認後、HuggingFace CLIでログイン:

```bash
huggingface-cli login
```

4. `sam3.pt`をダウンロード:

```bash
huggingface-cli download facebook/sam3 sam3.pt --local-dir .
```

## 使い方

### GUI

```bash
python gui.py
```

**STEP1: 抽出・選択**
1. 動画ファイルを選択（またはドラッグ&ドロップ）
2. 「解析開始」をクリック
3. 左側の候補一覧から口の形を選択
4. 右側のプレビューで確認し、カテゴリボタンで割り当て
   - **OPEN**: 大きく開いた口（必須）
   - **CLOSED**: 閉じた口（必須）
   - **HALF**: 半開きの口（必須）
   - **E**: 横長の口（任意）
   - **U**: すぼめた口（任意）
5. 必須3種類を選択したら「STEP2へ進む」

**STEP2: 出力**
1. マスク設定（楕円 or SAM3）、フェザー幅を調整
2. 位置調整（X/Y オフセット、倍率）で微調整
3. 「プレビュー更新」で確認
4. 「PNG出力」で保存

### CLI

```bash
python mouth_sprite_extractor.py --video your_video.mp4 --out output_dir/
```

オプション:
- `--video`, `-v`: 入力動画ファイル（必須）
- `--out`, `-o`: 出力ディレクトリ
- `--feather`: フェザー幅（デフォルト: 15px）
- `--padding`: 口検出パディング比率（デフォルト: 0.3）
- `--device`: デバイス指定（cuda/cpu/auto）

## 出力ファイル

```
output_dir/
├── open.png     # 大きく開いた口
├── closed.png   # 閉じた口
├── half.png     # 半開きの口
├── e.png        # 横長の口（任意）
└── u.png        # すぼめた口（任意）
```

出力されるPNGは正方形で、MotionPNGTuberでそのまま使用できます。

## MotionPNGTuberとの連携

1. 本ツールで口スプライト5枚を出力
2. MotionPNGTuberで動画の口トラッキングを実行（`face_track_anime_detector.py`など）
3. `calibrate_mouth_track.py`でスプライトとトラックの位置合わせ
4. MotionPNGTuberで口パク動画を生成

## ライセンス

### このプロジェクト

**AGPL-3.0 License**

このプロジェクトはUltralyticsパッケージを使用しているため、AGPL-3.0ライセンスが適用されます。

### SAM3モデル

SAM3モデルの使用には[Meta SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)が適用されます。

**主な制限事項:**
- 軍事・戦争目的での使用禁止
- 原子力産業での使用禁止
- 武器開発・スパイ活動への使用禁止
- 輸出規制対象の活動への使用禁止

商用利用は上記制限の範囲内で可能です。詳細は[LICENSE](https://github.com/facebookresearch/sam3/blob/main/LICENSE)を参照してください。

### Ultralytics

[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) はAGPL-3.0ライセンスで提供されています。
商用利用には[Enterprise License](https://www.ultralytics.com/license)が必要な場合があります。

## 謝辞

- **[ろてじん (@rotejin)](https://github.com/rotejin)** さん - [MotionPNGTuber](https://github.com/rotejin/MotionPNGTuber) の作者。素晴らしいPNGTuberツールの設計と実装に感謝いたします
- [Ultralytics](https://github.com/ultralytics/ultralytics) - SAM3のPython実装
- [facebookresearch/sam3](https://github.com/facebookresearch/sam3) - SAM3モデル

## 関連プロジェクト

- [MotionPNGTuber](https://github.com/rotejin/MotionPNGTuber) - anime-face-detectorを使用した口スプライト抽出ツール（Python 3.10）。本プロジェクトのベースとなったツールです
