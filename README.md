# MouthSpriteExtractor-SAM3

SAM3 (Segment Anything Model 3) を使用して、動画から口スプライト（5種類のPNG）を自動抽出するツールです。

PNGTuber / VTuber 向けの口パク素材作成に使用できます。

## 特徴

- **SAM3のテキストプロンプト**で口を自動検出（"mouth"プロンプト使用）
- anime-face-detector等の顔検出ライブラリ不要
- 5種類の口形状を自動選別（open, closed, half, e, u）
- 楕円マスク＋フェザーで自然な透過PNG出力
- GUIとCLIの両方に対応

## 必要環境

- Python 3.12以上
- PyTorch 2.5以上（CUDA 12.6推奨）
- CUDA対応GPU（推奨）

## セットアップ

### クイックスタート（uv使用・推奨）

```bash
# 1. リポジトリをクローン
git clone https://github.com/yourusername/MouthSpriteExtractor-SAM3.git
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

1. 動画ファイルを選択（またはドラッグ&ドロップ）
2. 「解析開始」をクリック
3. 候補フレームに1-5の数字を入力して割り当て
   - 1: open（大きく開いた口）
   - 2: closed（閉じた口）
   - 3: half（半開きの口）
   - 4: e（横長の口、「え」）
   - 5: u（すぼめた口、「う」）
4. 「プレビュー更新」で確認
5. 「PNG出力」で保存

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
├── mouth_open.png     # 大きく開いた口
├── mouth_closed.png   # 閉じた口
├── mouth_half.png     # 半開きの口
├── mouth_e.png        # 横長の口
└── mouth_u.png        # すぼめた口
```

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

- [Ultralytics](https://github.com/ultralytics/ultralytics) - SAM3のPython実装
- [facebookresearch/sam3](https://github.com/facebookresearch/sam3) - SAM3モデル
- [MotionPNGTuber](https://github.com/uezo/MotionPNGTuber) - オリジナルプロジェクト

## 関連プロジェクト

- [MotionPNGTuber](https://github.com/uezo/MotionPNGTuber) - anime-face-detectorを使用した口スプライト抽出ツール（Python 3.10）
