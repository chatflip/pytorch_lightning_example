# 🛠️ 開発者ガイド

本ドキュメントでは、開発環境のセットアップと開発ワークフローについて解説します。

## 開発環境のセットアップ

### 前提条件

- Python 3.12+
- CUDA対応GPU（推奨）
- [uv](https://docs.astral.sh/uv/) パッケージマネージャ

### インストール

```bash
# 依存関係のインストール
make install
```

## 開発ワークフロー

### コードフォーマット

```bash
# Ruffによるフォーマット
make format
```

### リント

```bash
# Ruffによるリント
make lint
```

## ディレクトリ構成

```
classification/
├── config/                           # 設定ファイル
│   ├── _base_/                       # ベース設定（継承元）
│   │   ├── default.yaml              # 共通デフォルト設定
│   │   ├── model/                    # モデル設定
│   │   ├── augmentation/             # オーギュメンテーション設定
│   │   ├── optimizer/                # オプティマイザー設定
│   │   └── trainer/                  # トレーナー設定
│   └── experiments/                  # 実験設定
├── src/                              # ソースコード
│   ├── train.py                      # エントリーポイント
│   ├── config/                       # 設定ローダー
│   ├── data/                         # データモジュール
│   ├── models/                       # モデル定義
│   └── builders/                     # ビルダー
├── tools/                            # ユーティリティツール
│   └── datasets/                     # データセット準備ツール
├── data/                             # データ
│   ├── raw/                          # ダウンロードした生データ
│   └── datasets/                     # 整形済みデータセット
├── outputs/                          # 学習出力（チェックポイント等）
├── mlruns/                           # MLFlow実験記録
└── docs/                             # ドキュメント
```

## モジュール構成

### src/config/

YAML継承を処理するローダー。

```python
from src.config import load_config

config = load_config("config/experiments/food101_efficientnet_b0.yaml")
```

### src/builders/

設定からオブジェクトを構築するビルダー群。

- `augmentation.py` - albumentationsパイプラインを構築
- `optimizer.py` - オプティマイザーとスケジューラーを構築
- `logger.py` - PyTorch Lightningロガーを構築

### src/data/

- `dataset.py` - ImageFolderDataset（汎用画像分類データセット）
- `datamodule.py` - ClassificationDataModule（LightningDataModule）

### src/models/

- `classifier.py` - ImageClassifier（LightningModule）

## データセットの追加

新しいデータセットを追加する場合:

1. `tools/datasets/` に準備スクリプトを作成
2. ImageFolder形式でデータを配置

```
data/datasets/my_dataset/
├── train/
│   ├── class_a/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── class_b/
│       └── ...
└── val/
    ├── class_a/
    │   └── ...
    └── class_b/
        └── ...
```

3. 実験設定ファイルを作成

```yaml
# config/experiments/my_dataset_resnet50.yaml
__base__:
  - ../_base_/default.yaml
  - ../_base_/model/resnet50.yaml
  - ../_base_/augmentation/basic.yaml
  - ../_base_/optimizer/adamw.yaml
  - ../_base_/trainer/default.yaml

exp_name: "my_dataset_resnet50"

data:
  dataset_root: "./data/datasets/my_dataset"
  num_classes: 2  # クラス数
  batch_size: 64
```

## 新しいモデルの追加

`config/_base_/model/` に新しいモデル設定を追加:

```yaml
# config/_base_/model/vit_base.yaml
model:
  name: "vit_base_patch16_224"
  pretrained: true
  drop_rate: 0.1
```

## 新しいオーギュメンテーションの追加

`config/_base_/augmentation/` に新しい設定を追加:

```yaml
# config/_base_/augmentation/mixup.yaml
augmentation:
  train:
    ops:
      - type: Resize
        height: 256
        width: 256
      - type: RandomResizedCrop
        height: 224
        width: 224
        scale: [0.8, 1.0]
      - type: HorizontalFlip
        p: 0.5
      - type: ColorJitter
        brightness: 0.4
        contrast: 0.4
        saturation: 0.4
        hue: 0.1
        p: 0.8
      - type: Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      - type: ToTensorV2
```

## MLFlow UI

実験の追跡にはMLFlowを使用しています。

```bash
# MLFlow UIを起動
uv run mlflow ui --port 5000
# ブラウザで http://localhost:5000 を開く
```
