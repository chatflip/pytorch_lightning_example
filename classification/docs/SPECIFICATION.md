# 画像分類学習システム 仕様書

## 1. 概要

DEIMv2を参考にしたYAML継承構造を持つ、汎用的な画像分類学習システム。
timmモデルとalbumentationsを使用し、すべての設定をYAMLで管理できる。

## 2. 技術スタック

| コンポーネント | ライブラリ |
|----------------|------------|
| 深層学習フレームワーク | PyTorch |
| 学習フレームワーク | PyTorch Lightning |
| モデル | timm |
| オーギュメンテーション | albumentations |
| 設定管理 | YAML + カスタムローダー |
| ロギング | MLFlow / TensorBoard |

## 3. ディレクトリ構成

```text
classification/
├── config/
│   ├── _base_/                          # ベース設定（継承元）
│   │   ├── default.yaml                 # 共通デフォルト設定
│   │   ├── model/
│   │   │   ├── efficientnet_b0.yaml
│   │   │   └── resnet50.yaml
│   │   ├── augmentation/
│   │   │   ├── basic.yaml               # 基本的なオーグメンテーション
│   │   │   └── strong.yaml              # 強いオーグメンテーション
│   │   ├── optimizer/
│   │   │   ├── sgd.yaml
│   │   │   └── adamw.yaml
│   │   └── trainer/
│   │       └── default.yaml
│   └── experiments/                     # 実験設定（継承して使用）
│       └── food101_efficientnet_b0.yaml
├── src/
│   ├── __init__.py
│   ├── train.py                         # エントリーポイント
│   ├── config/                          # 設定モジュール
│   │   ├── __init__.py
│   │   └── loader.py                    # YAML継承ローダー
│   ├── data/                            # データモジュール
│   │   ├── __init__.py
│   │   ├── datamodule.py                # LightningDataModule
│   │   └── dataset.py                   # ImageFolderDataset
│   ├── models/                          # モデルモジュール
│   │   ├── __init__.py
│   │   └── classifier.py                # LightningModule
│   ├── builders/                        # ビルダーモジュール
│   │   ├── __init__.py
│   │   ├── augmentation.py              # albumentationsビルダー
│   │   ├── optimizer.py                 # オプティマイザービルダー
│   │   └── logger.py                    # ロガービルダー
│   └── utils/                           # ユーティリティモジュール
│       ├── __init__.py
│       ├── metrics.py                   # メトリクス関連
│       └── timer.py                     # タイマー関連
├── datasets/                            # データセット配置場所
├── outputs/                             # 出力（チェックポイント等）
└── docs/
    └── SPECIFICATION.md                 # 本ドキュメント
```

## 4. データセット構造

ImageFolder形式を採用。データセットルートディレクトリの下に`train`と`val`ディレクトリがあり、
その中にクラス名のサブディレクトリがあり、画像が配置される。

```text
dataset_root/
├── train/
│   ├── class_a/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── class_b/
│   │   └── ...
│   └── ...
└── val/
    ├── class_a/
    │   └── ...
    ├── class_b/
    │   └── ...
    └── ...
```

## 5. YAML設定仕様

### 5.1 YAML継承の仕組み

`__base__` キーを使用してベース設定を継承し、必要な部分のみオーバーライドする。

```yaml
__base__:
  - ../_base_/default.yaml
  - ../_base_/model/efficientnet_b0.yaml
  - ../_base_/augmentation/basic.yaml

# オーバーライドする設定
exp_name: "my_experiment"
data:
  batch_size: 128
```

継承のルール:

1. `__base__` に指定された順番でベースファイルを読み込む
2. 後から読み込んだ設定が前の設定を上書きする
3. 実験設定ファイル自体の設定が最優先される
4. 辞書は再帰的にマージされる
5. リストは置換される（マージされない）

### 5.2 設定ファイル構造

#### 5.2.1 共通設定 (`_base_/default.yaml`)

```yaml
seed: 42
output_dir: "./outputs"
exp_name: "default"

data:
  num_workers: 8
  pin_memory: true

logger:
  type: "mlflow"
  experiment_name: "classification"
  log_model: true
  tracking_uri: null  # null = ローカル
```

#### 5.2.2 モデル設定 (`_base_/model/*.yaml`)

```yaml
model:
  name: "efficientnet_b0"    # timmのモデル名
  pretrained: true           # 事前学習済み重みを使用
  drop_rate: 0.0             # Dropout率
  drop_path_rate: 0.0        # DropPath率（ViT系）
```

#### 5.2.3 オーギュメンテーション設定 (`_base_/augmentation/*.yaml`)

DEIMv2スタイルの`ops`配列形式を採用。
各オペレーションは`type`でalbumentationsのクラス名を指定し、その他のキーが引数となる。

```yaml
augmentation:
  train:
    ops:
      - type: Resize
        height: 256
        width: 256
      - type: RandomCrop
        height: 224
        width: 224
      - type: HorizontalFlip
        p: 0.5
      - type: ColorJitter
        brightness: 0.2
        contrast: 0.2
        saturation: 0.2
        hue: 0.1
        p: 0.5
      - type: Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      - type: ToTensorV2
  val:
    ops:
      - type: Resize
        height: 256
        width: 256
      - type: CenterCrop
        height: 224
        width: 224
      - type: Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      - type: ToTensorV2
```

サポートするオーギュメンテーション（albumentationsのクラス）:

- **リサイズ系**: Resize, RandomResizedCrop, CenterCrop, RandomCrop, Crop
- **反転・回転**: HorizontalFlip, VerticalFlip, Rotate, RandomRotate90
- **色変換**: ColorJitter, RandomBrightnessContrast, HueSaturationValue, RGBShift
- **ノイズ**: GaussianBlur, GaussNoise, ISONoise
- **正規化**: Normalize, ToTensorV2
- **その他**: CoarseDropout, Cutout, GridDistortion, ElasticTransform

#### 5.2.4 オプティマイザー設定 (`_base_/optimizer/*.yaml`)

```yaml
optimizer:
  type: AdamW
  lr: 0.001
  weight_decay: 0.01
  betas: [0.9, 0.999]

scheduler:
  type: CosineAnnealingLR
  T_max: null          # null = max_epochsを使用
  eta_min: 0.00001
```

サポートするオプティマイザー:

- SGD, Adam, AdamW, RMSprop

サポートするスケジューラー:

- StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR
- CosineAnnealingWarmRestarts, OneCycleLR

#### 5.2.5 トレーナー設定 (`_base_/trainer/*.yaml`)

```yaml
trainer:
  max_epochs: 100
  accelerator: "gpu"
  devices: 1
  precision: "16-mixed"    # "32", "16-mixed", "bf16-mixed"
  strategy: "auto"         # "auto", "ddp", "deepspeed"
  deterministic: true
  log_every_n_steps: 10
  val_check_interval: 1.0  # 1.0 = 各エポック終了時

checkpoint:
  monitor: "val_loss"
  mode: "min"
  save_top_k: 3
  save_last: true
```

#### 5.2.6 実験設定 (`experiments/*.yaml`)

```yaml
__base__:
  - ../_base_/default.yaml
  - ../_base_/model/efficientnet_b0.yaml
  - ../_base_/augmentation/basic.yaml
  - ../_base_/optimizer/adamw.yaml
  - ../_base_/trainer/default.yaml

exp_name: "food101_efficientnet_b0"

data:
  dataset_root: "./datasets/food-101-imagefolder"
  num_classes: 101
  batch_size: 64

trainer:
  max_epochs: 50

# オーギュメンテーションの部分的オーバーライドも可能
augmentation:
  train:
    ops:
      - type: Resize
        height: 288
        width: 288
      - type: RandomResizedCrop
        height: 224
        width: 224
        scale: [0.8, 1.0]
      - type: HorizontalFlip
        p: 0.5
      - type: Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      - type: ToTensorV2
```

## 6. コンポーネント仕様

### 6.1 ConfigLoader

YAML継承を処理するローダー。

```python
from config_loader import load_config

config = load_config("config/experiments/food101_efficientnet_b0.yaml")
```

機能:

- `__base__` キーの解析と継承処理
- 辞書の再帰的マージ
- 相対パスの解決
- 環境変数の展開（`${ENV_VAR}` 形式）

### 6.2 AugmentationBuilder

YAML設定からalbumentationsパイプラインを構築。

```python
from builders.augmentation import build_transforms

train_transform = build_transforms(config["augmentation"]["train"])
val_transform = build_transforms(config["augmentation"]["val"])
```

入力: `ops` 配列を含む辞書
出力: `albumentations.Compose` オブジェクト

### 6.3 OptimizerBuilder

YAML設定からオプティマイザーとスケジューラーを構築。

```python
from builders.optimizer import build_optimizer, build_scheduler

optimizer = build_optimizer(config["optimizer"], model.parameters())
scheduler = build_scheduler(config["scheduler"], optimizer, max_epochs)
```

### 6.4 LoggerBuilder

YAML設定からPyTorch Lightningロガーを構築。

```python
from builders.logger import build_logger

logger = build_logger(config["logger"])
```

サポート:

- MLFlowLogger
- TensorBoardLogger
- WandbLogger

### 6.5 ImageFolderDataset

汎用的な画像分類データセット。

```python
from ImageFolderDataset import ImageFolderDataset

dataset = ImageFolderDataset(
    root="datasets/food-101-imagefolder/train",
    transform=train_transform
)
```

特徴:

- albumentationsトランスフォームを使用
- クラス名からラベルへの自動マッピング
- 画像ファイル拡張子のフィルタリング

### 6.6 ClassificationDataModule

PyTorch Lightning DataModule。

```python
from ClassificationDataModule import ClassificationDataModule

datamodule = ClassificationDataModule(config)
```

機能:

- train/val/testデータローダーの作成
- AugmentationBuilderを使用したトランスフォーム構築
- 設定からのバッチサイズ・ワーカー数の読み込み

### 6.7 ImageClassifier

PyTorch Lightning Module。

```python
from ImageClassifier import ImageClassifier

model = ImageClassifier(config)
```

機能:

- timmモデルの動的作成
- OptimizerBuilderを使用した動的オプティマイザー設定
- 訓練・検証・テストステップの実装
- メトリクス（accuracy, loss）のログ

## 7. 実行方法

### 7.1 学習の実行

```bash
# 設定ファイルを指定して学習
python src/train.py -c config/experiments/food101_efficientnet_b0.yaml

# 設定をコマンドラインでオーバーライド
python src/train.py -c config/experiments/food101_efficientnet_b0.yaml \
    --data.batch_size 128 \
    --trainer.max_epochs 100

# シードを指定
python src/train.py -c config/experiments/food101_efficientnet_b0.yaml \
    --seed 42
```

### 7.2 再開

```bash
# チェックポイントから再開
python src/train.py -c config/experiments/food101_efficientnet_b0.yaml \
    --resume outputs/food101_efficientnet_b0/checkpoints/last.ckpt
```

### 7.3 テストのみ

```bash
python src/train.py -c config/experiments/food101_efficientnet_b0.yaml \
    --test-only \
    --resume outputs/food101_efficientnet_b0/checkpoints/best.ckpt
```

## 8. 出力

```text
outputs/
└── {exp_name}/
    ├── checkpoints/
    │   ├── best.ckpt
    │   ├── last.ckpt
    │   └── epoch=XX-val_loss=X.XX.ckpt
    ├── logs/
    │   └── (TensorBoard/MLFlow logs)
    └── config.yaml              # 使用した設定のダンプ
```

## 9. 依存関係

```text
pytorch-lightning>=2.0.0
timm>=0.9.0
albumentations>=1.3.0
torch>=2.0.0
torchvision>=0.15.0
mlflow>=2.0.0
pyyaml>=6.0
loguru>=0.7.0
```
