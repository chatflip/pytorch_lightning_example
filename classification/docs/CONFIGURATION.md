# ⚙️ 設定ファイルガイド

本ドキュメントでは、YAML継承構造による設定管理の詳細を解説します。

## YAML継承の仕組み

`__base__`キーを使用してベース設定を継承し、必要な部分のみオーバーライドできます。

```yaml
# config/experiments/food101_efficientnet_b0.yaml
__base__:
  - ../_base_/default.yaml
  - ../_base_/model/efficientnet_b0.yaml
  - ../_base_/augmentation/basic.yaml
  - ../_base_/optimizer/adamw.yaml
  - ../_base_/trainer/default.yaml

exp_name: "food101_efficientnet_b0"

data:
  dataset_root: "./data/datasets/food101"
  num_classes: 101
  batch_size: 128

trainer:
  max_epochs: 50
```

### 継承のルール

1. `__base__` に指定された順番でベースファイルを読み込む
2. 後から読み込んだ設定が前の設定を上書きする
3. 実験設定ファイル自体の設定が最優先される
4. 辞書は再帰的にマージされる
5. リストは置換される（マージされない）

## 設定カテゴリ

| 設定カテゴリ | 説明 |
|--------------|------|
| `model` | timmのモデル名、事前学習、ドロップアウト率 |
| `data` | データセットパス、バッチサイズ、ワーカー数 |
| `augmentation` | train/valのオーギュメンテーションパイプライン |
| `optimizer` | オプティマイザータイプ、学習率、重み減衰 |
| `scheduler` | 学習率スケジューラー設定 |
| `trainer` | エポック数、精度、デバイス設定 |
| `logger` | ロガータイプ、実験名 |

## ベース設定ファイル

### 共通設定 (`_base_/default.yaml`)

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

### モデル設定 (`_base_/model/*.yaml`)

```yaml
model:
  name: "efficientnet_b0"    # timmのモデル名
  pretrained: true           # 事前学習済み重みを使用
  drop_rate: 0.0             # Dropout率
  drop_path_rate: 0.0        # DropPath率（ViT系）
```

利用可能なモデル:

- `efficientnet_b0` - 軽量で高精度なモデル
- `resnet50` - 安定した性能の標準モデル
- その他、timmでサポートされる全てのモデル

### オーギュメンテーション設定 (`_base_/augmentation/*.yaml`)

DEIMv2スタイルの`ops`配列形式を採用。各オペレーションは`type`でalbumentationsのクラス名を指定し、その他のキーが引数となります。

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

**サポートするオーギュメンテーション:**

- **リサイズ系**: Resize, RandomResizedCrop, CenterCrop, RandomCrop
- **反転・回転**: HorizontalFlip, VerticalFlip, Rotate, RandomRotate90
- **色変換**: ColorJitter, RandomBrightnessContrast, HueSaturationValue
- **ノイズ**: GaussianBlur, GaussNoise
- **正規化**: Normalize, ToTensorV2
- **その他**: CoarseDropout, GridDistortion, ElasticTransform

### オプティマイザー設定 (`_base_/optimizer/*.yaml`)

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

**サポートするオプティマイザー:**

- SGD, Adam, AdamW, RMSprop

**サポートするスケジューラー:**

- StepLR, MultiStepLR, ExponentialLR
- CosineAnnealingLR, CosineAnnealingWarmRestarts
- OneCycleLR

### トレーナー設定 (`_base_/trainer/*.yaml`)

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

## 実験設定の作成例

新しい実験設定を作成する場合:

```yaml
# config/experiments/my_experiment.yaml
__base__:
  - ../_base_/default.yaml
  - ../_base_/model/resnet50.yaml
  - ../_base_/augmentation/strong.yaml
  - ../_base_/optimizer/adamw.yaml
  - ../_base_/trainer/default.yaml

exp_name: "my_custom_experiment"

data:
  dataset_root: "./data/datasets/my_dataset"
  num_classes: 10
  batch_size: 64

trainer:
  max_epochs: 100
  precision: "bf16-mixed"

optimizer:
  lr: 0.0005
```

## コマンドラインオーバーライド

設定をコマンドラインでオーバーライドすることも可能です:

```bash
# バッチサイズとエポック数をオーバーライド
uv run python src/train.py -c config/experiments/food101_efficientnet_b0.yaml \
    --data.batch_size 128 \
    --trainer.max_epochs 100

# シードを指定
uv run python src/train.py -c config/experiments/food101_efficientnet_b0.yaml \
    --seed 42
```

## 詳細仕様

より詳細な技術仕様については、[SPECIFICATION.md](SPECIFICATION.md) を参照してください。
