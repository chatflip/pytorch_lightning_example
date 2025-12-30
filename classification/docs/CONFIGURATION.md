# ⚙️ 設定ファイルガイド

## YAML継承

```yaml
__base__:
  - ../_base_/default.yaml
  - ../_base_/model/efficientnet_b0.yaml

exp_name: "my_experiment"
data:
  batch_size: 128
```

**ルール**: 後から読み込んだ設定が上書き、辞書は再帰マージ、リストは置換

## 設定一覧

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
  tracking_uri: sqlite:///mlflow.db
```

### モデル (`_base_/model/*.yaml`)

```yaml
model:
  name: "efficientnet_b0"  # timmモデル名
  pretrained: true
  drop_rate: 0.0
```

### データ

```yaml
data:
  dataset_root: "data/datasets/food101"
  num_classes: 101
  batch_size: 64
  color_order: rgb  # rgb または bgr
```

### オーギュメンテーション (`_base_/augmentation/*.yaml`)

```yaml
augmentation:
  train:
    ops:
      - type: Resize
        height: 256
        width: 256
      - type: HorizontalFlip
        p: 0.5
      - type: Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      - type: ToTensorV2
```

### オプティマイザー (`_base_/optimizer/*.yaml`)

```yaml
optimizer:
  type: AdamW
  lr: 0.001
  weight_decay: 0.01

scheduler:
  type: CosineAnnealingLR
  eta_min: 0.00001
```

### トレーナー (`_base_/trainer/*.yaml`)

```yaml
trainer:
  max_epochs: 100
  accelerator: "gpu"
  devices: 1
  precision: "16-mixed"
  deterministic: true

checkpoint:
  monitor: "val_loss"
  mode: "min"
  save_top_k: 1
  save_last: true
```

## 実験設定例

```yaml
__base__:
  - ../_base_/default.yaml
  - ../_base_/model/efficientnet_b0.yaml
  - ../_base_/augmentation/basic.yaml
  - ../_base_/optimizer/adamw.yaml
  - ../_base_/trainer/default.yaml

exp_name: "food101_efficientnet_b0"
data:
  dataset_root: "data/datasets/food101"
  num_classes: 101
  batch_size: 128

trainer:
  max_epochs: 50
```

## 環境変数

```yaml
data:
  dataset_root: "${DATASET_ROOT}/food101"
```
