# 画像分類学習システム 仕様書

## ディレクトリ構成

```text
classification/
├── config/
│   ├── _base_/                    # ベース設定
│   │   ├── default.yaml
│   │   ├── model/
│   │   ├── augmentation/
│   │   ├── optimizer/
│   │   └── trainer/
│   └── experiments/               # 実験設定
├── src/
│   ├── train.py                   # トレーニング
│   ├── validate.py                # 検証
│   ├── config/                    # 設定ローダー・スキーマ
│   ├── data/                      # データモジュール
│   ├── models/                    # モデル
│   └── builders/                  # ビルダー
├── data/datasets/                 # データセット
├── outputs/                       # 出力
└── tools/datasets/                # データ準備ツール
```

## YAML設定仕様

### 継承の仕組み

```yaml
__base__:
  - ../_base_/default.yaml
  - ../_base_/model/efficientnet_b0.yaml

exp_name: "my_experiment"
data:
  batch_size: 128
```

**継承ルール**: 後から読み込んだ設定が上書き、辞書は再帰マージ、リストは置換

### 設定セクション

| セクション | 内容 |
|-----------|------|
| `model` | timmモデル名、事前学習 |
| `data` | データパス、バッチサイズ、color_order |
| `augmentation` | train/valのオーグメンテーション |
| `optimizer` | オプティマイザー、学習率 |
| `scheduler` | 学習率スケジューラー |
| `trainer` | エポック数、精度、デバイス |
| `checkpoint` | 保存設定 |
| `logger` | MLFlow/TensorBoard設定 |

### オーギュメンテーション形式

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

## 実行方法

```bash
# 学習
uv run python src/train.py -c config/experiments/food101_efficientnet_b0.yaml

# 学習 + 検証
uv run python src/train.py -c ... --validate

# 再開
uv run python src/train.py -c ... --resume outputs/.../checkpoints/last.ckpt

# 検証のみ
uv run python src/validate.py -c ... --run-id <RUN_ID>
```

## 出力

```text
outputs/{exp_name}/{run_id}/
├── checkpoints/
│   ├── best.ckpt
│   └── last.ckpt
├── config.yaml
└── validation/
    ├── predictions.csv
    ├── metrics.csv
    ├── confusion_matrix.png
    └── class_metrics.csv
```

## 検証メトリクス

- Top-1/Top-5 Accuracy
- Precision/Recall/F1（Macro, Weighted）
- クラスごとのメトリクス

## 依存関係

- PyTorch Lightning ~2.6.0
- timm >=1.0.22
- albumentations >=2.0.8
- mlflow >=3.8.1
- pydantic >=2.0.0
