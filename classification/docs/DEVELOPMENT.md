# 🛠️ 開発者ガイド

## セットアップ

```bash
make install        # 依存関係インストール
make format         # コードフォーマット
make lint           # リント
uv run pytest       # テスト
```

## ディレクトリ構成

```text
src/
├── train.py              # トレーニング
├── validate.py           # 検証
├── config/
│   ├── loader.py         # YAML継承ローダー
│   └── schema.py         # Pydanticバリデーション
├── data/
│   ├── dataset.py        # ImageFolderDataset
│   └── datamodule.py     # LightningDataModule
├── models/
│   └── classifier.py     # LightningModule
└── builders/
    ├── augmentation.py   # albumentationsビルダー
    ├── optimizer.py      # オプティマイザービルダー
    └── logger.py         # ロガービルダー
```

## データセット追加

1. `tools/datasets/` に準備スクリプトを作成
2. ImageFolder形式で配置:

```text
data/datasets/my_dataset/
├── train/
│   ├── class_a/
│   └── class_b/
└── val/
    ├── class_a/
    └── class_b/
```

1. 実験設定を作成:

```yaml
__base__:
  - ../_base_/default.yaml
  - ../_base_/model/resnet50.yaml
  - ../_base_/augmentation/basic.yaml
  - ../_base_/optimizer/adamw.yaml
  - ../_base_/trainer/default.yaml

exp_name: "my_dataset_resnet50"
data:
  dataset_root: "data/datasets/my_dataset"
  num_classes: 2
  batch_size: 64
```

## MLFlow UI

```bash
bash scripts/mlflow_ui.sh
# http://localhost:5000
```
