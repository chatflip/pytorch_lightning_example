# 🍽️ Classification - PyTorch Lightning 画像分類システム

YAML継承構造による設定管理を採用した、汎用的な画像分類学習システム。

## ✨ 主な特徴

- **YAML継承構造**: `__base__`による設定継承で実験設定を効率的に管理
- **timmモデル対応**: 豊富な事前学習済みモデルを利用可能
- **albumentations**: 高速で柔軟なデータオーギュメンテーション
- **PyTorch Lightning**: 分散学習、混合精度学習をサポート
- **MLFlow**: 実験追跡とメトリクスのロギング

## 🚀 クイックスタート

```bash
# インストール
make install

# データセット準備
uv run python tools/datasets/prepare_food101.py

# 学習
uv run python src/train.py -c config/experiments/food101_efficientnet_b0.yaml

# 学習 + 検証
uv run python src/train.py -c config/experiments/food101_efficientnet_b0.yaml --validate

# 検証のみ
uv run python src/validate.py \
  -c config/experiments/food101_efficientnet_b0.yaml --run-id <RUN_ID>

# MLFlow UI
bash scripts/mlflow_ui.sh
```

## 📈 出力構造

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

## 📚 ドキュメント

- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) - 設定ファイルガイド
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) - 開発者向けガイド
- [docs/SPECIFICATION.md](docs/SPECIFICATION.md) - 技術仕様書
