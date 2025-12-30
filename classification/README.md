# 🍽️ Classification - PyTorch Lightning 画像分類システム

YAML継承構造による設定管理を採用した、汎用的な画像分類学習システムです。

## ✨ 主な特徴

- **YAML継承構造**: DEIMv2スタイルの`__base__`による設定継承で、実験設定を効率的に管理
- **timmモデル対応**: 豊富な事前学習済みモデルをすぐに利用可能
- **albumentations**: 高速で柔軟なデータオーギュメンテーション
- **PyTorch Lightning**: 分散学習、混合精度学習を簡単に設定
- **MLFlow/TensorBoard**: 実験追跡とメトリクスのロギング
- **uvパッケージマネージャ**: 高速な依存関係管理

## 📦 技術スタック

| コンポーネント | ライブラリ |
|----------------|------------|
| 深層学習フレームワーク | PyTorch |
| 学習フレームワーク | PyTorch Lightning |
| モデル | timm |
| オーギュメンテーション | albumentations |
| 設定管理 | YAML + カスタムローダー |
| ロギング | MLFlow / TensorBoard |
| パッケージ管理 | uv |

## 🚀 クイックスタート

### 前提条件

- Python 3.12+
- CUDA対応GPU（推奨）
- [uv](https://docs.astral.sh/uv/) パッケージマネージャ

### インストール

```bash
make install
```

### データセットの準備

```bash
# Food101データセットのダウンロードと準備
uv run python tools/datasets/prepare_food101.py
```

### 学習の実行

```bash
# 設定ファイルを指定して学習
uv run python src/train.py -c config/experiments/food101_efficientnet_b0.yaml
```

### MLFlow UI

```bash
uv run mlflow ui --port 5000
# ブラウザで http://localhost:5000 を開く
```

## 📈 出力

学習後の出力は以下の構造で保存されます：

```text
outputs/{exp_name}/
├── checkpoints/
│   ├── epoch=XX-val_loss=X.XX.ckpt   # ベストモデル
│   └── last.ckpt                      # 最終モデル
└── config.yaml                        # 使用した設定
```

## 📚 ドキュメント

| ドキュメント | 説明 |
|-------------|------|
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | 設定ファイルの詳細ガイド |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | 開発者向けガイド |
| [docs/SPECIFICATION.md](docs/SPECIFICATION.md) | 技術仕様書 |
