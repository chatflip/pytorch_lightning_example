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

## 📁 ディレクトリ構成

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

## 🚀 インストール

### 前提条件

- Python 3.12+
- CUDA対応GPU（推奨）
- [uv](https://docs.astral.sh/uv/) パッケージマネージャ

### セットアップ

```bash
# 依存関係のインストール
make install
```

## 📊 使い方

### 1. データセットの準備

Food101データセットを使用する場合：

```bash
# データセットのダウンロードと準備
uv run python tools/datasets/prepare_food101.py
```

### 1. 学習の実行

```bash
# 設定ファイルを指定して学習
uv run python src/train.py -c config/experiments/food101_efficientnet_b0.yaml
```

### 1. MLFlow UI

```bash
# MLFlow UIを起動
uv run mlflow ui --port 5000
# ブラウザで http://localhost:5000 を開く
```

## ⚙️ 設定ファイル

### YAML継承の仕組み

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

### 主な設定項目

| 設定カテゴリ | 説明 |
|--------------|------|
| `model` | timmのモデル名、事前学習、ドロップアウト率 |
| `data` | データセットパス、バッチサイズ、ワーカー数 |
| `augmentation` | train/valのオーギュメンテーションパイプライン |
| `optimizer` | オプティマイザータイプ、学習率、重み減衰 |
| `scheduler` | 学習率スケジューラー設定 |
| `trainer` | エポック数、精度、デバイス設定 |
| `logger` | ロガータイプ、実験名 |

詳細は [docs/SPECIFICATION.md](docs/SPECIFICATION.md) を参照してください。

## 📈 出力

学習後の出力は以下の構造で保存されます：

```
outputs/{exp_name}/
├── checkpoints/
│   ├── epoch=XX-val_loss=X.XX.ckpt   # ベストモデル
│   └── last.ckpt                      # 最終モデル
└── config.yaml                        # 使用した設定
```

## 📝 開発

```bash
# コードフォーマット
make format

# リント
make lint
```

## 📄 ライセンス

MIT License
