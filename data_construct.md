# Stage 5評価出力構成

出力rootは通常、次の場所です。

```text
/mnt/data/3d_projects/stage5_evaluations/<EX_DATE>/<EXPERIMENT_NAME>/
```

## 全体構成

```text
<output_root>/
├── evaluation_data/
├── reference_ply/
├── checkpoint_epoch_0030/
├── checkpoint_epoch_0100/
├── checkpoint_epoch_0150/
├── best/
├── anonymized_metrics_SHARE_THIS/
├── anonymized_metrics_private_DO_NOT_SHARE/
├── checkpoint_summary.csv
├── h5_metrics_all_checkpoints.csv
├── window_metrics_all_checkpoints.csv
└── comparison_manifest.json
```

## `evaluation_data/`

評価対象の選択結果と参照PLYの対応情報です。

```text
evaluation_data/
├── selected_train_files.txt
├── validation_files.txt
├── reference_ply_manifest.csv
└── summary.json
```

- `selected_train_files.txt`: 固定train動画`20250626_124212_7300`とseed固定のランダム2件
- `validation_files.txt`: validation全H5
- `reference_ply_manifest.csv`: H5、コピー元PLY、コピー先PLY、GT positive点数の対応表
- `summary.json`: 選択seed、参照PLYディレクトリ、対象ファイル数など

## `reference_ply/`

checkpointに依存しないStage4教師データの可視化です。

```text
reference_ply/
├── train_sanity/
│   ├── pointcloud_foreground/
│   ├── pointcloud_annotated_foreground/
│   └── ground_truth_positive/
└── validation/
    ├── pointcloud_foreground/
    ├── pointcloud_annotated_foreground/
    └── ground_truth_positive/
```

- `pointcloud_foreground/`: Stage4のraw grayscale PLY
- `pointcloud_annotated_foreground/`: Stage4のannotation/source色付きPLY
- `ground_truth_positive/`: `valid_mask=True`かつ`point_label=1`の点だけを抽出したPLY

`train_sanity`には3動画、`validation`にはvalidation全動画が入ります。

## 各checkpointディレクトリ

`checkpoint_epoch_0030/`、`checkpoint_epoch_0100/`、`checkpoint_epoch_0150/`、`best/`は同じ構造です。

```text
<checkpoint>/
├── selected_train_files.txt
├── validation_files.txt
├── summary.json
├── h5_metrics.csv
├── window_metrics.csv
├── diagnostic_ply_legend.json
├── predictions/
│   ├── train_sanity/
│   └── validation/
└── ply/
    ├── train_sanity/
    └── validation/
```

`predictions/`には動画ごとの圧縮NPZがあります。

```text
predictions/<split>/<video>.npz
```

内容は以下です。

- `point_indices`: 元H5内の点index
- `prob_femur`: overlap集約後のpositive確率
- `pred_label`: overlap集約後の予測label
- `vote_count`: 各点が含まれたwindow数

`ply/`には動画ごとに3種類あります。

```text
<video>_probability.ply
<video>_diagnostic.ply
<video>_predicted_positive.ply
```

- `probability.ply`: `prob_femur`を青から赤の連続色で表示
- `diagnostic.ply`: TP、FP、FN、TN、ignore上のpositive予測などを分類色で表示
- `predicted_positive.ply`: `pred_label=1`の点だけを抽出。ignore領域上のpositive予測も含みます

診断色の対応は`diagnostic_ply_legend.json`に記録されます。

## Metrics

- `h5_metrics.csv`: overlap集約後の元点index単位・H5単位評価
- `window_metrics.csv`: aggregation前の各frame window単位評価
- `summary.json`: train sanityとvalidationそれぞれの全点集約評価

主な指標はprecision、recall、F1、femur IoU、FP/FN、valid点数、GTクラス比、ignore領域のpositive予測率・平均確率です。

## Checkpoint横断ファイル

- `checkpoint_summary.csv`: checkpointごとのtrain sanity/validation集約比較
- `h5_metrics_all_checkpoints.csv`: 全checkpointのH5単位metricsを結合
- `window_metrics_all_checkpoints.csv`: 全checkpointのwindow単位metricsを結合
- `comparison_manifest.json`: 比較したcheckpointと共通評価ファイル一覧

目視では、同一動画について`reference_ply/.../ground_truth_positive`と各checkpointの`ply/.../predicted_positive.ply`を並べると最も比較しやすくなります。

## 匿名化Metrics

Metricsを外部共有する場合は、出力root直下の元CSV/JSONではなく、
`anonymized_metrics_SHARE_THIS/`だけを共有します。

匿名化処理では、元の動画名、撮影日時形式のID、H5絶対パス、checkpoint絶対パスを
共有用ファイルから除去します。匿名IDは用途が分かるように以下の形式で付与します。

```text
train_sanity_fixed_001
train_sanity_random_001
train_sanity_random_002
validation_001
validation_002
...
```

- `train_sanity_fixed_NNN`: 固定選択した学習データのsanity check
- `train_sanity_random_NNN`: seed固定で追加選択した学習データのsanity check
- `validation_NNN`: 学習に使用していないvalidationデータの精度評価

### 共有用ディレクトリ

```text
anonymized_metrics_SHARE_THIS/
├── train_sanity_evaluation/
│   ├── train_sanity_h5_metrics.csv
│   ├── train_sanity_window_metrics.csv
│   └── train_sanity_checkpoint_summary.csv
├── validation_accuracy/
│   ├── validation_h5_metrics.csv
│   ├── validation_window_metrics.csv
│   └── validation_checkpoint_summary.csv
├── training_history/
│   ├── training_config_anonymized.json
│   └── training_epoch_metrics.jsonl
├── anonymized_metrics_manifest.json
├── anonymization_report.json
└── SHARE_THIS_DIRECTORY.txt
```

#### `train_sanity_evaluation/`

選択した学習データに対する推論結果です。モデルが訓練データを記憶・適合できているかを
確認するためのものであり、汎化性能としては扱いません。

- `train_sanity_h5_metrics.csv`: H5単位のtrain sanity評価
- `train_sanity_window_metrics.csv`: frame window単位のtrain sanity評価
- `train_sanity_checkpoint_summary.csv`: checkpoint単位のtrain sanity集約値

#### `validation_accuracy/`

hold-outされたvalidation全データに対する推論精度です。checkpoint選択や汎化性能の
比較にはこちらを使用します。

- `validation_h5_metrics.csv`: H5単位のvalidation評価
- `validation_window_metrics.csv`: frame window単位のvalidation評価
- `validation_checkpoint_summary.csv`: checkpoint単位のvalidation集約値

#### `training_history/`

対応する学習runの履歴です。

- `training_config_anonymized.json`: パス情報を`REDACTED_PATH`へ置換した学習設定
- `training_epoch_metrics.jsonl`: epochごとのtrain/validation metrics

#### Manifestと検査結果

- `anonymized_metrics_manifest.json`: checkpoint、匿名sample ID、共有ファイル一覧
- `anonymization_report.json`: 匿名化件数とprivacy check結果
- `SHARE_THIS_DIRECTORY.txt`: このディレクトリが共有対象であることを示す説明

`anonymization_report.json`では、少なくとも以下を確認します。

- 元動画IDが共有用ファイルに残っていない
- 撮影日時形式の動画IDが残っていない
- `/mnt/...`、`/home/...`などの絶対ホストパスが残っていない
- train sanityとvalidationが別々のファイルに分離されている

### 非共有ディレクトリ

```text
anonymized_metrics_private_DO_NOT_SHARE/
├── video_id_map_DO_NOT_SHARE.csv
└── DO_NOT_SHARE.txt
```

`video_id_map_DO_NOT_SHARE.csv`には匿名IDと元の動画名・H5パスの対応が残ります。
ローカルでPLYやH5と照合するためのファイルであり、外部共有してはいけません。

## 匿名化Metricsの生成

既存の評価結果から匿名版だけを生成する場合は、以下を実行します。PointNeXtの推論や
PLY生成は再実行されません。

```bash
bash /mnt/data/3d_projects/models/Stage5/export_anonymized_stage5_metrics.sh
```

標準とは異なる評価runを対象にする場合は、評価rootと学習runを明示します。

```bash
EVALUATION_OUTPUT_ROOT="/mnt/data/3d_projects/stage5_evaluations/<EX_DATE>/<EXPERIMENT_NAME>" \
RUN_DIR="/mnt/data/3d_projects/stage5_runs/<EX_DATE>/<EXPERIMENT_NAME>" \
bash /mnt/data/3d_projects/models/Stage5/export_anonymized_stage5_metrics.sh
```

`evaluate_stage5.sh`では、全checkpointの評価・集約後に匿名化Metricsも自動生成します。
自動生成を無効化する場合は、`EXPORT_ANONYMIZED_METRICS=0`を指定します。

```bash
EXPORT_ANONYMIZED_METRICS=0 \
bash /mnt/data/3d_projects/models/Stage5/evaluate_stage5.sh
```

## 共有時の境界

Metrics解析のために共有してよいのは、原則として以下だけです。

```text
anonymized_metrics_SHARE_THIS/
```

以下は点群形状、点単位予測、元ラベル、モデル重み、識別子対応を含むため、Metrics解析の
目的では共有しません。

```text
*.h5
*.ply
predictions/**/*.npz
*.pt
*.pth
reference_ply/
anonymized_metrics_private_DO_NOT_SHARE/
```
