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
├── checkpoint_epoch_0200/
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

評価対象のcheckpointは`evaluate_stage5.sh`の`CHECKPOINT_NAMES`で決まり、既定は
`checkpoint_epoch_0030.pt`、`checkpoint_epoch_0100.pt`、`checkpoint_epoch_0150.pt`、
`checkpoint_epoch_0200.pt`、`best.pt`です。ディレクトリ名は`.pt`を除いたものになり、
`last.pt`を評価した場合は`last/`が同じ構造で作られます。epoch数の短いrunなど、
列挙したcheckpointが1つでも存在しない場合は起動時に停止するため、
`CHECKPOINT_NAMES`を実在するものだけに絞って指定します。
(`STRICT_CHECKPOINT`はstate_dictのstrict loadの可否であり、別の設定です。)

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
├── ply/
│   ├── train_sanity/
│   └── validation/
└── prediction_frames/
```

`prediction_frames/`は`evaluate_stage5.sh`では作られません。後述の
`export_stage5_prediction_frames.sh`を実行した場合だけ追加されます。

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

## `prediction_frames/`

S5-15で追加した、予測とGTを元のlocal crop frame画像に重ねた可視化です。PLYの点だけでは
false positiveが解剖構造なのか器具なのかartifactなのか判断できないため、frame画像上で
確認するためのものです。既存の評価結果だけを読み、学習も推論もCUDAも使いません。

```text
<checkpoint>/prediction_frames/
├── train_sanity/
│   ├── <video>_frame_summary.csv
│   └── <video>/
│       ├── frame_00012_fp0431.png
│       └── ...
├── validation/
│   └── ...
├── summary.csv
├── manifest.json
└── DO_NOT_SHARE.txt
```

- `<video>/*.png`: local crop grayscale frameに診断色を重ねた画像。ファイル名は
  `frame_<frame_order>_fp<false positive数>.png`
- `<video>_frame_summary.csv`: frameごとのカテゴリ別点数、選択されたか、選択理由、PNG名
- `summary.csv`: 動画ごとのカテゴリ別合計と描画frame数
- `manifest.json`: 入力(評価ディレクトリ、教師H5、中間pseudo3d H5、`predictions/*.npz`)、
  描画設定、frame選択設定、可視化できなかった動画とその理由

診断色は`ply/`の`diagnostic.ply`と同じ6分類
(`true_positive`、`false_positive`、`false_negative`、`true_negative`、
`ignore_predicted_positive`、`ignore_other`)で、同一の関数から色を取ります。
`true_negative`と`ignore_other`は画面を埋めてしまうため既定では描画しません。

frameはfalse positive数の降順、同数なら`frame_order`の昇順で選ばれ、既定は動画ごとに
上位10枚です。全frame出力も可能ですが既定ではありません。

`pixel_xy`の座標系である中間pseudo3d H5の`local_encoder_images`を参照するため、
中間H5やその寸法が解決できない動画は、他動画の寸法を流用せずに
「可視化できなかった動画」として記録されます。点数の不一致、`pixel_xy`の範囲外、
画像枚数を超える`frame_order`は、skipせずに実行を停止します。

生成は次の通りです。まず`DRY_RUN=1`で入力解決とCSV/manifestだけを確認できます。

```bash
DRY_RUN=1 bash /mnt/data/3d_projects/models/Stage5/export_stage5_prediction_frames.sh
bash /mnt/data/3d_projects/models/Stage5/export_stage5_prediction_frames.sh
```

**このディレクトリはDO_NOT_SHAREです。** 患者frameそのものが描画され、ファイル名・CSV・
manifestに実動画名と絶対パスが残ります。共有する場合は既存の`video_alias`規約で
匿名化し直す必要があります。

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

以下は点群形状、点単位予測、元ラベル、モデル重み、識別子対応、患者frame画像を含むため、
Metrics解析の目的では共有しません。

```text
*.h5
*.ply
predictions/**/*.npz
*.pt
*.pth
reference_ply/
prediction_frames/
anonymized_metrics_private_DO_NOT_SHARE/
```

## S5-16 Step 0: split・封印成果物の3領域（2026-09-22）

B′分割の成果物は用途とアクセス権の異なる3領域へ分ける。1つのディレクトリにまとめない。
「通常処理の一時ファイル」と「封印対象の統計」を同じ場所に置くと、権限も後始末も区別できなくなるため。

```text
A: /mnt/data/3d_projects/stage5_private_work/            # 通常処理用private作業領域
   stage5_fixed_list_manifest.XXXXXX                     # train_core/validationのパス一覧（一時）
   teacher_preflight_log.json                            # preflightの観測値・期待値（診断用）

B: /mnt/data/3d_projects/stage5_splits/s5_16_step0_bprime/
   train_core_144.txt                                    # 非封印
   validation_18.txt                                     # 旧リストの写し。内容・順序とも不変
   s5_16_bprime.json                                     # split manifest（hash・件数・仕様）
   sealed_registry.json                                  # 封印集合の正本
   stratification_feasibility_SHARE.json                 # FL周辺度数のみ
   teacher_preflight_expected_162.json                   # 162件用の期待値。DO_NOT_SHARE

C: .../s5_16_step0_bprime/sealed/                        # mode 0700 / ファイル0600
   internal_test_18.txt
   stratification_DO_NOT_SHARE.json                      # 候補159件のGT統計・FL値・群
   allocation_DO_NOT_SHARE.json                          # セル別配分
   bootstrap_state.json                                  # 入力hash（再開照合用）
```

- **A**: 通常のtrain_core／validation処理の一時ファイルとログ。実H5パスを含むため共有しない。
  `train_stage5.sh`の`LIST_MANIFEST`は`/tmp`ではなくここへ作り、既存の`trap ... EXIT`で
  正常終了・異常終了のいずれでも削除する。internal_test情報は入らないため、封印領域と同じ場所にしない。
- **B**: 分割の成果物。共有してよいのは件数・hash・契約検査の成否まで。
  `teacher_preflight_expected_162.json`は共有しない（既にgitにある180件の定数との差から、
  封印18動画のQC統計が得られるため）。
- **C**: internal_test情報を含む成果物。S5-20cまで封印する。
  中間生成物は最初からここへ書き、`/tmp`や作業ディレクトリを経由して移動しない。

封印の担保範囲と限界は[Step 0報告書](s5-16/stage5_step0_report_to_policy_chat.md)10.5・11.4を参照する。
権限値の設定とsealedフラグの保存だけでは封印は成立せず、コードguard・出力境界・運用の3層で構成する。
元H5は移動・改名・権限変更をしないため、ファイルシステム上はinternal_testの元データへ到達可能である。

### 共有時の境界（S5-16 Step 0の追加分）

上記の共有禁止リストに次を加える。

```text
sealed/                                   # 領域C全体
teacher_preflight_expected_*.json         # 集合依存の期待値・観測値
stage5_private_work/                      # 領域A全体
```
