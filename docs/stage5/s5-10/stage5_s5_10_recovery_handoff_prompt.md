# Stage 5 S5-10 実装復旧・継続用引き継ぎプロンプト

作成日: 2026-09-11

このworkspaceでは、動画由来pseudo-3D point cloudから大腿骨をpoint-wise segmentationする
Stage 5を開発しています。以前のS5-10実装チャットは環境修復中に失われた可能性がありますが、
実装ファイル自体はworkspaceに残っています。既存実装を作り直さず、監査、必要な補強、実行検証、
結果記録を引き継いでください。

## 1. プロジェクト概要

Stage 5の処理は概ね次の構成です。

```text
Stage 4 annotated pseudo-3D H5
  -> frame_order基準のoverlap window Dataset
  -> official OpenPoints PointNeXt-S wrapper
  -> point-wise binary segmentation
  -> overlap prediction aggregation
  -> prediction/prob_femur, prediction/pred_label
```

- 現行model: official OpenPoints PointNeXt-Sを組み込んだStage 5 wrapper
- 初期重み: S3DIS PointNeXt-S checkpointからshape一致111/114 keyを転移
- 入力features: `intensity,confidence`
- window: frame size 16、stride 8、tailあり
- PointNeXtへのpadding: なし
- physical batch size: 1 window
- gradient accumulation: 8 windows
- loss: auto class weight付きCrossEntropyLoss、label smoothing 0.0
- production aggregation: mean probabilityを維持中
- 現在は200 epoch本学習へ進まず、normalization/BatchNorm問題を診断中

Stage 4 teacherはv6を使用しています。

```text
global_local_l75_w31_c12_area15_bboxrank_v6_
cvat_authoritative_xml_invalidation_v1
```

基準runは次です。

```text
/mnt/data/3d_projects/stage5_runs/260908/
pointnext_s_EX260908_260711_w16_s8_bboxrankv6_cvatxmlinv_glocal_
ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad/
```

- train: 163 files / 729 windows
- validation: 18 files / 79 windows
- 主診断checkpoint: `best.pt`、epoch 4
- 固定評価: train sanity 3動画 + validation 18動画

## 2. 最初に読む文書

次の順で確認してください。

1. `docs/stage5/stage5_revision_management_record.md`
   - Stage 5全体の進捗と意思決定の正本
   - 特にS5-09とS5-10
2. `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
   - 詳細な検証結果
   - 特に9.4節、9.5節の実装事項A/B
3. `docs/stage5/s5-08-09/stage5_overlap_aggregation_implementation_policy.md`
   - overlap aggregation検証の方針
4. `docs/stage5/s5-08-09/stage5_overlap_aggregation_handoff_prompt.md`
   - checkerとBatchNorm parityの詳細仕様
5. `docs/stage5/FILES.md`
   - Stage 5コード・生成物の配置
6. `docs/stage5/s5-10/stage5_message_to_implement_chat.md`
   - 失われた実装チャットへ渡した元のS5-10実装依頼

文書間に矛盾がある場合、現在の方針は
`docs/stage5/stage5_revision_management_record.md`を正としてください。ただし同文書には後述する
ステータス表記の未同期があります。

## 3. S5-10の背景と目的

S5-09 BatchNorm mode parityでは、同一windowでも`eval()`と`train()`で大きな差が確認されました。

- validation全valid点のclass disagreement: 約16.2%
- validation GT positive点のclass disagreement: 約38.2%
- aggregated recall: 3.87% -> 33.12%
- aggregated F1: 0.0341 -> 0.0541
- TP0動画数: 10/18 -> 0/18
- FP: 85,785 -> 757,945

normalization modeが重大な交絡要因であることは支持されています。しかし`train()`推論は評価window
自身のbatch statisticsを使うため、保存済みrunning statisticsの問題とtest-time adaptation効果を
分離できません。

S5-10ではmodel parameterを変更せず、training splitだけでBatchNorm running statisticsを再計算し、
そのmodelを通常の`eval()`で評価します。これは診断であり、production変更や再学習ではありません。

## 4. workspaceに残っている実装

次の新規ファイルが存在します。

```text
Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.py
Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.sh
```

確認時点の規模:

- Python: 921 lines
- bash: 152 lines
- 最終更新: 2026-09-10

既存実装は概ね次を実装済みです。

1. `best.pt`からoriginal/recalibratedの独立したmodel instanceを構築する。
2. 全parameterをfreezeする。
3. model全体を`eval()`にした後、BatchNorm moduleだけを`train()`へ切り替える。
4. BN running statisticsをresetし、`momentum=None`へ設定する。
5. `${RUN_DIR}/train_files.txt`を固定順序で読み、全windowを1回forwardする。
6. augmentation、backward、optimizer updateを行わず、`torch.inference_mode()`を使う。
7. calibration前後のparameter hashと非BN buffer hashの一致をassertする。
8. calibration train listとvalidation listの非重複を検査する。
9. original evalとrecalibrated evalを同じwindow・seedで比較する。
10. original evalを既存`h5_metrics.csv`と照合するmean baseline parity gateを持つ。
11. TP/FP/TN/FN、precision、recall、F1、IoU、FPR/FNR、predicted positive率、
    TP0動画数、video-level mean/medianを集計する。
12. probability差とclass disagreementをGT区分別に集計する。
13. BN module別running buffer差を出力する。
14. recalibrated checkpointを元checkpointとは別に`stage5_debug/`へ保存する。
15. share/private出力を分離し、匿名化self-checkを行う。
16. cumulative update、state不変性、validation混入拒否、determinismのsynthetic testを持つ。

再較正checkpointは`model_state_dict`キーで保存されます。既存の
`stage5/models/model_factory.py::_extract_state_dict`は`model`、`model_state_dict`、`state_dict`を
受理するため、保存形式は既存ローダーと互換です。

## 5. 現在の検証進捗

確認済み:

```text
python3 -m py_compile Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.py
bash -n Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.sh
git diff --check
```

上記はすべて成功しています。

未実行:

- synthetic self-test
- 実checkpoint/modelを使うsmoke run
- 163 train動画 / 729 windowsのfull recalibration
- train sanity 3件 + validation 18件のfull evaluation
- 匿名化済み結果bundleの確認
- 評価レポートへの結果追記
- 方針管理チャットへ返す完了報告

本引き継ぎ作成当時、`.tmp/`にはS5-10 recalibrationの結果bundleは存在しませんでした。現在の保存場所は`research/stage5/s5-10/`です。存在するBatchNorm metricsはS5-09の
mode parity結果であり、S5-10の結果と混同しないでください。

## 6. Gitと文書の状態

S5-10 checker 2ファイルは確認時点でuntrackedです。実装が失われないよう注意してください。
ただし、このworkspaceには本件以外の変更も多数存在します。

- 関係のない変更をrevertしない。
- dirty worktreeを前提に、対象ファイルだけを慎重に扱う。
- commitを求められていない段階では勝手にcommitしない。
- 既存のユーザー変更と生成物を削除しない。

`../stage5_revision_management_record.md`には状態の未同期があります。

- 冒頭: 「BatchNorm recalibration診断の実装前」
- タイムラインS5-10: `planned`
- S5-10本文: 「コード実装完了、実行未着手」

本文が現在の実態です。動的検証が終わるまでは、S5-10を完了扱いにしないでください。

## 7. 実行前に監査・補強する事項

まず既存コードを読み、重複実装を避けてください。その上で、次の補強が妥当か確認し、問題が
なければ実装してください。

1. 全BN moduleについて、recalibration後の
   `num_batches_tracked == calibration_processed_windows`をassertする。
2. 全BN moduleの`running_mean`と`running_var`がfiniteであることをassertする。
3. 全BN moduleの`running_var`に負値がないことをassertする。
4. calibration/validationの重複検査では、可能なら`Path.resolve()`後のpathでも比較する。
5. 上記assertに対応するsynthetic testまたは明確なfail-fast検証を追加する。

補足:

- 現在の実装はこれらの値をCSVへ記録していますが、full runの失敗条件としては一部未実装です。
- 163 files / 729 windowsは対象runの期待値です。override可能性を残す場合でも、実行結果には実数を
  必ず記録してください。
- `train_files.txt`のtrain sanity 3動画はcalibrationに含まれて構いません。
- validation 18動画はcalibrationに絶対に含めないでください。
- calibration処理で共通H5 loaderがlabelを読み込むこと自体は許容しますが、labelを更新式、選別、
  window skipに利用してはいけません。

## 8. 実行環境

実データとGPU検証はユーザー環境で行います。bashでは既存の設定を維持してください。

```bash
PYTHON="/home/kodaira/anaconda3/envs/dualtrack311/bin/python"
export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="12.0"

TORCH_LIB="$(${PYTHON} - <<'PY'
import torch
from pathlib import Path
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
```

PointNeXt/OpenPointsの`PYTHONPATH`も既存bashどおり設定してください。

このworkspace側のsystem Pythonには`numpy`、`torch`、`h5py`、`tqdm`がありません。そのため、
こちらで`py_compile`は実行できますが、synthetic self-testを含むruntime検証は上記conda環境で
行う必要があります。

## 9. 推奨する継続手順

### Step 1: 既存実装の再監査

- 仕様と実装の対応を確認する。
- 既存checkerから再利用している関数の契約を確認する。
- productionの`train_stage5.py`、`infer_stage5.py`、`evaluate_stage5.py`の既定挙動を変更して
  いないことを確認する。
- 第7節の補強事項を必要最小限で反映する。

### Step 2: 静的検証

```bash
python3 -m py_compile \
  Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.py

bash -n \
  Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.sh

git diff --check
```

### Step 3: synthetic self-test

ユーザー環境で実行します。

```bash
SELF_TEST=1 \
bash /mnt/data/3d_projects/models/Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.sh
```

期待ログ:

```text
Stage5 BatchNorm recalibration self-test passed.
```

### Step 4: train sanity smoke run

full calibrationは全163 train動画で行い、評価だけをtrain sanity 3動画に限定します。
既存bashのコメントと引数を確認し、validationを空listへ差し替えて実行してください。

```bash
EMPTY_VAL_LIST=/tmp/stage5_empty_val_list.txt
: > "${EMPTY_VAL_LIST}"

VAL_LIST="${EMPTY_VAL_LIST}" \
bash /mnt/data/3d_projects/models/Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.sh
```

smoke runでもcalibrationを全train windowへ実施するため、単なる数秒のテストではありません。

### Step 5: full run

```bash
bash /mnt/data/3d_projects/models/Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.sh
```

既定pathは対象v6 5 epoch pilotへ設定済みですが、実行前にbash冒頭のpathと実在ファイルを確認して
ください。

期待する主要入力:

```text
${RUN_DIR}/best.pt
${RUN_DIR}/train_files.txt
${RUN_DIR}/val_files.txt
${EVALUATION_DIR}/evaluation_data/selected_train_files.txt
${EVALUATION_DIR}/evaluation_data/summary.json
${EVALUATION_DIR}/best/h5_metrics.csv
```

期待する出力root:

```text
/mnt/data/3d_projects/stage5_debug/batchnorm_recalibration/
```

share側には匿名化metrics、private側には元ID/path mappingとrecalibrated diagnostic checkpointを
保存します。

## 10. 合格条件

最低限、次を満たした場合にchecker実行完了としてください。

1. synthetic self-testが通る。
2. calibration files/windowsが対象runでは163/729になる。
3. parameter hashが完全一致する。
4. 非BN buffer hashが完全一致する。
5. BN running buffersがfiniteかつvariance非負である。
6. 全BNの`num_batches_tracked`が想定forward回数と一致する。
7. calibrationとvalidationに重複がない。
8. original evalのTP/FP/TN/FNが既存`h5_metrics.csv`と完全一致する。
9. train sanity 3件とvalidation 18件の評価が完走する。
10. share bundleの匿名化self-checkが通る。

checkerの完走と、recalibration仮説の支持・棄却は別に判断してください。

## 11. 結果の解釈

- F1/IoU、recall、TP0動画数が改善し、FP/FPR増加が許容範囲なら、recalibrationを有力候補とする。
- recallだけ増え、FPもS5-09 train-mode同様に大幅増加するなら、性能改善ではなくprobability
  calibration shiftとして扱う。
- 変化が小さいなら、stale running statisticsを主要因から下げ、per-window statistics依存または
  normalization方式そのものを次候補とする。

checker側でproduction採否を決めないでください。recalibrated checkpointをproduction推論へ自動的に
接続せず、実装事項Aのaggregation方式も変更しないでください。

## 12. 実行後の文書更新

full run完了後、次を更新してください。

1. `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
   - S5-10の詳細条件、出力、主要metrics、仮説判断
2. `docs/stage5/stage5_revision_management_record.md`
   - 冒頭の現在段階
   - タイムラインのS5-10状態
   - S5-10本文の実施日、実装、結果、判断
3. `docs/stage5/FILES.md`
   - 既存記載を確認し、実際の出力構造と一致させる

production decisionやDecision recordは、方針管理チャットの判断前に確定しないでください。

## 13. 方針管理チャットへの完了報告

最後に、次を含む共有用Markdownを`docs/stage5/s5-10/`へ作成してください。

- 変更ファイル
- 実行したテストと成否
- full runの入力条件
- share/private出力先
- calibration file/window数
- BN module数とbuffer健全性
- parameter/non-BN buffer不変性
- original baseline parity
- original vs recalibratedのtrain sanity/validation主要metrics
- TP0動画数とvideo-level median F1/IoU
- S5-09 train-mode結果との関係
- 支持された仮説、棄却された仮説、未確定事項
- productionへ未反映であること
- 方針管理チャットで判断が必要な事項

実装や検証で不明点が見つかった場合は、事実、現在の実装、選択肢、推奨案を分けて提示して
ください。仕様上の重大な不明点がなければ、既存実装の監査から必要な補強、静的検証まで進め、
GPU/実H5が必要なコマンドをユーザーへ提示してください。
