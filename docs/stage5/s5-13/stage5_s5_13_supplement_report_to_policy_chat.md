# Stage 5 S5-13補足: Threshold-free診断と中間Class Weight確認 報告書

作成日: 2026-09-14
作成元: Stage 5実装チャット
状態: **評価完了**。補足検証1・2とも完了。**W-Cは不採用、W-Aを維持**という結論。新規5 epoch pilotは
1/2回のみ使用（2回目は不要と判断）。production反映は方針管理チャットの判断待ち

本書は`docs/stage5/s5-13/stage5_s5_13_supplement_implementation_handoff.md`（実装引き継ぎ、正本）を受けての
理解・現状・実装方針の記録として作成し、以後、検証が進むごとに本書へ結果を追記して方針管理チャット
への完了報告として使用する。実装引き継ぎ13章が別途指定する
`stage5_s5_13_supplement_report_to_policy_chat.md`という完了報告ファイル名と一致させてあるため、
新しい完了報告ファイルを別途作成する必要はない。

## 1. 経緯

S5-13の必須2-arm class weight ablation（W-A: 強いauto由来weight`[0.05963856, 1.94036150]`、W-B:
弱い固定weight`[0.5, 1.5]`）は完了し、方針管理チャットが次の3判断を行った。

1. W-Bを不採用とし、W-Aを`S5-13補足`およびS5-14の暫定class weightとして採用する。
2. S5-14前に、threshold-free診断（保存済みW-A/W-B checkpoint/predictionを再利用、再学習なし）と
   中間weight W-Cの限定pilot（1回、最大2回）を「S5-13補足」として実施する。
3. S5-13の必須A/B比較は完了と認定するが、固定threshold 0.5での全negativeが識別能力の消失なのか
   calibration shiftなのかは未分離であり、補足完了後に暫定weightを確定してS5-14へ進む。

これを踏まえ、S5-13補足として上記2つの限定検証を実施する依頼を受けた。

## 2. S5-13結果の要約（前提）

teacher v7、GroupNorm 8 groups、`bbox_noncontour_ignore`、同一split・seed・初期checkpointを固定し
比較した結果、W-Bはthreshold 0.5で全点をbackgroundと予測した。

| split | arm | recall | F1 | IoU | FPR | predicted positive | TP0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train sanity | W-A | 58.24% | 0.1156 | 0.0613 | 13.42% | 88,583 | 0/3 |
| train sanity | W-B | 0.00% | 0.0000 | 0.0000 | 0.00% | 0 | 3/3 |
| validation | W-A | 44.69% | 0.0714 | 0.0370 | 11.71% | 864,518 | 1/18 |
| validation | W-B | 0.00% | 0.0000 | 0.0000 | 0.00% | 0 | 18/18 |

config parity、train/val file list、初期checkpoint、epoch数、finite性、評価対象point集合はcheckerで
一致を確認済みであり、W-Bの結果は比較条件の不一致や実装上の故障ではない。詳細は
`docs/stage5/s5-13/stage5_s5_13_report_to_policy_chat.md`、数値正本は
`docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.7節（実装事項F）。

## 3. S5-13補足の目的と方針管理チャットの回答

1. 保存済みW-A/W-Bのdiscrimination（識別能力）とthreshold/calibration shiftを分離する。
2. W-Aより弱いがW-Bほどpositive寄与を失わない中間weight W-Cで、global FP/FPRを抑えられる余地を
   一度だけ確認する。
3. 結果からS5-14で固定する暫定class weightを決める。production thresholdの選定はS5-15まで
   行わない。

W-Cの固定値:

```text
W-C class weight = [0.11764706, 1.88235294]
positive:background weight比 = 16
想定positive weighted-denominator占有率 = 約16.7%（W-A約29.0%とW-B約3.6%の中間）
```

## 4. 固定する実験条件（S5-13から変更しないもの）

teacher v7（`bboxrank_v7_cvat_authoritative_crop_quality_v1`、180 H5）、S5-12/S5-13 Run Aと同一の
train 162 / validation 18 split、`label_policy=bbox_noncontour_ignore`、GroupNorm 8 groups、
W-A/W-Bと同一のGroupNorm S3DIS部分転移checkpoint、seed 42、window 16/8/tailあり、physical batch 1、
gradient accumulation 8（point-weighted normalization）、features `intensity,confidence`、
CrossEntropyLoss、label smoothing 0.0、AdamW lr 1e-3/weight decay 1e-4、dropout 0.0、epochs 5、
`model.eval()`、mean probability aggregation、primary threshold 0.5。W-Cはこれらをすべて揃え、
class weightだけを変える。

**検証量の上限（厳守）:** 新規5 epoch pilotは原則W-Cの1回、最大でも合計2回まで。W-A/W-Bは
再学習しない。no-weight、複数seed、広いweight grid、50〜200 epoch学習は行わない。2回目を実行する
前には必要性・変更する唯一の変数・期待する判定を報告する。

## 5. 実装前監査で確認した既存コードの状況

### 5.1 既存predictionの再利用可否（確認済み・再推論不要）

W-A/W-Bの`evaluate_stage5.sh`評価は`SAVE_PREDICTIONS=1`（既定値、未上書き）で実行済みのため、
各H5ごとに`prob_femur`/`pred_label`/`vote_count`を含む予測`.npz`
（`<evaluation_output_dir>/<checkpoint>/predictions/<split>/<video>.npz`、`evaluate_stage5.py`の
`write_probability_ply`等と同じ`prediction_path`規則）がすでに保存されている。GTラベル
（`point_label`/`valid_mask`）は`h5_metrics.csv`が記録する`h5_path`から元のH5を直接読み込めば
取得できる。したがって**W-A/W-Bのthreshold-free診断はモデルの再推論・CUDA・torchが一切不要**で
あり、h5py/numpyのみで計算できる（handoff文書6.1節「可能なら既存のprediction artifactを再利用し、
不足する場合だけcheckpointから再推論する」を満たす）。

### 5.2 再利用可能な既存実装

- `check_stage5_class_weight_ablation.py`の`compare_config`/`check_resolved_weight`/
  `check_history`/`check_checkpoint_present`は、W-A/B/C間のconfig parity・resolved weight・
  history finite性の検証にそのまま再利用できる（依頼書11章「既存checkerを拡張できる場合は重複
  実装を避ける」）。
- `evaluate_stage5.py`の`compute_metrics`/`metrics_from_counts`（TP/FP/TN/FN、precision/recall/F1/
  IoU/FPR算出ロジック）は、threshold別metrics計算でも同じ定義を再利用する。
- `train_stage5.sh`のclass-weight tag生成・output collision guard・manual weight処理（S5-13で実装
  済み）はW-Cにもそのまま使え、追加のshell変更は不要と見込む。

### 5.3 未確認・要監査の項目

- W-A/W-B評価出力ディレクトリ配下の`predictions/`の実ファイル名規則（`safe_name(video_name)`と
  `h5_metrics.csv`の`video_name`列の対応）をGPU側の実ファイルで確認する必要がある。
- `evaluate_stage5.py`のnpz保存が`best`/`last`両checkpointで別ディレクトリに分かれているか
  （Step F7では`CHECKPOINT_NAMES="last.pt best.pt"`で両方評価済みのため、primary比較には
  `last`側を使う）。

## 6. 想定する実装の形（未着手）

依頼書のStep 1〜11に沿って進める想定。

1. **新規checker**: `Stage5/checks/real_h5/check_stage5_class_weight_threshold_free.py/.sh`。
   既存の`.npz`予測artifactとH5 GTを読み込み、threshold 0.5のTP/FP/TN/FNが既存S5-13評価
   （`h5_metrics.csv`）と完全一致することをfail-fast gateにした上で、AUPRC（主指標）・AUROC
   （補助指標）・GT class別score percentile・PR curve・threshold別metrics・最大F1・
   `W-AのFPR以下で達成可能な最大recall`・固定FPR（1%/5%/10%）でのrecall/precisionを、
   split aggregate・video-levelで計算する。全pointをthresholdごとに再forwardせず、
   確率を一度収集してsort/cumulative countで計算する。W-A/W-Bのconfig・point-set parity検証には
   既存`check_stage5_class_weight_ablation.py`の関数を再利用する。synthetic test
   （既知score/GTでAUPRC、curve、tie処理、empty class、all-negative predictionを確認）を
   CPU静的テストとして追加する。
2. **静的検証**: 上記checkerのsynthetic test、`py_compile`/`bash -n`/`git diff --check`。
3. **補足検証1実行（GPU側、再学習なし）**: 保存済みW-A/W-Bに対して上記checkerを実行し、
   discriminationとcalibration shiftを分離する。
4. **W-C事前確認**: `train_stage5.sh`の出力先がW-A/Bと衝突しないこと、`CLASS_WEIGHT=
   0.11764706,1.88235294`のresolved値・tagを起動ログで確認する。
5. **W-C 5 epoch pilot（GPU側、新規学習1回目）**: 既存`train_stage5.sh`をそのまま使用する
   （コード変更不要の見込み）。
6. **W-C固定評価**: 既存`evaluate_stage5.sh`でepoch 5 checkpointをtrain sanity 3動画・
   validation 18動画へ適用し、threshold 0.5指標を取得する。
7. **W-C threshold-free診断**: 手順1のcheckerをW-Cにも適用する。
8. **比較・判定**: W-A/B/Cをthreshold 0.5・threshold-free両方でaggregate・video-level比較し、
   採否基準（7節）に従い判定する。
9. **文書更新・完了報告**: `../stage5_pointnext_s_training_evaluation_report.md` 9.7節、
   `../stage5_revision_management_record.md`のS5-13補足、`../FILES.md`を更新し、本書へ結果を追記する。

## 7. 判定基準（依頼書9章の要約）

**W-Cを暫定採用候補とする条件:** threshold 0.5でW-AよりFP/FPRとpredicted positive率が低下し、
recallが大幅に低下せずTP0動画数がW-Aの1/18から増加せず、video-level median F1/IoUが悪化せず
動画別傾向も改善を支持し、AUPRC/PR curveが単なるscore shiftでなく有効なtrade-offを支持し、
train sanityとvalidationの改善方向が大きく反転しない場合。

**W-Aを維持する条件:** W-Cが全negativeまたはそれに近い状態へ崩壊する、TP0動画数またはrecallが
明確に悪化する、FP低下がprecision/F1/IoU改善につながらない、threshold-free指標とvideo-level結果が
W-Cを支持しない、結果が混合的で限定検証の範囲内では優位性を証明できない、のいずれかに該当する場合。
混合的な場合は保守的にW-Aを維持する。

## 8. 非対象・禁止事項（依頼書10章の要約）

production既定値・thresholdの変更、mean以外のproduction aggregation採用、teacher/split/label
policy/normalization/group数の変更、learning rate/scheduler/window/sampling/feature/augmentation
変更、Focal/Dice/Hard Negative Mining、S5-12 label policy Run Bの再実行、S5-07〜S5-11のteacher v7
全面再実行、source H5書き換え、新規5 epoch pilotを3回以上実行すること、長期学習への移行は行わない。

## 9. 次のアクション

この文書の内容で認識に相違がなければ、新規checker
`check_stage5_class_weight_threshold_free.py/.sh`の実装（想定1）から着手する。実装過程で仕様上の
重大な不明点が見つかった場合は、実装を進めず事実・選択肢・推奨案を報告する。以降の実装・検証結果
（threshold-free診断結果、W-C config parity・学習履歴・threshold 0.5結果、W-A/B/C比較表、新規5
epoch実行回数、仮説判断、production未反映事項）は本書へ追記していく。

## 10. 関連文書

- `docs/stage5/s5-13/stage5_s5_13_supplement_implementation_handoff.md`（実装引き継ぎ、正本）
- `docs/stage5/s5-13/stage5_s5_13_report_to_policy_chat.md`（S5-13完了報告、W-A/B結果の数値根拠）
- `docs/stage5/s5-13/stage5_s5_13_class_weight_ablation_implementation_request.md`（S5-13固定条件、比較checker
  design、privacy要件）
- `docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.7節（実装事項F、S5-13数値正本）
- `docs/stage5/stage5_revision_management_record.md`（S5-13補足節、Decision record D-023〜D-025）

---

## 実施記録（以下、検証結果を随時追記する）

### Step 1〜3: threshold-free checker実装・CPU静的テスト（完了、2026-09-14）

変更・新規ファイル:

```text
Stage5/checks/real_h5/check_stage5_class_weight_threshold_free.py/.sh   (新規)
Stage5/checks/dummy/check_dummy_class_weight_threshold_free.py/.sh      (新規)
docs/stage5/FILES.md                                                     (変更)
```

**実装内容（想定通り）:** `evaluate_stage5.py`のW-A/W-B評価は`SAVE_PREDICTIONS=1`（既定）で実行済み
のため、各動画の`prob_femur`を含む予測`.npz`（`<evaluation_dir>/predictions/<split>/<video>.npz`）を
再利用し、GTは`h5_metrics.csv`の`h5_path`列から元のH5を直接読み込んで取得する設計で実装した
（**モデル再推論・CUDA・torch不要**）。`evaluate_stage5.py`自体はtorchをmodule scopeでimportするため、
このcheckerでは意図的にimportせず、`safe_name()`（predictionファイル名規則）のみ1行複製して整合性を
保っている。

主要機能:

- `binary_clf_curve()`: 降順scoreの重複値ごとの累積TP/FPを構築（distinct threshold単位、全点を
  thresholdごとに再走査しない）
- `average_precision()` / `roc_auc()`: 標準定義（`AP=Σ(R_n−R_{n−1})P_n`、台形則によるROC AUC）で実装
- `max_f1_from_curve()`: 全distinct thresholdにわたる最大F1とその診断threshold
- `recall_at_max_fpr()`: 指定FPR以下で達成可能な最大recall/precision（固定FPR候補・
  `--reference_fpr`双方に使用）
- `threshold_grid_table()`: 固定threshold grid（既定0.05刻み）でのTP/FP/TN/FN/precision/recall/F1/
  IoU/FPR/predicted positive rateをsplit aggregateおよびvideo-levelで出力
- `check_threshold_0p5_parity()`: 再計算したthreshold 0.5のTP/FP/TN/FNが既存`h5_metrics.csv`と
  完全一致することをfail-fast gateにする（6.3節要件）
- GT class別score percentile（mean/median/p10/p25/p75/p90/p95/p99）

**Step 3静的検証:**

- `checks/dummy/check_dummy_class_weight_threshold_free.py`: 14件の合成テストすべて合格。
  - sklearn文書の教科書例（AP=0.8333、AUROC=0.75）と一致することを確認
  - **discrimination保持 vs calibration shiftの分離を検証する核心テスト**: 同じ順位関係のまま
    全scoreを0.5未満へシフトしても（閾値0.5ではTP=FP=0へ「崩壊」する）、AUPRCは教科書例と完全に
    同じ値（0.8333）を保つことを確認した。これはS5-13補足の診断目的そのものを検証するテストである。
  - ランダム20点×5試行でO(N)ベクトル化AP計算とO(N²)愚直実装を突き合わせ、すべて一致（tol 1e-9）
  - tie処理（全score同値でAP=prevalence）、positiveクラスが空の場合のNaN返却、
    `max_f1_from_curve`とgrid searchの一致、`recall_at_max_fpr(0.0)`の境界動作、
    `threshold_grid_table`の境界threshold（0.0で全点positive、1.01で全点negative）、
    parity gateのpass/failケースをすべて確認した。
  - このdevコンテナには`numpy`/`h5py`が入っていないため、一時venv（`/tmp/.../scratchpad_numpy_venv`、
    セッション終了後に破棄）へ`pip install numpy h5py`して実行した。GPU実機のconda環境
    （`dualtrack311`）には両方とも既にインストール済みのため、本番実行では追加インストール不要と
    見込む。
- 合成H5（200点、10%positive、5%ignore）と合成予測`.npz`を使い、`main()`のCLIエンドツーエンドも
  検証した。GTと相関するがthreshold 0.5未満に留まる確率を与えたところ、期待通り
  threshold 0.5ではTP=FP=0（全negative予測）だがAUPRC=0.98・AUROC=0.996・max F1=0.95という結果になり、
  「calibration shiftで閾値0.5では検出できないが、識別能力自体は保持されている」ケースを正しく
  検出できることを確認した。`h5_metrics.csv`のcountを意図的に改ざんした場合、fail-fast gateが
  正しく例外を送出することも確認した。
- `py_compile`・`bash -n`・`git diff --check`はすべて合格。

### 次のアクション（GPU側、補足検証1の実行を依頼）

新規5 epoch pilotの回数(上限2回)はまだ0/2消化。以下は**再学習を伴わない**（保存済みW-A/Bの
predictionを再利用するだけの）診断であり、回数上限には影響しない。

```bash
# W-A (EX260914)
EVALUATION_DIR=/mnt/data/3d_projects/stage5_evaluations/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad/last \
CHECKPOINT=last \
REFERENCE_FPR=0.1171 \
  bash checks/real_h5/check_stage5_class_weight_threshold_free.sh

# W-B (EX260916)
EVALUATION_DIR=/mnt/data/3d_projects/stage5_evaluations/260916/pointnext_s_EX260916_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_cw_manual_0p5_1p5_lr1e3_ep5_bs1_acc8_nopad/last \
CHECKPOINT=last \
REFERENCE_FPR=0.1171 \
  bash checks/real_h5/check_stage5_class_weight_threshold_free.sh
```

`REFERENCE_FPR=0.1171`はW-Aのvalidation threshold-0.5 FPR（11.71%）で、W-B側の出力に
「W-AのFPR以下で達成可能な最大recall」を含めるために指定している。両方の出力（fail-fast gate結果、
`AUPRC`/`AUROC`/`max F1`のコンソールログ、`work_dirs/_class_weight_threshold_free/*.json`の内容）を
共有してもらえれば、W-Bのthreshold 0.5全negativeが識別能力の消失かcalibration shiftかを判定し
（handoff文書6.4節の解釈基準）、W-C事前確認・5 epoch pilot（補足検証2）へ進む。

### Step 4: 補足検証1（threshold-free診断）結果（完了、2026-09-14）

W-A（`EX260914`）・W-B（`EX260916`）とも`check_stage5_class_weight_threshold_free.sh`が
threshold-0.5 parity gateに合格した（既存`h5_metrics.csv`のTP/FP/TN/FNと完全一致）。

**split aggregate:**

| split | run | AUPRC | AUROC | max F1 |
| --- | --- | ---: | ---: | ---: |
| train_sanity | W-A | 0.0860 | 0.8519 | 0.1849 |
| train_sanity | W-B | 0.0707 | 0.8368 | 0.1566 |
| validation | W-A | 0.0574 | 0.8018 | 0.1162 |
| validation | W-B | 0.0367 | 0.7652 | 0.0885 |

（validation positive比率は約1.05%であり、ランダムなAUPRC期待値もこの水準。W-A/Bとも
これを大きく上回っており、W-Bも識別能力自体は明確に保持している。）

**固定FPRでのrecall/precision（validation）:**

| FPR | run | recall | precision |
| --- | --- | ---: | ---: |
| 1% | W-A | 11.60% | 10.92% |
| 1% | W-B | 8.40% | 8.15% |
| 5% | W-A | 28.35% | 5.65% |
| 5% | W-B | 22.54% | 4.55% |
| 10% | W-A | 41.16% | 4.17% |
| 10% | W-B | 34.24% | 3.49% |

**W-AのFPR（11.71%）以下で達成可能な最大recall（validation）:** W-A 44.70%（threshold≈0.500、元の
threshold 0.5結果とほぼ一致し整合性チェックとして機能）に対し、W-B 37.59%（threshold≈0.107、
threshold 0.5から大きく下げた場合）。

**GT class別score分布（validation）:** W-Aはpositive median 0.466・background median 0.116と
明確に分離している。W-Bはpositive median 0.071・background median 0.014と全体に大きく圧縮されて
いるが、positive p75(0.153)がbackground p90(0.118)を上回るなど分離自体は残っている。

**video-level（validation 18動画）:** AUPRC win countはW-A 12・W-B 6（tie無し）。ただしAUPRC
**median**はW-A 0.0422・W-B 0.0427とほぼ同値（meanはW-A 0.1046・W-B 0.0684とW-A優位）で、
S5-12でも見られたmean/median不一致のパターンが再度現れている。AUROC medianはW-A 0.8455・
W-B 0.8111とW-A優位。

**解釈（handoff文書6.4節）:** 「discrimination自体が失われた」（AUPRCがランダム水準へ崩壊）にも
「W-Aに近い」（数値がほぼ一致）にも該当しない中間的な結果だった。W-Bのthreshold 0.5全negativeは
**主にcalibration shift**が原因である（W-AのFPR以下という同一制約下で、thresholdを0.5から0.107へ
下げればrecall 37.59%まで回復する）が、**discrimination自体も中程度（相対15〜35%程度）に劣化して
おり、純粋なcalibration shiftだけでは説明しきれない**。video-level medianの近さは、この劣化が
一部の動画（AUPRCの高い動画）に偏っている可能性を示唆する。

handoff文書6.4節の方針通り、W-Bは現行threshold 0.5でのS5-14 controlには採用しない（既にD-023で
決定済み）。この結果はS5-15のthreshold診断用履歴候補として保持する。production thresholdの変更は
本補足でも行わない。

### 次のアクション（GPU側、補足検証2の準備）

W-Cの事前確認（handoff文書7.1節）に進む。出力先がW-A/Bと衝突しないこと、`CLASS_WEIGHT`の
resolved値が期待通りであることを起動ログで確認してから5 epoch本実行に進む。

### Step 5〜9: W-C 5 epoch pilot・固定評価・threshold-free診断・比較判定（完了、2026-09-14）

**新規5 epoch pilot回数: 1/2回**（handoff文書8章の上限を遵守。結果が明確なため2回目は実施しない）。

**Step 5（W-C事前確認・5 epoch pilot）:** handoff文書7.1節の推奨通り1 epoch smokeは省略し
（manual weight経路は既存機能で確認済み）、`CLASS_WEIGHT=0.11764706,1.88235294`、
`POINTNEXT_NORM=groupnorm`、W-A/Bと同一の初期checkpoint・split・seedで直接5 epoch本実行した
（`EX260917`）。学習ループのwindow単位running metricsでは、W-Bと異なりepoch 2〜3で一時的に
崩壊気味（fp≈0、f1≈0）になった後、epoch 4〜5で明確に回復し、val loss/F1/IoUがepoch 5まで
改善し続けた。best.pt/last.ptは同一epoch（5）を指す（val IoUがepoch5で最大）。

**Step 6（config parity）:** `check_stage5_class_weight_ablation.sh`（RUN_A_DIR=W-A、
RUN_B_DIR=W-C、`EXPECTED_WEIGHT_B=0.11764706,1.88235294`）が合格した。train_files.txt/
val_files.txt 162/18行一致、resolved weight `[0.11764706, 1.88235294]`が期待値と一致、
class weight以外のconfig差分なし。

**Step 7（固定評価）・Step 8（threshold-free診断）:** 既存`evaluate_stage5.sh`（train sanity
3動画+validation 18動画、mean probability aggregation、threshold 0.5）と新規
`check_stage5_class_weight_threshold_free.sh`（保存済みprediction再利用、再学習なし）を実行した。

**注記（作業ミス）:** 補足検証1・2とも、匿名化共有bundle（`anonymized_metrics_SHARE_THIS`）の
コピー指示で誤って`<checkpoint_stem>/`（`best/`等）を余計に含めたパスを提示した。
`evaluate_stage5.sh`のコードを確認したところ、これは私の指示ミスであり不具合ではない。
`export_anonymized_stage5_metrics.py`は両checkpoint評価が完了した後に1回だけ呼ばれ、
`summarize_stage5_evaluations.py`が生成した`OUTPUT_ROOT`直下の`*_all_checkpoints.csv`から
`anonymized_metrics_SHARE_THIS`を`OUTPUT_ROOT`直下（`<checkpoint_stem>/`の外側）に生成する。
生の`h5_metrics.csv`/`predictions/`は`best/`/`last/`個別ディレクトリに保存されるため上書きされず、
共有bundleには両checkpoint分の行が正しく含まれている（データ欠落なし）。

**threshold 0.5固定評価結果（split aggregate）:**

| split | run | TP | FP | FN | recall | precision | F1 | IoU | FPR | predicted positive | TP0動画数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train_sanity | W-A | 5,682 | 82,901 | 4,074 | 58.24% | 6.41% | 0.1156 | 0.0613 | 13.42% | 88,583 | 0/3 |
| train_sanity | W-C | 2,809 | 20,137 | 6,947 | 28.79% | 12.24% | 0.1718 | 0.0940 | 3.26% | 22,946 | 0/3 |
| validation | W-A | 33,531 | 830,987 | 41,502 | 44.69% | 3.88% | 0.0714 | 0.0370 | 11.71% | 864,518 | 1/18 |
| validation | W-C | 13,149 | 183,224 | 61,884 | 17.52% | 6.70% | 0.0969 | 0.0509 | **2.58%** | 196,373 | **5/18** |

**video-level（validation 18動画）:**

| 指標 | W-A mean/median | W-C mean/median |
| --- | --- | --- |
| F1 | 0.0672 / 0.0530 | 0.0805 / 0.0717 |
| IoU | 0.0355 / 0.0272 | 0.0435 / 0.0372 |
| precision | 0.0374 / 0.0282 | 0.0565 / 0.0627 |
| recall | 0.4741 / 0.4468 | 0.2125 / **0.0970** |
| FPR | 0.1153 / 0.1094 | **0.0260 / 0.0280** |
| predicted positive count | 48,029 / 33,980 | 10,910 / 9,174 |

F1 win count: W-A 9・W-C 8・1引き分け。IoU win count: W-A 9・W-C 8・1引き分け（ほぼ互角）。
**FPR win count（低いほうが勝ち）: 18動画すべてでW-Cが優位**。TP0動画数はW-Aの1/18からW-Cの
**5/18へ増加**。

**threshold-free診断結果（保存済みprediction、再学習なし）:**

| split | run | AUPRC | AUROC |
| --- | --- | ---: | ---: |
| train_sanity | W-A | 0.0860 | 0.8519 |
| train_sanity | W-C | 0.0940 | 0.8556 |
| validation | W-A | 0.0574 | 0.8018 |
| validation | W-C | **0.0445** | **0.7588** |

**固定FPRでのrecall（validation、同一FPR条件での直接比較）:**

| FPR | W-A recall | W-C recall |
| --- | ---: | ---: |
| 1% | 11.60% | 9.24% |
| 5% | 28.35% | 24.84% |
| 10% | 41.16% | 35.66% |
| 11.71%（W-A基準） | 44.70% | 38.75% |

**validation split では、同一FPRに揃えて比較すると全ての水準でW-AがW-Cを上回る。** これは
W-Cのthreshold 0.5での見かけ上のF1/precision/IoU改善が、モデルの識別能力（ROC/PR curveそのもの）
が向上した結果ではなく、より保守的な暗黙のoperating pointへ移動した結果であることを示す。
train_sanity（3動画のみ、高分散）では逆にFPR 10%以上でW-Cがわずかに上回ったが、動画数が
少なく信頼性は低い。

### 仮説判断・採否

handoff文書9章の基準に照らすと、次の通りである。

1. threshold 0.5でW-AよりFP/FPR・predicted positive率が低下する: **満たす**（validation FPR
   11.71%→2.58%、全18動画で低下）。
2. precision/F1/IoUの主要なvideo-level統計が改善する: **aggregate・mean/medianでは満たす**が、
   video別勝敗はほぼ互角（9対8）であり「動画別傾向も改善を支持する」とまでは言えない。
3. recallが大幅に低下しない: **満たさない**。aggregate 44.69%→17.52%、video median
   44.68%→9.70%と特にmedianで大幅に悪化。
4. TP0動画数がW-Aの1/18から増加しない: **満たさない**。1/18→5/18。
5. train sanityとvalidationの改善方向が大きく反転しない: 概ね一致（FP/FPR低下・precision/IoU
   向上・recall低下という同方向）。
6. AUPRC/PR curveが単なるscore shiftではなく有効なtrade-offを支持する: **満たさない**。
   validationの同一FPR比較でW-Aが全水準でW-Cを上回り、W-Cの改善は真の識別能力向上ではなく
   保守的な暗黙thresholdへの移動によるものと判断できる。

「W-Aを維持する条件」（TP0動画数またはrecallが明確に悪化する／threshold-free指標とvideo-level
結果がW-Cを支持しない／結果が混合的で限定検証の範囲内では優位性を証明できない）に複数該当する。
**W-Cは不採用、W-A（強いauto由来weight `[0.05963856, 1.94036150]`）を維持**と判断する。
「W-A/Cが混合的な場合は保守的にW-Aを維持する」というhandoff文書9章末尾の方針にも合致する。

結果が明確に不採用側へ判定できたため、2回目の新規5 epoch pilot（別の中間weight候補）は実施しない
（handoff文書8章「W-Cが明確に採用または不採用と判断できた時点で補足を終了し、2回目を消化する
必要はない」）。

### productionへ未反映であること

`train_stage5.py`/`infer_stage5.py`/`evaluate_stage5.py`の既定値（`class_weight=auto`）は変更して
いない。W-C checkpointはproduction checkpointへ接続していない。S5-13本比較の結論（W-A維持）は
本補足でも変わらない。

### S5-14へ進む前に方針管理チャットで判断が必要な事項

1. 本結果（W-C不採用、W-A維持）を正式に確定し、S5-13補足を完了としてS5-14へ進んでよいか。
2. W-Bと同様、W-Cのcheckpoint/結果もS5-15のthreshold診断用履歴候補として保持してよいか
   （W-Cは閾値を下げればW-Aの主要な指標に近づく可能性があるが、同一FPRで比較するとW-Aが常に
   上回るため、production threshold候補としての優先度はW-Bより低いと考える）。
3. 追加の中間weight探索（例: 比24等、W-AとW-Cの間）は行わず、S5-14（座標依存・frame-level・
   overlap loss診断）へ進んでよいか。
