# Stage 5 S5-13: GroupNorm固定Class Weight Ablation 報告書

作成日: 2026-09-14
作成元: Stage 5実装チャット
状態: **評価完了**。Step F1〜F7完了。W-B（弱い固定class weight）は不採用、W-A（強いauto由来weight）を
維持という明確な結論。production反映は方針管理チャットの判断待ち

本書は`docs/stage5/s5-13/stage5_s5_13_class_weight_ablation_implementation_request.md`（実装依頼書、正本）を
受けての理解・現状・実装方針の記録として作成し、以後、検証が進むごとに本書へ結果を追記して
方針管理チャットへの完了報告として使用する。実装依頼書12章が別途指定する
`stage5_s5_13_report_to_policy_chat.md`という完了報告ファイル名と一致させてあるため、
新しい完了報告ファイルを別途作成する必要はない。

## 1. 経緯

S5-12（GroupNorm固定Label policy ablation、teacher v7）の完了報告を受け、方針管理チャットが
次の3判断を行った。

1. S5-13では、より保守的なRun A（`bbox_noncontour_ignore`）を暫定label policyとして採用する。
2. S5-12を長期化せず、class weight ablationへ進む。
3. S5-07〜S5-11のteacher v7による全面再実行は行わない。v7のS5-12 Run Aを今後の比較baselineとし、
   必要な診断だけを個別に再実行する。

これを踏まえ、S5-13としてclass weightだけを変更した2条件（W-A: 強いauto由来weight、W-B: 弱い
固定weight）を比較する依頼を受けた。

## 2. S5-13の目的と仮説

teacher v7のvalid点はpositiveが少なく、S5-12で使った auto由来weight
`[0.05963856, 1.94036150]`（positive:background比 約32.5倍）は、minority positiveの学習と
TP0抑制に役立つ一方、GroupNorm条件で残る広範なFP/FPRを増やしている可能性がある。

比較する2条件:

| Arm | class weight | positive:background比 |
| --- | --- | ---: |
| W-A（control） | `[0.05963856, 1.94036150]` | 約32.54 |
| W-B | `[0.5, 1.5]` | 3.0 |

検証事項:

1. 弱いweightがglobal FP/FPRとpredicted positive率を下げるか。
2. FP低下が単なるpositive予測崩壊ではなく、precision/F1/IoU改善を伴うか。
3. positive recall、TP0動画数、video median F1/IoUを許容不能な程度に悪化させないか。
4. train sanityとvalidationで改善方向が一致するか。

`[1.0, 1.0]`（no weight）、別epsilonのauto weight、Focal/Dice lossは初回必須armに含めない。
W-A/W-Bで方向性を判定できない場合のみ、追加armを方針管理チャットへ提案する。

## 3. 固定する実験条件（S5-12から変更しないもの）

teacher v7（180 H5、`bboxrank_v7_cvat_authoritative_crop_quality_v1`）、S5-12 Run Aと同一の
train 162 / validation 18 split、`label_policy=bbox_noncontour_ignore`、GroupNorm 8 groups、
S5-12と同一のGroupNorm S3DIS部分転移checkpoint、seed 42、window 16/8/tailあり、physical batch 1、
gradient accumulation 8（point-weighted normalization）、features `intensity,confidence`、
CrossEntropyLoss、label smoothing 0.0、AdamW lr 1e-3/weight decay 1e-4、dropout 0.0、epochs 5、
`model.eval()`、mean probability aggregation、threshold 0.5。normalization group数、learning rate、
scheduler、window、sampling、feature、augmentation、label policy、threshold、production
aggregationは変更しない。

## 4. 実装前監査で確認した既存コードの状況

### 4.1 `train_stage5.sh`のrun命名に既知の不具合

`PREFIX`変数（train_stage5.sh L84付近）に文字列`"auto_weight"`が**ハードコード**されている。

```bash
PREFIX="w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep${EPOCHS}_bs${BATCH_SIZE}_acc${GRADIENT_ACCUMULATION_STEPS}_nopad"
```

`CLASS_WEIGHT`は既にenv override可能（S5-11/S5-12で`CLASS_WEIGHT="0.05963856,1.94036150"`として
利用済み）だが、`PREFIX`はこれを反映しないため、W-Bを`CLASS_WEIGHT="0.5,1.5"`で実行しても
出力先ディレクトリ名は`..._auto_weight_...`のまま表示される。さらに`PREFIX`がclass weightで
変わらないため、W-A/Bを同一`EX_DATE`で実行すると出力を上書きする危険がある（S5-11/S5-12では
`EX_DATE`を手動で分けることで回避していたが、依頼書7.1節はより恒久的な修正——weight tagの
run名反映、非empty出力先への書き込みガード——を求めている）。

### 4.2 再利用可能な既存実装

- class weight解決ロジック（`resolve_class_weight`/`parse_class_weight`/`is_auto_class_weight`、
  `train_stage5.py`）は、文字列`"auto"`と明示的な`"0.5,1.5"`形式の両方を既にサポートしており、
  大きな変更なく再利用できる。
- `metrics.jsonl`には既に`debug_class_weight_0/1`、`debug_class_count_0/1`、
  `debug_loss_normalizer`、`debug_pred_positive_ratio`等のepoch別診断フィールドが存在し、
  依頼書7.3節が求めるweighted denominator診断の大部分は追加実装なしで再利用できる見込み。
- `stage5/training/losses.py`の`CrossEntropySegmentationLoss`はclass_weightを
  `register_buffer`として保持するだけのシンプルな構造で、weight値の変更自体に対する
  追加実装は不要。
- S5-12の評価pipeline（`evaluate_stage5.sh`、mean baseline parity gate、匿名化export）は
  target（native teacher v7 label）が変わらないため、そのまま再利用できる。

### 4.3 未確認・要監査の項目

- S5-12 Run Aの実行時revisionと現在revisionのgit diffに、学習挙動へ影響する差分がないか
  （依頼書5節の再利用条件の一つ）。
- S5-12 Run Aの`train_files.txt`/`val_files.txt`の内容・順序がW-Bのものと完全一致するか
  （teacher v7 inventoryが変わっていないことを前提とするが、機械的確認が必要）。
- GroupNorm S3DIS部分転移checkpointのpath・SHA-256が同一か。

## 5. 想定する実装の形（未着手）

依頼書のStep F1〜F8に沿って進める想定。

1. **train_stage5.sh修正**: class weightからfilesystem-safeなtag
   （例: `cw_auto_resolved_0p05963856_1p94036150`、`cw_manual_0p5_1p5`）を生成し、run名・
   起動ログへ反映する。`OUTPUT_DIR`/`EXPERIMENT_NAME`の明示override、既存出力directoryが
   non-emptyな場合のfail-fastガードを追加する。production既定値（`CLASS_WEIGHT=auto`）自体は
   変更しない。
2. **比較checker新規実装**: `Stage5/checks/real_h5/check_stage5_class_weight_ablation.py/.sh`。
   train/val file list一致、teacher schema/label policy/normalization/model/features/window/
   seed/optimizer設定の一致、initialization checkpoint一致、class weight以外の差分がないこと、
   resolved weightが期待値と一致すること、両runがepoch 5まで完走しfiniteであることを検証する。
   S5-12のchecker群（`check_stage5_label_policy_bbox_preflight.py`等）と同じ
   share/private分離・匿名化self-checkの構成を踏襲する。
3. **W-A再利用判定（Step F4）**: 依頼書5節の条件（inventory、split、checkpoint、config、
   seed、window、batch、loss、optimizer、dropout、code revision）をS5-12 Run Aの記録と
   機械的に比較する。全一致ならRun AをW-Aとして再利用し、不一致があれば現在revisionでW-Aを
   5 epoch再実行してS5-12 Run Aはexternal referenceへ下げる。判定理由をJSON/CSVへ保存する。
4. **W-B 1 epoch smoke → 5 epoch pilot**: 既存`train_stage5.sh`を修正後の形で使用し、
   `CLASS_WEIGHT="0.5,1.5"`で実行する。
5. **固定評価**: 既存`evaluate_stage5.sh`でW-A/Bのepoch 5 checkpointを評価する
   （S5-12と同じtrain sanity 3動画+validation 18動画、mean aggregation、threshold 0.5、
   native teacher v7 labelがprimary target）。BBox non-contour region metricsは補助診断として
   任意で残す。
6. **集計・文書更新**: aggregate/video-level/TP0/weighted denominator診断を集計し、評価レポート・
   管理記録を更新した上で、本書（本節以降）へ結果を追記する。

## 6. 判定基準（依頼書9章の要約）

W-Bを有望とするには、validationで(1) FP/FPR・predicted positive率がW-Aより明確に低下、
(2) precision/F1/IoUの主要video-level統計が改善、(3) recallが大幅に低下しない、(4) TP0動画数が
`1/18`から増加しない、(5) train sanityとvalidationの改善方向が大きく反転しない、(6) 改善が
video別勝敗またはmedianでも支持される、を総合的に満たす必要がある。FP低下と引き換えにrecallが
崩壊する場合や、aggregateだけ改善しvideo-levelが悪化する場合は採用しない。

## 7. 非対象・禁止事項（依頼書10章の要約）

label policy Run Bの再学習・S5-12長期化、teacher v6復帰、S5-07〜S5-11一律再実行、production
既定値（normalization/label policy/class weight/threshold/aggregation）の変更、GroupNorm group数
変更・LayerNorm比較、Focal/Dice/Hard Negative Mining、learning rate/scheduler/window/sampling/
feature/augmentation変更、50〜200 epoch学習、source H5書き換えは行わない。

## 8. 次のアクション（更新: 2026-09-14、Step F1〜F8完了後）

Step F1〜F8はすべて完了した（詳細は本書末尾「実施記録」参照）。実装チャット側としてこれ以上の
armは追加せず、方針管理チャットの判断（末尾「方針管理チャットで判断が必要な事項」）を待つ。
再学習、label policy/normalization/threshold/aggregation変更、S5-07〜S5-11の再実行は実施していない。

## 9. 関連文書

- `docs/stage5/s5-13/stage5_s5_13_class_weight_ablation_implementation_request.md`（実装依頼書、正本）
- `docs/stage5/s5-12/stage5_s5_12_report_to_policy_chat.md`（S5-12完了報告、W-A候補の数値根拠）
- `docs/stage5/s5-12/stage5_s5_12_label_policy_implementation_handoff_prompt.md`（S5-12固定条件）
- `docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.6節・9.7節（実装事項F、数値正本）
- `docs/stage5/stage5_revision_management_record.md`（S5-13記録、Decision record D-018〜D-022）

---

## 実施記録（以下、検証結果を随時追記する）

### Step F1〜F3: 現状監査・shell/checker実装・CPU静的テスト（完了、2026-09-14）

変更・新規ファイル:

```text
Stage5/train_stage5.sh                                          (変更)
Stage5/checks/real_h5/check_stage5_class_weight_ablation.py/.sh (新規)
Stage5/checks/dummy/check_dummy_class_weight_tag.sh              (新規)
Stage5/checks/dummy/check_dummy_class_weight_ablation.py/.sh    (新規)
docs/stage5/FILES.md                                             (変更)
docs/stage5/stage5_revision_management_record.md                 (S5-13進捗、状態更新)
```

**train_stage5.sh修正（4.1節の不具合対応）:**

- `CLASS_WEIGHT`から`class_weight_tag()`関数でfilesystem-safeなtagを生成し（`auto`/
  `pointnext_auto`→`cw_auto`、空→`cw_none`、それ以外→`cw_manual_<値をp/_で置換>`）、
  `PREFIX`のハードコードされた`auto_weight`部分をこのtagへ置き換えた。
  `auto`は数値を解決せずtagに含めない（解決値は学習split依存でtrain_stage5.py内でのみ確定するため。
  config.json/checkpoint configの`class_weight_info`とlabel_policy_diagnostics.csvに既に記録済みの
  値を正本とする設計判断とした）。
- `PREFIX`/`EXPERIMENT_NAME`/`OUTPUT_DIR`を環境変数で明示override可能にした
  （`PREFIX="${PREFIX:-...}"`等）。
- 出力先directoryが存在しかつ非emptyな場合、`ALLOW_EXISTING_OUTPUT_DIR=1`を明示指定しない限り
  fail-fastするガードを追加した。
- 起動ログに`class weight   : <値> (tag=<tag>)`を追加した。
- production既定値`CLASS_WEIGHT=auto`自体は変更していない。resolved weight/mode/countsは
  既存の`config.json`/checkpoint configの`class_weight_info`に既に保存されており、追加実装不要だった
  （4.2節の監査どおり）。

**新規checker（7.2節の要件対応）:**

`check_stage5_class_weight_ablation.py/.sh`は2つのtrain run directoryを比較し、次をfail-fastで検証する。

1. train_files.txt/val_files.txtの内容・順序一致
2. config.json全体の一致（class weight関連キー`class_weight`/`class_weight_info`と、run/output由来で
   意図的に異なり得るキー`output_dir`/`train_list`/`train_dir`/`val_list`/`val_dir`/`max_train_files`/
   `max_val_files`/`val_fraction`を除く）→ これにより「変更対象がclass weightだけ」を機械的に保証
3. resolved class weightが期待値と一致すること、かつW-A/W-Bで実際に異なること
4. history.jsonが指定epoch数（既定5）まで記録され、train/val全metricsがfiniteであること
5. 指定checkpoint（既定`last.pt`）の存在
6. 初期checkpoint（`config.json`の`checkpoint`キー）のpath一致、かつ任意でSHA-256一致
   （`--skip_checkpoint_hash`で無効化可）
7. （任意）`evaluate_stage5.py`の`h5_metrics.csv`を2つ渡した場合、評価対象(split, video_name)集合と
   `total_point_count`/`valid_point_count`/`valid_positive_count`/`valid_background_count`の一致
   （Step F7の評価対象point集合一致に対応）

CPU/JSON/CSVのみで、torch/h5py/CUDAをimportしない（この開発コンテナ内でも実行可能）。

**CPU静的テスト（Step F3）:**

- `checks/dummy/check_dummy_class_weight_tag.sh`: `train_stage5.sh`から`class_weight_tag()`の
  関数定義を`sed`で直接抽出して評価し、8ケース（`auto`/`Auto`/`pointnext_auto`/空/`0.5,1.5`/
  `0.05963856,1.94036150`/`1,1`/空白混じり）すべて合格。
- `checks/dummy/check_dummy_class_weight_ablation.py`: checkerの比較関数（config差分、file list一致、
  resolved weight一致、history finite性・epoch数、evaluation target parity）に対する11件のpass/fail
  合成テストすべて期待通りに合格。
- 加えてcheckerのCLIエントリポイント自体もsubprocessで合成run directory一対に対して実行し、
  正常終了（exit 0）とJSON出力を確認した。
- `python3 -m py_compile`、`bash -n`、`git diff --check`をすべての変更・新規ファイルに対して実行し合格。

### Step F4: W-A再利用判定（一部完了、2026-09-14）

S5-12 Run Aの学習時revision（コミット`4b55e55`「Implement Stage 5 label policy ablation」、本報告書
作成時点のHEAD）と、Step F1〜F3実装後の現在の作業ツリーを`git diff`で比較した。差分は上記
`train_stage5.sh`のrun命名/衝突ガード変更のみであり、`train_stage5.py`へ渡す引数構築部分
（`cmd=(...)`）や`train_stage5.py`本体・`stage5/`配下のコードは無変更だった。したがって**現時点で
コードrevisionによる学習挙動差分はない**（依頼書5節の最終条件を満たす）。

未確認（GPU側でのみ確認可能、次のアクション参照）: S5-12 Run Aの実際のrun directoryにおける
`config.json`の全フィールド、`train_files.txt`/`val_files.txt`の実内容、初期checkpointのpath/SHA-256。
これらは実データ・実checkpointを含むため、本開発コンテナ（`/workspace`外へのアクセス不可）からは
確認できない。

### 次のアクション（GPU側での実行を依頼）

1. **Step F4残り**: 変更を取り込んだ上で、S5-12 Run Aのrun directoryを特定し、以下を実行してほしい。

   ```bash
   RUN_A_DIR=<S5-12 Run Aのrun directory> \
   RUN_B_DIR=<同じdirectoryを一時的に指定してよい（自己一致確認用）> \
     bash checks/real_h5/check_stage5_class_weight_ablation.sh
   ```

   まずはRUN_A_DIR自身のconfig.json/train_files.txt/val_files.txtが存在し読めることの確認だけでもよい。
   本格的な比較はW-B完走後（Step F6の後）に、`RUN_A_DIR=<S5-12 Run A> RUN_B_DIR=<W-B run>`で実行する。

2. **Step F5（W-B 1 epoch smoke）**:

   ```bash
   CLASS_WEIGHT="0.5,1.5" EPOCHS=1 EX_DATE=<新しい日付> \
     bash train_stage5.sh
   ```

   起動ログの`class weight   : 0.5,1.5 (tag=cw_manual_0p5_1p5)`とoutput_dir名に`cw_manual_0p5_1p5`が
   含まれることを確認してほしい。

3. **Step F6（W-B 5 epoch pilot）**: 同じINIT_CHECKPOINT/seed/splitで`EPOCHS=5`にして再実行する。

4. Step F6完了後、`check_stage5_class_weight_ablation.sh`をS5-12 Run A（W-A）とW-Bの両方に対して
   実行し、config parityを確認する。

5. Step F7（固定評価、既存`evaluate_stage5.sh`をW-A/W-Bのepoch 5 checkpointに対して実行）、
   Step F8（集計）は、Step F6完了・checker合格後に着手する。

### Step F4残り: W-A候補(EX260914)の個別確認（完了、2026-09-14）

S5-12 Run A相当のrun directory(`.../260914/pointnext_s_EX260914_..._auto_weight_..._nopad`、
teacher v7、GroupNorm、`bbox_noncontour_ignore`)のconfig.jsonを直接確認した。

```text
checkpoint = stage5_pointnext_s_s3dis_partial_init_groupnorm.pt
pointnext_norm = groupnorm, pointnext_norm_groups = 8
label_policy = bbox_noncontour_ignore
seed = 42, window=16/8, batch=1, grad_accum=8, lr=1e-3, weight_decay=1e-4, dropout=0.0
class_weight_info.resolved = [0.05963856, 1.9403615]（依頼書4節のW-A期待値と一致）
```

固定条件（6節）をすべて満たすため、このrunをW-Aとして再利用する。GPU側の初回確認では
誤ってS5-12 Run B(`label_policy=bbox_noncontour_background`、`.../260915/...`)と比較してしまい、
checkerが正しくlabel_policy不一致を検出して停止した(これはS5-13の比較対象ペアではないため
想定通りの挙動)。

### Step F5: W-B 1 epoch smoke（完了、2026-09-14）

`CLASS_WEIGHT="0.5,1.5" POINTNEXT_NORM=groupnorm INIT_CHECKPOINT=<W-Aと同じgroupnorm初期checkpoint> EPOCHS=1 EX_DATE=260915smoke bash train_stage5.sh`

- train 162 files/715 samples、val 18 files/87 samples（W-Aと同一split）
- `label_policy: bbox_noncontour_ignore`、変換点数0（no-op、W-Aの診断値と一致）
- `class weight: [0.5, 1.5]`（要求通り解決）
- output_dir名に`cw_manual_0p5_1p5`タグが正しく反映（train_stage5.shの命名修正が実runで機能することを確認）
- 既存runと衝突なし、1 epoch完走、finite loss、checkpoint保存を確認

### Step F6: W-B 5 epoch pilot（完了、2026-09-14）

`CLASS_WEIGHT="0.5,1.5" POINTNEXT_NORM=groupnorm INIT_CHECKPOINT=<同上> EPOCHS=5 EX_DATE=260916 bash train_stage5.sh`

出力: `/mnt/data/3d_projects/stage5_runs/260916/pointnext_s_EX260916_..._cw_manual_0p5_1p5_..._ep5_.../`
（`best.pt`＝`last.pt`と推定、後述のepoch別スコアより両者が同一epochを指す可能性が高い。要確認）

epoch別ログ:

| epoch | train loss | train f1 | train iou | train fp | train fn | val loss | val f1 | val iou | val fp | val fn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3089 | 0.0291 | 0.0147 | 4,514,560 | 1,209,511 | 0.1565 | 0.0000 | 0.0000 | 0 | 142,147 |
| 2 | 0.1506 | 0.0000 | 0.0000 | 0 | 1,295,189 | 0.1396 | 0.0000 | 0.0000 | 0 | 142,147 |
| 3 | 0.1386 | 0.0000 | 0.0000 | 0 | 1,295,189 | 0.1362 | 0.0000 | 0.0000 | 0 | 142,147 |
| 4 | 0.1322 | 0.0000 | 0.0000 | 0 | 1,295,189 | 0.1344 | 0.0000 | 0.0000 | 0 | 142,147 |
| 5 | 0.1281 | 0.0000 | 0.0000 | 0 | 1,295,189 | 0.1370 | 0.0000 | 0.0000 | 0 | 142,147 |

**重要な予備的観察（正式判定はStep F7の集計評価後に行う）:** epoch 2以降、train/valとも`fp=0`かつ`fn`が
固定値のまま推移しており、これはFP低下ではなくpositive予測の完全崩壊（全点をbackgroundと予測）を示唆する。
依頼書3章・9章が想定していた「FP低下と引き換えにrecallが崩壊する」ケースに該当する可能性が高いが、
この数値はtrain_stage5.py学習ループ内のwindow単位running metricsであり、公式比較に使う
`evaluate_stage5.py`のmean-probability集計評価（H5単位、Step F7）ではまだ確認していない。
production/採否の判断はStep F7〜F8の正式集計を経てから行う。

### 次のアクション（GPU側、Step F4本比較・F7評価を依頼）

1. **Step F4本比較**: W-A(EX260914)とW-B(EX260916、実runが揃った)でconfig parityを確認する。

   ```bash
   export RUN_A_DIR=/mnt/data/3d_projects/stage5_runs/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad
   export RUN_B_DIR=/mnt/data/3d_projects/stage5_runs/260916/pointnext_s_EX260916_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_cw_manual_0p5_1p5_lr1e3_ep5_bs1_acc8_nopad
   bash checks/real_h5/check_stage5_class_weight_ablation.sh
   ```

2. **Step F7（固定評価）**: 既存`evaluate_stage5.sh`をW-A/W-Bそれぞれに対して実行する
   （`CHECKPOINT_NAMES`は5 epoch runなので`checkpoint_epoch_*`が存在せず、`last.pt`/`best.pt`のみに絞る）。

   ```bash
   # W-A (EX260914)
   RUN_DIR=/mnt/data/3d_projects/stage5_runs/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad \
   EX_DATE=260914 \
   CHECKPOINT_NAMES="last.pt best.pt" \
     bash evaluate_stage5.sh

   # W-B (EX260916)
   RUN_DIR=/mnt/data/3d_projects/stage5_runs/260916/pointnext_s_EX260916_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_cw_manual_0p5_1p5_lr1e3_ep5_bs1_acc8_nopad \
   EX_DATE=260916 \
   CHECKPOINT_NAMES="last.pt best.pt" \
     bash evaluate_stage5.sh
   ```

3. 両方の評価完了後、`h5_metrics.csv`（train_sanity 3件+validation 18件）のログと、
   `EVALUATION_ROOT/.../summary`相当の集計を共有してほしい。Step F8（集計・報告）はその後に着手する。

### Step F4本比較・Step F7〜F8: config parity・固定評価・集計（完了、2026-09-14）

**Step F4本比較（`check_stage5_class_weight_ablation.sh`、W-A=EX260914 / W-B=EX260916）:** 合格。
train_files.txt/val_files.txt 162/18行が完全一致、W-A resolved weight
`[0.05963856, 1.9403615]`・W-B resolved weight`[0.5, 1.5]`とも期待値と一致、history.jsonは両arm
epoch 5まで記録・全metrics finite、class weight以外のconfig差分なし。

**Step F7（固定評価、`evaluate_stage5.sh`、`last.pt`/`best.pt`=epoch 5）:** 両arm完走。匿名化
共有ディレクトリ(`anonymized_metrics_SHARE_THIS`)のprivacy check(`original_identifiers_absent`/
`timestamp_like_video_ids_absent`/`absolute_host_paths_absent`)はいずれも`true`で合格を確認した。

**集計結果（split aggregate）:**

| split | run | TP | FP | FN | recall | precision | F1 | IoU | FPR | predicted positive数 | TP0動画数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train_sanity(3動画) | W-A | 5,682 | 82,901 | 4,074 | 58.24% | 6.41% | 0.1156 | 0.0613 | 13.42% | 88,583 | 0/3 |
| train_sanity(3動画) | W-B | 0 | 0 | 9,756 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.00% | 0 | **3/3** |
| validation(18動画) | W-A | 33,531 | 830,987 | 41,502 | 44.69% | 3.88% | 0.0714 | 0.0370 | 11.71% | 864,518 | 1/18 |
| validation(18動画) | W-B | 0 | 0 | 75,033 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.00% | 0 | **18/18** |

**video-level（validation 18動画）:** F1 mean/median: W-A 0.0672/0.0530、W-B 0.0000/0.0000。
recall mean/median: W-A 0.4741/0.4468、W-B 0.0000/0.0000。F1勝敗はW-A 17勝・W-B 0勝・1引き分け
（引き分けの1動画もW-A/Bともpredicted_positive_count=0で偶然一致）。train_sanity(3動画)も同様に
3動画すべてでW-B側TP0・recall 0%。

**Weighted CE denominator診断（epoch 1、train split。既存`debug_class_weight_*`/
`debug_loss_normalizer`/`debug_label_positive_count`等を再利用、新規実装なし）:**

| arm | class weight(pos, bg) | weight比(pos:bg) | loss normalizer | positiveの重み付きloss占有率 |
| --- | --- | ---: | ---: | ---: |
| W-A | (1.9403615, 0.05963856) | 32.54 | 8,657,719.6 | **29.03%** |
| W-B | (1.5, 0.5) | 3.00 | 53,457,991.0 | **3.63%** |

train split の valid点はbackground:positive ≈ 79.5:1（background 103,030,415点、positive
1,295,189点）。W-Aの重み比32.5はこの不均衡を部分的に相殺し、positiveがweighted lossの約29%を
占める水準を保つ。W-Bの重み比3.0では全く不十分で、positiveはweighted lossの3.6%しか占めず、CE
lossを最小化する最も単純な解が「常にbackgroundと予測する」になったと考えられる。W-Bのval側
`debug_mean_logit_margin_positive_minus_background`はepoch 1時点で-2.618（W-A: -0.991）と
大きく負に振れており、崩壊が学習の非常に早い段階（1 epoch以内）で起きたことと整合する。
optimizer step数（両arm epoch毎90 step）・empty-valid window数（両arm0件）は正常であり、
機構上の不具合ではなくweight設定自体が原因と判断できる。

### 仮説判断

依頼書9章の判定基準に照らすと、次の通りである。

1. FP/FPR・predicted positive率の低下: 形式的には満たすが、predicted positiveが完全に0になった
   結果のtrivialな低下であり、意味のある改善ではない。
2. precision/F1/IoUの改善: **満たさない**。いずれも0へ落ち込んだ。
3. recallの維持: **満たさない**。validationで44.69%→0.00%へ完全崩壊。
4. TP0動画数の維持: **満たさない**。1/18→18/18（train_sanityも0/3→3/3）。
5. train sanityとvalidationの方向一致: 両splitとも同一方向（完全崩壊）で一致するが、これは
   「改善方向の一致」ではなく「崩壊方向の一致」である。
6. video別勝敗/medianでの裏付け: **支持しない**。18動画中17動画でW-Aが優位、残り1動画は
   両arm0で実質的な優劣なし。

依頼書9章の不採用分岐「FP低下と引き換えにrecallが崩壊する場合、W-Bは採用しない」に明確に該当する。
S5-12のような判定不能ではなく、**W-Bは不採用、W-A（強いauto由来weight`[0.05963856,
1.94036150]`）を維持**という一貫した結論が得られた。

### productionへ未反映であること

`train_stage5.py`/`infer_stage5.py`/`evaluate_stage5.py`の既定値（`class_weight=auto`）は変更して
いない。W-A自体がS5-12以前からの既定`auto`解決値と同じ値であるため、今回の結論はproduction既定値の
変更を要求しない。W-A/W-B checkpointはいずれもproduction checkpointへ接続していない。

### 方針管理チャットで判断が必要な事項

1. 本結果（W-B不採用、W-A維持）を正式な暫定class weightとして確定してよいか。
2. 依頼書9章の分岐「W-BでFPは減るがrecall/TP0が悪化」に該当するため、中間weight
   （例: `[0.25, 1.75]`、weight比7）を追加で試すべきか、それとも今回の結果（3倍程度の弱いweightで
   即座に崩壊）から見て、これ以上弱いweightの探索は打ち切りS5-14（座標依存・frame-level診断）へ
   進むべきか。
3. class weightは今回の2 armで判断でき、S5-12同様に「判定不能につき追加検証」とはならなかった。
   S5-13を完了として扱い、次の実装をS5-14の指示（または別の優先事項）として進めてよいか。

再学習、label policy/normalization/threshold/aggregation設定の変更、S5-07〜S5-11の再実行は、
本報告時点では実施していない。
