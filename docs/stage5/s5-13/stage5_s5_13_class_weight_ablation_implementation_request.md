# Stage 5 S5-13: GroupNorm固定Class Weight Ablation 実装依頼書

作成日: 2026-09-14

Stage 5の次の改修フローとして、S5-13 class weight ablationを実装・検証してください。
S5-12でteacher v7、GroupNorm、BBox non-contour label policyの比較が完了したため、本事項では
Run A（`bbox_noncontour_ignore`）を暫定policyとして固定し、class weightだけを変更して
globalなFP/FPRとpositive recallのtrade-offを比較します。

これは5 epochの診断的比較です。production既定値、threshold、aggregation、loss形式、teacher、
normalization、label policyはまだ変更しません。

## 1. 最初に読む文書

次の順で確認してください。

1. `docs/stage5/stage5_revision_management_record.md`
   - 現在状態、S5-12、S5-13
   - Decision record D-018〜D-022
2. `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
   - 9.6節「実装事項E」のS5-12実装・結果・正式判断
   - 9.7節のclass weight方針
3. `docs/stage5/s5-12/stage5_s5_12_report_to_policy_chat.md`
   - S5-12完了報告と共通target metrics
4. `docs/stage5/s5-12/stage5_s5_12_label_policy_implementation_handoff_prompt.md`
   - S5-12で固定した実験条件と比較上の注意
5. `docs/stage5/FILES.md`
6. `docs/stage5/data_construct.md`
7. class weightとloss normalizationの現在実装
   - `Stage5/train_stage5.py`
   - `Stage5/train_stage5.sh`
   - `Stage5/stage5/training/losses.py`

現在の進捗と採用判断は`../stage5_revision_management_record.md`、数値根拠は
`../stage5_pointnext_s_training_evaluation_report.md`を正本としてください。

## 2. 方針管理チャットの正式判断

### 判断1: Run A（ignore）をS5-13の暫定label policyとする

S5-12 Run B（`bbox_noncontour_background`）はBBox non-contour領域のpositive予測率を
validationで40.63%から31.13%へ低下させましたが、全体FPRは11.71%から12.17%へ改善しませんでした。
対象領域はvalid background全体の約0.3%であり、global FP問題の主要因とは判断できません。

S5-13では、後方互換でvalidation aggregated F1/IoUとvideo別勝敗が優位だったRun A
（`bbox_noncontour_ignore`）を暫定採用します。Run Bは診断用alternateとして保持しますが、
S5-13の主比較へ混ぜません。

### 判断2: S5-12を長期化せずclass weight ablationへ進む

Run A/Bを100〜200 epochへ延長しません。次に、全valid点のpositive/negative gradient比へ直接作用する
class weightを比較します。thresholdは0.5、aggregationはmean probabilityのまま固定します。

### 判断3: S5-07〜S5-11をteacher v7で全面再実行しない

S5-07〜S5-11のteacher v6結果は構造診断の履歴として保持します。teacher v7のS5-12 Run Aを今後の
比較baselineとし、将来特定の仮説でteacher差が交絡する場合だけ対応する診断を個別再実行します。

## 3. S5-13の目的と仮説

teacher v7のtrain valid点ではpositiveが少なく、S5-12で使ったauto由来weight
`[0.05963856, 1.94036150]`はpositiveをbackgroundの約32.5倍に重み付けします。この強いweightは
minority positiveの学習とTP0抑制に役立つ一方、GroupNorm条件で残る広範なFP/FPRを増やしている
可能性があります。

S5-13では、S5-12 Run Aの強いweightと、より弱い`[0.5, 1.5]`（positive/background比3）を比較し、
次を検証します。

1. 弱いweightがglobal FP/FPRとpredicted positive率を下げるか。
2. FP低下が単なるpositive予測崩壊ではなく、precision/F1/IoU改善を伴うか。
3. positive recall、TP0動画数、video median F1/IoUを許容不能な程度に悪化させないか。
4. train sanityとvalidationで改善方向が一致するか。

## 4. 必須比較条件

### W-A: strong auto-derived control

```text
class weight = [0.05963856, 1.94036150]
positive/background weight ratio = 約32.54
```

S5-12 Run A（teacher v7、`bbox_noncontour_ignore`、5 epoch）を新しいcomparison baselineとして
使用します。既存runを再利用する前に、専用checkerまたはconfig比較で第6節の全固定条件が一致する
ことを証明してください。

### W-B: moderate fixed weight

```text
class weight = [0.5, 1.5]
positive/background weight ratio = 3.0
```

W-Aと同一の初期checkpoint・split・seedから5 epoch学習します。weightの平均は1であり、絶対scaleより
class間比率の効果を主に比較できます。

### 初回比較に含めない条件

`[1.0, 1.0]`（no class weight）、別epsilonのauto weight、Focal/Dice lossは初回の必須armに
含めません。W-A/W-Bで方向性を判定できない場合のみ、方針管理チャットへ結果と追加候補を報告して
ください。実装チャットの判断だけでarmを増やさないでください。

## 5. S5-12 Run Aをcontrolとして再利用する条件

S5-12 Run Aは、次がW-Bと一致すると機械的に確認できた場合に限り再利用できます。

- teacher v7 inventoryとH5 pattern
- train 162 / validation 18のfile list内容と順序
- initialization checkpointのpathと、可能ならSHA-256
- model configとGroupNorm 8 groups
- label policy `bbox_noncontour_ignore`
- seed 42
- window size 16 / stride 8 / tailあり
- physical batch size 1 / gradient accumulation 8
- point-weighted loss normalization
- input features `intensity,confidence`
- CE、label smoothing 0.0
- AdamW、lr `1e-3`、weight decay `1e-4`
- dropout 0.0
- schedulerその他のoptimizer挙動
- training code revisionに学習挙動を変える差分がないこと

configだけで確認できない項目は、runの`train_files.txt`、`val_files.txt`、checkpoint config、ログ、
git diffを使って確認してください。不一致または確認不能な項目があれば、W-Aも同じ現在revisionから
5 epoch再実行し、既存S5-12 Run Aはexternal referenceへ下げてください。

## 6. 固定する実験条件

| 項目 | 固定値 |
| --- | --- |
| teacher | Stage 4 teacher v7 `bboxrank_v7_cvat_authoritative_crop_quality_v1` |
| inventory | 180 H5 |
| split | S5-12 Run Aと同一のtrain 162 / validation 18 |
| label policy | `bbox_noncontour_ignore` |
| initialization | S5-12と同一のGroupNorm S3DIS部分転移checkpoint |
| model | official OpenPoints PointNeXt-S Stage 5 wrapper |
| normalization | `groupnorm`、8 groups |
| seed | 42 |
| features | `intensity,confidence` |
| window | size 16 / stride 8 / tailあり |
| physical batch | 1 window |
| gradient accumulation | 8 windows、point-weighted normalization |
| padding | PointNeXt入力へ追加しない |
| loss | CrossEntropyLoss |
| label smoothing | 0.0 |
| optimizer | AdamW、lr `1e-3`、weight decay `1e-4` |
| dropout | 0.0 |
| epochs | 5 |
| inference mode | `model.eval()` |
| aggregation | mean probability |
| threshold | 0.5 |

比較中にaugmentation、sampling、window、optimizer、scheduler、normalization group数などを変更しないで
ください。

## 7. 実装要件

### 7.1 training shellのrun識別

現在の`train_stage5.sh`は`CLASS_WEIGHT`を環境変数で上書きできますが、`PREFIX`に
`auto_weight`が固定で含まれている可能性があります。manual weightでも出力先がautoと表示されたり、
既存runを上書きしたりしないよう修正してください。

最低限、次を満たしてください。

- `CLASS_WEIGHT`からfilesystem-safeなweight tagを決定する。
- `auto`、none、manual weightをrun名と起動ログで区別する。
- `OUTPUT_DIR`または`EXPERIMENT_NAME`を環境変数で明示overrideできる。
- 既存出力directoryが非emptyなら、明示的なresume/overwrite指定なしに開始しない。
- resolved weight、requested mode、class countsを`config.json`とcheckpoint configへ保存する。

manual weight tagの例:

```text
cw_auto_resolved_0p05963856_1p94036150
cw_manual_0p5_1p5
cw_none
```

既存のproduction既定値`CLASS_WEIGHT=auto`自体は変更しないでください。

### 7.2 比較checker

S5-12 Run AとW-Bの比較条件を検証するcheckerを追加してください。既存のchecker配置規則に従い、
例えば次の場所を使用します。

```text
Stage5/checks/real_h5/check_stage5_class_weight_ablation.py
Stage5/checks/real_h5/check_stage5_class_weight_ablation.sh
```

checkerは少なくとも次をfail-fastで検査してください。

1. train/val file listの内容と順序が一致する。
2. teacher schema、label policy、normalization、model、features、window、seed、optimizer設定が一致する。
3. initialization checkpointが一致する。
4. 変更対象がclass weightだけである。
5. W-A/W-Bのrequested/resolved weightが期待値と一致する。
6. 両runがepoch 5まで完走し、loss/metricsがfiniteである。
7. primary checkpointが同じ学習量のepoch 5である。
8. evaluation対象動画とpoint集合が一致する。
9. 既存`evaluate_stage5.py`のmean baseline parityが成立する。

意図した差としてclass weight、run/output metadata、そこから派生するmetrics/checkpoint hashだけを許可し、
それ以外のconfig差分を一覧化してください。

### 7.3 weighted-loss診断

各weight条件について、train/validation split別に少なくとも次を記録してください。

- background/positive valid point countsとfrequency
- resolved class weightsとweight ratio
- background/positiveそれぞれのweighted denominator contribution
- positiveがweighted CE denominatorに占める割合
- epoch別loss、TP、FP、TN、FN、precision、recall、F1、IoU、FPR、FNR
- predicted positive count/rate
- optimizer step数、empty-valid window数、nonfinite count

既存metricsに同等の情報がある場合は再実装せず再利用してください。新しい集計を追加する場合は、
Datasetのeffective labelと同じ`bbox_noncontour_ignore` policyを使い、H5 raw labelとの不一致を
生じさせないでください。

## 8. 実装・検証手順

次の順序で進めてください。

### Step F1: 現状監査

- S5-12 Run Aのrun directory、checkpoint、config、file listsを特定する。
- `train_stage5.py`のmanual/auto/none weight処理とpoint-weighted gradient accumulationを確認する。
- `train_stage5.sh`のrun naming、output collision、env overrideを確認する。
- 既存機能で満たせる項目と追加実装が必要な項目を整理する。

### Step F2: shell/config/checker実装

- class weightを正しく表すrun tagと安全なoutput path処理を実装する。
- class weight比較checkerと必要最小限の集計を実装する。
- production既定値を変えない。

### Step F3: CPU/static test

- manual weight parseとrun tag生成をsynthetic inputで確認する。
- config parity checkerのpass/failケースを確認する。
- `py_compile`、`bash -n`、`git diff --check`を実行する。

### Step F4: control再利用判定

- 第5節の条件をS5-12 Run Aと現在のW-B設定で比較する。
- 全条件一致ならS5-12 Run AをW-Aとして再利用する。
- 不一致または確認不能ならW-Aを現在revisionで5 epoch再実行する。
- 判定理由と比較結果をmachine-readable JSON/CSVへ保存する。

### Step F5: W-B 1 epoch smoke

`CLASS_WEIGHT="0.5,1.5"`で1 epoch smokeを行い、次を確認します。

- teacher v7、split 162/18、Run A label policy
- GroupNorm 8 groups
- resolved weight `[0.5, 1.5]`
- loss denominator、gradient、metricsがfinite
- checkpoint/config/metricsが保存される
- output pathがW-Aや既存runと衝突しない

### Step F6: W-B 5 epoch pilot

W-Aと同一初期checkpoint・split・seedでW-Bを5 epoch学習します。primary比較用としてepoch 5の
`last.pt`または固定epoch checkpointを必ず保存してください。`best.pt`もsecondaryとして保存し、
best epochと選択metricを記録します。

### Step F7: 固定評価

W-A/W-Bのepoch 5 checkpointを次へ適用します。

- S5-12と同じ固定train sanity 3動画
- 同じvalidation 18動画全件
- mean probability aggregation
- threshold 0.5
- `model.eval()`

label policyと評価targetは同一なので、通常のteacher v7 native valid labelをprimary targetとします。
必要ならBBox non-contour region metricsを補助診断として残せますが、S5-13の採否はglobal metricsで
判断してください。

### Step F8: 集計と報告

split aggregateとvideo-levelの両方を出力します。

- TP、FP、TN、FN
- precision、recall、F1、IoU、FPR、FNR
- predicted positive count/rate
- TP0動画数
- video-level mean/median F1・IoU・precision・recall・FPR
- 動画別W-A/W-B勝敗数
- epoch別training/validation metrics
- bestとepoch 5の差
- class weight別weighted denominator構成

S5-12と同じprivacy契約で匿名化bundleを作り、実video IDや絶対入力pathをshareable metricsへ含めないで
ください。PLYは必須ではありません。数値が拮抗する場合やFP形状の違いが採否に関係する場合だけ、
固定aliasについて同じ座標・色規則で出力してください。

## 9. 判定基準

primary checkpointは同じ学習量のepoch 5とし、`best.pt`だけで異なるepochを比較しないでください。

W-Bを有望とするには、validationで次を総合的に満たす必要があります。

1. FP/FPRとpredicted positive率がW-Aより明確に低下する。
2. precision、F1、IoUの少なくとも主要なvideo-level統計が改善する。
3. recallがW-Aから大幅に低下しない。
4. TP0動画数がW-Aの`1/18`から増加しない。
5. train sanityとvalidationの改善方向が大きく反転しない。
6. 改善が少数動画だけでなく、video別勝敗またはmedianでも支持される。

FP低下と引き換えにrecallが崩壊する場合、W-Bは採用しません。aggregateだけ改善してvideo median、
TP0、動画別勝敗が悪化する場合も系統的改善とは判断しません。

結果による分岐:

- **W-Bが明確に有望:** S5-14以降の暫定weight候補として方針管理チャットへ提案する。
- **W-Aが優位:** strong auto-derived weightを維持する。
- **W-BでFPは減るがrecall/TP0が悪化:** 初回結果を報告し、必要なら中間weight
  （例: `[0.25,1.75]`）を追加するか方針管理チャットで判断する。
- **差が小さい、またはsplit/video-levelで符号不一致:** 無理に採用せず判定不能とし、追加seed、
  no-weight、長期学習のどれが必要かを方針管理チャットへ返す。

## 10. 非対象と禁止事項

本事項では次を行わないでください。

- label policy Run Bの再学習またはS5-12の長期化
- teacher v6への復帰
- S5-07〜S5-11の一律再実行
- production既定normalization/label policy/class weightの変更
- threshold tuning
- mean以外のproduction aggregation採用
- GroupNorm group数変更またはLayerNorm比較
- Focal loss、Dice loss、Hard Negative Mining
- learning rate、scheduler、window、sampling、feature、augmentation変更
- 50〜200 epoch学習
- source H5の書き換え

比較中に別の不具合を発見した場合は、比較を成立させる最小修正と追加実験を分離してください。
結果へ影響する修正後は両armの公平性を再確認し、片方だけを旧revisionのまま比較しないでください。

## 11. productionへの扱い

S5-13は実験条件の選定であり、完了だけを理由にproduction既定値を変更しません。

- production normalization既定値はS5-15まで現状維持
- production label policy既定値は現状維持
- threshold 0.5、mean aggregationを維持
- 採用weightはS5-14の最終pilotとS5-15の50 epoch中間判定を経て確定

## 12. 完了条件と成果物

完了時には次を満たしてください。

1. class weight以外のconfig parityが機械的に確認されている。
2. W-A再利用可否と根拠が記録されている。
3. W-B 1 epoch smokeと5 epoch pilotが完走している。
4. epoch 5固定checkpointでtrain sanity 3件・validation 18件を評価している。
5. aggregate、video-level、TP0、weighted denominator診断が揃っている。
6. mean baseline parityとprivacy checkが合格している。
7. 実装・検証結果を次へ追記している。
   - `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`のS5-13実装事項
   - `docs/stage5/stage5_revision_management_record.md`のS5-13
   - 必要に応じて`docs/stage5/FILES.md`
8. 方針管理チャット向け完了報告を次へ作成している。

```text
docs/stage5/s5-13/stage5_s5_13_report_to_policy_chat.md
```

完了報告には、実装ファイル、実行run/checkpoint、config parity、全主要metrics、動画別傾向、
仮説判断、production未反映事項、次の判断事項を記載してください。追加runやproduction変更は、
方針管理チャットの正式判断を待ってください。

