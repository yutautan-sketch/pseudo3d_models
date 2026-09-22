# Stage 5 S5-12: GroupNorm固定Label Policy Ablation 実装引き継ぎプロンプト

作成日: 2026-09-13

Stage 5の次の改修フローとして、S5-12のBBox non-contour label policy ablationを実装してください。
本件ではS5-11で有望と確認されたGroupNormを実験条件として固定し、BBox内かつpositive contour外の点を
ignoreにする場合とbackgroundとして学習する場合を比較します。

これはlabel policyの診断的比較です。production既定値の変更、threshold tuning、class weight比較、
長期学習はまだ行いません。

## 1. 最初に読む文書

次の順で確認してください。

1. `docs/stage5/stage5_revision_management_record.md`
   - 現在状態
   - S5-10、S5-11、S5-12
2. `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
   - 9.5節の実装事項C/D
   - 9.6節の実装事項E（S5-12の正本）
3. `docs/stage5/s5-11/stage5_s5_11_report_to_policy_chat.md`
   - S5-11完了報告
   - 特に13章の判断待ち事項
4. `docs/stage5/s5-11/stage5_s5_11_groupnorm_implementation_handoff_prompt.md`
   - GroupNorm実装時の固定条件と非対象
5. `docs/stage5/FILES.md`
6. `docs/stage5/data_construct.md`
7. Stage 4 teacher v6のlabel/provenance実装
   - `Stage2to4/pseudo3d/batch/annotation/batch_import_stage4_phase5_fullvideo_cvat.py`
   - `Stage2to4/pseudo3d/annotation/apply_deleted_xml_invalidations.py`
   - `Stage2to4/checks/stage4/check_stage4_bbox_ranked_label_policy.py`

現在の進捗・意思決定は`../stage5_revision_management_record.md`、S5-12の詳細仕様は
`../stage5_pointnext_s_training_evaluation_report.md`の実装事項Eを正としてください。

## 2. 現在の基準条件

- teacher: Stage 4 teacher v6
- model: official OpenPoints PointNeXt-S Stage 5 wrapper
- normalization: GroupNorm、8 groups（S5-12実験で暫定採用）
- input features: `intensity,confidence`
- window: frame size 16 / stride 8 / tailあり
- physical batch size: 1 window
- gradient accumulation: 8 windows、point-weighted normalization
- padding: PointNeXtへ入力しない
- dropout: 0.0
- loss: CrossEntropyLoss、label smoothing 0.0
- optimizer: AdamW、lr `1e-3`、weight decay `1e-4`
- inference: `model.eval()`
- aggregation: mean probability
- threshold: 0.5

基準のS5-11 GroupNorm pilot:

```text
/mnt/data/3d_projects/stage5_runs/260912/
pointnext_s_EX260912_260711_w16_s8_bboxrankv6_cvatxmlinv_glocal_
ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad/
```

- train: 163 files / 729 windows
- validation: 18 files / 79 windows
- seed: 42
- GroupNorm `best.pt`: epoch 4
- validation F1 / IoU / recall: `0.0878 / 0.0459 / 34.11%`
- validation TP0: `0/18`
- validation FP / FPR: `446,335 / 6.86%`

GroupNorm初期checkpointは、S5-11 Step D5で作成した
`stage5_pointnext_s_s3dis_partial_init_groupnorm.pt`を使用します。実際の絶対pathは既存transfer script、
S5-11 run config、ユーザー実機の出力を確認して決定し、推測で別checkpointを使用しないでください。

## 3. 方針管理チャットの正式判断

`docs/stage5/s5-11/stage5_s5_11_report_to_policy_chat.md` 13章の3事項について、以下を正式判断とします。

### 判断1: GroupNormを暫定採用する

GroupNormをS5-12以降の**実験用normalization方式として暫定採用**する。

根拠:

- train/eval logitsが完全一致
- validation TP0が`10/18 -> 0/18`
- validation median F1が`0 -> 0.0818`
- precisionとrecallが同時に改善
- validation lossの単調悪化が解消
- train sanityでもTP0が`1/3 -> 0/3`

これは小さな集約F1の変化だけではなく、動画単位でも一貫した改善である。LayerNorm比較は行わず保留する。

ただし、現時点では**production既定値をGroupNormへ変更しない**。S5-12以降の実験では明示的に
GroupNormを指定し、最終的なproduction採用はS5-15で決定する。

### 判断2: FP増加は未解決だが、threshold tuningは後回しにする

FPは`85,785 -> 446,335`、FPRは`1.32% -> 6.86%`であり、現時点でproduction上許容可能とは
判断しない。一方、次の改善があるためGroupNormを棄却する理由にもならない。

- recallは約8.8倍
- FP/FPRは約5.2倍
- precisionも`3.04% -> 5.04%`へ改善
- TP0動画が解消

単純なpositive確率の一様シフトではなく、検出能力の改善を含むと判断する。

S5-12のlabel policyとS5-13のclass weightはscore分布とFPを直接変えるため、S5-11直後にはthreshold
tuningを行わない。当面は`threshold=0.5`を固定する。

FP問題は次の順で扱う。

```text
S5-12 Label policy
  -> S5-13 class weight
  -> 必要ならStage 6入力としてのFP形状評価
  -> S5-15でthreshold sweep
```

### 判断3: 200 epochへ直接進まず、段階的な中間確認を行う

```text
S5-12 各label policyを5 epoch比較
  -> 採用policyを決定
S5-13 class weightを5 epoch比較
  -> 採用weightを決定
S5-14 必要な構造診断
  -> 5〜10 epoch最終pilot
S5-15 50 epoch中間判定
  -> 合格した場合のみ100〜200 epoch
```

50 epoch時点では次を確認する。

- validation lossが継続的に崩壊していない
- TP0動画数が再増加していない
- video-level median F1/IoUが5 epoch時点から悪化していない
- FP/FPRが学習とともに発散していない
- train sanityとvalidationの改善方向が大きく乖離していない

## 4. S5-12の目的と仮説

現行teacher v6では、BBox内かつpositive contour外の点が`point_label=-1`、`valid_mask=False`となり、
lossへ含まれません。この領域をpositiveと予測しても直接のpenaltyがないため、輪郭周辺FPの一因に
なっている可能性があります。

GroupNorm、初期parameter、split、seed、window、loss、class weight、threshold、aggregationを固定し、
BBox non-contour点をbackgroundとして学習へ加えることで、positive contourの検出を維持しながらFPを
抑制できるかを調べます。

## 5. 比較する2つのpolicy

同じteacher v6 H5を入力とし、source H5自体は変更・複製しません。

| Run | BBox内かつpositive contour外 | その他のlabel |
| --- | --- | --- |
| A: `bbox_noncontour_ignore` | `point_label=-1`、`valid_mask=False`を維持 | source v6のまま |
| B: `bbox_noncontour_background` | `point_label=0`、`valid_mask=True`へ変換 | source v6のまま |

Run Bで変更できるのは、teacher v6 provenance上「保存BBox内かつauthoritative/teacher positive外」と
確認できた点だけです。

次は必ず維持してください。

- CVAT authoritative positiveはBBox外またはno-BBox frameでもpositiveのまま
- no-BBoxという理由だけで既存labelをbackgroundへ上書きしない
- XML invalidation済みframeは既存の全backgroundを維持
- sourceの既存backgroundとpositive indexは維持
- points/features/frame metadata/window構造は維持

## 6. label policy実装方針

Stage 4 H5を書き換えず、Stage 5 Datasetでsource labelからeffective training labelを生成します。

推奨構成:

- Dataset、class count/debug集計、checkerから共用するlabel policy helperを追加
- training CLIへBBox non-contour専用policy optionを追加
- 未指定時は既存のignore policyを既定値とする
- policyと集計値を`config.json`、checkpoint config、起動ログ、metricsへ保存

命名は対象領域を明確にし、全てのignore理由を無差別に変換する一般的な
`ignore_to_background`にはしないでください。入力配列をin-place変更せず、source labelとeffective labelを
区別してください。

Runごとに最低限記録するもの:

- source/effective positive、background、ignore点数
- `-1 -> 0`変換点数
- policy対象frame/window数
- split別valid比、positive比、ignore比
- effective loss normalizer

class count処理がDatasetと異なるlabelを数えないよう、同じpolicy helperを利用してください。

## 7. teacher v6 provenance preflight

実学習前にteacher v6の全181 H5（train 163 + validation 18）を監査します。

最低限のassertion:

1. `valid_mask == (point_label != -1)`
2. policy対象maskとsource ignore点の関係を件数付きで記録
3. policy対象点が保存BBox内かつpositive contour外
4. Run Bでもpositive点数とpositive indexが完全一致
5. Run Bで変わる値が対象点の`-1 -> 0`と`False -> True`だけ
6. points、features、frame_order、point_indices、window一覧がRun A/Bで完全一致
7. 対象外ignore理由があれば変換せず、実学習前にfail-fastまたは明示的に別区分化

H5 schemaからignore理由を直接判定できない場合は、`frame_annotation/bbox_local_xyxy`、
`point_cloud/pixel_xy`、authoritative positive labelからmaskを再構築してください。全ignoreが
BBox non-contourであると監査で証明できた場合に限り、`point_label == -1`を対象maskとして簡略利用できます。

監査が成立しない場合、便宜的に全ignoreをbackground化して先へ進まないでください。発見したlabel理由、
件数、選択肢、推奨案をユーザーへ報告してください。

## 8. class weightの固定方法

方針上は「GroupNorm、threshold 0.5、auto class weight条件を固定」としています。ただしRun Bでは有効
background点が増えるため、文字列`auto`を各runで再計算するとresolved weightまで変わり、label policyと
class weightの二要因比較になります。

S5-12では、S5-11 GroupNorm controlで`auto`から解決済みの同一数値を両runへ明示指定してください。

```text
[0.05963856, 1.94036150]
```

つまり「auto baseline由来のclass weightを固定する」という意味です。Run A/Bのsource/effective class
countは診断値として記録しますが、weightを再計算しません。class weight自体の比較はS5-13で行います。

## 9. 固定する実験条件

| 項目 | 固定値 |
| --- | --- |
| teacher | Stage 4 teacher v6、同一H5 |
| split | 既存GroupNorm pilotと同じtrain 163 / validation 18 |
| initialization | 同一GroupNorm S3DIS部分転移checkpoint |
| normalization | `groupnorm`、8 groups |
| seed | 42 |
| window | size 16 / stride 8 / tailあり |
| physical batch | 1 window |
| gradient accumulation | 8 windows、point-weighted normalization |
| features | `intensity,confidence` |
| loss | CrossEntropyLoss |
| class weight | `[0.05963856, 1.94036150]` |
| label smoothing | 0.0 |
| optimizer | AdamW、lr `1e-3`、weight decay `1e-4` |
| dropout | 0.0 |
| epochs | 5 |
| inference | `model.eval()` |
| aggregation | mean probability |
| threshold | 0.5 |

既存`train_files.txt`/`val_files.txt`を明示的に再利用し、再splitしないでください。Run A/Bは同じ
code revisionと実行環境で両方実行します。既存S5-11 GroupNorm pilotは外部referenceとしてのみ残します。

## 10. 共通評価label

Run AとRun Bではnative `valid_mask`が異なるため、native validation loss/F1を直接比較できません。
同一predictionを次の共通targetで評価する専用ablation evaluator/checkerを用意してください。

1. `canonical_v6`
   - source teacher v6のvalid点だけを評価
   - 既存`evaluate_stage5.py`とのmean baseline parityを確認
2. `bbox_noncontour_as_background`
   - 監査済みBBox non-contour点をbackgroundかつvalidとして加えた同一target
   - S5-12のprimary comparison
3. `bbox_noncontour_region`
   - 対象領域だけのpredicted positive count/rateと`prob_femur`分布

positive contourのGT indexは全targetで同一にします。既存valid backgroundとBBox non-contourを分けて
集計し、FP低下がpositive予測全体の抑制か、対象hard negativeでの選択的改善かを確認してください。

## 11. checkpoint比較

policyごとにnative validation targetが異なるため、各runの`best.pt`だけを主比較にしません。

- primary: 同じ学習量のepoch 5固定checkpoint
- secondary: 各runの`best.pt`
- `save_every=1`で各epoch checkpointを保存
- best epoch、best metric、native targetの違いを記録

epoch 5固定checkpointが既存命名で保存されない場合は、`last.pt`とのepoch一致を検証し、比較artifactで
明示してください。

## 12. 実装・検証手順

### Step E1: 既存コード監査と実装方針

- Dataset/H5 loader、class count、training config、evaluation metricsを確認
- teacher v6のignore provenanceを確認
- 既存GroupNorm実装とcheckpoint復元を変更しない範囲を確認
- 短い実装方針をユーザーへ提示

### Step E2: synthetic label policy test

- Run Aがsource labelの完全なno-op
- Run Bが対象ignoreだけをbackground化
- positive index、points、features、frame/windowが不変
- 対象外ignoreを拒否または維持
- class countがeffective labelと一致

### Step E3: teacher v6全件preflight

- 全181 H5を監査
- 対象mask、ignore理由、変換予定点数をsplit別に出力
- assertionに失敗した場合は学習へ進まない

### Step E4: Dataset/batch parity checker

- Run A/Bでsample数、window順、point indices、points/featuresを比較
- 変化がlabels/valid_maskの対象点だけであることをhash/assertで確認
- Run Aが既存Datasetと完全一致することを確認

### Step E5: 1 epoch smoke

- Run BをGroupNorm、固定weight、既存splitで1 epoch実行
- loss/gradient/metrics/checkpointがfinite
- 変換件数、loss normalizer、effective label countを確認
- strict checkpoint reloadを確認

### Step E6: Run A/B 5 epoch

- 同じ初期checkpointとseedから両方を各5 epoch実行
- 毎epoch checkpointを保存
- 実行順による設定差がないようconfig diffを検証

### Step E7: 共通target評価

- epoch 5とbest checkpointを評価
- train sanity 3動画 + validation 18動画
- `canonical_v6`、`bbox_noncontour_as_background`、`bbox_noncontour_region`
- video-level集計とpolicy差分を出力

### Step E8: PLY可視化

- 固定train sanityと代表validationを同じ座標で出力
- GT、Run A/Run B全prediction、positive-only PLYを用意
- 輪郭周辺FPと大腿骨positive欠損を比較可能にする
- 実データPLYは共有用bundleへ含めない

## 13. 記録するmetrics

共通targetごとに次を記録してください。

- TP、FP、TN、FN
- precision、recall、F1、femur IoU
- FPR、FNR
- predicted positive count/rate
- TP0動画数
- video-level mean/median F1・IoU
- split別・動画別のRun B minus Run A差分

BBox non-contour regionでは次を追加します。

- 対象点数
- predicted positive count/rate
- `prob_femur` mean、p50、p90、p95、p99
- 動画別の改善/悪化方向

学習側では、追加background点がweighted loss denominatorへ占める割合、source/effective class比、
ignore比、windowごとの追加点分布も記録してください。

## 14. 判定基準

Run Bは次を満たす場合に有望とします。

- 共通`bbox_noncontour_as_background` targetで対象領域のpositive予測率が低下
- 全体FP/FPRも低下
- canonical positive contourのrecallを大きく失わない
- TP0動画数が再増加しない
- video-level median F1/IoUが大きく悪化しない
- train sanityとvalidationの方向が極端に乖離しない

FPが減ってもrecall collapseやTP0再発が大きい場合は採用しません。split集約だけが改善し、動画別の
符号が揃わない場合も系統的改善とは判定しません。

結果分岐:

1. Run Bが有望:
   BBox non-contour backgroundをS5-13以降の暫定training policyとする。
2. Run Bが悪化:
   Run Aのignoreを維持する。
3. Run A/B差が小さい:
   BBox non-contour ignoreをGroupNormの主要FP原因から下げ、Run Aを維持する。
4. 判定不能:
   事実、交絡、必要な追加検証を方針管理チャットへ返し、productionを変更しない。

いずれの場合も採用policyとGroupNormを固定してS5-13のclass weight比較へ進みます。threshold tuningは
score分布が定まった後、S5-15で行います。

## 15. 非対象・禁止事項

本件では次を行いません。

- production既定normalizationのGroupNorm化
- source Stage 4 H5の上書き
- threshold tuningまたはthreshold 0.5以外の比較
- class weightの再計算・比較
- GroupNorm group数や学習率のtuning
- aggregation/window/feature/teacher変更
- Dice、Focal、Hard Negative Mining追加
- 座標augmentation、center-only loss、overlap weight追加
- 50〜200 epoch学習
- LayerNorm比較
- unrelatedな既存変更のrevert

worktreeはdirtyであり、本件以外のユーザー変更が存在します。対象外の変更を削除・revertしないでください。

## 16. privacyと出力

共有用metricsは既存ルールに従って匿名化し、元動画ID、H5 path、座標、PLY、checkpointを含めないで
ください。

- share: 匿名video IDを使うCSV/JSONのみ
- private: 元ID mapping、絶対path、checkpoint、PLY
- canonical parity不一致やpreflight失敗の詳細はprivateへ保存
- anonymization self-checkを実行

## 17. 文書更新と方針管理チャットへの報告

実装・検証後は次を更新してください。

1. `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
   - 実装事項Eへ実施日、実装、preflight、smoke、5 epoch結果、仮説判断を追記
2. `docs/stage5/stage5_revision_management_record.md`
   - S5-11の暫定採用判断
   - S5-12の進捗、結果、次の分岐
   - 新しいdecision record
3. `docs/stage5/FILES.md`
   - 新規helper/checker/script/artifact構造

最後に`docs/stage5/s5-12/`へ方針管理チャット用の完了報告Markdownを作成してください。最低限、次を含めます。

- 変更ファイル
- teacher v6 provenance preflight結果
- 対象点・変換点数
- Dataset/batch parity
- class weightが両runで同一であること
- Run A/Bのconfig差分
- 1 epoch smokeと5 epochの実行条件
- epoch 5固定checkpointの共通target metrics
- best checkpointのsecondary metrics
- BBox non-contour regionの確率・positive率
- TP0、video median、動画別改善/悪化
- PLY目視確認用のprivate出力先
- 支持/棄却/未確定の仮説
- production未変更の確認
- S5-13へ持ち越すpolicy候補
- 方針管理チャットで判断が必要な事項

まず関連文書と既存コードを確認し、S5-12の実装方針を短く提示してください。重大な仕様上の不明点が
なければ、helper/CLI/checker/evaluator、静的・synthetic testまで実装し、teacher v6実データpreflightと
GPU学習に必要なコマンドをユーザーへ提示してください。preflightでprovenance契約を証明できない場合は、
推測で全ignoreを変換せず、その時点で相談してください。
