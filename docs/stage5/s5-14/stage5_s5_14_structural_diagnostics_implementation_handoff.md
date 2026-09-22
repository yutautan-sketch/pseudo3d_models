# Stage 5 S5-14: 座標依存・Frame-level・Overlap Exposure診断 実装依頼書

作成日: 2026-09-14

座標provenance調査追記: 2026-09-15

S5-13本比較とS5-13補足が完了したため、次の改修段階S5-14を実装・検証してください。

S5-14 coreは、train sanity PLYで観察された「似たXY位置のpositive集合が異なるframe群へ反復する」
現象について、座標事前分布、動画内時間位置、overlap windowによるtraining exposureの3仮説を
切り分ける診断段階です。core結果が出るまで再学習、loss変更、sampling変更、augmentation追加を
行わないでください。

## 1. 最初に読む文書

次の順で確認してください。

1. `docs/stage5/stage5_revision_management_record.md`
   - S5-13、S5-13補足、S5-14
   - Decision record D-026〜D-029
2. `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
   - 冒頭のtrain sanity定性評価
   - 9.4節のframe-level診断案
   - 9.5節のoverlap aggregation検証
   - 9.7節のS5-13本比較・補足結果
3. `docs/stage5/s5-13/stage5_s5_13_supplement_report_to_policy_chat.md`
   - W-A/W-B/W-Cのthreshold-free比較と判断事項
4. `docs/stage5/s5-13/stage5_s5_13_supplement_implementation_handoff.md`
   - S5-13補足の固定条件、parity、privacy要件
5. `docs/stage5/s5-08-09/stage5_overlap_aggregation_implementation_policy.md`
   - source point alignment、window accumulator、相対frame decileの既存仕様
6. `docs/stage5/FILES.md`
7. `docs/stage5/data_construct.md`
8. 関連コード
   - `Stage5/evaluate_stage5.py`
   - `Stage5/stage5/datasets/pseudo3d_pointcloud_dataset.py`
   - `Stage5/stage5/utils/frame_windows.py`
   - `Stage5/stage5/utils/feature_normalization.py`
   - `Stage5/checks/real_h5/check_stage5_overlap_aggregation.py`
   - `Stage5/checks/real_h5/check_stage5_class_weight_threshold_free.py`

現在の進捗と判断は`../stage5_revision_management_record.md`、過去の数値は
`../stage5_pointnext_s_training_evaluation_report.md`を正本としてください。

## 2. S5-13補足後の正式判断

### 判断1: W-Cを棄却しW-Aを維持する

**回答: W-Cを不採用とし、W-AをS5-14の暫定class weightとして確定します。**

W-C `[0.11764706,1.88235294]`はthreshold 0.5でvalidation FPRを11.71%から2.58%へ下げ、F1/IoUを
改善しましたが、recallは44.69%から17.52%、video median recallは44.68%から9.70%へ低下し、TP0は
1/18から5/18へ増加しました。validation AUPRC/AUROCおよび同一FPRでのrecallもW-Aを下回りました。

S5-14で使用するclass weight:

```text
W-A = [0.05963856, 1.94036150]
positive:background weight比 = 約32.54
```

これはS5-14の実験用暫定値であり、production採用の最終判断ではありません。

### 判断2: W-B/W-Cを診断履歴として保持する

**回答: checkpoint、config、prediction、metricsを診断履歴として保持します。**

W-B/W-CはS5-15でclass weightとcalibration/thresholdの関係を説明する資料として参照可能にします。
ただしactiveなproduction候補、S5-14の比較arm、追加学習の初期値にはしません。S5-14のprimary modelは
W-Aだけです。

### 判断3: 追加weightを試さずS5-14へ進む

**回答: weight比24などの追加中間weightは試しません。**

W-Cは同一FPRの全比較点でW-Aよりrecallが低く、AUPRC/AUROCも悪化しました。未使用の2回目の5 epoch枠は
消化せず、class weight探索を終了してS5-14へ進みます。threshold tuningはS5-15まで行いません。

## 3. S5-14の背景

train sanityのpositive-only PLYを側面から見ると、似たXY位置のpositive集合がpseudo-3DのZ方向、
すなわち異なるframe群へ2〜3回反復して見えました。

既に分かっていること:

- S5-04 batch integrityにより、Dataset、collate、shuffle、multi-workerで別sampleのGTやpointsが
  流用される不具合は強く除外されている。
- PLY上の反復点はexporterによる同一点複製ではなく、異なるframe由来のsource pointである可能性が高い。
- unique pointに対するwindow occurrence倍率はbackground約1.69、positive約1.90である。
- S5-08ではwindow境界距離とdisagreement率の相関は支持されなかった。
- 一方、動画内相対位置の後半ほどwindow disagreement率が高い傾向があった。
- 現行DatasetはH5全体のpointsを一度center/scaleしてからwindowを選択する。windowごとの再centerではなく、
  model入力には動画全体中心に対する相対XY/Z位置が残る。

したがって、見た目の反復を単純なbatch不具合として扱わず、座標、時間位置、window exposureを分離して
測定する必要があります。

## 4. 目的と検証仮説

W-Aの予測をsource pointとframeへ復元し、次の3仮説を再学習なしで検証します。

### 仮説A: 座標事前分布

GTの位置と十分に対応せず、modelが動画間で似たXY位置をpositiveにしやすい。特にno-GT frameや
GTから時間的に離れたframeでも同じXY binが繰り返しactiveになる可能性があります。

### 仮説B: 時間位置・frame phase

FP/FN、probability、window disagreementが動画内の特定区間、GT-positive区間の前後、動画後半などへ
偏る。これは座標よりも、時間とともに変わる見え方やwindow文脈で説明される可能性があります。

### 仮説C: Overlap exposure

source pointまたはframeがtraining windowへ含まれる回数の違いが、positive probability、FP/FN、
window disagreement、同一XY位置での時間的反復と対応する。positive classの高い重複倍率が、modelの
予測形状へ影響している可能性があります。

## 5. 固定baselineと対象範囲

| 項目 | 固定値 |
| --- | --- |
| teacher | v7 `bboxrank_v7_cvat_authoritative_crop_quality_v1`、180 H5 |
| model/checkpoint | S5-12/S5-13 W-A epoch 5、`best.pt`=`last.pt` |
| label policy | `bbox_noncontour_ignore` |
| normalization | GroupNorm、8 groups |
| class weight | `[0.05963856,1.94036150]` |
| window | size 16 / stride 8 / tailあり |
| inference mode | `model.eval()` |
| aggregation | mean probability |
| threshold | 0.5 |
| prediction対象 | 固定train sanity 3動画 + validation 18動画全件 |
| GT/exposure監査 | teacher v7全180 H5 |

W-B/W-Cは診断履歴として保存するだけで、S5-14 checkerのprimary入力にしないでください。

## 6. 座標系を必ず分離する

次を別々の列・artifactとして保持してください。

1. H5 raw `point_cloud/pixel_xy`
2. raw pseudo-3D `point_cloud/points`
3. model入力相当の`normalize_xyz(points)`
4. `point_cloud/frame_order`
5. 動画内相対frame位置

PLY上のraw XY反復とmodel入力座標上の依存を同一視しないでください。

動画間のXY比較には、推測値ではなく生成履歴で裏付けられたlocal image dimensionsを使って0〜1へ
正規化します。ただし、2026-09-14の実装調査により、teacher v7を含む最終combined H5にはcrop/imageの
幅・高さが直接保存されていないことが判明しました。一方、`point_cloud/pixel_xy`は
`local_encoder_images`上の座標であり、最終H5には元の中間H5を示す`source_h5`または
`source_pseudo3d_h5`が残る設計です。また、生成CLIの`local_crop_size`既定値は256ですが、これだけを
無検証の推測値として使用してはいけません。

このため、Step H3.1へ進む前にStep H2.5の座標provenance監査を行い、実寸法または生成run全体の共通寸法を
確定してください。既存`Stage5/stage5/utils/feature_normalization.py`の`normalize_pixel_xy()`は観測点の
最大値で除算するため、このcross-video解析には使用禁止です。寸法を確定できない動画はcross-video
heatmapから除外して件数と理由を記録します。同一動画内の解析はraw pixel座標でも継続できます。

## 7. 実装・検証手順

### Step H1: 現状監査とsynthetic test

既存checkerとprediction artifactの再利用可能範囲を確認してください。新規checkerは既存責務と重複させず、
必要なら次のような配置とします。

```text
Stage5/checks/real_h5/check_stage5_frame_spatial_overlap_diagnostics.py
Stage5/checks/real_h5/check_stage5_frame_spatial_overlap_diagnostics.sh
Stage5/checks/dummy/check_dummy_frame_spatial_overlap_diagnostics.py
Stage5/checks/dummy/check_dummy_frame_spatial_overlap_diagnostics.sh
```

synthetic testで少なくとも次を確認します。

- source `point_indices`への復元
- frame単位confusion counts
- GT/predicted positiveが空の場合のundefined処理
- XY重心・分散・距離
- normalized XY bin assignmentと境界値
- contiguous frame runの分割
- GT runとpredicted runの対応
- window exposure count
- vote countとwindow probability集約
- duplicate/missing pointのfail-fast
- nonfinite値のfail-fast

`py_compile`、`bash -n`、`git diff --check`も実行してください。

### Step H2: Teacher v7全件のGT・Exposure監査

GPU推論を使わず、teacher v7全180 H5を監査します。

記録項目:

- H5/videoごとのframe範囲とframe数
- frameごとのpositive/background/ignore点数
- BBox有無、GT-positive有無
- source pointごとのwindow occurrence count
- class別のunique point数とwindow occurrence総数
- class別exposure倍率
- frame相対位置decile別exposure
- train/validation split別および動画別統計
- positive frame runの数、長さ、間隔
- GT positiveのraw XY density（cross-video normalized densityはH2.5合格後に再計算する）

unique-point集計とwindow-occurrence集計を混ぜず、両方を明示してください。S5-08以前の値と異なる場合は、
teacher v6/v7、split、window設定の差を確認してから不具合と判断してください。

### Step H2.5: Stage 4 dataset inventoryとXY座標provenance監査

Step H1/H2の実装に続き、Step H3.1のcross-video XY解析へ進む前に本調査を実施してください。調査、監査
スクリプト、条件付き正規化の実装、synthetic test、結果報告まで実装チャットへ委任します。ただし、
teacher H5の書き換えや再生成、再学習、production変更は行わないでください。

#### H2.5-A: 保存dataset inventory

ユーザー実機の次のディレクトリを調べ、v2以降に保存されたStage 4 teacher datasetを列挙します。

```text
/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/
```

各datasetについて少なくとも次を記録してください。

- dataset/run名とteacher世代
- `annotated`、`collected`等のH5配置先
- H5件数
- 生成・派生pipeline
- 元datasetとの継承関係
- point cloud本体を再生成した世代か、annotation/labelだけを変更した世代か
- 利用可能なrun log、summary、manifest

絶対pathと実video IDを含むinventoryはprivate artifactとし、shareable summaryではdataset token、件数、
匿名video aliasだけを使用してください。

#### H2.5-B: H5属性とsource provenance

代表H5だけで結論を出さず、teacher v7全180 H5を主対象として次を監査してください。可能であれば対応する
v2以降の同一動画も照合します。

- 最終H5のroot属性と`point_cloud` group属性
- `source_h5`、`source_pseudo3d_h5`等のsource参照値
- 参照先中間H5の存在・可読性
- 中間H5の`local_encoder_images.shape`、`local_input_shape`
- `local_preprocess`、crop offset、resize scale、raw width/height
- `pixel_xy`のshape、finite性、軸別min/max
- `pixel_xy`がlocal image bounds内にあること
- 同一動画・世代間の`pixel_xy`、`points`、`frame_order`、point countの一致
- 必要に応じたdataset単位のchecksumまたはstreaming hash

巨大datasetを一括ロードせず、H5 datasetを必要な単位で読み、hashもchunk単位で計算してください。
source pathが古い絶対pathで解決できない場合は、basename/video対応による探索結果を別欄に記録し、暗黙に
同一ファイルとみなさないでください。

#### H2.5-C: 寸法確定の判定順序

cross-video heatmapの寸法は次の優先順で決定してください。

1. **選択肢3を優先する:** teacher v7全件についてsource中間H5を解決でき、
   `local_encoder_images.shape`等から実寸法を取得できる場合は、それをprovenanceの一次根拠とする。
2. **選択肢2を条件付きで使用する:** source中間H5が一部または全部失われていても、生成pipeline/run log、
   世代間point-cloud一致、全180 H5の`pixel_xy` boundsから共通local crop寸法を裏付けられる場合だけ、
   `--pixel_width 256 --pixel_height 256`等の明示的run-level定数を使用する。
3. **併用する場合:** source H5を読めた動画と読めない動画で別の座標規約を使わず、先にrun全体の共通寸法
   契約を確定する。source H5照合はその契約の実測検証とし、照合済み・未照合件数を報告する。
4. **根拠不足の場合:** 観測点のmax値、推定crop範囲、動画ごとの便宜的寸法では補わない。該当動画を
   cross-video解析から除外し、全件で確定不能ならcross-video heatmap自体を未実施として同一動画内解析
   のみに限定する。

生成CLIのdefaultが256であることだけでは選択肢2の採用根拠として不十分です。実際のrunでoverrideされて
いないこと、`pixel_xy`が共通local crop座標であること、全件が同じ寸法契約を満たすことを組み合わせて
示してください。

#### H2.5-D: 正規化とfail-fast

寸法を`width`、`height`として確定できた場合、pixel indexの両端を0と1へ対応させる次の規約を使用します。

```text
normalized_x = pixel_x / (width - 1)
normalized_y = pixel_y / (height - 1)
```

`width > 1`、`height > 1`を必須とし、全点についてfiniteかつ
`0 <= pixel_x < width`、`0 <= pixel_y < height`を検証してください。丸め前の浮動小数座標に対する許容差が
必要なら値と理由を設定へ保存し、bounds違反をclipして続行しないでください。synthetic testには少なくとも
四隅、境界直前、bounds外、nonfinite、寸法欠落を含めます。

#### H2.5-E: 必須artifactと停止条件

少なくとも次を出力してください。名称は既存checker構成に合わせて調整できます。

```text
stage4_dataset_coordinate_inventory.csv       # private
stage4_xy_source_resolution.csv               # private
stage4_xy_coordinate_provenance_summary.json  # shareable版も作成
stage4_xy_dimension_audit.csv                  # 匿名化したshareable版も作成
```

summaryには採用した選択肢、寸法、dimension source、全件数、source解決数、実寸法照合数、bounds合格数、
世代間point-cloud一致数、除外数と理由を保存してください。調査結果と選択肢2/3の採否を
`docs/stage5/s5-14/stage5_s5_14_report_to_policy_chat.md`へ明記します。

上記契約を満たせる場合は、その規約でStep H3.1を実装・実行して構いません。根拠不足、寸法不一致、
座標系混在、bounds違反が見つかった場合は推測で先へ進まず、同一動画内解析だけを継続して方針管理
チャットへ報告してください。

### Step H3: Aggregate PredictionのFrame/XY診断

保存済みW-A epoch 5のprediction `.npz`を優先して再利用し、固定21動画を解析します。必要なartifactが
不足する場合だけ再推論してください。

frame単位で次を記録します。

- TP、FP、TN、FN
- precision、recall、F1、IoU、FPR、FNR
- GT positive count、predicted positive count/rate
- valid/ignore count
- GT positive上、background上、ignore上の`prob_femur`統計
- GT/predicted positiveのXY重心と分散
- 両方存在する場合の重心距離
- no-GT frameのFP数とpositive probability
- 直近GT-positive frameまでの距離
- 動画内相対位置とdecile
- GT-positive区間のbefore/inside/after区分

GTまたはpredicted positiveが空の場合、重心や距離を0で埋めずnull/NaNとし、別途availability flagを
出してください。

### Step H3.1: XY DensityとTemporal Recurrence

normalized XY grid上で次を別々に集計します。

- GT positive density
- predicted positive density
- FP density
- FN density

GTとpredictionの位置相関、predicted centroidの動画間集中度、no-GT frameで同じXY binがactiveになる率を
計測してください。

各XY binについて、positiveを予測したframeを連続runへ分割し、次を記録します。

- predicted run数、各runの長さ
- GT positive runとのoverlap
- GTと重ならないpredicted run数・長さ
- unmatched run間のframe gap
- 同じXY binで時間的に離れた複数runが生じる率

grid解像度とrun定義をconfig/summaryへ保存してください。単一gridだけで結論を決めず、少なくとも主gridと
隣接する1つの解像度で結論の方向が変わらないか確認します。ただし網羅的grid searchは行いません。

### Step H4: Per-window Context診断

固定21動画についてのみ、W-Aでper-window probabilityを取得します。既存S5-08 overlap checkerの
accumulator、source point alignment、edge/center distance計算を再利用してください。

同じforward結果からmean probabilityを再構成し、保存済みW-A aggregate predictionと次を完全一致させます。

- point countとsource point order
- vote count
- threshold 0.5のTP/FP/TN/FN
- H5/video単位のaggregate metrics

CUDA非決定性等で不一致が出た場合はtoleranceで通さず、差分artifactを保存して停止し、原因を報告して
ください。

source point/frameごとに次を記録します。

- training window occurrence count
- inference vote count
- probability min/max/mean/std/range
- positive vote count/rate
- window間class disagreement
- window center/edge distance
- frame relative position/decile
- GT class、aggregate prediction class、FP/FN区分

次の層別比較を行います。

- valid positive / valid background / ignore
- TP / FP / FN / TN
- GT-positive frame / no-GT frame
- vote count / exposure count
- relative frame decile
- train sanity / validation

特に、unmatched XY recurrenceが高exposureまたは高disagreementへ集中するか、positiveの高い重複倍率が
FP/FNのどちらと対応するかを確認してください。

S5-08で棄却した「window境界に近いほどdisagreementが増える」という仮説は、teacher v7/W-Aでparityを
確認する以上に、根拠なく重み式を再設計しないでください。

### Step H5: 集計、可視化、完了報告

shareable出力には匿名video aliasだけを使用し、少なくとも次を作成してください。

```text
frame_metrics.csv
frame_position_deciles.csv
gt_overlap_exposure.csv
point_overlap_error_statistics.csv
xy_density_bins.csv または同等のNPZ
xy_temporal_recurrence.csv
video_summary.csv
stage5_s5_14_summary.json
```

必要に応じて次も出力できます。

- GT/prediction/FP/FNの全点PLY
- GT/prediction/FP/FNのpositive-only PLY
- XY density heatmap画像
- frame位置に対するTP/FP/FN/recall/FPRのplot

PLY、画像、実video mapping、絶対pathを含む出力はprivateに置き、匿名化bundleへ含めないでください。
shareable bundleにはprivacy self-checkを付け、timestamp形式のvideo IDとhost pathが存在しないことを
assertしてください。

## 8. 必須parityとfail-fast条件

1. W-Aのthreshold 0.5 aggregate TP/FP/TN/FNが既存`evaluate_stage5.py`結果と完全一致する。
2. H5 point数、frame_order、labels、valid_mask、pixel_xy、point_indicesのshapeが一致する。
3. 全source pointがaggregate predictionをちょうど1件持つ。
4. per-window voteを再集約したpointに欠落がない。
5. probabilityがfiniteかつ`[0,1]`内にある。
6. split/video alias/point集合が既存W-A評価と一致する。
7. teacher schemaがv7である。
8. checker設定、grid解像度、run定義、checkpoint identityをsummaryへ保存する。
9. cross-video XY解析を行う場合、Step H2.5のprovenance監査が合格し、採用寸法と根拠がsummaryにある。
10. `pixel_xy`が確定したlocal image bounds内にあり、観測点max正規化や暗黙の寸法fallbackを使用していない。

fail-fastを緩めたり、欠落値を0で埋めて続行したりしないでください。

## 9. 仮説判定基準

単一のaggregate値だけで結論を出さず、validation動画別結果、median、train sanityとの方向一致を
併記してください。

### 座標事前分布を支持する条件

- GT densityが低い共通XY領域へpredictionまたはFPが動画横断で集中する。
- predicted centroidが各frameのGT centroidより、動画横断のglobal prediction priorへ安定して近い。
- no-GT frameでも同じXY binに非連続なpredicted runが繰り返し現れる。
- raw point densityだけでは説明できず、densityで正規化したpositive rateでも傾向が残る。

### 時間位置・Frame phaseを支持する条件

- error/disagreementがrelative frame decileまたはGT-positive区間のbefore/inside/afterで一貫して変化する。
- 直近GT-positive frameまでの距離とFP/probabilityに系統的な関係がある。
- XY densityで層別化しても時間方向の傾向が残る。

### Overlap exposureを支持する条件

- exposure/vote countが増えるほどFP/FN/probability/disagreement/recurrenceが一貫して変化する。
- GT class、frame phase、動画を層別化しても関係が残る。
- positive/backgroundのexposure差が、単なるpoint density差ではなくerror差と対応する。

train sanity 3動画だけの傾向、少数の外れ値動画、raw point countだけでは支持と判定しないでください。
相関を因果と断定せず、次のablationで検証すべき候補として扱います。

## 10. 結果後の分岐

S5-14 coreの実装チャットは診断完了後、結果と選択肢を方針管理チャットへ返してください。自動的に
再学習へ進まないでください。

1. **座標事前分布が支持された場合**
   - まず再学習なしで座標変換に対するprediction equivariance診断を提案する。
   - その後に限りrotation、mirror等のaugmentationを1要因ずつ5 epoch比較する。
2. **overlap exposureが支持された場合**
   - inverse-occurrence loss weightingを第一候補とする。
   - center-only lossは第二候補とし、同時に導入しない。
   - 1方式だけの5 epoch比較案を提示する。
3. **時間位置だけが支持された場合**
   - window文脈とframe phaseを追加診断する。
   - 座標augmentationやoverlap lossを根拠なく導入しない。
4. **いずれも支持されない、または効果が小さい場合**
   - 構造変更を追加せずW-A baselineを維持する。
   - S5-15の5〜10 epoch最終pilot計画へ進む案を提示する。

追加ablationは方針管理チャットの承認後、1回に1要因、最大5 epochとします。

## 11. 非対象・禁止事項

S5-14 coreでは次を行わないでください。

- modelの再学習
- class weightの追加探索
- W-B/W-Cの再学習またはprimary比較への追加
- label policy変更
- GroupNorm group数またはnormalization方式変更
- center-only loss、inverse-occurrence loss weightingの実装
- sampling/window構成変更
- coordinate augmentation、feature ablation
- threshold tuningまたはproduction threshold変更
- mean以外のproduction aggregation採用
- Focal/Dice/Hard Negative Mining
- 50〜200 epoch学習
- source H5の書き換え
- source/teacher H5への寸法属性の後付け、dataset再生成
- 観測点maxによるcross-video XY正規化
- S5-07〜S5-11のteacher v7全面再実行

per-window probability取得のためのW-A再推論は許可しますが、固定21動画だけに限定し、同じforward結果を
全診断へ再利用してください。

## 12. Productionへの扱い

S5-14は診断段階です。次はすべて変更しません。

- production normalization既定値
- production label policy既定値
- production class weight既定値
- threshold 0.5
- mean probability aggregation
- training/inference CLIの既定動作

W-A、GroupNorm、`bbox_noncontour_ignore`は実験用暫定条件です。最終production採否はS5-15以降で
判断します。

## 13. 完了条件

1. synthetic testとstatic checkが合格している。
2. teacher v7全180 H5のGT/exposure監査が完了している。
3. v2以降のdataset inventoryとteacher v7全180 H5のXY座標provenance監査が完了している。
4. cross-video XY解析の採用可否、選択肢2/3、寸法、根拠、照合・除外件数が記録されている。
5. 固定21動画のframe/XY/temporal recurrence診断が完了している。
6. W-A per-window診断とmean baseline parityが合格している。
7. 3仮説それぞれについて支持・不支持・未確定を根拠付きで分類している。
8. aggregateだけでなくvideo-level、median、no-GT frameを評価している。
9. privacy self-checkが合格している。
10. core診断中に再学習・production変更を行っていない。
11. 次の文書を更新している。
   - `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
   - `docs/stage5/stage5_revision_management_record.md`
   - 新規ファイルがある場合は`docs/stage5/FILES.md`
12. 方針管理チャットへの完了報告を作成している。

```text
docs/stage5/s5-14/stage5_s5_14_report_to_policy_chat.md
```

## 14. 完了報告に含める内容

- 実装・変更ファイル
- 実行コマンドと対象checkpoint
- synthetic/static/parity結果
- teacher v7全件のclass別overlap exposure
- v2以降のStage 4 dataset inventoryと世代間point-cloud継承関係
- XY寸法provenance、source解決・実寸法照合・bounds検証の全件集計
- cross-video正規化に採用した選択肢2/3、寸法、根拠、除外件数
- frame-level train sanity/validation集計
- no-GT frameのFP/probability
- XY densityとtemporal recurrence結果
- per-window disagreementとexposure/errorの関係
- 3仮説の判定
- PLY定性確認がある場合はprivate artifactの案内
- production未変更の確認
- 次に行うべきablation候補と代替案
- 方針管理チャットで判断が必要な事項

core結果が得られた時点で停止し、方針管理チャットの判断を待ってください。
