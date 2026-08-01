Stage 4 Phase 7の実装・受け入れ確認が完了したため、次のPhaseとして **実教師データに基づくsampling parameter sweep基盤** を実装してください。

Phase 7でのStage 4 sampling実装そのものは合格とし、今回はsampling方式を追加・変更することが主目的ではありません。

今回の目的は、

**teacher-positiveを十分に保持するという条件を満たしながら、ignore点・冗長点・総点数を可能な限り減らすsampling parameterを実データから探索すること**

です。

重要な考え方として、単純に「BBox内positiveが何点拾えたか」だけを最大化しないでください。

極端には全pixelを採用すればpositive recallは最大になりますが、Stage 5へ大量の不要点・ignore点を渡すことになり、parameter tuningとして意味がありません。

したがって、今回のparameter sweepは **positive rescue性能とsampling costを同時に評価する制約付き/Pareto型の探索** としてください。

# 1. Phase 7時点の前提

現在の `combined_v2` samplingには以下が実装済みです。

* global evidence
* local percentile evidence
* top-hat evidence
* context grid
* source flagsによる重複なしunion
* sampling confidence
* evidence cleanup

  * opening
  * closing
  * connected component area filtering
* prior/corr geometryへの3D projection
* point cloud H5/PLY
* annotation H5へのschema伝播
* Stage 5 loader対応
* legacy互換性

source flags:

```text
bit 0 = global evidence
bit 1 = local percentile
bit 2 = top-hat
bit 3 = context grid
```

暫定parameter:

```text
sample_stride                    = 2

context_grid_stride              = 4
context_grid_phase               = origin

local_window_size                = 41
local_percentile                 = 80
local_min_contrast               = 4

tophat_kernel_size               = 21
tophat_percentile                = 85
tophat_min_response              = 2
tophat_morph_shape               = ellipse

evidence_open_ksize              = 3
evidence_close_ksize             = 5
evidence_morph_shape             = ellipse
evidence_min_component_area      = 100
```

これらは最終値ではなく、今回の調査対象です。

# 2. Phase 7から判明していること

legacy samplingでBBox内pointが0だった実データ3動画・51 target frameについて、暫定combined_v2では51/51 frameすべてでBBox内pointを回収できました。

一方、cleanup前のpoint count multiplierはlegacy比で約6.5〜6.7倍でした。

したがって、

```text
positiveを拾えるか
```

という第一段階の問題だけでなく、

```text
そのためにどれだけ不要点を増やしているか
```

を評価する必要があります。

また、strict contour annotationが0であってもraw BBox内にsampling pointが存在するケースがあります。

そのため、

* samplingがpositiveを落としたケース
* annotation生成側でteacher-positive自体が作れなかったケース

を必ず分離してください。

# 3. Parameter sweepの最重要原則

## 3.1 Samplingとteacher生成を分離する

sampling parameterの評価に、sampling後のpoint cloudから作られたannotationをそのまま使うと循環評価になります。

parameter sweepでは、可能な限りsamplingとは独立した **dense teacher label map** を作ってください。

各local frameについて概念的に:

```text
teacher_label_map[y, x]

  1 = femur positive
  0 = reliable background
 -1 = ignore / unknown
```

を用意し、sampling configによって選択されたpixelをこのdense mapへ照合します。

既存annotation処理がpoint list入力を前提としている場合は、既存のteacher生成ロジックを再利用しつつ、

```text
256x256 local image全pixelに対してteacher labelを作れる関数
```

へ切り出すことを検討してください。

重要:

* sampling mask生成にはBBoxやteacher labelを使用しない
* BBox/teacherはparameter評価にのみ使用する

# 4. BBoxを3種類に分ける

parameter sweepでは、BBoxを一括してzero-positive評価しないでください。

## A. teacher-valid BBox

```text
VOC BBoxあり
かつ
dense teacher label上にpositive pixelが1点以上存在
```

Stage 4 samplingのpositive rescue評価対象はこちらです。

## B. teacher-invalid BBox

```text
VOC BBoxあり
しかし
teacher-positive pixelが0
```

これはsampling failureとして数えないでください。

annotation/contour生成側の問題として別集計してください。

## C. BBoxなしframe

positive有無を断定できないため、positive rescue評価から原則除外します。

今後、

```text
zero-positive BBox
```

と呼ぶ場合は、

```text
teacher-positiveが存在するBBoxなのに
sampling後のselected positive countが0
```

であるものだけを意味するようにしてください。

# 5. Primary metric: Positive retention

最重要指標として以下を実装してください。

## zero_positive_bbox_count / rate

teacher-valid BBoxのうち、

```text
selected_positive_count == 0
```

となるBBox数・割合。

これは最優先で最小化します。

## positive_bbox_recall@K

teacher-valid BBoxのうち、少なくともK点のteacher-positive pixelをsamplingできた割合。

最低限:

```text
@1
@8
@16
@32
```

を出してください。

@1だけでは偶然1点拾ったconfigまで高評価になるため、複数Kを評価します。

## positive_pixel_recall

BBoxまたはframeごとに:

```text
selected teacher-positive pixels
--------------------------------
all teacher-positive pixels
```

を計算してください。

summaryとして最低限:

```text
mean
median
p10
p05
min
```

を出してください。

全体平均だけでなくworst-caseを重視します。

# 6. Endpoint/FL目的を考慮したpositive coverage

Stage 5/6の最終目的はFL endpoints / length推定なので、positiveの中央だけ大量に拾い、両端を落とすconfigも避けたいです。

可能であればteacher-positive pixelに対してPCAを適用し、

```text
positive_extent_recall
```

を追加してください。

概念:

```text
sampled positiveの第1主軸方向extent
-------------------------------------
teacher-positive全体の第1主軸方向extent
```

ただし、大腿骨が輪切り状に映るなどelongationが低いframeでは、このmetricは不安定です。

その場合は、

* elongationが一定以上のBBoxだけで計算
* それ以外はNaN / not_applicable

として構いません。

このmetricをsampling条件として使わず、評価指標としてのみ使用してください。

# 7. Sampling Cost metric

全pixel採用が最適にならないよう、以下を必ず評価してください。

## total points / frame

最低限:

```text
mean
median
p95
max
```

を出してください。

## legacy point count multiplier

```text
combined selected point count
-----------------------------
legacy point count
```

Phase 7では暫定configで約6.5〜6.7倍でした。

これを削減できるか評価します。

## selected ignore count / ratio

Stage 5 CEに直接使用しないlabel=-1点について:

```text
selected_ignore_count
selected_ignore_ratio
```

を計算してください。

これは今回の重要なcost metricです。

## useful point ratio

```text
N(label==1) + N(label==0)
-------------------------
N(selected)
```

を計算してください。

Stage 5 CEへ利用可能な点の割合として扱います。

# 8. Background / Context評価

positiveだけでなくbackground分類もStage 5で重要なので、

```text
backgroundが多い = 悪い
```

とは扱わないでください。

特にcontext gridはpositive rescueよりも、

**少ない点数でreliable background/contextを十分確保できるか**

で評価します。

最低限:

```text
selected_background_count
selected_background_ratio
```

に加え、context-onlyについて:

```text
context_background_yield
  = label==0 のcontext-only点 / context-only全点

context_ignore_yield
  = label==-1 のcontext-only点 / context-only全点
```

を計算してください。

これにより例えば、

```text
stride=4:
  backgroundは多いがignoreも大量

stride=8:
  backgroundは十分でignoreと総点数が少ない
```

といった比較が可能になります。

# 9. Source別評価

現在source_flagsが実装されているため、各sourceについて最低限以下を集計してください。

```text
global selected points
local selected points
top-hat selected points
context selected points

global positive points
local positive points
top-hat positive points
context positive points

global background
local background
top-hat background
context background

global ignore
local ignore
top-hat ignore
context ignore
```

source数はbitごとの集計なので、overlapした点が複数sourceへ含まれて構いません。

# 10. Source marginal contribution

source別の単純point数だけでなく、各sourceを除外した場合の性能差も調査できるようにしてください。

例:

```text
combined
without_global
without_local
without_tophat
without_context
```

について、

```text
zero_positive_bbox_count
positive_bbox_recall@K
positive_pixel_recall
total_points
```

を比較します。

特に、

```text
そのsourceがなければzero-positiveになるBBox数
```

を計算してください。

可能なら、

```text
marginal rescued BBoxes / 1000 added points
```

のような効率指標も出してください。

Phase 7の3動画だけでは、context gridとtop-hatの大腿骨回収寄与が小さく見えましたが、これは機能削除の根拠にはしません。

今回の全体調査によって判断します。

# 11. Parameter sweepを3段階に分ける

全parameterを一度にCartesian productすると探索数が大きくなりすぎるため、段階的に行える設計にしてください。

## Phase A: Evidence parameter sweep

context gridを一旦固定または無効にし、

```text
global
local percentile
top-hat
evidence cleanup
```

のparameterを評価します。

目的:

```text
positive retentionを高く維持
+
不要evidenceを減らす
```

候補parameter:

```text
local_window_size
local_percentile
local_min_contrast

tophat_kernel_size
tophat_percentile
tophat_min_response
tophat_morph_shape

evidence_open_ksize
evidence_close_ksize
evidence_morph_shape
evidence_min_component_area
```

調査範囲の初期候補:

```text
local_window_size:
  21, 31, 41, 61

local_percentile:
  70, 75, 80, 85, 90

local_min_contrast:
  0, 4, 8, 12

tophat_kernel_size:
  11, 21, 31, 41

tophat_percentile:
  80, 85, 90, 95

tophat_min_response:
  0, 2, 4, 8

evidence_open_ksize:
  0, 3, 5

evidence_close_ksize:
  0, 3, 5

evidence_min_component_area:
  0, 25, 50, 100, 200
```

ただし、最初から全直積を回さないでください。

coarse-to-fineまたは段階探索にしてください。

## Phase B: Context parameter sweep

Phase Aで上位のevidence configを固定し、context gridを比較します。

候補:

```text
context_grid_stride:
  4, 6, 8, 12

context_grid_phase:
  origin, centered, dual
```

評価の中心:

```text
background取得数
context_background_yield
context_ignore_yield
total point増加量
positive rescueへの追加寄与
```

## Phase C: Final combined comparison

Evidence上位config × Context上位configの少数候補だけを比較し、最終Pareto候補を抽出します。

# 12. Config選択は単一weighted scoreにしない

最初から、

```text
score = recall - alpha * point_count ...
```

のような恣意的weighted scoreだけで順位を決めないでください。

まずは制約条件で候補を絞ります。

例:

```text
zero_positive_bbox_rate が最小
positive_bbox_recall@16 が一定以上
positive_pixel_recall が一定以上
```

その条件を満たすconfigの中から、

```text
1. selected ignoreが少ない
2. total points/frameが少ない
3. legacy multiplierが低い
4. worst-video性能が高い
5. reliable backgroundが不足しない
```

順で比較できるようにしてください。

最終閾値値そのものは、実調査後に決定します。

# 13. Pareto front

summaryから少なくとも以下のPareto frontを抽出できるようにしてください。

例:

```text
x = mean points/frame
y = positive_bbox_recall@16
```

または:

```text
x = legacy multiplier
y = positive pixel recall
```

加えてignore costを含む比較もできるよう、CSVには各metricをすべて保持してください。

可能なら `pareto_front.csv` を生成してください。

# 14. Video単位 / BBox単位評価

全体平均だけではなく、最低限以下の粒度で出力してください。

```text
config summary
per-video
per-BBox
```

worst-case評価:

```text
worst video positive rescue
video p05
video p10
worst BBox positive recall
```

特定動画だけ大きく失敗するparameterを除外できるようにしてください。

# 15. Parameter sweepでは3D投影を原則行わない

Stage 4では、

```text
1 selected pixel
→
1 projected 3D point
```

の対応なので、parameter tuning自体は2D pixel空間で評価できます。

したがって、各configについて毎回:

```text
3D projection
H5 point cloud保存
PLY生成
```

を行う必要はありません。

parameter sweepでは、

```text
local image
→ sampling mask
→ source_flags
→ dense teacher mapとの比較
```

までを基本としてください。

最終候補数件についてのみ、必要なら既存Stage 4 exporterでH5/PLYを作成します。

これにより計算量・ストレージ量を抑えます。

# 16. Sampling処理は既存utilityを必ず再利用する

Phase 7で作成した:

```text
src/utils/pseudo3d_sampling.py
```

のsampling関数をparameter sweepでも使用してください。

調査用に同等処理を別実装しないでください。

本番Stage 4とparameter sweepでsampling結果が完全一致することが重要です。

必要なら既存関数を、

```text
2D mask/source_flagsのみ返す
```

用途へ少しリファクタリングして構いません。

ただしlegacy/combined_v2の既存挙動を壊さないでください。

# 17. Intermediate map cache

parameter数が増えるため、同じ画像に対する重い処理をconfigごとに無駄に再計算しない設計を検討してください。

キャッシュ候補:

```text
teacher dense label map
BBox map
local median map
local percentile関連map
top-hat response image
morphology前raw evidence mask
context grid mask
```

ただし、cacheによって本番samplingとの計算結果が変わらないことを優先してください。

# 18. 出力ファイル

推奨:

```text
sampling_parameter_sweep/
├── configs.csv
├── summary.csv
├── per_video.csv
├── per_bbox.csv
├── source_ablation.csv
├── pareto_front.csv
├── teacher_quality.csv
├── failure_cases.csv
├── search_space.yaml
└── visualizations/
```

`failure_cases.csv` には最低限:

```text
config_id
video_name
frame_index
BBox index
teacher_positive_count
selected_positive_count
selected_background_count
selected_ignore_count
selected_total_count
```

を保存してください。

# 19. Visualization

全configの画像は不要ですが、

```text
zero-positive
low positive recall
high ignore
high point count
representative good case
```

についてdebug visualizationを生成できるようにしてください。

overlay候補:

```text
local image
VOC BBox
teacher positive
teacher background
teacher ignore
global samples
local samples
top-hat samples
context samples
union samples
```

source別色は既存visualizationと可能な限り揃えてください。

# 20. Runtimeも計測する

Phase 8評価項目として、sampling速度も保存してください。

最低限:

```text
sampling time/frame
sampling time/video
teacher generation time
```

必要であればPhase A/B/Cごとの総探索時間も出せるようにしてください。

# 21. Train / validation / testの扱い

parameter tuningでtest setを使用しないでください。

基本:

```text
train:
  広いparameter探索

validation:
  Pareto候補比較と最終parameter選択

test:
  parameter固定後の最終評価のみ
```

既存dataset split/manifestが存在する場合はそれを利用してください。

存在しない場合は、勝手に新しいsplitを確定せず、実装側ではmanifestまたはvideo listを入力できるようにしてください。

# 22. 最初に作成してほしいもの

まずはparameter sweepを全面実行するのではなく、基盤実装を行ってください。

推奨新規ファイル例:

```text
pseudo3d/analysis/sweep_stage4_sampling_parameters.py
```

必要なら:

```text
src/utils/pseudo3d_sampling_evaluation.py
```

などにteacher生成・metric計算を分離してください。

役割:

```text
1. dataset/manifest読込
2. pseudo3D H5とVOC XML対応付け
3. dense teacher label map生成
4. teacher-valid / invalid BBox分類
5. sampling config生成
6. 既存pseudo3d_sampling.pyで2D sampling
7. metric計算
8. source ablation
9. per-BBox / per-video / summary集計
10. Pareto candidate出力
11. failure visualization
```

# 23. まず実装する範囲

今回のCodex作業では、まず以下まで実装してください。

1. 既存Stage 4 / annotationコードを確認
2. dense teacher label生成方法を設計・実装
3. teacher-valid / invalid BBox分類
4. 1 configについてsampling評価できる処理
5. positive/background/ignore metric
6. source別metric
7. source ablation
8. per-BBox / per-video / summary集計
9. 複数configを受け取るsweep runner
10. Phase A/B/Cへ分けられるsearch config設計
11. CSV出力
12. Pareto抽出
13. 少数データでのsmoke test

この段階では、まだ全データを使った本調査や最終parameter決定は行わないでください。

# 24. 最低限のテスト

syntheticまたはsmall real-data smoke testで以下を確認してください。

### Teacher

* positive/background/ignoreが期待通り生成される
* teacher-positive=0のBBoxはteacher-invalidになる
* teacher-invalid BBoxがsampling failureへ数えられない

### Sampling evaluation

* 全pixel selectionではpositive recallが1になる
* ただしpoint count / ignore costが最大方向になる
* empty samplingではzero-positive BBoxが最大になる
* known maskでpositive/background/ignore countが正しい

### Source ablation

* source bitを外した場合にpoint unionが正しく変化する
* overlap pixelが二重計上されない
* source別bit件数はoverlapを許容して正しく集計される

### Aggregation

* per-BBox → per-video → global summaryが一致する
* NaN / teacher-invalidの扱いが安定している
* worst-case / percentile計算が正しい

### Legacy compatibility

* Stage 4本番exporterの既存挙動を壊さない

# 25. 実装完了後に報告してほしい内容

* 変更ファイル一覧
* dense teacher labelの定義
* teacher-valid / invalid判定条件
* parameter config schema
* Phase A/B/Cの探索方式
* metric一覧と定義
* source ablationの実装方法
* Pareto判定方法
* output CSV schema
* smoke test結果
* 処理時間
* 本調査を実行するためのCLI例
* 本調査前に決める必要が残った事項

今回の目的は最終parameterを決めることではなく、

**Stage 4 samplingのpositive保持能力とsampling costを正しく比較できるparameter sweep基盤を完成させること**

です。
