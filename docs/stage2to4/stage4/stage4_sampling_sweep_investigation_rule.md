`stage4_invest_edit_prompt.md`を確認しました。調査は「基盤実装」と「実データparameter sweep」を分離し、以下の順序で進めるのが適切です。

## 全体構成

```text
I. 評価基盤の実装
  1. 入力manifestとteacher条件の固定
  2. dense teacher生成
  3. 1 config評価
  4. source ablation
  5. 集計・CSV・Pareto
  6. synthetic test
  7. small real-data smoke test

II. 実データ調査
  8. dataset/split監査
  9. Phase A: Evidence調査
 10. Phase B: Context調査
 11. Phase C: 最終候補比較
 12. validationで選定
 13. parameter固定後にtest評価
```

今回はまず手順7までを実装・確認し、全面的なparameter sweepはその後に行います。

## 1. 入力manifestとteacher条件の固定

最初に調査入力を明示するmanifestを定義します。

推奨列:

```text
video_name
pseudo3d_h5
voc_xml_root
split
enabled
notes
```

原則:

- `split` は `train`、`validation`、`test` のいずれか
- 実装側でsplitを勝手に生成しない
- XMLは `annotations_renamed` をstrict指定
- Phase 7の3件は `smoke` 用として別管理可能
- 同一動画の重複、H5欠落、XML対応ずれを開始前に検査

teacher生成条件もsampling sweepとは独立して固定し、設定とバージョンを出力へ保存します。

重要なのは、既存のsampling依存fallbackをteacher生成に使用しないことです。

```text
point_cloud=None
enable_fallback=False
strict_xml_annotation_dir=True
```

## 2. Dense teacher生成

既存の以下を再利用できます。

```python
build_frame_contour_mask_results(...)
build_full_frame_binary_mask(...)
build_largest_contour_mask_from_binary(...)
load_voc_bboxes(...)
xml_bbox_to_local(...)
```

各frameのdense teacher mapは次のように定義します。

```text
BBoxなしframe:
  全pixel = -1 ignore

BBoxありframe:
  BBox外             =  0 reliable background
  BBox内・contour外  = -1 ignore
  valid contour内    =  1 positive
```

複数BBoxがある場合:

- positiveはvalid contourのunion
- ignoreはBBox unionからpositiveを除いた領域
- per-BBox評価用maskは個別に保持
- frame全体評価用にはunion mapを使用

BBox分類:

```text
teacher-valid:
  local BBoxが有効
  かつteacher positive pixel > 0

teacher-invalid:
  BBoxは存在するがteacher positive pixel = 0

no-BBox:
  評価対象BBoxなし
```

`teacher-invalid`はsampling failureへ含めません。

出力するteacher品質情報:

```text
video_name
frame_order
frame_index
bbox_index
xml_path
local_bbox
teacher_status
teacher_positive_count
teacher_ignore_count
annotation_reason
contour_area
```

これは `teacher_quality.csv` に保存します。

## 3. 1 configの2D sampling評価

3D投影を行わず、exporterと同一の順序で2D samplingを実行します。

```text
local image
  -> make_texture_image
  -> build_global_evidence_mask
  -> build_sampling_source_flags
  -> selected_mask/source_flags
  -> dense teacher mapとの照合
```

必ず既存の [pseudo3d_sampling.py](/workspace/Stage2to4/src/utils/pseudo3d_sampling.py) を呼び出します。

評価用に同等のsamplingを別実装してはいけません。

また、global maskにはexporterと同様に `sample_stride` を適用します。context gridは `sample_stride` とは独立です。

### Positive metrics

teacher-valid BBoxについて:

```text
zero_positive_bbox_count/rate
positive_bbox_recall@1
positive_bbox_recall@8
positive_bbox_recall@16
positive_bbox_recall@32
positive_pixel_recall
```

positive pixel recallの集計:

```text
mean
median
p10
p05
min
```

### Endpoint coverage

teacher-positiveが十分に細長い場合だけPCAを使用します。

```text
positive_extent_recall
= sampled positiveのPC1 extent
  / teacher positive全体のPC1 extent
```

elongationが閾値未満、またはpixel数不足なら `NaN` とします。

### Cost metrics

全frameについて:

```text
points/frame mean, median, p95, max
legacy point count multiplier
selected_ignore_count/ratio
useful_point_ratio
selected_background_count/ratio
```

legacy point数が0のframeはframe別multiplierを `NaN` とし、別途zero-denominator frame数を記録します。全体multiplierは総combined点数/総legacy点数でも算出します。

## 4. Source別評価

各source bitについて以下を集計します。

```text
selected
positive
background
ignore
```

対象source:

```text
global
local
top-hat
context
context-only
```

source別件数ではoverlapを許容します。union総点数とは単純加算しません。

## 5. Source ablation

samplingを再計算せず、作成済み `source_flags` から対象bitを除外します。

```python
ablated_flags = source_flags & ~SOURCE_BIT
ablated_selected = ablated_flags != 0
```

評価対象:

```text
full combined
without_global
without_local
without_tophat
without_context
```

出力:

```text
zero-positive BBox増加数
positive_bbox_recall@K差
positive_pixel_recall差
削減point数
そのsourceがなければzero-positiveになるBBox数
marginal rescued BBoxes / 1000 unique added points
```

overlap点は他source bitが残っていれば選択状態を維持するため、二重計上を避けられます。

## 6. 集計と出力

新規ファイル構成案:

```text
src/utils/pseudo3d_sampling_evaluation.py
pseudo3d/analysis/sweep_stage4_sampling_parameters.py
pseudo3d/analysis/search_spaces/stage4_sampling_coarse.yaml
check_stage4_sampling_sweep_synthetic.py
```

出力:

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
├── run_metadata.json
└── visualizations/
```

各configには、全parameterのcanonical表現から安定した `config_id` を付与します。再実行時は同じconfigを識別し、途中再開できるようにします。

集計順序:

```text
per-BBox
  -> per-video
  -> split/global summary
```

全体平均とは別に以下を保存します。

```text
worst video
video p05/p10
worst BBox
teacher-invalid数
no-BBox frame数
```

## 7. Pareto抽出

単一weighted scoreは使用しません。

最初にpositive保持条件で候補を層別します。

優先順:

1. `zero_positive_bbox_rate` が最小
2. `positive_bbox_recall@16` が高い
3. positive pixel recallのp05/minが高い
4. selected ignoreが少ない
5. points/frameとlegacy multiplierが低い
6. reliable backgroundが不足しない
7. worst-video性能が高い

最低限、次の非劣解を抽出します。

```text
mean points/frame        minimize
legacy multiplier        minimize
selected ignore ratio    minimize
positive_bbox_recall@16  maximize
positive pixel recall    maximize
```

制約閾値そのものは実測分布を確認してから決定します。

## 8. Synthetic test

実データsmokeの前に以下を確認します。

### Teacher

- positive/background/ignoreの既知map
- teacher-positive=0ならteacher-invalid
- teacher-invalidがzero-positive集計へ入らない
- no-BBox frameはpositive評価から除外

### Sampling

- 全pixel選択ならpositive recall=1
- 全pixel選択ではpoint/ignore costが最大方向
- empty samplingでは全teacher-valid BBoxがzero-positive
- known maskでlabel別件数が手計算と一致

### Source ablation

- bit除外後のunion
- overlap点の維持
- source別bit件数
- marginal contribution

### Aggregation

- per-BBox→per-video→summary一致
- percentileとworst-case
- `NaN` とteacher-invalidの扱い
- config順序に依存しないdeterminism

### 本番一致

同じframe/configについて、評価runnerの `source_flags` と既存point-cloud exporterの保存結果を完全一致させます。これは調査基盤で最も重要な回帰テストです。

## 9. Small real-data smoke test

全面調査前に少数動画で実行します。

Phase 7の3件は次の用途に使えます。

- legacy BBox内0点の再現
- teacher-invalid分離
- mixed teacher-valid/invalid
- source visualization

ただし、最初の2件はstrict contourが全てteacher-invalidだったため、positive retentionのparameter比較だけには不十分です。

追加で以下を含む動画が必要です。

- teacher-valid BBoxが十分に多い
- 通常輝度
- 暗いframe
- 複数の形状・長さ
- BBoxなしframeを含む

smoke config:

```text
legacy/global baseline
現在のcombined_v2
context無効
local無効
top-hat無効
cleanup無効
```

ここでCSV、Pareto、failure visualizationまで生成できれば基盤実装を合格とします。

## 10. Phase A: Evidence sweep

contextを無効または固定し、次の順で調べます。

### A0. Source baseline/ablation

```text
global only
global + local
global + top-hat
global + local + top-hat
cleanup on/off
```

### A1. 一因子screening

現在値を中心に、1軸ずつ変更します。

- local window
- local percentile
- local min contrast
- top-hat kernel
- top-hat percentile
- top-hat min response
- morphology shape
- cleanup open/close
- cleanup min component area

この段階では全直積を作りません。

### A2. 少数組合せ

各系統の上位2〜3候補だけを組み合わせます。目安は最大20〜30 configです。

目的:

```text
positive retentionを維持
+
不要evidenceとignoreを削減
```

## 11. Phase B: Context sweep

Phase Aの上位evidence configを最大3件に絞り、以下を比較します。

```text
context disabled
stride = 4, 6, 8, 12
phase = origin, centered, dual
```

最大では:

```text
3 evidence configs × (1 + 12 context configs)
```

評価中心:

- background取得数
- context background/ignore yield
- unique point増加
- positive rescueへの追加寄与
- worst-video性能

## 12. Phase C: 最終候補比較

Evidence上位とContext上位の少数候補のみを比較します。

目安:

```text
5〜10 configs
```

trainでは候補生成、validationではPareto候補比較を行います。

必要ならこの段階でのみ以下も感度確認します。

```text
sample_stride
max_points_per_frame
```

`max_points_per_frame` を使う場合はseed固定とし、positive保持への影響を別に測定します。

## 13. 最終選定とtest

1. trainで探索
2. validationで最終configを1件または少数候補へ固定
3. config、teacher設定、manifest、code commitを保存
4. それ以降parameterを変更しない
5. testを一度だけ評価
6. 最終候補のみ既存Stage 4 exporterでH5/PLYを生成
7. annotation、collect、Stage 5 loaderまでend-to-end確認

## 実装開始時の順序

次の作業は以下の単位で進めるのが安全です。

1. dense teacher生成とteacher-valid分類
2. そのsynthetic test
3. 1 config評価とmetric
4. source ablation
5. aggregation/CSV
6. search configとrunner
7. Pareto
8. small real-data smoke
