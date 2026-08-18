# Stage 4 sampling parameter調査 進捗報告

最終更新: 2026-07-27  
対象: `combined_v2` Stage 4 sampling  
現在の状態: **global+local Stage 5学習ablationデータセット生成準備完了**

## 1. 調査目的

dense teacher上の大腿骨positiveを十分に保持しながら、Stage 5へ渡す総点数、
ignore点、冗長点およびsampling時間を削減する。

単純なpositive最大化ではなく、以下を同時に評価するPareto型探索とする。

1. `zero_positive_bbox_count`を最優先で0に保つ
2. `positive_bbox_recall@16`を高く保つ
3. `positive_pixel_recall`のp05/minを重視してhard caseを保護する
4. points/frame、legacy倍率、ignore比率を削減する
5. source追加に対するpositive rescue効率を確認する

sampling mask生成にはBBoxやteacher labelを使用していない。BBoxとdense teacherは
sampling後の評価にのみ使用している。

## 2. 固定した入力とteacher

### 2.1 学習データ

```text
pseudo3D H5:
/mnt/data/3d_projects/pseudo3d_dataset/pseudo3d_outputs/260711

VOC XML root:
/mnt/data/Data_hbl/predirectory_for_coco_annotation_leg/260630/260630_pseudo3d

split:
train
```

`260711`の全データを学習用として扱う。

実調査manifest:

```text
/mnt/data/3d_projects/pseudo3d_dataset/
stage4_sampling_parameter_sweep/260711/manifests/train_manifest.csv
```

preflight結果:

```text
enabled : 182
ok      : 182
failed  : 0
disabled: 0
splits  : {'train': 182}
```

### 2.2 Dense teacher

設定:

```text
pseudo3d/analysis/configs/stage4_dense_teacher_v1.yaml
fingerprint:
e837f6d32a3031752c22786955c02e9adb81a625672531a3d36e009fb4bd28cd
```

label定義:

| 値 | 意味 |
|---:|---|
| `1` | accepted contour内のteacher-positive |
| `0` | 有効BBox外のreliable background |
| `-1` | BBox内contour外、BBoxなしframe等のignore |

分類:

| 分類 | 条件 | sampling failure集計 |
|---|---|---|
| teacher-valid | BBox内teacher-positiveが1 pixel以上 | 対象 |
| teacher-invalid | BBoxはあるがteacher-positiveが0 | 対象外、別集計 |
| BBoxなし | 評価可能なBBoxなし | positive評価対象外 |

sampling依存fallbackは無効であり、strict
`annotations_renamed` XMLだけを使用している。

## 3. 調査設計

### 3.1 全train baseline

182動画全体で基準設定1件を実行し、teacher品質、全体傾向、source ablationを取得した。

```text
output:
/mnt/data/3d_projects/pseudo3d_dataset/
stage4_sampling_parameter_sweep/260711/phase_a/a0_reference
```

### 3.2 Train内screening subset

全32候補を182動画へ直接適用すると、基準実測から約54時間以上かかると見積もられた。
そのため、これは新しいvalidation/test splitではなく、train内のcoarse screening用subset
として36動画を決定論的に抽出した。

抽出対象:

- baselineで`recall@16 < 1`となるhard case
- positive pixel recallのmin/p05/microが低い動画
- selected ignore比率が高い動画
- points/frameが多い動画
- 固定seedによる補完例

```text
manifest:
/mnt/data/3d_projects/pseudo3d_dataset/
stage4_sampling_parameter_sweep/260711/manifests/train_screening_36.csv

seed:
stage4_phase_a_screening_v1
```

preflight結果:

```text
videos          : 36
frames          : 1,764
matched BBoxes  : 663
teacher-valid   : 598
teacher-invalid : 65
failed          : 0
```

coarse探索後の最終候補は、このsubsetだけで確定せず、182動画全体へ戻して確認する。

## 4. Phase A0: 全train baselineとsource ablation

### 4.1 基準設定

Phase A中はcontext gridを無効化し、evidence sourceだけを比較している。

```text
sample_stride=2

local:
  window_size=41
  percentile=80
  min_contrast=4

top-hat:
  kernel_size=21
  percentile=85
  min_response=2
  morph_shape=ellipse

cleanup:
  open=3
  close=5
  morph_shape=ellipse
  min_component_area=100
```

### 4.2 182動画での結果

| 指標 | 結果 |
|---|---:|
| videos | 182 |
| frames | 7,349 |
| teacher-valid BBox | 2,620 |
| teacher-invalid BBox | 452 |
| zero-positive BBox | 0 |
| positive BBox recall@16 | 0.999618 |
| positive pixel recall micro | 0.722166 |
| positive pixel recall p05 | 0.487256 |
| positive pixel recall min | 0.226087 |
| points/frame mean | 10,134.66 |
| legacy倍率 | 4.0499 |
| selected ignore ratio | 0.627715 |
| useful point ratio | 0.372285 |
| total selected points | 74,479,610 |
| total runtime | 6,297.54秒 |

`recall@16`未達は2,620 BBox中1件であり、zero-positiveはなかった。

### 4.3 Source ablation

| ablation | zero BBox | recall@16 | positive recall micro | selected points | source unique点 |
|---|---:|---:|---:|---:|---:|
| full | 0 | 0.999618 | 0.722166 | 74,479,610 | - |
| without global | 12 | 0.991603 | 0.631572 | 65,320,068 | 9,159,542 |
| without local | 0 | 0.999237 | 0.706566 | 59,881,702 | 14,597,908 |
| without top-hat | 0 | 0.998855 | 0.586697 | 58,247,872 | 16,231,738 |

判断:

- globalは12 BBoxをzero-positiveから救済しており、必須sourceとする。
- top-hatはglobalほどzero rescueを持たないが、positive pixel recallへの寄与が最も大きい。
- localは14,597,908 unique点を追加する一方、micro recallへの寄与は比較的小さい。
- ただし後述のsubset評価ではlocalがp05/minを改善しており、hard case用sourceとして評価を継続する。
- contextはPhase Aでは無効なので、この段階では評価しない。

## 5. Phase A0: screening subsetでのsource構造比較

```text
output:
.../phase_a/a0_screening_36
runtime:
4,097.73秒
```

| config | zero | recall@16 | micro | p05 | min | points/frame | legacy倍率 | ignore比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global+local+top-hat、cleanup有効 | 0 | 0.998328 | 0.606155 | 0.375026 | 0.226087 | 9,513.87 | 3.7966 | 0.623766 |
| global only | 0 | 0.996656 | 0.248329 | 0.231504 | 0.207248 | 2,505.90 | 1.0000 | 0.629887 |
| global+local | 0 | 0.998328 | 0.481697 | 0.241616 | 0.224138 | 7,677.88 | 3.0639 | 0.620488 |
| global+top-hat | 0 | 0.998328 | 0.595401 | 0.287735 | 0.226087 | 7,603.85 | 3.0344 | 0.630965 |
| global+local+top-hat、cleanup無効 | 0 | 1.000000 | 0.608488 | 0.449736 | 0.351304 | 11,671.90 | 4.6578 | 0.628501 |

判断:

- top-hat単独追加はlocal単独追加よりmicro recallへの寄与が大きい。
- localはtop-hat併用時のp05を`0.2877 → 0.3750`へ改善し、tail保護に寄与する。
- cleanup無効化は基準比で点数を22.7%増やす一方、p05を約7.47ポイント、
  minを約12.52ポイント改善した。
- 暫定cleanupの`min_component_area=100`はhard-case positiveを除去しすぎている。

全5候補がPareto frontに残った。これはpositive保持とcostのtrade-offが実在するためであり、
Pareto所属だけでは最終選定しない。

## 6. Phase A3: cleanup coarse sweep

local/top-hatは暫定値、contextは無効のまま、cleanup parameterだけを一因子比較した。

注意:

```text
min_component_area=0
```

はcleanup無効ではない。opening=3、closing=5は有効なまま、小component削除だけを
無効にする。closingによってraw evidenceのgapが埋まるため、cleanup無効より
positive recall microが高くなる場合がある。

| cleanup変更 | zero | recall@16 | micro | p05 | min | points/frame | legacy倍率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cleanup無効（A0参照） | 0 | 1.000000 | 0.608488 | 0.449736 | 0.351304 | 11,671.90 | 4.6578 |
| open=0 | 0 | 0.998328 | 0.633285 | 0.448063 | 0.227826 | 11,756.78 | 4.6916 |
| open=5 | 0 | 0.998328 | 0.556917 | 0.256183 | 0.226087 | 7,167.29 | 2.8602 |
| close=0 | 0 | 0.998328 | 0.570456 | 0.270985 | 0.226087 | 7,969.01 | 3.1801 |
| close=3 | 0 | 0.998328 | 0.579472 | 0.281757 | 0.226087 | 8,431.49 | 3.3647 |
| component area=0 | 0 | 1.000000 | 0.616885 | 0.445185 | 0.302609 | 10,893.65 | 4.3472 |
| component area=25 | 0 | 0.998328 | 0.615151 | 0.438811 | 0.275862 | 10,496.71 | 4.1888 |
| component area=50 | 0 | 0.998328 | 0.613114 | 0.434557 | 0.227826 | 10,109.40 | 4.0342 |
| component area=100（A0基準） | 0 | 0.998328 | 0.606155 | 0.375026 | 0.226087 | 9,513.87 | 3.7966 |
| component area=200 | 0 | 0.998328 | 0.573625 | 0.246474 | 0.224138 | 8,405.60 | 3.3543 |

判断:

- opening=5、closing縮小、component area=200はtail positiveの損失が大きい。
- `component area=0`はcleanup無効比で点数を6.67%削減しながら、
  `recall@16=1.0`、p05=`0.4452`を維持した。
- `component area=25/50`は追加の点数削減候補として有望だが、
  area増加とともにminが低下する。
- detector parameter調査中はpositiveを過剰除去しないようarea=0を固定する。
- detector確定後、area=`0/25/50`を最終候補として再比較する。

## 7. Phase A1: local-percentile coarse sweep

固定条件:

```text
top-hat:
  kernel=21
  percentile=85
  min_response=2
  shape=ellipse

cleanup:
  open=3
  close=5
  shape=ellipse
  component area=0
```

```text
output:
.../phase_a/a1_local_area0_screening_36
configs:
12
runtime:
15,174.82秒
```

| local設定 | zero | recall@16 | micro | p05 | min | points/frame | sec/frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| 41 / p80 / c4（基準） | 0 | 1.000000 | 0.616885 | 0.445185 | 0.302609 | 10,893.65 | 0.8206 |
| local無効 | 0 | 1.000000 | 0.606262 | 0.420978 | 0.252174 | 8,623.24 | 0.0022 |
| window=21 | 0 | 1.000000 | 0.611631 | 0.435355 | 0.354783 | 10,791.90 | 0.2516 |
| window=31 | 0 | 1.000000 | 0.613514 | 0.441866 | 0.340870 | 10,749.21 | 0.4971 |
| window=61 | 0 | 1.000000 | 0.634083 | 0.445185 | 0.297391 | 11,417.35 | 1.7807 |
| percentile=70 | 0 | 1.000000 | 0.656321 | 0.463121 | 0.384454 | 15,094.55 | 0.8459 |
| percentile=75 | 0 | 1.000000 | 0.632132 | 0.450023 | 0.373950 | 12,674.62 | 0.8350 |
| percentile=85 | 0 | 1.000000 | 0.610226 | 0.435669 | 0.271304 | 9,726.08 | 0.8023 |
| percentile=90 | 0 | 1.000000 | 0.607812 | 0.426105 | 0.252174 | 9,044.32 | 0.7779 |
| min contrast=0 | 0 | 1.000000 | 0.616885 | 0.445185 | 0.302609 | 15,832.24 | 0.8209 |
| min contrast=8 | 0 | 1.000000 | 0.616885 | 0.445185 | 0.302609 | 10,790.05 | 0.8211 |
| min contrast=12 | 0 | 1.000000 | 0.616838 | 0.445185 | 0.302609 | 10,564.96 | 0.8206 |

### 7.1 Localの判断

- local無効でもzeroとrecall@16は維持するが、p05/minが低下するため、
  hard case保護用としてlocalを残す。
- `window=31`はwindow=41比でsampling時間を約39.4%短縮し、
  points/frameを約1.33%削減する。p05低下は約0.33ポイントで、minは改善した。
- `min_contrast=12`は基準比でpoints/frameを約3.02%削減しながら、
  micro差は約-0.005ポイント、p05/minは同一だった。
- percentile=75はpositive保持を改善するが、points/frameが約16.35%増える。
- percentile=85/90は点数を減らすがtail recallが低下し、localを残す意義が薄くなる。

暫定local効率候補:

```text
window_size=31
percentile=80
min_contrast=12
```

positive保持寄り代替候補:

```text
window_size=31
percentile=75
min_contrast=12
```

後者の組合せはまだ直接評価しておらず、percentile=75の一因子結果を基にした候補である。

## 8. Phase A2: top-hat coarse sweep

状態: **完了**

config:

```text
pseudo3d/analysis/configs/
stage4_sampling_phase_a_tophat_local31_area0_search.yaml
```

固定条件:

```text
local:
  window=31
  percentile=80
  min_contrast=12

cleanup:
  open=3
  close=5
  shape=ellipse
  component area=0

context:
  disabled
```

探索対象:

```text
top-hat有効/無効
kernel_size     = 11, 21, 31, 41
percentile      = 80, 85, 90, 95
min_response    = 0, 2, 4, 8
morph_shape     = rect, ellipse, cross
```

展開設定数: 13

```text
output:
.../phase_a/a2_tophat_local31_area0_screening_36
configs:
13
runtime:
10,749.62秒
failures:
0
```

| top-hat設定 | zero | recall@16 | micro | p05 | min | points/frame | useful比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| k21 / p85 / r2 / ellipse（基準） | 0 | 1.000000 | 0.613232 | 0.440847 | 0.340870 | 10,200.36 | 0.375038 |
| top-hat無効 | 0 | 0.998328 | 0.483778 | 0.358858 | 0.275862 | 7,872.61 | 0.379557 |
| kernel=11 | 0 | 1.000000 | 0.558355 | 0.423760 | 0.356790 | 9,598.01 | 0.375506 |
| kernel=31 | 0 | 1.000000 | 0.618679 | 0.419906 | 0.340870 | 10,774.56 | 0.373719 |
| kernel=41 | 0 | 1.000000 | 0.643251 | 0.421617 | 0.340870 | 11,148.37 | 0.372811 |
| percentile=80 | 0 | 1.000000 | 0.669109 | 0.491530 | 0.353043 | 11,810.05 | 0.372782 |
| percentile=90 | 0 | 1.000000 | 0.547737 | 0.390326 | 0.334568 | 8,976.07 | 0.377664 |
| percentile=95 | 0 | 0.998328 | 0.494948 | 0.366030 | 0.275862 | 8,162.40 | 0.378722 |
| min response=0 | 0 | 1.000000 | 0.613232 | 0.440847 | 0.340870 | 10,200.36 | 0.375038 |
| min response=4 | 0 | 1.000000 | 0.613232 | 0.440847 | 0.340870 | 10,200.36 | 0.375038 |
| min response=8 | 0 | 1.000000 | 0.613232 | 0.440847 | 0.340870 | 10,200.36 | 0.375038 |
| shape=rect | 0 | 1.000000 | 0.603922 | 0.419399 | 0.340870 | 10,486.29 | 0.374664 |
| shape=cross | 0 | 0.998328 | 0.557282 | 0.403831 | 0.275862 | 9,395.27 | 0.376952 |

### 8.1 Top-hatの判断

- top-hat無効化では基準比でpoints/frameを22.82%削減できるが、microは約12.95ポイント、
  p05は約8.20ポイント、minは約6.50ポイント低下する。top-hatは維持する。
- kernel=31/41はmicroを増やすがp05を下げ、点数も増える。kernel=21を維持する。
- kernel=11は点数を5.91%削減するが、microを約5.49ポイント、p05を約1.71ポイント
  下げるため、最終候補には採用しない。
- percentile=80は基準比でpoints/frameが15.78%増える一方、microを約5.59ポイント、
  p05を約5.07ポイント改善する。positive保持候補とする。
- percentile=90/95は点数を削減するがtail性能の低下が大きく、local/top-hatを追加する
  目的に対して効率が悪い。
- min response=`0/2/4/8`は全metricと選択点数が完全一致した。このresponse範囲は、
  現在のpositive-response percentile処理下では識別不能である。暫定値2を維持する。
- ellipseはrect/crossよりp05が高い。rectは点数も増えるため不採用、crossは
  recall@16とminが低下するため不採用とする。

暫定top-hat効率候補:

```text
kernel_size=21
percentile=85
min_response=2
morph_shape=ellipse
```

positive保持寄り候補:

```text
kernel_size=21
percentile=80
min_response=2
morph_shape=ellipse
```

Pareto frontには同一metricのmin-response候補が重複して含まれた。これはPareto抽出が
同値configを統合しないためであり、実質的な別候補とは扱わない。

### 8.2 Phase A finalist screening

A1/A2/A3で残した候補だけを用い、以下の12設定をscreening subsetで比較する。

```text
local percentile:
  75, 80

top-hat percentile:
  80, 85

cleanup min component area:
  0, 25, 50
```

その他は次の値へ固定する。

```text
local window=31
local min contrast=12
top-hat kernel=21
top-hat min response=2
top-hat shape=ellipse
cleanup open/close=3/5
cleanup shape=ellipse
context disabled
```

config:

```text
pseudo3d/analysis/configs/
stage4_sampling_phase_a_finalist_screening_search.yaml
```

実行結果:

```text
output:
.../phase_a/a4_finalist_screening_36
configs:
12
runtime:
10,025.26秒
failures:
0
```

| local p | top-hat p | area | zero | recall@16 | micro | p05 | min | extent p05 | points/frame |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 75 | 80 | 0 | 0 | 1.000000 | 0.672966 | 0.500632 | 0.399322 | 0.959771 | 12,458.32 |
| 75 | 80 | 25 | 0 | 0.998328 | 0.671620 | 0.497470 | 0.275862 | 0.959771 | 12,014.45 |
| 75 | 80 | 50 | 0 | 0.998328 | 0.669915 | 0.491747 | 0.275862 | 0.959771 | 11,587.17 |
| 75 | 85 | 0 | 0 | 1.000000 | 0.622507 | 0.452175 | 0.386441 | 0.956433 | 11,166.73 |
| 75 | 85 | 25 | 0 | 0.998328 | 0.620681 | 0.448279 | 0.275862 | 0.955801 | 10,743.67 |
| 75 | 85 | 50 | 0 | 0.998328 | 0.618383 | 0.446831 | 0.275862 | 0.954879 | 10,321.53 |
| 80 | 80 | 0 | 0 | 1.000000 | 0.669109 | 0.491530 | 0.353043 | 0.959649 | 11,810.05 |
| 80 | 80 | 25 | 0 | 0.998328 | 0.667834 | 0.489598 | 0.275862 | 0.959649 | 11,375.07 |
| 80 | 80 | 50 | 0 | 0.998328 | 0.666132 | 0.487942 | 0.231304 | 0.959649 | 10,946.96 |
| 80 | 85 | 0 | 0 | 1.000000 | 0.613232 | 0.440847 | 0.340870 | 0.955801 | 10,200.36 |
| 80 | 85 | 25 | 0 | 0.998328 | 0.611382 | 0.436255 | 0.275862 | 0.954879 | 9,782.16 |
| 80 | 85 | 50 | 0 | 0.998328 | 0.609274 | 0.433993 | 0.226087 | 0.953674 | 9,351.06 |

### 8.3 Finalist screeningの判断

- 全12設定でzero-positive BBoxは0だった。
- area=0の4設定はすべてglobalおよびworst-videoの`recall@16=1.0`を維持した。
- area=25/50の8設定は、local/top-hat percentileに関係なく
  `recall@16=0.998328`、worst-video=`0.9375`となった。同じhard BBoxの
  component filteringが原因である可能性が高い。
- area=25はarea=0比でpoints/frameを約3.6〜4.1%削減するが、該当hard caseを含む
  min recallの低下が大きい。
- area=50は約7%前後の点数削減になるが、area=25に対するpositive保持低下も続く。
- local p75はp80比で、top-hat p80時に約5.5%、p85時に約9.5%点数を増やす。
  一方、特にmin recallを約4.6ポイント改善するため、hard-case保持候補として残す。
- top-hat p80はp85比で点数を約11.6〜15.8%増やすが、microとp05の改善が大きい。
- positive extent p05は全設定で約0.954〜0.960と高く、今回の候補選定では主要な
  差別化指標にならなかった。
- useful/ignore比率の差も小さく、主なtrade-offはpositive recallと総点数にある。
- 全12設定がPareto frontに残ったため、primary metricの優先順位とhard BBoxの
  実点数に基づく制約判断が必要である。

area=0を即確定すると、Phase 7で確認した孤立ノイズ除去を弱める可能性がある。
一方、area=25以上ではhard BBoxの`recall@16`が低下した。次にこのBBoxの
teacher-positive数、area別selected-positive数、動画/frameを監査してから、
areaの微調査またはfull-train finalistを決定する。

### 8.4 `recall@16`未達BBoxの監査

該当BBoxは1件だった。

```text
video_name:
20250403_114201_376

frame_order / frame_index:
8 / 8

XML:
.../20250403_114201_376/annotations_renamed/
20250403_114201_376_00009.xml

object:
leg

teacher_positive_count:
29

contour_area:
20.0
```

| local p | top-hat p | area=0 | area=25 | area=50 |
|---:|---:|---:|---:|---:|
| 75 | 80 | 18点 | 8点 | 8点 |
| 75 | 85 | 17点 | 8点 | 8点 |
| 80 | 80 | 18点 | 8点 | 8点 |
| 80 | 85 | 17点 | 8点 | 8点 |

判断:

- area=25はselected positiveを17〜18点から8点へ半減させており、
  「16点をわずかに下回る」ケースではない。
- teacher contour自体が29 pixel、contour area=20の小構造であり、
  area=25以上のcomponent filteringと相性が悪い。
- local/top-hat percentileを変更してもarea=25/50では8点で固定されるため、
  detector閾値よりcomponent filteringが直接原因と判断する。
- `remove_small_components`は面積がthreshold以上のcomponentだけを残す。
  area=1はarea=0と実質同じなので、fine sweepは`0/5/10/15/20/25`とする。
- area=25/50はPhase A finalistから除外する。孤立ノイズ対策との両立を調べるため、
  area=5〜20の破断点を保持候補・効率候補の2系列で評価する。

fine sweep:

```text
retention series:
  local p75
  top-hat p80

efficiency series:
  local p80
  top-hat p85

component area:
  0, 5, 10, 15, 20, 25
```

config:

```text
pseudo3d/analysis/configs/
stage4_sampling_phase_a_component_area_refine_search.yaml
```

### 8.5 Component area fine sweep結果

```text
output:
.../phase_a/a5_component_area_refine_screening_36
configs:
12
runtime:
10,044.81秒
failures:
0
```

| series | area | zero | recall@16 | micro | p05 | min | points/frame | hard BBox点 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| efficiency L80/T85 | 0 | 0 | 1.000000 | 0.613232 | 0.440847 | 0.340870 | 10,200.36 | 17 |
| efficiency L80/T85 | 5 | 0 | 1.000000 | 0.613215 | 0.440847 | 0.340870 | 10,197.95 | 17 |
| efficiency L80/T85 | 10 | 0 | 1.000000 | 0.612788 | 0.440847 | 0.325217 | 10,095.30 | 17 |
| efficiency L80/T85 | 15 | 0 | 0.998328 | 0.612227 | 0.440173 | 0.275862 | 9,993.90 | 8 |
| efficiency L80/T85 | 20 | 0 | 0.998328 | 0.611946 | 0.440173 | 0.275862 | 9,875.53 | 8 |
| efficiency L80/T85 | 25 | 0 | 0.998328 | 0.611382 | 0.436255 | 0.275862 | 9,782.16 | 8 |
| retention L75/T80 | 0 | 0 | 1.000000 | 0.672966 | 0.500632 | 0.399322 | 12,458.32 | 18 |
| retention L75/T80 | 5 | 0 | 1.000000 | 0.672960 | 0.500632 | 0.399322 | 12,455.44 | 18 |
| retention L75/T80 | 10 | 0 | 1.000000 | 0.672726 | 0.500632 | 0.397288 | 12,339.53 | 18 |
| retention L75/T80 | 15 | 0 | 1.000000 | 0.672465 | 0.500101 | 0.397288 | 12,230.93 | 18 |
| retention L75/T80 | 20 | 0 | 0.998328 | 0.672101 | 0.497470 | 0.275862 | 12,109.58 | 8 |
| retention L75/T80 | 25 | 0 | 0.998328 | 0.671620 | 0.497470 | 0.275862 | 12,014.45 | 8 |

判断:

- efficiency系列ではarea=10までhard BBox 17点と`recall@16=1.0`を維持し、
  area=15で8点へ破断した。最大安全値を10とする。
- retention系列ではarea=15までhard BBox 18点と`recall@16=1.0`を維持し、
  area=20で8点へ破断した。最大安全値を15とする。
- area=5はarea=0比の点数削減が約0.02%に留まり、実質的な差がない。
- area=10はefficiency系列で約1.03%、retention系列で約0.95%点数を削減する。
- retention area=15はarea=0比で約1.83%点数を削減し、p05/min低下は小さい。
- component areaの追加調査は完了とし、local windowの再調査も行わない。

全trainへ進める2候補:

```text
Efficiency:
  local window=31
  local percentile=80
  local min contrast=12
  top-hat kernel=21
  top-hat percentile=85
  top-hat min response=2
  shape=ellipse
  cleanup open/close=3/5
  cleanup component area=10

Retention:
  local window=31
  local percentile=75
  local min contrast=12
  top-hat kernel=21
  top-hat percentile=80
  top-hat min response=2
  shape=ellipse
  cleanup open/close=3/5
  cleanup component area=15
```

full-train config:

```text
pseudo3d/analysis/configs/
stage4_sampling_phase_a_full_train_finalists_search.yaml
```

### 8.6 Phase A finalistの全train確認

```text
output:
.../phase_a/a6_full_train_finalists
videos:
182
teacher-valid / invalid:
2620 / 452
configs:
2
runtime:
7,724.79秒
failures:
0
```

| candidate | zero | @1 | @8 | @16 | @32 | micro | p05 | min | extent p05 | points/frame | legacy倍率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| efficiency L80/T85/A10 | 0 | 1.0 | 1.0 | 1.0 | 0.999237 | 0.720584 | 0.492824 | 0.325217 | 0.960155 | 10,547.52 | 4.2149 |
| retention L75/T80/A15 | 0 | 1.0 | 1.0 | 1.0 | 0.999237 | 0.779025 | 0.546684 | 0.397288 | 0.967419 | 12,845.15 | 5.1330 |

両候補とも:

```text
zero-positive BBox = 0
under 16 positive points = 0
worst-video recall@16 = 1.0
```

retention候補はefficiency候補比で:

- points/frameが2,297.62点、21.78%増加
- positive recall microが約5.84ポイント増加
- p05が約5.39ポイント増加
- minが約7.21ポイント増加
- extent p05が約0.73ポイント増加
- sampling時間は約1.74%増加
- useful/ignore比率はほぼ同じ

Phase Aの2D sampling制約としては両方合格である。追加点がStage 5学習へ実際に有効かは、
sampling metricだけで最終決定できない。そのため、まず追加sourceをlocalだけに限定した
Stage 5学習ablationを行い、legacy/global-onlyからの学習改善を確認する方針とする。

### 8.7 Local-only学習ablation方針

「local-only」はglobalを除去する意味ではなく、従来のglobal evidenceへ
local-percentileだけを追加する。

```text
control:
  legacy/global evidence

experiment:
  global + tuned local-percentile

disabled:
  top-hat
  context grid
```

localは保持候補側を使用する。

```text
sample_stride=2
local window=31
local percentile=75
local min contrast=12
cleanup open/close=3/5
cleanup shape=ellipse
cleanup component area=15
```

データセット生成前に、A6の既存source ablation結果からretention候補の
`without_tophat`を確認する。これは新規parameter sweepではなく、すでに生成済みの
source flagsからtop-hat bitを除いた安全監査である。

source ablation結果:

| 構成 | zero | @1 | @8 | @16 | @32 | micro | points/frame | total points |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global+local+top-hat | 0 | 1.0 | 1.0 | 1.000000 | 0.999237 | 0.779025 | 12,845.15 | 94,399,001 |
| global+local | 0 | 1.0 | 1.0 | 0.999237 | 0.999237 | 0.624515 | 9,552.45 | 70,200,919 |
| global+top-hat | 0 | 1.0 | 1.0 | 1.000000 | 0.999237 | 0.772703 | 11,457.73 | 84,202,874 |

global+localでは2,620 teacher-valid BBox中2件が16点未満となるが、zeroは0、
recall@8は1.0である。local追加効果を分離する初回学習ablationとして許容する。

一方、2D metricではglobal+top-hatがglobal+localより高い。したがって今回の
global+local学習は最終性能候補の決定ではなく、local追加がStage 5学習へ実際に
寄与するかを測るcontrolled experimentと位置づける。

dataset build pipeline:

```text
pseudo3d/pipelines/
build_stage4_global_local_ablation_dataset.sh
```

処理:

```text
182 pseudo3D H5
  -> global+local point-cloud H5（PLYなし）
  -> strict annotations_renamedによるannotation H5
  -> Stage 5用flat collected directory
```

### 8.8 Local追加点とannotation teacherの構造的不整合

global+localデータセットのannotation textureを目視した結果、
local-percentile-onlyの追加点にはpositive segmentationがほとんど対応せず、
global由来点とsegmentationが強く一致した。

これはsource flagの伝搬ミスではない。現在のpoint annotationはすべてのsource点を
同じcontour maskで判定するが、dense teacherのpositive mask自体が従来globalと同系統の
`percentile85_area100`で作られている。そのため、global mask外を補完したlocal-only点は
strict XML BBox内であっても原則`ignore`となり、Stage 5のpositive教師にはならない。

また、Phase Aのpositive recallはこのglobal系teacherに対する保持率であり、
globalが見落とした新規大腿骨領域をlocalが救済したことの直接評価ではない。

したがって、現在のcollectedデータでStage 5学習を始める前に、
globalとlocalの輪郭候補をstrict XML BBoxとの整合度で比較する
annotation policyへ変更して再生成する。

globalとlocalは先にunionせず、各binary maskからBBox内の接続成分を独立候補として抽出する。
各候補に対して少なくとも次を計算する。

1. BBox中心と輪郭重心の距離をBBox対角長で正規化した値
2. filled contour area / BBox area
3. 従来のabsolute contour area

小さすぎる候補をarea ratioとabsolute areaで除外し、残った候補から
中心距離と面積十分性のscoreが最良のものをsourceに関係なく採用する。
採用されたlocal contourはweak point追加ではなく、filled mask全体を
通常のpositive segmentation teacherとする。global候補を優先するtie-breakは設けない。

`20250403_114201_376`を必須回帰例とし、この例でlocal候補がglobal候補より
高いBBox整合scoreで選ばれることを確認する。選択source、候補数、両候補のscore、
中心距離、area ratio、選択理由はH5の`frame_annotation`に保存する。

2026-08-01に`bbox_ranked_global_local`をteacher v2として実装した。v1の
`contour_in_bbox`はlegacy互換のため保持し、v2出力は既存データを上書きしない
`global_local_l75_w31_c12_area15_bboxrank_v2`へ分離する。

v2の初期ranking設定:

```text
min absolute contour area=20
min area ratio=0.02
sufficient area ratio=0.10
max normalized center distance=0.50
center weight=0.75
area weight=0.25
source-priority tie break=none
```

同一maskがglobal/local両方から得られた場合は`shared`、同一rankでmaskが異なる
完全同率はsource優先を避けるためinvalidとする。

`20250403_114201_376`の目視確認後、v2再annotation pipelineは260711の
enabled 182件を既定とし、入力manifest、annotated出力、collected出力のそれぞれで
182件を強制確認する構成に移行した。その後、Stage 5用ラベル境界の確定に伴い、
`train_stage5.sh`は次の再生成版を既定入力とする。

```text
/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/
global_local_l75_w31_c12_area15_bboxrank_v2_nobbox_bg/collected
```

Stage 5開始前に全H5の`label_mode=bbox_ranked_global_local`、
`ranked_teacher_schema=stage4_bbox_ranked_teacher_v2`、`no_bbox_label=0`、
`bbox_inside_non_contour_label=ignore`、video nameの一意性をpreflightする。

Stage 5開始前の最終目視用として、既存point-cloud H5から次のPLYを全182件生成する。

```text
global_local_l75_w31_c12_area15_bboxrank_v2_nobbox_bg/
  pointcloud_foreground/             # raw grayscale PLY
  pointcloud_annotated_foreground/   # BBox-ranked label/source colored PLY
```

raw/annotated H5と生成後の両PLYはそれぞれ182件を強制検証する。

## 9. 現時点の暫定判断

### 採用方向

| 項目 | 暫定値・方針 | 根拠 |
|---|---|---|
| global | 維持 | 除去時に全trainで12 zero-positive BBox |
| local | 維持 | subsetのp05/min改善 |
| local window | 31 | window=41から大幅高速化、tail低下が小さい |
| local percentile | 80を効率候補、75を保持候補 | p75は点数増と引き換えにminを改善 |
| local min contrast | 12 | recallをほぼ維持して点数削減 |
| top-hat kernel | 21 | 31/41は点数増とp05低下、11はmicro低下 |
| top-hat percentile | 85を効率基準、80を保持候補 | p80は点数増と引き換えにtail改善 |
| top-hat min response | 2を維持 | 0/2/4/8が完全同一 |
| top-hat shape | ellipse | rect/crossよりtail保持が良い |
| cleanup open/close | 3/5を調査基準として維持 | area軸を先に分離して評価 |
| cleanup component area | 効率候補10、保持候補15 | 各系列でrecall@16を維持する最大値 |
| context | Phase A中は無効 | evidenceとcontext costを分離 |

### 保留

| 項目 | 候補 |
|---|---|
| Phase A効率候補 | local p80、top-hat p85、area10 |
| Phase A保持候補 | local p75、top-hat p80、area15 |
| 初回Stage 5 ablation | global + local p75、window31、contrast12、area15。top-hat/contextは除外 |
| context stride/phase | Phase Bで調査 |

### 却下方向

| 候補 | 理由 |
|---|---|
| global無効 | zero-positive BBoxが12件発生 |
| cleanup component area=200 | tail recall低下が大きい |
| cleanup component area=25/50 | hard BBoxのselected positiveが17〜18点から8点へ低下 |
| efficiency area>=15 | hard BBoxが17点から8点へ低下 |
| retention area>=20 | hard BBoxが18点から8点へ低下 |
| cleanup open=5 | tail recall低下が大きい |
| local min contrast=0 | recall改善なしで約45%点数増加 |
| local percentile=90 | local無効に近いtail性能 |
| top-hat無効 | positive micro/p05/minの低下が大きい |
| top-hat kernel=31/41 | 点数増に対してp05が低下 |
| top-hat percentile=95 | top-hat無効に近いtail性能 |
| top-hat shape=rect/cross | ellipseよりtail保持が低い |

## 10. 今後の手順

1. retention候補の`without_tophat` source ablationを既存A6出力から確認
2. global/local独立輪郭候補をBBox整合scoreで選ぶannotation policyを実装
3. 既知例とscreening subsetで選択source・score・textureを確認し、閾値を1回調整
4. global+local annotation/collectedデータを再生成し、source別label数とtextureを確認
5. legacy/global-onlyを対照にStage 5を一度学習
6. Stage 5結果からlocal追加の実効性を判断
7. Phase B context調査は初回学習後に再開するか判断
8. 継続する場合、full evidence候補およびcontextを段階的に追加

## 10.1 Stage 5入力用ラベル方針

Stage 5へ渡す全182件の再生成版では、ラベル境界を次のように固定する。

| 点の位置 | ラベル | 理由 |
|---|---:|---|
| BBoxが存在しないフレーム | background (0) | 明示的な非対象フレームとして負例学習に使用 |
| BBox内かつ選択contour内 | positive (1) | BBox-ranked teacherが選んだ大腿骨候補 |
| BBox内かつ選択contour外 | ignore (-1) | BBox内の未抽出領域を誤って背景学習しないため |
| BBoxありフレームのBBox外 | background (0) | 対象BBox外の点 |

旧`bboxrank_v2`出力は上書きせず、再生成版を
`global_local_l75_w31_c12_area15_bboxrank_v2_nobbox_bg`として分離する。
Stage 5の既定入力もこの版へ切り替え、H5属性に加えてBBoxなしフレームの
全点が実際にbackgroundであることを訓練前preflightで検査する。

## 11. 関連ファイル

### 評価基盤

```text
pseudo3d/analysis/stage4_sampling_sweep_config.py
pseudo3d/analysis/stage4_sampling_evaluation.py
pseudo3d/analysis/sweep_stage4_sampling_parameters.py
pseudo3d/analysis/validate_stage4_sampling_manifest.py
pseudo3d/analysis/build_stage4_sampling_manifest.py
pseudo3d/analysis/select_stage4_sampling_screening_subset.py
pseudo3d/pipelines/build_stage4_global_local_ablation_dataset.sh
pseudo3d/pipelines/build_stage4_bbox_ranked_annotations.sh
pseudo3d/pipelines/export_stage4_bbox_ranked_pointcloud_visualizations.sh
pseudo3d/batch/export/batch_export_annotation_mask_visualization.sh
pseudo3d/batch/export/batch_export_point_cloud_ply.py
checks/stage4/check_stage4_bbox_ranked_teacher.py
checks/stage4/check_stage4_bbox_ranked_realdata.py
checks/stage4/check_stage4_bbox_ranked_label_policy.py
```

### Config

```text
pseudo3d/analysis/configs/stage4_dense_teacher_v1.yaml
pseudo3d/analysis/configs/stage4_bbox_ranked_teacher_v2.yaml
pseudo3d/analysis/configs/stage4_sampling_phase_a_coarse_search.yaml
pseudo3d/analysis/configs/stage4_sampling_phase_a_local_area0_search.yaml
pseudo3d/analysis/configs/stage4_sampling_phase_a_tophat_local31_area0_search.yaml
pseudo3d/analysis/configs/stage4_sampling_phase_a_finalist_screening_search.yaml
pseudo3d/analysis/configs/stage4_sampling_phase_a_component_area_refine_search.yaml
pseudo3d/analysis/configs/stage4_sampling_phase_a_full_train_finalists_search.yaml
```

### 出力root

```text
/mnt/data/3d_projects/pseudo3d_dataset/
stage4_sampling_parameter_sweep/260711
```

## 12. 更新履歴

| 日付 | 更新 |
|---|---|
| 2026-07-26 | A0全train、screening subset、cleanup coarse、local coarseの結果を初回記録。A2 top-hatを実行中として登録。 |
| 2026-07-26 | A2 top-hat全13設定を追記。kernel=21、ellipseを固定し、p85/p80を最終候補へ選定。 |
| 2026-07-26 | A4 finalist 12設定を追記。全設定zero=0、area=25以上で同一hard case由来とみられるrecall@16低下を確認。 |
| 2026-07-26 | hard BBoxを監査。area=25でselected positiveが17〜18点から8点へ低下することを確認し、area=5〜20のfine sweepを追加。 |
| 2026-07-27 | component area fine sweepを追記。効率候補area10、保持候補area15を選定し、2候補の全train確認へ移行。 |
| 2026-07-27 | A6全train比較を追記。両候補でzero=0、under16=0を確認し、最初のStage 5実験をglobal+local ablationとする方針を追加。 |
| 2026-07-27 | retention source ablationを追記。global+localでzero=0、recall@8=1.0を確認し、再現可能なdataset build pipelineを追加。 |
| 2026-07-27 | global+local学習データの抽出点とstrict annotation segmentationを重ねるバッチ可視化手順を追加。 |
| 2026-08-01 | annotation texture目視により、global系dense teacherがlocal-only救済点をpositive化できない構造的不整合を確認。Stage 5前のannotation policy修正を必須化。 |
| 2026-08-01 | global保存を廃し、global/local輪郭候補をBBox中心距離と面積十分性で同列選択するlocal-aware teacher方針に更新。 |
| 2026-08-01 | `bbox_ranked_global_local` teacher v2、H5選択メタデータ、source別輪郭色分け、既存point-cloud再利用pipeline、synthetic checkを実装。 |
| 2026-08-01 | v2の目視確認完了を受け、260711の182件全件buildを強制する件数検証とStage 5 trainingのv2専用入力preflightを追加。 |
| 2026-08-01 | Stage 5前の全件目視用にraw/annotated PLYを既存H5から各182件生成・件数検証するバッチpipelineを追加。 |
| 2026-08-01 | Stage 5入力ラベルを`BBoxなし=background`、`BBox内contour外=ignore`へ固定。旧版を保持したまま`bboxrank_v2_nobbox_bg`として182件を再生成するpipelineと訓練前実データ検査を追加。 |
