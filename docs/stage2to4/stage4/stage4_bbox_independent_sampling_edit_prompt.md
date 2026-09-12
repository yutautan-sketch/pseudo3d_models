Stage 5の学習データを確認したところ、現在のStage 4サンプリングでは、輝度依存のforeground抽出によってVOC BBox内の大腿骨pixelが点群から完全に落ち、教師上positive点が存在しないフレームが複数発生していることが判明しました。

この問題への対応として、まず `export_annotated_point_cloud_ply.py` 周辺に、点が存在しないフレームが複数発生してBBox非依存のStage 4 sampling Step 1〜3を実装してください。

ただし、この後に実教師データを用いたパラメータ調査基盤を実装する予定です。そのため、今回の値は最終設定ではなく暫定値です。各処理を再利用可能・設定可能・評価可能な形で実装してください。

# 1. 今回の方針

Stage 4のサンプリングは学習後の推論時にも使用するため、サンプリング処理自体をVOC BBoxや教師ラベルへ依存させません。

今回実装するのは以下の3処理です。

* Step 1: Base Grid Context
* Step 2: Local Percentile Evidence
* Step 3: Top-hat Evidence

既存のglobal/texture-alpha foreground抽出は、高信頼evidenceとして残します。

今回は以下を実装しません。

* BBox Rescue sampling
* 大腿骨を線状構造と仮定したline/vesselness抽出
* frame-level fallback sampling
* パラメータ探索そのもの

BBoxは、後続のパラメータ調査やannotation評価において、選択された点が教師positiveをどれだけ救えているかを測るために使用します。sampling maskの生成条件には使用しないでください。

# 2. 方法A〜Hとの対応

今回のStep 1〜3は、これまでの候補手法と以下のように対応します。

Step 1: Base Grid Context

* 方法E: foreground + context/background sampling
* 方法H: low-resolution base grid + evidence oversampling

Step 2: Local Percentile Evidence

* 方法B: local percentile / local rank
* 方法F: multi-source samplingの一部
* 方法Aのglobal thresholdを補完する処理

Step 3: Top-hat Evidence

* 方法Dを「線状構造強調」ではなく、「局所背景より明るい小〜中サイズ構造の抽出」として採用
* 方法F: multi-source samplingの一部

保留・不採用:

* 方法C BBox Rescue: Stage 4には入れない
* 方法G Frame Fallback: Base Gridと役割が近く、positive保証でもないため保留
* 線状性・elongationを必須条件にする処理: 採用しない

# 3. 実装位置に関する注意

最初に、現在の `export_annotated_point_cloud_ply.py` と関連ファイルの役割を確認してください。

今回の処理は、輝度閾値でpixelが破棄される前、つまり以下へアクセスできる段階で行う必要があります。

* local_encoder_images
* pixel座標
* pixel_to_image
* prior/corr tracking geometry
* frame_order / frame_index

もし `export_annotated_point_cloud_ply.py` が、すでに抽出済みの点群h5をPLYへ変換するだけのファイルであり、元画像上で失われたpixelを復元できない構造なら、そのファイル内部だけに無理に実装しないでください。

その場合は、例えば以下の共有utilityを新設してください。

* `src/utils/pseudo3d_sampling.py`

そして、必要に応じて以下から同じ関数を呼ぶ構造にしてください。

* point cloud生成処理
* annotated point cloud生成処理
* `export_annotated_point_cloud_ply.py`
* 後続のparameter sweepスクリプト

調査用と本番用でsampling処理を別実装しないことが重要です。

# 4. Sampling全体の構造

新方式は、以下のようなpixel-level sampling maskを作ります。

1. 既存global/texture-alpha evidence
2. Base Grid Context
3. Local Percentile Evidence
4. Top-hat Evidence
5. 同一pixelを重複させずsource情報を統合
6. 選択pixelを3Dへ投影

概念的には以下です。

selected_mask =
global_evidence_mask
OR context_grid_mask
OR local_percentile_mask
OR top_hat_mask

ただし、単なるbool maskだけでなく、各pixelがどのsourceで選択されたかを保持してください。

# 5. Step 1: Base Grid Context

## 目的

画像全体から低密度gridでpixelを残し、global thresholdで落ちた領域もcontext pointとして保持します。

これはpositiveを直接保証する処理ではありませんが、BBox内のpixelが点群上から完全に消える可能性を下げます。また、Stage 5でbackground/context分類を強化するためにも利用します。

## 実装要件

* 画像全体に対する規則的grid sampling
* BBox非依存
* evidence maskとは独立
* 同じpixelがevidenceにも選ばれた場合は重複点を作らない
* context-only点とevidence点を区別する

CLIまたは設定項目の候補:

* `include_context_grid`
* `context_grid_stride`
* `context_grid_phase`

`context_grid_phase` は、後続の調査で比較できるよう、可能なら以下に対応してください。

* `origin`
* `centered`
* `dual`

定義例:

* origin: offset=(0, 0)
* centered: offset=(stride//2, stride//2)
* dual: originとcenteredのunion

frame-cycledは今回は必須ではありません。

暫定値:

* `context_grid_stride = 4`
* `context_grid_phase = origin`

ただし、これを最終推奨値として固定しないでください。後続調査で4, 6, 8, 12等を比較します。

256×256画像の場合の概算:

* stride=4: 4096 points/frame
* stride=8: 1024 points/frame

点数増加を後で測定できる統計を保存してください。

# 6. Step 2: Local Percentile Evidence

## 目的

画像全体では高輝度でなくても、局所近傍の中では相対的に目立つpixelを抽出します。

大腿骨が線状、短楕円状、輪切り状のどの見え方でも利用できる、形状非依存の処理にしてください。

## 基本定義

各pixelについて、周辺window内のpercentile閾値を計算します。

概念:

local_threshold(x, y)
= percentile(window around (x, y), p)

local_mask(x, y)
= image(x, y) >= local_threshold(x, y)

ただし、ほぼ一様な暗い領域で同値pixelが大量に選択されるのを防ぐため、局所contrast条件も用意してください。

例:

local_contrast
= image - local_median

local_mask
= percentile_condition
AND local_contrast >= min_local_contrast

## 実装要件

設定項目:

* `enable_local_percentile`
* `local_window_size`
* `local_percentile`
* `local_min_contrast`

暫定値:

* `local_window_size = 41`
* `local_percentile = 80`
* `local_min_contrast = 4`

制約:

* window sizeは正の奇数
* 画像境界の扱いを明示する
* Pythonのpixel単位二重ループは避ける
* 既存依存関係を確認し、利用可能なら効率的なpercentile filterを使う
* SciPy等を新たに使う場合は、既存環境・requirementsとの整合を確認する
* SciPyがない場合も、極端に遅いnaive実装にはせず、patch/histogram等の効率的な方法を検討する

後続のparameter sweepでwindow size、percentile、min contrastを変更できるよう、処理を純粋関数として切り出してください。

できれば以下のような返り値にしてください。

* `local_mask`
* `local_threshold_map`
* `local_median_map`
* `local_contrast_map`

本番では不要なmapを常時保存する必要はありませんが、調査・debug時に取得できる構造が望ましいです。

# 7. Step 3: Top-hat Evidence

## 目的

Top-hatを線状構造専用処理としてではなく、局所背景より明るい小〜中サイズの構造を抽出するために使用します。

対象は以下を含みます。

* 線状断面
* 点状断面
* 短楕円状断面
* 輪切り状の断面

elongationやline-likenessを条件にしないでください。

## 基本定義

white top-hat:

top_hat = image - morphological_opening(image)

## 実装要件

設定項目:

* `enable_tophat`
* `tophat_kernel_size`
* `tophat_percentile`
* `tophat_min_response`
* `tophat_morph_shape`

暫定値:

* `tophat_kernel_size = 21`
* `tophat_percentile = 85`
* `tophat_min_response = 2`
* `tophat_morph_shape = ellipse`

percentile thresholdは、zero responseが大量に存在する場合に閾値が0にならないよう注意してください。

推奨挙動:

1. top-hat responseを計算
2. 正のresponseが存在しなければ空mask
3. percentileはpositive response上で計算
4. `response >= percentile_threshold`
5. かつ `response >= tophat_min_response`

後続調査で以下を比較する予定です。

* kernel size: 11, 21, 31, 41
* percentile: 80, 85, 90, 95
* min response: 0, 2, 4, 8

そのため、値をコードへ固定しないでください。

# 8. Global Evidenceの扱い

既存のtexture-alpha / percentile / Otsu等によるforeground抽出は削除しないでください。

新方式では、既存抽出を高信頼global evidenceとして扱います。

既存互換を保つため、sampling modeを分けることを推奨します。

例:

* `legacy`

  * 現在の挙動を維持
* `combined_v2`

  * global evidence
  * Base Grid Context
  * Local Percentile Evidence
  * Top-hat Evidence

既存CLIのデフォルト挙動は可能な限り壊さないでください。新方式は明示的に有効化できるようにしてください。

# 9. Source flagsと重複統合

同じframe内の同じpixelが複数sourceで選ばれる場合、複数の3D点を作らず、1点に統合してください。

各pixelにuint16等のbit flagsを持たせます。

例:

* bit 0: global_evidence
* bit 1: local_percentile
* bit 2: top_hat
* bit 3: context_grid

将来追加用bitは空けておいて構いません。

例:

```python
SOURCE_GLOBAL = 1 << 0
SOURCE_LOCAL_PERCENTILE = 1 << 1
SOURCE_TOPHAT = 1 << 2
SOURCE_CONTEXT_GRID = 1 << 3
```

処理方法としては、frameごとに画像サイズと同じ `source_flags_map` を作り、各maskに対応するbitをORする形が分かりやすいです。

```python
source_flags_map[global_mask] |= SOURCE_GLOBAL
source_flags_map[local_mask] |= SOURCE_LOCAL_PERCENTILE
source_flags_map[tophat_mask] |= SOURCE_TOPHAT
source_flags_map[context_mask] |= SOURCE_CONTEXT_GRID

selected_mask = source_flags_map != 0
```

これにより、pixel重複を自然に排除できます。

出力には少なくとも以下を追加してください。

* `source_flags [K] uint16`

既存の `source_type` は互換性のため残して構いません。

`source_type` を残す場合の代表source優先順位案:

1. global evidence
2. local percentile
3. top-hat
4. context grid only

ただし、正確な情報は `source_flags` を正としてください。

# 10. Confidenceの扱い

既存の `alpha` や `confidence` の意味を、黙って変更しないでください。

現在のフィールドの意味を確認し、必要なら新たに:

* `sampling_confidence`

を追加してください。

暫定的なsource confidence例:

* global evidence: 既存alphaまたは既存confidence
* local percentile: 0.7
* top-hat: 0.7
* context-only: 0.0

複数sourceに該当する場合は最大値を使って構いません。

ただし、後続調査では固定confidence値よりもsource flagsを主に使う予定です。固定値を学習上の真の信頼度とはみなさないでください。

# 11. 出力統計

後続のparameter sweepで点数増加を評価するため、少なくとも以下を計算・表示・metadata保存できるようにしてください。

全体およびframe単位:

* total selected points
* legacy/global evidence points
* local percentile selected points
* top-hat selected points
* context grid selected points
* context-only points
* evidence points
* 複数source overlap points
* points per frame min / mean / median / max
* legacy方式に対するpoint count multiplier

ここでsource別件数は、各bitが立っている点数として数えてください。同じ点が複数source件数へ含まれて構いません。

一方、総点数は重複排除後の一意pixel数です。

# 12. CLI案

現在のスクリプト構成に合わせて調整して構いませんが、概ね以下を用意してください。

* `--sampling_mode legacy|combined_v2`

Base Grid:

* `--include_context_grid`
* `--context_grid_stride`
* `--context_grid_phase origin|centered|dual`

Local Percentile:

* `--enable_local_percentile`
* `--local_window_size`
* `--local_percentile`
* `--local_min_contrast`

Top-hat:

* `--enable_tophat`
* `--tophat_kernel_size`
* `--tophat_percentile`
* `--tophat_min_response`
* `--tophat_morph_shape rect|ellipse|cross`

暫定実行例:

```bash
python pseudo3d/export_annotated_point_cloud_ply.py \
  ...既存引数... \
  --sampling_mode combined_v2 \
  --include_context_grid \
  --context_grid_stride 4 \
  --context_grid_phase origin \
  --enable_local_percentile \
  --local_window_size 41 \
  --local_percentile 80 \
  --local_min_contrast 4 \
  --enable_tophat \
  --tophat_kernel_size 21 \
  --tophat_percentile 85 \
  --tophat_min_response 2 \
  --tophat_morph_shape ellipse
```

実際のファイル名や既存CLIに合わせて修正してください。

# 13. 後続のParameter Sweepを考慮した設計

次の開発では、実教師pseudo3dデータとVOC annotationを用いて、以下を比較するparameter sweep基盤を実装します。

* positive frame rescue率
* positive pixel recall
* BBox coverage
* zero-positive frame数
* total points/frame
* context/evidence点数
* point count multiplier
* 動画単位のworst-case性能

そのため、今回のsampling処理は以下を満たしてください。

* パラメータを直書きしない
* config/dataclass/dictとして渡せる
* 同じ関数を本番処理とsweepで利用できる
* deterministicである
* 2D maskだけを生成できる
* 必要ならintermediate score/mapを返せる
* 3D投影処理とsampling mask生成を分離する
* 教師BBoxを受け取らなくても動作する
* source別maskを個別に取得できる
* 全mask統合後のsource flagsを取得できる

推奨関数構成例:

```python
@dataclass
class PointSamplingConfig:
    sampling_mode: str
    include_context_grid: bool
    context_grid_stride: int
    context_grid_phase: str
    enable_local_percentile: bool
    local_window_size: int
    local_percentile: float
    local_min_contrast: float
    enable_tophat: bool
    tophat_kernel_size: int
    tophat_percentile: float
    tophat_min_response: float
    tophat_morph_shape: str
```

```python
def build_context_grid_mask(...):
    ...

def build_local_percentile_evidence(...):
    ...

def build_tophat_evidence(...):
    ...

def build_sampling_source_flags(
    image,
    global_mask,
    config,
    *,
    return_debug_maps=False,
):
    ...
```

返り値例:

```python
{
    "selected_mask": ...,
    "source_flags": ...,
    "global_mask": ...,
    "local_percentile_mask": ...,
    "tophat_mask": ...,
    "context_grid_mask": ...,
    "sampling_confidence": ...,
    "debug_maps": ...,
}
```

実際の既存コード構造に合わせて調整して構いません。

# 14. 最低限のテスト

可能なら、小さいsynthetic imageを使った単体テストまたは確認コードを追加してください。

確認したい内容:

1. Base Grid

   * strideとphaseに応じて期待点数になる
   * origin/centered/dualが正しく異なる

2. Local Percentile

   * global thresholdでは落ちるが、暗い背景中で相対的に明るいpixelを拾える
   * 一様画像では `local_min_contrast` により大量選択されない

3. Top-hat

   * 小さい明るい円形構造を拾える
   * 短い棒状構造も拾える
   * 線状性を条件にしていないことを確認する
   * responseがすべて0なら空maskになる

4. Duplicate merge

   * 同じpixelが複数sourceに該当しても1点だけになる
   * source flagsには複数bitが立つ

5. Legacy compatibility

   * legacy modeで既存の選択結果・出力形式が壊れない

# 15. 実装完了時に報告してほしい内容

* 変更したファイル一覧
* 新設したutilityやdataclass
* sampling modeごとの処理フロー
* source flagsのbit定義
* 既存出力schemaとの差分
* 新規CLI引数
* 暫定実行コマンド
* synthetic testまたは簡易動作確認結果
* 後続parameter sweepで再利用する関数
* 未解決事項や追加依存関係

今回はStep 1〜3の実装までとし、VOC BBoxを用いたparameter sweepや最終パラメータ決定はまだ行わないでください。

---

# Phase 7: Stage 4実装確認・実データ受け入れ確認

## 1. Phase 7の位置づけ

Phase 1〜6で追加したBBox非依存sampling実装について、synthetic test、CLI、H5 schema、annotation、PLY、可視化、batch処理までの整合性を確認してください。

Phase 7は実装の確認段階です。新しいsampling方式の最終パラメータ決定や、VOC BBoxを用いた大規模parameter sweepはまだ行いません。ただし、少数の実データと既存annotationを読み取り専用の評価対象として使用し、後続sweepへ進める状態かを判定することは含みます。BBoxや教師ラベルをsampling mask生成へ入力してはいけません。

今回確認する実装範囲は以下です。

* Phase 1〜2: `src/utils/pseudo3d_sampling.py` とpoint cloud生成処理へのStep 1〜3統合
* Phase 3: `check_pseudo3d_sampling_synthetic.py` によるsynthetic確認
* Phase 4〜5: annotation H5、annotated PLY、annotation mask visualizationへの新schema伝播とsource別表示
* Phase 6: single/batch CLI、shell script、pipeline間の引数・ファイル名・summary伝播

## 2. Phase 7の原則

* 最初に現状を検査し、問題が見つかった場合だけ最小限の修正を行う
* `legacy` の既存挙動を基準として維持する
* `combined_v2` は `foreground` point modeでのみ確認する
* samplingは `local_encoder_images` 上で、輝度閾値によるpixel破棄より前に行われることを確認する
* sampling mask生成関数へVOC BBox、教師label、annotation maskを渡さない
* 同一frame・同一pixelは一意の点とし、複数sourceは `source_flags` のbit ORで表す
* `confidence` の既存意味を変えず、新sampling由来の値は `sampling_confidence` で扱う
* 暫定パラメータの性能を最終結論として扱わない
* 実データが利用できない環境では、その項目を失敗扱いにせず「未実施」として、必要な実行コマンドと入力条件を残す

## 3. 対象ファイル

最低限、以下を相互に照合してください。

* `src/utils/pseudo3d_sampling.py`
* `pseudo3d/export/export_pseudo3d_point_cloud.py`
* `pseudo3d/batch/export/batch_export_pseudo3d_point_cloud.py`
* `check_pseudo3d_sampling_synthetic.py`
* `pseudo3d/annotation/annotate_pseudo3d_point_cloud.py`
* `pseudo3d/batch/annotation/batch_annotate_pseudo3d_point_cloud.py`
* `pseudo3d/export/export_annotated_point_cloud_ply.py`
* `pseudo3d/batch/export/batch_export_annotated_point_cloud_ply.py`
* `pseudo3d/export/export_annotation_mask_visualization.py`
* 関連するsingle/batch/pipelineの `.sh`
* `scripts/utils/collect_annotated_pseudo3d_h5.py` と関連shell script
* Stage 5側でannotated H5を読むdataset、inference、visualization処理

## 4. 環境と静的確認

以下を確認してください。

1. `numpy`、`opencv-python`、`scipy`、`h5py` 等の既存依存でimportできる
2. `scipy.ndimage.percentile_filter` と `median_filter` がrequirements/実行環境と整合する
3. 対象Pythonファイルがcompile/importできる
4. CLIの `--help` がsingle/batchとも正常に表示される
5. shell scriptからPython CLIへ存在しない引数を渡していない
6. singleとbatchでsampling configのdefaultおよびoverride結果が一致する
7. `legacy` は追加オプションを指定してもStep 1〜3を有効にしない
8. `combined_v2` のdefaultは現在の暫定値を使用し、各BooleanOptionalActionで明示的に無効化できる

入力値validationも確認してください。

* strideは1以上
* local window sizeとtop-hat kernel sizeは正の奇数
* phaseとmorph shapeは定義済みchoicesのみ
* image/global mask/confidence mapのshape不一致は明示的に失敗する
* `combined_v2` と `point_mode=grid|dense` の不正な組み合わせは明示的に失敗する

## 5. Synthetic test

既存の `check_pseudo3d_sampling_synthetic.py` を実行し、少なくとも以下を確認してください。

1. Base Grid

   * origin、centered、dualの点数とoffsetが正しい
   * dualは重複を含まずunionになる
   * strideが画像サイズを割り切らない場合やstride=1でも正しい

2. Local Percentile

   * 一様暗画像は `local_min_contrast` により空maskになる
   * 局所的に明るい点、短い棒、円形領域を拾える
   * debug指定時だけthreshold/median/contrast mapを返す
   * 境界付近でも出力shapeとdtypeが維持される

3. Top-hat

   * responseが全て0なら空maskになる
   * percentileはpositive responseだけで計算される
   * 点状、円形、短い棒状構造を拾い、線状性を必須にしない
   * rect、ellipse、crossが実行できる
   * debug mapのresponse、opened、thresholdが妥当である

4. Source merge

   * 複数sourceに該当するpixelは1点だけ生成される
   * bit 0〜3が正しくORされる
   * `source_type` の代表source優先順位がglobal、local、top-hat、contextの順になる
   * context-only、evidence、overlapの統計が手計算と一致する

5. Confidence

   * globalは既存alpha由来、local/top-hatは暫定値、context-onlyは0.0
   * 複数sourceでは最大値になる
   * `confidence` と `sampling_confidence` が混同されない

6. Projection

   * legacyとcombined_v2のpixel座標が同じ `pixel_to_image` とtracking geometryで3D投影される
   * frame index、frame order、pixel_xy、全point-level配列の長さが一致する
   * `max_points_per_frame` 適用後もsource flagsと統計が実際の保存点に対応する

既存テストに不足する境界条件があれば追加してください。random subsamplingを確認する場合はseedを固定し、deterministicであることも確認してください。

## 6. Legacy互換性確認

同じpseudo3d入力、geometry、texture設定、stride、seedを使い、変更前相当のlegacy出力と現行 `--sampling_mode legacy` を比較してください。

最低限比較する項目:

* point数とframe別point数
* `points`、`pixel_xy`、`frame_index`、`frame_order`
* `intensity`、`rgb`、`alpha`、`confidence`
* tracking/geometry関連metadata
* annotation結果のlabel、BBox内外判定、frame対応
* PLYの点数と既存property

浮動小数点は適切なtoleranceで比較し、それ以外は完全一致を原則としてください。新規fieldが追加されること自体は許容します。

legacy出力の新規fieldについては以下を確認してください。

* foreground点の `source_flags` はglobal bit
* grid/dense点の `source_flags` はcontext bit
* 既存互換のためlegacy `source_type` は従来値を維持してよい
* `sampling_confidence` は既存alpha/confidenceと整合する

## 7. combined_v2の実データsmoke test

代表的な少数動画を選び、同一入力からlegacyとcombined_v2を出力してください。可能なら以下を含めます。

* 従来zero-positive frameが発生した動画
* 暗いframeを含む動画
* 通常輝度の動画
* frame数または画像サイズが異なる動画

暫定設定:

```bash
python pseudo3d/export/export_pseudo3d_point_cloud.py \
  --input_h5 INPUT_PSEUDO3D_H5 \
  --geometry_key corr \
  --preset percentile85_area100 \
  --output_h5 OUTPUT_POINTCLOUD_H5 \
  --output_ply OUTPUT_POINTCLOUD_PLY \
  --point_mode foreground \
  --sample_stride 1 \
  --sampling_mode combined_v2 \
  --include_context_grid \
  --context_grid_stride 4 \
  --context_grid_phase origin \
  --enable_local_percentile \
  --local_window_size 41 \
  --local_percentile 80 \
  --local_min_contrast 4 \
  --enable_tophat \
  --tophat_kernel_size 21 \
  --tophat_percentile 85 \
  --tophat_min_response 2 \
  --tophat_morph_shape ellipse
```

各出力で以下を確認してください。

* 全frameの処理が完了する
* context gridにより、global evidenceが空のframeでも選択点が存在する
* `pixel_xy` が画像範囲内で、同一frame内に重複座標がない
* source別point数の和と総point数を混同していない
* point count multiplierが保存後の一意点数に基づく
* frame別配列の長さがnum_frames、point別配列の長さがnum_points
* NaN/Infがpoints、confidence、sampling confidenceに存在しない
* point数とメモリ・処理時間の増加が運用不能な規模でない

## 8. H5 schema・metadata確認

point cloud H5で以下を確認してください。

Point-level dataset:

* `source_flags [K] uint16`
* `sampling_confidence [K] float32`
* `source_type [K]`
* 既存のpoints/rgb/intensity/alpha/frame/pixel/confidence datasets

Frame-level dataset:

* `per_frame_counts`
* `per_frame_global_counts`
* `per_frame_local_percentile_counts`
* `per_frame_tophat_counts`
* `per_frame_context_grid_counts`
* `per_frame_context_only_counts`
* `per_frame_evidence_counts`
* `per_frame_overlap_counts`

Metadata:

* sampling modeと全sampling parameter
* source flagのbit定義
* total/source別/context-only/evidence/overlap統計
* points per frameのmin/mean/median/max
* legacy point count multiplier

dataset shape、dtype、metadata値、実データから再計算した統計が一致することを確認してください。特に `max_points_per_frame` 後の統計であることを確認してください。

## 9. Annotation・収集・Stage 5伝播確認

新しいpoint cloud H5をannotation処理へ渡し、以下を確認してください。

* `source_flags`、`sampling_confidence`、frame-level count datasetsがannotated H5へ欠落せず伝播する
* 古いH5に新fieldがない場合はfallbackが働き、その事実と理由がmetadataへ記録される
* fallbackで推定した値を、実際に保存されていた値として誤表示しない
* point数とpoint orderがannotation前後で維持される
* batch annotation summaryへsampling modeとfallback有無が出る
* collect処理後も新datasetとmetadataが維持される
* Stage 5 datasetが未知の追加datasetで壊れず、既存feature/label読込が同じである
* 必要ならStage 5から `source_flags` と `sampling_confidence` を参照できるが、今回それを学習特徴へ強制追加しない

annotation評価では、教師をsampling条件に使わず、結果の測定だけに使用してください。少なくとも参考値として以下をlegacyとcombined_v2で比較し、Phase 8以降のsweep基盤が必要なことを確認します。

* zero-positive frame数
* positive点が新たに存在するようになったframe数
* BBox内point数
* source別のBBox内point数
* 総point数とpoint count multiplier

この少数例の結果だけでパラメータを変更・確定しないでください。

## 10. PLY・可視化確認

通常PLY、annotated PLY、annotation mask visualizationを出力し、以下を確認してください。

* PLY headerのproperty数・型・vertex数がbodyと一致する
* `source_flags` と `sampling_confidence` がPLYへ出力される
* legacy H5のfallbackでもPLY exportが失敗しない
* source表示でglobal、local percentile、top-hat、context-only、複数sourceが識別できる
* annotation source表示でpositive/negativeとsourceの関係を確認できる
* point sizeや描画順による見かけだけで重複点があると誤判定しない
* frame visualization上のpixel位置とpoint cloudの `pixel_xy` が一致する
* context gridのphase/strideが視覚的にもCLI指定と一致する

少なくとも1つのlegacy出力、1つのcombined_v2出力、1つの古いschemaからのfallback出力を確認してください。

## 11. Batch・shell・pipeline確認

single exportとbatch exportへ同じ設定を与え、同一動画のH5内容と統計が一致することを確認してください。

* `sampling_mode` を出力ファイル名/tagへ含め、legacyを誤って上書きしない
* `--skip_existing` が異なるsampling modeのファイルを取り違えない
* summary CSVにsampling mode、source別統計、frame統計、失敗理由が出る
* shell変数がsingle/batch/annotation/PLY/visualizationへ一貫して伝播する
* batchは1件の失敗時に指定どおり停止または継続する
* pipeline defaultが意図せず既存legacy運用をcombined_v2へ変更していない

## 12. 合格基準

以下を全て満たせば、Phase 7を合格としてください。

* synthetic testと追加した境界testが全て成功する
* 対象Pythonファイルのcompile/importとCLI helpが成功する
* legacyの既存point選択・投影・annotation結果に意図しない差分がない
* combined_v2で4 sourceが設定どおり動作し、重複pixelが生成されない
* source flags、confidence、全統計が保存内容からの再計算と一致する
* point cloudからannotation、collect、Stage 5、PLY、可視化までschemaが伝播する
* singleとbatchで同じ入力・設定の結果が一致する
* 実データsmoke testで例外、NaN/Inf、shape不整合、明らかな座標ずれがない
* sampling mask生成がBBox/教師非依存である
* 未実施項目、性能上の懸念、既知の制約が明示されている

## 13. Phase 7完了時の報告形式

以下を報告してください。

1. 実行環境と使用commit
2. 実行したコマンド一覧
3. synthetic/static/legacy/combined_v2/schema/annotation/PLY/batchの各結果
4. 実データに使用した動画と選定理由
5. legacy対combined_v2のpoint数・frame統計・参考annotation指標
6. source別件数とoverlap/context-only件数
7. 変更したファイルと修正理由
8. 合格、不合格、未実施の項目一覧
9. Phase 8以降へ持ち越す事項

Phase 7で不具合を修正した場合は、修正前に再現testを追加し、修正後に関連するsingle/batch/end-to-end確認を再実行してください。

## 14. Phase 7の対象外

以下はPhase 7では実装しません。

* 全データを対象としたparameter sweep
* 最終sampling parameterの決定
* BBox Rescue sampling
* frame-level fallback sampling
* line/vesselness/elongation条件
* source confidenceの学習的calibration
* Stage 5モデル構造やlossへのsource情報の本格統合
* BBoxまたは教師labelに依存する本番sampling

これらはPhase 7の確認結果を受けた後続Phaseで別途設計してください。
