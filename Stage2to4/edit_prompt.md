今回の編集目的は、Stage 5の本学習用入力として、foreground evidence点群だけでなく、閾値処理でピクセルを削除しないgrid / semi-dense slice point cloudを扱えるようにすることです。

重要な設計方針は以下です。

1. `export_pseudo3d_point_cloud.py` に `--point_mode` を追加する。
2. `--point_mode foreground` は現行挙動を維持する。
3. `--point_mode grid` では、輝度・alpha thresholdで点を削除せず、local image上の規則格子点を3D world座標へ投影する。
4. `grid` でも `intensity`, `alpha`, `confidence`, `frame_index`, `frame_order`, `pixel_xy`, `source_type`, `per_frame_counts` は保存する。
5. `alpha` / `confidence` は点の削除条件ではなく、モデル入力特徴量・foreground priorとして保存する。
6. `annotate_pseudo3d_point_cloud.py` では、point cloudが `foreground` 以外、特に `grid` の場合、現行のlargest contour抽出が失敗したとき、またはpositive点数が少なすぎるときに、閾値を緩めてmask取得を再試行するfallback機能を追加する。
7. fallbackに失敗した場合、BBox内は `ignore`、BBox外は `background` とするのを基本方針とする。
8. `grid` では、BBox内かつcontour外の点を強いbackgroundにすると、輪郭抽出で落ちた大腿骨部分を誤ってbackground学習する危険があるため、初期値では `ignore` にできるようにする。

具体的な編集内容は以下です。

## 1. `export_pseudo3d_point_cloud.py` の編集

### 1-1. CLI引数を追加

`build_parser()` に以下を追加してください。

```python
parser.add_argument(
    "--point_mode",
    type=str,
    default="foreground",
    choices=["foreground", "grid", "dense"],
    help=(
        "Point selection mode. "
        "foreground: current behavior; select foreground pixels using alpha/intensity. "
        "grid: keep regular grid pixels without alpha/intensity removal. "
        "dense: same as grid with effective sample_stride=1."
    ),
)
```

### 1-2. grid用の点選択関数を追加

現在の `_choose_foreground_pixels()` はそのまま残し、新たに `_choose_grid_pixels()` を追加してください。

仕様：

```python
def _choose_grid_pixels(
    image: np.ndarray,
    texture: np.ndarray,
    *,
    sample_stride: int,
    dense: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

挙動：

* `image_to_uint8_gray(image)` でgrayを作る。
* `texture` にalpha channelがあれば、そのalphaを使う。
* alpha channelがなければ、全点のalphaを255にする。
* `dense=True` の場合は `sample_stride=1` 相当。
* `grid` の場合は `sample_stride` 間隔で全local pixelを選ぶ。
* alpha値は選ばれたgrid点に対応するものを返す。
* 点の選択にalphaやintensity thresholdは使わない。

擬似コード：

```python
gray = image_to_uint8_gray(image)
h, w = gray.shape

if dense:
    step = 1
else:
    step = max(1, int(sample_stride))

ys, xs = np.mgrid[0:h:step, 0:w:step]
xs = xs.reshape(-1).astype(np.float32)
ys = ys.reshape(-1).astype(np.float32)

if texture.ndim == 3 and texture.shape[-1] == 4:
    alpha = texture[..., 3].astype(np.uint8)
else:
    alpha = np.full(gray.shape, 255, dtype=np.uint8)

alpha_values = alpha[ys.astype(np.int64), xs.astype(np.int64)]
return xs, ys, alpha_values
```

### 1-3. 点選択の分岐を追加

`build_frame_point_cloud()` に `point_mode: str` 引数を追加してください。

現在は各frameで `_choose_foreground_pixels()` を呼んでいますが、以下のように分岐してください。

```python
if point_mode == "foreground":
    xs, ys, alpha_values = _choose_foreground_pixels(...)
elif point_mode == "grid":
    xs, ys, alpha_values = _choose_grid_pixels(
        image,
        texture,
        sample_stride=sample_stride,
        dense=False,
    )
elif point_mode == "dense":
    xs, ys, alpha_values = _choose_grid_pixels(
        image,
        texture,
        sample_stride=1,
        dense=True,
    )
else:
    raise ValueError(...)
```

`max_points_per_frame` によるsubsampleは、foreground/grid/denseすべてで共通に適用してください。

### 1-4. metadataにpoint_modeを保存

`meta` に以下を追加してください。

```python
"point_mode": args.point_mode,
```

また、`build_frame_point_cloud()` 呼び出し時にも `point_mode=args.point_mode` を渡してください。

### 1-5. 既存挙動を壊さない

`--point_mode` を指定しない場合は `foreground` なので、既存コマンドは同じ結果になるようにしてください。

## 2. `annotate_pseudo3d_point_cloud.py` の編集

### 2-1. point cloudのpoint_modeを読む

`load_point_cloud_h5()` はすでに `point_cloud_attrs` を返しているので、main内で以下のように取得してください。

```python
point_mode = str(point_cloud_attrs.get("point_mode", "foreground"))
```

古いH5には `point_mode` がないため、defaultは `"foreground"` としてください。

### 2-2. CLI引数を追加

`build_parser()` に以下を追加してください。

```python
parser.add_argument(
    "--enable_contour_fallback",
    action="store_true",
    help=(
        "Enable relaxed contour extraction fallback when the initial contour "
        "is invalid or yields too few positive points. Mainly intended for "
        "non-foreground point clouds such as point_mode=grid."
    ),
)

parser.add_argument(
    "--fallback_only_non_foreground_point_cloud",
    action="store_true",
    default=True,
    help=(
        "Apply contour fallback only when point_cloud attrs indicate "
        "point_mode != foreground."
    ),
)

parser.add_argument(
    "--fallback_percentiles",
    type=str,
    default="85,80,75,70",
    help="Comma-separated percentile thresholds tried during contour fallback.",
)

parser.add_argument(
    "--fallback_min_positive_points",
    type=int,
    default=5,
    help=(
        "If a valid contour yields fewer than this number of positive points "
        "on the current point cloud, try fallback when enabled."
    ),
)

parser.add_argument(
    "--bbox_inside_non_contour_label",
    type=str,
    default="ignore",
    choices=["ignore", "background"],
    help=(
        "Label assigned to points inside a VOC bbox but outside the accepted "
        "contour. For grid point clouds, ignore is safer because contour "
        "extraction may miss true femur pixels."
    ),
)
```

必要であれば、`--fallback_min_component_areas` などは後で追加する。初期実装ではpercentile fallbackだけでよい。

文字列parserとして、既存の `_parse_offsets()` に似た `_parse_float_list()` を追加してください。

```python
def _parse_float_list(text: str) -> list[float]:
    values = []
    for token in re.split(r"[, ]+", text.strip()):
        if token:
            values.append(float(token))
    return values
```

### 2-3. positive点数を評価する関数を追加

fallback判定では、輪郭がvalidかどうかだけでなく、その輪郭に対応する点群上のpositive点数も見る必要があります。

以下のようなhelperを追加してください。

```python
def count_points_in_mask(
    *,
    point_cloud: dict[str, np.ndarray],
    frame_order_value: int,
    mask: np.ndarray,
) -> int:
    frame_order = point_cloud["frame_order"].astype(np.int32)
    pixel_xy = point_cloud["pixel_xy"].astype(np.float32)

    frame_mask = frame_order == int(frame_order_value)
    if not np.any(frame_mask):
        return 0

    xy = np.rint(pixel_xy[frame_mask]).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, mask.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, mask.shape[0] - 1)

    return int(mask[xy[:, 1], xy[:, 0]].sum())
```

### 2-4. fallback用contour抽出関数を追加

初回の `build_largest_contour_mask_from_binary()` はそのまま使い、fallbackではthreshold percentileを変えた `AlphaTextureConfig` を作って再試行してください。

例：

```python
def try_contour_with_fallbacks(
    *,
    image: np.ndarray,
    bbox: LocalBBox,
    base_texture_config: AlphaTextureConfig,
    min_alpha: int,
    min_contour_area: float,
    point_cloud: dict[str, np.ndarray],
    frame_order_value: int,
    enable_fallback: bool,
    fallback_percentiles: list[float],
    fallback_min_positive_points: int,
) -> dict[str, Any]:
    ...
```

挙動：

1. まず `base_texture_config` で現行方式のmaskを作る。
2. `result["valid"] == True` かつ `num_positive_points >= fallback_min_positive_points` なら採用。
3. 条件を満たさず、fallbackが無効ならそのまま返す。
4. fallbackが有効なら、`fallback_percentiles` を順に試す。
5. 各fallbackでは、`base_texture_config` をコピーし、以下のように変更する。

   * `texture_style="threshold_alpha"`
   * `threshold_mode="percentile"`
   * `percentile=<fallback percentile>`
   * denoise / open / close / morph / min_component_area はbaseを基本的に維持
6. validかつpositive点数が条件を満たしたものを採用。
7. 結果dictには以下を追加する。

   * `fallback_used`
   * `fallback_success`
   * `annotation_strategy`
   * `annotation_attempt_index`
   * `annotation_percentile`
   * `num_positive_points_on_cloud`

注意：

* dataclassではないので、dictに追加情報を入れて返せばよい。
* `AlphaTextureConfig` のcopyには `copy.deepcopy()` を使ってよい。

### 2-5. `build_frame_contour_mask_results()` を拡張

`build_frame_contour_mask_results()` に以下の引数を追加してください。

```python
point_cloud: dict[str, np.ndarray],
enable_contour_fallback: bool,
fallback_percentiles: list[float],
fallback_min_positive_points: int,
```

各BBoxについて、現在の `build_largest_contour_mask_from_binary()` 呼び出し部分を、fallback対応関数に置き換えてください。

ただし、fallbackを使うかどうかはmainで決めて渡してください。

main側の判定例：

```python
point_mode = str(point_cloud_attrs.get("point_mode", "foreground"))
enable_fallback = bool(args.enable_contour_fallback)

if args.fallback_only_non_foreground_point_cloud and point_mode == "foreground":
    enable_fallback = False
```

### 2-6. BBox内contour外ラベルの扱いを変更

`build_contour_point_annotations()` では、現在frame_labelsを全てbackgroundで初期化し、contour内だけpositiveにしています。

grid点群では、BBox内かつcontour外をbackgroundにすると危険なので、以下のオプションを反映してください。

* `bbox_inside_non_contour_label == "ignore"` の場合：

  * frame内の基本値はbackground
  * VOC BBox内の点をいったんignoreにする
  * その後、contour内の点をpositiveにする
* `bbox_inside_non_contour_label == "background"` の場合：

  * 現行に近く、contour外はbackground

実装イメージ：

```python
frame_labels = np.full(num_points, LABEL_BACKGROUND, dtype=np.int8)

if bbox_inside_non_contour_label == "ignore":
    for bbox in local_bboxes[order]:
        x1, y1, x2, y2 = bbox.local_xyxy
        xy = np.rint(pixel_xy[frame_mask]).astype(np.int64)
        inside_bbox = (
            (xy[:, 0] >= x1)
            & (xy[:, 0] <= x2)
            & (xy[:, 1] >= y1)
            & (xy[:, 1] <= y2)
        )
        frame_labels[inside_bbox] = LABEL_IGNORE

# contour_union内はpositiveで上書き
frame_labels[inside] = LABEL_FEMUR_CANDIDATE
labels[frame_mask] = frame_labels
```

複数BBoxがある場合はunionとして扱ってください。

### 2-7. fallback失敗時の扱い

fallbackしてもvalid contourが得られない場合は、以下の扱いにしてください。

* `bbox_inside_non_contour_label == "ignore"` の場合：

  * BBox内はignore
  * BBox外はbackground
* `bbox_inside_non_contour_label == "background"` の場合：

  * 現行互換に近く、frame全体background

ただし、XMLがないframeは既存の `no_bbox_label` に従ってください。

### 2-8. H5に監査情報を保存

`frame_annotation` に以下の項目を追加してください。

```python
"fallback_used"
"fallback_success"
"annotation_strategy"
"annotation_attempt_index"
"annotation_percentile"
"num_positive_points_on_cloud"
```

既存の `annotation_reason`, `valid_contour`, `contour_area`, `binary_area`, `foreground_ratio_in_bbox`, `num_labeled_points` などは維持してください。

dtypeは以下のようにしてください。

* `fallback_used`: bool
* `fallback_success`: bool
* `annotation_strategy`: object/string
* `annotation_attempt_index`: int32
* `annotation_percentile`: float32, 不明なら NaN
* `num_positive_points_on_cloud`: int32

### 2-9. root attrs / metaにも設定を保存

`meta` に以下を追加してください。

```python
"point_mode": point_mode,
"enable_contour_fallback": bool(enable_fallback),
"fallback_percentiles": args.fallback_percentiles,
"fallback_min_positive_points": int(args.fallback_min_positive_points),
"bbox_inside_non_contour_label": args.bbox_inside_non_contour_label,
```

## 3. 期待する動作

### 既存互換

既存のコマンド：

```bash
python export_pseudo3d_point_cloud.py \
  --input_h5 input.h5 \
  --output_h5 point_cloud.h5
```

は、`point_mode=foreground` として現行と同じ挙動になること。

既存のannotationコマンドも、fallbackを明示しなければ基本的に現行と同じ挙動になること。

### grid point cloud生成

例：

```bash
python export_pseudo3d_point_cloud.py \
  --input_h5 pseudo3d.h5 \
  --output_h5 point_cloud_grid_s4.h5 \
  --geometry_key corr \
  --point_mode grid \
  --sample_stride 4 \
  --preset percentile90_area100
```

この場合、alpha/intensity thresholdで点を削除せず、4 pixel間隔のgrid点を3D化する。`alpha` と `confidence` はforeground priorとして保存される。

### grid annotation with fallback

例：

```bash
python annotate_pseudo3d_point_cloud.py \
  --point_cloud_h5 point_cloud_grid_s4.h5 \
  --pseudo3d_h5 pseudo3d.h5 \
  --voc_xml_root annotations \
  --video_name example \
  --output_h5 annotated_grid_s4.h5 \
  --geometry_key corr \
  --enable_contour_fallback \
  --fallback_percentiles 85,80,75,70 \
  --fallback_min_positive_points 5 \
  --bbox_inside_non_contour_label ignore
```

期待される挙動：

* 初回は現行presetでlargest contourを取得する。
* invalidまたはpositive点数が少なすぎる場合、percentileを緩めて再試行する。
* contour内点は `femur_candidate=1`。
* BBox内かつcontour外は `ignore`。
* BBox外は `background=0`。
* XMLがないframeは既存の `--no_bbox_label` に従う。

## 4. 注意点

* Stage 5ではendpointやlengthは扱わない。`measurement` groupは現状通りempty / not computedでよい。
* `foreground` modeの既存挙動を壊さないことを最優先にする。
* `grid` modeでは点数が増えるため、`sample_stride` と `max_points_per_frame` が正しく機能することを確認する。
* H5出力の既存キー名は変更しない。
* 追加キー・追加attrsは後方互換的に追加する。
* 可能であれば、簡単なdry runとして小さなdummy h5または既存サンプルに対して、以下を確認する。

  * foreground modeが従来通り動く
  * grid modeで点数が増える
  * annotation H5に `point_mode`, `fallback_used`, `bbox_inside_non_contour_label` などが保存される
  * `annotation/point_label` に -1, 0, 1 が想定通り入る
