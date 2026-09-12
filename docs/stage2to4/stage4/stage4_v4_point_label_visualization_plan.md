# Stage 4 teacher v4 point-label可視化計画

作成日: 2026-09-05  
最終更新日: 2026-09-05  
編集対象ディレクトリ: `/workspace/Stage2to4`  
状態: v4適用範囲不整合を確認・authoritative snapshot対応計画へ移行

## 1. 目的

teacher v4 H5に保存された最終学習ラベルを直接可視化し、CVAT manual correctionと
automatic contourのどちらが最終positiveを生成したかを判別できるようにする。

既存`annotation_textures`の赤mask・輪郭線は、保存済みv4 labelではなく画像・BBox・configから
再計算したautomatic contourである。このため、既存表示でglobal輪郭が不適切に見えても、
v4 H5ではCVAT maskにより修正済みの可能性がある。本可視化をBBox境界接触ロジックの変更前に
実施し、H5再作成の要否を判断する。

## 2. 安全方針

- 既存v4 H5、CVAT ZIP、Task backup、snapshot、XML、point cloudを読み取り専用とする
- 既存`annotation_textures`を上書きしない
- XMLからBBoxを再読込しない
- global/local二値maskや輪郭を再計算しない
- 表示用の点半径・alphaは変えてよいが、ラベル値と集計は変更しない
- 出力は新しい専用rootへ生成する

## 3. 固定入力

対象run:

```text
global_local_l75_w31_c12_area15_bboxrank_v4_manual_fullvideo_v1
```

対象ファイル:

```text
annotated/<video_name>/*bboxrank_v4_manual_fullvideo_v1.h5
```

H5から直接使用する値:

| H5 field | 用途 |
|---|---|
| `point_cloud/pixel_xy` | 点のlocal画像座標 |
| `point_cloud/frame_order` | 点とフレームの対応 |
| `annotation/point_label` | 最終3値学習ラベル |
| `annotation/valid_mask` | ignoreを除く有効点 |
| `frame_annotation/bbox_local_xyxy` | 最終H5が参照するBBox |
| `frame_annotation/frame_order` | BBoxとフレームの対応 |
| `frame_annotation/bbox_index` | 複数BBoxの識別 |
| `frame_annotation/selected_contour_source` | automatic/manual provenance |
| `frame_annotation/annotation_reason` | 採用・削除理由 |

背景画像はH5 metadataが指すsource pseudo3D H5の`local_encoder_images`を使用する。frame indexも
source pseudo3D H5から取得するが、BBox geometryは最終annotated H5の保存値を使用する。

CVATのdense maskはteacher v4 H5自体には保存されていないため、final buildに使用した
`cvat_exports_after_textfree_review_v2`の修正済みSegmentation Mask 1.1 ZIPから読み込む。
snapshot manifest SHA、Task ID、annotation ZIP SHAをv4 H5の`manual_review_fullvideo` provenanceと
照合する。現行実装は同groupに記録された実適用frameだけを描画するため、context frameの
CVAT maskを表示・適用できないことが判明した。修正版ではsnapshot対象動画の全Task frameを
frame単位provenanceで照合し、target/contextを問わずCVAT maskを描画する。

最終positiveも同じCVAT mask内だけから生成する。詳細は次を正本とする。

- `docs/stage4/stage4_cvat_snapshot_authoritative_label_revision_plan.md`

## 4. 描画契約

主要なframe画像ではsource別の色を使わず、最終classだけを描画する。

| 表示 | RGB | 意味 |
|---|---:|---|
| background point | `80,140,200` | `point_label=0` |
| ignore point | `255,210,0` | `point_label=-1` |
| corrected CVAT mask | `0,255,255` | manual適用frameのdense segmentation mask |
| positive point | `255,32,32` | `point_label=1` |
| saved BBox | `160,255,80` | `frame_annotation/bbox_local_xyxy` |

点は視認性のため半径1程度で描画する。描画円のpixel数ではなく、入力point数をCSVの件数とする。
画像上には文字を重ねず、frame/BBox/provenanceの詳細はsidecar CSVへ保存する。

同一frameに複数BBoxがある場合、point labelは保存済みframe unionとして一度だけ描く。
BBoxは保存行ごとに描き、source/reasonはBBox単位でCSVへ記録する。

### 4.1 各表示色の読み方

このPNGはpixel単位の塗りつぶしsegmentation maskではなく、point cloudに存在する点へ
`annotation/point_label`を重ねた確認用overlayである。

- 赤は最終`positive`（`point_label=1`）であり、Stage 5で大腿骨候補として学習対象になる。
- 黄は`ignore`（`point_label=-1`）であり、`valid_mask=False`としてloss計算から除外される。
  backgroundとして学習される点ではない。
- 青は`background`（`point_label=0`）であり、`valid_mask=True`の負例として学習対象になる。
  BBoxがないframeでは全点がこのclassでなければならない。
- 水色はCVATから返却されたdense segmentation maskである。修正版ではsnapshot対象動画の
  全Task frameでSHA/Task/frame対応を照合して表示する。これはpoint markerではなくpixel領域で、
  empty maskでは表示領域がない。
- 緑線は最終H5に保存されたBBoxであり、segmentation境界やpositive領域そのものではない。
  manual correctionでは人が確定した赤positiveが元のweak BBox外へ延びることがあり、それだけで
  異常とは判定しない。`selected_contour_source`と併せて判断する。
- 色の付いていないgray部分は背景画像であり、そのpixel位置に保存点がないことを示す。
  background labelを意味するとは限らない。

点は標準設定で半径1の表示markerへ拡張し、alpha 0.85でgray画像へ合成する。この拡張範囲は
見やすさのためだけのもので、周囲のpixelを追加pointや追加labelとして数えない。markerが重なる
場合は青point、黄point、水色CVAT mask、赤pointの順に描く。したがってCVAT maskは黄の上、
赤の下となり、最終positive pointは常に確認できる。緑BBoxは最後に描くためBBox線上では緑が
点色またはmask色を覆う場合がある。

global、local、Phase 3、CVATの由来はPNGの色では区別しない。各BBoxの由来は同じvideo配下の
`frame_labels.csv`にある`selected_contour_source`と`annotation_reason`で確認する。複数BBox frameの
point countはframe全体のunion値であり、BBox行ごとに同じ件数が記録されるため、CSV行を単純合計
しない。video全体の重複なし集計には`summary.json`またはbatch `summary.csv`を使用する。

## 5. 出力構成

```text
global_local_l75_w31_c12_area15_bboxrank_v4_manual_fullvideo_v1/
└── annotation_textures_v4_labels/
    ├── summary.csv
    └── <video_name>/
        ├── frames/
        │   └── annotation_frame_<frame_order:05d>.png
        └── frame_labels.csv
```

`frame_labels.csv`は少なくとも次を持つ。

```text
video_name
frame_order
frame_index
bbox_index
bbox_local_xyxy
selected_contour_source
annotation_reason
valid_contour
positive_points
ignore_points
background_points
cvat_mask_applied
cvat_mask_pixels
```

複数BBoxのないframeもframe単位のlabel countを記録する。BBoxなしframeではBBox rowとは別に
frame summaryを残し、全点backgroundであることを確認できるようにする。

## 6. provenanceの解釈

`selected_contour_source`は少なくとも次を区別する。

| source | 判断 |
|---|---|
| `manual_cvat_fullvideo_v1` | CVAT maskを最終positiveへ適用済み |
| `manual_cvat_fullvideo_empty_v1` | CVATで空mask・削除を確定 |
| `phase3_*` | Phase 3 proposalを使用 |
| `global` | v2 global automatic contourを維持 |
| `local_percentile` | v2 local automatic contourを維持 |
| `shared` | global/localが同一mask |
| `none` | 有効輪郭なし |

既存automatic visualizationで緑globalが表示されても、本可視化の赤positiveが正しく、sourceが
`manual_cvat_fullvideo_v1`ならH5再作成は不要である。赤positiveもBBox辺まで膨張し、sourceが
`global`なら境界接触ロジックの監査対象とする。

## 7. 実装内容

既存automatic contour可視化を変更せず、専用処理を追加する。

```text
pseudo3d/export/export_stage4_point_label_visualization.py
pseudo3d/batch/export/batch_export_stage4_point_label_visualization.py
pseudo3d/pipelines/export_stage4_v4_point_label_visualizations.sh
checks/stage4/check_stage4_point_label_visualization.py
```

single-frame描画、single-H5 export、batch/pipelineを追加し、全入口で同じpure rendering
functionを共有した。入力H5とsource pseudo3D H5のSHA-256を実行前後で照合し、video単位の
一時directoryからatomicに配置する。再開時は保存済みsummary、両入力checksum、PNG数を
照合してから`skipped_verified`とする。

`point_label`をframeごとに再探索せず、`frame_order`で一度だけ安定sortして索引化するため、
全181動画でもpoint数×frame数の反復走査を避ける。BBoxなしframeにignore/positiveが存在する
入力は、v4の3値label policy違反として描画前に拒否する。

## 8. 検証手順

### Step 1: 入力契約

- `point_label`が`-1/0/1`だけである
- `valid_mask == (point_label != -1)`である
- point-level配列長が一致する
- `pixel_xy`が画像範囲内である
- frame/BBox indexとshapeが有効である
- provenance sourceが既知値である

### Step 2: Synthetic check

- 3値labelが指定色へ対応する
- 黄point、CVAT水色mask、赤pointの重なり順が固定される
- 修正済みCVAT ZIPのSHA、Task、frame stemをv4 H5 provenanceと照合する
- CVAT dense maskをsnapshot対象の全Task frameへ表示し、target/contextの適用漏れを許さない
- 保存済みBBoxだけを描画する
- automatic contour再計算関数を呼ばない
- manual、manual-empty、automatic provenanceをCSVへ保持する
- 複数BBox、BBoxなし、空point frameを処理する
- H5 label countとCSV countが一致する
- 同一入力からPNG/CSVが決定的に生成される
- 入力H5のhashが変化しない

2026-09-05、`dualtrack311`環境で全項目が成功した。

```text
[OK] direct ignore/background/positive color mapping
[OK] saved labels, saved BBoxes, provenance CSV, and deterministic output
[OK] label/provenance/no-BBox validation and no contour/XML recomputation
[OK] batch CLI, summary, and verified resume
v4_point_label_visualization_synthetic_exit_code=0
```

上記は3値point overlay実装時の結果である。その後、水色CVAT dense maskを黄pointと赤pointの
間へ追加したため、同じSynthetic checkを再実行してCVAT ZIP checksum、manual frame alignment、
overlay順を含む更新後contractを確認する。

### Step 3: 181動画の一括出力

先に全件を同一条件で出力し、`summary.csv`と各`frame_labels.csv`を代表例選定の母集団とする。

```text
videos                 : 181
failed                 : 0
input_files_modified   : 0
label_count_mismatches : 0
invalid_valid_masks    : 0
unknown_sources        : 0
```

を完了条件とする。

### Step 4: 代表データの目視確認

一括出力後、CSVのprovenanceと既知の目視事例から次の少数例を選ぶ。

- global過採用に見えた例
- CVATで明確に修正した例
- CVATでmaskを削除した例
- localまたはsharedの良好例
- BBoxなしframeを含む例

色、点半径、BBox、CSV対応だけを確認する。描画の視認性調整によってH5やlabelを変更しない。

## 9. 目視結果の分類

目視結果は次の3分類で記録する。

1. `visualization_only`: 旧automatic表示だけが不適切でv4 labelは正しい
2. `automatic_global_overfill`: v4 sourceがglobalでpositiveもBBox辺まで過抽出
3. `manual_label_issue`: CVAT由来の最終label自体に問題がある

`automatic_global_overfill`が実際に残ることを確認してから、
`stage4_bbox_ranked_border_contact_revision_plan.md`のborder guardへ進む。現在は方針記録までとし、
自動輪郭選択コード、H5、CVAT artifactは変更しない。

## 10. 実行順

実装の合成検査:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_point_label_visualization.py

echo "v4_point_label_visualization_synthetic_exit_code=$?"
```

Step 2通過後、全181動画を専用rootへ生成する。

```bash
VIDEO_NAMES="" \
EXPECTED_FILES=181 \
SKIP_EXISTING=0 \
bash pseudo3d/pipelines/export_stage4_v4_point_label_visualizations.sh
```

CVAT mask追加前の出力が存在する場合は一度だけ`SKIP_EXISTING=0`で専用可視化rootを更新する。
更新後に中断・再開する場合は`SKIP_EXISTING=1`へ戻すと、入力checksum、CVAT ZIP checksum、PNG数を
照合した正常videoが`skipped_verified`となる。全件出力後、`summary.csv`と`frame_labels.csv`から
Step 4の代表例を選ぶ。
