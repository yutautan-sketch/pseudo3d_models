# Stage 4 Phase 5 full-video CVAT review 実装手順

作成日: 2026-08-26  
最終更新日: 2026-09-07  
編集対象ディレクトリ: `/workspace/Stage2to4`  
CVAT実行環境: ローカルMac（`http://localhost:8080`）

## 0. 最新のlabel authority方針（2026-09-05）

teacher v4の目視検証により、full-video Taskのcontext frameで修正されたCVAT maskがH5へ
適用されず、旧automatic positiveが残る問題を確認した。今後はsnapshot対象動画の全Task frameで、
CVAT snapshot maskを最終positiveの唯一の根拠とする。

- target/contextを問わず全Task frameへ適用する
- CVAT mask外のautomatic positiveは維持しない
- empty maskでもautomatic positiveへfallbackしない
- BBoxはmask外pointのignore/background区分に使い、CVAT positiveをclipしない

本書のうち「129 targetだけを反映する」「contextをH5へ反映しない」「target外はv3を保持する」
という記述は、当時のv4実装履歴としてのみ残し、今後の実装仕様としては無効とする。新契約と
修正手順は次を正本とする。

- `docs/stage4/stage4_cvat_snapshot_authoritative_label_revision_plan.md`

CVAT snapshot作成後に誤BBoxのXMLを削除したframeを、保存済みH5/CVATより上位の明示的な
annotation tombstoneとして反映する後続方針は次を参照する。

- `docs/stage4/stage4_deleted_xml_annotation_invalidation_plan.md`

## 1. 目的

Phase 3で選ばれた129 BBoxだけを並べたCVAT Taskでは、動画内での大腿骨断面の移動を
確認できず、crop境界で見切れたBBoxを次のいずれかに判定しにくい。

- 現在のcrop内でmaskを修正すべき例
- 動画内に十分な良好frameがあり、crop端のframeを教師から除外してよい例
- crop位置そのものを再生成すべき例

この問題を解決するため、対象となった60動画を1動画1 CVAT Taskに分け、各動画の
全local-crop frameを時系列で確認できるreview packageを作る。

本書は実装前の固定仕様である。既存Task 5、編集前backup、selected-frame package、
teacher v3 H5は上書きしない。

## 2. 固定入力と既存artifact

教師入力:

```text
global_local_l75_w31_c12_area15_bboxrank_v3_refined_auto_v1/
├── annotated/       # 182 H5
├── collected/       # manual review前の中間出力
└── manual_review_cvat_phase5_all_phase3_v1/
```

Phase 3固定件数:

```text
selected BBoxes : 129
auto_accept     : 31
auto_refine     : 35
manual_review   : 63
selected videos : 60
```

既存CVAT Task 5は129候補だけを格納したselected-frame baselineとして保持する。

```text
task_id       : 5
frames        : 129
backup sha256 : cbce5290afa34f968553b85a78646aeb3c082e463199755bea978f82c08720ce
```

Task 5で既に修正を始めている場合は、作業を止める前に途中結果を次の名前でexportする。

```text
reviewed_segmentation_mask_1_1_all_phase3_master_partial_v1.zip
```

このpartial ZIPは新packageの初期maskへ任意にmergeできる入力として保存し、暗黙には
適用しない。

## 3. 新しい出力root

既存`manual_review_cvat_phase5_all_phase3_v1`を置換せず、次へ出力する。

```text
global_local_l75_w31_c12_area15_bboxrank_v3_refined_auto_v1/
└── manual_review_cvat_phase5_fullvideo_v2/
```

構成:

```text
manual_review_cvat_phase5_fullvideo_v2/
├── review_index.csv
├── video_progress.csv
├── crop_review_decisions.csv
├── export_summary.json
├── checksums.csv
└── videos/
    └── <video_name>/
        ├── images/                  # 全local-crop frame、BBox境界線だけを描画
        ├── masks/                   # target初期mask、contextはall-background
        ├── overlays/                # 目視補助
        ├── cases/                   # 129 targetだけの詳細
        ├── review_frames.csv
        ├── review_bboxes.csv
        ├── crop_review_decisions.csv
        ├── context_only_stems.txt
        ├── export_summary.json
        ├── checksums.csv
        ├── review_session_template.yaml
        └── cvat/
            ├── annotations_segmentation_mask_1_1.zip
            └── conversion_summary.json
```

全frame数は実データpreflightで集計し、summaryへ固定する。

## 4. frame選択と順序

対象video集合は既存`review_index.csv`に含まれる60動画と完全一致させる。182動画すべてを
CVATへ送らない。

各対象videoではpseudo3D H5の`local_encoder_images`と`frame_indices`を読み、全frameを
exportする。画像名は次に固定し、lexicographical sortと時間順を一致させる。

```text
<video_name>__fo<frame_order:05d>__fi<frame_index:08d>.png
```

各frameに次を記録する。

- `frame_order`
- `frame_index`
- `review_target`の有無
- strict XML BBox数
- Phase 3 selected BBox index
- source pseudo3D/annotated H5とSHA-256
- image、initial mask、overlayのSHA-256

CVATへimportする画像はlocal crop座標系だけとする。raw/full-width画像はmask座標が異なるため
同じSegmentation Taskへ混在させない。

## 5. BBoxの描画規則

CVATでmanual segmentationに用いる画像は、元のlocal cropと同じshape・座標を維持し、
BBoxの細い境界線だけを次の色で描く。

| 表示 | RGB | 意味 |
|---|---|---|
| yellow | `255,255,0` | fully drawableなreview target |
| magenta | `255,0,255` | crop境界でpartially clippedされたreview target |
| cyan | `0,255,255` | 同frameの非target XML BBox |
| red edge tick | `255,0,0` | target BBoxがcrop外で描画不能 |

### 5.1 ラベル文字重複の解消（描画契約実装済み）

full-video Taskの目視確認で、画像上端のframe情報とBBox別ラベルがBBoxや解剖構造へ重なり、
manual segmentationを妨げる例が確認された。このためCVAT入力画像から次の文字をすべて除去する。

```text
TARGET / CONTEXT
frame_order / frame_index
bbox_index
crop_status
visible_fraction
```

表示契約は次に固定する。

- CVAT入力画像のheight、width、frame stem、frame順を既存packageから変更しない
- 元画像をresize、crop、pad、平行移動しない
- 画像上にはBBox境界線とcrop外方向markerだけを描き、文字を一切描かない
- fully outside BBoxを示す全幅・全高red lineは画像端の短いtickまたはarrowへ変更し、
  解剖構造やmask境界と誤認しない表示にする
- frame/BBoxの詳細は`review_frames.csv`、`review_bboxes.csv`、
  `crop_review_decisions.csv`、`overlays/`、`cases/`で確認する
- initial maskのshape、pixel座標、内容を変更しない
- 元のH5、XML、point cloud座標を変更しない

この変更ではCVAT画像だけが変わり、Segmentation Mask 1.1のstem、shape、座標は変わらない。
したがって、既にCVATで修正済みのmask ZIPは座標変換やpaddingなしで新Taskへimportできる。

CVATへupload済みのTask mediaはローカルPNGの置換だけでは更新されないため、既存Taskは
編集済みZIPとbackupを保存してから、新しい画像を使うTaskへ置き換える。旧Task/packageを
上書きせず、例えば次へ版を分離する。

```text
manual_review_cvat_phase5_fullvideo_v3_textfree/
stage4_phase5_fullvideo_v3_textfree__<video_name>
```

実装時には画像上端・下端・左右端、小さいBBox、複数BBox、partially clipped、fully outsideを
synthetic checkへ追加する。CVAT画像とmaskのshape・stem・順序が旧packageと一致すること、
既存修正ZIPを無変換でimportできること、H5座標が不変であることを確認してから実データpackageを
再生成する。文字と重なっていたframeだけは新Task上で再確認する。

2026-09-03に、既存60 Taskの修正済みSegmentation Mask ZIPとpost-review Task backupの
一括exportを完了した。これをtext-free Taskへ移行する前の復元点とする。続いてfull-video用
描画関数から全テキストを除去し、fully-outside表示を全辺の赤線から短い内向きtickへ変更した。
frame番号・role・crop metricはCSVに残し、画像pixelへは書き込まない。次の確認対象は
`check_stage4_cvat_manual_roundtrip.py`でのtext-free描画contractと、旧/new package間の
stem・shape・mask不変性である。

同日、text-free描画contractのsynthetic checkがexit code 0で完了した。さらに、既存v2
packageを読み取り専用入力とし、保存済み`review_frames.csv`のsource pseudo3D H5とBBox
geometryから画像だけを再構築するmigration処理を追加した。この処理はXML・Phase 3判定・
initial maskを再計算せず、per-video mask PNGとSegmentation Mask ZIPのバイト一致、frame
stem・shape・順序、source H5 SHA-256を検証する。旧task state・修正済みsnapshotは新packageへ
混入させず、新しいMac転送manifestを生成する。

```text
pseudo3d/batch/export/rebuild_stage4_phase5_textfree_review_package.py
pseudo3d/pipelines/rebuild_stage4_phase5_fullvideo_textfree_review.sh
```

text-free packageの実データ生成は60動画・129 BBoxを保持してexit code 0で完了した。Mac側の
新Task作成では、旧60 Taskから一括exportした修正済みsnapshot manifestを必須入力にできる。
各snapshot ZIPを新しい同一stem/shapeの画像群に対して検証し、package初期ZIPの代わりにimport
する。また`stage4_video_exclusions_v1.csv`を適用し、`20250626_090758_8000`は新Taskを作らず
progress上で`excluded`とする。新Task名は次に分離する。

```text
stage4_phase5_fullvideo_v3_textfree__<video_name>
```

実行入口:

```text
pseudo3d/pipelines/create_stage4_phase5_fullvideo_textfree_cvat_tasks.sh
```

## 6. crop metric

XML BBoxをlocal crop座標へ変換した後、clip前とclip後を分けて保存する。

```text
projected_bbox_xyxy
clipped_bbox_xyxy
projected_area
visible_intersection_area
visible_fraction = visible_intersection_area / projected_area
```

追加flag:

```text
touches_left
touches_right
touches_top
touches_bottom
fully_outside_crop
partially_clipped
fully_visible
```

`x_max == image_width`など境界に一致するBBoxは自動エラーにしない。`touches_right`と
`visible_fraction`を記録し、人力判断へ渡す。非有限値、負の面積、座標変換不能だけを
validation failureとする。

## 7. frame roleと人力判定

全frameをCVATへ入れるが、全frameを教師修正対象にはしない。

### 7.1 frame role

```text
review_target : Phase 3の129 BBoxを含むframe
context_only  : 同じ60動画内のその他のframe
```

context-only frameにはCVAT import用のall-background maskを用意する。CVAT exportでそのmaskが
省略されても、Task image inventoryとの厳格な照合に成功したstemだけはempty maskとして扱う。
返却snapshotではcontext-onlyも最終label適用対象であり、非空maskはpositive、空maskは
zero-positiveの人力確定結果とする。

### 7.2 review disposition

129 BBoxごとに`crop_review_decisions.csv`の`review_disposition`を完成させる。

| 値 | 意味 | H5反映 |
|---|---|---|
| `pending` | 未確認 | 最終import不可 |
| `annotate` | 現crop内でmaskを修正 | BBox内にnonempty positiveを要求 |
| `exclude_crop_boundary` | 良好frameが別にあり、端の見切れ例は不要 | 対象BBox内をignore、positiveを残さない |
| `needs_recrop` | 動画内でも十分な表示がなくcrop再生成が必要 | 対象BBox内をignore、recrop queueへ追加 |

必須列:

```text
video_name
frame_order
frame_index
bbox_index
phase3_decision
crop_status
visible_fraction
review_disposition
reviewer_notes
```

empty maskだけから除外を推論しない。`exclude_crop_boundary`または`needs_recrop`の明示がない
actionable BBoxは、従来どおりnonempty maskを必須とする。

## 8. CVAT初期annotation

label contractは維持する。

```text
format     : Segmentation Mask 1.1
background : 0
femur      : 1
polygon化  : false
```

- `review_target`にはteacher v3の初期maskを入れる
- `context_only`にはall-background maskを入れる
- mask shapeはlocal crop imageと完全一致させる
- 返却snapshotでは、129 target以外を含む全Task frameのmaskをH5へ反映する
- 複数BBox frameではBBoxごとのintersectionと重なりをpreflightする

Task 5のpartial修正をseedに使う場合は、ZIP SHA-256を記録し、該当129 stemだけを明示的に
置換する。自動mergeと手動修正済みmaskの上書きは禁止する。

## 9. Mac上のCVAT Task

CVATはMac、teacher H5と実装repositoryは管理端末にあるため、工程を分離する。

1. 管理端末でfull-video packageを生成・検査する
2. `manual_review_cvat_phase5_fullvideo_v2`をMacへ転送する
3. Macの`localhost:8080`へ60 standalone Taskを作る
4. videoごとに編集前Task backupを保存する
5. 修正済みZIP、review session、decision CSVを管理端末へ戻す

Task名:

```text
stage4_phase5_fullvideo__<video_name>
```

各Taskは`femur` labelだけを持つstandalone Taskとする。Task作成処理は次を満たす。

- `localhost`/loopback以外を拒否する
- `video_progress.csv`へTask ID、URL、statusを保存する
- 同名Taskを暗黙に再利用しない
- Task作成、annotation import、backupの各段階で状態を保存する
- 中断後に再開できる
- 画像数とframe basename/orderをCVAT APIから再確認する
- backup SHA-256をvideo単位で保存する

## 10. CVATでの確認手順

video Task内を時間順に移動し、target前後の断面移動を確認する。

1. yellow targetは通常のmask修正候補とする
2. magenta targetは前後frameと比較して`annotate`またはcrop判定を選ぶ
3. red/outside targetは無理にmaskを描かず、動画内の良好frame有無で除外かrecropを選ぶ
4. cyan BBoxは参照用だが、CVAT semantic maskはBBox種別にかかわらず最終positiveの正本とする
5. context-only frameも最終mask確認対象とし、空のまま確定した場合はzero-positiveとして扱う
6. 全targetのdispositionを完成させてからTaskを完了する

修正export名:

```text
reviewed_segmentation_mask_1_1_<video_name>_fullvideo_v2.zip
```

## 11. manual import contract

importerはroot packageとvideo packageのSHA-256、Task backup、CVAT export、review session、
decision CSVを検証してからH5を作る。

### `annotate`

- corrected maskはBBox union外にpositiveを持たない
- 対象BBox内に少なくとも1 positive pixelを要求する
- point labelはmask内を`1`、BBox内mask外を`-1`とする

### `exclude_crop_boundary`

- 対象BBoxについてempty correctionを許可する
- 対象BBox内の既存positiveを除去し`-1`へ戻す
- `annotation_reason=manual_exclude_crop_boundary`を保存する

### `needs_recrop`

- 現cropの対象BBox内を`-1`にする
- `recrop_queue.csv`へsource H5、frame、XML BBox、crop metric、notesを保存する
- 自動的な別crop生成はこのPhaseでは行わない

この節の旧v4 importerはcontext-only frameおよび129 target外をteacher v3から保持していた。
この部分適用契約は廃止し、snapshot対象動画では全Task frameをCVAT maskから再構築する。
詳細は`stage4_cvat_snapshot_authoritative_label_revision_plan.md`を参照する。source H5は上書きせず、
修正版は新しいversioned rootへ生成する。

## 12. 実装段階

### Step 1: data contractとexporter

- full-video optionをreview exporterへ追加する
- 60動画と129 targetの固定集合を検証する
- crop metric、描画、frame role、decision templateを生成する
- 既存selected-frame modeを壊さない

### Step 2: synthetic check

- fully visible、partial clip、fully outside BBox
- `x_max == width`境界
- target/context mask分離
- frame順、複数BBox、非連続frame index
- 既存selected-frame modeの互換性

### Step 3: real-data export preflight

- 60動画の全frame数を集計する
- 129 targetの欠落・重複がないことを確認する
- crop status別件数とvideo別frame数を出す
- package生成前後でsource H5/XMLが不変であることを確認する

### Step 4: full package生成

- 新rootへvideo packageを生成する
- ZIP、CSV、checksum、summaryを検査する
- Mac転送前manifestを保存する

### Step 5: Mac Task作成

- Task creatorをmaster-reference前提からstandalone 60 Taskへ拡張する
- 最初の1動画でsmoke確認する
- 残りを一括作成し、全Task backupを取得する

### Step 6: CVAT review

- videoごとのmaskとcrop dispositionを完成させる
- 修正ZIP、review session、decision CSVを保存する
- pendingが0であることを確認する

### Step 7: final import/build

- 2026-09-03実装済み。
- Macから返却された59 Taskのannotation ZIP・post-review backup・task map・package checksumを
  H5作成前に一括検証する。Mac上の絶対パスは参照せず、Linux上のsnapshot rootから
  `video_name`と`task_id`で安全に再解決する。
- 全frame画像、初期mask、BBox manifestを含む元review packageはLinux上の
  `bboxrank_v3_refined_auto_v1/manual_review_cvat_phase5_fullvideo_v3_textfree`から読み、
  Mac返却先は`task_management/task_map.csv`とreview後snapshotだけを読む。返却snapshotへ
  `videos/`の複製を要求しない。
- `20250626_090758_8000`は明示的な除外manifestで外し、最終対象を181動画とする。
- CVAT maskは除外後に描画可能な112 target BBoxだけへ適用する。full-videoのcontext frame、
  target外BBox、crop外のtarget BBoxはv3を保持する。
- 空maskは意図的なpositive削除として許可し、そのBBox内をignoreへ戻す。単一target BBoxへ
  一意に帰属できる人力maskは元の弱いBBoxより優先し、BBox外positiveも保持する。初回返却
  snapshotでは7 frame・1,815 pxを確認したが、CVAT再修正後は新snapshotから再集計し、件数と
  画素数をprovenanceへ記録する。複数target BBoxでBBox外部分の帰属が曖昧な場合はpreflightで
  拒否する。
- reviewed 59動画と継承122動画を別runの`annotated`へ生成し、`collected`、3値label監査、
  `annotation_textures`まで一括生成する。

実装入口は次の2ファイルである。

```text
pseudo3d/batch/annotation/batch_import_stage4_phase5_fullvideo_cvat.py
pseudo3d/pipelines/build_stage4_bbox_ranked_v4_manual_fullvideo.sh
```

最終run名は
`global_local_l75_w31_c12_area15_bboxrank_v4_manual_fullvideo_v1`、H5 teacher tokenは
`bboxrank_v4_manual_fullvideo_v1`とする。合成契約検査は
`checks/stage4/check_stage4_phase5_fullvideo_final_import.py`に分離した。

## 13. 検証項目

- 60 Taskのframe総数がsource H5と一致する
- 129 targetが各1回だけ現れる
- context-onlyの非空・空CVAT maskがH5へ伝播する
- snapshot対象frameでCVAT mask外positiveが0である
- `annotate`だけnonempty maskを要求する
- crop除外とrecrop指定にpositiveが残らない
- no-BBox/background、BBox内非positive/ignoreを維持する
- point cloud、sampling、measurement配列を保持する
- source H5、XML、Task 5を変更しない
- 全Taskのbackup、export、session、decision SHAを記録する
- 同一入力からのpackageと最終H5が決定的である

## 14. 直近の実装対象

Step 1は2026-08-28に実装済み。

- exporterへ`--include_all_video_frames`と`--expected_selected_videos`を追加
- full-video modeを`all_phase3`かつvideo分割時だけ許可
- 60動画、129 target、Phase 3 decision件数を実行前に検証
- 全local-crop frame、crop metric、frame role、BBox状態描画を出力
- root/video単位の`crop_review_decisions.csv`を`pending`で生成
- target frameだけを`cases/`へ出し、context frameはall-background maskとする
- `manual_review_cvat_phase5_fullvideo_v2`専用pipelineを追加
- 従来のselected-frame modeは既定値として維持

Step 2は2026-08-28に実装済み。

- fully visible、partial clip、fully outsideを数値検証
- `x_max == width`をfully visibleかつ`touches_right`として検証
- 非有限座標と非正面積を拒否
- target/contextの初期mask分離を検証
- 複数BBox、全frame順、非連続frame indexを統合検証
- full-videoでは全frameを出し、`cases/`をtarget frameだけに限定
- root/video decision CSV、video progress、件数、source不変性を検証
- 従来selected-frame modeが既定値であり、decision CSVを追加しないことを検証

同日、ユーザー環境でsynthetic checkのexit code 0を確認済み。

Step 3は2026-08-28に実装済み。

- 60動画・129 target・decision件数をPhase 3 artifactと照合
- selected videoの全frame数とvideo別frame数を集計
- targetをstrict XMLへ一意に解決し、欠落・重複・frame不整合を拒否
- target単位のcrop metricとcrop status件数を出力
- v2/v3 annotation schemaとteacher metadataをread-only検査
- selected videoのpseudo3D H5、v2/v3 H5、strict XMLをSHA-256 manifest化
- 実行中のsource size/mtime不変性を検証
- `--verify_source_checksums`でStep 4後にpreflight manifestとの完全一致を再確認可能

同日、実データpreflightのexit code 0を確認済み。

```text
selected videos : 60
total frames    : 3048
target frames   : 129
target BBoxes   : 129
fully visible  : 102
partial clip   : 11
fully outside  : 16
source files   : 1350
```

実行入口:

```text
pseudo3d/pipelines/preflight_stage4_phase5_fullvideo_cvat_review.sh
```

出力先:

```text
manual_review_cvat_phase5_fullvideo_v2_preflight/
├── preflight_summary.json
├── video_summary.csv
├── target_bboxes.csv
├── source_checksums.csv
├── report_checksums.csv
└── run_config.yaml
```

Step 4は2026-08-28に実装済み。

- full-video exporterをStep 3 preflight必須のpipelineへ統合
- 60 video package、3,048 frame、129 target/caseを相互照合
- root/video CSV、summary、frame stem/order、crop metricをpreflightと照合
- context-only frameの初期maskがall-backgroundであることを検証
- 60個のCVAT ZIPを画像・reference maskとpixel単位で検証
- video packageの`checksums.csv`についてmember、size、SHA-256を検証
- Step 3の1,350 source H5/XMLを再ハッシュし、不変性を確認
- Mac転送対象を`mac_transfer_manifest.csv`とsummaryへ固定

追加validator:

```text
pseudo3d/analysis/validate_stage4_phase5_fullvideo_cvat_package.py
```

生成後の追加artifact:

```text
manual_review_cvat_phase5_fullvideo_v2/
├── package_validation_summary.json
├── mac_transfer_manifest.csv
└── mac_transfer_summary.json
```

実行コマンド:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

OVERWRITE=0 \
bash pseudo3d/pipelines/export_stage4_phase5_fullvideo_cvat_review.sh

status=$?
echo "phase5_fullvideo_step4_exit_code=${status}"
```

`mac_transfer_manifest.csv`は転送対象となるtop-level metadataと`videos/`以下を列挙する。
root直下の重複staging directory（`images/`、`masks/`、`overlays/`、`cvat/`）は除外する。
manifest自身、`mac_transfer_summary.json`、`package_validation_summary.json`は自己参照を避けるため
manifest行には含めないが、転送時にはreview rootのcontrol fileとして併せてコピーする。

同日、ユーザー環境でfull-video exportとpackage validatorが完了し、exit code 0を確認した。

```text
Stage 4 Phase 5 full-video package validation passed.
Stage 4 Phase 5 full-video export complete.
phase5_fullvideo_step4_exit_code=0
```

これによりStep 4を完了とする。次はMacへ検証済みpackageを転送し、standalone Taskを
最初の1動画でsmoke確認してから残り59動画へ展開するStep 5へ進む。Task 5と既存
selected-frame artifactは変更しない。

Step 5は2026-08-28に実装済み。

- Step 4の`mac_transfer_manifest.csv`をMac上で再検証する
- `video_progress.csv`と各video packageの`full_video`契約を検証する
- Task 5を参照・変更せず、`stage4_phase5_fullvideo__<video_name>`をstandaloneで作成する
- labelを`femur`だけに固定し、画像数・basename・時系列順をCVAT APIから再照合する
- `Segmentation Mask 1.1`をpolygon化せずimportする
- videoごとの編集前Task backupをZIP検査し、bytes・member数・SHA-256を保存する
- Task作成、annotation import、backup取得の各段階でstateをatomicに保存する
- `video_progress.csv`へTask ID、URL、status、backup path/SHAを追記する
- 同名の未追跡Taskを暗黙採用せず、中断後は追跡済みTaskだけを再開する
- `--only_video`で1動画smoke、解除後に残り59動画を作成できる
- `localhost`、`127.0.0.1`、`::1`以外への接続を拒否する
- macOS標準Bash 3.2と`set -u`でも空optional配列を展開しない引数構築にする

追加・更新ファイル:

```text
pseudo3d/batch/export/batch_create_stage4_phase5_cvat_tasks.py
pseudo3d/pipelines/create_stage4_phase5_fullvideo_cvat_tasks.sh
checks/stage4/check_stage4_phase5_fullvideo_cvat_tasks.py
```

Synthetic checkではfake CVAT clientだけを使用し、外部通信なしでdry-run、1動画smoke、
残りへのresume、再実行時のskip、backup ZIP/SHA検証、転送後改変の拒否を確認した。

```text
[OK] standalone dry-run, one-video smoke, resume-all, and idempotent backup verification
[OK] transferred package tampering rejection
Stage 4 Phase 5 full-video CVAT task synthetic checks passed.
```

Mac上でのdry-runと1動画smoke:

```bash
cd /path/to/Stage2to4

PHASE5_FULLVIDEO_REVIEW_ROOT=/Users/yutakodaira/Desktop/manual_review_cvat_phase5_fullvideo_v2 \
CVAT_USERNAME=yutau \
SMOKE_ONLY=1 \
SMOKE_VIDEO=1-3_14 \
APPLY=0 \
bash pseudo3d/pipelines/create_stage4_phase5_fullvideo_cvat_tasks.sh

PHASE5_FULLVIDEO_REVIEW_ROOT=/Users/yutakodaira/Desktop/manual_review_cvat_phase5_fullvideo_v2 \
CVAT_USERNAME=yutau \
SMOKE_ONLY=1 \
SMOKE_VIDEO=1-3_14 \
APPLY=1 \
bash pseudo3d/pipelines/create_stage4_phase5_fullvideo_cvat_tasks.sh
```

1動画の画像順、初期mask、Task backupを確認後、残りをdry-runしてから作成する。

```bash
PHASE5_FULLVIDEO_REVIEW_ROOT=/Users/yutakodaira/Desktop/manual_review_cvat_phase5_fullvideo_v2 \
CVAT_USERNAME=yutau \
SMOKE_ONLY=0 \
APPLY=0 \
bash pseudo3d/pipelines/create_stage4_phase5_fullvideo_cvat_tasks.sh

PHASE5_FULLVIDEO_REVIEW_ROOT=/Users/yutakodaira/Desktop/manual_review_cvat_phase5_fullvideo_v2 \
CVAT_USERNAME=yutau \
SMOKE_ONLY=0 \
APPLY=1 \
bash pseudo3d/pipelines/create_stage4_phase5_fullvideo_cvat_tasks.sh
```

Step 5の実データ完了条件は`task_creation_state.json`の`status=complete`、
`completed_tasks=60`、および全行の`cvat_task_creation_status=backup_verified`である。

同日、Mac上の`cvat-sdk 2.73.0`とlocal CVATを使用した1動画smoke dry-runが成功した。

```text
video_packages    : 60
target_bboxes     : 129
selected_now      : 1
only_video        : 1-3_14
transfer_files    : 16548
smoke frames      : 22
smoke targets     : 2
new_tasks_planned : 1
apply             : False
phase5_step5_smoke_preflight_exit_code=0
```

このdry-runではTask、state、backupを作成していない。次は同じscopeを`APPLY=1`で実行し、
1 Taskの画像順・初期mask・編集前backupを確認する。

初回smoke applyではTask作成後、CVAT APIのformat名を`Segmentation Mask 1.1`と渡したため、
serverから`Unknown input format`で拒否された。API識別子は大文字小文字を区別するため、
`Segmentation mask 1.1`へ修正した。既存stateの旧表記はcase-insensitiveに移行し、
`task_created`の既存Taskを削除・再作成せずannotation importからresumeする。

修正後のresumeは成功し、Task #6へ22 frame・2 targetの初期annotationをimportして、
編集前backupを検証・記録した。

```text
video             : 1-3_14
task_id           : 6
action            : resume from task_created
new_tasks_planned : 0
backup_verified   : 1
completed         : 1/60
```

次はTask #6で画像順、target mask、context frameを目視確認する。問題がなければ
`SMOKE_ONLY=0, APPLY=0`で残り59 Taskをdry-runし、その後`APPLY=1`で作成する。

Task #6の22 frame、2 target、初期mask、contextは目視確認で問題なしと判定した。
続く全件dry-runも成功した。

```text
video_packages    : 60
target_bboxes     : 129
selected_now      : 60
tracked task      : Task #6 / 1-3_14 / backup_verified
new_tasks_planned : 59
apply             : False
```

同名の未追跡Task衝突はなく、残り59 Taskを`SMOKE_ONLY=0, APPLY=1`で作成可能である。
途中失敗時は同じコマンドを再実行し、stateに保存済みの段階からresumeする。

## 15. Phase 5最終実行結果（2026-09-05）

text-free画像を用いた59 TaskのCVAT修正とsnapshot再export後、返却snapshot v2から
full-video manual correctionを反映した。`20250626_090758_8000`はcrop不良の除外manifestに
従って対象外とし、残り181動画をteacher v4へ構築した。

```text
output run : global_local_l75_w31_c12_area15_bboxrank_v4_manual_fullvideo_v1
annotated  : 181 files
collected  : 181 files
excluded   : 1 video (20250626_090758_8000)
exit code  : 0
```

出力先:

```text
/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/
└── global_local_l75_w31_c12_area15_bboxrank_v4_manual_fullvideo_v1/
    ├── annotated/
    ├── collected/
    ├── annotation_textures/
    └── logs/
```

### 15.1 現在の検証状態

- final import、181件のH5生成、collect、3値label監査は完了した
- source v3 H5、CVAT返却snapshot、除外manifestは上書きしていない
- `annotation_textures`の生成自体は完了した
- ただし現行textureのmask・輪郭はv4 point label/manual maskではなく、自動輪郭の再計算結果である
- よってv4ラベルの目視検証は別途、保存済み`point_label`を直接描画する方式で行う
- 目視中に見つかった相談事項は未修正であり、方針合意前にH5・XML・描画処理を変更しない

現時点ではStep 7のbuild処理は成功済みだが、Phase 5の受入完了条件であるv4ラベルの目視確認は
保留中である。次は相談事項を整理した後、最終H5の`annotation/point_label`と`valid_mask`を
直接可視化して確認する。

自動輪郭で確認されたBBox境界接触globalの過採用は、full-video CVAT運用とは分離して次へ記録する。
既存CVAT修正済みmaskを再編集せず再利用することも同文書の必須契約とする。

- `docs/stage4/stage4_bbox_ranked_border_contact_revision_plan.md`

teacher v4の最終受入に使うpoint-label直接可視化は、automatic contour表示と分離して次へ記録する。

- `docs/stage4/stage4_v4_point_label_visualization_plan.md`

### 15.2 CVAT適用範囲の不整合と受入保留

保存済みv4 labelと返却CVAT dense maskの比較により、`1-3_14/frame_order=10`でCVAT mask外に
185 positive pointが残ることを確認した。frame 10はfull-video Taskのcontext frameで、旧v4
importerがframe 12/15だけをmanual適用対象としていたことが原因である。過去のmask描画残りではない。

```text
CVAT mask pixels       : 198
v4 positive points     : 343
positive inside CVAT   : 158
positive outside CVAT  : 185
saved source           : global
manual applied         : false
```

この結果により、v4 build成功はartifact生成記録として保持するが、最終教師としての受入は保留する。
修正版ではsnapshot対象動画の全Task frameでCVAT maskを唯一のpositive authorityとし、既存automatic
positiveを利用しない。CVAT Taskと返却snapshotは再編集せず、新しいversioned runへ再適用する。

- `docs/stage4/stage4_cvat_snapshot_authoritative_label_revision_plan.md`

## 16. 後続補正と最終学習データ（2026-09-07）

Phase 5後に確認された2つのラベル不整合は、既存CVAT Taskを再編集せず、versioned成果物として
順に解消した。

1. teacher v5で、返却full-video CVAT snapshotを対象Task全frameのpositive authorityとして適用した
2. teacher v6で、CVAT snapshot作成後に削除された誤BBox XML 7件を明示的に無効化した

v6はcrop不良として除外した`20250626_090758_8000`を含まず、annotated/collected各181 H5で構成する。
保存point labelの一括可視化と目視確認も完了し、CVAT mask外positive、削除済みBBox、旧maskの描画残りが
ないことを確認した。したがって、今回の学習入力の正本は次である。

```text
/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/
  global_local_l75_w31_c12_area15_bboxrank_v6_cvat_authoritative_xml_invalidation_v1/
    collected/
```

詳細な無効化契約、構築件数、可視化および最終受入結果は次を参照する。

- `docs/stage4/stage4_deleted_xml_annotation_invalidation_plan.md`
