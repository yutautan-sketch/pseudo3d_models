# Stage 4 BBox-aware輪郭教師・positive判定 改修計画

作成日: 2026-08-17  
最終更新日: 2026-08-17  
編集対象ディレクトリ: `/workspace/Stage2to4`（主要）、`/workspace/Stage5`（入力契約・事前検査のみ）

## 1. 目的

Stage 4のforeground点取得後に行う、アノテーションBBox内の輪郭抽出と
positive点判定を改善する。

foreground抽出は推論時にも利用し得る汎用処理である一方、本改修対象は教師データ
作成専用である。そのため、VOC XMLのBBox、対象が大腿骨断面であるという知識、
既存候補間の比較を利用してよい。ただし、自動処理で信頼できない例を無理にpositive化
せず、人力修正へ送れる構成にする。

### 1.1 人力修正環境の固定方針

人力修正には、ローカルホスト上で運用するCVAT Community Editionを使用する。
画像、mask、manifestを外部SaaS、クラウドストレージ、外部AI/ML backendへ送信しない。
Stage 4パイプラインもCVAT APIやネットワークへ接続せず、ローカルimport用artifactの
生成までを担当する。CVATへのuploadと修正済みannotationのexportは、利用者が
ローカル環境で明示的に実行する。

CVATとのannotation交換形式は、次の仕様に固定する。

- CVAT format: `Segmentation Mask 1.1`
- class: `background=0`、`femur=1`
- CVAT import時の`Convert masks to polygons`: OFF
- 詳細仕様: `docs/stage4/cvat_segmentation_mask_1_1_import_spec.md`

## 2. 現在の固定条件

### 2.1 foreground点

当面は、調査済みの次のforeground点群を固定入力として再利用する。

```text
sampling run: global_local_l75_w31_c12_area15
global       : enabled
local        : window=31, percentile=75, min_contrast=12
top-hat      : disabled
context grid : disabled
cleanup area : 15
```

本改修ではforeground H5を再生成せず、輪郭教師、point label、annotation metadataのみを
再生成する。これにより、foreground抽出の効果と教師改善の効果を分離する。

### 2.2 point label

次の意味を維持する。

| 条件 | label | 意味 |
|---|---:|---|
| BBoxがないフレーム | `0` | background |
| BBoxありフレームのBBox外 | `0` | background |
| BBox内かつ採用輪郭内 | `1` | positive |
| BBox内かつ採用輪郭外 | `-1` | ignore |

輪郭が得られないBBoxを、そのBBox内すべてbackgroundとして扱ってはならない。
有効な修正maskがない場合はBBox内をignoreのまま保持する。

## 3. 現在の課題

現在のBBox-ranked teacher v2はglobal/local候補をBBox中心距離と面積で比較できるが、
次の過抽出・不確実性を十分に扱っていない。

1. foregroundまたは選択輪郭がBBoxの大部分を覆い、点が残され過ぎる例
2. 輪郭がBBox境界へ広く接触し、対象外組織まで結合している例
3. BBox内に複数成分が存在し、単一閾値では分離できない例
4. global/local候補が拮抗し、自動選択の信頼性が低い例
5. どの閾値でも適切な輪郭が得られず、人力修正が必要な例

教師側でpositiveを過大にすると、Stage 5は誤った領域を強い正例として学習する。
一方、厳しすぎる閾値は大腿骨断面を欠落させるため、単純に閾値を一律で高くする
方針も採用しない。

## 4. 改修後の判定構造

BBoxごとに、最終判定を次の3状態へ分類する。

| 状態 | 処理 |
|---|---|
| `auto_accept` | 現在のglobal/local候補をそのまま採用 |
| `auto_refine` | BBox内で追加閾値候補を生成し、再rankingして採用 |
| `manual_review` | 自動候補を採用せず、人力修正対象として出力 |

`manual_review`でも、レビュー前のH5ではBBox内をignoreにして誤教師化を防ぐ。

## 5. BBox単位で計測する指標

### 5.1 面積と被覆

- foreground mask面積 / BBox面積
- 選択filled contour面積 / BBox面積
- BBox内foreground点数 / BBox内候補点数
- positive点数 / BBox内point数
- absolute contour area

点群密度はsampling条件に依存するため、判定の主軸はpixel mask面積とし、point数は
補助指標として保存する。

### 5.2 位置と境界

- 輪郭重心とBBox中心の距離をBBox対角長で正規化した値
- 輪郭のうちBBox境界へ接触するpixelの割合
- 上下左右それぞれのBBox境界への接触有無
- 輪郭BBoxとannotation BBoxのextent比

### 5.3 形状

- 接続成分数
- solidity
- extent
- perimeter、compactness
- 主軸方向とaspect ratio

これらは単独で大腿骨らしさを決定するhard ruleにはせず、過抽出検出、候補ranking、
manual review理由の説明に利用する。

### 5.4 不確実性

- global/local候補のscore差
- 1位候補と2位候補のscore差
- 閾値を変更したときのmask IoUと重心移動量
- 採用候補を支持する閾値設定数
- 候補sourceとthreshold method

## 6. BBox内の自動再抽出

### 6.1 発動条件

以下のいずれかを満たすBBoxを`auto_refine`候補とする。数値閾値は初回監査結果を
見て決定し、実装前に固定値を推測しない。

- foreground/contour area ratioが上位tailにある
- BBox境界接触率が高い
- BBox全体に近い単一成分が得られている
- global/localの候補差が小さい
- 現候補の中心距離または形状指標が悪い

### 6.2 追加候補

BBox crop内で、少なくとも次を候補として比較する。

1. percentile thresholdの段階的な厳格化
2. Otsu threshold
3. adaptive/local threshold
4. morphologyと小成分除去を適用した各候補
5. 必要に応じたdistance transformまたはwatershedによる結合成分の分離

探索範囲は初回監査で絞り、候補数を無制限に増やさない。

### 6.3 ranking

候補は次を組み合わせてrankingする。

- BBox中心との近さ
- 小さすぎず大きすぎないarea ratio
- BBox境界接触へのpenalty
- solidity/compactnessなどの形状安定性
- 近隣閾値間でのmask IoU
- absolute contour areaの下限

中心に近いだけの微小成分や、面積が大きいだけのBBox全体maskを選ばないよう、
eligibility判定とranking scoreを分離する。global/local/追加閾値間に固定のsource優先順位は
設けない。

## 7. manual reviewへの振り分け

次の例は`manual_review`とする。

- 有効候補がない
- すべての候補がBBoxを過度に覆う
- 1位と2位が拮抗し、位置または形状が大きく異なる
- 閾値変更による重心移動またはmask変化が大きい
- BBox中心付近にeligible候補がない
- 面積下限を満たす成分がない
- 複数BBoxまたは複数成分の対応が曖昧
- 自動refine後も品質条件を満たさない

判定理由は単一文字列ではなく、複数の`reason_code`として保存する。

## 8. CVAT人力修正データ契約

### 8.1 セキュリティと運用境界

- CVAT Community Editionをlocalhostまたは外部通信を遮断した閉域LANで運用する
- `cvat.ai`などのhosted serviceは使用しない
- CVATの外部cloud storage、webhook、外部AI/ML連携は使用しない
- CVAT data volume、export ZIP、元画像、backupはローカルの保護対象領域に置く
- annotation uploadはTask内の既存annotationを置換し得るため、upload前に必ずexportしてbackupする
- Stage 4はCVATへ自動送信せず、import用ZIPと補助資料だけを生成する

### 8.2 review単位とframe mask

teacher decisionはBBox単位だが、`Segmentation Mask 1.1`の`SegmentationClass`は
画像単位のsemantic maskである。この差を次のように扱う。

1. 1個以上のBBoxが`manual_review`になったframeをCVAT Taskの対象画像とする
2. CVATへ渡す初期maskは、そのframeにある全BBoxの採用候補をunionしたframe maskとする
3. `review_manifest.csv`にはframe内の全BBoxを記録し、`manual_review`対象を明示する
4. 同一frameの`auto_accept`/`auto_refine`領域も初期maskへ含め、annotation uploadによる消失を防ぐ
5. 修正結果はframe maskとして受け取り、元のBBox群と照合してpoint labelへ反映する

複数instanceの区別が必要なframeでは`SegmentationObject`を使用し、object indexと
`bbox_index`の対応をmanifestへ保存する。Stage 5がsemantic positiveを利用する限り、
`SegmentationClass`のfemur領域を最終的な正解maskとする。

### 8.3 画像名と座標系

CVAT Task作成後の画像名とannotation ZIP内のstemは完全一致させる。衝突を避けるため、
画像stemは次のようなvideoを含む一意な形式に固定する。

```text
{video_name}__fo{frame_order:05d}__fi{frame_index:08d}
```

- `images/`、`masks/`、`SegmentationClass/`、`SegmentationObject/`で同じstemを使う
- Task作成後に画像をrenameしない
- 元画像とmaskをリサイズしない
- local frame pixel座標をCVAT編集座標とし、raw/local変換情報はmanifestへ保存する
- source H5、XML、video、frame order/index、BBox indexへ逆引きできるようにする

### 8.4 Stage 4側のreview artifact

Stage 4パイプラインは、runごとに独立した次のartifactを生成する。

```text
manual_review_cvat/
├── images/                         # CVAT Taskへ登録する無描画PNG
├── overlays/                       # BBox、候補、score、reasonを描いた参照画像
├── masks/                          # teacher実行時の初期binary mask（0/255）
├── cvat/
│   ├── annotations_segmentation_mask_1_1.zip
│   └── unpacked_reference/
│       ├── labelmap.txt
│       ├── ImageSets/Segmentation/default.txt
│       ├── SegmentationClass/*.png
│       └── SegmentationObject/*.png
└── review_manifest.csv
```

annotation ZIPには画像本体や外側の余分なdirectoryを含めず、ZIP root直下を
`labelmap.txt`、`ImageSets`、`SegmentationClass`、`SegmentationObject`とする。
`images/`は既存CVAT Taskの作成用、`overlays/`は判断支援用であり、annotation import
ZIPには含めない。

### 8.5 labelmapとmask値

初期versionのlabelmapを次に固定する。

```text
background:0,0,0::
femur:255,0,0::
```

- staging用`masks/`は可読性と既存処理との互換性のためbinary `0/255`を許可する
- CVAT ZIP内maskはgrayscale `uint8`のindexed mask `0/1`へ変換する
- `0=background`、`1=femur`以外のclass indexを含めない
- `SegmentationClass`と`SegmentationObject`を同じ画像shapeで生成する
- 単一instanceの初期形式ではobject maskも`0/1`とする
- 複数instanceを扱う場合はobject indexを決定的に割り当て、manifestへ記録する

### 8.6 proposal maskの保存方針

CVAT用の初期maskを、後からsparseなpoint labelから再構成してはならない。
teacher実行時にrankingへ使用したfull-resolution pixel maskを、その場でreview staging
maskとして保存する。これにより、後日のcode/config変更でCVAT初期annotationが変化する
ことを防ぐ。

各maskについて少なくとも次をmanifestへ保存する。

- mask relative path、image shape、mask SHA-256
- teacher policy/config fingerprint
- selected source、threshold method/value、candidate score
- decision、reason codes、対象BBox一覧
- source H5/XML pathと可能ならchecksum

`manual_review`対象は、レビュー前の学習用H5ではBBox内をignoreにする。一方、CVATの
初期annotationには人が修正しやすいbest-effort proposalを入れてよい。このproposalは
確定positiveではなく、修正開始点としてのみ扱う。

### 8.7 ZIP生成時のstrict validation

ZIP作成前に次を検査し、1件でも不整合があればそのrunを失敗させる。

- image/mask stemの一対一一致
- duplicate、missing mask、extra mask、未対応拡張子がないこと
- imageとmaskのwidth/heightが完全一致すること
- maskが2-D grayscale `uint8`で、CVAT側のunique indexが`{0, 1}`の部分集合であること
- `default.txt`が拡張子なしstemを重複なく決定的順序で列挙していること
- `labelmap.txt`のindexとmask値が一致すること
- ZIP memberが定義済みroot以外に出ないこと
- 0件のmanual reviewに対して、不正な空annotation ZIPを生成しないこと

再現性のため、member順、PNG生成、ZIP timestamp/permissionを固定し、同一入力から
同一内容とchecksumを生成できる設計にする。

### 8.8 CVATでの最小導入確認

全件運用の前に2〜3画像のgolden sampleで次を確認する。

1. ローカルCVAT Taskを同じ画像名と`femur` labelで作成する
2. 既存annotationをbackupする
3. `Segmentation Mask 1.1`、`Convert masks to polygons=OFF`でZIPをimportする
4. femur mask、画像対応、向き、座標、instance表示を目視確認する
5. CVATから同形式でexportし、mask値、shape、stem、pixel領域を比較する

### 8.9 修正結果のH5 import

CVATからexportした修正済み`Segmentation Mask 1.1` ZIPをH5へ戻す処理は、review
exportとは別の明示的pipelineとして設計する。元H5を直接上書きせず、新しいrunへ出力する。

- ZIPを安全に展開し、path traversal、未知member、重複memberを拒否する
- `review_manifest.csv`を基準にvideo/frame/BBoxをstrict matchingする
- image stem、shape、labelmap、class/object index、checksumを検証する
- 非zero pixelが元annotation BBoxのunion外にある場合は、初期versionでは自動clipせず拒否する
- 修正mask内をpositive、BBox内mask外をignore、BBox外をbackgroundに設定する
- 空mask、missing frame、extra frame、重複mask、shape不一致、未知indexを拒否する
- point配列、frame alignment、source flag、confidenceの順序と値を変更しない
- 修正者、更新日時、CVAT export ZIP SHA-256、Task識別情報、import versionを保存する

manual maskのprovenanceは`manual_cvat_segmentation_mask_1_1_v1`とし、自動teacherと
明確に区別する。補正済みZIPの適用はidempotentにし、同一入力を二重適用してもpoint
labelが変化しないことを検証する。

## 9. H5へ追加するmetadata案

既存schemaを壊さず、`frame_annotation`へ次を追加する。

```text
teacher_policy_version
teacher_decision                 # auto_accept / auto_refine / manual_review
teacher_reason_codes
initial_contour_source
selected_contour_source
selected_threshold_method
selected_threshold_value
foreground_area_ratio
selected_area_ratio
bbox_border_contact_ratio
center_distance_norm
solidity
extent
compactness
candidate_score_margin
threshold_stability_iou
manual_review_required
manual_annotation_provenance
manual_review_mask_sha256
cvat_export_sha256
cvat_task_identifier
```

可変長の候補詳細は、必要なら`contour_candidates` groupまたは監査CSVへ保存し、
point-level datasetを候補数だけ複製しない。

## 10. 実装段階

### Phase 1: read-only全件監査

現在の182件を変更せず、BBox単位の指標をCSVへ出力する。

- 過被覆上位
- BBox境界接触上位
- global/local拮抗例
- invalid contour例
- 小面積・低point例

カテゴリごとに上位20件程度のoverlayを生成する。

### Phase 2: CVAT最小converterとgolden round trip

2〜3画像のbinary maskを`Segmentation Mask 1.1`へ変換し、ZIP構造、labelmap、stem、
shape、indexを検証する。ローカルCVATへのimport/export round tripを行い、仕様を固定する。

### Phase 3: auto-refine prototype

監査対象の代表例でBBox内追加閾値を実装し、候補metricsとrankingを可視化する。
既知の良好例を壊さないことを確認してから判定閾値を固定する。

### Phase 4: review exportとmanual import

`manual_review` frameをimages、overlays、初期mask、CVAT ZIP、manifestとして一括出力する。
CVATからexportした修正済みmaskを別runのH5 annotationへ反映し、point label、valid mask、
metadataの整合性を検査する。自動maskとmanual maskの上書き規則を明文化する。

### Phase 5: 全182件のteacher v3再生成

既存foreground H5を再利用し、annotation、collected、可視化を新しいrun名で生成する。

候補run名:

```text
global_local_l75_w31_c12_area15_bboxrank_v3_refined
global_local_l75_w31_c12_area15_bboxrank_v3_refined_manual_v1
```

既存の`bboxrank_v2_nobbox_bg`は上書きしない。

### Phase 6: Stage 5比較

同じStage 5設定で少なくとも次を比較する。

1. `bboxrank_v2_nobbox_bg`
2. `bboxrank_v3_refined`
3. manual修正が十分ある場合は`bboxrank_v3_refined_manual_v1`

foreground点、split、seed、モデル、loss設定を固定し、教師変更だけを比較する。

## 11. 実装予定ファイル

### 新規候補

```text
pseudo3d/analysis/audit_stage4_contour_teacher.py
pseudo3d/export/convert_masks_to_cvat_segmentation_mask_1_1.py
pseudo3d/batch/export/batch_export_stage4_manual_review_cvat.py
pseudo3d/annotation/import_cvat_segmentation_mask_corrections.py
pseudo3d/batch/annotation/batch_import_cvat_segmentation_mask_corrections.py
pseudo3d/pipelines/build_stage4_bbox_ranked_v3_annotations.sh
checks/stage4/check_stage4_contour_teacher_audit.py
checks/stage4/check_stage4_cvat_segmentation_mask_export.py
checks/stage4/check_stage4_cvat_manual_roundtrip.py
```

### 主な既存編集対象

```text
pseudo3d/annotation/annotate_pseudo3d_point_cloud.py
pseudo3d/batch/annotation/batch_annotate_pseudo3d_point_cloud.py
pseudo3d/batch/export/batch_export_annotation_mask_visualization.py
pseudo3d/batch/export/batch_export_annotation_mask_visualization.sh
pseudo3d/pipelines/build_stage4_bbox_ranked_annotations.sh
docs/stage4/stage4_sampling_investigation_progress.md
../Stage5/train_stage5.sh
```

## 12. 検証項目

### Synthetic

- overfilled BBoxがrefine対象になる
- 適正輪郭はauto_acceptされる
- 厳格化閾値で対象成分が分離される
- 小さすぎる中心成分が採用されない
- 候補拮抗時にmanual reviewとなる
- BBoxなし/background、BBox内contour外/ignoreを維持する
- 複数BBoxでlabel unionとmetadataが整合する
- 同一入力で結果が決定的である
- 0/255 binary maskがCVAT用0/1 indexed maskへ正しく変換される
- CVAT ZIPのroot構造、labelmap、default.txt、stem、member順が仕様どおりである
- missing/extra/duplicate mask、shape不一致、未知index、危険なZIP pathを拒否する
- 同一入力のCVAT artifactとchecksumが決定的である
- manual reviewが0件の場合に不正な空ZIPを作らない

### Real data

- 182件すべてで処理失敗がない
- decision、reason、scoreの欠損・非有限値を監査する
- v2からpositiveが増減したBBoxを一覧化する
- 過被覆上位例のarea ratioと境界接触が改善する
- v2の良好例を壊していない
- manual review数と理由別件数を記録する
- golden sampleをローカルCVATへimportでき、mask表示と画像対応が正しい
- CVATから再exportしたmaskのshape、stem、index、pixel領域が一致する
- CVAT修正maskとH5 positive maskが一致する
- BBox union外のmanual positiveを拒否し、暗黙にclipしない

### Stage 5 contract

- point配列、source flag、confidence、frame alignmentを保持する
- labelは`-1/0/1`のみ
- `valid_mask == (point_label != -1)`
- no-BBox点はすべてbackground
- BBox内の非positive点はすべてignore
- teacher policyとmanual provenanceをpreflightする

## 13. 完了条件

1. 全182件のBBox監査結果とreview画像が生成される
2. 自動refineとmanual reviewの判定理由をBBox単位で説明できる
3. 過被覆例で輪郭が縮小・分離され、目視で改善が確認できる
4. 良好なv2輪郭を不必要に変更しない
5. 人力修正対象をCVAT `Segmentation Mask 1.1`でexport/importできる
6. label policyとpoint配列整合性が全H5で通る
7. v2/v3のStage 5比較を再現可能なrun設定で実施できる
8. 画像、mask、manifestが外部serviceへ送信されず、local artifactのみで工程が完結する

## 14. 直近の着手内容

最初にPhase 1として、既存`bboxrank_v2_nobbox_bg`の182件を対象にread-only監査を
実装する。初回監査では自動採否のhard thresholdを決めず、分布と上位tailを出力する。
その結果から代表例を選び、Phase 2のCVAT golden sampleとPhase 3の閾値探索範囲を
確定する。golden round tripが通るまで全件review ZIP生成には進まない。

本書更新時点では方針とdata contractの固定のみを行い、converter、pipeline、importerは
まだ実装しない。
