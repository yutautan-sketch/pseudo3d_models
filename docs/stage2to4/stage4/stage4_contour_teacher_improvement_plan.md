# Stage 4 BBox-aware輪郭教師・positive判定 改修計画

作成日: 2026-08-17  
最終更新日: 2026-09-05
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
通常のStage 4教師生成パイプラインはCVAT APIへ接続せず、ローカルimport用artifactの
生成までを担当する。Phase 5のTask作成だけは独立した明示実行スクリプトから
`localhost`のCVAT APIへ接続し、`--apply`指定時に限ってuploadする。外部hostや
cloud storageは使用しない。

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

#### Phase 1実装目的

既存の`stage4_bbox_ranked_teacher_v2`による輪郭選択結果を変更せず、182件の全BBoxに
ついて候補mask、選択結果、形状、位置、過抽出、不確実性をread-onlyで監査する。

Phase 1では以下を行わない。

- point labelやannotation H5の再生成
- 自動refine
- `auto_accept`などのhard threshold確定
- CVAT用review ZIPの生成
- 既存H5やXMLへの書き込み

#### Phase 1実装予定ファイル

```text
pseudo3d/analysis/audit_stage4_contour_teacher.py
checks/stage4/check_stage4_contour_teacher_audit.py
```

必要な場合のみ、既存teacher実装へread-onlyな候補取得関数を追加する。

```text
pseudo3d/annotation/annotate_pseudo3d_point_cloud.py
```

候補生成やranking処理を監査スクリプトへ複製せず、annotation生成時と同じ処理を共有する。

#### Phase 1固定入力

- 182件の入力manifest
- 元pseudo3D H5
- `bboxrank_v2_nobbox_bg` annotated H5
- strict VOC XML
- `stage4_bbox_ranked_teacher_v2.yaml`
- teacher config fingerprint
- foreground条件:
  - global: enabled
  - local window: 31
  - local percentile: 75
  - local min contrast: 12
  - cleanup area: 15
  - top-hat/context: disabled

監査開始時に、H5、XML、video、frame、BBoxの対応関係をstrict preflightする。

#### 既存処理の再利用

以下の既存処理を再利用する。

- XMLとframeのstrict matching
- raw/local BBox変換
- global binary mask生成
- local-percentile mask生成
- cleanup処理
- global/localの全外部成分抽出
- BBox-ranked eligibility判定
- rankingと現行候補選択

監査用処理は全候補を取得できるようにするが、現行の
`build_bbox_ranked_contour_mask`の選択結果を変更しない。

#### BBox監査レコード

`bbox_audit.csv`は1 BBoxにつき1行とし、次のキーで一意にする。

```text
video_name + frame_order + frame_index + bbox_index
```

識別情報として次を記録する。

- video name
- source H5、annotated H5、XML path
- frame order、frame index
- object index、BBox index、object name
- XML/raw/local BBox座標
- image shape、BBox幅、高さ、面積

保存済みteacher結果として次を記録する。

- valid contour
- annotation reason
- selected contour source
- contour selection score
- contour area
- selected area ratio
- foreground ratio in BBox
- frame point数
- BBox内point数
- positive point数

候補情報として次を記録する。

- global/local候補数
- global/local eligible候補数
- source別best score
- 1位・2位候補のscore
- score margin
- 1位・2位候補のsource
- 1位・2位maskのIoU
- global/local best maskのIoU
- 候補間の重心距離

過抽出・形状指標として次を記録する。

- source foreground / BBox面積比
- filled contour / BBox面積比
- positive point / BBox内point比
- BBox中心と輪郭重心の正規化距離
- BBox境界接触pixel数・接触率
- 上下左右それぞれの境界接触
- 接続成分数
- solidity
- extent
- perimeter
- compactness
- contour bounding rectangle / BBox extent

閾値変更に対するstabilityはPhase 3で計測し、Phase 1では現行global/local候補間の差だけを
扱う。

#### 保存済み結果との一致検証

再計算したteacher結果と、annotated H5に保存された以下の値を照合する。

- valid contour
- annotation reason
- selected source
- contour area
- selected area ratio
- center distance
- positive point数
- global/local候補数
- source別score

許容誤差を超える差があれば監査結果として続行せず、設定または入力の不一致として
失敗させる。

#### 監査カテゴリ

hard thresholdは設定せず、metricによる順位として抽出する。

- `overfilled`: 面積比が大きい
- `border_contact`: BBox境界接触率が高い
- `ambiguous`: score marginが小さく候補maskが異なる
- `off_center`: 中心距離が大きい
- `invalid`: eligible候補なし、または曖昧性による棄却
- `small_contour`: 面積・positive点数が小さい
- `candidate_good_control`: 中心・面積・境界接触が比較的安定

各カテゴリの上位件数はCLIで指定し、初期値を20とする。同じBBoxが複数カテゴリに
入ることを許可する。

#### Overlay生成

ランキング対象について、次を描画する。

- 元local frame
- annotation BBox
- global binary
- local binary
- global候補輪郭
- local候補輪郭
- 現在の採用輪郭
- BBox中心と候補重心
- candidate sourceと順位
- score、面積比、中心距離、境界接触率
- audit category

元画像とoverlayを混同しないよう、監査専用ディレクトリへ出力する。

#### Phase 1出力構造

```text
stage4_contour_teacher_audit/
├── bbox_audit.csv
├── video_summary.csv
├── category_summary.csv
├── failures.csv
├── audit_summary.json
├── run_config.yaml
└── overlays/
    ├── overfilled/
    ├── border_contact/
    ├── ambiguous/
    ├── off_center/
    ├── invalid/
    ├── small_contour/
    └── candidate_good_control/
```

`run_config.yaml`またはsummaryへ、入力manifest、teacher config、fingerprint、実行日時、
対象件数、スクリプトversionを保存する。

#### CLI方針

最低限、以下を引数化する。

- input manifest
- pseudo3D/annotated H5 root
- teacher config
- output root
- categoryごとのtop-K
- overlay生成の有無
- overwrite禁止または明示的許可
- preflight-only
- continue-on-error

最終監査では`failure_rows=0`を必須とする。

#### Phase 1 Synthetic check

- 既知形状でarea ratio、重心、境界接触率が正しい
- solidity、extent、compactnessが有限である
- 空maskとinvalid BBoxを安全に扱う
- 複数候補のscore marginとIoUが正しい
- global/local候補のsourceを混同しない
- 複数BBoxと非連続frame indexが整合する
- audit追加前後で現行teacherの選択結果が変わらない
- CSV行順とoverlay選択が決定的である
- 入力H5を更新しない

#### Phase 1完了条件

- manifest上の182件をすべて監査できる
- 全XML BBoxに一意な監査行がある
- 保存済みteacher結果と再計算結果が一致する
- unexpected NaN/Infがない
- invalid値にはreason codeがある
- failure rowsが0件
- 既存H5/XMLが変更されていない
- 各カテゴリの上位例と良好control例を目視できる
- Phase 2用CVAT golden sampleを2〜3件選べる
- Phase 3で調査するrefine条件と閾値範囲を提案できる

#### Phase 1実装状況（2026-08-18）

次のファイルへPhase 1を実装した。

```text
pseudo3d/analysis/audit_stage4_contour_teacher.py
checks/stage4/check_stage4_contour_teacher_audit.py
pseudo3d/annotation/annotate_pseudo3d_point_cloud.py
```

annotation実装へ追加したのは、teacher v2と同じ候補生成およびsort keyを監査側から
read-onlyで利用する共有APIである。既存の候補eligibility、ranking、tie判定は変更せず、
annotation生成と監査が同じ計算経路を使う。

監査CLIは次を実装済みである。

- manifest、teacher config、annotated H5、strict XMLのpreflight
- teacher configとannotated H5 metadataの一致検査
- global/local候補と現行選択結果の再計算
- 保存済みframe annotationおよびpoint labelとの完全照合
- BBox単位の位置、面積、境界接触、形状、候補差metric
- metric rankingによる7カテゴリの上位抽出
- category別overlay、BBox/video/category summaryの生成
- 入力H5/XMLのsizeとmtimeが実行前後で不変であることの検査
- failure row、run config、teacher/manifest fingerprintの保存

Synthetic checkは、既知形状、候補比較、ranking決定性に加え、複数BBox、非連続frame
index、BBoxなしframeを含む小型H5/XMLのend-to-end監査を対象とする。

実データ監査前に次を実行する。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_contour_teacher_audit.py
```

182件のpreflightは次のコマンドで行う。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

RUN_ROOT=/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/global_local_l75_w31_c12_area15_bboxrank_v2_nobbox_bg

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  pseudo3d/analysis/audit_stage4_contour_teacher.py \
  --manifest /mnt/data/3d_projects/pseudo3d_dataset/stage4_sampling_parameter_sweep/260711/manifests/train_manifest.csv \
  --annotated_root "${RUN_ROOT}/annotated" \
  --teacher_config pseudo3d/analysis/configs/stage4_bbox_ranked_teacher_v2.yaml \
  --output_root "${RUN_ROOT}/contour_teacher_audit_phase1_preflight" \
  --expected_videos 182 \
  --preflight_only \
  --continue_on_error
```

preflight通過後、`--preflight_only`を外し、正式な`contour_teacher_audit_phase1`へ全監査結果を
生成する。preflight専用rootと分離するため`--overwrite`は不要である。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

RUN_ROOT=/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/global_local_l75_w31_c12_area15_bboxrank_v2_nobbox_bg

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  pseudo3d/analysis/audit_stage4_contour_teacher.py \
  --manifest /mnt/data/3d_projects/pseudo3d_dataset/stage4_sampling_parameter_sweep/260711/manifests/train_manifest.csv \
  --annotated_root "${RUN_ROOT}/annotated" \
  --teacher_config pseudo3d/analysis/configs/stage4_bbox_ranked_teacher_v2.yaml \
  --output_root "${RUN_ROOT}/contour_teacher_audit_phase1" \
  --expected_videos 182 \
  --top_k 20 \
  --continue_on_error
```

同じoutput rootで再実行する場合に限り、内容を置換する明示的な`--overwrite`を付ける。

#### Phase 1 Synthetic check結果（2026-08-19）

`dualtrack311`環境で次の全項目が成功した。

- shape、center、BBox border metric
- candidate comparisonとteacher-selection isolation
- audit category rankingの決定性
- 複数BBox、非連続frame index、BBoxなしframeを含むend-to-end監査
- 保存済みannotation H5と再計算teacherの一致
- 2回の独立実行におけるCSVとoverlayの決定性
- 入力pseudo3D H5、annotated H5、XMLの不変性
- teacher configおよびmanifest validation failure

結果は次のとおりである。

```text
videos_checked        : 1/1
bbox_rows             : 2
valid_contours        : 2
invalid_contours      : 0
failure_rows          : 0
input_files_unchanged : true
phase1_synthetic_exit_code=0
```

これによりPhase 1の実装基盤は確認済みとする。Phase 1全体の完了には、引き続き182件の
preflight、全監査、category上位overlayの目視確認が必要である。

### Phase 2: CVAT最小converterとgolden round trip

2〜3画像のbinary maskを`Segmentation Mask 1.1`へ変換し、ZIP構造、labelmap、stem、
shape、indexを検証する。ローカルCVATへのimport/export round tripを行い、仕様を固定する。

#### Phase 2の実装範囲

Phase 2では、Stage 4全件のmanual-review package生成に先立ち、CVATとの最小data contractを
独立したconverterで確定する。対象はflat directoryに置かれた2〜3画像と、同stemの
single-class binary maskである。

Phase 2に含める。

- 元画像とbinary maskのstrict pairing
- binary `0/255`からindexed `0/1`への変換
- `Segmentation Mask 1.1` ZIPの決定的生成
- 生成ZIPを再度開くself validation
- CVATからexportした同形式ZIPのread-only validation
- 2〜3画像のlocal CVAT golden round trip

Phase 2には含めない。

- Phase 1 audit結果からのmanual-review対象自動選択
- H5からのreview画像・proposal mask抽出
- 複数class、RGB mask、任意class ID
- 複数instanceとBBox indexの対応付け
- CVAT APIへの接続または自動upload
- CVAT修正maskのH5 import
- 元H5、元画像、元maskの変更

これらはPhase 4のreview export/manual importで実装する。

#### Phase 2実装ファイル

```text
pseudo3d/export/convert_masks_to_cvat_segmentation_mask_1_1.py
checks/stage4/check_stage4_cvat_segmentation_mask_export.py
```

converter内部のZIP validation関数は、Phase 4のbatch exporterおよびmanual importerから
再利用できるpure/read-only APIとして設計する。

#### Phase 2入力契約

初期versionでは次の構成だけを受け付ける。

```text
images/
├── image_00001.png
├── image_00002.jpg
└── image_00003.png

masks/
├── image_00001.png
├── image_00002.png
└── image_00003.png
```

- imageはflat directoryの`.png`、`.jpg`、`.jpeg`に限定する
- maskはflat directoryの`.png`に限定する
- image/maskは拡張子を除くstemで対応させる
- stemはdirectory内で一意でなければならない
- imageとmaskのstem集合は完全一致させる
- maskは2-D、1-channel、`uint8`とする
- mask pixelは`{0, 255}`の部分集合だけを許可する
- `0=background`、`255=femur proposal`とする
- imageとmaskのwidth/heightは完全一致させる
- symlink、nested directory、未対応拡張子を拒否する
- 自動rename、自動resize、補間、輪郭化を行わない

all-background maskは形式上有効とする。golden sample全体では、変換確認のため少なくとも
1枚にforeground pixelを含める。

#### Phase 2 CLI案

converterは少なくとも`convert`と`validate`の2 modeを提供する。

```text
convert:
  --images_dir
  --masks_dir
  --output_zip
  --label_name femur
  --background_value 0
  --foreground_value 255
  --summary_json
  --keep_unpacked_dir
  --overwrite

validate:
  --input_zip
  --images_dir
  --summary_json
```

`convert`はZIP生成後に必ず同じvalidatorを実行し、validationが失敗した場合は最終ZIPを
残さない。`validate`はCVATから再exportしたZIPをread-onlyで検査するために使用する。
既存ZIPの置換は明示的な`--overwrite`がある場合だけ許可する。

#### Phase 2変換処理

stemをlexicographical sortし、各maskを次のように変換する。

```text
input mask : uint8 {0, 255}
output mask: uint8 {0, 1}

0   -> 0 (background)
255 -> 1 (femur)
```

出力maskはgrayscale/indexed PNGとし、palette画像やRGB画像へ暗黙変換しない。

`labelmap.txt`は次の内容に固定する。

```text
background:0,0,0::
femur:255,0,0::
```

foreground label名を変更可能にする場合も、改行、区切り文字、空文字、重複、path文字を
検査し、mask index `1`との対応を維持する。

`ImageSets/Segmentation/default.txt`には、sort済みstemを拡張子なしで1行ずつ記録する。

#### Phase 2 ZIP契約

ZIP rootを次に固定する。

```text
annotations_segmentation_mask_1_1.zip
├── labelmap.txt
├── ImageSets/
│   └── Segmentation/
│       └── default.txt
├── SegmentationClass/
│   ├── image_00001.png
│   ├── image_00002.png
│   └── image_00003.png
└── SegmentationObject/
    ├── image_00001.png
    ├── image_00002.png
    └── image_00003.png
```

Phase 2はsingle foreground objectを前提とし、`SegmentationClass`と
`SegmentationObject`へ同じ`0/1` maskを格納する。複数instance frameへの一般化は
golden sampleでCVAT exportの挙動を確認した後、Phase 4で設計する。

ZIPには元画像、summary、外側の親directory、絶対path、`..`を含むmemberを入れない。

#### 決定的ZIP生成

同一入力からbyte-identical ZIPを生成できるよう、次を固定する。

- memberの順序
- stemのsort順
- text fileのUTF-8 encodingとLF改行
- PNG encoding
- ZIP timestamp
- ZIP permission/external attributes
- compression methodとcompression level

最終ZIPは一時fileへ生成・検査してからatomicに配置し、途中失敗時に不完全なZIPを
残さない。ZIP SHA-256をsummaryへ保存する。

#### Phase 2 strict validation

converter実行前に次を検査する。

- image/maskのmissing、extra、duplicate stem
- unsupported extension、nested file、symlink
- image decode failure、mask decode failure
- image/mask shape mismatch
- maskのdimension、dtype、unique value
- label名と出力先

生成後またはCVAT export後のZIPについて次を検査する。

- ZIP member pathにabsolute path、`..`、backslash、余分なrootがない
- duplicate memberがない
- `labelmap.txt`がroot直下に1個だけ存在する
- `ImageSets/Segmentation/default.txt`が存在する
- `SegmentationClass`と`SegmentationObject`に全stemのPNGが存在する
- default stem、class stem、object stemの集合が完全一致する
- default stemに拡張子、空行、重複がない
- class/object PNGが2-D `uint8`である
- unique indexが`{0, 1}`の部分集合である
- class/object PNGのshapeが元画像と一致する
- labelmapの0-based行indexとmask indexが一致する
- 宣言外memberが存在しない

validation結果は標準出力とZIP外のJSONへ保存する。

```text
images
masks
classes
unique_indices
size_mismatches
missing_masks
extra_masks
zip_members
zip_sha256
status
```

#### Phase 2 Synthetic check

`check_stage4_cvat_segmentation_mask_export.py`で次を確認する。

- 3画像の正常変換
- input `0/255`とoutput `0/1`のpixel単位一致
- class/object maskのshape、dtype、unique index
- labelmapとdefault.txtの内容・改行
- ZIP root構造とmember集合
- ZIPに元画像や余分なparent directoryが含まれない
- 入力順やfile作成順に依存しない決定的出力
- 同一入力から生成した2 ZIPがbyte-identical
- all-background maskを保持できる
- missing/extra/duplicate stemを拒否する
- unsupported extension、nested file、symlinkを拒否する
- shape mismatchを拒否し、自動resizeしない
- RGB/non-uint8/未知pixel値maskを拒否する
- invalid label名を拒否する
- 既存outputを`--overwrite`なしで置換しない
- validation失敗時に不完全な最終ZIPを残さない
- validatorが危険path、duplicate member、未知member、未知indexを拒否する

Synthetic checkは一時directoryだけを使用し、実画像をrepositoryへ追加しない。

#### Phase 2 local CVAT golden round trip

Synthetic check通過後、外部へ送信できない2〜3枚のlocal sampleを使って次を行う。

1. globally uniqueなstemへ事前に固定した画像と0/255 maskを準備する
2. converterで`Segmentation Mask 1.1` ZIPを生成する
3. local CVATに`femur` labelを持つtest Taskを作成する
4. Taskへ同じbasename/stemの画像を登録する
5. 既存annotationがあれば先にexportしてbackupする
6. `Segmentation Mask 1.1`、`Convert masks to polygons=OFF`でimportする
7. 画像対応、向き、位置、label、mask境界を目視確認する
8. 編集前に同形式でexportし、validatorでstem、shape、index、pixel領域を比較する
9. Brush/Eraserで局所編集し、保存後に再exportできることを確認する
10. 編集後ZIPもvalidatorで形式が維持されていることを確認する

golden round tripでは次を記録する。

- converter input/output SHA-256
- CVAT version
- Task label設定
- import formatとpolygon変換OFF
- import前backup path
- no-edit exportのvalidation summary
- edit後exportのvalidation summary
- 目視確認結果

CVAT Taskへのuploadとexportはユーザーがlocalhost上で手動実行する。converterやtestから
CVAT API、外部network、cloud storageへ接続しない。

#### Phase 2完了条件

- converter Synthetic checkがexit code 0で通る
- 2〜3画像のZIPが仕様どおり決定的に生成される
- local CVATへmaskとしてimportできる
- image/stem/shape/label/pixel位置が正しい
- no-edit exportがvalidatorを通り、class maskが入力と一致する
- Brush/Eraserで編集して再exportできる
- edit後exportも`Segmentation Mask 1.1` validationを通る
- patient画像またはmaskが外部serviceへ送信されていない
- Phase 4で再利用するZIP contractとvalidator APIが固定される

#### Phase 2実装状況（2026-08-19）

次の2ファイルを実装した。

```text
pseudo3d/export/convert_masks_to_cvat_segmentation_mask_1_1.py
checks/stage4/check_stage4_cvat_segmentation_mask_export.py
```

converterには次を実装済みである。

- `convert`と`validate`の2 subcommand
- flat image/mask directoryのstrict pairingと入力検査
- binary `0/255` maskからindexed `0/1` class/object maskへの変換
- 固定labelmap、default set、member順、timestamp、permission、PNG圧縮による決定的ZIP生成
- temporary file上でのself validationとatomicな最終ZIP配置
- SHA-256、stem、index、比較件数を含むJSON summary
- ZIP path traversal、symlink、duplicate/unknown member、未知index、過大memberの拒否
- 元画像とのshape照合、および任意のreference maskとのpixel単位照合
- CVATからexportしたno-edit/edit済みZIPを検査するread-only validator

Synthetic checkには正常系、byte identity、入力不変性、all-background mask、filesystem/pairing、
shape/dtype/value、label、overwrite、異常ZIPを含めた。実行コマンドは次のとおりである。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_cvat_segmentation_mask_export.py

echo "phase2_synthetic_exit_code=$?"
```

#### Phase 2 Synthetic check結果（2026-08-19）

`dualtrack311`環境で次の全項目が成功した。

- binary `0/255`からindexed `0/1`への変換とZIP contract
- file作成順に依存しない決定的出力とbyte identity
- missing、extra、duplicate stemの拒否
- unsupported extension、nested directory、symlinkの拒否
- shape mismatch、RGB、`uint16`、未知pixel値maskの拒否
- 失敗時に不完全な最終ZIPを残さないこと
- unknown member、危険path、duplicate member、未知indexを持つZIPの拒否
- label validationと明示的なoverwrite

結果は次のとおりである。

```text
[OK] deterministic 0/255 -> 0/1 conversion and exact ZIP contract
[OK] missing/extra/duplicate/unsupported/nested/symlink rejection
[OK] shape/RGB/uint16/unknown-value rejection without partial ZIP
[OK] unknown/unsafe/duplicate/unknown-index ZIP rejection
[OK] label validation and explicit overwrite behavior
Stage 4 CVAT Segmentation Mask 1.1 synthetic checks passed.
phase2_synthetic_exit_code=0
```

これによりPhase 2のconverter、validator、決定的artifact生成、およびstrict validationの
実装基盤は確認済みとする。

次に、local CVAT golden sampleを次のように生成する。`GOLDEN_ROOT/images`と
`GOLDEN_ROOT/masks`には、同じstemを持つ2〜3件の画像と0/255 maskを配置する。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

GOLDEN_ROOT=/mnt/data/3d_projects/pseudo3d_dataset/stage4_contour_teacher_cvat_golden

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  pseudo3d/export/convert_masks_to_cvat_segmentation_mask_1_1.py convert \
  --images_dir "${GOLDEN_ROOT}/images" \
  --masks_dir "${GOLDEN_ROOT}/masks" \
  --output_zip "${GOLDEN_ROOT}/cvat/annotations_segmentation_mask_1_1.zip" \
  --label_name femur \
  --summary_json "${GOLDEN_ROOT}/cvat/conversion_summary.json" \
  --keep_unpacked_dir "${GOLDEN_ROOT}/cvat/unpacked_reference"
```

CVATでは同じ画像と`femur` labelを持つlocal Taskを作り、`Segmentation Mask 1.1`、
`Convert masks to polygons=OFF`でimportする。編集せずに同形式でexportしたZIPは、入力maskと
pixel単位で照合する。

```bash
/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  pseudo3d/export/convert_masks_to_cvat_segmentation_mask_1_1.py validate \
  --input_zip "${GOLDEN_ROOT}/cvat/cvat_no_edit_export.zip" \
  --images_dir "${GOLDEN_ROOT}/images" \
  --reference_masks_dir "${GOLDEN_ROOT}/masks" \
  --label_name femur \
  --summary_json "${GOLDEN_ROOT}/cvat/no_edit_validation.json"
```

Brush/Eraserで変更したexportは入力maskと異なるため、`--reference_masks_dir`を指定せず形式と
画像shapeを検査する。

```bash
/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  pseudo3d/export/convert_masks_to_cvat_segmentation_mask_1_1.py validate \
  --input_zip "${GOLDEN_ROOT}/cvat/cvat_edited_export.zip" \
  --images_dir "${GOLDEN_ROOT}/images" \
  --label_name femur \
  --summary_json "${GOLDEN_ROOT}/cvat/edited_validation.json"
```

実装、静的検査、`dualtrack311`でのSynthetic checkは完了している。Phase 2全体の完了には、
localhost上のCVATを用いたgolden import/no-edit export/edit exportの確認が必要である。

### Phase 3: auto-refine prototype

監査対象の代表例でBBox内追加閾値を実装し、候補metricsとrankingを可視化する。
既知の良好例を壊さないことを確認してから判定閾値を固定する。

#### Phase 3の目的

Phase 1の監査結果から、`overfilled`、`border_contact`、`ambiguous`、`off_center`、
`invalid`と判定された代表BBoxを対象に、BBox内の追加threshold候補を生成する。
現行teacher v2のglobal/local候補をbaselineとして同じ候補集合へ残し、追加候補が品質と
安定性の両方で明確に改善した場合だけ`auto_refine`を提案する。

Phase 3は候補生成、metric、判定規則を調査するread-only prototypeとし、annotated H5、
point label、元pseudo3D H5、XMLを変更しない。Phase 3の`teacher_decision`は評価用の
`proposed_decision`であり、学習用H5へ反映するのはPhase 5とする。

#### Phase 3の実装範囲

Phase 3に含める。

- Phase 1監査結果からの代表BBoxの決定的選択
- 現行v2 global/local候補の完全な再現
- BBox crop内のboundedな追加threshold候補生成
- 候補maskの重複排除、metric計算、eligibility、ranking
- 近接threshold間のmask stability計測
- `auto_accept`、`auto_refine`、`manual_review`の提案
- v2選択maskと提案maskの比較CSV、binary mask、overlay
- 入力file不変性と出力決定性の検査

Phase 3には含めない。

- annotated H5またはpoint labelの更新
- 182件すべてのteacher v3生成
- CVAT review packageの生成またはCVAT修正結果のimport
- watershedなど候補数と挙動が大きく変わる分離処理
- Stage 5学習
- foreground samplingの再計算

watershedまたはdistance transformは、boundedなthreshold/morphology候補で改善しない代表例が
複数確認された場合に限り、Phase 3後半または別configとして追加する。

#### Phase 3実装ファイル

```text
pseudo3d/annotation/contour_teacher_refinement.py
pseudo3d/analysis/prototype_stage4_contour_auto_refine.py
pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3.yaml
checks/stage4/check_stage4_contour_auto_refine.py
```

必要な既存編集対象は次のとおりである。

```text
pseudo3d/annotation/annotate_pseudo3d_point_cloud.py
pseudo3d/analysis/audit_stage4_contour_teacher.py
```

候補生成、metric、decisionは`contour_teacher_refinement.py`のpure APIへ置く。runnerや
Synthetic checkへ処理を複製しない。現行v2候補の生成とsort keyは
`annotate_pseudo3d_point_cloud.py`の共有APIをそのまま使用する。Phase 1監査のinput解決処理を
共通化する必要が生じた場合も、Phase 1の出力schemaと選択結果を変えない回帰検査を行う。

#### Phase 3入力契約

prototypeは次を入力とする。

- Phase 1で使用したtrain manifest
- Phase 1の`bbox_audit.csv`、`category_summary.csv`、`run_config.yaml`
- `bboxrank_v2_nobbox_bg` annotated H5 root
- `stage4_bbox_ranked_teacher_v2.yaml`
- Phase 3 refine config

実行前に次をstrict validationする。

- Phase 1 auditが成功し、`failure_rows=0`である
- manifest、teacher config、annotated rootのpathとfingerprintがPhase 1記録と一致する
- BBox key `(video_name, frame_order, frame_index, bbox_index)`が一意である
- selection rowが`bbox_audit.csv`に存在し、保存metricと一致する
- pseudo3D H5、annotated H5、strict XMLが存在する
- frame数、frame index、画像shape、BBox座標が整合する
- annotated H5がteacher v2 schemaと固定foreground条件を持つ
- refine configの全threshold、kernel、weight、quality gateが有限で有効範囲内にある

代表BBoxは`category_summary.csv`からカテゴリ別上位をunionして重複排除し、
`candidate_good_control`を必ず含める。自動選択結果は`selection_resolved.csv`へ保存する。
必要な場合は同schemaの`--selection_csv`で目視選択例を追加できるようにするが、manifest外の
video/BBoxは受け付けない。

#### Phase 3 refine config

設定はYAMLへ固定し、少なくとも次を持たせる。

```text
schema_version
config_name
base_teacher_config_name
base_teacher_fingerprint
trigger.*
candidate_generation.percentile_values
candidate_generation.otsu_enabled
candidate_generation.adaptive_methods
candidate_generation.adaptive_block_sizes
candidate_generation.adaptive_c_values
candidate_generation.cleanup_variants
eligibility.*
ranking.weights.*
decision.minimum_refine_score_gain
decision.minimum_score_margin
decision.minimum_stability_iou
decision.minimum_support_count
```

`trigger`、`eligibility`、`decision`のproduction数値はPhase 1の実データ分布から固定し、
計画時点で推測しない。config templateで未確定値を`null`にする場合、通常実行では拒否し、
候補分布だけを出す明示的な`--candidate_generation_only`でのみ許可する。Synthetic checkでは
既知形状用の独立した固定configを使用する。

`base_teacher_config_name`は必須とし、`base_teacher_fingerprint`は特定のcanonical YAMLを
追加で固定するときだけ指定する。通常はPhase 1 auditに保存されたteacher fingerprintと
現在のteacher fingerprintをCLIが完全照合するため、refine config側のfingerprintは`null`を
許可する。config fileのSHA-256とcanonical fingerprintを全出力へ保存し、CLIから個別閾値を暗黙に
上書きしない。調査条件を変更する場合は別config名と別output rootを使用する。

#### 代表BBoxとrefine発動条件

初期prototypeでは次のいずれかを満たすBBoxをrefine評価対象とする。

- selected contour/BBox area ratioが設定上限以上
- foreground/BBox area ratioが設定上限以上
- BBox border contact ratioが設定上限以上
- center distance normが設定上限以上
- 1位・2位候補のscore marginが設定下限以下でmaskが異なる
- v2候補がinvalid
- Phase 1で`overfilled`、`border_contact`、`ambiguous`、`off_center`、`invalid`に選ばれた

`candidate_good_control`は発動条件に該当しなくても評価へ含め、baselineがそのまま
`auto_accept`されることを確認する。カテゴリ名だけで置換を決めず、必ず元metricから
発動条件を再計算する。

#### 追加候補の生成

各BBoxを画像範囲へclipし、現行teacherと同じfull-resolution local imageからBBox cropを
取得する。resize、補間、BBox外pixelの参照を行わない。threshold maskはBBox外を常に0とする。

boundedな初期候補集合を次に限定する。

1. BBox crop内のhigh-intensity percentile threshold
2. BBox crop内のOtsu threshold
3. 設定したblock size/Cだけのadaptive meanまたはGaussian threshold
4. 上記へ設定済みmorphology open/closeと小成分除去を適用したvariant
5. 現行v2のglobal/local候補

percentile、adaptive block size、morphology kernel、minimum component areaはconfigに列挙した
有限個だけを使用する。入力imageのdtype/rangeを明示的に正規化し、同じframe/BBox/configから
常に同じbinary maskを得る。

各binary maskではexternal connected componentを独立候補として抽出し、filled contourを
作る。複数BBoxは独立に処理し、別BBoxの中心やmaskをrankingへ利用しない。候補maskはBBoxへ
clipし、mask byte列のSHA-256で重複排除する。同一maskが複数method/thresholdから得られた場合は
候補を複製せず、全provenanceと`support_count`を保持する。

#### 候補metricとstability

候補ごとに少なくとも次を保存する。

- source、threshold method/value、cleanup variant、component index
- mask SHA-256、supporting parameter一覧、support count
- contour pixel数、filled area、area/BBox ratio
- contour centroid、center distance norm
- border contact ratioと上下左右の接触flag
- contour BBox extent、solidity、perimeter、compactness、aspect ratio
- BBox内point数、候補内point数、positive point ratio
- v2 selected maskとのIoU、area差、centroid移動量
- 近接threshold候補との最大/中央値IoU
- eligibility、rejection reason codes、各score項、total score

stabilityは同じmethod familyの隣接parameterで得た候補をmask IoUにより対応付けて評価する。
単独parameterだけで現れる候補を高信頼にしない。異なるmethodで同一maskが得られた場合も
supportとして記録するが、support数だけで品質gateを迂回させない。

#### eligibilityとranking

eligibilityをrankingより先に適用する。少なくとも次をhard rejectionとする。

- 空mask、非有限metric、無効contour
- absolute areaまたはarea ratioが下限未満
- area ratioが上限を超える
- center distance normが上限を超える
- border contactが許容上限を超える
- BBox内の保存pointを1点も含まない

ranking scoreは次の独立項をconfig weightで合成する。

- BBox中心との近さ
- 設定した許容area band内でのarea score
- border non-contact score
- solidity/compactness/extentによるshape score
- threshold stability score
- 複数parameter/methodによるsupport score

面積を単純に最大化せず、中心に近い微小成分も選ばない。global/local/refine sourceに固定の
優先順位を設けない。決定性のため、score各項とmetricのsort keyを固定する。ただし、許容差内で
同点かつmaskが異なる候補はhash順で自動採用せず、ambiguousとして`manual_review`へ送る。

#### proposed decision

BBox単位の判定を次に固定する。

`auto_accept`:

- v2選択maskがquality gateを満たす
- refine triggerが発動しない、または追加候補が必要改善幅を満たさない
- v2 maskをpixel単位でそのまま維持する

`auto_refine`:

- 追加候補がすべてのeligibility/quality gateを満たす
- v2候補より`minimum_refine_score_gain`以上改善する
- 2位候補との差が`minimum_score_margin`以上ある
- stability IoUとsupport countが設定下限以上である
- mask変更理由をreason codeとscore差で説明できる

`manual_review`:

- eligible候補がない
- 全候補がoverfilled、off-center、border-contact、too-smallのいずれかで棄却される
- 1位・2位が拮抗してmaskが異なる
- threshold間のmask/centroid変動が大きい
- 複数BBoxまたは複数成分の対応が曖昧である
- v2候補は不適切だが、追加候補の改善幅が採用条件に届かない

refine triggerが発動してもbaselineが最良で品質gateを満たす場合は`auto_accept`へ戻せる。
追加候補の存在だけを理由に`auto_refine`しない。Phase 3では判定とproposal maskを出力するだけで、
`manual_review` BBoxのpoint label変更やignore化は行わない。

#### Phase 3出力契約

出力は既存runと分離した次の構造とする。

```text
contour_auto_refine_phase3/
├── selection_resolved.csv
├── bbox_decisions.csv
├── candidate_metrics.csv
├── video_summary.csv
├── category_summary.csv
├── failures.csv
├── refine_summary.json
├── run_config.yaml
├── proposal_masks/
│   └── {video}__fo.....__bbox....png
└── overlays/
    ├── auto_accept/
    ├── auto_refine/
    ├── manual_review/
    └── regressions/
```

`bbox_decisions.csv`はv2とproposalのsource、mask SHA-256、全主要metric、score差、decision、
reason codesを1 BBox 1行で保持する。`candidate_metrics.csv`は1候補1行とする。
proposal maskは評価用binary `0/255` PNGで、学習用teacherまたはCVAT確定maskとして扱わない。

overlayは最低限、元画像、BBox、v2 mask、proposal mask、追加/削除pixel、候補centroid、
主要metric、decision/reasonを表示する。`auto_refine`と`manual_review`は全件、
`auto_accept`はカテゴリ別control上位を出力する。

summaryにはmanifest、Phase 1 audit、teacher/refine configのpath・SHA-256・fingerprint、
入力件数、decision/reason別件数、failure数、入力不変性を保存する。作成日時を除くCSV、mask、
overlayの内容は同一入力で決定的にし、比較用checksum manifestも保存する。

#### Phase 3 CLI案

```bash
cd /mnt/data/3d_projects/models/Stage2to4

RUN_ROOT=/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/global_local_l75_w31_c12_area15_bboxrank_v2_nobbox_bg

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  pseudo3d/analysis/prototype_stage4_contour_auto_refine.py \
  --manifest /mnt/data/3d_projects/pseudo3d_dataset/stage4_sampling_parameter_sweep/260711/manifests/train_manifest.csv \
  --annotated_root "${RUN_ROOT}/annotated" \
  --phase1_audit_root "${RUN_ROOT}/contour_teacher_audit_phase1" \
  --teacher_config pseudo3d/analysis/configs/stage4_bbox_ranked_teacher_v2.yaml \
  --refine_config pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3.yaml \
  --output_root "${RUN_ROOT}/contour_auto_refine_phase3" \
  --top_per_category 20
```

既存output rootを暗黙に追記・置換しない。同じ条件を再実行する場合だけ明示的な
`--overwrite`を必要とする。`--continue_on_error`を指定した場合も最後にfailure件数が1件以上なら
non-zero exitとし、不完全な結果を成功扱いしない。

#### Phase 3 Synthetic check

`check_stage4_contour_auto_refine.py`で次を確認する。

- 適正なv2輪郭がpixel単位で維持され`auto_accept`になる
- BBoxを覆う結合成分が厳格thresholdで分離され`auto_refine`になる
- BBox中心に近いだけの微小成分をrejectする
- 大面積だがBBox境界へ広く接触する成分をrejectする
- 同じmaskを作る複数parameterが重複排除されsupportへ集約される
- 隣接thresholdのIoUとcentroid stabilityが正しい
- 同点の異なるmaskをhash順で採用せず`manual_review`にする
- eligible候補なし、unstable、改善幅不足がそれぞれ期待decision/reasonになる
- 複数BBoxを独立処理し、maskとmetricを混同しない
- BBoxなしframe、非連続frame index、empty frameを安全に扱う
- v2 candidate APIの選択結果がPhase 3追加前後で変わらない
- point labelと入力H5/XMLのsize、mtime、checksumが変化しない
- 同一入力のCSV、mask、overlay、decisionが決定的である
- invalid config、fingerprint mismatch、未知BBox keyを拒否する

実装後の確認コマンドは次とする。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_contour_auto_refine.py

echo "phase3_synthetic_exit_code=$?"
```

#### Phase 3実データ確認

最初から182件のteacherを再生成せず、Phase 1カテゴリ上位とgood controlを合わせた代表集合で
実行する。確認順序は次とする。

1. `--candidate_generation_only`で候補metric分布を得る
2. area、center、border、stability、score marginの分布からconfig閾値を固定する
3. 同じselectionでdecision modeを実行する
4. `auto_refine`、`manual_review`、`regressions` overlayを全件目視する
5. v2 good controlが不必要に変更されていないことを確認する
6. overfilled/border-contact例でareaと境界接触が改善し、対象断面を欠落していないことを確認する
7. 閾値をわずかに変えた感度確認を1回行い、decisionが大きく反転しないことを確認する

目視で誤った`auto_refine`が見つかった場合は、個別video名による例外規則を追加せず、
quality gate、stability、manual-review条件のいずれで一般化して防ぐかを検討する。

#### Phase 3完了条件

- Synthetic checkがexit code 0で通る
- representative selectionと全候補が入力/configから決定的に再現できる
- baseline good controlのmaskが不必要に変更されない
- 明確な過抽出例でproposalが縮小または分離され、目視でも改善している
- 不安定または拮抗した例が`manual_review`へ送られる
- 全decisionに機械可読なreason codeと比較metricがある
- 入力H5/XMLと既存annotated H5が変更されていない
- Phase 4/5で再利用できるcandidate/decision APIとconfig schemaが固定される

#### Phase 3実装順序

1. refine config loaderとstrict validation
2. pureな追加candidate生成、重複排除、metric/stability API
3. eligibility、ranking、proposed decision API
4. Phase 1 audit入力を使うread-only prototype CLI
5. CSV、mask、overlay、summaryの決定的出力
6. Synthetic check
7. representative実データのcandidate-only調査
8. threshold固定後のdecision/目視確認

#### Phase 3実装状況（2026-08-19）

次の4ファイルを実装した。

```text
pseudo3d/annotation/contour_teacher_refinement.py
pseudo3d/analysis/prototype_stage4_contour_auto_refine.py
pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3.yaml
checks/stage4/check_stage4_contour_auto_refine.py
```

実装済み内容は次のとおりである。

- v2 global/local候補を変更せずbaselineとして再利用するpure refine engine
- BBox内percentile、Otsu、adaptive thresholdとbounded cleanup variant
- connected component単位のfilled mask生成とmask SHA-256による重複排除
- 同一maskを支持する複数parameterのprovenance/support集約
- area、center、border、shape、point、baseline差、stability metric
- eligibilityを先に適用するconfig-driven ranking
- `auto_accept`、`auto_refine`、`manual_review`のreason付き提案
- Phase 1 manifest/audit/config fingerprintをstrictに照合するread-only CLI
- selection、BBox decision、全候補、video/category summary、mask、overlay、checksum出力
- 入力H5/XML/Phase 1 artifactのsize・mtime不変性検査
- good control、overfilled、tiny、ambiguous、deduplication、end-to-end決定性のSynthetic check

初期configのtrigger、eligibility、ranking、decision値はscreening用の暫定値である。
`decision.production_thresholds_fixed: false`に固定しているため、通常decision modeは失敗し、
実データでは次のcandidate-only調査だけを許可する。これにより、暫定値がteacher v3へ誤って
採用されることを防ぐ。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

RUN_ROOT=/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/global_local_l75_w31_c12_area15_bboxrank_v2_nobbox_bg

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  pseudo3d/analysis/prototype_stage4_contour_auto_refine.py \
  --manifest /mnt/data/3d_projects/pseudo3d_dataset/stage4_sampling_parameter_sweep/260711/manifests/train_manifest.csv \
  --annotated_root "${RUN_ROOT}/annotated" \
  --phase1_audit_root "${RUN_ROOT}/contour_teacher_audit_phase1" \
  --teacher_config pseudo3d/analysis/configs/stage4_bbox_ranked_teacher_v2.yaml \
  --refine_config pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3.yaml \
  --output_root "${RUN_ROOT}/contour_auto_refine_phase3_screening" \
  --top_per_category 20 \
  --candidate_generation_only
```

#### Phase 3 Synthetic check結果（2026-08-19）

`dualtrack311`環境で次の全項目が成功した。

- good controlの`auto_accept`とmask維持
- overfilled baselineから安定した追加候補への`auto_refine`
- tiny componentのreject
- score同点・異mask候補の`manual_review`
- mask単位の候補重複排除、support/stability集約、決定的ranking
- Phase 1 manifest/audit/teacher contractとのend-to-end連携
- 入力H5/XMLの不変性
- 独立した2回の実行におけるCSV、mask、overlay、checksumの決定性
- invalid configの拒否
- project teacher config名、任意fingerprint、candidate-only safety gate

結果は次のとおりである。

```text
[OK] good-control auto-accept and stable overfilled auto-refine
[OK] tiny-component rejection and equal-score ambiguity isolation
[OK] mask-level candidate deduplication and deterministic ranking
[OK] end-to-end Phase 1 contract, read-only inputs, and deterministic outputs
[OK] strict Phase 3 config validation
[OK] project teacher fingerprint and candidate-only safety gate
Stage 4 contour-teacher Phase 3 synthetic checks passed.
phase3_synthetic_exit_code=0
```

Synthetic checkでは2 BBoxから各run 10候補を生成し、2回ともfailureなし、入力不変で完了した。
これによりPhase 3の実装基盤は確認済みとする。次はPhase 1全監査出力を用いた
candidate-only実データ調査を行い、その分布とoverlayからproduction decision閾値を固定する。

#### Phase 1/3実データscreening結果（2026-08-23）

Phase 1本監査は182/182 videos、3,072 BBox、failure 0、入力不変で完了した。

```text
videos_checked         : 182/182
bbox_rows              : 3072
valid_contours         : 3046
invalid_contours       : 26
failure_rows           : 0
input_files_unchanged  : true
phase1_realdata_audit_exit_code=0
```

Phase 1カテゴリ上位から重複を除いた60 videos、129 BBoxに対しcandidate-only screeningを実行し、
9,698候補をfailureなしで生成した。

```text
videos_checked         : 60/60
selected_bboxes        : 129
processed_bboxes       : 129
candidate_rows         : 9698
failure_rows           : 0
input_files_unchanged  : true
phase3_candidate_screening_exit_code=0
```

129 BBox中98件がtriggerされ、108件でeligible proposalが得られ、21件はeligible候補なしだった。
triggered proposalの代表分布は次のとおりである。

| metric | p05 | p50 | p95 |
|---|---:|---:|---:|
| area ratio | 0.0545 | 0.1493 | 0.2749 |
| center distance norm | 0.0137 | 0.0552 | 0.2266 |
| border contact ratio | 0.0000 | 0.0000 | 0.2372 |
| stability IoU | 0.7160 | 0.8333 | 1.0000 |
| score gain | 0.0309 | 0.2215 | 0.4150 |
| score margin | 0.0001 | 0.0064 | 0.0419 |

decision gateをCSV上で比較した結果は次のとおりである。

| 条件 | auto accept | auto refine | manual review |
|---|---:|---:|---:|
| current conservative | 31 | 9 | 89 |
| balanced | 31 | 44 | 54 |
| retention | 31 | 59 | 39 |

目視前にnear-tie候補を過剰採用しないこと、候補なし21件を確実にmanual reviewへ送ること、
maskのparameter重複数よりstabilityを優先することから`balanced`をproduction候補として採用した。
screening configは安全ゲートとして変更せず、次を追加した。

```text
pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3_production_v1.yaml
```

固定したdecision値は次のとおりである。

```yaml
decision:
  production_thresholds_fixed: true
  minimum_refine_score_gain: 0.05
  minimum_score_margin: 0.005
  minimum_stability_iou: 0.70
  minimum_support_count: 1
  equal_score_tolerance: 1.0e-12
```

trigger、candidate generation、eligibility、rankingはscreening configと同一である。次のcommandで
同じ代表129 BBoxをdecision mode実行し、想定件数とoverlayを確認する。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

RUN_ROOT=/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/global_local_l75_w31_c12_area15_bboxrank_v2_nobbox_bg

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  pseudo3d/analysis/prototype_stage4_contour_auto_refine.py \
  --manifest /mnt/data/3d_projects/pseudo3d_dataset/stage4_sampling_parameter_sweep/260711/manifests/train_manifest.csv \
  --annotated_root "${RUN_ROOT}/annotated" \
  --phase1_audit_root "${RUN_ROOT}/contour_teacher_audit_phase1" \
  --teacher_config pseudo3d/analysis/configs/stage4_bbox_ranked_teacher_v2.yaml \
  --refine_config pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3_production_v1.yaml \
  --output_root "${RUN_ROOT}/contour_auto_refine_phase3_production_v1_decisions" \
  --expected_videos 182 \
  --top_per_category 20

echo "phase3_production_decision_exit_code=$?"
```

production config追加後のSynthetic checkは`dualtrack311`環境で成功した。

```text
[OK] end-to-end Phase 1 contract, read-only inputs, and deterministic outputs
[OK] strict Phase 3 config validation
[OK] project screening safety gate and balanced production config
Stage 4 contour-teacher Phase 3 synthetic checks passed.
phase3_updated_synthetic_exit_code=0
```

同じ代表129 BBoxをproduction configでdecision mode実行した実測結果は次のとおりである。

```text
videos_checked         : 60/60
selected_bboxes        : 129
processed_bboxes       : 129
candidate_rows         : 9698
decision_counts        : {'auto_accept': 31, 'auto_refine': 35, 'manual_review': 63}
failure_rows           : 0
input_files_unchanged  : true
phase3_production_decision_exit_code=0
```

CSV上の事前simulationは`31 / 44 / 54`だったが、production実runは`31 / 35 / 63`となった。
これは`minimum_support_count`が最終decision gateだけでなくcandidateのsupport scoreにも使われ、
`2`から`1`への変更によってranking、selected proposal、score marginが再計算されたためである。
したがって、Phase 4へ渡す正式なproduction decisionはend-to-end実runの`31 / 35 / 63`とする。
この差をCLI不整合とは扱わず、以後の集計と代表例選択には実run artifactを使用する。

### Phase 4: review exportとmanual import

`manual_review` frameをimages、overlays、初期mask、CVAT ZIP、manifestとして一括出力する。
CVATからexportした修正済みmaskを別runのH5 annotationへ反映し、point label、valid mask、
metadataの整合性を検査する。自動maskとmanual maskの上書き規則を明文化する。

#### Phase 4の目的

Phase 3で自動判定できなかったBBoxを、localhost上のCVATへ安全かつ再現可能な形式で渡し、
人力修正済みsemantic maskをStage 4のpoint labelへ戻す最小round tripを確立する。

Phase 4は次の3工程を明示的に分離する。

1. `review export`: Stage 4からlocal artifactを生成する
2. `local CVAT review`: 利用者がlocalhost CVATへ手動importし、Brush/Eraserで修正してexportする
3. `manual import`: CVAT export ZIPをstrict validationし、新しいannotated H5へ反映する

Stage 4 codeからCVAT API、外部network、cloud storageへ接続しない。CVAT Taskの作成、annotation
upload、backup、修正、exportは利用者が明示的に行う。

#### Phase 4開始条件

実装・実データround tripの前に次を満たす。

- Phase 2 Synthetic checkが成功している
- Phase 2 local CVAT golden import/no-edit export/edit exportが成功している
- golden結果に基づき`SegmentationClass`と`SegmentationObject`のcontractが固定されている
- Phase 3 Synthetic checkが成功している
- Phase 3 production configの`production_thresholds_fixed`が`true`である
- Phase 3 representative decision runがfailureなしで完了している
- `manual_review`対象とreason codeがBBox単位で確定している

Phase 2 goldenでCVAT exportのobject maskがPhase 2のsingle-object仮定と異なる場合は、Phase 4を
実装する前にconverter/validator contractを更新する。CVATの実挙動をimport側で推測して
暗黙に許容しない。

#### Phase 4の実装範囲

Phase 4に含める。

- representative Phase 3 runからの`manual_review` frame選択
- CVAT登録用の無描画画像、参照overlay、初期binary maskの生成
- Phase 2 converterを利用した`Segmentation Mask 1.1` ZIP生成
- frame/BBox/sourceへ逆引きできるreview manifest
- CVAT export ZIPのread-only validation
- semantic correction maskとBBox/manifestのstrict matching
- source annotated H5を複製した別runへのpoint label反映
- point配列、frame alignment、sampling schema、measurementの保存検査
- manual provenance、review/export/import checksumの保存
- batch export/import summaryと決定性検査

Phase 4には含めない。

- CVAT APIによるTask作成、upload、download
- hosted CVAT、外部SaaS、外部AI/ML backendの利用
- source H5/XML/Phase 3 artifactの上書き
- BBox union外maskの暗黙clip
- missing/extra frame、未知label、shape mismatchの自動補正
- 全182件のteacher v3生成
- instance-aware Stage 5学習
- Stage 5学習または評価

Phase 4はrepresentative manual round tripのcontract確定までとし、全件生成はPhase 5で行う。

#### Phase 4実装予定ファイル

```text
pseudo3d/annotation/stage4_manual_review.py
pseudo3d/batch/export/batch_export_stage4_manual_review_cvat.py
pseudo3d/annotation/import_cvat_segmentation_mask_corrections.py
pseudo3d/batch/annotation/batch_import_cvat_segmentation_mask_corrections.py
pseudo3d/analysis/configs/stage4_manual_review_cvat_v1.yaml
checks/stage4/check_stage4_cvat_manual_roundtrip.py
```

再利用・必要最小限の編集対象は次のとおりである。

```text
pseudo3d/export/convert_masks_to_cvat_segmentation_mask_1_1.py
pseudo3d/annotation/annotate_pseudo3d_point_cloud.py
pseudo3d/batch/export/batch_export_annotation_mask_visualization.py
```

frame/BBox manifest、mask合成、mask-to-point label変換、provenance検証は
`stage4_manual_review.py`のpure APIへ置き、single/batch CLIとSynthetic checkで共有する。
Phase 2 ZIP validatorとPhase 3 decision schemaを再実装しない。

#### Phase 4A: review export入力契約

review exporterは次を入力とする。

- Phase 3で使用したtrain manifest
- source pseudo3D H5
- source `bboxrank_v2_nobbox_bg` annotated H5
- Phase 1 audit root
- production閾値を固定したPhase 3 config
- Phase 3 decision runの`bbox_decisions.csv`、`candidate_metrics.csv`、proposal masks
- Phase 3 `run_config.yaml`、`refine_summary.json`、`checksums.csv`
- Stage 4 manual-review config

実行前に次をstrict validationする。

- Phase 3がcandidate-onlyではなくdecision modeで完了している
- Phase 3 `failure_rows=0`かつ全selected BBoxが処理済みである
- manifest、teacher、refine config、Phase 1 auditのpath/checksum/fingerprintが一致する
- Phase 3 checksum manifestと全proposal maskが一致する
- `manual_review` rowにreason code、proposal mask、元BBoxが存在する
- BBox keyとframe stemが一意である
- source annotated H5がteacher v2 schemaおよび固定foreground samplingを持つ
- source point labelが`-1/0/1`、`valid_mask == (point_label != -1)`を満たす

Phase 3 representative runが一部BBoxだけを評価した場合、review frame内でPhase 3 rowを持たない
他BBoxは現行v2 baselineを再計算して初期maskへ含める。これにより、annotation uploadがTask内
annotationを置換しても、同じframeの非review femur領域が消えない。

#### review frameの選択

1個以上のBBoxが`manual_review`となったframeを1 review imageとする。同一frameに複数の
`manual_review` BBoxがあっても画像とCVAT class maskは1枚だけ生成する。

frame stemを次に固定する。

```text
{video_name}__fo{frame_order:05d}__fi{frame_index:08d}
```

- stemは全review package内で一意である
- image、mask、overlay、CVAT ZIP、manifestで同じstemを使う
- 拡張子はimage/maskとも`.png`に固定する
- source imageをリサイズ、crop、補間しない
- local encoder imageを既存の正規化関数で`uint8` grayscaleへ変換する
- source imageは変更せず、CVAT Task用の複製画像だけへstrict BBox境界を描く
- `manual_review` BBoxは黄、同一frameの他BBoxはシアンの1 pixel線とし、文字やmaskは描かない

#### 初期frame maskの合成規則

review frame内の全strict XML BBoxを処理し、初期semantic maskをunionする。

| BBox状態 | 初期maskへ含める領域 |
|---|---|
| `auto_accept` | v2 baseline mask |
| `auto_refine` | Phase 3 selected proposal mask |
| `manual_review` | Phase 3 best-effort proposal mask |
| Phase 3対象外 | 再計算したv2 baseline mask |

`manual_review` proposalは修正開始点であり確定positiveではない。review前の学習用H5へは反映
しない。全BBox maskをannotation BBoxへclipし、frame maskのBBox union外を0にする。

同一frameに複数BBoxがある場合も、初期versionはStage 5が使用するsemantic `femur` unionを
正解とする。`SegmentationClass`と`SegmentationObject`の具体的なindex表現はPhase 2 goldenで
固定したcontractに従う。instance IDを推測して新たに割り当てない。

#### review artifact出力契約

runごとに独立した次の構造を生成する。

```text
manual_review_cvat/
├── images/
│   └── {frame_stem}.png              # strict BBox描画済みCVAT review画像
├── masks/
│   └── {frame_stem}.png              # binary uint8 0/255
├── overlays/
│   └── {frame_stem}.png
├── cvat/
│   ├── annotations_segmentation_mask_1_1.zip
│   ├── conversion_summary.json
│   └── unpacked_reference/
├── review_frames.csv                 # 1 frame 1 row
├── review_bboxes.csv                 # 1 BBox 1 row
├── review_session_template.yaml
├── checksums.csv
├── export_summary.json
└── run_config.yaml
```

`review_frames.csv`は少なくとも次を持つ。

- frame stem、video、frame order/index、image shape
- image/mask/overlay relative pathとSHA-256
- source pseudo3D/annotated H5 pathとchecksum
- XML root、frame内BBox数、manual-review BBox数
- initial frame mask pixel数
- teacher/refine/manual-review config fingerprint

`review_bboxes.csv`は少なくとも次を持つ。

- frame stem、BBox key、object name/index、XML path/checksum
- XML/raw/local BBox座標
- Phase 3 decisionとreason codes
- baseline/proposal source、score、mask SHA-256
- `manual_review_required`
- 初期frame mask内のBBox positive pixel数

overlayには元画像、全BBox、各decision色、baseline/proposal contour、manual-review reason、score、
area/center/border/stabilityを表示する。overlayは判断支援用でCVAT Task/annotation ZIPへ含めない。

#### 0件時の動作

`manual_review`が0件の場合は正常終了し、summaryへ0件を記録する。ただし、空のimages/masks、
空のCVAT annotation ZIP、空の`default.txt`は生成しない。`status=no_manual_review`として
Phase 5へ進めることを明示する。

#### review exportの決定性と安全性

- image/mask/frame/BBoxを決定的にsortする
- PNG encoding、ZIP timestamp/permission/member順をPhase 2 contractへ従わせる
- 同一入力からbyte-identical mask、CVAT ZIP、CSV、checksumを生成する
- output rootはsource H5、Phase 1/3 artifact、CVAT backupと分離する
- 既存outputは`--overwrite`なしで変更しない
- converter失敗時に不完全な最終ZIPを残さない
- source H5/XML/Phase 3 artifactのsize、mtime、checksumを前後比較する

#### review export CLI案

```bash
cd /mnt/data/3d_projects/models/Stage2to4

RUN_ROOT=/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/global_local_l75_w31_c12_area15_bboxrank_v2_nobbox_bg

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  pseudo3d/batch/export/batch_export_stage4_manual_review_cvat.py \
  --manifest /mnt/data/3d_projects/pseudo3d_dataset/stage4_sampling_parameter_sweep/260711/manifests/train_manifest.csv \
  --annotated_root "${RUN_ROOT}/annotated" \
  --phase1_audit_root "${RUN_ROOT}/contour_teacher_audit_phase1" \
  --phase3_root "${RUN_ROOT}/contour_auto_refine_phase3_production_v1_decisions" \
  --teacher_config pseudo3d/analysis/configs/stage4_bbox_ranked_teacher_v2.yaml \
  --refine_config pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3_production_v1.yaml \
  --manual_review_config pseudo3d/analysis/configs/stage4_manual_review_cvat_v1.yaml \
  --output_root "${RUN_ROOT}/manual_review_cvat_phase4"
```

#### Phase 4B: localhost CVATでの操作

review export完了後、利用者が次を行う。

1. local CVAT version、Task ID/name、利用者識別子を記録する
2. Task labelを`femur`として作成する
3. `images/*.png`だけをTask画像として登録する
4. 既存annotationがあれば、import前に同形式でローカルbackupする
5. `Segmentation Mask 1.1`、`Convert masks to polygons=OFF`でZIPをimportする
6. stem、frame対応、向き、mask位置、labelを確認する
7. overlayを参照し、全`manual_review` BBoxをBrush/Eraserで修正する
8. 同一frameの非review BBox領域を誤って消していないことを確認する
9. 保存後、`Segmentation Mask 1.1`でローカルexportする
10. export ZIPを元artifactと別directoryへ保存する
11. `review_session_template.yaml`をコピーし、Task/reviewer/CVAT version/backup/export情報を記入する

`review_session.yaml`には少なくとも次を記録する。

```text
review_schema_version
review_package_sha256
cvat_version
cvat_task_identifier
reviewer_identifier
imported_zip_sha256
pre_import_backup_sha256
exported_zip_sha256
review_started_utc
review_completed_utc
all_manual_bboxes_reviewed
notes
```

実名を不要とする運用では、ローカル管理されたpseudonymous reviewer IDを使用する。外部service
のuser ID、URL、token、credentialをmanifestやH5へ保存しない。

#### Phase 4C: manual import入力契約

manual importerは次を入力とする。

- review export root
- CVATからexportした`Segmentation Mask 1.1` ZIP
- 記入済み`review_session.yaml`
- source annotated H5 root
- 新規output root

import前にPhase 2 validatorを使い、次をstrict validationする。

- ZIP pathが安全でduplicate/unknown memberがない
- labelmapが`background=0`、`femur=1` contractと一致する
- default stem、class/object mask、review frame stem集合が完全一致する
- maskが2-D `uint8`でindexが`{0,1}`の部分集合である
- mask shapeがreview imageおよびmanifestと一致する
- review package、CVAT import ZIP、CVAT export ZIPのchecksum chainが一致する
- `review_session.yaml`が完了状態で全manual BBox確認済みである
- source H5/XML/config checksumがreview export時から変化していない
- unknown/missing/extra frameがない

edited exportは初期maskと異なるため、Phase 2 validatorのreference pixel equalityは使用しない。
代わりに初期maskとのadded/removed pixel数とIoUを保存する。

#### correction maskのBBox制約

各frame maskのnonzero pixelはstrict XML BBox union内だけを許可する。union外pixelが1点でも
存在した場合はimportを失敗させ、暗黙にclipしない。

さらに各`manual_review` BBoxについて、corrected maskとのintersectionが1 pixel以上あることを
初期contractとする。修正の結果「対象なし」と確定する運用が必要になった場合は、empty maskを
暗黙に意味付けせず、review sessionへBBox単位の明示的な`confirmed_absent`状態を追加してから
contract versionを更新する。

同一frameの複数BBoxが重なる場合、Stage 5用semantic labelはcorrected frame maskのunionを使用
する。BBox別metadataは`corrected mask AND BBox`で計測する。重複pixelをinstance IDへ推測分割
しない。

#### point labelへの反映規則

review済みframeではpoint labelを次の順序で再構成する。

1. frame内pointを初期値background `0`とする
2. strict XML BBox union内pointをignore `-1`とする
3. corrected semantic mask内pointをpositive `1`とする

したがって固定policyは次のまま維持される。

- BBoxなしframe: background
- BBox外: background
- BBox内かつcorrected mask外: ignore
- corrected mask内: positive
- `valid_mask == (point_label != -1)`

review対象外frameはsource H5のlabelをpixel単位で保持する。Phase 4 representative prototypeでは
未選択BBoxをteacher v3へ暗黙更新しない。

#### 新規H5の保存規則

source H5を直接変更せず、別run名へ保存する。

```text
{video_name}_pointcloud_annotated_bboxrank_v2_phase4_manual_v1.h5
```

次をbyte-levelまたはarray-levelでsourceと一致させる。

- `point_cloud`以下の全dataset、dtype、shape、順序、compression
- points、intensity、alpha、confidence
- frame order/index、pixel coordinates
- source type/flags、sampling confidence
- frame-level sampling statistics
- measurement group
- foreground/sampling metadata

変更可能なのは次だけとする。

- `annotation/point_label`
- `annotation/valid_mask`
- review frameに対応する`frame_annotation` annotation metric
- manual review provenance metadata

追加metadata案:

```text
manual_annotation_provenance: manual_cvat_segmentation_mask_1_1_v1
manual_review_schema_version
manual_review_package_sha256
manual_review_session_sha256
cvat_export_sha256
cvat_task_identifier
reviewer_identifier
manual_reviewed_frame_count
manual_reviewed_bbox_count
manual_added_positive_pixels
manual_removed_positive_pixels
source_annotated_h5_sha256
teacher_fingerprint
refine_fingerprint
```

reviewed BBox rowにはdecisionを`manual_review_completed`、selected sourceを
`manual_cvat_segmentation_mask_1_1_v1`として保存し、初期proposal source/reasonは別fieldへ保持
する。自動teacher由来とmanual mask由来を同じsource名で上書きしない。

#### idempotenceと再適用

同じsource H5、review package、CVAT export ZIP、review sessionからはbyte-equivalentなannotation
arrayとmetadataを生成する。同一outputへの暗黙再適用は拒否し、`--overwrite`付き再生成でも
point labelが変化しないことを検査する。

すでにmanual provenanceを持つH5を入力にする場合は、source package/export SHAが完全一致する
idempotence check以外を拒否する。異なる修正版を適用する場合は元source H5から新しいrunを
作り、revision IDを更新する。

#### manual import CLI案

```bash
cd /mnt/data/3d_projects/models/Stage2to4

RUN_ROOT=/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/global_local_l75_w31_c12_area15_bboxrank_v2_nobbox_bg

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  pseudo3d/batch/annotation/batch_import_cvat_segmentation_mask_corrections.py \
  --review_root "${RUN_ROOT}/manual_review_cvat_phase4" \
  --cvat_export_zip "${RUN_ROOT}/manual_review_cvat_phase4/cvat_exports/reviewed_segmentation_mask_1_1.zip" \
  --review_session "${RUN_ROOT}/manual_review_cvat_phase4/cvat_exports/review_session.yaml" \
  --source_annotated_root "${RUN_ROOT}/annotated" \
  --manual_review_config pseudo3d/analysis/configs/stage4_manual_review_cvat_v1.yaml \
  --output_root "${RUN_ROOT}/annotated_phase4_manual_v1" \
  --summary_csv "${RUN_ROOT}/annotated_phase4_manual_v1/summary.csv"
```

#### Phase 4 Synthetic check

`check_stage4_cvat_manual_roundtrip.py`で次を確認する。

- manual-review BBoxを含むframeだけが1回ずつexportされる
- unique frame stemとimage/mask/BBox alignmentが正しい
- frame内のauto-accept/refine/未選択v2 BBoxが初期maskへ保持される
- initial maskがBBox union外へ出ない
- Phase 2 converter/validatorを通る決定的CVAT ZIPが生成される
- 0件時に空ZIPを生成しない
- missing/extra/duplicate stem、shape/index/labelmap不一致を拒否する
- unsafe ZIP path、stale manifest、checksum mismatchを拒否する
- BBox union外manual pixelを拒否し、clipしない
- manual-review BBoxのempty correctionを拒否する
- corrected mask内pointだけがpositiveになる
- BBox内mask外がignore、BBox外/BBoxなしがbackgroundになる
- `valid_mask == (point_label != -1)`を維持する
- review対象外frameのlabelを完全保持する
- point-level/frame-level sampling配列とmeasurementを完全保持する
- manual provenanceとframe/BBox metadataが一致する
- 同一入力のexport/import結果が決定的かつidempotentである
- source H5/XML/Phase 1/3 artifactが変更されない
- failure時に不完全な最終H5を残さない

実装後の確認コマンドは次とする。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_cvat_manual_roundtrip.py

echo "phase4_synthetic_exit_code=$?"
```

#### Phase 4 local CVAT round trip確認

最初は2〜3 frame、少なくとも次を含むrepresentative packageで確認する。

- overfilledから`manual_review`になった例
- ambiguousまたはunstable例
- 同一frameにmanual/non-manual BBoxが共存する例
- 可能なら複数BBox frame

確認順序:

1. review packageの画像、mask、overlay、manifestを目視する
2. local CVATへimportし、no-edit exportをvalidatorで確認する
3. Brush/Eraserで追加・削除を各1例以上行う
4. edited exportをvalidatorとmanual importerへ通す
5. corrected mask、point label visualization、annotated PLYを目視する
6. source/manual H5のschema propagationとStage 5 loader互換性を確認する
7. BBox外/background、BBox内ignore、positiveの3値を確認する

目視不一致がある場合はCVAT側で無理に運用回避せず、stem、座標、mask index、frame合成、import
規則のどこに原因があるかをartifactとchecksumから特定する。

#### Phase 4完了条件

- Phase 2 local CVAT golden contractが確定している
- Phase 4 Synthetic checkがexit code 0で通る
- manual-review frameだけの決定的review packageを生成できる
- localhost CVATでmaskのままimport、編集、exportできる
- 修正済みZIPをstrict validationして別run H5へ適用できる
- source point/sampling/measurement schemaを保持できる
- fixed label policyとStage 5 loader contractを満たす
- manual provenanceとchecksum chainから全修正を追跡できる
- source H5/XMLおよび既存runを変更しない
- 画像、mask、manifestが外部serviceへ送信されていない
- Phase 5で全件運用するexport/import APIとschemaが固定される

#### Phase 4実装順序

1. Phase 2 local CVAT golden round trip完了とZIP contract固定
2. manual-review configとframe/BBox manifest schema
3. pureなframe mask合成・stem・checksum API
4. batch review exporterとPhase 2 converter接続
5. review session schemaとmanual ZIP preflight
6. mask-to-point label変換と新規H5 writer
7. batch importer、summary、idempotence
8. Synthetic check
9. 2〜3 frameのlocalhost CVAT round trip
10. visualization、schema propagation、Stage 5 loader確認

#### Phase 4実装状況（2026-08-19）

次のファイルを実装した。

```text
pseudo3d/annotation/stage4_manual_review.py
pseudo3d/batch/export/batch_export_stage4_manual_review_cvat.py
pseudo3d/annotation/import_cvat_segmentation_mask_corrections.py
pseudo3d/batch/annotation/batch_import_cvat_segmentation_mask_corrections.py
pseudo3d/analysis/configs/stage4_manual_review_cvat_v1.yaml
checks/stage4/check_stage4_cvat_manual_roundtrip.py
```

また、Phase 2のstrict ZIP contractをmanual importでも唯一の解釈系として再利用するため、次の
public read APIを追加した。

```text
pseudo3d/export/convert_masks_to_cvat_segmentation_mask_1_1.py
  read_cvat_segmentation_class_masks(...)
```

実装済み内容は次のとおりである。

- Phase 3 production decision artifact、checksum、manifest、Phase 1 fingerprintのstrict照合
- `manual_review` BBoxを含むframeだけのunique stemによるreview export
- 同一frameのauto-accept、auto-refine、manual-review、Phase 3対象外BBoxの初期mask合成
- BBox単位clip後のsemantic union mask、strict BBox描画済みCVAT review画像、参照overlayの生成
- Phase 2 converterによる決定的`Segmentation Mask 1.1` ZIP生成
- frame/BBox manifest、session template、run config、checksum chainの生成
- `manual_review=0`を正常終了し、空CVAT ZIPを生成しない動作
- 記入済みreview sessionとCVAT export ZIPのstrict validation
- missing/extra stem、unsafe ZIP、未知index、shape、labelmap、checksum不一致の拒否
- strict BBox union外pixelとmanual BBox内empty correctionの拒否
- review frameの`background/ignore/positive` label再構成
- review対象外frameとpoint/sampling/measurement schemaの保持
- source H5を変更しない一時ファイル経由のatomicな別run H5生成
- manual provenance、review/package/session/export checksum、reviewer/task情報の保存
- 同じsourceへの異なるmanual revisionの暗黙再適用防止
- repeat importのannotation array・metadata決定性検査

実データreview exportはPhase 3の`production_thresholds_fixed: true`かつdecision mode完了を必須と
する。candidate-only調査には従来のscreening configを使い続け、review exportにはbalanced値を
固定した`stage4_contour_auto_refine_phase3_production_v1.yaml`と、そのconfigで生成したdecision
artifactだけを使用する。これによりscreening artifactを確定annotationとして誤用しない。

Synthetic checkは次で実行する。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_cvat_manual_roundtrip.py

echo "phase4_synthetic_exit_code=$?"
```

#### Phase 4 Synthetic check結果（2026-08-21）

`dualtrack311`環境でSynthetic round tripを実行し、`exit_code=0`で完了した。

確認済み項目は次のとおりである。

- mask-to-point labelの`background/ignore/positive`規則
- strict BBox union外pixelの拒否
- manual-review BBoxのempty correction拒否
- Phase 1 auditとPhase 3 production decision fixtureのend-to-end連携
- manual-review frame、初期mask、overlay、CVAT ZIPの決定性
- `manual_review=0`時に空CVAT ZIPを生成しない動作
- edited CVAT ZIPから別run annotated H5への反映
- source H5のpoint cloud、sampling、measurement schema保持
- point label、valid mask、manual provenanceの整合性
- 同一入力を独立したoutputへimportした場合のannotation arrayとmetadataの決定性
- project Phase 4 configと固定label contract

結果は次のとおりである。

```text
[OK] strict mask-to-point labels, BBox union, and empty correction
[OK] deterministic review frame export and exact CVAT ZIP
[OK] zero manual-review result does not create an empty CVAT ZIP
[OK] edited CVAT ZIP import, H5 schema preservation, labels, and provenance
[OK] deterministic repeated import annotation arrays and metadata
[OK] project Phase 4 config and fixed label contract
Stage 4 CVAT manual round-trip synthetic checks passed.
phase4_synthetic_exit_code=0
```

これによりPhase 4のコード実装とSynthetic contractは確認済みとする。詳細なreason集計とoverlay
全件監査は省略するが、CVAT export/importの実データ契約を確定するため、2〜3 frameの代表round trip
だけは自動refine版teacher v3の生成前に行う。

#### 擬似教師データ生成を優先する短縮方針（2026-08-23）

`stage4_contour_auto_refine_phase3_production_v1.yaml`を現設定で固定し、代表CVAT round trip後に
次の規則で全182件の自動refine版teacher v3を生成する。

- `auto_accept`: 現行v2 maskを使用する
- `auto_refine`: Phase 3 proposalへ置換する
- `manual_review`: 現行v2 maskを維持する
- baseline invalidかつ候補なし: 現行のinvalid/ignoreを維持する

#### 残りのステップ

1. 既存のCVAT修正済み代表ZIPを再編集せず、context-only許可付きvalidatorとmanual importerへ通す。
2. 修正mask、3値label、schema保持を最低限確認し、代表CVAT確認を完了する。
3. 全182件について自動refine版teacher v3を別runへ生成する。`manual_review`は現行v2 mask、
   baseline invalidかつ候補なしは現行invalid/ignoreを維持する。
4. 全件成功、H5 schema、label件数、Stage 5 loader、数例のannotation textureだけを確認して学習へ進む。

全件のCVAT修正は必須とせず、学習結果から改善が必要と判断した難例に限定して後から実施する。

#### CVAT context-only画像契約（2026-08-24）

- CVAT taskには代表選定した全画像を入れ、前後関係や同一task内の文脈を保持する。
- 描画可能なstrict BBoxを持つ画像だけをannotation-requiredとし、黄色BBoxを画像へ焼き込む。
- 描画可能なmanual BBoxがない画像は`context_only_stems.txt`へ記録し、H5更新対象外とする。
- CVAT exportでcontext-only画像のClass/Object PNGが両方省略された場合だけ、空maskとして補完する。
- annotation-required画像のmask欠落、Class/Object片側だけの欠落、未知stemは従来どおり拒否する。
- context-only maskがZIPに含まれる場合は初期maskと同一であることを要求し、編集を反映しない。
- `context_only` CSV列より前に生成した代表packageでは、明示した`context_only_stems.txt`を
  human decisionとして優先する。未知stem、ZIP mask契約、初期mask不変条件は同様に検査する。
- 今回の代表確認は既存修正済みZIPを使用し、追加のCVAT編集は行わない。

代表実データZIPのcontext-only validationは成功した。3画像のうち2画像にmaskがあり、描画可能な
manual BBoxを持たない1画像は`context_only_stems.txt`に基づくmask省略として受理された。

```text
images        : 3
masks         : 2
missing_masks : 1
unique_indices: [0, 1]
zip_sha256    : 2cabad381171eb3da93ecec38ae6992105f677571302696812f9fd42d02411bc
exit_code     : 0
```

これによりCVAT側での追加編集・再exportは行わず、このZIPを代表manual importへ使用する。

代表manual importも同じZIPから成功し、Phase 4を完了とする。

```text
videos                   : 2
reviewed_frames          : 2
reviewed_bboxes          : 2
context_only_frames      : 1
cvat_missing_context_masks: 1
added_positive_pixels    : 0
removed_positive_pixels  : 763
failure_rows             : 0
manual_import_exit_code  : 0
```

修正対象は`1-3_14`と`1-3_15_03`の各1 frame/1 BBoxである。context-onlyの
`20250625_161030_0550__fo00051__fi00000051`はH5更新対象から除外し、元annotationを保持した。
source H5を上書きせず、代表manual出力を別runへ生成できることを確認した。

### Phase 5: 全182件のteacher v3再生成

Phase 5はCVATによる全manual-review対象の確認・修正までを範囲とする。
既存foreground H5を再利用し、自動事前annotation、ケース別CVAT package、
manual反映済みteacherをそれぞれ別runで生成する。

最初に生成するrun名:

```text
global_local_l75_w31_c12_area15_bboxrank_v3_refined_auto_v1
```

CVAT修正を後から反映する場合のrun名:

```text
global_local_l75_w31_c12_area15_bboxrank_v3_refined_manual_v1
```

既存の`bboxrank_v2_nobbox_bg`は上書きしない。

#### Phase 5実装（2026-08-24）

Phase 3 production runの代表129 BBoxは`auto_accept=31`、`auto_refine=35`、
`manual_review=63`で固定されている。Phase 5では35件の`auto_refine`だけをproposal maskへ
置換し、他のdecisionとPhase 3未選択BBoxはv2をpixel単位で維持する。全182 H5は新runへ
出力し、source H5を上書きしない。

追加ファイル:

```text
pseudo3d/batch/annotation/batch_apply_stage4_contour_refinement.py
pseudo3d/pipelines/build_stage4_bbox_ranked_v3_refined_auto.sh
checks/stage4/check_stage4_contour_teacher_phase5_apply.py
```

安全条件:

- Phase 1/3のmanifest、config fingerprint、decision checksumを照合する
- production閾値固定済み・candidate-onlyではないPhase 3 artifactだけを許可する
- 変更frameの全BBoxについてv2 maskを再構築し、source point labelとの完全一致を要求する
- proposal maskのshape、値、semantic SHA、BBox、主要metricを再検証する
- frame内の全BBox maskをunionして`background/ignore/positive`を再構築する
- point cloud、sampling、measurementと変更対象外frame/BBoxを保持する
- H5へdecision、mask SHA、score、source/config fingerprintを保存する
- atomicな別H5出力と`--skip_existing`時のprovenance照合を行う

Synthetic check:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_contour_teacher_phase5_apply.py

echo "phase5_synthetic_exit_code=$?"
```

Synthetic checkは`dualtrack311`環境で成功した。

```text
videos_written_or_verified: 1
refined_bboxes           : 1
refined_frames           : 1
added_positive_points    : 0
removed_positive_points  : 13
failure_rows             : 0
Phase 5 apply passed.
Phase 5 synthetic checks passed.
phase5_synthetic_exit_code=0
```

これによりproposal適用、v2保持、point/measurement schema、provenance、repeat runの決定性を
確認済みとし、全182件buildへ進む。

全件生成、collect、label policy監査、annotation texture生成:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

SKIP_EXISTING=1 \
EXPORT_VISUALIZATIONS=1 \
bash pseudo3d/pipelines/build_stage4_bbox_ranked_v3_refined_auto.sh

echo "phase5_full_build_exit_code=$?"
```

主要出力:

```text
stage4_training_ablation/260711/
└── global_local_l75_w31_c12_area15_bboxrank_v3_refined_auto_v1/
    ├── annotated/
    ├── collected/
    ├── annotation_textures/
    └── logs/
```

Phase 5-Aの自動事前アノテーションbuildは`dualtrack311`環境で全182件完了した。

```text
annotated                    : 182
collected                    : 182
phase5_full_build_exit_code  : 0
```

`annotated`、一時`collected`、目視確認用の`annotation_textures`を、source runを
上書きせず`bboxrank_v3_refined_auto_v1`下に生成済みである。ただしこの
`collected`はmanual review前の中間生成物であり、Phase 5完了品としてStage 5へ渡さない。

#### Phase 5-B: Phase 3全ケースのCVAT確認

Phase 3 production decisionの対象となった129 BBoxを全件対象とする。
`auto_accept=31`、`auto_refine=35`、`manual_review=63`のすべてをCVATで確認可能にし、
自動判定ケースも必要なら修正する。Phase 3監査で未選択の残り2,943 BBoxは
現時点のCVAT対象に含めない。

- 1ケースは1 review frameとし、同一frameの複数BBoxは分割しない
- 出力はvideoごとの独立ディレクトリとCVAT taskに分ける
- 各video配下にケース別のimage、初期mask、overlay、BBox/判定情報を保存する
- 黄BBoxをCVAT入力画像に直接描画し、必要なコンテキストframeも同梱する
- `auto_accept`、`auto_refine`、`manual_review`の元判定を各ケースに明記する
- video単位で確認済み・修正済みを記録し、中断・再開可能にする
- CVAT export ZIP、task backup、review session、SHA-256をvideo単位で保存する

##### Phase 5-B export実装（2026-08-24）

代表試験用の既存export contractを維持しつつ、次を追加した。

- `review_scope=all_phase3`でPhase 3の129 BBoxすべてをreview対象にする
- correction/importの基点H5を`bboxrank_v3_refined_auto_v1`として記録する
- 全体packageに加え、`videos/<video_name>`に独立import可能なCVAT packageを生成する
- `cases/<frame_stem>`にBBox描画済み画像、初期mask、overlay、metadataを保存する
- review対象BBoxは元のPhase 3 decisionにかかわらずCVAT画像上で黄色にする
- `review_index.csv`と`progress.csv`でBBox/video単位の進捗を管理する
- 描画不能なinvalid BBoxも129件のindexに残し、`context_only/not_applicable`として確認対象にする
- expected countを`129/31/35/63`に固定し、古いPhase 3 artifactの混入を拒否する

編集ファイル:

```text
pseudo3d/batch/export/batch_export_stage4_manual_review_cvat.py
pseudo3d/pipelines/export_stage4_phase5_cvat_review_cases.sh
checks/stage4/check_stage4_cvat_manual_roundtrip.py
```

Synthetic check:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_cvat_manual_roundtrip.py

echo "phase5b_export_synthetic_exit_code=$?"
```

全件export:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

OVERWRITE=0 \
bash pseudo3d/pipelines/export_stage4_phase5_cvat_review_cases.sh

echo "phase5b_export_exit_code=$?"
```

出力先:

```text
global_local_l75_w31_c12_area15_bboxrank_v3_refined_auto_v1/
└── manual_review_cvat_phase5_all_phase3_v1/
    ├── review_index.csv
    ├── progress.csv
    ├── partition_summary.json
    └── videos/
        └── <video_name>/
            ├── cases/<frame_stem>/
            ├── images/
            ├── masks/
            ├── overlays/
            └── cvat/annotations_segmentation_mask_1_1.zip
```

##### Phase 5-B CVAT master Task確認（2026-08-25）

CVAT `2.73.1`のTask `5`は`1-3_14`だけの代表Taskではなく、Phase 3全129ケースを
格納したmaster Taskである。したがって、60個のvideo packageはローカルでの整理・
修正結果の再分割に使い、CVAT上に60 Taskを重複作成しない。

- master Task 5をroot `images/`の129画像、root annotation ZIPと照合する
- root画像集合と60 video packageの画像集合が完全一致することを確認する
- `progress.csv`、画像枚数、ZIP SHA-256、Project、`femur` labelを検査する
- master確認modeではread-onlyとし、`--apply`を拒否する
- per-video Task作成機能は将来必要になった場合の任意機能として残す
- 接続先は`localhost`、`127.0.0.1`、`::1`だけを許可する
- MacのFinderが生成する`.DS_Store`/AppleDoubleは画像集合から除外する

master preflight結果:

```text
CVAT version       : 2.73.1
task_id            : 5
task_name          : stage4_phase5__1-3_14
task scope         : all_phase3_master
workspace          : standalone
master frames      : 129
selected BBoxes    : 129
video packages     : 60
preflight          : passed
```

Task名は代表video名を含むが、実体は全129ケースのmaster Taskである。Taskを追加作成せず、
Task 5を全件確認・修正に使用する。

編集前Task backupもMac側で作成・検証済みである。

```text
filename : task_5_all_phase3_pre_review_backup.zip
contents : data/images, data/manifest.jsonl, task.json, annotations.json
zip test : passed
sha256   : cbce5290afa34f968553b85a78646aeb3c082e463199755bea978f82c08720ce
```

追加ファイル:

```text
pseudo3d/batch/export/batch_create_stage4_phase5_cvat_tasks.py
pseudo3d/pipelines/create_stage4_phase5_cvat_tasks.sh
```

#### Phase 5-C: full-video CVAT review

selected-frameだけを格納したTask 5では、crop境界に達するまでの大腿骨断面の移動を
判断しにくいことが分かった。Task 5はbaselineとして保存し、以後はPhase 3対象を含む
60動画について、1動画1 Taskで全local-crop frameを時系列に格納する。

人力確認では、修正対象を`annotate`、十分な良好frameがあるcrop端例を
`exclude_crop_boundary`、crop再生成が必要な例を`needs_recrop`に分類する。
見切れたBBoxを一律にpositive化せず、判断結果をmaskとは別のCSVに保存する。

実装手順、data contract、出力構成、crop metric、CVAT運用、再importの詳細は次へ分離する。

- `docs/stage4/stage4_phase5_fullvideo_cvat_review_implementation.md`

2026-08-28時点でStep 1、Step 2を実装し、合成検査のexit code 0を確認済み。実データを
変更しないpreflight（Step 3）も60動画・3,048 frame・129 targetで成功済み。full-video
package生成・CVAT ZIP検査・source再照合・Mac転送manifest生成（Step 4）も実データで
exit code 0を確認して完了した。Mac上でTask 5と分離した60 standalone Taskを作成し、
1動画smokeから残りへ再開可能に展開するStep 5も実装済みである。次はMacへpackageと
更新コードを転送し、dry-run、1動画smoke、残り59動画の順に実行する。

#### Phase 5-D: manual反映済み全件runの構築

2026-09-03、59 Taskの返却snapshotを取り込む実装を追加した。crop不良として登録済みの
`20250626_090758_8000`を除外し、reviewed 59動画と非review 122動画から181動画を構築する。
CVAT修正は除外後に描画可能な112 target BBoxだけへ限定し、context・target外・crop外BBoxはv3を
保持する。空maskは削除修正として許可する。単一targetへ一意に帰属するmanual maskは弱い
BBoxより優先する。BBox外positiveはCVAT再exportごとに再集計し、provenance付きで保持する。
最終runは
`bboxrank_v4_manual_fullvideo_v1`とし、preflight、annotated、collected、3値label監査、可視化を
`pseudo3d/pipelines/build_stage4_bbox_ranked_v4_manual_fullvideo.sh`から実行する。実データでの
build成功と目視確認をもってPhase 5完了とする。

##### Phase 5-D実データ結果（2026-09-05）

CVATで再修正した59 Taskのsnapshot v2をLinux環境へ返却し、crop不良として登録した
`20250626_090758_8000`を除外した最終buildが成功した。元の182動画のうち181動画を、
手動修正反映済みteacher v4として別runへ生成した。

```text
run name       : global_local_l75_w31_c12_area15_bboxrank_v4_manual_fullvideo_v1
annotated H5   : 181
collected H5   : 181
excluded video : 20250626_090758_8000
build exit code: 0
```

主要出力:

```text
stage4_training_ablation/260711/
└── global_local_l75_w31_c12_area15_bboxrank_v4_manual_fullvideo_v1/
    ├── annotated/
    ├── collected/
    ├── annotation_textures/
    └── logs/
```

`annotated`と`collected`の生成、3値label policy監査、既存可視化のexportまでは完了した。
ただし、現行`annotation_textures/<video>/frames`の赤maskと輪郭線は、保存済みv4
`annotation/point_label`やCVAT manual maskを直接描くものではなく、画像・strict XML BBox・
保存configから`bbox_ranked_global_local`を再実行した自動輪郭である。したがって、この画像は
sampling source、XML BBox、自動輪郭の確認には使えるが、v4手動修正ラベルの最終確認には
使わない。

Phase 5の状態は「teacher v4 build完了、v4ラベル専用の目視検証は未完了」とする。
目視中に見つかった追加相談事項は、内容と対応方針を合意するまで未修正として扱い、現時点の
H5、XML、可視化処理へ変更を加えない。

自動輪郭の目視では、適切に見えるlocal候補を、BBox辺まで膨張したglobal候補が内包して
globalが採用される例が確認された。原因調査、境界接触による最小変更案、既存CVAT maskの
再利用契約は次の独立文書へ分離する。現時点では未実装である。

- `docs/stage4/stage4_bbox_ranked_border_contact_revision_plan.md`

H5再作成の要否を判断する前に、保存済みteacher v4の3値point labelとprovenanceを直接表示する。
既存automatic contour可視化とは分離し、次の計画に従う。

- `docs/stage4/stage4_v4_point_label_visualization_plan.md`

その比較により、full-video CVAT Taskのcontext frameでCVAT mask外の旧automatic positiveが
残ることを確認した。旧Phase 5-Dの「112 target BBoxだけへ部分適用する」契約は廃止し、
snapshot対象動画の全Task frameでCVAT maskをpositiveの唯一の根拠にする。修正仕様と実装手順は
次を正本とする。

- `docs/stage4/stage4_cvat_snapshot_authoritative_label_revision_plan.md`

### Phase 6: Stage 5比較

v4ラベルを直接表示する可視化での目視確認と、保留中の相談事項の扱いを確定するまで開始しない。

同じStage 5設定で少なくとも次を比較する。

1. `bboxrank_v2_nobbox_bg`
2. `bboxrank_v3_refined_auto_v1`
3. manual修正が十分ある場合は`bboxrank_v3_refined_manual_v1`

foreground点、split、seed、モデル、loss設定を固定し、教師変更だけを比較する。

## 11. 実装予定ファイル

### 新規候補

```text
pseudo3d/analysis/audit_stage4_contour_teacher.py
pseudo3d/analysis/prototype_stage4_contour_auto_refine.py
pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3.yaml
pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3_production_v1.yaml
pseudo3d/annotation/contour_teacher_refinement.py
pseudo3d/annotation/stage4_manual_review.py
pseudo3d/export/convert_masks_to_cvat_segmentation_mask_1_1.py
pseudo3d/batch/export/batch_export_stage4_manual_review_cvat.py
pseudo3d/annotation/import_cvat_segmentation_mask_corrections.py
pseudo3d/batch/annotation/batch_import_cvat_segmentation_mask_corrections.py
pseudo3d/analysis/configs/stage4_manual_review_cvat_v1.yaml
pseudo3d/pipelines/build_stage4_bbox_ranked_v3_annotations.sh
checks/stage4/check_stage4_contour_teacher_audit.py
checks/stage4/check_stage4_contour_auto_refine.py
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
../../../Stage5/train_stage5.sh
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
