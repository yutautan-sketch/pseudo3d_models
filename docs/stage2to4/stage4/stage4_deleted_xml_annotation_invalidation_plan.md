# Stage 4 削除済みXMLアノテーション無効化計画

作成日: 2026-09-06  
最終更新日: 2026-09-07  
編集対象ディレクトリ: `/workspace/Stage2to4`  
状態: Step 1〜Step 6完了、teacher v6最終受入済み・学習利用可

関連文書:

- `docs/stage4/stage4_cvat_snapshot_authoritative_label_revision_plan.md`
- `docs/stage4/stage4_phase5_fullvideo_cvat_review_implementation.md`
- `docs/stage4/stage4_bbox_ranked_border_contact_revision_plan.md`

## 1. 背景

teacher v5のCVAT-authoritative構築と受入検査は成功し、返却CVAT mask外に旧automatic positiveが
残る問題は解消した。一方、CVAT review packageとsource H5を作成した後に、誤BBoxとしてVOC XMLを
削除したframeでは、v5可視化にBBoxとpositiveが残っている。

これは次のartifactがXML削除前の状態を保持しているためである。

- source v3 H5の`frame_annotation`に保存されたBBoxと`xml_path`
- text-free full-video review packageのBBox inventory
- CVAT Taskへ投入・修正後に返却されたsemantic mask
- v5 H5のCVAT-authoritative frame labelとprovenance

v5は再現性確保のため構築時に現在のXMLディレクトリを再走査しない。したがって、XMLファイルの
削除だけでは既存artifactのBBoxやCVAT maskは消えない。この挙動自体はv5契約どおりである。

## 2. 決定する優先順位

明示的に「誤アノテーションとして削除したXML」は、返却済みCVAT maskより上位の無効化指示とする。

```text
明示的annotation invalidation manifest
    > reviewed CVAT snapshot
    > automatic teacher
```

ただし、単に保存済み絶対パス上でXMLが見つからないことだけを自動削除条件にはしない。mount変更、
データ移動、ファイル名変更を誤削除と判定する危険があるため、欠落XML監査は候補抽出にだけ使い、
実際の適用対象はversion管理されたmanifestへ明記する。

## 3. 無効化単位とラベル規則

今回の操作はXMLファイル自体の削除であり、そのXMLが表していたframe全体を無効化単位とする。
該当frameでは有効な大腿骨BBoxが存在しないものとして扱う。

```text
point_label = 0 (background)  # 該当frameの全point
valid_mask = True
保存BBox row = 0
最終positive = 0
最終ignore = 0
```

元CVAT maskは監査用checksum/provenanceとして保持できるが、最終point labelの生成および通常の
mask overlayには使用しない。これにより、削除済みBBox、赤positive、水色の旧CVAT maskが通常の
最終ラベル可視化へ残らない。

1つのXMLに複数の正しいBBoxと誤BBoxが混在し、誤objectだけを除外したい場合はframe全体を無効化
してはならない。その場合はXMLを削除せずobject-level correctionを別途定義する。今回の
frame-level tombstoneとは分離する。

## 4. 無効化manifest

新規CSVをリポジトリ内のconfigとして管理する。予定する最低限の列は次のとおり。

```text
video_name
frame_order
frame_index
frame_stem
expected_xml_name
expected_saved_bbox_rows
action                 # invalidate_entire_frame
reason_code            # deleted_incorrect_bbox_xml
notes
```

適用前に次を厳密に照合する。

- video/frameがcrop-clean train manifestの有効動画に属する
- frame order/index/stemがpseudo3D H5およびCVAT Task inventoryと一致する
- source v5の`frame_annotation/xml_path` basenameが`expected_xml_name`と一致する
- 対象frameの保存BBox row数が`expected_saved_bbox_rows`と一致する
- 現在の正規VOC XML rootに対象XMLが存在しない
- manifest内に重複frameがない
- manifestにない欠落XMLは勝手に無効化しない

該当video/frameの具体値は実装開始時に監査結果とユーザー確認から固定する。

## 5. versioned出力

成功済みv5を上書きしない。v5 annotated H5を読み取り専用sourceとし、無効化だけを適用した新しい
versioned teacher rootを作る。仮称は次とする。

```text
teacher token: bboxrank_v6_cvat_authoritative_xml_invalidation_v1
run root    : global_local_l75_w31_c12_area15_bboxrank_v6_cvat_authoritative_xml_invalidation_v1
```

CVAT Task、返却snapshot v2、source v3、teacher v4/v5、元XML directoryは変更しない。

## 6. H5補正契約

対象frameだけに次を適用する。

1. `annotation/point_label`を全backgroundへ変更する
2. `annotation/valid_mask`を全trueへ変更する
3. `frame_annotation`の対象frame行を全datasetで同じselectorにより除去する
4. BBox単位`manual_review_fullvideo`の対応行を除去する
5. `manual_review_fullvideo_frames`のframe行は監査証跡として保持する
6. frame authorityを`xml_deletion_manifest`へ変更し、override reasonとmanifest SHAを保存する
7. point cloud、sampling、geometry、他frameのlabelを変更しない
8. root/annotation統計を再計算する

別groupに、除去前BBox、元XML path、元CVAT mask status/SHA、変更前後のlabel数、manifest SHAを
保存し、削除理由を追跡可能にする。

## 7. 実装手順

### Step 1: read-only欠落XML監査

181本のv5 H5について、保存済み`frame_annotation/xml_path`と現在の正規VOC XML rootを照合する。
候補CSVにはvideo、frame、BBox数、v5 positive/ignore数、CVAT mask状態を出す。H5は変更しない。

2026-09-07実装済み。H5保存パスとtrain manifestから再構成したcanonical pathを別々に確認し、
次の状態を区別する。

```text
canonical
canonical_only
saved_only
canonical_saved_identical
canonical_saved_conflict
missing
```

`missing`だけを候補CSVへ出すが、この段階では無効化を適用しない。同一frameにmissing XMLと
現存XMLが混在する場合は`frame_has_mixed_xml_presence=True`として、frame全体削除候補から分離する。
v5 frame provenanceからCVAT mask状態とpositive pixel数を読み、保存済みpoint labelから候補frameの
positive/ignore/background数も記録する。v5 H5、manifest、除外manifest、現存XMLはSHA-256を実行後に
再照合し、H5はread-onlyで開く。

実装入口:

```text
pseudo3d/analysis/audit_stage4_deleted_xml_annotations.py
pseudo3d/pipelines/audit_stage4_v5_deleted_xml_annotations.sh
checks/stage4/check_stage4_deleted_xml_annotation_audit.py
```

Synthetic check:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_deleted_xml_annotation_audit.py

echo "deleted_xml_step1_synthetic_exit_code=$?"
```

実データ監査:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

OVERWRITE=0 \
bash pseudo3d/pipelines/audit_stage4_v5_deleted_xml_annotations.sh

echo "deleted_xml_step1_realdata_exit_code=$?"
```

出力はv5 run root内の`deleted_xml_annotation_audit_step1`へ保存する。

```text
audit_summary.json
xml_inventory.csv
missing_xml_candidates.csv
video_summary.csv
input_checksums.csv
```

2026-09-07、実データ181動画の監査がexit code 0で完了した。

```text
videos                           : 181
bbox_rows                        : 3059
xml_entries                      : 2743
missing_xml_entries              : 7
missing_frames                   : 7
all_xml_missing_frames           : 7
mixed_xml_presence_frames        : 0
all_missing_frame_positive_points: 2124
```

欠落7 XMLはユーザーが意図的に削除した件数（同一ケース6件と別ケース1件）と一致した。7件はすべて
別frameで、そのframe内のXMLがすべて欠落しており、現存XMLとの混在はない。したがって、今回の
7 frameはobject単位の曖昧性なしに`invalidate_entire_frame`へ登録できる。v5では合計2124 positive
pointが残っているため、後続補正で0へ変更する対象となる。

### Step 2: invalidation manifestの固定

監査候補から、ユーザーが意図的に削除したXMLだけをCSVへ登録する。件数、frame stem、XML basename、
保存BBox数をpreflightし、未登録の欠落は警告または失敗として分離する。

2026-09-07実装済み。Step 1の5成果物を証拠として再利用し、次を固定する。

- `missing_xml_candidates.csv`の7行をframe単位manifestへ決定的な順序で変換する
- 7行が`missing`、全XML欠落、混在なし、保存/正規XMLとも不存在であることを確認する
- crop-clean train manifest上で対象動画が有効かつ除外対象外であることを確認する
- Step 1の全入力checksumを再照合し、監査後のH5/XML/manifest変更を拒否する
- 該当動画の現在のv5 H5をread-onlyで開き、frame identity、XML basename、BBox数、label数を再照合する
- 7 frameの重複、復元済みXML、複数XMLが混在するframe、候補CSV改変を拒否する
- 合計2124 positive pointを後続で除去する対象として固定する
- 同一内容の既存manifestは`verified_existing`として受理し、異なる内容は明示的`OVERWRITE=1`なしで拒否する

versioned config:

```text
pseudo3d/analysis/configs/stage4_deleted_xml_invalidations_v1.csv
```

実装入口:

```text
pseudo3d/analysis/build_stage4_deleted_xml_invalidation_manifest.py
pseudo3d/pipelines/build_stage4_deleted_xml_invalidation_manifest.sh
checks/stage4/check_stage4_deleted_xml_invalidation_manifest.py
```

Synthetic check:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_deleted_xml_invalidation_manifest.py

echo "deleted_xml_step2_synthetic_exit_code=$?"
```

実データmanifest固定:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

OVERWRITE=0 \
bash pseudo3d/pipelines/build_stage4_deleted_xml_invalidation_manifest.sh

echo "deleted_xml_step2_realdata_exit_code=$?"
```

出力はversioned configとv5 run root内のStep 2 summaryに分離する。

```text
pseudo3d/analysis/configs/stage4_deleted_xml_invalidations_v1.csv
deleted_xml_annotation_invalidation_step2/manifest_summary.json
```

Step 2ではH5を書き換えない。manifestには`invalidate_entire_frame`と
`deleted_incorrect_bbox_xml`を固定し、summaryにはmanifest SHA-256、semantic fingerprint、Step 1証拠
SHA-256、該当v5 H5 SHA-256を保存する。

2026-09-07、実データmanifest固定がexit code 0で完了した。

```text
invalidations      : 7
affected videos    : 2
positive to remove : 2124
manifest status    : written
summary status     : written
fingerprint        : 723e1818c60baa7d05ea590ed1825131f100bcdac6038f07e968bd80aa4e16c4
```

対象は`20250626_090828_2100`の1 frameと`20250626_094835_7950`の連続6 frameであり、ユーザーが
意図的に削除した内訳と一致する。これにより、後続処理が参照する無効化対象、期待XML basename、
保存BBox数、action、reason codeがversioned configとして固定された。Step 2ではH5を書き換えていない。

### Step 3: Synthetic contract

少なくとも次を固定する。

- CVAT positiveを持つ削除XML frameが全backgroundになる
- BBox/ignoreが残らない
- 非対象frameはv5と完全一致する
- 複数BBox XMLのframe-level削除
- manifest重複、frame不一致、XMLが現存する誤指定を拒否する
- verified resumeと入力checksum不変性

2026-09-07実装済み。Step 4のbatch処理から利用する単一H5補正関数と、合成H5による契約検査を
追加した。この段階では実データ181本のv6構築は行わない。

実装入口:

```text
pseudo3d/annotation/apply_deleted_xml_invalidations.py
checks/stage4/check_stage4_deleted_xml_invalidation_apply.py
```

補正処理は対象frameについて次を行う。

- 全pointをbackgroundへ変更し、`valid_mask=True`とする
- `frame_annotation`の全BBox行を同一selectorで除去する
- BBox単位`manual_review_fullvideo`の対応行も除去する
- 除去した両groupの全datasetを`xml_annotation_invalidation`配下へ退避する
- frame単位`manual_review_fullvideo_frames`は保持し、対象行のauthorityとlabel数を更新する
- 元CVAT mask status/pixel数は監査情報として保持する
- source v5 H5、非対象point label、point cloud、その他groupを変更しない
- 対象外動画もlabel/BBoxを変えずv6 schemaへ伝播し、全manifest fingerprintを各H5へ保存する
- v6 token、source/manifest SHA-256、semantic fingerprint、変更前後のpoint数を保存する

Synthetic check:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_deleted_xml_invalidation_apply.py

echo "deleted_xml_step3_synthetic_exit_code=$?"
```

検査では、複数BBoxを持つ1 XMLのframe-level削除、退避データ、非対象データ不変、決定的な再生成、
verified resumeに加え、BBox数不一致、重複frame、復元済みXML、既存出力provenance破損を拒否する。

### Step 4: versioned H5補正と収集

v5をsourceとしてv6 annotated/collectedを生成する。CVAT snapshotの再importや再編集は行わない。

2026-09-07実装済み。crop-clean train manifest、動画除外manifest、Step 2で固定した無効化manifestと
summaryを入力とし、181動画を1回のversioned buildで処理する。

- 7対象frameにはStep 3のframe tombstoneを適用する
- 残りframeおよび対象外174動画はpoint label/BBoxを変更せずv6へ伝播する
- `20250626_090758_8000`はcrop-clean manifestどおり除外し、出力しない
- Step 2に保存した該当v5 H5 checksum、manifest SHA-256、semantic fingerprintを再照合する
- annotatedは動画別directory、collectedはflat directoryへ保存する
- collected H5は対応するannotated H5とのバイト同一性をSHA-256で検証する
- verified resumeでは既存v6 provenanceとannotated/collected checksumを再検証し、不一致を拒否する
- summaryには181動画、2対象動画、7 frame、除去BBox 7行、除去positive 2124点を記録する

実装入口:

```text
pseudo3d/batch/annotation/batch_apply_stage4_deleted_xml_invalidations.py
pseudo3d/pipelines/build_stage4_bbox_ranked_v6_xml_invalidation.sh
checks/stage4/check_stage4_deleted_xml_invalidation_batch.py
```

Step 3完了実績:

```text
[OK] all-background tombstone, BBox removal/archive, provenance, non-target preservation, and deterministic resume
[OK] unaffected-video v5-to-v6 propagation without label/BBox changes
[OK] BBox mismatch, duplicate frame, and restored XML rejection
[OK] corrupt verified-resume provenance rejection
deleted_xml_step3_synthetic_exit_code=0
```

Step 4 synthetic check:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_deleted_xml_invalidation_batch.py

echo "deleted_xml_step4_synthetic_exit_code=$?"
```

実データv6構築:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

SKIP_EXISTING=1 \
bash pseudo3d/pipelines/build_stage4_bbox_ranked_v6_xml_invalidation.sh

echo "deleted_xml_step4_realdata_exit_code=$?"
```

出力先:

```text
/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/
  global_local_l75_w31_c12_area15_bboxrank_v6_cvat_authoritative_xml_invalidation_v1/
    annotated/
    collected/
    invalidation_summary.csv
    invalidation_summary.json
    logs/01_deleted_xml_invalidation_apply.log
```

Step 4ではv5、CVAT snapshot、XMLを変更せず、可視化もまだ生成しない。実データ構築後のsaved-label
可視化はStep 5でversioned出力として行う。

2026-09-07、実データv6構築がexit code 0で完了した。

```text
videos             : 181
invalidated frames : 7
removed BBoxes     : 7
removed positive   : 2124
annotated files    : 181
collected files    : 181
```

Step 2で固定した期待値とすべて一致し、v5とは別のversioned rootへ保存された。

### Step 5: saved-label可視化

v6 H5の保存済みlabelを描画する。無効化frameでは旧CVAT maskを通常overlayせず、BBox、positive、
ignoreが0であることをframe CSV/summaryへ記録する。その他のreviewed frameは従来どおりCVAT maskを
表示する。

2026-09-07実装済み。既存saved-label rendererをv5/v6両対応にし、v6では
`xml_annotation_invalidation`を先に検証してから描画する。

- 無効化frameは`label_authority=xml_deletion_manifest`であることを要求する
- 保存pointが全background、BBox rowとmanual BBox rowが0であることを描画前に検証する
- 返却CVAT ZIPと保存済みstatus/pixel数は証拠として照合する
- 無効化frameの旧CVAT maskは描画対象から除外し、`suppressed_by_xml_invalidation`と記録する
- 他frameでは従来どおり水色CVAT mask、赤positive、黄ignore、青background、緑BBoxを描画する
- `frame_labels.csv`へauthority、無効化action/reason、mask suppressionを追加する
- 全181動画でmanifest SHA/fingerprintが1種類、無効化7 frame/2動画、suppression 7 frameであることを
  batch側でも検証する
- 自動輪郭やXMLは再計算せず、H5、pseudo3D H5、CVAT ZIPを変更しない

実装入口:

```text
pseudo3d/export/export_stage4_point_label_visualization.py
pseudo3d/batch/export/batch_export_stage4_point_label_visualization.py
pseudo3d/pipelines/export_stage4_v6_xml_invalidation_point_label_visualizations.sh
checks/stage4/check_stage4_deleted_xml_invalidation_visualization.py
```

Synthetic check:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_deleted_xml_invalidation_visualization.py

echo "deleted_xml_step5_synthetic_exit_code=$?"
```

実データ一括可視化:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

SKIP_EXISTING=1 \
bash pseudo3d/pipelines/export_stage4_v6_xml_invalidation_point_label_visualizations.sh

echo "deleted_xml_step5_realdata_exit_code=$?"
```

出力先:

```text
global_local_l75_w31_c12_area15_bboxrank_v6_cvat_authoritative_xml_invalidation_v1/
  annotation_textures_v6_xml_invalidation_labels/
    summary.csv
    <video_name>/summary.json
    <video_name>/frame_labels.csv
    <video_name>/frames/annotation_frame_*.png
```

### Step 6: 受入検査（完了）

2026-09-07、Step 4で構築したteacher v6について、Step 5の保存済みpoint-label可視化を
全動画へ出力し、フレーム画像の目視確認を完了した。描画内容に問題は認められなかった。

最終受入では次を確認した。

- XML削除対象7 frameでは、削除済みBBox、赤positive、黄ignore、水色の旧CVAT maskが残っていない
- 対象frameのpointは全てbackgroundで、BBox rowも除去されている
- 対象外frameでは、CVAT snapshotを唯一のpositive authorityとするv5の修正結果が維持されている
- CVAT修正済みmask外へautomatic positiveが残る問題は再発していない
- XML削除7件は固定invalidation manifestと一致し、別のframeへ波及していない
- crop不良の`20250626_090758_8000`は引き続き除外され、成果物は181動画である
- `annotated`と`collected`は各181 H5で、v5や返却CVAT snapshotを上書きしていない

以上により、XML削除とCVAT修正は両方とも最終教師ラベルへ反映済みと判断する。teacher v6を
今回のStage 5学習に使用する正本として受け入れる。

学習入力:

```text
/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/
  global_local_l75_w31_c12_area15_bboxrank_v6_cvat_authoritative_xml_invalidation_v1/
    collected/
```

用途の区分:

- `collected/`: Stage 5の学習入力
- `annotated/`: 動画別H5の監査・追跡用正本
- `annotation_textures_v6_xml_invalidation_labels/`: 最終目視確認の証拠
- v4/v5成果物: 変更履歴と回帰比較用であり、今回の学習入力には使用しない

## 8. 今回行わないこと

- XMLファイルの復元・編集
- CVAT Taskの再編集またはsnapshotの再export
- foreground sampling、crop、global/local輪郭選択の変更
- 欠落XML候補を確認なしで一括削除扱いにすること
- 成功済みv5成果物の上書き
