# Stage 4 CVAT snapshot authoritative label 改修計画

作成日: 2026-09-05  
最終更新日: 2026-09-07  
編集対象ディレクトリ: `/workspace/Stage2to4`  
状態: Step 1〜6完了、v5検証済み・後続teacher v6を最終学習データとして受入済み

## 1. 決定事項

full-video CVAT snapshotを持つ動画では、Taskに含まれる全frameについて、CVATから返却された
semantic segmentation maskを最終positiveの唯一の根拠とする。

```text
final positive = sampled pointのpixel座標がCVAT femur mask内にある
```

次は禁止する。

- CVAT maskと既存automatic positiveのunion
- CVAT mask外にあるglobal/local/Phase 3 positiveの維持
- `review_target`だけをCVAT適用対象とし、`context_only`の編集を無視すること
- BBoxでCVAT maskをclipしてからpositiveを決めること
- 空のCVAT maskを理由にautomatic positiveへfallbackすること

automatic maskはCVAT編集開始時のseed、比較用provenance、非snapshot動画のteacherとしては利用できる。
ただし、snapshot対象frameの最終positive計算には利用しない。

## 2. 方針変更の根拠

teacher v4の保存済みpoint labelと返却CVAT maskを比較したところ、次の不整合を確認した。

```text
video                     : 1-3_14
frame_order               : 10
frame role                : context
CVAT mask pixels          : 198
final positive points     : 343
positive inside CVAT      : 158
positive outside CVAT     : 185
saved contour source      : global
manual applied frame      : false
```

現行importerはPhase 3のactionable BBoxを持つframeだけを更新し、full-video Task上で人が追加修正した
context frameをv3から継承していた。これは描画残りではなく、H5に旧automatic positiveが保存された
適用範囲の問題である。

## 3. 適用範囲

### 3.1 snapshot対象動画

返却snapshotに含まれる59動画では、各CVAT Taskの画像inventoryに含まれる全frameを適用対象とする。
`review_target`、`context_only`、Phase 3 decision、BBoxの有無でpositive authorityを切り替えない。

- maskが非空なら、そのmask内のsampled pointだけをpositiveにする
- maskが空なら、そのframeのfinal positiveを0にする
- CVAT exportでall-background mask PNGが省略された場合も、Task image stemとの照合に成功した
  正規のmissing maskだけをempty maskとして扱う
- unknown stem、extra mask、shape不一致、Task image欠落は失敗とし、emptyへ暗黙補完しない

### 3.2 snapshot非対象動画

現在の返却snapshotは59動画だけを覆い、crop不良の1動画は除外されている。残り122動画には
CVAT snapshotがないため、本契約だけからCVAT-authoritative labelを生成することはできない。

181動画をまとめる場合は、次をmetadataとmanifestで明確に区別する。

```text
snapshot-covered 59 videos : CVAT mask authoritative
non-snapshot 122 videos     : automatic teacher
excluded 1 video            : dataset対象外
```

「全181動画のpositiveがCVAT由来」と表現してはならない。dataset全体をCVAT-onlyにする場合は、
残る122動画にもfull-video CVAT snapshotを作成する必要がある。本改修では新しいannotationを
捏造せず、59動画の適用不整合だけを修正する。

## 4. point label契約

frameのCVAT femur maskを`M`、保存済みstrict XML BBoxのframe unionを`B`、sampled pointを`p`とする。
`pixel_xy`は既存処理と同じくround-to-nearestで整数pixelへ対応させる。

```text
M[p] == 1                    -> point_label =  1 (positive)
M[p] == 0 and B[p] == 1      -> point_label = -1 (ignore)
M[p] == 0 and B[p] == 0      -> point_label =  0 (background)
valid_mask                   -> point_label != -1
```

重要な帰結:

- `point_label == 1`なら必ず`M[p] == 1`である
- 旧automatic positiveはframeごと一度破棄し、上式からlabelを再構築する
- CVAT maskがBBox外へ延びても、その部分は人が確定したpositiveとして保持する
- BBoxはCVAT positiveのclip条件ではなく、mask外pointのignore/background区分にだけ使う
- BBoxなしframeでもCVAT mask内のpointはpositiveにできる
- semantic maskはframe unionであり、複数BBoxをinstance別には分解しない
- empty maskではBBox内をignore、BBox外をbackgroundとし、positiveを残さない

## 5. 入力・provenance契約

既存artifactは上書きせず、次を読み取り専用入力とする。

- text-free full-video review package
- 59 Taskの返却snapshot v2
- snapshot `export_manifest.csv`
- annotation ZIPとpost-review Task backup
- source v3 annotated H5
- strict XML BBoxとpseudo3D local image geometry
- crop除外manifest

H5作成前に次を照合する。

- snapshot manifest SHA-256
- video name、Task ID、Task image数、frame stem、frame order/index
- annotation ZIP SHA-256とTask backup SHA-256
- image/mask shapeとCVAT label/index contract
- source H5のpoint arrays、`pixel_xy`、frame alignment、BBox geometry
- exclusion manifestと対象video集合

新H5にはBBox単位の既存provenanceだけでなく、全Task frameを表せるframe単位provenanceを保存する。
少なくとも次を記録する。

```text
frame_stem
frame_order
frame_index
frame_role
cvat_mask_status          # positive / empty / omitted_as_empty
cvat_mask_positive_pixels
cvat_annotation_zip_sha256
cvat_task_id
label_authority           # cvat_snapshot
positive_points
ignore_points
background_points
```

no-BBox context frameも記録できるよう、BBox行だけに依存したschemaにはしない。

## 6. 実装手順

### Step 1: 現行不整合をSynthetic contractへ固定

- actionableではないcontext frameに非空CVAT maskを置く
- source H5にはCVAT mask外のautomatic positiveを置く
- 現行v4相当の不整合を再現し、修正後はmask外positiveが0になることを期待値にする
- context empty、mask省略、BBoxなし、複数BBoxもfixtureへ含める

2026-09-05実装済み。pureな期待値oracleでsource labelに依存しない3値再構築を固定し、現行
end-to-end importerでは非空context maskが無視される不整合を意図的に再現する。Step 1時点では
importer本体を変更しないため、検査成功は「現行動作が正しい」という意味ではない。Step 3で
不整合再現assertionをauthoritative適用成功のassertionへ置き換える。

同日、ユーザー環境でStep 1 Synthetic checkのexit code 0を確認した。

```text
[OK] pre-fix context-mask omission reproduced with exclusion, provenance, and verified resume
Stage 4 CVAT snapshot-authoritative Step 1 synthetic contracts passed
cvat_authoritative_step1_synthetic_exit_code=0
```

主な検査対象:

```text
checks/stage4/check_stage4_phase5_fullvideo_final_import.py
checks/stage4/check_stage4_point_label_visualization.py
```

### Step 2: snapshot readerを全Task frame契約へ変更

- package `review_frames.csv`をTask frame inventoryの正本とする
- annotation ZIPを全stemについて検証・読込する
- target/contextを問わずframeごとのmask状態を返す
- 正規のall-background省略だけを`omitted_as_empty`として明示記録する
- BBox/actionable行がないframeも失わない

2026-09-05実装済み。`read_authoritative_task_frame_masks()`を追加し、`review_frames.csv`の
stem/order/index/role/shapeとCVAT ZIPを照合して、全frameを`positive`、`empty`、
`omitted_as_empty`へ分類する。reader結果はpackage preflightへ接続したが、Step 2ではH5 label
更新loopへはまだ接続しない。Synthetic fixtureには非空context、明示empty target、省略empty、
重複frame、role不整合を追加した。

### Step 3: final importerをframe-authoritativeに変更

- actionable BBoxだけを回す現行loopを、snapshot対象Taskの全frame loopへ変更する
- 各frameの旧labelを使用せず、CVAT maskと保存BBox unionから3値labelを再構築する
- mask外の旧global/local/Phase 3 positiveをすべて除去する
- nonempty context mask、empty context mask、BBox外manual maskを同じ規則で処理する
- snapshot非対象動画のH5は変更しない
- frame単位provenanceと集計値を保存する

主な実装対象:

```text
pseudo3d/batch/annotation/batch_import_stage4_phase5_fullvideo_cvat.py
pseudo3d/pipelines/build_stage4_bbox_ranked_v5_cvat_authoritative.sh
```

既存teacher v4は上書きせず、新しいrun token/rootへ出力する。

2026-09-05実装済み。importerに
`bboxrank_v5_cvat_authoritative_v1`モードを追加し、snapshot-covered動画では
`review_frames.csv`の全Task frameを順番に処理する。各frameはsource labelを参照せず、CVAT
maskをpositive、保存済み全BBoxのunion内かつmask外をignore、それ以外をbackgroundとして
全面再構築する。したがって、target/context、明示empty、省略empty、no-BBox、BBox外human
positiveを同じ規則で処理し、CVAT mask外の旧global/local/Phase 3 positiveは残らない。

H5にはBBox単位の従来provenanceに加えて`manual_review_fullvideo_frames`を保存する。ここには
frame role、mask状態、CVAT positive pixel数、各3値point数、
`positive_outside_cvat_mask_points`を記録する。snapshot非対象動画はsource labelを保持し、空の
frame provenanceと`inherited` authorityを記録する。

既存v4成果物と再現経路を保護するため、importerの既定値は従来v4互換のままとし、v5専用
pipelineから次を明示する。

```text
--output_teacher_token bboxrank_v5_cvat_authoritative_v1
```

新規構築入口:

```text
pseudo3d/pipelines/build_stage4_bbox_ranked_v5_cvat_authoritative.sh
```

保存済み3値label監査には、CVAT provenanceを持つframeだけno-BBox positiveを許可する
`--allow_cvat_authoritative_no_bbox_positive`を追加した。Step 3 Synthetic checkでは、Step 2まで
意図的に再現していたcontext omissionをauthoritative適用成功へ置き換える。

### Step 4: 可視化を全snapshot frameへ対応

- H5 frame provenanceと同じsnapshot ZIPをSHA照合して読む
- target/contextを問わず、適用したCVAT maskを水色で表示する
- 赤positiveを水色maskより後に描画する
- 集計ではmarker半径ではなくpoint中心を使い、`positive_outside_cvat_mask=0`を必須にする
- frame 10を回帰確認対象として固定する

主な実装対象:

```text
pseudo3d/export/export_stage4_point_label_visualization.py
pseudo3d/batch/export/batch_export_stage4_point_label_visualization.py
pseudo3d/pipelines/export_stage4_v5_cvat_authoritative_point_label_visualizations.sh
```

2026-09-05実装済み。`bboxrank_v5_cvat_authoritative_v1`ではBBox単位provenanceではなく
`manual_review_fullvideo_frames`を正本として、review Taskの全frameに対応するCVAT maskを読む。
H5に保存したframe stem/order/index、mask status、mask pixel数、各3値point数とsnapshot ZIP SHAを
照合し、point中心で次を必須検査する。

```text
saved positive == sampled CVAT mask
positive_outside_cvat_mask_points == 0
cvat_point_mask_mismatch_points == 0
```

描画順はbackground/ignore point、水色CVAT mask、赤positive、保存済みBBoxとする。marker半径は
表示だけに使用し、上記検査には使用しない。v5専用pipelineでは赤markerの見かけ上のmask外膨張を
避けるため半径0とする。frame CSVにはmask状態とframe単位の不整合数、summaryにはauthority、
全Task frame数、mask状態別件数を保存する。

既存v4経路を維持し、新規v5一括入口を追加した。

```text
pseudo3d/pipelines/export_stage4_v5_cvat_authoritative_point_label_visualizations.sh
```

この入口は`--require_cvat_authoritative`を指定し、v4 H5の混入と古い可視化summaryのverified
resumeを拒否する。`VIDEO_NAMES=1-3_14`で回帰対象だけを先行出力できる。

Step 4 Synthetic check:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_point_label_visualization.py
```

### Step 5: Synthetic checkとread-only実データpreflight

Syntheticでは少なくとも次を確認する。

- nonempty target/context maskの完全適用
- empty target/context maskでfinal positiveが0になる
- mask外automatic positiveが残らない
- BBox外CVAT positiveをclipしない
- BBoxなしframeのmanual positive
- 複数BBox frameのsemantic union
- 正規missing-as-emptyと不正missingの分離
- `valid_mask == (point_label != -1)`
- point cloud、sampling、measurement schemaの保存
- input不変性、atomic output、決定性、verified resume

実データpreflightではH5を書かず、59 Task全frameについて次を集計する。

```text
CVAT positive/empty/omitted frame数
CVAT mask positive pixels
旧automatic positive points
旧positiveのCVAT内/外件数
新規positive/削除positiveの見込み
BBoxなしCVAT-positive frame数
```

2026-09-06実装済み。source v3 H5、full-video review package、snapshot manifest、annotation
ZIP、Task backup、task map、除外manifestを既存final importerと同じ契約で検証する。その後、59
動画の全Task frameをpoint中心で比較し、次を`frame_metrics.csv`、`video_summary.csv`、
`preflight_summary.json`へ保存する。

対象frame数は、元package 60動画・3048 frameからcrop除外動画
`20250626_090758_8000`の34 frameを除いた、59動画・3014 frameに固定する。

```text
source positiveのCVAT mask内/外point数
CVAT-authoritative化後のprojected positive point数
retained / added / removed positive point数
positive / empty / omitted_as_empty frame数
review_target / context_only frame数
BBoxなしCVAT-positive frame数とprojected positive point数
```

入力H5は常にread-onlyで開き、source H5、返却ZIP、Task backup、各package contract file等の
SHA-256を実行前後で照合する。生成物は監査reportだけで、`h5_files_written=0`を記録する。

実データでは`review_target=True, context_only=True`となるframeが存在する。これは矛盾ではなく、
Phase 3 targetだがcrop内に描画可能なmanual BBoxがない`unreviewable target`を意味する。Step 5
ではactionable target、通常context、unreviewable targetを分離して集計し、後者も正規の
omitted-as-empty候補として扱う。

実装入口:

```text
pseudo3d/analysis/preflight_stage4_cvat_authoritative_labels.py
pseudo3d/pipelines/preflight_stage4_v5_cvat_authoritative_labels.sh
checks/stage4/check_stage4_cvat_authoritative_preflight.py
```

Step 5 Synthetic check:

適用・schema・決定性は検査済みのStep 3/4 checkで担保し、Step 5固有checkではpoint比較の
partition集計、empty/context見込み、geometry拒否、read-only H5 openを確認する。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_cvat_authoritative_preflight.py
```

2026-09-06、実データpreflightのexit code 0を確認した。出力先はv5 run root内の
`cvat_authoritative_preflight_step5`であり、`preflight_summary.json`、`frame_metrics.csv`、
`video_summary.csv`を生成した。このpreflight成果物をStep 6の構築前提および受入期待値として
再利用する。

### Step 6: versioned rebuildと受入検査

- snapshot v2を再利用し、CVATで再編集しない
- 新しいversioned rootへannotated、collected、label visualizationを生成する
- 59 snapshot動画の全Task frameで`positive_outside_cvat_mask=0`を確認する
- `1-3_14/frame_order=10`で赤pointがCVAT mask外にないことを確認する
- snapshot非対象122動画と除外1動画の扱いをsummaryへ明記する
- source v3、teacher v4、CVAT ZIP、Task backupを変更しない

2026-09-06実装済み。Step 5の`preflight_summary.json`と入力checksum inventoryを構築前提とし、
次の5段階を単一pipelineへ接続した。

```text
1. v5 annotated H5を181動画へ構築
2. collectedへ181 H5を収集
3. 保存済み3値label policyを監査
4. 保存labelとCVAT maskを全動画で可視化
5. Step 5予測値、H5、収集物、可視化を横断して受入検査
```

受入検査では、snapshot対象59動画・全3014 Task frame、snapshot非対象122動画、除外1動画を
明示的に分離する。59動画ではCVAT mask状態別件数とpositive総数がStep 5予測に一致し、
`positive_outside_cvat_mask_points=0`かつ`cvat_point_mask_mismatch_points=0`であることを必須とする。
122動画ではsource v3の`point_label`/`valid_mask`が完全一致すること、collected H5はannotated H5と
バイト一致すること、全181動画の可視化summaryとPNG件数がH5に対応することを確認する。
`1-3_14/frame_order=10`は固定回帰対象とする。

実装入口:

```text
pseudo3d/pipelines/build_stage4_bbox_ranked_v5_cvat_authoritative.sh
checks/stage4/check_stage4_v5_cvat_authoritative_acceptance.py
```

実データ構築コマンド:

```bash
cd /mnt/data/3d_projects/models/Stage2to4

SKIP_EXISTING=1 \
bash pseudo3d/pipelines/build_stage4_bbox_ranked_v5_cvat_authoritative.sh

echo "cvat_authoritative_step6_build_exit_code=$?"
```

既定の`SKIP_EXISTING=1`では、既存H5と可視化をchecksum/provenance付きで検証して再開する。
入力や設定の変更後に同じv5出力rootを作り直す用途では、既存成果物を退避したうえで新しい
versioned rootを指定する。既存v3/v4成果物は上書きしない。

2026-09-06、実データ181動画についてStep 6のexit code 0を確認した。annotated/collectedは
各181 H5、snapshot対象は59動画・3014 Task frame、snapshot非対象は122動画、crop除外は1動画
である。`annotation_textures_v5_cvat_authoritative_labels`の目視確認でも、CVAT mask外に旧automatic
positiveが残る問題は解消した。

この確認後、過去に誤BBoxとしてXML自体を削除したframeでは、v5が保存済みBBoxと返却済みCVAT
maskを再現したため、削除意図が反映されずBBox/positiveが残る別問題を確認した。これは
CVAT-authoritative適用の失敗ではなく、v5の入力artifactがXML削除前に固定されていることによる。
対処は次の独立文書へ分離する。

- `docs/stage4/stage4_deleted_xml_annotation_invalidation_plan.md`

2026-09-07、上記の後続計画はStep 1〜Step 6まで完了した。削除済みXML 7件を明示的に無効化した
teacher v6を181動画で構築し、保存ラベル可視化の目視確認にも問題がないことを確認した。これにより、
CVAT snapshotをpositiveの唯一の根拠とする修正と、後発のXML削除意図が両立した。今回のStage 5学習では
v5ではなく、次のv6 `collected`を使用する。

```text
/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/
  global_local_l75_w31_c12_area15_bboxrank_v6_cvat_authoritative_xml_invalidation_v1/
    collected/
```

## 7. 受入条件

- snapshot対象frameのpositive point中心がCVAT maskの部分集合である
- 非空context maskがH5へ反映される
- empty context maskにautomatic positiveが残らない
- snapshot対象frameのlabel計算がautomatic contour maskを参照しない
- BBox外の人力maskを暗黙clipしない
- frame単位でmask authorityとchecksumを追跡できる
- 59 Task全件でstem、shape、SHA、label contractが通る
- 既存artifactを上書きせず、同一入力から決定的に再構築できる

## 8. 今回行わないこと

- CVAT Taskまたはsnapshotの再編集
- global/local輪郭選択ロジックの変更
- foreground sampling、crop、`pixel_xy`、XML BBoxの変更
- CVAT mask外automatic positiveの救済
- snapshotのない122動画へCVAT labelを推測すること

自動輪郭のBBox境界接触見直しは別課題として維持するが、snapshot対象frameの最終positiveには
影響させない。
