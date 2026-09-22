# Stage 4管理チャットへの調査依頼: teacher v6 2動画のcrop/BBox整合性問題

作成日: 2026-09-13
作成元: Stage 5実装チャット（S5-12 GroupNorm固定Label policy ablation）

## 1. 経緯

Stage 5のS5-12（GroupNorm固定でBBox内・positive contour外の点をignoreのままにするRun Aと、
backgroundとして学習に含めるRun Bを比較する診断）の実装中、学習前preflightとして
teacher v6の全181 H5（train 163 + validation 18）を監査した
（`Stage5/checks/real_h5/check_stage5_label_policy_bbox_preflight.py`、S5-12 Step E3）。

このpreflightは「`point_label=-1`（ignore）の点は、保存された`frame_annotation/bbox_local_xyxy`の
いずれかのBBox内に必ず収まる」という契約を、`pixel_xy`の丸め込み+frame単位のBBox union判定
（`Stage2to4/checks/stage4/check_stage4_bbox_ranked_label_policy.py`と同じ幾何ロジックを
Stage 5側の`Stage5/stage5/utils/label_policy.py`へ移植したもの）で検証する。契約が成立しない場合、
該当点をbackgroundへ変換せずfail-fastする設計にしている。

全181ファイル中2ファイルでこの契約が破れ、checkerは設計通り停止した。

## 2. 検出内容（Stage 5側、匿名化済み）

集計は`Stage5/checks/real_h5/check_stage5_label_policy_bbox_preflight.py`の出力
（`label_policy_bbox_preflight_summary.json`、`label_policy_bbox_preflight_per_file.csv`）による。
`video_alias`はStage 5側で採番した匿名IDで、train_files.txt/val_files.txtの並び順に対応する
（`train_XXX`はN番目のtrainファイル、`val_XXX`はN番目のvalファイル、0始まり3桁）。実video IDは
Stage 5側の非共有ファイル`label_policy_bbox_preflight_video_id_map_DO_NOT_SHARE.csv`にのみ存在し、
本文書には含めていない。**Stage 4側で対応する場合、まずこのローカルの対応表からalias→実video ID
への変換が必要**（Stage 5チャット側では実video IDを保持していない）。

### 2.1 全体集計（181ファイル）

| 項目 | 値 |
| --- | ---: |
| 全体ignore点数 | 262,347 |
| stray ignore点数（保存BBoxのどれにも属さないignore点） | 16（0.006%） |
| 該当ファイル数 | 2/181 |

schema属性は全ファイルで想定通り一致した
（`label_mode=bbox_ranked_global_local`、`no_bbox_label=0`、`bbox_inside_non_contour_label=ignore`、
`contour_teacher_schema=bboxrank_v6_cvat_authoritative_xml_invalidation_v1`）。no-BBoxフレーム上の
ignore点は0件（`no_bbox_ignore_count=0`、全181ファイル）で、no-BBoxフレームがbackground限定という
契約自体は保たれている。

### 2.2 該当2ファイルの内訳

| video_alias | split | ignore点数 | BBox内（想定target）点数 | stray点数 | stray率 |
| --- | --- | ---: | ---: | ---: | ---: |
| train_068 | train | 97 | 87 | 10 | 10.3% |
| val_009 | validation | 843 | 837 | 6 | 0.7% |

## 3. Stage 5側での目視調査結果（ユーザーによる可視化画像確認、2026-09-13）

frame単位の詳細（`frame_order`、丸め込み後`pixel_xy`、そのframeの保存BBox座標）を出力する診断
オプション（`--dump_stray_alias`）を追加し、該当2動画のannotation可視化済みframe画像と突き合わせて
確認した。

- **train_068**: 動画の大部分でBBoxがStage 2のcrop範囲外にあった。strayが発生したframeは、
  そのBBoxの一部だけが偶然crop範囲内に入っていたケースだった。
- **val_009**: frame番号が進むにつれてBBoxが画面右端へ移動し、stray発生frameでcrop範囲外へ
  出ていった。該当frameでは、BBoxの一辺または自動annotationの端部と思われる描画が確認された。
  それ以前のframeでは、BBoxは正しくcrop範囲内に捉えられていた。

## 4. 根本原因の仮説（Stage 5側での文書・コード調査）

Stage 4側の関連コード・文書を確認したところ、これは既知の失敗モードと一致する可能性が高い。

### 4.1 既存のcrop可視性監査ツール

`Stage2to4/pseudo3d/batch/export/batch_export_stage4_manual_review_cvat.py`の
`_bbox_crop_metrics()`（クリップ前のBBoxをcrop範囲へ再投影し、`visible_fraction`、
`touches_left/right/top/bottom`、`crop_status`（`fully_visible`/`partially_clipped`/
`fully_outside_crop`）を計算）が既に存在し、
`Stage2to4/pseudo3d/analysis/preflight_stage4_phase5_fullvideo_cvat_review.py`の
`run_preflight`がこれをBBox単位・動画単位で集計して`target_bboxes.csv`、`video_summary.csv`、
`preflight_summary.json`へ出力している。

### 4.2 過去の類似除外事例

`Stage2to4/pseudo3d/analysis/configs/stage4_video_exclusions_v1.csv`に、動画
`20250626_090758_8000`が理由`local_crop_tracking_drift`（`evidence_frames=15-23`、
`scope=stage4_teacher`、`recoverable=true`）で既に除外されている。備考は「frame 14までは
見えているが、以降femurのBBoxがlocal cropの左側へ出ていく」——**train_068・val_009と同じ
パターン**である。この除外は`Stage2to4/pseudo3d/analysis/build_stage4_exclusion_manifest.py`で
適用され、`docs/stage2to4/stage4/stage4_phase5_fullvideo_cvat_review_implementation.md`、
`stage4_deleted_xml_annotation_invalidation_plan.md`、`stage4_contour_teacher_improvement_plan.md`
に記録されている。

### 4.3 crop窓とBBoxの座標系

crop（Stage 2）は`Stage2to4/src/utils/pseudo3d_processing.py`の`crop_frames_with_offset()`
（L450）/`resize_shorter_then_offset_crop()`（L526）で生成され、**動画単位で固定**のcrop offset
（CLI引数、H5属性`local_crop_left`/`local_crop_top`/`local_resize_scale`/`local_crop_clipped`
として保存）を全frameへ一律適用する。一方、被写体（femur）はframeごとに動くため、固定crop窓が
動画後半で被写体から外れていく構造的弱点がある。

`bbox_local_xyxy`は`Stage2to4/pseudo3d/annotation/annotate_pseudo3d_point_cloud.py`の
`xml_bbox_to_local()`（L456）で生成される。元frame（crop前）のXML/VOC BBoxを保存済みH5属性で
local-crop座標へ再投影した後、`[0, local_w-1] x [0, local_h-1]`へ**クリップ**し、
`x2>x1 and y2>y1`のときのみ`valid`とする。**このクリップにより、crop範囲外へ大きく外れたBBoxでも
crop端に張り付いた小さい「有効」BBoxとして保存され、`bbox_local_xyxy`だけを見てもズレの深刻さが
分からない**（`_bbox_crop_metrics()`のようなクリップ前の`visible_fraction`を見て初めて分かる）。

point_labelのignore/positive付与が、保存済み（クリップ済み）`bbox_local_xyxy`ではなく、
クリップ前のBBox・contourを基準にしている場合、この2つの基準のズレがstray ignore点として
現れた可能性がある。

## 5. Stage 4管理チャットへの検証依頼事項

1. **既存crop可視性ツールの実行**: `_bbox_crop_metrics()`
   （`batch_export_stage4_manual_review_cvat.py`）または
   `preflight_stage4_phase5_fullvideo_cvat_review.py`を、train_068・val_009に対応する実video ID
   （Stage 5側の非共有対応表から特定）へ適用し、frame別`visible_fraction`/`crop_status`を取得する。
   Stage 5側でこの幾何を再実装せず、既存の権威あるツールの出力を正本とすることを想定している。
2. **train_068の重篤度を既存除外事例と比較**: `fully_outside_crop`の割合が
   `20250626_090758_8000`（既存の`local_crop_tracking_drift`除外事例）と同等かそれ以上であれば、
   同じ理由コードで`stage4_video_exclusions_v1.csv`へ追加することを検討する。
3. **val_009の遷移frame特定**: `crop_status`が`partially_clipped`から`fully_outside_crop`へ
   変わる正確なframeを特定し、動画全体の除外ではなく、該当frameのみを対象とした
   frame単位の是正（既存の`apply_deleted_xml_invalidations.py`のXML invalidationの枠組みが
   使える可能性がある）で十分かを判断する。
4. **既存Stage 5結果への遡及影響の確認**: train_068はteacher v6の163 train fileの1つとして、
   S5-07（teacher v6移行）以降の全run（S5-07〜S5-11のBatchNorm/GroupNorm比較を含む）に
   既に使用されている。除外を決定した場合、これらの既存結果への影響（train 163→162ファイルへの
   変更、または軽微な性能差の可能性）を記録し、再学習の要否を判断する必要がある。

## 6. Stage 5側の現状の対応方針

- `Stage5/stage5/utils/label_policy.py`のfail-fast（BBox外のignore点が1点でもあれば
  `bbox_noncontour_background`への変換を拒否する）は**緩めない**。今回の事象は座標の丸め誤差
  ではなくラベル品質の実問題である可能性が高く、Stage 5側で機械的にstray点をスキップする
  workaroundは問題を隠蔽しかねないため。
- S5-12のRun A/B学習（Step E5/E6）は、この調査結果と対応方針が決まるまで保留する。
- 上記4項目の検証・対応はStage 4側のtooling・manifestで行うのが保守性の観点から適切と判断し、
  本文書で依頼する。

## 7. 参照

### Stage 5側

- `Stage5/checks/real_h5/check_stage5_label_policy_bbox_preflight.py`（本事象を検出したchecker、
  `--dump_stray_alias`でframe単位detail出力も可能）
- `Stage5/stage5/utils/label_policy.py`（BBox内外判定・label policy適用ロジック）
- `docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.6節 実装事項E
  （Step E3 preflight結果の記録）
- `docs/stage5/stage5_revision_management_record.md` S5-12（進捗・保留理由の記録）

### Stage 4側（本調査で参照したファイル）

- `Stage2to4/pseudo3d/batch/export/batch_export_stage4_manual_review_cvat.py`
  （`_project_bbox_to_local_unclipped()`、`_bbox_crop_metrics()`）
- `Stage2to4/pseudo3d/analysis/preflight_stage4_phase5_fullvideo_cvat_review.py`
  （`run_preflight`、`target_bboxes.csv`/`video_summary.csv`/`preflight_summary.json`出力）
- `Stage2to4/pseudo3d/analysis/configs/stage4_video_exclusions_v1.csv`（既存除外manifest）
- `Stage2to4/pseudo3d/analysis/build_stage4_exclusion_manifest.py`（除外manifest適用スクリプト）
- `Stage2to4/src/utils/pseudo3d_processing.py`（`crop_frames_with_offset()`、
  `resize_shorter_then_offset_crop()`）
- `Stage2to4/pseudo3d/annotation/annotate_pseudo3d_point_cloud.py`（`xml_bbox_to_local()`）
- `Stage2to4/pseudo3d/annotation/apply_deleted_xml_invalidations.py`（frame単位XML invalidation）
- `docs/stage2to4/stage4/stage4_phase5_fullvideo_cvat_review_implementation.md`
- `docs/stage2to4/stage4/stage4_deleted_xml_annotation_invalidation_plan.md`
- `docs/stage2to4/stage4/stage4_contour_teacher_improvement_plan.md`
