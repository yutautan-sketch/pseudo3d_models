# Stage 5 S5-12向け Stage 4 crop品質修正レポート

作成日: 2026-09-14  
作成元: Stage 2–4実装管理チャット  
引継先: Stage 5実装管理チャット（S5-12 GroupNorm固定Label policy ablation）  
状態: teacher v7構築・受入・可視化・Stage 5入力preflight完了

## 1. 結論

`docs/stage2to4/stage4/stage5_s5_12_stage4_crop_quality_investigation_request.md`で報告された、teacher v6の2動画・5 frame・16点のstray ignore問題をStage 4側で解消した。

- `20250626_090652_6340`は動画全体のlocal crop品質不良として除外した。
- 既存除外動画`20250626_090758_8000`は引き続き除外した。
- `20250625_161030_0550`はframe order/index 51だけをcrop-quality invalidationし、全pointをbackground化してactive BBoxを除去した。
- 退化BBoxを1 pixel線として扱わないよう、BBoxの正面積判定を統一した。
- 補正済みteacher v7を新規rootへ構築し、全180 H5でstray ignore、no-BBox ignore、BBox内background、CVAT mask外positiveがすべて0であることを確認した。
- Stage 5の既定入力をteacher v7へ切り替え、入力preflight-only実行がexit code 0で完了した。

以後のStage 5固有preflight、split/list生成、S5-12 Run A/Bの実行判断はStage 5実装管理チャットへ引き継ぐ。

## 2. 発端と根本原因

Stage 5のteacher v6全181 H5監査で、保存された有効BBox unionの外側に`point_label=-1`が16点見つかった。

| Stage 5 alias | video_name | 問題frame_order | stray ignore |
|---|---|---|---:|
| `train_068` | `20250626_090652_6340` | 38, 43, 44, 46 | 10 |
| `val_009` | `20250625_161030_0550` | 51 | 6 |

read-only監査により、16点はすべて「幅0または高さ0へ退化したcrop後BBoxを、一部経路が1 pixel線として扱う」という境界条件差で説明できた。source v5とv6のstray件数は同一であり、v6のdeleted-XML invalidationが新たに生成した問題ではない。

crop可視性の監査結果は次のとおりである。

| video_name | XML BBox | fully visible | partial | outside | 採否 |
|---|---:|---:|---:|---:|---|
| `20250626_090652_6340` | 10 | 0 | 4 | 6 | 動画除外 |
| `20250625_161030_0550` | 14 | 9 | 4 | 1 | frame 51だけ無効化 |
| `20250626_090758_8000` | 13 | 0 | 4 | 9 | 既存除外を維持 |

`20250625_161030_0550`ではframe 47–50が右側へ段階的にclipされ、frame 51で完全にcrop外となった。frame 51の保存BBoxは`[255.0, 117.706, 255.0, 135.824]`の幅0で、visible fractionは0、保存contourもinvalidだった。このframeだけを無効化し、CVAT確認済みmaskが存在するframe 47–50は維持した。

## 3. 固定manifestと入力inventory

次のversioned manifestを作成した。

- `Stage2to4/pseudo3d/analysis/configs/stage4_video_exclusions_v2.csv`
  - `20250626_090758_8000`
  - `20250626_090652_6340`
- `Stage2to4/pseudo3d/analysis/configs/stage4_crop_quality_invalidations_v1.csv`
  - `20250625_161030_0550`, frame order/index 51
  - action: `invalidate_entire_frame`
  - reason: `local_crop_fully_outside`
  - expected stray ignore: 6

生成済み学習manifest:

```text
/mnt/data/3d_projects/pseudo3d_dataset/stage4_sampling_parameter_sweep/260711/
manifests/train_manifest_cropclean_v2.csv
```

元manifest 182行に対してenabled 180、disabled 2である。

## 4. 実装内容

### 4.1 監査・manifest固定

- `Stage2to4/pseudo3d/analysis/audit_stage4_crop_quality_stray_ignore.py`
- `Stage2to4/checks/stage4/check_stage4_crop_quality_stray_ignore.py`
- `Stage2to4/pseudo3d/pipelines/audit_stage4_crop_quality_stray_ignore.sh`
- `Stage2to4/pseudo3d/analysis/validate_stage4_crop_quality_manifests.py`
- `Stage2to4/checks/stage4/check_stage4_crop_quality_manifests.py`
- `Stage2to4/pseudo3d/pipelines/build_stage4_crop_quality_manifests_v2.sh`

### 4.2 BBox境界条件とteacher v7構築

- BBox unionの有効条件を`right > left and bottom > top`へ統一し、退化BBoxを1 pixelのlabel領域として扱わないようにした。
- `Stage2to4/pseudo3d/annotation/apply_crop_quality_invalidations.py`
- `Stage2to4/pseudo3d/batch/annotation/batch_apply_stage4_crop_quality_invalidations.py`
- `Stage2to4/checks/stage4/check_stage4_crop_quality_invalidation_apply.py`
- `Stage2to4/pseudo3d/pipelines/build_stage4_bbox_ranked_v7_crop_quality.sh`

frame invalidationでは対象frameのpoint labelを全background、`valid_mask`を全trueとし、active `frame_annotation`およびmanual-review BBox rowを除去した。補正前BBox・CVAT情報はprovenanceへ退避し、v6のXML invalidation履歴と対象外データは保持した。source v6、XML、CVAT snapshotは上書きしていない。

### 4.3 受入検査

- `Stage2to4/checks/stage4/check_stage4_crop_quality_v7_acceptance.py`

HDF5 scalar datasetは`dataset[()]`、配列datasetは`dataset[:]`で読み分けるようにし、scalar dataspaceへの不正sliceも解消した。

### 4.4 保存ラベル可視化

- `Stage2to4/pseudo3d/export/export_stage4_point_label_visualization.py`
- `Stage2to4/pseudo3d/batch/export/batch_export_stage4_point_label_visualization.py`
- `Stage2to4/checks/stage4/check_stage4_crop_quality_invalidation_visualization.py`
- `Stage2to4/pseudo3d/pipelines/export_stage4_v7_crop_quality_point_label_visualizations.sh`

可視化は保存済みpoint labelとactive BBoxのみを使用し、contour/XMLを再計算しない。XMLまたはcrop-qualityで無効化されたframeでは旧CVAT maskとarchive済みBBoxを描画しない。

### 4.5 Stage 5入力切り替え

次をteacher v7既定へ変更した。

- `Stage5/train_stage5.sh`
- `Stage5/infer_stage5.sh`
- `Stage5/evaluate_stage5.sh`

主な変更は次のとおりである。

- 既定teacher: `bboxrank_v7_cvat_authoritative_crop_quality_v1`
- 既定run名: `global_local_l75_w31_c12_area15_bboxrank_v7_cvat_authoritative_crop_quality_v1`
- 既定入力数: 180
- CVAT provenance期待値: 58動画、2960 frame
- XML invalidationとcrop-quality invalidationを別々にfail-fast検証
- 除外2動画の非混入を検証
- 旧v6 runとの衝突を避けるため既定prefixを`bboxrankv7_cvatcropq`へ変更
- 学習を開始しない`PREFLIGHT_ONLY=1`を追加

学習方式、normalization、label policy、class weightなどの学習条件は、この入力切り替えでは変更していない。

## 5. teacher v7成果物

schema/token:

```text
bboxrank_v7_cvat_authoritative_crop_quality_v1
```

run root:

```text
/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/
global_local_l75_w31_c12_area15_bboxrank_v7_cvat_authoritative_crop_quality_v1
```

主要成果物:

- `annotated/`: 180 H5
- `collected/`: 180 H5（Stage 5入力）
- `crop_quality_invalidation_summary.csv`
- `crop_quality_invalidation_summary.json`
- `crop_quality_acceptance_step5.json`
- `annotation_textures_v7_crop_quality_labels/summary.csv`

teacher v7構築集計:

| 項目 | 値 |
|---|---:|
| 出力動画 | 180 |
| 除外動画 | 2 |
| crop無効化frame | 1 |
| 除去BBox row | 1 |
| ignore→background | 6 |
| positive除去 | 0 |

## 6. 検証結果

### 6.1 全180 H5の機械受入

`check_stage4_crop_quality_v7_acceptance.py`はexit code 0で完了した。

| 項目 | 結果 |
|---|---:|
| source v6 videos | 181 |
| v7 videos | 180 |
| affected videos | 1 |
| unaffected semantic-identity videos | 179 |
| invalidated frames | 1 |
| removed BBox rows | 1 |
| removed ignore points | 6 |
| stray ignore points | 0 |
| no-BBox ignore points | 0 |
| BBox内background points | 0 |
| positive outside CVAT mask points | 0 |

### 6.2 保存ラベル可視化

合成検査、対象動画先行出力、全180動画一括出力がすべて通過した。対象frame 51にactive BBox、positive、ignore、旧CVAT maskが描画されず、保存済みbackgroundだけであることを目視確認した。

全件集計:

```text
inputs=180, processed=179, skipped_verified=1, failed=0
xml_invalidation_frames=7, xml_invalidation_videos=2
crop_invalidation_frames=1, crop_invalidation_videos=1
suppressed_cvat_mask_frames=8
input_files_modified=0
```

### 6.3 Stage 5 teacher v7入力preflight

次を実行し、学習を開始せずexit code 0で完了した。

```bash
cd /mnt/data/3d_projects/models/Stage5

PREFLIGHT_ONLY=1 \
bash train_stage5.sh
```

確定した出力:

```text
files=180, videos=180
cvat_videos=58, cvat_frames=2960
invalidated_videos=2, invalidated_frames=7
removed_positive=2124, removed_bbox_rows=7
crop_invalidated_videos=1, crop_invalidated_frames=1
crop_removed_ignore=6, crop_removed_bbox_rows=1
```

入力dirとpatternもteacher v7を指していることを確認済みである。

## 7. Stage 5実装管理チャットへの引継事項

S5-12の続きはteacher v7を正本入力として行う。

1. Stage 5固有のlabel-policy BBox preflightをv7の180 H5または新しいtrain/validation listへ向けて再実行し、stray ignore=0をStage 5側でも確認する。
2. v7 inventoryを用いてtrain/validation listとDataset parityを固定する。旧v6の181ファイル前提・163/18 splitを暗黙に再利用しない。
3. S5-12 Run A/Bの共通条件を再確認する。
   - Run A: `bbox_noncontour_ignore`
   - Run B: `bbox_noncontour_background`
   - GroupNormなど、S5-12で既に固定した比較条件はStage 5側の管理文書を正本とする。
4. 学習開始前に、v7 prefixの新規runであること、旧v6 checkpoint/runを上書きしないことを確認する。
5. inference/evaluationはv7 checkpoint生成後に行う。評価用PLY/reference directoryが必要な場合は、v7 rootに対応成果物が存在するかを別途preflightする。

Stage 5側のfail-fastは緩めない。今回の修正によって入力側で契約違反を除去したため、stray pointを黙ってbackgroundへ変換する回避策は不要である。

## 8. 既存結果の扱い

- S5-07〜S5-11はteacher v6による既存結果として保持する。
- teacher v6の181動画とteacher v7の180動画は入力inventoryが異なるため、指標比較時にteacher版を明記する。
- v6成果物、source pseudo3D H5、VOC XML、返却CVAT snapshotは変更していない。
- 本修正だけを理由に既存runを一律再学習するかは決めない。S5-12以降の目的と比較可能性に基づきStage 5側で判断する。

## 9. 関連文書

- 元調査依頼: `docs/stage2to4/stage4/stage5_s5_12_stage4_crop_quality_investigation_request.md`
- 詳細計画・実施記録: `docs/stage2to4/stage4/stage4_crop_quality_stray_ignore_investigation_plan.md`
- Stage 5 S5-12実装引継ぎ: `docs/stage5/s5-12/stage5_s5_12_label_policy_implementation_handoff_prompt.md`
- Stage 5学習・評価記録: `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
- Stage 5改修管理記録: `docs/stage5/stage5_revision_management_record.md`
