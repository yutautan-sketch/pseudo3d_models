> 配置注記（2026-09-22）：この一覧のファイル名・パス・Git状態はコミット分割調査当時の記録です。旧配置を現在のパスへ置き換えず保持しています。現在の成果物は[コミット準備資料](../../research/development/commit-preparation/README.md)を参照してください。

## 付録A. 全118エントリの区分割当（2026-09-12詳細確認）

状態は初回inventoryと同じ。区分IDは第6節に対応し、複数IDは差分/節の分割が必要。
区分03は実装内容確認により独立区分から解除し、10・12・13へ再配属した。

| 状態 | 対象（移動先） | 区分 | 判断・分割上の注意 |
| --- | --- | --- | --- |
| ` M` | `.gitignore` | 01 | 環境設定。Dockerfileは移設＋Claude Code導入、別差分として扱う |
| ` D` | `Stage2to4/Dockerfile.codex` | 01 | 環境設定。Dockerfileは移設＋Claude Code導入、別差分として扱う |
| ` M` | `Stage2to4/checks/stage4/check_stage4_bbox_ranked_label_policy.py` | 12 | no-BBox positiveをv5 provenance付きで許可 |
| ` M` | `Stage2to4/checks/stage4/check_stage4_sampling_sweep_manifest.py` | 10 | 動画除外契約とその検査 |
| ` M` | `Stage2to4/pseudo3d/pipelines/export_stage4_bbox_ranked_pointcloud_visualizations.sh` | 13, 16 | v6 collected H5参照と最終label色への切替 |
| ` M` | `Stage5/checks/dummy/check_dummy_pointnext_s_training.py` | 15, 18 | 勾配蓄積の検査とGroupNorm選択・reloadを分割 |
| ` M` | `Stage5/checks/dummy/check_dummy_pointnext_s_training.sh` | 15, 18 | 勾配蓄積の検査とGroupNorm選択・reloadを分割 |
| ` M` | `Stage5/checks/dummy/check_dummy_training.py` | 15 | loss_sum/normalizerと勾配蓄積の契約 |
| ` M` | `Stage5/evaluate_stage5.py` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| ` M` | `Stage5/evaluate_stage5.sh` | 16 | v6入力/run参照への移行。評価checkpoint既定値の変更を含む |
| ` M` | `Stage5/export_anonymized_stage5_metrics.sh` | 16 | v6入力/run参照への移行。評価checkpoint既定値の変更を含む |
| ` M` | `Stage5/infer_stage5.py` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| ` M` | `Stage5/infer_stage5.sh` | 16 | v6入力/run参照への移行。評価checkpoint既定値の変更を含む |
| ` M` | `Stage5/stage5/models/pointnext_s_segmentor.py` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| ` M` | `Stage5/stage5/training/losses.py` | 15 | loss_sum/normalizerと勾配蓄積の契約 |
| ` M` | `Stage5/train_stage5.py` | 15, 18 | 勾配蓄積とGroupNormのhunk分割 |
| ` M` | `Stage5/train_stage5.sh` | 15, 16, 18 | 勾配蓄積・v6 preflight・GroupNorm・run名が交錯 |
| `RM` | `docs/README.md` | 02, 各機能の目次 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage2to4/docs/README.md`。 |
| `R ` | `docs/stage2to4/legacy/dualtrack_legacy.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage2to4/docs/legacy/dualtrack_legacy.md`。 |
| `R ` | `docs/stage2to4/stage4/cvat_segmentation_mask_1_1_import_spec.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage2to4/docs/stage4/cvat_segmentation_mask_1_1_import_spec.md`。 |
| `R ` | `docs/stage2to4/stage4/edit_prompt.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage2to4/docs/stage4/edit_prompt.md`。 |
| `R ` | `docs/stage2to4/stage4/stage4_bbox_independent_sampling_edit_prompt.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `stage4_edit_prompt.md`。 |
| `RM` | `docs/stage2to4/stage4/stage4_contour_teacher_improvement_plan.md` | 02, 04, 05, 06, 07, 08, 09, 10 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage2to4/docs/stage4/stage4_contour_teacher_improvement_plan.md`。 |
| `RM` | `docs/stage2to4/stage4/stage4_phase7_report.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage2to4/docs/stage4/stage4_phase7_report.md`。 |
| `R ` | `docs/stage2to4/stage4/stage4_sampling_investigation_progress.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage2to4/docs/stage4/stage4_sampling_investigation_progress.md`。 |
| `R ` | `docs/stage2to4/stage4/stage4_sampling_sweep_investigation_edit_prompt.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `stage4_invest_edit_prompt.md`。 |
| `R ` | `docs/stage2to4/stage4/stage4_sampling_sweep_investigation_rule.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `stage4_invest_rule.md`。 |
| `RM` | `docs/stage2to4/submission/README.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage2to4/docs/submission/README.md`。 |
| `RM` | `docs/stage5/FILES.md` | 02, 14, 15, 16, 17, 18 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage5/FILES.md`。 |
| `R ` | `docs/stage5/TRAINING_IMPROVEMENT_PLAN.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage5/TRAINING_IMPROVEMENT_PLAN.md`。 |
| `R ` | `docs/stage5/data_construct.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `../stage5/data_construct.md`。 |
| `R ` | `docs/stage5/stage5_edit_prompt.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `../stage5/stage5_edit_prompt.md`。 |
| `AM` | `docs/stage5/s5-08-09/stage5_overlap_aggregation_handoff_prompt.md` | 17 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う |
| `??` | `Dockerfile.codex` | 01 | 環境設定。Dockerfileは移設＋Claude Code導入、別差分として扱う |
| `??` | `Stage2to4/checks/stage4/check_stage4_contour_auto_refine.py` | 05 | checkerはscreening/production両configを参照 |
| `??` | `Stage2to4/checks/stage4/check_stage4_contour_teacher_phase5_apply.py` | 07 | review共通処理とexporter helperが先行依存 |
| `??` | `Stage2to4/checks/stage4/check_stage4_cvat_authoritative_preflight.py` | 12 | v5検査・preflight・pipeline |
| `??` | `Stage2to4/checks/stage4/check_stage4_cvat_manual_roundtrip.py` | 06, 09 | full-video描画・textfree移行もimport/検査。全量を初期段階へ追加しない |
| `??` | `Stage2to4/checks/stage4/check_stage4_cvat_segmentation_mask_export.py` | 04 | context-only欠落許可等も含む現行converter契約 |
| `??` | `Stage2to4/checks/stage4/check_stage4_deleted_xml_annotation_audit.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/checks/stage4/check_stage4_deleted_xml_invalidation_apply.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/checks/stage4/check_stage4_deleted_xml_invalidation_batch.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/checks/stage4/check_stage4_deleted_xml_invalidation_manifest.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/checks/stage4/check_stage4_deleted_xml_invalidation_visualization.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/checks/stage4/check_stage4_phase5_cvat_task_snapshots.py` | 10 | snapshot保存・v4構築 |
| `??` | `Stage2to4/checks/stage4/check_stage4_phase5_fullvideo_cvat_tasks.py` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage2to4/checks/stage4/check_stage4_phase5_fullvideo_final_import.py` | 10, 12 | v4/v5同居。共有readerもv5対応済み |
| `??` | `Stage2to4/checks/stage4/check_stage4_point_label_visualization.py` | 11, 12, 13 | 保存label描画とCVAT全frame・XML無効化対応が同居 |
| `??` | `Stage2to4/checks/stage4/check_stage4_v5_cvat_authoritative_acceptance.py` | 12 | v5検査・preflight・pipeline |
| `??` | `Stage2to4/pseudo3d/analysis/audit_stage4_deleted_xml_annotations.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/analysis/build_stage4_deleted_xml_invalidation_manifest.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/analysis/build_stage4_exclusion_manifest.py` | 10 | 動画除外契約とその検査 |
| `??` | `Stage2to4/pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3.yaml` | 05 | checkerはscreening/production両configを参照 |
| `??` | `Stage2to4/pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3_production_v1.yaml` | 05 | checkerはscreening/production両configを参照 |
| `??` | `Stage2to4/pseudo3d/analysis/configs/stage4_deleted_xml_invalidations_v1.csv` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/analysis/configs/stage4_manual_review_cvat_v1.yaml` | 06 | importerからreview exporterのhelper参照あり |
| `??` | `Stage2to4/pseudo3d/analysis/configs/stage4_video_exclusions_v1.csv` | 10 | 動画除外契約とその検査 |
| `??` | `Stage2to4/pseudo3d/analysis/preflight_stage4_cvat_authoritative_labels.py` | 12 | v5検査・preflight・pipeline |
| `??` | `Stage2to4/pseudo3d/analysis/preflight_stage4_phase5_fullvideo_cvat_review.py` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage2to4/pseudo3d/analysis/prototype_stage4_contour_auto_refine.py` | 05 | checkerはscreening/production両configを参照 |
| `??` | `Stage2to4/pseudo3d/analysis/validate_stage4_phase5_fullvideo_cvat_package.py` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage2to4/pseudo3d/annotation/apply_deleted_xml_invalidations.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/annotation/contour_teacher_refinement.py` | 05 | checkerはscreening/production両configを参照 |
| `??` | `Stage2to4/pseudo3d/annotation/import_cvat_segmentation_mask_corrections.py` | 06 | importerからreview exporterのhelper参照あり |
| `??` | `Stage2to4/pseudo3d/annotation/stage4_manual_review.py` | 06 | importerからreview exporterのhelper参照あり |
| `??` | `Stage2to4/pseudo3d/batch/annotation/batch_apply_stage4_contour_refinement.py` | 07 | review共通処理とexporter helperが先行依存 |
| `??` | `Stage2to4/pseudo3d/batch/annotation/batch_apply_stage4_deleted_xml_invalidations.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/batch/annotation/batch_import_cvat_segmentation_mask_corrections.py` | 06 | importerからreview exporterのhelper参照あり |
| `??` | `Stage2to4/pseudo3d/batch/annotation/batch_import_stage4_phase5_fullvideo_cvat.py` | 10, 12 | v4/v5同居。共有readerもv5対応済み |
| `??` | `Stage2to4/pseudo3d/batch/export/batch_create_stage4_phase5_cvat_tasks.py` | 08, 09 | selected/master/fullvideo/修正snapshot再利用が同居 |
| `??` | `Stage2to4/pseudo3d/batch/export/batch_export_stage4_manual_review_cvat.py` | 06, 07, 08, 09 | 基本export・v3で使うhelper・全ケース・full-videoが同居 |
| `??` | `Stage2to4/pseudo3d/batch/export/batch_export_stage4_phase5_cvat_task_snapshots.py` | 10 | snapshot保存・v4構築 |
| `??` | `Stage2to4/pseudo3d/batch/export/batch_export_stage4_point_label_visualization.py` | 11, 12, 13 | 保存label描画とCVAT全frame・XML無効化対応が同居 |
| `??` | `Stage2to4/pseudo3d/batch/export/rebuild_stage4_phase5_textfree_review_package.py` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage2to4/pseudo3d/export/convert_masks_to_cvat_segmentation_mask_1_1.py` | 04 | context-only欠落許可等も含む現行converter契約 |
| `??` | `Stage2to4/pseudo3d/export/export_stage4_point_label_visualization.py` | 11, 12, 13 | 保存label描画とCVAT全frame・XML無効化対応が同居 |
| `??` | `Stage2to4/pseudo3d/pipelines/audit_stage4_v5_deleted_xml_annotations.sh` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/pipelines/build_stage4_bbox_ranked_v3_refined_auto.sh` | 07 | review共通処理とexporter helperが先行依存 |
| `??` | `Stage2to4/pseudo3d/pipelines/build_stage4_bbox_ranked_v4_manual_fullvideo.sh` | 10 | snapshot保存・v4構築 |
| `??` | `Stage2to4/pseudo3d/pipelines/build_stage4_bbox_ranked_v5_cvat_authoritative.sh` | 12 | v5検査・preflight・pipeline |
| `??` | `Stage2to4/pseudo3d/pipelines/build_stage4_bbox_ranked_v6_xml_invalidation.sh` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/pipelines/build_stage4_deleted_xml_invalidation_manifest.sh` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/pipelines/create_stage4_phase5_cvat_tasks.sh` | 08 | 全ケースレビューの入口 |
| `??` | `Stage2to4/pseudo3d/pipelines/create_stage4_phase5_fullvideo_cvat_tasks.sh` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage2to4/pseudo3d/pipelines/create_stage4_phase5_fullvideo_textfree_cvat_tasks.sh` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage2to4/pseudo3d/pipelines/export_stage4_phase5_cvat_review_cases.sh` | 08 | 全ケースレビューの入口 |
| `??` | `Stage2to4/pseudo3d/pipelines/export_stage4_phase5_fullvideo_cvat_review.sh` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage2to4/pseudo3d/pipelines/export_stage4_phase5_fullvideo_cvat_task_snapshots.sh` | 10 | snapshot保存・v4構築 |
| `??` | `Stage2to4/pseudo3d/pipelines/export_stage4_v4_point_label_visualizations.sh` | 11 | v4可視化入口 |
| `??` | `Stage2to4/pseudo3d/pipelines/export_stage4_v5_cvat_authoritative_point_label_visualizations.sh` | 12 | v5検査・preflight・pipeline |
| `??` | `Stage2to4/pseudo3d/pipelines/export_stage4_v6_xml_invalidation_point_label_visualizations.sh` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/pipelines/preflight_stage4_phase5_fullvideo_cvat_review.sh` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage2to4/pseudo3d/pipelines/preflight_stage4_v5_cvat_authoritative_labels.sh` | 12 | v5検査・preflight・pipeline |
| `??` | `Stage2to4/pseudo3d/pipelines/rebuild_stage4_phase5_fullvideo_textfree_review.sh` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage5/checks/dummy/check_dummy_pointnext_s_groupnorm.py` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| `??` | `Stage5/checks/dummy/check_dummy_pointnext_s_groupnorm.sh` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| `??` | `Stage5/checks/real_h5/check_stage5_batch_integrity.py` | 14 | 診断checkerとshellを組にする |
| `??` | `Stage5/checks/real_h5/check_stage5_batch_integrity.sh` | 14 | 診断checkerとshellを組にする |
| `??` | `Stage5/checks/real_h5/check_stage5_batchnorm_mode_parity.py` | 17 | overlap→BN parity→recalibrationのimport順を守る |
| `??` | `Stage5/checks/real_h5/check_stage5_batchnorm_mode_parity.sh` | 17 | overlap→BN parity→recalibrationのimport順を守る |
| `??` | `Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.py` | 17 | overlap→BN parity→recalibrationのimport順を守る |
| `??` | `Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.sh` | 17 | overlap→BN parity→recalibrationのimport順を守る |
| `??` | `Stage5/checks/real_h5/check_stage5_overlap_aggregation.py` | 17 | overlap→BN parity→recalibrationのimport順を守る |
| `??` | `Stage5/checks/real_h5/check_stage5_overlap_aggregation.sh` | 17 | overlap→BN parity→recalibrationのimport順を守る |
| `??` | `Stage5/checks/real_h5/check_stage5_padding_parity.py` | 14 | 診断checkerとshellを組にする |
| `??` | `Stage5/checks/real_h5/check_stage5_padding_parity.sh` | 14 | 診断checkerとshellを組にする |
| `??` | `Stage5/checks/transfer/check_stage5_batchnorm_to_groupnorm_transfer.py` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| `??` | `Stage5/checks/transfer/check_stage5_batchnorm_to_groupnorm_transfer.sh` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| `??` | `Stage5/stage5/models/norm_layers.py` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| `??` | `Stage5/stage5/models/pointnext_decoder_patch.py` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| `??` | `docs/stage2to4/stage4/stage4_bbox_ranked_border_contact_revision_plan.md` | 計画のみ | 未実装計画。機能実装コミットにはしない |
| `??` | `docs/stage2to4/stage4/stage4_cvat_snapshot_authoritative_label_revision_plan.md` | 12, 13 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う |
| `??` | `docs/stage2to4/stage4/stage4_deleted_xml_annotation_invalidation_plan.md` | 13 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う |
| `??` | `docs/stage2to4/stage4/stage4_phase5_fullvideo_cvat_review_implementation.md` | 09, 10, 12, 13 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う |
| `??` | `docs/stage2to4/stage4/stage4_v4_point_label_visualization_plan.md` | 11, 12 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う |
| `??` | `docs/stage5/s5-08-09/stage5_overlap_aggregation_implementation_policy.md` | 17 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う |
| `??` | `docs/stage5/stage5_pointnext_s_training_evaluation_report.md` | 14, 15, 16, 17, 18 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う |
| `??` | `docs/stage5/stage5_revision_management_record.md` | 14, 15, 16, 17, 18 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う |
