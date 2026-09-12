# プロジェクト documentation

このプロジェクト（胎児大腿骨検出・計測パイプライン）のMarkdown文書はこのディレクトリに集約する。
Stage2to4由来の文書は`stage2to4/`配下に、Stage5由来の文書は`stage5/`配下に格納する。

## Stage2to4 / Stage 4

- `stage2to4/stage4/edit_prompt.md`: Stage 4の初期編集要件
- `stage2to4/stage4/stage4_phase7_report.md`: Phase 7の受け入れ確認報告
- `stage2to4/stage4/stage4_sampling_investigation_progress.md`: sampling parameter調査の進捗
- `stage2to4/stage4/stage4_contour_teacher_improvement_plan.md`: BBox-aware輪郭教師の改修計画
- `stage2to4/stage4/stage4_phase5_fullvideo_cvat_review_implementation.md`: Phase 5 full-video CVAT運用・実装記録
- `stage2to4/stage4/stage4_cvat_snapshot_authoritative_label_revision_plan.md`: full-video CVAT snapshotを最終positiveの唯一の根拠にする反映改修計画
- `stage2to4/stage4/stage4_deleted_xml_annotation_invalidation_plan.md`: 削除済み誤BBox XMLの無効化、teacher v6構築・可視化・最終受入記録
- `stage2to4/stage4/stage4_bbox_ranked_border_contact_revision_plan.md`: BBox境界接触を用いた自動輪郭選択の小規模見直し案
- `stage2to4/stage4/stage4_v4_point_label_visualization_plan.md`: teacher v4の保存済み3値point labelを直接確認する可視化計画
- `stage2to4/stage4/cvat_segmentation_mask_1_1_import_spec.md`: CVATセグメンテーションマスク1:1インポート仕様
- `stage2to4/stage4/stage4_bbox_independent_sampling_edit_prompt.md`: BBox非依存Stage4 samplingの編集指示
- `stage2to4/stage4/stage4_sampling_sweep_investigation_edit_prompt.md`: 実教師データに基づくsampling parameter sweep基盤の編集指示
- `stage2to4/stage4/stage4_sampling_sweep_investigation_rule.md`: 上記調査の進め方に関する方針整理

## Stage2to4 / その他

- `stage2to4/legacy/dualtrack_legacy.md`: 旧DualTrack実行手順
- `stage2to4/submission/README.md`: submission utilitiesの案内

## Stage 5

- `stage5/stage5_revision_management_record.md`: Stage 5の概要、時系列の改修履歴、採用済み判断、現在状態、次の実施順を管理する正本
- `stage5/stage5_pointnext_s_training_evaluation_report.md`: PointNeXt-Sのtrain sanity、validation、overlap window、paddingに関する評価・原因調査
- `stage5/FILES.md`: Stage5リポジトリのファイル構成一覧
- `stage5/TRAINING_IMPROVEMENT_PLAN.md`: Stage5学習改善方針（初期プラン。現在の優先順には全体改修・管理記録を使用）
- `stage5/stage5_overlap_aggregation_handoff_prompt.md`: overlap aggregation検証・実装のチャット引継ぎプロンプト
- `stage5/data_construct.md`: Stage5評価出力構成
- `stage5/stage5_edit_prompt.md`: Stage5暫定モデル実装の編集指示

## 検査スクリプト

- `../Stage2to4/checks/stage4/`: Stage 4およびpseudo3Dの検査
- `../Stage2to4/checks/dualtrack/`: DualTrackモデルの検査
- `../Stage5/checks/`: Stage5学習・推論のスモークテスト/可視化補助

## 補助ツール

- `../Stage2to4/scripts/data/analyze_local_preprocess_candidates.py`: local前処理候補の解析
- `../Stage2to4/pseudo3d/analysis/visualize_pseudo3d_h5.py`: pseudo3D H5の可視化
