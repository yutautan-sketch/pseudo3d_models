# Stage5 文書案内

現在の判断・優先順位は[改修管理記録](stage5_revision_management_record.md)、数値的根拠は[訓練・評価レポート](stage5_pointnext_s_training_evaluation_report.md)を参照してください。各ステップの依頼・報告には当時の状態も含まれます。

運用は**総括管理チャット1つ＋各ステップの実装チャット**とします（管理記録D-038）。
Step 0の担当開始には[実装依頼書](s5-16/stage5_step0_implementation_handoff.md)、
残務・管理返信・実行結果には[Step 0報告書](s5-16/stage5_step0_report_to_policy_chat.md)を使用します。

## 全体管理・共通資料

- [FILES.md](FILES.md)：Stage5 File Layout
- [TRAINING_IMPROVEMENT_PLAN.md](TRAINING_IMPROVEMENT_PLAN.md)：Stage 5 学習改善方針
- [data_construct.md](data_construct.md)：Stage 5評価出力構成
- [stage5_edit_prompt.md](stage5_edit_prompt.md)：stage5_edit_prompt
- [stage5_pointnext_s_training_evaluation_report.md](stage5_pointnext_s_training_evaluation_report.md)：Stage 5 PointNeXt-S学習・評価調査報告
- [stage5_revision_management_record.md](stage5_revision_management_record.md)：Stage 5 全体改修・管理記録

## S5-08-09

- [stage5_overlap_aggregation_handoff_prompt.md](s5-08-09/stage5_overlap_aggregation_handoff_prompt.md)：Stage 5 overlap aggregation検証・実装チャット引継ぎプロンプト
- [stage5_overlap_aggregation_implementation_policy.md](s5-08-09/stage5_overlap_aggregation_implementation_policy.md)：実装事項A: overlap probability / aggregation checker 実装方針記録
- [stage5_report_to_policy_chat.md](s5-08-09/stage5_report_to_policy_chat.md)：Stage5 実装事項A・B 完了報告（2026-09-10）

## S5-10

- [stage5_message_to_implement_chat.md](s5-10/stage5_message_to_implement_chat.md)：Stage 5 S5-10: BatchNorm recalibration診断
- [stage5_s5_10_recovery_handoff_prompt.md](s5-10/stage5_s5_10_recovery_handoff_prompt.md)：Stage 5 S5-10 実装復旧・継続用引き継ぎプロンプト
- [stage5_s5_10_report_to_policy_chat.md](s5-10/stage5_s5_10_report_to_policy_chat.md)：Stage 5 S5-10 完了報告: BatchNorm recalibration診断

## S5-11

- [stage5_s5_11_groupnorm_implementation_handoff_prompt.md](s5-11/stage5_s5_11_groupnorm_implementation_handoff_prompt.md)：Stage 5 S5-11: GroupNorm normalization比較 実装引き継ぎプロンプト
- [stage5_s5_11_report_to_policy_chat.md](s5-11/stage5_s5_11_report_to_policy_chat.md)：Stage 5 S5-11 完了報告: GroupNorm normalization比較

## S5-12

- [stage5_s5_12_label_policy_implementation_handoff_prompt.md](s5-12/stage5_s5_12_label_policy_implementation_handoff_prompt.md)：Stage 5 S5-12: GroupNorm固定Label Policy Ablation 実装引き継ぎプロンプト
- [stage5_s5_12_report_to_policy_chat.md](s5-12/stage5_s5_12_report_to_policy_chat.md)：Stage 5 S5-12 完了報告: GroupNorm固定Label policy ablation

## S5-13

- [stage5_s5_13_class_weight_ablation_implementation_request.md](s5-13/stage5_s5_13_class_weight_ablation_implementation_request.md)：Stage 5 S5-13: GroupNorm固定Class Weight Ablation 実装依頼書
- [stage5_s5_13_report_to_policy_chat.md](s5-13/stage5_s5_13_report_to_policy_chat.md)：Stage 5 S5-13: GroupNorm固定Class Weight Ablation 報告書
- [stage5_s5_13_supplement_implementation_handoff.md](s5-13/stage5_s5_13_supplement_implementation_handoff.md)：Stage 5 S5-13補足: Threshold-free診断と中間Class Weight確認 実装引き継ぎ
- [stage5_s5_13_supplement_report_to_policy_chat.md](s5-13/stage5_s5_13_supplement_report_to_policy_chat.md)：Stage 5 S5-13補足: Threshold-free診断と中間Class Weight確認 報告書

## S5-14

- [stage5_s5_14_report_to_policy_chat.md](s5-14/stage5_s5_14_report_to_policy_chat.md)：Stage 5 S5-14: 座標依存・Frame-level・Overlap Exposure診断 報告書
- [stage5_s5_14_step_h4_parity_boundary_case_decision_request.md](s5-14/stage5_s5_14_step_h4_parity_boundary_case_decision_request.md)：Stage 5 S5-14 Step H4: Parity Gate境界ケースの扱いに関する判断依頼
- [stage5_s5_14_structural_diagnostics_implementation_handoff.md](s5-14/stage5_s5_14_structural_diagnostics_implementation_handoff.md)：Stage 5 S5-14: 座標依存・Frame-level・Overlap Exposure診断 実装依頼書
- [stage5_s5_14_supplement2_report_to_policy_chat.md](s5-14/stage5_s5_14_supplement2_report_to_policy_chat.md)：Stage 5 S5-14補足2: 座標変換診断（prediction equivariance） 報告書
- [stage5_s5_14_supplement_implementation_handoff.md](s5-14/stage5_s5_14_supplement_implementation_handoff.md)：Stage 5 S5-14補足: 点密度補正と動画別・時間別再集計 実装依頼書
- [stage5_s5_14_supplement_report_to_policy_chat.md](s5-14/stage5_s5_14_supplement_report_to_policy_chat.md)：Stage 5 S5-14補足: 点密度補正と動画別・時間別再集計 報告書

## S5-15

- [stage5_prediction_frame_visualization_implementation_handoff.md](s5-15/stage5_prediction_frame_visualization_implementation_handoff.md)：Stage 5: 予測positiveのフレーム画像可視化 実装依頼書
- [stage5_s5_15_report_to_policy_chat.md](s5-15/stage5_s5_15_report_to_policy_chat.md)：Stage 5 S5-15: 回転augmentation短期比較・R0 50 epoch評価 終了報告書
- [stage5_s5_15_rotation_augmentation_implementation_handoff.md](s5-15/stage5_s5_15_rotation_augmentation_implementation_handoff.md)：Stage 5 S5-15: 回転augmentation比較と段階的学習 実装依頼書

## S5-16

- [stage5_s5_16_implementation_handoff.md](s5-16/stage5_s5_16_implementation_handoff.md)：Stage 5 S5-16: 汎化不足への改善方針策定と限定比較 ハンドオフ
- [stage5_s5_16_to_s5_20_policy_chat_transfer.md](s5-16/stage5_s5_16_to_s5_20_policy_chat_transfer.md)：Stage 5: S5-16〜S5-20 総括管理チャットへの引き継ぎ
- [stage5_step0_report_to_policy_chat.md](s5-16/stage5_step0_report_to_policy_chat.md)：S5-16文書同期・Step 0残務と受入記録
- [stage5_step0_implementation_handoff.md](s5-16/stage5_step0_implementation_handoff.md)：Step 0の分割・封印・train_core専用weight・メタ情報確認の実装依頼

## 関連資料

- [Stage4 crop品質調査依頼](../stage2to4/stage4/stage5_s5_12_stage4_crop_quality_investigation_request.md)
- [Stage4 crop品質修正報告](../stage2to4/stage4/stage5_s5_12_stage4_crop_quality_correction_report.md)
- [調査成果物一覧](../../research/stage5/README.md)
