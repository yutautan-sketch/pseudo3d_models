# s5-13: Class weight W-A/W-B/W-C・threshold-free診断

[調査一覧](../README.md)

## 対応する文書

- [Stage 5 S5-13: GroupNorm固定Class Weight Ablation 実装依頼書](../../../docs/stage5/s5-13/stage5_s5_13_class_weight_ablation_implementation_request.md)
- [Stage 5 S5-13: GroupNorm固定Class Weight Ablation 報告書](../../../docs/stage5/s5-13/stage5_s5_13_report_to_policy_chat.md)
- [Stage 5 S5-13補足: Threshold-free診断と中間Class Weight確認 実装引き継ぎ](../../../docs/stage5/s5-13/stage5_s5_13_supplement_implementation_handoff.md)
- [Stage 5 S5-13補足: Threshold-free診断と中間Class Weight確認 報告書](../../../docs/stage5/s5-13/stage5_s5_13_supplement_report_to_policy_chat.md)

## 保存成果物

元の実験名・日付・パッケージ階層を保持しています。圧縮版と展開版は重複排除せず保管しています。

- [s5_13_supplement_wa_threshold_free.json](s5_13_supplement_wa_threshold_free.json)
- [s5_13_supplement_wb_threshold_free.json](s5_13_supplement_wb_threshold_free.json)
- [s5_13_supplement_wc_threshold_free.json](s5_13_supplement_wc_threshold_free.json)
- [s5_13_wa_share_metrics/SHARE_THIS_DIRECTORY.txt](s5_13_wa_share_metrics/SHARE_THIS_DIRECTORY.txt)
- [s5_13_wa_share_metrics/anonymization_report.json](s5_13_wa_share_metrics/anonymization_report.json)
- [s5_13_wa_share_metrics/anonymized_metrics_manifest.json](s5_13_wa_share_metrics/anonymized_metrics_manifest.json)
- [s5_13_wa_share_metrics/train_sanity_evaluation/train_sanity_checkpoint_summary.csv](s5_13_wa_share_metrics/train_sanity_evaluation/train_sanity_checkpoint_summary.csv)
- [s5_13_wa_share_metrics/train_sanity_evaluation/train_sanity_h5_metrics.csv](s5_13_wa_share_metrics/train_sanity_evaluation/train_sanity_h5_metrics.csv)
- [s5_13_wa_share_metrics/train_sanity_evaluation/train_sanity_window_metrics.csv](s5_13_wa_share_metrics/train_sanity_evaluation/train_sanity_window_metrics.csv)
- [s5_13_wa_share_metrics/training_history/training_config_anonymized.json](s5_13_wa_share_metrics/training_history/training_config_anonymized.json)
- [s5_13_wa_share_metrics/training_history/training_epoch_metrics.jsonl](s5_13_wa_share_metrics/training_history/training_epoch_metrics.jsonl)
- [s5_13_wa_share_metrics/validation_accuracy/validation_checkpoint_summary.csv](s5_13_wa_share_metrics/validation_accuracy/validation_checkpoint_summary.csv)
- [s5_13_wa_share_metrics/validation_accuracy/validation_h5_metrics.csv](s5_13_wa_share_metrics/validation_accuracy/validation_h5_metrics.csv)
- [s5_13_wa_share_metrics/validation_accuracy/validation_window_metrics.csv](s5_13_wa_share_metrics/validation_accuracy/validation_window_metrics.csv)
- [s5_13_wb_share_metrics/SHARE_THIS_DIRECTORY.txt](s5_13_wb_share_metrics/SHARE_THIS_DIRECTORY.txt)
- [s5_13_wb_share_metrics/anonymization_report.json](s5_13_wb_share_metrics/anonymization_report.json)
- [s5_13_wb_share_metrics/anonymized_metrics_manifest.json](s5_13_wb_share_metrics/anonymized_metrics_manifest.json)
- [s5_13_wb_share_metrics/train_sanity_evaluation/train_sanity_checkpoint_summary.csv](s5_13_wb_share_metrics/train_sanity_evaluation/train_sanity_checkpoint_summary.csv)
- [s5_13_wb_share_metrics/train_sanity_evaluation/train_sanity_h5_metrics.csv](s5_13_wb_share_metrics/train_sanity_evaluation/train_sanity_h5_metrics.csv)
- [s5_13_wb_share_metrics/train_sanity_evaluation/train_sanity_window_metrics.csv](s5_13_wb_share_metrics/train_sanity_evaluation/train_sanity_window_metrics.csv)
- [s5_13_wb_share_metrics/training_history/training_config_anonymized.json](s5_13_wb_share_metrics/training_history/training_config_anonymized.json)
- [s5_13_wb_share_metrics/training_history/training_epoch_metrics.jsonl](s5_13_wb_share_metrics/training_history/training_epoch_metrics.jsonl)
- [s5_13_wb_share_metrics/validation_accuracy/validation_checkpoint_summary.csv](s5_13_wb_share_metrics/validation_accuracy/validation_checkpoint_summary.csv)
- [s5_13_wb_share_metrics/validation_accuracy/validation_h5_metrics.csv](s5_13_wb_share_metrics/validation_accuracy/validation_h5_metrics.csv)
- [s5_13_wb_share_metrics/validation_accuracy/validation_window_metrics.csv](s5_13_wb_share_metrics/validation_accuracy/validation_window_metrics.csv)
- [s5_13_wc_share_metrics/SHARE_THIS_DIRECTORY.txt](s5_13_wc_share_metrics/SHARE_THIS_DIRECTORY.txt)
- [s5_13_wc_share_metrics/anonymization_report.json](s5_13_wc_share_metrics/anonymization_report.json)
- [s5_13_wc_share_metrics/anonymized_metrics_manifest.json](s5_13_wc_share_metrics/anonymized_metrics_manifest.json)
- [s5_13_wc_share_metrics/train_sanity_evaluation/train_sanity_checkpoint_summary.csv](s5_13_wc_share_metrics/train_sanity_evaluation/train_sanity_checkpoint_summary.csv)
- [s5_13_wc_share_metrics/train_sanity_evaluation/train_sanity_h5_metrics.csv](s5_13_wc_share_metrics/train_sanity_evaluation/train_sanity_h5_metrics.csv)
- [s5_13_wc_share_metrics/train_sanity_evaluation/train_sanity_window_metrics.csv](s5_13_wc_share_metrics/train_sanity_evaluation/train_sanity_window_metrics.csv)
- [s5_13_wc_share_metrics/training_history/training_config_anonymized.json](s5_13_wc_share_metrics/training_history/training_config_anonymized.json)
- [s5_13_wc_share_metrics/training_history/training_epoch_metrics.jsonl](s5_13_wc_share_metrics/training_history/training_epoch_metrics.jsonl)
- [s5_13_wc_share_metrics/validation_accuracy/validation_checkpoint_summary.csv](s5_13_wc_share_metrics/validation_accuracy/validation_checkpoint_summary.csv)
- [s5_13_wc_share_metrics/validation_accuracy/validation_h5_metrics.csv](s5_13_wc_share_metrics/validation_accuracy/validation_h5_metrics.csv)
- [s5_13_wc_share_metrics/validation_accuracy/validation_window_metrics.csv](s5_13_wc_share_metrics/validation_accuracy/validation_window_metrics.csv)
