# Stage 5 S5-10 完了報告: BatchNorm recalibration診断

作成日: 2026-09-12

方針管理チャットの判断待ち事項があるため、実装チャットからの完了報告として共有する。
数値の正本は`docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.5節
「実装事項C」、進捗管理は`docs/stage5/stage5_revision_management_record.md` S5-10に記録済み。

## 1. 経緯

以前のS5-10実装チャットは環境復旧中に失われたが、実装ファイル自体は残っていた
（`docs/stage5/s5-10/stage5_s5_10_recovery_handoff_prompt.md`から引き継ぎ）。本チャットでは
既存実装を作り直さず、監査・補強・実行検証・結果記録のみを行った。

## 2. 変更ファイル

```text
Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.py
Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.sh
docs/stage5/stage5_pointnext_s_training_evaluation_report.md   (9.5節へ実装事項Cを追記)
docs/stage5/stage5_revision_management_record.md               (S5-10を完了記録へ移動、3/4章更新)
```

補強内容（引き継ぎ文書7節の指摘への対応）:

1. 全BN moduleの`num_batches_tracked == calibration_processed_windows`をfail-fast assertion化。
2. 全BN moduleの`running_mean`/`running_var`のfinite性をfail-fast assertion化。
3. 全BN moduleの`running_var`非負をfail-fast assertion化。
4. calibration/validation重複検査に`Path.resolve()`後の比較を追加。
5. 上記1〜4に対応するsynthetic testを4件追加（既存4件と合わせ計8件）。

`train_stage5.py`/`infer_stage5.py`/`evaluate_stage5.py`の既定挙動は変更していない。

## 3. 実行したテストと成否

| テスト | 結果 |
| --- | --- |
| `python3 -m py_compile check_stage5_batchnorm_recalibration.py` | 合格（実装チャット） |
| `bash -n check_stage5_batchnorm_recalibration.sh` | 合格（実装チャット） |
| `git diff --check` | 合格（実装チャット） |
| synthetic self-test（8ケース） | 合格（ユーザー実機） |
| train sanity 3件のみのsmoke run（calibrationは163動画全件） | 合格（ユーザー実機） |
| full run（train sanity 3件+validation 18件、計21動画） | 合格（ユーザー実機） |
| 匿名化self-check（`assert_share_bundle_anonymous`＋実装チャット側の独立`grep`確認） | 合格 |

## 4. full runの入力条件

- checkpoint: teacher v6 5 epoch pilot`best.pt`（epoch 4）
- calibration list: `${RUN_DIR}/train_files.txt`
- 評価対象: 固定train sanity 3動画 + validation全18動画
- window: size 16 / stride 8、tailあり、physical batch size 1
- aggregation: mean probability（現行production方式）
- device: cuda、seed: 42

## 5. calibration規模とBN module健全性

- calibration processed files/windows: **163 / 729**（対象runの期待値と一致）
- BN module数: 17。全moduleで`recalibrated_num_batches_tracked = 729`（processed windowsと完全一致）
- 全moduleで`running_mean`/`running_var`がfinite、負のvarianceは0件
- running statistics自体の変化量は大きい: `running_var_abs_diff_max`最大約469.9
  （`pointnext.decoder.decoder.3.0.convs.0.1`）、`running_mean_abs_diff_max`最大約12.3
- `original_num_batches_tracked`は全module16859（事項Bの記録と一致）

## 6. parameter/非BN buffer不変性、original baseline parity

- parameter hash: recalibration前後で完全一致
- 非BN buffer hash: recalibration前後で完全一致
- mean baseline parity: original evalのTP/FP/TN/FNが既存`h5_metrics.csv`と完全一致
  （train_sanity: TP407/FP16602/TN600976/FN9349、validation: TP2691/FP85785/TN6421265/FN66760）
- calibration（train_files.txt）とvalidationの重複: なし（`Path.resolve()`後も確認）

## 7. original vs recalibrated 主要metrics

### probability差（GT区分別）

| split | GT区分 | disagreement率 | mean abs diff |
| --- | --- | ---: | ---: |
| train_sanity | 全valid点 | 2.43% | 0.045 |
| train_sanity | GT positive点のみ | 13.27% | 0.105 |
| validation | 全valid点 | 2.63% | 0.045 |
| validation | GT positive点のみ | 4.84% | 0.060 |

（参考: 事項Bのeval/train disagreement率はGT positiveでtrain_sanity 51.16%、validation 38.15%）

### aggregated（mean probability集約後）

| split | model | TP | FP | recall | F1 | IoU | TP0動画数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train_sanity | original | 407 | 16,602 | 4.17% | 0.0304 | 0.0154 | 1/3 |
| train_sanity | recalibrated | 298 | 13,331 | 3.05% | 0.0255 | 0.0129 | 2/3 |
| validation | original | 2,691 | 85,785 | 3.87% | 0.0341 | 0.0173 | 10/18 |
| validation | recalibrated | 2,819 | 73,374 | 4.06% | 0.0387 | 0.0197 | 10/18 |

### window-granularity（集約前）

| split | model | TP | FP | recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| train_sanity | original | 2,516 | 28,784 | 14.19% | 0.1026 |
| train_sanity | recalibrated | 1,252 | 23,541 | 7.06% | 0.0589 |
| validation | original | 10,991 | 343,123 | 8.54% | 0.0455 |
| validation | recalibrated | 9,653 | 290,511 | 7.50% | 0.0450 |

### video単位のTP0反転

validationのTP0動画数は10/18のまま変化なしだが、構成が入れ替わった。

- `validation_004`: TP 59→0（悪化）
- `validation_016`: TP 0→31（改善）
- train_sanityでは`train_sanity_random_002`がTP 190→0（悪化）、TP0動画数1/3→2/3

`validation_011`（F1 0.0115→0.2584）、`validation_013`（F1 0.0009→0.0641）のような大きな改善と、
`validation_005`（F1 0.0808→0.0379）のような悪化が同時に生じており、split集約の小さな正味変化の
内側で動画ごとの符号は一致していない。

## 8. S5-09（train-mode）結果との関係

事項Bのtrain-mode推論で観測された大幅なrecall/FP同時増加
（validation aggregated recall 3.87%→33.12%、FP+672,160点）は、recalibrationでは
**再現されなかった**。recalibrationはrecallとFPがおおむね同方向（両方微減または横ばい）に動き、
train-modeとは質的に異なる、小さく符号が一致しない変化だった。

## 9. 支持された仮説・棄却された仮説・未確定事項

**支持:**

- BN module 17個全てでrunning statisticsは仕様通り正しく再計算された
  （`num_batches_tracked`一致、finite、非負variance、parameter/非BN buffer不変）。
- 保存済みrunning statisticsのstalenessは、S5-09で観測した規模のeval/train乖離の主要因からは
  後退する（running statistics自体は大きく変化したが、production指標への効果は小さい）。

**棄却（暫定）:**

- 「recalibrationだけでS5-09のtrain-mode相当の改善が得られる」という想定は支持されなかった
  （TP0動画数はvalidationで変化なし、train_sanityではむしろ悪化）。

**未確定:**

- S5-09の効果の主要因が、test-time adaptation（各windowが自身のbatch statisticsを使う効果）に
  あるのか、normalization方式そのもの（physical batch size 1）にあるのかは、本checkerでは
  分離できていない。
- validationで観測されたvideo単位の改善・悪化の入れ替わりが、系統的な変化かnoiseかは未検証。

## 10. productionへの影響

なし。recalibrated checkpointはユーザー実機の`stage5_debug/`下にdiagnostic artifactとして
保存されており、production `train_stage5.py`/`infer_stage5.py`/`evaluate_stage5.py`の既定挙動、
production aggregation（mean probability）はいずれも変更していない。

## 11. share/private出力先

```text
共有（匿名化済み、実装チャットで独立grep確認済み）:
  research/stage5/s5-10/batchnorm_recalibration_share/
    batchnorm_recalibration_summary.json
    batchnorm_recalibration_probability_difference.csv
    batchnorm_recalibration_window_metrics.csv
    batchnorm_recalibration_aggregation_metrics.csv
    batchnorm_recalibration_checkpoint_summary.csv
    batchnorm_recalibration_model_delta.csv
    batchnorm_recalibration_buffer_diff.csv

ユーザー実機（非共有）:
  /mnt/data/3d_projects/stage5_debug/batchnorm_recalibration/
    pointnext_s_EX260908_260711_..._best/share_metrics/   (上記と同内容)
    pointnext_s_EX260908_260711_..._best/private_DO_NOT_SHARE/
      recalibrated_checkpoint_DO_NOT_SHARE.pt
      video_id_map_DO_NOT_SHARE.csv
      batchnorm_recalibration_run_info_DO_NOT_SHARE.json
      mean_baseline_parity_failure.csv（今回は不発生）
```

## 12. 方針管理チャットで判断が必要な事項

1. S5-11（BatchNorm対策の選定）の優先候補を、recalibrationのcheckpoint運用ではなく、
   per-window statisticsを使う正規化やGroupNorm/LayerNorm等への置換へ変更するか。
2. train_sanityで観測された悪化（TP0動画数1/3→2/3、window F1 0.1026→0.0589）を、
   train_sanity 3動画という小標本のnoiseとして扱うか、追加検証（動画数を増やす、
   複数seedでのsensitivity case）が必要と判断するか。
3. Label policy ablation（9.6節）・class weight比較（9.7節）を、S5-11のBatchNorm対策比較より
   先に行うか。

再学習、loss変更、threshold tuning、production aggregationおよびBatchNorm設定の変更は、
本報告時点でも実施していない。
