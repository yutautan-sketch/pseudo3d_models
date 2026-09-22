# Stage5 実装事項A・B 完了報告（2026-09-10）

対象: run `pointnext_s_EX260908_260711_..._ep5_bs1_acc8_nopad` の `best.pt`（epoch 4、teacher v6 5 epoch pilot）。

実装・実行・レポート記録は完了。詳細は`docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.5節（実装事項A・Bそれぞれの検証結果）に記録済み。本書はその要約。

## 1. mean baseline parityの成否

事項A・Bとも合格。checkerが独立に計算したmean_probability（事項A）／eval-mode mean集約（事項B）のTP/FP/TN/FNが、既存`evaluate_stage5.py`の記録値と完全一致した。

| 対象 | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: |
| train sanity（3動画合計） | 407 | 16,602 | 600,976 | 9,349 |
| validation（18動画合計） | 2,691 | 85,785 | 6,421,265 | 66,760 |

## 2. 実装事項A: overlap window間のdisagreement率

- GT positive点のdisagreement率はvideo間で0.2%〜79.7%と大きくばらつく。disagreement点の大半がmean集約でbackground判定になっている。
- **window境界距離（`min_edge_distance`）との相関はほぼ無し**（5.9〜6.0%で横ばい）。`center_weighted`/`center_nearest`の設計前提（境界に近いほど不一致増）は支持されない。
- 動画内相対位置（時間経過）とはある程度相関あり（前半decile ~2%→後半decile ~7%）。

## 3. suppressed-positive数と割合

`max_probability > 0.5`かつ`mean_probability <= 0.5`（mean集約で見かけ上background化する点）。

| split | 対象 | 点数 | suppressed-positive数 | 割合 |
| --- | --- | ---: | ---: | ---: |
| train_sanity | GT positive点のみ | 9,756 | 2,101 | 21.5% |
| validation | GT positive点のみ | 69,451 | 7,331 | 10.6% |

## 4. 4 aggregation方式の比較（split集約）

| split | method | F1 | IoU | recall | FP | TP0動画数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train_sanity | mean | 0.030 | 0.015 | 4.2% | 16,602 | 1/3 |
| train_sanity | max | 0.125 | 0.067 | 25.7% | 27,848 | 0/3 |
| train_sanity | center_nearest | 0.113 | 0.060 | 18.9% | 21,111 | 1/3 |
| train_sanity | center_weighted | 0.051 | 0.026 | 7.3% | 17,589 | 1/3 |
| validation | mean | 0.034 | 0.017 | 3.9% | 85,785 | 10/18 |
| validation | max | 0.049 | 0.025 | 14.4% | 328,572 | 7/18 |
| validation | center_nearest | 0.043 | 0.022 | 8.8% | 210,119 | 8/18 |
| validation | center_weighted | 0.037 | 0.019 | 5.0% | 117,206 | **11/18** |

`center_weighted`はvalidationのsplit集約IoU/F1がmean比で改善するが、TP0動画数はmean（10/18）より悪化（11/18）。`max`はrecall/F1最大だがFPも急増（診断用、採用候補外）。

**→ 方針管理チャットの既決定事項**: production aggregationは`mean`を維持、`center_weighted`の重み式も再設計しない（境界距離相関が薄いため）。

## 5. 実装事項B: BatchNorm train/eval probability差とclass disagreement率

同一window・同一点順序・同一RNG seedで、eval-mode（保存済みrunning statistics）とtrain-mode（各windowのbatch statistics）を比較。**train sanity・validationいずれも、train-mode側が大幅に良い結果**になった。

| split | GT区分 | disagreement率 | mean abs diff |
| --- | --- | ---: | ---: |
| train_sanity | 全valid点 | 15.4% | 0.143 |
| train_sanity | GT positive点のみ | 51.2% | 0.285 |
| validation | 全valid点 | 16.2% | 0.156 |
| validation | GT positive点のみ | 38.2% | 0.260 |

| split | granularity | mode | recall | F1 | IoU | TP0動画数 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| train_sanity | aggregated | eval | 4.17% | 0.030 | 0.015 | 1/3 |
| train_sanity | aggregated | train | 61.01% | 0.118 | 0.063 | 0/3 |
| validation | aggregated | eval | 3.87% | 0.034 | 0.017 | 10/18 |
| validation | aggregated | train | 33.12% | 0.054 | 0.028 | **0/18** |

BatchNorm module 17個の`running_mean`/`running_var`は全moduleでfinite、負のvarianceも無く、`num_batches_tracked`も全module一致（16859）。**保存されている統計量の値自体は壊れていない**が、production（eval-mode）はtrain-modeのbatch statisticsを使った場合と比べて著しく性能が低い。validation（学習に使っていない18動画）でも同じ傾向が再現しており、train sanityへの過学習的な現象ではない。

train-modeのFPも大幅に増える（validation aggregated FPはeval比+672,160点）ため、「train-modeの統計量をそのまま採用すればよい」という意味ではない点に注意。

**→ 6.3節の判定: 「train-statistics側だけ大幅に良い」に該当。BatchNorm running statisticsまたはphysical batch size 1学習の影響を支持。**

## 6. checkerが示した事実と、まだ判断できない事項

**事実として確認できたこと:**
- mean集約によるpositive抑制は実在する（事項A）。
- window境界距離という設計仮説は棄却される（事項A）。
- eval-modeのBatchNorm running statisticsが、production性能低下の主要因の一つである可能性が高い（事項B）。validationでも再現するため汎化性能の問題ではなくproduction推論経路固有の問題。

**まだ判断できない事項:**
- BatchNorm対策（recalibration・freeze・別normへの変更など）をどう実施するか。
- overlap不一致・aggregation方式の問題とBatchNorm問題の相対的な寄与度（両方が独立に効いている可能性が高いが、定量的な切り分けは未実施）。
- aggregation方式の優劣は最終モデル確定後に再評価が必要（今回の4方式比較を恒久的結論として扱わない）。

## 7. 変更ファイル・実行コマンド・出力先・テスト結果

**新規ファイル:**
```
Stage5/checks/real_h5/check_stage5_overlap_aggregation.py / .sh
Stage5/checks/real_h5/check_stage5_batchnorm_mode_parity.py / .sh
docs/stage5/s5-08-09/stage5_overlap_aggregation_implementation_policy.md
```

**更新ファイル:**
```
docs/stage5/s5-08-09/stage5_overlap_aggregation_handoff_prompt.md（進捗記録を追記）
docs/stage5/stage5_pointnext_s_training_evaluation_report.md（9.5節へ事項A・Bの結果を追記）
docs/stage5/FILES.md
```

**テスト結果:** 両checkerともsynthetic self-test（事項A: 8ケース、事項B: 5ケース）、smoke run、full run（train sanity 3件+validation 18件）を完走。mean baseline parityは両方とも合格。

**出力先（ユーザー実機、`/mnt/data`配下）:**
```
/mnt/data/3d_projects/stage5_debug/overlap_aggregation/
/mnt/data/3d_projects/stage5_debug/batchnorm_mode_parity/
```
各`share_metrics/`配下は匿名化済み（checker自身のanonymization self-checkおよび実装チャット側の独立grep確認済み）。

## 8. 未決定事項（判断を仰ぎたいもの）

1. BatchNorm対策（recalibration診断への着手可否、対策の方向性）をどう進めるか。
2. 対策を先に検討する場合、Label policy ablation（9.6節）・class weight比較（9.7節）の着手はどのタイミングにするか。
3. overlap不一致とBatchNormの寄与度切り分けを追加で行うか、それとも対策実施後の再測定に委ねるか。

再学習、loss変更、threshold tuning、production aggregationおよびBatchNorm設定の変更は未実施。
