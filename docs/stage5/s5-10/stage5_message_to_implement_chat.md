# Stage 5 S5-10: BatchNorm recalibration診断

Stage 5の次の改修フローとして、BatchNorm running statisticsのrecalibration診断を実装してください。

## 最初に読む文書

1. `docs/stage5/stage5_revision_management_record.md`
   - 3章の現在状態
   - S5-09の結果
   - S5-10の実装方針
2. `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
   - 9.4節
   - 9.5節の実装事項A・B
3. `docs/stage5/s5-08-09/stage5_overlap_aggregation_handoff_prompt.md`
   - 6章のBatchNorm parity仕様
4. `docs/stage5/FILES.md`

全体方針は`../stage5_revision_management_record.md`を正としてください。

## 背景

teacher v6 5 epoch pilotの`best.pt`について、同一windowの`eval()`と`train()`で次の大きな差が確認されました。

- validation全valid点のclass disagreement: 16.2%
- validation GT positive点: 38.2%
- aggregated recall: 3.87%から33.12%
- aggregated F1: 0.0341から0.0541%
- TP0動画数: 10/18から0/18
- FP: 85,785から757,945

normalization modeが重大な交絡要因であることは支持されています。ただし、`train()`は評価window自身のbatch statisticsを使うため、保存済みrunning statisticsの問題とtest-time adaptation効果が分離されていません。

## 目的

最終model parameterを変更せず、training splitだけを用いてBatchNorm running statisticsを再計算します。そのmodelを`eval()`で評価し、running statisticsの再計算だけで改善する範囲を測定してください。

production変更や再学習ではなく、診断用実装です。

## 対象

- run: teacher v6 5 epoch padding-free pilot
- checkpoint: `best.pt`、epoch 4
- calibration: `${RUN_DIR}/train_files.txt`の163動画、729 windows
- evaluation:
  - 固定train sanity 3動画
  - validation全18動画
- window: size 16、stride 8、tailあり
- physical batch size: 1
- aggregation: 現行`mean_probability`

## 新規ファイル

以下を基本案とします。

- `Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.py`
- `Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.sh`

既存のmodel構築、H5読み込み、window生成、metrics、匿名化処理を再利用し、重複実装を避けてください。

## Recalibration要件

1. `best.pt`から診断用model copyを構築する。
2. 全parameterをfreezeし、calibration前後のparameter hash完全一致をassertする。
3. `model.eval()`を基準とし、BatchNorm moduleだけをtrain modeへ切り替える。
4. dropoutやその他のstochastic layerを有効化しない。
5. BatchNormのrunning statisticsをresetする。
6. primary caseは`momentum=None`によるdeterministic cumulative updateとする。
7. `train_files.txt`の順序とwindow順序を固定し、全train windowを1回だけforwardする。
8. augmentation、label、validation dataはcalibrationに使用しない。
9. `torch.inference_mode()`を使用し、backwardやoptimizer updateを行わない。
10. 変更を許可するstateはBNの`running_mean`、`running_var`、`num_batches_tracked`だけとする。
11. 元checkpointを上書きしない。recalibrated checkpointは`stage5_debug/`下のdiagnostic artifactとする。
12. BN module数、元のmomentum、recalibration policy、処理file/window数を記録する。

通常momentumによるsensitivity caseは、primary結果が不明瞭な場合だけ追加候補とし、最初からproduction候補にはしないでください。

## 評価と比較

同じ評価対象について以下を比較してください。

- original checkpointの`eval()`
- recalibrated checkpointの`eval()`
- 実装事項Bのtrain-mode結果は既存記録をreferenceとして扱う

最低限、次を記録してください。

- probabilityのmean/max absolute differenceとpercentile
- class disagreement count/rate
- TP、FP、TN、FN
- precision、recall、F1、IoU、FPR、FNR
- predicted positive count/rate
- TP0動画数
- video-level mean/median F1、IoU
- BN module別running buffer差
- parameter hashと非BN bufferの不変性
- calibrationとvalidation listが非重複であること

original evalのTP/FP/TN/FNは既存`h5_metrics.csv`と完全一致させてください。不一致時は差分artifactを保存して停止します。

## テスト

- synthetic BN recalibration test
- BNだけが更新されること
- parameterと非BN bufferが不変であること
- cumulative updateが期待値と一致すること
- validation混入を拒否すること
- deterministic再実行
- Python compile
- bash `-n`
- train sanityによるsmoke run
- 163 train動画でrecalibration後、train sanity 3件とvalidation 18件によるfull run
- 匿名化self-check

## 判定上の注意

recalibration後にrecallだけが増え、FPもtrain-mode同様に急増した場合は、性能改善ではなく確率校正の移動として報告してください。

checker側ではproduction採否を決定しません。事実と未確定事項を分けて報告してください。productionの`train_stage5.py`、`infer_stage5.py`、`evaluate_stage5.py`の既定挙動は変更しないでください。

## 文書更新

実行完了後は次を行ってください。

- 詳細結果を`../stage5_pointnext_s_training_evaluation_report.md`へ追記
- `../stage5_revision_management_record.md`のS5-10へ実施日、実装、結果、仮説判断を追記
- 新規checkerを`../FILES.md`へ追加
- production decisionとDecision recordは方針管理チャットの判断まで確定しない

最後に、変更ファイル、テスト、出力先、主要metrics、仮説判断、方針管理チャットへ返す判断事項をまとめた共有用Markdownを作成してください。

まず既存コードを確認して実装方針を提示し、仕様上の不明点がなければ実装まで進めてください。