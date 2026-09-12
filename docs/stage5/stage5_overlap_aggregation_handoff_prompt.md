# Stage 5 overlap aggregation検証・実装チャット引継ぎプロンプト

> **進捗注記（2026-09-10）:** 本書の実装事項A・Bは完了済みである。現在状態、採用済み判断、
> BatchNorm recalibration以降の実施順は`stage5_revision_management_record.md`を正とし、本書は
> A・Bの仕様・実装・検証履歴として参照する。

## 0. この文書の使い方

この文書は、Stage 5の実装と検証を別チャットへ引き継ぐための作業プロンプトである。
新しい実装チャットでは、最初に本書と「1. 関連文書」に示す正本文書を読み、既存コードを
確認してから作業を開始すること。

チャットの役割を次のように分ける。

- 実装チャット: checker実装、synthetic/static test、実行bash作成、ユーザー実行結果の機械的解析、検証レポートへの事実記録を担当する。
- 方針管理チャット: 検証結果に基づく採用aggregation、再学習、BatchNorm対策などの最終判断を担当する。

実装チャットは、本書で指定した診断を完了する前にproduction推論方式を変更したり、200 epoch
学習を開始したりしないこと。結果から次の方針判断が必要になった時点で、成果物と要約を
方針管理チャットへ返す。

### 0.1 実装前確認事項への確定回答（2026-09-09）

実装チャットから提示されたF1からF3は、次の内容で確定した。

- F1: `edge_distance`は事前bucket化せず、生の整数値をgroup keyとして保存する。
- F1: window境界距離と動画内時間位置は混合せず、独立した2つの内訳表として出力する。
- F1: 動画内時間位置は生の`frame_order`で動画間集約せず、0から1へ正規化したdecileを使用する。
- F2: 既存評価の`${EVALUATION_DIR}/best/h5_metrics.csv`と突き合わせるOption Aを採用する。
- F2: TP/FP/TN/FNは完全一致を要求する。差が出た場合は許容誤差で通さず、差分artifactを保存してcheckerを停止し、CUDA非決定性と実装差を切り分ける。
- F3: aliasを明示指定する薄いPLY export optionを今回実装し、既定値はoffとする。
- F3: 未検証code pathにはしない。固定train sanity 1件でoptionのsmoke testを行い、期待する方式別probability/positive-only PLYの存在とheaderを確認する。
- F3: full runでは固定train sanity aliasだけを指定できるようにし、validation PLYは定量結果から対象を選んだ後に必要なH5だけ再実行する。

F3のテスト追加以外は、実装チャットの推奨内容をそのまま採用する。

### 0.2 実装事項Aのコード実装完了（2026-09-09）

次の2ファイルを新規作成した。

```text
Stage5/checks/real_h5/check_stage5_overlap_aggregation.py
Stage5/checks/real_h5/check_stage5_overlap_aggregation.sh
```

実装チャットのdevコンテナ（numpy/torch/h5pyが無い）で完了した検証。

- `python3 -m py_compile check_stage5_overlap_aggregation.py` 合格。
- `bash -n check_stage5_overlap_aggregation.sh` 合格。

実装の要点。

- per-point accumulator（vote_count、prob_sum/sum_sq、prob_min/max、positive_vote_count、
  center_weighted_sum/weight_sum、center_nearest_prob/distance/window_id、min_edge_distance）を
  `OverlapAccumulator`として実装し、windowごとにNumPy fancy indexingでH5内O(N)更新する。
  モデルforwardは1 windowにつき1回、1 H5全体でも4方式分の再実行はしない（制約1）。
- mean baseline parityは`evaluate_stage5.compute_metrics`をそのまま再利用し、
  `--reference_h5_metrics_csv`（既存の`${EVALUATION_DIR}/best/h5_metrics.csv`）とTP/FP/TN/FNを
  完全一致比較する（F2で確定したOption A）。不一致時は
  `private_output_dir/mean_baseline_parity_failure.csv`へ差分を保存し、
  `ParityMismatchError`で即座に停止する。
- `min_edge_distance`はbucket化せず整数値のままgroup keyにし、動画内時間位置は
  `disagreement_by_relative_frame_decile.csv`として境界距離表と別に出力する（F1）。
- PLY exportは`--export_ply_alias`（alias明示指定、既定off）として実装し、要求aliasが
  実際に処理されたか事前・事後の両方で照合するfail-fastチェックを入れた（F3）。
- alias付与は`export_anonymized_stage5_metrics.build_aliases`を再利用。全H5を読み込む前に、
  H5ルート属性（`h5py.File(...).attrs`）だけを読む軽量pre-passでvideo_nameを解決し、
  点群配列の二重読み込みを避けている。
- Step A1のsynthetic testは`--self_test`引数として同スクリプト内に実装した。次の8ケースを
  カバーする: 基本累積(vote_count/sum/sum_sq/min/max/positive_vote_count)、2-vote統計
  (mean/std/range)、all_negative/all_positive/disagreement分類、center_nearestと
  center_weightedの計算、center_nearestタイブレーク（順方向・逆方向で結果が変わらないこと）、
  tail windowのedge_distance、vote_count=1での4方式一致、probability=0.5のtieがbackgroundに
  なること。

Step A1完了（2026-09-09、ユーザー実機）。

```text
$ SELF_TEST=1 bash checks/real_h5/check_stage5_overlap_aggregation.sh
Stage5 overlap aggregation self-test (Step A1)
Stage5 overlap aggregation self-test passed.
```

8ケース全て合格。accumulator蓄積、mean/std/range、all_negative/all_positive/disagreement分類、
center_nearest/center_weighted、center_nearestタイブレーク、tail windowのedge_distance、
vote_count=1での4方式一致、probability=0.5のtie、いずれも実装通りの結果になることを確認した。

Step A2試行時に判明した`.sh`の不具合を修正した（2026-09-09）。空の`VAL_LIST`
（train-sanityのみのsmoke run用）を渡すと、存在確認が`-s`（非空要求）になっていたため
`Validation list not found or empty`で誤って停止していた。`check_stage5_overlap_aggregation.sh`の
検証を`-f`（存在確認のみ）へ変更し、空ファイルの場合は停止せず警告表示のみとした。
Python側`read_path_list`は元々空ファイルを許容しており（0件のリストを返す）、
`main()`側も`selected_train_paths`か`val_paths`のどちらかが非空であれば処理を続行する
設計だったため、この修正でPython側の変更は不要だった。

Step A2 smoke run（train sanity 3件、PLY: `train_sanity_fixed_001`）が完走した
（2026-09-09、ユーザー実機、`Stage5 overlap aggregation checker passed.`）。ただし完走直後の
レビューで、共有可能とするはずの`share_output_dir/overlap_aggregation_summary.json`に
checkpointと`reference_h5_metrics_csv`の**絶対パス**が含まれていることを発見した。制約9・
8章の「共有しないもの: 元H5や絶対データパスの一覧」に反する実装上の不備であり、
ユーザーへ共有を依頼する前にこちらで検出・修正した。

修正内容（2026-09-09）。

- `overlap_aggregation_summary.json`（share側）は`checkpoint_name`・`run_name`
  （いずれもbasenameのみ）と`mean_baseline_parity_reference`（`h5_metrics.csv`という
  ファイル名のみ）に置き換え、絶対パスを含めないようにした。
- 絶対パス（checkpoint、reference CSV、selected_train_list、val_list）は
  `private_output_dir/overlap_aggregation_run_info_DO_NOT_SHARE.json`へ分離した。
- parity失敗時の`summary["parity_failure_csv"]`も絶対パスをやめ、ファイル名のみの参照に変更した。
- 実行完了時に`export_anonymized_stage5_metrics.assert_share_bundle_anonymous`を再利用した
  自己検査を追加し、`share_output_dir`配下に絶対パス・タイムスタンプ型video ID・private
  mappingが残っていないことをchecker自身が毎回確認するようにした（8章の検査項目と同じ基準）。

この修正により、直前のsmoke runで生成された`share_metrics/overlap_aggregation_summary.json`
（絶対パスを含む版）は共有不可であり、再実行して置き換える必要がある。CSV各種
（`overlap_probability_statistics.csv`等）はvideo_aliasのみを含み、この問題の影響を
受けていない。

修正後にoutput directoryを削除して再実行し、Step A2完了を確認した（2026-09-09、
ユーザー実機）。

- `Stage5 overlap aggregation checker passed.`まで到達し、anonymization self-check
  （`assert_share_bundle_anonymous`）もエラーなく通過した。`overlap_aggregation_summary.json`に
  絶対パスは含まれていない。
- mean baseline parity合格。train sanity（3動画合計、mean_probability方式）の
  `TP=407, FP=16602, TN=600976, FN=9349`が、本書2.5節に既に記録済みのbest.pt評価結果と
  完全一致した。checkerの独立実装が本番`evaluate_stage5.py`の結果を正確に再現できることを
  確認した。
- PLY smoke test合格。`train_sanity_fixed_001`について4方式×probability/positive-onlyの
  8ファイルが生成され、全てPLY headerとして正しい形式
  （`ply`/`format ascii 1.0`/`element vertex N`/xyz+rgb properties/`end_header`）だった。
  probability PLY（全点）は4方式とも188301頂点で一致（同一動画・同一点群のため一致が正しい）。
  positive-only PLYの頂点数はmean(13161) < center_weighted(14218) < center_nearest(17251) <
  max(22408)であり、3動画合計のcheckpoint_summaryにおける`predicted_positive_count`の並び
  （17009 < 18305 < 22959 < 30356）およびtp_zero_video_count（mean/center系=1動画、
  max=0動画）と整合していた。3動画のsmoke sampleに留まるため結論扱いはしないが、
  調査対象の仮説（mean集約によるpositive抑制）の方向性と一致する参考情報として記録する。

Step A3（full run、train sanity 3件+validation 18件、計21動画）完了を確認した
（2026-09-09、ユーザー実機）。`Stage5 overlap aggregation checker passed.`まで到達し、
anonymization self-checkも合格した。`share_metrics/`一式（8ファイル）を
`/workspace/.tmp/overlap_aggregation_share/`へ配置してもらい、実装チャット側でも
絶対パス・timestamp型video IDが含まれていないことを独立に`grep`確認した上で内容を精査した。

mean baseline parityはvalidation split（TP=2691, FP=85785, TN=6421265, FN=66760,
valid=6576501, ignore=19050）についても本書2.5節の記録値と完全一致した。固定train sanity動画
単体のF1（0.0293）も9.4節の記録値と一致した。

Step A4として、`stage5_pointnext_s_training_evaluation_report.md`の9.5節へ
「実装事項A: overlap probability / aggregation checker 検証結果（2026-09-09）」を追記した。
suppressed-positive率（GT positive点でtrain 21.54%/validation 10.56%）、video別disagreement率
（0.2%〜79.7%と大きくばらつく）、`min_edge_distance`別disagreement率がほぼ一定（5.9〜6.0%、
window境界仮説を支持しない）、動画内相対位置別disagreement率が後半で緩やかに上昇する傾向、
4 aggregation方式のsplit集約比較表（F1/IoU/recall/FP/TP0動画数）を記録した。詳細は同節を参照。

未実施（GPU・`/mnt/data`・numpy/torch/h5pyが必要で、実装チャットのdevコンテナでは実行できない）。

- 実装事項B（BatchNorm train/eval parity）: 6章の仕様通り、事項Aの記録後に着手する。
  方針管理チャットが事項Aの結果を確認し、着手を承認してからとする。

ユーザー実機で次の順に実行する（Step A1〜A3は完了済み、参考として残す）。

```bash
cd /mnt/data/3d_projects/models/Stage5

# Step A1: synthetic accumulator test（checkpoint/H5不要、numpyのみで完結）
SELF_TEST=1 bash checks/real_h5/check_stage5_overlap_aggregation.sh

# Step A2: train sanity 3件のみのsmoke run + 固定train sanity 1件でPLY smoke test
#   （validationを一時的に空リストにしてGPU時間を絞る）
: > /tmp/stage5_empty_val_list.txt
VAL_LIST=/tmp/stage5_empty_val_list.txt \
EXPORT_PLY_ALIASES="train_sanity_fixed_001" \
bash checks/real_h5/check_stage5_overlap_aggregation.sh

# Step A3: full run（既定のRUN_DIR/EVALUATION_DIRのまま、train sanity 3件+validation 18件）
bash checks/real_h5/check_stage5_overlap_aggregation.sh
```

Step A2・A3とも、事前に`evaluate_stage5.sh`でbest.ptの評価が完了しており
`${EVALUATION_DIR}/best/h5_metrics.csv`と`${EVALUATION_DIR}/evaluation_data/summary.json`が
存在している必要がある（mean baseline parityの参照元）。

### 0.3 方針管理チャットの判断（2026-09-10）

実装事項Aの結果報告に対する方針管理チャットの判断は次の通り。

1. production aggregationは変更しない。現行の`mean_probability`を維持する。
   理由: `max`はFP増加が大きく、`center_nearest`もvalidation FPが約2.4倍。`center_weighted`は
   TP0動画数が悪化しており、明確な採用候補がない。
2. `center_weighted`の重み式は再設計しない。理由: 境界距離との相関が無く現在の重み式に
   根拠がない。動画後半ほど不一致が多いという結果も「どちらのwindow予測が正しいか」を
   示すものではないため、時間位置による重み付けにも直結させない。
3. 次は実装事項B（BatchNorm train/eval parity）へ進む。best.ptを対象に、再学習不要で
   label policy/class weight実験に残る交絡を先に切り分ける。
4. Ablationの順序: 事項B → Label policy ablation（9.6節） → Class weight比較（9.7節）。
   200 epoch学習はまだ行わない。

実装指示（2026-09-10）。

- 実装事項Aは受入完了。production aggregationはmeanのまま変更せず、center weightingも
  再設計しない。
- 実装事項Bへ進む。best.ptを対象に、独立model instanceによる`eval()`と`train()`の同一window
  比較を行い、train sanity 3件とvalidation 18件を評価する。`dropout=0`をassertし、checkpointや
  productionコードは変更しない。probability差、class disagreement、window/mean集約metrics、
  TP0動画数、BatchNorm running statisticsを匿名化して記録する。
- 判定後の分岐: train/eval差が大きい場合はLabel policy実験前にBatchNorm対策を検討する。
  差が小さい場合はBatchNormを主要因から外し、同一split・初期重み・seedによる5 epochの
  Label policy ablationへ進む。
- aggregation方式の優劣はモデルを再学習すると変わり得るため、最終モデル確定後に改めて比較する
  （今回の4方式比較結果を恒久的な結論として扱わない）。

## 1. 関連文書と読み方

### 1.1 Stage 5の現在状態を示す正本

- `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
  - Stage 5調査結果の正本。
  - 特に9.1のbatch integrity、9.2のpadding parity、9.4のpadding-free baseline、
    9.5のoverlap検証方針を読むこと。
  - 「Teacher v6 5 epoch pilot結果」と「Teacher v6 5 epoch checkpoint評価」が直近状態である。
  - 今後の検証結果も、既存記録を消さず同文書へ追記する。

- `docs/stage5/FILES.md`
  - Stage 5のコード、checker、work directoryの配置案内。
  - 新規checkerは`Stage5/checks/real_h5/`へ配置する。

### 1.2 Stage 4 teacher v6の正本

- `docs/stage2to4/stage4/stage4_cvat_snapshot_authoritative_label_revision_plan.md`
  - CVAT snapshot対象frameではCVAT maskをpositive authorityとする契約の正本。

- `docs/stage2to4/stage4/stage4_deleted_xml_annotation_invalidation_plan.md`
  - 削除済みXMLをframe-level tombstoneとして反映したteacher v6の正本。
  - v6は最終受入済みで、Stage 5学習利用可能である。

- `docs/stage2to4/stage4/stage4_phase5_fullvideo_cvat_review_implementation.md`
  - CVAT full-video reviewの実装履歴。
  - 冒頭の「最新のlabel authority方針」を優先すること。
  - 「129 targetだけ反映」「contextは反映しない」などの古い記述は履歴であり、現在仕様ではない。

### 1.3 履歴資料

- `docs/stage5/TRAINING_IMPROVEMENT_PLAN.md`
  - 初期調査時の計画であり、履歴資料としてのみ参照する。
  - grid=1、empty-valid window、paddingありbatchを前提とする箇所は最新状態ではない。
  - 現在の優先順位はStage 5評価レポートの9.5を正とする。

- 過去runのmetricsや一時共有artifact
  - teacher v2やpaddingありrunを現在baselineと誤認しないこと。
  - `/workspace/.tmp/`は一時共有場所であり、正本文書として扱わない。

## 2. 現在の確定状態

### 2.1 データセット

現在のStage 5入力正本はteacher v6である。

```text
/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/
  global_local_l75_w31_c12_area15_bboxrank_v6_cvat_authoritative_xml_invalidation_v1/
    collected/
```

H5 pattern:

```text
*_pointcloud_annotated_foreground_combined_v2_global_local_l75_w31_c12_area15_bboxrank_v6_cvat_authoritative_xml_invalidation_v1.h5
```

固定件数と契約:

- H5は181動画。
- 59動画・3014 Task frameでCVAT snapshotをpositive authorityとして使用。
- XML invalidationは2動画・7 frame。
- invalidationによりpositive 2124点とBBox 7行を除去。
- invalidated frameは全点`point_label=0`かつ`valid_mask=True`。
- crop不良の`20250626_090758_8000`は除外済み。
- `train_stage5.sh`のteacher v6 provenance preflightは1 epoch/5 epoch runで合格済み。

### 2.2 現在のモデルと学習条件

- model: official OpenPoints PointNeXt-Sを使う`pointnext_s` wrapper
- input features: `intensity,confidence`
- window: frame_order基準、size 16、stride 8、tailあり
- physical batch size: 1
- gradient accumulation: 8 windows
- padding: なし
- loss: class-weighted CrossEntropyLoss
- class weight: auto、`[0.0596385561, 1.9403614998]`
- label smoothing: 0.0
- optimizer: AdamW、lr `1e-3`、weight decay `1e-4`
- initial weight: S3DIS PointNeXt-Sからshape一致部分を転移したcheckpoint

Datasetはwindowごとに元H5内の`point_indices`を保持している。batch integrity testはtrain/validation
全件で合格済みであり、H5、window、point index、labelの取り違えは強く除外されている。

padding parity testでは、可変長sampleのzero paddingが実点logits、gradient、BatchNorm統計へ
大きく影響することを確認した。このため現在はphysical batch size 1とgradient accumulationを使い、
train/validationともpadding point countは0である。

### 2.3 現在のtraining run

```text
RUN_DIR=/mnt/data/3d_projects/stage5_runs/260908/pointnext_s_EX260908_260711_w16_s8_bboxrankv6_cvatxmlinv_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad
```

- `best.pt`: epoch 4、学習中validation window IoU最大
- `last.pt`: epoch 5
- train: 163 files / 729 windows
- validation: 18 files / 79 windows
- `train_files.txt`と`val_files.txt`はrun directory内に保存済み

このrunは5 epochすべてで次を満たした。

- padding point count 0
- empty-valid window 0
- train microbatch 729
- optimizer step 92 = `ceil(729 / 8)`
- confusion matrix、valid/ignore/total、weighted loss normalizerが再計算値と一致

### 2.4 現在のevaluation run

```text
EVALUATION_DIR=/mnt/data/3d_projects/stage5_evaluations/260908/pointnext_s_EX260908_260711_w16_s8_bboxrankv6_cvatxmlinv_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad
```

既存評価対象:

- train sanity: 固定1動画`20250626_124212_7300` + seed固定random 2動画
- validation: `val_files.txt`の全18動画
- checkpoint: `best.pt`と`last.pt`
- inference mode: `model.eval()`、windowごとにphysical batch size 1
- current aggregation: 元H5の同一点に対するclass probabilityの単純平均

再利用するlist:

```text
${RUN_DIR}/train_files.txt
${RUN_DIR}/val_files.txt
${EVALUATION_DIR}/evaluation_data/selected_train_files.txt
```

### 2.5 直近の評価結果

| checkpoint | train window F1 / IoU | train mean集約 F1 / IoU | val window F1 / IoU | val mean集約 F1 / IoU |
| --- | ---: | ---: | ---: | ---: |
| best, epoch 4 | 0.1026 / 0.0541 | 0.0304 / 0.0154 | 0.0455 / 0.0233 | 0.0341 / 0.0173 |
| last, epoch 5 | 0.0363 / 0.0185 | 0.0140 / 0.0070 | 0.0394 / 0.0201 | 0.0314 / 0.0160 |

`best.pt`の現行mean aggregation baseline count:

```text
train sanity:
  TP=407, FP=16602, TN=600976, FN=9349
  valid=627334, ignore=4966

validation:
  TP=2691, FP=85785, TN=6421265, FN=66760
  valid=6576501, ignore=19050
```

重要な観測:

- bestのtrain sanity recallはwindow集計14.19%からmean集約4.17%へ低下。
- bestのvalidation recallはwindow集計8.54%からmean集約3.87%へ低下。
- validation 18動画中10動画でTP 0、video-level median F1は0。
- validationのGT-positive window 54個中36個でTP 0。
- negative-only window 25個中16個にbestのFPが存在。
- ignore上のbest positive予測はtrain 2/4966点、validation 520/19050点。
- valid background FPはtrain 16602点、validation 85785点であり、ignoreだけではFPを説明できない。

現時点の第一検証候補は、同一点に対するoverlap window間予測の不一致と、単純平均による
positive予測の打ち消しである。BatchNormによる全面的なeval崩壊は第一候補ではないが、まだ
棄却していない。

## 3. 実装上の制約

1. 事項1から3は同じforward結果から計算し、方式ごとにモデルを再実行しない。
2. まず`best.pt`だけを対象とする。`last.pt`は必要になった場合のみ追加する。
3. productionの`infer_stage5.py`、`evaluate_stage5.py`の既定aggregationは診断結果が出るまで変更しない。
4. 再学習、loss変更、class weight変更、threshold tuningを行わない。
5. Focal Loss、Dice Loss、Hard Negative Mining、random point samplingを追加しない。
6. point cloudやH5を書き換えない。入力はread-onlyで扱う。
7. 1 H5ずつ処理し、全動画のper-point配列を同時に保持しない。
8. 既存のdirty worktreeやユーザー変更をrevertしない。変更はStage 5 checker、必要な共有export、レポートへ限定する。
9. 元動画名、絶対パス、座標、per-point probabilityを共有用artifactへ含めない。
10. checker完走と仮説支持・棄却を区別して報告する。

binary class判定は既存の`np.argmax([p0, p1])`と一致させる。厳密なtieではbackgroundとなるため、
positive判定をスカラーで実装する場合は`p1 > 0.5`を使用する。

## 4. 実装事項A: overlap probability / aggregation checker

次を新規作成する。

```text
Stage5/checks/real_h5/check_stage5_overlap_aggregation.py
Stage5/checks/real_h5/check_stage5_overlap_aggregation.sh
```

既存実装から再利用するもの:

- `evaluate_stage5.py`のcheckpoint/model構築
- H5 loaderとfeature構築
- `generate_frame_order_windows`
- `point_indices_for_window`
- metrics計算
- train sanity/validationのlist

### 4.1 事項1: 重複点のwindow間確率分布

各H5について、元H5内の点indexごとに次を蓄積する。

```text
vote_count
probability_sum
probability_sum_squared
probability_min
probability_max
positive_vote_count
```

導出値:

```text
probability_mean = sum / vote_count
probability_std  = sqrt(max(sum_squared / vote_count - mean^2, 0))
probability_range = max - min
positive_vote_ratio = positive_vote_count / vote_count
```

集計区分:

- valid background
- valid positive
- ignore
- 全valid点
- vote count 1
- vote count 2以上

最低限記録する統計:

- point countとoverlap point count/rate
- mean/std/rangeの平均
- std/rangeのp50、p90、p95、p99
- `max_probability > 0.5`かつ`mean_probability <= 0.5`となるsuppressed-positive数
- split別、匿名video別、GT class別

### 4.2 事項2: window間class不一致

vote count 2以上の点を次へ分類する。

```text
all_negative: 全windowでbackground
all_positive: 全windowでpositive
disagreement: positiveとbackgroundがwindow間で混在
```

次を出力する。

- GT positive/background/ignore別のclass disagreement count/rate
- disagreement点のうちmean aggregationでbackgroundとなる点数
- positive vote ratioの分布

境界距離と動画内時間位置は別の要因なので、次の2表へ分離する。

```text
disagreement_by_edge_distance.csv
disagreement_by_relative_frame_decile.csv
```

`disagreement_by_edge_distance.csv`では、各window内のvoteについて次を計算する。

```python
edge_distance = min(
    frame_order - window_start,
    window_end - frame_order,
)
```

同一点が複数windowへ含まれる場合、その点のgroup keyには全vote中の最小値
`min_edge_distance`を使用する。値は事前bucket化せず、生の非負整数として記録する。
size 16 / stride 8では通常0から3が中心になるが、tail windowや将来のwindow設定を考慮し、
値域をhard-codeしない。

`disagreement_by_relative_frame_decile.csv`では、動画ごとに次の相対位置を計算する。

```python
relative_frame_position = (
    (frame_order - video_min_frame_order)
    / (video_max_frame_order - video_min_frame_order)
)
frame_decile = min(floor(relative_frame_position * 10), 9)
```

`video_max_frame_order == video_min_frame_order`の場合は`frame_decile = 0`とする。動画長が異なるため、
生の`frame_order`を動画間で直接group化しない。境界距離表とrelative decile表は、いずれも
split、GT class、group keyごとのpoint count、disagreement count/rateを最低限含める。

### 4.3 事項3: 4 aggregation方式の比較

同じforward結果から次を計算する。

1. `mean_probability`: 現行単純平均
2. `max_probability`: positive probabilityの最大値
3. `center_nearest`: 点のframe_orderにwindow中心が最も近いwindowの予測
4. `center_weighted`: window内部ほど大きい重みでの加重平均

`center_nearest`のwindow中心は`(start_frame + end_frame) / 2`とし、距離が同じ場合は小さい
`window_id`を選ぶ。

`center_weighted`の初期重みは次に固定する。端でも重みを0にしない。

```python
weight = 1 + min(
    frame_order - window_start,
    window_end - frame_order,
)
```

各方式について、split集約とvideo別に次を出力する。

- TP、FP、TN、FN
- precision、recall、F1、femur IoU、FPR、FNR
- predicted positive count/rate
- video-level mean/median F1、IoU
- TP 0のvideo数
- GT-positive windowまたはvideoの検出率

現行`mean_probability`は、同じprocess内で`predict_h5`を再実行して作る自己参照値ではなく、
既存評価の`${EVALUATION_DIR}/best/h5_metrics.csv`と突き合わせる。匿名aliasとsplitを対応付け、
video別およびsplit集約のTP/FP/TN/FNを完全一致させる。丸めを許容するのはcountから導出した
float metricだけとする。

不一致の場合は許容誤差で通さず、他方式の比較を採用可能な結果として扱わない。最低限、
alias別count差分、checker設定、比較元CSV情報をparity failure artifactへ保存して停止する。
必要ならthreshold 0.5近傍の点数を追加診断として記録し、その後にCUDA非決定性と実装差を
切り分ける。事前にcount toleranceを導入しない。

`max_probability`はrecall上限を見る診断用であり、FPが大幅に増える場合は採用候補にしない。
方式比較ではF1/IoUだけでなくFP増加、TP 0動画数、固定train sanityの変化を併記する。

### 4.4 推奨出力

```text
overlap_aggregation_summary.json
overlap_probability_statistics.csv
overlap_class_disagreement.csv
disagreement_by_edge_distance.csv
disagreement_by_relative_frame_decile.csv
aggregation_checkpoint_summary.csv
aggregation_h5_metrics.csv
aggregation_comparison.csv
```

ローカル確認用として、匿名aliasを明示指定する薄いPLY export optionを追加する。例えば
repeat可能な`--export-ply-alias`とし、未指定時はoff、既定出力はmetricsだけとする。指定aliasでは
4方式それぞれのprobability PLYとpositive-only PLYを、同じforwardで得た集約結果から出力する。
既存のPLY writerを再利用し、PLYを共有bundleへ含めない。

共有用artifactには既存命名と同じ匿名aliasを使う。

```text
train_sanity_fixed_001
train_sanity_random_001
train_sanity_random_002
validation_001 ... validation_018
```

元IDとの対応はprivate outputへ分離し、共有しない。

## 5. 実装事項Aのテストと実行順

### Step A1: synthetic accumulator test

小さい人工frame/window/probability配列で次を検証する。

- vote count、sum、sum squared、min、max
- mean、std、range
- all-negative/all-positive/disagreement
- mean/max/center-nearest/center-weighted
- center-nearest tie break
- tail window
- vote count 1の点は全方式で元予測と一致
- probability 0.5のtieはbackground
- PLY export aliasの選択と方式別出力filename

### Step A2: static/smoke test

- Python compile
- bash `-n`
- 既存dummy testへの回帰がないこと
- train sanity 1件またはvalidation 1件でsmoke run
- mean baseline parity assertion
- NaN/Infなし
- 固定train sanity 1件でPLY optionを有効化し、4方式 x probability/positive-onlyの8ファイルを確認
- 各PLYのheader、元点群を保持するPLYのvertex count、positive-only PLYのvertex countを確認

workspace環境から`/mnt/data`へアクセスできない場合、実データrunを無理に行わず、ユーザー向け
実行コマンドを提示する。

### Step A3: full run

- `best.pt`
- train sanity 3件
- validation 18件
- GPU、physical batch size 1
- 全方式を1回のforwardから算出
- PLYが必要な場合も固定train sanity aliasだけを指定し、validation PLYは定量結果を見て後から対象を選ぶ

### Step A4: 結果記録

Stage 5評価レポート9.5へ次の形式で追記する。

```text
実施日
対象run/checkpoint
checkerの完走状態
mean baseline parity
事項1の結果
事項2の結果
事項3の比較表
仮説に対する暫定判断
方針管理チャットへ返す判断事項
```

## 6. 実装事項B: BatchNorm train/eval parity

事項Aの結果記録後に実装する。

```text
Stage5/checks/real_h5/check_stage5_batchnorm_mode_parity.py
Stage5/checks/real_h5/check_stage5_batchnorm_mode_parity.sh
```

### 6.1 比較条件

同じ`best.pt`から独立したモデルinstanceを2個構築する。

- A: `model.eval()`で保存済みrunning statisticsを使用
- B: `model.train()`で各windowのbatch statisticsを使用
- `dropout=0.0`であることをconfigからassert
- `torch.inference_mode()`を使用し、backward/optimizer updateを行わない
- Bが更新するrunning bufferがAや後続caseへ影響しないようinstanceを分離する
- 同じ入力window、同じpoint順序、physical batch size 1で比較する

### 6.2 記録内容

- positive probabilityのmean/max absolute difference
- p50、p90、p95、p99 absolute difference
- class disagreement count/rate
- GT positive/background/ignore別の差
- window単位F1/IoU
- 現行mean aggregation後のF1/IoU
- BatchNorm module数
- running mean/variance、`num_batches_tracked`のfinite/妥当性

必要なら、direct parityで大差が出た場合だけ、train H5をforwardしてrunning statisticsを再計算した
コピーを評価するBN recalibration診断を追加する。recalibrated checkpointをproductionへ保存・採用する
判断は方針管理チャットへ戻す。

### 6.3 判定

- train-statistics側だけ大幅に良い: BatchNorm running statisticsまたはphysical batch size 1の影響を支持。
- 両modeが近い: BatchNormを主要因から下げ、overlap不一致とper-window検出不足を優先。
- mode差とaggregation差が両方大きい: それぞれを独立要因として報告し、一度にproduction変更しない。

結果はStage 5評価レポート9.5へ、事項Aと同じ形式で追記する。

### 6.4 コード実装完了（2026-09-10）

次の2ファイルを新規作成した。

```text
Stage5/checks/real_h5/check_stage5_batchnorm_mode_parity.py
Stage5/checks/real_h5/check_stage5_batchnorm_mode_parity.sh
```

devコンテナで完了した検証。

- `python3 -m py_compile check_stage5_batchnorm_mode_parity.py` 合格。
- `bash -n check_stage5_batchnorm_mode_parity.sh` 合格。

実装の要点。

- 同じ`best.pt`から`evaluate_stage5.model_from_checkpoint`を2回独立に呼び出し、eval-mode
  instance（A）とtrain-mode instance（B、`.train()`を明示的に呼ぶ）を構築する。
  `load_state_dict`は値をコピーするため、同一checkpoint dictを2回使ってもA/B間でtensorは
  aliasしない。dropoutはconfigから`0.0`であることをassertする。
- 各windowについて、同じ`points`/`features`/`window`、同じRNG seed（`set_seed`、
  `checks.real_h5.check_stage5_padding_parity`から再利用）でAとBを`torch.inference_mode()`下で
  forwardし、直接差分（positive probabilityのabs diff、class disagreement、GT class別内訳）を
  window出現単位で記録する。同時に、window単位の混同行列（重複除去なし）と、元点ごとの
  mean probability集約後の混同行列（現行production集約と同じ方式）を、A/B独立に集計する。
- A（eval-mode、mean集約後）の結果は、事項Aで実装した`load_reference_h5_metrics`/
  `check_mean_probability_parity`（`checks.real_h5.check_stage5_overlap_aggregation`から再利用）で
  既存`evaluate_stage5.py`の記録値と完全一致することを確認する。Aは本来production評価と
  同一の経路のはずであり、これが崩れている場合はA/B比較そのものが無効であるため、
  事項Aと同じくhard gateとして扱い、不一致ならparity failure artifactを保存して停止する。
- BatchNorm moduleごとのrunning_mean/running_var finite性、running_var負値件数、
  `num_batches_tracked`を記録する`inspect_batchnorm_modules`を新規実装した（eval-mode
  instanceから取得。eval-mode forwardはrunning bufferを更新しないため、これは
  checkpointに保存された値そのものである）。
- 出力ファイル: `batchnorm_mode_parity_summary.json`、
  `batchnorm_probability_difference.csv`（GT class別diff統計）、
  `batchnorm_window_metrics.csv`、`batchnorm_aggregation_metrics.csv`、
  `batchnorm_checkpoint_summary.csv`（window単位/aggregated両方、TP0動画数含む）、
  `batchnorm_mode_delta.csv`（train－eval差分）、`batchnorm_buffer_report.csv`。
  事項Aと同じくshare/private分離とprivacy self-check（`assert_share_bundle_anonymous`）を
  適用し、絶対パスは`private_output_dir`側のJSONのみに含める（事項Aで発見した不備の再発防止）。
- synthetic self-testを`--self_test`として実装した（BatchNorm buffer異常値検知、diff集計、
  空maskでのNone/0安全処理、mean集約の手計算一致、dropout assertionの動作、計5ケース）。

self-test完了（2026-09-10、ユーザー実機）。

```text
$ SELF_TEST=1 bash checks/real_h5/check_stage5_batchnorm_mode_parity.sh
Stage5 BatchNorm mode parity self-test
Stage5 BatchNorm mode parity self-test passed.
```

5ケース全て合格。BatchNorm buffer異常値検知、diff集計、空maskでのNone/0安全処理、mean集約の
手計算一致、dropout assertionの動作、いずれも実装通りの結果になることを確認した。

train sanity 3件のみのsmoke run完了（2026-09-10、ユーザー実機、
`Stage5 BatchNorm mode parity checker passed.`）。mean baseline parity（eval-modeのmean集約結果と
既存`evaluate_stage5.py`の一致）も合格した。share側出力を`/workspace/.tmp/batchnorm_mode_parity_share/`
へ配置してもらい、絶対パス・timestamp型video IDが含まれていないことを実装チャット側で独立に
`grep`確認した上で内容を精査した。

**train sanityのみだが、極めて大きなmode差が観測された**（train sanity 3動画合計）。

| granularity | mode | TP | FP | FN | recall | F1 | IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| window | eval | 2516 | 28784 | 15210 | 14.19% | 0.1026 | 0.0541 |
| window | train | 11259 | 128376 | 6467 | 63.52% | 0.1431 | 0.0771 |
| aggregated | eval | 407 | 16602 | 9349 | 4.17% | 0.0304 | 0.0154 |
| aggregated | train | 5952 | 84789 | 3804 | 61.01% | 0.1185 | 0.0630 |

aggregated（現行mean集約後）のTP0動画数は、eval-modeで1/3だったのがtrain-modeでは0/3になった。
probability差はGT positive点でmean abs diff 0.19〜0.32、disagreement率21.7%〜64.4%と、
数値誤差ではなく大きな乖離だった。BatchNorm module 17個の`running_mean`/`running_var`は
全moduleでfinite、負のvarianceも無く、`num_batches_tracked`は全module一致（16859）で
異常な値ではなかった。

train sanityの範囲では、6.3節の判定基準のうち「train-statistics側だけ大幅に良い」
（BatchNorm running statisticsまたはphysical batch size 1の影響を支持）に明確に該当する。
ただしtrain sanity（学習に使ったデータ）だけでは汎化の確認にならないため、validationでも
同じ傾向が出るかをfull runで確認する必要がある。

full run完了（2026-09-10、ユーザー実機、`Stage5 BatchNorm mode parity checker passed.`）。
mean baseline parityも合格。**validation（学習に使っていない18動画）でも同じ傾向が確認された**。

| split | granularity | TP0動画数（eval→train） | recall（eval→train） |
| --- | --- | --- | --- |
| train_sanity | aggregated | 1/3 → 0/3 | 4.17% → 61.01% |
| validation | aggregated | 10/18 → 0/18 | 3.87% → 33.12% |

validation全体でGT positive点のdisagreement率は38.15%、mean abs diff 0.260。詳細な比較表と
判定は`stage5_pointnext_s_training_evaluation_report.md` 9.5節「実装事項B: BatchNorm train/eval
parity 検証結果」に記録した。6.3節の判定基準「train-statistics側だけ大幅に良い」に該当し、
BatchNorm running statisticsまたはphysical batch size 1学習の影響を支持する結果が、train sanity
だけでなくvalidationでも再現した。

事項Bはこれで完了。実装・実行・レポート記録が完了したので、方針管理チャットへ結果を返す。

未実施（GPU・`/mnt/data`・numpy/torch/h5pyが必要で、実装チャットのdevコンテナでは実行できない）。

- 6.2節のBN recalibration診断（方針管理チャットがBatchNorm対策の検討を承認した場合のみ着手）。

ユーザー実機で次の順に実行する（self-test・smoke run・full runは完了済み、参考として残す）。

```bash
cd /mnt/data/3d_projects/models/Stage5

# self-test（checkpoint/H5不要、numpyとtorchのみで完結。完了済み）
SELF_TEST=1 bash checks/real_h5/check_stage5_batchnorm_mode_parity.sh

# train sanity 3件のみのsmoke run（validationを一時的に空リストにしてGPU時間を絞る）
: > /tmp/stage5_empty_val_list.txt
VAL_LIST=/tmp/stage5_empty_val_list.txt \
bash checks/real_h5/check_stage5_batchnorm_mode_parity.sh

# full run（既定のRUN_DIR/EVALUATION_DIRのまま、train sanity 3件+validation 18件）
bash checks/real_h5/check_stage5_batchnorm_mode_parity.sh
```

事前に`evaluate_stage5.sh`でbest.ptの評価が完了しており`${EVALUATION_DIR}/best/h5_metrics.csv`と
`${EVALUATION_DIR}/evaluation_data/summary.json`が存在している必要がある
（eval-mode sanity parityの参照元、事項Aと共通）。

## 7. 実行環境

ユーザー実行環境はDockerではなく、SSH接続した実機である。

```text
Python: /home/kodaira/anaconda3/envs/dualtrack311/bin/python
PyTorch: 2.7.1+cu128
GPU: NVIDIA GeForce RTX 5090
CUDA available: True
```

PointNeXt/OpenPointsと`pointnet2_batch_cuda`を使うbashでは次を設定する。

```bash
SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
PYTHON="/home/kodaira/anaconda3/envs/dualtrack311/bin/python"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  CONDA_PREFIX="$(dirname "$(dirname "${PYTHON}")")"
fi

export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="12.0"

TORCH_LIB="$("${PYTHON}" - <<'PY'
import torch
from pathlib import Path
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${SCRIPT_DIR}/external/PointNeXt:${SCRIPT_DIR}/external/PointNeXt/openpoints:${PYTHONPATH:-}"
```

`libtinfo.so.6: no version information available`というbash warningは既知であり、処理が継続する場合は
今回の失敗条件ではない。`pointnet2_batch_cuda`のimport/forwardは確認済みである。

## 8. Privacyと共有

実データ結果を別チャットへ共有するときは、匿名化済みsummary CSV/JSONだけを
`/workspace/.tmp/`へ配置する。

共有してよいもの:

- 匿名aliasだけを含む集約CSV
- 個人・動画識別子を除いたsummary JSON
- 実行設定の匿名版
- checker logの識別情報を除いた部分

共有しないもの:

- private alias mapping
- 元H5や絶対データパスの一覧
- checkpoint
- prediction NPZ/H5
- PLY
- point座標やper-point probability

匿名化exportでは最低限次を検査する。

- original identifiers absent
- timestamp-like video IDs absent
- absolute host paths absent
- share bundleにprivate mappingが含まれない

## 9. 今回の停止条件と方針管理チャットへ返す内容

事項AとBの実装・実行・レポート記録が完了したら、次を要約して方針管理チャットへ返す。

1. mean baseline parityの成否
2. GT positive/background別のwindow disagreement率
3. suppressed-positive数と割合
4. 4 aggregation方式のtrain sanity/validation F1、IoU、FP、recall、TP 0動画数
5. BatchNorm train/eval probability差とclass disagreement率
6. checkerが示した事実と、まだ判断できない事項
7. 変更ファイル、実行コマンド、出力先、テスト結果

この時点では、次のいずれも自動決定・実装しない。

- production aggregationの変更
- center-only loss
- overlap出現回数の逆数weight
- BatchNorm freeze/recalibration/別normへの変更
- 追加学習または200 epoch学習

これらは、方針管理チャットで検証結果を比較した後に決定する。

## 10. 実装品質の完了条件

- 既存コードを読み、共通処理を必要以上に複製していない。
- 事項1から3を同一forwardで計算している。
- mean baseline countが既存評価と完全一致する。
- synthetic testがaggregation式とtail/tieを覆う。
- 実データは1 H5ずつ処理し、メモリ使用量が点数に対して線形である。
- checkerの出力にNaN/Infがない。
- share/private出力が分離されている。
- `git diff --check`、Python compile、bash syntax checkが通る。
- 既存のユーザー変更や無関係ファイルをrevertしていない。
- Stage 5評価レポートへ結果を追記した。
- 最後に、方針管理チャットが判断できる短い結果要約を提示した。
