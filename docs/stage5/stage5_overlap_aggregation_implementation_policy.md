# 実装事項A: overlap probability / aggregation checker 実装方針記録

> **位置づけ注記（2026-09-10）:** 本書は実装事項Aの設計根拠を保存する補助資料である。
> Stage 5全体の現在状態と次の実施順は`stage5_revision_management_record.md`を参照する。

## 0. この文書の位置づけ

この文書は、`stage5_overlap_aggregation_handoff_prompt.md`の実装事項A（4章・5章）を
実装する前に、実装チャットが提示した方針（A〜F）と、その根拠・比較検討をそのまま
記録したものである。確定仕様は`stage5_overlap_aggregation_handoff_prompt.md`
（特に0.1、4章、5章）であり、本書はその設計判断の理由を残すための補助資料である。
本書と正本が食い違う場合は正本を優先する。

## A. ファイル構成と既存コードの再利用方針

新規作成は次の2ファイルのみとする。

```text
Stage5/checks/real_h5/check_stage5_overlap_aggregation.py
Stage5/checks/real_h5/check_stage5_overlap_aggregation.sh
```

既存コードから再利用し、複製しないもの。

| 用途 | 再利用元 |
| --- | --- |
| checkpoint/model構築 | `evaluate_stage5.model_from_checkpoint` |
| window生成・点index取得 | `stage5.utils.frame_windows.generate_frame_order_windows` / `point_indices_for_window` |
| H5読み込み | `stage5.utils.h5_io.load_stage5_pointcloud_h5` / `read_path_list` |
| video名・匿名化 | `evaluate_stage5.path_video_name`、`export_anonymized_stage5_metrics.build_aliases` / `anonymize_metric_rows` |
| 混同行列→指標変換 | `evaluate_stage5.compute_metrics` / `metrics_from_counts` / `MetricTotals` |
| train sanity選定 | `evaluate_stage5.select_train_paths`（既存`selected_train_files.txt`があればそれを使う） |
| PLY出力（optionのみ） | `stage5.utils.visualization_export.write_probability_ply` / `write_selected_probability_ply` |

`compute_metrics`をそのまま再利用することで、mean_probability方式のTP/FP/TN/FNが
`evaluate_stage5.py`と構造的に同じ計算経路を通る。ただしこれだけでは実行を跨いだ
再現性の保証にはならないため、Cで述べる既存CSVとの突合を別途行う。

## B. 事項1〜3のストリーミング計算設計

### B.1 per-point accumulator

forwardループは`evaluate_stage5.predict_h5`と同じwindow列挙構造を使い、確率を
捨てずに次のper-point配列（各H5内でshape `[N]`、O(N)で保持しO(N×window数)には
しない）を更新する。

```text
vote_count
prob_sum, prob_sum_sq
prob_min (+inf初期), prob_max (-inf初期)
positive_vote_count
center_weighted_sum, center_weight_sum
center_nearest_prob, center_nearest_distance (+inf初期), center_nearest_window_id (+inf初期)
min_edge_distance (+inf初期)
```

各windowについて`indices = point_indices_for_window(...)`で得た点集合に対し、
NumPyのfancy indexingで一括更新する（Pythonループなし）。

```python
p1 = probabilities[:, 1]
positive = p1 > 0.5  # np.argmax([p0, p1]) と同義（tie=0.5はbackground）

vote_count[indices] += 1
prob_sum[indices] += p1
prob_sum_sq[indices] += p1 ** 2
prob_min[indices] = np.minimum(prob_min[indices], p1)
prob_max[indices] = np.maximum(prob_max[indices], p1)
positive_vote_count[indices] += positive.astype(np.int32)

edge_distance = np.minimum(
    frame_order[indices] - window.start_frame,
    window.end_frame - frame_order[indices],
)
weight = 1 + edge_distance
center_weighted_sum[indices] += weight * p1
center_weight_sum[indices] += weight
min_edge_distance[indices] = np.minimum(min_edge_distance[indices], edge_distance)

window_center = (window.start_frame + window.end_frame) / 2.0
dist_to_center = np.abs(frame_order[indices] - window_center)
better = (dist_to_center < center_nearest_distance[indices]) | (
    (dist_to_center == center_nearest_distance[indices])
    & (window.window_id < center_nearest_window_id[indices])
)
center_nearest_prob[indices] = np.where(better, p1, center_nearest_prob[indices])
center_nearest_distance[indices] = np.where(better, dist_to_center, center_nearest_distance[indices])
center_nearest_window_id[indices] = np.where(better, window.window_id, center_nearest_window_id[indices])
```

### B.2 導出値

```python
mean = prob_sum / vote_count
std = sqrt(max(prob_sum_sq / vote_count - mean ** 2, 0))
range_ = prob_max - prob_min
positive_vote_ratio = positive_vote_count / vote_count

all_negative = positive_vote_count == 0
all_positive = positive_vote_count == vote_count
disagreement = (vote_count >= 2) & ~all_negative & ~all_positive

suppressed_positive = (prob_max > 0.5) & (mean <= 0.5)

center_weighted_probability = center_weighted_sum / center_weight_sum
```

4方式の点ごとの確率（`mean` / `prob_max` / `center_nearest_prob` / `center_weighted_probability`）を
それぞれ`compute_metrics(labels, valid_mask, pred_label=(p > 0.5), prob_femur=p, ignore_index=...)`へ
渡し、H5単位・split単位（`MetricTotals`）・video単位の指標を得る。1 H5内で4方式まとめて
計算するため、モデルforwardは1回のみで済む（制約1）。

### B.3 edge_distanceの値域（window=16, stride=8での実測的根拠）

`window_size_frames=16`のとき、window内オフセット`offset = f - start`は0〜15、
`edge_distance = min(offset, 15 - offset)`は0〜7の8値のみを取る。

連続する2つのoverlap window（start=s1とs1+8）の両方に入る点について、
window1内のedge_distanceとwindow2内のedge_distanceの和は常に7になる
（`edge_distance_w1 + edge_distance_w2 = 7`）。したがって、vote_count=2の点では
`min_edge_distance`は実質0〜3の4値しか取らず、`max_edge_distance = 7 - min_edge_distance`
という双対関係にある。この事実がF1の結論（bucket化しない、frame_orderとwindow境界距離を
分離する）の根拠になっている。

## C. mean baseline parityの担保方法

`${EVALUATION_DIR}/best/h5_metrics.csv`（既存`evaluate_stage5.sh`のbest.pt評価出力）を
`--reference_h5_metrics_csv`として読み込み、mean_probability方式で計算したTP/FP/TN/FNを
video名で突き合わせて完全一致を要求する。

比較方式として次の2案を検討した。

- **Option A（採用）**: 既存`h5_metrics.csv`（別プロセス・別時刻に完走した本番評価の成果物）と
  突き合わせる。新checkerのwindow生成・feature構築・正規化・model構築が本番`evaluate_stage5.py`と
  実質同一であることを、実行を跨いで検証できる。
- **Option B（不採用）**: 同一プロセス内で`predict_h5`を追加で呼び出し、その場で計算した値と
  比較する。実装差分の検出力がOption Aより弱く（「新checkerが本番と一致するか」ではなく
  「同じ関数を2回呼んで一致するか」の確認になる）、モデルforwardの重複実行にもなる。

Option Aの既知リスク: PointNeXt/OpenPointsのfurthest point sampling・ball queryは
`pointnet2_batch_cuda`のCUDAカーネルを使っており、eval mode・dropoutなし・勾配計算なしでも
スレッド実行順序に依存する浮動小数点差が理論上あり得る。確率がちょうど0.5近傍の点があると、
`p1 > 0.5`判定が実行間でわずかに反転しTP/FP/TN/FNのcountが1点ずれる可能性がある。これは
実装バグとは別カテゴリの事象であり、発生した場合は許容誤差で通さず、差分artifact
（alias別count差分、checker設定、比較元CSV情報）を保存してcheckerを停止し、その後で
CUDA非決定性と実装差を切り分ける。事前のcount tolerance導入はしない。

## D. 出力ファイル設計

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

- `overlap_probability_statistics.csv`: split×匿名video×GT区分(valid_positive/valid_background/
  ignore/all)×vote区分(1/2以上)でmean/std/rangeの平均、std/rangeのp50/p90/p95/p99、
  overlap point数・率、suppressed-positive数。
- `overlap_class_disagreement.csv`: GT positive/background/ignore別のclass disagreement
  count/rate、disagreement点のうちmean aggregationでbackgroundとなる点数、positive vote
  ratio分布。
- `disagreement_by_edge_distance.csv` / `disagreement_by_relative_frame_decile.csv`:
  境界距離（アーキテクチャ仮説）と動画内時間位置（データ/window生成仮説）は原因の種類が
  異なるため、1つの表に混ぜず分離する。境界距離はbucket化せず生の整数`min_edge_distance`を
  group keyにする。時間位置は動画ごとに`(frame_order - video_min) / (video_max - video_min)`を
  計算し、10分位（`frame_decile = min(floor(relative_position * 10), 9)`、
  `video_max == video_min`なら0）でgroup化する。動画長が異なるため生のframe_orderで
  動画間集約はしない。
- `aggregation_checkpoint_summary.csv` / `aggregation_h5_metrics.csv` / `aggregation_comparison.csv`:
  4方式×split（h5_metricsのみvideo単位）でTP/FP/TN/FN/precision/recall/F1/IoU/FPR/FNR/
  predicted positive率/video-level mean・median F1・IoU/TP0動画数/GT-positive検出率。

### PLY export option（F3）

既定offの、aliasを明示指定する薄いoptionとして実装する（例: repeat可能な
`--export_ply_alias`）。指定aliasについて、4方式それぞれのprobability PLYと
positive-only PLYを、事項1-3計算と同じforward結果から出力する。既存の
`write_probability_ply` / `write_selected_probability_ply`を再利用し、PLYは共有bundleへ
含めない。

## E. テスト・実行の役割分担

実装チャットのdevコンテナには`numpy` / `torch` / `h5py`が無く、GPU・`/mnt/data`にも
アクセスできない。このため各Stepの実行主体は次のようになる。

- **Step A1（synthetic accumulator test）**: numpy依存のためdevコンテナでは実行できない。
  期待値をコード内にassertとして埋め込み、ユーザー実機で実行するコマンドを提示する。
- **Step A2（static/smoke test）**: `python -m py_compile`と`bash -n`はdevコンテナで
  実行し検証する。real H5 smoke run、PLY option smoke testはユーザー実機のみ。
- **Step A3（full run）**: GPU・`/mnt/data`が必要なため完全にユーザー実機。実行コマンドを
  提示する。
- **Step A4**: ユーザーが実行したJSON/CSV結果を受け取り、評価レポート9.5への追記文を作成する。

## F. 未確定事項の検討経緯と結論

### F1. edge_distanceのbucket分け

**検討**: 現行設定（window=16, stride=8）でのedge_distanceの値域は0〜7であり、
B.3で示した通りvote_count=2の点では実質0〜3の4値しか取らない。事前提案していた
`0 / 1-2 / 3-4 / 5+`という区切りは値域の実態に対して粗く、根拠のない境界を持ち込む。

「frame_orderとwindow中心/境界距離でgroup化」という原文は、中心距離と境界距離が
`center_distance ≈ (window_size-1)/2 - edge_distance`の線形な双対関係にあることから、
2つの独立軸ではなく「境界距離（アーキテクチャ仮説: window端は文脈が少なく不安定）」と
「動画内時間位置（データ/window生成仮説: tail windowや特定区間への偏り）」という
性質の異なる2軸だと解釈した。

**結論**:
- `min_edge_distance`はbucket化せず生の整数値をgroup keyにする。
- 境界距離表と動画内時間位置表は別々のCSVに分離する。
- 動画内時間位置は動画間で長さが異なるため、生のframe_orderではなく0〜1へ正規化した
  相対位置のdecileを使う。

### F2. mean baseline parityの比較方式

**検討**: Option A（既存`h5_metrics.csv`との突合）とOption B（同一プロセス内で
`predict_h5`を再実行して比較）を比較した。Option Bは実装差分の検出力が弱く、
モデルforwardの重複実行という無駄も生む。Option Aは仕様書4.3の文言
（「既存評価CSVと完全一致」）にも合致し、実行を跨いだ真の回帰・整合性チェックになる。
一方でGPU上の浮動小数点非決定性により、閾値0.5近傍の点でcountが偶発的にずれる
リスクがある。

**結論**: Option Aを採用する。TP/FP/TN/FNは完全一致を要求し、不一致時は許容誤差で
通さず差分artifactを保存してcheckerを停止する。CUDA非決定性の切り分けは不一致が
実際に発生した場合の追加診断とし、事前のtolerance導入はしない。

### F3. PLYオプションの実装タイミング

**検討**: 「必要なら追加する」という条件付き記述と、完了条件（10章）・Step A1〜A4の
テストにPLY関連項目が無いことから、当初は「今回は作らない」「flagだけ用意し既定offで
最小限実装するが未検証のまま残す」の2案を提示した。後者は再利用コストがほぼゼロで
ある一方、テストされないコードパスが残る点が懸念だった。

**結論**: 折衷案として、alias明示指定・既定offのPLY export optionを実装しつつ、
未検証のコードパスにはしない。固定train sanity 1件でoptionを有効化し、4方式×
probability/positive-onlyの8ファイルが期待通り出力されること（存在、header、
vertex count）をStep A2のsmoke testに追加する。Step A3のfull runでは固定train
sanity aliasだけを指定し、validation PLYは定量結果を見て対象を選んだ後に必要な
H5だけ再実行する。
