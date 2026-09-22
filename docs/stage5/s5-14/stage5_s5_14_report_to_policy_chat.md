
# Stage 5 S5-14: 座標依存・Frame-level・Overlap Exposure診断 報告書

作成日: 2026-09-15
作成元: Stage 5実装チャット
状態: **S5-14 core（Step H1〜H5）完了**。GPU側実run・privacy self-checkとも合格。3仮説の
最終判定確定（仮説A強く示唆、仮説B明確に支持、仮説C限定的に支持）。次の改修方針は方針管理
チャットの判断待ち

本書は`docs/stage5/s5-14/stage5_s5_14_structural_diagnostics_implementation_handoff.md`（実装依頼書、正本）を
受けての理解・現状・実装方針の記録として作成し、以後、検証が進むごとに本書へ結果を追記して
方針管理チャットへの完了報告として使用する。実装依頼書13章が別途指定する
`stage5_s5_14_report_to_policy_chat.md`という完了報告ファイル名と一致させてあるため、新しい
完了報告ファイルを別途作成する必要はない。

## 1. 経緯

S5-13本比較・S5-13補足が完了し、方針管理チャットは次の3判断を行った（D-026〜D-028）。

1. W-Cを不採用とし、W-A（`[0.05963856, 1.94036150]`）をS5-14の暫定class weightとして確定する。
2. W-B/W-Cはproduction候補やS5-14の比較armにはせず、S5-15でのcalibration/threshold診断用の
   履歴として保持する。
3. 追加の中間weight探索は行わず、class weight探索を終了してS5-14へ進む（D-029）。

S5-14 coreは、train sanity positive-only PLYで観察された「似たXY位置のpositive集合が異なるframe群へ
2〜3回反復する」現象について、座標事前分布・動画内時間位置・overlap windowによるtraining exposureの
3仮説を、再学習なしで切り分ける診断段階である。

## 2. 目的と検証する3仮説

W-A baseline（S5-12/S5-13で使用したGroupNorm epoch 5 checkpoint、`best.pt`=`last.pt`）の予測を
source pointとframeへ復元し、次を検証する。

- **仮説A（座標事前分布）:** GTの位置と十分に対応せず、modelが動画間で似たXY位置をpositiveにしやすい。
- **仮説B（時間位置・frame phase）:** FP/FN・probability・window disagreementが動画内の特定区間、
  GT-positive区間の前後、動画後半などへ偏る。
- **仮説C（overlap exposure）:** source point/frameのtraining window出現回数の違いが、positive
  probability・FP/FN・window disagreement・同一XY位置での時間的反復と対応する。

## 3. 固定baseline（S5-14 core全体を通して変更しない）

teacher v7（`bboxrank_v7_cvat_authoritative_crop_quality_v1`、180 H5）、S5-12/S5-13 W-A epoch 5
checkpoint（GroupNorm 8 groups、`bbox_noncontour_ignore`、class weight`[0.05963856,1.94036150]`、
window 16/stride 8/tailあり）、`model.eval()`、mean probability aggregation、threshold 0.5、
予測対象は固定train sanity 3動画+validation 18動画全件、GT/exposure監査はteacher v7全180 H5。
W-B/W-Cは診断履歴としてのみ保持し、S5-14 checkerのprimary入力にしない。modelの再学習、class
weightの追加探索、label policy/normalization方式変更、threshold tuning、production設定変更は
行わない。

座標系は次を別々に保持する：(1) H5 raw `pixel_xy`、(2) raw pseudo-3D `points`、(3) model入力相当の
`normalize_xyz(points)`、(4) `frame_order`、(5) 動画内相対frame位置。PLY上のraw XY反復とmodel入力
座標上の依存を同一視しない。

## 4. Step H1・H2: 実装内容と検証結果（完了）

### 4.1 変更・新規ファイル

```text
Stage5/checks/real_h5/check_stage5_frame_spatial_overlap_diagnostics.py/.sh   (新規)
Stage5/checks/dummy/check_dummy_frame_spatial_overlap_diagnostics.py/.sh      (新規)
docs/stage5/FILES.md                                                          (変更)
docs/stage5/stage5_revision_management_record.md                             (S5-14進捗記録)
```

### 4.2 Step H1: 共有primitiveとsynthetic test

依頼書7章Step H1が求める項目のうち、GT/exposure監査（Step H2）に必要な範囲を実装した。

- `find_runs()`: 単調増加のframe値配列をcontiguous runへ分割（gapで区切る）。
- `per_frame_label_counts()`: frameごとのpositive/background/ignore点数（`np.unique`+`np.add.at`で
  ベクトル化）。
- `window_occurrence_counts()`: `stage5.utils.frame_windows.generate_frame_order_windows`/
  `point_indices_for_window`を再利用した、source pointごとの純geometryなwindow出現回数
  （model forward不要）。
- `relative_frame_decile()`: 動画内相対位置のdecile化（0除算とならないよう単一frame動画も処理）。
- `class_exposure_summary()`: class別のunique point数とwindow occurrence総数を明示的に分離して記録
  （依頼書の「unique-point集計とwindow-occurrence集計を混ぜない」要件）。
- `positive_frame_runs()`: GT positiveを含むframeのcontiguous run（数・長さ・間隔）。
- `require_no_duplicate_or_missing_points()`/`require_finite()`: fail-fastヘルパー。

これらはteacher v7 H5・predictionを一切読み込まない、純粋なnumpy関数として実装しており、torch/CUDAに
依存しない。

**synthetic test（`checks/dummy/check_dummy_frame_spatial_overlap_diagnostics.py`）:** 8種類の
単体テスト（`find_runs`の基本/gap/単一/空ケースと不正入力拒否、`per_frame_label_counts`、
`window_occurrence_counts`のoverlap帯の出現回数、`relative_frame_decile`の境界値と単一frame動画、
`class_exposure_summary`の通常/空classケース、`positive_frame_runs`、重複/欠落point・nonfinite値の
fail-fast pass/fail）に加え、手作りの合成H5（24 frame、frame 5-7と15がGT-positiveの2 run）による
`audit_h5()`の統合テストを実装し、**この開発コンテナ内**（一時venvへ`numpy`/`h5py`をインストール）で
全ケース合格を確認した。`py_compile`・`bash -n`・`git diff --check`もすべて合格した。

### 4.3 Step H2: teacher v7全180 H5のGT/exposure監査

`check_stage5_frame_spatial_overlap_diagnostics.py`の`audit_h5()`/`main()`で、H5ごとに以下を計算し
CSV/JSONへ出力する: frame範囲・frame数、window数、positive-frame数、class別のunique point数・
window occurrence総数・exposure倍率、positive frame runの数・長さ・間隔、frame相対decile別の
positive/background点数と平均occurrence。CPU/h5pyのみで動作し、GPU/predictionを必要としない。

なお、依頼書Step H2の「GT positiveのnormalized XY density」は、Step H2.5でcross-video正規化の
寸法契約が確定してから計算する（4.4節参照）。raw XY densityの取得自体はStep H3.1で改めて実装する。

**GPU側実行結果（2026-09-14、ユーザー実機、CPU/h5pyのみで動作）:**

```text
files audited: 180
positive exposure multiplier (mean of per-video means): 1.8516
background exposure multiplier (mean of per-video means): 1.6019
```

依頼書3章に記載されたS5-08時点の概算値（background約1.69倍、positive約1.90倍）と近い水準であり、
teacher v7・window 16/stride 8設定でも同様のoverlap構造が再現されていることを確認した。出力は
`work_dirs/_frame_spatial_overlap_diagnostics/gt_overlap_exposure.csv`/`.json`。

### 4.4 発見事項: cross-video XY正規化に使える寸法がteacher v7最終H5に存在しない

Step H2.5着手前の調査（subagentによるStage 2〜4パイプライン追跡）で、Stage 5が入力とする最終
combined H5（`annotate_pseudo3d_point_cloud.py`が生成）にはcrop/image寸法（幅・高さ）を示す属性が
一切保存されていないことが判明した。crop offset・resize scale等は生成pipeline中間段階の個別動画H5
（`Stage2to4/src/utils/pseudo3d_processing.py`が書き込む`local_crop_top`/`local_crop_left`/
`local_resize_scale`/`local_input_shape`等のroot属性）にのみ存在し、最終H5には引き継がれていない。
crop sizeそのもの（`--local_crop_size`、既定256）は生成run全体で共有される外部CLI定数であり、
H5から読み取れる値ではない。既存`stage5/utils/feature_normalization.py`の`normalize_pixel_xy()`は
観測点のmax値で正規化しており、依頼書が禁止する「観測点のmax値を推測的に代用」に該当し使用できない。

この発見を受け、方針管理チャットはStep H3.1（cross-video XY解析）へ進む前に、中間H5を遡って実寸法を
回収できるか調査するStep H2.5の実施を決定した（詳細は5節）。

## 5. Step H2.5: Stage 4 dataset inventoryとXY座標provenance監査（方針把握のみ、実装は次の段階）

依頼書のStep H2.5は5つのサブステップで構成される。teacher/source H5の書き換えや再生成、再学習、
production変更は行わない、監査専用の段階である。

### H2.5-A: 保存dataset inventory

`/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/`配下のv2以降のStage 4
teacher datasetを列挙し、dataset/run名・teacher世代、H5配置先、H5件数、生成・派生pipeline、元
datasetとの継承関係、point cloud本体を再生成した世代かannotation/labelだけを変更した世代か、
利用可能なrun log/summary/manifestを記録する。絶対pathと実video IDを含むinventoryはprivate、
shareable summaryはdataset token・件数・匿名video aliasのみとする。

### H2.5-B: H5属性とsource provenance

teacher v7全180 H5を主対象に、最終H5のroot属性・`point_cloud` group属性、`source_h5`/
`source_pseudo3d_h5`等のsource参照値、参照先中間H5の存在・可読性、中間H5の
`local_encoder_images.shape`/`local_input_shape`/`local_preprocess`（crop offset、resize scale、
raw width/height）、`pixel_xy`のshape・finite性・軸別min/max、`pixel_xy`がlocal image bounds内に
あること、同一動画・世代間の`pixel_xy`/`points`/`frame_order`/point count一致を監査する。巨大
datasetの一括ロードを避け、hashもchunk単位で計算する。sourceパスが解決できない場合はbasename/video
対応による探索結果を別欄に記録し、暗黙に同一ファイルとみなさない。

事前調査で、`Stage2to4/src/utils/pseudo3d_processing.py`が生成する個別動画の中間H5には上記の
crop/resize属性が実際に残っており、`annotate_pseudo3d_point_cloud.py`が`attrs = dict(f.attrs)`と
して読み込んでいることを確認済みである。したがって選択肢3（中間H5からの実寸法回収）が現実的に
狙える可能性が高いが、`source_h5`/`source_pseudo3d_h5`参照が現存ファイルとして解決できるか、
teacher v7の180ファイル全件でこの経路が有効かは未確認であり、これがH2.5-Bの核心的な調査対象である。

### H2.5-C: 寸法確定の判定順序

1. **選択肢3を優先:** teacher v7全件でsource中間H5を解決でき、実寸法を取得できる場合はそれを
   provenanceの一次根拠とする。
2. **選択肢2を条件付きで使用:** source中間H5が一部/全部失われていても、生成pipeline/run log、
   世代間point-cloud一致、全180 H5の`pixel_xy` boundsから共通local crop寸法を裏付けられる場合
   だけ、`--pixel_width 256 --pixel_height 256`等の明示的run-level定数を使用する。生成CLIの既定値が
   256であることだけでは根拠として不十分（実際にoverrideされていないこと等の複合的裏付けが必要）。
3. **併用する場合:** run全体の共通寸法契約を先に確定し、source H5照合はその実測検証とする。
4. **根拠不足の場合:** 観測点max値・推定crop範囲・動画ごとの便宜的寸法で補わない。該当動画を
   cross-video解析から除外し、全件で確定不能ならcross-video heatmap自体を未実施として同一動画内
   解析のみに限定する。

### H2.5-D: 正規化とfail-fast

寸法を`width`/`height`として確定できた場合、`normalized_x = pixel_x / (width - 1)`、
`normalized_y = pixel_y / (height - 1)`で正規化する（`width > 1`・`height > 1`必須）。全点について
finiteかつ`0 <= pixel_x < width`、`0 <= pixel_y < height`を検証し、bounds違反をclipせずfail-fastする。
synthetic testで四隅・境界直前・bounds外・nonfinite・寸法欠落を網羅する。

### H2.5-E: 必須artifactと停止条件

```text
stage4_dataset_coordinate_inventory.csv       (private)
stage4_xy_source_resolution.csv               (private)
stage4_xy_coordinate_provenance_summary.json  (shareable版も作成)
stage4_xy_dimension_audit.csv                  (匿名化shareable版も作成)
```

summaryには採用した選択肢、寸法、dimension source、全件数、source解決数、実寸法照合数、bounds合格数、
世代間point-cloud一致数、除外数と理由を保存する。契約を満たせる場合はその規約でStep H3.1を実装・
実行し、根拠不足・寸法不一致・座標系混在・bounds違反が見つかった場合は推測で先へ進まず、同一動画内
解析だけを継続して方針管理チャットへ報告する。

## 6. 次のアクション

Step H2.5の実装（新規監査script、synthetic test、CPU/h5py静的検証）に着手する。実装過程で仕様上の
重大な不明点（特にH2.5-Bのsource参照解決可否）が見つかった場合は、実装を進めず事実・選択肢・推奨案を
報告する。以降の実装・検証結果（inventory監査結果、source provenance解決率、採用した寸法決定選択肢、
Step H3以降への影響）は本書へ追記していく。

## 7. 関連文書

- `docs/stage5/s5-14/stage5_s5_14_structural_diagnostics_implementation_handoff.md`（実装依頼書、正本。
  2026-09-15にStep H2.5を追記した版）
- `docs/stage5/s5-13/stage5_s5_13_supplement_report_to_policy_chat.md`（S5-13補足完了報告、W-A/B/C結果）
- `docs/stage5/s5-13/stage5_s5_13_supplement_implementation_handoff.md`（S5-13補足の固定条件・privacy要件）
- `docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.7節（S5-13本比較・補足結果）
- `docs/stage5/s5-08-09/stage5_overlap_aggregation_implementation_policy.md`（source point alignment、
  window accumulator、相対frame decileの既存仕様、Step H4で再利用予定）
- `docs/stage5/stage5_revision_management_record.md`（S5-14進捗、Decision record D-026〜D-029）

---

## 実施記録（以下、検証結果を随時追記する）

### Step H1・H2: 実装・検証結果（完了、2026-09-14）

4節に記載の通り。合成テスト・GPU側実行とも合格。

### Step H2.5: 方針把握（完了、2026-09-15）

5節に記載の通り。

### Step H2.5: 実装・静的検証結果（完了、2026-09-15）

変更・新規ファイル:

```text
Stage5/checks/real_h5/check_stage5_xy_coordinate_provenance.py/.sh   (新規)
Stage5/checks/dummy/check_dummy_xy_coordinate_provenance.py/.sh      (新規)
docs/stage5/FILES.md                                                  (変更)
docs/stage5/stage5_revision_management_record.md                     (S5-14進捗記録)
```

**H2.5-A（dataset inventory）:** `scan_dataset_inventory()`が
`stage4_training_ablation/<date>/`配下の各run directoryを列挙し、`collected`/`annotated`
サブディレクトリのH5件数と、manifest/summary/README等それらしいファイル名を記録する
（内容は読まず、存在確認のみ）。

**H2.5-B（H5属性とsource provenance）:** `audit_video_provenance()`が最終H5の`video_name`/
`source_pseudo3d_h5`属性を読み、`resolve_intermediate_h5()`で中間H5を解決する
（記録済みsource属性を優先、解決できなければ`pseudo3d_outputs/<date>/`配下のbasename一致
へfallback。両者の解決手段は結果に明示的に区別して記録し、複数candidate一致時は
`basename_fallback_ambiguous`として片方を暗黙に選ばない）。`read_intermediate_local_dimensions()`
が中間H5の`local_input_shape`属性（文字列化されたtensor shape）を解析し、実寸法
（`(height, width)`、全経路で正方形になることを確認済み）を取得、`pixel_xy`のbounds適合を検証する。

**H2.5-C（寸法決定の優先順位）:** `decide_dimension_policy()`が次を実装した。

- 全動画が個別解決できれば選択肢3（per-video実寸法をそのまま使用）。
- 一部未解決でも、解決済み動画が単一の共通寸法へ一致し、かつ未解決動画のpixel_xyがその寸法へ
  strictly収まる場合のみ選択肢2として個別に拡張（収まらない動画は個別に除外、一つも拡張できなければ
  選択肢4）。
- 何も解決できなければ選択肢4（全件除外、cross-video heatmap断念）。

**H2.5-D（正規化とfail-fast）:** `normalize_pixel_xy_with_dimensions()`は
`width>1`/`height>1`必須、nonfinite・bounds外（`0<=x<width`等）をclipせずfail-fastする。

**H2.5-E（artifact）:** private（`stage4_dataset_coordinate_inventory.csv`、
`stage4_xy_source_resolution.csv`、実video ID・絶対pathを含む）とshareable
（`stage4_xy_coordinate_provenance_summary.json`、`stage4_xy_dimension_audit.csv`、
video_alias・寸法・解決手段・bounds合否のみ）を分離して出力する。

**静的検証:** `checks/dummy/check_dummy_xy_coordinate_provenance.py`で15件の合成テスト
（shape文字列parseの正常/異常系、中間H5次元読取の正常/属性欠落/parse不能/非正方形/ファイル欠落、
source解決の記録属性優先/basename fallback/複数一致拒否/完全未解決、正規化のcorner case・
fail-fastケース4種、寸法決定ロジックの選択肢3/2/4分岐とその境界ケース、手作りfinal+intermediate
H5ペアでの`audit_video_provenance()`統合テスト2種）をこの開発コンテナ内（一時venv使用）で
すべて合格を確認した。`py_compile`・`bash -n`・`git diff --check`もすべて合格。

### 次のアクション（GPU側、Step H2.5の実行を依頼）

```bash
bash checks/real_h5/check_stage5_xy_coordinate_provenance.sh
```

既定でteacher v7全180 H5（S5-12/S5-13 W-A runのtrain/val list）を対象に、
`/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/`のdataset inventoryと、
`/mnt/data/3d_projects/pseudo3d_dataset/pseudo3d_outputs/260711/`をfallback rootとしたsource
provenance監査を実行する。完了したらコンソール出力（特に`resolved dimension count`、
`resolution methods seen`、`decision: option N`、`excluded videos`の行）と、
`stage4_xy_coordinate_provenance_summary.json`の内容を共有してほしい。

### Step H2.5: GPU側実行結果（完了、2026-09-15）

匿名化bundle（`SHARE_THIS/`）のprivacy check（実video IDパターン・絶対host pathとも検出なし）を
確認した上で、内容を確認した。

```text
videos audited: 180
resolved dimension count: 180 / 180
resolution methods seen: ['recorded_source_attr']
decision: option 3 -- Per-video real dimensions resolved for every video from intermediate H5 provenance
excluded videos: 0
```

`stage4_xy_dimension_audit.csv`の内訳: 全180動画が`recorded_source_attr`（最終H5の
`source_pseudo3d_h5`属性）だけで解決でき、basename fallbackは一度も必要なかった。寸法は
**全180動画で(256, 256)に完全一致**し、`pixel_xy_bounds_ok`も全180動画でTrue（bounds違反なし）。
split内訳はtrain 162・val 18で、S5-12/S5-13のsplitと一致する。

**管理チャット二重検査での指摘・修正（2026-09-15）:** `decide_dimension_policy()`が、選択肢3判定時に
`pixel_xy_bounds_ok`を条件に含めておらず、将来的にbounds違反がある動画でも黙って選択肢3に含まれ得る
という指摘を受けた。修正として、`dimension_status=="ok"`だが`pixel_xy_bounds_ok`が`True`でない
動画が1件でもあればfail-fastするよう`decide_dimension_policy()`を変更した（解決済みdimensionでの
bounds違反は「未解決」として選択肢2/4へ静かに回すのではなく、データ/ロジックの矛盾として明示的に
停止する）。synthetic testにpolicy-levelのbounds違反ケースを追加し（`decide_dimension_policy`が
正しく`AssertionError`を送出することを確認）、修正後のロジックを実際の監査結果CSV（180行）へ
再適用しても結論（選択肢3、除外0件）が変わらないことを確認した。

**結論:** H2.5-Cの優先順位における最良のケース（選択肢3、個別解決率100%、除外0件）で確定した。
`local_crop_size`の生成CLI既定値256を推測で採用したのではなく、全180動画それぞれの中間H5
provenanceから実測値として得られた一致である。したがってStep H3.1のcross-video XY正規化は
**動画除外なしで**、`normalized_x = pixel_x / 255`、`normalized_y = pixel_y / 255`という単一の
規約で全180動画（および固定21動画のframe/XY診断対象）に適用できる。

### 次のアクション

Step H2.5が最良の結果で完了したため、Step H3（保存済みW-A predictionを使ったframe/XY診断、固定21
動画）とStep H3.1（XY density/temporal recurrence）の実装に進む。方針管理チャットへの報告は
Step H3〜H5完了後にまとめて行う想定だが、本節の確定事実（cross-video正規化を除外なしで採用）は
先に記録しておく。

### Step H3・H3.1: 実装・静的検証結果（完了、2026-09-15）

変更・新規ファイル:

```text
Stage5/checks/real_h5/check_stage5_frame_xy_diagnostics.py/.sh   (新規)
Stage5/checks/dummy/check_dummy_frame_xy_diagnostics.py/.sh      (新規)
docs/stage5/FILES.md                                              (変更)
docs/stage5/stage5_revision_management_record.md                 (S5-14進捗記録)
```

**実装内容:** 既存の`check_stage5_class_weight_threshold_free.py`（prediction読込・parity gate・
`safe_name`）、`check_stage5_frame_spatial_overlap_diagnostics.py`（`find_runs`・
`relative_frame_decile`）、`check_stage5_xy_coordinate_provenance.py`
（`normalize_pixel_xy_with_dimensions`）を再利用し、重複実装を避けた。

- **Step H3（frame単位診断）:** `frame_confusion_counts()`（frameごとのTP/FP/TN/FN）、
  `frame_probability_stats()`（GT positive/background/ignore別の`prob_femur`平均・標準偏差、
  空groupは0埋めせずNone）、`xy_centroid_and_variance()`（GT/predicted positiveのXY重心・分散、
  空集合はNone）、`centroid_distance()`（両方存在する場合のみ計算）、
  `nearest_positive_frame_distance()`（直近GT-positive frameまでの距離）、
  `classify_relative_to_runs()`（GT-positive runに対するinside/before/after/no_gt_run分類）を
  実装し、`build_frame_metrics_rows()`で1動画分のframe単位行を組み立てる。
- **Step H3.1（XY density/temporal recurrence）:** `assign_xy_grid_bin()`（正規化XY座標の
  grid bin割当、範囲外・非finite値はfail-fast）、`compute_xy_density()`（GT/predicted/FP/FN density
  heatmap）、`bin_frame_activity()`（binごとのactive frame集合、`(bin_row,bin_col,frame)`重複除去で
  ベクトル化）、`temporal_recurrence_for_bin()`（predicted runとGT runの重なり判定、
  unmatched run数・run間gap）を実装した。既定でgrid解像度16・8の2種類を計算し、単一gridに依存しない。
- 動画ごとに既存W-A predictionとのthreshold 0.5 parity gate（`check_threshold_0p5_parity`の再利用）を
  実行し、fail-fastする。

**静的検証:** `checks/dummy/check_dummy_frame_xy_diagnostics.py`で15件の合成テスト
（frame混同行列、確率統計の空group処理、XY重心のNone可用性・0埋め回避、frame間距離の境界・空配列、
run分類のinside/before/after/no_gt_run、grid bin境界値（0・境界直前・ちょうど1・中間）とbounds
違反拒否、density集計、bin_frame_activityの空mask処理、temporal recurrenceのmatched/unmatched/gap
計算、frame-level availability flag統合テスト、手作りH5+predictionでのparity gate pass/fail）を
実装し、この開発コンテナ内（一時venv）ですべて合格を確認した。`py_compile`・`bash -n`・
`git diff --check`もすべて合格。

### 次のアクション（GPU側、Step H3・H3.1の実行を依頼）

W-Aのevaluation出力ディレクトリ（S5-13で使用した`.../260914/..._nopad/last`）に対して実行する。

```bash
EVALUATION_DIR=/mnt/data/3d_projects/stage5_evaluations/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad/last \
CHECKPOINT=last \
PIXEL_WIDTH=256 \
PIXEL_HEIGHT=256 \
  bash checks/real_h5/check_stage5_frame_xy_diagnostics.sh
```

`PIXEL_WIDTH`/`PIXEL_HEIGHT`はStep H2.5で確定した全180動画共通の値（256×256）をそのまま使用する。
完了したらコンソール出力（`videos`、`frame rows`、`density rows`、`recurrence rows`の行数）と、
生成された`frame_metrics.csv`/`video_summary.csv`/`xy_density_bins.csv`/
`xy_temporal_recurrence.csv`のファイルサイズ・先頭数行を共有してほしい。

### 重大インシデント: video_aliasへの実video ID混入と修正（2026-09-15）

GPU側実行後の初回共有ファイルで、`video_alias`列に実video ID（タイムスタンプ形式、例:
`20250626_124212_7300`）がそのまま出力されていることが発覚した。原因は
`check_stage5_frame_xy_diagnostics.py`の`main()`が、`evaluate_stage5.py`が出力する**匿名化前の生**
`h5_metrics.csv`の`row["video_name"]`（実video ID）をそのまま`video_alias`として全出力行へ書き込んで
いたバグである（これまで分析してきた`anonymized_metrics_SHARE_THIS`は別途匿名化ステップ
`export_anonymized_stage5_metrics.py`を経由していたため、このcheckerで初めて顕在化した）。

**修正:** `main()`で実video名はH5/predictionファイルパス解決にのみ内部使用し、出力行には
`f"{split}_{index:03d}"`という列挙ベースaliasのみを書き込むよう変更した。**回帰テストを追加**し
（タイムスタンプ形式の実video ID風文字列を含む合成`h5_metrics.csv`でCLIをエンドツーエンド実行し、
出力ファイルにその文字列が一切含まれないことを検証。修正前のコードでは確実に失敗することを確認
済み）、合成テスト16件・`py_compile`・`git diff --check`すべて合格した。ユーザー側で該当4ファイル
（`.tmp/`とGPU機`work_dirs/`双方）を削除し、修正版checkerで再実行、再実行後の出力に実video ID・
絶対host pathが含まれないことを確認した。

### Step H3・H3.1: GPU側実行結果と分析（完了、2026-09-15）

実行結果: `videos: 21, frame rows: 834, grid resolutions: [16, 8], density rows: 6720,
recurrence rows: 2917`。全21動画でthreshold 0.5 parity gateに合格した（fail-fastなし）。

**仮説B（時間位置・frame phase）の分析結果:** validation 18動画・770 frame行を対象に、
動画内相対位置decile別のrecall/FPRを集計した。

| decile | mean recall | mean FPR |
| --- | ---: | ---: |
| 0（動画序盤） | 64.2% | 10.8% |
| 3 | 70.4% | 11.7% |
| 6 | 49.3% | 10.9% |
| 8 | **7.9%** | 11.3% |
| 9（動画終盤） | N/A（GT不足） | 10.5% |

**recallがdecile 0の64%からdecile 8の7.9%へ、動画終盤にかけて系統的に急落する**一方、FPRは
10.5%〜14.2%とdecile間でほぼ一定だった。`gt_interval_position`別でもFPRは
before(13.0%) > inside(11.9%) > after(10.2%)という緩やかな傾向が見られた。これはS5-08の
既知傾向（動画内相対位置の後半ほどdisagreement率が高い）と整合し、**時間位置による性能劣化は
主にrecall側に現れ、FP側にはほぼ影響しない**ことを示す新しい知見である。

**仮説A（座標事前分布）の分析結果:** grid解像度16でpooled（全21動画合算）density mapを構築した。
GT positive density・predicted positive density・FP densityの空間相関はGT-FP間0.568、
GT-predicted間0.604と中程度の正相関だった。より直接的な検証として、**個々の動画でGT positiveが
一切存在しないbin**（その動画にとってローカルな根拠が皆無の領域）だけに絞り、動画横断でGTが
密集する「globally hot」bin（pooled GT密度の上位25%）と「globally cold」bin（下位25%）で
predicted positive率を比較したところ、**ローカルな根拠が皆無であるにもかかわらず、globally hot
binではglobally cold binの約8.7倍のpredicted positive/FP率**を示した（mean predicted:
320.8 vs 36.8）。これは動画横断の座標事前分布が、個々の動画の実際の構造とは無関係に予測へ影響を
与えていることを強く示唆する。

**留保事項:** handoff文書9章の座標事前分布支持条件は「raw point densityだけでは説明できず、
densityで正規化したpositive率でも傾向が残る」ことを求めているが、現在の`xy_density_bins.csv`は
bin単位のraw point総数（valid point count）を保持しておらず、この正規化確認はまだ行っていない。
globally hot領域はそもそも点群サンプリング密度自体が高い可能性があり、この交絡を完全には排除
できていない。強い示唆はあるが、正式な「支持」の確定にはbin単位point密度の追加集計が必要。

**仮説C（overlap exposure）:** Step H3.1だけでは、exposure（window出現回数）とframe/bin単位の
誤り率を直接結びつける集計を行っていない（Step H2で得たclass別exposure倍率はvideo単位の
構造情報に留まる）。source point/frame単位でexposure countとFP/FN/disagreementを層別化する
Step H4が、仮説Cを検証する主要な手段として引き続き必要である。

**PLYで見えた「反復」現象の定量化:** grid16でpredicted positiveがactiveなbin 2,192件のうち、
**36.1%がGT runと重ならない複数の時間的に分離したpredicted run（multiple_unmatched_runs）**を
持ち、**32.0%のbinはGT runと一度も重ならない**（純粋に誤検出だけのXY位置）。これはtrain sanity
PLYで観察された「似たXY位置のpositive集合が異なるframe群へ反復する」現象を、定量的にほぼ全動画
共通の頻出パターンとして確認するものである。

**中間まとめ:** Step H3・H3.1単独では、仮説Bは明確に支持され、仮説Aは強く示唆される（density
正規化確認が未了）、仮説Cは未検証（Step H4待ち）という状態である。依頼書の完了条件が求める
「3仮説それぞれの支持・不支持・未確定の根拠付き分類」は、Step H4完了後に確定させる。

### Step H4: 実装・静的検証結果（完了、2026-09-15）

変更・新規ファイル:

```text
Stage5/checks/real_h5/check_stage5_per_window_context_diagnostics.py/.sh   (新規)
Stage5/checks/dummy/check_dummy_per_window_context_diagnostics.py/.sh      (新規)
docs/stage5/FILES.md                                                        (変更)
docs/stage5/stage5_revision_management_record.md                            (S5-14進捗記録)
```

**実装内容:** S5-08 `check_stage5_overlap_aggregation.py`の`OverlapAccumulator`/
`run_overlap_forward`（per-window forward + accumulate）/`classify_overlap`/`gt_class_masks`/
`relative_frame_deciles`を再利用した。per-window forwardに必要なtorch importは
`process_video_h4()`内のlocal importに限定し、parity判定・層別集計・XY bin集計・Step H3.1
recurrence CSVとのjoinはすべてtorch非依存で実装した（この開発コンテナ内でtorch未インストールの
まま動作確認済み）。

**no-tolerance parity gate（handoff文書の要求通り、CUDA非決定性でも通さない）:**

1. このrunのvote_countと保存済みW-A predictionのvote_countが完全一致
2. Step H2と同じwindow構成のtraining window occurrence countとこのrunのvote_countが完全一致
   （training exposureとinference vote countの内部整合性チェック）
3. mean probability（threshold 0.5）から再導出したpredicted classと保存済みpredictionの
   TP/FP/TN/FN・per-point predicted classが完全一致

いずれか1つでも満たさなければ即座に停止する。

**層別集計・XY bin診断:** GT class（valid_positive/valid_background/ignore）・TP/FP/FN/TN・
relative frame decile・vote count bucket（1/2/3-4/5+）を組み合わせた層別集計
（`point_overlap_error_statistics.csv`相当）と、predicted positive点（TP+FP）のXY bin単位
exposure/disagreement集計（`bin_exposure_disagreement.csv`）を出力する。`--xy_temporal_recurrence_csv`
にStep H3.1の出力を渡すと、`multiple_unmatched_runs`のある/なしでmean training window occurrence・
mean disagreement rateを比較するcross-check（`recurrence_exposure_cross_check.json`）を自動実行する。

**静的検証:** 15件の合成テスト（parity gateの正常系・vote count不一致・training occurrence
不一致・predicted class不一致・改ざんCSV拒否の4種の異常系、vote countバケット境界値、
層別集計のscope_mask処理、TP/FP/FN/TN分類でのignore点除外、XY bin集計の正常/空mask、
Step H3.1 recurrenceとのjoin）をこの開発コンテナ内ですべて合格を確認した。`py_compile`・
`bash -n`・`git diff --check`も合格。

### 次のアクション（GPU側、Step H4の実行を依頼）

```bash
CHECKPOINT=/mnt/data/3d_projects/stage5_runs/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad/last.pt \
EVALUATION_DIR=/mnt/data/3d_projects/stage5_evaluations/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad/last \
XY_TEMPORAL_RECURRENCE_CSV=/mnt/data/3d_projects/models/Stage5/work_dirs/_frame_xy_diagnostics/xy_temporal_recurrence.csv \
PIXEL_WIDTH=256 \
PIXEL_HEIGHT=256 \
  bash checks/real_h5/check_stage5_per_window_context_diagnostics.sh
```

**事前に知っておいてほしいこと:** handoff文書の指示通り、parity gateにtoleranceは一切設けていない。
PointNeXtのCUDA kernel（farthest point samplingなど）が非決定的な場合、理論上は初回実行でparity
不一致により停止する可能性がある。その場合はエラーメッセージ（どの一致条件で失敗したか、
`max_abs_prob_diff`の値）をそのまま共有してほしい。原因（真の非決定性か、設定の取り違えか）を
切り分けてから対応を判断する（toleranceを勝手に追加することはしない）。

## Step H4.1: parity gate不一致の原因訂正と集約規約統一（2026-09-15）

### 経緯

Step H4のGPU側実行で、validation動画（video_alias `validation_000`）のpredicted class parityが
1点だけ不一致となり、fail-fastが正しく発動した（`max abs probability diff=2.980e-08`、
diff artifact記録済み: `point_index=290130, frame_order=34, abs_probability_diff=0.0`）。
実装チャットは当初、vote_count・training window occurrence countが完全一致していたこと、不一致が
1点のみだったこと、該当点の確率が0.5近傍だったことから「GPU/CUDA kernelの非決定性による境界
ケース」と推測し、`docs/stage5/s5-14/stage5_s5_14_step_h4_parity_boundary_case_decision_request.md`として
方針管理チャットへ判断を依頼した。

### 方針管理チャットによる精査結果（同文書8節に追記）

方針管理チャットは報告書とコードを確認し、「CUDA非決定性」という結論を時期尚早と判断した。
実装チャット側の報告にも誤りがあった。

- `max_abs_prob_diff`は動画内の全点についての最大絶対差であり、不一致点自身の差ではない
  （該当点自身の`abs_probability_diff`は診断artifact上0.0）。この誤った紐付けを報告してしまった。
- 保存時（`evaluate_stage5.py`）とH4 checkerでpredicted classの導出規約が異なっていた。

### 実装チャットによる独立検証（2026-09-15、コード確認）

方針管理チャットの指摘をコードで検証し、正しいことを確認した。

| 項目 | 保存時 `evaluate_stage5.py: predict_h5()` | Step H4（修正前） |
| --- | --- | --- |
| 集約対象 | 両クラス（`probability_sum`は`[N, num_classes]`） | positiveクラスのみ（`OverlapAccumulator.p1`） |
| window単位のcast | softmax出力を直後に`float32`化してから蓄積 | softmax出力をそのまま`float64`化 |
| 最終判定直前 | `float64`で合算・平均後、`float32`へ変換 | `float64`のまま |
| predicted class | 2クラスの`argmax`（同値はindex 0=background） | `mean_probability >= 0.5`（同値はpositive） |

さらに、再利用元の`check_stage5_overlap_aggregation.py`には既存テスト
`test_probability_half_is_background()`（`p1 == 0.5`はbackground、`p1 > 0.5`の厳密不等号を検証）が
既に存在しており、これも実装チャットが使った`>=0.5`規約と矛盾していた。事前に確認すべきだった。

**結論: 不一致点の確率値そのものは両run間で完全に一致（diff=0.0）しており、GPU非決定性ではなく、
実装チャットが新設したcheckerの集約・判定規約が正本と異なっていたことが原因である。** 当初の
「CUDA非決定性」という推測は撤回する。

### 修正方針（未実装、方針管理チャット8.5節への対応）

1. エラーメッセージを訂正し、「video全体の最大差」と「該当点自身の差」を明確に区別する。
2. `run_overlap_forward`（S5-08、positiveクラスのみ・float64直接cast）を予測class判定に使うのを
   やめ、`predict_h5()`と同じ手順（window単位float32化→両クラスをfloat64で加算→最終float32化→
   2クラスargmax）を再現する新しいforward処理をH4 checkerへ実装する。backgroundは`1-positive`で
   復元せず、softmaxの実出力をそのまま保持する。同じforward 1回分の出力を、既存
   `OverlapAccumulator`（edge distance等の診断用）と新しい2クラス集約の両方へ供給し、
   forwardの二重実行を避ける。
3. parity比較・TP/FP/TN/FN層別集計を、この正本互換のpredicted classへ統一する
   （`classify_prediction()`を外部から渡されたpred_label配列を使う形に変更）。per-window単位の
   `positive_vote_ratio`等（診断専用、S5-08の`p1 > 0.5`規約のまま）とは意味が異なることを明記する。
4. CPU synthetic testを追加する: (a) 両クラスがfloat64で厳密に0.5/0.5のタイ→background、
   (b) float64では非タイだがfloat32変換後にタイになるケース→変換後の規約に従う、(c) 素朴な
   `positive確率>=0.5`と正しい2クラスargmax（float32変換後）が食い違う境界ケース→修正後の実装が
   正本の規約と一致することを確認する。期待値は`predict_h5()`のロジックから直接計算する。
5. 該当点（`validation_000`、point_index 290130、frame_order 34）を除外せず、toleranceも追加せず
   再検証する。
6. 合格すれば21動画全体のStep H4へ進む。差が残る場合はpre/post cast値・保存済み値・差分を
   artifactへ保存して停止し、「非決定性」と断定せず原因と選択肢を改めて報告する。

保存済みW-A prediction・checkpointやproductionの判定規約は変更しない。今回の修正はH4 checker側の
比較条件を既存評価処理へ合わせるものである。

### 次のアクション

方針管理チャットの確認を経てから、上記方針に沿って実装する。

### Step H4.1: 実装結果（完了、2026-09-15）

`Stage5/checks/real_h5/check_stage5_per_window_context_diagnostics.py`と対応する
`Stage5/checks/dummy/check_dummy_per_window_context_diagnostics.py`を修正した。

**1. 報告の訂正:** `verify_h4_parity()`のfail-fastメッセージを、
「video-wide max abs positive-probability diff」（動画全体の最大値、informational）と
「per-point diff artifactの`abs_probability_diff`列」（該当点自身の差）を明確に区別する文言へ
修正した。

**2. 正本互換の2クラス集約を新設（`run_h4_forward()`/`canonical_predicted_class()`）:**
`check_stage5_overlap_aggregation.run_overlap_forward()`（S5-08、positiveクラスのみ・
window単位でfloat64直接cast）はpredicted class判定には使わないことにした。新設した
`run_h4_forward()`は同じforward 1回分の出力から、(a) 既存`OverlapAccumulator`（診断専用、
edge distance等）と、(b) `predict_h5()`と同じ手順の2クラスfloat64累積（window単位で
softmaxをfloat32へ直後cast→float64で加算）を同時に更新する。`canonical_predicted_class()`が
最終的に「float64合計/vote_count→float32変換→2クラスargmax」を行い、backgroundは
`1-positive`で復元せずsoftmaxの実出力（2列）をそのまま保持する。

**3. parity・TP/FP/TN/FN層別の統一:** `verify_h4_parity()`と`classify_prediction()`はいずれも
`canonical_predicted_class()`が返す`pred_label`を受け取る形に変更し、内部で独自の閾値判定を
行わないようにした。per-window単位の`positive_vote_ratio`/`classify_overlap()`（S5-08の
`p1 > 0.5`規約、診断専用）とは意味が異なることをdocstringで明記した。

**4. CPU synthetic testを追加（3件、方針管理チャット指定の境界ケース）:**

- (a) 両クラスがfloat64で厳密に0.5/0.5のタイ→background（`pred_label=[0]`）
- (b) float64では非タイ（positiveがわずかに高く、素朴なfloat64 argmaxならpositiveを選ぶ）だが
  float32変換後に厳密に0.5/0.5のタイになるケース→変換後の規約通りbackgroundとなることを確認
  （数値例: `background_sum=1.0-1e-9`, `positive_sum=1.0+1e-9`、float64平均は
  `0.4999999995`/`0.5000000005`で非タイだが、float32へcastすると両方とも`0.5`になる）
- (c) (b)と同じデータで、素朴な`positive確率>=0.5`（修正前の実装）と正しい2クラスargmax
  （float32変換後）が実際に食い違うことを直接確認（`naive=True`だが`canonical=False`）。
  これは今回のインシデントを引き起こしたシナリオそのものを再現するテストである。

これらに加え既存16件（parity gate正常系・4異常系・diff artifact機能・vote countバケット・
層別集計・TP/FP/FN/TN分類・XY bin集計・recurrence join）と合わせて計19件のsynthetic testを
この開発コンテナ内（torch未インストール、モジュールが引き続きtorchに依存しないことも確認）で
すべて合格を確認した。`py_compile`・`bash -n`・`git diff --check`も合格。

**状態:** 実装・静的検証完了。

### Step H4.1: GPU側再検証結果（完了、2026-09-15）

修正版で再実行した結果、**全21動画でparity gateが完全一致**した（除外・tolerance追加なし）。

```text
videos: 21, stratified rows: 1226, bin rows: 2192
max abs mean-probability diff vs saved W-A prediction (informational only): 0.000e+00
```

これにより、当初の「CUDA非決定性」という診断は完全に誤りであり、方針管理チャットが指摘した
「checkerの集約・判定規約が正本と異なっていた」ことが不一致の唯一の原因だったことが実証された。
GPU forward自体に非決定性由来の残差は検出されなかった（bit-exact一致）。

**recurrence/exposure cross-check結果（Step H3.1のXY bin recurrenceとの結合）:**

| 区分 | bin数 | mean training window occurrence | mean disagreement rate |
| --- | ---: | ---: | ---: |
| multiple_unmatched_runsあり | 791 | 1.619 | 0.273 |
| multiple_unmatched_runsなし | 1,401 | 1.473 | 0.252 |

multiple unmatched runsのあるbinは、ないbinよりtraining window occurrence（約+10%）・
disagreement rate（約+8%）とも高いが、差は小さい。仮説A（座標事前分布）で見られた8.7倍という
効果に比べると、overlap exposureとの相関は弱いか限定的である可能性がある。

### Step H4: 層別集計の詳細分析（完了、2026-09-15）

`point_overlap_error_statistics.csv`（1,226行、TP/FP/FN/TN×relative_frame_decile/
vote_count_bucketの層別集計）を分析した。window 16/stride 8では、点ごとのvote_countは
構造上**1または2のみ**（overlap率50%のため3重以上の重複は発生しない）。

**vote_count別のTP/FP/TN/FN分析（validation、点数加重平均）:**

| vote_count | recall | FP率（background中） | FP disagreement rate | TN disagreement rate |
| --- | ---: | ---: | ---: | ---: |
| 1（重複なし） | 39.2% | 11.88% | 0.0%（重複なしのため定義上0） | 0.0% |
| 2（重複あり） | 45.3% | **11.64%** | **34.1%** | **7.2%** |

**FP率（background点がFPになる割合）はvote_count 1と2でほぼ同一（11.88% vs 11.64%）**であり、
「window重複が多いほどFPが増える」という単純な関係は支持されなかった。むしろrecallは
vote_count=2の方がやや高い（45.3% vs 39.2%）。

一方、**vote_count=2の内部だけで比較すると、FPのdisagreement rate（34.1%、2つのwindowが
positive/negativeで割れる率）はTNのdisagreement rate（7.2%）よりはるかに高い**。これは
「overlapが多いほど誤りが増える」という単純な話ではなく、「2つのwindow予測が食い違う
（不安定な）境界例が、FPという誤りへ集中しやすい」というより限定的なメカニズムを示唆する。

**relative_frame_decile別の分析:** disagreement rateとtraining window occurrenceは、TP/FP/FN/TN
いずれの層でも動画の**中盤（decile 4〜6）でピーク、両端（decile 0・9）で低い**という対称的な
パターンを示した。これはwindow境界の構造（動画両端は重複windowが少ない）による幾何学的な
効果であり、Step H3で確認した「recallが動画終盤にかけて単調に低下する（64%→7.9%）」という
非対称な傾向とは形状が異なる。したがって、Step H3のrecall低下は主にexposureパターンの
副産物ではなく、独立した時間位置効果である可能性が高い。

**仮説Cの判定（暫定）:** overlap exposureは、FP率を直接押し上げる単純な要因としては支持されない
（vote_count 1/2間でFP率がほぼ同一）。ただし、(1) vote_count=2の中でFPがdisagreement（window間の
予測不一致）へ強く偏る、(2) XY bin単位でmultiple unmatched runsのあるbinがtraining window
occurrence・disagreement rateともやや高い（+8〜10%）、という2つの限定的な根拠がある。総合すると
**仮説Cは「主要因ではないが、限定的に支持される」**と判断する。仮説Aの8.7倍という効果と比べると
明確に小さい。

## 3仮説の最終判定（Step H1〜H4完了時点）

| 仮説 | 判定 | 根拠 |
| --- | --- | --- |
| A. 座標事前分布 | **強く示唆される**（正式な「支持」確定にはdensity正規化確認が必要） | 個々の動画でGT positiveが皆無のbinでも、動画横断でGTが密集するbinはpredicted positive/FP率が約8.7倍。ただしbin単位point密度での正規化は未実施で、点群サンプリング密度の交絡を完全には排除できていない。 |
| B. 時間位置・frame phase | **明確に支持される** | recallが動画内相対decile 0の64.2%からdecile 8の7.9%へ単調に急落する一方、FPRはdeciles間でほぼ一定。この低下パターンはStep H4で見たexposureの対称的な分布（動画両端で低い）とは形状が異なり、独立した時間効果と判断できる。 |
| C. overlap exposure | **限定的に支持される（主要因ではない）** | FP率はvote_count 1/2間でほぼ同一で、単純な「exposureが多いほど誤り増加」は支持されない。ただしvote_count=2内でのFP disagreement rate（34.1%）はTNの約4.7倍、XY bin recurrenceとexposureにも弱い正の関連（+8〜10%）があり、限定的なメカニズムとして残る。 |

3仮説はいずれか1つに単純化できず、**座標事前分布（仮説A）が最も強い効果、時間位置（仮説B）が
recallに限定した明確な効果、overlap exposure（仮説C）は補助的・限定的な効果**という、複合的な
結論となった。依頼書10章「結果後の分岐」に照らすと、複数仮説が支持される場合の扱いは方針管理
チャットへの報告事項である。

## 方針管理チャットの回答（2026-09-15）

1. Step H4.1の修正・parity検証を承認する。
2. Step H4の集計結果をS5-14の診断資料として採用する。
3. Step H5の総括で仮説Aの密度正規化と仮説B/Cの層別結果を整理してから、次の改修を決定する。

これによりStep H5（集計・可視化・完了報告）へ着手した。

## Step H5: 実装・静的検証結果（完了、2026-09-15）

変更・新規ファイル:

```text
Stage5/checks/real_h5/check_stage5_s5_14_summary_export.py/.sh   (新規)
Stage5/checks/dummy/check_dummy_s5_14_summary_export.py/.sh      (新規)
docs/stage5/FILES.md                                              (変更)
docs/stage5/stage5_revision_management_record.md                 (S5-14進捗記録)
```

**実装内容:** 既存のStep H2（`gt_overlap_exposure.csv`）・H2.5（`stage4_xy_coordinate_
provenance_summary.json`）・H3（`frame_metrics.csv`、`video_summary.csv`）・H3.1
（`xy_density_bins.csv`、`xy_temporal_recurrence.csv`）・H4（`point_overlap_error_statistics.csv`、
`video_summary.csv`、`stage5_s5_14_h4_summary.json`）の出力（すべてvideo_alias化済み）を入力とし、
依頼書が要求するfile listのうち未生成だった2つを新規生成する。

- `frame_position_deciles.csv`: `frame_metrics.csv`のframe単位行を`split × relative_frame_decile`
  で集計し、recall/precision/F1/IoU_femur/FPR/FNRの平均・中央値・有効frame数を算出する。
  recallが未定義（GT positiveなし）のframeは0扱いせず、平均・中央値の計算対象から除外する。
- `stage5_s5_14_summary.json`: 3仮説の判定、Step H2〜H4の主要数値（parity結果、recurrence/
  exposure cross-check、XY座標provenance解決率）、fileマニフェストを1つのJSONへ集約する。

加えて、Step H3とH4で別々に出力されていた`video_summary.csv`を`(video_alias, split)`単位で
1本へmergeし（キー集合が完全一致しない場合はfail-fast）、仮説Cの根拠数値を再現可能にする
`vote_count_error_rates.csv`（vote count bucket別のrecall・FP率・FP/TN disagreement rateを
`point_overlap_error_statistics.csv`から再集計）を追加出力する。最後にbundle全体へ
timestamp形式video ID・絶対host pathのprivacy self-checkを実行する。

**静的検証:** 5件の合成テスト（frame decile集計での未定義値除外・全未定義ケース、video_summary
mergeの正常系とkey不一致拒否、vote count別error rateの手計算検算、privacy self-checkのpass/fail
3種）をこの開発コンテナ内ですべて合格を確認した。`py_compile`・`bash -n`・`git diff --check`も
合格。

### 次のアクション（GPU側、Step H5の実行を依頼）

すべての入力ファイルは既定パス（これまでのStep H2〜H4実行で生成済み）を使うため、追加の環境変数
指定なしで実行できる。

```bash
bash checks/real_h5/check_stage5_s5_14_summary_export.sh
```

完了したらコンソール出力（`videos`、`privacy check`の行）と、出力先
（既定`work_dirs/_s5_14_summary/SHARE_THIS/`）のファイル一覧を共有してほしい。

### Step H5: GPU側実行結果（完了、2026-09-15）

```text
videos: 21
privacy check: {'status': 'passed', 'files_checked': 9}
files written: [gt_overlap_exposure.csv, frame_metrics.csv, frame_position_deciles.csv,
  xy_density_bins.csv, xy_temporal_recurrence.csv, point_overlap_error_statistics.csv,
  video_summary.csv, vote_count_error_rates.csv, stage5_s5_14_summary.json]
```

出力bundle（`Stage5/work_dirs/_s5_14_summary/SHARE_THIS/`）の内容を確認し、privacy check
（timestamp形式video ID・絶対host pathとも検出なし）を独立に再確認した。`vote_count_error_rates.
csv`・`frame_position_deciles.csv`の値は、これまで実装チャット側で手計算していた数値
（recall/FP率/disagreement rate、decile別recall）と完全に一致し、独立した集計経路で再現性が
確認できた。train_sanityでも同様の傾向（vote_count=2でFP disagreement rate 35.2% vs TN
disagreement rate 6.7%）が見られ、仮説Cの限定的な支持はvalidationだけでなくtrain_sanityでも
方向が一致した。

`docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.8節へS5-14全体（Step H1〜H5）の
数値正本を記録した。S5-14 coreはこれで完了とする。次の改修（座標equivariance診断、
inverse-occurrence loss weighting等）は方針管理チャットの判断を待つ。
