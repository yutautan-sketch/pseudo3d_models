# Stage 5 S5-14補足: 点密度補正と動画別・時間別再集計 報告書

作成日: 2026-09-15
最終更新日: 2026-09-15
作成元: Stage 5実装チャット
状態: **本補足（Step S1〜S5）は実装・実機再集計完了。方針管理チャットが受入れ、
仮説A優先で座標変換診断（S5-14補足2）を委任した（D-032、D-033）。
補足2の方針・実施記録は`docs/stage5/s5-14/stage5_s5_14_supplement2_report_to_policy_chat.md`へ分離**

本書は`docs/stage5/s5-14/stage5_s5_14_supplement_implementation_handoff.md`（実装依頼書、正本）を受けての
理解・現状・実装方針の記録として作成し、以後、検証が進むごとに本書へ結果を追記して方針管理
チャットへの完了報告として使用する。実装依頼書9章が別途指定する
`stage5_s5_14_supplement_report_to_policy_chat.md`という完了報告ファイル名と一致させてあるため、
新しい完了報告ファイルを別途作成する必要はない。

## 1. 経緯

S5-14 core（Step H1〜H5）の成果物・parity検証・privacy checkは方針管理チャットに受け入れられた
（D-030）が、3仮説の解釈には以下の方法論的な留保が指摘された。

- **仮説A（座標事前分布）:** 「globally hot/cold binで約8.7倍」という数値は、mean predicted
  点**数**320.8対36.8の比較であり、background点数を分母とするFPR（率）ではない。bin内の
  background点数自体が多ければpredicted positive/FP点数も増えるため、分母を揃えない限り
  座標事前分布の学習を断定できない。
- **仮説B（時間位置）:** decile 0のmean recall 64.2%とdecile 8の7.9%には対象動画・GT点数構成の
  差が含まれ得る。decile 3は70.4%でdecile 0を上回っており、「単調低下」という記述は不正確。
  同一動画内での前半/後半比較が必要。
- **仮説C（overlap exposure）:** vote count 1/2間のFPR差（11.88%→11.64%）は小さく、vote
  count=2内でのFP disagreement率の高さは予測不一致と誤りの関連を示すのみで、training
  exposureがFPを増やす因果関係の証明にはならない。動画・時間位置を考慮した層別が必要。

S5-14 core報告書（`docs/stage5/s5-14/stage5_s5_14_report_to_policy_chat.md`）とH5 JSON（`stage5_s5_14_summary.
json`）の`hypothesis_verdicts`固定文字列は、新たな検証結果ではなく履歴として扱い、本補足の
再集計から独立に判断する（D-030、D-031）。

## 2. 目的

点密度・動画構成・時間位置とexposureの交絡を、**既存H5・保存済みprediction・H4成果物のみ**を
使った一度の限定的な再集計で確認し、次に行うべき診断（座標変換診断、時間文脈診断、
inverse-occurrence loss weighting比較、またはS5-15 pilotへの直行）を決定する。新規学習0回、
GPU再推論0回。

## 3. 固定条件（本補足全体を通して変更しない）

teacher v7（`bboxrank_v7_cvat_authoritative_crop_quality_v1`、180 H5、train 162 / validation 18、
既存file list使用）、W-A epoch 5 checkpoint（GroupNorm 8 groups、`bbox_noncontour_ignore`、
class weight`[0.05963856, 1.94036150]`、window 16/stride 8/tailあり）、aggregate判定は
**mean probabilityから保存済み2クラスargmax（同値はbackground）を使った既存pred_labelそのもの**
（`>=0.5`での再生成は行わない）、XY座標はH2.5で確認済みの実寸法256×256（`pixel_xy / 255`）、
予測評価対象は固定train sanity 3動画+validation 18動画、grid解像度は16と8のみ。

## 4. 実装前監査で確認した既存コードの再利用可能性

- **`check_stage5_class_weight_threshold_free.py`の`load_prediction()`**: 保存済み`.npz`から
  `prob_femur`（診断用）と`pred_label`（判定の正本、2クラスargmax済み）を読み込む。本補足でも
  `pred_label`をそのまま使い、確率からの再判定は行わない。
- **`check_stage5_xy_coordinate_provenance.py`の`normalize_pixel_xy_with_dimensions()`**: H2.5で
  確定した256×256を使ったXY正規化（bounds外・非finite値はfail-fast）をそのまま再利用できる。
- **`check_stage5_frame_xy_diagnostics.py`の`assign_xy_grid_bin()`**: 正規化XY座標のgrid bin
  割当（境界値の扱い含む）を再利用できる。同ファイルの`classify_prediction()`（GT class×
  predicted classからTP/FP/FN/TN判定）も、引数を保存済み`pred_label`にすればそのまま使える。
- **`check_stage5_per_window_context_diagnostics.py`**: `bucket_vote_count()`
  （vote count 1/2の分類）、`run_h4_forward()`が生成したStep H2の`window_occurrence_counts()`
  （training exposure、純geometry・model forward不要）を再利用できる。GPU modelをimportする
  経路（`run_h4_forward`本体、`main()`のtorch import）は本補足では使わない。
- **`check_stage5_s5_14_summary_export.py`**: CSV読み書き・privacy self-check
  （`assert_bundle_anonymous`）のパターンをそのまま踏襲できる。
- **`check_stage5_frame_xy_diagnostics.py`の`frame_metrics.csv`**: 動画×decileのTP/FP/TN/FN・
  GT点数・frame数はここから再集計する（Step S3）。依頼書が指摘する通り、既存のframe等重み
  mean recallと点数から計算するrecallは区別して両方保持する。

### 未確認・要監査の項目

- H4の既存artifact（`point_overlap_error_statistics.csv`等）に、動画×時間区分×vote countの
  **joint**集計（周辺集計ではなく）が含まれるか。含まれない場合、保存済みprediction・H5・
  frame_orderから独立に再構成できる範囲を確認する必要がある（依頼書Step S4）。
- per-window（window単位）のclass disagreementは、H4が生成した`OverlapAccumulator`由来の
  disagreement rate（video×bin単位で集計済み）はあるが、これを動画×時間区分×vote countの
  joint層まで分解した粒度で保持しているかは未確認。集計し直せない場合は「未検証」として報告し、
  GPU再推論で補わない（依頼書の明示的な禁止事項）。
- train 162動画のみからXY priorを作る際、train sanity 3動画（trainの一部でもある）を
  prior作成から除外する具体的な実装方法（対象動画のGT点をtrain集計から差し引く、または
  対象動画を除いた162-3=159動画で再集計する等）を確定する必要がある。

## 5. 実装した内容（Step S1〜S5、完了）

依頼書のStep S1〜S5に沿って実装した。新規checker
`Stage5/checks/real_h5/check_stage5_s5_14_supplement.py/.sh`と
`Stage5/checks/dummy/check_dummy_s5_14_supplement.py/.sh`をCPU/NumPy/h5pyのみで実装し、
GPU modelをimportする経路は使っていない（`import torch`は本モジュールに一切ない）。

1. **Step S1（分母付きXY bin統計）**: 動画×grid×bin単位で全点数・valid点数・valid positive/
   background数・ignore数・TP/FP/TN/FN・predicted positive数（ignore上のものは別記）・
   `FPR=FP/(FP+TN)`・`recall=TP/(TP+FN)`を再集計する。分子・分母は必ず同じ点集合から計算し、
   分母0はnull/NaN+availability flagとする。各動画のbin合計を既存W-A confusion counts
   （`h5_metrics.csv`等）と照合し、fail-fastで一致を確認する。
2. **Step S2（train-only XY priorとhot/cold比較）**: train 162動画（train sanity 3動画は
   prior作成対象から除外）のGTのみから、grid別の共通XY分布を2定義（raw GT-positive点数 /
   valid点数で正規化したGT-positive率）で別々に作成する。上位/下位25%をhot/cold候補とし、
   同値境界の扱い・実際のbin数を保存する。上下境界が一致し分離不能な定義は未定義として報告する。
   各評価動画で**GT positive数0のbin**を対象に、valid background点に対するhot/cold FPRを
   点数加重・動画等重みの両方で比較し、比較可能動画数・増減方向件数を報告する。cold FPRが0の
   場合はepsilonで有限倍率を作らず、比は未定義としFPR差を使う。train sanityとvalidationは
   別々に集計する。
3. **Step S3（時間位置の動画内paired比較）**: `frame_metrics.csv`から動画×decileのTP/FP/TN/FN・
   GT点数・frame数を集計し、split×decileの参加動画数・GT-positive frame数/点数・recall計算
   可能動画数を明示する。前半（decile 0-4）と後半（decile 5-9）**の両方にvalid GT-positive点が
   ある同一動画**でrecallの後半-前半差を計算し、動画別値・mean/median・低下/上昇/同値件数を
   報告する。片側にしかGTがない動画はpaired比較から除外し理由・数を保存する。可能な範囲で
   動画×前半/後半×vote count 1/2の比較も追加し、層が疎な場合はデータ不足と明記する。
4. **Step S4（exposure/disagreementの層別整理）**: 動画×時間区分でvote count 1/2のFPR・recallを
   比較する（Step S3のjoint集計を再利用）。disagreementは保存済みartifactで動画×時間区分×
   vote countのjoint層別が再構成できる場合のみ同じ層で集計し、できない場合はH4の既存確認済み
   範囲を引用して「joint層別は未検証」と明記する。FPとdisagreementの関連、exposureと誤り率の
   関連は別々に結論づけ、observationalな関連だけでinverse-occurrence loss weightingの効果を
   確定しない。
5. **Step S5（検算・集約・報告）**: synthetic testで少なくとも次を確認する。
   - raw点数差が大きくてもFPRが同じになる例／分母補正後もFPR差が残る例
   - valid/ignoreの分離、分母0、cold FPR0、空bin
   - train-only prior、train sanity自身の寄与除外、quantile同値の扱い
   - 動画構成が変わるとpooled時間傾向が動画内傾向と異なる例、paired対象選択
   - joint集計と周辺集計の区別、既存confusion countsとの一致
   - 匿名化出力の実video ID・host path検出（`assert_bundle_anonymous`の再利用）

   `py_compile`・`bash -n`・`git diff --check`を実行し、実機で一度の限定再集計を行う。

## 6. 出力ファイル（依頼書6章）

```text
xy_bin_denominators.csv
train_xy_prior.csv
xy_hot_cold_video_metrics.csv
video_decile_counts.csv
paired_temporal_metrics.csv
video_time_exposure_metrics.csv
stage5_s5_14_supplement_summary.json
```

summaryにはdataset/checkpoint識別情報、split件数、grid、prior定義、分母、quantile、時間区分、
入力artifactとの対応、parity、未定義数・未検証項目を記録する。仮説判定は根拠数値と留保を伴う
分析結果として報告し、`hypothesis_verdicts`のような固定文字列を無条件出力する実装にしない。
実path・video mappingはprivate、共有出力は既存匿名aliasを使用し、schemaで許可した列のみを
shareableへ出してbundle全体のprivacy self-checkに合格させる。

## 7. 判定基準（依頼書8章の要約）

1. 仮説Aの偏りが分母補正後も動画別・両grid（16/8）で残る場合、再学習なしの座標変換診断を
   次案とする。
2. 仮説Bの低下が同一動画内のpaired比較でも残る場合、window文脈・GT形状/輝度・動画全体での
   XYZ正規化との関係を調べる限定診断を次案とする。
3. exposure固有の悪影響が動画・時間区分を考慮しても残る場合だけ、inverse-occurrence loss
   weightingの1要因・5 epoch比較案を検討する（本補足では学習を開始しない）。
4. 強い構造的根拠が残らない場合、W-Aを維持してS5-15の5〜10 epoch pilot計画を返す。

支持・不支持・データ不足を区別し、いずれでも報告して停止する。複数仮説が残る場合も倍率の大小
だけで主要因を決めず、次の診断の切り分け能力と実施量を説明する。

## 8. 非対象・禁止事項（依頼書7章の要約）

新規学習、GPU再推論、grid 16/8以外の追加、teacher/source H5・保存済みpredictionの書き換え、
threshold tuning、aggregation変更、loss変更、augmentation、座標変換forward診断、長期学習は
本補足の対象外。予測結果を見てからのquantile/grid調整、結果を改善するための追加arm探索も
行わない。保存済み`pred_label`を確率から再生成しない。

## 9. 次のアクション

実装（Step S1〜S5）とsynthetic/static検証が完了した。joint集計の再構成可否については、実装前
監査で懸念した「未確認・要監査の項目」（4章）のうち、per-windowのdisagreementのjoint層別
再構成は不可能と結論した（10章の項目1）。次のアクションは、ユーザー実機で
`Stage5/checks/real_h5/check_stage5_s5_14_supplement.sh`を実行し、実数値を本書へ追記して
方針管理チャットへ報告することである。実機実行には次が必要:

- `EVALUATION_DIR`: W-A epoch 5のevaluate_stage5.py出力（`h5_metrics.csv`と`predictions/`を含む）
- `TRAIN_LIST`: W-Aのtrain file list（162 H5パス）
- `FRAME_METRICS_CSV`: 既存のStep H3 `frame_metrics.csv`（Step S3が再利用する）

実装過程で仕様上の重大な不明点は見つからなかった（10章に判断の記録を残す）。

## 10. 実装判断の記録

4章「未確認・要監査の項目」を実装時に次のように解消した。

1. **H4既存artifactのjoint集計可否:** `point_overlap_error_statistics.csv`は
   （動画×split×stratum×relative_frame_decile）と（動画×split×stratum×vote_count_bucket）を
   別々の`group_name`で持つ周辺集計であり、（動画×時間区分×vote_count）のjoint集計は含まない。
   `bin_exposure_disagreement.csv`も（動画×bin）粒度のdisagreement_rateのみで、時間区分や
   vote_countとの交差を持たない。per-windowのdisagreement判定（`OverlapAccumulator`由来）は
   H4のGPU再推論（`run_h4_forward`）でのみ生成され、保存済みH5・prediction・既存CSVからは
   再構成できない（disagreementは同一点への複数window予測間の不一致であり、集約後の
   `pred_label`/`vote_count`だけからは復元不能）。したがって、Step S4のFPR/recallは保存済み
   `pred_label`/`vote_count`から動画×時間区分×vote_countのjointを直接再構成する一方、
   disagreementのjoint層別は実装せず、`stage5_s5_14_supplement_summary.json`の
   `joint_disagreement_status`にH4既存範囲（動画×bin、動画×decile、動画×vote_countbucketの
   別々の集計）を引用したうえで「未検証」と明記する実装にした。二つの周辺集計を結合してjoint
   集計を捏造することはしていない。
2. **train sanity自己除外の実装方法:** train file list（162動画）から動画単位で
   raw GT-positive点数とvalid点数のgrid別bin配列を保持し（`build_train_prior`の`per_video`）、
   評価対象動画のH5パスがtrain listに含まれる場合（train sanityの3動画）だけ、その動画自身の
   bin配列をtotalから差し引いた残差を用いる（`prior_excluding_video`）。validation動画（train
   listに含まれない）は残差計算をせず、全162動画分のpriorをそのまま使う。hot/cold quantile
   分類は、train sanity動画では残差priorから再分類し、validation動画では共有の全体prior分類を
   使う（動画ごとに異なる分類になり得るが、各動画の自己参照を避けることを優先した）。
3. **quantile同値境界の扱い:** 75/25パーセンタイル（`numpy.quantile`、`linear`補間）を
   閾値とし、閾値と同値のbinはhot/coldそれぞれへ包含的に（`>=`/`<=`）判定する。座標順による
   恣意的な分割は行わない。上位下位の閾値が一致する場合（分布の大半が同一値、典型的には
   ゼロ点が大半を占めるbin群）は`status: "undefined_boundary_collision"`として比較を未定義に
   する。synthetic testで両ケースを確認済み（`test_classify_hot_cold_bins_ties_and_boundary_
   collision`）。

## 11. 関連文書

- `docs/stage5/s5-14/stage5_s5_14_supplement_implementation_handoff.md`（実装依頼書、正本）
- `docs/stage5/s5-14/stage5_s5_14_report_to_policy_chat.md`（S5-14 core完了報告、旧解釈の履歴）
- `docs/stage5/s5-14/stage5_s5_14_step_h4_parity_boundary_case_decision_request.md`（H4.1の経緯）
- `docs/stage5/s5-14/stage5_s5_14_structural_diagnostics_implementation_handoff.md`（H2.5の座標provenanceと
  privacy契約）
- `docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.8節（S5-14 core数値正本）
- `docs/stage5/stage5_revision_management_record.md`（S5-14補足節、Decision record D-030・D-031）

---

## 実施記録

### 2026-09-15: 実装・synthetic/static検証

**実装ファイル:**

- `Stage5/checks/real_h5/check_stage5_s5_14_supplement.py`（Step S1〜S5のロジック本体）
- `Stage5/checks/real_h5/check_stage5_s5_14_supplement.sh`（実機実行用bashラッパー）
- `Stage5/checks/dummy/check_dummy_s5_14_supplement.py`（synthetic pass/fail検証、13テスト関数）
- `Stage5/checks/dummy/check_dummy_s5_14_supplement.sh`

**static check:** `py_compile`（両.pyファイル）、`bash -n`（両.shファイル）、`git diff --check`
（対象4ファイル、trailing whitespace等なし）すべて合格。

**synthetic検証:** この実装環境にはh5py/numpyが未導入だったため、検証用に一時的な仮想環境
（`/tmp/s5_14_check_venv`、numpy 2.5.3 + h5py 3.16.0、実装・teacher/predictionデータとは無関係、
このworkspace外のスクラッチ領域）を作成して実行した。13個のsynthetic testすべて合格:

1. 分母付きXY binのFPR/recall（10倍のraw FP点数差が分母補正で同一FPRに収束する例、および
   分母が揃っていても差が残る例）
2. valid/ignoreの分離と、ignore側predicted positive数の別集計
3. 分母0・空binでnull+availability flag（0埋めしない）
4. bin合計とh5_metrics.csv confusion countsの一致（改竄行のfail-fast拒否を含む）
5. hot/cold quantile分類の同値境界処理（q75=q25の`undefined_boundary_collision`、
   および真の分布での閾値包含的な同値binの扱い）
6. train-only prior構築とtrain sanity自己除外（train listに含まれない動画は無変更）
7. prior 2定義（raw_count / rate）の区別（validが0のbinはNaN、GTが0でvalidがあるbinは0.0）
8. hot/cold FPR比較でのcold FPR=0時の比未定義・差は定義可能な扱い
9. 動画横断のhot/cold集計（点数加重pooled値と動画等重みmean/medianの区別）
10. （時間区分×vote_count）jointテーブルが周辺集計の積ではなく真のjoint cellであることの確認
11. paired前半/後半recall比較（片側GTのみの動画を理由付きで除外し、動画構成の異なる別動画に
    引きずられないことを確認）
12. 点数加重recallとframe等重みmean recallの区別
13. H5+prediction .npzを使った統合テスト（構造的`vote_count`再計算によるparity gate、改竄された
    `vote_count`のfail-fast拒否）、およびCLI end-to-end実行でのprivacy/alias検証
    （タイムスタンプ風video ID・絶対hostパスが出力に含まれないこと）

**実機再集計:** ユーザー実機で下記の通り実行し、正常終了を確認した。

```text
EVALUATION_DIR=/mnt/data/3d_projects/stage5_evaluations/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad/last
TRAIN_LIST=/mnt/data/3d_projects/stage5_runs/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad/train_files.txt
FRAME_METRICS_CSV=/mnt/data/3d_projects/models/Stage5/work_dirs/_frame_xy_diagnostics/frame_metrics.csv
  bash checks/real_h5/check_stage5_s5_14_supplement.sh
```

`status: passed`、評価21動画（train sanity 3 + validation 18）、train prior 162動画、
grid 16/8、`vote_count_parity.max_mismatch_any_video: 0`（構造的`window_occurrence_counts()`
再計算と保存済み`vote_count`が全点で一致、再推論なしでのparity確認）。出力行数は
`xy_bin_denominators.csv` 6720行（21動画×(16²+8²)）、`train_xy_prior.csv` 640行、
`xy_hot_cold_video_metrics.csv` 84行（21動画×2 grid×2定義）、`video_decile_counts.csv` 210行
（21動画×10 decile）、`paired_temporal_metrics.csv` 21行、`video_time_exposure_metrics.csv` 84行
（21動画×2 half×vote_count{1,2}）で、いずれも想定通りの件数と一致した。出力ディレクトリへの
privacy scan（タイムスタンプ風video ID・絶対hostパスパターン）は0件。

### 2026-09-15: 実機再集計の結果と分析

#### Step S1/S2: 分母付きXY bin統計とhot/cold FPR比較（仮説A）

`train_xy_prior.csv`のquantile分類は8組（grid{16,8}×definition{raw_count,rate}）すべてで
`status: ok`（`undefined_boundary_collision`は0件）。hot/coldはそれぞれ全eligible binの
ちょうど25%（grid16: 63/256bin、grid8: 16/64bin）で、恣意的な境界分割は発生しなかった。
grid16 `rate`定義の閾値はq75=1.557%、q25=0.216%（train validポイントに対するGT-positive率）。

`xy_hot_cold_video_metrics.csv`（84行）のうち`quantile_status`は全行`ok`、
`self_excluded_from_prior`は12行（train sanity 3動画×2 grid×2定義）で`True`。

**評価対象動画のGT positive数0のbinに限定したhot/cold FPR比較（`FP / (FP+TN)`、動画等重み/
点数加重の両方）は、grid・prior定義・splitの全8通りの集計単位で、比較可能な全動画がhot>coldと
なった（例外なし）:**

| grid | 定義 | split | 比較可能動画数 | hot>cold | hot<cold | 点数加重pooled hot FPR | 点数加重pooled cold FPR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | raw_count | train_sanity | 3/3 | 3 | 0 | 22.2% | 0.044% |
| 16 | raw_count | validation | 18/18 | 18 | 0 | 23.4% | 0.328% |
| 16 | rate | train_sanity | 3/3 | 3 | 0 | 25.0% | 0.011% |
| 16 | rate | validation | 18/18 | 18 | 0 | 24.7% | 0.478% |
| 8 | raw_count | train_sanity | 3/3 | 3 | 0 | 19.2% | 0.162% |
| 8 | raw_count | validation | 18/18 | 18 | 0 | 20.8% | 0.416% |
| 8 | rate | train_sanity | 3/3 | 3 | 0 | 19.6% | 0.126% |
| 8 | rate | validation | 18/18 | 18 | 0 | 20.7% | 0.461% |

grid16 `rate`定義・validationの動画別FPR（一例）は、hot FPR 17.2%〜36.4%、cold FPR
0%〜3.8%（18動画中10動画はcold FPRちょうど0で比は未定義、差のみ使用）で、
`fpr_diff_hot_minus_cold`は全動画で+0.17〜+0.36の範囲に収まり、符号の反転や0近傍への収束は
一件もなかった。動画等重みmean/median（表中略、+0.18〜+0.25の範囲）と点数加重pooled値は
近い水準で一致し、点数の多い少数動画に引きずられた結果ではないことを確認した。

**評価:** globally hot/cold binの「約8.7倍」という当初の指摘は、実際にはmean predicted
**点数**の比だった（仮説の前提上の留保）。しかし、分母をvalid background点数に揃え、
GT positive数0のbinに限定し、train-onlyかつtrain sanity自己除外済みのpriorでhot/cold領域を
定義し直した後も、**hot領域のFPRはcold領域よりも一貫して極めて高い**（差にして約17〜36
ポイント、pooled比で言えば数十倍〜数千倍のオーダーになる動画が多数、ただしcold FPR=0の場合は
比を計算せず差のみを使用）。この効果はgrid16/8・2つのprior定義・train sanity/validation両split・
21動画全てで方向が一貫しており、単一動画やbinの外れ値による偶然とは考えにくい。**分母補正・
自己参照排除という指摘された全ての方法論的懸念を反映した後も、座標事前分布バイアスの構造的な
根拠は強く残った。**

#### Step S3: 動画内前半/後半paired recall比較（仮説B）

`video_decile_counts.csv`（210行=21動画×10decile）から集計した`paired_temporal_metrics.csv`
（21行）:

| split | 対象動画数 | paired可能 | 除外 | 低下 | 上昇 | 同値 | mean diff | median diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train_sanity | 3 | 3 | 0 | 3 | 0 | 0 | -38.4% | -46.0% |
| validation | 18 | 12 | 6 | 7 | 5 | 0 | +1.2% | -1.2% |

除外6動画の理由は全て片側decile（前半または後半）にGT-positive点が存在しないため
（`no_gt_positive_in_first_half`4件、`no_gt_positive_in_second_half`2件）で、0埋めせず除外した。

個別動画のrecall diff（後半-前半）を見ると、validationでは-35.1%（validation_012）から
+57.7%（validation_011）まで大きくばらつき、方向も7勝5敗と割れている。train sanity 3動画は
全て低下（-46.0%、-48.5%、-20.7%）で一貫するが、n=3であり、うち1動画は依頼書固定の
`FIXED_TRAIN_VIDEO`、残り2動画はseed固定の乱数選択であって代表性を主張できるサンプルではない。

**評価:** S5-14 core報告の「decile 0の64.2%からdecile 8の7.9%への単調急落」は、複数動画の
frame集合をdecile単位でpoolした集計であり、動画構成（どの動画のどのdecileにGT-positive点が
多いか）の影響を受け得る、という当初の留保が実機データでも裏付けられた。同一動画内でpairする
と、**validationでは系統的な低下は再現されず**（動画間で方向が割れ、mean/medianともほぼ0）、
**train sanity（n=3、非代表サンプル）でのみ強い一貫した低下が見られる**。この構成差により、
仮説Bの「時間位置による系統的なrecall低下」は、pooled集計で見えていたほど動画横断で一般化
できる効果ではないと考えられる。train sanityの3動画固有の要因（GT形状、動画長、window境界
との相互作用等）である可能性が残るが、本補足の再集計範囲では動画横断の一般的な時間効果としては
支持されない。

#### Step S4: vote_count×時間区分の層別FPR/recall（仮説C）

`video_time_exposure_metrics.csv`（84行）を動画横断でpoolした結果:

| split | 時間区分 | vote_count | FPR | recall | valid background数 | valid positive数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train_sanity | 前半 | 1 | 15.34% | 77.33% | 228,436 | 1,213 |
| train_sanity | 前半 | 2 | 11.94% | 66.46% | 70,059 | 3,799 |
| train_sanity | 後半 | 1 | 11.40% | 20.24% | 161,431 | 573 |
| train_sanity | 後半 | 2 | 13.39% | 50.42% | 157,652 | 4,171 |
| validation | 前半 | 1 | 12.40% | 39.22% | 1,365,134 | 7,919 |
| validation | 前半 | 2 | 12.15% | 53.89% | 2,231,552 | 25,714 |
| validation | 後半 | 1 | 10.78% | 未定義（GT-positive点0） | 638,118 | 0 |
| validation | 後半 | 2 | 11.23% | 40.02% | 2,864,156 | 41,400 |

FPRはvote_count 1/2間、前半/後半間のいずれで層別しても10.8%〜15.3%の狭いレンジに収まり、
S5-14 core報告のvote_count単独比較（11.88%/11.64%）と整合する結果が時間区分で層別しても
崩れなかった。recallはvote_count 2の方が高い傾向が複数セルで見られるが、window境界・tail
window構成に起因する構造的な交絡（vote_count自体がframe位置と相関し得る）を排除できておらず、
本補足の範囲では原因を特定しない。

disagreementのjoint層別（動画×時間区分×vote_count）は、4章で判断した通り既存H4 artifactの
周辺集計から再構成不能なため実装せず、`joint_disagreement_status`に理由を記録して「未検証」の
まま維持した。

**評価:** FPRはvote_count・時間区分のいずれで層別してもほぼ一定であり、「exposureが多いほど
FPが増える」という単純な関係は本補足でも支持されない。disagreement側のjoint検証は依然として
未実施であり、training exposureとFPの因果関係についての結論は変わらず持ち越しとなる。

#### 3仮説の総括（本補足の再集計結果に基づく、数値と留保付き）

| 仮説 | 本補足での結果 | 根拠 | 留保 |
| --- | --- | --- | --- |
| A. 座標事前分布 | 分母補正・自己参照排除後も**強く残る** | hot FPR 17〜36pt高い、21動画・grid16/8・prior 2定義全てで方向一貫 | bin単位のraw point密度自体の交絡（依頼書仮説Aの原文の懸念）は本補足でも未確認 |
| B. 時間位置 | validationでは**再現されず**（train sanity n=3のみ低下） | validation mean diff+1.2%・median-1.2%、7勝5敗で方向不定 | train sanity 3動画固有の要因の可能性は未確認。pooled集計は動画構成の交絡を含んでいた可能性が高い |
| C. overlap exposure | FPRへの効果は**時間区分別でも小さいまま** | 全8セルでFPR 10.8〜15.3%のレンジ内 | disagreementのjoint検証は未実施のまま。recallのvote_count差の原因は未特定 |

依頼書8章の分岐に照らすと、**仮説Aの偏りが分母補正後も動画別・両gridで残った**ため、
再学習なしの座標変換診断（local pixel座標とmodel入力XYZの違いを考慮したもの）を次案とすることが
妥当と考えられる。仮説Bの低下は同一動画内でも一般には再現されなかったため、時間文脈の限定診断を
仮説Aと同等の優先度で進める根拠は本補足の範囲では得られなかった。仮説Cのexposure固有の悪影響も
時間区分を考慮して残らなかったため、inverse-occurrence loss weightingの比較案を優先する根拠も
得られなかった。倍率の大小のみで主要因を順位付けせず、Aについては複数の独立した層別（動画別・
grid別・定義別）で方向が一貫した点を重視した。

## 12. 判断と次工程（追記）

方針管理チャットは本補足の結果を受け入れ、仮説Aの座標事前分布バイアスを優先し、再学習なしの
座標変換診断（S5-14補足2）を委任した（管理記録 Decision record D-032、D-033）。
`docs/stage5/s5-14/stage5_s5_14_supplement_implementation_handoff.md` 10章に補足2の固定条件・8変換条件・
実装手順が追記されている。本書（補足1）はここまでの内容で完結とし、以後の実装・検証・実機結果は
新しい完了報告書 `docs/stage5/s5-14/stage5_s5_14_supplement2_report_to_policy_chat.md` へ記録する
（依頼書10.6節の指定どおり、本書と成果物は保持する）。
