# Stage 5 S5-14補足: 点密度補正と動画別・時間別再集計 実装依頼書

作成日: 2026-09-15

最終更新日: 2026-09-15

作成元: Stage 5方針管理チャット

状態: S5-14補足は実装・実機検証完了、管理チャットによる受入済み。補足2は方針決定済み、実装・検証未着手。

管理チャット追記（2026-09-15）: **次の委任対象は10章のS5-14補足2（座標変換診断）です。**
1〜9章は完了した初回補足の依頼履歴として保持します。その「GPU再推論0回」は初回補足にのみ
適用し、補足2では10章の限定forwardを許可します。新規学習・production変更は禁止のままです。

S5-14 core（Step H1〜H5）の成果物・parity検証は受け入れました。本書は、次の改修を選ぶために
残った点密度・動画構成・時間位置の影響を確認する、限定的な補足検証の実装依頼です。
既存H5・保存済みprediction・H4成果物を使用し、新規学習0回、GPU再推論0回で実施してください。

## 1. 最初に読む文書

1. `docs/stage5/stage5_revision_management_record.md`
   - 6章「S5-14補足 点密度補正と動画別・時間別再集計」
   - Decision record D-030、D-031
2. `docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.8節
3. `docs/stage5/s5-14/stage5_s5_14_report_to_policy_chat.md`
   - H3/H3.1の数値・留保、H4.1の修正、H4層別集計、H5実施結果
4. `docs/stage5/s5-14/stage5_s5_14_step_h4_parity_boundary_case_decision_request.md` 8〜9節
5. `docs/stage5/s5-14/stage5_s5_14_structural_diagnostics_implementation_handoff.md`
   - H2.5の座標provenanceとprivacy契約
6. `docs/stage5/FILES.md`、`docs/stage5/data_construct.md`

管理記録を現行方針の正本、評価レポートを数値の正本としてください。core報告の旧見出しや
「最終判定」は履歴として読み、今回の留保を反映して補足の判定を返してください。

## 2. 実装理由と目的

### 仮説A: 点数と率を区別する

globally hot/cold binの「8.7倍」はmean predicted点数320.8対36.8の比較です。
bin内のbackground点数が多ければFP点数も増えるため、これだけでは座標事前分布の学習を断定できません。
必要なのは分母を揃えたFPR比較です。GT positiveがない領域にもignore点は存在し得るため、
predicted positiveの全点数をそのままFPとして扱ってはいけません。

### 仮説B: 同じ動画内で時間的な低下を確認する

decile 0のmean recall 64.2%とdecile 8の7.9%には対象動画・GT点数の構成差が含まれ得ます。
decile 3は70.4%であり、「単調低下」とは記述しません。decile 9の未定義recallも0で埋めません。
同一動画内の前半/後半比較と、各層の対象数を使って関連が残るか確認します。

### 仮説C: Exposureとdisagreementを区別する

vote count 1/2のFPRは11.88%/11.64%で大きく変わりません。一方、vote count 2のFPでdisagreement率が
高いことは、予測の不一致と誤りの関連を示しますが、training exposureがFPを増やした証明ではありません。
動画・時間位置を考慮した集計で、次のloss比較を支持する根拠が残るか確認します。

H5の`hypothesis_verdicts`は既存解釈の固定文字列です。今回の再集計から独立に判断し、倍率の単位や
分母が異なるA/B/Cを、倍率の大小だけで順位付けしないでください。

## 3. 正式判断と固定条件

1. S5-14 coreの完了を維持し、今回の作業はS5-14補足として別に記録する。
2. W-Aを暫定baselineとして維持する。
3. S5-15の学習前に、一度の補足再集計を行う。
4. 再学習、GPU再推論、augmentation・loss変更は本補足の対象外とする。

| 項目 | 固定値 |
| --- | --- |
| teacher | v7 `bboxrank_v7_cvat_authoritative_crop_quality_v1`、180 H5 |
| split | W-Aのtrain 162 / validation 18。既存file listを使用 |
| checkpoint | W-A epoch 5、既存predictionの生成checkpoint |
| normalization | GroupNorm、8 groups |
| class weight | `[0.05963856, 1.94036150]` |
| label policy | `bbox_noncontour_ignore` |
| window | size 16 / stride 8 / tailあり |
| aggregate判定 | mean probability、保存済み2クラスargmax、同値background |
| XY座標 | H2.5で確認したlocal image 256×256、`pixel_xy / 255` |
| prediction評価 | 固定train sanity 3動画 + validation 18動画 |
| prior作成 | train 162動画のGTのみ。train sanityは対象動画を除外 |
| grid | 16と8のみ |

## 4. 入力監査と実装配置

既存checkerの責務と再利用可能な関数を先に確認してください。

```text
Stage5/checks/real_h5/check_stage5_frame_xy_diagnostics.py
Stage5/checks/real_h5/check_stage5_per_window_context_diagnostics.py
Stage5/checks/real_h5/check_stage5_xy_coordinate_provenance.py
Stage5/checks/real_h5/check_stage5_s5_14_summary_export.py
```

新規checkerは例えば次へ配置し、対応bashに実機の入出力パスを設定できるようにします。

```text
Stage5/checks/real_h5/check_stage5_s5_14_supplement.py
Stage5/checks/real_h5/check_stage5_s5_14_supplement.sh
Stage5/checks/dummy/check_dummy_s5_14_supplement.py
Stage5/checks/dummy/check_dummy_s5_14_supplement.sh
```

既存bashのPython指定・path overrideの形式を踏襲してください。CPU/NumPy/h5pyで完結させ、
GPU modelをimportする経路を避けます。実機の外部パスへこのworkspaceからアクセスできない場合は、
実行bashと確認コマンドを用意してユーザー実機で検証します。

W-A file list、評価対象manifest、H2.5寸法監査、H3〜H5 artifactのsplit/video対応を確認します。
匿名aliasは既存mappingを使用し、ファイルの並び順だけで別checkerのaliasを対応させないでください。
予測の`point_indices`、点数・順序、H5 labels/valid mask、vote countを検証します。
保存済み`pred_label`を正本とし、positive確率の`>=0.5`で再生成しないでください。

## 5. 実装・検証手順

### Step S1: 分母付きXY bin統計

H5とpredictionから動画×grid×bin単位で次を再集計します。

- 全点数、valid点数、valid positive数、valid background数、ignore数
- TP、FP、TN、FN
- 全点predicted positive数と、ignore上のpredicted positive数
- `FPR = FP / (FP + TN)`、`recall = TP / (TP + FN)`
- valid点中のpredicted positive率とGT positive率

同一率の分子・分母は必ず同じ点集合で計算し、分母0はnull/NaNとavailability flagを出します。
各動画のbin合計が既存W-A confusion countsに完全一致することを確認してください。

### Step S2: Train GT由来のXY priorとhot/cold比較

train 162動画からgrid別の共通GT分布を作成します。validation GTをpriorの作成やhot/cold領域の選定に
使用しません。train sanityの3動画は各対象動画の寄与をtrain集計から引いて評価します。
モデルを追加学習する工程はありません。

priorは次の2定義を別々に出力します。

1. raw GT-positive point count（coreの点数分布に対応する参考値）
2. valid point countで割ったGT-positive rate（分母付きの主要比較）

各定義についてtrain valid点が存在するbinを対象に上位/下位25%をhot/cold候補とし、quantileの
計算法・同値境界の扱い・実際のbin数を保存してください。同値binを座標順で恣意的に分割しないこと。
上下境界が一致し領域を分離できない場合は、その定義の比較を未定義として報告します。
予測結果を見てquantileやgridを追加調整しないでください。

各評価動画でGT positive数0のbinを主要対象とし、valid background点に対するhot/cold FPRを計算します。
この対象条件は評価時の層別にのみ使い、train priorは変更しません。ignore上の予測は別表にします。
hot/cold両方にbackground分母がある動画でFPR差・比を比較し、点数加重の全体値と動画等重みの
mean/median、比較可能動画数・増減方向の件数を報告します。cold FPRが0の場合にepsilonを足して
有限の倍率を作らず、比は未定義としFPR差を使用します。train sanityとvalidationは別々に集計します。

### Step S3: 時間位置の動画内比較

既存`frame_metrics.csv`から動画×decileのTP/FP/TN/FN、GT点数、frame数を集計します。
既存のframe等重みmean recallも参考値として保持し、点数から計算するrecallと区別してください。
split×decileでは参加動画数、GT-positive frame数・点数、recall計算可能な動画数を明示します。

主要なpaired比較は前半decile 0〜4と後半5〜9です。両方にvalid GT-positive点がある同一動画で
recallの後半-前半差を計算し、動画別値、mean/median、低下/上昇/同値件数を報告します。
片側にしかGTがない動画はpaired比較から除外し、理由・数を保存します。少数GT点に基づく率は
点数を併記して解釈し、都合のよい最小点数閾値を後から探索しないでください。

H5のframe_orderから既存window関数でoccurrence countを復元し、保存済みvote_countと一致を確認して、
可能な範囲で動画×前半/後半×vote count 1/2の同様の比較を追加します。
比較可能な層が少なければ、独立した時間効果とは断定せずデータ不足と報告します。

### Step S4: Exposureとdisagreementの層別整理

動画×時間区分でvote count 1/2のFPR・recallを比較します。S3のjoint集計を再利用してください。
H4の既存CSVが時間位置とvote countの周辺集計しか持たない場合、両表を結合してjoint集計を捏造しないこと。

disagreementは保存済みper-point/window artifactがある場合だけ同じjoint層で集計します。
aggregate predictionのmean/vote countから各windowのclassを逆算することはできません。
復元不能なら既存H4の確認済み範囲を引用し、joint層別は未検証とします。補うためのGPU再推論は行いません。

FPとdisagreementの関連、exposureと誤り率の関連を別々に結論付けてください。
観察的な関連だけでinverse-occurrence loss weightingの効果を確定しないでください。

### Step S5: 検算・集約・報告

少なくとも次のsynthetic検証を行います。

- raw点数差が大きくてもFPRが同じになる例と、分母補正後もFPR差が残る例
- valid/ignoreの分離、分母0、cold FPR0、空bin
- train-only prior、train sanity自身の寄与除外、quantile同値の扱い
- 動画構成が変わるとpooled時間傾向が動画内傾向と異なる例、paired対象の選択
- joint集計と周辺集計の区別、既存confusion countsとの一致
- 匿名化出力の実video ID・host path検出

static check（`py_compile`、`bash -n`、`git diff --check`）を行い、実機で限定再集計を一度実施します。
実装不具合の修正・再検算は可能ですが、結果を改善するための追加arm・grid・閾値探索は行いません。

## 6. 出力とprivacy

新しい補足用出力先を使用し、core成果物を保持してください。出力例:

```text
xy_bin_denominators.csv
train_xy_prior.csv
xy_hot_cold_video_metrics.csv
video_decile_counts.csv
paired_temporal_metrics.csv
video_time_exposure_metrics.csv
stage5_s5_14_supplement_summary.json
```

summaryにdataset/checkpoint識別情報、split件数、grid、prior定義、分母、quantile、時間区分、
入力artifactとの対応、parity、未定義数・未検証項目を記録します。仮説判定は根拠数値と留保を伴う
分析結果として報告し、無条件に`clearly_supported`等を出力する実装にしないでください。

実path・video mappingはprivate、共有出力は既存匿名aliasを使用します。schemaで許可した列のみを
shareableへ出し、bundle全体のprivacy self-checkに合格させてください。

## 7. 検証量・非対象

本補足は一度の限定再集計です。新規学習0回、GPU再推論0回、grid 16/8のみとします。
不足artifactは一覧化し、再集計不能な項目を未検証として返してください。
teacher/source H5・保存済みpredictionの書き換え、threshold tuning、aggregation変更、loss変更、
augmentation、座標変換forward診断、長期学習は実施しません。

## 8. 判定と結果後の分岐

1. Aの偏りが分母補正後も動画別・両gridで残る場合、再学習なしの座標変換診断を次案にする。
   local pixel座標とmodel入力XYZの違いも考慮し、相関だけで座標記憶を確定しない。
2. Bの低下が同一動画内でも残る場合、window文脈、GT形状・輝度、動画全体でのXYZ正規化との
   関係を調べる限定診断を次案にする。
3. exposure固有の悪影響が動画・時間区分を考慮しても残る場合だけ、inverse-occurrence loss weighting
   の1要因・5 epoch比較案を検討する。今回の依頼では学習を開始しない。
4. 強い構造的根拠が残らない場合、W-Aを維持してS5-15の5〜10 epoch pilot計画を返す。

支持・不支持・データ不足を区別し、いずれでも報告して停止します。複数仮説が残る場合も倍率の大小で
主要因を決めず、次の診断の切り分け能力と実施量を説明してください。

## 9. 完了条件と記録先

1. synthetic/static検証と実機再集計の結果を記録している。
2. 既存W-Aのsplit・point alignment・confusion countsと一致している。
3. 分母付きXY比較、動画内時間比較、exposure/disagreementの確認可能範囲が示されている。
4. 対象数・未定義・未検証項目を明記し、privacy self-checkが合格している。
5. 仮説ごとの根拠・留保・次の選択肢を記述している。
6. 次の文書を更新している。
   - `docs/stage5/stage5_revision_management_record.md`: 補足進捗・結果要約。未承認の改修を採用済みにしない。
   - `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`: 9.8節に補足数値とcore解釈の訂正・留保を追記。
   - `docs/stage5/FILES.md`: 実装ファイルと出力の配置。
7. 方針管理チャットへの報告を次に作成している。

```text
docs/stage5/s5-14/stage5_s5_14_supplement_report_to_policy_chat.md
```

完了報告には実行条件、入力件数、検算結果、分母付き指標、paired比較可能動画数、各仮説の判定、
未検証理由、次の推奨ステップを含めてください。補足結果の報告後、方針管理チャットの判断を待ちます。

## 10. 管理チャット追記: S5-14補足2 座標変換診断

方針決定・委任日: 2026-09-15。担当: Stage 5実装チャット。状態: 実装・検証未着手。

### 10.1 前補足の受入と今回の目的

`docs/stage5/s5-14/stage5_s5_14_supplement_report_to_policy_chat.md`の完了を受け入れます。
分母補正とtrain sanityのleave-one-video-out後も、全21動画・grid16/8でhot領域のFPRがcoldより
高い結果でした。background点数の違いだけでは説明できませんが、座標の直接的な記憶と
局所形状・輝度・点間隔への反応は未分離です。「密度の交絡未検証」と一括せず、点数分母は補正済み、
局所密度等の交絡は未分離、と区別してください。

時間位置の低下はtrain sanityで残るもののvalidationでは一貫せず、exposure別FPRにも明確な
悪影響がないため、時間文脈診断・inverse-occurrence loss比較を先行させません。
joint disagreementは未検証のままです。今回のforwardを利用して探索対象を広げないでください。

補足2では同じ点群を剛体変換し、相対距離・点の特徴・ラベルを固定した状態で予測の感度を調べます。
これはaugmentationの採用試験ではなく、次に試す改修を選ぶための再学習なしの診断です。
管理記録6章「S5-14補足2」、D-032/D-033と本章を現行方針としてください。

### 10.2 固定条件と座標契約

3章のW-A条件、固定train sanity3動画＋validation18動画を維持します。実際のcheckpoint/configを
照合し、features=`intensity,confidence`、GroupNorm8、全動画XYZ正規化、window16/8・tailあり、
physical batch1・paddingなし・samplingなし、eval modeを確認してください。

実装前に次の経路を読み、どこで変換するかを報告・コードに明記してください。

- `Stage5/stage5/utils/feature_normalization.py`の`normalize_xyz()`。
- Datasetと`Stage5/evaluate_stage5.py`の全動画正規化・window抽出経路。
- `Stage5/stage5/models/pointnext_s_segmentor.py`の`points -> pos`と特徴の受け渡し。
- H4 checkerのcanonical accumulationとH4.1 parity修正。

変換対象は**全動画正規化後のモデル入力XYZ**です。全動画へ同じ変換を適用してから元のwindowを
取り出し、変換後に再正規化・再中心化・clipをしません。正規化前の一様平行移動はcenteringで
相殺されるため、今回の位置感度テストには使いません。

H2.5の`pixel_xy / 255`は画像上の診断座標です。これだけ変更してもW-Aの入力は変わりません。
XYZのX/Y/Zと画像pixelの軸の対応は生成コードから確認し、対応未確認ならモデル軸としてのみ解釈します。
画像座標・frame_order・point_indices・GT・valid_mask・intensity/confidenceは元点に付随して固定し、
元のhot/cold分類も固定します。変換されたXYZを画像pixelへ推測的に換算しないでください。

### 10.3 条件と実施量の上限

正規化XYZをpとし、以下の8条件だけを事前固定します。平行移動量は動画正規化後の単位です。

| 条件 | 変換 |
| --- | --- |
| identity | 元のp。既存baselineとのparity確認 |
| repeat_identity | 同じpを独立に再forward。実行上の揺らぎのcontrol |
| translate_x_plus / minus | p + (±0.1, 0, 0) |
| translate_y_plus / minus | p + (0, ±0.1, 0) |
| rotate_z_plus / minus | 原点中心のZ軸回転、±15度。回転行列を保存 |

正規化前translationが正規化で相殺される確認はCPU synthetic testに限定します。
初回はreflection・scale・shear・点削除・特徴の変更を行いません。振幅、軸、seed、checkpointの
追加探索も禁止です。rotationは相対距離を保存しますが、方向性のある局所形状も変わるため、
translationと同じ意味の「位置だけの介入」とは扱いません。

新規学習0回、最大21動画×8条件＝168動画相当のforward走査とします。1走査はその動画の全windowを
処理する単位であり、GPU呼び出し168回という意味ではありません。固定順のtrain sanity1動画と
validation1動画でpreflightし、正常出力を本集計へ再利用してください。同一条件の不要な再実行は
避け、失敗修正に伴う再実行は理由と実施量を記録します。追加実験には管理チャットの承認が必要です。

### 10.4 実装・検証の手順

1. **T1 入力監査とCPU synthetic test**: checkpoint/config・元prediction・点対応・window一覧を照合。
   identity、平行移動、回転・逆変換、相対距離保存、特徴/GT/index不変性を検証する。
   距離検算は全点の二乗距離行列を作らず、固定seedの点対等を使い、float32誤差基準を明記する。
2. **T2 identity preflight**: H4.1と同じcanonical経路で既存保存値と比較する。
   2クラス確率をfloat64で加算・平均しfloat32化後argmax、同値backgroundとする。
   点ごとのpred_label・vote count・confusion countsは完全一致を要求し、確率差も記録する。
   `p1 >= 0.5`への置換は禁止。repeat identityの判定不一致やbaseline parity不合格は停止・原因報告し、
   自動的な閾値緩和・境界点除外で通さない。
3. **T3 限定変換forward**: 同じ前処理と点順序を保持し、モデル入力XYZだけを変更する。
   seed/RNG条件とbackend設定はarm間で揃えて記録する。変換行列、点数、座標範囲、単位球外点率も保存。
   frame_orderによるwindow所属とvotesがidentityと一致することを検算する。
4. **T4 全21動画のpaired集計**: 元H5のpoint_indicesで同じ点を比較する。
   変換後PLYを出す場合も元座標表示を既定とし、GTとの対応を崩さない。
5. **T5 記録・匿名化・報告**: 後述の完了条件を確認して停止し、次の学習は開始しない。

既存checkerの入出力・canonical集約を再利用し、productionコードの既定挙動は変更しません。
例えば`Stage5/checks/real_h5/check_stage5_coordinate_transform_diagnostics.py`と対応bashを追加します。
bashは既存GPU checkerに合わせ、dualtrack311のPYTHON、対応するCONDA_PREFIX/CUDA_HOME、
TORCH_CUDA_ARCH_LIST=12.0、torch/libとconda/lib・lib64のLD_LIBRARY_PATHを設定してください。
入力・出力パスはbash内で指定可能とし、既存H5・checkpoint・予測を上書きしません。

### 10.5 必須指標と解釈の限界

動画×変換ごとに、identityに対する次の指標を出してください。

- positive確率差のsigned mean、absolute mean/p95/max、判定反転率と方向別点数。
- valid GT別のTP/FP/FN/TN、precision/recall/FPR/F1/IoUと差分、TP0動画数。
- 元点のhot/cold領域別FPRと差分。前補足のtrain-only prior、train sanity自己除外、分母条件を再利用。
- ignore領域の予測positive率は別集計とし、FPとは呼ばない。
- split合算と動画別median・改善/悪化動画数。未定義率は0補完せず対象動画/点数を併記。

変換差はrepeat identityの揺らぎと並べ、translationの正負・軸間、rotationで分けて解釈します。
感度が大きくても、PointNeXtが厳密な平行移動/回転不変を保証するという前提は置きません。
正規化後の分布外入力、方向性のある局所形状、浮動小数点によるFPS/近傍選択の差も影響し得ます。
「不具合確定」「絶対座標の暗記確定」「augmentation採用」と飛躍させないでください。
逆に、限定した小変換で差が小さいことも全ての座標依存の否定にはなりません。

### 10.6 完了条件・報告先・次の判断

synthetic検証、canonical parity、点対応/距離/特徴不変性、全21動画の条件別指標、実施量、
未定義・未検証項目を記録してください。匿名化bundleは数値と匿名IDのみを共有し、実video ID・
絶対path・元H5/PLY/prediction・対応表を含めないprivacy self-checkを行います。

管理記録6章と評価レポート9.8節には「S5-14補足2」として別項目で追記し、`docs/stage5/FILES.md`に
配置を反映してください。初回補足の報告書や成果物は保持し、新しい完了報告を次へ作成します。

```text
docs/stage5/s5-14/stage5_s5_14_supplement2_report_to_policy_chat.md
```

完了後の選択肢は、根拠が得られた場合のaugmentation単独5 epoch比較案、またはW-Aを維持した
S5-15 pilot計画です。どちらも管理チャットが改めて判断します。本章は学習開始、loss/label policy変更、
production normalization/aggregation/threshold変更を承認するものではありません。
