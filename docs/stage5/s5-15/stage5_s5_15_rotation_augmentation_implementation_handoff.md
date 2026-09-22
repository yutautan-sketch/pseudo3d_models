# Stage 5 S5-15: 回転augmentation比較と段階的学習 実装依頼書

作成日: 2026-09-15
最終更新日: 2026-09-15
作成元: Stage 5方針管理チャット
担当: Stage 5実装チャット
状態: 方針策定済み。実装・実行未着手。

## 1. 今回の依頼と承認境界

S5-14全体（core・補足1〜3）はユーザー判断により完了・コミット済みです。
管理記録6章「S5-15 長期学習とproduction候補確定」に従い、**P1〜P3の準備・実装・検証**を
お願いします。最初は既存コードとW-Aの実条件を監査し、P1の結果とP2の実装方針を提示してください。
明確な範囲のコード実装・CPUテストは進めて構いませんが、GPU preflightと5 epoch学習は、
実施対象・回数・出力先を示したうえで明示的な実行承認を得てください。

P1、P2、P3の各段階で結果を返し、次段階への承認を確認します。P4以降は見通しを共有するための
計画であり、今回の委任で10〜200 epochへ自動継続することは認めません。
同じコマンドの再実行も計算量の追加です。修正後のやり直しを「元の承認範囲内」と自己判断せず、
理由・対象・回数を提示して追加承認を得てください。実行コマンドの提示と実行承認を区別します。

## 2. 最初に読む文書

1. `docs/stage5/stage5_revision_management_record.md` 6章S5-15（P1〜P6）、D-034。
   方針・現在状態の正本です。本依頼との齟齬があれば開始前に相談してください。
2. `docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.8.3〜9.8.5節。
   診断結果・留保の正本です。初期teacherや旧padding runを今回のbaselineと混同しないでください。
3. `docs/stage5/s5-14/stage5_s5_14_supplement2_report_to_policy_chat.md`の「S5-14補足3 完了報告」。
   旧集計・訂正履歴も含みます。生成CSVとの転記留保は文書照合で処理し、S5-14を再開しません。
4. `docs/stage5/s5-13/stage5_s5_13_report_to_policy_chat.md`および
   `docs/stage5/s5-13/stage5_s5_13_supplement_report_to_policy_chat.md`。W-A採用、W-B/W-C不採用の経緯。
5. `docs/stage5/FILES.md`、`docs/stage5/data_construct.md`。実装配置・データ構造・共有範囲。

Stage5はStage2to4生成のpseudo-3D H5を使うpoint-wise segmentationです。
動画全体XYZを正規化し、frame_order基準のoverlap windowをDataset sampleとして学習します。
可変点数に対するzero paddingの影響を避け、physical batch1＋gradient accumulation8としています。
実データ・checkpointは実機の`/mnt/data/3d_projects/`以下です。workspaceに存在しない場合は
実行済みとせず、実機で必要な監査コマンド・スクリプトを用意してください。

## 3. 背景と正式判断

- S5-14で平行移動への低感度、回転への明確な感度がvalidation単独でも確認されました。
- 回転でFPが減ってもrecallが低下する条件があり、絶対座標の暗記や不具合は確定していません。
- 推論時の＋15度が有利だったことを根拠に、片方向回転や推論時回転を採用しません。
- 今回は学習時のランダム回転だけを変え、有効性を短期比較します。採用は結果を見て別途判断します。
- W-Aをcontrolとして維持します。loss、label policy、class weight、normalization方式、
  sampling、aggregation、thresholdを同時に変えません。
- 追加の座標診断・seed/角度探索・長期学習は自動で行いません。

## 4. P1: 固定条件と実験manifest

| 項目 | 条件 |
| --- | --- |
| teacher | v7 `bboxrank_v7_cvat_authoritative_crop_quality_v1`、180 H5 |
| split | W-A保存済みtrain162 / validation18 file list。再分割禁止 |
| 評価対象 | 固定train sanity3動画＋全validation18動画 |
| モデル | PointNeXt-S、GroupNorm8 groups、W-Aと同一width/radius/nsample等 |
| 初期重み | W-Aの学習開始に使ったGroupNorm転移初期checkpoint |
| features | intensity,confidence |
| label policy | bbox_noncontour_ignore |
| loss / weight | CE、smoothing0、固定weight [0.05963856, 1.94036150] |
| optimizer | AdamW、lr1e-3、weight_decay1e-4、grad_clip_norm10 |
| window | size16 / stride8 / tailあり |
| batch | physical1 / accumulation8、paddingなし |
| sampling | overlapの実点を維持。random point removalなし |
| 評価 | augmentationなし、eval mode、mean probability、既存2クラスargmax |
| seed | W-A実configと同一。augmentation用乱数は独立 |
| 保存 | 各epoch checkpoint、best/last、config、metrics、file lists、manifest |

W-Aの**epoch5学習済みcheckpointから開始しないでください**。両armの初期checkpoint hashを
照合します。GroupNormと初期重みを明示し、`train_stage5.sh`の既定値を無検査で流用しません。
固定weightの精度はW-A実configと照合し、auto再計算による変更を避けてください。

manifestにはGT・file list・初期重みの同一性、seed、実config、コードrevisionと未コミット差分の
有無、ライブラリ/GPU環境、run/output paths、予定実施量を記録します。
旧W-Aとの不一致があれば学習前に報告し、差分を黙って新しいbaselineへ持ち込まないでください。

## 5. P2: 実装契約

### 5.1 挿入場所と変換

`Stage5/train_stage5.py`、Dataset/loader、`Stage5/stage5/utils/feature_normalization.py`、
PointNeXt wrapper、`Stage5/evaluate_stage5.py`を読んで、既存の責務に沿う最小変更を選んでください。
診断checkerの変換をそのままtrainingへ流用するのではなく、epoch/workerへの伝達を設計します。

全動画XYZ正規化後・window抽出前に、動画重心に対応する原点を中心としてモデルZ軸回転を行います。
train動画×epochごとに一つの角度をUniform[-15度,+15度]で生成し、同じ動画の全windowに共有します。
元のpixel_xyは画像上の診断座標であり、モデルXYZと混同しません。ラベルと点対応は変えません。

設定はnone / random_z_rotation、回転範囲、augmentation seedに限定し、無効を既定値にします。
引数名は既存CLIに合わせて選び、config/checkpointへ保存してください。
評価CLIが学習configを読む場合でもaugmentationを評価へ適用してはいけません。

### 5.2 再現性と不変条件

- base seed・epoch・安定した動画IDから角度を生成し、Pythonのプロセス依存hashを使わない。
- augmentation用RNGをモデル・shuffle用RNGから分離し、無効時に乱数列を余分に消費しない。
- worker数、取得順、persistent workerの有無で同一動画/epochの角度が変わらないようにする。
- epoch情報のworkerへの伝達を検証し、角度が全epoch固定になる不具合を防ぐ。
- 元点・cacheをin-placeで書き換えず、回転が累積しないようにする。
- 距離、点順、intensity/confidence、GT、valid_mask、frame_order、point_indicesを維持する。
- window所属・overlap点の対応を維持し、windowごとの再中心化・別角度付与をしない。
- 回転後の再正規化・clip、translation/scale/reflection、点削除を追加しない。
- validation、train sanity評価、inferenceは常に無変換とする。

### 5.3 必須テストとpreflight

CPU syntheticで角度範囲・距離保存・同一動画/epochの一致・epoch更新・worker再現性・元データ
不変性・none時の既存出力との一致を検証してください。学習/eval切替とcheckpoint/config保存も確認します。
回転行列の許容誤差と、点index/label等の完全一致を区別してください。

GPU preflightはdummy forward/backwardと実H5少数stepに限定します。finite loss/gradient、
optimizer更新、点対応、評価無変換を確認し、旧runへ保存しません。対象動画/step数、実行回数、
停止条件をP1報告後に提示して承認を得てください。フル1 epoch smokeを無条件に追加しません。
NaN/Inf、GT/点対応不一致、none時の回帰は停止・報告し、再実行は別承認とします。

## 6. P3: 学習・評価の手順

### 6.1 比較armと実施上限

| arm | augmentation | 期間 |
| --- | --- | --- |
| R0 | none | 同一転移初期重みから5 epoch |
| R1 | random_z_rotation、[-15度,+15度] | 同じ初期重み・seedから5 epoch |

通常は最大2 run、計10 training epochs。R0に既存W-Aを再利用する場合は、P1/P2で初期重み・
データ・config・seed・optimizer更新条件・環境・none時の経路に実質的な差がないことを確認し、
根拠を提示して了承を得てください。再利用不能ならR0を新規学習し、旧W-Aは履歴とします。
再利用した場合の新規学習はR1のみ5 epochです。追加seed・角度探索・3本目のrunは認めません。

学習開始前に両armのコマンド、manifest、出力先と予定実施量を提示してください。
各epochのloss・confusion・ignore/weight関連の既存debug metricsを保持し、各epoch checkpointと
best/lastを保存します。新規scheduler等は追加しません。

### 6.2 固定21動画の評価

主比較は両armのepoch5。bestは同じ選択規則で保存・副次評価し、異なるepochのbestだけで
augmentation効果を判定しません。epoch5とbestが同一なら重複評価を避け、その事実を記録します。
評価checkpoint・動画数・再利用範囲も実行前に示してください。

既存pipelineで無変換・mean aggregationの予測、metrics、GT/予測/positive-only PLYを作成します。
元座標・同じ動画/視点で比較し、raw/annotated reference PLYの収集も既存仕様を維持します。
以下をtrain sanityとvalidationで分けて報告してください。

- TP/FP/FN/TN合算からのpooled precision/recall/FPR/F1/IoU。
- 動画別F1/IoU中央値、R1-R0のpaired差分中央値、中央値同士の差（別名で区別）。
- TP0動画数、改善/悪化/同値数、recall/FPRの変化、定義可能な動画数・点数。
- ignore上のpositive率（valid GT上のFPとは別集計）。
- 同一動画での欠損、背景FP、時間方向の反復のPLY比較。目視未実施なら未実施と記録。

pooled recallの重みはGT positive数、FPRはGT background数です。未定義を0で埋めず、差分の
符号・単位・mean/medianを検算します。表は可能ならCSVから生成して転記誤りを防いでください。
0.5近辺では既存の2クラスfloat64加算/平均→float32→argmax・同値backgroundを保持し、
`p1 >= 0.5`へ置き換えません。

### 6.3 事前選定基準

R1暫定採用の基本条件は管理記録P3と同じです。

1. validationのpaired F1差分中央値が正。
2. validationのsplit median IoUとpooled F1がR0以上で、TP0が増えない。
3. recall低下やFP増加だけで説明される悪化がなく、両指標のトレードオフを明示できる。
4. train sanityのmedian F1/IoU・TP0が悪化しない。

全条件を満たしても単一seedでの暫定判断です。微差・トレードオフ・train sanityとの不一致が
残る場合はR0を維持する提案とし、管理チャットへ返してください。結果を見て角度・閾値・
採用基準を変更せず、学習を延長して採用条件を満たそうとしないでください。

## 7. P4〜P6の見通し（今回の実行対象外）

- P4: 採用条件の5〜10 epoch最終pilot。健全なP3の5 epoch結果を再利用し、必要時のみ累計10へ。
- P5: pilot受入後に50 epoch中間判定。pilot/epoch25/epoch50の動画評価・PLY、各epoch metrics、
  10 epochごととepoch25・best/lastの保存を行う計画。継続悪化は中断・報告する。
- P6: 50 epoch合格後のみ100へ、100 epochの再判定後に必要なら200へ。各延長は別承認。

現在の`train_stage5.py --checkpoint`はモデル初期化用であり、完全resumeではありません。
P4以降の延長にはoptimizer・epoch・RNG・データ順序・augmentation epochを復元する経路の
実装検証、または初期値から通算期間を再実行する別計画が必要です。P1〜P3に不要なresume改修は
先行実装せず、モデル重みだけの再読込を連続学習と呼ばないでください。

threshold sweep（0.1〜0.9、0.1刻み、0.5を保持）はモデル固定後のみ。aggregationはmean基準で、
変更検討時のみ4方式を副次比較し、maxは診断用のままです。これらを今回の学習比較へ混ぜません。
productionのnormalization/aggregation/threshold採否も別判断です。
既存validation18動画は調整に使用済みであり独立testではありません。最終性能確認には未使用動画群が
必要で、確保不能なら汎化評価の限界を明記します。

## 8. 実装配置・実行環境・保全

既存Dataset/helper/CLIの責務を優先し、manual editはapply_patchで行ってください。
外部PointNeXt本体の改造、無関係なリファクタ、production既定値変更は対象外です。
テストは`Stage5/checks/dummy/`と必要なら`Stage5/checks/real_h5/`、対応bashも同じ配置にします。
比較用bashは既存の実験用配置を確認し、入出力パス・条件を内部定数で指定できる形にします。
既存`train_stage5.sh`の一般利用向け設定を実験値で上書きしないでください。

GPU bashは既存checkerの環境設定を踏襲してください。

```bash
PYTHON="/home/kodaira/anaconda3/envs/dualtrack311/bin/python"
# CONDA_PREFIXは上記PYTHONに対応する環境であることを確認して設定する。
CONDA_PREFIX="/home/kodaira/anaconda3/envs/dualtrack311"
export CONDA_PREFIX
export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="12.0"
TORCH_LIB="$("${PYTHON}" - <<'PY'
import torch
from pathlib import Path
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
```

元H5・初期重み・旧run・旧評価を上書きしません。出力先の既存成果物を検出したら、削除・混在させず
停止して確認してください。中断後の再実行でmetricsが混在することを避けます。
dirty worktreeの他作業を取り消さず、`.tmp/`、生データ、checkpoint、PLYをコミットしません。

## 9. 完了条件と管理チャットへの報告

P1監査、P2実装・CPU検証と承認済みGPU preflight、P3の承認済み比較・評価が揃った時点で、
P1〜P3完了として停止します。学習未実行なら実装完了と分け、次の報告書を段階的に更新してください。

```text
docs/stage5/s5-15/stage5_s5_15_report_to_policy_chat.md
```

報告には次を含めます。

1. 実装ファイル、code revision、テスト結果、実機検証/ローカル確認の区別。
2. P1の差分監査、R0再利用可否の根拠、実行条件・hash付きmanifest。
3. 角度決定・epoch/worker伝達・評価時無効化の検証。
4. 実行承認の対象/回数、実際のtraining epochs・GPU preflight/評価の実施量、失敗再実行履歴。
5. 同epoch5のsplit別比較、pooled/動画別指標、TP0、PLY所見と未実施項目。
6. R0維持またはR1暫定採用の提案、基準ごとの充足/非充足、留保とP4への判断依頼。

進捗・採否は管理記録6章S5-15、数値は評価レポートの新しいS5-15節、配置は`../FILES.md`へ記録します。
共有metricsは既存匿名化規約を使い、実video ID・絶対path・対応表・生データを含めないprivacy
self-checkを行います。未承認の採用や未実行テストを完了と書かないでください。

## 10. 最初に返してほしい内容

まず関連文書とコードを確認し、W-Aの実config/初期重み/file listへアクセスできるか、追加する
引数とaugmentation挿入位置、epoch/worker伝達方法、CPUテスト計画、限定GPU preflight案、
R0再利用可否の監査方法を提示してください。不明点は推測で埋めず、この範囲で相談してください。
その後、段階承認に従ってP1〜P3を進めてください。
