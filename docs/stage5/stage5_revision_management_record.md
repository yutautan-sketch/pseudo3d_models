# Stage 5 全体改修・管理記録

## 0. 文書情報

| 項目 | 内容 |
| --- | --- |
| 文書種別 | Stage 5の進捗・意思決定・改修履歴を管理する正本 |
| 作成日 | 2026-09-10 |
| 最終更新日 | 2026-09-12 |
| 対象 | pseudo-3D point cloudからのpoint-wise大腿骨segmentation |
| 現在の段階 | S5-10完了。S5-11 GroupNorm比較はStep D6（1 epoch smoke）まで合格、5 epoch pilot（Step D7）判定待ち |
| production readiness | 未到達。診断中 |

本書は、Stage 5の目的、現在状態、過去の改修、検証結果、次の実施順を一か所から追えるように
するためのliving documentである。各checkerの全数値や実装詳細を複製する文書ではなく、
各段階の問題、目的、実装、結果、判断と、根拠文書への参照を管理する。

文書間の優先関係は次の通りとする。

1. 本書: 現在の進捗、採用済み判断、次の実施順
2. `stage5_pointnext_s_training_evaluation_report.md`: 検証結果と数値的根拠
3. 個別のimplementation policy / handoff文書: checkerや実験の詳細仕様
4. `FILES.md`: コードと生成物の配置
5. `TRAINING_IMPROVEMENT_PLAN.md`: 初期計画。現在状態と矛盾する場合は履歴資料として扱う

## 1. Stage 5の概要

### 1.1 役割

Stage 5は、Stage 4で作成した動画由来のpseudo-3D point cloudを入力し、各点が大腿骨に属する
確率とclass labelを推定する段階である。

```text
Stage 4 annotated pseudo-3D H5
  -> frame_order overlap window Dataset
  -> PointNeXt-S point-wise segmentation
  -> overlap prediction aggregation
  -> prediction/prob_femur, prediction/pred_label
  -> Stage 6 axis fitting / endpoint extraction / FL measurement
```

Stage 5は大腿骨長の計測、axis fitting、endpoint抽出を行わない。これらはStage 6の責務である。

### 1.2 入出力契約

主要入力は、Stage 4が生成したannotation付きH5である。

```python
sample = {
    "points": Tensor[Nw, 3],
    "features": Tensor[Nw, C],
    "labels": Tensor[Nw],
    "valid_mask": Tensor[Nw],
    "frame_order": Tensor[Nw],
    "point_indices": Tensor[Nw],
    "window_start": int,
    "window_end": int,
}
```

- `points`: pseudo-3D XYZ
- `features`: 現行設定では`intensity,confidence`
- `labels`: background `0`、femur `1`、ignore `-1`
- `valid_mask`: lossと通常metricsへ含める点
- `frame_order`: window生成の基準
- `point_indices`: overlap予測を元H5内の点へ戻すためのindex

推論時はannotationの有無に依存せず全windowを処理する。annotation付きH5は定量評価に、
annotation前H5は実運用推論に使用できる。

### 1.3 現在の主要実装

- Dataset: `Stage5/stage5/datasets/pseudo3d_pointcloud_dataset.py`
- frame window: `Stage5/stage5/utils/frame_windows.py`
- collate: `Stage5/stage5/datasets/collate.py`
- model wrapper: `Stage5/stage5/models/pointnext_s_segmentor.py`
- lightweight baseline: `Stage5/stage5/models/mlp_baseline_segmentor.py`
- loss / metrics: `Stage5/stage5/training/`
- training CLI: `Stage5/train_stage5.py`
- inference CLI: `Stage5/infer_stage5.py`
- evaluation pipeline: `Stage5/evaluate_stage5.py`、`Stage5/evaluate_stage5.sh`
- privacy export: `Stage5/export_anonymized_stage5_metrics.py`

## 2. 管理原則

1. 一度に変更する主要因は一つとし、変更前baselineを保存する。
2. checkerの完走と、仮説の支持・棄却を別々に記録する。
3. train/validation split、初期checkpoint、seed、window条件を固定して比較する。
4. production設定は診断結果だけで直ちに変更しない。
5. 200 epoch学習は、データ経路、normalization、評価方法が安定してから行う。
6. 当面はCrossEntropyLossを維持し、Dice、Focal、Hard Negative Miningを追加しない。
7. annotation依存のrandom point samplingや、必要な点を恣意的に削除する処理を追加しない。
8. 実データ共有は匿名化済み集約metricsに限り、H5、PLY、座標、checkpoint、元動画IDを共有しない。
9. 完了済み記録を上書きせず、後続判断で変更された場合は新しい日付のdecision recordを追記する。

本書では状態を次の語で統一する。

| 状態 | 意味 |
| --- | --- |
| accepted | 実装と検証が完了し、後続作業の前提として採用済み |
| diagnostic | 診断には使用できるがproduction採用ではない |
| planned | 方針決定済みで未実装 |
| deferred | 前提条件が整うまで保留 |
| rejected | 根拠不足または悪化により採用しない |

## 3. 現在状態の要約

基準日: 2026-09-10。

| 項目 | 現在状態 | 状態 |
| --- | --- | --- |
| teacher | Stage 4 teacher v6、181動画 | accepted |
| model | official OpenPoints PointNeXt-SのStage 5 wrapper | accepted |
| initialization | S3DIS PointNeXt-Sからshape一致111/114 keyを転移 | accepted |
| window | `frame_order`基準、size 16、stride 8、tailあり | accepted |
| physical batch | 1 window | accepted |
| gradient accumulation | 8 windows、point-weighted normalization | accepted |
| padding | PointNeXt入力へ入れない | accepted |
| loss | auto class weight付きCrossEntropyLoss、smoothing 0.0 | diagnostic baseline |
| optimizer | AdamW、lr `1e-3`、weight decay `1e-4` | diagnostic baseline |
| production aggregation | mean probability | 維持。ただし性能上の問題あり |
| alternate aggregation | max / center-nearest / center-weighted | diagnostic only |
| BatchNorm | train/evalで大差を確認。recalibration診断では変化小さく、stalenessは主要因から後退。S5-11でGroupNorm比較を実装中 | 未解決 |
| long training | 200 epochへ進まない | deferred |

現在の基準runは次である。

```text
/mnt/data/3d_projects/stage5_runs/260908/
  pointnext_s_EX260908_260711_w16_s8_bboxrankv6_cvatxmlinv_glocal_
  ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad/
```

- train: 163 files / 729 windows
- validation: 18 files / 79 windows
- 主診断checkpoint: `best.pt`、epoch 4
- current production evaluation: `model.eval()`、physical batch size 1、mean aggregation

## 4. 改修タイムライン

| ID | 実施期間 | 段階 | 状態 |
| --- | --- | --- | --- |
| S5-00 | 2026-06-30 | Stage 5基盤とMLP baseline | accepted |
| S5-01 | 2026-07-07〜2026-07-10 | frame_order overlap window | accepted |
| S5-02 | 2026-07-11 | official PointNeXt-S導入・S3DIS転移 | accepted |
| S5-03 | 2026-07〜2026-08-18 | loss、metrics、training/inference/evaluation整備 | accepted |
| S5-04 | 2026-08-29〜2026-08-30 | 精度問題の監査とbatch integrity | accepted |
| S5-05 | 2026-08-30 | padding parity検証 | accepted |
| S5-06 | 2026-08-30〜2026-09-03 | padding-free学習経路 | accepted |
| S5-07 | 2026-09-08 | Stage 4 teacher v6移行と5 epoch pilot | diagnostic |
| S5-08 | 2026-09-09 | overlap aggregation検証、実装事項A | accepted diagnostic |
| S5-09 | 2026-09-10 | BatchNorm mode parity、実装事項B | accepted diagnostic |
| S5-10 | 2026-09-10〜2026-09-12 | BatchNorm recalibration診断 | checker accepted。production変更は未実施 |
| S5-11 | 2026-09-12〜 | GroupNorm normalization比較 | 実装中。Step D6合格（1 epoch有望）、Step D7が判定点 |

## 5. 段階別の改修記録

### S5-00 Stage 5基盤とMLP baseline

| メタ情報 | 内容 |
| --- | --- |
| 実施日 | 2026-06-30 |
| Git節目 | `5a196a0` Initial clean commit |
| 状態 | accepted |

**問題:** Datasetから学習、checkpoint、推論、H5/PLY出力までを通すStage 5固有の共通基盤が
必要だった。また、初期モデルはPointNeXt風MLPであり、official PointNeXt-Sと呼ぶには構造が
異なっていた。

**目的:** モデル固有実装とデータ・学習・推論処理を分離し、まず軽量モデルでend-to-endの
契約を検証する。

**実装:** H5 loader、Dataset、point sampling、feature normalization、model factory、
`BasePointSegmentor`、拡張可能なloss class、metrics、training/inference CLI、checkpoint、
prediction H5/PLY出力、dummy training/inferenceを追加した。暫定モデルは
`mlp_baseline_segmentor.py`へ整理し、`pointnext_s`という名前をofficial wrapper用に確保した。

**結果:** dummy dataでtraining、checkpoint再読込、inference、H5/PLY出力が完走した。
MLP baselineはpipeline regression用として保持し、本格精度評価には使わない方針を確定した。

**参照:** `stage5_edit_prompt.md`、`FILES.md`

### S5-01 frame_order overlap window

| メタ情報 | 内容 |
| --- | --- |
| 実施期間 | 2026-07-07〜2026-07-10 |
| Git節目 | `c8db26e`、`648558b` |
| 状態 | accepted |

**問題:** 動画全体由来のpseudo-3D point cloudは点数が大きく、動画全体を一度にモデルへ
入力できない。3D座標分割では時間的連続性も失われる。

**目的:** 学習と推論で共通利用できる、`frame_order`基準のoverlap windowを作る。

**実装:** `generate_frame_order_windows`、元H5 indexを保持する`point_indices`、tail window、
可変長collate、window summary、DataLoader batchのPLY exportを追加した。初期既定値はsize 12、
stride 6、軽量確認値は8/4とした。overlap modeではrandom point samplingを無効化した。
training CLIは2026-07-10にwindow sampleへ接続した。

**結果:** 1 H5から複数windowが生成され、frame範囲、overlap、元点index、DataLoader batch、
MLP forwardを確認した。後の実データ学習ではGPU memoryと時間範囲を考慮して16/8を採用した。

**参照:** `stage5_pointnext_s_training_evaluation_report.md` 6章・7.1節、`FILES.md`

### S5-02 official PointNeXt-S導入とS3DIS転移

| メタ情報 | 内容 |
| --- | --- |
| 実施日 | 2026-07-11 |
| Git節目 | `65daf6d` |
| 状態 | accepted |

**問題:** MLP baselineにはset abstraction、ball query、FPS、hierarchical downsampling、
feature propagationがなく、PointNeXt-Sの精度検証には使えなかった。OpenPointsのCUDA extensionも
PyTorch CUDA 12.8とhost CUDA 13.2の不一致を解消する必要があった。

**目的:** official repositoryのPointNeXt-SをStage 5共通interfaceへ接続し、既知データと既知重みで
モデル実装の正常性を確認してからStage 5へ転移する。

**実装:** `Stage5/external/PointNeXt/`のOpenPoints実装、`pointnet2_batch_cuda`、
`pointnext_s_segmentor.py`を導入した。wrapperはStage 5の`[B,N,3]` pointsと`[B,N,C]` featuresを
OpenPoints形式へ変換し、logitsを`[B,N,num_classes]`へ戻す。bashではConda CUDAとTorch libraryを
明示してextensionを読み込むようにした。

**検証と結果:** S3DIS PointNeXt-S checkpointはofficial `BaseSeg`へ114 keyをstrict loadでき、
1部屋推論のPLYはGTと概ね対応した。Stage 5 wrapperにはshape一致111 keyを転移し、missing 2 key、
shape mismatch 1 keyを新規初期化側に残した。dummy window、実H5 window、loss、training CLIまで
forwardが完走した。

**参照:** `FILES.md`のS3DIS/transfer checker一覧、`stage5_edit_prompt.md`

### S5-03 loss、metrics、training/inference/evaluation整備

| メタ情報 | 内容 |
| --- | --- |
| 実施期間 | 2026-07〜2026-08-18 |
| Git節目 | `8c35b13`、`481d8c8` |
| 状態 | accepted。精度設定はdiagnostic baseline |

**問題:** class imbalance、ignore点、checkpoint比較、train sanityとvalidationの固定評価、
推論結果の可視化を一貫して扱う必要があった。初期metricsにはF1計算不具合もあった。

**目的:** PointNeXt公式実装から大きく外れないweighted CrossEntropyLossを基準に、後からlossを
差し替えられる構造と、再現可能な評価・可視化経路を整える。

**実装:** CrossEntropyLoss class、manual/auto class weight、label smoothing、AdamW、FP/FNを含む
metrics、F1修正、詳細debug metrics、定期checkpoint保存を追加した。label smoothingは0.0を本命値と
した。2026-08-01にはno-BBox frameをbackgroundとして学習するlabel policyを既定化した。
2026-08-18には固定train sanity 3動画とvalidation全件を評価する`evaluate_stage5.sh`、H5/window/
aggregated metrics、診断PLY、checkpoint比較、匿名化exportを整備した。

**結果:** 複数の短期・長期runを比較できる状態になった。一方、teacher v2系の150/200 epoch runでは
train改善に対してvalidation lossが悪化し、FP、動画間ばらつき、過学習が残ったため、単なるepoch追加や
複雑なloss追加ではなく、データ経路とbatch処理の監査へ移行した。

**参照:** `TRAINING_IMPROVEMENT_PLAN.md`、`data_construct.md`、
`stage5_pointnext_s_training_evaluation_report.md` 2〜5章

### S5-04 精度問題の監査とbatch integrity

| メタ情報 | 内容 |
| --- | --- |
| 実施期間 | 2026-08-29〜2026-08-30 |
| 状態 | accepted |

**問題:** train sanity PLYで似たXY位置のpositiveが複数frame群へ反復し、validationでは検出不足と
背景FPが併存した。GTまたは点群がbatch間で誤って流用された可能性を先に除外する必要があった。

**目的:** H5、window、`point_indices`、labels、collate、shuffle、multi-workerの対応を全件監査する。

**実装:** `check_stage5_batch_integrity.py/.sh`を追加し、sample hash、元H5再取得、ordered scan、
shuffle + multi-worker、異なる点数を含むmanual batchを検査した。

**結果:** train/validation全件で合格した。GTや点群の別sampleへの流用は強く除外され、PLY上の反復は
異なるframeの点、overlap、座標事前分布、model挙動などを調べるべき現象と判断した。

**参照:** `stage5_pointnext_s_training_evaluation_report.md` 3章・7.1節・9.1節

### S5-05 padding parity

| メタ情報 | 内容 |
| --- | --- |
| 実施日 | 2026-08-30 |
| 状態 | accepted |

**問題:** 可変長windowをbatch内最大点数へzero paddingしていたが、PointNeXt-Sへ実点maskを渡して
いなかった。FPS、ball query、BatchNormがpaddingを実点として扱う可能性があった。

**目的:** paddingだけを追加したときのlogits、class、loss、gradient、BatchNorm統計への影響を分離する。

**実装:** `check_stage5_padding_parity.py/.sh`でrepeat control、同一shape batch-size sensitivity、
manual zero padding、実在peer padding、train-mode gradient/BN比較を実施した。

**結果:** positive targetで最大確率差0.658〜0.848、class反転0.61〜3.76%、gradient cosine
0.09/0.34を確認した。zero paddingは無視できない実装上の問題と判定し、paddingありbatchを現在の
PointNeXt-S学習から廃止する方針を採用した。

**参照:** `stage5_pointnext_s_training_evaluation_report.md` 7.2節・9.2節

### S5-06 padding-free学習経路

| メタ情報 | 内容 |
| --- | --- |
| 実施期間 | 2026-08-30〜2026-09-03 |
| 状態 | accepted |

**問題:** physical batch size 1ではpaddingを除去できるが、単純なgradient accumulationでは点数と
class weight分母が異なるwindowを正しく結合できない。

**目的:** PointNeXtへpadding点を渡さず、8 window相当のoptimizer更新をpoint-weighted CEとして
再現する。

**実装:** physical batch size 1、gradient accumulation 8、loss numeratorとclass-weighted有効分母の
蓄積、step直前のgradient正規化、端数step処理を追加した。microbatch、optimizer step、padding、
valid/ignore、class count、loss normalizerをmetricsへ記録した。

**結果:** teacher v2の1 epoch smoke testで733 train window、92 optimizer step、train/validation
padding 0、checkpoint保存を確認した。empty-valid windowも0だったため、初期計画にあった
`skip_empty_valid_windows`は現行teacherの優先課題ではない。

**参照:** `stage5_pointnext_s_training_evaluation_report.md` 9.4節

### S5-07 Stage 4 teacher v6移行と5 epoch pilot

| メタ情報 | 内容 |
| --- | --- |
| 実施日 | 2026-09-08 |
| 状態 | diagnostic baseline accepted |

**問題:** 旧teacher v2には、CVAT full-video修正と削除済み誤BBox XMLの無効化が反映されていなかった。
最終教師データを確定せずに長期学習を続けると比較が無効になる。

**目的:** Stage 4で受入済みのteacher v6をStage 5正本入力とし、padding-free経路を再検証する。

**実装:** 次の181 H5 datasetへtraining/inference/evaluation bashとprovenance preflightを更新した。

```text
global_local_l75_w31_c12_area15_bboxrank_v6_
cvat_authoritative_xml_invalidation_v1/collected/
```

teacher v6は59動画・3014 Task frameでCVAT snapshotをpositive authorityとして使用し、2動画・7 frameの
削除XMLを全background・全validへ変更し、crop不良1動画を除外している。

**結果:** 1 epoch smoke testと5 epoch pilotが完走した。全epochでpadding 0、empty-valid 0、
train 729 microbatch、92 optimizer step、loss/metrics再計算一致を確認した。bestはepoch 4で、
validation window F1/IoUは0.0455/0.0233だった。mean overlap集約後は0.0341/0.0173へ低下し、
train/validationの乖離、低recall、背景FPが残った。200 epochへは進まず診断を優先した。

**参照:** Stage 4の3文書と`stage5_pointnext_s_training_evaluation_report.md` 9.4節

### S5-08 overlap aggregation検証、実装事項A

| メタ情報 | 内容 |
| --- | --- |
| 実施日 | 2026-09-09 |
| 対象 | teacher v6 5 epoch pilot `best.pt`、train sanity 3 + validation 18動画 |
| 状態 | checker accepted。alternate aggregationはdiagnostic only |

**問題:** window単位よりmean aggregation後のrecallが低く、同一点に対する別windowの予測がpositiveを
打ち消している可能性があった。

**目的:** 同一forwardからwindow間確率分布、不一致、suppressed-positive、4 aggregation方式を比較する。

**実装:** `check_stage5_overlap_aggregation.py/.sh`を追加し、mean、max、center-nearest、
center-weightedを比較した。既存評価CSVとのmean count完全一致、匿名化、方式別PLY optionを検証した。

**結果:** GT positiveのsuppressed-positiveはtrain sanity 21.5%、validation 10.6%だった。
window間disagreementは動画ごとに0.2〜79.7%と大きく異なった。境界距離別rateは5.9〜6.0%で横ばい、
動画内相対位置では前半約2%から後半約7%へ増える傾向だった。maxはvalidation recall 14.4%まで
上げる一方FPを85,785から328,572へ増やした。center系にも明確な優位性はなかった。

**判断:** meanによるpositive抑制は支持されたが、production aggregationはmeanを維持する。
境界距離を根拠とするcenter weightingは再設計しない。最終モデル確定後に方式比較をやり直す。

**参照:** `stage5_overlap_aggregation_implementation_policy.md`、
`stage5_overlap_aggregation_handoff_prompt.md`、`stage5_pointnext_s_training_evaluation_report.md` 9.5節

### S5-09 BatchNorm mode parity、実装事項B

| メタ情報 | 内容 |
| --- | --- |
| 実施日 | 2026-09-10 |
| 対象 | S5-08と同じ21動画、`best.pt` |
| 状態 | checker accepted。production変更は未実施 |

**問題:** physical batch size 1学習では各windowのbatch statisticsを使用する一方、productionの
`eval()`はrunning statisticsを使用する。5 epoch pilotのtrain/eval乖離にBatchNormが寄与する可能性が
あった。

**目的:** 同一window、同一点順序、同一RNG seedで`eval()`と`train()`の確率差を測る。

**実装:** `check_stage5_batchnorm_mode_parity.py/.sh`で独立model instance、dropout 0 assertion、
BN buffer監査、window/mean集約metrics、既存mean baseline parity、匿名化を実装した。

**結果:** 全valid点のclass disagreementはtrain sanity 15.4%、validation 16.2%、GT positiveでは
51.2%/38.2%だった。aggregated recallはtrain sanity 4.17%から61.01%、validation 3.87%から
33.12%へ上がったが、validation FPも85,785から757,945へ増えた。17 BN moduleのrunning mean/varは
finiteで、variance負値はなく、`num_batches_tracked=16859`で一致した。

**判断:** normalization modeは重大な交絡要因である。ただしtrain-modeは評価window自身の統計を
使うtest-time adaptationと確率校正の変化も含むため、「running statisticsだけが壊れている」または
「汎化問題ではない」とはまだ断定しない。train-mode推論をproduction採用せず、recalibrationで
保存統計の不適合とper-window normalization効果を切り分ける。

**参照:** `stage5_overlap_aggregation_handoff_prompt.md` 6章、
`stage5_pointnext_s_training_evaluation_report.md` 9.5節の実装事項B

### S5-10 BatchNorm recalibration診断

| メタ情報 | 内容 |
| --- | --- |
| 実施期間 | 2026-09-10〜2026-09-12 |
| 対象 | S5-08/S5-09と同じ21動画、`best.pt` |
| 状態 | checker accepted。production変更は未実施 |

**問題:** S5-09はnormalization mode依存を示したが、保存済みrunning statisticsのstalenessと、
各test window固有のbatch statisticsを使う効果を分離していない。

**目的:** 最終重みを固定したままtraining splitだけでrunning statisticsを再計算し、
recalibrated modelを`eval()`で評価して、保存統計の再計算だけで改善する範囲を測る。

**実装内容:** `check_stage5_batchnorm_recalibration.py/.sh`を追加した。`evaluate_stage5.
model_from_checkpoint`、事項Aのmean baseline parity gate、事項Bの`run_window_forward`/
`summarize_diff_group`、`check_stage5_batch_integrity`のhash比較を再利用し（事項Bのcheckerファイル
自体は変更していない）、modelはBatchNorm sub-moduleだけ`.train()`へ切り替え、それ以外は`.eval()`を
維持する方式（事項Bの「model全体を`.train()`にしてdropout=0でassert」より厳密）とした。
calibration前後のparameter hash・non-BN buffer hashの不変性、全BN moduleの`num_batches_tracked`が
calibration処理window数と一致すること、`running_mean`/`running_var`がfiniteかつ非負varianceで
あることをfail-fast assertionとして実装し、対応するsynthetic testを追加した
（`Path.resolve()`後のcalibration/validation重複検査を含め計8ケース）。

**検証方法:** self-test（8ケース）、train sanity 3件のみのsmoke run（calibrationは163動画全件を
実施したうえで評価を限定）、full run（train sanity 3件+validation 18件、計21動画）をユーザー実機で
実行し、original eval・recalibrated evalを同じwindow・seedで比較した。

**結果:** self-test・smoke run・full runはいずれも完走し、匿名化self-checkも合格した。
calibrationは163 files/729 windowsを処理し、17 BN module全てで`recalibrated_num_batches_tracked=729`
（calibration processed windowsと完全一致）、`running_mean`/`running_var`はfiniteかつ負のvarianceも
0件だった。parameter hash・非BN buffer hashは不変で、original evalのmean baseline parityも
既存`h5_metrics.csv`と完全一致した。

running statistics自体の絶対的な変化量は大きい（`running_var_abs_diff_max`最大約469.9）一方、
original-vs-recalibrated probability disagreement率はGT positive点でtrain_sanity 13.27%、
validation 4.84%と、事項Bのeval/train disagreement率（同51.16%/38.15%）よりはるかに小さかった。
aggregated F1/recallの変化はvalidationで微増（F1 0.0341→0.0387、recall 3.87%→4.06%、
FP 85,785→73,374）、train_sanityではむしろ悪化した（F1 0.0304→0.0255、TP0動画数1/3→2/3）。
validationのTP0動画数は10/18のまま変わらず、video単位では改善（`validation_016`: TP 0→31）と
悪化（`validation_004`: TP 59→0）が両方向に生じた。事項Bのtrain-modeで観測された
recall/FPの大幅同時増加（validation aggregated recall 3.87%→33.12%、FP+672,160点）は
recalibrationでは再現されなかった。

**仮説判断:** BN moduleのrunning statistics自体は仕様通り大きく再計算されたが、production寄りの
指標（aggregated/window F1・recall・TP0動画数）への効果は小さく、split間・video間で符号が
一致しない。このため、保存済みrunning statisticsのstalenessは、S5-09で観測した規模のeval/train
乖離の主要因からは後退し、各test windowが自身のbatch statisticsを使うtest-time adaptation効果
（またはphysical batch size 1のnormalization方式そのもの）が相対的な優先候補として残る。
ただし本checkerはこの2要因を直接分離する設計ではなく、断定はしない。

**productionへの影響:** なし。recalibrated checkpointはユーザー実機の`stage5_debug/`下へ
diagnostic artifactとして保存されており、`train_stage5.py`/`infer_stage5.py`/`evaluate_stage5.py`の
既定挙動、production aggregation（mean probability）は変更していない。

**次のアクション:** S5-11（BatchNorm対策の選定）の優先候補見直しを方針管理チャットへ諮る。
Label policy ablation（9.6節）・class weight比較（9.7節）との順序も含め、方針管理チャットの
判断を待つ。

**関連文書・出力先:** `stage5_pointnext_s_training_evaluation_report.md` 9.5節の実装事項C、
`.tmp/batchnorm_recalibration_share/`（匿名化済み共有metrics）、ユーザー実機
`/mnt/data/3d_projects/stage5_debug/batchnorm_recalibration/`（share/private出力とdiagnostic
checkpoint）。

## 6. 次の改修フロー

### S5-11 GroupNorm normalization比較

| メタ情報 | 内容 |
| --- | --- |
| 方針決定日 | 2026-09-12 |
| 状態 | 実装中（Step D6の1 epoch smokeまで合格。5 epoch pilot（Step D7）で最終判定） |
| production変更 | 禁止。5 epoch比較で有望と判断されるまで既定値`batchnorm`を変更しない |

**問題:** S5-10により、保存済みrunning statisticsのstalenessはS5-09で観測した規模のeval/train乖離の
主要因からは後退した。physical batch size 1でも train/evalで同じ挙動となる正規化方式が
次の優先候補となる。

**方針管理チャットの判断（2026-09-12）:** running statistics recalibrationを主要候補から外し、
diagnostic artifactとしてのみ保存する（recalibrated checkpointはproduction採用しない）。S5-10の
追加seed・追加動画によるsensitivity検証は行わない。Label policy ablationより先にS5-11を短く
実施する。第一候補はGroupNorm、不採用時はLayerNormを次候補とする。詳細は
`.tmp/stage5_s5_11_groupnorm_implementation_handoff_prompt.md`。

**目的:** 現行PointNeXt-SのBatchNormだけをGroupNormへ置き換え、physical batch size 1、可変点数
window、train/eval modeの条件に依存しない正規化へ変更した場合の学習安定性と固定評価性能を測る。
教師label、class weight、loss、threshold、aggregation、split、seed、window、入力featureは固定する。

**実装進捗（2026-09-12）:** wrapper/CLI/checkpoint config経路とGroupNorm adapterのコード実装、および
GPU非依存の静的・synthetic検証まで完了した。詳細は
`stage5_pointnext_s_training_evaluation_report.md` 9.5節の実装事項Dを参照。

```text
Stage5/stage5/models/norm_layers.py                                        (新規)
Stage5/stage5/models/pointnext_decoder_patch.py                            (新規)
Stage5/stage5/models/pointnext_s_segmentor.py                              (変更)
Stage5/train_stage5.py / infer_stage5.py / evaluate_stage5.py              (変更)
Stage5/train_stage5.sh                                                     (変更、POINTNEXT_NORM knob追加)
Stage5/checks/dummy/check_dummy_pointnext_s_training.py/.sh                (変更)
Stage5/checks/dummy/check_dummy_pointnext_s_groupnorm.py/.sh               (新規)
Stage5/checks/transfer/check_stage5_batchnorm_to_groupnorm_transfer.py/.sh (新規)
```

`py_compile`・`bash -n`・`git diff --check`はdevコンテナで確認済み。CPU only synthetic self-testは
ユーザー実機で合格した。

ユーザー実機のStep D2/D3 full run初回実行で、`pointnext_norm=groupnorm`でもdecoder内8 module（4 stage
×2 convs）がBatchNormのまま残る不具合を検出した。原因はofficial OpenPoints `PointNextDecoder`が
`norm_args`/`act_args`を`**kwargs`で受け取りながら実際には使用せず、`FeaturePropogation`自身の
hardcoded既定値`{'norm': 'bn1d'}`が常に使われていたため。external PointNeXt cloneは変更せず、
`Stage5/stage5/models/pointnext_decoder_patch.py`（新規）で`PointNextDecoder`を継承した
`Stage5PointNextDecoder`を実装し、`_make_dec()`だけをoverrideして`norm_args`を正しく伝播させた
（`MODELS`へ新しい名前で登録、冪等）。batchnorm既定経路は`create_norm`のdimension接尾辞処理により
元のhardcoded挙動と同一の`nn.BatchNorm1d`を構築するため、既存checkpointの互換性に影響しない。
再発防止のsynthetic testを追加済み。詳細は
`stage5_pointnext_s_training_evaluation_report.md` 9.5節の実装事項D。

修正後、self-test・Step D2/D3 full runともユーザー実機で合格した（2026-09-12）。
`Stage5 PointNeXt-S GroupNorm structure check passed.`、batchnorm module数17に対しgroupnorm
module数も17で1:1置換を確認し、train()/eval() logitsのmax abs diffは0.000e+00（完全一致）だった。
詳細は`stage5_pointnext_s_training_evaluation_report.md` 9.5節の実装事項D。

Step D4（dummy training smoke）・Step D5（BatchNorm→GroupNorm初期化転移）もユーザー実機で合格した
（2026-09-12）。Step D4はbatchnorm/groupnorm双方でtrain/val lossが有限、checkpoint保存・strict
reloadまで完走。Step D5は既存BatchNorm S3DIS部分転移checkpoint（114 key）からGroupNormモデル
（63 key）へ転移し、loaded 63/63・除外keyは17 BN module×3 buffer=51個のみ（予期しない
missing/unexpected/shape不一致は0件）で完全一致した。詳細は
`stage5_pointnext_s_training_evaluation_report.md` 9.5節の実装事項D。

Step D6（実H5 1 epoch smoke）もユーザー実機で合格した（2026-09-12、
`EX260912_..._ep1_bs1_acc8_nopad`）。microbatch 729・optimizer step 92・padding 0はBatchNorm
controlと一致。1 epoch checkpointへ既存評価pipelineを先行実施した参考値では、validation
aggregated F1 0.0508・recall 17.81%・TP0動画数2/18・median F1 0.0313となり、BatchNorm control
（5 epoch pilot、F1 0.0341・recall 3.87%・TP0 10/18・median F1 0）を1 epochの時点で上回った。
一方FPも85,785→405,149（+4.7倍）へ増加し、recall増加（+4.6倍）とほぼ同じ倍率であるため、
検出改善か確率校正シフトかはこの時点では未確定。学習量（1 epoch vs 5 epoch）も揃っていない。
判定はStep D7（5 epoch pilot、BatchNorm controlと同epoch数）を待つ。詳細は
`stage5_pointnext_s_training_evaluation_report.md` 9.5節の実装事項D。

Step D7（5 epoch pilot）の評価は未実行。

候補の優先順位は、recalibrationのcheckpoint運用、fine-tuning時のBN freeze、常時per-window
statisticsを使うnormalization、GroupNorm/LayerNorm等への置換のうち、GroupNormを第一候補、
LayerNormを次候補、per-window BatchNorm推論とBN freeze/recalibrated checkpoint運用を低優先度とする
（S3DISとStage 5のdomain差、physical batch 1、可変点数を考慮し、train-mode推論をそのまま採用しない）。
方式変更を行う場合は5 epoch pilotからやり直し、production inferenceと同一normalization条件で
validationする。

### S5-12 Label policy ablation

状態: deferred。normalization方針確定後に実施する。

- Run A: BBox内かつcontour外をignore
- Run B: BBox内かつcontour外をbackground
- no-BBox、CVAT-authoritative frame、XML invalidationのteacher v6契約は維持する
- 同一split、seed、初期重み、window、normalization、class weightで5 epoch比較する
- 両runを同一の固定評価targetで比較し、ignore領域上のpositive予測も診断値として残す

label policyによって評価対象自体が変わるため、異なるvalid maskで得たF1を直接比較するだけでは
不十分である。共通評価labelと、BBox内非contour領域の予測率を分けて記録する。

### S5-13 Class weight比較

状態: deferred。S5-12後に実施する。

auto weightと、より弱い固定weightを同じ教師・normalization条件で比較する。positive recallだけでなく、
FP/FPR、predicted positive率、TP 0動画数、video-level medianを重視する。threshold tuningや複雑なlossは
この比較へ混ぜない。

### S5-14 座標依存・frame-level・overlap loss診断

状態: deferred。

似たXY位置のpositive反復について、frame単位TP/FP/FN、GT/predicted positive重心、動画内相対位置を
記録する。必要になった場合に限り、座標augmentationまたは座標feature ablationを一因ずつ比較する。
center-only lossやoverlap出現回数の逆数weightは、最終normalizationとaggregation評価が固まるまで
実装しない。

### S5-15 長期学習とproduction候補確定

状態: deferred。

S5-10〜S5-14のうち主要な構造問題が解消し、5〜10 epoch pilotでtrain sanityとvalidationの両方に
改善根拠が得られた場合だけ、50 epochの中間判定を経て100〜200 epochへ進む。最終checkpointで
aggregation 4方式を再評価し、production aggregation、normalization、thresholdを確定する。

## 7. Decision record

| ID | 日付 | 決定 | 根拠 |
| --- | --- | --- | --- |
| D-001 | 2026-06-30 | MLP modelを`mlp_baseline`として保持 | official PointNeXt-S構造ではない |
| D-002 | 2026-07-07 | 3D位置ではなく`frame_order`でwindow分割 | 動画時間文脈と元点対応を維持するため |
| D-003 | 2026-07-11 | official OpenPoints PointNeXt-S wrapperを本命化 | S3DIS strict loadと実データforwardに成功 |
| D-004 | 2026-08 | CE、label smoothing 0.0を基準にする | 複雑なloss前にデータ経路を検証するため |
| D-005 | 2026-08-30 | PointNeXt入力からzero paddingを除く | logits、gradient、BNへ大きな影響を確認 |
| D-006 | 2026-09-08 | teacher v6をStage 5入力正本とする | CVAT authorityとXML invalidation受入完了 |
| D-007 | 2026-09-09 | production aggregationはmeanを維持 | alternate方式にFP/TP0を含む明確な優位性なし |
| D-008 | 2026-09-09 | center weightingを再設計しない | 境界距離とdisagreementの相関なし |
| D-009 | 2026-09-10 | train-mode inferenceをproduction採用しない | recallと同時にFPが大幅増加 |
| D-010 | 2026-09-10 | Label/class weight前にBN recalibrationを診断 | normalization modeが重大な交絡要因 |
| D-011 | 2026-09-12 | recalibrated checkpointをproduction採用しない | validation改善が小さく、train sanity・動画別結果で符号が揃わない |
| D-012 | 2026-09-12 | S5-10の追加seed・追加動画検証は行わない | validation 18動画でもTP0は改善せず、追加検証で判断が変わる可能性は低い |
| D-013 | 2026-09-12 | Label policyより先にS5-11（normalization比較）を実施する | normalizationを固定しないままlabel policyを比較すると、採用後のnormalizationへ結果を持ち越せない可能性がある |
| D-014 | 2026-09-12 | S5-11の第一候補をGroupNormとする | physical batch size 1に依存せずtrain/evalで同じ挙動となり、PointNeXtの1D/2D tensorにも適用できる |

## 8. 文書更新ルール

新しい改修または検証を行うたびに、本書へ次の形式で追記する。

```text
### S5-XX タイトル

実施期間:
担当段階:
対象dataset/run/checkpoint:
状態:

問題:
目的:
実装内容:
検証方法:
結果:
仮説判断:
productionへの影響:
次のアクション:
関連文書・出力先:
```

更新時には次も行う。

1. 4章のtimelineと3章の現在状態を更新する。
2. 方針決定があれば7章へdecision recordを追加する。
3. 詳細metricsは評価レポートへ記録し、本書には判断に必要な要約だけを書く。
4. 実装ファイルを追加した場合は`FILES.md`を更新する。
5. 従来判断を変更する場合は削除せず、新しいdecision IDから旧IDを参照する。

## 9. 関連文書

### Stage 5

- `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
  - 精度問題、batch/padding監査、teacher v6 pilot、事項A/Bの数値的正本
- `docs/stage5/stage5_overlap_aggregation_handoff_prompt.md`
  - 事項A/Bの実装・実行履歴、環境、privacy契約
- `docs/stage5/stage5_overlap_aggregation_implementation_policy.md`
  - overlap accumulator、parity、出力設計の実装根拠
- `docs/stage5/stage5_edit_prompt.md`
  - MLP baselineの名称整理とofficial PointNeXt-S移行方針
- `docs/stage5/TRAINING_IMPROVEMENT_PLAN.md`
  - 初期改善計画。現在の優先順には本書を使用する
- `docs/stage5/data_construct.md`
  - evaluation outputと匿名化bundleの構造
- `docs/stage5/FILES.md`
  - Stage 5コード、checker、work directoryの索引

### Stage 4 teacher v6

- `docs/stage2to4/stage4/stage4_cvat_snapshot_authoritative_label_revision_plan.md`
- `docs/stage2to4/stage4/stage4_deleted_xml_annotation_invalidation_plan.md`
- `docs/stage2to4/stage4/stage4_phase5_fullvideo_cvat_review_implementation.md`

Stage 4文書では最新のlabel authorityを優先し、履歴上の旧target-only方針をteacher v6の現行仕様と
混同しない。
