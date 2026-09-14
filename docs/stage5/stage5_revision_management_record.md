# Stage 5 全体改修・管理記録

## 0. 文書情報

| 項目 | 内容 |
| --- | --- |
| 文書種別 | Stage 5の進捗・意思決定・改修履歴を管理する正本 |
| 作成日 | 2026-09-10 |
| 最終更新日 | 2026-09-14 |
| 対象 | pseudo-3D point cloudからのpoint-wise大腿骨segmentation |
| 現在の段階 | S5-12完了。teacher v7・GroupNorm・`bbox_noncontour_ignore`を実験条件としてS5-13 class weight比較へ進む |
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
| BatchNorm/normalization | S5-11でGroupNormが同epoch数比較で有望（TP0動画数10/18→0/18等）。実験用normalizationとして暫定採用済み、production既定値はS5-15まで`batchnorm` | 暫定採用。FP増加（+5.2倍）は未解決、S5-12/S5-13で原因を調査中 |
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

S5-12以降はcrop品質修正済みteacher v7（`bboxrank_v7_cvat_authoritative_crop_quality_v1`、
180動画、train 162 / val 18）を入力とする（D-018）。上記の基準runはS5-07〜S5-11（teacher v6）の
記録であり、v6/v7混同を避けるため両者を区別して扱う。

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
| S5-11 | 2026-09-12 | GroupNorm normalization比較 | checker accepted。方針管理チャットが実験用normalizationとして暫定採用 |
| S5-12 | 2026-09-13〜2026-09-14 | GroupNorm固定Label policy ablation | completed。Run A（ignore）をS5-13の暫定policyとして採用 |

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

### S5-11 GroupNorm normalization比較

| メタ情報 | 内容 |
| --- | --- |
| 方針決定日 | 2026-09-12 |
| 状態 | checker accepted。GroupNormが判定基準の分岐1（有望）に該当。production変更は未実施、採否は方針管理チャット判断待ち |
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

**Step D7（5 epoch pilot）結果（2026-09-12、ユーザー実機）:** BatchNorm controlと同一条件・
同一epoch数（`best.pt`はいずれもepoch 4）で比較した。

| split | model | recall | F1 | IoU | FP | TP0動画数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train_sanity | BatchNorm control | 4.17% | 0.0304 | 0.0154 | 16,602 | 1/3 |
| train_sanity | GroupNorm | 42.33% | 0.1463 | 0.0789 | 42,589 | **0/3** |
| validation | BatchNorm control | 3.87% | 0.0341 | 0.0173 | 85,785 | 10/18 |
| validation | GroupNorm | 34.11% | 0.0878 | 0.0459 | 446,335 | **0/18** |

validation median F1は0→0.0818、validation lossはBatchNorm controlの単調増加
（0.5985→0.9208）と異なり0.539〜0.509の範囲で安定した。FPは+5.2倍に増加したが、precisionは
3.04%→5.04%へ改善しており、recall増加率（+8.8倍）がFPR増加率（+5.2倍）を上回る。S5-09/S5-10で
確認された「recallだけ増えFPが同等以上に急増する」calibration shiftパターンとは異なる。

12章の判定基準の「GroupNormが構造testを満たし、train sanity/validation双方で有望」（分岐1）に
該当する事実が確認された。ただしFP/FPR増加の許容可否、200 epoch時点の挙動、Label
policy/class weightとの相互作用は未検証であり、production採否は方針管理チャットの判断に委ねる。
詳細は`stage5_pointnext_s_training_evaluation_report.md` 9.5節の実装事項D。

候補の優先順位は、recalibrationのcheckpoint運用、fine-tuning時のBN freeze、常時per-window
statisticsを使うnormalization、GroupNorm/LayerNorm等への置換のうち、GroupNormを第一候補、
LayerNormを次候補、per-window BatchNorm推論とBN freeze/recalibrated checkpoint運用を低優先度とする
（S3DISとStage 5のdomain差、physical batch 1、可変点数を考慮し、train-mode推論をそのまま採用しない）。
方式変更を行う場合は5 epoch pilotからやり直し、production inferenceと同一normalization条件で
validationする。

### S5-12 GroupNorm固定Label policy ablation

| メタ情報 | 内容 |
| --- | --- |
| 方針決定日 | 2026-09-13 |
| 状態 | 完了。teacher v7でStep E2〜E7合格。Run AをS5-13の暫定policyとして採用 |
| production変更 | 禁止。Run Bが有望と判断されるまで既定`bbox_noncontour_ignore`を変更しない |

**問題:** teacher v6では、BBox内かつpositive contour外の点が`point_label=-1`、`valid_mask=False`
のためlossに含まれない。この領域をpositiveと予測しても直接のpenaltyがなく、GroupNorm採用後に
増加したFPの一因である可能性がある。

**方針管理チャットの判断（2026-09-13）:** S5-11の判断待ち3事項に対する正式回答として、GroupNormを
S5-12以降の実験用normalizationとして暫定採用する（production既定値はS5-15まで`batchnorm`のまま）。
FP増加（validation FP 85,785→446,335、FPR 1.32%→6.86%）はproduction上許容済みとは判断せず、
S5-12/S5-13で原因と抑制可能性を調べる。threshold tuningはS5-12/S5-13後、S5-15まで行わない。
200 epochへ直接進まず、S5-12→S5-13→必要な追加診断→5〜10 epoch最終pilot→50 epoch中間判定を経て
100〜200 epochの可否を決める。詳細は`.tmp/stage5_s5_12_label_policy_implementation_handoff_prompt.md`。

**目的:** GroupNorm、初期parameter、split、seed、window、loss、class weight、threshold、aggregationを
固定し、Run A（`bbox_noncontour_ignore`、現状維持）とRun B（`bbox_noncontour_background`、対象点を
backgroundとして学習）を比較する。class weightは両runとも`[0.05963856, 1.94036150]`に明示固定し
（`auto`再計算はしない）、label policyとclass weightの2要因が同時に変わることを避ける。

**実装内容:** `Stage2to4/checks/stage4/check_stage4_bbox_ranked_label_policy.py`と同じ幾何
（pixel_xyの丸め込み、frame単位BBox union判定）をStage 5側の`stage5/utils/label_policy.py`
（新規）へ移植し、`compute_bbox_inside_mask()`/`apply_bbox_noncontour_label_policy()`
（`bbox_noncontour_ignore`は完全no-op、`bbox_noncontour_background`は監査済み対象点のみ変換）/
`require_no_stray_ignore_outside_bbox()`（BBox外のignore点があればfail-fast）を実装した。
`stage5/utils/h5_io.py`へ`frame_annotation`のBBox配列読み込みを追加し、
`Pseudo3DPointCloudDataset`へ`label_policy`引数を追加して`_load()`内で一度だけ変換を適用する
（source配列は別keyで保持）。`train_stage5.py`へ`--label_policy`（既定`bbox_noncontour_ignore`、
後方互換）を追加し、`config.json`へsource/effective点数・変換点数の診断情報を記録する。

```text
Stage5/stage5/utils/label_policy.py                                          (新規)
Stage5/stage5/utils/h5_io.py                                                 (変更)
Stage5/stage5/datasets/pseudo3d_pointcloud_dataset.py                        (変更)
Stage5/train_stage5.py / train_stage5.sh                                     (変更)
Stage5/checks/dummy/check_dummy_label_policy.py/.sh                         (新規、Step E2)
Stage5/checks/real_h5/check_stage5_label_policy_bbox_preflight.py/.sh       (新規、Step E3)
Stage5/checks/real_h5/check_stage5_label_policy_dataset_parity.py/.sh       (新規、Step E4)
Stage5/checks/real_h5/check_stage5_label_policy_ablation_eval.py/.sh        (新規、Step E7/E8)
```

Step E3（preflight）・E4（Dataset parity）はCPU/h5pyのみで動作し、GPUを必要としない。Step E7の
共通target評価は既存`evaluate_stage5.predict_h5`を1回forwardし、`canonical_v6`
（既存`evaluate_stage5.py`とのmean baseline parity gate付き）、`bbox_noncontour_as_background`
（主比較target）、`bbox_noncontour_region`（対象領域診断）の3 targetへ同じ予測結果を適用する設計
とした（追加のforwardコストなし）。`--export_ply_alias`でStep E8のGT/prediction診断PLYも出力できる。
Step E5（1 epoch smoke）・E6（Run A/B 5 epoch）は既存`train_stage5.sh`へ
`LABEL_POLICY=bbox_noncontour_background`を指定するだけで実行でき、専用checkerを追加していない。
`last.pt`は5 epoch run終了時点でepoch 5と一致するため、`save_every`変更なしにStep E6の
「epoch 5固定checkpoint」要件を満たす。

`py_compile`・`bash -n`・`git diff --check`はdevコンテナで確認済み。

**Step E3 preflight結果（2026-09-13、ユーザー実機）:** teacher v6の全181 H5（train 163 + val 18）を
監査し、`checker`は設計通りfail-fastした。全体ではignore点262,347点中、stray（BBox外）ignoreは
16点（0.006%）のみで、schema属性も想定通り
（`label_mode=bbox_ranked_global_local`、`contour_teacher_schema=bboxrank_v6_cvat_authoritative_xml_invalidation_v1`）
だった。stray点は2 H5（`train_068`: ignore 97点中stray 10点、`val_009`: ignore 843点中stray 6点）に
集中していた。

対応方針として3案（(1) 該当2ファイルを除外、(2) 該当strayをignoreのまま残しRun Bの変換対象から
明示的に除く、(3) 実データを精査し原因を特定してから決める）を提示し、方針管理チャット側は
(3) を選択した。該当2動画のアノテーション可視化フレームを目視確認した結果、丸め誤差ではなく
Stage 2のcrop窓固定と被写体移動によるずれ（`local_crop_tracking_drift`、既存除外事例
`20250626_090758_8000`と同型）に起因することが判明したため、Stage 4管理チャットへ調査・修正を
依頼した（`.tmp/stage5_s5_12_stage4_crop_quality_investigation_request.md`）。

**Stage 4側修正結果（2026-09-14、Stage 4管理チャットより受領）:**
`.tmp/stage5_s5_12_stage4_crop_quality_correction_report.md`に詳細。要点は次の通り。

- `train_068`（実video: `20250626_090652_6340`）は動画全体をcrop品質不良として除外した
  （既存除外`20250626_090758_8000`と合わせて計2動画除外）。
- `val_009`（実video: `20250625_161030_0550`）はframe order 51のみを無効化し、全pointを
  backgroundへ変換した（CVAT確認済みframe 47-50は維持）。
- 退化BBox（幅または高さ0）を1 pixel領域として扱っていた境界条件バグを修正した
  （`right > left and bottom > top`へ統一）。Stage 5側`label_policy.py`の
  `compute_bbox_inside_mask()`にも同じ修正を反映した。
- 補正済みteacher v7（`bboxrank_v7_cvat_authoritative_crop_quality_v1`、180 H5）を新規構築し、
  全180 H5でstray ignore・no-BBox ignore・BBox内background・CVAT mask外positiveが
  いずれも0であることを確認した。
- `Stage5/train_stage5.sh`/`infer_stage5.sh`/`evaluate_stage5.sh`の既定入力をteacher v7へ
  切り替え済み（`PREFLIGHT_ONLY=1`実行はexit code 0で完了）。

Stage 4側からStage 5への引継ぎ事項（要対応）:

1. Stage 5側のlabel-policy BBox preflightをv7の180 H5へ再実行し、stray ignore=0を独立に確認する。
2. v7 inventory（180ファイル）でtrain/validation listを新規生成する。**旧v6の181ファイル・163/18
   splitは再利用しない**（v6/v7でファイル構成が異なるため）。
3. S5-12 Run A/Bの比較条件（GroupNorm等）はStage 5側の本文書を正本として再確認する。
4. v7 prefixの新規runであることを確認し、v6 runを上書きしない。

S5-07〜S5-11はteacher v6ベースの既存結果として保持し、v6/v7を指標比較時に明記する。今回の修正
だけを理由にS5-07〜S5-11を一律再学習するかはまだ決めていない（S5-12以降の目的と比較可能性に
基づき判断する）。Stage 5側のfail-fastは緩めない方針を維持する（入力側で契約違反が解消された
ため、stray pointを黙って変換する回避策は不要）。

**teacher v7でのStep E2〜E4再実行結果（2026-09-14、ユーザー実機）:** v7 inventory（180 H5）から
`train_stage5.py`の既存split関数（val_fraction 0.1、seed 42）でtrain 162 / val 18のfile listを
新規生成し、Step E2（synthetic test）・E3（v7 180 H5監査）・E4（Dataset parity、180ファイル全件）
をすべて合格した。E3ではstray ignore points=0をStage 5側から独立に確認し、Stage 4の修正が
Stage 5の契約を完全に満たすことを検証した（ignore点総数262,244=BBox内target点数262,244で一致）。
詳細は`stage5_pointnext_s_training_evaluation_report.md` 9.6節の実装事項E。

**Step E5結果（2026-09-14、ユーザー実機）:** Run B（`bbox_noncontour_background`）1 epoch smoke
が完走した。train 162 files/715 samples、val 18 files/87 samplesはsplitと一致し、
変換点数train 238,377/val 23,867点、class weight固定値`[0.05963856, 1.9403615]`を確認した。
train/val loss・F1・IoUは有限でcheckpoint保存まで完走した。

**Step E6結果（2026-09-14、ユーザー実機）:** Run A（`EX260914_..._ep5`）・Run B
（`EX260915_..._ep5`）とも5 epoch完走し、`best.pt`/`last.pt`を保存した。window単位の
train/val loss・F1・IoUはいずれも有限だった。

| run | epoch | val loss | val F1 | val IoU |
| --- | ---: | ---: | ---: | ---: |
| Run A（ignore） | 5 | 0.4735 | 0.0727 | **0.0377**（5epoch中最大） |
| Run B（background） | 5 | 0.4963 | 0.0680 | **0.0352**（5epoch中最大） |

両runともval IoUがepoch 5で最大のため、`best_metric=iou_femur`により`best.pt`と`last.pt`は
同一epochを指す見込みである（Step E7で確認）。この値はRun A/Bそれぞれのnative（実効）labelに
対する window-level集計であり、label policyでvalid_maskが異なるため直接比較はできない
（Step E7の共通targetで比較する）。

**Step E7結果（2026-09-14、ユーザー実機）:** 両runの`last.pt`（epoch 5）を、既存
`evaluate_stage5.sh`によるcanonical参照値と`check_stage5_label_policy_ablation_eval.py`の
3 target評価で比較した（`canonical_v6`のmean baseline parityは両runとも合格）。実行中に本
checker（`.sh`）の`RUN_DIR`/`EVALUATION_DIR`既定値が旧v6 S5-11 pilot runへハードコードされた
ままだったバグを発見・修正した（`CHECKPOINT`/`REFERENCE_H5_METRICS_CSV`の親ディレクトリから
自動導出するよう変更）。

| split | run | recall | F1 | IoU | FPR | video median F1 | video median IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train_sanity | A（ignore） | 58.24% | 0.1156 | 0.0613 | 13.42% | 0.0568 | 0.0292 |
| train_sanity | B（background） | 65.65% | 0.1335 | 0.0715 | 12.93% | 0.0555 | 0.0632 |
| validation | A（ignore） | 44.69% | 0.0714 | 0.0370 | 11.71% | 0.0530 | 0.0272 |
| validation | B（background） | 43.15% | 0.0666 | 0.0345 | **12.17%** | 0.0623 | 0.0322 |

BBox non-contour region（対象点）のpositive予測率: validation 40.63%→**31.13%**
（相対-23%、動画別21件中12件改善・6件悪化・3件同値）、train_sanity 44.34%→40.80%。

train_sanityはRun Bが全指標で優位。validation aggregatedはRun Aがわずかに優位で、FPRは
Run Bで悪化した（11.71%→12.17%）。canonical F1のvideo単位勝敗は18動画中A12勝・B5勝・
1引き分けだが、median F1/IoUはRun Bが上回る（分布形状差、矛盾ではない）。TP0動画数は
両runとも1/18で差なし。

**仮説判断:** 判定基準の「region positive率**と**全体FP/FPRを低下」のうち前者のみ支持され
（核心メカニズムは機能）、後者（全体FP/FPR低下）は確認されなかった。分岐1（有望）・
分岐2（悪化）のいずれにも明確に該当せず、分岐3（差が小さい）〜分岐4（判定不能）に近い。
5 epoch・teacher v7移行後初回比較という条件も踏まえ、production採否は実装チャット側では
決定しない。詳細は`stage5_pointnext_s_training_evaluation_report.md` 9.6節の実装事項E、
方針管理チャットへの報告は`.tmp/stage5_s5_12_report_to_policy_chat.md`。

**完了判断（2026-09-14）:** Run BはBBox non-contour領域のpositive予測率をvalidationで
40.63%から31.13%へ抑制したが、対象領域はvalid backgroundの約0.3%に限られ、全体FPRは
11.71%から12.17%へ改善しなかった。validation aggregated F1/IoUとvideo別勝敗はRun Aが優位、
video medianはRun Bが優位、TP0は同数であり、Run Bを標準化する一貫した根拠は得られなかった。
S5-13では後方互換なRun A（`bbox_noncontour_ignore`）を暫定採用し、Run Bは診断用alternateとして
保持する。S5-12の長期化は行わずclass weight比較へ進む。teacher v7移行だけを理由とするS5-07〜S5-11
の一律再実行も行わず、必要な仮説だけ個別に再検証する。checker出力名`canonical_v6`はhistoricalな
target IDであり、本比較の実データはteacher v7 native labelである。

## 6. 次の改修フロー

### S5-13 Class weight比較

状態: next。S5-12で採用したteacher v7・GroupNorm・`bbox_noncontour_ignore`を固定して実施する。

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
| D-015 | 2026-09-13 | GroupNormをS5-12以降の実験用normalizationとして暫定採用する（production既定値はS5-15まで`batchnorm`のまま） | train/eval logits完全一致、validation TP0が10/18→0/18、median F1が0→0.0818、precision/recall同時改善、validation lossの単調悪化解消 |
| D-016 | 2026-09-13 | FP増加（+5.2倍）を理由にGroupNormを棄却しない。threshold tuningはS5-12/S5-13後、S5-15まで行わない | recall増加（+8.8倍）がFPR増加（+5.2倍）を上回り、precisionも同時改善しているため単純な確率一様シフトではない |
| D-017 | 2026-09-13 | 200 epochへ直接進まず、S5-12→S5-13→必要な追加診断→5〜10 epoch最終pilot→50 epoch中間判定の順で段階確認する | 構造要因（normalization、label policy、class weight）を1つずつ切り分けてから長期学習に進むため |
| D-018 | 2026-09-14 | S5-12以降の入力をteacher v6（181動画）からteacher v7（180動画、crop品質修正済み）へ切り替える。旧v6の163/18 splitは再利用せず、v7 inventoryで新規生成する | S5-12 Step E3が検出したstray ignore点（16点/2動画）がStage 2のcrop窓固定に起因する既知パターン（`local_crop_tracking_drift`）と判明し、Stage 4がteacher v7で修正・全180 H5での契約充足を確認済みのため |
| D-019 | 2026-09-14 | S5-07〜S5-11（teacher v6ベース）は既存結果として保持し、この修正だけを理由に一律再学習しない | v6/v7はファイル構成が異なる別inventoryであり、再学習要否はS5-12以降の目的と比較可能性に基づき個別に判断するため |
| D-020 | 2026-09-14 | S5-13の暫定label policyにRun A（`bbox_noncontour_ignore`）を採用し、Run Bは診断用alternateとして保持する | Run Bは対象領域を抑制したがglobal FPRを改善せず、validation aggregated F1/IoUと動画別勝敗はRun Aが優位だったため |
| D-021 | 2026-09-14 | S5-12のA/Bを長期化せず、teacher v7・GroupNorm・Run A固定でS5-13 class weight比較へ進む | BBox non-contour領域はvalid backgroundの約0.3%であり、global FPにはclass weightの方が直接的に作用すると考えられるため |
| D-022 | 2026-09-14 | teacher v7移行だけを理由にS5-07〜S5-11を一律再実行せず、必要な診断のみ個別に再実行する | 構造診断を履歴として保持しつつ、teacher v7 S5-12 Run Aを今後の比較baselineにできるため |

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
