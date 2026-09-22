# Stage 5 全体改修・管理記録

## 0. 文書情報

| 項目 | 内容 |
| --- | --- |
| 文書種別 | Stage 5の進捗・意思決定・改修履歴を管理する正本 |
| 作成日 | 2026-09-10 |
| 最終更新日 | 2026-09-19 |
| 対象 | pseudo-3D point cloudからのpoint-wise大腿骨segmentation |
| 現在の段階 | S5-15 P1〜P3完了（2026-09-19）。回転augmentation単独5 epoch比較の結果、**R0（augmentation=none）を維持しR1を採用しない**と判断。R1は比較履歴として保持。次はR0条件での新規50 epoch学習（承認済み、開始前にCPU突合が必要）。production設定は未変更 |
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

最新方針（2026-09-15）: S5-14全体（core・補足1〜3）の完了をユーザー判断により受け入れた。
S5-15は6章の段階別計画に従い、回転augmentation単独5 epoch比較から進める。
本更新は方針策定であり、実装・学習の完了や長期学習の一括承認ではない。
以下は補足1の結果要約で、補足2・3と最終判断は6章を参照する。
teacher v7全180 H5（train 162 / val 18）、W-A epoch 5、GroupNorm 8 groups、`bbox_noncontour_ignore`
を固定し、保存済み予測で再集計した。分母補正・自己参照排除後も仮説A（座標事前分布）の偏りは
残存し、仮説B（時間位置）の低下は動画内paired比較では再現されず、仮説C（overlap exposure）の
効果は時間区分別でも小さいままだった。続く補足2の座標変換診断も完了したが、座標の直接的な
記憶は確定しておらず、局所形状・輝度との交絡を留保する。

以下は2026-09-10時点の基準runを含む履歴要約であり、補足の入力は6章の固定条件を優先する。

| 項目 | 現在状態 | 状態 |
| --- | --- | --- |
| teacher | Stage 4 teacher v6、181動画 | accepted |
| model | official OpenPoints PointNeXt-SのStage 5 wrapper | accepted |
| initialization | S3DIS PointNeXt-Sからshape一致111/114 keyを転移 | accepted |
| window | `frame_order`基準、size 16、stride 8、tailあり | accepted |
| physical batch | 1 window | accepted |
| gradient accumulation | 8 windows、point-weighted normalization | accepted |
| padding | PointNeXt入力へ入れない | accepted |
| loss | auto class weight付きCrossEntropyLoss、smoothing 0.0。S5-13で弱い固定weight`[0.5,1.5]`（比3.0）はpositive予測が完全崩壊（recall 0%、TP0動画数18/18）し不採用、強いauto由来weight（比32.5）を維持 | diagnostic baseline |
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
| S5-13 | 2026-09-14 | GroupNorm固定Class weight ablation | completed。W-B（弱い固定weight）は不採用、W-A（強いauto由来weight）を暫定採用 |
| S5-13補足 | 2026-09-14 | Threshold-free診断と中間class weight限定確認 | completed。W-C（比16）不採用、W-A維持。新規5 epoch pilot 1/2回で終了 |
| S5-14 | 2026-09-14〜2026-09-15 | 座標依存・frame-level・overlap exposure診断 | completed（core、Step H1〜H5）。成果物・parityを受入。仮説解釈はD-030に従い補足で再評価 |
| S5-14補足 | 2026-09-15 方針決定、実装・実機再集計完了、受入 | 点密度補正と動画別・時間別再集計 | Aの空間的偏りは分母補正後も残存。Bの一般的な時間低下・Cのexposure対策は優先しない。学習・GPU再推論0回 |
| S5-14補足2 | 2026-09-15 実装・実機再実行完了、受入 | 座標変換診断 | W-A固定・学習0回。自己除外・split分離を修正。回転感度を確認、既知のGPU実施量は計336動画条件相当 |
| S5-14補足3 | 2026-09-15 CPU検証完了、受入 | 残存集計・記録確認 | pooled/動画別集計を分離し誤記を訂正。GPU・学習0回。core・補足1〜3を含むS5-14全体を完了 |

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

**参照:** `s5-08-09/stage5_overlap_aggregation_implementation_policy.md`、
`s5-08-09/stage5_overlap_aggregation_handoff_prompt.md`、`stage5_pointnext_s_training_evaluation_report.md` 9.5節

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

**参照:** `s5-08-09/stage5_overlap_aggregation_handoff_prompt.md` 6章、
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
`research/stage5/s5-10/batchnorm_recalibration_share/`（匿名化済み共有metrics）、ユーザー実機
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
`docs/stage5/s5-11/stage5_s5_11_groupnorm_implementation_handoff_prompt.md`。

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
100〜200 epochの可否を決める。詳細は`docs/stage5/s5-12/stage5_s5_12_label_policy_implementation_handoff_prompt.md`。

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
依頼した（`docs/stage2to4/stage4/stage5_s5_12_stage4_crop_quality_investigation_request.md`）。

**Stage 4側修正結果（2026-09-14、Stage 4管理チャットより受領）:**
`docs/stage2to4/stage4/stage5_s5_12_stage4_crop_quality_correction_report.md`に詳細。要点は次の通り。

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
方針管理チャットへの報告は`docs/stage5/s5-12/stage5_s5_12_report_to_policy_chat.md`。

**完了判断（2026-09-14）:** Run BはBBox non-contour領域のpositive予測率をvalidationで
40.63%から31.13%へ抑制したが、対象領域はvalid backgroundの約0.3%に限られ、全体FPRは
11.71%から12.17%へ改善しなかった。validation aggregated F1/IoUとvideo別勝敗はRun Aが優位、
video medianはRun Bが優位、TP0は同数であり、Run Bを標準化する一貫した根拠は得られなかった。
S5-13では後方互換なRun A（`bbox_noncontour_ignore`）を暫定採用し、Run Bは診断用alternateとして
保持する。S5-12の長期化は行わずclass weight比較へ進む。teacher v7移行だけを理由とするS5-07〜S5-11
の一律再実行も行わず、必要な仮説だけ個別に再検証する。checker出力名`canonical_v6`はhistoricalな
target IDであり、本比較の実データはteacher v7 native labelである。

### S5-13 GroupNorm固定Class weight ablation

| メタ情報 | 内容 |
| --- | --- |
| 方針決定日 | 2026-09-14 |
| 状態 | 完了。W-Bは不採用、W-A（強いauto由来weight）をS5-13補足およびS5-14の暫定controlとして採用 |
| production変更 | 禁止。W-Aが従来のauto既定値と同じ値のため、既定`CLASS_WEIGHT=auto`は変更していない |

**問題:** S5-12で採用したteacher v7・GroupNorm・`bbox_noncontour_ignore`のもとでも、validation FPR
（11.71%）は未解決のまま残っている。S5-12で使ったauto由来weight`[0.05963856, 1.94036150]`
（positive:background比約32.5）が、minority positiveの学習とTP0抑制に役立つ一方でFP/FPRを
増やしている可能性がある。

**目的:** teacher v7・GroupNorm・`bbox_noncontour_ignore`・split・seed・初期checkpointを固定し、
class weightだけを変えてglobal FP/FPRとpositive recallのtrade-offを比較する。

| Arm | class weight | positive:background weight比 |
| --- | --- | ---: |
| W-A（control、S5-12 Run Aを再利用） | `[0.05963856, 1.94036150]` | 約32.54 |
| W-B | `[0.5, 1.5]` | 3.0 |

**実装内容（Step F1〜F3）:** `train_stage5.sh`のrun命名不具合（`PREFIX`が実際の`CLASS_WEIGHT`に
関わらず`auto_weight`固定）を修正し、`CLASS_WEIGHT`からfilesystem-safeなtag（`cw_auto`/
`cw_manual_<value>`/`cw_none`）を生成してrun名・起動ログへ反映、`OUTPUT_DIR`/`EXPERIMENT_NAME`の
明示override、非emptyな既存出力先への`ALLOW_EXISTING_OUTPUT_DIR`ガードを追加した
（production既定値`CLASS_WEIGHT=auto`は変更していない）。新規checker
`checks/real_h5/check_stage5_class_weight_ablation.py/.sh`を追加し、2つのrun directory間で
config.json（class weight関連キーとrun/output由来キーを除く全一致）・train_files.txt/val_files.txt・
resolved class weight・history.jsonのepoch数/finite性・初期checkpointのpath/SHA-256・（任意で）
evaluate_stage5.pyのh5_metrics.csvによる評価対象point集合の一致を検証する（CPU/JSON/CSVのみ、
torch/h5py/CUDA不要）。静的検証として`checks/dummy/check_dummy_class_weight_tag.sh`（8ケース）と
`checks/dummy/check_dummy_class_weight_ablation.py/.sh`（11ケース）を追加し、いずれも合格した。

```text
Stage5/train_stage5.sh                                          (変更)
Stage5/checks/real_h5/check_stage5_class_weight_ablation.py/.sh (新規)
Stage5/checks/dummy/check_dummy_class_weight_tag.sh              (新規)
Stage5/checks/dummy/check_dummy_class_weight_ablation.py/.sh    (新規)
```

**Step F4（W-A再利用判定）:** S5-12 Run Aの学習時revision（コミット`4b55e55`）と現revisionの
`git diff`は`train_stage5.sh`の命名/衝突ガード変更のみで、`train_stage5.py`への引数構築や
`train_stage5.py`本体・`stage5/`配下は無変更（学習挙動に影響する差分なし）。S5-12 Run A
（`EX260914`、teacher v7、GroupNorm 8 groups、`bbox_noncontour_ignore`、`class_weight=
[0.05963856, 1.9403615]`、seed 42、window 16/8、batch 1、grad accum 8、lr 1e-3、weight decay
1e-4、dropout 0.0）のconfig.jsonを確認し、6節の固定条件をすべて満たすためW-Aとして再利用した。

**Step F5〜F6結果（2026-09-14、ユーザー実機）:** W-B（`CLASS_WEIGHT=0.5,1.5`、同一初期checkpoint・
split・seed）は1 epoch smoke・5 epoch pilotとも完走した。train 162 files/715 samples、val 18
files/87 samplesはW-Aと一致、label policy変換点数0（no-op）、`class weight: [0.5, 1.5]`を確認、
output_dir名に`cw_manual_0p5_1p5`タグが正しく反映され既存runと衝突しなかった。ただしepoch 2以降、
train/valとも`fp=0`かつ`fn`が一定値に固定される（positive予測の完全崩壊）window単位running metrics
が観測された。val側はepoch 1終了時点で既に崩壊していた。

**Step F4本比較・Step F7結果（2026-09-14、ユーザー実機）:** `check_stage5_class_weight_ablation.sh`
をW-A（`EX260914`）・W-B（`EX260916`）に対して実行し、config parityはすべて合格した。既存
`evaluate_stage5.sh`（train sanity 3動画+validation 18動画、mean probability aggregation、
threshold 0.5、`last.pt`=`best.pt`＝epoch 5）による公式評価でも、W-Bの崩壊が完全に再現された。

| split | run | recall | precision | F1 | IoU | FPR | predicted positive数 | TP0動画数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train_sanity | W-A | 58.24% | 6.41% | 0.1156 | 0.0613 | 13.42% | 88,583 | 0/3 |
| train_sanity | W-B | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.00% | 0 | **3/3** |
| validation | W-A | 44.69% | 3.88% | 0.0714 | 0.0370 | 11.71% | 864,518 | 1/18 |
| validation | W-B | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.00% | 0 | **18/18** |

video-level（validation 18動画）: F1勝敗はW-A 17勝・W-B 0勝・1引き分け（全動画でW-B側
predicted_positive_count=0）。既存`debug_*`診断（再実装なし）から、positiveがweighted CE
denominatorに占める割合はW-A 29.03%に対しW-B 3.63%と算出され、raw点数比（background:positive
約79.5:1）に対してW-Bの重み比3.0では不均衡を全く相殺できず、CE lossを最小化する解が
「常にbackgroundと予測する」になったことを裏付けた。optimizer step数・empty-valid window数は
両armとも正常で、機構上の不具合ではなくweight設定自体が原因と判断できる。詳細な数値は
`stage5_pointnext_s_training_evaluation_report.md` 9.7節の実装事項F。

**仮説判断:** 依頼書9章の判定基準に対し、FP/FPR低下は起きたが（predicted positiveが0のため
trivial）、precision/F1/IoUは改善せず0へ、recallはW-Aの44.69%(validation)から0%へ完全崩壊、
TP0動画数はW-Aの1/18から18/18へ増加、video別勝敗もW-B側の改善を支持しない。「FP低下と引き換えに
recallが崩壊する場合、W-Bは採用しない」という9章の不採用分岐に明確に該当し、判定不能ではなく
**W-Bは不採用、W-A（強いauto由来weight）を維持**という明確な結論が得られた。production反映
（`CLASS_WEIGHT`既定値の変更）は行わず、方針管理チャットの判断を待つ。方針管理チャットへの報告は
`docs/stage5/s5-13/stage5_s5_13_report_to_policy_chat.md`。

### S5-13補足 Threshold-free診断と中間class weight限定確認

状態: 完了。W-C（比16）不採用、W-A（強いauto由来weight）を維持。新規5 epoch pilotは1/2回で終了。

**背景:** S5-13ではW-B（`[0.5,1.5]`、positive:background weight比3）がthreshold 0.5で全点を
backgroundと予測し、validation recall 0%、TP0動画数18/18となったため不採用とした。一方、class
weightの変更はlogitの基準位置とcalibrationも変える。今回の固定threshold評価だけでは、W-Bでpositive/
backgroundの順位識別まで消失したのか、scoreが0.5未満へ移動した影響が中心なのかを分離できない。
また、weight比32.54のW-Aと比3のW-Bは間隔が広く、W-Aが最適であることまでは2点比較から確定しない。

**目的:** 追加計算を抑えながら、(1) 保存済みW-A/W-Bの識別能力とcalibration shiftを分離し、
(2) W-Aより弱いがW-Bほどpositive寄与を失わない中間weightでglobal FP/FPRを抑えられる余地を一度だけ
確認する。結果からS5-14以降で固定する暫定class weightを決める。production thresholdの選定は
S5-15まで行わない。

**補足検証1（再学習なし）:** 保存済みW-A/W-Bの同一validation 18動画・train sanity 3動画に対し、
集約後`prob_femur`からAUPRCを主指標、AUROCを補助指標として計算する。GT class別score分布・percentile、
PR curve、threshold別TP/FP/FN、最大F1、recall at fixed FPRなども記録する。これはdiscriminationと
calibrationを診断するためのsweepであり、production thresholdを変更する操作ではない。従来の
threshold 0.5 metricsもprimary operating-point結果として維持する。

**補足検証2（中間weight、原則1 run）:** teacher v7、GroupNorm 8 groups、
`bbox_noncontour_ignore`、S5-12/S5-13と同一split・seed・初期checkpoint・window・optimizer・lossを
固定し、次のW-Cを5 epochだけ学習・評価する。

```text
W-C class weight = [0.11764706, 1.88235294]
positive:background weight比 = 16
想定positive weighted-denominator占有率 = 約16.7%
```

W-CはW-Aの約29.0%とW-Bの約3.6%の中間のpositive loss寄与を狙う。epoch 5固定checkpointをW-A/W-Bと
同じtrain sanity・validationへ適用し、threshold 0.5のTP/FP/TN/FN、precision、recall、F1、IoU、FPR、
predicted positive率、TP0動画数、video-level mean/medianと勝敗に加えてthreshold-free指標を比較する。

**検証量の上限:** 本段階はS5-13の補足でありclass-weight sweepへ拡大しない。新規5 epoch pilotは
**原則W-Cの1回、最大でも合計2回まで**とする。2回目は、W-Cが明確な不具合なく境界的結果となり、
もう1点で採否を確定できる合理的根拠がある場合、または比較公平性を損なう実行上の問題により再実行が
必要な場合に限る。no-weight、複数seed、広いweight grid、50〜200 epoch学習は行わない。保存済み
W-A/W-Bは再学習せず再利用する。

**判定:** threshold 0.5でW-CがW-AよりFP/FPRを下げ、recall・TP0・video median F1/IoUを大きく
損なわず、改善方向がthreshold-free指標と動画別結果でも支持される場合に限りW-Cを暫定採用候補とする。
W-Cが全negativeへ崩壊する、TP0が増える、または改善が一貫しない場合はW-Aを維持する。W-Bは
threshold-free識別が残っていても現行threshold 0.5のS5-14 controlには採用せず、S5-15のthreshold診断で
参照可能な履歴候補としてのみ保持する。

**固定事項:** production既定値、threshold 0.5、mean aggregation、teacher、label policy、GroupNorm
group数、learning rate、scheduler、window、sampling、feature、augmentation、loss形式は変更しない。
補足完了後に結果と採用判断を評価レポートおよび本管理記録へ追記してからS5-14へ進む。

**補足検証1結果（2026-09-14、ユーザー実機、再学習なし）:** 新規checker
`check_stage5_class_weight_threshold_free.py/.sh`（`evaluate_stage5.py`の既存prediction `.npz`と
H5 GTのみ使用、モデル再推論・CUDA不要）で、保存済みW-A/Bのthreshold-0.5 TP/FP/TN/FNが既存
`h5_metrics.csv`と完全一致することを確認した上で、AUPRC/AUROC/固定FPR別recall/precisionを算出した。

| split | run | AUPRC | AUROC | max F1 |
| --- | --- | ---: | ---: | ---: |
| train_sanity | W-A | 0.0860 | 0.8519 | 0.1849 |
| train_sanity | W-B | 0.0707 | 0.8368 | 0.1566 |
| validation | W-A | 0.0574 | 0.8018 | 0.1162 |
| validation | W-B | 0.0367 | 0.7652 | 0.0885 |

W-AのFPR（11.71%）以下で達成可能な最大recall（validation）はW-A 44.70%に対しW-B 37.59%
（thresholdを0.5から約0.107へ下げた場合）。video-level AUPRC win countはW-A 12・W-B 6（validation
18動画）だが、median AUPRCはW-A 0.0422・W-B 0.0427とほぼ同値（meanはW-A優位）で、S5-12と同様の
mean/median不一致が見られた。

**判定:** 「discrimination消失」（ランダム水準への崩壊）にも「W-Aに近い」にも該当しない中間的結果。
W-Bのthreshold 0.5全negativeは主にcalibration shiftが原因（threshold再設定でrecall 37.59%まで
回復）だが、discrimination自体も中程度（相対15〜35%程度）に劣化しており、純粋なcalibration
shiftだけでは説明しきれない。D-023の判断（W-Bを現行threshold 0.5のS5-14 controlに採用しない）は
維持し、この結果はS5-15のthreshold診断用履歴候補として保持する。詳細は
`stage5_pointnext_s_training_evaluation_report.md` 9.7節、報告は
`docs/stage5/s5-13/stage5_s5_13_supplement_report_to_policy_chat.md`。

**補足検証2結果（W-C、weight比16、新規5 epoch pilot 1/2回、2026-09-14、ユーザー実機）:**
config parityは合格（class weight以外の差分なし）。threshold 0.5固定評価（validation aggregate）は
recall 44.69%(A)→17.52%(C)、FPR 11.71%(A)→**2.58%(C)**、F1 0.0714(A)→0.0969(C)、
IoU 0.0370(A)→0.0509(C)。video-levelではF1/IoU win countがW-A 9・W-C 8・1引き分けとほぼ互角な
一方、FPRは18動画すべてでW-Cが低く、recall medianは44.68%→**9.70%**と大幅悪化、TP0動画数は
W-Aの1/18からW-Cの**5/18へ増加**した。threshold-free診断ではvalidation AUPRC 0.0574(A)→0.0445(C)、
AUROC 0.8018(A)→0.7588(C)といずれもW-Cが下回り、**同一FPRに揃えて比較すると1%/5%/10%/11.71%の
すべての水準でW-AがW-Cのrecallを上回った**。これは、W-Cのthreshold 0.5での見かけ上のF1/precision/
IoU改善が識別能力そのものの向上ではなく、より保守的な暗黙operating pointへ移動した結果であることを
示す。

**判定:** handoff文書9章の「W-Aを維持する条件」（TP0動画数またはrecallの明確な悪化、
threshold-free指標とvideo-level結果がW-Cを支持しない、結果が混合的）に複数該当し、
**W-Cは不採用、W-A（強いauto由来weight）を維持**と判断した。結果が明確なため2回目の新規5 epoch
pilotは実施しなかった。S5-13本比較の結論（W-A維持）はこの補足でも変わらない。production反映は
方針管理チャットの判断を待つ。

**方針管理チャットの完了判断（2026-09-14）:** S5-13補足を完了とし、W-Cは不採用、W-A
（`[0.05963856,1.94036150]`）をS5-14の暫定class weightとして確定する。W-B/W-Cのcheckpoint・config・
prediction・threshold-free metricsはS5-15で参照可能な診断履歴として保持するが、activeなproduction
候補にはしない。W-Cは同一FPRでW-Aよりrecallが低く、AUPRC/AUROCも悪化したため、weight比24などの
追加中間weight探索は行わない。未使用の2回目の5 epoch枠は消化せず、S5-14へ進む。threshold 0.5と
mean aggregationは維持し、production threshold tuningはS5-15まで延期する。

### S5-14 座標依存・frame-level・overlap exposure診断

状態: 完了（core、Step H1〜H5）。3仮説の最終判定: 仮説A（座標事前分布）強く示唆（density正規化
確認は残課題）、仮説B（時間位置）明確に支持、仮説C（overlap exposure）限定的に支持（主要因では
ない）。次の改修（座標equivariance診断、inverse-occurrence loss weighting等）は方針管理チャットの
判断待ち。

**背景:** train sanityのpositive-only PLYを側面から見たとき、同じようなXY位置のpositive集合が
異なるframe群（pseudo-3DのZ方向）へ2〜3回反復して見えた。S5-04のbatch integrity検証により、GTや
point cloudが別sampleへ流用されるDataset/collate/shuffle上の不具合は強く除外されている。一方、
現在のwindow学習ではunique pointのwindow出現倍率がbackground約1.69倍、positive約1.90倍で、
frame位置によりlossへの露出回数が異なる。S5-08ではwindow境界距離とdisagreementの相関はなかったが、
動画内相対位置の後半ほどdisagreement率が高い傾向があった。座標事前分布、時間的な見え方の変化、
overlapによる不均一なtraining exposureをまだ分離できていない。

**目的:** W-A baselineの予測をsource point・frameへ戻し、次の3仮説を再学習なしで切り分ける。

1. **座標事前分布:** GTの位置と無関係に、modelが動画間で似たXY位置をpositiveにしやすい。
2. **時間位置・frame phase:** 誤検出または見逃しが動画内の特定区間、GT-positive区間の前後、
   no-GT frameへ偏る。
3. **overlap exposure:** windowへの出現回数またはwindow間disagreementが、point/frame単位の
   probability・FP・FN・反復構造と対応する。

**固定baseline:** teacher v7、GroupNorm 8 groups、`bbox_noncontour_ignore`、W-A class weight
`[0.05963856,1.94036150]`、window 16/stride 8/tailあり、mean probability aggregation、threshold 0.5、
`model.eval()`を固定する。対象checkpointはS5-12/S5-13で再利用したW-A epoch 5（`best.pt`と
`last.pt`が同一epoch）とする。aggregate評価はvalidation 18動画全件と固定train sanity 3動画を使う。
GT/exposureだけで完結するdataset監査はteacher v7全180 H5へ拡張する。

**座標系の分離:** H5のraw `pixel_xy`、raw pseudo-3D `points`、modelへ渡される`normalize_xyz(points)`、
`frame_order`を別項目として保持する。現行DatasetはH5全体のpointsを一度center/scaleしてからwindowを
選ぶため、windowごとの再centerではない。したがってmodel入力には動画全体中心に対する相対XY/Z位置が
残る。PLY上のraw座標反復とmodel入力座標上の依存を同一視せず、両方を報告する。動画間XY比較はStage 4
schemaに記録されたcrop/image寸法で0〜1へ正規化し、寸法が利用できない場合は推測で補わず、その動画を
cross-video heatmapから除外して件数を報告する。

**Step H1: synthetic/parity test。** 小さな合成videoでframe集計、重心、空のpositive class、
contiguous frame run、XY bin、window exposure count、source `point_indices`への復元を検証する。
W-Aのthreshold 0.5 aggregate TP/FP/TN/FNが既存`evaluate_stage5.py`の結果と完全一致することを
real-data fail-fast gateにする。point数、frame_order、label、valid_mask、vote countの欠落・重複、
nonfinite値があれば停止する。

**Step H2: teacher v7全件のGT/exposure監査（CPU）。** 全180 H5について、frameごとのGT positive/
background/ignore点数、BBox/GT-positive有無、各source pointのwindow出現回数、class別出現倍率、
frame相対位置decile別のexposureを記録する。unique-point集計とwindow-occurrence集計を分け、train/val
splitおよび動画別に出力する。これにより、反復がteacher GT自体またはwindow構成だけで生じ得る程度を
model predictionと独立に確認する。

**実装状況（2026-09-14）:** `checks/real_h5/check_stage5_frame_spatial_overlap_diagnostics.py/.sh`
（Step H2本体、CPU/h5py only）と`checks/dummy/check_dummy_frame_spatial_overlap_diagnostics.py/.sh`
（Step H1のsynthetic test、8ケース＋合成H5でのaudit_h5()統合テスト）を実装し、すべて合格を確認した
（`py_compile`/`bash -n`/`git diff --check`含む）。実装した共有primitiveは`find_runs`（contiguous
frame run分割）、`per_frame_label_counts`、`window_occurrence_counts`（`generate_frame_order_windows`
を再利用した純geometry、model forward不要）、`relative_frame_decile`、`class_exposure_summary`
（unique point数とwindow occurrence総数を明示的に分離）、重複/欠落point・nonfinite値のfail-fast。
Step H2の実行（teacher v7全180 H5）はGPU側（実際にはCPU/h5pyのみで動作するため、GPU機の
Python環境でよい）での実行待ち。

**既知の課題（Step H3.1のcross-video XY正規化について、2026-09-14調査）:** 座標系分離の方針
（本節冒頭）が前提とする「Stage 4 schemaに記録されたcrop/image寸法」を調査したところ、
**teacher v7を含むStage 5入力H5（`annotate_pseudo3d_point_cloud.py`が生成する最終combined H5）には
crop/image寸法（幅・高さ）を示す属性が一切保存されていない**ことが判明した。crop offset・resize
scale等は生成pipeline中間段階の個別動画h5（`pseudo3d_processing.py`が書き込む`local_crop_top`/
`local_crop_left`/`local_resize_scale`等のroot属性）にのみ存在し、Stage 5が使う最終H5の`point_cloud`
group属性には引き継がれていない。crop sizeそのものは`batch_infer_video_to_pseudo3d_h5.py`の
`--local_crop_size`（既定256）という生成run全体で共有される外部定数であり、H5から読み取れる値では
ない。したがって「寸法が利用できない動画をcross-video heatmapから除外する」という想定は、
一部動画の欠落ではなく**全180動画がこの対象**になる。既存`stage5/utils/feature_normalization.py`の
`normalize_pixel_xy()`は観測点のmax値で正規化しており、真のcrop寸法によるfixed-frame正規化ではない
ため、handoff文書が禁止する「観測点のmax値を推測的に代用」に該当し使用できない。同一動画内の
raw pixel座標での解析（handoff文書6節が許容する代替）は影響を受けない。この制約の扱い
（cross-video正規化heatmapを断念するか、Stage 2/4側に外部定数として`local_crop_size`の確認を依頼するか）
は方針管理チャットへ確認予定。

**方針管理チャットの判断（2026-09-15）:** 諦めずに中間H5を遡って実寸法を回収できるか調査する
Step H2.5（Stage 4 dataset inventoryとXY座標provenance監査）を実施する。詳細は
`docs/stage5/s5-14/stage5_s5_14_structural_diagnostics_implementation_handoff.md`の2026-09-15追記版。

**Step H2.5実装（2026-09-15、実装・静的検証完了）。**
`checks/real_h5/check_stage5_xy_coordinate_provenance.py/.sh`
（H2.5-A dataset inventory scan、H2.5-B source provenance解決と中間H5属性読取、H2.5-C寸法決定の
優先順位ロジック、H2.5-D正規化とbounds fail-fast、H2.5-E artifact出力）と
`checks/dummy/check_dummy_xy_coordinate_provenance.py/.sh`（15件superset合成テスト、手作り
final+intermediate H5ペアでの統合テスト含む）を実装し、すべて合格を確認した。事前調査
（`Stage2to4/src/utils/pseudo3d_processing.py`のroot属性書込みと`annotate_pseudo3d_point_cloud.py`
の`save_annotated_h5()`によるmeta属性継承を追跡）により、中間H5の`local_input_shape`属性
（文字列化されたtensor shape、`(num_frames, 1, height, width)`）が各動画のlocal crop実寸法を
直接与えること、かつ全前処理分岐でcrop出力が常に正方形（`local_crop_size`×`local_crop_size`、
既定256）になることを確認し、これを寸法解決の一次情報源として実装した（依頼書H2.5-C選択肢3）。
最終H5の`source_pseudo3d_h5`属性（絶対path）を第一の解決手段とし、解決できない場合は
`pseudo3d_outputs/<date>/`配下でのbasename一致を第二の解決手段とする（両者は明示的に区別して
記録し、暗黙に同一視しない）。`decide_dimension_policy()`は、全動画が個別解決できれば選択肢3、
一部未解決でも解決済み動画が単一の共通寸法に一致しかつ未解決動画のpixel_xyがその寸法内に収まる
場合のみ選択肢2として当該動画へ拡張し、それ以外は動画単位で除外する（推測による一律採用はしない）。

**Step H2.5 GPU側実行結果（2026-09-15、ユーザー実機、CPU/h5pyのみで動作）。** teacher v7全180
H5を監査し、**全180動画が`source_pseudo3d_h5`属性だけで中間H5を解決でき（basename fallback
不要）、寸法は全180動画で(256, 256)に完全一致、`pixel_xy` boundsも全180動画で合格**した
（除外0件、選択肢3で確定）。splitはtrain 162・val 18でS5-12/S5-13と一致。`local_crop_size`の
生成CLI既定値256を推測採用したのではなく、全180動画それぞれの中間H5 provenanceから実測値として
確認した結果である。これによりStep H3.1のcross-video XY正規化は`normalized_x = pixel_x / 255`、
`normalized_y = pixel_y / 255`を**動画除外なしで**全180動画・固定21診断対象動画に適用できる。
詳細は`docs/stage5/s5-14/stage5_s5_14_report_to_policy_chat.md`。

**管理チャット二重検査での指摘・修正（2026-09-15）:** `decide_dimension_policy()`が選択肢3判定時に
`pixel_xy_bounds_ok`を条件に含めておらず、将来bounds違反があっても選択肢3として通り得るという指摘を
受けた。dimension解決済みだがbounds違反の動画が1件でもあればfail-fastするよう修正し、policy-level
のsynthetic testを追加、実際の監査結果CSVへ再適用しても結論（選択肢3、除外0件）が変わらないことを
確認した。

**Step H3: aggregate predictionのframe/XY診断（再推論不要を優先）。** 保存済みW-A prediction
`.npz`とteacher v7 H5を使い、固定21動画についてframe単位のTP/FP/TN/FN、precision、recall、F1、IoU、
FPR、predicted positive率、GT/predicted positive点数を出力する。GT positiveとpredicted positiveの
XY重心・分散・重心距離は、両集合が存在するframeだけで計算し、undefinedを0で埋めない。no-GT frame
ではFP数、positive probability、直近GT-positive frameまでの距離を記録する。動画内相対位置decile、
GT-positive区間の前/中/後、train sanity/validationを分けて集計する。

normalized XY grid上にGT positive density、predicted positive density、FP density、FN densityの
heatmap用CSV/NPZを作り、GTとpredictionの位置相関、predicted centroidの動画間集中度、no-GT frameで
同じXY binが繰り返しactiveになる率を計測する。XY binごとのpositive frameを連続runへ分解し、GT runと
重ならないpredicted run数・長さ・frame間隔を記録することで、PLYで見えた「同じXY位置の複数frame群」
を定量化する。grid解像度やrun定義は設定と出力へ保存し、単一の恣意的なgridだけで仮説を確定しない。

**Step H3/H3.1実装（2026-09-15、実装・静的検証完了、実runはGPU側待ち）。**
`checks/real_h5/check_stage5_frame_xy_diagnostics.py/.sh`と
`checks/dummy/check_dummy_frame_xy_diagnostics.py/.sh`を実装し、すべて合格を確認した。
`check_stage5_class_weight_threshold_free.py`の`load_prediction`/`safe_name`/
`load_h5_metrics_rows`/`check_threshold_0p5_parity`/`binary_counts`と、
`check_stage5_frame_spatial_overlap_diagnostics.py`の`find_runs`/`relative_frame_decile`、
`check_stage5_xy_coordinate_provenance.py`の`normalize_pixel_xy_with_dimensions`を再利用し
重複実装を避けた。GT/predicted positiveのXY重心・重心距離はどちらかが空のframeでは`None`とし
0で埋めない（availability flag `centroid_distance_available`を別途出力）。frame単位で
`classify_relative_to_runs()`によりGT-positive runに対するinside/before/after/no_gt_runを分類する。
XY binごとのpositive frame活動は`bin_frame_activity()`（`(bin_row,bin_col,frame)`の重複除去による
ベクトル化）で抽出し、`temporal_recurrence_for_bin()`でGT runとの重なり判定・unmatched run数・
run間gapを計算する。grid解像度は既定で16・8の2種類を計算し、単一gridに依存しない設計とした。
既存W-A predictionとのthreshold 0.5 parity gate（`check_threshold_0p5_parity`）を動画ごとに実行する。
synthetic testは15件（frame混同行列、確率統計の空group処理、XY重心のNone可用性、frame間距離、
run分類のinside/before/after/no_gt_run、grid bin境界値、density集計、bin_frame_activityの空mask
処理、temporal recurrenceのmatched/unmatched/gap計算、availability flag統合テスト、手作りH5+
predictionでのparity gate pass/fail）を実装し、この開発コンテナ内（一時venv）ですべて合格を確認した。

**インシデント（2026-09-15、修正済み）:** GPU側初回実行の出力で`video_alias`に実video ID
（タイムスタンプ形式）がそのまま書き込まれるバグを発見した。`main()`が匿名化前の生
`h5_metrics.csv`の`video_name`をそのまま出力していたことが原因。実video名はpath解決にのみ内部
使用し、出力には`{split}_{index:03d}`の列挙aliasのみを書き込むよう修正、タイムスタンプ形式の
文字列がCLI出力に一切現れないことを検証する回帰テストを追加した。ユーザー側で漏洩済み4ファイル
（`.tmp/`・GPU機`work_dirs/`双方）を削除し、修正版で再実行・再確認済み。

**Step H3/H3.1 GPU側実行・分析結果（2026-09-15、ユーザー実機）:** 21動画・834 frame行・
grid16/8で6720 density行・2917 recurrence行を生成し、全動画でparity gateに合格した。

- **仮説B（時間位置）: 明確に支持。** validation動画のrecallは動画内相対decile 0（序盤）の64.2%
  からdecile 8（終盤）の7.9%へ系統的に急落する一方、FPRはdecile間でほぼ一定（10.5%〜14.2%）。
  時間位置による性能劣化はrecall側に集中し、FP側にはほぼ影響しないことが新たに分かった。
- **仮説A（座標事前分布）: 強く示唆されるが正式な「支持」は未確定。** 個々の動画でGT positiveが
  皆無のbinに限定しても、動画横断でGTが密集する「globally hot」binは「globally cold」binの
  約8.7倍のpredicted positive/FP率を示した。ただしhandoff文書が求める
  「density正規化後も傾向が残る」確認（bin単位point密度での正規化）は未実施であり、点群
  サンプリング密度自体の交絡を排除できていないため留保する。
- **仮説C（overlap exposure）: 未検証。** exposureとframe/bin単位誤り率を直接結びつける集計は
  Step H4（per-window診断）が主要な検証手段であり、Step H3.1だけでは判定できない。
- **PLY反復現象の定量化:** grid16でpositiveがactiveな2,192 binのうち36.1%が
  GT runと重ならない複数の時間分離predicted run（multiple_unmatched_runs）を持ち、32.0%は
  GT runと一度も重ならない。train sanity PLYで見えた反復現象を定量的に裏付けた。

3仮説の最終判定はStep H4完了後に確定する。詳細は`docs/stage5/s5-14/stage5_s5_14_report_to_policy_chat.md`。

**Step H4: per-window context診断（固定21動画のみ、必要な1回の再推論）。** 既存S5-08 overlap
checkerのaccumulatorとsource point alignmentを再利用し、teacher v7・GroupNorm・W-Aでwindowごとの
`prob_femur`を取得する。同じforward結果からmean aggregateを再構成し、既存W-A predictionとの
TP/FP/TN/FN parityを要求する。source point/frameごとにvote count、probability min/max/mean/std、
positive vote ratio、window間class disagreement、各window内の中心/境界距離を記録する。

point/frame errorをvote count、training exposure、動画内相対位置、GT class、no-GT/GT-positive frameで
層別化する。特に、同じXY binのunmatched predicted runが高いexposureまたは高いdisagreementへ集中するか、
positiveがbackgroundより多く重複することがFP/FNのどちらと対応するかを確認する。S5-08で棄却した
「window境界ほど不一致が多い」仮説は、teacher v7/W-A上のparity確認を除き、根拠なく再設計しない。

**Step H4実装（2026-09-15、実装・静的検証完了、実runはGPU側待ち）。**
`checks/real_h5/check_stage5_per_window_context_diagnostics.py/.sh`と
`checks/dummy/check_dummy_per_window_context_diagnostics.py/.sh`を実装した。S5-08
`check_stage5_overlap_aggregation.py`の`OverlapAccumulator`/`run_overlap_forward`/
`classify_overlap`/`gt_class_masks`/`relative_frame_deciles`を再利用し（重複実装なし）、
per-window forward自体が必要とするtorch importは`process_video_h4()`内のlocal importに限定し、
それ以外のロジック（parity判定、層別集計、XY bin exposure/disagreement集計、Step H3.1
recurrence CSVとのjoin）はtorch非依存で実装した。

**parity gateはno-toleranceで実装**: (1) このrunのvote_countと保存済みW-A predictionの
vote_countが完全一致すること、(2) Step H2と同じwindow構成で計算したtraining window occurrence
countとこのrunのvote_countが完全一致すること（training exposureとinference vote countの内部
整合性チェック）、(3) mean probability（threshold 0.5）から再導出したpredicted classと保存済み
predictionのTP/FP/TN/FN・per-point predicted classが完全一致すること、の3点を1点でも満たさなければ
即座にfail-fastする（CUDA非決定性が疑われる場合もtoleranceで通さない、というhandoff文書の要求通り）。

synthetic test 15件（parity gateの正常系・4種の不一致系、vote countバケット境界、層別集計の
scope_mask処理、TP/FP/FN/TN分類でのignore点除外、XY bin集計、Step H3.1 recurrenceとのjoin）を
実装し、この開発コンテナ内（torch未インストール、モジュールがtorchに依存しないことも合わせて
確認）ですべて合格を確認した。`py_compile`・`bash -n`・`git diff --check`も合格。

**GPU側初回実行での不一致と対応（2026-09-15）:** validation_000でTP数が1点だけ不一致
（recomputed=8456 vs 保存済み8455）し、fail-fastが正しく発動した。vote_count・training window
occurrence countは完全一致していたため、window構成やcoverageの問題ではなく、1点のprobabilityが
threshold 0.5境界の極近傍でrun間にわずかに揺れた可能性が高いと判断した。ただし当初の実装は
fail-fastするだけで診断artifactを保存していなかった（handoff文書の要求を満たしていなかった）ため、
`verify_h4_parity()`を拡張し、predicted class不一致が起きた場合はraiseする前に該当点のindex・
GT label・両run のprobability・0.5境界からの距離・vote countを記録したCSVを保存するよう修正した
（pass/fail判定自体へのtolerance追加ではない）。回帰テストを追加し、この開発コンテナ内で合格を
確認した。

**Step H4.1: 原因訂正と集約規約統一（2026-09-15）。** 上記「CUDA非決定性」という当初診断は
誤りだった。方針管理チャットの精査（`docs/stage5/s5-14/stage5_s5_14_step_h4_parity_boundary_case_decision_request.md`
8節）とコード確認により、原因はH4 checkerの集約・判定規約が正本`evaluate_stage5.py:predict_h5()`と
異なっていたことと判明した：(1) `predict_h5()`は両クラスの確率を集約するが、S5-08
`run_overlap_forward()`はpositiveクラスのみ保持、(2) `predict_h5()`はwindow単位でfloat32化してから
集約するが、S5-08はfloat64直接cast、(3) `predict_h5()`は2クラスargmax（同値はbackground）、H4は
`mean_probability >= 0.5`（同値はpositive）。不一致点（point_index 290130）の確率は両run間で
完全一致（`abs_probability_diff=0.0`）しており、純粋にタイブレーク規約の違いによる見かけ上の
不一致だった。

修正として、S5-08 `run_overlap_forward()`をpredicted class判定に使うのをやめ、新設した
`run_h4_forward()`/`canonical_predicted_class()`が`predict_h5()`と同じ手順（window単位float32化→
両クラスfloat64加算→最終float32変換→2クラスargmax、backgroundは`1-positive`で復元せずsoftmax
実出力を保持）を再現するようにした。`verify_h4_parity()`・`classify_prediction()`はいずれも
この正本互換の`pred_label`を受け取る形に統一し、per-window診断用の`p1 > 0.5`規約とは明確に
区別した。方針管理チャット指定の3種の境界ケース（float64での厳密タイ、float64非タイだが
float32変換後にタイになるケース、素朴な`>=0.5`と正しいargmaxが食い違うケース）を含む
synthetic test 19件をこの開発コンテナ内ですべて合格を確認した。

**GPU側再検証結果（2026-09-15）:** 修正版を再実行した結果、**全21動画でparity gateが完全一致**
した（除外・tolerance追加なし、`max abs diff=0.000e+00`）。「CUDA非決定性」という当初診断は
完全に誤りであり、原因はcheckerの集約・判定規約の不一致のみだったことが実証された。
recurrence/exposure cross-checkでは、multiple unmatched runsのあるbin（791件）はないbin
（1,401件）よりtraining window occurrence約+10%・disagreement rate約+8%高いが、仮説A
（座標事前分布、8.7倍）と比べると効果は小さい。詳細は`docs/stage5/s5-14/stage5_s5_14_report_to_policy_chat.md`
のStep H4.1節。

**Step H4層別分析と3仮説の最終判定（2026-09-15）。** `point_overlap_error_statistics.csv`の
詳細分析により、window 16/stride 8ではvote_countが構造上1または2のみ（重複率50%）であることを
確認した。vote_count 1/2間でFP率（background点中）はほぼ同一（11.88%→11.64%）で「exposureが
多いほどFPが増える」という単純な関係は支持されなかったが、vote_count=2の中ではFPの
disagreement rate（34.1%）がTNの4.7倍（7.2%）に達し、window間予測の食い違いがFPへ偏る
限定的なメカニズムが見られた。またrelative_frame_decile別のdisagreement/exposureは動画中盤で
ピーク・両端で低いという対称パターンを示し、Step H3で確認したrecallの単調な低下
（decile0の64.2%→decile8の7.9%）とは形状が異なることから、後者は独立した時間効果と判断した。

3仮説の最終判定: **仮説A（座標事前分布）は強く示唆される**（bin単位point密度での正規化確認が
残課題）、**仮説B（時間位置）は明確に支持される**（recall低下、FPRへの影響なし）、
**仮説C（overlap exposure）は限定的に支持される**（FP率の単純な押し上げ効果はないが、
disagreementとの関連あり、主要因ではない）。単一の仮説に単純化できない複合的な結論となった。
Step H5（集計・報告書作成）へ進む前に、この複合的な結論の扱いを方針管理チャットへ報告する。

**Step H5: 可視化と出力。** shareable出力は匿名video aliasを使用し、少なくとも次を作成する。

```text
frame_metrics.csv
frame_position_deciles.csv
gt_overlap_exposure.csv
point_overlap_error_statistics.csv
xy_density_bins.csv または同等のNPZ
xy_temporal_recurrence.csv
video_summary.csv
stage5_s5_14_summary.json
```

固定train sanityと代表validationについて、GT/prediction/FP/FNの全点およびpositive-only PLYを任意出力
できるようにする。PLYと実video mappingはprivate出力とし、匿名化bundleへ含めない。既存のprivacy
self-check、mean baseline parity、同じpoint setの検証を維持する。

**仮説の判定:** 単一のsplit aggregateだけで判断せず、validation動画別勝敗・median・no-GT frame・
train sanityとの方向一致を確認する。

**Step H5実装（2026-09-15、実装・静的検証完了、GPU側実run待ち）。** 方針管理チャットは
Step H4.1の修正・parity検証を承認し、H4集計結果をS5-14の診断資料として採用した上で、Step H5で
「Aの密度正規化とB/Cの層別結果を整理」してから次の改修を決定する方針を示した。
`checks/real_h5/check_stage5_s5_14_summary_export.py/.sh`と対応する
`checks/dummy/check_dummy_s5_14_summary_export.py/.sh`を実装した。既存のStep H2/H2.5/H3/H3.1/H4
出力（すべてvideo_alias化済み）を入力とし、未生成だった`frame_position_deciles.csv`（frame単位
recall/precision/F1/IoU/FPR/FNRをsplit×relative_frame_decileで集計、未定義値は0埋めせず除外）と
`stage5_s5_14_summary.json`（3仮説の判定・headline数値・fileマニフェスト）を新規生成し、
Step H3/H4で別々だった`video_summary.csv`を1本へmergeし、仮説Cの根拠となる
`vote_count_error_rates.csv`（vote_count bucket別recall/FP率/disagreement rate、手計算で
検算済み）も併せて出力する。bundle全体に対しtimestamp形式video ID・絶対host pathの
privacy self-checkを実行する。synthetic test 5件（frame decile集計の未定義値除外、
video_summary mergeのkey不一致拒否、vote count別error rateの検算一致、privacy self-checkの
pass/fail）をこの開発コンテナ内ですべて合格を確認した。`py_compile`・`bash -n`・
`git diff --check`も合格。

**Step H5 GPU側実行結果（2026-09-15、ユーザー実機）。** 全21動画・privacy self-check合格
（`files_checked: 9`、timestamp形式video ID・絶対host pathとも検出なし）で完了した。
`vote_count_error_rates.csv`・`frame_position_deciles.csv`の値は、実装チャット側の手計算結果と
完全に一致し、独立した集計経路での再現性を確認した。train_sanityでも同様の傾向
（vote_count=2でFP disagreement rate 35.2% vs TN disagreement rate 6.7%）が見られ、仮説Cの
限定的な支持はvalidationだけでなくtrain_sanityでも方向が一致した。**S5-14 core（Step H1〜H5）は
これで完了とする。** 数値正本は`docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
9.8節、完了報告は`docs/stage5/s5-14/stage5_s5_14_report_to_policy_chat.md`。次の改修（座標equivariance診断、
inverse-occurrence loss weighting等）は方針管理チャットの判断を待つ。

- GT densityが低い共通XY領域へprediction/FPが動画横断で集中し、centroidがGTよりglobal priorへ近い場合、
  座標事前分布仮説を支持する。
- errorが動画内相対位置またはGT-positive区間からの距離と一貫して変化し、XY集中だけでは説明できない
  場合、時間位置・frame phase仮説を支持する。
- error/recurrenceがsource pointのwindow出現回数またはdisagreementと一貫して増え、GT class・時間位置で
  層別化しても残る場合、overlap exposure仮説を支持する。
- train sanity 3動画だけの傾向、少数動画、raw point density差だけで全体仮説を採用しない。

**結果による分岐:** core S5-14ではtraining code、loss、sampling、augmentationを変更しない。

1. 座標事前分布が支持された場合は、座標変換に対するprediction equivarianceの再学習なし診断を先に
   提案し、その後に限りrotation/mirror等のaugmentationを1要因ずつ5 epoch比較する。
2. overlap exposureが支持された場合は、inverse-occurrence loss weightingを第一候補、center-only lossを
   第二候補として、同時導入せず1方式だけの5 epoch比較案を提示する。
3. 時間位置だけが支持された場合は、window文脈とframe phaseの関係を追加診断し、座標augmentationや
   overlap lossを根拠なく導入しない。
4. いずれも支持されない、または効果が小さい場合は構造変更を追加せず、W-A baselineを維持して
   S5-15の5〜10 epoch最終pilot計画へ進む。

いずれの分岐でも、S5-14 core結果の報告前に再学習を開始しない。追加ablationは1回に1要因、最大5 epoch
とし、方針管理チャットの承認を別途得る。threshold tuning、aggregation変更、長期学習はS5-14の対象外
とする。

**productionへの影響:** なし。GroupNorm、W-A class weight、`bbox_noncontour_ignore`は引き続き実験用の
暫定条件であり、production既定値は変更しない。mean aggregationとthreshold 0.5を維持する。

## 6. 次の改修フロー

### S5-14補足 点密度補正と動画別・時間別再集計

方針決定日: 2026-09-15。状態: 実装完了（Step S1〜S5、synthetic/static検証済み）、実機再集計完了。
方針管理チャットによる完了受入済み（2026-09-15）。担当: Stage 5実装チャット。

**実装状況（2026-09-15）:** `Stage5/checks/real_h5/check_stage5_s5_14_supplement.py/.sh`と
`Stage5/checks/dummy/check_dummy_s5_14_supplement.py/.sh`を実装した。分母付きXY bin統計（Step S1）、
train-onlyのXY prior・hot/cold FPR比較（Step S2、train sanity自身の寄与除外を含む）、frame_metrics.csv
を再利用した動画内前半/後半paired recall比較（Step S3）、vote count 1/2のjoint層別FPR/recall
（Step S4）を、保存済みprediction（`pred_label`、`vote_count`）とH5から再構成する。per-windowの
disagreementはH4既存artifactの周辺集計からjoint層別を捏造できないため「未検証」として明記する実装
とし、GPU再推論は行わない。`vote_count`は`window_occurrence_counts()`による純geometry再計算とも
突き合わせ、一致しなければfail-fastする。py_compile・bash -n・git diff --checkと、h5py/numpyを
導入した検証環境でのsynthetic統合テスト（13ケース、CLI end-to-endのprivacy/alias検証含む）は
全て合格した。

**実機再集計の結果（2026-09-15）:** ユーザー実機で21動画（train sanity 3 + validation 18）・
train prior 162動画・grid 16/8で正常終了、`vote_count`の構造的再計算とのparity不一致0件、
既存confusion countsとのbin合計parityも合格した。詳細な数値は
`docs/stage5/s5-14/stage5_s5_14_supplement_report_to_policy_chat.md`の「実施記録」に記載。要点:

- **仮説A（座標事前分布）:** 分母補正・GT positive数0のbin限定・train sanity自己参照排除の
  全てを適用した後も、hot bin FPRはcold bin FPRより一貫して高い（差にして約17〜36ポイント）。
  grid16/8・prior 2定義（raw_count/rate）・train sanity/validation両split・比較可能な21動画
  全てで方向が一貫し、例外は0件。当初の「8.7倍」という点数比の指摘は分母補正後も同方向の効果
  として残った。
- **仮説B（時間位置）:** 同一動画内の前半(decile 0-4)/後半(decile 5-9)paired比較では、
  validation 18動画中12動画が比較可能で、7動画が低下・5動画が上昇と方向が割れ、
  mean diff +1.2%・median diff -1.2%とほぼ0に収束した。強い一貫した低下（-46.0%、-48.5%、
  -20.7%）が見られたのはtrain sanity 3動画（非代表サンプル、n=3）のみ。core報告の「単調急落」は
  pooled集計の動画構成差を反映していた可能性が高く、動画横断の一般的な時間効果としては
  本補足で再現されなかった。
- **仮説C（overlap exposure）:** vote_count 1/2間のFPR差は、前半/後半で層別しても10.8%〜15.3%の
  狭いレンジに収まり、S5-14 coreの結果と整合する形で小さいまま残った。per-windowの
  disagreementのjoint層別（動画×時間区分×vote_count）は、既存H4 artifactの周辺集計から
  復元不能なため実装せず「未検証」のまま維持した（捏造・追加GPU再推論なし）。

依頼書8章の分岐に照らすと、Aのみが分母補正後も動画別・両gridで一貫して残ったため、
再学習なしの座標変換診断を次案とする根拠が最も強い。Bは同一動画内では一般に再現されず、
時間文脈の限定診断を同等優先度で進める根拠は得られなかった。Cのexposure固有効果も
時間区分別で残らなかった。最終判断は方針管理チャットが行う。

**実装理由:** core成果物の再現性・parity・privacy checkは受け入れるが、分析上の留保は残る。
仮説Aの「8.7倍」はmean predicted点数320.8対36.8の比較であり、background点数を分母とするFPRではない。
仮説Bのdecile別recallは対象動画・GT点数の構成差を含み得るうえ、decile 3の70.4%はdecile 0の64.2%を
上回るため「単調低下」とは記述しない。仮説CのFP/disagreement関連は、training exposureがFPを増やす
因果関係とは区別する。H5 JSONの固定判定文字列は、新たな検証結果ではない。

**目的:** 点密度、動画構成、時間位置とexposureの交絡を限定的な再集計で確認し、座標変換診断、
時間文脈診断、loss比較、またはS5-15 pilotのどれを次に行うか決定する。

**固定条件:** teacher v7全180 H5（train 162 / val 18）、W-A epoch 5、GroupNorm 8 groups、
class weight `[0.05963856,1.94036150]`、`bbox_noncontour_ignore`、window 16/stride 8/tailあり。
予測評価は固定train sanity 3動画とvalidation 18動画。mean aggregationと保存済み2クラスargmaxの
判定規約を維持する。XYはH2.5の実寸法256×256に従い`pixel_xy / 255`とする。

**検証内容:**

1. XY binへ全点数、valid positive/background数、ignore数、TP/FP/TN/FNを追加する。
   主指標は`FP / valid background数`と`TP / valid positive数`。分母0は未定義とし、ignore予測は別記する。
2. train 162動画のGTから共通XY分布を作り、validation 18動画へ適用する。train sanityでは対象動画自身を
   prior作成から除く。grid 16/8でhot/cold領域のFPRを動画別、動画等重み、点数加重で比較する。
   raw GT点数による分布とvalid点数で補正したGT率の分布を区別し、定義・分母・対象数を保存する。
3. 各decileの動画数、GT-positive frame数・点数を併記し、同じ動画の前半/後半でrecallを比較する。
   可能な範囲でvote count 1/2別にも比較し、比較不能・疎な層を明示する。
4. Cは動画・時間位置で層別化したFPR/recallとdisagreementを分けて評価する。必要なjoint集計が既存H4
   CSVにない場合は保存済みprediction、H5、既存per-point artifactから再構成できる範囲に限定する。
   復元不能なdisagreement層別は未検証とし、再推論で補わない。

**検証量の上限:** 一度の限定的な補足再集計とする。新規学習0回、GPU再推論0回、gridは16/8のみ。
支持・不支持・データ不足のいずれでも報告して終了し、追加探索を自動で始めない。

**完了条件:** 全件・split・point alignmentと既存confusion countsが整合し、分母付き数値、動画別の
比較対象数・方向・median、未定義/未検証項目、privacy検証結果が揃うこと。3仮説の判定は固定文字列で
自動確定せず、数値と留保を添えて方針管理チャットへ返す。単位・分母の異なる倍率から主要因を順位付けしない。

**結果による分岐:** Aの偏りが密度補正後も残れば再学習なしの座標変換診断、Bの低下が動画内でも残れば
時間文脈の限定診断、exposure固有の悪影響が残ればinverse-occurrence loss weightingの5 epoch比較案を
検討する。強い構造的根拠が残らなければW-Aを維持しS5-15の5〜10 epoch pilotを計画する。
いずれも本補足の完了報告後に判断する。

**実機再集計後の分岐評価（2026-09-15、判断は方針管理チャットが行う）:** Aは分母補正後も
動画別・grid16/8の両方で一貫して残ったため、座標変換診断の分岐条件を満たす。Bは同一動画内の
paired比較では一般に残らなかった（train sanity n=3のみ）ため、時間文脈の限定診断を同等優先度で
進める根拠は本補足の範囲では得られなかった。Cのexposure固有の悪影響も時間区分別のFPRでは
残らなかった（disagreementのjoint層別は未検証のまま）ため、inverse-occurrence loss weighting
比較案を優先する根拠も得られなかった。

**productionへの影響:** なし。teacher・保存済みpredictionの変更、threshold tuning、loss変更、
augmentation追加は対象外。

**関連文書:** `docs/stage5/s5-14/stage5_s5_14_supplement_implementation_handoff.md`（実装依頼）、
`docs/stage5/s5-14/stage5_s5_14_report_to_policy_chat.md`（core報告）、
`docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.8節（数値正本）。
補足報告先は`docs/stage5/s5-14/stage5_s5_14_supplement_report_to_policy_chat.md`。

### S5-14補足2 座標変換診断

方針決定日: 2026-09-15。状態: parity合格・実機実行は完了（修正後のコードで再実行済み）。
分析仕様の残る確認事項はS5-14補足3として対応済み（次節）。2026-09-15にS5-14全体として受入済み。
担当: Stage 5実装チャット。

**実装状況（2026-09-15）:** `Stage5/checks/real_h5/check_stage5_coordinate_transform_
diagnostics.py/.sh`と`Stage5/checks/dummy/check_dummy_coordinate_transform_diagnostics.py/.sh`
を実装した。変換注入位置は`evaluate_stage5.predict_h5()`の`normalize_xyz()`直後・window抽出前
（監査済み、10.2節の契約どおり）。8条件の変換適用、identity/repeat_identityのS5-14 core H4
canonical parity再利用（no tolerance）、全条件でのvote_count構造的検算、前補足のhot/cold共通
分類（`train_xy_prior.csv`）再利用によるFPR比較を実装した。py_compile・bash -n・
git diff --checkと、synthetic test 13件（transform数学的性質、正規化前平行移動の相殺確認、
per-point比較指標、hot/cold FPR再利用、条件別集計）は全て合格した。

**実機実行結果（2026-09-15、旧集計・診断履歴として保持）:** ユーザー実機で21動画×8条件
（168 video-forward相当）を実行し、正常終了を確認した。T2のidentity/repeat_identity
canonical parityは全21動画でno-tolerance合格（`private_DO_NOT_SHARE` diff artifactは
生成されず、この結果は下記の修正の影響を受けない）。T3/T4は平行移動でほぼ無反応・回転
（特にhot bin）で明確な反応という非対称性を示したが、下記の通りhot/cold分類とsplit集計に
仕様不一致があったため、この結果は旧集計として保持し、正式判断には使用しない。

**集計仕様の修正（2026-09-15、管理チャット指摘への対応）:** 管理チャットの精査により、
(1) train sanity 3動画のhot/cold分類が、S5-14補足1と同じleave-one-video-out自己除外を
反映しておらず全train共通分類になっていたこと、(2) condition summary/hot_cold summaryが
train sanityとvalidationを分離せず21動画混合になっていたこと、の2点が指摘された。
`check_stage5_coordinate_transform_diagnostics.py`を修正し、`prior_excluding_video()`
（補足1、無改変で再利用）による自己除外をtrain sanity動画へ適用し、split別の主集計
（`condition_summary_by_split`等）と21動画混合の参考値（`*_all_videos_reference_mixes_
splits`）を分離した。CLI引数も`--train_xy_prior_csv`から`--train_list`へ変更した。
既存の保存済み出力には条件別・bin別のper-point/per-bin中間データがなく、train sanityの
修正後hot/cold FPRはCPU再集計だけでは復元できないため、D-033承認範囲内（21動画×8条件）の
再実行が必要と判断した。synthetic test 16件（追加3件含む）・static checkは全て合格。

**修正後の実機再実行結果（2026-09-15、split別、正式集計）:** ユーザー実機で修正後のコードを
再実行し、正常終了・privacy scan 0件を確認した。T2 parityは両splitとも差分0を維持。

validation（n=18、train_sanityの自己参照問題を受けない代表的なサンプル）単独で、修正前と
同じ質的パターンが再現された: 平行移動（±0.1）は全体・hot bin（grid16, rate）のFPR diff
中央値がほぼ0（それぞれ+0.003pt、+0.009pt）で方向も一定しないのに対し、回転（±15度）は
全体flip_rate中央値が平行移動の約25〜35倍、hot bin FPR diff中央値は`rotate_z_plus`で-2.88pt
（2/18動画のみ増加）、`rotate_z_minus`で-3.82pt（18動画中18動画が減少）と、方向が強く
一貫していた。train_sanity（n=3、自己除外適用後）も同方向だが、recallの低下幅（`rotate_z_
plus`で中央値-4.58pt、`rotate_z_minus`で-9.59pt）がvalidationより大きく出た。
**train sanityを分離した後もvalidation単独で同じ傾向が残ったことから、元の21動画混合の結果は
train sanity（n=3）の外れ値的な影響で駆動されたものではないと確認できた。**一方、
train sanityの自己除外適用により数値自体（hot bin FPR diffの大きさ等）は修正前から変化して
おり、自己除外が数値に実質的な影響を与えたことも確認された。

recallはvalidationで`rotate_z_plus`が中央値+0.16pt（9/18動画増加・9/18動画低下、ほぼ相殺）と
ほぼ無変化だが、`rotate_z_minus`は中央値-1.94pt（14/18動画が低下）と明確に低下しており、
hot bin FP減少を「回転による精度改善」と解釈しないよう留意した。正規化後の平行移動の低感度は
`normalize_xyz()`の中心化（正規化前の平行移動にのみ働く別の性質）とは無関係の独立した観測と
して記述し、hot/cold間の変化も「FPの再配分」ではなく別々の観測として扱った。

詳細な動画別数値・split別テーブル・解釈上の留保は
`docs/stage5/s5-14/stage5_s5_14_supplement2_report_to_policy_chat.md`を参照。

**補足結果への正式判断:** 前補足の完了を受け入れる。全21動画・grid16/8でhot領域のFPRが
cold領域より高く、background点数による分母の交絡だけでは説明できない。一方、局所的な点密度・
形状・輝度と座標の関係は未分離であり、「絶対座標の暗記」とは断定しない。時間位置低下は
train sanityでは残るがvalidation全体では一貫せず、exposure固有の悪影響も明確でないため、
時間文脈診断・inverse-occurrence loss比較は優先しない。joint disagreementは未検証のまま保持する。

**目的:** 同じ点の相対距離・特徴・GTを保ったままモデル入力XYZの位置・向きを変え、
予測の変化を元点単位で比較する。再学習前に座標依存への対策を試す根拠があるか確認する。

**固定条件と範囲:** teacher v7、W-A epoch5、GroupNorm8、元のsplit・train sanity3＋validation18、
window16/8、physical batch1・paddingなし、features intensity/confidence、mean aggregationを維持する。
全動画XYZ正規化の後・window抽出の前に一つの剛体変換を適用し、再正規化しない。
`pixel_xy / 255`は元点の診断座標であり、モデル入力XYZと混同しない。

**限定比較:** identityとrepeat identity、正規化XYZのX/Y各軸の±0.1平行移動（4条件）、
原点を中心としたZ軸±15度回転（2条件）の計8条件。scale・reflection・shear・振幅探索は行わない。
学習0回、最大21動画×8条件の168動画相当forward走査（各走査内で全windowを処理）とする。
小規模preflightの正常出力は本集計へ再利用し、追加seed・checkpoint・動画探索は行わない。

**検証と判定:** identityの既存保存値とのcanonical parityを先に確認する。点index・特徴・GT・
window/vote対応と距離保存を検算し、元点ごとの確率差・判定反転・動画別FPR/recall/F1を記録する。
変換感度そのものはバグや座標暗記の証明ではない。入力分布外への移動、方向性のある局所形状、
浮動小数点・近傍選択の影響も留保し、translationとrotationを分けて解釈する。
完了後、augmentationの単独5 epoch比較案またはW-AによるS5-15 pilotのどちらへ進むか再判断する。
本委任に学習・production変更の承認は含めない。

**関連文書:** `docs/stage5/s5-14/stage5_s5_14_supplement_implementation_handoff.md` 10章が実装依頼の詳細。
前補足報告は`docs/stage5/s5-14/stage5_s5_14_supplement_report_to_policy_chat.md`。
補足2報告先は`docs/stage5/s5-14/stage5_s5_14_supplement2_report_to_policy_chat.md`、
数値は評価レポート9.8節へ別項目として追記する。

### S5-14補足3 残存集計・記録の確認

方針決定日: 2026-09-15。状態: 完了。担当: Stage 5実装チャット。GPU再推論・新規学習は
一切行っていない（CPU専用のCSV再集計・記録確認）。

**委任内容:** 補足2の split別修正後の結果を方針管理チャットが確認した際、(1) split合算の
confusion countsから算出するpooled precision/recall/FPR/F1/IoU（動画別rateの平均とは別の
統計）が未算出、(2) F1/IoUの動画別diff中央値と、split内中央値同士の差を区別していない、
(3) 報告文に増減の取り違え等の誤記がある、(4) 再推論（初回・修正後の計336 video-condition相当）
の事前承認記録が未確認、という4点が指摘され、補足2の最終受入を保留してS5-14補足3として
委任された。

**実装:** `Stage5/checks/real_h5/check_stage5_coordinate_transform_reconciliation.py/.sh`
（既存`coordinate_transform_point_metrics.csv`のみを入力とするCPU専用の再集計）と
`Stage5/checks/dummy/check_dummy_coordinate_transform_reconciliation.py/.sh`
（synthetic test 7件）を新規実装した。この入力CSVはH5/checkpoint/torchを必要としないため、
この実装環境内で実データに対して直接実行し、split別confusion counts整合性・identity自己diff=0
の検算に合格した。

**主な結果:** pooled（点数加重）recallは`rotate_z_minus`で両splitとも明確に低下した
（train_sanity約-9.2pt、validation約-6.8pt。厳密な差分の転記留保は下記）。一方、動画等重みのF1中央値差分
（median_of_per_video_diffs）はほぼ変化なし〜わずかに正（validation +0.06pt、18動画中11動画で
F1改善）。pooled recallは各動画のGT positive数（TP+FN）を重みとするrecallの加重平均であり、
背景点数や全点数を重みにするものではない。pooled FPRの重みはGT background数（FP+TN）である。
この分母別のpooled統計と、動画ごとに均等な重みのF1統計とで異なる
側面が見えており、precision改善（FPR低下）がrecall低下を動画単位では部分的に相殺している
可能性がある。「hot bin FP減少」「pooled recall低下」「動画別F1はほぼ中立」はいずれも
矛盾しない別々の観測として報告する。

**報告文の訂正:** T2 parityの記述（利用可能動画数が0であるかのような誤記）、grid8・raw_count
定義での増減の取り違え（train_sanityの「減少」を実際には「増加」の数値で記載していた）、
train_sanity recall diffのmean/median取り違え（正しい中央値は`rotate_z_plus`-4.58pt、
`rotate_z_minus`-9.59pt）の3件を本書・管理記録へ反映した。

**再推論の承認経緯:** 方針管理チャットによる事前の明示的承認記録は本書上に存在しないため、
「承認記録未確認」として報告する。実施量は初回・修正後の計2回、21動画×8条件=168
video-condition相当×2＝336 video-condition相当。これは得られた数値を無効化する趣旨ではなく、
運用上の記録確認である。

詳細・全数値は`docs/stage5/s5-14/stage5_s5_14_supplement2_report_to_policy_chat.md`の
「S5-14補足3 完了報告」を参照。

### S5-14全体の完了判断と留保（2026-09-15）

ユーザー判断によりcore（H1〜H5）・補足1〜3を完了として受け入れる。補足2のparity・自己除外・
split別再集計、補足3のCPU集計と7件のsynthetic検証を完了記録とし、追加GPU診断は行わない。
完了は診断工程の完了であり、精度問題の解消・座標暗記の確定・augmentation採用を意味しない。

**数値転記の留保:** 補足3報告のvalidation recall表示値44.71%→37.88%は約-6.83ptだが、
差分欄は-6.81ptで一致しない。他の差分欄も含め厳密な引用には生成CSVとの照合が必要であり、
本書では約-6.8ptとして扱う。CSV未照合の値を確定値へ置換しない。これは文書整理上の留保であり、
新規検証段階・追加推論の理由にはしない。

回転感度とFP/recallのトレードオフはvalidation単独でも確認された。ただし正負回転で結果が異なり、
変換の移動量と方向依存の局所特徴を分離できていない。＋15度だけを事後的に有利な設定として採用しない。
次候補は回転augmentation単独5 epoch比較とし、条件を事前固定してW-Aと比較する方針を別途策定する。
今回の完了判断は学習開始の承認ではない。production設定・W-Aを維持する。

再推論の事前承認記録未確認は運用上の留保として保持し、ユーザー自身の実行と区別する。
把握済みの2回・計336動画条件相当以外の実行有無は未確認。これをもって診断結果を無効とはしない。
詳細な数値と留保の正本は評価レポート9.8.4〜9.8.5節。

### S5-15 長期学習とproduction候補確定

方針策定日: 2026-09-15。状態: 計画確定、実装・実行未着手。担当: Stage 5実装チャット。
S5-14全体はユーザー判断により完了・コミット済み。以下の段階ごとに結果を管理チャットへ返し、
次段階へ進む承認を得る。200 epochを一括実行する計画ではない。

#### 目的と判断原則

回転augmentationを一要因で比較して学習条件を固定し、短期pilot、50 epoch中間判定を経て
長期学習・production候補を評価する。S5-14で確認したのは推論時の回転感度であり、学習時
augmentationの有効性ではない。＋15度の推論結果が良かったことを理由に、片方向回転や
推論時回転を採用しない。座標暗記の因果的証明を追加診断で追求する段階にも戻らない。

#### P1: 固定条件・実験manifestの監査

| 項目 | 短期比較の固定条件 |
| --- | --- |
| teacher / split | teacher v7、既存W-Aのtrain162 / validation18。保存済みfile listを使用し再分割しない |
| 評価対象 | 固定train sanity3動画＋全validation18動画。GT・元点対応も維持 |
| モデル | PointNeXt-S、GroupNorm8 groups、W-Aと同じwidth/radius/nsample等 |
| 初期重み | W-Aを学習した際と同一のGroupNorm転移初期checkpoint。両armでhashを一致させる |
| 特徴 / label policy | intensity,confidence / bbox_noncontour_ignore |
| Loss / class weight | CE、smoothing0、W-Aの固定weight [0.05963856, 1.94036150] |
| Optimizer | AdamW、lr=1e-3、weight_decay=1e-4、grad_clip_norm=10。W-A実configと照合 |
| 入力 / batch | 全動画XYZ正規化、window16/stride8/tailあり、physical batch1、accumulation8 |
| Sampling | overlap windowの実点を維持。random point removal・zero paddingなし |
| 評価 / 集約 | eval mode、augmentationなし、mean probability、2クラスargmax・同値background |
| 乱数 | W-Aのseed・shuffle条件を再使用。augmentationの乱数はshuffle/modelの乱数から分離 |
| 保存 | 独立した新規run dir、config・metrics・file lists・best/lastと各epoch checkpoint |

既存bashの既定値がW-Aと一致するとは仮定しない。特にGroupNormとその初期checkpointを明示し、
W-A epoch5の学習済み重みを新しいarmの初期値にしない。auto weightを再算出して比較条件を
変えることも避ける。入力file list・GT/初期重みの同一性、コードrevision、実行環境、主要引数を
manifestへ記録し、旧出力を上書きしない。W-A実configとの差があれば開始前に報告する。

#### P2: Training-only回転augmentationの実装・preflight

追加する設定はaugmentationのnone / random_z_rotationと回転範囲・seedに限定する。
production既定値はnoneのまま。実験用bashでは明示的に指定し、config/checkpointにも保存する。

- 全動画XYZ正規化後・window抽出前に、原点（動画重心）中心のZ軸回転を適用する。
- 角度はtrain動画×epochごとに一つ、Uniform[-15度, +15度]から生成する。同一動画の全windowで
  同じ角度を使い、epoch間では更新する。動画内のoverlap点を異なる角度で学習させない。
- base seed・epoch・安定した動画識別子から再現可能に生成する。Pythonのプロセス依存hashや
  workerの呼出順に依存しない。角度生成用RNGをモデルやDataLoaderのRNGと共有しない。
- persistent workerを含むDataLoaderへのepoch伝達を検証する。キャッシュ済み点群をin-placeで
  回し続けない。距離・点順・特徴・GT・valid_mask・frame_order・point_indicesを維持する。
- 回転後の再正規化・clip・sampling・translation/scale/reflection追加はしない。
  validation・train sanity評価・inferenceでは常に無変換にする。

CPU syntheticで角度範囲、距離保存、同一動画/epochの一致、epoch更新、worker間再現性、
元データ不変、無効時の既存経路との一致を検証する。GPUでは限定したdummy forward/backwardと
実H5の少数stepでfinite loss/gradient・点対応を確認する。preflightはフル1 epoch学習を
別途追加するものではなく、実施動画/step数と実行回数を実行前に提示する。

#### P3: 5 epoch一要因比較

| arm | augmentation | 学習開始・期間 |
| --- | --- | --- |
| R0 control | none | W-Aと同一の初期重みから5 epoch |
| R1 candidate | random_z_rotation、角度[-15度,+15度] | 同じ初期重み・seedから5 epoch |

通常は2 run、計10 training epochsを上限とする。既存W-Aの5 epoch runをR0として再利用できるのは、
初期重み・teacher/split・config・seed・optimizer更新条件・実行環境・augmentation無効時の経路に
実質的な差がないことをP1/P2で確認できた場合だけとし、再利用理由を報告する。
再利用不能ならR0を新規実行し、旧W-Aは履歴にする。追加seed・角度探索・3本目の5 epoch runは
自動追加しない。preflight失敗のやり直しも、理由・実施量を記録して追加実行の承認を得る。

比較の主対象は両armのepoch5。bestは同じ選択規則で保存し副次評価とする。異なるepochのbestだけを
比べてaugmentation効果としない。固定21動画の無変換評価で、pooled confusion-derived指標と
動画別F1/IoU中央値・paired差分・TP0・recall/FPRを併記する。ignore上のpositiveはFPと区別する。
PLYは元座標のGT/予測/positive-onlyを既存パイプラインで出力し、同じ動画・視点で比較する。

**事前の選定基準:** validationの動画別F1/IoU、pooled F1、TP0を主指標とし、recall/FPRと
train sanityを安全側の確認に使う。R1暫定採用は、validationのpaired F1差分中央値が正、
split median IoUとpooled F1がR0以上、TP0が増えず、recall低下やFP増加だけで説明される
悪化がなく、train sanityでもmedian F1/IoU・TP0が悪化しない場合を基本とする。
全条件を満たしても単一seedの暫定判断にとどめる。微差・指標間のトレードオフ・train sanityとの
不一致が残る場合はR0を維持し、結果を見て角度や判定基準を変更して採用しない。
pooled recallはGT positive数加重、FPRはGT background数加重で解釈する。

#### P4: 採用条件固定と最終pilot（5〜10 epoch）

P3の選定を管理チャットへ返し、noneまたはrandom_z_rotationを固定する。同一条件のP3の5 epoch
runが健全なら最終pilotの5 epoch部分として再利用し、儀礼的に再学習しない。追加確認が必要な場合だけ
累計10 epochまでの延長案を出す。label policy・weight・LR・scheduler等を同時に変更しない。

**継続学習の注意:** 現在の`train_stage5.py --checkpoint`はモデル初期化用であり、optimizer等を
復元するresumeとは扱わない。延長する場合はepoch・optimizer・必要なRNG/データ順序・augmentation
epochを復元できるresume経路を実装・検証するか、同じ初期値から通算期間を再実行する計画を
事前承認する。モデル重みだけの再読込を「5→10→50 epochの連続学習」と記録しない。

#### P5: 50 epoch中間判定

pilotを受け入れた場合だけ50 epochまでの実行を承認する。継続か再開始かをmanifestへ明記し、
optimizer更新回数・通算epochを比較可能にする。短期比較ではschedulerを追加しない。
50 epochでもまず採用条件を固定し、LR変更が必要なら別の変更として管理チャットへ相談する。

各epochのtrain/validation loss・FP/FN/TP/TN・weight/ignore情報を保存し、10 epochごととbest/lastを
保持する。動画単位の評価とPLYはpilot時点・epoch25・epoch50を基本とし、epoch25も明示的に保存する。
validation lossの継続的な悪化、TP0の増加、video median F1/IoUのpilot比低下、FP/FPRの増大、
train sanityとvalidationの改善方向の乖離を確認する。NaN/Inf・点対応破損は即停止。
品質指標の継続悪化は次の保存区切りで中断・報告し、epochを増やすだけで解決しようとしない。
単一のbest値だけで継続を判断せず、時系列と動画別結果を管理チャットへ返す。

#### P6: 100〜200 epochとproduction候補の評価

50 epoch判定が良好な場合のみ100 epochへ、100 epochの再判定後に必要なら200 epochへ進む。
各延長は別承認とし、同じ監視・保存を続ける。最終モデルは事前固定のvalidation選択規則で選び、
lastと選択checkpointの両方を報告する。

モデルを固定した後にのみthreshold sweepを行う。候補は0.1〜0.9（0.1刻み）とし、baseline0.5を
必ず残す。両クラスargmax・float32同値backgroundという既存判定との整合をテストする。
FP/FPR、recall、TP0、video median指標、必要に応じStage6入力としてのFP形状を併せて判断する。
Stage6側の許容基準が未定なら「production上許容」と断定しない。

aggregationはmeanを基準に維持し、変更を検討する場合のみ同じcheckpointで4方式を副次比較する。
maxは診断用で、S5-14以前の判断どおり既定の採用候補に戻さない。aggregationとthresholdを同時に
探索して改善要因を混ぜず、変更候補がなければ4方式の再実行は必須にしない。
production採用の最終判断はnormalization・augmentation学習設定・aggregation・thresholdを明示して
別途行い、それまでは既存既定値を変更しない。

validation18動画はすでに多数の方式選択に使用しており、最終の独立testではない。thresholdや
checkpoint選択後の数値を未見データの汎化保証と呼ばない。最終的な性能確認には、調整に使っていない
動画群を別途確保し、動画単位の分離とteacher版を明記する。確保不能ならその限界を報告する。

#### 成果物・記録・実施境界

実装依頼はP1〜P3から開始し、P4以降は結果を受けた段階承認とする。本節の方針策定だけで実行を
開始しない。実装チャットへ渡す依頼書は別途作成する。
各段階でコードrevision、run manifest、config、metrics、checkpoint、固定21動画の評価・PLY、
匿名化metrics、実施量と判断を残す。bash内で実機のパスを指定可能にし、GPU実行時は既存の
dualtrack311/CUDA_HOME/TORCH_CUDA_ARCH_LIST/LD_LIBRARY_PATH設定を踏襲する。
元H5・旧run・旧評価を上書きせず、`.tmp/`や生データ・重みをコミットしない。

進捗・採否は本管理記録、数値と検証は評価レポートの新しいS5-15節、実装配置は`FILES.md`へ記録する。
S5-14の転記上の留保は生成CSVとの文書照合として処理し、新たなGPU診断や補足段階へ戻さない。

#### S5-15 P1〜P3の結果と現在の状態（2026-09-19追記）

P1（固定条件監査）・P2（実装・CPU検証・限定GPU preflight）・P3（R0/R1各5 epoch比較と
固定21動画評価）を完了した。詳細な数値は評価レポート9.9節、経緯は
`docs/stage5/s5-15/stage5_s5_15_report_to_policy_chat.md`。

**判断: R0を維持し、R1を今回は採用しない。** validation pooledでFPRが12.53%→7.91%へ下がる一方
recallが45.08%→32.41%へ下がり、pooled F1は6.77%→7.36%、TP0は1→0と改善するものの、
動画別F1は改善9/悪化9、paired差分中央値は約+0.03ptにとどまる。微差と大きな
recall/FPトレードオフが残るため、事前方針どおりR0を維持する。
「R1に効果がない」「回転augmentationが一般に無効」とは結論しない。FP削減効果は観測されており、
threshold調整やStage6のFP許容基準が定まった段階では評価が変わりうる。
今回の判断をthreshold調整で後から覆さず、production設定も変更しない。

**保存仕様の逸脱（解消せず記録する）:** P3の両armは`train_stage5.sh`が環境変数`SAVE_EVERY`を
素の代入で上書きしていたため、意図`SAVE_EVERY=1`に対し実効`save_every=10`で動作し、
**per-epoch checkpointが1件も保存されなかった**。5 epoch分のmetricsはhistory.jsonに残るが、
**中間epochの重みは復元不能**である。epoch5の`last.pt`と`best.pt`（評価モデルとして同一）は
揃っており主比較は成立するため、欠落重みの再生成だけを目的とした再学習は行わない。
既存のmanifest・config・重みを修正して当初から正しく保存されていたように見せない。

再発防止として`train_stage5.sh`の`SAVE_EVERY`/`SEED`/`GRAD_CLIP_NORM`/`NUM_WORKERS`/`PYTHON`を
環境変数上書き可能な形式へ変更し、launcherがexportする全変数が採用されることを検査する静的テストと、
実行中のrunの**実効config**と定期保存の動作を早期に確認するCPU checkerを追加した（commit `cd3b886`）。
静的テストはshellの解析であり、値がPythonへ届くことの実行時検証とは区別する。

**次段階:** 同一のGroupNorm転移初期重みからR0条件で新規50 epochを1 run実行する（承認済み）。
P3の`last.pt`からの継続ではなく、resume実装も行わない。開始前に保存済みmanifestと実効configの
CPU突合を行う。確認は開始時・epoch25・epoch50へ集約し、50 epochで固定21動画評価とPLY確認を行う。
100〜200 epoch延長・追加seed・R1長期比較・threshold tuning・production変更は未承認。

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
| D-023 | 2026-09-14 | W-B（`[0.5,1.5]`）を不採用とし、W-A（`[0.05963856,1.94036150]`）をS5-13補足およびS5-14の暫定class weightとして採用する | W-Bはthreshold 0.5でvalidation recall 0%、TP0 18/18へ完全崩壊し、W-Aは検出能力を維持したため |
| D-024 | 2026-09-14 | S5-14前にS5-13補足としてthreshold-free診断と中間weight W-C（比16）の限定確認を行う | W-A/W-Bの間隔が広く、固定threshold結果だけではdiscrimination消失とcalibration shiftを分離できないため |
| D-025 | 2026-09-14 | S5-13補足の新規5 epoch pilotを原則1回、最大2回に制限する | 補足検証を広いweight sweepや長期学習へ拡大せず、S5-14前の判断に必要な最小量へ抑えるため |
| D-026 | 2026-09-14 | W-C（`[0.11764706,1.88235294]`、比16）を不採用とし、W-AをS5-14の暫定class weightとして確定する | W-CはFPRを下げたがrecall medianとTP0が悪化し、AUPRC/AUROCおよび同一FPRでのrecallもW-Aを下回ったため |
| D-027 | 2026-09-14 | W-B/W-CはS5-15で参照可能な診断履歴として保持するが、activeなproduction候補にはしない | calibration shiftの資料にはなるが、両者ともW-Aよりdiscriminationが低く、現行thresholdで検出不足が大きいため |
| D-028 | 2026-09-14 | weight比24などの追加中間weightを試さず、未使用の2回目の5 epoch枠を消化しない | W-Cは同一FPRの全比較点でW-Aよりrecallが低く、追加のweight弱化を支持する根拠が得られなかったため |
| D-029 | 2026-09-14 | S5-14 coreを再学習なしの座標・frame・overlap exposure診断として開始し、結果報告前にloss/augmentationを変更しない | PLY上のXY反復について座標事前分布、時間位置、window重複露出の3仮説が未分離であり、先に原因を定量化する必要があるため |
| D-030 | 2026-09-15 | S5-14 core成果物を受入れ、3仮説の解釈はS5-14補足で再評価する | Aの8.7倍は点数比で密度補正未了、Bは動画構成差が残り単調低下でもない、Cのdisagreement関連はexposureの因果効果を示さないため |
| D-031 | 2026-09-15 | S5-14補足をS5-15前に一度の限定再集計として実施する。学習・GPU再推論0回、grid 16/8、W-A固定 | 点密度と動画・時間構成を確認してから次の改修を選び、根拠不足でも追加探索を自動拡大しないため |
| D-032 | 2026-09-15 | S5-14補足を受入れ、空間的な誤分類の偏りを次の診断対象にする | 分母補正・自己参照排除後も全動画で偏りが残る一方、一般的な時間低下・exposure対策の根拠は弱いため |
| D-033 | 2026-09-15 | S5-14補足2の座標変換診断を委任する。W-A固定・学習0回、21動画×8条件以内。S5-15は保留 | 点の対応と相対距離を維持して入力XYZへの感度を調べ、augmentation比較の必要性を判断するため。production変更は承認しない |
| D-034 | 2026-09-15 | ユーザー判断によりS5-14 core・補足1〜3を完了として受入。追加GPU診断は終了し、回転augmentation単独5 epoch比較を次候補とする | parity・自己除外・split/pooled集計・記録確認を完了。座標暗記・augmentation効果は未確定、転記と承認経緯の留保を保持。学習開始・production変更は未承認 |

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
- `docs/stage5/s5-08-09/stage5_overlap_aggregation_handoff_prompt.md`
  - 事項A/Bの実装・実行履歴、環境、privacy契約
- `docs/stage5/s5-08-09/stage5_overlap_aggregation_implementation_policy.md`
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
