# Stage 5 PointNeXt-S学習・評価調査報告

> **位置づけ注記（2026-09-10）:** 本書は検証結果と数値的根拠の正本である。Stage 5全体の
> 現在状態、採用済み判断、次の実施順は`stage5_revision_management_record.md`を参照する。

## 1. 目的

Stage 4で生成したpseudo-3D教師データを用いたStage 5 PointNeXt-Sについて、次の現象を定量・定性の両面から整理する。

- train sanityでは大腿骨付近を検出する一方、輪郭付近を中心にFalse Positive（FP）が多い。
- train sanityのpositive点群を側面から見ると、似たXY位置の検出が複数frame群で反復して見える。
- validationではpositive判定が少なく、多くのデータで大腿骨を検出できない。
- checkpointの進行に伴う改善と過学習の関係を確認する。
- Dataset、window分割、batch化、評価時aggregationに、GTや予測を誤って複製する処理がないか確認する。

本報告は2026-08-29時点の実装と、匿名化済み評価metricsを対象とする。元H5、元PLY、動画ID、絶対パスなどの識別情報は解析対象に含めていない。

## 2. 評価対象

### 2.1 学習条件

| 項目 | 設定 |
| --- | --- |
| model | PointNeXt-S Stage 5 wrapper |
| input features | `intensity,confidence` |
| frame window | 16 frames |
| window stride | 8 frames |
| include tail | enabled |
| batch size | 8 |
| epochs | 150 |
| learning rate | `1e-3` |
| weight decay | `1e-4` |
| loss | weighted CrossEntropyLoss |
| label smoothing | `0.0` |
| class weight | auto: `[0.062828, 1.937172]` |
| label policy | no-BBox frame = background、BBox内かつcontour外 = ignore |

学習データ全体のuniqueな有効ラベル数は、background 62,639,449点、positive 803,862点であり、positive比率は約1.27%である。auto class weightはpositiveをbackgroundの約30.8倍に重み付けする。

### 2.2 評価対象checkpoint

- `checkpoint_epoch_0030.pt`
- `checkpoint_epoch_0100.pt`
- `checkpoint_epoch_0150.pt`
- `best.pt`（validation基準ではepoch 45）

評価対象はtrain sanity 3動画、validation 18動画である。train sanityには固定選択1動画と、seed固定のランダム選択2動画を使用した。

## 3. 定性的な観察

PLYの目視確認では、train sanityについて次を確認した。

- 大腿骨のGT周辺がおおむねpositiveになるよう学習されている。
- 明確な輪郭線付近に誤ったpositiveが多い。
- positive点群を上面から見ると、GTの大腿骨付近にほぼ直線状に集まる。
- 側面から見ると、似たXY位置のpositive点群が2～3個のframe群で反復して見える。

validationでは、全体としてpositive判定が少なく、ほぼnegativeとなるデータが多かった。一部データではGT大腿骨領域の一部をpositiveとして検出できていた。

反復して見えるpositiveについては、「あるwindowのGTが別windowや別batchへ誤って流用された」という実装不具合と、「重複window、座標事前分布、paddingなどが組み合わさった学習挙動」の両方を候補として調査した。

## 4. checkpoint別の集約結果

### 4.1 Train sanity

| checkpoint | Precision | Recall | F1 | IoU | FPR | predicted positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch 30 | 0.1060 | 0.9841 | 0.1914 | 0.1058 | 0.1246 | 86,083 |
| epoch 100 | 0.1544 | 0.9969 | 0.2674 | 0.1544 | - | 59,866 |
| epoch 150 | 0.1964 | 0.9943 | 0.3281 | 0.1962 | - | 46,943 |
| best (epoch 45) | 0.1495 | 0.9707 | 0.2591 | 0.1489 | - | 60,202 |

epoch 150ではTP 9,221、FP 37,722、FN 53である。Recallは十分高いが、positive予測の約80.4%が有効background上のFPであり、train sanityが解決したとはいえない。

ignore点へのpositive予測も存在するが、epoch 150では推定2,457点であり、`valid background FP + ignore positive`の約6.1%にとどまる。したがって、BBox内かつcontour外をignoreとする方針は輪郭付近のFPに寄与し得るものの、広範囲のFPを単独では説明できない。

ignore領域のpositive予測率にはサンプル差が大きく、epoch 150の3動画では約0.128、0.811、0.981であった。教師領域や点群分布への依存が強いことを示す。

### 4.2 Validation

| checkpoint | Precision | Recall | F1 | IoU | predicted positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| epoch 30 | 0.0528 | 0.2098 | 0.0844 | 0.0441 | 296,844 |
| epoch 100 | 0.0712 | 0.0524 | 0.0603 | 0.0311 | - |
| epoch 150 | 0.1519 | 0.0715 | 0.0973 | 0.0511 | - |
| best (epoch 45) | 0.1328 | 0.1062 | 0.1180 | 0.0627 | - |

`best.pt`が4 checkpoint中で最も高い集約IoUを示した。ただし、18動画のmacro IoUは0.0473、median IoUは0.0031であり、動画間のばらつきが大きい。

`best.pt`では18動画中8動画がRecall 0、10動画がRecall 0.01未満だった。一方、5動画ではRecall 0.1を超えている。したがって、単純な「常にnegativeを出す崩壊」ではなく、データごとの位置合わせまたは特徴分布の不一致が強い。

79個のvalidation windowのうち、GT-positiveを含むwindowは56個、negative-only windowは23個だった。`best.pt`では次の結果となった。

- GT-positive windowの32/56で1点以上のTPを検出した。
- GT-positive windowの24/56はRecall 0だった。
- negative-only windowの18/23でpositiveを誤検出した。
- negative-only window上のFPは合計39,521点だった。

validationの目視結果とmetricsは整合しており、検出不能windowと背景FPが併存している。

## 5. 学習履歴

train IoUはepoch 30の0.153からepoch 150の0.334まで上昇した。一方、validation lossはepoch 8の約0.506を最小として、その後epoch 45で約1.117、epoch 100で約1.609、epoch 150で約2.081まで増加した。

これはvalidation loss計算だけの不具合よりも、強い過学習を示す挙動である。validation IoUは単調には低下せず、epoch 45付近で相対的に良い結果を示すため、lossと閾値後のIoUの最良epochが一致しないこと自体は異常ではない。ただし、epoch 100以降を採用する根拠は弱い。

## 6. Overlap windowの影響

16-frame window、stride 8では中央frameが複数windowに含まれる。train sanity 3動画を調べた結果、各動画は2 windowからなり、GT-positiveの重複は次の通りだった。

| sample | unique GT-positive | window 1 | window 2 | overlap | tail-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 1,661 | 1,661 | 1,489 | 1,489 | 0 |
| random 1 | 1,377 | 1,377 | 1,140 | 1,140 | 0 |
| random 2 | 6,236 | 5,663 | 5,630 | 5,057 | 573 |

学習データ全体では、unique点に対するwindow内出現回数の倍率はbackground約1.69倍、positive約1.90倍だった。結果として、unique点で1.27%だったpositive比率はwindow集計では1.42%になる。

現状のlossはwindowごとに同じ点を再度数えるため、中央frameとpositive点へ相対的に強い重みがかかる。これは同じXY位置付近の検出がframe方向に反復して見える現象の一因になり得る。ただし、metricsに座標情報がないため、この因果関係はまだ確定できない。

## 7. 実装監査

### 7.1 Datasetとbatch化

Datasetのsample indexは、H5 indexとwindow indexを個別に保持している。`__getitem__`は該当H5と該当windowから点、label、元H5内の`point_indices`を選択する。collateも各sampleを独立にpaddingしている。

この範囲では、1つのGTをbatch内の全sampleへ明示的にコピーする処理は確認されなかった。train sanityの3動画でGT数と予測傾向が異なることも、単純なGT一括流用とは整合しにくい。

### 7.2 Padding

可変長windowはbatch内最大点数までzero paddingされる。一方、現在のPointNeXt-S wrapperは実点数やpadding maskをOpenPointsモデルへ渡していない。このためPointNeXtのFPS、近傍探索、BatchNormがpadding点を実点として扱う可能性がある。

確認対象runでは、sampleの点数がtrainで約52,000～190,000点、validationで約73,000～173,000点と大きく異なり、padding率は全体で約14.3%だった。評価はbatch size 1で行われるためpaddingがなく、学習時と評価時の入力条件も一致していない。

これは現在確認された中で優先度の高い実装上のリスクである。

### 7.3 座標の利用

XYZ正規化は動画全体に対して行われ、その後にwindowが切り出される。このためwindow間で正規化座標系が共有され、PointNeXtは動画内の相対位置を利用できる。

現状では座標の平行移動、回転、mirrorなどのaugmentationがない。入力featureも`intensity`と`confidence`のみであるため、モデルが「右下」などの座標事前分布へ依存する可能性がある。側面表示で似たXY位置の検出が反復するという目視結果と整合するが、frame単位の座標metricsによる確認が必要である。

### 7.4 推論時のoverlap aggregation

評価処理は各windowの予測確率を元H5の`point_indices`へ加算し、vote数で平均する。したがって、overlap点が出力PLYへ複製される処理にはなっていない。

PLY上で反復して見える点群は、exporterによる同一点の複製ではなく、異なるframeに由来する別のpseudo-3D点である可能性が高い。

## 8. 原因候補の評価

| 原因候補 | 現時点の判断 | 根拠 |
| --- | --- | --- |
| GTを全batchへ誤って複製 | 直接的な証拠なし | Dataset/collateに該当処理がなく、サンプル別metricsも異なる |
| BBox内かつcontour外のignore | 部分的に寄与 | 輪郭付近のhard negativeを学習しないが、ignore上のpositiveだけではFP総数を説明できない |
| auto class weight | FPを増幅する可能性あり | positiveへ約30.8倍の重みを与える |
| overlap window | 寄与する可能性が高い | positiveがbackgroundより高頻度に重複し、中央frameが反復してlossへ入る |
| paddingを実点として処理 | 優先度の高い不具合候補 | 可変長batchにmaskがなく、学習とbatch size 1評価で条件が異なる |
| 絶対的な座標事前分布 | 寄与する可能性あり | 全windowが共通座標系で、座標augmentationがない |
| 過学習 | 明確 | train指標改善とvalidation loss悪化が同時進行 |
| no-BBox background不足 | 主因ではない | 現在はno-BBox frameをbackgroundとして学習済み |

現時点では、単一原因よりも、padding、overlapによる重複重み、強いclass weight、座標事前分布、ignore領域、過学習が複合していると考えるのが妥当である。

## 9. 次の検証手順

再学習条件を増やす前に、以下を順番に実施する。

各事項には、検証後に「実施日」「対象」「結果」「判断」「次のアクション」を追記する。検査スクリプトが完走したことと、検証仮説が支持または棄却されたことは分けて記録する。

### 9.1 実H5 batch integrity test

各batch sampleについてH5 index、window範囲、`point_indices`、label hashを記録し、元H5から直接再取得した配列と一致することを検証する。

実装済みの検査は次のbashから実行する。

```bash
bash /mnt/data/3d_projects/models/Stage5/checks/real_h5/check_stage5_batch_integrity.sh
```

検査結果は既定で`/mnt/data/3d_projects/stage5_debug/batch_integrity/<run_name>/`へ保存される。`integrity_summary.json`の全splitが`passed`であることに加え、sample単体、shuffle + multi-worker、手動batchの各検査結果を確認する。

#### 検証結果（2026-08-30）

| 項目 | 内容 |
| --- | --- |
| 対象run | `pointnext_s_EX260801_260711_w16_s8_bboxrankv2_nobboxbg_glocal_ce_smooth00_auto_weight_lr1e3_ep150_bs8` |
| 対象split | train全件、validation全件 |
| 実行状態 | `Stage5 real-H5 batch integrity check passed.` |
| summary | `integrity_summary.json`: `status=passed` |
| Dataset照合 | 合格 |
| ordered source check | 合格 |
| shuffle + multi-worker check | 合格 |
| 手動batch check | 合格 |

各windowについて、`points`、`features`、`labels`、`valid_mask`、`frame_order`、`point_indices`が元H5から直接取得した期待値と一致した。collate後の実点部分とpadding値も一致し、shuffle + multi-workerの1 epoch走査でsampleの欠落、重複、hash変化は検出されなかった。同一H5のoverlap window、異なるH5、点数差が大きいsampleを組み合わせた手動batchも合格した。

結果ファイルは次のディレクトリに保存した。

```text
/mnt/data/3d_projects/stage5_debug/batch_integrity/
  pointnext_s_EX260801_260711_w16_s8_bboxrankv2_nobboxbg_glocal_ce_smooth00_auto_weight_lr1e3_ep150_bs8/
    integrity_summary.json
    train_sample_integrity.csv
    train_batch_integrity.csv
    val_sample_integrity.csv
    val_batch_integrity.csv
```

**判断:** Dataset、overlap window index、collate、shuffle、multi-workerの範囲では、GTや点群を別sampleへ流用する不具合は強く除外できる。PLYで観察された同一XY付近のpositive反復を、単純なbatch/GT対応不具合で説明する根拠はない。

**次のアクション:** モデルへ渡されたzero paddingが実点のlogits、gradient、BatchNorm統計へ影響するか、9.2のpadding parity testで検証する。

### 9.2 Padding parity test

同じwindowを単独でforwardした場合と、点数の異なるwindowとbatch化した場合で、実点部分のlogitsを比較する。batch内の順序と相手sampleも変更し、予測変化を測る。

実装済みの検査は次のbashから実行する。

```bash
bash /mnt/data/3d_projects/models/Stage5/checks/real_h5/check_stage5_padding_parity.sh
```

`padding_parity_summary.json`の`status`は検査が完走したか、`verdict`はpadding影響の診断結果を表す。`verdict`が`padding_effect_detected`の場合、zero paddingが実点の予測、gradient、またはBatchNorm統計へ影響している。`inconclusive_no_padding_control_failed`の場合は、同一shapeの`repeat_single`でも再現性がないためCUDA演算やFPSの再現性を先に調べる。`inconclusive_no_padded_case`の場合は、純粋なmanual padding caseを構成できていないため、選択targetより長いpeerを確保して再実行する。

#### 検証結果（2026-08-30）

| 項目 | 内容 |
| --- | --- |
| 対象run | `pointnext_s_EX260801_260711_w16_s8_bboxrankv2_nobboxbg_glocal_ce_smooth00_auto_weight_lr1e3_ep150_bs8` |
| checkpoint | `best.pt` |
| 対象 | train/validationの最短windowおよびpositiveを含む最短window、計4 target |
| 実行状態 | `status=passed` |
| スクリプト判定 | `inconclusive_no_padding_control_failed` |
| 調査後判定 | **padding effect detected** |

同一shapeで同じ入力を再実行する`repeat_single`は、4 targetすべてでlogits、確率、予測labelが完全一致した。したがって、同じ実行条件における基本的な再現性は確認できた。

`duplicate_no_padding`ではbatch sizeが1から2へ変わり、最大確率差0.000275～0.009631、2つのpositive targetで15点と8点の予測反転が生じた。これはpaddingなしでもbatch shapeに対する小規模な感度があることを示すが、同一shapeの再実行失敗ではない。初期スクリプトはこのcaseを必須の再現性controlとして扱ったため、全体を`inconclusive_no_padding_control_failed`と判定した。

一方、`manual_zero_padding`はbatch size 1のままtarget末尾へcollateと同じzero paddingを追加するため、paddingだけを分離した比較である。結果は次の通りだった。

| split/target | 実点数 | padding点数 | padding率 | 最大確率差 | 平均確率差 | 予測反転 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train/min | 52,266 | 137,704 | 72.49% | 0.010574 | 0.000233 | 0 (0.00%) |
| train/positive_short | 55,721 | 134,249 | 70.67% | 0.847865 | 0.032723 | 2,095 (3.76%) |
| val/min | 73,335 | 99,605 | 57.60% | 0.312768 | 0.003712 | 0 (0.00%) |
| val/positive_short | 87,001 | 85,939 | 49.69% | 0.658092 | 0.009180 | 530 (0.61%) |

manual paddingの最大確率差は`duplicate_no_padding`の約34.8～95.6倍だった。negative-onlyの2 targetではclass labelの反転はなかったが、確率値は最大0.0106および0.3128変化した。positive targetでは最大0.658～0.848の確率差と実際のlabel反転が生じた。

実在peerで最大長へpaddingした`peer_max`と、zeroだけを追加した`manual_zero_padding`の最大確率差はほぼ一致した。両者の差は4 targetで約`1.2e-7`、`0.00198`、`5.6e-5`、`0.00198`だった。targetをbatchの先頭と末尾で入れ替えても`peer_max`の結果は一致した。このことから、主要因はpeerの内容やbatch内位置ではなく、targetへ追加されたpaddingと入力点数の変化である。

train modeでは、同一の2-sample batchをpaddingなしと手動paddingありで比較した。

| target | loss（なし→あり） | 最大確率差 | 予測反転率 | gradient相対L2差 | gradient cosine | BN最大絶対差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train/min | 0.1562 → 0.3795 | 0.9818 | 19.66% | 3.1989 | 0.0914 | 442.76 |
| train/positive_short | 3.7964 → 5.0022 | 0.9599 | 5.82% | 1.3510 | 0.3403 | 67.81 |

lossからpadding labelをignoreしていても、paddingは実点logits、gradient、BatchNorm running statisticsへ大きく影響した。特にgradient cosine similarityが0.09および0.34であり、paddingの有無によって更新方向自体が大幅に変化している。

結果ファイルは次のディレクトリに保存した。

```text
/mnt/data/3d_projects/stage5_debug/padding_parity/
  pointnext_s_EX260801_260711_w16_s8_bboxrankv2_nobboxbg_glocal_ce_smooth00_auto_weight_lr1e3_ep150_bs8_best/
    padding_parity_summary.json
    padding_parity_targets.csv
    eval_padding_parity.csv
    train_padding_parity.csv
```

**判断:** 初期スクリプトのverdictはcontrolの分類方法による過度に保守的な判定である。B=1のmanual padding比較とB=2のtrain-mode比較の双方から、現在のzero paddingはPointNeXt-Sの学習・推論に無視できない影響を与えると判断する。学習はbatch size 8のpaddingあり、評価推論はbatch size 1のpaddingなしで行われており、学習時と評価時の入力条件も一致していない。

**対応（2026-08-30）:** parity checkerを修正し、`repeat_single`を再現性control、`duplicate_no_padding`をbatch-size sensitivity、`manual_zero_padding`を純粋なpadding controlとして分離した。同じ結果に対する修正後の判定は`padding_effect_detected`となる。

**次のアクション:** 9.4のbatch size 1 + gradient accumulationによるpaddingなしbaselineを9.3より先に実施し、cleanな学習条件を確立してからframe-level metricsを評価する。

### 9.3 Frame-level metrics

`frame_order`ごとにGT/predicted positive数、TP、FP、FN、TN、ignore上のpositive数を記録する。可能ならGTと予測positiveのXY重心、重心間距離も匿名化された派生値として保存する。

### 9.4 Paddingの影響を除いたbaseline

まずbatch size 1とgradient accumulationで学習し、PointNeXtへpadding点を入力しない条件を作る。その後、必要に応じてpoint数bucket batchingまたはmask対応を検討する。

#### 実装（2026-08-30）

`train_stage5.py`へpoint-weighted gradient accumulationを追加した。各microbatchのCrossEntropyLossを単純平均せず、有効点のclass weight合計を分母としてloss numeratorとgradientを蓄積する。このため、点数とpositive比率が異なるwindowを蓄積しても、概念上それらを結合して計算したweighted CEと一致する。全点がignoreのmicrobatchは分母へ加えず、optimizer step直前にgradientを累積分母で正規化してからgradient clippingを行う。

`train_stage5.sh`の既定値は次の通りとした。

| 設定 | 値 |
| --- | ---: |
| physical batch size | 1 |
| gradient accumulation steps | 8 |
| effective batch size | 8 windows |
| padding | なし |
| window size / stride | 16 / 8 frames |
| class weight / smoothing | auto / 0.0 |

`metrics.jsonl`には`debug_microbatch_count`、`debug_optimizer_step_count`、`debug_gradient_accumulation_steps`、`debug_effective_batch_size_samples`、`debug_loss_normalizer`を追加した。padding-free条件ではtrain/validationとも`debug_padding_ratio=0`になることを確認対象とする。

以下はteacher v2でpadding-free経路を確認した際の1 epoch smoke testコマンドである。

```bash
EX_DATE=260830 EPOCHS=1 \
bash /mnt/data/3d_projects/models/Stage5/train_stage5.sh
```

このsmoke testでCUDA OOM、NaN、optimizer step数、padding ratio、checkpoint保存を確認した。teacher v2による200 epoch学習は、後述するteacher v6への移行により実施しない。

勾配蓄積はoptimizer更新あたりのwindow数を従来のbatch size 8へ合わせるが、BatchNormは各windowを個別に観測する。この差はpaddingを除去するために意図した条件変更であり、旧runとの比較時に明記する。

#### 1 epoch smoke test結果（2026-09-03）

次のrunで実H5を用いた1 epoch学習が完走し、`best.pt`と`last.pt`の保存を確認した。

```text
pointnext_s_EX260830_260711_w16_s8_bboxrankv2_nobboxbg_glocal_ce_smooth00_auto_weight_lr1e3_ep1_bs1_acc8_nopad
```

設定はtrain 164 files / 733 windows、validation 18 files / 79 windows、physical batch size 1、gradient accumulation 8である。class weightは`[0.06282799, 1.93717206]`、label smoothingは0.0で、旧batch size 8 runと同じsplit、seed、window設定、初期checkpointを使用している。

| 検査項目 | train | validation | 判定 |
| --- | ---: | ---: | --- |
| microbatch count | 733 | 79 | sample数と一致 |
| optimizer step count | 92 | 0 | `ceil(733 / 8)=92`と一致 |
| padding point count | 0 | 0 | 合格 |
| padding ratio | 0.0 | 0.0 | 合格 |
| empty-valid window | 0 | 0 | 合格 |
| ignore ratio | 0.2444% | 0.2393% | 整合 |
| confusion matrix合計 | 107,244,397 | 11,170,491 | valid point countと完全一致 |
| weighted loss normalizer | 9,593,603.68 | 967,459.11 | class countとweightからの再計算値と丸め誤差内で一致 |

1 epoch終了時のtrain loss/F1/IoUは`0.6066 / 0.0381 / 0.0194`、validationは`0.6047 / 0.0073 / 0.0037`だった。trainではGT positive率1.42%に対してpredicted positive率13.67%、validationではGT positive率1.27%に対してpredicted positive率0.395%だった。ただしtrain metricsは、学習中に変化するモデルをtrain modeで順次集計した値であり、validation metricsはepoch末のモデルをeval modeで集計した値である。したがって、この差だけから汎化性能や不具合を判断しない。

**判断:** padding-free学習経路、point-weighted gradient accumulation、metrics、checkpoint保存は正常に動作している。旧batch size 8 runとはBatchNormの観測単位とepoch lossの集約方法も異なるため、最終比較ではF1、IoU、FP/FN、推論PLYを主指標とする。teacher v2での長期学習の要否は、利用する最終教師データを確定してから判断する。

#### Teacher v6への移行（2026-09-08）

Stage 4でCVAT full-video修正と削除済みXML無効化が完了し、次のteacher v6がStage 5学習入力の正本として受入済みとなった。

```text
/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/
  global_local_l75_w31_c12_area15_bboxrank_v6_cvat_authoritative_xml_invalidation_v1/
    collected/
```

v6は181動画で構成される。59動画・3014 Task frameではCVAT snapshotをpositive authorityとしてラベルを再構築し、snapshot非対象frameではsource teacherを継承する。さらに、削除済み誤BBox XMLに対応する2動画・7 frameを全backgroundかつ全validへ変更し、positive 2124点とBBox 7行を除去している。crop不良の1動画は含まない。

teacher v2の1 epoch結果はpadding-free実装の動作証拠として保持するが、精度baselineには使用しない。padding影響は9.2のlogits/gradient/BatchNorm parity testで直接立証済みであるため、teacher v2の200 epoch学習は省略し、v6で1 epoch smoke testから再開する。

Stage5の実行bashは次のv6既定値へ更新した。

- `train_stage5.sh`: v6 `collected/`、181 H5、v6 provenance preflight
- `infer_stage5.sh`: v6 H5 suffixとv6 padding-free run
- `evaluate_stage5.sh`: v6 run、v6 reference PLY root、200 epoch checkpoint群
- `export_anonymized_stage5_metrics.sh`: v6 run
- `Stage2to4/pseudo3d/pipelines/export_stage4_bbox_ranked_pointcloud_visualizations.sh`: v6 H5からraw/annotation PLYを生成

学習preflightではteacher token、label/valid整合、CVAT mask外positiveが0であること、CVAT対象件数、XML無効化件数、manifest同一性、除外動画の不在を確認する。旧teacher v2の「BBoxがないframeは全background」という条件は、BBox外またはno-BBoxのCVAT-authoritative positiveを誤って拒否するため使用しない。

次の実行はv6の1 epoch smoke testとする。

```bash
EX_DATE=260908 EPOCHS=1 \
bash /mnt/data/3d_projects/models/Stage5/train_stage5.sh
```

#### Teacher v6 1 epoch smoke test結果（2026-09-08）

次のrunでteacher v6のprovenance preflightと実H5による1 epoch学習が完走し、`best.pt`と`last.pt`の保存を確認した。

```text
pointnext_s_EX260908_260711_w16_s8_bboxrankv6_cvatxmlinv_glocal_ce_smooth00_auto_weight_lr1e3_ep1_bs1_acc8_nopad
```

設定はtrain 163 files / 729 windows、validation 18 files / 79 windows、physical batch size 1、gradient accumulation 8である。class weightは`[0.05963856, 1.94036150]`、label smoothingは0.0で、PointNeXt-Sの構成、初期checkpoint、window設定、optimizer設定はteacher v2 smoke testと同じである。

| 検査項目 | train | validation | 判定 |
| --- | ---: | ---: | --- |
| microbatch count | 729 | 79 | sample数と一致 |
| optimizer step count | 92 | 0 | `ceil(729 / 8)=92`と一致 |
| padding point count | 0 | 0 | 合格 |
| padding ratio | 0.0 | 0.0 | 合格 |
| empty-valid window | 0 | 0 | 合格 |
| ignore ratio | 0.4389% | 0.3240% | 集計値と一致 |
| confusion matrix合計 | 106,464,696 | 11,126,891 | valid point countと完全一致 |
| weighted loss normalizer | 8,810,516.08 | 905,706.49 | label countとweightからの再計算値とfloat丸め範囲内で一致 |

`valid + ignore = total`、`predicted positive + predicted background = valid`、`unpadded = total`もtrain/validationの双方で完全一致した。trainの最後の1 windowを含む端数accumulationもoptimizer stepへ反映されている。

1 epoch終了時のtrain loss/F1/IoUは`0.5903 / 0.0361 / 0.0184`、validationは`0.6183 / 0.0040 / 0.0020`だった。trainではGT positive率1.23%に対してpredicted positive率12.45%、validationではGT positive率1.16%に対してpredicted positive率0.247%であり、初期epochではvalidation予測がbackgroundへ強く偏っている。ただし、train metricsは学習中のモデルをtrain modeで集計し、validation metricsはepoch末のモデルをeval modeで集計した値であるため、この差は実装不具合の証拠ではなく、1 epoch時点の精度評価にも使用しない。

teacher v2 smoke testと比べ、class weight算出元のtrain H5ではbackgroundが322,889点（0.52%）、positiveが109,988点（13.68%）減少した。これはCVAT authoritative修正、XML無効化、crop不良動画の除外と整合する方向だが、入力ファイル数が182から181へ変わるとseed 42のindex shuffleによるsplit membershipも変化し得る。そのため、v2/v6のloss、F1、FP/FN差をteacher変更だけの効果として解釈しない。

**判断:** teacher v6の入力検証、padding-free DataLoader経路、point-weighted gradient accumulation、loss/metrics集計、checkpoint保存は正常に動作している。重大な構造的不整合、NaN、空window、padding混入は認められない。次は同じv6設定で5～10 epochの短期pilotを行い、loss、positive予測率、F1/IoUの推移と固定train sanity/validation評価を確認してから200 epoch runへ進む。

#### Teacher v6 5 epoch pilot結果（2026-09-08）

同じ設定を5 epochまで学習した次のrunを解析した。

```text
pointnext_s_EX260908_260711_w16_s8_bboxrankv6_cvatxmlinv_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad
```

| epoch | train loss | train F1 | train IoU | val loss | val F1 | val IoU | val predicted positive率 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5909 | 0.0344 | 0.0175 | 0.5985 | 0.0172 | 0.0087 | 0.347% |
| 2 | 0.5150 | 0.0637 | 0.0329 | 0.6711 | 0.0275 | 0.0139 | 1.283% |
| 3 | 0.4772 | 0.0782 | 0.0407 | 0.7027 | 0.0280 | 0.0142 | 4.453% |
| 4 | 0.4445 | 0.0862 | 0.0451 | 0.7583 | **0.0455** | **0.0233** | 3.183% |
| 5 | 0.4101 | **0.1004** | **0.0529** | 0.9208 | 0.0394 | 0.0201 | 2.698% |

train lossは全epochで低下し、F1、IoU、recallも概ね上昇した。trainのGT positiveに対する平均positive確率は0.403から0.516へ上昇し、GT backgroundでは0.343から0.191へ低下したため、train上ではpositive/backgroundの分離が進んでいる。一方、train predicted positive率はepoch 5でも15.11%であり、GT positive率1.23%に対して大幅に高く、FPは15,213,169点残っている。

validation lossは0.5985から0.9208へ全epochで増加した。validationのGT positiveに対する平均positive確率が0.237から0.111へ低下し、recallもepoch 4の8.54%を頂点にepoch 5で6.57%へ低下しているため、loss増加は集計不具合ではなく、positiveに対する確信度低下を反映している。GT backgroundの平均positive確率も0.198から0.066へ低下しており、モデル全体がvalidationではbackground側へ強く移動している。

auto class weightのpositive/background比は約32.54倍である。validationのweighted CE分母では、全点の1.16%しかないpositiveが約27.58%を占める。このため、background確率の改善と同時にpositive確率が低下すると、閾値後F1が一時改善していてもvalidation lossは増加し得る。今回のlossとIoUの不一致はこの性質で説明可能である。

全5 epochでpadding点0、empty-valid window 0、microbatch 729、optimizer step 92を維持した。confusion matrix、予測class count、valid/ignore/total、およびweighted loss normalizerも全epochで再計算値と一致した。したがって、5 epoch pilotから新たなデータ経路・勾配蓄積・metrics集計の不具合は検出されなかった。

`best_metric=iou_femur`であるため、このrunの`best.pt`はvalidation IoUが最大のepoch 4、`last.pt`はepoch 5に対応する。なお、train metricsはepoch中に更新されるモデルを`train()`モードで集計し、validation metricsはepoch末のモデルを`eval()`モードで集計している。PointNeXt-SはBatchNormを使用するため、この表だけでは通常の過学習とphysical batch size 1におけるBatchNorm running statisticsの影響を分離できない。

**判断:** 学習自体は進んでいるが、5 epoch以内にtrain/validationの乖離が明確になった。現時点では200 epochへ進まず、`best.pt`と`last.pt`を同じ`eval()`経路で固定train sanity subsetと全validationへ適用する。eval modeでもtrain sanityが良好なら汎化・split・座標依存を主因候補とし、train sanityまでbackgroundへ崩れるならBatchNorm running statisticsまたはtrain/eval条件差を優先して検証する。

#### Teacher v6 5 epoch checkpoint評価（2026-09-08）

`best.pt`（epoch 4）と`last.pt`（epoch 5）を、固定train sanity 1動画、seed固定random train 2動画、validation全18動画へ適用した。いずれも`eval()`かつphysical batch size 1で各windowを推論し、最終H5 metricsでは元H5の同一点に対するpositive確率をwindow間で平均してから閾値判定した。

| checkpoint | train window F1 / IoU | train aggregated F1 / IoU | val window F1 / IoU | val aggregated F1 / IoU |
| --- | ---: | ---: | ---: | ---: |
| best, epoch 4 | 0.1026 / 0.0541 | **0.0304 / 0.0154** | 0.0455 / 0.0233 | **0.0341 / 0.0173** |
| last, epoch 5 | 0.0363 / 0.0185 | **0.0140 / 0.0070** | 0.0394 / 0.0201 | **0.0314 / 0.0160** |

`best.pt`の固定train sanity動画では、aggregated precision 1.65%、recall 13.06%、F1 0.0293、FP 12,944点だった。random train 2動画のうち1動画はTP 0であり、もう1動画のF1も0.0387にとどまった。`last.pt`では固定動画のF1が0.0248へ低下し、random train 2動画はいずれもTP 0だった。したがって、5 epoch時点のdeploy時推論はtrain sanityを十分に再現できておらず、200 epochへ進む根拠にはならない。

validationのaggregated metricsは`best.pt`でprecision 3.04%、recall 3.87%、F1 0.0341、IoU 0.0173だった。18動画中10動画でTPが0で、video-level F1のmedianも0だった。最大F1は0.0960で、F1が正だったのは8動画に限られる。`last.pt`でもTP 0は10/18動画、median F1は0であり、集約IoUは0.0160へ低下した。動画ごとの差は大きく、lastで改善した動画3件、同値9件、悪化6件だった。

window単位では、validationのGT-positive window 54個中36個がTP 0だった。negative-only window 25個のうちFPを含むものはbestで16個、lastで9個だった。単純なall-background崩壊ではなく、検出できるwindowが限定される一方、GT-positiveを含まないwindowにも局所的FPが出ている。

特に重要なのは、mean probability aggregationによる性能低下である。`best.pt`ではtrain sanity F1がwindow集計の0.1026から元点集約後の0.0304へ低下し、recallは14.19%から4.17%へ低下した。validationでもF1は0.0455から0.0341、recallは8.54%から3.87%へ低下した。平均化によりFPRも低下するが、TP/recallの損失が大きい。これは同じ点が異なる時間window文脈で一貫したpositive確率を得ておらず、一方のwindowのpositive予測を他方が打ち消している可能性を示す。

一方、`best.pt`のtrain sanityは同じ`eval()`モードでもwindow単位F1 0.1026を保ち、epoch 4の学習中train F1 0.0862を下回っていない。この結果から、BatchNorm running statisticsによる全面的なeval-mode崩壊は現時点の第一候補ではない。ただし対象はtrain 163動画中3動画のみであり、BatchNormの影響を棄却するには同一windowのtrain/eval parity testが別途必要である。

ignore点へのaggregated positive予測は、bestでtrain 2/4,966点、validation 520/19,050点だった。valid background上のFPはそれぞれ16,602点、85,785点であり、現時点のFPはignore領域だけでは説明できない。

**判断:** `best.pt`は`last.pt`より集約F1/IoUが両splitで高いため、後続診断ではbestを主対象とする。次は再学習ではなく、元点ごとのwindow vote数、positive確率のmin/max/mean/std、window間class不一致率をGT class別に記録し、mean、max、時間中心優先または中心重み付きaggregationを同一推論結果で比較する。これにより、モデル自体の検出不足とaggregationによるpositive消失を分離する。その後に同一windowのBatchNorm train/eval parityを確認する。

### 9.5 Overlap lossの補正

まず再学習なしで、同一点に対するwindow間予測のばらつきとclass不一致率を計測し、現行mean probabilityを対照としてmax、時間中心優先、中心重み付きaggregationを比較する。その後、推論側で良好な方式と整合するcenter-only lossまたは各点のwindow出現回数の逆数によるloss weightingを比較する。

#### 実装事項A: overlap probability / aggregation checker 検証結果（2026-09-09）

実施日: 2026-09-09。対象run/checkpoint:
`pointnext_s_EX260908_260711_w16_s8_bboxrankv6_cvatxmlinv_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad`
の`best.pt`（epoch 4）。

**checkerの完走状態:** Step A1（synthetic accumulator test、8ケース）、Step A2（train sanity 3件
smoke run + 固定train sanity 1件でのPLY 8ファイルsmoke test）、Step A3（full run: train sanity 3件
+ validation 18件、計21動画）をいずれもユーザー実機で完走した。`check_stage5_overlap_aggregation.py`の
anonymization self-check（`assert_share_bundle_anonymous`）も合格した。

**mean baseline parity:** 合格。checkerが独立に計算したmean_probability方式のTP/FP/TN/FNが、
既存`evaluate_stage5.py`の記録値と完全一致した。

| 対象 | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: |
| train sanity（3動画合計） | 407 | 16602 | 600976 | 9349 |
| validation（18動画合計） | 2691 | 85785 | 6421265 | 66760 |
| 固定train sanity動画のみ | 217 | 12944 | — | — |

固定train sanity動画のF1（0.0293）も、9.4節の5 epoch checkpoint評価で記録済みの値
（「aggregated precision 1.65%, recall 13.06%, F1 0.0293」）と一致した。

**事項1の結果（重複点のwindow間確率分布）:** `max_probability > 0.5`かつ`mean_probability <= 0.5`と
なるsuppressed-positive（mean集約で見かけ上backgroundへ落ちる点）は次の通りだった。

| split | 対象 | 点数 | suppressed-positive数 | 割合 |
| --- | --- | ---: | ---: | ---: |
| train_sanity | GT positive点のみ | 9756 | 2101 | 21.54% |
| train_sanity | 全valid点 | 627334 | 13347 | 2.13% |
| validation | GT positive点のみ | 69451 | 7331 | 10.56% |
| validation | 全valid点 | 6576501 | 250118 | 3.80% |

GT positive点に限ると、train sanityで5点に1点以上、validationでも10点に1点が、いずれかのwindowで
positive確率0.5超と判定されながらmean集約でbackground側へ転じている。

**事項2の結果（window間class不一致）:** video別のdisagreement率（vote count 2以上の点のうち、
window間でpositive/backgroundが混在する点の割合）はvideoによって大きくばらついた。
GT positive点でのdisagreement率は、train_sanity_random_001の0.21%からvalidation_017の79.66%まで
分布し、disagreement点の大半（train_sanity_fixed_001で738点中674点、validation_017で1187点中
894点など）がmean集約でbackground判定になっていた。

`min_edge_distance`（window境界からの距離、0〜3）別のdisagreement率は、全split・全video合算で
次の通りほぼ一定だった。

| min_edge_distance | 点数（vote 2以上） | disagreement数 | rate |
| ---: | ---: | ---: | ---: |
| 0 | 1,206,258 | 71,271 | 5.91% |
| 1 | 1,203,828 | 71,858 | 5.97% |
| 2 | 1,198,948 | 71,136 | 5.93% |
| 3 | 1,198,589 | 70,753 | 5.90% |

window境界に近い点ほど不一致が増えるという仮定は、このデータでは支持されなかった
（0.059〜0.060の範囲でほぼ横ばい）。一方、動画内相対位置（decile 0〜9）別では次の通り、
前半（decile 1）の1.99%から後半（decile 6、9）の7%前後まで緩やかな上昇傾向が見られた。

| decile | 点数（vote 2以上） | disagreement数 | rate |
| ---: | ---: | ---: | ---: |
| 0 | 37,242 | 0 | 0.00% |
| 1 | 233,399 | 4,636 | 1.99% |
| 2 | 433,275 | 14,907 | 3.44% |
| 3 | 565,741 | 37,175 | 6.57% |
| 4 | 736,813 | 40,188 | 5.45% |
| 5 | 725,617 | 45,747 | 6.30% |
| 6 | 744,941 | 52,248 | 7.01% |
| 7 | 683,512 | 46,691 | 6.83% |
| 8 | 451,578 | 29,690 | 6.57% |
| 9 | 195,505 | 13,736 | 7.03% |

decile 0の0%は点数が少なく（37,242点、他decileの1/6〜1/20）、単独では解釈しない。

**事項3の比較表（4 aggregation方式、split集約）:**

| split | method | F1 | IoU | recall | FP | TP0動画数 | video_mean_F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train_sanity | mean | 0.0304 | 0.0154 | 4.17% | 16,602 | 1/3 | 0.0227 |
| train_sanity | max | 0.1250 | 0.0667 | 25.71% | 27,848 | 0/3 | 0.1037 |
| train_sanity | center_nearest | 0.1130 | 0.0599 | 18.94% | 21,111 | 1/3 | 0.0876 |
| train_sanity | center_weighted | 0.0510 | 0.0262 | 7.34% | 17,589 | 1/3 | 0.0364 |
| validation | mean | 0.0341 | 0.0173 | 3.87% | 85,785 | 10/18 | 0.0198 |
| validation | max | 0.0491 | 0.0252 | 14.43% | 328,572 | 7/18 | 0.0423 |
| validation | center_nearest | 0.0426 | 0.0217 | 8.75% | 210,119 | 8/18 | 0.0357 |
| validation | center_weighted | 0.0367 | 0.0187 | 5.03% | 117,206 | 11/18 | 0.0244 |

`max_probability`はrecall/F1/IoUをmean比で最大化するが、FPがtrain sanityで+11,246点、
validationで+242,787点増加する（4.3節の設計通り、これは診断用でありFP急増のため採用候補には
しない）。`center_nearest`と`center_weighted`はmean比でF1/IoUを改善するが、validationの
`center_weighted`はTP0動画数がmean（10/18）よりも悪化した（11/18）。split集約IoUの改善と
video単位のTP0動画数悪化が同時に起きており、集約指標だけでは方式の優劣を判断できないことを
示している。

**仮説に対する暫定判断:**

- 支持: mean probability集約によるpositive予測の打ち消しは、suppressed-positive数
  （GT positive点の21.5%/10.6%）とGT-positive点のdisagreement率（video間で0.2%〜79.7%と
  大きくばらつく）の両方から定量的に確認された。これは9.4節で観測した
  「window集計 recall > mean集約後 recall」という既存の疑いと整合する。
- 棄却（暫定）: 「window境界に近いほど不一致が増える」という、`center_weighted`/`center_nearest`
  の設計根拠となっていた仮説は、`min_edge_distance`別disagreement率がほぼ一定だったことから
  このデータでは支持されない。動画内相対位置（時間経過）とはある程度相関があり、原因は
  window境界そのものではなく、時間的な文脈（BBox遷移や見え方の変化など）にある可能性を示唆する。
- 未検証: BatchNorm running statistics由来のtrain/eval崩壊仮説は、事項A単体では検証していない
  （事項Bで別途検証する）。

**方針管理チャットへ返す判断事項（未決定、実装チャットでは決定しない）:**

1. production aggregation（`evaluate_stage5.py`/`infer_stage5.py`の既定mean probability）を
   変更するか、変更する場合はどの方式を採用するか。
2. `center_weighted`の重み式（`weight = 1 + min_edge_distance`）は、edge_distanceとdisagreement率の
   相関が薄いという今回の結果を踏まえて再設計するか。
3. 事項B（BatchNorm train/eval parity）に進むか。
4. Label policy ablation（9.6節）やclass weight比較（9.7節）を先に行うか。

再学習、loss変更、threshold tuning、production aggregationの変更は、本節の記録時点では実施していない。

方針管理チャットの判断（2026-09-10）: production aggregationは変更せず`mean_probability`を維持する。
`center_weighted`の重み式も再設計しない（境界距離相関が無いため根拠がない）。次は実装事項Bへ進み、
その後Label policy ablation（9.6節）→ class weight比較（9.7節）の順とする。200 epoch学習はまだ
行わない。

#### 実装事項B: BatchNorm train/eval parity 検証結果（2026-09-10）

実施日: 2026-09-10。対象run/checkpointは実装事項Aと同じ`best.pt`（epoch 4）。

**checkerの完走状態:** synthetic self-test（5ケース）、train sanity 3件のみのsmoke run、
full run（train sanity 3件+validation 18件、計21動画）をいずれもユーザー実機で完走した。
anonymization self-checkも合格した。

**mean baseline parity（eval-mode instanceの健全性確認）:** 合格。同じ`best.pt`から構築した
eval-mode instance（A）でmean集約した結果のTP/FP/TN/FNが、既存`evaluate_stage5.py`の記録値
（本レポート2.5節・9.5節事項Aの数値）と完全一致した。これにより、以下のtrain-mode（B）との
比較が正しくproductionのeval経路を基準にしていることを確認できた。

**記録内容（6.2節）:**

BatchNorm module 17個全てで`running_mean`/`running_var`はfinite、負のvarianceも無く、
`num_batches_tracked`は全module一致（16859）だった。checkpointに保存されたBatchNorm統計量
自体に異常値は見られない。

同一window・同一点順序・同一RNG seedで、eval-mode instance（A）とtrain-mode instance（B、
各windowのbatch statisticsを使用）を比較した結果は次の通り。

| split | GT区分 | occurrence数 | disagreement率 | mean abs diff |
| --- | --- | ---: | ---: | ---: |
| train_sanity | 全valid点 | 872,417 | 15.39% | 0.143 |
| train_sanity | GT positive点のみ | 17,726 | 51.16% | 0.285 |
| validation | 全valid点 | 11,163,057 | 16.20% | 0.156 |
| validation | GT positive点のみ | 128,735 | 38.15% | 0.260 |

window単位・mean集約後のF1/IoU/recallは次の通り、eval-modeとtrain-modeで大きく乖離した。

| split | granularity | mode | TP | FP | recall | F1 | IoU | TP0動画数 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train_sanity | window | eval | 2,516 | 28,784 | 14.19% | 0.1026 | 0.0541 | — |
| train_sanity | window | train | 11,259 | 128,376 | 63.52% | 0.1431 | 0.0771 | — |
| train_sanity | aggregated | eval | 407 | 16,602 | 4.17% | 0.0304 | 0.0154 | 1/3 |
| train_sanity | aggregated | train | 5,952 | 84,789 | 61.01% | 0.1185 | 0.0630 | 0/3 |
| validation | window | eval | 10,991 | 343,123 | 8.54% | 0.0455 | 0.0233 | — |
| validation | window | train | 51,674 | 1,707,227 | 40.14% | 0.0547 | 0.0281 | — |
| validation | aggregated | eval | 2,691 | 85,785 | 3.87% | 0.0341 | 0.0173 | 10/18 |
| validation | aggregated | train | 23,001 | 757,945 | 33.12% | 0.0541 | 0.0278 | 0/18 |

validationのwindow単位eval-mode recall（8.54%）は9.4節の記録値と一致し、mean baseline parityの
数値的裏付けにもなっている。

train-modeへ切り替えるとFPも大幅に増える（validation aggregated FPはeval比+672,160点）ため、
train-modeのBatchNorm統計量をそのまま採用すればよいという意味ではない。ここで観測しているのは
「production（eval-mode、保存済みrunning statistics）が、同じ重みで各windowのbatch statisticsを
使った場合と比べて著しく性能が低い」という差そのものである。

**判定（6.3節）:** 「train-statistics側だけ大幅に良い」に該当する。validation
（学習に使っていない18動画）でもaggregated recallが3.87%→33.12%、TP0動画数が10/18→0/18、
video-level median F1が0→0.047へ改善しており、train sanityだけの現象ではなく汎化データでも
再現することを確認した。BatchNorm running statisticsまたはphysical batch size 1学習の影響を
支持する結果であり、9.2節のpadding parity testで確認した「physical batch size 1がBatchNorm
統計へ与える影響」という既存の懸念と整合する。

**方針管理チャットへ返す判断事項（未決定、実装チャットでは決定しない）:**

1. BatchNorm対策（running statistics recalibration、freeze、別normへの変更など）を
   Label policy ablationより先に検討するか（6.3節の分岐および方針管理チャットの
   既定方針「train/eval差が大きい場合はLabel policy実験前にBatchNorm対策を検討する」に該当）。
2. 対策を検討する場合、6.2節に規定されたBN recalibration診断（train H5をforwardしてrunning
   statisticsを再計算したコピーを評価する）に進むか。
3. train-modeの統計量をそのまま採用することは、FPも大幅に増える（validation aggregated FPは
   eval比+672,160点）ため推奨しない。対策はrecalibrationや別のbatch正規化戦略を含めて
   別途比較検討が必要。

再学習、loss変更、threshold tuning、production aggregationおよびBatchNorm設定の変更は、
本節の記録時点では実施していない。

#### 実装事項C: BatchNorm recalibration診断 検証結果（2026-09-12）

実施日: 2026-09-12。対象run/checkpointは実装事項A/Bと同じ`best.pt`（epoch 4）。

**checkerの完走状態:** synthetic self-test（8ケース、requirement 1〜3の`num_batches_tracked`一致・
finite性・非負varianceのfail-fast assertionを含む）、train sanity 3件のみのsmoke run
（calibrationは163動画全件を実施したうえで評価をtrain sanityへ限定）、full run（train sanity 3件+
validation 18件、計21動画）をいずれもユーザー実機で完走した。anonymization self-checkも
（実装チャット側の独立`grep`確認を含め）合格した。

**calibration規模:** `train_files.txt`から163 files / 729 windowsを処理した（対象runの期待値と一致）。
augmentationなし、label/validationはcalibrationに使用していない。

**parameter/非BN buffer不変性:** 合格。`model_recalibrated`のparameter hashと非BN buffer hashは
recalibration前後で完全一致した。

**mean baseline parity（original evalの健全性確認）:** 合格。`model_original`のeval-mode・mean集約
結果のTP/FP/TN/FNが、既存`evaluate_stage5.py`の記録値（train_sanity: TP407/FP16602/TN600976/FN9349、
validation: TP2691/FP85785/TN6421265/FN66760）と完全一致した。

**BatchNorm module健全性（17 module全て）:** 全moduleで`recalibrated_num_batches_tracked=729`となり、
calibration processed windows（729）と完全一致した（本チャットで追加したrequirement 1のfail-fast
assertionが実データでも成立することを確認）。`running_mean`/`running_var`は全moduleでfinite、
負のvarianceも0件だった（requirement 2・3）。一方、running statistics自体の絶対的な変化量は大きく、
`running_var_abs_diff_max`は`pointnext.decoder.decoder.3.0.convs.0.1`（256ch）で約469.9、
`running_mean_abs_diff_max`で約12.3に達した。`original_num_batches_tracked`は全module16859で
事項Bの記録と一致する。

**probability差（original eval vs recalibrated eval、GT区分別）:**

| split | GT区分 | occurrence数 | disagreement率 | mean abs diff |
| --- | --- | ---: | ---: | ---: |
| train_sanity | 全valid点 | 863,015 | 2.44%（全点ベース2.69%） | 0.045〜0.046 |
| train_sanity | GT positive点のみ | 17,726 | 13.27% | 0.105 |
| validation | 全valid点 | 11,126,891 | 2.63%（全点ベース2.66%） | 0.045 |
| validation | GT positive点のみ | 128,735 | 4.84% | 0.060 |

事項B（eval/train-mode parity）のdisagreement率（train_sanity GT positive 51.16%、validation GT
positive 38.15%）と比べ、recalibrationによるoriginal-vs-recalibrated差はGT positive点で約1/4〜1/8
に留まる。running statistics自体の変化量（上記buffer差）は小さくないため、この結果は
「保存統計のstalenessだけでは事項Bで観測した規模の差を再現しない」ことを示唆する。

**aggregated（mean probability集約後）metrics:**

| split | model | TP | FP | recall | F1 | IoU | TP0動画数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train_sanity | original | 407 | 16,602 | 4.17% | 0.0304 | 0.0154 | 1/3 |
| train_sanity | recalibrated | 298 | 13,331 | 3.05% | 0.0255 | 0.0129 | 2/3 |
| validation | original | 2,691 | 85,785 | 3.87% | 0.0341 | 0.0173 | 10/18 |
| validation | recalibrated | 2,819 | 73,374 | 4.06% | 0.0387 | 0.0197 | 10/18 |

**window-granularity（集約前）metrics:**

| split | model | TP | FP | recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| train_sanity | original | 2,516 | 28,784 | 14.19% | 0.1026 |
| train_sanity | recalibrated | 1,252 | 23,541 | 7.06% | 0.0589 |
| validation | original | 10,991 | 343,123 | 8.54% | 0.0455 |
| validation | recalibrated | 9,653 | 290,511 | 7.50% | 0.0450 |

事項Bのtrain-mode推論（validation aggregated recall 3.87%→33.12%、window recall 8.54%→40.14%、
いずれもFPが大幅増加）とは対照的に、recalibrationはrecallとFPがおおむね同方向（両方微減または
横ばい）に動き、train-modeのような大幅なrecall/FP同時急増は再現しなかった。

**video単位のTP0反転:** validationのTP0動画数はoriginal・recalibratedとも10/18で変わらないが、
構成が入れ替わった。`validation_004`はTP 59→0（悪化）、`validation_016`はTP 0→31（改善）。
train_sanityでは`train_sanity_random_002`がTP 190→0（悪化）となり、train_sanityのTP0動画数は
1/3→2/3へ悪化した。個別動画では`validation_011`（F1 0.0115→0.2584）や`validation_013`
（F1 0.0009→0.0641）のような大きな改善も、`validation_005`（F1 0.0808→0.0379）のような悪化も
双方観測され、split集約の小さな正味変化の内側で動画ごとの符号が一致していない。

**仮説に対する暫定判断:**

- 支持: BN module 17個全てで`num_batches_tracked`が期待forward回数（729）と完全一致し、
  running buffer（`running_mean`/`running_var`）はfiniteかつ非負varianceを維持したまま、
  実際に大きく変化した（`running_var_abs_diff_max`最大約469.9）。recalibration自体は
  仕様通りに機能している。
- 支持（暫定）: 保存済みrunning statisticsのstalenessは、事項Bで観測した規模のeval/train乖離を
  単独では説明しない。running statisticsを大きく動かしても、original-vs-recalibrated
  disagreement率（GT positiveでtrain_sanity 13.27%、validation 4.84%）は事項Bのeval/train
  disagreement率（同51.16%/38.15%）よりはるかに小さく、aggregated F1/recallの変化もvalidationで
  ±0.5ポイント程度、train_sanityではむしろ悪化した。
- 未解決: validationのTP0動画数は10/18で変化せず、video単位では改善・悪化が両方向に生じている。
  これはaggregated指標の小さな正味変化が、モデルの系統的な改善ではなく、動画ごとに異なる方向へ
  ずれるnoise的な再配分である可能性を示す。この点はcheckerの範囲では切り分けられない。
- 保留: 事項Bのtrain-mode推論で観測された大幅なrecall/FP同時増加は、recalibrationでは再現
  されなかった。したがって、事項Bの効果の主要因は保存統計のstalenessよりも、
  各評価windowが自身のbatch statisticsを使うtest-time adaptation効果（または
  physical batch size 1のnormalization方式そのもの）にある可能性が、今回のデータでは
  相対的に高まった。ただし本checkerはこの2つの効果を直接分離する設計ではないため、断定はしない。

**方針管理チャットへ返す判断事項（未決定、実装チャットでは決定しない）:**

1. BatchNorm対策（S5-11）として、running statistics recalibrationを主要候補から外し、
   per-window statisticsを使う正規化（test-time batch normalization相当）や
   GroupNorm/LayerNorm等への置換を優先候補とするか。
2. train_sanityで悪化が観測された点（TP0動画数1/3→2/3、window F1 0.1026→0.0589）を、
   recalibration自体の欠陥ではなくtrain_sanity 3動画という小標本のnoiseとして扱うか、
   追加検証（video数を増やす、複数seedでのsensitivity case）が必要と判断するか。
3. Label policy ablation（9.6節）・class weight比較（9.7節）へ進むか、S5-11のBatchNorm対策比較を
   先に行うか。

再学習、loss変更、threshold tuning、production aggregationおよびBatchNorm設定の変更は、
本節の記録時点でも実施していない。recalibrated checkpointはユーザー実機の`stage5_debug/`配下に
diagnostic artifactとして保存されており、productionへは接続していない。

#### 実装事項D: GroupNormによるnormalization比較 実装計画（2026-09-12）

**状態:** planned。S5-11として実装・短期比較を行う。production設定は比較結果が出るまで変更しない。

**方針管理チャットの判断（2026-09-12）:** running statistics recalibrationは主要候補から外し、
diagnostic artifactのままproductionへ採用しない。事項Cで観測されたtrain sanityの悪化は、
小標本noiseと断定はしないが、系統的改善を示さない不安定な変化として扱い、S5-10の追加video・
複数seed検証は行わない。Label policy ablationとclass weight比較へ進む前に、physical batch size 1でも
train/evalで同じ挙動となるnormalizationをS5-11で比較する。第一候補はGroupNorm、LayerNormは
GroupNorm不採用時の次候補とする。per-window BatchNormは事項Bの診断結果をreferenceとして残すが、
recallとFPが同時に大幅増加したため、そのままproduction候補にはしない。BN freezeとrecalibrated
checkpoint運用は低優先度とする。

**目的:** 現行PointNeXt-SのBatchNormだけをGroupNormへ置き換え、physical batch size 1、可変点数window、
train/eval modeの条件に依存しない正規化へ変更した場合の学習安定性と固定評価性能を測る。教師label、
class weight、loss、threshold、aggregation、splitなどは固定し、normalization方式の効果だけを
切り分ける。

**wrapper / CLI実装方針:** `pointnext_s_segmentor.py`とtraining/inference/evaluationのmodel config経路へ
次を追加する。

```text
--pointnext_norm batchnorm   # 既存既定値
--pointnext_norm groupnorm
--pointnext_norm_groups 8
```

既存checkpointにこれらのfieldがない場合は`batchnorm`として解釈し、既存modelの構造、state dict、
forward結果との後方互換性を維持する。normalization指定はencoder、decoder、segmentation headの全てへ
明示的に渡し、decoderのOpenPoints既定BatchNormを残さない。

OpenPointsの`create_norm()`は文字列`gn`をそのまま使用するとdimension suffixと`nn.GroupNorm`の
constructor形式が一致しない。このため、external PointNeXt cloneは変更せず、Stage 5 wrapper側へ
`num_channels`と`num_groups`を受け取るGroupNorm adapterを置き、OpenPointsの`norm_args`へcallableとして
渡す。adapterは1D/2D feature tensorの両方で`nn.GroupNorm`を使用する。channel数がgroup数で割り切れない
場合は暗黙にgroup数を変更せず、model構築時に明示的に停止する。現行width 32構成の初期値は8 groupsと
する。

**初期parameterの統一:** GroupNorm版をscratchから開始すると、normalizationと事前学習の2要因が
同時に変わるため採用しない。現行teacher v6 BatchNorm controlと同じStage 5用S3DIS部分転移checkpointを
初期値として使用し、convolution/linear parameter、入力層、binary head、shapeが一致するnorm affine
`weight`/`bias`をGroupNormへ読み込む。BatchNorm固有の`running_mean`、`running_var`、
`num_batches_tracked`だけを除外する。ロード結果をartifactへ記録し、BN running buffer以外のmissing、
unexpected、shape mismatchがあればfail-fastとする。単なる`strict=False`による無言の読み飛ばしには
しない。

**固定比較条件:** 次の条件はteacher v6 5 epoch padding-free pilotから変更しない。

| 項目 | 固定値 |
| --- | --- |
| teacher | Stage 4 teacher v6 |
| split | train 163 files / validation 18 files（同一file list） |
| seed | 42 |
| window | size 16 / stride 8 / tailあり |
| physical batch | 1 window |
| gradient accumulation | 8 windows、point-weighted normalization |
| features | `intensity,confidence` |
| loss | CrossEntropyLoss |
| class weight | `auto` |
| label smoothing | 0.0 |
| optimizer | AdamW、lr `1e-3`、weight decay `1e-4` |
| dropout | 0.0 |
| inference mode | `model.eval()` |
| aggregation | mean probability |
| class threshold | 0.5 |

既存BatchNorm v6 5 epoch pilotをcontrolとする。共通model構築経路を変更した場合は、旧checkpointを
BatchNorm既定値でstrict reloadできることと、同一入力・seedのforward parityを先に確認する。
GroupNorm比較のためにlabel policy、class weight、threshold、augmentation、座標feature、window、
aggregationを同時に変更しない。

**実装・検証手順:** 次の順序で進める。

1. 既存BatchNorm後方互換testを追加し、既存checkpointのstrict loadとforward parityを確認する。
2. GroupNorm modelを構築し、全normalization siteが置換され、`_BatchNorm`が0 moduleであること、
   GroupNorm module数が期待値と一致することを検査する。
3. dropout 0.0、同一入力・同一seedでGroupNorm modelの`train()`/`eval()` logitsが許容誤差内で一致する
   ことを確認する。
4. dummy window batchでforward、CE loss、backward、optimizer stepを行い、finite性を確認する。
5. checkpoint保存後のstrict reloadとlogits parityを確認する。
6. S3DIS部分転移parameterのロードreportを出力し、許可したBN buffer以外の欠落がないことを確認する。
7. 実H5による1 epoch smoke testを行い、loss/gradient/checkpoint/metricsがfiniteで完走することを確認する。
8. 同じsplitとseedで5 epoch GroupNorm pilotを実行する。
9. `best.pt`とepoch 5の固定checkpointを、既存BatchNorm controlと同じtrain sanity 3動画・validation
   18動画へ適用する。

**評価項目:** epoch中のtrain metricsだけで結論を出さず、保存checkpointを全て同じ`eval()`・現行mean
aggregation経路で比較する。最低限、TP、FP、TN、FN、precision、recall、F1、femur IoU、FPR、FNR、
predicted positive率、TP0動画数、video-level mean/median F1・IoU、ignore領域上のpositive予測率を
記録する。train/eval logits parity、normalization module count、checkpoint transfer reportも
構造上の受入条件として保存する。

**判定基準:** GroupNormはtrain/eval一致を満たすだけでは採用しない。validationの集約F1だけの微増では
なく、TP0動画数またはvideo-level median F1/IoU、recallの改善が確認でき、FP/FPRとpredicted positive率が
許容不能に増えず、train sanityが崩壊しないことを重視する。split集約値とvideo別変化の符号が一致しない
場合は、事項Cと同様に系統的改善とは判定しない。

**結果による分岐:** GroupNormが構造testを満たし、train sanityとvalidationの双方で有望なら、
normalization候補として暫定採用し、同方式を固定して9.6節のLabel policy ablationへ進む。train/eval差は
解消するが精度が悪化する場合はGroupNormを不採用とし、同じ固定条件でLayerNormを次候補として検討する。
GroupNormとBatchNormの差が小さい場合はnormalizationを主要因から下げ、既定方式を変更せずLabel policy
ablationへ進む。per-window BatchNorm、BN freeze、recalibration checkpointを追加比較するのは、
GroupNorm/LayerNormの結果だけでは判断できない場合に限定する。

**非対象:** 本事項では200 epoch学習、Dice/Focal/Hard Negative Mining、class weight変更、threshold
tuning、production aggregation変更、label policy変更、座標augmentation、Stage 4 teacher変更を行わない。
実装チャットはcheckerの完走と仮説判断を分けて報告し、production採否を独断で確定しない。

**実装進捗（2026-09-12）:** wrapper/CLI/checkpoint config経路とGroupNorm adapterのコード実装、および
GPU非依存の静的・synthetic検証まで完了した。実行（Step D3以降のGPU実行、Step D6/D7）は未着手。

- `Stage5/stage5/models/norm_layers.py`（新規）: `Stage5GroupNorm`（`nn.GroupNorm`を直接継承。
  `.weight`/`.bias`がBatchNormと同じdotted keyへ載るため、既存BatchNorm checkpointからのnorm affine
  重み転移がkey remapなしで成立する）と`resolve_pointnext_norm_args()`を実装した。OpenPoints
  `create_norm()`は`norm_args["norm"]`が文字列以外（callable）の場合、dimension接尾辞処理を経ず
  `norm(channels, **残りkwargs)`を直接呼ぶ仕様であるため、文字列`"gn"`のconstructor不一致
  （`nn.GroupNorm`の第1引数は`num_groups`）を回避できることをコード監査で確認済み。
  `pointnext_norm="batchnorm"`（既定）は従来通り`{"norm": "bn"}`をそのまま返し、既存コードパスを
  変更しない。
- `Stage5/stage5/models/pointnext_s_segmentor.py`: `build_pointnext_s_config()`/
  `PointNeXtSSegmentor`/`pointnext_s()`へ`pointnext_norm`/`pointnext_norm_groups`を追加した。
  `decoder_args`にも`norm_args`を明示指定する（`BaseSeg`は`encoder_args`をdeepcopyして
  `decoder_args`へ`update()`するため暗黙にも継承されるが、要件通り明示化した）。
- `train_stage5.py`/`infer_stage5.py`/`evaluate_stage5.py`: `--pointnext_norm`
  （`batchnorm`/`groupnorm`、既定`batchnorm`）・`--pointnext_norm_groups`（既定8）をCLIとcheckpoint
  config復元経路（`model_from_checkpoint`/`resolve_model_kwargs`）の両方へ追加した。
  `train_stage5.py`の`config = vars(args).copy()`により、checkpointの`config.json`へは追加作業なしで
  新fieldが保存される。`infer_stage5.py`/`evaluate_stage5.py`はcheckpoint configから同じnormalization
  設定を復元するため、`infer_stage5.sh`/`evaluate_stage5.sh`自体の変更は不要だった。
  `train_stage5.sh`（editable launcher）にのみ`POINTNEXT_NORM`/`POINTNEXT_NORM_GROUPS`変数を追加した
  （既定`batchnorm`/8のため既存runの挙動は変わらない）。
- `Stage5/checks/dummy/check_dummy_pointnext_s_training.py/.sh`: 既存のdummy training smoke test
  （`train_stage5.py`をsubprocess経由で実行しcheckpoint reloadまで検証する既存checker）へ
  `--pointnext_norm`/`--pointnext_norm_groups`を追加した（既定値は従来通りのため既存呼び出しは
  無変更で動作する）。`.sh`は`POINTNEXT_NORM`/`POINTNEXT_NORM_GROUPS`環境変数で切り替えられ、
  Step D4（dummy training）をBatchNorm/GroupNorm両方で共用する。
- `Stage5/checks/dummy/check_dummy_pointnext_s_groupnorm.py/.sh`（新規、Step D2/D3）:
  `--self_test`はCPU only（CUDA不要）で、`Stage5GroupNorm`の`[B,C,N]`/`[B,C,*,*]`両形状での
  forward、divisibility違反のfail-fast、train()/eval()出力の完全一致、`resolve_pointnext_norm_args`の
  batchnorm/groupnorm/不正値応答、`build_pointnext_s_config`のencoder/decoder/head全サイトへの
  norm_args伝播（3サイトが独立dictであることの検査を含む）を検証する。full run（CUDA必須、dummy H5）は
  batchnorm既定modelのBatchNormモジュール数記録、groupnorm modelでの同数`Stage5GroupNorm`置換、
  train/eval logits完全一致、checkpoint保存・strict reload・logits一致、不正group数のfail-fastを
  検証し、任意で実運用checkpoint（`--legacy_checkpoint`）のstrict loadによる後方互換性再確認も行う。
- `Stage5/checks/transfer/check_stage5_batchnorm_to_groupnorm_transfer.py/.sh`（新規、Step D5）:
  既存のStage5 BatchNorm S3DIS部分転移checkpoint（`stage5_pointnext_s_s3dis_partial_init.pt`）から
  GroupNorm modelへ、key一致＋shape一致のconv/linear/head/norm affine parameterだけを転移する。
  target側でBatchNorm転移checkpointに対応キーがない場合、source側でBN buffer suffix
  （`running_mean`/`running_var`/`num_batches_tracked`）以外の未使用keyがある場合、shape不一致がある
  場合はいずれもfail-fastとする。`--self_test`はCPU onlyの合成state dictでkey分類ロジックのみを検証する
  （正常系・非BN bufferの unexpected key検知・target側missing key検知の3ケース）。

静的検証（devコンテナ、`torch`非搭載のため`py_compile`/`bash -n`/`git diff --check`のみ）:
すべて合格。synthetic self-test（CPU only、torch要）はユーザー実機で合格した。

**Step D2/D3実行で発見・修正した不具合（2026-09-12）:** ユーザー実機での初回full run
（`check_dummy_pointnext_s_groupnorm.sh`）で、`pointnext_norm=groupnorm`指定にもかかわらず
BatchNorm moduleが8個残るエラーが発生した。原因を調査したところ、official OpenPointsの
`PointNextDecoder.__init__`は`norm_args`/`act_args`を`**kwargs`として受け取るが実際には一切
使用せず、`_make_dec()`は常に`FeaturePropogation`をnorm_args省略で構築するため、
`FeaturePropogation`自身のhardcoded既定値`{'norm': 'bn1d'}`が`decoder_args`の設定と無関係に
使われ続けることが判明した（8個はdecoderの4 stage×2 convsに一致し、encoder 8個・head 1個は
正しくGroupNormへ置換されていた）。

external PointNeXt cloneを直接編集せずに解消するため、`Stage5/stage5/models/
pointnext_decoder_patch.py`（新規）へ`PointNextDecoder`を継承した`Stage5PointNextDecoder`を実装し、
`_make_dec()`だけをoverrideして`norm_args`/`act_args`を`FeaturePropogation`へ明示的に渡すようにした。
`openpoints.models.build.MODELS`へ新しい名前`"Stage5PointNextDecoder"`として登録し（`force`なし、
`MODELS.get()`による冪等性チェック付き）、`build_pointnext_s_config()`の`decoder_args["NAME"]`を
これへ差し替えた。`norm_args={"norm": "bn"}`は`create_norm`のdimension接尾辞処理により
`FeaturePropogation`の元のhardcoded既定値`{'norm': 'bn1d'}`と同一の`nn.BatchNorm1d`を構築するため、
batchnorm既定経路のmodule構造・state_dict keyは変更前と完全に同一であり、既存checkpointの
strict loadに影響しない（この等価性は静的に確認済みで、Step D2の実運用checkpoint再確認
（`--legacy_checkpoint`）でも再検証する）。

再発防止のため、`check_dummy_pointnext_s_groupnorm.py`の`--self_test`へ
`test_stage5_pointnext_decoder_honors_norm_args`を追加した（`Stage5PointNextDecoder`単体を
CPU上でbatchnorm/groupnorm両方の`norm_args`で構築し、norm module型と数が期待通りであることを
直接検証する。ball query/FPSを使わないためCUDA不要）。

修正後の`py_compile`/`git diff --check`はdevコンテナで再確認済み。

**Step D2/D3 full run結果（2026-09-12、ユーザー実機）:** 修正後の再実行で
`Stage5 PointNeXt-S GroupNorm structure check passed.`を確認した。

| 項目 | 結果 |
| --- | --- |
| batchnorm module count（control） | 17 |
| groupnorm module count | 17（1:1置換、encoder 8 + decoder 8 + head 1と整合） |
| train/eval logits max abs diff | 0.000e+00（完全一致） |

BatchNorm module数とGroupNorm module数が完全一致し、修正が全normalization site（encoder・decoder・
head）へ及んだことを確認した。GroupNormのtrain()/eval() logitsが数値誤差なく完全一致したことは、
GroupNormがrunning statisticsに依存せずphysical batch size 1でもtrain/evalで同じ挙動になるという
S5-11の狙い（D-014の根拠）を構造面で裏付ける結果である。

**Step D4 dummy training smoke結果（2026-09-12、ユーザー実機）:** 既存`check_dummy_pointnext_s_training.py/.sh`
（train 2 files/6 samples、val 1 file/3 samples、physical batch 1、gradient accumulation 2、1 epoch）を
`batchnorm`/`groupnorm`双方で実行し、いずれも`PointNeXt-S Stage5 training CLI check passed.`
（config.json/metrics.jsonl/checkpoint保存とstrict reloadの既存assertion含む）を確認した。

| norm | train loss | train F1 | train IoU | train FP/FN | val loss |
| --- | ---: | ---: | ---: | --- | ---: |
| batchnorm | 0.6462 | 0.0589 | 0.0303 | 95/864 | 0.6793 |
| groupnorm | 0.6267 | 0.1133 | 0.0600 | 122/833 | 0.6307 |

train/val lossはいずれも有限で、forward・backward・optimizer step・checkpoint保存・strict reloadが
両方式で正常に完走した（val F1/IoUが両方0なのは、val 3 sampleのみの合成dummyデータでpositiveが
偶然含まれなかったためであり、精度上の懸念ではない）。

**Step D5 BatchNorm→GroupNorm転移結果（2026-09-12、ユーザー実機）:** 既存Stage5 BatchNorm
S3DIS部分転移checkpoint（`stage5_pointnext_s_s3dis_partial_init.pt`、114 key）から
GroupNormモデル（63 key）へ転移し、`Stage5 BatchNorm -> GroupNorm transfer check passed.`を確認した。

- loaded keys: 63/63（target側のmissing・shape不一致は0件）
- excluded BatchNorm buffer keys: 51（17 BN module × `running_mean`/`running_var`/`num_batches_tracked`
  の3 bufferと完全一致し、それ以外の予期しない未使用keyは0件）
- forward loss: 0.720794（有限）
- 出力: `stage5_pointnext_s_s3dis_partial_init_groupnorm.pt`、
  `stage5_pointnext_s_s3dis_partial_init_groupnorm.transfer_report.json`

51 = 17×3という正確な一致は、`Stage5GroupNorm`が`nn.GroupNorm`を直接継承しaffine
`weight`/`bias`をBatchNormと同じdotted keyに配置する設計、およびfail-fast転移ロジックの両方が
意図通り動作していることを裏付ける。

**Step D6 実H5 1 epoch smoke結果（2026-09-12、ユーザー実機）:** Step D5の
`stage5_pointnext_s_s3dis_partial_init_groupnorm.pt`から、teacher v6・padding-free・
window 16/8・physical batch 1・gradient accumulation 8という既存BatchNorm controlと同一条件で
1 epoch学習した（`pointnext_norm=groupnorm`）。次のrunである。

```text
pointnext_s_EX260912_260711_w16_s8_bboxrankv6_cvatxmlinv_glocal_ce_smooth00_auto_weight_lr1e3_ep1_bs1_acc8_nopad
```

train 163 files / 729 windows、validation 18 files / 79 windowsで、microbatch数729・optimizer
step数92・padding比率0・empty-valid window 0はBatchNorm controlと同様に確認した。class weight
`[0.05963856, 1.94036150]`もBatchNorm control（teacher v6）と完全一致し、データ経路自体は
normalization方式を切り替えても変わっていないことを確認した。

1 epoch終了時点のwindow単位train/val: train loss/F1/IoU `0.5680/0.0365/0.0186`、
val loss/F1/IoU `0.5398/0.0522/0.0268`だった。既存記録（9.4節）のBatchNorm control 1 epoch
smoke testのvalidation `0.6183/0.0040/0.0020`と比べ、GroupNormは1 epoch時点でval loss・F1・IoUの
いずれも上回った。

`evaluate_stage5.sh`で固定train sanity 3動画+validation 18動画の`eval()`・mean aggregation評価も
実行した（本来Step D7の5 epoch pilotで行う予定だった評価を、1 epoch checkpointに対して先行実施した
参考値）。

| split | TP | FP | recall | F1 | IoU | TP0動画数 | video median F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train_sanity（3動画） | 2,383 | 54,646 | 24.43% | 0.0714 | 0.0370 | 0/3 | — |
| validation（18動画） | 12,367 | 405,149 | 17.81% | 0.0508 | 0.0261 | 2/18 | 0.0313 |

参考として、既存BatchNorm control（5 epoch pilot`best.pt`、epoch 4）の同一評価は
validation TP 2,691・FP 85,785・recall 3.87%・F1 0.0341・IoU 0.0173・TP0動画数10/18・
median F1 0だった（9.4/9.5節）。GroupNormは**1 epochの時点で**、BatchNorm controlの**5 epoch**より
recall・F1・IoU・TP0動画数（10/18→2/18）・median F1（0→0.0313）のいずれも上回った。ただし
FPも85,785→405,149（+4.7倍）へ増加しており、recallの増加（3.87%→17.81%、+4.6倍）とほぼ同じ倍率で
動いている。この比率の近さは、S5-09/S5-10で警戒した「検出性能の向上ではなく、閾値0.5に対する
確率校正が全体的にpositive側へ移動しただけ」という可能性を否定できない。また、1 epoch
（GroupNorm）と5 epoch（BatchNorm）という学習量が異なる比較であるため、この時点では
同epoch数での比較ではない。

**現時点での暫定所見（結論ではない）:** GroupNormは1 epochの早い段階から、BatchNorm 5 epoch
pilotが抱えていた「validationの半数以上でTPが0」という崩壊パターンを示していない。これは有望な
兆候だが、(a) FP増加がrecall増加とほぼ比例している点、(b) 学習量が同一でない比較である点の
2つから、判定基準（12章）に基づく最終判断はStep D7の5 epoch pilot（BatchNorm controlと同じ
epoch数）を待って行う。

### 9.6 Label policy ablation

BBox内かつcontour外をbackgroundとするrunと、ignoreのままにするrunを、同一の固定評価データで比較する。

### 9.7 Class weightと座標依存の比較

構造上の問題を除いた後で、`auto`とより弱い固定weightを比較する。続いて座標augmentationまたは座標feature ablationで、位置事前分布への依存を測る。

Hard Negative Mining、Dice loss、Focal lossなどは、上記のデータ経路とbatch処理を確認するまでは導入しない。

## 10. 結論

- PointNeXt-Sはtrain sanityで大腿骨周辺を学習しているが、valid background上のFPが多く、memorization確認としても未解決である。
- validationでは一部データを検出できるものの、median IoUはほぼ0であり、汎化性能は低い。
- 旧teacher v2・paddingありrunでは`best.pt`（epoch 45）が比較checkpoint中で最良だった。teacher v6・paddingなし5 epoch pilotでは`best.pt`はepoch 4だが、いずれも実用精度には達していない。
- train/validation全件のbatch integrity testに合格し、Dataset、overlap window index、collate、shuffle、multi-workerによるGT・点群の取り違えは強く除外された。
- padding parity testにより、可変長windowのzero paddingが実点logits、loss、gradient、BatchNorm統計を大きく変えることを確認した。
- batch size 8のpaddingあり学習とbatch size 1のpaddingなし評価が不一致であり、現在の学習結果を解釈する上で重大な交絡要因となっている。
- batch size 1 + gradient accumulationによるpaddingなし経路は、受入済みteacher v6の5 epoch pilotでも構造上の検査に合格した。一方でepoch 1から5にtrain lossが低下する間、validation lossは単調増加した。
- v6 5 epoch checkpoint評価では、bestのtrain sanity F1がwindow単位0.1026からoverlap平均後0.0304へ、validation F1が0.0455から0.0341へ低下した。200 epoch学習前にwindow間予測不一致とaggregation方式を定量化する。
- 反復するXY位置のpositiveは、overlapによる重複教師と座標事前分布で説明できる可能性があり、frame-level metricsで検証する必要がある。
