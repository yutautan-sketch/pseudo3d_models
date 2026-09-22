# Stage 5 PointNeXt-S学習・評価調査報告

> **位置づけ注記（2026-09-10）:** 本書は検証結果と数値的根拠の正本である。Stage 5全体の
> 現在状態、採用済み判断、次の実施順は`stage5_revision_management_record.md`を参照する。

> **最終更新・完了注記（2026-09-15）:** S5-14 core・補足1〜3をユーザー判断により完了として受入。
> 最終的な解釈・転記上の留保は9.8.5節を参照。診断完了は精度問題の解消やproduction採用を意味しない。

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

**Step D7 実H5 5 epoch pilot結果（2026-09-12、ユーザー実機）:** Step D6と同一の
`INIT_CHECKPOINT`（GroupNorm転移checkpoint）・split・seed・window・class weight・lossで
`EPOCHS=5`まで学習した。

```text
pointnext_s_EX260912_260711_w16_s8_bboxrankv6_cvatxmlinv_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad
```

全5 epochでpadding点0、empty-valid window 0、microbatch 729、optimizer step 92を維持し、class
weight`[0.05963856, 1.94036150]`もBatchNorm controlと完全一致した。データ経路・勾配蓄積・metrics
集計に不具合は検出されなかった。

| epoch | train loss | train F1 | train IoU | val loss | val F1 | val IoU | val predicted positive率 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5678 | 0.0368 | 0.0187 | 0.5394 | 0.0491 | 0.0252 | 8.23% |
| 2 | 0.5005 | 0.0729 | 0.0378 | 0.5088 | 0.0824 | 0.0429 | 9.14% |
| 3 | 0.4668 | 0.0846 | 0.0442 | 0.5070 | 0.0696 | 0.0361 | 12.66% |
| 4 | 0.4398 | 0.0920 | 0.0482 | 0.4801 | 0.0865 | **0.0452** | 8.75% |
| 5 | 0.4313 | 0.0961 | 0.0505 | 0.4996 | 0.0764 | 0.0397 | 12.27% |

`best_metric=iou_femur`によりGroupNormの`best.pt`もepoch 4となり、BatchNorm control（epoch 4）と
同一epoch数での比較が成立する。9.4節のBatchNorm control表と比べ、最も大きな違いはvalidation
lossの挙動である。BatchNorm controlはvalidation lossが5 epoch全てで単調増加した
（0.5985→0.9208）のに対し、GroupNormのvalidation lossは0.539〜0.509の範囲で安定し、単調な
悪化を示さなかった。

`evaluate_stage5.sh`で固定train sanity 3動画+validation 18動画の`eval()`・mean aggregation評価を
実行した（BatchNorm controlと完全に同一の評価pipeline・同一epoch数の`best.pt`）。

| split | model | TP | FP | precision | recall | F1 | IoU | TP0動画数 | video median F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train_sanity | BatchNorm control | 407 | 16,602 | 2.39% | 4.17% | 0.0304 | 0.0154 | 1/3 | — |
| train_sanity | GroupNorm | 4,130 | 42,589 | 8.84% | 42.33% | 0.1463 | 0.0789 | **0/3** | — |
| validation | BatchNorm control | 2,691 | 85,785 | 3.04% | 3.87% | 0.0341 | 0.0173 | 10/18 | 0 |
| validation | GroupNorm | 23,687 | 446,335 | 5.04% | 34.11% | 0.0878 | 0.0459 | **0/18** | **0.0818** |

GroupNormはBatchNorm controlと同じepoch数（4）で、validation aggregated F1
（0.0341→0.0878、+2.57倍）、IoU（0.0173→0.0459、+2.65倍）、recall（3.87%→34.11%、+8.8倍）の
いずれも上回った。FPは85,785→446,335（+5.2倍）へ増加したが、precisionは低下せずむしろ
3.04%→5.04%へ改善しており、recallの増加率（+8.8倍）がFPRの増加率（1.32%→6.86%、+5.2倍）を
上回っている。これはS5-09のtrain-modeやS5-10のrecalibrationで見られた「recallとFPがほぼ
同じ比率で動く」パターンとは異なり、単純な確率校正の一様シフトでは説明しにくい。

video-level では、validationのTP0動画数が**10/18から0/18へ**、median F1が**0から0.0818へ**
改善し、18動画全てで最低限のTP（93点以上）を検出した。train_sanityもTP0動画数が1/3から0/3へ
改善し（歴史的にTP0だった動画も含む）、崩壊は確認されなかった。

**仮説判断:** 12章の判定基準に照らすと、「GroupNormが構造testを満たし、train sanity/validation
双方で有望」（分岐1）に該当する事実が確認された。TP0動画数の解消、video-level median F1/IoUの
改善、precisionとrecallの同時改善、validation lossの安定化（BatchNorm controlで見られた単調増加が
消失）は、いずれも構造的な改善を支持する。FP/FPRの増加（+5.2倍）は小さくないが、recallの増加が
上回りprecisionも改善しているため、S5-09/S5-10で確認された「recallだけが増えFPが同等以上に急増する
calibration shift」パターンとは異なる。

**未確定事項:** FP/FPRの増加が実運用（Stage 6の入力として）許容できる範囲かどうかは、本checkerの
範囲では判定しない。5 epochのみの比較であり、200 epochまで学習した場合の挙動、Label policy・
class weightとの相互作用は未検証。GroupNormのgroup数（8）や学習率など、正規化方式変更後の
再チューニングも行っていない。

**productionへの影響:** なし。production既定値は引き続き`batchnorm`のままであり、
`train_stage5.py`/`infer_stage5.py`/`evaluate_stage5.py`の既定挙動、production aggregation
（mean probability）は変更していない。GroupNormをnormalization候補として暫定採用するかどうかは
方針管理チャットの判断に委ねる。

### 9.6 Label policy ablation

#### 実装事項E: GroupNorm固定のBBox non-contour label policy比較 実装計画（2026-09-13）

**状態:** completed（2026-09-14）。teacher v7を用いて2条件を各5 epochで比較し、共通target評価まで
完了した。結果に基づく方針管理チャットの判断は本節末尾に記録する。production label policy、
normalization既定値、thresholdは変更していない。

**方針管理チャットの判断（2026-09-13）:** S5-11でGroupNormはtrain/eval logits完全一致、
validation TP0動画数10/18から0/18、video median F1 0から0.0818、precision/recall同時改善を示したため、
S5-12以降の実験用normalizationとして暫定採用する。production既定値はS5-15まで`batchnorm`のまま
維持する。GroupNormで増加したFP/FPR（validation FP 85,785から446,335、FPR 1.32%から6.86%）は
production上許容済みとは判断せず、S5-12/S5-13で原因と抑制可能性を調べる。ただしlabel policyと
class weightがscore分布を変えるため、thresholdは0.5に固定し、この時点ではtuningしない。
200 epochへ直接進まず、S5-12/S5-13を各5 epoch、必要なS5-14診断、5〜10 epoch最終pilot、50 epoch
中間判定を経てから100〜200 epochの可否を決める。

**問題:** 現行teacher v6では、CVAT/teacher contourのpositive点以外に、BBox内かつcontour外の点を
`point_label=-1`、`valid_mask=False`として保存している。この領域はlossへ入らないため、輪郭線周辺の
点をpositiveと予測しても直接のpenaltyがない。GroupNormは大腿骨検出を大幅に改善した一方でvalid
background上のFPも増加しており、BBox non-contour hard negativeを学習へ含めることでFPを抑制できるかを
切り分ける必要がある。ただし従来集計ではignore上のpositive予測はFPに含まれないため、異なる
`valid_mask`の通常F1だけを比較してはいけない。

**目的:** GroupNorm、初期parameter、split、seed、window、loss、class weight、threshold、aggregationを
固定し、BBox内かつcontour外の点をignoreのままにする場合とbackgroundとして学習する場合を比較する。
positive contourの検出能力を維持しながら、BBox周辺および既存valid background上のFPが減少するかを
測る。

**比較するlabel policy:** 同じteacher v6 H5を入力とし、元H5自体は書き換えない。

| Run | training時のBBox内かつcontour外 | その他のlabel |
| --- | --- | --- |
| A: `bbox_noncontour_ignore` | `point_label=-1`、`valid_mask=False`を維持 | source teacher v6のまま |
| B: `bbox_noncontour_background` | `point_label=0`、`valid_mask=True`へ変換 | source teacher v6のまま |

Run Bで変更してよいのは、teacher v6のprovenance上「BBox内かつpositive contour外」と確認できた点だけと
する。CVAT authoritative maskのpositive点はBBox外またはno-BBox frameに存在する場合も含めてpositiveの
まま維持し、no-BBoxという理由だけでbackgroundへ上書きしない。XML invalidationによりbackground化された
frame、既存background、positive contour、frame metadataは変更しない。

**label policy実装方針:** Stage 4 H5の複製・破壊的変更は行わず、Stage 5のDatasetでsource labelから
effective training labelを作る。policy名は対象領域が明確なものとし、全てのignore理由を一括変換する
一般的な`ignore_to_background`にはしない。policy未指定時は既存挙動の`ignore`を既定値とし、既存run/
checkpointとの後方互換性を維持する。変換処理はDataset、class count/debug集計、checkerから共通利用できる
一つのhelperへ集約し、入力配列をin-place変更しない。

実データ適用前にteacher v6全H5をpreflightし、少なくとも次をassertする。

1. `valid_mask == (point_label != -1)`である。
2. policy対象maskとsourceのignore点の関係を件数付きで記録する。
3. policy対象点が保存BBox内かつauthoritative/teacher positive外である。
4. Run Bでpositive点数とpositive indexがRun Aから変化しない。
5. Run Bで変更されるのは対象点の`-1 -> 0`と`False -> True`だけである。
6. points、features、frame_order、point_indices、window一覧は両runで完全一致する。
7. 対象外のignore理由が存在する場合、それらをbackgroundへ変換せずfail-fastまたは別区分として残す。

H5 schemaだけでignore理由を一意に識別できない場合は、`frame_annotation/bbox_local_xyxy`と
`point_cloud/pixel_xy`、authoritative positive labelから対象maskを再構築して検証する。全ignoreが
BBox non-contourであることを監査で証明できた場合に限り、`point_label == -1`を対象maskの簡略表現として
利用できる。

**CLI / config要件:** training CLIへBBox non-contour policyを明示するoptionを追加し、checkpointの
`config`、`config.json`、起動ログ、metrics debug fieldへ保存する。Runごとにsource/effectiveのpositive、
background、ignore点数、変換点数、対象frame/window数を出力する。inferenceはlabelをmodel入力に使わない
ため予測動作を変更しないが、評価側はcheckpointがどのpolicyで学習されたかを表示できるようにする。

**class weightの固定:** Run Bでは有効background点が増えるため、`auto`を各runで再計算するとlabel policyと
class weightの2要因が同時に変わる。S5-12では両runとも、S5-11 GroupNorm controlで解決済みの同一weight
`[0.05963856, 1.94036150]`を明示指定する。source/effective class countは診断値として記録するが、weightの
再計算には使わない。class weight自体の比較はS5-13で行う。

**固定比較条件:** Run A/Bは次を完全に揃える。

| 項目 | 固定値 |
| --- | --- |
| teacher | Stage 4 teacher v6、同一H5 |
| train / validation split | 既存GroupNorm pilotと同じ163 / 18 files |
| initialization | S5-11で作成した同一GroupNorm S3DIS部分転移checkpoint |
| normalization | GroupNorm、8 groups |
| seed | 42 |
| window | size 16 / stride 8 / tailあり |
| physical batch | 1 window |
| gradient accumulation | 8 windows、point-weighted normalization |
| features | `intensity,confidence` |
| loss | CrossEntropyLoss |
| class weight | `[0.05963856, 1.94036150]` |
| label smoothing | 0.0 |
| optimizer | AdamW、lr `1e-3`、weight decay `1e-4` |
| dropout | 0.0 |
| epochs | 5 |
| inference | `model.eval()` |
| aggregation / threshold | mean probability / 0.5 |

既存S5-11 GroupNorm pilotは外部referenceとして残すが、S5-12の主比較では同じコードrevisionと実行環境で
Run A/Bを両方再実行する。file listを再splitせず、既存`train_files.txt`/`val_files.txt`を明示的に再利用
する。window数とshuffle順がpolicy間で一致することを検査する。

**共通評価label:** native validation loss/F1はRun Aではignoreを除外し、Run Bでは同じ点をbackgroundとして
含めるため、直接比較しない。両checkpointを同一動画・同一point prediction・同一aggregationで推論し、
次の共通targetを両方へ適用する専用ablation evaluator/checkerを用意する。

1. `canonical_v6`: source teacher v6のvalid点だけを評価し、従来metricsとのparityを確認する。
2. `bbox_noncontour_as_background`: 監査済みBBox non-contour点をbackgroundかつvalidとして加えた共通target
   で両modelを評価する。S5-12の主比較targetとする。
3. `bbox_noncontour_region`: 対象領域だけについてpredicted positive count/rate、mean/percentile
   `prob_femur`、FP/FPR相当値を記録する。

positive contourのGT indexは全targetで同一とする。通常metricsに加え、既存valid backgroundとBBox
non-contourを分けて集計し、FP減少が単なるpositive予測全体の抑制か、対象hard negativeで選択的に
起きたかを確認する。既存`evaluate_stage5.py`のcanonical mean baseline parityを維持し、不一致時は停止する。

**checkpoint比較:** policyごとにnative validation targetが異なるため、各run固有の`best.pt`だけを主比較に
しない。primary comparisonは同じ学習量の`checkpoint_epoch_0005.pt`または同等のepoch 5固定checkpointと
する。各runの`best.pt`もsecondaryとして保存・評価し、best epoch、選択metric、native targetの違いを
明記する。必要なら`save_every=1`として全epoch checkpointを残す。

**実装・検証手順:** 次の順序で進める。

1. synthetic H5でRun Aが完全なno-opであること、Run Bが対象ignoreだけをbackground化すること、
   positive/points/windowが不変であることをtestする。
2. teacher v6全181 source H5またはtrain/validation対象全件をpreflightし、policy対象maskとlabel契約を
   監査する。対象外ignoreが見つかった場合は実学習へ進まず報告する。
3. Dataset sample、class count、window、collateを両policyで比較するcheckerを実行する。
4. 既存GroupNorm checkpoint/configのstrict reloadとRun A backward parityを確認する。
5. Run Bで1 epoch smokeを実行し、変換件数、loss normalizer、gradient、metrics、checkpointがfiniteで
   あることを確認する。
6. 同じ初期checkpointとseedからRun A/Bを各5 epoch実行し、毎epoch checkpointを保存する。
7. 両runのepoch 5とbest checkpointを固定train sanity 3動画・validation 18動画へ適用する。
8. 3種類の共通評価target、video-level集計、BBox non-contour region診断を出力する。
9. 固定train sanityと代表validationについて、GT、両modelの全prediction、positive-only PLYを同じ座標で
   書き出し、輪郭周辺FPと大腿骨positive欠損を目視確認できるようにする。

**評価項目:** 共通targetごとにTP、FP、TN、FN、precision、recall、F1、femur IoU、FPR、FNR、predicted
positive率、TP0動画数、video-level mean/median F1・IoUを記録する。BBox non-contour regionについては
点数、predicted positive count/rate、probability mean、p50/p90/p95/p99、動画別値を記録する。
Run Bで追加されたbackground点がloss denominatorへ占める割合、source/effective class比、ignore比も
split別に残す。

**判定基準:** Run Bは、共通`bbox_noncontour_as_background` targetでBBox non-contour regionのpositive
予測率と全体FP/FPRを低下させつつ、canonical positive contourのrecall、TP0動画数、video median F1/IoUを
大きく悪化させない場合に有望とする。FPが減ってもrecall低下またはTP0再発が大きい場合は採用しない。
Run A/B差が小さい場合はBBox non-contour ignoreをGroupNormの主要FP原因から下げ、既存ignore policyを
維持してS5-13へ進む。split集約だけ改善しvideo別符号が揃わない場合は系統的改善と判断しない。

**結果による分岐:** Run Bが有望ならBBox non-contour backgroundをS5-13以降の暫定training policyとして
固定する。有望でなければRun Aのignoreを維持する。いずれの場合も、同じ採用policyとGroupNormを固定して
S5-13のclass weight比較へ進む。threshold tuningはS5-13後、最終score分布が定まってからS5-15で行う。
各後続設定は5〜10 epoch pilotで確認し、長期学習は50 epoch中間判定を通過した場合だけ100〜200 epochへ
進める。

**非対象:** 本事項ではGroupNorm group数、学習率、class weight、threshold、aggregation、window、入力
feature、Stage 4 teacherを変更しない。Dice/Focal/Hard Negative Mining、座標augmentation、center-only
loss、overlap回数weight、200 epoch学習を追加しない。source H5とproduction既定値を上書きせず、
checkerの完走、仮説判断、production採否を分けて報告する。

**実装進捗（2026-09-13）:** helper/CLI/preflight/parity/evaluatorのコード実装、および
GPU非依存の静的検証まで完了した。GPU実行（Step E5〜E7）とsynthetic self-test（Step E2）は未実行。

- `Stage5/stage5/utils/label_policy.py`（新規）: `compute_bbox_inside_mask()`
  （`Stage2to4/checks/stage4/check_stage4_bbox_ranked_label_policy.py`と同じ幾何ロジック
  ——pixel_xyの丸め込み、frame単位のBBox union判定——をStage 5側へ移植し、両stageの
  「BBox内」定義を一致させた）、`apply_bbox_noncontour_label_policy()`
  （`bbox_noncontour_ignore`は完全no-op、`bbox_noncontour_background`は監査済み対象点のみ変換し、
  入力配列を破壊しない）、`require_no_stray_ignore_outside_bbox()`
  （BBox外のignore点が1点でもあればfail-fast）を実装した。
- `Stage5/stage5/utils/h5_io.py`: `frame_annotation/frame_order`・`frame_annotation/bbox_local_xyxy`
  の読み込みを追加した（既存の`OPTIONAL_POINT_CLOUD_KEYS`パターンと同様、存在する場合のみ）。
- `Stage5/stage5/datasets/pseudo3d_pointcloud_dataset.py`: `label_policy`引数を追加し、`_load()`内で
  一度だけ変換を適用する。sourceの`point_label`/`valid_mask`は`source_point_label`/`source_valid_mask`
  として別keyに保持し、診断に使えるようにした。
- `Stage5/train_stage5.py`: `--label_policy`（既定`bbox_noncontour_ignore`、後方互換）を追加し、
  `config.json`へsource/effective点数・変換点数を記録する`label_policy_diagnostics`を追加した
  （`resolve_class_weight`のauto経路とは独立した別スキャン。S5-12はclass weightを明示指定するため
  autoスキャンは実行されない）。
- `Stage5/train_stage5.sh`: `LABEL_POLICY`環境変数knobを追加し、`CLASS_WEIGHT`を
  env override可能にした（既定値は変更していないため既存runの挙動は不変）。
- `Stage5/checks/dummy/check_dummy_label_policy.py/.sh`（新規、Step E2相当）: pure array test
  （`compute_bbox_inside_mask`の幾何、no-op保証、対象外ignoreのfail-fast、独立copy保証）と、
  手動構築した4-frame合成H5によるDataset統合test（Run Aが生H5と完全一致、Run Bが監査済み対象点
  だけを変換しpositive/points/features/window構造を変えないこと、対象外ignoreを含むH5でのfail-fast）
  を実装した。CUDA/PointNeXtのimportが一切ないため、torchさえあればGPU無しで実行できる。
- `Stage5/checks/real_h5/check_stage5_label_policy_bbox_preflight.py/.sh`（新規、Step E3）:
  teacher v6のtrain/validation全H5を監査し、`valid_mask == (point_label != -1)`、BBox外ignore点数
  （0であることを既定でfail-fast gate）、no-BBoxフレーム上のlabel内訳、schema属性
  （`label_mode`等、実データでの値は未確認のため診断表示のみ）を記録する。CPU/h5pyのみで実行できる。
- `Stage5/checks/real_h5/check_stage5_label_policy_dataset_parity.py/.sh`（新規、Step E4）:
  実teacher v6 H5上でRun A/B Datasetを比較し、points/features/frame_order/point_indices/window境界の
  hash一致、labelsの差分が監査済み対象点に限られること、positive集合が不変であることを検証する。
  CPU/h5pyのみで実行できる。
- `Stage5/checks/real_h5/check_stage5_label_policy_ablation_eval.py/.sh`（新規、Step E7）:
  既存`evaluate_stage5.predict_h5`を1回forwardし、同じ`aggregated_probability`/`pred_label`を
  `canonical_v6`（既存`evaluate_stage5.py`とのmean baseline parity gate付き）、
  `bbox_noncontour_as_background`（共通比較target）、`bbox_noncontour_region`
  （対象領域のみのprobability分布・predicted positive率）の3種類の target label/valid_maskへ
  適用する設計とした（追加のGPU forwardなし）。`--export_ply_alias`/`--ply_output_dir`で
  Step E8のGT/prediction診断PLY（`write_diagnostic_segmentation_ply`を再利用）も出力できる。

Step E5（1 epoch smoke）・Step E6（Run A/B 5 epoch）は、既存`train_stage5.sh`へ`LABEL_POLICY=
bbox_noncontour_background`を指定するだけで実行できる（追加のcheckerは不要）。`last.pt`は
5 epoch run終了時点でepoch 5と一致するため、Step E6が要求する「epoch 5固定checkpoint」は
`save_every`変更なしに`last.pt`で満たされる。

静的検証（devコンテナ、`torch`非搭載のため`py_compile`/`bash -n`/`git diff --check`のみ）:
すべて合格。

**Step E3 preflight結果（2026-09-13、ユーザー実機）:** teacher v6の全181 H5（train 163 + validation
18）を`check_stage5_label_policy_bbox_preflight.py`で監査した。checkerは設計通り動作し、
`valid_mask == (point_label != -1)`、no-BBoxフレーム上のignore点0件（no_bbox_ignore_count=0）、
schema属性（`label_mode=bbox_ranked_global_local`、
`contour_teacher_schema=bboxrank_v6_cvat_authoritative_xml_invalidation_v1`、
`bbox_inside_non_contour_label=ignore`）は想定通りだったが、次のfail-fast条件に該当した。

| 項目 | 値 |
| --- | ---: |
| 全体ignore点数 | 262,347 |
| stray ignore点数（BBox外） | 16（0.006%） |
| 該当ファイル数 | 2/181 |

| video | ignore点数 | bbox内（target）点数 | stray点数 | stray率 |
| --- | ---: | ---: | ---: | ---: |
| train_068 | 97 | 87 | 10 | 10.3% |
| val_009 | 843 | 837 | 6 | 0.7% |

データセット全体（69,849,282点）に対しては16点（0.00002%）と無視できる規模だが、
train_068単体ではignore点の10.3%を占める。3つの対応案（(1) 該当2ファイル除外、(2) stray点を
ignoreのまま残しRun Bの変換対象から明示的に除いて続行、(3) 実データ精査で原因特定してから決定）を
提示し、方針管理チャットは(3)を選択した。

**原因判明とStage 4側修正（2026-09-13〜14）:** 該当2動画のannotation可視化frameを目視確認した
結果、丸め誤差ではなく、Stage 2のcrop窓（動画単位で固定）と被写体（femur）の相対移動による
ずれ——既存除外事例`20250626_090758_8000`（`local_crop_tracking_drift`）と同型のパターン——と
判明した。詳細調査依頼を`docs/stage2to4/stage4/stage5_s5_12_stage4_crop_quality_investigation_request.md`として
Stage 4管理チャットへ提出し、次の修正を受領した
（`docs/stage2to4/stage4/stage5_s5_12_stage4_crop_quality_correction_report.md`）。

- train_068（実video `20250626_090652_6340`）: 動画全体をcrop品質不良として除外
  （既存除外1動画と合わせ計2動画除外）。
- val_009（実video `20250625_161030_0550`）: frame order 51のみを無効化し全pointをbackground化
  （CVAT確認済みframe 47-50は維持）。
- 退化BBox（幅または高さ0）を1 pixel領域として誤って有効扱いしていた境界条件バグを修正
  （`right > left and bottom > top`へ統一）。Stage 5側`stage5/utils/label_policy.py`の
  `compute_bbox_inside_mask()`にも同じ修正を反映した。
- 補正済みteacher v7（`bboxrank_v7_cvat_authoritative_crop_quality_v1`、180 H5）を構築し、
  全180 H5でstray ignore・no-BBox ignore・BBox内background・CVAT mask外positiveが
  いずれも0であることを確認した。
- `train_stage5.sh`/`infer_stage5.sh`/`evaluate_stage5.sh`の既定入力をteacher v7へ切り替え済み。

Stage 5側の対応（未実施、次に行う）:

1. `check_stage5_label_policy_bbox_preflight.py`をv7の180 H5へ再実行し、stray ignore=0を
   独立に確認する。
2. v7 inventory（180ファイル）でtrain/validation listを新規生成する。旧v6の163/18 splitは
   再利用しない（v6/v7でファイル構成が異なるため、単純な使い回しは不整合を生む）。
3. v7で改めてStep E2（既に完了）・E4（Dataset parity）・E5（1 epoch smoke）・E6（Run A/B
   5 epoch）・E7（共通target評価）を実行する。

`label_policy.py`のfail-fast設計自体は変更しない（緩めない）。今回、入力側の契約違反が
解消されたため、stray pointを黙って変換するworkaroundは不要という判断をStage 4・Stage 5双方で
共有している。S5-07〜S5-11（teacher v6ベース）は既存結果として保持し、v6/v7を指標比較時に
明記する。

**teacher v7でのStep E2〜E4再実行結果（2026-09-14、ユーザー実機）:** v7 inventory（180 H5）から
`train_stage5.py`の`collect_h5_paths_from_dir`/`split_train_val_paths`（val_fraction 0.1、seed 42）
を直接呼び出し、実際の学習開始時と同一ロジックでtrain 162 / val 18のfile listを新規生成した上で、
Step E2〜E4を再実行した。

| Step | 結果 |
| --- | --- |
| E2（synthetic test） | 合格 |
| E3（v7 180 H5監査） | 合格。stray ignore points = **0**（Stage 4修正をStage 5側からも独立確認）。ignore点総数262,244 = BBox内target点数262,244で完全一致。schema属性も`contour_teacher_schema=bboxrank_v7_cvat_authoritative_crop_quality_v1`で想定通り |
| E4（Dataset parity、180ファイル全件） | 合格。総window数802、BBox non-contour target点数262,244（E3の集計と一致） |

v6時点のignore点総数262,347から262,244への減少（-103点）は、除外2動画分のignore点と、
val_009 frame 51の6点がbackground化された分の合計として整合する。E3/E4がいずれもGPU非依存
（CPU/h5pyのみ）で完走したことも確認できた。

**Step E5結果（2026-09-14、ユーザー実機）:** Run B（`bbox_noncontour_background`）を1 epoch
smoke実行した（`EX260914smoke_..._ep1_bs1_acc8_nopad`）。train 162 files/715 samples、
val 18 files/87 samplesで実際のsplitと一致した。変換点数はtrain 238,377点、val 23,867点
（`-1 -> 0`）で、class weightは固定値`[0.05963856, 1.9403615]`（auto再計算なし）を確認した。
epoch 1: train loss/F1/IoU `0.5801/0.0319/0.0162`、val loss/F1/IoU `0.5651/0.0428/0.0218`で、
finite値かつcheckpoint保存まで完走した。

**Step E6結果（2026-09-14、ユーザー実機）:** teacher v7・v7 split（162/18）・GroupNorm・
class weight固定値`[0.05963856, 1.94036150]`で、Run A（`bbox_noncontour_ignore`、
`EX260914_..._ep5`）とRun B（`bbox_noncontour_background`、`EX260915_..._ep5`）を各5 epoch
学習した。

| run | epoch | train loss | train F1 | train IoU | val loss | val F1 | val IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 1 | 0.5786 | 0.0329 | 0.0167 | 0.5596 | 0.0294 | 0.0149 |
| A | 2 | 0.5043 | 0.0719 | 0.0373 | 0.5253 | 0.0535 | 0.0275 |
| A | 3 | 0.4709 | 0.0809 | 0.0421 | 0.4998 | 0.0679 | 0.0351 |
| A | 4 | 0.4494 | 0.0902 | 0.0472 | 0.5003 | 0.0674 | 0.0349 |
| A | 5 | 0.4323 | 0.0995 | 0.0523 | 0.4735 | 0.0727 | **0.0377** |
| B | 1 | 0.5800 | 0.0323 | 0.0164 | 0.5685 | 0.0222 | 0.0112 |
| B | 2 | 0.5091 | 0.0705 | 0.0366 | 0.5291 | 0.0550 | 0.0283 |
| B | 3 | 0.4785 | 0.0810 | 0.0422 | 0.5187 | 0.0490 | 0.0251 |
| B | 4 | 0.4529 | 0.0882 | 0.0461 | 0.4947 | 0.0607 | 0.0313 |
| B | 5 | 0.4376 | 0.0955 | 0.0501 | 0.4963 | 0.0680 | **0.0352** |

両runとも全epochでloss/F1/IoUが有限で、val IoUはepoch 5で最大（Run A 0.0377、Run B 0.0352）
だった。`best_metric=iou_femur`により`best.pt`・`last.pt`はいずれも同一epochを指す見込みで
あり、Step E7で確認する。ここでのF1/IoUはRun A/Bそれぞれの実効label（Run Bはvalid_maskが
異なる）に対するwindow単位集計であるため、この時点でA/Bを直接比較しない
（Step E7の共通targetで比較する）。

**Step E7結果（2026-09-14、ユーザー実機）:** Run A・Run Bとも`last.pt`（epoch 5、Step E6で
val IoU最大epochと一致）に対し、既存`evaluate_stage5.sh`でcanonical参照値を取得した後、
`check_stage5_label_policy_ablation_eval.py`で3 target評価を実行した。`canonical_v6`の
mean baseline parity gateはRun A/Bとも合格し、既存`evaluate_stage5.py`との数値一致を確認した。

実行中に、本checker（`check_stage5_label_policy_ablation_eval.sh`）の`RUN_DIR`/`EVALUATION_DIR`
既定値が、teacher v7導入前に設定した旧v6 S5-11 GroupNorm pilot run（`EX260912`）のpathへ
ハードコードされたままになっているバグを発見した。`CHECKPOINT`/`REFERENCE_H5_METRICS_CSV`を
明示指定してもこの既定値が使われた結果、`VAL_LIST`が修正前v6の`val_files.txt`（val_009の
frame 51 stray点を含む）を指してしまい、Run Aの評価中にfail-fastした。`RUN_DIR`/
`EVALUATION_DIR`を`CHECKPOINT`/`REFERENCE_H5_METRICS_CSV`の親ディレクトリから自動導出する
よう修正し、以後は評価対象のrunと矛盾しないpathが常に使われるようにした。

**主要metrics（`canonical_v6` target、window/point集約後）:** `canonical_v6`はcheckerに残る
historicalな評価target IDであり、本評価の入力はteacher v7である。実体はteacher v7 H5のnative valid
labelで、teacher v6の181-file inventoryを評価したものではない。

| split | run | TP | FP | recall | precision | F1 | IoU | FPR | TP0動画数 | video median F1 | video median IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train_sanity | A（ignore） | 5,682 | 82,901 | 58.24% | 6.41% | 0.1156 | 0.0613 | 13.42% | 0/3 | 0.0568 | 0.0292 |
| train_sanity | B（background） | 6,405 | 79,822 | 65.65% | 7.43% | 0.1335 | 0.0715 | 12.93% | 0/3 | 0.0555 | 0.0632 |
| validation | A（ignore） | 33,531 | 830,987 | 44.69% | 3.88% | 0.0714 | 0.0370 | 11.71% | 1/18 | 0.0530 | 0.0272 |
| validation | B（background） | 32,376 | 864,239 | 43.15% | 3.61% | 0.0666 | 0.0345 | 12.17% | 1/18 | 0.0623 | 0.0322 |

train_sanityはF1・IoU・recallいずれもRun Bが上回り、FPも減少した。一方validationの
aggregated指標はRun Aがわずかに上回り、FPR はRun Bでむしろ悪化した（11.71%→12.17%）。
video単位のcanonical F1では18動画中Run Aが12勝、Run Bが5勝、1引き分けだったが、
median F1/IoUはRun Bが上回った（分布の裾（F1が低い動画が多い）と中央値の位置が異なるため。
矛盾ではなく、Run Aの勝ちが小差で多く、Run Bの勝ちが中央付近に位置することによる）。
TP0動画数は両runとも1/18で差がない。

**BBox non-contour region診断（`bbox_noncontour_region`、対象点でのpredicted positive率）:**

| split | run | target点数 | predicted positive数 | predicted positive率 |
| --- | --- | ---: | ---: | ---: |
| train_sanity | A | 4,966 | 2,202 | 44.34% |
| train_sanity | B | 4,966 | 2,026 | 40.80% |
| validation | A | 23,867 | 9,697 | 40.63% |
| validation | B | 23,867 | 7,429 | **31.13%** |

対象領域のpositive予測率はRun Bで明確に低下した（validation: 40.63%→31.13%、相対-23%）。
video別では21動画中12動画で改善、6動画で悪化、3動画で同値だった。これはRun Bの学習が、
狙い通りBBox non-contour領域でのpositive予測を抑制する方向に働いたことを支持する。ただし
9.6節冒頭で述べた通り、この領域抑制は全体FP/FPRの低下には直結しなかった（該当領域点数が
valid background全体（約720万点）に対して0.3%程度と少なく、領域内の改善が全体指標に埋もれた
可能性がある）。

**仮説判断:** 9.6節の判定基準（「BBox non-contour regionのpositive予測率**と**全体FP/FPRを
低下させつつ、canonical positive contourのrecall・TP0動画数・video median F1/IoUを大きく
悪化させない場合に有望」）に照らすと、領域内positive予測率の低下という核心メカニズムは支持
されたが、全体FP/FPRの低下は確認されず、validation aggregatedのrecall/F1/IoUはRun Aがやや
上回った。これは判定基準の「分岐1（有望）」「分岐2（悪化）」のいずれにも明確には該当せず、
「分岐3（差が小さい）」または「分岐4（判定不能）」に近い。5 epochという短い学習量、
teacher v7移行後で初めての比較であること、split集約とvideo別勝敗数・medianが一致しない点も
踏まえ、production採否を実装チャット側では決定しない。詳細な判断材料は
`docs/stage5/s5-12/stage5_s5_12_report_to_policy_chat.md`にまとめ、方針管理チャットへ返す。

**結果解析と方針管理チャットの判断（2026-09-14）:** Run Bは意図した局所的なlabel mechanismを
学習できた一方、それがStage 5全体のFP問題を解決するという仮説は支持されなかった。validationの
BBox non-contour positive予測率は40.63%から31.13%へ低下したが、対象点はvalid background約720万点の
約0.3%であり、全体FPRは11.71%から12.17%へ改善しなかった。canonical aggregated F1/IoUとvideo別F1の
勝敗はRun Aが優位で、Run Bのvideo median F1/IoUが高いことは一部動画への有効性を示すものの、TP0
動画数は同じである。したがって、Run Bを次段階の標準policyとする一貫した根拠には不足する。

以上から次を正式判断とする。

1. S5-13ではRun A（`bbox_noncontour_ignore`）を暫定label policyとして採用する。これはRun Bの局所効果を
   否定する判断ではなく、global validationを改善しなかった変更を固定せず、後方互換なpolicyで
   class weightの影響を単独評価するための保守的な選択である。Run Bは診断用alternateとして保持する。
2. S5-12のA/Bを長期学習へ延長せず、両policyを200 epochまで持ち越さない。対象領域が全backgroundの
   約0.3%である以上、学習期間の延長だけで局所抑制がglobal FP改善へ転じる見込みは限定的である。
   threshold 0.5、mean aggregationを維持し、次にglobalなpositive/negative trade-offへ直接作用する
   S5-13 class weight比較を行う。
3. teacher v7への修正だけを理由にS5-07〜S5-11を一律再実行しない。S5-11までの構造診断はteacher v6の
   履歴結果として保持し、teacher v7のS5-12 Run Aを今後の比較baselineとする。特定の仮説の解釈に
   teacher差が交絡する場合だけ、対応するcheckerまたはpilotをv7で個別再実行する。

この判断によりS5-12は完了とし、次段階をteacher v7、GroupNorm 8 groups、Run A label policy、
threshold 0.5、mean aggregation固定のS5-13 class weight ablationとする。GroupNormおよびlabel policyの
production既定値は引き続き変更せず、最終採否はS5-15で判断する。

### 9.7 Class weightと座標依存の比較

構造上の問題を除いた後で、`auto`とより弱い固定weightを比較する。続いて座標augmentationまたは座標feature ablationで、位置事前分布への依存を測る。

Hard Negative Mining、Dice loss、Focal lossなどは、上記のデータ経路とbatch処理を確認するまでは導入しない。

#### 実装事項F: Class weight ablation（GroupNorm固定、teacher v7）実装・検証結果（2026-09-14）

**目的:** S5-12でRun A（`bbox_noncontour_ignore`）を暫定採用した後、teacher v7・GroupNorm・
`bbox_noncontour_ignore`を固定し、class weightだけを変えてglobal FP/FPRとpositive recallの
trade-offを調べる。比較armは次の2つ。

| Arm | class weight | positive:background weight比 |
| --- | --- | ---: |
| W-A（control、S5-12 Run Aを再利用） | `[0.05963856, 1.94036150]` | 約32.54 |
| W-B | `[0.5, 1.5]` | 3.0 |

**実装内容:** `train_stage5.sh`の`PREFIX`が実際の`CLASS_WEIGHT`に関わらず文字列`auto_weight`を
固定で含んでいた不具合を修正し、`CLASS_WEIGHT`からfilesystem-safeなtag（`cw_auto`/
`cw_manual_<値>`/`cw_none`）を生成してrun名・起動ログへ反映、`OUTPUT_DIR`/`EXPERIMENT_NAME`の
明示override、非emptyな既存出力先への書き込みガード（`ALLOW_EXISTING_OUTPUT_DIR`）を追加した
（production既定値`CLASS_WEIGHT=auto`は変更していない）。新規checker
`checks/real_h5/check_stage5_class_weight_ablation.py/.sh`を追加し、2 run directory間で
config.json（class weight関連キーと意図的に異なり得るrun/output由来キーを除く全一致）、
train_files.txt/val_files.txt、resolved class weight、history.jsonのepoch数・finite性、
初期checkpointのpath/SHA-256、（任意で）評価対象point集合の一致をfail-fastで検証する
（CPU/JSON/CSVのみ、torch/h5py/CUDA不要）。合成テスト
（`checks/dummy/check_dummy_class_weight_tag.sh`8ケース、
`checks/dummy/check_dummy_class_weight_ablation.py/.sh`11ケース）はすべて合格した。

```text
Stage5/train_stage5.sh                                          (変更)
Stage5/checks/real_h5/check_stage5_class_weight_ablation.py/.sh (新規)
Stage5/checks/dummy/check_dummy_class_weight_tag.sh              (新規)
Stage5/checks/dummy/check_dummy_class_weight_ablation.py/.sh    (新規)
```

**Step F4（W-A再利用判定）:** S5-12 Run Aの学習時revisionと現revisionの`git diff`は上記
`train_stage5.sh`の命名/衝突ガード変更のみで、`train_stage5.py`への引数構築や`train_stage5.py`
本体・`stage5/`配下は無変更であることを確認した（学習挙動に影響する差分なし）。S5-12 Run A
（`EX260914`、teacher v7、GroupNorm、`bbox_noncontour_ignore`、`class_weight=[0.05963856,
1.9403615]`）の`config.json`を確認し、6節の固定条件をすべて満たすことを確認したため、これを
W-Aとして再利用した。

**Step F5〜F6結果（2026-09-14、ユーザー実機）:** W-B（`CLASS_WEIGHT=0.5,1.5`、同一初期checkpoint・
split・seed）で1 epoch smoke・5 epoch pilotとも完走した。train 162 files/715 samples、val 18
files/87 samplesはW-Aと一致、`label_policy: bbox_noncontour_ignore`（変換点数0、no-op）、
`class weight: [0.5, 1.5]`を確認した。output_dir名に`cw_manual_0p5_1p5`タグが正しく反映され、
既存runと衝突しなかった（naming修正の実run確認）。

epoch別のwindow単位running metrics（5 epoch pilot）:

| epoch | train loss | train F1 | train FP | train FN | val loss | val F1 | val FP | val FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3089 | 0.0291 | 4,514,560 | 1,209,511 | 0.1565 | 0.0000 | 0 | 142,147 |
| 2 | 0.1506 | 0.0000 | 0 | 1,295,189 | 0.1396 | 0.0000 | 0 | 142,147 |
| 3 | 0.1386 | 0.0000 | 0 | 1,295,189 | 0.1362 | 0.0000 | 0 | 142,147 |
| 4 | 0.1322 | 0.0000 | 0 | 1,295,189 | 0.1344 | 0.0000 | 0 | 142,147 |
| 5 | 0.1281 | 0.0000 | 0 | 1,295,189 | 0.1370 | 0.0000 | 0 | 142,147 |

epoch 2以降、train/valともFP=0かつFNが一定値に固定されており、positive予測が完全に消失した
（全点をbackgroundと予測）。valはepoch 1終了時点で既に崩壊している。

**Step F4本比較・Step F7結果（2026-09-14、ユーザー実機）:** `check_stage5_class_weight_ablation.sh`
をW-A（`EX260914`）・W-B（`EX260916`）に対して実行し、config parity（file list、resolved weight、
history finite性・epoch数、初期checkpoint path一致）はすべて合格した。既存`evaluate_stage5.sh`
（train sanity 3動画+validation 18動画、mean probability aggregation、threshold 0.5、`last.pt`=
`best.pt`＝epoch 5）で両armを評価した結果、W-Bの崩壊は学習ループのrunning metricsだけでなく、
公式の集計評価でも完全に再現された。

| split | run | TP | FP | FN | recall | precision | F1 | IoU | FPR | predicted positive数 | TP0動画数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train_sanity | W-A | 5,682 | 82,901 | 4,074 | 58.24% | 6.41% | 0.1156 | 0.0613 | 13.42% | 88,583 | 0/3 |
| train_sanity | W-B | 0 | 0 | 9,756 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.00% | 0 | **3/3** |
| validation | W-A | 33,531 | 830,987 | 41,502 | 44.69% | 3.88% | 0.0714 | 0.0370 | 11.71% | 864,518 | 1/18 |
| validation | W-B | 0 | 0 | 75,033 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.00% | 0 | **18/18** |

video-level（validation 18動画）: F1勝敗はW-A 17勝・W-B 0勝・1引き分け（全動画でW-Bの
predicted_positive_count=0）。W-A meanF1=0.0672/medianF1=0.0530に対しW-B mean/median F1とも0.0000。
train_sanity（3動画）も同様に3/3動画でTP0。

**Weighted CE denominatorの内訳（epoch 1、train split、既存`debug_*`診断を再利用、再実装なし）:**

| arm | weight比(pos:bg) | 生の点数比(bg:pos) | positiveが占める重み付きloss denominator比率 |
| --- | ---: | ---: | ---: |
| W-A | 32.54 | 79.55 | **29.03%** |
| W-B | 3.00 | 79.55 | **3.63%** |

background点はvalid点の約98.8%を占める（raw比 約79.5:1）。W-Aの重み比32.5はこの不均衡を部分的に
相殺し、positiveがweighted lossの約29%を占める状態を保つ。W-Bの重み比3.0では相殺が全く不十分で、
positiveはweighted lossのわずか3.6%しか占めず、CE lossを最小化する最も簡単な解が「常にbackgroundと
予測する」になったと考えられる。実際、W-Bのval側`debug_mean_logit_margin_positive_minus_background`
はepoch 1時点で-2.618（W-A: -0.991）と大きく負に振れており、崩壊が学習初期から起きていたことと
整合する。optimizer step数・empty-valid window数は両armとも正常（W-A/Bともepoch毎90 step、
empty-valid sample 0件）で、機構上の不具合ではなく重み設定そのものが原因と判断できる。

**仮説判断:** 依頼書9章の判定基準に対し、(1) FP/FPR・predicted positive率はW-Bで明確に低下した
（trivially、predicted positiveが0のため）が、(2) precision/F1/IoUは改善せず0へ落ち込んだ、
(3) recallがW-Aの44.69%(validation)から0%へ完全に崩壊した、(4) TP0動画数がW-Aの1/18から18/18へ
増加した、(6) video別勝敗・medianもW-B側の系統的改善を全く支持しない。「FP低下と引き換えに
recallが崩壊する場合、W-Bは採用しない」という9章の不採用分岐に明確に該当し、判定不能ではなく
**W-Bは不採用、W-A（強いauto由来weight）を維持**という明確な結論が得られた。production反映は
方針管理チャットの判断を待つ。詳細な報告は`docs/stage5/s5-13/stage5_s5_13_report_to_policy_chat.md`。

#### S5-13補足: Threshold-free診断と中間Class Weight（W-C）検証結果（2026-09-14）

方針管理チャットはW-A維持を正式決定（D-023）した上で、S5-14前の限定的な補足検証（D-024/D-025）を
指示した。新規checker`check_stage5_class_weight_threshold_free.py/.sh`は`evaluate_stage5.py`が
`SAVE_PREDICTIONS=1`で既に保存したprediction `.npz`（`prob_femur`）とH5 GTのみを使い、モデル再推論
・CUDA・torchを一切使わずに実装した（threshold-0.5 TP/FP/TN/FNが既存`h5_metrics.csv`と完全一致する
ことをfail-fast gateとして検証）。

**補足検証1（threshold-free診断、保存済みW-A/B再利用、再学習なし、2026-09-14）:**

| split | run | AUPRC | AUROC | max F1 |
| --- | --- | ---: | ---: | ---: |
| train_sanity | W-A | 0.0860 | 0.8519 | 0.1849 |
| train_sanity | W-B | 0.0707 | 0.8368 | 0.1566 |
| validation | W-A | 0.0574 | 0.8018 | 0.1162 |
| validation | W-B | 0.0367 | 0.7652 | 0.0885 |

validation positive比率は約1.05%であり、W-A/Bとも対応するランダム基準を大きく上回る。固定FPRでの
recall（validation）はFPR 1%でW-A 11.60%/W-B 8.40%、FPR 5%でW-A 28.35%/W-B 22.54%、FPR 10%で
W-A 41.16%/W-B 34.24%と、いずれもW-Bが一貫してW-Aを下回った。W-AのFPR（11.71%）以下で達成可能な
最大recallは、W-A 44.70%（threshold≈0.500、元のthreshold 0.5結果とほぼ一致し整合性チェックとして
機能）に対し、W-B 37.59%（threshold≈0.107）。video-level（validation 18動画）のAUPRC win countは
W-A 12・W-B 6だが、median AUPRCはW-A 0.0422・W-B 0.0427とほぼ同値（meanはW-A優位）で、S5-12でも
見られたmean/median不一致が再度現れた。

**判定:** 「discrimination消失」（ランダム水準への崩壊）にも「W-Aに近い」にも該当しない中間的結果。
W-Bのthreshold 0.5全negativeは主にcalibration shiftが原因（threshold再設定でrecall 37.59%まで
回復する）だが、discrimination自体も中程度（相対15〜35%程度）に劣化しており、純粋なcalibration
shiftだけでは説明しきれない。D-023（W-Bを現行threshold 0.5のS5-14 controlに採用しない）は維持し、
この結果はS5-15のthreshold診断用履歴候補として保持する。production thresholdの変更は本補足でも
行わない。

**補足検証2（W-C 5 epoch pilot、weight比16、新規5 epoch pilot1/2回、2026-09-14）:** W-A/Bと同一の
teacher v7・GroupNorm・`bbox_noncontour_ignore`・split・seed・初期checkpointで、
`class_weight=[0.11764706, 1.88235294]`を5 epoch学習した。config parityは
`check_stage5_class_weight_ablation.sh`で合格（class weight以外の差分なし）。

threshold 0.5固定評価（`evaluate_stage5.sh`、split aggregate）:

| split | run | recall | precision | F1 | IoU | FPR | TP0動画数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | W-A | 44.69% | 3.88% | 0.0714 | 0.0370 | 11.71% | 1/18 |
| validation | W-C | 17.52% | 6.70% | 0.0969 | 0.0509 | **2.58%** | **5/18** |

video-level（validation 18動画）: F1/IoU win countはW-A 9・W-C 8・1引き分けとほぼ互角だが、FPRは
18動画すべてでW-Cが低い。一方recallはmean 47.41%→21.25%、**median 44.68%→9.70%**と大幅に悪化し、
TP0動画数もW-Aの1/18からW-Cの5/18へ増加した。

threshold-free診断（保存済みprediction再利用、再学習なし）: validation AUPRC 0.0574(A)→0.0445(C)、
AUROC 0.8018(A)→0.7588(C)といずれもW-Cが下回った。さらに**同一FPRに揃えて比較すると、1%/5%/10%/
11.71%のすべての水準でW-AがW-Cのrecallを上回った**（例: FPR 11.71%でA 44.70% vs C 38.75%）。
これは、W-Cのthreshold 0.5での見かけ上のF1/precision/IoU改善が、識別能力（ROC/PR curveそのもの）の
向上ではなく、より保守的な暗黙operating pointへ移動した結果であることを示す。train_sanity
（3動画のみ、高分散）ではFPR 10%以上で逆転が見られたが、動画数が少なく信頼性は低い。

**仮説判断:** handoff文書9章の基準に照らし、FP/FPR低下は明確に満たすが、recall・TP0動画数の
明確な悪化、および同一FPR比較でのW-A優位（AUPRC/AUROC含む）により、「W-Aを維持する条件」に
複数該当する。video-level F1/IoU win countがほぼ互角である点も「改善が動画別傾向でも支持される」
とは言えない。**W-Cは不採用、W-A（強いauto由来weight）を維持**と判断し、結果が明確なため2回目の
新規5 epoch pilotは実施しなかった。S5-13本比較の結論（W-A維持）はこの補足でも変わらない。
production反映は方針管理チャットの判断を待つ。詳細な報告は
`docs/stage5/s5-13/stage5_s5_13_supplement_report_to_policy_chat.md`。

### 9.8 座標依存・Frame-level・Overlap Exposure診断（S5-14）

S5-13本比較・補足の結論（W-A `[0.05963856, 1.94036150]`を暫定class weightとして維持、D-023〜
D-025）を受け、train sanity positive-only PLYで観察された「似たXY位置のpositive集合が異なる
frame群へ反復する」現象について、座標事前分布（仮説A）・動画内時間位置（仮説B）・overlap
windowによるtraining exposure（仮説C）の3仮説を、再学習なしで切り分けた。固定baselineは
S5-12/S5-13 W-A epoch 5 checkpoint（teacher v7、GroupNorm 8 groups、`bbox_noncontour_ignore`、
window 16/stride 8、mean probability aggregation、threshold 0.5）。

#### Step H1・H2: 共有primitiveとteacher v7全180 H5のGT/exposure監査

`check_stage5_frame_spatial_overlap_diagnostics.py`（CPU/h5pyのみ、GPU不要）で、frame単位の
GT positive/background/ignore点数、class別window occurrence exposure倍率、positive frame run
統計を実装した。teacher v7全180 H5を監査した結果、positive exposure倍率（mean of per-video
means）1.8516、background exposure倍率1.6019となり、S5-08時点の概算値（それぞれ約1.90/1.69）と
近い水準で、window 16/stride 8構成下でも同様のoverlap構造が再現されることを確認した。

#### Step H2.5: XY座標provenance監査

Stage 5が使う最終H5には`pixel_xy`の正規化に使えるcrop/image寸法が保存されていないことが判明した
（中間pseudo3d H5にのみ`local_crop_top`等のroot属性が残る）。`check_stage5_xy_coordinate_
provenance.py`が、最終H5の`source_pseudo3d_h5`属性（または`pseudo3d_outputs/<date>/`配下の
basename fallback）から中間H5を解決し、`local_input_shape`属性から実寸法を取得する監査を実装した。
teacher v7全180 H5で監査した結果、**全180動画が`source_pseudo3d_h5`属性だけで解決でき
（basename fallback不要）、寸法は全180動画で(256, 256)に完全一致、bounds違反も0件**だった
（選択肢3、除外0件）。これにより、Step H3.1のcross-video XY正規化は`normalized_x = pixel_x /
255`、`normalized_y = pixel_y / 255`を動画除外なしで全180動画・固定21診断対象動画へ適用できる
ことが確定した。

#### Step H3・H3.1: Frame/XY診断

保存済みW-A predictionを再利用し（再推論不要）、固定21動画（train sanity 3+validation 18）で
frame単位のTP/FP/FN/precision/recall/F1/IoU/FPR、GT/predicted positiveのXY重心・重心距離
（空集合は0埋めせずNone）、正規化XY grid上のdensity・temporal recurrenceを算出した。

**仮説B（時間位置）:** validation動画のrecallは動画内相対decile 0（序盤）の64.2%からdecile 8
（終盤）の7.9%へ系統的に急落する一方、FPRはdecile間でほぼ一定（10.5%〜14.2%）だった。時間位置
による性能劣化はrecall側に集中し、FP側にはほぼ影響しない。

**仮説A（座標事前分布）:** grid解像度16でpooled（全21動画合算）density mapを構築し、GT positive
densityとFP density・predicted positive densityの空間相関（それぞれ0.568、0.604）を確認した。
より直接的な検証として、個々の動画でGT positiveが一切存在しないbinに絞り、動画横断でGTが密集
する「globally hot」bin（pooled GT密度の上位25%）と「globally cold」bin（下位25%）で
predicted positive率を比較したところ、**ローカルな根拠が皆無であるにもかかわらず、globally hot
binではglobally cold binの約8.7倍のpredicted positive/FP率**を示した（mean predicted:
320.8 vs 36.8）。ただし、bin単位のraw point密度で正規化した確認は未実施であり、点群サンプリング
密度自体の交絡を完全には排除できていない。

**PLY反復現象の定量化:** grid16でpredicted positiveがactiveなbin 2,192件のうち、36.1%がGT run
と重ならない複数の時間的に分離したpredicted run（`multiple_unmatched_runs`）を持ち、32.0%のbin
はGT runと一度も重ならない。

#### Step H4・H4.1: Per-window Context診断とparity gateインシデント

固定21動画についてW-Aでper-window probabilityを再取得し（S5-14で許可された唯一の再推論）、
既存W-A aggregate predictionとの厳密parity（vote count、training window occurrence count、
threshold 0.5 predicted class）をno-toleranceで検証する設計とした。

初回のGPU実行では、validation動画1件で1点だけpredicted classが不一致となり、fail-fastが
正しく発動した。実装チャットは当初「CUDA非決定性」と推測したが、方針管理チャットの精査と
コード確認により、**実際の原因はH4 checkerの集約・判定規約が正本`evaluate_stage5.py:
predict_h5()`と異なっていたこと**と判明した（`predict_h5()`は両クラスの確率をwindow単位で
float32化してから集約・2クラスargmaxで判定するのに対し、H4は再利用したS5-08
`run_overlap_forward()`がpositiveクラスのみをfloat64直接集約し`>=0.5`で判定していた。不一致点の
確率自体は両run間で完全一致していた）。`predict_h5()`と同じ集約規約を再現する
`canonical_predicted_class()`を実装して修正した結果、**全21動画でparity gateが完全一致した
（`max abs diff=0.000e+00`、除外・tolerance追加なし）**。

修正後の層別分析（`point_overlap_error_statistics.csv`、TP/FP/FN/TN×relative_frame_decile/
vote_count_bucket）では、window 16/stride 8下でvote_countは構造上1または2のみと判明した。

| vote_count | recall（validation） | FP率（background中） | FP disagreement rate | TN disagreement rate |
| --- | ---: | ---: | ---: | ---: |
| 1（重複なし） | 39.2% | 11.88% | 0%（重複なしのため定義上0） | 0% |
| 2（重複あり） | 45.3% | 11.64% | **34.1%** | **7.2%** |

**仮説C（overlap exposure）:** FP率はvote_count 1/2間でほぼ同一（11.88%→11.64%）であり、
「exposureが多いほどFPが増える」という単純な関係は支持されなかった。ただしvote_count=2の内部
だけで見ると、FPのdisagreement rate（34.1%）はTNの約4.7倍（7.2%）に達し、window間予測の
食い違いがFPへ偏るという限定的なメカニズムが見られた。Step H3.1のXY bin recurrenceとの
cross-checkでも、multiple unmatched runsのあるbin（791件）はないbin（1,401件）よりtraining
window occurrence約+10%・disagreement rate約+8%高いが、効果は小さい。

relative_frame_decile別のdisagreement/exposureは動画中盤でピーク・両端で低い対称パターンを
示し、これは仮説Bで確認したrecallの単調な低下とは形状が異なることから、仮説Bのrecall低下は
exposureパターンの副産物ではなく独立した時間効果と判断した。

#### Step H5: 集計と3仮説の最終判定

`check_stage5_s5_14_summary_export.py`がStep H2〜H4の出力を集約し、`frame_position_deciles.csv`
（frame単位metricsをsplit×decileで集計、未定義recallは0埋めせず除外）、`stage5_s5_14_summary.
json`、統合`video_summary.csv`、`vote_count_error_rates.csv`を生成し、bundle全体のprivacy
self-checkに合格した。

**3仮説の最終判定:**

| 仮説 | 判定 | 根拠 |
| --- | --- | --- |
| A. 座標事前分布 | 強く示唆される（density正規化確認が残課題） | globally hot binがglobally cold binの約8.7倍のpredicted positive/FP率（ローカルGT根拠皆無でも）。bin単位point密度での正規化は未実施。 |
| B. 時間位置・frame phase | 明確に支持される | recallがdecile 0の64.2%からdecile 8の7.9%へ単調に急落、FPRはほぼ一定。exposureパターンとは独立。 |
| C. overlap exposure | 限定的に支持される（主要因ではない） | FP率はvote_count間でほぼ一定だが、vote_count=2内でFPのdisagreement rateがTNの4.7倍、XY bin recurrenceとの弱い正の関連（+8〜10%）。 |

単一の仮説に単純化できない複合的な結論であり、座標事前分布（仮説A）が最も強い効果、時間位置
（仮説B）がrecallに限定した明確な効果、overlap exposure（仮説C）は補助的・限定的な効果という
整理となった。production設定（normalization/label policy/class weight/threshold/aggregation）は
本診断でも変更していない。次の改修（座標equivariance診断、inverse-occurrence loss weighting等）
は方針管理チャットの判断を待つ。詳細は`docs/stage5/s5-14/stage5_s5_14_report_to_policy_chat.md`。

**上記「3仮説の最終判定」表とD-030以前の記述は履歴として読むこと。** 方針管理チャットの精査で、
以下の方法論的な留保が指摘され（D-030）、S5-14補足（本節末尾のサブセクション）で再評価する
方針となった。

- 仮説Aの「約8.7倍」はmean predicted**点数**320.8対36.8の比較であり、valid background点数を
  分母とするFPR（率）ではない。bin内のbackground点数自体が多ければpredicted positive/FP点数も
  増えるため、分母を揃えない限り座標事前分布の学習を断定できない。
- 仮説Bのdecile 0（64.2%）とdecile 8（7.9%）の比較には対象動画・GT点数構成の差が含まれ得る。
  decile 3（70.4%）はdecile 0を上回っており、「単調な急落」という記述は不正確。同一動画内での
  前半/後半比較が必要。
- 仮説CのFP率（vote_count 1: 11.88%、vote_count 2: 11.64%）の差は小さく、vote_count=2内での
  FP disagreement rateの高さ（34.1%）は予測不一致と誤りの関連を示すのみで、training exposureが
  FPを増やす因果関係の証明にはならない。動画・時間位置を考慮した層別が必要。

#### 9.8.1 補足: 点密度補正と動画別・時間別再集計（S5-14補足、実装内容）

上記の留保を受け、`Stage5/checks/real_h5/check_stage5_s5_14_supplement.py`（Step S1〜S5）を
実装した。既存H5・保存済みprediction（`pred_label`、`vote_count`）・Step H3の`frame_metrics.csv`
のみを使い、新規学習・GPU再推論は行わない。

- **Step S1（分母付きXY bin統計）:** 動画×grid×bin単位で全点数・valid点数・valid positive/
  background数・ignore数・TP/FP/TN/FN・`FPR = FP / (FP + TN)`・`recall = TP / (TP + FN)`を
  再集計し、各動画のbin合計を既存`h5_metrics.csv`の confusion counts と bin-for-bin で突き合わせる
  （fail-fast）。分母0は null + availability flag とし、0埋めや未定義比率のepsilon補完は行わない。
- **Step S2（train-onlyのXY prior）:** train 162動画のGTのみから、raw GT-positive点数と
  valid点数で正規化したGT-positive率の2定義でgrid別分布を作成する。train sanityの3動画は
  対象動画自身の寄与をtrain集計から差し引いてから評価する（自己参照を避ける）。上位/下位25%の
  hot/cold bin判定は同値境界を座標順で恣意的に分割せず、上下境界が一致する場合はその定義の比較を
  未定義として報告する。各評価動画の GT positive数0の bin に限定し、hot/cold の FPR 差を
  点数加重・動画等重みの両方で比較し、cold FPRが0の場合は比を未定義としFPR差のみを使う。
- **Step S3（時間位置のpaired比較）:** 既存`frame_metrics.csv`から動画×decileのTP/FP/TN/FN・
  GT点数を再集計し、同一動画内の前半（decile 0-4）と後半（decile 5-9）の両方にvalid GT-positive
  点がある動画だけでrecallの後半-前半差を計算する。片側にしかGTがない動画はpaired比較から除外し
  理由を記録する。frame等重みmean recallと点数から計算するrecallは区別して両方保持する。
- **Step S4（exposure/disagreementの層別）:** 動画×時間区分でvote count 1/2のFPR・recallを
  保存済みprediction/H5から直接再構成する。`vote_count`は`window_occurrence_counts()`による
  純geometry再計算（モデルforward passなし）とも突き合わせて一致を確認する。per-windowの
  disagreementは、H4既存artifactが持つ周辺集計（動画×bin、動画×decile、動画×vote_count
  bucketの別々の集計）からjoint層別を復元できないため、捏造せず「未検証」として明記する。

Step S5として、raw点数差が大きくてもFPRが同じになる例／分母補正後もFPR差が残る例、
valid/ignore分離、分母0・cold FPR0・空binの扱い、train-only prior・train sanity自己除外・
quantile同値境界の扱い、動画構成が異なるとpooled傾向と動画内傾向が乖離する例、joint集計と
周辺集計の区別、既存confusion countsとの一致、匿名化出力の実video ID・host path検出を含む
synthetic統合テスト（`Stage5/checks/dummy/check_dummy_s5_14_supplement.py`、13ケース、CLI
end-to-end実行を含む）を、h5py/numpyを導入した検証環境で実施し全て合格した。
`py_compile`・`bash -n`・`git diff --check`も合格した。

#### 9.8.2 実機再集計の結果（2026-09-15）

ユーザー実機で`Stage5/checks/real_h5/check_stage5_s5_14_supplement.sh`を実行し、21動画
（train sanity 3 + validation 18）・train prior 162動画・grid 16/8で完了した。構造的な
`window_occurrence_counts()`再計算と保存済み`vote_count`の不一致は0件、bin合計と既存
confusion countsのparityも全動画・両gridで一致した。

**仮説A（座標事前分布）:** GT positive数0のbinに限定し、train-onlyかつtrain sanity自己参照
排除済みのpriorでhot/cold領域を定義し直したうえで、valid background点を分母とするFPRを
比較した。結果、grid{16,8}×prior定義{raw_count, rate}×split{train_sanity, validation}の
全8通りで、比較可能な動画（train_sanity 3/3、validation 18/18）**すべてがhot FPR > cold FPR**
となった。例えばgrid16・`rate`定義・validationでは、点数加重pooled hot FPR 24.7%に対しpooled
cold FPR 0.48%（動画別のFPR差は+17〜+36ポイントの範囲に一貫して収まり、符号反転は0件）。
当初の「約8.7倍」という指摘はmean predicted点数の比であって率の比ではないという留保は妥当
だったが、分母補正・GT-positive-0限定・train sanity自己参照排除という指摘された懸念を全て
反映した後も、**座標事前分布バイアスの構造的根拠は強く残った**。bin単位のraw point密度自体の
交絡（依頼書が指摘した残課題）は本補足でも未確認のまま。

**仮説B（時間位置）:** 同一動画内の前半(decile 0-4)/後半(decile 5-9)でrecallをpaired比較した
結果、validationは18動画中12動画が比較可能（片側decileにGT-positive点がない6動画は除外）で、
7動画が低下・5動画が上昇と方向が割れ、mean diff +1.2%・median diff -1.2%とほぼ0に収束した。
強い一貫した低下（-46.0%、-48.5%、-20.7%）が見られたのはtrain sanity 3動画（非代表サンプル、
n=3）のみだった。9.8節の「decile 0の64.2%からdecile 8の7.9%への単調急落」は、複数動画のframe
集合をdecile単位でpoolした集計であり、動画構成の影響を受け得るという当初の留保が実機データでも
裏付けられた形となった。**同一動画内paired比較では、validationにおける系統的な時間低下は
再現されなかった。**

**仮説C（overlap exposure）:** vote_count{1,2}のFPRを前半/後半で層別しても10.8%〜15.3%の
レンジに収まり、S5-14 core報告のvote_count単独比較（11.88%/11.64%）と整合する形で小さいまま
残った。per-windowのdisagreementのjoint層別（動画×時間区分×vote_count）は、既存H4 artifactの
周辺集計（動画×bin、動画×decile、動画×vote_countbucketの別々の集計）から復元不能なため、
捏造もGPU再推論による補完も行わず「未検証」のまま維持した。**exposureがFPRへ与える効果は
時間区分別でも小さいまま**で、disagreement側の因果関係についての結論は持ち越しとなった。

依頼書8章の分岐基準に照らすと、Aの偏りが分母補正後も動画別・両grid（16/8）で一貫して残ったため、
再学習なしの座標変換診断（local pixel座標とmodel入力XYZの違いを考慮したもの）を次案とする根拠が
最も強い。Bの低下は同一動画内では一般に再現されなかったため時間文脈の限定診断を同等優先度で
進める根拠は得られず、Cのexposure固有効果も時間区分別で残らなかったためinverse-occurrence loss
weighting比較案を優先する根拠も得られなかった。次の改修の最終選択は方針管理チャットの判断を待つ。
詳細な動画別数値・synthetic検証内容は`docs/stage5/s5-14/stage5_s5_14_supplement_report_to_policy_chat.md`を
参照。

#### 9.8.3 S5-14補足2: 座標変換診断（prediction equivariance、2026-09-15修正後の実機実行結果）

9.8.2節の結果（仮説Aの偏りが分母補正後も残存）を受け、方針管理チャットは再学習なしの座標変換
診断を委任した（Decision record D-033）。`Stage5/checks/real_h5/check_stage5_coordinate_
transform_diagnostics.py`が、W-Aの`normalize_xyz()`済みモデル入力XYZへ、window抽出前に
8条件（`identity`、`repeat_identity`、X/Y軸±0.1平行移動4条件、Z軸±15度回転2条件）のいずれかを
一括適用し、`evaluate_stage5.predict_h5()`と同じ集約規約で再推論する。`identity`/
`repeat_identity`は保存済みW-A予測とno-toleranceで一致することを要求し（全21動画で合格）、
全条件で`vote_count`が構造的`window_occurrence_counts()`再計算と一致することも検算した
（window所属はXYZ変換に依存しないという契約の確認）。この検算結果は下記の集計仕様修正の
影響を受けない。

**初回実機実行結果（2026-09-15、旧集計・診断履歴）:** 平行移動（±0.1、X/Y軸）はflip_rate中央値
0.1〜0.2%とほぼ無反応、回転（±15度、Z軸）はflip_rate中央値約4.5%（平行移動の約30倍）で
明確に反応し、hot bin FPRの減少幅が全体population平均の3〜5倍という、hot領域に偏った反応が
観測された。ただし、この初回集計はtrain sanity 3動画のhot/cold分類がS5-14補足1と同じ
leave-one-video-out自己除外を反映しておらず、また集計がtrain sanityとvalidationを分離
していなかったことが方針管理チャットの精査で判明したため、**正式な判断には使用しない**。

**集計仕様の修正（2026-09-15）:** `check_stage5_coordinate_transform_diagnostics.py`を修正し、
S5-14補足1の`prior_excluding_video()`（無改変で再利用）によるtrain sanity自己除外と、
train_sanity/validationのsplit別集計（21動画混合は参考値として明示的に分離）を実装した。
synthetic test 16件・static checkは合格。

**修正後の実機再実行結果（2026-09-15、split別、正式集計）:** ユーザー実機で再実行し、
正常終了・privacy scan 0件を確認した。T2 parityは両splitとも差分0を維持。
**validation（n=18、train sanityの自己参照問題を受けない代表的サンプル）単独でも**、
平行移動（±0.1）は全体・hot bin（grid16、`rate`定義）のFPR diff中央値がほぼ0（それぞれ
+0.003pt、+0.009pt）で方向も一定しないのに対し、回転（±15度）は全体flip_rate中央値が
平行移動の約25〜35倍、hot bin FPR diff中央値は`rotate_z_plus`で-2.88pt（18動画中2動画のみ
増加）、`rotate_z_minus`で-3.82pt（18動画中18動画が減少）という、方向の強く一貫した反応を
示した。**train sanityを分離した後もvalidation単独で同じ質的パターンが再現されたことから、
初回（split混合）の結果がtrain sanity（n=3）の外れ値的な影響で駆動されたものではないことを
確認した。**一方、train sanityの自己除外適用により数値自体は初回から変化しており
（例: train_sanity側hot bin FPR diff中央値は`rotate_z_plus`で-1.84pt、`rotate_z_minus`で
-0.96pt）、自己除外の適用が結果に実質的な影響を与えたことも確認された。

recallはvalidationで`rotate_z_plus`が中央値+0.16pt（9/18動画増加・9/18動画低下、ほぼ相殺）と
ほぼ無変化だが、`rotate_z_minus`は中央値-1.94pt（14/18動画が低下）と明確に低下しており、
hot bin FP減少を「回転による精度改善」とは解釈しない。正規化後に適用した平行移動の低感度は、
`normalize_xyz()`の中心化（正規化**前**の平行移動にのみ働く別の性質）とは無関係の独立した
経験的観測として記述し、hot bin側の減少とcold bin側の増加も、FPが保存量として
「再配分」されたことを意味する表現ではなく、別々に観測された変化として扱う。

**総合（確定的な結論は出さない）:** 平行移動でほぼ無反応・回転（特にhot bin）で明確な反応
という非対称性は、より大きく代表性のあるvalidationサンプル単独でも再現され、仮説A
（座標事前分布）に対する相関にとどまらない操作的な追加根拠として残る。ただし
（1）回転は動画重心から離れた点ほど平行移動より絶対移動量が大きくなる交絡があり感度倍率を
変換種類固有の効果と単純化できない、（2）PointNeXtは厳密な回転不変性を設計上保証しないため
方向依存の局所特徴量への感度である可能性を否定できない、（3）recallも同時に変化する
（特に`rotate_z_minus`で明確に低下する）ため「hot bin FP減少＝精度改善」と言えない、という
3つの留保により、「座標記憶の確定」「augmentation採用」への飛躍はしない。次工程
（augmentation単独5 epoch比較案の検討、またはW-A維持でのS5-15 pilot）の選択は方針管理チャット
が行う。詳細な動画別数値・split別テーブルは
`docs/stage5/s5-14/stage5_s5_14_supplement2_report_to_policy_chat.md`を参照。

#### 9.8.4 S5-14補足3: 残存集計・記録の確認（CPU専用、2026-09-15完了）

方針管理チャットの精査により、split合算confusion countsから算出するpooled（点数加重）
precision/recall/FPR/F1/IoUが未算出であること、F1/IoUの動画別diff中央値とsplit内中央値同士の
差が未区別であること、報告文に増減の取り違え等の誤記があることが指摘され、GPU再推論なし・
CPU専用の追加集計と記録訂正としてS5-14補足3が委任された。`Stage5/checks/real_h5/
check_stage5_coordinate_transform_reconciliation.py`を新規実装し、既存の
`coordinate_transform_point_metrics.csv`のみを入力として再集計した。

**pooled recallと動画別F1中央値の乖離:** `rotate_z_minus`について、pooled（点数加重）recallは
両splitで明確に低下した（train_sanity約-9.2pt、validation約-6.8pt。転記留保は9.8.5節）が、動画等重みのF1差分の
中央値（`median_of_per_video_diffs`）はほぼ変化しない〜わずかに正だった（validation +0.06pt、
18動画中11動画でF1改善）。pooled recallは各動画のGT positive数（TP+FN）を重みとする
recallの加重平均であり、背景点数や全点数で加重されるものではない。pooled FPRはGT background数
（FP+TN）で加重される。この分母別の集計と、動画ごとに均等な重みのF1統計では異なる側面が
見えており、precision改善（FPR低下だけでprecision改善が保証されるわけではない）がrecall低下を動画単位では
部分的に相殺している可能性がある。「hot bin FP減少」「pooled recallの低下」「動画別F1は
ほぼ中立」はいずれも矛盾しない別々の観測であり、単一の結論に単純化しない。

**報告文の訂正:** T2 parityの記述、grid8・raw_count定義での増減の取り違え、train_sanity
recall diffのmean/median取り違え（正しい中央値は`rotate_z_plus`-4.58pt、`rotate_z_minus`
-9.59pt）の3件を訂正した。再推論（初回・修正後の計336 video-condition相当）の事前承認記録は
本書上に確認できず、「承認記録未確認」として報告している（得られた数値自体を無効化する趣旨
ではない）。詳細・全数値は`docs/stage5/s5-14/stage5_s5_14_supplement2_report_to_policy_chat.md`の
「S5-14補足3 完了報告」を参照。

#### 9.8.5 S5-14全体の完了判断と残る留保（2026-09-15）

ユーザー判断によりS5-14 core（H1〜H5）と補足1〜3を完了として受け入れた（管理記録D-034）。
自己除外・split分離を反映した座標変換診断と、CPUによるpooled集計・動画別差分の検算を完了し、
追加GPU診断は終了する。補足3は実データCPU再集計、synthetic test 7件、static/privacy検証の
合格報告に基づく。管理チャットでの文書・コード確認と実装チャットでの実行検証は区別する。

**現時点の結論:** 限定した平行移動への反応は小さく、回転への感度はvalidation単独でも残る。
ただし−15度ではFPとrecallが同時に低下し、＋15度でもtrain sanityの動画別F1は2/3動画で悪化する。
局所特徴の方向依存・点の移動量・正規化後の分布変化を分離できておらず、座標暗記・不具合の確定、
推論時回転や学習時augmentationの採用根拠とはしない。pooled recallと動画別F1の違いを背景点数で
説明する旧報告の解釈は採用しない（正しい分母は9.8.4節）。乖離の原因自体は未特定である。

**数値転記の留保:** 補足3報告のvalidation pooled recallは表示値44.71%→37.88%であり、
その差は約-6.83ptだが、報告の差分欄は-6.81ptで一致しない。ほかの差分欄も含め、厳密な引用には
`coordinate_transform_split_pooled_metrics.csv`等の生成CSVとの照合が必要。CSV未照合のまま
片方を正しい値と断定せず、本書では当該低下を約-6.8ptとする。元報告は履歴として保持する。
これは記録上の留保であり、新たな補足実験やGPU再推論を要求するものではない。

**運用上の留保:** 把握済みのGPU実施量は初回・修正後の2回、計336動画条件相当。
ユーザー自身による実行と、管理チャットによる事前承認記録未確認を区別する。
その他のpreflight等の有無は未確認のまま保持するが、この理由だけで得られた数値を無効とはしない。

次候補は回転augmentationだけを変更する5 epoch比較。＋15度だけを事後的に選ばず、条件と判定基準を
事前固定する依頼を別途作成する。現段階は学習開始未承認であり、W-A・production設定・長期学習の
保留を維持する。完了判断の正本は`stage5_revision_management_record.md`6章およびD-034。

### 9.9 S5-15: 回転augmentation単独5 epoch比較（P1〜P3、2026-09-19完了）

固定条件はW-Aと同一（teacher v7、保存済みtrain162/val18、GroupNorm転移初期checkpoint
SHA-256 `55ec6e6b...438b`、固定class weight `[0.05963856, 1.94036150]`、GroupNorm 8 groups、
window16/stride8/tail、batch1/accumulation8、seed42、lr1e-3、weight_decay1e-4）。
R0は`augmentation=none`、R1は`random_z_rotation`（±15度、解決済みbase_seed 500042）で、
**両armの差はaugmentationのみ**。比較checkerがconfig parity・初期重みhash・file list一致
（W-A保存済みリストとの全64桁一致を含む）をFAIL 0で確認した。

評価は固定21動画（train sanity 3＋validation 18）のepoch5。両armとも`best.pt`と`last.pt`は
**評価モデルとして同一**（state_dict 63キー完全一致）であったため、依頼書6.2に従い
`last.pt`のみを評価した（42動画条件）。

#### pooled（点数加重、合算TP/FP/TN/FNから導出）

| split | 指標 | R0 | R1 | 差分 |
| --- | --- | ---: | ---: | ---: |
| validation | precision | 3.66% | 4.15% | +0.49pt |
| validation | recall | 45.08% | 32.41% | **-12.67pt** |
| validation | FPR | 12.53% | 7.91% | **-4.63pt** |
| validation | F1 | 6.77% | 7.36% | +0.59pt |
| validation | IoU | 3.51% | 3.82% | +0.32pt |
| train sanity | recall | 70.30% | 60.92% | -9.38pt |
| train sanity | FPR | 14.88% | 8.82% | -6.06pt |
| train sanity | F1 | 12.64% | 16.94% | +4.30pt |

ignore領域のpositive率（valid GT上のFPとは別集計）はvalidation 40.08%→28.62%、
train sanity 66.94%→21.99%といずれもR1で低下した。

#### 動画別（動画等重み）

| split | 指標 | median-of-diffs | diff-of-medians |
| --- | --- | ---: | ---: |
| validation | F1 | **+0.03pt** | +1.04pt |
| validation | recall | -6.17pt | **-24.25pt** |
| validation | FPR | -4.54pt | -4.29pt |
| train sanity | F1 | +1.66pt | +1.66pt |

validationのF1は**改善9／悪化9**（同値0）で拮抗。TP0動画数はvalidation 1→0、train sanityは0→0。
recallでmedian-of-diffsとdiff-of-mediansが大きく乖離しており、動画ごとのrecall変化が
不均一であることを示す（S5-14補足3で分離した2統計の意義が再び現れた）。

#### 判断

事前選定基準の機械判定は、基準1 satisfied（ただしcheckerが`marginal`と付す+0.03pt）、
基準2 satisfied、基準3 requires_judgment、基準4 satisfied、総合`requires_policy_chat_judgment`。
**R0を維持しR1を採用しない**と判断した（管理記録参照）。単一seed・5 epochの暫定結果であり、
F1約7%・IoU約3.5%という初期段階のモデル同士の比較であるため、50〜200 epochでの優劣を
予測するものではない。validation18動画は既に多数の方式選択に使用済みで独立testではない。

**PLY所見: 未実施。** 評価パイプラインはPLYを出力したが目視確認は行っていない。

#### 実施量と保存仕様の逸脱

学習はoptimizer更新900回（90/epoch×5×2 arm、予定値と一致）、forward/backward 3,575組/run。
GPU preflightは4回（うちStage B 1回はpreflightスクリプトの実装不具合による失敗。学習コードとは
無関係で、修正後の再実行1回で合格）。**学習runの失敗・再実行は0回。**

`train_stage5.sh`が環境変数`SAVE_EVERY`を素の代入で上書きしていたため、意図`SAVE_EVERY=1`に対し
実効`save_every=10`で動作し、**per-epoch checkpointが保存されなかった**。5 epoch分のmetricsは
history.jsonに残るが中間epochの重みは復元不能である。主比較はepoch5の`last.pt`で成立するため
再学習は行わず、逸脱として記録する。再発防止はcommit `cd3b886`。

## 10. 結論

本章以下は初期調査の履歴要約を含む。2026-09-15時点のS5-14最終判断は9.8.5節を優先する。

- PointNeXt-Sはtrain sanityで大腿骨周辺を学習しているが、valid background上のFPが多く、memorization確認としても未解決である。
- validationでは一部データを検出できるものの、median IoUはほぼ0であり、汎化性能は低い。
- 旧teacher v2・paddingありrunでは`best.pt`（epoch 45）が比較checkpoint中で最良だった。teacher v6・paddingなし5 epoch pilotでは`best.pt`はepoch 4だが、いずれも実用精度には達していない。
- train/validation全件のbatch integrity testに合格し、Dataset、overlap window index、collate、shuffle、multi-workerによるGT・点群の取り違えは強く除外された。
- padding parity testにより、可変長windowのzero paddingが実点logits、loss、gradient、BatchNorm統計を大きく変えることを確認した。
- batch size 8のpaddingあり学習とbatch size 1のpaddingなし評価が不一致であり、現在の学習結果を解釈する上で重大な交絡要因となっている。
- batch size 1 + gradient accumulationによるpaddingなし経路は、受入済みteacher v6の5 epoch pilotでも構造上の検査に合格した。一方でepoch 1から5にtrain lossが低下する間、validation lossは単調増加した。
- v6 5 epoch checkpoint評価では、bestのtrain sanity F1がwindow単位0.1026からoverlap平均後0.0304へ、validation F1が0.0455から0.0341へ低下した。200 epoch学習前にwindow間予測不一致とaggregation方式を定量化する。
- 反復するXY位置のpositiveは、overlapによる重複教師と座標事前分布で説明できる可能性があり、frame-level metricsで検証する必要がある。
