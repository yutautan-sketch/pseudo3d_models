# Stage 5 S5-13補足: Threshold-free診断と中間Class Weight確認 実装引き継ぎ

作成日: 2026-09-14

S5-13の必須2-arm class weight ablationは完了しました。本書は、完了報告の3つの判断事項に対する
方針管理チャットの正式回答と、S5-14前に行う限定的な補足検証の実装・実行依頼です。

本補足はclass weightの広範な再探索ではありません。保存済みcheckpointを最大限再利用し、新規の
5 epoch pilotは原則1回、最大2回に制限してください。

## 1. 最初に読む文書

次の順で確認してください。

1. `docs/stage5/stage5_revision_management_record.md`
   - S5-13の完了記録
   - 「S5-13補足 Threshold-free診断と中間class weight限定確認」
   - Decision record D-023〜D-025
2. `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
   - 9.7節「実装事項F」のS5-13結果
3. `docs/stage5/s5-13/stage5_s5_13_report_to_policy_chat.md`
   - S5-13完了報告と判断待ち3事項
4. `docs/stage5/s5-13/stage5_s5_13_class_weight_ablation_implementation_request.md`
   - S5-13の固定条件、比較checker、privacy要件
5. `docs/stage5/FILES.md`
6. `docs/stage5/data_construct.md`

現在の進捗・決定事項は`../stage5_revision_management_record.md`、S5-13の数値は
`../stage5_pointnext_s_training_evaluation_report.md`を正本としてください。

## 2. S5-13結果の要約

S5-13ではteacher v7、GroupNorm 8 groups、`bbox_noncontour_ignore`、同一split・seed・初期checkpointを
固定し、次の2条件を各epoch 5時点で比較しました。

| Arm | class weight | positive:background比 | positive weighted denominator比 |
| --- | --- | ---: | ---: |
| W-A | `[0.05963856, 1.94036150]` | 約32.54 | 29.03% |
| W-B | `[0.5, 1.5]` | 3.0 | 3.63% |

W-Bはthreshold 0.5で全点をbackgroundと予測しました。

| split | arm | recall | F1 | IoU | FPR | predicted positive | TP0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train sanity | W-A | 58.24% | 0.1156 | 0.0613 | 13.42% | 88,583 | 0/3 |
| train sanity | W-B | 0.00% | 0.0000 | 0.0000 | 0.00% | 0 | 3/3 |
| validation | W-A | 44.69% | 0.0714 | 0.0370 | 11.71% | 864,518 | 1/18 |
| validation | W-B | 0.00% | 0.0000 | 0.0000 | 0.00% | 0 | 18/18 |

config parity、train/val file list、初期checkpoint、epoch数、finite性、評価対象point集合はcheckerで
一致を確認済みです。W-Bの結果は比較条件の不一致や実装上の故障ではありません。

## 3. 3つの判断事項への正式回答

### 判断1: W-Aを暫定class weightとして採用する

**回答: 採用します。**

W-B `[0.5,1.5]`は現行のthreshold 0.5でvalidation recall 0%、TP0 18/18となり、S5-13の受入基準を
明確に満たしません。W-Bは不採用とし、W-A `[0.05963856,1.94036150]`をS5-13補足およびS5-14の
暫定controlとして採用します。

これはproduction設定の最終採用ではありません。GroupNorm、label policy、class weight、thresholdの
production採否は後続段階で決定します。既定`CLASS_WEIGHT=auto`も本補足では変更しません。

### 判断2: 中間weightを限定的に追加検証する

**回答: 実施します。**

W-Aのweight比32.54とW-Bの比3は間隔が広く、この2点だけではW-Aが最適とは確定できません。一方、
報告書で例示された`[0.25,1.75]`（比7）はpositiveのweighted denominator占有率が約8.1%に留まり、
W-Bと同じ全negative崩壊へ近づく懸念があります。

初回の中間weightは次の1条件に固定します。

```text
W-C class weight = [0.11764706, 1.88235294]
positive:background weight比 = 16
想定positive weighted denominator占有率 = 約16.7%
```

W-C以外のweightは同時に追加しません。W-C結果だけでは境界的で、もう1点の確認により判断が確定すると
合理的に説明できる場合のみ、2回目の5 epoch条件を方針管理チャットへ提案してください。

### 判断3: S5-13本比較を完了とし、S5-14前に補足を行う

**回答: S5-13の必須A/B比較は完了と認定します。**

ただし、固定threshold 0.5での全negativeが、順位識別の消失なのかcalibration shift中心なのかは
未分離です。S5-14へ進む前に、本書のthreshold-free診断とW-C限定pilotを「S5-13補足」として
実施してください。補足完了後に暫定weightを確定し、S5-14へ進みます。

## 4. 補足検証の目的

1. 保存済みW-A/W-Bについて、discriminationとthreshold/calibration shiftを分離する。
2. W-Aよりpositive weightを弱めつつ、W-Bほどpositive寄与を失わないW-Cでglobal FP/FPRを
   抑えられるか確認する。
3. S5-14で固定するclass weightを、追加計算を抑えた上で決定する。

threshold-free診断はproduction threshold tuningではありません。公式operating-point比較は引き続き
threshold 0.5を使用し、production thresholdの選定はS5-15まで行いません。

## 5. 固定する条件

W-CはW-A/W-Bとclass weight以外を完全に揃えてください。

| 項目 | 固定値 |
| --- | --- |
| teacher | v7 `bboxrank_v7_cvat_authoritative_crop_quality_v1`、180 H5 |
| split | S5-12/S5-13と同一のtrain 162 / validation 18 |
| label policy | `bbox_noncontour_ignore` |
| model | official OpenPoints PointNeXt-S Stage 5 wrapper |
| normalization | GroupNorm、8 groups |
| initialization | W-A/W-Bと同一のGroupNorm S3DIS部分転移checkpoint |
| seed | 42 |
| features | `intensity,confidence` |
| window | size 16 / stride 8 / tailあり |
| physical batch | 1 window |
| gradient accumulation | 8、point-weighted normalization |
| padding | PointNeXt入力へ追加しない |
| loss | CrossEntropyLoss、label smoothing 0.0 |
| optimizer | AdamW、lr `1e-3`、weight decay `1e-4` |
| dropout | 0.0 |
| epochs | 5 |
| inference | `model.eval()` |
| aggregation | mean probability |
| primary threshold | 0.5 |

## 6. 補足検証1: Threshold-free診断

### 6.1 対象

保存済みのW-A/W-B epoch 5 checkpointを再利用します。再学習しないでください。

- train sanity: S5-12/S5-13と同じ3動画
- validation: 同じ18動画全件
- point prediction: overlap windowのmean probability集約後
- GT: teacher v7 native valid label

可能なら既存のprediction artifactを再利用し、不足する場合だけcheckpointから再推論してください。

### 6.2 必須出力

split aggregateとvideo-levelで次を記録してください。

- AUPRC / Average Precision（主指標）
- AUROC（補助指標。class imbalanceが強いため単独で採否に使わない）
- GT positive/background別`prob_femur`のmean、median、p10、p25、p75、p90、p95、p99
- PR curveとthreshold列
- threshold別TP、FP、TN、FN、precision、recall、F1、IoU、FPR、predicted positive率、TP0
- maximum F1とその診断threshold
- W-AのFPR 11.71%以下で達成可能な最大recall
- 固定FPR候補（例: 1%、5%、10%）でのrecall/precision
- video別AUPRC、最大F1、score分布

threshold候補は十分な分解能で評価しつつ、全pointをthresholdごとに再forwardしない実装にしてください。
予測確率を一度保存・集約し、sort/cumulative countまたは同等の方法で計算します。

### 6.3 parityとprivacy

- threshold 0.5のTP/FP/TN/FNが既存S5-13評価と完全一致することをfail-fast gateにする。
- W-A/W-Bで評価対象video、point数、valid/positive/background数が一致することを確認する。
- nonfinite probability、範囲外probability、point index重複/欠落があれば停止する。
- shareable出力では実video IDと絶対host pathを匿名化する。
- private mappingとshareable metricsを分離する。

### 6.4 解釈

- W-BのAUPRCとclass別score分布も崩壊している場合、weight比3ではdiscrimination自体が失われたと
  判断できる。
- W-BのAUPRCや順位分離がW-Aに近い場合、threshold 0.5での全negativeはcalibration shiftの寄与が
  大きい。W-BをS5-14 controlには採用しないが、S5-15のthreshold診断用履歴候補として保持する。
- diagnostic best thresholdをproductionや通常評価の既定値へ反映しない。

## 7. 補足検証2: W-C 5 Epoch Pilot

### 7.1 事前確認

既存`train_stage5.sh`のclass-weight tag、output collision guard、manual weight処理を再利用します。
W-C開始前に、出力directory名がW-A/W-Bと衝突せず、次の値を起動ログで確認してください。

```text
CLASS_WEIGHT=0.11764706,1.88235294
LABEL_POLICY=bbox_noncontour_ignore
POINTNEXT_NORM=groupnorm
EPOCHS=5
```

必要なら1 epoch smokeを行えますが、5 epoch本実行と同一設定であり、出力衝突を避けてください。
1 epoch smokeは5 epoch pilotの回数上限には数えませんが、既存機能で既にmanual weight経路は確認済み
なので、コード変更がなければ省略を優先します。

### 7.2 学習とcheckpoint

- W-Cを5 epochだけ学習する。
- epoch 5の`last.pt`または固定epoch checkpointをprimary comparisonにする。
- `best.pt`はsecondaryとして保存し、best epochとmetricを明記する。
- W-A/W-Bの保存済みrunは再学習しない。
- class weight以外のconfig parityを既存checkerで検証する。
- resolved weightとpositive weighted denominator占有率を記録する。

### 7.3 固定評価

W-C epoch 5をW-A/W-Bと同じtrain sanity 3動画・validation 18動画へ適用してください。

primary operating point:

```text
aggregation = mean probability
threshold = 0.5
model mode = eval
```

次をsplit aggregate・video-levelで比較します。

- TP、FP、TN、FN
- precision、recall、F1、IoU、FPR、FNR
- predicted positive count/rate
- TP0動画数
- mean/median F1、IoU、precision、recall、FPR
- W-A/W-Cの動画別勝敗
- epoch別train/validation metrics
- weighted denominator内訳
- 第6節と同じAUPRC/AUROC/PR curve/score分布

## 8. 検証量の制限

本段階はS5-13の補足です。次を厳守してください。

1. 新規5 epoch pilotは**原則W-Cの1回**とする。
2. 技術的な再実行または追加の中間weightを含めても、**新規5 epoch実行は合計2回まで**とする。
3. 2回目を実行する前に、必要性、変更する唯一の変数、期待する判定を報告する。
4. no-weight、複数seed、広いweight gridは実施しない。
5. 50〜200 epochへ延長しない。
6. W-A/W-Bを再学習しない。
7. threshold-free診断のための再推論は5 epoch実行回数に含めないが、forwardの重複は避ける。

W-Cが明確に採用または不採用と判断できた時点で補足を終了し、2回目を消化する必要はありません。

## 9. 採否基準

### W-Cを暫定採用候補とする条件

- threshold 0.5でW-AよりFP/FPRとpredicted positive率が低下する。
- recallが大幅に低下せず、TP0動画数がW-Aの1/18から増加しない。
- video-level median F1/IoUが悪化せず、aggregateだけでなく動画別傾向も改善を支持する。
- AUPRCまたはPR curveが、単なるscore shiftではなく有効なtrade-offを支持する。
- train sanityとvalidationの改善方向が大きく反転しない。

### W-Aを維持する条件

- W-Cが全negativeまたはそれに近い状態へ崩壊する。
- TP0動画数またはrecallが明確に悪化する。
- FP低下がprecision/F1/IoU改善につながらない。
- threshold-free指標とvideo-level結果がW-Cを支持しない。
- 結果が混合的で、限定検証の範囲内ではW-Cの優位性を証明できない。

W-A/W-Cが混合的な場合は、保守的にW-Aを維持してください。補足で完全な最適weight探索を目指さず、
採用根拠が明確な場合だけW-Cへ変更します。

## 10. 非対象・禁止事項

- production既定値の変更
- production thresholdの選定または変更
- mean以外のproduction aggregation採用
- teacher、split、label policy、normalization、group数の変更
- learning rate、scheduler、window、sampling、feature、augmentationの変更
- Focal loss、Dice loss、Hard Negative Mining
- S5-12 label policy Run Bの再実行
- S5-07〜S5-11のteacher v7全面再実行
- source H5の書き換え
- 新規5 epoch pilotを3回以上実行すること
- 長期学習へ移ること

実装中に比較へ影響する不具合を発見した場合は、最小修正と実験を分離し、修正後のconfig parityを
再確認してください。

## 11. 実装物の推奨構成

既存checkerを拡張できる場合は重複実装を避けてください。責務が異なる場合は、例えば次を追加します。

```text
Stage5/checks/real_h5/check_stage5_class_weight_threshold_free.py
Stage5/checks/real_h5/check_stage5_class_weight_threshold_free.sh
```

最低限必要なもの:

- probabilityとGTの収集または既存prediction artifact読込
- threshold-free/threshold curve集計
- threshold 0.5 baseline parity
- W-A/W-B/W-Cのpoint-set・config parity
- aggregate/video-level CSV/JSON
- privacy self-check
- synthetic test（既知score/GTでAUPRC、curve、tie処理、empty class、all-negative predictionを確認）

既存`evaluate_stage5.py`のproduction挙動は変更せず、診断checker側で実装することを優先してください。

## 12. 実施順序

1. 関連文書と既存S5-13 checker/outputを監査する。
2. threshold-free checkerの仕様と既存artifact再利用可否を整理する。
3. synthetic testとstatic checkを実装・実行する。
4. 保存済みW-A/W-Bでthreshold-free診断を実行する。
5. 診断結果を記録する。ここではproduction thresholdを決めない。
6. W-Cのconfig parityを事前確認する。
7. W-Cを5 epochだけ学習する。
8. epoch 5固定checkpointをthreshold 0.5とthreshold-free指標で評価する。
9. W-A/W-B/W-Cをaggregate・video-levelで比較する。
10. 採否基準に従いW-A維持またはW-C候補を報告する。
11. 文書を更新し、方針管理チャットへ完了報告を返す。

## 13. 文書更新と完了報告

結果は次へ追記してください。

- `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
  - 9.7節のS5-13補足として、実装、parity、threshold-free結果、W-C結果、仮説判断を記録
- `docs/stage5/stage5_revision_management_record.md`
  - S5-13補足の状態、結果、暫定weight判断を更新
  - 必要なら新しいDecision recordを追加
- `docs/stage5/FILES.md`
  - 新規checker/artifactの索引を追加

方針管理チャットへの完了報告は次へ作成してください。

```text
docs/stage5/s5-13/stage5_s5_13_supplement_report_to_policy_chat.md
```

完了報告には次を含めてください。

1. 変更ファイルと検証コマンド
2. W-A/W-B threshold-free aggregate・video-level結果
3. W-Bのdiscrimination消失とcalibration shiftの判断
4. W-Cのconfig parity、学習履歴、threshold 0.5結果
5. W-Cのthreshold-free結果
6. W-A/W-B/W-C比較表
7. 新規5 epoch実行回数（上限2回を守ったこと）
8. W-A維持またはW-C採用候補の根拠
9. productionへ未反映であること
10. S5-14へ進む前に方針管理チャットで判断が必要な事項

補足の実装・評価が完了しても、実装チャットだけでproduction設定を変更したり、2回目の追加weight、
S5-14、長期学習へ自動的に進んだりしないでください。

