# Stage 5 S5-12 完了報告: GroupNorm固定Label policy ablation

作成日: 2026-09-14

方針管理チャットの判断待ち事項があるため、実装チャットからの完了報告として共有する。
数値の正本は`docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.6節
「実装事項E」、進捗管理は`docs/stage5/stage5_revision_management_record.md` S5-12に記録済み。

## 1. 経緯の要約

S5-11でGroupNormが暫定採用された後、S5-12としてBBox内・positive contour外の点を
ignoreのまま学習するRun Aと、backgroundとして学習に含めるRun Bを比較する予定だった。
実装後のteacher v6全181 H5監査（Step E3）で、2ファイル・16点のstray ignore（保存BBoxの
どれにも属さないignore点）を検出し、fail-fastが正しく機能した。原因調査の結果、これは
Stage 2のcrop窓固定と被写体移動による既知の失敗パターン（`local_crop_tracking_drift`）と
判明し、Stage 4管理チャットへ調査を依頼した
（`docs/stage2to4/stage4/stage5_s5_12_stage4_crop_quality_investigation_request.md`）。

Stage 4側は該当2動画のうち1動画を全体除外、もう1動画は該当1 frameのみを無効化し、退化BBoxの
境界条件バグも修正した、補正済みteacher v7（180 H5）を構築した
（`docs/stage2to4/stage4/stage5_s5_12_stage4_crop_quality_correction_report.md`）。Stage 5側もv7でStep E2〜E4を
独立に再実行し、stray ignore=0を確認した上でStep E5〜E7（GPU学習・評価）を実施した。

## 2. 変更ファイル

```text
Stage5/stage5/utils/label_policy.py                                          (新規)
Stage5/stage5/utils/h5_io.py                                                 (変更)
Stage5/stage5/datasets/pseudo3d_pointcloud_dataset.py                        (変更)
Stage5/train_stage5.py / train_stage5.sh                                     (変更)
Stage5/checks/dummy/check_dummy_label_policy.py/.sh                         (新規、Step E2)
Stage5/checks/real_h5/check_stage5_label_policy_bbox_preflight.py/.sh       (新規、Step E3)
Stage5/checks/real_h5/check_stage5_label_policy_dataset_parity.py/.sh       (新規、Step E4)
Stage5/checks/real_h5/check_stage5_label_policy_ablation_eval.py/.sh        (新規、Step E7/E8)
docs/stage5/stage5_pointnext_s_training_evaluation_report.md                 (9.6節実装事項E)
docs/stage5/stage5_revision_management_record.md                            (S5-12記録)
```

実装チャット側の不具合として、`check_stage5_label_policy_ablation_eval.sh`の`RUN_DIR`/
`EVALUATION_DIR`既定値が旧v6 S5-11 pilot runへハードコードされたままになっており、Run Aの
評価実行中に古いv6の`val_files.txt`（修正前のstray点を含む）を誤って参照してfail-fastした。
`CHECKPOINT`/`REFERENCE_H5_METRICS_CSV`の親ディレクトリから自動導出するよう修正済み。

`train_stage5.py`/`infer_stage5.py`/`evaluate_stage5.py`の**既定値は変更していない**
（`--label_policy`既定は引き続き`bbox_noncontour_ignore`）。

## 3. teacher v7 provenance preflight結果（Stage 5側の独立確認）

v7 inventory（180 H5）からtrain 162 / val 18のfile listを新規生成し
（`train_stage5.py`の`split_train_val_paths`、val_fraction 0.1、seed 42を直接呼び出し、
実際の学習開始時と同一ロジック）、`check_stage5_label_policy_bbox_preflight.py`で監査した。

| 項目 | 値 |
| --- | ---: |
| 監査ファイル数 | 180 |
| ignore点総数 | 262,244 |
| stray ignore点数（BBox外） | **0** |
| BBox内target点数 | 262,244（ignore点総数と完全一致） |

schema属性は`contour_teacher_schema=bboxrank_v7_cvat_authoritative_crop_quality_v1`で
想定通り。Stage 4の修正がStage 5の契約を完全に満たすことを独立に確認した。

## 4. 対象点・変換点数

Step E5（1 epoch smoke、Run B）のログより:

- train 162 files/715 samples、val 18 files/87 samples（実際のsplitと一致）
- 変換点数（`-1 -> 0`）: train 238,377点、val 23,867点

## 5. Dataset/batch parity

`check_stage5_label_policy_dataset_parity.py`をv7全180ファイルで実行し合格した。

- 総window数: 802
- BBox non-contour target点数: 262,244（Step E3の集計と一致）
- Run A（points/features/frame_order/point_indices/window境界）とRun Bで、labelsが監査済み
  対象点以外で完全一致することを確認した。

## 6. class weightが両runで同一であること

両runとも明示的に`--class_weight 0.05963856,1.94036150`を指定し、`auto`再計算は行っていない
（config/ログで確認済み）。

## 7. Run A/Bのconfig差分

固定: teacher v7、split（train 162/val 18、seed 42）、GroupNorm 8 groups、初期checkpoint
（GroupNorm S3DIS部分転移）、window 16/8、physical batch 1、gradient accumulation 8、
class weight固定値、epochs 5。

差分: `--label_policy`のみ（Run A=`bbox_noncontour_ignore`、Run B=`bbox_noncontour_background`）。

## 8. 実行条件

- Step E5（1 epoch smoke、Run B）: `EX260914smoke`。完走、finite値、checkpoint保存確認。
- Step E6（Run A/B各5 epoch）: Run A=`EX260914`、Run B=`EX260915`。両runとも5 epoch完走。
  val IoUはいずれもepoch 5で最大（Run A 0.0377、Run B 0.0352）のため、`best.pt`と`last.pt`は
  同一epochを指す。

## 9. epoch 5固定checkpointの共通target metrics

`canonical_v6`（native teacher v7 label、既存`evaluate_stage5.py`とのmean baseline parity
gate付き、両run合格）:

| split | run | TP | FP | recall | precision | F1 | IoU | FPR | TP0動画数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train_sanity | A（ignore） | 5,682 | 82,901 | 58.24% | 6.41% | 0.1156 | 0.0613 | 13.42% | 0/3 |
| train_sanity | B（background） | 6,405 | 79,822 | 65.65% | 7.43% | 0.1335 | 0.0715 | 12.93% | 0/3 |
| validation | A（ignore） | 33,531 | 830,987 | 44.69% | 3.88% | 0.0714 | 0.0370 | 11.71% | 1/18 |
| validation | B（background） | 32,376 | 864,239 | 43.15% | 3.61% | 0.0666 | 0.0345 | **12.17%** | 1/18 |

train_sanityはRun Bが全指標で優位（recall/F1/IoU向上かつFP減少）。validation aggregatedは
Run Aがわずかに優位で、FPRはRun Bでむしろ悪化した。

`bbox_noncontour_as_background` target（BBox non-contour点をbackground+validとして加えた
共通target）でも同様の傾向だった（詳細は評価レポート9.6節）。

best checkpointはlast.ptと同一epochのため、secondary比較は主要比較と一致する。

## 10. BBox non-contour regionの確率・positive率

| split | run | target点数 | predicted positive数 | predicted positive率 |
| --- | --- | ---: | ---: | ---: |
| train_sanity | A | 4,966 | 2,202 | 44.34% |
| train_sanity | B | 4,966 | 2,026 | 40.80% |
| validation | A | 23,867 | 9,697 | 40.63% |
| validation | B | 23,867 | 7,429 | **31.13%**（相対-23%） |

動画別方向性（21動画）: 12動画で改善、6動画で悪化、3動画で同値。

## 11. TP0、video median、動画別改善/悪化

- TP0動画数: train_sanity・validationとも両run同一（train_sanity 0/3、validation 1/18）。
- canonical F1のvideo単位勝敗（validation 18動画）: Run A 12勝、Run B 5勝、1引き分け。
- video median F1/IoU: Run Aが0.0530/0.0272、Run Bが0.0623/0.0322とRun Bが上回る
  （F1分布の裾が低い動画に偏っていることによるもので、勝敗数の逆転とは矛盾しない）。

## 12. PLY目視確認用の出力先

本ラウンドでは`--export_ply_alias`を使用しておらず、PLY診断出力は生成していない。必要な場合は
`check_stage5_label_policy_ablation_eval.sh`の`EXPORT_PLY_ALIAS`/`PLY_OUTPUT_DIR`で
個別動画を指定して再実行できる（出力は`private_DO_NOT_SHARE`側、非共有）。

## 13. 支持・棄却・未確定の仮説

**支持:** Run Bの学習は、狙い通りBBox non-contour領域でのpositive予測率を低下させた
（validation 40.63%→31.13%、相対-23%）。これは「対象領域をbackgroundとして学習することで、
その領域のFPを抑制できる」という核心メカニズム仮説を支持する。train_sanityでは
recall/F1/IoU/FPすべてでRun Bが優位だった。

**棄却（暫定）:** 「Run Bで全体FP/FPRも低下する」は支持されなかった。validation FPRは
むしろ悪化した（11.71%→12.17%）。対象領域点数がvalid background全体（約720万点）の
0.3%程度と小さく、領域内の改善が全体指標に埋もれた可能性がある。

**未確定:** validation aggregatedのrecall/F1/IoUはRun Aがわずかに優位、video median F1/IoUは
Run Bが優位という、split集約とvideo別統計が一致しない状態であり、5 epoch・21動画という規模では
系統的な優劣を判定できない。200 epochへ向かう際にこの傾向が維持されるかは未検証。

## 14. productionへ未反映であること

`train_stage5.py`/`infer_stage5.py`/`evaluate_stage5.py`の既定値（`label_policy=
bbox_noncontour_ignore`、`normalization=batchnorm`、mean probability aggregation、
threshold 0.5）はいずれも変更していない。Run A/B checkpointはいずれもproduction
checkpointへ接続していない。

## 15. S5-13へ持ち越すpolicy候補

judgment基準（9.6節）の分岐1（有望）・分岐2（悪化）のいずれにも明確に該当せず、
分岐3（差が小さい）〜分岐4（判定不能）に近い結果のため、実装チャット側でRun A/Bの
どちらを暫定policyとして固定するかを決定していない。方針管理チャットの判断を仰ぐ。

## 16. 方針管理チャットで判断が必要な事項

1. Run A（ignore、現状維持）とRun B（background）のどちらをS5-13以降の暫定label policyと
   するか、あるいは判定不能として追加検証（より長いepoch数、または200 epoch pilotまで
   両方持ち越す）を先に行うか。
2. 「region内positive率は改善するが全体FP/FPRは改善しない」という結果を、領域点数が
   全体の0.3%程度と小さいことによる統計的な埋没とみなし、より長い学習やより大きい対象領域
   （例えばBBox全体ではなく複数frameにまたがる集計）で再評価する価値があるか。
3. Stage 4のteacher v7移行（D-018/D-019）を受けて、S5-07〜S5-11（teacher v6ベース）の
   一部または全部をv7で再実行し、v7ベースの新しいbaselineを確立するか。

再学習、loss変更、threshold tuning、production aggregationおよびlabel policy/normalization
設定の変更は、本報告時点では実施していない。

## 17. 関連文書

- `docs/stage5/s5-12/stage5_s5_12_label_policy_implementation_handoff_prompt.md`（実装引き継ぎ）
- `docs/stage2to4/stage4/stage5_s5_12_stage4_crop_quality_investigation_request.md`（Stage 4への調査依頼）
- `docs/stage2to4/stage4/stage5_s5_12_stage4_crop_quality_correction_report.md`（Stage 4からの修正報告）
- `docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.6節（実装事項E、数値正本）
- `docs/stage5/stage5_revision_management_record.md` S5-12（進捗記録、Decision record D-018/D-019）
