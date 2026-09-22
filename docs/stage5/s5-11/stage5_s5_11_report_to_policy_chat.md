# Stage 5 S5-11 完了報告: GroupNorm normalization比較

作成日: 2026-09-12

方針管理チャットの判断待ち事項があるため、実装チャットからの完了報告として共有する。
数値の正本は`docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.5節
「実装事項D」、進捗管理は`docs/stage5/stage5_revision_management_record.md` S5-11に記録済み。

## 1. 経緯

S5-10（BatchNorm recalibration診断）の結果を受け、方針管理チャットが次の4判断を行い、
S5-11（GroupNorm normalization比較）の実装を指示した
（`docs/stage5/s5-11/stage5_s5_11_groupnorm_implementation_handoff_prompt.md`）。

1. recalibrated checkpointはproduction採用しない
2. S5-10の追加seed・追加動画検証は行わない
3. Label policyより先にS5-11を短く実施する
4. S5-11の第一候補はGroupNorm

## 2. 変更ファイル

```text
Stage5/stage5/models/norm_layers.py                                        (新規)
Stage5/stage5/models/pointnext_decoder_patch.py                            (新規)
Stage5/stage5/models/pointnext_s_segmentor.py                              (変更)
Stage5/train_stage5.py / infer_stage5.py / evaluate_stage5.py              (変更)
Stage5/train_stage5.sh                                                     (変更)
Stage5/checks/dummy/check_dummy_pointnext_s_training.py/.sh                (変更)
Stage5/checks/dummy/check_dummy_pointnext_s_groupnorm.py/.sh               (新規)
Stage5/checks/transfer/check_stage5_batchnorm_to_groupnorm_transfer.py/.sh (新規)
docs/stage5/stage5_pointnext_s_training_evaluation_report.md               (9.5節に実装事項Dを追記)
docs/stage5/stage5_revision_management_record.md                          (S5-11の記録・状態更新)
```

`train_stage5.py`/`infer_stage5.py`/`evaluate_stage5.py`の**既定値は`batchnorm`のまま変更していない**。
`--pointnext_norm`/`--pointnext_norm_groups`は新規追加のopt-inフラグである。

## 3. 実装の要点

- `Stage5GroupNorm`（`nn.GroupNorm`を直接継承）: `.weight`/`.bias`がBatchNormと同じdotted key位置に
  載るため、既存BatchNorm checkpointからのnorm affine重み転移がkey remapなしで成立する。
- OpenPointsの`create_norm()`は`norm_args["norm"]`にcallableを渡すとdimension接尾辞処理を経ず
  `norm(channels, **kwargs)`を直接呼ぶ仕様を利用し、external PointNeXt cloneは変更していない。
- **実装中に発見・修正したバグ**: official OpenPointsの`PointNextDecoder`は`norm_args`/`act_args`を
  `**kwargs`で受け取るが実際には一切使用せず、decoderは常に`FeaturePropogation`のhardcoded既定値
  `{'norm': 'bn1d'}`を使っていた（Step D2/D3の初回実行でdecoder内8 moduleがGroupNorm化されない
  形で発覚）。external cloneは変更せず、`PointNextDecoder`を継承した`Stage5PointNextDecoder`を
  `pointnext_decoder_patch.py`へ実装し、`_make_dec()`だけをoverrideして解消した。
  `norm_args={"norm": "bn"}`は元のhardcoded既定値と同一の`nn.BatchNorm1d`を構築するため、
  batchnorm既定経路の互換性には影響しない（既存checkpointのstrict loadに影響しないことを
  static解析および後述のStep D2で確認）。

## 4. 実行したテストと成否（すべてユーザー実機、GPU）

| Step | 内容 | 結果 |
| --- | --- | --- |
| self-test（2種） | `Stage5GroupNorm`/`resolve_pointnext_norm_args`/decoder norm伝播のCPU only synthetic test | 合格 |
| D2/D3 | BatchNorm後方互換・GroupNorm構造・train/eval parity（dummy H5、CUDA要） | 合格（バグ修正後） |
| D4 | dummy training smoke（batchnorm/groupnorm両方） | 合格 |
| D5 | BatchNorm→GroupNorm初期化転移 | 合格 |
| D6 | 実H5 1 epoch smoke + 先行評価 | 合格（参考値） |
| D7 | 実H5 5 epoch pilot + 固定評価 | 合格（主要比較点） |

## 5. GroupNorm構造とtrain/eval parity（Step D2/D3）

- batchnorm既定modelのBatchNorm module数: 17
- groupnorm指定modelのGroupNorm module数: 17（1:1置換、encoder 8 + decoder 8 + head 1）
- train()/eval() logits max abs diff: **0.000e+00**（完全一致。GroupNormはrunning statisticsに
  依存しないため、physical batch size 1でもtrain/evalで挙動が変わらないというS5-11の狙いを
  構造面で裏付ける）

## 6. transfer key数と除外理由（Step D5）

既存Stage5 BatchNorm S3DIS部分転移checkpoint（114 key）からGroupNormモデル（63 key）へ転移。

- loaded keys: **63/63**（target側のmissing・shape不一致は0件）
- excluded keys: **51**（17 BN module × `running_mean`/`running_var`/`num_batches_tracked`の
  3 bufferと完全一致。それ以外の予期しない未使用keyは0件）
- forward loss: 0.720794（有限）

51 = 17×3という正確な一致は、fail-fast転移ロジックが意図通り動作していることを裏付ける。

## 7. smoke/full pilotの実行条件

- teacher v6、padding-free、window 16/8、physical batch 1、gradient accumulation 8、
  train 163 files/729 windows、validation 18 files/79 windows、seed 42、初期checkpointは
  Step D5のGroupNorm転移checkpoint（BatchNorm control同様S3DIS部分転移由来）
- Step D6: 1 epoch。Step D7: 5 epoch（`best_metric=iou_femur`によりepoch 4がbest、
  BatchNorm controlと同一epoch）
- class weight`[0.05963856, 1.94036150]`はBatchNorm controlと完全一致（データ経路は不変）

## 8. BatchNorm controlとの主要metrics比較（Step D7、同一epoch数=4）

| split | model | recall | precision | F1 | IoU | FP | TP0動画数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train_sanity | BatchNorm control | 4.17% | 2.39% | 0.0304 | 0.0154 | 16,602 | 1/3 |
| train_sanity | GroupNorm | 42.33% | 8.84% | 0.1463 | 0.0789 | 42,589 | **0/3** |
| validation | BatchNorm control | 3.87% | 3.04% | 0.0341 | 0.0173 | 85,785 | 10/18 |
| validation | GroupNorm | 34.11% | 5.04% | 0.0878 | 0.0459 | 446,335 | **0/18** |

validation median F1は0→0.0818。validation lossはBatchNorm controlが5 epoch全てで単調増加した
（0.5985→0.9208）のに対し、GroupNormは0.539〜0.509の範囲で安定した。

**FP増加の性質:** FPは+5.2倍（85,785→446,335）に増加したが、precisionは低下せずむしろ
3.04%→5.04%へ改善した。recallの増加率（+8.8倍）がFPRの増加率（+5.2倍）を上回っており、
S5-09（train-mode）やS5-10（recalibration）で見られた「recallだけが増えFPが同等以上に急増する」
calibration shiftパターンとは異なる。

## 9. video-level結果とTP0動画数

- validation: TP0動画数**10/18→0/18**。18動画全てで最低93点以上のTPを検出した。
- train_sanity: TP0動画数**1/3→0/3**（歴史的にTP0だった動画を含め全て検出）。

## 10. GroupNorm採否に関する事実

12章の判定基準に照らすと、「GroupNormが構造testを満たし、train sanity/validation双方で有望」
（分岐1）に該当する事実が確認された。

- 支持: TP0動画数の解消、video-level median F1/IoUの改善、precisionとrecallの同時改善、
  validation lossの安定化（単調増加の消失）。
- 支持（構造面）: train/eval logits完全一致、GroupNorm module 17個の1:1置換、
  transfer key数の正確な一致。

## 11. 未確定事項

- FP/FPRの増加（+5.2倍）が実運用（Stage 6入力として）許容できる範囲かは、本checkerの範囲では
  判定していない。
- 5 epochのみの比較であり、200 epoch時点の挙動は未検証。
- Label policy・class weightとの相互作用は未検証。
- GroupNormのgroup数（8）や学習率など、正規化方式変更後の再チューニングは行っていない。

## 12. productionへ未反映であること

`train_stage5.py`/`infer_stage5.py`/`evaluate_stage5.py`の既定値は引き続き`batchnorm`である。
production aggregation（mean probability）、threshold、label policy、class weightはいずれも
変更していない。GroupNorm転移checkpoint・5 epoch pilot checkpointはいずれもproduction
checkpointへ接続していない。

## 13. 方針管理チャットで判断が必要な事項

1. GroupNormをnormalization方式の暫定候補として採用し、S5-12（Label policy ablation）を
   GroupNorm条件で進めるか（12章の想定フロー）。
2. FP/FPR増加（+5.2倍）を、この時点で許容範囲と判断するか、追加の閾値調整や比較検証
   （例: threshold tuningはS5-11の対象外だったため、閾値0.5以外での比較）を先に行うか。
3. 200 epochへの本格移行前に、GroupNorm条件でも同様の短期pilotによる中間確認を挟むか
   （9章S5-15の既定方針）。

再学習、loss変更、threshold tuning、production aggregationおよびnormalization設定の変更は、
本報告時点では実施していない。
