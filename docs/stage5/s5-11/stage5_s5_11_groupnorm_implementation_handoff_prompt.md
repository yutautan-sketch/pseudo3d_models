# Stage 5 S5-11: GroupNorm normalization比較 実装引き継ぎプロンプト

作成日: 2026-09-12

Stage 5の次の改修フローとして、PointNeXt-SのGroupNorm対応と短期比較基盤を実装してください。
本件はS5-11であり、normalization方式だけを切り分ける診断的な5 epoch比較です。GroupNormの
production採用を事前に確定する作業ではありません。

## 1. 最初に読む文書

次の順で確認してください。

1. `docs/stage5/stage5_revision_management_record.md`
   - 現在状態
   - S5-09、S5-10、S5-11
2. `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
   - 9.5節の実装事項B/C
   - 2026-09-12追記の実装事項D
3. `docs/stage5/s5-10/stage5_s5_10_report_to_policy_chat.md`
   - S5-10完了報告
   - 特に12章の判断待ち事項
4. `docs/stage5/FILES.md`
5. `Stage5/stage5/models/pointnext_s_segmentor.py`
6. `Stage5/external/PointNeXt/openpoints/models/layers/norm.py`

現在の進捗・採用済み判断は`../stage5_revision_management_record.md`、実装事項Dの仕様は
`../stage5_pointnext_s_training_evaluation_report.md`を正としてください。

## 2. Stage 5の現在状態

- model: official OpenPoints PointNeXt-SのStage 5 wrapper
- teacher: Stage 4 teacher v6
- input features: `intensity,confidence`
- window: frame size 16 / stride 8 / tailあり
- physical batch size: 1 window
- gradient accumulation: 8 windows、point-weighted normalization
- padding: PointNeXtへ入力しない
- loss: auto class weight付きCrossEntropyLoss、label smoothing 0.0
- optimizer: AdamW、lr `1e-3`、weight decay `1e-4`
- dropout: 0.0
- production inference: `model.eval()`
- production aggregation: mean probability
- threshold: 0.5
- 200 epoch学習: 保留

基準run:

```text
/mnt/data/3d_projects/stage5_runs/260908/
pointnext_s_EX260908_260711_w16_s8_bboxrankv6_cvatxmlinv_glocal_
ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad/
```

- train: 163 files / 729 windows
- validation: 18 files / 79 windows
- 主診断checkpoint: `best.pt`、epoch 4
- 固定評価: train sanity 3動画 + validation 18動画

## 3. 方針管理チャットで採用した4つの判断

以下は再検討事項ではなく、S5-11の前提として扱ってください。

### 判断1: recalibrated checkpointはproduction採用しない

validationでの改善が小さく、train sanityと動画別結果で符号が揃っていない。running statistics
recalibrationは主要なBN対策候補から外し、S5-10のdiagnostic artifactとしてのみ保存する。

### 判断2: S5-10の追加seed・追加動画検証は行わない

train sanity 3動画は小標本だが、validation 18動画でもTP0は10/18から改善しなかった。追加検証を
行っても「再較正だけでは不十分」という判断が変わる可能性は低いため、S5-10を拡張しない。

### 判断3: Label policyより先にS5-11を短く実施する

normalizationは依然として重大な交絡要因である。normalizationを固定せずにlabel policyを比較すると、
採用後のnormalizationへ結果を持ち越せない可能性がある。まずS5-11の短期比較を完了する。

### 判断4: S5-11の第一候補はGroupNorm

GroupNormはphysical batch size 1に依存せず、train/evalで同じ挙動となる。PointNeXtの1D/2D feature
tensorにも適用できるため、LayerNormより先に検討する。

## 4. S5-10報告12章の3つの判断事項への正式回答

### 回答1: S5-11の候補優先順位

running statistics recalibrationを主要候補から外し、次の順で扱う。

1. GroupNormを第一候補として5 epoch比較する。
2. GroupNorm不採用時にLayerNormを次候補とする。
3. per-window BatchNormは診断用referenceに限定する。
4. BN freezeとrecalibrated checkpoint運用は低優先度とする。

S5-09のper-window statistics相当のtrain-mode推論ではrecallとFPが同時に大幅増加したため、現在の
状態のままproductionへ採用しない。

### 回答2: train sanity悪化の扱い

S5-10のtrain sanity悪化を「noiseだった」と断定はしない。一方、validationでも系統的改善がなく、
S5-10のproduction不採用判断は十分に可能である。したがって「小標本を含む不安定な変化で、系統的
改善を示さない」と記録し、追加video・複数seedによるS5-10感度検証は行わない。

### 回答3: 後続作業の順序

```text
S5-11 GroupNorm normalization比較
  -> normalization方式を暫定固定
  -> S5-12 Label policy ablation
  -> S5-13 class weight比較
  -> S5-14 座標依存・frame-level診断
```

## 5. S5-11の目的

現行PointNeXt-SのBatchNormだけをGroupNormへ置き換え、physical batch size 1、可変点数window、
train/eval modeの条件に依存しない正規化へ変更した場合の学習安定性と固定評価性能を測る。

教師label、class weight、loss、threshold、aggregation、split、seed、window、入力featureを固定し、
normalization方式だけを比較する。

## 6. wrapper / CLI実装要件

PointNeXt-S wrapperとtraining/inference/evaluationのcheckpoint config復元経路へ、少なくとも次を追加する。

```text
--pointnext_norm batchnorm   # 既存既定値
--pointnext_norm groupnorm
--pointnext_norm_groups 8
```

要件:

1. 既存checkpointにfieldがない場合は`batchnorm`として解釈する。
2. 既存BatchNorm modelの構造、state dict、strict load、forward結果との後方互換性を維持する。
3. normalization指定をencoder、decoder、segmentation headの全てへ明示的に渡す。
4. decoder内部へOpenPoints既定のBatchNormを残さない。
5. `pointnext_norm`と`pointnext_norm_groups`をcheckpointの`config`へ保存する。
6. `infer_stage5.py`と`evaluate_stage5.py`はcheckpoint configから同じmodelを再構築する。
7. production既定値は`batchnorm`のままとする。

## 7. GroupNorm adapter

OpenPoints external cloneは変更しないでください。

OpenPointsの`create_norm()`には`gn`登録があるが、文字列`gn`を渡すだけではdimension suffixと
`nn.GroupNorm`のconstructor形式が一致しない。Stage 5 wrapper側へ、概ね次の契約を持つadapterを
実装する。

```python
Stage5GroupNorm(num_channels, num_groups=8, ...)
```

- 内部では`nn.GroupNorm`を使用する。
- `[B, C, N]`と`[B, C, *, *]`の両方をそのまま処理できる形にする。
- OpenPointsの`norm_args`へcallableとして渡す。
- `num_channels % num_groups != 0`では分かりやすいエラーで停止する。
- group数を暗黙に減らしたり変更したりしない。
- 現行width 32構成では8 groupsを初期値とする。

外部PointNeXtコードの直接編集やmonkey patchは避け、Stage 5 wrapper内で完結させる。

## 8. 初期parameterの統一

GroupNorm版をscratchから開始しない。現行teacher v6 BatchNorm controlと同じStage 5用S3DIS部分転移
checkpointを初期値として使う。

GroupNorm版へ読み込むもの:

- convolution / linear parameter
- Stage 5用入力層とbinary segmentation head
- shapeが一致するnorm affine `weight` / `bias`

除外を許可するもの:

```text
*.running_mean
*.running_var
*.num_batches_tracked
```

ロードしたkey、除外したBN buffer、missing、unexpected、shape mismatchをreportへ保存する。
許可したBN running buffer以外にmissing/unexpected/shape mismatchがあればfail-fastとする。
汎用loaderの`strict=False`だけで無言に読み飛ばしてはいけない。

これにより、GroupNormとBatchNormの比較でtrainable parameterの初期値を可能な限り一致させる。

## 9. 固定比較条件

| 項目 | 固定値 |
| --- | --- |
| teacher | Stage 4 teacher v6 |
| split | 既存runのtrain/val file listを再利用 |
| seed | 42 |
| train / validation | 163 files / 18 files |
| window | size 16 / stride 8 / tailあり |
| physical batch | 1 window |
| gradient accumulation | 8 windows、point-weighted normalization |
| features | `intensity,confidence` |
| loss | CrossEntropyLoss |
| class weight | `auto` |
| label smoothing | 0.0 |
| optimizer | AdamW |
| learning rate | `1e-3` |
| weight decay | `1e-4` |
| dropout | 0.0 |
| inference | `model.eval()` |
| aggregation | mean probability |
| threshold | 0.5 |

GroupNorm比較へlabel policy、class weight、threshold tuning、augmentation、座標feature、window、
aggregation変更を混ぜない。

既存teacher v6 BatchNorm 5 epoch pilotをcontrolとする。共通model構築経路を変更した場合は、
旧checkpointが既定`batchnorm`でstrict reloadでき、同一入力・seedのforward結果が変更前と一致することを
最初に確認する。

## 10. 推奨する実装・検証順

### Step D1: 実装前監査

- `pointnext_s_segmentor.py`のencoder/decoder/head config伝播を確認する。
- `train_stage5.py`、`infer_stage5.py`、`evaluate_stage5.py`のmodel kwargs/config復元箇所を確認する。
- 既存dummy/checkスクリプトの配置規則に合わせて新規testを設計する。
- external PointNeXtへ変更が入らない方針を確認する。

### Step D2: BatchNorm後方互換

- field未指定時に従来と同じBatchNorm modelが構築されること。
- encoder/decoder/headのmodule typeとstate dict keyが変わらないこと。
- 既存checkpointのstrict loadが通ること。
- 同一入力・同一seedで変更前baselineとforward parityが成立すること。

### Step D3: GroupNorm構造test

- GroupNorm modelを構築できること。
- 全normalization siteがGroupNormへ置換されていること。
- `_BatchNorm` moduleが0個であること。
- GroupNorm module数が期待値と一致すること。
- 不正なgroup数をfail-fastで拒否すること。
- dropout 0.0、同一入力・seedで`train()`/`eval()` logitsが許容誤差内で一致すること。

### Step D4: dummy training / checkpoint test

- dummy overlap window batchでforwardする。
- CE loss、backward、optimizer stepがfiniteであること。
- checkpoint configにnorm設定が保存されること。
- 保存checkpointをstrict reloadできること。
- reload前後のlogits parityが成立すること。

### Step D5: S3DIS部分転移test

- BN controlと同じStage 5用S3DIS部分転移checkpointを使用する。
- trainable parameterが期待どおり読み込まれることをkey単位で確認する。
- 除外がBN running bufferだけであることをreport/assertする。
- 初期化済みGroupNorm checkpointを元checkpointとは別に保存する。

### Step D6: 実H5 1 epoch smoke

- 既存v6 splitを使用する。
- physical batch 1、gradient accumulation 8、window 16/8を維持する。
- loss、gradient、metricsがfiniteであること。
- best/last/config/metrics/checkpoint保存とstrict reloadを確認する。

### Step D7: 5 epoch GroupNorm pilot

- 同じsplit、seed、初期parameter、学習条件で5 epoch実行する。
- `best.pt`だけでなくepoch 5の固定checkpointも残す。
- 既存BatchNorm controlと同じ評価pipelineを実行する。
- train sanity 3動画とvalidation 18動画を評価する。

## 11. 評価項目

epoch中のtrain metricsだけで結論を出さず、保存checkpointを同じ`eval()`・mean aggregation経路で
比較する。

最低限記録するもの:

- TP、FP、TN、FN
- precision、recall、F1、femur IoU
- FPR、FNR
- predicted positive count/rate
- TP0動画数
- video-level mean/median F1・IoU
- ignore領域上のpositive予測率
- train/eval logits parity
- normalization module count
- checkpoint transfer report
- BatchNorm controlとの差分

## 12. 判定基準

GroupNormはtrain/eval一致を満たすだけでは採用しない。

- validation集約F1だけの微増で採用しない。
- TP0動画数またはvideo-level median F1/IoU、recallの改善を重視する。
- FP/FPRとpredicted positive率が許容不能に増えていないことを確認する。
- train sanityが崩壊していないことを確認する。
- split集約値とvideo別変化の符号が一致しない場合、系統的改善と判定しない。

結果分岐:

1. GroupNormが構造testを満たし、train sanity/validation双方で有望:
   normalization候補として暫定採用し、同方式を固定してS5-12へ進む。
2. train/eval差は解消するが精度が悪化:
   GroupNormを不採用とし、LayerNormを次候補として検討する。
3. BatchNormとの差が小さい:
   normalizationを主要因から下げ、既定方式を変更せずS5-12へ進む。
4. 結果が不明瞭:
   事実と未確定事項を方針管理チャットへ返す。production判断を実装チャットで行わない。

## 13. 非対象・禁止事項

本事項では次を行わない。

- recalibrated checkpointのproduction接続
- per-window train-mode推論のproduction採用
- 200 epoch学習
- Label policy変更
- class weight変更
- threshold tuning
- production aggregation変更
- Dice / Focal / Hard Negative Mining追加
- 座標augmentationやfeature ablation
- Stage 4 teacher変更
- external PointNeXt cloneの直接編集
- unrelatedな既存変更のrevert

worktreeはdirtyであり、本件以外のユーザー変更が存在する。対象外の変更を削除・revertしないこと。

## 14. 文書更新と完了報告

実装・検証結果は次へ追記する。

1. `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
   - 実装事項Dへ実施日、実装、test、結果、仮説判断を追記
2. `docs/stage5/stage5_revision_management_record.md`
   - S5-11の状態、結果、次の分岐を更新
3. `docs/stage5/FILES.md`
   - 新規checker、script、artifact構造を追加

最後に、方針管理チャットへ共有するMarkdownを`docs/stage5/s5-11/`へ作成する。最低限、次を含める。

- 変更ファイル
- 後方互換test
- GroupNorm構造とtrain/eval parity
- transfer key数と除外理由
- smoke/full pilotの実行条件
- BatchNorm controlとの主要metrics比較
- video-level結果とTP0動画数
- GroupNorm採否に関する事実
- 未確定事項
- production未変更の確認
- 方針管理チャットで判断が必要な事項

まず既存コードと実装事項Dを確認し、短い実装方針を提示してください。重大な仕様上の不明点がなければ、
後方互換性を保ったGroupNorm対応、静的test、dummy/checker実装まで進め、GPUと実H5を必要とする実行
コマンドをユーザーへ提示してください。
