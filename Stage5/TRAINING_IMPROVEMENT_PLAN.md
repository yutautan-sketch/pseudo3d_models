# Stage 5 学習改善方針

## 1. 目的

本書では、Stage 5 PointNeXt-Sの学習・推論を改善するために、今後行う調査と実装の順序を定める。

当面はDice Loss、Focal Loss、Hard Negative Miningなどの複雑な処理を追加しない。まず評価方法、window選択、損失集計を信頼できる状態にし、各変更の効果を個別に判断できるようにする。

## 2. 現状の整理

目視結果と保存済みmetricsから、モデルは訓練データを学習できている一方、validation動画への汎化性能が低いと考えられる。

`grid=1`データによる200 epoch学習では、以下の結果が得られた。

- train femur IoU最高値: 約`0.448`
- 学習終盤のtrain recall: 約`0.974`から`0.979`
- val femur IoU最高値: epoch 27で約`0.063`
- epoch 200のval femur IoU: 約`0.014`
- valid点が存在しないtrain window: 約`54.5%`
- valid点が存在しないval window: 約`70.1%`
- valid点比率: train約`26.3%`、val約`19.2%`

`grid=1`では点数が約4倍になったが、独立した教師情報は増えていない。クラス比とauto class weightも従来データとほぼ同じだった。

`point_label=-1`または`valid_mask=False`の点は、CrossEntropyLossと通常のmetricsの両方から除外されている。そのため、ignore領域をpositiveと予測しても直接的なペナルティは発生しない。

valid点がないwindowも現在はPointNeXt-Sのforwardを通過する。損失はゼロになるが、BatchNorm統計やAdamWの更新に影響する可能性がある。また、padding点もPointNeXt-Sへの入力に含まれており、公式モデルにはpadding maskを渡していない。

## 3. 基本原則

1. 一度に変更する主要因は一つとする。
2. 初期調査ではCrossEntropyLossと`label_smoothing=0.0`を維持する。
3. 保持したwindow内の点は削除しない。現段階ではrandom samplingを導入しない。
4. frame windowの境界生成には`frame_order`のみを使用する。
5. labelに基づくwindow除外は、frame window生成後の学習・評価用sample選択として実施する。
6. 推論ではannotationの有無にかかわらず全windowを処理する。
7. window除外後は1 epoch当たりのbatch数が変わるため、epochだけでなくoptimizer step数と累積valid点数でもrunを比較する。

## 4. Phase 1: 評価・可視化の整備

### 4.1 固定評価データ

同じH5を使ってcheckpoint間と実験間を比較する。

- 訓練データ例: `20250626_124212_7300`
- validationデータ: 学習runの`val_files.txt`に記録された全ファイルを使用する

各H5について、少なくとも以下を比較する。

- `checkpoint_epoch_0030.pt`
- `checkpoint_epoch_0100.pt`
- `checkpoint_epoch_0200.pt`
- `best.pt`

これにより、学習途中の未収束、訓練データの記憶、未見動画への汎化失敗を区別する。

### 4.2 評価出力

アノテーション済みH5の推論評価に、以下を追加する。

- valid領域のprecision、recall、F1、femur IoU
- validなpositive点数、background点数とその比率
- ignore領域におけるpositive予測数・予測率
- ignore領域における平均`prob_femur`
- H5単位のmetrics
- window単位のmetrics
- `point_indices`でoverlap aggregationした後の元点index単位metrics

ignore点には信頼できる正解labelがないため、ignore領域のpositive予測率は誤分類率ではなく診断値として扱う。

### 4.3 PLY可視化

既存の確率PLYに加えて、以下を色分けした診断用PLYを出力する。

- True Positive
- valid background上のFalse Positive
- False Negative
- True Negative
- ignore領域上のpositive予測
- その他のignore点

### 4.4 完了条件

同じ訓練H5・validation H5についてcheckpoint間の挙動を比較でき、valid領域のFalse Positiveとignore領域のpositive予測を別々に確認できること。

### 4.5 評価スクリプト

`evaluate_stage5.sh`から以下を自動実行する。

- 学習runの`train_files.txt`と`val_files.txt`を使用する
- train sanityは`20250626_124212_7300`と、固定seedで選んだ追加2ファイルを使用する
- validationは`val_files.txt`の全ファイルを使用する
- checkpointごとにH5単位・window単位・元点index集約後のmetricsを保存する
- Stage 4のraw PLYとannotation-colored PLYを評価対象分だけ共通参照ディレクトリへコピーする
- GT positive点だけを含む共通参照PLYを保存する
- probability PLY、診断カテゴリPLY、predicted positive点だけのPLYを保存する
- 全checkpointのsplit/H5/window metricsを比較用CSVへ結合する

デフォルトの比較checkpointは、現在の150 epoch runに合わせて以下とする。

- `checkpoint_epoch_0030.pt`
- `checkpoint_epoch_0100.pt`
- `checkpoint_epoch_0150.pt`
- `best.pt`

200 epoch runを評価する場合は、bash実行時の`CHECKPOINT_NAMES`を変更する。

## 5. Phase 2: Empty-valid windowの除外

### 5.1 除外条件

最初に`frame_order`のみからframe windowを生成し、その後に学習用sampleを以下の条件で絞り込む。

```python
valid_count = ((labels != ignore_index) & valid_mask).sum()
keep = valid_count > 0
```

positive点の存在は条件にしない。validなbackground点だけを含むwindowも有効な教師データとして保持する。

保持されたwindow内のignore点は空間的contextとしてPointNeXt-Sへ入力する。削除するのはvalid点が一つもないwindow全体だけとする。

### 5.2 CLI

以下のような明示的なオプションを追加する。

```text
--skip_empty_valid_windows
--no_skip_empty_valid_windows
```

初期設定は以下とする。

- train: `train_stage5.sh`で有効化
- validation: 当初は全windowを保持するが、all-empty batchは安全にskipする
- inference: 常に無効

### 5.3 ログ

以下を`config.json`またはepoch metricsへ記録する。

- 生成されたwindow数
- 保持されたwindow数
- skipされたwindow数と比率
- all-empty batch数
- optimizer step数
- 学習に使用した累積valid点数

### 5.4 安全処理

all-empty batchが残った場合は、model forward、backward、`optimizer.step()`を実行しない。これにより、教師信号がないbatchによるBatchNorm更新とAdamW weight decayを防ぐ。

### 5.5 完了条件

- frame windowの境界が従来と一致する。
- 保持されたwindow内の点が削除されていない。
- train DataLoaderにempty-valid sampleが含まれない。
- MLP baselineとPointNeXt-Sの両方でforward・loss計算が通る。
- dummy trainingと実H5による1 epoch testが通る。

## 6. Phase 3: Loss集計と診断ログ

学習目的はclass weight付きCrossEntropyLossのままとし、この段階では勾配計算方法を変更しない。

現在のepoch lossはbatch lossの単純平均であり、batchごとのvalid点数差やall-empty batchの影響を受ける。そのため、学習に用いるlossとは別に、valid点数またはclass weight適用後の有効な分母を考慮したmonitoring lossを追加する。

例えば以下を区別して保存する。

- `optimizer_loss_mean`: 従来のbatch loss平均
- `valid_weighted_loss`: valid点数等を考慮した監視用loss

併せて以下を維持・追加する。

- valid点数
- ignore点数と比率
- positive/backgroundのlabel数
- positive/backgroundの予測数
- valid positive点上の平均positive確率
- valid background点上の平均positive確率
- ignore点上の平均positive確率

validation lossがall-empty batchによって見かけ上小さくならず、metricsから集計分母を確認できる状態を完了条件とする。

## 7. Phase 4: 短期A/B学習

`grid=1`データ、固定seed、固定train/validation split、同じS3DIS由来初期checkpoint、同じloss設定を使用する。

- Run A: 現行のwindow処理
- Run B: `skip_empty_valid_windows=True`

最初は30から50 epochで比較し、以下を確認する。

- optimizer step数
- 累積valid点数
- val femur IoUとF1の最高値
- best checkpointのepochとstep
- trainとvalidationの性能差
- ignore領域のpositive予測率
- 固定H5のPLY結果

Run Bでは1 epoch当たりのbatch数が減るため、epoch数だけでは比較しない。短期runが正常で改善傾向を示した場合にのみ、100から200 epochの学習へ進む。

## 8. Phase 5: 点密度とPointNeXt近傍設定

`grid=1`では点密度が増えている一方、PointNeXt-Sは`radius=0.1`、`nsample=32`のままである。固定`nsample`がより狭い実空間の点だけで埋まり、必要な空間contextを失っている可能性がある。

empty-valid window対策の評価後、以下を短期runで比較する。

```text
radius=0.10, nsample=32
radius=0.10, nsample=64
radius=0.10, nsample=128
radius=0.15, nsample=64
```

可能な限り`radius`と`nsample`の片方だけを変更する。この比較中はrandom samplingや恣意的な点削除を導入しない。

## 9. Phase 6: Annotationとignore方針

empty-valid windowを除外しても、ignore領域のpositive予測を直接抑制することはできない。問題が残る場合はStage2to4側のannotation定義を見直す。

以下の3種類を区別する。

- 信頼できる大腿骨positive点
- 信頼できるbackground点
- 境界または未アノテーションで判定不能なignore点

意味的に確実なbackground領域だけをignoreからlabel 0へ変更する。曖昧な輪郭境界はignoreのまま残す。この変更はStage 5のHard Negative Miningではなく、annotation生成時のlabel定義として実装する。

## 10. 後回しにする内容

以下は上記の調査が完了するまで追加しない。

- Focal Loss
- Dice Loss
- Hard Negative Mining
- annotation依存のpositive oversampling
- モデル改善の代わりに行う確率threshold調整
- 系統的な誤分類を隠すための幾何学的post-processing

これらは将来の候補として残すが、現状のデータと学習挙動の解析を妨げないようにする。

## 11. 直近の実装順序

1. H5単位・ignore領域を含む評価出力を追加する。
2. 診断カテゴリ別PLY出力を追加する。
3. frame window生成後に`skip_empty_valid_windows`を実装する。
4. all-empty batchをmodel forward前にskipする。
5. valid点を考慮したlossとoptimizer stepの診断ログを追加する。
6. dummy testと実H5による1 epoch testを行う。
7. 30から50 epochのA/B学習を行う。
8. A/B結果から、PointNeXt近傍設定とannotation方針のどちらを次に検討するか決定する。
