前回までに、Stage 5の暫定モデルとして以下のファイルを作成しました。

* `base_segmentor.py`
* `model_factory.py`
* `pointnext_segmentor.py`

ただし、モデル構造を再確認した結果、現在の `pointnext_segmentor.py` は **公式PointNeXt-S準拠モデルではない** と判断しました。

現在の実装は、以下のような構成です。

```text
stem:
  1x1 Conv / MLP

blocks:
  ResidualPointMLPBlock × depth

head:
  1x1 Conv / MLP

optional:
  global max context
```

これは、PointNeXt風のinverted residual MLPを持つ軽量baselineではありますが、公式PointNeXt-Sに必要な以下の構成を含んでいません。

```text
- set abstraction
- ball query / neighborhood grouping
- farthest point sampling
- hierarchical downsampling
- feature propagation decoder
- PointNextEncoder
- PointNextDecoder
- SegHead
```

したがって、現在のモデルを **PointNeXt-S公式実装** として扱うのは不正確です。

## 今後の方針

現時点では、以下の方針で進めてください。

```text
Step 1:
  現在の暫定モデルを、公式PointNeXt-Sではなく、
  Stage 5 pipeline疎通確認用のlightweight baselineとして整理する。

Step 2:
  このbaselineで、Dataset / DataLoader / loss / metrics / checkpoint / inference / H5出力 / PLY出力の
  end-to-end pipelineが通ることを確認する。

Step 3:
  1〜5 epoch程度、または50〜200 iteration程度の短い学習テストだけ行う。
  目的は性能評価ではなく、pipeline sanity check。

Step 4:
  pipeline確認後、公式PointNeXt-S / OpenPoints実装をcloneまたはsubmodule導入し、
  現在の `BasePointSegmentor` interfaceに合わせたwrapperを作成する。

Step 5:
  本格学習・評価は公式PointNeXt-S wrapperで行う。

Step 6:
  将来的には同じinterfaceでPTv3または軽量PTv3系モデルへ置き換えられるようにする。
```

## 具体的に行ってほしい修正

### 1. 暫定モデルの名称変更

現在の `pointnext_segmentor.py` のモデルは、公式PointNeXt-Sではないため、名称を変更してください。

推奨名：

```text
PointNeXtLiteSegmentor
```

または

```text
Stage5PointMLPBaseline
```

今回は、PointNeXtに着想を得たlightweight baselineであることが分かるように、

```text
PointNeXtLiteSegmentor
```

を推奨します。

対応例：

```python
class PointNeXtLiteSegmentor(BasePointSegmentor):
    ...
```

builder名も以下のように変更してください。

```python
def pointnext_lite(...):
    ...
```

### 2. `model_factory.py` の登録名変更

現在の登録名が `pointnext_s` になっている場合、これは公式PointNeXt-S用に空けてください。

変更後の登録例：

```python
_MODEL_BUILDERS = {
    "pointnext_lite": pointnext_lite,
    "mlp_baseline": pointnext_lite,
}
```

将来的に公式PointNeXt-S wrapperを追加する際に、以下を使えるようにしてください。

```python
_MODEL_BUILDERS = {
    "pointnext_lite": pointnext_lite,
    "mlp_baseline": pointnext_lite,
    "pointnext_s": openpoints_pointnext_s,
    "pointnext_s_official": openpoints_pointnext_s,
}
```

この段階では、`pointnext_s` を未登録にしてもよいです。
ただし、ユーザーが誤って `--model pointnext_s` を指定したときに、現在のlightweight baselineが呼ばれないようにしてください。

### 3. コメント・docstringの修正

現在のdocstringに `PointNeXtSegmentor` や `pointnext_s` として公式準拠に見える表現がある場合、以下のように修正してください。

避ける表現：

```text
PointNeXt-S segmentor
official PointNeXt
PointNeXt implementation
```

推奨表現：

```text
PointNeXt-inspired lightweight point-wise MLP baseline
Pure-PyTorch lightweight baseline for Stage 5 pipeline validation
No set abstraction / ball query / feature propagation is implemented
```

### 4. pipeline sanity check用として残す

このbaselineは削除せず、以下の用途で使います。

```text
- Dataset / DataLoader確認
- H5読み込み確認
- points / features / labels / valid_mask shape確認
- loss計算確認
- ignore label処理確認
- metrics計算確認
- checkpoint保存・読み込み確認
- inference H5 / PLY出力確認
- OpenPoints導入前のregression test
```

本格性能評価用のモデルではありません。

### 5. 短い学習テストのみ実施

このbaselineでは、まず以下の確認まで進めてください。

```text
- 1 batch train stepが通る
- 数十iterationでlossがNaNにならない
- checkpoint保存・再読み込みができる
- inferenceで prediction/prob_femur, prediction/pred_label をH5に保存できる
- 可能ならPLY出力も確認する
```

学習時間は短くて構いません。

目安：

```text
1〜5 epoch
または
50〜200 iteration
```

この段階では高い精度は期待しません。

## 公式PointNeXt-S wrapperへの移行方針

pipeline疎通確認後、公式PointNeXt-S実装を組み込みます。

候補：

```text
- OpenPoints
- PointNeXt official repository
```

ただし、Stage 5側の共通interfaceは維持してください。

共通入力：

```python
batch = {
    "points": Tensor[B, N, 3],
    "features": Tensor[B, N, C],
    "labels": Tensor[B, N],
    "valid_mask": Tensor[B, N],
    "meta": ...
}
```

共通出力：

```python
output = {
    "logits": Tensor[B, N, num_classes],
}
```

公式PointNeXt-S wrapperは、例えば以下のようなクラスにしてください。

```python
class OpenPointsPointNeXtSegmentor(BasePointSegmentor):
    ...
```

登録名は以下を想定します。

```text
pointnext_s
pointnext_s_official
```

内部でOpenPoints / official実装が要求する形式にbatchを変換し、出力をStage 5共通形式 `[B, N, num_classes]` に戻してください。

## 重要な注意

* 現在のlightweight baselineで本格評価をしない。
* 現在のlightweight baselineを公式PointNeXt-Sとして記録しない。
* ただしpipeline確認用としては有用なので残す。
* `pointnext_s` という名前は公式準拠wrapper用に空ける。
* Dataset / loss / metrics / inferenceは、lightweight baseline・公式PointNeXt-S・将来のPTv3で再利用できるようにする。
* Stage 5ではendpointやFL lengthは扱わない。
* Stage 5の出力はpoint-wise femur probabilityまで。
* Stage 6でaxis fitting / endpoint extraction / FL measurementを扱う。
