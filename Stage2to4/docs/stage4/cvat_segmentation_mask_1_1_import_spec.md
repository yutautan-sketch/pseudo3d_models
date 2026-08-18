# CVAT Segmentation Mask 1.1 Import 仕様メモ

## 1. 目的

既存の画像セグメンテーションマスクを CVAT に読み込み、CVAT 上で Brush / Eraser 等を用いて人手修正するための、`Segmentation Mask 1.1` 形式 ZIP の作成仕様をまとめる。

想定ワークフロー:

```text
元画像
  +
既存アノテーションマスク
        ↓
CVAT Segmentation Mask 1.1 ZIP へ変換
        ↓
既存 CVAT Task に Upload annotations
        ↓
Mask annotation として読み込み
        ↓
CVAT 上で修正
        ↓
Save
        ↓
Segmentation Mask 1.1 等で再 Export
```

基本方針として、既存の raster mask は **Polygon へ事前変換せず Mask のまま Import** する。
必要な場合のみ、CVAT Import 時の `Convert masks to polygons` を使用する。

---

## 2. CVAT 側で使用する Import 形式

使用形式:

```text
Segmentation Mask 1.1
```

CVAT の Segmentation Mask 形式は Pascal VOC segmentation 形式をベースにした CVAT 独自形式で、以下を扱える。

- Semantic segmentation
- Instance segmentation
- Panoptic segmentation
- 1-channel grayscale PNG mask
- 3-channel RGB PNG mask

Import 可能な annotation:

- Mask
- Polygon
  - Import 時に `Convert masks to polygons` を有効にした場合

Import では以下はサポートされない。

- Attributes
- Tracks

---

## 3. Import ZIP の基本ディレクトリ構造

ZIP のルート直下を次の構造にする。

```text
archive.zip
├── labelmap.txt
├── ImageSets/
│   └── Segmentation/
│       └── default.txt
├── SegmentationClass/
│   ├── image_00001.png
│   ├── image_00002.png
│   └── ...
└── SegmentationObject/
    ├── image_00001.png
    ├── image_00002.png
    └── ...
```

CVAT 公式仕様では `default.txt` である必要はなく、

```text
ImageSets/Segmentation/<any_subset_name>.txt
```

という任意の subset 名を使用できる。

このプロジェクトでは特別な理由がなければ、

```text
default.txt
```

に統一することを推奨する。

### ZIP 作成時の注意

以下のように、さらに外側のフォルダを余分に入れない。

```text
NG:
archive.zip
└── cvat_import/
    ├── labelmap.txt
    ├── ImageSets/
    └── ...

OK:
archive.zip
├── labelmap.txt
├── ImageSets/
├── SegmentationClass/
└── SegmentationObject/
```

---

## 4. `default.txt` の仕様

`ImageSets/Segmentation/default.txt` には、対象画像名を **拡張子なし** で1行ずつ記述する。

例:

元画像:

```text
image_00001.jpg
image_00002.jpg
image_00003.jpg
```

なら、

```text
image_00001
image_00002
image_00003
```

とする。

対応するマスクは PNG として、

```text
SegmentationClass/image_00001.png
SegmentationClass/image_00002.png
SegmentationClass/image_00003.png
```

および必要に応じて、

```text
SegmentationObject/image_00001.png
SegmentationObject/image_00002.png
SegmentationObject/image_00003.png
```

を作る。

### 推奨

変換スクリプトでは、元画像と mask の対応確認を必ず行う。

チェック例:

- 元画像 stem と mask stem が一致しているか
- mask が不足していないか
- 余分な mask が存在しないか
- 画像ごとに width / height が一致するか
- 同名 stem が重複していないか

CVAT への Import を単純化するため、可能なら **Task 内で basename / stem が一意になるファイル名**を使用する。

---

## 5. `labelmap.txt` の基本形式

基本構文:

```text
<label_name>:R,G,B::<改行>
```

例:

```text
background:0,0,0::
femur:255,0,0::
```

公式仕様上の構造は:

```text
label : color (RGB) : 'body' parts : actions
```

今回は segmentation mask の Import が目的なので、後半2フィールドを空にして、

```text
label:RGB::
```

とする。

---

# 6. 1-channel mask を使用する場合

## 6.1 推奨形式

既存 mask を 1-channel PNG に統一する。

例: 1クラス segmentation

```text
0 = background
1 = femur
```

データ型は通常、

```text
uint8
```

でよい。

### 重要

一般的な binary mask:

```text
0   = background
255 = foreground
```

を **そのまま CVAT の 1-channel index mask として使わないことを推奨する**。

代わりに、

```text
0   → 0
255 → 1
```

へ変換する。

理由は、CVAT の 1-channel Segmentation Mask Import では **画素値を label index として扱う**ためである。

---

## 6.2 `labelmap.txt` と画素値の対応

1-channel mask では、

```text
mask pixel value == labelmap.txt の 0-based 行インデックス
```

となる。

例:

```text
labelmap.txt

background:0,0,0::   # index 0
femur:255,0,0::      # index 1
```

mask:

```text
0 = background
1 = femur
```

となる。

RGB 値は 1-channel mask の index 対応そのものには使用されないが、**各行で異なる色**にする。

---

## 6.3 index に欠番を作らない

CVAT 公式仕様では、1-channel mask について、

- 使用する index を `labelmap.txt` にすべて定義する
- index に gap を作らない
- 行順を index 順にする
- 欠番が存在するなら dummy label で埋める

必要がある。

例えば mask が、

```text
0 = background
1 = class_A
4 = class_B
```

を使用する場合、

```text
background:0,0,0::      # 0
class_A:10,10,10::      # 1
_dummy2:20,20,20::      # 2
_dummy3:30,30,30::      # 3
class_B:40,40,40::      # 4
```

のように 2, 3 を埋める必要がある。

したがって実装では、原則として入力ラベル ID を、

```text
0, 1, 2, ..., N
```

へ再マッピングしてから ZIP を作成することを推奨する。

---

# 7. 3-channel RGB mask を使用する場合

CVAT は RGB PNG mask も Import できる。

この場合は画素の RGB 色が `labelmap.txt` の RGB と対応する。

例:

```text
labelmap.txt

background:0,0,0::
femur:255,0,0::
other:0,255,0::
```

mask pixel:

```text
(0, 0, 0)     = background
(255, 0, 0)   = femur
(0, 255, 0)   = other
```

### 条件

RGB mask 内で使用される **すべての annotation 色を `labelmap.txt` に宣言する**。

### 推奨

元データが RGB semantic mask でなければ、実装は 1-channel indexed PNG に統一した方が単純である。

---

# 8. `SegmentationClass` と `SegmentationObject`

## 8.1 `SegmentationClass`

```text
SegmentationClass/
```

は **class segmentation mask** を表す。

同一 class に属する複数 object は同じ class 値 / 色を持つ。

例:

```text
0 = background
1 = femur
2 = vessel
```

同じ画像内に femur object が複数あっても、

```text
class mask 上ではすべて 1
```

となる。

---

## 8.2 `SegmentationObject`

```text
SegmentationObject/
```

は **instance segmentation mask** を表す。

同じ class の object が複数ある場合でも、instance ごとに区別する。

概念例:

```text
0 = background
1 = femur instance A
2 = femur instance B
3 = vessel instance A
...
```

`SegmentationClass` と `SegmentationObject` を組み合わせることで、

- その pixel が何 class か
- どの instance に属するか

を表現できる。

---

## 8.3 単一 foreground object の場合

例えば各画像につき `femur` が最大1領域であり、

```text
0 = background
1 = femur
```

だけで十分な場合は、シンプルな semantic / single-instance mask として扱える。

初期実装ではまず、

```text
SegmentationClass/
```

を正しく生成することを優先する。

既存の CVAT Export 結果を reference として、必要なら同じデータから `SegmentationObject/` も生成する。

### 実装時の強い推奨

手書きで仕様を推測するより、CVAT で同じラベル構成の小さな Task を作り、

```text
Segmentation Mask 1.1
```

で一度 Export した ZIP を **golden sample / reference** として比較する。

今回すでに CVAT から `Segmentation Mask 1.1` の Export が確認できているため、その ZIP の:

```text
labelmap.txt
default.txt
SegmentationClass/*.png
SegmentationObject/*.png
```

を変換スクリプトの期待出力と比較するのが最も安全。

---

# 9. Task 側の label

既存 Task に annotation を Import する場合は、Task 側の label と `labelmap.txt` の label 名を一致させる。

例:

CVAT Task:

```text
femur
```

`labelmap.txt`:

```text
background:0,0,0::
femur:255,0,0::
```

### 推奨

- 大文字 / 小文字
- 空白
- `_`
- `-`

を含め、label 名を完全一致させる。

変換スクリプトには label 名を引数または設定ファイルとして明示的に与える。

---

# 10. CVAT への Import 手順

既存 Task に対して:

```text
Tasks
  ↓
対象 Task
  ↓
Actions
  ↓
Upload annotations
  ↓
Format:
Segmentation Mask 1.1
  ↓
作成した ZIP を選択
```

Job 単位でも Import 可能。

Job の場合:

```text
Task
  ↓
対象 Job の ...
  ↓
Import annotations
```

---

# 11. 非常に重要: 既存 annotation は置換される

CVAT 公式仕様では、Task / Job に annotation を Upload すると、

```text
既存 annotations
        ↓
削除 / 置換
        ↓
Import annotations
```

となる。

そのため、本番 Task に Import する前に必ず既存 annotation を Export してバックアップする。

推奨:

```text
before_import_backup/
└── task_xxx_CVAT_for_images_1.1.zip
```

または、

```text
before_import_backup/
└── task_xxx_Segmentation_Mask_1.1.zip
```

を保存する。

---

# 12. Mask のまま Import するか Polygon に変換するか

## 推奨: Mask のまま Import

今回の目的は、

```text
既存 segmentation mask
        ↓
CVAT
        ↓
人手で修正
```

なので、基本的には Mask のまま Import する。

Import 時:

```text
Convert masks to polygons = OFF
```

を推奨。

CVAT 上では Brush tool を利用して、

- Brush: mask 領域を追加
- Eraser: mask 領域を削除
- Add polygon: polygon 選択範囲を mask に追加
- Remove polygon: polygon 選択範囲を mask から削除

できる。

---

## Polygon にしたい場合

必要な場合のみ、

```text
Convert masks to polygons = ON
```

とする。

ただし raster mask を contour 化すると、

- 頂点数が多くなる
- pixel 境界と polygon 表現に差が生じる可能性がある
- 細い領域や小領域の形状が変化し得る

ため、pixel mask の修正目的では Mask のまま扱う方が適している。

---

# 13. 変換スクリプトに要求する処理

別チャットで実装するスクリプトには、最低限以下を実装する。

## 必須入力

例:

```text
--images-dir
--masks-dir
--output
--labels
```

必要に応じて:

```text
--foreground-value
--background-value
--label-name
--mapping
```

等を用意する。

---

## 必須処理

### A. ファイル対応確認

```text
元画像 stem ↔ mask stem
```

を対応付ける。

チェック:

- image without mask
- mask without image
- duplicate stem
- unsupported extension

問題があれば warning ではなく原則エラー終了する。

---

### B. 画像サイズ確認

各ペアについて、

```text
image width  == mask width
image height == mask height
```

を確認する。

異なる場合は Import 前にエラーにする。

自動 resize はデフォルトでは行わない。

理由:

annotation mask の resize は教師データを暗黙に変更するため。

必要なら明示的な option として別実装する。

---

### C. mask の dimensionality / dtype 確認

想定:

```text
H x W
```

の 1-channel mask。

RGB 入力の場合:

```text
H x W x 3
```

として別処理する。

binary mask の典型例:

```text
{0, 255}
```

は、

```text
{0, 1}
```

へ明示的に変換する。

---

### D. label index の正規化

推奨:

```text
background = 0
class_1    = 1
class_2    = 2
...
```

に連番化する。

元 mask が、

```text
0, 10, 50
```

の場合は可能なら、

```text
0 → 0
10 → 1
50 → 2
```

へ変換する。

変換 mapping を標準出力または metadata として必ず記録する。

---

### E. `labelmap.txt` 生成

例:

```text
background:0,0,0::
femur:255,0,0::
```

multi-class の場合も index 順に生成する。

1-channel mask では **行番号と mask index が対応すること**を保証する。

---

### F. `default.txt` 生成

各画像 stem を1行ずつ記述。

例:

```text
image_00001
image_00002
image_00003
```

出力順は、再現性のため自然順または lexicographical sort で固定する。

---

### G. `SegmentationClass/*.png` 生成

変換済み class-index mask を PNG 保存する。

推奨:

```text
uint8
```

ただし label index 数が 255 を超える可能性がある場合は、CVAT 仕様および PNG 表現を再確認して設計すること。

通常の少数 class 用途では 0..255 内に収める。

---

### H. `SegmentationObject/*.png` 生成

instance が必要な場合に生成。

semantic segmentation のみの入力の場合は、まず CVAT から Export した reference ZIP の挙動を確認してから生成方式を合わせる。

---

### I. ZIP 生成

最終 ZIP:

```text
output.zip
├── labelmap.txt
├── ImageSets/
│   └── Segmentation/
│       └── default.txt
├── SegmentationClass/
│   └── *.png
└── SegmentationObject/
    └── *.png
```

ZIP root に余分なトップディレクトリを作らない。

---

# 14. 変換後に必ず行う validation

ZIP を作成するだけでなく、スクリプト内で validation を行う。

## 推奨 validation 項目

```text
[ ] ZIP に labelmap.txt がある
[ ] ImageSets/Segmentation/default.txt がある
[ ] SegmentationClass がある
[ ] default.txt の各 stem に対応する PNG がある
[ ] PNG のサイズが元画像と一致
[ ] PNG が 1-channel または 3-channel
[ ] 1-channel の全 unique 値が labelmap index 範囲内
[ ] 1-channel index に意図しない値がない
[ ] RGB mask の全 unique color が labelmap に存在
[ ] background の扱いが統一されている
[ ] image/mask 対応数が一致
[ ] 出力 ZIP 内に不要な親ディレクトリがない
```

validation summary 例:

```text
Images             : 120
Masks              : 120
Classes            : 2
Mask mode          : grayscale/indexed
Unique indices     : [0, 1]
Background index   : 0
Foreground labels  : {1: "femur"}
Size mismatches    : 0
Missing masks      : 0
Extra masks        : 0
Output             : cvat_import.zip
```

---

# 15. 最初の実装で推奨する最小仕様

まずは multi-class / RGB / instance を全部一般化せず、次の仕様から実装する。

## Input

```text
images/
├── image_00001.*
├── image_00002.*
└── ...

masks/
├── image_00001.png
├── image_00002.png
└── ...
```

mask:

```text
0   = background
255 = foreground
```

label:

```text
femur
```

## Conversion

```text
0   → 0
255 → 1
```

## Output

```text
cvat_import.zip
├── labelmap.txt
│   background:0,0,0::
│   femur:255,0,0::
│
├── ImageSets/
│   └── Segmentation/
│       └── default.txt
│
├── SegmentationClass/
│   └── *.png
│
└── SegmentationObject/
    └── *.png
```

CVAT Import:

```text
Segmentation Mask 1.1
Convert masks to polygons = OFF
```

この最小構成が正常に Import できることを確認してから、

- multi-class
- arbitrary input IDs
- RGB masks
- multiple instances
- nested directories

へ拡張する。

---

# 16. 推奨テスト方法

いきなり全データを変換せず、最初は 2～3 枚だけ使用する。

```text
test_images/
├── image_00001.png
├── image_00002.png
└── image_00003.png

test_masks/
├── image_00001.png
├── image_00002.png
└── image_00003.png
```

テスト手順:

```text
1. 変換スクリプトで cvat_import_test.zip を作成
2. CVAT にテスト用 Task を作成
3. 同じ3枚の画像を登録
4. Segmentation Mask 1.1 として annotation ZIP を Import
5. Convert masks to polygons = OFF
6. Job を開く
7. mask が正しい画像位置・ラベルで表示されることを確認
8. Brush / Eraser で編集できることを確認
9. Save
10. Segmentation Mask 1.1 で再 Export
11. Import 前 mask と Export 後 mask を比較
```

さらに、CVAT から手作業で作った正解 ZIP とスクリプト生成 ZIP のディレクトリ構造・PNG unique 値を比較するとよい。

---

# 17. 実装時に避けるべきこと

## 0/255 binary mask を index mask としてそのまま使用

```text
NG:
0   = background
255 = foreground
```

をそのまま 1-channel indexed mask として投入する。

理由:
255 を index 255 と解釈させる構成になり、`labelmap.txt` 側にも 0～255 の index 対応が必要になるため不適切。

推奨:

```text
0 → 0
255 → 1
```

---

## 自動 resize

mask と元画像サイズが異なる場合に暗黙に resize しない。

原則:

```text
size mismatch → error
```

とする。

---

## Import 前の既存 annotation バックアップ忘れ

Task への Upload annotations は既存 annotation を置換するため、本番データでは必ず Export してから Import する。

---

## いきなり Polygon 化

既存 pixel mask の人手修正が目的なら、

```text
Convert masks to polygons = OFF
```

から開始する。

---

# 18. 別チャットでの実装依頼に渡す要点

実装チャットには最低限、以下を伝える。

```text
目的:
既存 segmentation mask を CVAT Segmentation Mask 1.1 の
annotation import ZIP に変換する Python スクリプトを作成する。

初期対象:
- 元画像ディレクトリ
- 元画像と同 stem の binary PNG mask
- mask は 0/255
- 1 foreground class
- foreground label 名を引数指定
- 0/255 → 0/1 に変換
- grayscale 1-channel PNG として出力

ZIP:
labelmap.txt
ImageSets/Segmentation/default.txt
SegmentationClass/*.png
SegmentationObject/*.png

重要仕様:
- 1-channel mask の画素 index と labelmap.txt の 0-based 行 index を一致
- background=0
- foreground=1
- image/mask stem と size の厳密検証
- 不整合時はエラー終了
- ZIP root に余分なディレクトリを入れない
- validation summary を表示
- CVAT Import は Segmentation Mask 1.1
- Convert masks to polygons は OFF を前提
```

---

# 19. 公式仕様参照

確認日: 2026-08-17

CVAT 公式ドキュメント:

- Segmentation Mask
  - `docs.cvat.ai/docs/dataset_management/formats/format-smask/`
- Import annotations and data to CVAT
  - `docs.cvat.ai/docs/dataset_management/import-datasets/`
- Annotation with brush tool
  - `docs.cvat.ai/docs/annotation/manual-annotation/shapes/annotation-with-brush-tool/`

特に重要な公式仕様:

1. Segmentation Mask Import は 1-channel / 3-channel PNG の双方に対応。
2. 1-channel mask は label index と `labelmap.txt` の行 index を対応させる。
3. 1-channel mask の index に gap がある場合は dummy label で埋める必要がある。
4. RGB mask は使用する全色を `labelmap.txt` に宣言する。
5. Mask のまま Import 可能。
6. `Convert masks to polygons` を有効にすると Polygon として Import 可能。
7. Task / Job へ annotation を Upload すると既存 annotation は削除・置換される。
8. Brush tool では既存 mask に対して追加・削除・polygon add/remove が可能。
