# Stage 5: 予測positiveのフレーム画像可視化 実装依頼書

作成日: 2026-09-19
作成元: Stage 5実装チャット（S5-15 P3/P4担当）
担当: 本件専用の実装チャット
状態: 方針・調査済み。実装未着手。

## 1. 目的と背景

S5-15の評価で、学習済みモデルのfalse positive（FP）が多いことが分かっている
（epoch 6のvalidation pooled FPR 7.41%、precision 5.27%）。
**「FPがどのようなオブジェクト上で発生しているのか」を知りたい**が、
現在出力されるPLYは3D点群のため、解剖学的・器具的に何を誤検出しているのかを読み取りにくい。

そこで、**モデルの予測positiveとGTのpositiveを元のフレーム画像上に重ねて描画し、
2D画像として比較できるディレクトリ**を、既存の評価出力へ追加で生成する。

本件は**可視化の追加のみ**であり、学習・再推論・production設定の変更を含まない。

## 2. 承認境界（重要）

- **新規学習は行わない。GPU推論も行わない。** 既存の評価成果物と既存のH5のみを入力とする。
- 既存の評価出力・run・H5・checkpointを**上書き・改変しない**。追加ディレクトリへ書くだけ。
- `evaluate_stage5.py` / `evaluate_stage5.sh` / `train_stage5.*` の既存挙動を変更しない
  （呼び出しを追加する場合も、既定では無効かつ後段の独立したステップとする）。
- Stage2to4の既存exporterを**改変しない**。再利用する場合はimportのみ。
- 実行はCPUで完結する見込み。CUDAは不要。

## 3. 最初に読む文書・コード

1. `docs/stage5/FILES.md`（実装配置の規約）、`docs/stage5/data_construct.md`（データ構造）。
2. `docs/stage5/s5-15/stage5_s5_15_report_to_policy_chat.md` の8.12.10〜8.12.12
   （本件の動機となった評価結果。epoch 6とepoch 50の挙動差）。
3. `Stage2to4/pseudo3d/export/export_stage4_point_label_visualization.py`（**最重要**。
   同じ座標変換・同じ画像ソースでGTラベルを既に描画している既存実装）。
4. `Stage5/evaluate_stage5.py` の予測保存部（468〜476行目）とPLY出力部（478〜497行目）。
5. `Stage5/checks/real_h5/check_stage5_xy_coordinate_provenance.py`
   （`pixel_xy`の座標系を確定させたS5-14の監査。103〜180行目、329〜339行目）。

## 4. 調査済みの技術的前提（推測ではなく確認済み）

本節は実装チャットが再調査せずに前提としてよい。ただし**実機のデータで成立することは
最初に確認すること**（4.6節）。

### 4.1 予測は保存済みで、再推論は不要

`evaluate_stage5.py`は`--no_save_predictions`を付けない限り、動画ごとに次を保存する
（`evaluate_stage5.sh`の既定は保存する）。

```text
<評価出力>/<checkpoint名>/predictions/<split>/<safe_name(video)>.npz
  point_indices : int64  [N]   0..N-1
  prob_femur    : float32[N]   mean-probability集約後のpositive確率
  pred_label    : uint8  [N]   最終判定（0/1）
  vote_count    : int32  [N]
```

`N`は当該動画の全点数で、**H5の点配列と同じ並び**である。したがって
H5の`pixel_xy`・`frame_order`・`point_label`・`valid_mask`と**インデックスで直接対応づく**。

`safe_name()`は`re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")`（`evaluate_stage5.py:230`）。

### 4.2 `pixel_xy`はlocal-crop画像の座標系

S5-14 Step H2.5で確定済み。`pixel_xy`は**動画ごとのlocal-crop画素空間**の座標であり、
`[0, width) × [0, height)`に収まる（`check_stage5_xy_coordinate_provenance.py:329-339`）。
元フレーム全体の座標**ではない**。

### 4.3 フレーム画像は中間pseudo3d H5の中にある

外部の画像ディレクトリを探す必要はない。中間pseudo3d H5が`local_encoder_images`
データセットを持ち、**これが`pixel_xy`と同じ座標系の画像**である
（`export_stage4_point_label_visualization.py:526-529, 1525`）。

中間H5のattrsから次が読める（`check_stage5_xy_coordinate_provenance.py:103-145`）。

| attr | 意味 |
| --- | --- |
| `local_input_shape` | `(num_frames, channels, height, width)`。`pixel_xy`の空間 |
| `raw_width` / `raw_height` | 元フレームの寸法 |
| `local_crop_top` / `local_crop_left` | 元フレーム内でのcrop原点 |
| `local_resize_scale` | resize倍率 |

**今回の用途ではlocal-crop画像上に描くだけでよい**ため、元フレームへの逆変換は必須ではない。
ただし「何を誤検出したか」の判断に周辺文脈が要る場合は、上記attrsで元フレーム座標へ戻せる。
逆変換を行う場合は`local_resize_scale`→`local_crop_left/top`の順で戻すこと。

### 4.4 中間H5の場所

最終H5（teacher v7）のattr `source_pseudo3d_h5` に記録されている。
解決できない場合のfallback（basename一致、glob）を含む実装が
`check_stage5_xy_coordinate_provenance.py:148-180`にある。**この解決ロジックを再利用すること。**
解決方法（記録attr経由か、fallbackか）は**出力に記録し、同一視しない**。

### 4.5 点とフレームの対応

`frame_order`（`[N]`、0始まりのフレーム順序）で対応づける。既存exporterは
`np.argsort(frame_order)`と`np.bincount`でフレームごとの点インデックスを作り、
`local_images[frame_order]`と突き合わせている（1510〜1525行目）。**同じ方法を使うこと。**

### 4.6 最初に確認すべきこと

- 21動画すべてについて中間pseudo3d H5が実機に存在するか。**存在しない動画があれば、
  その動画は「可視化不可」として記録し、代替の推測（別動画の寸法流用など）をしない。**
- `predictions/<split>/*.npz`が`best`/`last`の両方に存在するか。
- `pixel_xy`が`local_input_shape`の範囲に収まるか（既存の境界チェックを流用）。

## 5. 実装方針

### 5.1 描画仕様

**GTと予測を1枚の画像で比較できることを最優先とする。** 点ごとに次の分類で色を付ける。
Stage5の既存PLY診断色（`stage5/utils/visualization_export.py:8-27`）と**同じ配色を使い**、
PLYと画像で見え方が食い違わないようにする。

| 分類 | 条件 | 色（RGB） |
| --- | --- | --- |
| true_positive | valid かつ GT positive かつ 予測positive | 緑 `(30,180,70)` |
| **false_positive** | valid かつ GT background かつ 予測positive | **赤 `(230,45,45)`** |
| false_negative | valid かつ GT positive かつ 予測background | 黄 `(245,190,35)` |
| true_negative | valid かつ GT background かつ 予測background | 青 `(65,105,180)` |
| ignore_predicted_positive | ignore領域で予測positive | 紫 `(200,65,180)` |
| ignore_other | ignore領域その他 | 灰 `(145,145,145)` |

- 背景画像は`image_to_uint8_gray(local_images[frame_order])`でグレースケール化したもの。
- **true_negativeと ignore_other は既定で描画しない**（FPを見たい用途で画面が埋まるため）。
  描画するかは引数で切り替え可能にする。
- 点の半径・不透明度は引数化する。既存の`render_point_labels`（1275行目）が
  半径・alphaつきの描画を実装しているので、**同等の描き方を再利用または踏襲**する。
- 既存の`_draw_saved_bbox`でBBoxも重ねられる。参考情報として任意で描けるようにする。

### 5.2 全フレームは出さない。FPの多い順に絞る

21動画×全フレームではPNGが数千枚になり、目視に耐えない。**フレーム選択機構を必須とする。**

- 動画ごとに**フレーム単位のFP数**（および TP/FN数）を集計したCSVを出力する。
- 既定では**FP数の多い上位N フレーム**（`--top_frames_per_video`、既定10程度）を出力する。
- `--all_frames`で全フレーム出力も可能にする（既定では使わない）。
- 「FPゼロのフレーム」「TPゼロのフレーム」も別途少数サンプリングできると比較しやすい。

### 5.3 出力先

既存の評価出力ディレクトリ配下へ**追加**する。既存ファイルは触らない。

```text
<評価出力>/<checkpoint名>/prediction_frames/<split>/<video>/
    frame_00012_fp0431.png          （フレーム番号とFP数を名前に含めると並べ替えやすい）
    ...
<評価出力>/<checkpoint名>/prediction_frames/<split>/<video>_frame_summary.csv
<評価出力>/<checkpoint名>/prediction_frames/summary.csv      全動画のフレーム別集計
<評価出力>/<checkpoint名>/prediction_frames/manifest.json    入力hash・解決方法・実施量
```

`best`と`last`で同じ構成にし、**同一動画・同一フレーム番号で並べて比較**できるようにする。

### 5.4 コードの配置（**最初に方針を確認すること**）

判断が必要な点である。実装チャットは着手前に方針管理チャットへ確認すること。

- **案A（推奨）: `Stage5/`側に新規モジュールを置き、Stage2to4の描画関数をimportする。**
  入力の主体はStage5の予測（`predictions/*.npz`）であり、出力先もStage5の評価ディレクトリである。
  ただしStage2to4への依存が生じる（既にStage5は`external/PointNeXt`へ依存しているため前例はある）。
- 案B: `Stage2to4/pseudo3d/export/`側に置く。描画・画像・座標の資産はすべてこちらにある。
  ただしStage5の出力構造に依存する関数がStage2to4に入る。
- どちらでも**既存exporterは改変しない**。共通化のための大規模リファクタリングは今回の対象外。

### 5.5 CLIとbash

- Python CLIは`--evaluation_dir`（`<評価出力>/<checkpoint名>`）、`--h5_list`または
  評価時の`evaluation_data/`から動画→H5対応を取得、`--output_subdir`、`--top_frames_per_video`、
  `--radius` / `--alpha` / `--draw_true_negative` 等を持つ。
- bashは既存checkerの規約（`PYTHON`変数、実機パスの内部定数化）に従う。
  **GPU用の環境変数設定は不要**（CUDAを使わないため）。

## 6. テスト

`Stage5/checks/dummy/`（または案Bなら`Stage2to4`側の対応する場所）に合成テストを置く。
**CUDA・実データ不要**で完結させること。

- 合成H5（`point_cloud`/`annotation`スキーマ）と合成中間H5（`local_encoder_images`と
  必要なattrs）を作り、既知の点配置で分類が正しく色付けされることを検証する。
  TP/FP/FN/TN/ignore-positiveの5分類それぞれについて、**期待した画素に期待した色**が入ること。
- `frame_order`によるフレーム振り分けが正しいこと（あるフレームの点が別フレームに描かれない）。
- フレーム選択（FP上位N）が正しく並ぶこと。FP数が同数のときの決定的な順序も固定する。
- `pixel_xy`が範囲外・中間H5が見つからない・`predictions/*.npz`の点数がH5と一致しない場合に
  **停止して報告する**こと（無言のスキップや推測補完をしない）。
- 既存の評価出力を変更しないこと（入力ディレクトリのmtime・内容が不変）。

## 7. privacy（重要）

- **フレーム画像は患者由来データである。** 出力ディレクトリは既定で
  `DO_NOT_SHARE`相当の扱いとし、共有用の匿名化ディレクトリへは**入れない**。
- ファイル名・CSV・manifestに実video名が入る。共有が必要な場合は
  既存の`video_alias`規約（`validation_000`等）に従って別途匿名化すること。
  匿名化していない出力を共有可能ディレクトリへ書かない。
- manifestを共有する場合は、既存checkerと同じprivacy self-check
  （`[0-9]{8}_[0-9]{6}_[0-9]+`と`(/mnt/data|/home/[A-Za-z0-9_.-]+)`の正規表現スキャン）を通すこと。

## 8. 完了条件

1. 合成テストが全件合格し、`py_compile`・`bash -n`・`git diff --check`が通る。
2. 実機で、S5-15の`best`（epoch 6）と`last`（epoch 50）の両方に対して実行し、
   validationの代表動画についてFP上位フレームのPNGが生成される。
3. フレーム別集計CSVで「FPが集中しているフレーム」が特定できる。
4. `docs/stage5/FILES.md`に配置を追記する。
5. 実施内容・入力hash・未可視化動画（あれば）を報告する。

## 9. 本件で答えを出したい問い

可視化の完成が目的ではなく、次の問いに答えるための材料を得ることが目的である。

- **FPはどのオブジェクト上で起きているか。** 他の骨、軟部組織、プローブ由来のアーティファクト、
  画像端、といった分類ができるか。
- FPは特定のフレーム・特定の領域に集中しているか、全体に散っているか。
- epoch 6とepoch 50で、FPの発生場所の性質が変わっているか
  （epoch 50はvalidationで予測positiveが極端に少ないため、
  「FPが減った」のか「positiveを出さなくなった」だけなのかを画像で確認できる）。
- GT positiveが取れていない領域（FN）に共通の特徴があるか。

これらは定性的な観察であり、**数値評価の結論を置き換えるものではない**。
観察結果は`docs/stage5/s5-15/`の報告書へ記録し、production設定の変更は別途判断する。

## 10. 実装ステップ（2026-09-20 追記。着手前の段取り確定版）

本節は、5.4節および4節の未確定事項に対する方針管理チャットの回答（2026-09-20）を反映した
実装手順である。**5節までの仕様と矛盾する場合は本節を優先する。**

### 10.0 確定した方針（再調査不要）

| 項目 | 決定 |
| --- | --- |
| コード配置（5.4節） | **案A**。`Stage5/`側に新規モジュールを置く |
| Stage2to4からのimport | **`image_to_uint8_gray`のみ**。exporter本体はimportしない |
| 実機のStage2to4パス | `/mnt/data/3d_projects/models/Stage2to4` |
| 中間H5のfallback root/suffix | 既存checkerと同一（`pseudo3d_outputs/260711`、`_ts448_oym96_corr.h5`）。**全21動画に適用可**|
| 対象split | **`train_sanity`（3動画）と`validation`（18動画）の両方** |

`export_stage4_point_label_visualization.py`をimportしない理由は、同モジュールが冒頭で
`sys.path`を書き換えたうえで`cv2`・`imageio`・CVATマスク変換モジュールまで芋づるで読み込むためである。
必要な描画ロジックは6分類化が必須で（10.1参照）どのみち書き下ろすことになるため、
依存面を`image_to_uint8_gray`一点に絞る。

### 10.1 調査で判明した再利用可能資産（実装前に把握しておくこと）

**(a) 6分類ロジックは既にStage5に存在する。新規実装してはならない。**

`stage5/utils/visualization_export.py`の`diagnostic_segmentation_categories()`が、
5.1節の表と完全に同じ6分類を`uint8`のカテゴリ配列`[N]`として返す
（0=true_positive, 1=false_positive, 2=false_negative, 3=true_negative,
4=ignore_predicted_positive, 5=ignore_other）。色は同ファイルの
`DIAGNOSTIC_CATEGORY_COLORS`で、5.1節の表のRGBと一致することを確認済みである。

**この関数と色表をそのまま使うこと。** 診断PLY（`write_diagnostic_segmentation_ply`）が
同じ関数を通っているため、PNGとPLYの見え方の一致は分類ロジックの共有によって
構造的に保証される。分類条件をPNG側で書き直すと、この保証が失われる。

**(b) `render_point_labels`は6分類には使えない。**

既存の`render_point_labels`（`export_stage4_point_label_visualization.py:1275`）は
background/ignore/positiveの**3ラベル固定**であり、色引数も3つしかない。
したがって「reuse」ではなく「**踏襲**」となる。踏襲すべき描き方は次の3点である。

1. `np.rint(pixel_xy).astype(np.int64)`で整数画素へ丸める。
2. カテゴリごとに`uint8`のmarker画像を作り、`cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2r+1, 2r+1))`
   で`cv2.dilate`して半径を与える。
3. `(1-alpha) * 元画素 + alpha * 色`でブレンドする（`_blend_mask`、同1259行）。

描画順序は**重なったとき何を見たいか**で決める。既存実装がbackground→ignore→positiveの順で
positiveを最後に描いているのと同じ考え方で、本件は
`true_negative → ignore_other → false_negative → true_positive → ignore_predicted_positive → false_positive`
の順とし、**false_positiveが最後（最前面）**に来るようにする。
本件の目的はFPの観察であるため、FPが他の色に隠れてはならない。

**(c) 中間H5の解決ロジックはcheckerからimportできる。**

`checks/real_h5/check_stage5_xy_coordinate_provenance.py`は`REPO_ROOT`を`sys.path`へ入れる以外に
import時副作用がなく、`checks/real_h5/__init__.py`も存在するため、

```python
from checks.real_h5.check_stage5_xy_coordinate_provenance import (
    read_intermediate_local_dimensions,
    resolve_intermediate_h5,
)
```

でそのまま再利用できる（4.4節の「この解決ロジックを再利用すること」を、複製ではなくimportで満たす）。
`resolve_intermediate_h5`は`method`として`recorded_source_attr` / `basename_fallback_exact` /
`basename_fallback_glob` / `basename_fallback_ambiguous` / `unresolved`を返すため、
4.4節の「解決方法を出力に記録し、同一視しない」要件はこの`method`をmanifestとCSVへ
そのまま書けば満たせる。

**留意点:** CLIが`checks/`配下へ依存する向きは本リポジトリに前例がない。
実装中にこの向きが問題だと判断した場合は、**勝手に既存checkerを移動・改変せず**、
方針管理チャットへ確認すること。

**(d) 動画→H5の対応は`h5_metrics.csv`から取る。**

5.5節は`--h5_list`または`evaluation_data/`を挙げているが、評価出力直下の`h5_metrics.csv`が
`split` / `video_name` / `h5_path`をそのまま持っている（`evaluate_stage5.py:437-442`）。
**評価時に実際に使われたH5**であることが保証されるため、これを既定の入力とする
（`--h5_list`は上書き用として残してよい）。

### 10.2 対象ディレクトリの構造（確認済み）

ユーザー指定の
`/mnt/data/3d_projects/stage5_evaluations/260919/pointnext_s_EX260919_s5_15_r0long50_none_gn8_cwfixed_lr1e3_ep50_bs1_acc8_nopad`
は`evaluate_stage5.sh`の`OUTPUT_ROOT`であり、`<評価出力>/<checkpoint名>`は
その直下の`best/`と`last/`である（`checkpoint_stem`＝`best.pt`→`best`）。

```text
<OUTPUT_ROOT>/
  evaluation_data/ , reference_ply/
  best/   predictions/{train_sanity,validation}/<safe_name>.npz , ply/ , h5_metrics.csv , summary.json
  last/   （同構成）
```

**split名は`train_sanity`と`validation`である**（`evaluate_stage5.py:429`）。
5.3節の`<split>`はこの2つを指す。出力は`best/prediction_frames/`と`last/prediction_frames/`の2系統。

### 10.3 実装手順

着手順に並べる。各ステップの完了物を明示する。

**Step 1: 描画コアモジュール**

`Stage5/stage5/utils/prediction_frame_render.py`（新規）。

- `render_diagnostic_categories(gray, *, pixel_xy, categories, radius, alpha, draw_categories)` →
  `[H,W,3] uint8`。10.1(b)の描き順・描き方に従う。`draw_categories`で
  true_negative・ignore_otherを既定offにする。
- `frame_point_index_ranges(frame_order, num_frames)` →
  `np.argsort(frame_order, kind="stable")`＋`np.bincount`＋`cumsum`で
  フレームごとの点indexを返す（4.5節、exporter 1511-1525行と同じ方法）。
- 分類は`diagnostic_segmentation_categories`をimportして使う（10.1(a)）。
- `image_to_uint8_gray`は`Stage2to4`側からimportする。
  `sys.path`への追加は**CLI側**（Step 2）で一箇所に集約し、コアモジュールでは行わない。

**Step 2: CLI**

`Stage5/export_stage5_prediction_frames.py`（新規、トップレベルCLI）。
既存の`export_anonymized_stage5_metrics.py`と同じ位置づけ。

- 引数: `--evaluation_dir`（`<OUTPUT_ROOT>/best`等）、`--stage2to4_root`（既定
  `/mnt/data/3d_projects/models/Stage2to4`）、`--pseudo3d_outputs_root`、`--fallback_suffix`、
  `--splits`（既定`train_sanity validation`）、`--output_subdir`（既定`prediction_frames`）、
  `--top_frames_per_video`（既定10）、`--all_frames`、`--zero_fp_samples`、`--zero_tp_samples`、
  `--radius`、`--alpha`、`--draw_true_negative`、`--draw_ignore_other`、`--draw_bbox`、
  `--h5_list`（`h5_metrics.csv`の上書き用）。
- 処理順: h5_metrics.csv読込 → 動画ごとに npz/最終H5/中間H5を解決 →
  整合検証（Step 3）→ フレーム別集計 → フレーム選択（Step 4）→ 描画・PNG書き出し →
  CSV・manifest書き出し。
- `--stage2to4_root`を`sys.path`へ挿入するのはここだけ。解決できない場合は
  **その場で停止し、探したパスを明示して報告する**（黙ってフォールバックしない）。

**Step 3: 停止条件（6節の要求。無言のスキップ・推測補完を一切しない）**

以下はすべて**例外を送出して停止**し、何が不一致かを具体値付きで出力する。

- `predictions/<split>/<safe_name>.npz`の点数 ≠ 最終H5の点数。
- `pixel_xy`が`local_input_shape`の`[0,width) × [0,height)`外
  （既存の境界チェックの判定式を流用する）。
- `frame_order`の最大値 ≥ `local_encoder_images`のフレーム数。
- 中間H5が`unresolved`または`basename_fallback_ambiguous`。
  → ただし4.6節どおり、**この動画のみ「可視化不可」として記録し、他動画の寸法を流用しない**。
  全動画が不可の場合は異常として停止する。動画単位の不可は
  manifestの`unvisualized_videos`へ理由付きで記録し、最後にまとめて報告する。

**Step 4: フレーム選択（5.2節）**

- 動画ごとにフレーム単位でTP/FP/FN/TN/ignore各数を集計。
- 既定は**FP数降順**で上位N。**FP数が同数の場合は`frame_order`昇順**で決定的に並べる
  （6節が求める決定的順序の固定）。
- `--zero_fp_samples` / `--zero_tp_samples`で比較用の少数サンプルを追加抽出する。
  サンプリングは固定seedで決定的にする。
- `--all_frames`指定時のみ全フレーム。既定では使わない。

**Step 5: 出力（5.3節）**

```text
<評価出力>/<checkpoint名>/prediction_frames/<split>/<video>/frame_00012_fp0431.png
<評価出力>/<checkpoint名>/prediction_frames/<split>/<video>_frame_summary.csv
<評価出力>/<checkpoint名>/prediction_frames/summary.csv
<評価出力>/<checkpoint名>/prediction_frames/manifest.json
```

manifestに入れるもの: 入力（npz・最終H5・中間H5）のsha256、中間H5の`resolution_method`、
使用した描画引数、動画ごとの選択フレーム数、`unvisualized_videos`、実行日時、
Stage2to4 root。`best`と`last`で同一動画・同一フレーム番号が並ぶこと。

**既存ファイルには一切書かない。** 既存の評価出力のmtimeと内容が不変であることを
Step 7のテストで検証する。

**Step 6: bashラッパ**

`Stage5/export_stage5_prediction_frames.sh`。既存checker規約に従う
（`SCRIPT_DIR`を実機パスで内部定数化、`PYTHON="${PYTHON:-...}"`、各パスは環境変数で上書き可能、
実行前に設定値をechoし、最後に出力先をecho）。**CUDA関連の環境変数は設定しない。**
`best`と`last`を順に処理できるようにする。

**Step 7: 合成テスト**

`Stage5/checks/dummy/check_dummy_prediction_frame_visualization.py` および同名`.sh`。
CUDA・実データ不要。合成の最終H5（`point_cloud`/`annotation`スキーマ）と
合成中間H5（`local_encoder_images`＋必要attrs）と合成npzを一時ディレクトリに作る。

検証項目（6節）:

1. TP/FP/FN/TN/ignore_predicted_positiveの5分類について、**既知の点座標の画素に期待した色**が入る
   （`radius=0`・`alpha=1.0`で厳密一致を見るケースを必ず含める）。
2. TN・ignore_otherが既定で描かれず、フラグ指定時に描かれる。
3. `frame_order`による振り分けが正しい（あるフレームの点が別フレームに描かれない）。
4. FP上位N選択の順序が正しく、FP同数時に`frame_order`昇順で決定的。
5. Step 3の各停止条件でそれぞれ例外になる（点数不一致・範囲外・中間H5未解決）。
6. 入力ディレクトリのmtimeと内容が実行前後で不変。
7. FP最前面の描画順序（FPとTPが同一画素に来たときFP色になる）。

**Step 8: 文書更新**

`docs/stage5/FILES.md`へ`export_stage5_prediction_frames.py` / `.sh` /
`stage5/utils/prediction_frame_render.py` / dummy checkを追記する。

### 10.4 実行環境の制約（重要。着手前に必ず確認すること）

**開発コンテナ（/workspace）には`numpy`・`h5py`・`cv2`・`imageio`のいずれも入っておらず、
`/mnt/data`も見えない。** したがって開発環境で可能な検証は

- `python3 -m py_compile`、`bash -n`、`git diff --check`

までであり、**Step 7の合成テストの実行と、実機の評価ディレクトリへの出力は、
どちらも実機（`dualtrack311`環境）で行う。** 8節の完了条件1が求める「合成テスト全件合格」は
実機での実行結果をもって満たす。開発環境でテストが動かないことを
「テスト不要」と読み替えてはならない。

また`cv2`・`imageio`はStage5がこれまで使っていない依存である
（Stage5配下のPythonにimport実績なし）。Stage2to4が同じ環境で両方を使っているため
実機では利用可能と見込まれるが、**実機での最初の作業として両モジュールのimport可否を確認する**こと。
利用できない場合は方針管理チャットへ報告する（numpyだけで円形スタンプを描く代替はあるが、
既存の見え方との一致が崩れるため独断で切り替えない）。

### 10.5 実機での実行順（着手後）

1. import可否確認（`cv2`・`imageio`・Stage2to4 root）。
2. Step 7の合成テストを実行し全件合格させる。
3. 21動画すべてについて中間H5が解決するかを、描画なしのdry-runで先に確認する（4.6節）。
   不可の動画があればここで確定させ、報告に含める。
4. `best`（epoch 6）に対して`validation`で実行。FP上位フレームのPNGとCSVを確認。
5. 同じ引数で`last`（epoch 50）に対して実行。
6. `train_sanity`（3動画）についても同様に実行。
7. 出力は7節どおり`DO_NOT_SHARE`扱い。共有が必要な場合のみ`video_alias`規約で別途匿名化し、
   manifestを共有する場合は既存checkerと同じprivacy self-check正規表現を通す。
8. 9節の問いに対する観察を`docs/stage5/s5-15/`の報告書へ記録する。

## 11. 実施結果と定性観察（2026-09-20）

本節はセクション10の手順で実装・実行した結果と、生成画像の目視観察の記録である。
完了条件2・3・5への回答を兼ねる。

### 11.1 実施内容

| 項目 | 内容 |
| --- | --- |
| 対象 | `stage5_evaluations/260919/pointnext_s_EX260919_s5_15_r0long50_none_gn8_cwfixed_lr1e3_ep50_bs1_acc8_nopad` |
| checkpoint | `best`（epoch 6）と`last`（epoch 50）の両方 |
| split | `train_sanity`（3動画）と`validation`（18動画）＝全21動画 |
| フレーム | **全フレーム出力**（`ALL_FRAMES=1`）。834フレーム×2 checkpoint＝1,668 PNG |
| 描画設定 | `radius=2`、`alpha=0.7`、TN・ignore_otherは非描画 |
| 中間H5の解決 | **21動画すべてが`recorded_source_attr`**。fallbackを要した動画はゼロ |
| 可視化不可 | **0件** |
| 環境確認 | numpy 2.2.2 / h5py 3.13.0 / cv2 4.13.0 / imageio、Stage2to4の`image_to_uint8_gray`すべて利用可 |
| 合成テスト | 実機で全30項目合格（`check_dummy_prediction_frame_visualization.sh`） |

**フレーム選択方針は実行前に変更した。** 5.2節は「21動画×全フレームでは数千枚になるため
上位N件に絞る」を前提にしていたが、実際の総フレーム数は**834**（1動画あたり20〜128）であり、
前提が成り立たなかった。加えて11.2のとおりFPがほぼ全フレームに存在するため、
「FP上位10フレーム」では代表性が得られない。よって全フレーム出力を採用した。
上位N件の機構自体は残してあり、`TOP_FRAMES_PER_VIDEO`で従来どおり使える。

以下で用いる`validation_000`等のaliasは、split内でvideo名を昇順に並べた順序である。
**この対応表はどこにも保存していない**（7節のprivacy方針による）。再現するには同じ並べ替えを行う。

### 11.2 集計から確認された事実（定量。画像ではなくCSVから）

**(a) FPは特定フレームに集中していない。全フレームに散っている。**

`best`では、FPを1点以上含むフレーム数がほぼ全フレーム数と一致した
（`validation_005`は36/36、`validation_008`は66/66、`validation_009`は59/59）。
9節の問い「FPは特定のフレームに集中しているか、全体に散っているか」への答えは
**「散っている」**である。

**(b) epoch 50は「FPが減った」のではなく「positiveを出さなくなった」。**

| checkpoint | TP | FP | FN | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| best (epoch 6) | 35,304 | 574,513 | 49,485 | 5.8% | 41.6% |
| last (epoch 50) | 14,729 | 176,839 | 70,060 | 7.7% | 17.4% |

FPは69%減ったがTPも58%減り、FNは41%増えた。決定的なのは、
**`last`ではvalidation 18動画中10動画でTPがちょうど0**でありながらFPは出し続けている点である
（例: `validation_003`はTP=0/FP=9,531）。FPを含むフレーム数もframes数を下回る動画が増えた
（`validation_007`は5/28）。8.12.11の「positiveを出さなくなっただけではないか」に対し、
**動画単位では「出さなくなった」が正しい**という材料になる。

**(c) `validation_005`は`best`でもTP=0である**（FP=24,120、FN=5,397）。
epoch 6の時点から一度も当たっていない動画が存在する。

**(d) エクスポート自体の正しさの傍証。**
全21動画で`TP + FN`が`best`と`last`で完全に一致した
（例: `validation_017`は2038+10 = 0+2048 = 2,048）。`TP+FN`はGT positive数であり
checkpointに依存しない量なので、これが21動画×2条件で合致することは、
npzとH5のindex対応・GT読み出し・6分類が正しく噛み合っていることの実データ側の裏付けである
（合成テストとは独立した証拠）。

### 11.3 目視による定性観察（`validation`、`best`中心）

**本節は画像を見た印象を文章化したものである。** 観察者自身が「表現が不足・わかりにくい箇所が
ありうる」と留保している。数値的な裏づけは取っていない。11.5に確認が必要な点を列挙する。

**(1) checkpoint間の差**

`best`のほうが`last`よりTP領域の割合が多い。`last`はFPが減ったというより
**positiveの出力自体が消極的になった**と読める。11.2(b)の数値と整合する。
以下(2)〜(5)はすべて`best`についての観察である。

**(2) FPは各フレームの輝度が高い領域・輪郭に発生する**

FPが出やすい場所として次が挙げられた。

- 大腿骨に近い足（太腿や脛）の領域、およびその外周輪郭
- 頭蓋骨・腹部の輪郭。円状の輪郭がところどころ途切れており、
  結果として**少し曲がった細長い線状**に見える
- 大腿骨断面に似た形状の細長いアーティファクト

いずれも**その場所の輝度が高いほどFPが発生しやすい**傾向に見えた。

**(3) 輝度の「高さ」はフレーム内の相対値らしい**

`best`の予測は、**どのフレームからも、そのフレーム画像内で輝度が高い箇所を
できるだけ拾おうとする**傾向がある。その結果、**動画全体で見れば明らかに輝度が低い箇所**にも
FPが発生していた。つまり閾値が動画全体の絶対輝度ではなく、フレーム内の相対的な明るさに
連動しているように見える。

**(4) 細かなノイズ状のFPが点在する**

(2)(3)のような構造物に乗るFPとは別に、微細で散在的なFPも見られた。

**(5) 画面端寄りのアーティファクトにはFPが出ない**

画面端寄りのアーティファクトに対するFPは見られなかった。重要なのは、
**(3)で見られた「明らかに輝度が低い領域へのFP」が存在する一方で、
それより輝度が高い画面端のアーティファクトにはFPが出ていない**という点である。
輝度だけではこの差は説明できない。

### 11.4 観察が示唆すること（解釈であり、観察そのものではない）

以下は11.3からの推論である。**検証されていない仮説として扱うこと。**

- **モデルは「大腿骨」ではなく「フレーム内で相対的に明るい細長い構造」を学習している可能性がある。**
  (2)の3類型（足の輪郭、途切れた円状輪郭が作る細長い線、大腿骨断面に似たアーティファクト）は
  解剖学的には無関係だが、**見た目の形状と輝度が似ている**という共通点を持つ。
- **輝度に加えて位置（画面端からの距離）が効いている。** (5)は輝度単独では説明できず、
  モデルが端領域を抑制する何かを学んでいることを示唆する。GT positiveが端に分布しない
  ことを学習した結果である可能性がある。
- (3)が正しければ、フレーム単位の正規化（前処理・特徴量・モデルのいずれか）が
  絶対輝度の情報を失わせている可能性がある。これは**確認すべき実装上の論点**であり、
  現時点では推測にすぎない。

### 11.5 表現が曖昧で、確認が必要な点

観察者の留保を踏まえ、次は**観察の記述として曖昧**であり、再確認または定量化を要する。

1. **(2)の「領域」と「輪郭」の区別。** FPが乗っているのは明るい面なのか、その縁（エッジ）なのか。
   記述は両方を含むように読めるが、両者は意味が異なる（面＝テクスチャ、縁＝勾配）。
2. **(2)の「大腿骨断面に似た形状」が効いているのか、単に輝度が高いのか**が分離できていない。
   形状の寄与を主張するには、同程度の輝度で形状が異なる領域との比較が要る。
3. **「頭蓋骨・腹部の輪郭が少し曲がった細長い線状に見える」**が、輪郭自体の見え方の記述なのか、
   FPクラスタの形状の記述なのかが一意でない。
4. **(3)の「動画全体では輝度が低い」**は全体の輝度分布と比べた印象であり、数値で確認していない。
5. **(5)の「画面端寄り」の範囲が未定義。** 端から何画素・全体の何割かが示されていない。
6. (4)の「細かなノイズ状」が、単独点なのか小クラスタなのかが未記述。

### 11.6 9節の問いへの現時点の回答

| 問い | 現時点の回答 |
| --- | --- |
| FPはどのオブジェクト上で起きているか | **分類できた。** 足（太腿・脛）の領域と輪郭、頭蓋骨・腹部の途切れた輪郭、大腿骨断面に似た細長いアーティファクト、および微細なノイズ。共通項は「フレーム内で相対的に明るい」こと（11.5-1、11.5-2の留保付き） |
| 集中か分散か | **分散。** ほぼ全フレームにFPがある（11.2(a)、定量的に確定） |
| epoch 6と50でFPの性質が変わったか | **「FPが減った」ではなく「positiveを出さなくなった」。** 10/18動画でTP=0（11.2(b)、定量的に確定） |
| FNに共通の特徴があるか | **未回答。** 今回の観察はFP中心であり、FNの分布は体系的に見ていない |

### 11.7 次に取りうる検証（提案。未承認・追加学習不要・CPU完結）

11.4の仮説はいずれも**既存の成果物だけで定量的に検証できる**。実施する場合は別途承認を得ること。

- **輝度とFPの関係の定量化。** 各点の`pixel_xy`位置の画素輝度を中間H5から読み、
  `prob_femur`（閾値非依存）との関係を、GTクラス別に集計する。
  輝度を**フレーム内percentile**と**動画全体percentile**の両方で表現すれば、
  (3)の「フレーム内相対値らしい」が数値で決着する。
- **画面端からの距離とFPの関係。** 端からの距離（またはcrop中心からの正規化半径）と
  予測positive率の関係を、**輝度を揃えたうえで**比較すれば(5)が検証できる。
- どちらも`predictions/*.npz`・teacher H5・中間H5のみを入力とし、再推論を伴わない。
  S5-13補足の閾値非依存checkerと同じ枠組みで実装できる。

### 11.8 privacy

生成した1,668枚のPNGは**患者フレーム画像そのもの**であり、
`<評価出力>/{best,last}/prediction_frames/`（`DO_NOT_SHARE.txt`を同梱）から
共有可能ディレクトリへ移してはならない。本節の記述は画像を含まず、
実video名も含まない（aliasのみ）。11.2の集計値は匿名化した状態で
privacy self-check（`[0-9]{8}_[0-9]{6}_[0-9]+`と`(/mnt/data|/home/[A-Za-z0-9_.-]+)`）を
通過したものである。
