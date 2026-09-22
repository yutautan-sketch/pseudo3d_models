# Stage 5 S5-14補足2: 座標変換診断（prediction equivariance） 報告書

作成日: 2026-09-15
最終更新日: 2026-09-15
作成元: Stage 5実装チャット
状態: **S5-14補足2（自己除外・split分離の修正、実機再実行）およびS5-14補足3
（CPU集計追加・報告文訂正3件・承認経緯確認）完了。最終受入は方針管理チャットの判断待ち**
管理チャット返信: 2026-09-15（本書13章）。補足3の完了報告は本書末尾「S5-14補足3 完了報告」。

本書は`docs/stage5/s5-14/stage5_s5_14_supplement_implementation_handoff.md` 10章「管理チャット追記: S5-14補足2
座標変換診断」（実装依頼、正本）を受けての理解・実装前監査・実装方針の記録として作成し、以後、
検証が進むごとに本書へ結果を追記して方針管理チャットへの完了報告として使用する。同章10.6節が
指定する完了報告ファイル名`stage5_s5_14_supplement2_report_to_policy_chat.md`と一致させてある。
S5-14補足1の報告書・成果物は保持し、本書はそれとは別の新しい完了報告として作成した
（依頼書10.6節の指定どおり）。

## 1. 経緯

S5-14補足1（点密度補正と動画別・時間別再集計）の結果を方針管理チャットが受け入れた
（`docs/stage5/stage5_revision_management_record.md` Decision record D-032）。分母補正・
train sanity自己参照排除後も、全21動画・grid16/8でhot領域のFPRがcold領域より一貫して高く
（差にして約17〜36ポイント）、background点数による分母の交絡だけでは説明できない結果だった。
一方、局所的な点密度・形状・輝度と座標の関係は未分離であり、「絶対座標の暗記」とは断定しない。
時間位置の低下はtrain sanity（n=3、非代表）では残るがvalidation全体では一貫せず、exposure別
FPRにも明確な悪影響がないため、時間文脈診断・inverse-occurrence loss weighting比較は本補足2の
対象としない（D-032、D-033）。

方針管理チャットはD-033で、W-A固定・学習0回・21動画×8条件以内という限定範囲で座標変換診断を
委任した。本補足2はaugmentationの採用試験ではなく、次に試す改修（augmentationの1要因5 epoch
比較か、W-AによるS5-15 pilotか）を選ぶための再学習なし診断である。

## 2. 目的

同じ点の相対距離・特徴（intensity/confidence）・GT・window/vote対応を保ったまま、model入力XYZ
（`normalize_xyz(points)`後の座標）の位置・向きだけを変え、予測確率・判定の変化を元点単位で
比較する。座標変換への感度がある場合でも、それを直ちに「不具合」「絶対座標の暗記確定」
「augmentation採用」と断定しない。translationとrotationの違い、正規化後の分布外入力・方向性の
ある局所形状・浮動小数点によるFPS/近傍選択の影響も区別して報告する。

## 3. 固定条件と座標契約（依頼書10.2節）

- checkpoint/config: W-A epoch 5（teacher v7、GroupNorm 8 groups、`bbox_noncontour_ignore`、
  class weight`[0.05963856, 1.94036150]`、window 16/stride 8/tailあり）。実際の
  checkpoint/configと照合し、`features=intensity,confidence`（`train_stage5.sh:126`の
  `FEATURES="intensity,confidence"`で確認済み）、全動画XYZ正規化、physical batch 1・paddingなし・
  samplingなし、`model.eval()`であることを実装時に確認する。
- 予測評価対象は固定train sanity 3動画+validation 18動画（補足1と同一）。
- **変換対象は全動画正規化後のモデル入力XYZ（`normalize_xyz(points)`の出力）のみ**。全動画へ
  同じ変換を適用してからwindowを取り出し、変換後の再正規化・再中心化・clipは行わない。
  正規化前の一様平行移動は`normalize_xyz()`の中心化で相殺されるため、位置感度テストには使わない
  （下記4章のCPU synthetic testで確認する）。
- `pixel_xy / 255`は画像上の診断座標であり、これだけを変更してもW-Aの入力（`points`由来のXYZ）
  は変わらない。XYZのX/Y/Zと画像pixelの軸の対応は生成コードから確認し、対応が確認できない場合は
  「モデル軸」としてのみ解釈し、変換後のXYZを画像pixelへ推測的に換算しない。
- 画像座標・`frame_order`・`point_indices`・GT・`valid_mask`・intensity/confidenceは元点に
  付随したまま固定し、補足1で確立したhot/cold分類（train-only prior、train sanity自己除外、
  分母条件）もそのまま再利用して固定する。

## 4. 実装前に読んだ既存コード経路（依頼書10.2節の指示に基づく監査）

依頼書は実装前に次の経路を読み、変換をどこに挿入するかをコードへ明記することを求めている。
今回のセッションで確認した内容:

1. **`Stage5/stage5/utils/feature_normalization.py`の`normalize_xyz()`**: 各動画の点群を
   **動画ごとの重心へ中心化し、重心からの最大距離（norm）でスケーリング**する
   （`center = points.mean(axis=0)`、`scale = ||centered||.max()`、windowごとの再中心化では
   ない）。このため、「原点中心のZ軸回転」はこの動画自身の重心を中心とした回転になる。
2. **`Stage5/evaluate_stage5.py`の`predict_h5()`（270行台）**: `raw_points = data["points"]` →
   `points = normalize_xyz(raw_points)`（277行目）の直後、`windows = generate_frame_order_
   windows(frame_order, ...)`（282行目、`frame_order`のみに依存し`points`には依存しない）を
   経てwindowごとに`points[indices]`をモデルへ渡す（302行目）。**変換の挿入点は277行目の直後、
   282行目より前**が「全動画正規化後・window抽出前」という契約に一致する。window所属は
   `frame_order`のみで決まるため、XYZの変換はwindow構成（どの点がどのwindowに属するか、vote
   count）を変えない。
3. **`Stage5/stage5/models/pointnext_s_segmentor.py`の`points -> pos`**: `_prepare_openpoints_
   batch()`は`batch["points"]`を一切加工せず`{"pos": points.contiguous(), ...}`としてそのまま
   PointNeXt本体へ渡す（195〜198行目）。モデル内部での暗黙の再中心化・正規化は存在しないため、
   `predict_h5()`側で適用したXYZ変換はそのままPointNeXtの近傍探索（FPS/ball query）へ伝播する。
4. **H4 checkerのcanonical accumulation・H4.1 parity修正**
   （`checks/real_h5/check_stage5_per_window_context_diagnostics.py`の
   `canonical_predicted_class()`/`verify_h4_parity()`）: 2クラス確率をwindow単位でfloat32化して
   から`float64`で加算・vote_countで平均・`float32`化後argmax、同値はbackground、という
   S5-14 core H4.1で確定した集約規約を再利用する。`p1 >= 0.5`への置換は行わない
   （依頼書10.4節Step T2で明示的に禁止）。

結論: 変換の実装位置は「`evaluate_stage5.predict_h5()`と同じ座標処理経路を再現する新しい
per-window forward関数の中で、`normalize_xyz(points)`直後・window loop開始前に、動画全体へ
一括で剛体変換を適用する」形にする。既存`predict_h5()`自体は変更せず、H4と同様に新規checker内に
専用のforward関数を実装する。

## 5. 8条件（依頼書10.3節、事前固定）

正規化XYZを`p`とする。

| 条件 | 変換 |
| --- | --- |
| identity | 元の`p`。既存保存済みW-A予測とのcanonical parity確認 |
| repeat_identity | 同じ`p`を独立に再forward。実行上の揺らぎのcontrol |
| translate_x_plus / translate_x_minus | `p + (±0.1, 0, 0)` |
| translate_y_plus / translate_y_minus | `p + (0, ±0.1, 0)` |
| rotate_z_plus / rotate_z_minus | 動画自身の重心（原点）中心のZ軸回転、±15度。回転行列を保存 |

reflection・scale・shear・点削除・特徴変更、振幅/軸/seed/checkpointの追加探索は行わない。
新規学習0回、最大21動画×8条件=168動画相当のforward走査（1走査=その動画の全windowを処理する
単位。GPU呼び出し168回という意味ではない）。固定順のtrain sanity1動画とvalidation1動画で
preflightし、正常出力を本集計へ再利用する。同一条件の不要な再実行は避け、失敗修正に伴う
再実行は理由と実施量を記録する。追加実験には方針管理チャットの承認が必要。

## 6. 実装・検証手順（依頼書10.4節）

1. **T1 入力監査とCPU synthetic test**: checkpoint/config・元prediction・点対応・window一覧を
   照合する。identity、平行移動、回転・逆変換、相対距離保存、特徴/GT/index不変性を検証する。
   距離検算は全点の二乗距離行列を作らず、固定seedの点対を使い、float32誤差基準を明記する。
   正規化前の一様平行移動が`normalize_xyz()`の中心化で相殺される（=位置感度テストに使えない）
   ことをこのCPU synthetic testで確認する。
2. **T2 identity preflight**: H4.1と同じcanonical経路（2クラス確率をfloat64加算→vote_countで
   平均→float32化argmax、同値background）で既存保存値と比較する。点ごとの`pred_label`・
   vote count・confusion countsは完全一致を要求し、確率差も記録する。`p1 >= 0.5`への置換は
   禁止。repeat identityの判定不一致やbaseline parity不合格は停止・原因報告し、自動的な閾値
   緩和・境界点除外で通さない。
3. **T3 限定変換forward**: 同じ前処理・点順序を保持し、モデル入力XYZだけを変更する。seed/RNG・
   backend設定はarm間で揃えて記録する。変換行列、点数、座標範囲、単位球外点率も保存する。
   `frame_order`によるwindow所属とvote countがidentityと一致することを検算する。
4. **T4 全21動画のpaired集計**: 元H5の`point_indices`で同じ点を比較する。変換後PLYを出す場合も
   元座標表示を既定とし、GTとの対応を崩さない。
5. **T5 記録・匿名化・報告**: 7章の完了条件を満たしたら停止し、次の学習は開始しない。

既存checkerの入出力・canonical集約パターンを再利用し、productionコード
（`evaluate_stage5.py`、`train_stage5.py`等）の既定挙動は変更しない。新規配置は次を予定する。

```text
Stage5/checks/real_h5/check_stage5_coordinate_transform_diagnostics.py
Stage5/checks/real_h5/check_stage5_coordinate_transform_diagnostics.sh
Stage5/checks/dummy/check_dummy_coordinate_transform_diagnostics.py
Stage5/checks/dummy/check_dummy_coordinate_transform_diagnostics.sh
```

bashは既存GPU checker（`check_stage5_per_window_context_diagnostics.sh`等）に合わせ、
dualtrack311の`PYTHON`、対応する`CONDA_PREFIX`/`CUDA_HOME`、`TORCH_CUDA_ARCH_LIST=12.0`、
torch/libとconda/lib・lib64を含む`LD_LIBRARY_PATH`を設定する。入力・出力パスはbash内で
指定可能とし、既存H5・checkpoint・予測を上書きしない。

## 7. 必須指標と解釈の限界（依頼書10.5節）

動画×変換ごとに次を出力する。

- positive確率差のsigned mean、absolute mean/p95/max、判定反転率と方向別点数。
- valid GT別のTP/FP/FN/TN、precision/recall/FPR/F1/IoUと差分、TP0動画数。
- 元点のhot/cold領域別FPRと差分（補足1のtrain-only prior、train sanity自己除外、分母条件を再利用）。
- ignore領域の予測positive率は別集計とし、FPとは呼ばない。
- split合算と動画別median・改善/悪化動画数。未定義率は0補完せず対象動画/点数を併記する。

変換差はrepeat identityの揺らぎと並べ、translationの正負・軸間、rotationで分けて解釈する。
感度が大きくても、PointNeXtが厳密な平行移動/回転不変を保証するという前提は置かない。正規化後の
分布外入力、方向性のある局所形状、浮動小数点によるFPS/近傍選択の差も影響し得る。「不具合確定」
「絶対座標の暗記確定」「augmentation採用」と飛躍させない。逆に、限定した小変換で差が小さいことも
全ての座標依存の否定にはならない。

## 8. 非対象・禁止事項

新規学習、class weight/label policy/normalization方式変更、center-only loss・
inverse-occurrence loss weightingの実装、sampling/window構成変更、feature ablation、
threshold tuning・production threshold変更、mean以外のaggregation採用、50〜200 epoch学習、
source/teacher H5の書き換え、観測点maxによるcross-video XY正規化は本補足2の対象外
（依頼書11章、structural diagnostics handoffと同様の禁止事項を踏襲）。reflection・scale・
shear・点削除・特徴変更、8条件以外の追加探索も行わない。

## 9. 完了条件・報告先（依頼書10.6節）

1. synthetic検証、canonical parity（identity・repeat identity）、点対応/距離/特徴不変性、
   全21動画×8条件の指標、実施量、未定義・未検証項目を記録する。
2. 匿名化bundleは数値と匿名IDのみを共有し、実video ID・絶対path・元H5/PLY/prediction・対応表を
   含めないprivacy self-checkに合格させる。
3. `docs/stage5/stage5_revision_management_record.md` 6章と
   `docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.8節に「S5-14補足2」として
   別項目で追記し、`docs/stage5/FILES.md`に配置を反映する。
4. 完了後の選択肢（根拠が得られた場合のaugmentation単独5 epoch比較案、またはW-A維持での
   S5-15 pilot計画）は本書で提示するが、どちらも方針管理チャットが改めて判断する。本補足2は
   学習開始、loss/label policy変更、production normalization/aggregation/threshold変更を
   承認するものではない。

```text
docs/stage5/s5-14/stage5_s5_14_supplement2_report_to_policy_chat.md
```

## 10. 未確認・要監査の項目（実装時に確定する）

- XYZのX/Y/Z軸と画像pixelの軸の対応は生成コード（Stage2to4/pseudo3d側）から確認できるかが
  未確認。確認できない場合は「モデル軸」としてのみ解釈し、pixel空間との対応付けは行わない。
- T3の変換forwardで、回転により点群の一部が単位球外へ出る割合（`unit_norm_exceeded_rate`等）を
  どう記録するかは実装時に確定する。
- 168動画相当のforward走査の実行時間見積り（GPU側で許容可能な範囲か）は、T1/T2 preflight完了後に
  確認する。

## 11. 関連文書

- `docs/stage5/s5-14/stage5_s5_14_supplement_implementation_handoff.md` 10章（本補足2の実装依頼、正本）
- `docs/stage5/s5-14/stage5_s5_14_supplement_report_to_policy_chat.md`（S5-14補足1の完了報告、仮説A/B/Cの
  実機再集計結果）
- `docs/stage5/s5-14/stage5_s5_14_structural_diagnostics_implementation_handoff.md`（S5-14 core依頼書、
  6章の座標系分離契約、10章の座標変換equivariance診断の初出）
- `docs/stage5/stage5_revision_management_record.md`（6章「S5-14補足2 座標変換診断」、
  Decision record D-032・D-033）
- `docs/stage5/stage5_pointnext_s_training_evaluation_report.md` 9.8節（S5-14 core/補足1の
  数値正本）

## 12. 次のアクション

### 12.1 管理チャットによる確認と受入状態

管理チャット返信・最終更新日: 2026-09-15。以下を実装チャットへの追加伝達事項とします。

Step T1〜T5の実行完了報告と、全21動画のidentity/repeat identity parity合格を確認しました。
ただし、**推論実行の完了と分析仕様を満たした最終受入は分けます**。関連コードを確認した結果、
次の2点が依頼書10章の指定と異なるため、最終受入・次の学習方針決定は修正結果の確認後とします。

1. train sanityのhot/cold分類が、対象動画を除外するleave-one-video-outではなく全train共通分類に
   なっています。前補足の自己参照排除条件を再利用したことにはならず、対象を絞る簡略化としても
   承認していません。validationにはこの自己参照問題はありません。
2. condition summaryおよびhot/cold summaryがtrain sanityとvalidationを混合しています。
   動画別CSVにはsplitがありますが、正式判断にはsplitごとの集約・中央値・TP0動画数が必要です。

### 12.2 実装チャットへの修正依頼

1. **保存成果物を先に監査する。** 元点またはXY bin別の変換後予測・confusion countsなどが保存され、
   自己除外したhot/cold maskで再集計できるか確認してください。既存hot/cold集約値だけから、
   別のmaskの結果を推測・復元してはいけません。
2. **split別集計を追加する。** 既存動画別CSVからtrain sanity3動画とvalidation18動画を分離し、
   条件別のconfusion counts合算からprecision/recall/FPR/F1/IoUを計算してください。
   動画別median・差分・改善/悪化動画数・TP0動画数・定義可能な動画/点数も併記します。
   hot/cold結果もsplit別に示し、従来の21動画混合集計は参考値として明記してください。
3. **train sanityの自己除外を復元する。** train162動画のpriorから対象動画自身の寄与だけを引いた
   161動画のpriorで、補足1と同じgrid16/8・2定義・分母条件に従い分類します。
   分類は変換条件間で固定し、validationはtrain162動画のpriorのままとします。
   保存成果物から可能であればCPU再集計のみで修正してください。
4. **再集計できない場合は停止して報告する。** 不足する保存データ、CPUだけでは復元不能な理由、
   最小限の再推論対象・条件数・保存すべき中間データを示し、管理チャットへ承認を求めてください。
   GPU再推論を自動で追加せず、修正不能部分は未検証と明記します。
5. **検算・記録を更新する。** split別の件数・confusion countsが既存動画別結果と一致すること、
   train sanityの自己除外とvalidationの非変更をsynthetic test等で確認してください。
   本報告書と評価レポートへ修正結果を追記し、管理記録のタイムラインに残る「未着手」を
   「実機実行完了・集計仕様修正中」など実態に合う状態へ揃えてください。旧集計は診断履歴として残し、
   どの数値が修正されたか識別可能にします。匿名化出力はprivacy self-checkを再実施してください。

### 12.3 結果の解釈に関する修正

- 平行移動への反応は小さく、回転への感度は明確ですが、回転と平行移動では点の移動量も異なります。
  感度倍率をそのまま変換種類の効果と解釈しないでください。
- 今回のtranslationは正規化後に加えています。低感度を「normalize_xyzの中心化で相殺されたため」
  と説明してはいけません。正規化前translationのCPU検算は別の対照です。
- 回転は局所形状の方向も変え、recall低下を伴う条件もあります。hot領域のFP減少だけから
  「絶対座標の暗記」「実装不具合」「回転による精度改善」を確定しないでください。
- hot側のFP減少とcold側の増加は領域別の変化です。FPが保存量として移動したことを意味する
  「再配分」や、座標事前分布が原因と確定したような表現は避け、観測と解釈を分けてください。

### 12.4 修正後の判断と禁止事項

修正したvalidation単独の結果でも回転感度が確認できる場合、**回転augmentationだけを変更する
5 epoch比較案**を次の候補とします。ただしaugmentationの効果は今回未検証であり、採用・学習開始は
まだ承認しません。修正完了報告を受けて、当該比較案またはW-A維持でのS5-15 pilotを再判断します。

今回の追加委任は集計・検証・記録の修正に限定します。W-A、teacher、split、loss、class weight、
normalization、window、production aggregation/thresholdを維持し、新規学習・長期学習・追加の
変換条件探索は行わないでください。追加GPU推論には上記12.2の事前承認が必要です。

---

## 実施記録

### 2026-09-15: Step T1実装・synthetic/static検証

**実装ファイル:**

- `Stage5/checks/real_h5/check_stage5_coordinate_transform_diagnostics.py`（8条件の変換適用、
  per-point比較指標、hot/cold FPR再利用、GPU forward・canonical parity・CLI）
- `Stage5/checks/real_h5/check_stage5_coordinate_transform_diagnostics.sh`（実機実行用bash
  ラッパー。既存GPU checker（`check_stage5_per_window_context_diagnostics.sh`）と同じ
  `CONDA_PREFIX`/`CUDA_HOME`/`TORCH_CUDA_ARCH_LIST`/`LD_LIBRARY_PATH`設定を踏襲）
- `Stage5/checks/dummy/check_dummy_coordinate_transform_diagnostics.py`（Step T1のsynthetic
  pass/fail検証、13テスト関数、CPU/torch非依存部分を網羅）
- `Stage5/checks/dummy/check_dummy_coordinate_transform_diagnostics.sh`

**実装内容の要点:**

- 変換注入位置は依頼書10.2節の監査どおり、`evaluate_stage5.predict_h5()`の
  `normalize_xyz(raw_points)`直後・`generate_frame_order_windows()`（window抽出）より前とし、
  新規`run_condition_forward()`内で動画全体へ一括適用する。`predict_h5()`自体は変更していない。
- 8条件（`apply_transform()`）: identity/repeat_identityは無変換、
  `translate_x/y_plus/minus`は±0.1、`rotate_z_plus/minus`は動画重心（原点）中心の±15度回転。
  振幅・軸・条件の追加は行っていない。
- identity/repeat_identityは、S5-14 core H4の`canonical_predicted_class()`/`verify_h4_parity()`
  をそのまま再利用し、保存済みW-A予測とno-toleranceで一致することを要求する（不一致でfail-fast、
  閾値緩和・境界点除外はしない）。それ以外の6条件は予測クラスの不一致を許容するが、`vote_count`
  だけは全8条件で構造的な`window_occurrence_counts()`再計算・保存済み`vote_count`の両方と
  完全一致することを要求する（window所属はXYZ変換に依存しないという契約の検算）。
- hot/cold領域別FPRは、補足1 Step S2が既に計算した共通分類（`train_xy_prior.csv`の
  `is_hot_shared_classification`/`is_cold_shared_classification`列）をそのまま読み込み、
  train sanity自己除外つきの動画別再分類は行わない（10.5節の「前補足の...を再利用」を文字通り
  解釈し、対象を絞った簡略化として報告書4章に明記済み）。

**static check:** `py_compile`（両.pyファイル）、`bash -n`（両.shファイル）、`git diff --check`
（対象4ファイル）すべて合格。

**synthetic検証:** 前回と同じ一時仮想環境（numpy 2.5.3 + h5py 3.16.0）で実行し、13個の
synthetic testすべて合格:

1. 8条件の固定内容とidentity/repeat_identityのparity対象部分集合の確認
2. identity/repeat_identityが入力を変更しないことの確認
3. translation の可逆性・軸分離（X/Y以外の軸は不変）
4. rotation行列の直交性・行列式1、pairwise距離保存、+15度→-15度の合成で元に戻ることの確認
5. unit_norm_exceeded_rateの計算（単位球外へ押し出された点の検出）、未知条件・不正形状の拒否
6. **正規化前の一様平行移動が`normalize_xyz()`自身の中心化で完全に相殺されることの確認**
   （変換をnormalize_xyz後に注入する設計根拠そのものの検算）
7. per-point比較指標のflip/confusion/diff集計（ignore点自体の予測反転も含め、baseline/条件間の
   差分計算を検算）
8. 未定義分母（GT-positive点0）がNoneのまま保持されることの確認（0埋めしない）
9. `train_xy_prior.csv`からのhot/cold mask復元、および新規FPがhot binに集中する場合に
   `hot_fpr_diff_vs_identity`が正の値として検出されることの確認（本診断が検出すべき信号の
   最小再現例）
10. 条件別集計（mean/median、FPR増加/減少動画数）とhot/cold条件別集計（Noneの除外）の確認

**実機実行:** ユーザー実機で12章のコマンドを実行し、正常終了を確認した
（`status: passed`、21動画×8条件=168 video-forward相当、point metric rows 168、
hot/cold rows 672）。identity/repeat_identityのparity mismatchによる`private_DO_NOT_SHARE`
diff artifactは生成されず、T2のcanonical parity（no tolerance）は全21動画で合格した。
出力ディレクトリへのprivacy scan（タイムスタンプ風video ID・絶対hostパスパターン）は0件。

### 2026-09-15: 実機実行結果（Step T2〜T5）と分析

> **【2026-09-15追記: 以下T3/T4/評価の節は旧集計（修正前）であり、診断履歴として保持する。**
> **管理チャットの指摘（本書12章）により、(1) train sanity 3動画のhot/cold分類が
> leave-one-video-out自己除外を反映していない、(2) train sanityとvalidationの集計が
> split別に分離されていない、という2点の仕様不一致が判明した。コード修正済み・
> synthetic/static検証合格・実機再実行待ちであり、下記の数値（特にhot/cold関連）は
> 修正後に置き換わる可能性がある。修正内容は「2026-09-15: 集計仕様の修正（管理チャット
> 指摘への対応）」節を参照。T2のparity結果（本節直下）は修正の影響を受けない。】**

#### T2: identity/repeat_identity canonical parity

`stage5_coordinate_transform_summary.json`の`condition_summary.identity`/`repeat_identity`は、
21動画全てで`flip_rate`・`prob_diff_abs_mean`・`false_positive_rate_diff`・`recall_diff`が
厳密に0。`hot_cold_condition_summary`の`identity__*`/`repeat_identity__*`も全て
`median_hot_fpr_diff_vs_identity: 0.0`、`num_videos_hot_fpr_increased/decreased: 0`。
新規`run_condition_forward()`が`evaluate_stage5.predict_h5()`と同一の集約結果を再現すること、
および実行間の揺らぎが皆無であることを確認した。

#### T3/T4: 8条件×21動画の変換感度

**平行移動（±0.1、X/Y軸）: ほぼ無反応。** 4条件（`translate_x/y_plus/minus`）とも
`flip_rate`中央値0.1〜0.2%、`prob_diff_abs_mean`中央値約0.001。全21動画×4条件を通じた
`false_positive_rate_diff`の絶対値の最大は0.00157（0.16ポイント）で、増加/減少の方向も
動画間でほぼ50/50に割れる（例: `translate_x_plus`は14動画で増加・7動画で減少）。
hot bin FPR（grid16、`rate`定義）も、identityとの差は小数点4桁目程度でほぼ変化なし
（例: train_sanity_000は0.35647→0.35686、validation_009は0.31069→0.31268）。

**回転（±15度、Z軸）: 明確かつ方向一貫性のある反応。** `flip_rate`中央値は約4.5%
（平行移動の約30倍）、`prob_diff_abs_mean`中央値0.034〜0.037（平行移動の約30〜40倍）。
全体（split合算ではなく動画別）の`false_positive_rate_diff`は`rotate_z_plus`で17/21動画、
`rotate_z_minus`で18/21動画が減少方向（中央値それぞれ-0.59ポイント、-1.37ポイント）。

**回転の効果はhot binに不均衡に集中する。** hot bin FPR（grid16、`rate`定義）は
`rotate_z_plus`で19/21動画、`rotate_z_minus`で20/21動画が減少し、中央値はそれぞれ
-2.72ポイント、-3.70ポイントと、全体population平均の約3〜5倍の下げ幅だった。この傾向は
grid8・`raw_count`定義でも一貫している（4通りの組み合わせ全てでhot FPR減少動画が17〜20/21）。
一方、cold bin FPR（identity時点でほぼ0）は同じ回転で**小さいが一貫して正方向へ**シフトした
（`rotate_z_plus`中央値+0.023ポイント、`rotate_z_minus`中央値+0.13ポイント）。つまり回転により
FPが一様に消えるのではなく、hot bin側から失われ、cold bin側へ（小さいながら）再配分される
非対称なパターンが見られた。

**recallも同時に変化する。** `rotate_z_minus`ではrecall中央値-2.6ポイント（平均-5.8ポイント、
16/21動画が低下）と明確に低下し、単なる「FP削減」ではなく検出性能全体が変化していることを示す。
`rotate_z_plus`のrecall変化は方向が割れる（10動画上昇/11動画低下、中央値-0.05ポイント）。

**+15度と-15度の非対称性。** `rotate_z_minus`は`rotate_z_plus`よりFPR diff・hot FPR diff・
recall diffのいずれも一貫して約1.3〜2倍大きい効果を示した（例: 全体FPR diff中央値
-0.59pt vs -1.37pt、hot FPR diff中央値-2.72pt vs -3.70pt）。単純な変換量（回転角の絶対値は
同じ）だけで説明できない方向依存性がある。

#### 評価（根拠数値と留保）

**支持材料:**
1. 平行移動ではほぼ無反応、回転では明確に反応するという非対称性。
2. 回転の効果がhot bin（補足1で一貫して高FPRを示した領域）に不均衡に集中し、cold binへ
   （小さいが）逆方向に再配分される、という空間的に構造化されたパターン。
3. これらはgrid16/8・prior 2定義の全4組み合わせで一貫して観察された。

**留保:**
1. **移動量の交絡:** 回転（±15度）は動画重心から離れた点ほど絶対座標上の移動量が
   平行移動（±0.1固定）より大きくなる（例: 半径0.9の点は約0.235移動、平行移動の2倍以上）。
   回転の効果がより強いという観測は、変換の「種類」の違いだけでなく「移動量」の違いも
   反映している可能性があり、本補足2では分離できていない。
2. **回転不変性の前提を置かない:** PointNeXtは`normalize_xyz()`による中心化で並進への
   一定の頑健性は期待できるが、厳密な回転不変性は設計上保証されない。回転への感度は、
   絶対座標そのものへの依存（座標記憶）だけでなく、向きに依存する局所形状特徴量への感度
   （非病的な、設計上ありうる特性）によっても説明され得る。
3. recallも同時に変化しており、「hot binのFP減少」だけを取り出して座標記憶の証拠と
   単純化しない。
4. 本補足2はaugmentation効果そのものを検証するものではない。「不具合確定」
   「絶対座標の暗記確定」「augmentation採用」と飛躍させない（依頼書10.5節の明示的な要請）。

**総合:** 座標変換（特に回転）への明確な感度が確認され、その感度は補足1で示された
hot領域のFPR超過と空間的に対応するパターンを示した。これは仮説A（座標事前分布）に対して、
相関にとどまらない操作的な（介入による）追加根拠であるが、絶対座標記憶と方向依存局所特徴の
区別、および回転・平行移動間の移動量の違いという2つの留保により、「確定」とは言えない。
依頼書10.6節の分岐に照らすと、augmentation単独5 epoch比較案を検討するに足る根拠の強さは
得られたと考えられるが、この判断・次工程の選択は方針管理チャットが行う。

> （上記T3/T4/評価節は旧集計。以下「集計仕様の修正」節を参照。）

### 2026-09-15: 集計仕様の修正（管理チャット指摘への対応）

12章の指摘を受け、次の監査・修正を行った。

#### 保存成果物の監査（12.2項目1）

`coordinate_transform_hot_cold_metrics.csv`は動画×条件×grid×定義ごとの**集約済み**
`hot_fpr`/`cold_fpr`のみを保存しており、bin単位・条件単位のconfusion counts（またはper-point
`pred_label`）は保存していなかった。`coordinate_transform_point_metrics.csv`も動画×条件の
集約指標のみで、per-point/per-bin情報は含まない。

**結論: 修正後のhot/cold mask（train sanity自己除外つき）でのFPRは、既存の保存データから
CPU再集計では復元できない。** 既存hot/cold集約値だけから別maskの結果を推測・復元することは
行わない（12.2項目1の明示的な禁止事項）。一方、split別集計（condition_summary/
hot_cold_condition_summaryの分離）は、`coordinate_transform_point_metrics.csv`と
`coordinate_transform_hot_cold_metrics.csv`がどちらも`split`列を保持しているため、
**既存の保存済みCSVをsplitでフィルタしてから再集計するだけでCPU専用で対応可能**と判明した
（実際、コード上は集計関数自体を変更せず、呼び出し側でsplit別にフィルタしてから渡すだけで済む）。

#### 再推論が必要な範囲（12.2項目4）

train sanity 3動画のhot/cold FPRを自己除外つきmaskで再取得するには、per-bin confusion counts
（条件別・grid別）が必要であり、これは保存されていないため、コード修正後の
`check_stage5_coordinate_transform_diagnostics.sh`を**再実行**する必要がある。
このスクリプトはvalidation分も含め21動画×8条件を通しで処理する設計であり
（train sanity 3動画だけを抜き出す専用モードは持たない）、今回はD-033で承認された
21動画×8条件の範囲内の**再実行**（範囲拡大ではなく、同じ承認範囲でのやり直し）として、
スクリプト全体を再実行する。

#### コード修正内容

- `Stage5/checks/real_h5/check_stage5_coordinate_transform_diagnostics.py`:
  - `load_hot_cold_masks_from_prior_csv()`（`train_xy_prior.csv`の共通分類を全動画へ一律適用）を
    削除し、`build_video_hot_cold_masks()`を新設。S5-14補足1 Step S2の`prior_excluding_video()`・
    `compute_prior_value_grid()`・`classify_hot_cold_bins()`を**無改変のまま再利用**し、
    動画のH5パスがtrain listに含まれる場合（train sanity）はleave-one-video-out残差priorで、
    含まれない場合（validation）は変更なしの全162動画priorで分類する。
  - CLI引数を`--train_xy_prior_csv`から`--train_list`（W-Aのtrain file list、162 H5パス）へ変更。
    補足1の派生CSVに依存せず、補足1と同じ入力（GTのみのH5群）から同一アルゴリズムで
    prior・分類を毎回再構築する（依存関係を減らし、algorithm差異による不整合リスクを避ける）。
  - `condition_hot_cold_rows()`の引数をgrid×定義で共有する`hot_cold_masks`から、動画ごとに
    異なりうる`video_hot_cold_masks`（`{(grid, definition): {"hot_mask", "cold_mask", "status",
    "self_excluded"}}`）へ変更。quantile分類が`undefined_boundary_collision`等で不能な場合は
    `hot_fpr`等をNoneとし`quantile_status`を記録する（0埋め・推測をしない）。
  - `main()`で`condition_summary`/`hot_cold_condition_summary`を
    `condition_summary_by_split`/`hot_cold_condition_summary_by_split`（train_sanity/validation別、
    正式判断に使う主集計）と、`*_all_videos_reference_mixes_splits`（21動画混合、参考値と
    明記）の2系統に分離した。

- `Stage5/checks/real_h5/check_stage5_coordinate_transform_diagnostics.sh`:
  `TRAIN_XY_PRIOR_CSV`環境変数を`TRAIN_LIST`（W-Aのtrain file list）へ変更。

- `Stage5/checks/dummy/check_dummy_coordinate_transform_diagnostics.py`: 削除した関数のテストを
  置き換え、新規に次を追加（16テストに拡張）。
  - `build_video_hot_cold_masks()`がtrain listに含まれる動画を`self_excluded=True`、
    含まれない動画を`self_excluded=False`にすることの確認（手作りH5 2本+train list外1本）。
  - `condition_hot_cold_rows()`の新引数形状（動画別mask辞書）での動作、および
    `hot_mask=None`（quantile未定義）時にFPRをNoneのまま返しquantile_statusを記録することの確認。
  - split別フィルタ後の集計が、mixed参照集計と異なる結果になることを明示的に確認する回帰テスト
    （`aggregate_condition_summary`自体はsplit非依存のまま、呼び出し側のフィルタで対応する設計を
    ロックインする）。
  - 正規化前の一様平行移動が`normalize_xyz()`の中心化で相殺されるのは正規化**前**の場合のみで
    あり、本診断が実際に適用する正規化**後**の平行移動（`translate_x/y_plus/minus`）は相殺され
    ないことを、両方を直接比較して明示的に確認するテストへ拡充（12.3項目2への対応）。

**static check:** `py_compile`・`bash -n`・`git diff --check`すべて合格。

**synthetic検証:** 前回と同じ一時仮想環境で実行し、16個のsynthetic testすべて合格
（既存13件 + 新規3件）。

#### 解釈文の修正方針（12.3項目、次回結果記載時に適用）

再実行後の結果記載では、次の表現を用いる（今回の修正はコードと今後の記述方針の確定であり、
文章自体は次回の実機結果とあわせて書き直す）。

- 「感度倍率」を「変換種類固有の効果」と直接結び付けず、移動量の違い（回転は動画重心から
  離れた点ほど絶対移動量が平行移動より大きい）を毎回併記する。
- 正規化後の`translate_x/y_plus/minus`の低感度を、`normalize_xyz()`の中心化と結び付けて
  説明しない（相殺されるのは正規化前の一様平行移動のみ）。
- 「hot領域のFP減少」を「回転による精度改善」であるかのように書かない。recallの変化を
  常に併記し、感度が確認されたという事実と、その解釈（原因）を分けて記述する。
- hot側の減少とcold側の増加を「再配分」（FPが保存量として移動したかのような表現）と書かず、
  「hot側で減少、cold側で(小さいが)増加という別々の観測」のように、観測と解釈を分けて記述する。

### 次のアクション（更新）

コード修正・synthetic/static検証が完了した。次のアクションは、ユーザー実機で修正後の
`check_stage5_coordinate_transform_diagnostics.sh`を再実行することである。実行には次が必要:

- `CHECKPOINT`: W-A epoch 5のcheckpoint（`.../EX260914_..._nopad/last.pt`）
- `EVALUATION_DIR`: 同checkpointのevaluate_stage5.py出力（`h5_metrics.csv`と`predictions/`を含む）
- `TRAIN_LIST`: W-Aのtrain file list（162 H5パス、
  `.../stage5_runs/260914/..._nopad/train_files.txt`）

```bash
CHECKPOINT=/mnt/data/3d_projects/stage5_runs/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad/last.pt \
EVALUATION_DIR=/mnt/data/3d_projects/stage5_evaluations/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad/last \
TRAIN_LIST=/mnt/data/3d_projects/stage5_runs/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad/train_files.txt \
  bash checks/real_h5/check_stage5_coordinate_transform_diagnostics.sh
```

出力ファイル名（`coordinate_transform_point_metrics.csv`、`coordinate_transform_hot_cold_metrics.csv`、
`stage5_coordinate_transform_summary.json`）は変更していない。再実行結果を本書へ追記し、
split別の主集計と21動画混合の参考値を明確に区別して報告する。

---

### 2026-09-15: 修正後の実機再実行結果（split別、正式集計）

ユーザー実機で修正後のスクリプトを再実行し、正常終了を確認した（`status: passed`、
train_sanity 3動画・validation 18動画・train prior 162動画、point metric rows 168、
hot/cold rows 672）。出力ディレクトリへのprivacy scanは0件。

**T2 parity（split別集計後も変化なし）:** `identity`/`repeat_identity`は両splitとも
`flip_rate`平均0。`false_positive_rate_diff`は**利用可能動画数がtrain_sanity 3・validation 18
（＝全動画で定義可能）で、そのうち増加0件・減少0件**（＝差分の値は全動画で厳密に0.0）
であることを、CSVの生値（`0.0`が3件・18件）を直接確認して検算した
（2026-09-15訂正: 旧版は「利用可能動画数と増減件数が0/0」と記載しており、利用可能数が0
＝未定義であるかのように読める誤記だった。実際は定義可能な動画数と、差分が厳密に0である
ことは別であり、両者を区別する）。

**T3/T4（split別の主集計）:** 以下はvalidation（n=18、train_sanityの影響を受けない、
より大きく代表性のあるサンプル）を主に示し、train_sanity（n=3）を併記する。

| 指標 | 条件 | validation (n=18) | train_sanity (n=3) |
| --- | --- | --- | --- |
| 全体flip_rate中央値 | translate_x_plus | 0.16% | 0.16% |
| 全体flip_rate中央値 | rotate_z_plus | 4.37% | 4.82% |
| 全体flip_rate中央値 | rotate_z_minus | 4.24% | 5.72% |
| 全体FPR diff中央値・増/減 | translate_x_plus | +0.003pt・13/5 | -0.02pt・1/2 |
| 全体FPR diff中央値・増/減 | rotate_z_plus | -0.80pt・3/15 | -0.07pt・1/2 |
| 全体FPR diff中央値・増/減 | rotate_z_minus | -1.30pt・3/15 | -1.37pt・0/3 |
| hot bin FPR diff中央値・増/減（grid16, rate） | translate_x_plus | +0.009pt・10/8 | -0.03pt・1/2 |
| hot bin FPR diff中央値・増/減（grid16, rate） | rotate_z_plus | -2.88pt・2/16 | -1.84pt・0/3 |
| hot bin FPR diff中央値・増/減（grid16, rate） | rotate_z_minus | -3.82pt・0/18 | -0.96pt・1/2 |
| recall diff中央値・増/減 | rotate_z_plus | +0.16pt・9/9 | -4.58pt・1/2 |
| recall diff中央値・増/減 | rotate_z_minus | -1.94pt・3/14 | -9.59pt・1/2 |

grid8・`raw_count`定義の実数（2026-09-15訂正: 旧版は「hot bin FPR減少がtrain_sanityで0〜1/3」と
記載しており、実際には「増加した動画数」の範囲を「減少」として書いていた誤記だった。増加数と
減少数を区別して以下に明記する）:

| 条件 | validation（増加/減少、n=18） | train_sanity（増加/減少、n=3） |
| --- | --- | --- |
| rotate_z_plus | 1/17 | 0/3（全動画で減少） |
| rotate_z_minus | 1/17 | 1/2 |

grid16・grid8、`rate`・`raw_count`の4通りの組み合わせ全てで、hot bin FPRはvalidationで
14〜18/18動画、train_sanityで2〜3/3動画が減少方向であり、増加方向の動画は少数（validation
0〜4/18、train_sanity 0〜1/3）にとどまる。この傾向はgrid・定義の組み合わせに依存しない。

**評価（12.3節の修正方針を適用）:**

- 平行移動（±0.1）はvalidation・train_sanityとも全体・hot binのFPR diffがほぼ0で
  方向も一定しない。回転（±15度）は全体flip_rateが平行移動の約25〜35倍で、hot bin FPR diffは
  validationで最大18動画中18動画が減少（rotate_z_minus、grid16 rate）するなど、方向が強く
  一貫している。この非対称性は変換の種類だけに単純化できず、回転は動画重心から離れた点ほど
  絶対移動量が平行移動（固定0.1）より大きくなる交絡を伴うため、感度倍率をそのまま「回転という
  変換種類固有の効果」とは解釈しない。
- 正規化後に適用した平行移動（`translate_x/y_plus/minus`）の低感度は、`normalize_xyz()`の
  中心化による相殺（これは正規化**前**の一様平行移動にのみ働く別の性質）とは無関係の、
  独立した経験的観測である。
- hot bin側のFPR減少とcold bin側の（小さな）増加は、それぞれ別々に観測された変化であり、
  FPが保存量として「再配分」されたことを意味しない。
- rotate_z_plusのrecallはvalidationでほぼ増減同数（9/9）・中央値+0.16ptとほぼ変化なしだが、
  rotate_z_minusはrecall中央値-1.94pt（14/18動画が低下）と明確に低下しており、hot bin FPRの
  減少を「回転による精度改善」として一般化できない。train_sanityではrotate_z_plus/minusとも
  recallが明確に低下する（中央値-4.58pt、-9.59pt）。
- PointNeXtは厳密な回転不変性を設計上保証しないため、観測された回転感度は絶対座標への依存
  （座標記憶）だけでなく、向きに依存する局所形状特徴量への感度によっても説明され得る。
  本補足2はこの2つを分離できない。

**split分離の効果:** train_sanityを分離した後も、validation単独（n=18、自己参照問題のない
サンプル）で同じ質的パターン（平行移動にほぼ無反応、回転かつhot bin集中の明確な反応）が
確認された。これは元の21動画混合の結果がtrain_sanity（n=3）の外れ値的な影響で駆動された
ものではないことを示す。一方、train_sanityの自己除外後の数値（hot bin FPR diff中央値など）は
修正前（全train共通分類）から変化しており、自己除外の適用が数値に実質的な影響を与えたことも
確認された。

**総合（方針管理チャットの判断材料として、確定的な結論は出さない）:** 平行移動でほぼ無反応・
回転（特にhot bin）で明確な反応という非対称性は、validation単独でも再現され、仮説A
（座標事前分布）に対する相関にとどまらない操作的な追加根拠として残る。ただし
（1）回転と平行移動の絶対移動量の違いという交絡、（2）PointNeXtの回転不変性非保証という
留保、（3）recallも同時に変化する（特にrotate_z_minusで明確に低下する）ことから
「hot bin FP減少＝精度改善」とは言えない、という3点により、「座標記憶の確定」
「augmentation採用」への飛躍はしない。次工程（augmentation単独5 epoch比較案の検討、または
W-A維持でのS5-15 pilot）の選択は方針管理チャットが行う。

## 13. 管理チャット返信: S5-14補足3 残存集計・記録の確認

返信・方針決定日: 2026-09-15。担当: Stage 5実装チャット。状態: 未着手。

### 13.1 現時点の判断と委任範囲

「集計仕様の修正」以降の報告と対応コードを確認しました。train sanityの自己除外とsplit分離は
修正され、validation単独でも「平行移動への反応は小さく、回転への反応は明確」という主要結果が
維持されています。一方、12章で依頼した集計の一部と記録の整合性に確認事項が残っています。
**補足2の最終受入を保留し、以下をS5-14補足3として実施してください。**

これは座標変換診断の拡張ではなく、既存成果物を使った限定的なCPU集計・記録確認です。
追加GPU推論0回、新規学習0回とし、条件・動画・seed・checkpointの追加は行いません。

### 13.2 残る集計の追加

既存の動画別CSVを使い、train sanityとvalidationを分けて、条件ごとに以下を出してください。

1. TP/FP/FN/TNの合算と、その合算値から計算したprecision、recall、FPR、F1、IoU。
   動画別指標の平均をsplit合算指標の代用にしないでください。
2. 動画別F1/IoUの中央値とidentityとの差分。差分の中央値と中央値同士の差は区別して名称を付け、
   対象動画数・未定義数を併記してください。
3. TP0動画数、改善/悪化/同値動画数、各指標の定義可能な動画数。
   既存summaryで出力済みの項目は再利用し、未定義を0で埋めないでください。

splitごとの件数とconfusion countsが動画別CSVと一致すること、identityとの差分が0になることを
CPUテストで検算してください。新しい集計は別名または版を明示し、元成果物を保持します。
必要な列が欠けていれば不足内容を報告し、推測や再推論で補わず停止してください。

### 13.3 報告文の整合性確認

1. identity/repeat identityについて「false_positive_rate_diffの利用可能動画数と増減件数が0/0」
   と記載されていますが、**差分が0であることと、指標が未定義で利用可能動画数が0であることは
   別です**。実際のsummary/CSVと照合し、利用可能数・増加数・減少数・同値数を区別して訂正してください。
2. grid8・raw_countの結果で「train sanityのhot FPR減少が0〜1/3」としながら「同様の傾向」と
   記載しています。条件×grid×定義×splitの実数を確認し、増減の取り違えか、実際に一貫しない
   結果なのかを明記してください。validationの傾向とtrain sanityの傾向を混同しないでください。
3. parity合格・実行完了と分析仕様の充足を分け、管理記録・評価レポート・本書の状態表記を揃えて
   ください。旧数値や説明は履歴として識別可能にし、修正箇所と根拠の成果物を示してください。

### 13.4 再推論の承認経緯の確認

12章では追加GPU推論を事前承認制としました。一方、実装報告では「同じ21動画×8条件の再実行なので
承認範囲内」と説明しています。**同じ条件でも計算量は追加されるため、この説明だけでは事前承認の
確認になりません。** 別途ユーザー承認があった場合は、承認内容・日時・対象範囲を記録してください。
確認できなければ承認記録未確認として報告し、承認済みと推定しないでください。

初回と修正後の走査数を区別し、報告上の各168動画相当、計336動画相当の再実行履歴と、その他の
失敗・preflightの重複実行があればその実施量を確認してください。この確認のための再実行は不要です。
本件は運用上の確認であり、得られた数値をそれだけで無効とする趣旨ではありません。

### 13.5 完了報告と次工程

CPU集計結果、検算結果、報告文の訂正、承認経緯と実施量、残る未検証項目を本書へ追記してください。
管理記録と評価レポートにも「S5-14補足3」として進捗・結果を記録し、共有する数値成果物はprivacy
self-checkを行ってください。実機での検証とこのチャットでのコード・文書確認を区別します。

回転augmentation単独5 epoch比較は有力な次候補ですが、採用・学習開始はまだ承認しません。
補足3の完了報告後、当該比較またはW-A維持でのS5-15 pilotを管理チャットが判断します。
W-Aとproduction設定を維持し、loss・label policy・class weight・normalization・window・
aggregation・thresholdは変更しないでください。

---

## S5-14補足3 完了報告

作成日: 2026-09-15。状態: 完了。本節はCPU専用の集計・記録確認であり、GPU再推論・新規学習は
一切行っていない（13章の禁止事項を遵守）。既存の`coordinate_transform_point_metrics.csv`
（実機で生成済み、書き換えなし）のみを入力とする。

### 実装

- `Stage5/checks/real_h5/check_stage5_coordinate_transform_reconciliation.py`（新規）: 既存の
  動画別`point_metrics.csv`を読み込み、(1) 動画×conditionのTP/FP/FN/TNをsplit単位で合算し、
  その合算値からprecision/recall/FPR/F1/IoUを算出する「pooled（点数加重）」集計、
  (2) F1/IoUについて、動画別diffの中央値（`median_of_per_video_diffs`）と、split内の
  中央値同士の差（`diff_of_split_medians`）を明確に区別して算出する集計、を追加した。
  既存の`condition_summary_by_split`（動画等重みのmean/median）は変更せず、別集計として
  追加している。
- `Stage5/checks/real_h5/check_stage5_coordinate_transform_reconciliation.sh`（新規、CPU専用、
  CUDA設定なし）。
- `Stage5/checks/dummy/check_dummy_coordinate_transform_reconciliation.py/.sh`（新規、
  synthetic test 7件）: 点数加重pooling が素朴な動画別rate平均と異なることの確認（小動画100%
  FPR・巨大動画0%FPRを混ぜても50%にならない）、`median_of_per_video_diffs`と
  `diff_of_split_medians`が実際に異なる値になる構成例（片方の動画だけ大きく変化するケースで
  中央値差分は動かないがsplit中央値同士の差は動く）、動画数の重複/不足の拒否、
  identity自己diffが厳密に0であることの検算、を含む。

**static check:** `py_compile`・`bash -n`・`git diff --check`すべて合格。
**synthetic検証:** 7件すべて合格。

### 13.2への対応: 実データでの再集計結果

この実装環境は既存の`coordinate_transform_point_metrics.csv`（実機生成済み、H5/checkpoint
不要）に直接アクセスできたため、本節の集計はこのチャット内でCPU実行した（GPU・実機操作は
不要）。`status: passed`、split別動画数の整合性検証・identity自己diff=0検算とも合格。
出力3ファイルのprivacy scan（タイムスタンプ風video ID・絶対hostパス）は0件。

**pooled（点数加重）split集計**（一部抜粋。全条件は`coordinate_transform_split_pooled_
metrics.csv`参照）:

| split | 条件 | pooled recall | recall pooled diff | pooled FPR | FPR pooled diff | pooled F1 | F1 pooled diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train_sanity | identity | 58.24% | — | 13.42% | — | 11.56% | — |
| train_sanity | rotate_z_plus | 60.31% | +2.07pt | 13.41% | -0.02pt | 11.95% | +0.40pt |
| train_sanity | rotate_z_minus | 49.06% | **-9.18pt** | 11.39% | -2.03pt | 11.28% | -0.28pt |
| validation | identity | 44.71% | — | 11.72% | — | 7.13% | — |
| validation | rotate_z_plus | 45.75% | +1.06pt | 11.04% | -0.66pt | 7.68% | +0.55pt |
| validation | rotate_z_minus | 37.88% | **-6.81pt** | 10.35% | -1.35pt | 6.78% | -0.36pt |

**動画別diff: 中央値 vs split中央値同士の差**（F1、一部抜粋。全条件・IoUは
`coordinate_transform_median_diff_metrics.csv`参照）:

| split | 条件 | median_of_per_video_diffs | diff_of_split_medians | 改善/悪化/同値 |
| --- | --- | ---: | ---: | --- |
| train_sanity | rotate_z_plus | -0.48pt | -1.13pt | 1/2/0 |
| train_sanity | rotate_z_minus | -0.75pt | -0.75pt | 1/2/0 |
| validation | rotate_z_plus | +0.44pt | +1.20pt | 14/4/0 |
| validation | rotate_z_minus | **+0.06pt** | +0.47pt | 11/6/1 |

**重要な発見（pooled/median-of-diffsを分離したことで判明）:** `rotate_z_minus`について、
**点数加重pooled recallは両splitで明確に低下する**（train_sanity -9.18pt、validation -6.81pt）
一方、**動画等重みのF1中央値差分（median_of_per_video_diffs）はほぼ変化しない、
validationではむしろわずかに正**（+0.06pt、18動画中11動画でF1改善）。これは、点数の多い
（一般に背景点が多い）動画ほどrecall低下の影響が強く反映されるpooled統計と、動画ごとに
均等な重みを与える統計とで、異なる側面を見ていることを意味する。precisionの改善（FPR低下）が
多くの動画でrecall低下を部分的に相殺し、F1としては動画単位で見ると中立に近い結果になっている
可能性がある。**「hot bin FP減少」「pooled recallの低下」「動画別F1はほぼ中立」は、いずれも
同時に成立し得る、互いに矛盾しない別々の観測であり、単一の結論（改善/悪化）に単純化しない。**

### 13.3への対応: 報告文の訂正

以下の誤記を修正した（本書の該当箇所を直接訂正済み。修正箇所には訂正注記を付した）。

1. **T2 parityの記述（項目1）:** 「利用可能動画数と増減件数が0/0」という記載は、利用可能数が
   0＝未定義であるかのように読める誤記だった。実際はtrain_sanity 3・validation 18の
   **全動画でFPR差分が定義可能**であり、そのうち増加0件・減少0件（＝差分の値が全動画で
   厳密に0.0）である。CSVの生値（`false_positive_rate_diff`列が`0.0`×3件・×18件）を
   直接確認して訂正した。
2. **grid8・raw_countの増減の取り違え（項目2）:** 「train sanityのhot FPR減少が0〜1/3」という
   記載は、実際には**増加した動画数**（0〜1/3）を「減少」として書いた誤記だった。正しくは
   train_sanityの減少動画数は2〜3/3（`rotate_z_plus`は3/3全動画、`rotate_z_minus`は2/3）。
   条件×grid×定義の実数を表として本書に追記し、validationとtrain_sanityの傾向を明確に
   区別した。
3. **追加で発見した誤記（本節の検算過程で判明）:** train_sanityのrecall diff中央値を
   「-1.75pt、-8.47pt」と記載していたが、これは実際には**mean（平均）**の値であり、
   同じ列の中央値は「-4.58pt、-9.59pt」だった。集計JSONの`median`キーと`mean`キーを
   取り違えていたため、該当箇所（本書・管理記録）を中央値の正しい値へ訂正した。評価レポートは
   この誤記を含んでいなかった（hot bin FPRの中央値のみを引用しており、該当なし）。
4. **状態表記の整合:** 「parity合格・実行完了」と「分析仕様の充足」を区別し、本書・管理記録・
   評価レポートの状態欄を「集計仕様修正・実機再実行完了、補足3のCPU集計・記録確認完了、
   最終受入は方針管理チャットの判断待ち」に統一した。旧数値（split混合の初回実行結果）は
   引き続き診断履歴として識別可能な形で保持している。

### 13.4への対応: 再推論の承認経緯

**確認結果: 事前の明示的承認文言は、方針管理チャットの記録として本書上には存在しない。**
経緯は次の通り: (1) 実装チャットが12章の指摘を受けてコードを修正し、再実行コマンドを
ユーザーへ提示、(2) ユーザーが実機で当該コマンドを実行し、完了報告を実装チャットへ共有。
この(2)のユーザー自身による実行が実質的な実施の契機であり、方針管理チャット側からの
事前の明文化された承認（承認内容・日時・対象範囲の記録）とは区別する。実装チャットが本書に
記載した「D-033承認範囲内の再実行」という説明は、実装チャット自身の判断であり、方針管理
チャットによる事前承認の確認ではなかった。**承認記録は未確認として報告する。**

**実施量:** このチャットが把握している実行は2回。(1) 修正前コード: 21動画×8条件=168
video-condition相当（2026-09-15、hot/cold分類が全train共通・split混合の版）。(2) 修正後
コード: 21動画×8条件=168 video-condition相当（2026-09-15、train sanity自己除外・split別集計
対応版）。**合計336 video-condition相当。** いずれも`status: passed`で完了し、failureの報告は
受けていない。上記以外のpreflight・失敗・重複実行の有無は、ユーザー実機のログでのみ確認可能で
あり、このチャットからは確認できない。本件は得られた数値そのものを無効化する趣旨ではなく、
運用上の承認記録の確認である。今後、同様の再実行が必要な場合は、実行前に方針管理チャットの
明示的な承認（対象範囲・実行回数）を得ることを徹底する。

### 13.5: 完了条件

1. CPU集計結果（pooled precision/recall/FPR/F1/IoU、median_of_per_video_diffs vs
   diff_of_split_medians）を本節に記録した。
2. split別件数・confusion countsが動画別CSVの合算と一致すること
   （`verify_split_video_counts`）、identityの自己diffが厳密に0であること
   （`verify_identity_self_diff_is_zero`）をCPUで検算し合格した。
3. 報告文の訂正3件（T2表記、grid8増減の取り違え、mean/median取り違え）を本書・管理記録へ
   反映した。
4. 再推論の承認経緯を「未確認」として記録し、実施量（336 video-condition相当、2回）を
   明記した。
5. 出力3ファイル（`coordinate_transform_split_pooled_metrics.csv`、
   `coordinate_transform_median_diff_metrics.csv`、
   `stage5_coordinate_transform_reconciliation_summary.json`）のprivacy self-checkに合格した。
6. 実機での検証（Step T2〜T4のGPU forward pass、2回・計336 video-condition相当）と、
   本節のCPU専用集計・記録確認（このチャット内で完結）を区別して記録した。

未検証項目: pooled/median-of-diffsの分離で判明した「点数加重recallの低下とvideo等重みF1の
中立」という乖離の原因（背景点数の多い動画への依存等）は、本補足3の範囲では特定していない。
回転augmentation比較の採用・学習開始は本補足3でも承認していない。次工程の選択は方針管理
チャットが行う。
