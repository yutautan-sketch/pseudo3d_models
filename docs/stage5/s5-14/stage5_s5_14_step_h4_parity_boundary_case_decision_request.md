# Stage 5 S5-14 Step H4: Parity Gate境界ケースの扱いに関する判断依頼

作成日: 2026-09-15
作成元: Stage 5実装チャット

最終更新日: 2026-09-15

管理チャット返信: 第8節にStage 5方針管理チャットによる確認結果・正式判断・実装依頼を追記済み。
第1〜7節は判断依頼時の記録として保持する。第3節の原因診断と第5節の推奨は未承認であり、
今後の対応は第8節に従うこと。

## 1. 経緯

S5-14 Step H4（per-window context診断）では、実装依頼書の要求通り、W-A（S5-12/S5-13で使用した
GroupNorm epoch 5 checkpoint）の保存済みaggregate predictionに対し、per-window forwardを
再実行して次の3点を**no-toleranceで**一致させるparity gateを実装した。

1. vote_count（source pointごとのwindow投票数）の完全一致
2. training window occurrence count（Step H2の構造的exposure count）とこのrunのvote_countの
   完全一致
3. mean probability（threshold 0.5）から再導出したpredicted classと、保存済みpredictionの
   TP/FP/TN/FN・per-point predicted classの完全一致

依頼書は「CUDA非決定性等で不一致が出た場合はtoleranceで通さず、差分artifactを保存して停止し、
原因を報告してください」と明記しており、これに従って実装した。

## 2. 発生した事象

GPU側実行（W-Aのvalidation動画、動画alias `validation_000`）で、3の条件が1点だけ不一致となり、
fail-fastが正しく発動した。

```text
1 point(s) have a different threshold-0.5 predicted class between this re-inference's
mean_probability and the saved W-A prediction .npz (max abs probability diff=2.980e-08)
```

不一致点の診断artifact（自動保存、実装済みのdiff artifact機能による）:

```text
point_index,gt_label,valid,this_run_mean_probability,saved_prob_femur,abs_probability_diff,distance_from_0.5_this_run,distance_from_0.5_saved,this_run_vote_count,saved_vote_count,frame_order
290130,1,True,0.5,0.5,0.0,0.0,0.0,2,2,34
```

## 3. 診断

以下の3点から、実装上のバグではなく、GPU/CUDA kernelの非決定性に起因する境界ケースであると
判断した。

- **vote_count・training window occurrence countは完全一致**している。window構成やcoverageの
  問題ではない。
- 監査対象の全point中、**この1点のみ**が不一致。系統的な誤りであれば、より多くの点、または
  特定のパターン（例えば特定window・特定frameに集中する等）で不一致が出るはずである。
- 不一致点の`this_run_mean_probability`と`saved_prob_femur`は、CSV表示上どちらも`0.5`（実際には
  約2.98e-08だけ異なる、float64精度でほぼ隣接するビット）であり、**この点のモデル予測確率が
  ちょうど閾値0.5の真上に位置している**。この点はvote_count=2、すなわち2つのwindow予測の平均で
  あり、2つの確率を合算する際の浮動小数点加算順序がGPU実行ごとにわずかに変わることで、最後の
  数ビットが変化し、0.5のどちら側に落ちるかが入れ替わったと考えられる。

この点自体は、モデルにとって「femurかbackgroundか、確率的にほぼ五分五分」の症例であり、
`0.5`という単一の决定的な閾値で二値化する限り、どちらの分類になってもおかしくない。

## 4. 依頼書の制約との関係

依頼書は「不一致が出た場合はtoleranceで通さず、差分artifactを保存して停止し、原因を報告する」
ことを求めている。今回、diff artifact保存・停止・原因報告（本文書）はすべて実施済みである。

問題は、この先どう進めるかである。以下の選択肢を提示する。

### 選択肢1: 限定的な境界ケース除外を実装する（実装チャットの推奨案）

predicted classの不一致が検出された場合、**その点の確率が0.5から極めて近い（例: 距離1e-6未満）
場合に限り**、「境界ケースとして記録し、その点をStep H4の厳密比較対象から明示的に除外する」
処理を追加する。0.5から明確に離れた値での不一致（真のバグを示唆する可能性が高い）は、これまで
通り即座にfail-fastする。除外された点は常にログ・artifactへ記録し、黙って握りつぶさない。

**利点:** 「一般的なtolerance」ではなく、数学的に退化した特殊ケース（確率がほぼ厳密に0.5）
だけを対象とするため、依頼書が禁止する「不一致をtoleranceで通す」こととは質的に異なると
考えられる。Step H4の残り20動画・数百万点規模の診断を、1点の境界ケースのために完全に止めずに
進められる。

**懸念:** 依頼書の文言を厳密に読めば「toleranceで通さない」という指示への部分的な例外にあたる。
閾値1e-6の妥当性、および将来この仕組みが誤用され得るリスクをどう見るか。

### 選択肢2: 決定的な推論設定を試す

`torch.use_deterministic_algorithms(True)`やcuDNN determinism設定などを追加し、GPU forward自体を
完全決定的にすることで、今回のような境界ケースの再現性を確保できないか試す。

**懸念:** 保存済みW-A predictionを生成した元のevaluate_stage5.py実行自体が決定的設定で行われた
保証がなく、今から決定的設定にしても元の値と一致する保証はない。決定的アルゴリズムへの変更は
速度低下を伴う場合があり、S5-14の診断目的に対して過剰な対応になる可能性もある。

### 選択肢3: この動画（validation_000）をStep H4対象から除外する

該当1点だけでなく、動画単位でStep H4の対象から除外し、残り20動画で診断を進める。

**懸念:** 1点のための動画丸ごと除外は過剰であり、その動画の他の診断価値（no-GT frame FP等）を
失う。

## 5. 実装チャットの推奨

選択肢1を推奨する。数学的に退化したケース（確率がほぼ厳密に0.5）に限定した、透明性のある
例外処理であり、Step H4の本来の目的（仮説A/B/Cの検証）を大きく妨げずに進められるため。

## 6. 判断が必要な事項

1. 選択肢1〜3のいずれを採用するか（または別案）。
2. 選択肢1を採用する場合、境界許容距離（提案値: 1e-6）は適切か。
3. 今回の1点（video_alias `validation_000`、frame_order 34、point_index 290130）を除外して
   Step H4を続行してよいか。

## 7. 関連文書

- `docs/stage5/s5-14/stage5_s5_14_structural_diagnostics_implementation_handoff.md`（実装依頼書、正本、
  Step H4のparity要求は7章）
- `docs/stage5/s5-14/stage5_s5_14_report_to_policy_chat.md`（S5-14進捗報告、Step H1〜H4の実装状況）
- `Stage5/checks/real_h5/check_stage5_per_window_context_diagnostics.py`（`verify_h4_parity()`が
  該当箇所）

## 8. 方針管理チャットからの返信（2026-09-15）

### 8.1 結論

報告書と関連コードを確認した。parity gateが不一致を検出して停止したことは適切である。
ただし、現時点で「CUDA非決定性が原因」とは判断できない。保存時とH4 checkerの判定規則・
集約後のdtypeが異なっており、まずこの比較条件を揃える必要がある。

本返信は報告書とコードに基づく判断であり、管理チャットで実機の差分CSVそのものを再確認したり、
GPU再実行を行ったりした結果ではない。

### 8.2 確認できた比較処理の違い

| 項目 | 保存時のevaluate_stage5.py | H4 checker |
| --- | --- | --- |
| 集約対象 | background/positiveの両クラス | positive側のみ |
| 集約精度 | float64で加算・平均 | float64で加算・平均 |
| 判定直前 | 平均確率をfloat32へ変換 | float64平均を使用 |
| predicted class | 2クラスのargmax | positive確率 >= 0.5 |
| 両クラスが0.5の場合 | background（index 0） | positive（index 1） |

参照箇所:

- `Stage5/evaluate_stage5.py`: `predict_h5()`の`aggregated_probability`生成と`np.argmax()`。
- `Stage5/checks/real_h5/check_stage5_per_window_context_diagnostics.py`:
  `verify_h4_parity()`の`pred_label_from_mean = (mean_probability >= 0.5)`。
- 同H4ファイルの`classify_prediction()`も`>= 0.5`を使用しており、parity gateだけでなく
  TP/FP/TN/FN層別の判定規約も確認が必要。
- `Stage5/checks/real_h5/check_stage5_overlap_aggregation.py`:
  `test_probability_half_is_background()`は既に「0.5はbackground」という規約を検証している。

同じ確率が得られても、この同値時の扱いだけで今回の1点不一致は発生し得る。
不一致点が少数であることは比較ロジックの不具合を否定する根拠にはならない。

また、`>=`を単に`>`へ変更するだけでは厳密な同等性を保証できない。浮動小数点では両クラスの
確率の和が厳密に1になる保証がなく、平均後のfloat32丸めも判定に影響する。
保存時と同じ両クラス集約・float32変換・argmaxを再現すること。

### 8.3 確率差に関する報告の訂正

`verify_h4_parity()`の`max_abs_prob_diff`は、動画内の全点について求めた最大絶対差である。
エラーメッセージの`2.980e-08`を不一致点自身の差と解釈してはいけない。

第2節の不一致点artifactには次の値が記録されている。

```text
this_run_mean_probability = 0.5
saved_prob_femur          = 0.5
abs_probability_diff     = 0.0
```

CSV writerは`float()`で値を出力し、桁数を制限する明示的な書式指定はしていない。
したがって「CSV表示上は同じだが、この点自身は約2.98e-08異なる」という第3節の説明は、
提示されたartifactとコードからは導けない。実機artifactの値・dtypeと保存済み`pred_label`を
改めて確認すること。

さらに、0.5近傍の`2.98e-08`はfloat32の刻みに相当する大きさであり、float64でほぼ隣接する
ビット程度の差ではない。今回の全点最大差はfloat64平均と保存時のfloat32丸めの比較でも
説明可能である。window確率の集約もコード上はCPU/NumPyで順次実行されるため、
「2つのwindow確率の加算順序がGPU実行ごとに変わった」と断定する根拠はない。
GPU forward由来の差が残るかどうかは、比較条件を揃えた後に切り分ける。

### 8.4 第6節の3事項への正式回答

1. **別案を採用する。** 保存時とH4の集約・dtype・predicted class判定を統一し、厳密parityを
   再検証する。境界ケース除外、決定的推論設定への変更、動画除外は現段階では採用しない。
2. **1e-6の許容距離は採用しない。** 比較条件の不一致が未解決であり、許容幅の導入を正当化する
   根拠がない。境界点だけの除外もparity要件を変更する例外にあたる。
3. **該当1点を除外して続行することは承認しない。** `validation_000`のpoint_index 290130、
   frame_order 34を含めて再検証する。厳密parityが合格した場合にH4の残り対象へ進む。

### 8.5 実装チャットへの次の依頼

1. 該当点の保存済み`pred_label`、今回の判定値、両側の確率とdtypeを確認し、全点最大差と
   当該点の差を区別して報告する。
2. 同じper-window forwardからbackground/positiveの両クラス確率を保持し、保存時と同じ順序で
   float64加算・平均、float32変換、argmaxを行う。background確率を`1 - positive確率`で
   復元せず、softmaxの実出力を使用する。
3. parity比較とH4のTP/FP/TN/FN層別に同じaggregate predicted classを使用する。
   診断用float64平均・分散等を保持する場合は、保存処理を再現した判定用確率と区別する。
   per-window positive vote等の判定規約も確認し、aggregate classとの意味の違いを明記する。
4. CPUで検証可能なsynthetic testを追加する。少なくとも両クラスが0.5の同値ケース、
   float64平均をfloat32へ変換すると同値になるケース、0.5近傍でpositive確率の単独比較と
   両クラスargmaxが異なるケースを含める。期待値は保存時の処理に基づいて確認する。
5. 該当動画でsource point alignment、vote_count、training occurrence、全点predicted class、
   TP/FP/TN/FNを除外・toleranceなしで再検証する。確率差は同じdtype同士の差と丸めに由来する差を
   分けて記録する。
6. 合格したら固定21動画のH4を継続する。差が残る場合は、両クラス平均の変換前後、
   保存済み確率・label、point別差分をartifactへ保存して停止し、改めて原因と選択肢を報告する。

保存済みW-A prediction・checkpointやproductionの判定規約をcheckerへ合わせて変更しないこと。
今回の修正はcheckerの比較条件を既存評価処理へ揃えるものとして扱う。

修正・検証後は本書の実施記録、`docs/stage5/s5-14/stage5_s5_14_report_to_policy_chat.md`、
`docs/stage5/stage5_revision_management_record.md`へ原因・変更内容・結果を追記すること。
第3節の「CUDA非決定性」という当初仮説は、確定事実として後続文書へ転載しないこと。

## 9. 検証結果（実装チャット、2026-09-15）

### 9.1 原因の確定

第3節の「CUDA非決定性」という当初仮説は誤りであり、**撤回する**。方針管理チャットの指摘通り、
実際の原因はH4 checkerの集約・判定規約が正本`evaluate_stage5.py: predict_h5()`と異なっていた
ことだった。不一致点（point_index 290130）の確率は両run間で完全に一致（`abs_probability_diff=
0.0`）しており、GPU forward自体の非決定性は関与していなかった。

コード確認で判明した3つの差異:

| 項目 | 保存時 `predict_h5()` | H4（修正前） |
| --- | --- | --- |
| 集約対象 | 両クラス | positiveクラスのみ |
| window単位のcast | softmax直後にfloat32化 | float64直接cast |
| predicted class | 2クラスargmax（同値はbackground） | `>=0.5`（同値はpositive） |

再利用元`check_stage5_overlap_aggregation.py`の既存テスト`test_probability_half_is_background()`
（`p1 > 0.5`厳密不等号）とも矛盾していた規約であり、事前に確認すべきだった。

### 9.2 実装内容（8.5節への対応）

`Stage5/checks/real_h5/check_stage5_per_window_context_diagnostics.py`を修正した。

1. `verify_h4_parity()`のエラーメッセージを訂正し、「video全体の最大差」（informational）と
   「該当点自身の差」（diff artifactの`abs_probability_diff`列）を明確に区別した。
2. `run_overlap_forward()`（S5-08、positiveクラスのみ・float64直接cast）をpredicted class判定に
   使うのをやめ、新設した`run_h4_forward()`/`canonical_predicted_class()`が`predict_h5()`と
   同じ手順（window単位float32化→両クラスfloat64加算→最終float32変換→2クラスargmax）を再現する
   ようにした。backgroundは`1-positive`で復元せず、softmaxの実出力（2列）をそのまま保持する。
   同じforward1回分の出力を、既存`OverlapAccumulator`（診断専用）と新しい2クラス集約の両方へ
   供給し、forwardの二重実行を避けた。
3. `verify_h4_parity()`と`classify_prediction()`（TP/FP/TN/FN層別）はいずれもこの正本互換の
   `pred_label`を受け取る形に統一した。per-window単位の`positive_vote_ratio`等（診断専用、
   S5-08の`p1 > 0.5`規約のまま）とは意味が異なることをdocstringで明記した。
4. CPU synthetic testを3件追加した: (a) 両クラスがfloat64で厳密に0.5/0.5のタイ→background、
   (b) float64では非タイ（positiveがわずかに高く、素朴な比較ならpositiveを選ぶ）だが
   float32変換後にタイになるケース→変換後の規約通りbackground、(c) (b)と同じデータで
   「素朴な`positive確率>=0.5`」と「正しい2クラスargmax」が実際に食い違うことを直接確認
   （今回のインシデントの再現）。既存分と合わせ計19件のsynthetic testをこの開発コンテナ内
   （torch未インストール）ですべて合格を確認した。`py_compile`・`bash -n`・`git diff --check`も
   合格。

保存済みW-A prediction・checkpoint・production側の判定規約は変更していない。

### 9.3 GPU側再検証結果

修正版をvalidation_000を含む固定21動画全体で再実行した。**全21動画でparity gateが完全一致した
（除外・tolerance追加は一切行っていない）。**

```text
videos: 21, stratified rows: 1226, bin rows: 2192
max abs mean-probability diff vs saved W-A prediction (informational only): 0.000e+00
```

video別の`max_abs_prob_diff`も全21動画で`0.0`であり、vote_count・training window occurrence・
predicted classとも1点の不一致もなかった。

### 9.4 layered分析と3仮説の最終判定

固定21動画のTP/FP/FN/TN×relative_frame_decile/vote_count_bucket層別集計、およびStep H3.1の
XY bin recurrenceとのcross-checkを行った。

- window 16/stride 8では、vote_countは構造上1または2のみ（重複率50%）。
- **vote_count 1/2間でFP率（background点中）はほぼ同一**（11.88%→11.64%）であり、
  「overlap exposureが多いほどFPが増える」という単純な関係は支持されなかった。recallはむしろ
  vote_count=2の方がやや高い（45.3% vs 39.2%）。
- ただし**vote_count=2の内部だけで見ると、FPのdisagreement rate（34.1%）はTNの4.7倍
  （7.2%）**に達し、window間予測の食い違いがFPへ偏るという限定的なメカニズムが見られた。
- relative_frame_decile別のdisagreement/exposureは動画**中盤でピーク・両端で低い対称
  パターン**（window境界の幾何学的効果）を示し、Step H3で確認した**recallの単調な低下
  （decile0の64.2%→decile8の7.9%）とは形状が異なる**。したがってStep H3のrecall低下は
  exposureパターンの副産物ではなく、独立した時間効果と判断した。
- Step H3.1のXY bin recurrenceとのcross-check: multiple unmatched runsのあるbin（791件）は
  ないbin（1,401件）よりtraining window occurrence約+10%・disagreement rate約+8%高いが、
  効果は小さい。

**3仮説の最終判定:**

| 仮説 | 判定 |
| --- | --- |
| A. 座標事前分布 | 強く示唆される（bin単位point密度での正規化確認が残課題） |
| B. 時間位置・frame phase | 明確に支持される |
| C. overlap exposure | 限定的に支持される（主要因ではない） |

単一の仮説に単純化できない複合的な結論となった。詳細な数値・分析過程は
`docs/stage5/s5-14/stage5_s5_14_report_to_policy_chat.md`のStep H4.1節・「3仮説の最終判定」節に記録している。
本書はこれ以上更新せず、以後の経過は同報告書と`docs/stage5/stage5_revision_management_record.md`
へ追記する。
