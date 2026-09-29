# S5-17 全体報告書: 幾何表現のtraining-free診断と共通評価器

作成日: 2026-09-29  
状態: **S5-17クローズ時点のまとめ（2026-09-29管理受入）。** 新しい判断・解釈・実データ処理は含まない  
基準コミット: `9902b3d`（評価器・合成テスト）、`04de18e`（S17-4受入・S17-5承認の文書）、`b77484d`（補完集計）。本書はその後の文書同期と同時にコミットする想定  
想定読者: S5-18〜S5-20の実装チャットと総括管理チャット（S5-17の経緯を読んでいない前提）

## 0. 本書の位置づけ

S5-17の成果を、後続段階（特にS5-20）が使える形に一か所へまとめた報告書である。

| 文書 | 役割 | 優先関係 |
| --- | --- | --- |
| 本書 | S5-17の成果・結果・限界・引継ぎのまとめ | 下の2文書と矛盾する場合は、そちらを優先する |
| [S5-17管理書](stage5_s5_17_implementation_management.md) | 確定した仕様・解釈（4.5節）・実行条件・状態の正本 | 仕様と解釈の正本 |
| [S5-17報告書](stage5_s5_17_report_to_policy_chat.md) | 提案・管理判断・実装・テスト・実機結果の履歴と証拠 | 経緯と証拠の正本 |
| [全体管理記録](../stage5_revision_management_record.md) 5章「S5-17」 | Stage 5全体における位置づけ | 全体方針の正本 |
| [評価レポート](../stage5_pointnext_s_training_evaluation_report.md) 9.13節 | 検証結果の要約 | 数値的根拠の正本 |

数値は、実機で生成された共有JSON（10章）と報告書13〜15章で照合済みの値だけを使う。
共有JSONの配置は後で変更する予定のため、10章の場所は2026-09-29時点のものである。

## 1. 目的と、途中で変わった前提

**当初の目的（S5-16 v3 10.1）:** segmentationと後処理でFLに必要な幾何情報（存在・軸・端点・長さ）を扱えるかを、学習・GPU・再推論なしで診断する。どの幾何表現を後続へ渡すかを決め、S5-18〜S5-20の共通評価器を作る。

**前提の訂正（依頼書4章、報告書1〜3章）:**

- アノテーション済み180動画に**臨床実測FLはない。** 手元のFL値（`femur_traj_len`）は、BBoxアノテーションから導いた「大腿骨断面の重心の軌跡長」（正規化座標）である。
- そのため旧H17-1「臨床FLと最もよく一致する代理FLを選ぶ」は実行不能になった。次のように再設計した。
  - H17-1: GT由来表現の**情報保持・整合性・摂動安定性**。
  - H17-2〜H17-5: 予測由来幾何とGT由来幾何の一致、FPの空間分布、prior-only、複数instance（いずれも記述）。
- **撮像条件（ユーザー説明、報告書14.11）:** 断面は動画によって異なり、基本は横断像（点・楕円状）か斜めの断面像（線状）である。sweepは一定方向に進み、骨の一端から他端までを覆う。**sweepの方向と骨の長軸の方向には直接の関係がない。** pseudo-3Dで骨が曲がって見えるのは、Stage 2（dualtrack）の精度の影響を含む。

## 2. 実施範囲と固定条件

| 項目 | 内容 |
| --- | --- |
| 分割 | Step 0で確定したtrain_core144（identity `fa8429e3…`）・validation18（`de3f3d51…`）・internal_test18（S5-20cまで封印、S5-17では一切読まない） |
| **後続でのデータ利用の制限** | **S5-15の旧checkpoint（S5-17で使ったR0長期runのbest・lastを含む）は、internal_testの18件を学習済みである。S5-20cでも、これらのcheckpointをinternal_testの18件で評価することは禁止**（seal registryの`prohibited_checkpoint_rule`、D-037）。internal_testでの評価は、train_coreだけで学習した構成に限る。**実測FL付きの5例（D-040）**は、固定した方法による探索的比較とmm換算の健全性確認の補助に限り、**学習・較正・方式選択・採否には使わない**（S5-17の入力にも含めていない） |
| teacher | v7 `bboxrank_v7_cvat_authoritative_crop_quality_v1` |
| 予測 | S5-15 R0長期runの保存済み予測。best（epoch6）を主、last（epoch50）を補助。validation18とsanity3（in-sample、別表）。**新規学習・GPU・再推論なし** |
| 用途別の入力 | GT幾何・prior・後処理定数: train_core144。予測診断: validation18。in-sample: sanity3 |
| 座標 | pixelモード（元frame pixel、x/y等方を仮定）。mm評価は行わない（mm/pixelの契約が未確定） |
| 実施量 | coverage登録・メタ監査（S17-4）各1回、本診断（S17-5）1回、共有集計の補完1回。いずれも停止なし、再実行なし |

## 3. 成果物（コードと検証）

### 3.1 コード

| パス | 内容 | 依存 |
| --- | --- | --- |
| `Stage5/stage5/geometry/types.py` | 座標空間`CoordSpace`（LOCAL_CROP_PX／RAW_FRAME_PX／RAW_FRAME_NORM／MM_XY／PSEUDO3D）、`Points2D`、`InstanceGeometry`（重心`centroid`と矩形中心`center`を区別）、`FrameGeometry`、`VideoFLEstimate`。停止（`GeometryContractError`）・入力評価不能（`InputUnevaluable`）・手法側の失敗（`FailureReason`）の3区分 | numpyのみ |
| `.../transform.py` | `CropTransform`（`local_preprocess_effective`で分岐するcrop逆変換）、`PixelToMM`（x/y別、既定値なし、供給元hash必須） | numpy |
| `.../frame_geometry.py` | 連結成分（local 4 px以下で連結、5点未満を除外）、PCAによるinstance幾何（2〜98分位の矩形、中心線上の端点） | numpy |
| `.../fl_estimate.py` | 表現(a)〜(e)と動画集約。(d)のframe選択はGT・実用・oracleを別関数で分離 | numpy |
| `.../postprocess.py` | 後処理P0〜P3、P1+P2と、train_core由来の定数 | numpy |
| `.../matching.py` | Hungarian法とゲート付きM3 matching（重心間距離、ゲート＝GT長×0.5） | numpy |
| `.../prior.py` | prior-only（元frame正規化座標で構築・逆写像、identityによる自己除外） | numpy |
| `.../metrics.py` | 未定義と0の区別、frame存在判定・instance検出・FP空間分類（分母を明示）、保持被覆率、摂動15通り、F-A規則（最近順位法）、主候補選定、分母の内訳 | numpy |
| `Stage5/stage5/utils/geometry_inputs.py` | 生配列を検査してから変換する読込関数（teacher H5・中間H5・NPZ、window数の再計算、混同行列） | numpy・h5py |
| `Stage5/evaluate_stage5_geometry.py`・`.sh` | 評価CLI: `register-coverage`・`audit`・`run`。封印guard、処理目的別の許可リスト、hashに紐づくcoverage、privacy検査付きの共有出力 | 上記 |
| `Stage5/supplement_stage5_geometry_aggregates.py`・`.sh` | private記録から共有集計を補うCLI（`register`・`aggregate`）。実行者確認の完全hashを必須とし、出力確定前に入力hashを再照合 | 上記 |

### 3.2 合成テスト（すべて実機で合格）

| テスト | 項目数 | 主な内容 |
| --- | ---: | --- |
| `checks/dummy/check_dummy_geometry_core.*` | 142 | 手計算の既知座標でのcrop逆変換、Stage 4の順変換式との整合、異方的なmm換算、既知形状、退化、(b)(c)の領域一致、F-Aの境界、prior正規化 |
| `checks/dummy/check_dummy_geometry_cli_guard.*` | 144 | 合成の契約で封印の読込前拒否、許可リスト、coverage登録、点対応の整合性検査、`register-coverage`→`audit`→`run`の完走、出力境界 |
| `checks/dummy/check_dummy_geometry_supplement.*` | 80 | 実行者確認の完全hash、解析前の保証と解析後の検出の区別、集計値、出力確定前の再照合 |
| `checks/dummy/observe_geometry_holdout_coverage.*` | 観測 | 既知形状での保持被覆率（合否判定なし。報告書11.1） |

## 4. 入力の確認と来歴

### 4.1 S17-4（メタ監査、報告書13章）

- 162件（train_core144・validation18）すべてで、必要なdatasetが揃い、入力側で評価可能だった。入力評価不能は0件。
- 中間H5は、162件すべて記録attr（`source_pseudo3d_h5`）で解決した。前方一致のglobは使っていない。
- crop modeは162件すべて`resize_shorter_then_offset_crop`で、有限・正のscaleで逆変換を構築できた。`offset_crop_fallback_resize`は0件（Stage 4のBBox変換の論点は今回のデータでは生じない）。
- 評価runのbest・lastとも、`summary.json`・`h5_metrics.csv`の動画集合、checkpoint名（`best.pt`／`last.pt`）・epoch（6／50）・window（16／8／tail）が一致した。NPZは21件×2のすべてで必要なキーが揃っていた。
- sanityリストは、pin済みmanifestに記録されたStep 0のhashと一致した。

### 4.2 S17-5（本診断の冒頭検査、報告書14.1）

162件と21件×2のすべてで、次の検査を通過した。

- 値域、`valid_mask`と`point_label≠-1`の整合、`frame_order`・`pixel_xy`の範囲、逆変換後の範囲。
- NPZの配列長、`point_indices=arange(N)`、確率とlabelの整合（±1e-6の境界帯を除く）。
- window数の再計算、混同行列と`h5_metrics.csv`の一致。

集計レベルの確認として、P0（後処理なし）の点単位FP総数（best 525,800、last 128,800）がS5-15の公式集計（評価レポート9.10.2）と一致した。

### 4.3 来歴の限界（結果とともに保持する）

| 項目 | 状態 |
| --- | --- |
| 評価時点のgit revision | 記録なし（`unknown`） |
| best・lastのcheckpoint SHA-256 | 記録なし（`unknown`） |
| teacher H5の動画別SHA-256 | 過去記録（teacher v7生成時のCSV）の存在は確認したが、封印対象のGT統計を含むため**未照合**（`unknown`。記録なしとは区別） |
| 補完集計のprivate記録 | 実行者による生成・保管経緯の確認と完全SHA-256の固定を、今回限りの運用上の来歴として受入（生成時hashではない） |

coverageや整合性検査の通過を、評価当時のコード・checkpointの同一性の証明とは扱わない。

## 5. 結果

値はGT由来量どうしの比較であり、長さはraw px（x/y等方を仮定）で、mmではない。**臨床FL精度ではない。**

### 5.0 本書で使う方式の定義

| 記号 | 定義 |
| --- | --- |
| instance | frame内の点（GT陽性点、または予測・後処理後の陽性点）を、local crop座標で距離4 px以下の単連結で成分に分け、5点未満の成分を除いたもの |
| M1 | 保持されたinstanceのうち点数最大のもの（同点は最小の点index） |
| M2 | 保持された全instanceを1つとして統合したもの |
| M3 | GTのinstance集合と予測のinstance集合のmatching（Hungarian法、重心間距離、ゲート＝GTのinstance長×0.5。軸が使えないときはAABB長辺×0.5）。ゲート外の組は未対応として残す |
| P0 | 後処理なし（保存済み`pred_label`） |
| P1 | frameごとに最大の連結成分だけを残す（連結・最小点数はinstanceと同じ） |
| P2 | frame間の持続性: 前後2 frame以内に、重心が15.46 raw px（train_coreのGT長中央値×0.5）以内の成分がある成分だけを残す |
| P1+P2 | P1の後にP2 |
| P3 | 保存済み`pred_label`に陽性点が1点以上あるframeだけを対象に、そのframeの全点から`prob_femur`上位K＝230点（train_coreの陽性frameあたり有効GT陽性点数の中央値）を陽性とする（0.5未満の点も含む）。**予測陽性のないframeは対象外で、そのFNは救済しない。** P3の対象frameと、後処理後に5点以上の成分が成立する幾何的な存在frameは別物 |
| (d)のframe選択 | 各frameのM1が次の条件を満たす場合だけ選択可能: crop境界（local 2 px以内）に接していない、(c)の長さが定義できる（軸が曖昧・共線でない）。選択可能なframeのうち、GT表現ではM1の点数（保持評価では構築側Aの点数だけ）、実用では予測のM1の点数×平均`prob_femur`が最大のframeを選び、同点は最小の`frame_order`。oracleはGTで選んだframeを予測に適用する（別表） |

### 5.1 train_coreのGTから得た基礎量

| 量 | 値 | 備考 |
| --- | --- | --- |
| GTのper-frame長さ（M1の(b)）の中央値 | 約30.9 raw px | P2の距離15.46 raw px（＝0.5×中央値）から。長さが定義できたframe 2,143、未定義3 |
| 有効GT陽性点数／frameの中央値 | 230点 | 有効GT陽性点が1点以上あるframe 2,146（P3のK） |
| 複数instanceのGT frame | 35動画・269 frame（陽性frameの約12.5%、分母の定義が異なる参考値） | そのうち98.9%（266 frame）で、M1とM2の(b)長さが10%超異なる |
| 点GTとBBoxの整合性（動画別中央値の分布） | IoU中央値0.731（四分位0.609〜0.820、P90 0.891）、長辺の相対差（点GT − BBox）中央値−0.059 | 全BBoxを包む領域と全GT instanceを包む領域の比較。独立な検証ではない |
| (e) pseudo-3D参考値 | 144件すべて定義、中央値82.7（四分位55.3〜109.1、P90 128.1） | pseudo3d_units。候補外、mmでも骨長でもない |

### 5.2 H17-1: GT由来表現の摂動安定性（train_core144）

摂動は15条件（各3 seed）: **間引き（保持率50%・75%）**＝各frameのGT陽性n点から`max(ceil(保持率×n), min(n,5))`点を非復元で残す（小点数frameでは最低5点の制約がかかる）、背景点の追加10%・30%（同frameの有効背景点から非復元で付け替え）、揺らぎσ1 local px。
F-A規則（失敗は最悪値扱い、最近順位法、区分ごとに中央値≤5%かつP90≤15%）で判定した。

| 表現 | 定義 | 規則 | 選定用P90の最大値 | 手法側の失敗 | 保持被覆率（記述、中央値。定義／未定義の動画数） | 相対膨張率（記述、中央値） |
| --- | --- | --- | ---: | --- | ---: | ---: |
| (a) | frame AABB長辺、動画q90 | 合格 | 0.095（揺らぎ） | 0/144 | 0.997（140／4） | 1.42 |
| (b) | PCA 2〜98分位矩形の主軸長、動画q90 | 合格 | 0.028（保持率50%） | 0/144 | 0.893（140／4） | 1.03 |
| (c) | (b)の中心線の両端点、長さの動画q90 | 合格 | 0.028（保持率50%） | 0/144 | 0.893（140／4） | — |
| (d) | (c)の長さを選択した1 frameで取る（選択条件は5.0） | 合格 | 0.047（保持率50%） | 1/144（`no_usable_frame`） | 0.902（139／5） | — |

**選定: 主(c)、副(b)(d)。** (c)と(b)は同値で、事前順位により(c)。(a)は合格したが副候補の上限（2）により候補外。
区分別の値（中央値／P90）: (b)(c)は間引き（保持率50%） 0.011／0.028、揺らぎ 0.008／0.023、背景点の追加30% 0.000／0.005。(a)は揺らぎ 0.049／0.095。

### 5.3 H17-2〜H17-4: validation18（best＝主）

H17-2の表現はH17-1の候補（(c)・(b)・(d)）。`τ_rel`＝10%は記述のみで`T_FL`ではない。最良後処理はvalidationで選んだ開発上の選択（楽観的）。

| 後処理 | frame存在 P／R | instance検出 P／R | (b)(c) 10%以内 | (b)(c) \|相対誤差\| 中央値（符号付き） | (d) 10%以内 | (d) \|相対誤差\| 中央値（符号付き） | (d) oracle 符号付き中央値 |
| --- | --- | --- | ---: | --- | ---: | --- | --- |
| P0 | 0.379／1.000 | 0.036／0.515 | 2/18 | 1.405（+1.405） | 3/18 | 1.263（+1.263） | +0.331 |
| P1 | 0.379／1.000 | 0.077／0.179 | 2/18 | 1.405（+1.405） | 3/18 | 1.263（+1.263） | +0.331 |
| P2 | 0.380／1.000 | 0.039／0.509 | 2/18 | 1.405（+1.405） | 3/18 | 1.263（+1.263） | +0.331 |
| P1+P2 | 0.399／0.958 | 0.081／0.173 | 2/18 | 1.436（+1.436） | 3/18 | 1.263（+1.263） | +0.542（15件） |
| P3 | 0.379／1.000 | 0.063／0.651 | 2/18 | 0.455（+0.237） | **6/18** | 0.351（−0.004） | −0.218 |

- frame存在（P0の集計）: GTのある287 frameをすべて検出した一方、GTの無い483 frameのうち470で予測が存在した。P1・P2・P3も287 frameをすべて検出したが、P1+P2は287 frame中275（FN 12）だった。
- prior-only: 10%以内は(a)(b)(c)が2/18、(d)が0/18。符号付き相対誤差の中央値は+0.36〜+0.45。
- 手法側の失敗は、全表現・全後処理で0件。
- last（補助）: frame存在はP 0.39〜0.42／R 0.59〜0.68。(b)(c)の10%以内は1〜3/18、(d)は0〜5/18。(d)のoracleは定義件数が8〜10/18にとどまる。

**H17-3 FPの空間分類（best・P0、有効背景上のFP 525,800点が分母）:** GT近傍0.3%、GTと同じframeの遠方38.3%、GTの無いframe61.5%。予測陽性全体に占めるignore上の割合は1.3%。距離による分類であり、別の解剖構造とは断定しない。

**matching成立ペアの誤差（条件付き。未検出・未対応は含まない）:** 各列は**成立ペア間の中央値**である。「端点誤差（平均）」は、各ペアで2つの端点誤差を平均した値の、ペア間での中央値（全ペアの平均ではない）。best・P0とP2では、端点・角度・長さの誤差に各1件の未定義がある（軸が使えないペア。中心誤差は未定義なし）。

| checkpoint | 後処理 | ペア数（動画数） | 中心誤差 | 端点誤差（平均） | 角度誤差 | 長さ相対誤差 符号付き／絶対値 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| best | P0 | 167（16） | 3.29 px | 8.38 px（未定義1） | 6.93°（未定義1） | −0.193／0.372（未定義1） |
| best | P1 | 58（8） | 4.57 px | 18.06 px | 4.12° | +0.166／0.427 |
| best | P2 | 165（15） | 3.28 px | 8.27 px（未定義1） | 6.93°（未定義1） | −0.192／0.365（未定義1） |
| best | P1+P2 | 56（7） | 4.57 px | 18.80 px | 4.12° | +0.229／0.466 |
| best | P3 | 211（16） | 3.51 px | 6.90 px | 7.75° | −0.352／0.420 |
| last | P0 | 29（7） | 6.72 px | 11.04 px | 12.90° | −0.556／0.556 |
| last | P3 | 52（9） | 7.44 px | 11.50 px | 11.13° | −0.549／0.549 |

best・P0では、動画単位の(c)の長さが過大（+140%）だが、matching成立ペアの長さは過小（−19%）になる。成立ペアの長さの向きは後処理で異なる（bestのP0・P2・P3は過小、P1・P1+P2は過大）。動画単位の過大評価は、GTとmatchしないまとまった予測領域がM1（最大instance）になることによる可能性がある［推論］。

### 5.4 sanity3（in-sample、別表）

bestのP0〜P2で(b)(c)の\|相対誤差\|の中央値は0.107、10%以内1/3。lastのP0〜P2では0.39〜0.41。
matching成立ペアの絶対相対誤差の中央値は、lastのP0・P1・P2・P1+P2で約0.9〜1.6%、P3で34.2%。学習済み動画での値であり、validationと合算しない。

## 6. 解釈（管理書4.5の確定事項）

- (a)〜(d)は「元frame平面内のGT陽性領域に由来する2Dの広がり・長さ」、主に**各frameに映る骨断面の2Dの広がり**である。臨床FLや骨の解剖学的全長と同一視しない。約30.9 raw pxという絶対値だけで断面の種類を判定しない。
- **H17-1は、事前の摂動・判定規則に対するGT由来幾何量の安定性試験として成立した。** 主(c)はこの量の表現としての安定性を示すにとどまり、FLに必要な情報の保持は証明していない。
- (b)と(c)は長さ・領域が同じ表現群であり、独立した2方式の支持とは数えない。(c)が代表、(b)は同じ幾何をboxで表す補助形式、(d)はframe選択を含む別構成。(a)はH17-1に合格したが選定上限により候補外となった記録。
- (d)の「最良frame」は、骨全長の測定断面を含むことを意味しない。複数frameで骨全体を観測することと、frameの進行方向が骨の長軸に一致することは別である。
- H17-2〜H17-4: bestの存在recallは高いが、GTの無いframeへの予測が多く、GT由来長さとの一致も限定的だった。(d)+P3の6/18は、validationで選んだ開発上の記述結果であり、後処理の一般的な優位性・臨床FL精度・production採用を示さない。lastで誤差が小さくなる条件は、存在recallの低下とoracleの定義件数の減少と併記する。
- FP摂動は「同一frameの有効背景から一様に点を加える事前固定の摂動」に限った頑健性である。まとまった領域・時間的に連続する誤検出・別instanceが最大成分になる場合への頑健性は示していない。
- (e) pseudo-3Dは単位制約付きの参考値・候補外。sweep距離、frame数、重心軌跡長をそのまま骨長としない。

## 7. S5-20で必要となる情報

### 7.1 最初に決める必要があること（S5-17では決まっていない）

1. **何の長さを測るか。** 各frameの断面の2D広がり（S5-17の(a)〜(d)）か、骨全体を覆う複数frameの情報を統合して得る全長か。後者は全長推定に向けて検討する意義があるが、frame間の位置・姿勢と物理スケールの対応が必要である。
2. **座標と単位の契約。** pseudo-3Dの単位と物理幾何の対応は未確定である。mm/pixelの供給元・形式・完全一致キー・粒度・frame対応も未確定（UNCONFIRMED）。
3. **許容誤差`T_FL`。** ユーザーが設定する事項で、未設定である。
4. **dualtrack由来の形状歪みの扱い。** 曲がりを解剖学的形状とも追跡誤差とも断定しない。誤差量は測っていない。

S5-17の結果は、この選択を決める根拠ではない。2D断面由来量についての診断として引き継ぐ。

### 7.2 S5-20で使える確定事項

| 事項 | 内容 | 根拠 |
| --- | --- | --- |
| 2D断面由来量の表現 | (c)（PCA 2〜98分位、端点は矩形の中心線上）を代表とする表現は、事前の摂動に安定。(b)は同じ表現群、(d)はframe選択を含む別構成、(a)は合格したが候補外の記録 | 5.2 |
| 表現の定義上の性質 | (b)と(c)は領域・長さが定義上一致する（端点を重心線ではなく矩形の中心線上に置くため）。副軸方向の分位範囲は重心に対して非対称になり得る | 管理書4.2、コア合成テスト[4] |
| 複数instance | 複数instance frameでは、M1（最大instance）とM2（全instanceの統合）の長さがほぼ常に10%以上異なる（定義は5.0）。**S5-20では複数instanceの扱い（instance選択か集合matchingか）を明示的に設計する必要がある** | 5.1（記述） |
| 予測の傾向（S5-15 R0 best） | GTの無いframeへの予測が多い（P0の存在precision 0.379）。best・P0では、FPの61.5%がGTの無いframe、38.3%が同frameの遠方で、動画単位の長さは過大（+140%）。matching成立ペアの長さは、bestのP0・P2・P3で過小、P1・P1+P2で過大であり、後処理全般に共通する向きはない。frameの存在判定（objectness）とcontextを設計する際の記述的な材料になる（採否の根拠ではない） | 5.3 |
| frame選択 | 実用の(d)（予測の点数×平均確率で選ぶ）とoracleの(d)（GTで選んだframe）で誤差が異なる（P0で+1.263対+0.331）。frame選択の質が長さに効く可能性がある［推論］ | 5.3 |
| BBoxと点GTの関係 | 点GTの外接矩形はBBoxより長辺が6%程度短い傾向（IoU中央値0.73）。BBoxをtargetにする場合と点GTから作る場合で、同じ量にはならない | 5.1 |
| prior-only | 元frame正規化座標で作ったpriorの10%以内は0〜2/18。bestのP0〜P2と同程度 | 5.3 |

### 7.3 座標系について確認できた事実（S5-20aの`GeometryMetadata`・`coord_space`契約の材料）

- teacher H5の`pixel_xy`はlocal crop座標である。元frameへの逆変換は`x=(u+left)/s`、`y=(v+top)/s`（`resize_shorter_then_offset_crop`・`offset_crop`の場合）。
- この逆変換は中間H5に記録されたscale（`local_resize_scale`）に基づく。実際のresize後の寸法の丸め（軸ごとに最大0.5/辺程度）と、`cv2.INTER_AREA`によるsub-pixelのずれは厳密には再現しない。確認したのは、手計算の既知座標とStage 4の順変換式との整合（合成テスト）と、実データの座標が範囲内にあること（逆変換後は1 px の余裕で検査）までである。**物理座標の正確性（mmでの位置精度）までは確認していない。**
- 今回の162件はすべて`resize_shorter_then_offset_crop`だった。`resize`（x/y異方）や`offset_crop_fallback_resize`（Stage 4側の変換規約と食い違う可能性、報告書1.3 F4）が将来のデータに現れた場合は、`CropTransform`で別扱いになる（後者は入力評価不能）。
- 逆変換に必要なattr（`raw_width`・`raw_height`・`local_crop_left`・`local_crop_top`・`local_resize_scale`・`local_preprocess_effective`・`local_input_shape`）は、中間H5から全件読めた。モードは要求値の`local_preprocess`ではなく、実効値の`local_preprocess_effective`で判定する必要がある。
- `frame_order`はlocal frame数（`local_input_shape`の先頭）未満である（全件で検査済み）。
- 元frame正規化座標（`x/W, y/H`）は等方ではない。正規化空間で長さを平均すると、解像度・視野の異なる動画で物理的な意味が変わる（prior-onlyの限界）。
- PSEUDO3D（teacher `points`）は単位未確定であり、MM_XYと演算させない（評価器は型で拒否する）。

### 7.4 S5-20の比較に使える評価器の機能

| 機能 | 所在 |
| --- | --- |
| 2D幾何（存在・中心・軸・端点・長さ・幅）の抽出とGTとの誤差（端点交換・軸符号に不変） | `stage5/geometry/frame_geometry.py`・`metrics.py` |
| frame存在判定（TN含む）、instance検出（M3）、FP空間分類（分母明示） | `metrics.py` |
| 動画単位の長さ（(a)〜(d)）と失敗理由、oracleの分離 | `fl_estimate.py` |
| prior-only baseline（元frame正規化座標） | `prior.py` |
| 摂動安定性とF-A規則 | `metrics.py` |
| 封印・許可リスト・coverage・出力境界 | `evaluate_stage5_geometry.py` |

新runに使う前の改修（S5-18依頼書の冒頭タスク、8.2）が必要である。

## 8. S5-18・S5-19での使い方

### 8.1 使用契約

- 主判定は、S5-16 v3の既定（AUPRC、同一FPRでのrecall、動画別paired差分等）を維持する。**S5-17の幾何量は副次診断とする。**
- 副次診断として使うもの: frame存在判定のP／R、instance検出のP／R、FPの空間分類、(c)・(d)の長さのGTとの相対誤差（2D断面由来量の一致の診断値。FL精度とは呼ばない）、matching成立ペアの誤差（条件付き、ペア数と併記）。
- 未定義は0にせず件数で示し、分母を明記する。validationで選んだ条件は開発上の選択として扱う。

### 8.2 再利用の前に必要な改修（S5-18依頼書の冒頭タスク、今回は未実施）

1. run識別・checkpoint/epoch・来歴の根拠を新runに適用できるようにする（現在はS5-15長期runに固有の前提がある: epoch 6／50、S5-15 launcherのvalidationリストhashを来歴の証拠とすること）。
2. 評価器が**生成時に**private記録のhashと対象集合を結びつけるcoverageを記録する。動画別identityを含むcoverageはprivate側に置き、共有側には成果物hash・集合fingerprint・件数などだけを記録する（S5-17では生成時hashがなく、後から来歴を立てる必要が生じた。報告書14.13）。
3. 共有JSONの項目と仕様の対応を合成テストで検査し、集計漏れを防ぐ（S5-17では4項目が漏れ、補完が必要になった。報告書14.8）。

## 9. 限界

1. GT由来量どうしの比較であり、臨床FL精度ではない。長さはx/y等方を仮定したraw pxで、mmではない。
2. (a)〜(d)は各frameの骨断面の2D広がりであり、骨の解剖学的全長ではない。断面は動画によって異なる。
3. ファイル単位で、同一検査の兄弟ファイルを含む。旧train／validationにまたがる同一検査は4グループ（両集合で計17ファイル）で、validation側の件数は未確認。動画別の統計は検査単位のばらつきを過小評価する。
4. validationは開発で使った集合で、最良後処理もvalidationで選んだ。学習runもseedも1つだけ。
5. 来歴の限界は4.3のとおり。
6. FP摂動は散在点の一様な追加に限った条件。
7. H17-1b（BBox整合性）はframe単位のプール統計がなく、動画ごとの中央値の分布だけ。
8. pseudo-3Dにはdualtrack由来の形状歪みがあり得る。誤差量は未測定。
9. 保持被覆率は記述量であり、点数に強く依存する（合成観測、報告書11.1）。(d)のframe選択を含む挙動は観測していない。

## 10. 記録ファイル

### 10.1 リポジトリ内の文書

| 文書 | 内容 |
| --- | --- |
| [S5-17依頼書](stage5_s5_17_implementation_handoff.md) | 背景、Step 0からの引継ぎ、前提の訂正 |
| [S5-17管理書](stage5_s5_17_implementation_management.md) | 確定仕様（4章）、S17-5結果の解釈（4.5節）、失敗・未定義の規約（5章）、ステップと状態（6章） |
| [S5-17報告書](stage5_s5_17_report_to_policy_chat.md) | 1〜3章: H17-1の再設計と管理判断／5〜12章: 実装・合成テスト／13章: S17-4の結果／14章: S17-5の結果・管理判断・補完集計／15章: 結果整理・引継ぎ・クローズ |
| [全体管理記録](../stage5_revision_management_record.md) 5章「S5-17」、6章 | Stage 5全体での位置づけ、S5-18の冒頭タスク |
| [評価レポート](../stage5_pointnext_s_training_evaluation_report.md) 9.13節 | 結果の要約 |
| [FILES.md](../FILES.md)、[data_construct.md](../data_construct.md) | コード・出力の配置と共有境界 |

### 10.2 共有JSON（集計のみ、privacy検査済み、共有可）

開発コンテナの写しは2026-09-29に[research/stage5/s5-17/shared/](../../../research/stage5/s5-17/shared/)へ移動した。内容・SHA-256は移動前後で一致している。
実機での生成場所は`/mnt/data/3d_projects/stage5_private_work/s5_17_geometry/shared/`。

| ファイル | 生成 | 内容 | SHA-256 |
| --- | --- | --- | --- |
| [coverage_best.json](../../../research/stage5/s5-17/shared/coverage_best.json) | S17-4 `register-coverage`（best） | 評価runの登録: 件数・hash先頭16桁・証拠の成否 | `b36266f769c869518df24f12326ccacdc3a57fe20b9fe8d192c4c0aea66f902f` |
| [coverage_last.json](../../../research/stage5/s5-17/shared/coverage_last.json) | S17-4 `register-coverage`（last） | 同上 | `9846c411cebf99b6b03e22005d48e292344c8a0f5fad09e6ebd02ff2fc224540` |
| [audit.json](../../../research/stage5/s5-17/shared/audit.json) | S17-4 `audit` | メタ監査の件数（dataset、入力評価不能、crop mode、解決方法、NPZキー、hash状態） | `ddc38a83f13a4b5ca4520a8a5523d9cfacd3f90957ecdba12faa571d7359f62e` |
| [run.json](../../../research/stage5/s5-17/shared/run.json) | S17-5 `run` | H17-1（表現別の分母・安定性区分・候補規則・保持被覆率・膨張率・選定）、H17-2〜H17-4（validation・sanity×best・last×後処理×表現の表、prior-only、FP空間分類）、train_core由来の定数、来歴状態、注記、限界 | `3348f49823967676d2ded96504725b91e4dda8659def089a1094ecbfcfd9035d` |
| [supplement_registration.json](../../../research/stage5/s5-17/shared/supplement_registration.json) | 補完 `register` | private記録の登録（完全hashの一致、実行者確認の参照先、メタ情報の検査結果） | `30e40fdac9d095ab1efb579d0ed80081d8605f5357625e120cf53249f7c63af9` |
| [supplement.json](../../../research/stage5/s5-17/shared/supplement.json) | 補完 `aggregate` | (e) pseudo-3D参考値、H17-1b BBox整合性、H17-5 複数instance、H17-2 matching成立ペアの誤差（validation・sanity×best・last×後処理） | `0141d323f497245985709b0cadc3e72dcf241035bfb0ea4d3f4760d8db60c3f4` |

`run.json`の主なキー: `h17_1.per_representation.<表現>`（`denominators`・`stability_cells`・`candidate_rule`・`holdout_coverage_descriptive`・`relative_inflation_descriptive`）、`h17_1.selection`、`tables.<validation|sanity_in_sample>.checkpoints.<best|last>.postprocess.<P0〜P3|P1+P2>`（`presence_pooled`・`detection_pooled`・`fp_classes_pooled`・`representations`）、`tables.*.prior_only`、`constants_from_train_core`、`provenance_hash_states`。

### 10.3 privateの記録（DO_NOT_SHARE、実機のみ）

場所: `/mnt/data/3d_projects/stage5_private_work/s5_17_geometry/`（mode 0700）。**共有しない。**

| ファイル | 内容 | 備考 |
| --- | --- | --- |
| `geometry_run_DO_NOT_SHARE.json` | alias対応表（実ID・パス）、動画別の幾何・推定値・誤差・摂動の変化・matchingペア | 完全SHA-256 `93881f0fe6b0fe613317dbfc7d0e979a648c8465402349819ef480e414441b65`（実行者確認で固定、報告書14.15） |
| `artifact_coverage_DO_NOT_SHARE.json` | 評価runの`summary.json`・`h5_metrics.csv`（best・last）とprivate記録のcoverage（動画identity付き） | 補完の登録で1件追記 |
| `input_audit_DO_NOT_SHARE.json` | S17-4の動画別メタ情報 | — |
| `run_resource_usage.txt` | S17-5の計測（経過2分53秒、最大常駐メモリ約4.4 GB、終了コード0） | — |

入力として読んだもの（変更していない）: 評価出力`/mnt/data/3d_projects/stage5_evaluations/260919/pointnext_s_EX260919_s5_15_r0long50_none_gn8_cwfixed_lr1e3_ep50_bs1_acc8_nopad/{best,last}/`（`summary.json`、`h5_metrics.csv`、`predictions/`）、分割リスト`/mnt/data/3d_projects/stage5_splits/s5_16_step0_bprime/`（train_core・validation）、sanityリスト（上記評価出力の`evaluation_data/selected_train_files.txt`）、teacher v7 H5、中間H5の属性。

## 11. 未決事項（後続段階へ）

| 事項 | 扱う段階 |
| --- | --- |
| 何の長さを測るか（断面の2D広がりか、frame統合による全長か）と、その座標・単位の契約 | S5-20a |
| mm/pixelの供給元・形式・完全一致キー・粒度・frame対応 | mm評価を行う前 |
| `T_FL` | ユーザー設定、S5-20の判定前 |
| pseudo-3Dの物理幾何との対応、dualtrack由来の歪みの扱い | S5-20a以降 |
| 複数instanceの扱い（instance選択か集合matchingか） | S5-20a |
| 共通評価器の再利用改修（8.2） | S5-18依頼書の冒頭 |
| 実測FL付き5例（D-040）の比較単位 | S5-18の学習中にアノテーションし、比較は算出方法・手順・checkpointを固定してから行う。**用途は固定した方法による探索的比較とmm換算の健全性確認の補助に限り、学習投入・補正係数の較正・方式選択・task/production採否には使わない**（結果を見て方法を変えた場合、そのデータは開発利用として記録する） |
| 共有JSONの配置の確定と本書の修正 | 完了 |
