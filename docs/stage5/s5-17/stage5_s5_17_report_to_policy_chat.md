# S5-17 幾何表現のtraining-free診断・共通評価器 報告と進捗管理

作成日: 2026-09-27  
最終更新日: 2026-09-29（12.10: 12.9の管理書同期、run経路の固定、R6のrepo内確認、S17-4の実行手順）  
開始時HEAD: `f31cdd8444d08adea522e350e0d37b96717c03c4`（`f31cdd8 docs(stage5): record Step 0 acceptance and synchronise Stage 5 records`）  
担当: S5-17実装チャット（管理は総括管理チャット、D-038）  
状態: **S17-1・S17-2完了。S17-3は管理側受入済み、S17-4はユーザー・管理側承認済み（12.9）。S4-2・S4-3は確定、S4-4はR6と監査結果を確認して判断する。R1〜R4は事前回答を必須とせずS17-4で確認、R5はpixelモードでは不要、R6は既存記録を確認して報告する。R7・R8はユーザー回答済み。S17-4の実施結果は未報告、S17-5は未承認。12.10で管理書を版6へ同期し、S4-2・S4-3をコードに反映（CLI合成テストの再実行待ち）。**  
現行仕様: [S5-17実装・検証管理書](stage5_s5_17_implementation_management.md)。本書1〜3章は提案履歴、4章は承認・文書運用の決定記録として保持する。


本書は[S5-17実装依頼書](stage5_s5_17_implementation_handoff.md)8章の報告先である。
節は「提案／管理決定／実装／合成テスト／実機実行／受入」の順に追記し、過去の節は上書きしない。

---

## 1. 初回報告: 仕様・入力契約の監査と具体案（2026-09-27）

担当: S5-17実装チャット。状態: **提案。管理判断待ち。**
本章の作業は、リポジトリ内の文書・コード・保存形式の読取りだけである。
実H5・予測NPZ・中間H5・封印物・private JSONは読んでいない。`/mnt/data`へのアクセス、学習、GPU、推論、
合成テストの実行は行っていない（本開発コンテナにはnumpy/h5pyが入っていない。1.9.1）。

### 1.1 開始時の状態

| 項目 | 内容 |
| --- | --- |
| HEAD | `f31cdd8`。依頼書3章の基準コミット`392b0ab`・`b10dd87`・`f31cdd8`はこの祖先に含まれる |
| 作業ツリー | 管理側の未コミット文書差分5件（管理記録、評価レポート、Step 0報告37章、FILES.md、README.md）と未追跡の`docs/stage5/s5-17/`（本依頼書）。**本チャットはこれらを変更していない**（本報告書の新規作成のみ） |
| pin | `Stage5/stage5/config/split_contract_pins.json`に`seal_registry`・`s5_16_bprime`の2件。依頼書3章と一致 |

確認した文書: 依頼書全文、管理記録0章・3.1・6章現行フロー・D-035〜D-041、Step 0報告15章・24章・29〜37章、
S5-16 v3の6.1・10.0〜10.4・11〜12章、評価レポート9.10〜9.12、FILES.mdの封印期間節、data_construct.md。

確認したコード: `stage5/utils/{split_contract,split_identity,file_list_mode,h5_io,label_policy}.py`、
`checks/real_h5/check_stage5_{gt_component_count,xy_coordinate_provenance,coordinate_transform_reconciliation,s5_14_supplement}.py`、
`checks/real_h5/audit_stage5_train_core_mm_metadata.py`、`evaluate_stage5.py`（予測NPZの書出し）、
`export_stage5_prediction_frames.py`（NPZ読込）、`Stage2to4/src/utils/pseudo3d_processing.py`（crop/resize）、
`Stage2to4/pseudo3d/annotation/annotate_pseudo3d_point_cloud.py`（BBoxのlocal座標変換）。

### 1.2 理解と旧10.1からの差分

**目的:** 既存GTと保存済み予測だけから、(i) GT由来の幾何表現がFLに必要な情報（存在・軸・端点・長さ）を
どれだけ安定に保持するか、(ii) 保存済み予測＋非学習後処理がそのGT由来幾何にどこまで近づくかを測り、
S5-18/19の副次指標とS5-20の候補表現を渡す。**臨床的な正しさ、学習可能性、geometry headの汎化は示さない。**

**固定条件（再実施・変更しない）:** 144／18／18分割と封印、class weight（S5-17では使用しない）、
teacher v7、R0 best epoch6／last epoch50の保存予測、学習・GPU・再推論なし、production未変更。

| 旧10.1の項目 | 扱い | 理由・変更内容 |
| --- | --- | --- |
| H17-1「臨床FLと最もよく一致する代理FL定義」 | **変更** | 180動画に臨床実測FLが無い。1.5で「情報保持・整合性・摂動安定性」の問いへ置換 |
| 採否(1)「train_coreで臨床FL誤差最小を1つ選ぶ」 | **変更** | 実行不能。1.5.4の足切り＋事前順位＋判定不能を許す規則へ |
| 採否(2)「T_FL以内の動画がvalidation過半」 | **修正して維持** | H17-2（予測→GT由来幾何）の判定としてのみ残す。mm未確定時は相対許容で代替し、T_FLとは呼ばない（1.5.5） |
| 採否(3)「prior-onlyとの差が小さい表現は外す」 | **維持** | 差の定義を1.6.7で固定 |
| 採否(4)「FPの主体が離れた別構造ならobjectness必須」 | **修正** | 距離で「別構造」と断定しない。「GTから遠いFP」とし、解剖学的解釈は可視化等の別根拠が要る（1.6.8） |
| H17-2〜H17-5 | 維持（細部修正） | H17-2はoracle／実用を分離。H17-5は規則比較の事前固定 |
| 表現(a)〜(e) | 維持（上限付き） | (e)は単位制約付き参考値に限定し、採否の対象外 |
| 後処理4種 | 維持（上限付き） | 組合せは5通りに限定（1.6.5） |
| metrics「臨床FLとのbias/MAE/Bland–Altman」 | **削除** | 臨床FLのない集合で臨床MAE・一致限界を作らない |
| metrics（mm単位の誤差） | **保留** | mm/pixel確定まで元frame pixelと相対量で報告 |
| fail-fast「臨床FL欠損は除外として記録」 | **削除** | 全件欠損で完了扱いにしない（依頼書4.1） |
| fail-fast（その他） | 維持＋追加 | 1.8.5に列挙 |
| decision gate | 維持（修正） | 「代理FL定義を固定」→「採用・保留・判定不能のいずれかを事前規則で記録」 |

### 1.3 repo監査で判明した事項

実データを読まずにコードから判明した事項で、S5-17の設計に影響するものを挙げる。
**F2・F3・F4はStep 0の既存報告の読み方に関わる。** Step 0のやり直しを求めるものではない。

| # | 所見 | 根拠 | S5-17での扱い |
| --- | --- | --- | --- |
| F1 | 予測NPZの`point_indices`は`np.arange(N)`で、NPZにはcheckpoint・H5のhashが記録されていない。点対応の証拠は点数一致だけ | `evaluate_stage5.py:483-489` | 点数一致に加え、NPZ＋H5から混同行列を再計算し、同じcheckpointディレクトリの`h5_metrics.csv`の動画行と**整数で完全一致**することを点対応・checkpoint紐付けの条件にする（1.4.3、M-3） |
| F2 | `resolve_intermediate_h5`は記録attrで解決できない場合に`{video_name}*.h5`のglobへfallbackする。**前方一致**なので`…_091`が`…_091_02`の中間H5へ解決され得る | `check_stage5_xy_coordinate_provenance.py:148-178` | S5-17では**使わない**。許可するのは記録attr（識別子の完全一致を追加検査）と完全一致basenameだけ。glob・曖昧は停止。Step 0のmm監査144/144がどの方法で解決されたかの**方法別件数**（共有可能）の確認を依頼する（1.4.4） |
| F3 | mm監査の「crop逆変換メタ全field完備」は`is not None`の判定で、`local_resize_scale`が**NaN**（`resize`、`offset_crop_fallback_resize`モード）でも完備と数える。`local_preprocess_effective`と`local_resized_h/w`は集計していない | `audit_stage5_train_core_mm_metadata.py:177-182`、`pseudo3d_processing.py:473-483,630-643` | 144/144は「fieldが読める」であり、**数値的に逆変換可能**の件数ではない（35.3-1の補正と整合）。S5-17の実機fail-fastで`local_preprocess_effective`別件数と有限scale件数を確認する |
| F4 | Stage 4のBBox→local変換は要求モード`local_preprocess`で分岐し、非resizeモードでNaN scaleを1.0に置き換える。frameがcropより小さい`offset_crop_fallback_resize`に該当する動画があれば、GT BBoxのlocal座標がずれ得る | `annotate_pseudo3d_point_cloud.py:465-503` | teacher側の潜在的な論点として**報告のみ**。S5-17は修正しない。該当動画があればS5-17では変換不能として除外し、件数を報告する。該当0件なら論点は消える |
| F5 | BBoxのraw座標はCVAT XMLの画像寸法から`raw_width/raw_height`へ縮尺している。「元frame」がCVAT画像か復号したraw frameかで寸法が異なり得る | 同上`:480-483` | mm/pixelの基準がどちらのframeかをユーザーに確認する（1.7.1-2） |
| F6 | `resize_shorter_then_offset_crop`は等方の`scale`を記録するが、実際のresize後寸法は`round(h*scale)`・`round(w*scale)`で、軸別の実効倍率は最大`0.5/h`程度ずれる。補間は`cv2.INTER_AREA` | `pseudo3d_processing.py:541-555,432-447` | 逆変換はStage 4のBBox変換と同じ規約`x_raw=(x_local+left)/scale`に揃える。この丸め差は許容誤差の見積りに入れる（1.7.3） |
| F7 | `ActiveContract.assert_paths_allowed`は封印集合の**拒否**だけで、処理目的ごとの**許可リスト照合**は行わない | `split_contract.py:305-329` | S5-17のCLIで目的別許可リスト（GT prior構築＝train_core、予測診断＝validation、in-sample診断＝train_core∩予測あり）を追加する（1.8.2） |
| F8 | `check_stage5_gt_component_count`の`privacy_self_check`はtimestamp3区切りの旧パターンで、case形式・第4セグメントを検出できない。`read_gt_points`は`astype(int64)`でframe番号を検証前に丸める | `check_stage5_gt_component_count.py:43,100-116,246-251` | `component_sizes`（純関数）だけを関数import候補とする。privacy検査は`split_identity.contains_video_identity`を使う新規実装。frame番号は整数性を検証してから変換する |
| F9 | S5-14のpriorはlocal crop座標を共通寸法で正規化し、自己除外を`Path.resolve()`で判定する | `check_stage5_s5_14_supplement.py:178-250` | local crop座標は動画間でcrop位置が異なり揃わないため流用しない。priorは元frame正規化座標で作り、自己除外はvideo identityで判定する（1.6.7） |
| F10 | **文書間の不整合:** D-041とStep 0報告24.2の「4グループ・17ファイル」は**train/validationにまたがるグループの合計**で、validation側の件数は「少なくとも4、最大13、正確な数は未取得」（24.3）。依頼書4.3と評価レポート9.12は「validation18のうち17ファイル」と記載している | Step 0報告24.2-24.3、依頼書4.3、評価レポート9.12 | 本チャットでは訂正しない。validation側の正確な件数（共有可能な集計1値）の確認と、記載の訂正を管理判断に上げる（M-8） |

### 1.4 入力契約

#### 1.4.1 必要な保存物

「repo」はコードで形式を確認済み、「実機」は所在・件数・内容が未確認であることを示す。

| 保存物 | 用途 | 対象集合 | 必要な項目 | 単位・座標 | 状態 |
| --- | --- | --- | --- | --- | --- |
| pins／seal registry／`s5_16_bprime.json` | 契約解決・許可リスト | — | pin hash、`lists.*.identity_sha256` | — | repo（pin）、実機（manifest本体） |
| `train_core_144.txt`、`validation_18.txt` | 対象集合の確定 | train_core／validation | パス一覧 | — | 実機（Step 0で作成済みと報告） |
| teacher v7 H5 | GT幾何・prior・予測の照合 | train_core144、validation18 | `point_cloud/{pixel_xy,frame_order,points}`、`annotation/{point_label,valid_mask}`、`frame_annotation/{frame_order,bbox_local_xyxy}`、file attrs `video_name`・`source_pseudo3d_h5` | `pixel_xy`＝local crop pixel。`points`＝pseudo-3D、物理単位未確認 | repo（形式）、実機（件数は162件のpreflight通過で確認済み） |
| 中間pseudo-3D H5（attrsのみ） | crop逆変換、frame数照合 | 同上 | `raw_width/height`、`local_crop_top/left`、`local_resize_scale`、`local_preprocess`、`local_preprocess_effective`、`local_resized_h/w`、`local_input_shape`、`num_frames` | 元frame pixel | repo（形式）、実機（train_core144はfield読取り済み。validation18は未監査。F2・F3） |
| 予測NPZ | H17-2〜H17-5 | validation18（主）、sanity3（in-sample補助） | `pred_label`、`prob_femur`、`vote_count`、`point_indices` | 点index＝H5の行番号 | repo（形式）、**実機（所在・run・epoch・coverage未確認）** |
| checkpointディレクトリの`h5_metrics.csv`、`summary.json` | 点対応とcheckpointの紐付け | 同上 | 動画別TP/FP/TN/FN、checkpoint名・epoch | — | repo（形式）、実機未確認 |
| mm/pixel | mm換算 | 未定 | 未定（1.7.1） | 元frame基準（ユーザー回答） | **供給元・形式・キー・粒度が未確定** |

`intermediate spacing`（全件1.0/1.0）は使わない。5例（D-040）は入力に含めない。

#### 1.4.2 予測の対象と coverage の想定

評価レポート9.10.1から、S5-15長期runの評価は**train sanity3＋validation18の21動画、best／last**である。
train_core144全体の予測は**無いと想定する**。不足をGPU再推論や別checkpointで埋めない。

- 主: R0長期run best（epoch6）のvalidation18。補助: 同last（epoch50）のvalidation18。
- sanity3の予測は**in-sample診断**として別表にだけ出す（学習済み動画であり、validationと合算しない）。
- 5 epochごとのcheckpointの評価予測があっても使わない（比較対象を増やさない）。

#### 1.4.3 点対応の検査（F1への対応）

各動画で次をすべて満たさなければ、その動画ではなく**実行全体を停止**する。

1. NPZのファイル名から得たidentityが、許可リストのH5のidentityと完全一致する（`extract_video_identities`、ファイル名は`safe_name(video_name)`）。
2. NPZの各配列長＝H5の点数、`point_indices`＝`arange(N)`、`pred_label∈{0,1}`、`prob_femur`が有限かつ[0,1]、`vote_count≥1`。
3. `pred_label`と`prob_femur≥0.5`が一致する（2クラスargmaxの再現。同値時の扱いは評価コードの規約に合わせる）。
4. NPZ＋H5の`point_label`・`valid_mask`から再計算したTP/FP/TN/FNが、同じcheckpointディレクトリの`h5_metrics.csv`の当該動画行と整数で完全一致する。
5. `summary.json`のcheckpoint名・epochが、指定したbest（6）／last（50）と一致する。

4と5は、NPZ自身が持たないcheckpointとの紐付けを外部の保存物で補うためのものである。
`h5_metrics.csv`は複数動画を含む保存物なので、読込前の扱いを1.8.2とM-3で決める。

#### 1.4.4 実機で確認が必要な事項（個人情報を含まない所在確認）

ユーザーに依頼したい確認を、共有可能な情報だけで答えられる形にした。**GT統計・動画別の値は求めない。**

| # | 確認事項 | 求める回答の形 |
| --- | --- | --- |
| R1 | S5-15長期runの評価出力root（`<EX_DATE>/<EXPERIMENT_NAME>`）と、`best/`・`last/`の`predictions/validation/`・`predictions/train_sanity/`の存否 | 存否とNPZ件数（例: 18／3） |
| R2 | 同ディレクトリの`h5_metrics.csv`・`summary.json`の存否 | 存否 |
| R3 | train_core144の中間H5解決方法の方法別件数（Step 0 mm監査のprivate JSONにある`intermediate_resolution_method`） | 例: recorded_source_attr 144／glob 0 |
| R4 | validation18の中間H5の所在（train_coreと同じrootか） | はい／いいえ |
| R5 | mm/pixelの供給元（1.7.1） | 1.7.1の各項目 |

R1〜R4は、実装後の実機fail-fastでも機械的に検査する。事前に聞くのは、所在が違えば入力契約と設計が変わるためである。

### 1.5 H17-1の再設計

#### 1.5.1 答えられる問いと答えられない問い

臨床正解がない条件で**答えられる**のは、同じアノテーションから作った表現同士の性質である。

| 問い | 答えられるか | 理由 |
| --- | --- | --- |
| GT点群を各表現へ変換したとき、GT点をどれだけ覆い、どれだけ余分な領域を含むか | はい | GT点とその表現だけで閉じる |
| 点GT由来の表現と、保存済みBBoxアノテーション（`bbox_local_xyxy`）由来の表現がどれだけ一致するか | はい（ただし限定的） | 同じ注記系列の2つの見え方の**整合性**。**独立な検証ではない** |
| 点の間引き・偽陽性ノイズ・境界の揺らぎで、動画単位の長さ推定がどれだけ変わるか | はい | 摂動は合成で、真値を必要としない |
| xy異方スケールの誤差が長さにどう効くか | はい（解析的） | 軸角度の関数として算出できる |
| どの表現が真の大腿骨長に最も近いか | **いいえ** | 参照値が無い。GT由来量同士の一致は臨床的な正しさではない |
| 摂動に安定な表現が臨床的に正しいか | **いいえ** | 安定性は必要条件の候補に過ぎない |

#### 1.5.2 改訂H17-1（提案）

> **H17-1（改訂）:** train_core144のGTについて、候補表現(a)〜(d)のうち、
> **(1) GT点の情報を保持し、(2) 摂動に対して動画単位の長さ推定が安定する表現**はどれか。
> (3) 点GTとBBoxアノテーションの整合性は記述量として併記する。
> 外部的な正しさは判定しない。**差が事前規則の幅に収まれば「判定不能」とし、複数候補を後続へ渡す。**

H17-1a〜cを分けて測る。(e)は単位制約付きの参考値として同じ量を出すが、採否の対象にしない。

| ID | 問い | 主指標（提案） |
| --- | --- | --- |
| H17-1a 情報保持 | 表現がGT点をどれだけ覆うか、どれだけ膨らむか | 被覆率`C`＝表現内に入るGT陽性点の割合（frame単位→動画中央値→全体中央値）。膨張率`I`＝表現の面積／GT陽性点の占有pixel数（丸めたpixelの異なり数）。端点系(c)(d)は線分からの垂直距離が`w/2`以内の点を被覆とする |
| H17-1b 整合性（記述） | 点GT由来と`bbox_local_xyxy`由来が一致するか | frame単位のAABB IoU、長辺長の相対差。**採否には使わない**（同一注記系列のため） |
| H17-1c 摂動安定性 | 動画単位の長さ推定の変化 | 相対変化`|ΔL|/L`の動画中央値と90パーセンタイル。摂動は1.5.3 |

#### 1.5.3 摂動の固定（提案）

seedは固定（0,1,2の3反復）、種類と強さは以下に限る。結果を見て追加しない。

| 摂動 | 内容 |
| --- | --- |
| 間引き | 各frameのGT陽性点を一様に50%・75%保持 |
| FPノイズ | 各GT陽性frameに、そのframeのGT点数の10%・30%の点を、frameの有効点（`valid_mask`）から一様抽出して陽性に加える |
| 境界の揺らぎ | GT陽性点の`pixel_xy`に標準偏差1 local pixelのガウスノイズ |

FPノイズを有効点から抽出するのは、画像外の座標を作らないためである。
遠方FPの影響は実際の予測（H17-2）で測られるため、ここでは大きさの比較に限る。

#### 1.5.4 採否規則（提案）

旧規則の「臨床FL誤差最小の1つを選ぶ」を次の3段に置き換える。**閾値は提案値で、管理判断で確定する。**

1. **足切り（情報保持）:** `C`の全体中央値≥0.95。端点系は`I`を適用しない。箱系は`I`を記述量として報告する。
2. **足切り（安定性）:** 全摂動条件で`|ΔL|/L`の動画中央値≤5%、かつ90パーセンタイル≤15%。
3. **残った表現の扱い:**
   - 残った表現をすべて「候補」とし、最大3つまでS5-20へ渡す。3つを超える場合は、事前順位
     (c) → (d) → (b) → (a)（端点・長さを直接与え、Stage 6の軸・端点推定へ接続しやすい順）で上位3つを残す。
   - この順位は**正しさの順位ではない**。実データ結果を見る前に固定する選択の規約である。
   - 残った表現が0の場合: 「判定不能・候補なし」を記録し、進行可否を管理判断へ返す。
     足切りを緩めて再評価しない。

**「1案に絞る」ことを完了条件にしない。** 1案を選ぶ根拠は、外部参照が得られる後続
（A案、S5-20c）に委ねる。

#### 1.5.5 旧「T_FL以内の動画が過半」規則の扱い

| 要素 | 扱い |
| --- | --- |
| 何と比べるか | 旧: 予測由来代理FL対臨床FL → **新: 予測由来の長さ対同じ表現のGT由来の長さ**（H17-2、validation） |
| 許容幅 | mm/pixelが確定し、ユーザーが`T_FL`を設定した場合だけmm単位の`T_FL`を使う。それ以外は相対許容`τ_rel=10%`（提案値）を使い、**診断用であり`T_FL`ではない**と出力に明記する |
| 分母 | validation18の**全件**。予測なし・幾何失敗の動画は「許容外」に数える |
| 「過半」 | 維持（10/18以上）。ただし同一検査重複（D-041、F10）で実効標本数が18より小さく、偶然の変動を過小評価することを結果に併記する |
| 判定の意味 | 満たせば「segmentation＋後処理をS5-20のbaseline候補にする」。満たさなくても、S5-19はS5-17の結果によらず実施する（依頼書2章） |
| 主checkpoint | best（epoch6）で判定する。last（epoch50）は補助として同じ量を報告し、判定に使わない |

#### 1.5.6 限界として結果に付けるもの

- GT由来量同士の一致・安定性であり、臨床的な正しさや骨長の真値との一致を示さない。
- train_core144はファイル単位で、同一検査の兄弟ファイルを含む。動画別統計は検査単位のばらつきを過小評価する。
- validationは開発で方式選択済みの18件で、同一検査重複を含む。後処理の選択をvalidationで行うため、その値は楽観側に偏る。
- 単一の学習run・単一seedの予測である。

### 1.6 評価器の設計

#### 1.6.1 構成

```text
Stage5/stage5/geometry/            # numpyのみ。torch・h5py非依存
  __init__.py
  types.py          # CoordSpace, Unit, InstanceGeometry, FrameGeometry, VideoFLEstimate, FailureReason
  transform.py      # CropTransform（local↔元frame）, PixelToMM（元frame pixel↔mm）
  frame_geometry.py # 点集合→instance（連結成分）→PCA・端点・箱
  fl_estimate.py    # 表現(a)〜(e)と動画単位の集約
  postprocess.py    # P0〜P3
  matching.py       # 複数instanceの選択・matching
  prior.py          # train_core由来のprior-only
  metrics.py        # 幾何誤差・存在判定・FP距離・安定性
Stage5/evaluate_stage5_geometry.py  # 契約解決・許可リスト・H5/NPZ I/O・出力境界
Stage5/evaluate_stage5_geometry.sh
Stage5/checks/dummy/check_dummy_geometry_{core,transform,postprocess,cli_guard}.py/.sh
```

H5の読込と契約検査はCLI側だけに置く。`stage5/geometry/`は配列を受け取る純関数とし、S5-18〜S5-20の学習・評価から再利用できるようにする。
新しいmodel/head/loss、Stage 6本実装は作らない。

#### 1.6.2 座標空間と単位の型

| `CoordSpace` | 意味 | 生成 |
| --- | --- | --- |
| `LOCAL_CROP_PX` | `pixel_xy`そのもの | H5 |
| `RAW_FRAME_PX` | 元frame pixel | `CropTransform.local_to_raw` |
| `RAW_FRAME_NORM` | 元frameを`raw_width/height`で割った[0,1]² | S5-20 canonical候補 |
| `MM_XY` | 元frame基準mm | `PixelToMM`が確定した場合だけ |
| `PSEUDO3D` | `points`のXYZ | H5。単位は`pseudo3d_units`で、mmと混在させない |

すべての座標・長さに`coord_space`と`unit`を持たせ、異なる空間の値を演算すると例外にする。
未換算の値は`None`とし、0や他動画の値で埋めない。

**幾何の計算は`RAW_FRAME_PX`で行う。** cropモードでは等方scaleなのでlocalと相似だが、`resize`モードはx/y異方であり、
local空間でPCAすると軸・長さが歪むためである。mm確定後は`MM_XY`でPCAをやり直す（異方scaleではPCA結果が変わる）。

#### 1.6.3 データ型（案）

```python
@dataclass(frozen=True)
class InstanceGeometry:
    n_points: int
    center: tuple[float, float]
    axis_angle: float | None       # [0, π)。符号を持たない
    length: float | None           # 主軸方向の頑健な広がり（q=0.02〜0.98）
    width: float | None            # 副軸方向の頑健な広がり
    endpoints: tuple[tuple[float, float], tuple[float, float]] | None  # 正規順（下記）
    aabb: tuple[float, float, float, float]
    axis_ambiguous: bool           # λ2/λ1 > 0.9
    touches_crop_border: bool      # local cropの端2 pixel以内に点がある
    coord_space: CoordSpace
    unit: Unit

@dataclass(frozen=True)
class FrameGeometry:
    frame_order: int
    present: bool
    instances: tuple[InstanceGeometry, ...]   # 点数の降順
    status: str                     # ok / empty / insufficient_points / transform_unavailable

@dataclass(frozen=True)
class VideoFLEstimate:
    value: float | None
    unit: Unit
    coord_space: CoordSpace
    representation: str             # a_aabb / b_obb / c_axis / d_bestframe / e_pseudo3d
    aggregation: str                # q90 / best_frame
    instance_rule: str              # M1 / M2
    frames_used: tuple[int, ...]
    selection_score: float | None   # 最良frame選択の内部score。確率・信頼度ではない
    failure: FailureReason | None

class CropTransform:   # 1動画1個。local_preprocess_effectiveで分岐
    def local_to_raw(self, xy): ...
    def raw_to_local(self, xy): ...

class PixelToMM:       # mm_per_px_x, mm_per_px_y, 粒度（video/frame）, 供給元の出所hash
    def raw_px_to_mm(self, xy): ...
    def mm_to_raw_px(self, xy): ...
```

- **信頼度は置かない。** 最良frameの選択に使う量は`selection_score`と呼び、正解確率ではないと型の説明に書く。
- 端点の正規順: 主軸へ射影して値の小さい方を`e0`とし、`axis_angle`は[0,π)へ折り返す。比較時は端点の2通りの対応のうち誤差の小さい方を使う（交換不変）。
- 軸角度の誤差は`min(|θ1−θ2|, π−|θ1−θ2|)`（符号不変）。平均は2倍角の(cos, sin)で取る。

`CropTransform`の分岐（F3・F4・F6）:

| `local_preprocess_effective` | local→元frame | 扱い |
| --- | --- | --- |
| `offset_crop`、`resize_shorter_then_offset_crop` | `x=(u+left)/s`、`y=(v+top)/s` | 有効。`s`は有限・正 |
| `resize` | `x=u·raw_w/local_w`、`y=v·raw_h/local_h` | 有効（x/y異方） |
| `offset_crop_fallback_resize` | 同上の異方変換が正しい可能性が高い | **無効として除外**（F4によりGT BBox側の変換規約と一致しない可能性がある） |
| 上記以外・fieldの欠落・非有限 | — | 除外し、理由を記録 |

#### 1.6.4 候補表現と上限

| ID | 表現 | frame単位の長さ | 動画単位の集約 |
| --- | --- | --- | --- |
| (a) | frame軸平行BBox | 長辺 | 存在frameの長さの90パーセンタイル（`q90`） |
| (b) | oriented box | PCA主軸方向の頑健な広がり | `q90` |
| (c) | 頑健な主軸端点 | 端点間距離（(b)と同じ長さ定義に端点を付けたもの） | `q90` |
| (d) | 最良frame＋端点 | (c) | `selection_score`最大のframe（1.6.6） |
| (e) | pseudo-3D軸 | — | 動画全体のGT点のPCA主軸の頑健な広がり（`pseudo3d_units`、参考値） |

(b)と(c)の長さは同じ定義で、違いは端点を持つかどうかである。H17-1aの被覆判定（箱か線分か）と、H17-2の端点誤差の有無で差が出る。
長さの差が無いことを実データ前に確認しておくため、合成テストで一致を検査する。

GT表現は**5通りに限定**する。分位点（0.02/0.98、q90）、最小点数（5）、軸曖昧の閾値（0.9）は提案値で、結果を見て変えない。

#### 1.6.5 後処理と組合せの上限

| ID | 後処理 | 利用する情報 |
| --- | --- | --- |
| P0 | なし（`pred_label`） | 予測のみ |
| P1 | frameごとの最大連結成分 | 予測のみ。連結はlocal pixel距離4以内の単連結、成分の最小点数5 |
| P2 | frame間持続性: 前後2 frame以内に、中心距離がtrain_coreのGT長中央値の0.5倍以内の成分がある成分だけ残す | 予測＋train_core由来の定数 |
| P3 | 確率上位: frameごとに`prob_femur`上位`K`点（`K`＝train_coreのGT陽性frameあたり点数の中央値）を陽性とし、`prob_femur<0.5`でも採る | 予測＋train_core由来の定数 |
| P1+P2 | P1の後にP2 | 同上 |

threshold探索は行わない。P3の`K`とP2の距離はtrain_coreのGTだけから1回算出して固定する。

**比較順序と上限:**

1. train_coreでH17-1を判定し、候補表現を最大3つに絞る。
2. validationで、候補表現（≤3）×後処理（5）×checkpoint（best主・last補助の2）＝**最大30組**を1回の本実行で算出する。
3. 「最良後処理」は、bestの候補表現ごとに、1.5.5の許容内動画数が最大の後処理を選ぶ（同数なら単純な順P0→P1→P2→P1+P2→P3）。validationで選ぶため楽観的になることを明記する。

#### 1.6.6 最良frameの選択とoracleの分離

| 区分 | frame選択に使う情報 | 出力 |
| --- | --- | --- |
| 実用 | 予測由来の幾何と`prob_femur`だけ。`selection_score`＝主instanceの点数×平均`prob_femur`、`touches_crop_border`のframeは除く | 主表 |
| oracle | GTの最良frame（GT由来の`selection_score`最大）を予測に適用 | `oracle_*`列。実用性能に混ぜない |

GT側の(d)の`selection_score`は、主instanceの点数（GT陽性点数）とする。GTには確率が無いため、予測側と同じ式にしない。

#### 1.6.7 prior-only baseline（H17-4）

- 座標系: `RAW_FRAME_NORM`。local cropは動画ごとにcrop位置が異なり、動画間で揃わないため（F9）。
- 構築: train_core144のGT幾何だけから、動画単位の(中心、軸角度の2倍角平均、長さ)の平均と、正規化時刻10区間ごとのGT存在率を作る。
- 予測: validationの各動画に、同じ幾何（元frame寸法で逆正規化）を与え、存在率>0.5の区間のframeを「存在」とする。
- 自己除外: train_coreへ適用するin-sample表では、対象動画をvideo identityで除いた平均を使う。同一検査の兄弟ファイルは除かないため、in-sample値は楽観的になると明記する。
- 採否(3)の「差が小さい」: 候補表現×最良後処理の許容内動画数がprior-onlyより**3件以上多くない**場合、その表現を候補から外す（提案値）。

#### 1.6.8 FPの空間分類（H17-3）

予測陽性でGT陽性でない点を、次の排他的な区分に分ける。距離は`RAW_FRAME_PX`で測り、同じframeのGT陽性点までの最近傍距離とする。

| 区分 | 定義 |
| --- | --- |
| ignore上 | `point_label==-1`または`valid_mask==False`（valid GT上のFPに加えない） |
| GT近傍 | 同frameにGTがあり、距離≤そのframeのGT長×0.1 |
| GT同frame・遠方 | 同frameにGTがあり、上記より遠い |
| GT無しframe | 同frameにGT陽性点が無い |

「遠方」「GT無しframe」を**別の解剖構造とは呼ばない**。採否(4)は「遠方＋GT無しframeが予測陽性（valid上）の過半なら、S5-20でobjectness／contextの設計を必須とする」に修正する。
解剖学的な解釈には、S5-15の可視化のような別の根拠が要る。

#### 1.6.9 複数instance（H17-5）

GTも予測も、frame内の陽性点を1.6.5-P1と同じ規則で連結成分に分ける。

| 規則 | 内容 | 用途 |
| --- | --- | --- |
| M1 | 最大instance（点数最大）だけで幾何を作る | 動画単位の長さの主規則 |
| M2 | 全instanceを1つとして扱う | 比較 |
| M3 | 予測とGTのinstanceを中心距離でHungarian matching（ゲート＝GT長×0.5） | frame単位の存在・中心・端点誤差 |

H17-5の事前規則（提案）: GTの複数instance frameでM1とM2の長さの相対差が10%を超えるframeが、複数instance frameの半数を超えるなら、
「S5-20では複数instanceの扱いを明示的に設計する」と記録する。M1/M2のどちらが正しいかは判定しない。

#### 1.6.10 失敗・除外と分母

| 事象 | 扱い |
| --- | --- |
| 予測陽性が0のframe | そのframeは「予測なし」。存在判定ではFN |
| 動画全体で予測幾何が得られない | 動画単位の長さは失敗。分母に残し、許容外に数える |
| 点数不足・軸曖昧 | frameを`insufficient_points`／`axis_ambiguous`として記録。軸誤差は軸曖昧frameを除いて算出し、除いた件数を併記 |
| crop逆変換不能（1.6.3） | その動画を除外し、理由と件数を報告。H17-1では分母から除いた件数を明記し、H17-2では許容外として数える |
| 点対応の不一致（1.4.3） | 実行全体を停止 |

集計は、成功動画だけの値と失敗率を必ず並べて出す。pooledの値と動画別中央値・勝敗数も併記する（S5-16 10.0）。

#### 1.6.11 指標

| 区分 | 指標 | 単位 |
| --- | --- | --- |
| 存在判定 | frame単位precision／recall（M3のmatching後）、動画単位の検出有無 | 率 |
| 幾何誤差 | 中心誤差、端点誤差（交換不変）、軸角度誤差（符号不変）、長さの相対誤差、長さの誤差 | `RAW_FRAME_PX`、度、比。mm確定後に`MM_XY` |
| 動画単位 | 1.5.5の許容内動画数（/18）、失敗率、長さ相対誤差の中央値 | 件、率 |
| 頑健性 | H17-1cの`|ΔL|/L` | 比 |
| FP | 1.6.8の区分別割合 | 率 |

臨床MAE・Bland–Altman一致限界は作らない。mmを使う指標は、mm確定前は出力に含めない（列ごと出さない）。

### 1.7 mmとT_FL

#### 1.7.1 ユーザーへ必要な情報

1. **供給元と形式:** ファイル種別、列名、単位（mm/pixel）。x/y別か共通か。
2. **基準frame:** 復号したraw frameの寸法か、CVATでアノテーションした画像の寸法か（F5）。両者が同じか。
3. **突合キー:** 動画を一意に示すキー。Step 0の2系統の命名（timestamp形式の全trailing数値、case形式）と**完全一致**で突合できるか。
4. **粒度:** 動画内で一定か、frame別か。frame別なら、どのframe番号（raw frame番号、`frame_order`、stride後の番号）に対応するか。
5. **対象範囲:** train_core・validationのどれを覆うか。欠ける動画があるか。

#### 1.7.2 mm未確定でも進められる範囲

H17-1a〜cとH17-2〜H17-5は、`RAW_FRAME_PX`と相対量だけで実行できる。
長さの相対誤差・被覆率・安定性は等方scaleの差に依存しない。**ただし元frameのx/y mm/pixelが異なる場合、
pixelでの長さ・角度はmmでの値と一致しない。** このため、mm未確定時の結果には「x/y等方を仮定したpixel値」と明記し、
異方比`k=sy/sx`が`1±ε`のときの長さの変化を軸角度の関数として解析的に併記する（合成テストで検証）。

`T_FL`は改訂H17-1には不要である。H17-2のmm判定にだけ使い、未設定なら`τ_rel`で代替する（1.5.5）。推定・既定値の設定はしない。

#### 1.7.3 往復誤差の許容値（実行前に固定する提案）

| 変換 | 許容 |
| --- | --- |
| local→元frame→local（float64） | 最大絶対誤差≤1e-6 local pixel |
| 元frame pixel→mm→元frame pixel（x/y別） | 最大相対誤差≤1e-9 |
| 合成の異方scale（`sx≠sy`）での往復 | 同上。x/yを取り違えると失敗する値（例: `sx=0.1`、`sy=0.13`）で検査 |
| 実データの`pixel_xy`の範囲 | `0≤u<local_w`、`0≤v<local_h`（`local_input_shape`）。外れれば停止 |
| 逆変換後の元frame座標 | `-1≤x≤raw_w`、`-1≤y≤raw_h`（F6の丸め差を許す幅）。外れれば停止 |

これらは実装の算術誤差と契約違反を検出するための値で、resize補間（`INTER_AREA`）によるsub-pixelの位置ずれを較正するものではない。
その位置ずれ（最大で元frameの1 pixel程度、［推論］）は限界として記録する。

#### 1.7.4 実行を止める条件（mm関連）

- mm換算を指定したのに`PixelToMM`が得られない動画がある（0・他動画の値・1.0で埋めない）。
- キーの完全一致が得られない、または1キーに複数の値がある。
- frame別の値でframe対応が定義されていない。
- 1.7.3の往復誤差を超える。
- `PSEUDO3D`の量を`MM_XY`と演算しようとした。

### 1.8 実装・検証計画

#### 1.8.1 変更ファイル案

| 区分 | ファイル |
| --- | --- |
| 新規（幾何コア） | `Stage5/stage5/geometry/*.py`（1.6.1） |
| 新規（CLI） | `Stage5/evaluate_stage5_geometry.py`、`.sh` |
| 新規（合成テスト） | `Stage5/checks/dummy/check_dummy_geometry_{core,transform,postprocess,cli_guard}.py`・`.sh` |
| 更新（文書、実装後） | `docs/stage5/FILES.md`、`docs/stage5/data_construct.md`（出力構成・共有境界）、本報告書 |
| 変更しない | 既存の`split_contract.py`・`split_identity.py`・学習／評価コード・pins・Stage2to4 |

#### 1.8.2 契約と許可リスト（CLI）

実行順を固定する。**1〜5はH5・NPZ・CSVを開く前に行う。**

1. `resolve_active_contract(manifest_key="s5_16_bprime")`でpins→registry→manifestを解決（直接パス指定は不可）。
2. `--train_core_list`・`--validation_list`を読み、`list_identity_sha256`がmanifestの`lists.train_core`／`lists.validation`の`identity_sha256`と一致することを確認。
3. 処理目的ごとの許可リストを作る: `gt_prior`＝train_core、`prediction_diagnosis`＝validation、`in_sample_diagnosis`＝train_core。
4. 読む予定の全ファイル（teacher H5、中間H5、NPZ）について`contract.assert_paths_allowed`（封印拒否）を実行し、さらに各identityが目的の許可リストに**完全一致で**含まれることを確認。
5. NPZは許可リストから期待パスを組み立てて開く。**ディレクトリを列挙して見つかったものを読むことはしない**（directory-modeへのfallbackなし）。期待NPZが無ければ停止。
6. 複数動画を含む`h5_metrics.csv`は、読込前に同じディレクトリの`validation_files.txt`・`selected_train_files.txt`（パス一覧）の各行へ手順4を適用し、全行が許可リスト内であることを確認してから読む（M-3）。
7. 中間H5は記録attrまたは完全一致basenameで解決し、そのファイル名のidentityがteacher H5のidentityと完全一致することを確認（F2）。

identityの抽出は`split_identity.extract_video_identities`だけを使い、独自の正規表現・前方一致を使わない。

#### 1.8.3 既存部品の再利用

| 部品 | 再利用の仕方 |
| --- | --- |
| `split_contract.resolve_active_contract`、`ActiveContract.assert_paths_allowed` | そのまま。許可リスト照合はCLIで追加（F7） |
| `split_identity.{extract_video_identities,contains_video_identity,masked_shape}` | そのまま。privacy検査にも使う |
| `file_list_mode.{read_file_list,list_identity_sha256}` | そのまま |
| `h5_io.load_stage5_pointcloud_h5` | そのまま（`bbox_local_xyxy`・`frame_annotation`も返す） |
| `check_stage5_gt_component_count.component_sizes` | 関数importの候補（O(n²)、1frame 20000点上限）。CLI・`privacy_self_check`・`read_gt_points`は使わない（F8） |
| `check_stage5_xy_coordinate_provenance.parse_shape_string`、`read_intermediate_local_dimensions` | 関数import。後者は`local_preprocess`・`local_resized_h/w`・`num_frames`を返さないため、不足分はCLIで追加に読む。`resolve_intermediate_h5`は使わない（F2） |
| S5-14のprior関数 | 使わない（F9） |

`stage5/geometry/`がこれらのchecker（h5py依存）をimportしないようにし、依存はCLI側に閉じる。

#### 1.8.4 合成テスト案

本番pins・実データに依存せず、テスト内で合成した契約（空pins・合成registry・manifest）と合成H5/NPZを使う（Step 0 31.4の教訓）。
**成功経路を必ず通す**（Step 0 31.3の教訓）。

| ファイル | 主な検査 |
| --- | --- |
| `core` | 既知の線分・楕円・回転矩形からの長さ・幅・角度・端点が解析値と一致。符号・端点交換の不変性。退化（1点、同一点、共線、`λ2/λ1>0.9`）。空GT・空予測。(b)と(c)の長さ一致。M1/M2/M3（2 instance、matchingのゲート外）。単位・空間の混在で例外 |
| `transform` | 3モードの往復（1.7.3）。異方`PixelToMM`の往復とx/y取り違えの検出。`offset_crop_fallback_resize`・非有限scaleの除外。範囲外`pixel_xy`で停止。異方比による長さ変化の解析式 |
| `postprocess` | P1の最大成分、P2の孤立frame除去、P3の上位K（0.5未満も採る）。FP区分（ignore上／近傍／遠方／GT無しframe）の境界値。prior-onlyの自己除外 |
| `cli_guard` | 封印identityを含む入力が**存在しないパス**でも読込前に封印例外になる。許可リスト外（train_coreを予測診断へ渡す等）の拒否。list identityの不一致で停止。期待NPZの欠落で停止し、ディレクトリ内の別NPZを拾わない。`…_091`と`…_091_02`の併合が起きない。点数不一致・混同行列の不一致で停止。**完走してJSONを書く成功経路**と、共有出力に動画ID（両命名規則・第4セグメント）・絶対パス・動画別値が入らないこと |

#### 1.8.5 実機の手順とCPU量

依頼書7章の順（実装→合成テスト→結果報告→実機fail-fast→限定CPU診断→管理受入）に従う。

| 段階 | 読むもの | 出力 | 回数 |
| --- | --- | --- | --- |
| 合成テスト | 合成データのみ | 合否 | 必要に応じて（実機で実行、1.9.1） |
| 実機fail-fast（入力監査） | 契約・リスト、162件のteacher H5のメタ（点数・dataset存否・frame範囲）、中間H5のattrs、NPZのキー・形状、`summary.json` | 件数・方法別件数・合否のみ（GT統計なし） | 1回。本実行とは別コマンドで、読むのはメタだけ |
| 本実行 | train_core144のGT（H17-1、prior）、validation18のGT＋NPZ（best・last）、sanity3のNPZ（in-sample） | 1.8.7 | **1回** |
| 不具合修正後の再実行 | 同上 | 同上 | **1回まで** |

本実行はfail-fastの全検査を冒頭でもう一度行う（読取りの重複はメタのみ）。
計算量の見積り［推論］: 1動画あたりGTの幾何抽出はframe数×連結成分計算で、摂動は3種×2強度×3 seedの18通り。
CPUで数十分以内を想定するが、実機で計測して報告する。**結果を見て条件を変えて再実行しない。**

停止条件（抜粋）: 1.4.3の点対応、1.7.4のmm関連、`frame_order`が中間H5の`num_frames`以上、`pixel_xy`の範囲外、
list identityの不一致、許可リスト外・封印の検出、期待ファイルの欠落、NaN/Inf。

#### 1.8.6 privacy

- private（領域Aに準じる新規ディレクトリ）: 動画別の幾何・長さ・端点・FP区分、alias対応表。
- 共有: 集計値（件数、中央値、分位点、率）と照合情報（件数・hashの先頭）だけ。
- 共有出力は書出し前に`contains_video_identity`と絶対パスの検査を通し、失敗すれば書かない。
- stdout・例外メッセージは位置と件数だけを出す（`split_contract`の既存規約）。図は作らない。

#### 1.8.7 共有出力の例（形式のみ。値は架空）

```json
{
  "schema": "stage5_s5_17_geometry_summary_v1",
  "contract": {"manifest": "s5_16_bprime", "train_core_identity16": "fa8429e30b7c8724",
               "validation_identity16": "de3f3d513fc0bf41"},
  "coord_space": "RAW_FRAME_PX", "mm_available": false,
  "h17_1": {"n_videos": 144, "n_excluded": {"transform_unavailable": 0},
            "representations": {"c_axis": {"coverage_median": 0.0, "stability_median_rel": 0.0,
                                           "stability_p90_rel": 0.0, "admitted": null}}},
  "h17_2": {"checkpoint": "best_epoch6", "tolerance": {"kind": "tau_rel", "value": 0.10,
            "note": "diagnostic, not T_FL"}, "within_tolerance": "k/18", "failure_rate": 0.0},
  "limits": ["GT-derived quantities only; no clinical reference", "file-level units include same-exam siblings"]
}
```

### 1.9 判断依頼

Step 0で決定済みの事項（分割、封印、class weight、5例の用途制限、抽選単位）は再質問しない。

| # | 判断事項 | 推奨 | 代替案と差分 |
| --- | --- | --- | --- |
| M-1 | 改訂H17-1（1.5.2）と採否規則（1.5.4）。閾値`C≥0.95`、`|ΔL|/L`中央値≤5%・P90≤15%、最大3候補、事前順位(c)(d)(b)(a) | 推奨案で確定 | (i) 足切りを安定性だけにする: 箱系の膨張を見なくなる。(ii) 事前順位を置かず全通過表現を渡す: S5-20の候補が3を超え得る |
| M-2 | 旧「T_FL以内が過半」の扱い（1.5.5）。mm未確定時は`τ_rel=10%`、分母は18全件、判定はbest | 推奨案で確定 | (i) mm確定までH17-2の判定を保留し記述だけ出す: 判定は遅れるが相対許容の恣意性を避けられる |
| M-3 | 点対応を`h5_metrics.csv`との混同行列一致で確定すること、その読込前にパス一覧で許可リストを確認する手順（1.4.3、1.8.2-6） | 推奨案で確定 | (i) 点数一致だけで進める: checkpointとの紐付けが保証されない（F1）。(ii) coverage記録を新たに作る: 既存保存物の由来確認が先に要り、手間が大きい |
| M-4 | 候補5表現×後処理5種、validationの最大30組、比較順序（1.6.4・1.6.5） | 推奨案で確定 | (i) P3を外す: 確率をargmax以外で使う唯一の経路が無くなる。(ii) lastを算出しない: 過学習後の幾何劣化が見えなくなる |
| M-5 | prior-onlyを元frame正規化座標で作り、差の基準を「許容内動画数+3件」とする（1.6.7） | 推奨案で確定 | (i) local crop座標で作る: S5-14と揃うが動画間で座標が揃わない |
| M-6 | `offset_crop_fallback_resize`の動画を変換不能として除外（1.6.3、F4） | 推奨案で確定。件数は実機fail-fastで報告 | (i) 異方変換で扱う: GT BBox側の変換（NaN→1.0）と規約が食い違う可能性が残る |
| M-7 | 1.4.4の所在確認（R1〜R5）をユーザーへ依頼すること | 依頼する | — |
| M-8 | F10の文書不整合の訂正と、validation側の同一検査件数（共有可能な1値）の取得 | 総括管理で扱う | 本チャットで文書を直すことはしない |
| M-9 | 実施範囲: 本報告の承認後、実装→合成テスト→報告までを進め、実機fail-fast以降は結果報告後に改めて承認を受ける | 推奨案で確定 | (i) 実機fail-fastまでを一括承認: 往復が1回減るが、合成テストの結果を管理が見る前に実機を読むことになる |

#### 1.9.1 実行環境の制約

本開発コンテナにはnumpy・h5pyが無く、合成テストも実行できない。Step 0と同様に、合成テストと実機工程はユーザーの実機
（`/mnt/data/3d_projects/models/Stage5`、既存`.sh`の`PYTHON`）で実行してもらう想定である。
本チャットは構文と静的な整合の確認までに留め、実行結果を受け取って報告する。
コンテナ内で合成テストを実行したい場合は、numpy/h5pyの導入可否を別途判断してほしい。

### 1.10 未確認・未解決の事項（区別して記録）

| 区分 | 事項 |
| --- | --- |
| 未確認（実機） | 予測NPZの所在・件数・checkpoint（R1・R2）、中間H5の解決方法の内訳（R3）、validation18の中間H5（R4）、`local_preprocess_effective`別の件数（F3） |
| 未確定（ユーザー設定） | mm/pixelの供給元・形式・基準frame・キー・粒度（1.7.1）、`T_FL` |
| 管理判断待ち | M-1〜M-9 |
| 文書の不整合 | F10（validation側の同一検査件数の記載） |
| 潜在的なteacher側の論点 | F4（該当動画が無ければ消える） |
| 持越し（S5-17の範囲外） | 5例の比較単位（D-040）。S5-17の入力・方式選択・較正に使わない |

本章の作成は提案であり、H17-1の再設計・mm評価・実行条件の未決事項が解決したことを意味しない。

---

## 2. 管理評価への対応と修正版提案（2026-09-27）

担当: S5-17実装チャット。状態: **修正版提案。管理判断待ち。**
本章の作業は文書とコードの読取りだけである。実データ・`/mnt`・封印物は読んでおらず、実装・合成テストも行っていない。

### 2.1 管理評価の記録

総括管理は1章を確認し、**M-1〜M-9の一括承認を見送った。** 評価の要旨は次のとおり。

| 区分 | 内容 |
| --- | --- |
| 採用できる方向性 | H17-1を情報保持・整合性・摂動安定性へ改めること。幾何コアをnumpyのみとし、I/O・封印guardをCLIへ分離すること。中間H5の前方一致を廃止し、identityを完全一致で照合すること。mm未確定の診断とmm評価、oracleと実用処理を分けること。bestを主・lastを補助とし、比較数を事前に制限すること |
| 実装前の修正要求 | M-3（点対応の確定）、1.5.4（最終選択を最終testへ委ねる記述）、M-1（被覆率・膨張率の単位と定義）、1.5.3（FP摂動の抽出元と件数）、存在判定・FP指標の分母、M-2・M-5（提案値の位置づけ）、M-6（除外後の判定不能条件） |
| 実装上の補足要求 | 既知座標の期待値照合、`pred_label`と確率の照合規約、CSVの読込前coverageと読込後の行identity照合、生配列検査と既存loaderの整合 |
| F10 | 指摘どおりの誤記。管理側で訂正済み（2.12） |
| F2〜F4 | Step 0の受入を取り消すものではない。取得可否監査と、S5-17で必要な数値的な変換検証を分け、追加監査へ引き継ぐ（2.11） |

### 2.2 1章からの変更一覧

| 1章の箇所 | 2章での扱い | 置換先 |
| --- | --- | --- |
| 1.4.3（点対応の検査）、M-3 | 「確定」を撤回し、整合性検査に限定。来歴と組み合わせる | 2.3、2.10.2、2.10.3 |
| 1.5.4（採否規則）の「1案を選ぶ根拠はA案・S5-20cに委ねる」 | **撤回。** 方式選択を開発側で完結させる | 2.4 |
| 1.5.2 H17-1aの被覆率・膨張率 | 単位を元frame pixelに統一し、定義をやり直す | 2.5 |
| 1.5.3（摂動） | FPの抽出元を有効な背景点に限定。件数を15通りに訂正 | 2.6 |
| 1.6.8、1.6.10、1.6.11（存在判定・FP） | frame存在判定、instance検出、点単位FPを分離し、分母を明示 | 2.7 |
| 1.5.5、1.6.7、1.6.9、1.6.8の閾値 | 診断用提案値として記述的比較に使う。候補除外に使わない | 2.8 |
| 1.6.3のfallback resize除外（M-6） | 保守的な扱いとして維持し、評価可能件数不足の判定不能条件を追加 | 2.9 |
| 1.7.3（往復誤差） | 既知座標の期待値照合を追加 | 2.10.1 |
| 1.4.3-3（`pred_label`と`prob≥0.5`の一致） | argmaxの同値処理とfloat32の丸めを踏まえた規約へ | 2.10.2 |
| 1.8.2-6（CSVの隣のリストで確認） | 読込前の根拠を来歴に置き、読込後の行identity照合を追加 | 2.10.3 |
| 1.8.3の`h5_io.load_stage5_pointcloud_h5`「そのまま」 | **撤回。** 生配列を検証してから明示的に変換する読込関数をCLIに置く | 2.10.4 |
| 1.3 F2〜F4 | 取得可否監査（Step 0、受入済み）と数値的変換検証（S5-17）を分ける | 2.11 |
| 1.3 F10、1.5.6 | 管理側の訂正を反映 | 2.12 |
| 1.9 判断依頼 | 修正版に置き換え | 2.13 |

1章のその他の記述（1.1〜1.3の監査記録、1.6.1〜1.6.6の構成・型・候補・後処理・oracle分離、1.7.1〜1.7.2・1.7.4、1.8の手順）は維持する。

### 2.3 点対応: 混同行列一致は整合性検査に限る（M-3の修正）

**混同行列の一致は点対応を確定しない。** 同じ動画の中で点の対応がずれていても、TP/FP/TN/FNの合計が一致する場合がある。
したがって「点対応」は、次の来歴の組合せで**根拠づける**ものとし、混同行列の一致はそれと矛盾しないことを確かめる**整合性検査**とする。
来歴のどれかが確認できない場合は、その旨を限界として記録する。確認できない来歴を整合性検査で埋め合わせない。

| # | 根拠 | 確認方法 | 確認できる範囲 |
| --- | --- | --- | --- |
| P1 | 保存時の点順規約 | `evaluate_stage5.py`は`load_stage5_pointcloud_h5(path)`で読んだH5の行順のまま全点を推論し、`point_indices=arange(N)`で保存する（`evaluate_stage5.py:444-489`） | コード上の規約。S5-15評価を実行した時点のコミットで同じであることは、runの記録するgit revisionで確認する（R6） |
| P2 | 対象H5 | NPZのファイル名identity＝H5のidentity（完全一致）。同じ評価ディレクトリの`summary.json`の`validation_files`・`selected_train_files`に、そのH5パスがあること（読込前後の扱いは2.10.3） | 評価時に**そのパスのH5**を読んだこと |
| P3 | H5が評価後に変わっていないこと | 評価時点の各H5のcontent hashが記録されていれば照合する（R6）。記録が無ければ、teacher v7の受入以降にH5を再生成していないという運用記録と、Step 0 preflightの集合単位の期待値との整合に留まる | 記録の有無による。**無い場合は限界として明記** |
| P4 | run/checkpoint | `summary.json`の`checkpoint`（パス）・`checkpoint_epoch`がbest（6）／last（50）と一致すること。checkpointのcontent hashが評価記録またはrun manifestにあれば照合する（R6） | checkpointのパスとepoch。内容の同一性はhash記録がある場合だけ |
| C1 | 整合性検査: 点数 | NPZの全配列長＝H5の点数、`point_indices`＝`arange(N)` | 必要条件 |
| C2 | 整合性検査: 予測の内部整合 | 2.10.2 | 必要条件 |
| C3 | 整合性検査: 混同行列 | NPZ＋H5から再計算した動画別TP/FP/TN/FNが、同じcheckpointディレクトリの`h5_metrics.csv`の行と整数で一致 | 必要条件。**点対応の確定ではない** |

C1〜C3のいずれかが不一致なら停止する（不一致は来歴のどこかが誤っていることを示すため）。
P3・P4のhash記録が無い場合でも、実行するかどうかは管理判断とする（M-C）。

### 2.4 方式選択を開発側で完結させる（1.5.4の修正）

1.5.4の「1案を選ぶ根拠は、外部参照が得られる後続（A案、S5-20c）に委ねる」を撤回する。
最終test（S5-20cのinternal_test、A案）は**選定済みの方式を評価するためのもの**であり、その結果を見て方式を選ばない。

修正版の流れ（提案）:

1. **S5-17（train_core、H17-1）:** 事前規則（2.8で位置づけを確認するH17-1の足切り）を通過した表現を「候補」とする。
   候補の中から**主候補を1つ**、事前に固定した規則で決める: 摂動安定性の90パーセンタイル（2.6の5区分の最大値）が最小の表現。
   差が1パーセンタイルポイント以内なら事前順位(c)→(d)→(b)→(a)。残りの候補（最大2つ）は副候補とする。
2. **S5-17（validation、H17-2）:** 主候補・副候補のそれぞれについて予測由来の幾何を記述的に報告する。
   validationの結果で主候補を入れ替えない。
3. **S5-20a/b（開発側）:** 副候補を残す場合、その比較は**S5-20開始前に比較規則を固定した上で**validationで行う。
   規則はS5-20の依頼書で確定する（本報告では決めない）。
4. **S5-20c:** 3.までに固定した**1つの構成だけ**をinternal_testで1回評価する。結果を見て選び直さない。

| 事象 | 扱い |
| --- | --- |
| 候補が1つ | それを主候補とする |
| 候補が複数 | 1.の規則で主候補を決め、副候補は3.の事前固定比較へ |
| 候補が0、または2.9の判定不能 | 「判定不能」を記録し、S5-20へ進むかどうかと進め方を管理判断へ返す。足切りを緩めて再評価しない。**最終testに判断を委ねない** |

### 2.5 H17-1aの指標の再定義（M-1の修正）

**単位:** 幾何・面積はすべて`RAW_FRAME_PX`（面積はraw px²）で計算する。local pixel単位の量と比べない。

**GT点はマスクではない。** 教師点群はサンプリングされた点であり、点数や占有pixel数はマスク面積の推定量ではない。
1章の膨張率`I`（表現の面積／GT点の占有pixel数）は撤回する。

| 指標 | 定義 | 用途 |
| --- | --- | --- |
| 保持被覆率`C_ho` | frameごとに、GT陽性点を固定seedで半分ずつA/Bに分け、Aから表現を作り、Bのうち表現の領域に入る割合。seedは0,1,2。frame→動画中央値→全体中央値 | 主指標。同じ点から作った表現が同じ点を覆う割合は分位点の選び方で決まり、表現の比較にならないため、保持側で測る |
| 相対膨張率`I_hull` | 表現の面積／同じinstanceのGT陽性点の凸包面積（ともにraw px²）。凸包が定義できないframe（非共線の3点未満）は除き、件数を記録 | 箱系(a)(b)の記述量。凸包も点から作る量であり、マスク面積の代わりではない |

各表現の「領域」を明示する。

| 表現 | 被覆判定に使う領域 |
| --- | --- |
| (a) | 軸平行矩形`[x_min,x_max]×[y_min,y_max]` |
| (b) | 主軸方向の範囲`[t_0.02, t_0.98]`×副軸方向の範囲`[s_0.02, s_0.98]`の回転矩形 |
| (c) | 端点`e0,e1`を結ぶ線分について、**主軸方向の射影が`[t(e0), t(e1)]`に入り、かつ副軸方向の距離が`w/2`以内**の領域（長さ方向の範囲を含む） |
| (d) | (c)と同じ領域を、選択した1 frameだけで評価 |

(c)の領域は(b)の回転矩形と一致するため、**`C_ho`では(b)と(c)を区別できない。** (c)の追加の情報は端点であり、その評価はH17-2（端点誤差）で行う。
これを限界として結果に明記する。

### 2.6 摂動の再定義（1.5.3の修正）

| 摂動 | 強度 | 定義 |
| --- | --- | --- |
| 間引き | 50%、75%保持 | frameごとにGT陽性点`n`点から`max(ceil(r·n), min(n,5))`点を非復元抽出で残す |
| FPノイズ | GT点数の10%、30% | 同じframeの**有効な背景点**（`valid_mask==True`かつ`point_label==0`。GT陽性点・ignore点を含まない）から`m=round(ρ·n)`点を非復元抽出し、陽性に付け替える。既存の点を付け替えるため重複は生じない。候補が`m`点未満なら候補すべてを付け替え、不足として記録 |
| 境界の揺らぎ | σ=1 local pixel | GT陽性点の`pixel_xy`にガウスノイズを加えてから元frameへ変換。範囲検査は元データにだけ適用し、揺らぎ後の座標は切り詰めない |

各強度をseed 0,1,2で3反復する。**件数は間引き2×3＝6、FPノイズ2×3＝6、揺らぎ1×3＝3の計15通り**（1章の「18通り」は誤り）。

FPノイズについて、条件ごとに**要求した付け替え点数・実際に付け替えた点数・不足が生じたframe数**を集計して報告する。
安定性の集計は、5区分（間引き2、FP2、揺らぎ1）ごとに動画単位の`|ΔL|/L`を3 seedの中央値にまとめ、動画間の中央値と90パーセンタイルを出す。

### 2.7 存在判定・検出・FP指標の分離

3つを別の指標として出し、混ぜない。

**(1) frame存在判定。** 対象frameは、H5に点が1点以上あるframe（`frame_order`の異なり値）。
GTの存在＝最小点数5以上のinstanceが1つ以上。予測の存在＝後処理後に同じ条件を満たす。

| GT | 予測 | 区分 |
| --- | --- | --- |
| あり | あり | TP |
| あり | なし | FN |
| なし | あり | FP |
| なし | なし | **TN** |

1章の「予測なしframeはFN」は、GTがある場合に限る。

**(2) instance検出（M3のmatching後）。** GTか予測の少なくとも一方があるframeで、matchしたGT–予測の組をTP、matchしない予測instanceをFP、matchしないGT instanceをFNとする。
中心・端点・軸・長さの誤差はmatchした組だけで算出し、組数を併記する。frame存在判定の値とは別表にする。

**(3) 点単位のFP空間分類。** 分母を列名に含めて出す。

| 記号 | 定義 |
| --- | --- |
| `N_pred_all` | 予測陽性の全点（ignore・無効点を含む） |
| `N_pred_ignore` | そのうち`point_label==-1`または`valid_mask==False` |
| `N_pred_valid` | `N_pred_all − N_pred_ignore` |
| `N_FP` | `valid_mask==True`かつ`point_label==0`かつ予測陽性 |

- ignore上の割合は`N_pred_ignore / N_pred_all`。
- 距離区分（GT近傍／GT同frame・遠方／GT無しframe）は`N_FP`を排他的に分ける。割合は**`N_FP`を主の分母**とし、`N_pred_valid`を分母とする値を副として併記する。
- 1章の採否(4)「予測陽性の過半」は、2.8により記述量へ変更する。

### 2.8 提案値の位置づけ（M-2・M-5ほか）

1章の閾値はすべて**診断用の提案値**であり、臨床的妥当性や統計的優越性の基準ではない。用途を次のように分ける。

| 提案値 | 1章の用途 | 修正後の用途 |
| --- | --- | --- |
| `τ_rel=10%`、「過半（10/18）」（1.5.5、M-2） | H17-2の判定 | **記述的比較のみ。** 許容内動画数を`k/18`として報告し、候補除外やbaseline候補の判定に使わない |
| prior-onlyとの差「+3件」（1.6.7、M-5） | 表現を候補から外す | **記述的比較のみ。** prior-onlyとの差（件数、動画別paired差の中央値・勝敗）を報告し、除外しない |
| H17-5の10%・半数（1.6.9） | 複数instance設計の要否 | 記述的比較のみ |
| FP遠方の過半（旧採否(4)） | objectness設計の要否 | 記述的比較のみ |
| FP近傍の`0.1·L`、P2の`0.5·`GT長中央値、連結距離4 pixel、最小点数5、分位点0.02/0.98・q90、軸曖昧0.9 | 指標・後処理の定義 | 定義上の定数として事前固定（結果を見て変えない）。判定基準ではない |
| H17-1の足切り（保持被覆率`C_ho`中央値≥0.95、安定性の中央値≤5%・90パーセンタイル≤15%） | 候補の絞込み | **管理判断（M-A）。** 推奨は、S5-20へ渡す候補を最大3つに絞るための事前固定の規約として使うこと。ただし臨床的な基準ではないと明記する |

候補除外まで行う根拠が必要なもの（上表で「記述的比較のみ」としたもの）は、S5-17の結果を受けて管理が別途判断する。
S5-17の実装では、それらを判定値として出力せず、記述量と分母だけを出す。

### 2.9 評価可能件数が不足した場合の判定不能（M-6への追加）

`offset_crop_fallback_resize`・非有限scale等の除外（1.6.3）は、当面の保守的な扱いとして維持する。
除外後の少数例だけで採否を決めないため、次の条件を事前に固定する（数値は提案値）。

| 対象 | 判定不能とする条件 |
| --- | --- |
| H17-1（train_core） | 評価可能な動画が130件未満（144件の約90%）。または、除外が特定の`local_preprocess_effective`に集中し、そのモードの動画が1件も評価できない場合で、そのモードが144件の10%以上を占める |
| H17-2（validation） | 評価可能な動画が16件未満（18件中）。この場合、許容内動画数等の記述量も「評価可能n件中」として出し、18件の値として扱わない |

判定不能の場合は2.4の「候補が0」と同じ扱いにする。除外件数は理由別・`local_preprocess_effective`別の件数（共有可能な集計）で報告する。

### 2.10 実装上の補足

#### 2.10.1 既知座標の期待値照合

往復だけでは、順変換と逆変換が同じ誤りをした場合に検出できない。合成テストでは、手計算した期待値と照合する。
期待値はテストに定数で書き、実装のコードから再計算しない。

| モード | 入力attrs | 期待値 |
| --- | --- | --- |
| `offset_crop` | raw 800×600、`left=272`、`top=172`、`s=1` | local(0,0)→raw(272,172)。local(255.5,10)→raw(527.5,182) |
| `resize_shorter_then_offset_crop` | raw 640×480、`s=448/480`、`left=170`、`top=0` | local(0,0)→raw(182.142857…,0)。local(56,112)→raw(242.142857…,120) |
| `resize` | raw 640×480、local 256×256 | local(128,64)→raw(320,120)（x/y異方） |
| `PixelToMM` | `mm_per_px_x=0.1`、`mm_per_px_y=0.13` | raw(100,200)→(10.0,26.0) mm。x/yを取り違えると(13.0,20.0)になり検出される |

加えて、Stage 4がGT BBoxに使う順変換の式（`local=raw·s−left`、`annotate_pseudo3d_point_cloud.py:495-503`）を**テスト内に別の式として書き**、
逆変換の結果に適用して元のlocal座標へ戻ることを確認する（GT側の規約との整合）。

実データでは、中間H5またはteacher H5にBBoxの元frame座標が保存されていれば（`LocalBBox.raw_xyxy`相当。保存の有無は未確認）、
切り詰められていないBBoxについて逆変換の結果と照合する。保存されていなければ、この照合はできないと記録する（実機fail-fastで存否を確認）。

#### 2.10.2 `pred_label`と確率の照合規約

評価コードは2クラスの平均確率をfloat64で求めてfloat32へ丸め、`argmax`で`pred_label`を決める（同値はindex 0＝background）。
保存されるのはpositive列`p1`（float32）だけで、background列`p0`の丸め後の値は保存されない（`evaluate_stage5.py:333-337`）。
このため`p1`だけからは、0.5付近の`pred_label`を再現できない。

照合規約（提案）: `δ=1e-6`として、

- `p1 > 0.5+δ`なら`pred_label==1`、`p1 < 0.5−δ`なら`pred_label==0`でなければ停止する。
- `|p1−0.5| ≤ δ`の点はどちらでも許し、件数を記録する。
- `p1`が非有限または`[0,1]`の外なら停止する。

`δ`は、float32の0.5付近の刻み（約6e-8）とwindow間平均での和の丸めを覆う幅として選んだ値で、較正ではない。

#### 2.10.3 複数動画を含む保存物（`summary.json`、`h5_metrics.csv`）のcoverage

CSVの隣にあるリストは、CSVの収録対象を保証しない。読込前の根拠と読込後の照合を分けて明示する。

| 段階 | 内容 |
| --- | --- |
| 読込前の根拠 | **来歴**: 両ファイルは`evaluate_stage5.py`の1回の実行が、同じ実行の入力リストについてだけ書く（`summary.json`は`selected_train_files`・`validation_files`、`h5_metrics.csv`は同じ2リストの各動画の行。`evaluate_stage5.py:441-546`）。この評価runはS5-15の固定21動画（sanity3＋validation18）で実行されたと記録されている（評価レポート9.10.1）。したがって封印18件を含まないことの根拠は、コードの書出し規約とrunの記録であり、ファイル名ではない |
| 読込後の照合 | `summary.json`の2リストの全パスについて、identityが許可リストに完全一致で含まれ、集合がsanity3＋validation18と**過不足なく一致**すること。`h5_metrics.csv`の全行の`h5_path`・`video_name`について同じ照合を行い、行集合がcheckpointごとにsanity3＋validation18と一致すること |
| 不一致時 | 実行全体を停止する。不一致の行の内容（値・ID・パス）は出力・ログに書かず、件数と位置だけを出す |

この手順では、来歴の記録が誤っていた場合、照合前に封印動画の行がメモリへ読み込まれ得る。その行は出力されないが、
「読込前に拒否する」保証はここでは成り立たない。これを限界として明記する。
代替として、`h5_metrics.csv`を読まずに整合性検査C3を省く案がある（M-C）。

#### 2.10.4 生配列の検査と読込関数

`h5_io.load_stage5_pointcloud_h5`は読込時に型変換する（`point_label`→int64、`valid_mask`→bool、`pixel_xy`・`points`→float32）。
このため、非整数のlabel、0/1以外の`valid_mask`、float64の`pixel_xy`の精度低下を検出できない。
1章の「そのまま再利用」を撤回し、CLIに次の読込関数を置く。

1. h5pyで生配列を読み、dtype・形状・点数の一致を確認する。
2. 値域を検査する: `point_label∈{-1,0,1}`かつ整数値、`valid_mask`はboolまたは`{0,1}`、`frame_order`は有限・整数値・非負、`pixel_xy`は有限、`prob_femur`は2.10.2。
3. 検査に合格した後で、float64／int64／boolへ明示的に変換する。

`frame_annotation`（`bbox_local_xyxy`）も同じ手順で読む。既存loaderは変更しない。

### 2.11 F2〜F4の位置づけ

F2〜F4は**Step 0の受入を取り消すものではない。** 次の2つを分けて扱う。

| 区分 | 担当 | 内容 | 状態 |
| --- | --- | --- | --- |
| 取得可否監査 | Step 0（S0-6） | crop逆変換に必要なfieldが144/144で読めること、spacingが既定値であること | 受入済み。変更しない |
| 数値的変換検証 | S5-17（実機fail-fastの追加監査） | 中間H5の解決方法（記録attr・完全一致basenameだけを許可し、方法別件数を出す）、`local_preprocess_effective`別件数、有限・正のscale、`pixel_xy`と逆変換後座標の範囲、2.10.1の実データ照合（可能な場合） | 本章で提案。実装後の実機fail-fastで実施 |

F4（Stage 4のBBox変換が要求モードで分岐し、NaN scaleを1.0に置き換える点）は、該当動画の件数を実機fail-fastで報告するまでに留める。
S5-17では修正しない。該当が1件以上あれば、Stage 4側の扱いは管理の別判断とする。

### 2.12 F10の訂正の反映

管理側で、Step 0報告38章、S5-17依頼書v2の4.3、評価レポート9.12、S5-15報告10.1、D-041が訂正された。
正しい記録は「**旧train162とvalidation18にまたがる同一検査は4グループで、所属ファイルは両集合の合計17件。validation側の正確な件数は未確認（既存記録上の範囲は4〜13件）**」である。

S5-17の結果に付ける限界（1.5.6）の表現はこれに合わせる。validation側の件数を推測で補わない。
これは旧分割についての記録であり、新train_coreとvalidationの間の重複件数を確定したものでもない。

### 2.13 修正後の判断依頼

1章のM-1〜M-9を次のとおり置き換える。

| # | 判断事項 | 推奨 | 代替案と差分 |
| --- | --- | --- | --- |
| M-A | H17-1の指標（2.5）と摂動（2.6）の定義。H17-1の足切りを、S5-20へ渡す候補を最大3つに絞る事前固定の規約として使うか（2.8） | 定義を確定し、足切りを規約として使う | (i) 足切りも記述のみとする: 候補数が3を超え得るため、S5-20a前に別の絞込み規則が要る |
| M-B | 開発側で完結する方式選択（2.4）。主候補の規則と、副候補を事前固定比較としてS5-20a/bへ渡す扱い | 推奨案で確定 | (i) 副候補を持たず主候補だけを渡す: 単純だが、H17-1がGT由来量だけで決まるため選択の根拠が弱い |
| M-C | 点対応を来歴P1〜P4で根拠づけ、C1〜C3を整合性検査とする（2.3）。`h5_metrics.csv`の扱い（2.10.3）。P3・P4のhash記録が無い場合に実行するか | C3を含めて実施し、hash記録が無い場合は限界を明記して実行する | (i) `h5_metrics.csv`を読まずC3を省く: 読込前拒否の限界は消えるが、checkpointとの整合性検査が1つ減る。(ii) hash記録が無ければ実行しない: 予測診断ができなくなる |
| M-D | M-2・M-5ほかの提案値を記述的比較だけに使い、S5-17では候補除外に使わない（2.8） | 推奨案で確定 | — |
| M-E | 評価可能件数不足の判定不能条件（2.9）。train_core 130件、validation 16件、モード10% | 推奨案で確定 | (i) 件数だけで判定しモード条件を外す: 除外の偏りを見落とし得る |
| M-F | `pred_label`と確率の照合規約（2.10.2）。`δ=1e-6` | 推奨案で確定 | — |
| M-G | 生配列を検査する読込関数をCLIに置き、既存loaderは変更しない（2.10.4） | 推奨案で確定 | — |

維持する事項: 1章のM-4（候補5表現×後処理5種、validation最大30組）、M-6（fallback resizeの除外。2.9を追加）、M-9（承認後は実装→合成テスト→報告まで。実機fail-fast以降は改めて承認）。
M-7（所在確認の依頼）には次を追加する。M-8はF10の訂正が管理側で完了したため閉じる（validation側の件数は未確認のまま）。

| # | 確認事項 | 求める回答の形 |
| --- | --- | --- |
| R6 | S5-15長期runの評価時点のgit revision、checkpointのcontent hash、teacher H5の動画別content hashが、run manifest・評価出力のどこかに記録されているか（2.3のP1・P3・P4） | 記録の有無と記録先の種類（ファイル名は不要） |

### 2.14 本章で変えていないこと

- 分割・封印・class weight・5例の用途制限・ファイル単位の抽選（Step 0決定事項）。
- 幾何コアの構成、座標空間の型、候補表現(a)〜(e)、後処理P0〜P3、oracleと実用の分離、best主・last補助。
- 実機の手順（本実行1回、不具合修正後の再実行1回まで）と、学習・GPU・再推論・production変更・封印解除を行わないこと。

本章は修正版の提案であり、実行条件の確定や実装開始の承認を意味しない。

---

## 3. 2章への管理評価と再修正（2026-09-27）

担当: S5-17実装チャット。状態: **再修正版提案。M-A・M-C・M-Eについて管理判断待ち。**
本章の作業は文書の修正だけである。実データ・`/mnt`・封印物は読んでおらず、実装・合成テストも行っていない。

### 3.1 管理評価の記録

| 区分 | 内容 |
| --- | --- |
| 改善を確認した点 | 混同行列一致を整合性検査に限定したこと、最終testで方式選択しないこと、FP摂動・存在判定・分母の整理、F10の訂正の反映、Step 0受入を維持する扱い |
| 採用可能と評価（方向性） | **M-B**（開発側で完結する方式選択）、**M-D**（提案値の記述的比較）、**M-F**（確率照合の許容幅`δ=1e-6`）、**M-G**（生配列検査） |
| 修正後に確定 | **M-A**（保持被覆率の95%基準と足切り）、**M-C**（複数動画を含む保存物の読込み）、**M-E**（評価可能件数と分母） |
| 修正事項 | 95%基準と箱の定義の衝突、保持側評価の成立条件、(b)と(c)の領域一致の条件、M-Cの限界の記述の誤り、入力側の評価不能と手法側の失敗の区別 |
| 実装時の明記事項 | 未定義値の扱い、整数変換時の型範囲検査、主候補選定の閾値の表現、hash欠落と不一致の区別 |

### 3.2 2章からの変更一覧

| 2章の箇所 | 3章での扱い | 置換先 |
| --- | --- | --- |
| 2.8のH17-1足切り「`C_ho`中央値≥0.95」、M-A | **95%を撤回し、確定しない。** 合成既知形状の期待値を先に求める | 3.3 |
| 2.5の保持側評価 | 成立条件（instanceの固定、分割前後の最小点数、評価不能、seed集約、(d)のframe選択）を定義 | 3.4 |
| 2.5の「(c)の領域は(b)と一致」 | 一致の条件（端点を置く副軸位置と矩形中心）を定義 | 3.5 |
| 2.10.3、2.13 M-Cの代替(i)「C3を省けば読込前拒否の限界が消える」 | **訂正。** `summary.json`にも同じ問題が残る。両保存物に共通の手順を定義 | 3.6 |
| 2.9、2.13 M-E | 入力側の評価不能と手法側の失敗を分け、18件の内訳を併記 | 3.7 |
| 2.4-1「差が1パーセンタイルポイント以内」 | 表現を訂正 | 3.8.3 |
| 2.3のP3・P4「hash記録が無い場合」 | 一致／不一致／不明の3状態に分ける | 3.8.4 |
| 2.10.4 | 整数変換前の型範囲検査を追加 | 3.8.2 |
| 2.7、2.5、2.6（指標全般） | 未定義値の扱いを追加 | 3.8.1 |

### 3.3 保持被覆率の基準（M-Aの修正）

#### 3.3.1 衝突の確認

(b)の回転矩形は主軸・副軸それぞれを2〜98分位で切る。各軸の周辺分布で4%ずつが外れるため、両軸を同時に満たす割合は
**`1−0.04−0.04＝0.92`以上、`0.96`以下**になる。両軸の座標が独立なら`0.96×0.96≈0.922`である（［推論］、解析値）。
一様な矩形状の点やガウス分布の楕円状の点では、PCAの軸で2座標がほぼ無相関になるため、同じ点から作った場合でも約92%になる。
**したがって2章の「中央値≥0.95」は、(b)(c)(d)を定義上ほぼ常に落とす。** 幾何定義と足切りが衝突している。
(a)の軸平行矩形は最小値・最大値で切るため、同じ点からは100%、保持側でも点が密なら100%近くになり、この基準は(a)に有利に働く。

#### 3.3.2 修正方針

- **95%を撤回し、S5-17の現段階では被覆率の足切り値を確定しない。**
- 合成テスト（`check_dummy_geometry_core`）で、既知形状ごとに各表現の`C_ho`を求める。形状は次の4種に固定する。
  一様な矩形、ガウス楕円（長短比5:1）、弓状に曲がった帯、2つに分かれた領域（M1の最大instanceの挙動確認用）。
  各形状で点数を20・100・1000点とし、3.4の手順（分割・seed）を同じに適用する。
- この合成結果を管理へ報告し、**実データを読む前に**、被覆率を足切りに使うかどうかと使う場合の値を決める。
  3.3.1の解析値と合成結果が食い違えば、実装の誤りとして先に直す。
- 分位点（0.02/0.98）を被覆率の基準に合わせて変更することはしない（定義を基準へ合わせる逆算になるため）。

| 案 | 内容 | 差分 |
| --- | --- | --- |
| **推奨** | 候補の絞込みは摂動安定性だけで行い、`C_ho`は合成既知形状の期待値と並べる記述量にする | 表現の定義と衝突しない。(a)に有利な偏りが判定に入らない |
| 代替 | 合成結果から表現ごとの期待値を求め、「期待値から一定幅以上下回る」ことを足切りにする。幅は合成結果の報告後、実データ前に管理が決める | 形状の違いへの感度は残るが、表現ごとの基準で衝突を避けられる |

### 3.4 保持側評価の成立条件（2.5の修正）

| 項目 | 定義（提案） |
| --- | --- |
| instanceの固定 | **分割前に**、frameの全GT陽性点で連結成分を求め（連結距離4 local pixel、最小5点）、主instance（M1、点数最大）を決める。分割後に連結成分を求め直さない |
| 分割 | 主instanceの点を固定seedで並べ替え、先頭`ceil(n/2)`点をA（構築側）、残り`floor(n/2)`点をB（保持側）とする |
| 分割前の最小点数 | `n≥10`。これ未満のframeは保持評価不能とする |
| 分割後の最小点数 | Aが5点以上、Bが5点以上（`n≥10`で両方満たす） |
| 退化 | Aの`λ2/λ1>0.9`（軸曖昧）または共線で幅0のとき、(b)(c)(d)はそのframeで保持評価不能。(a)は幅・高さのどちらかが0なら評価不能 |
| 評価不能の扱い | 0にしない。表現ごと・理由ごとに件数を記録する |
| 共通frame集合 | 全表現で評価可能なframeの集合を主の比較対象とする。表現ごとの評価可能frameでの値は副として件数付きで併記する |
| 動画の成立 | seedごとに、共通frame集合の評価可能frameが3つ以上ある動画だけで動画中央値を求める。満たさない動画は保持評価不能として件数を記録する |
| seed間の集約 | 動画ごとに、seedごとの動画中央値の中央値（3 seed）をその動画の値とする。全体は動画値の中央値と分布（四分位）を報告する。seedごとの全体中央値も併記し、seed間の幅を示す |
| (d)のframe選択 | **Aだけから求めた`selection_score`（Aの点数）で選ぶ。Bは選択に使わない。** 選んだframeのBで被覆を測る |

(d)は1動画1 frameしか使わないため、動画単位の`C_ho`は1 frameの値になる。frame単位の(a)〜(c)と分布の性質が異なることを結果に明記する。

### 3.5 (b)と(c)の領域一致の条件（2.5の修正）

副軸方向の分位範囲`[s_0.02, s_0.98]`は、重心の両側に対称とは限らない。重心を通る主軸上に端点を置いて`w/2`の帯を取ると、(b)の矩形と一致しない。
一致させるため、次のとおり定義を揃える。座標はinstanceの重心を原点とし、主軸方向を`t`、副軸方向を`s`とする。

| 量 | 定義 |
| --- | --- |
| 矩形の範囲（(b)） | `t∈[t_0.02, t_0.98]`、`s∈[s_0.02, s_0.98]` |
| 矩形の中心`center` | `(t_mid, s_mid)`を元の座標へ戻した点。`t_mid=(t_0.02+t_0.98)/2`、`s_mid=(s_0.02+s_0.98)/2` |
| 幅`w` | `s_0.98−s_0.02` |
| 端点（(c)(d)） | `(t_0.02, s_mid)`と`(t_0.98, s_mid)`。**重心を通る線上ではなく、矩形の中心線上に置く** |
| (c)の領域 | `t∈[t(e0), t(e1)]`かつ`|s−s_mid|≤w/2` |

この定義では(c)の領域は(b)の矩形と一致する。重心は`centroid`として`center`とは別に保持する（`InstanceGeometry`に追加）。
端点が重心を通る主軸からずれ得ること（ずれ量は`|s_mid|`）は定義上の性質として記録する。
合成テストで、副軸方向に非対称な分布（片側に裾の長い分布）について(b)と(c)の被覆判定が点ごとに一致することを確認する。

### 3.6 複数動画を含む保存物の読込み（M-Cの修正）

#### 3.6.1 2章の記述の訂正

2.13 M-Cの代替(i)「`h5_metrics.csv`を読まずC3を省けば、読込前拒否の限界は消える」は**誤り**であり、訂正する。
`summary.json`も複数動画の情報（入力リストと集合単位の集計）を持つ保存物で、2.3のP2・P4の確認に使うため、同じcoverageの問題が残る。
2.10.3も`summary.json`と`h5_metrics.csv`で手順を分けて書いていなかった点を含め、次の共通手順に置き換える。

#### 3.6.2 共通手順

対象: 評価出力のcheckpointディレクトリ（`best/`、`last/`）にある、ファイル名から動画を特定できない保存物。S5-17で読むのは`summary.json`と`h5_metrics.csv`の2つだけとする。

**読込みを許可する単位は「hashで固定した個々のファイル」とする。** ディレクトリ単位やファイル名単位では許可しない。
既存の`split_identity.ArtifactCoverage`（`stage5_artifact_coverage_v1`、ファイル名とcontent SHA-256に紐づくcoverage記録）を使う。

| 段階 | 内容 |
| --- | --- |
| 1. 来歴の確認（内容を読まない） | 次の3つを確認する。(i) 対象ディレクトリが、S5-15長期runの評価出力としてユーザーが指定したものであること。(ii) git管理下の記録: S5-15の起動スクリプトに固定されたvalidationリストのhash（`checks/real_h5/run_stage5_s5_15_arm.sh`の`EXPECTED_VAL_LIST_SHA256`）と、長期run評価が固定21動画で行われたという記録（評価レポート9.10.1）。(iii) 同じディレクトリの`predictions/train_sanity/`・`predictions/validation/`の**ファイル名の集合**（内容は開かない）を`extract_video_identities`で照合し、sanity3＋validation18と過不足なく一致すること。1つでも許可リスト外の名前があれば停止する |
| 2. coverage記録の作成 | 1.が成立したとき、対象2ファイルのSHA-256（バイト列のhashで、内容は解釈しない）と、動画集合（許可リストから得たsanity3＋validation18のidentity）を`stage5_artifact_coverage_v1`で記録する。記録は来歴の根拠（1.の各項目の結果）を併記し、privateに置く。**作成は独立した登録コマンドとし、本実行のCLIでは作らない** |
| 3. 読込前の拒否 | 本実行のCLIは、2.のcoverage記録を`resolve_active_contract(coverage_path=…)`へ渡し、`assert_paths_allowed`でhash照合と封印拒否を行う。さらに記録の動画集合が目的の許可リストに完全一致で含まれることを確認する。hashが一致しない（登録後に変更された）ファイルは読まずに停止する |
| 4. 読込後の照合 | `summary.json`の`selected_train_files`・`validation_files`の全パス、`h5_metrics.csv`の全行の`h5_path`・`video_name`について、identityが許可リストに完全一致で含まれ、集合がcoverage記録の動画集合と過不足なく一致することを確認する |
| 5. 不一致時 | 実行全体を停止する。不一致の内容（値・ID・パス）は出力・ログへ書かず、件数と位置だけを出す |

#### 3.6.3 残る限界

1.の来歴は保存物の外にある証拠だが、保存物の内容そのものを検証するものではない。来歴が誤っていた場合、
4.の照合より前に封印動画の情報がメモリへ読み込まれ得る（出力はされない）。この限界は**2ファイルに共通**であり、
どちらか一方を読まないことでは解消しない。

| 案 | 内容 | 差分 |
| --- | --- | --- |
| **推奨** | 3.6.2の手順で2ファイルとも読む | P2・P4の確認とC3の整合性検査ができる。限界は上記のとおり残る |
| 代替 | 2ファイルとも読まない | 3.6.3の限界は生じない。代わりにP2（評価時に読んだH5）とP4（checkpoint・epoch）を保存物から確認できず、ディレクトリ名とNPZのファイル名だけが根拠になる。C3も行えない |

「`h5_metrics.csv`だけを読まない」案は、限界を減らさずに整合性検査だけを失うため、提案から外す。

### 3.7 評価可能件数と分母（M-Eの修正）

入力側の評価不能と手法側の失敗を分ける。**手法側の失敗は評価可能件数から落とさず、失敗として残す。**

| 区分 | 該当する事象 | 分母での扱い |
| --- | --- | --- |
| 入力側の評価不能 | crop逆変換不能（`offset_crop_fallback_resize`、非有限・非正のscale、fieldの欠落）、mm換算を指定した場合の`PixelToMM`の欠落 | 評価可能件数から除く。理由別・`local_preprocess_effective`別の件数を報告 |
| 手法側の失敗（予測） | 予測陽性なし、後処理後に幾何なし、全frameが点数不足・退化 | 評価可能件数に**含める**。失敗として数え、許容内には数えない |
| 手法側の失敗（GT表現） | GT幾何が得られない、保持評価の成立条件（3.4）を満たさない | 評価可能件数に含める。表現ごとの失敗として数える |
| 停止 | 点対応の整合性検査C1〜C3の不一致、期待ファイルの欠落、範囲外座標、封印・許可リスト違反 | 件数に入れず、実行全体を停止 |

報告の形式（validationの例）:

```text
全18件 = 入力側の評価不能 n_in + 評価可能 n_eval
評価可能 n_eval = 成功 n_ok + 手法側の失敗 n_fail（理由別）
記述統計（長さ相対誤差の中央値・四分位など）: 成功 n_ok 件の値、および n_eval を分母とする失敗率
```

判定不能の条件（2.9）は**入力側の評価不能だけで数える**ように修正する。

| 対象 | 判定不能とする条件（提案値） |
| --- | --- |
| H17-1（train_core） | 入力側で評価可能な動画が130件未満。または、入力側の除外が特定の`local_preprocess_effective`に集中し、そのモードが144件の10%以上を占めるのに1件も評価できない |
| H17-2（validation） | 入力側で評価可能な動画が16件未満 |

手法側の失敗がいくら多くても判定不能にはしない。失敗が多いこと自体が結果である。

### 3.8 実装時の明記事項

#### 3.8.1 未定義値

- `L=0`（相対誤差・相対変化の分母）、分母0の割合、空のmatching結果（precision・recall・誤差の対象がない場合）は、**0ではなく「未定義」（`None`）**とし、未定義の件数を指標ごとに記録する。
- 未定義の値は中央値・分位点の計算から除き、除いた件数を併記する。
- 共有出力でも`null`と件数で表し、0や空欄で置き換えない。

#### 3.8.2 整数変換時の型範囲

2.10.4の手順2に次を加える。整数へ変換する前に、整数値であることに加えて**変換先の型の範囲内**であることを検査する。

| 配列 | 変換先 | 範囲検査 |
| --- | --- | --- |
| `point_label` | int64 | 値が`{-1,0,1}`（型範囲内を含意） |
| `frame_order` | int64 | `0≤値≤2^63−1`、かつ中間H5の`num_frames`未満 |
| `frame_annotation/frame_order` | int64 | 同上 |
| `vote_count` | int64 | `1≤値`、かつ動画のwindow数以下 |
| `pred_label` | uint8 | `{0,1}` |

浮動小数点の配列を整数へ変換する場合は、非有限値を先に拒否し、範囲外の値でラップアラウンドや飽和が起きないようにする。

#### 3.8.3 主候補選定の閾値の表現

2.4-1の「差が1パーセンタイルポイント以内」を、**「長さの相対変化率（`|ΔL|/L`）の90パーセンタイルの差が1パーセントポイント以内」**に訂正する。
例: (c)が4.2%、(b)が5.0%なら差は0.8パーセントポイントで、事前順位により(c)を主候補とする。

#### 3.8.4 hashの照合結果

2.3のP3（H5）・P4（checkpoint）、3.6.2のcoverage記録について、照合結果を3状態で記録する。

| 状態 | 意味 | 扱い |
| --- | --- | --- |
| 一致 | 記録があり、現在のファイルのhashと一致 | 進める |
| **不一致** | 記録があり、現在のファイルのhashと一致しない | **停止する。** hash欠落と同じ扱いで許容しない |
| 不明 | 記録が無い | 「不明」と記録し、限界として結果に付ける。進めるかどうかはM-Cの判断による |

### 3.9 修正後の判断依頼

| # | 判断事項 | 推奨 | 代替案と差分 |
| --- | --- | --- | --- |
| M-A | 95%基準を撤回し、合成既知形状の結果を先に報告する手順（3.3.2）。候補の絞込みを摂動安定性だけで行い、`C_ho`を記述量にするか | 推奨案（安定性だけで絞込み） | 表現ごとの合成期待値からの乖離を足切りにする。幅は合成結果の報告後、実データ前に決める |
| M-A2 | 保持側評価の成立条件（3.4）と、(b)(c)の領域一致の定義（3.5） | 推奨案で確定 | — |
| M-C | 複数動画保存物の共通手順（3.6.2）。hashで固定した個々のファイル単位で許可し、coverage記録は独立した登録コマンドで作る | 推奨案で2ファイルとも読む | 2ファイルとも読まない（P2・P4・C3を失う） |
| M-E | 入力側の評価不能と手法側の失敗の区別、18件の内訳の報告形式、入力側だけで数える判定不能条件（3.7） | 推奨案で確定 | — |

採用可能と評価されたM-B・M-D・M-F・M-Gは、3.8.3（M-Bの閾値の表現）と3.8.2（M-Gの型範囲）の明記を加えて、2章の内容のまま維持する。
M-4・M-6・M-9と所在確認（R1〜R6）も2章のとおり維持する。

本章は再修正版の提案であり、実行条件の確定や実装開始の承認を意味しない。

## 4. 実装着手承認と専用管理書の運用（2026-09-27）

担当: 総括管理チャット。ユーザーは、現行の実装・検証方針とステップを専用管理書へ集約し、
更新ログ・修正事項・依頼内容は本報告書へ集約するよう指示した。
[実装・検証管理書](stage5_s5_17_implementation_management.md)を新設し、確定仕様・実行条件・段階別状態を記載した。
管理書は現行内容と最終更新日・版・状態だけを更新し、変更履歴の章を作らない。
依頼書は背景・引継ぎ、全体管理記録はStage 5全体の方針を担う。

### 4.1 管理判断

3章の主要修正を確認し、**実装・合成テスト・結果報告までを承認する**。
M-Aは95%基準を撤回し、被覆率は記述量、摂動安定性を選定の事前規約とする。
M-A2の保持側評価・領域定義、M-Cのファイル単位coverage登録、M-Eの失敗と分母の区別を採用する。
M-B・M-D・M-F・M-Gも確定する。実機fail-fast・本診断は合成テストと入力確認の報告後に判断する。

### 4.2 四つの必須補足

1. 安定性の成功例だけで候補を合格させない。全件未定義は選定不能。
   一部失敗の通過条件は合成テスト報告で提案し、実データ前に固定する。
2. 92〜96%は有限標本・保持側被覆率の保証ではない。既知形状の厳密検査と確率的検査を分け、
   理想化した解析値との差だけで実装不良とはしない。
3. 必須coverageの欠落・登録後のhash不一致は停止する。過去H5・checkpoint hashの不明とは区別する。
4. mm指定対象の必要値が欠ける場合は全体停止。除外して続行、pixelへの自動切替はしない。

これらは管理書5章へ反映した。この補足の反映だけを理由に再度の仕様承認待ちにしない。
未定義の扱いは分母・対象の有無で判断し、matchingが0件でも分母が正のprecision/recallは0とする。

### 4.3 次の作業

実装チャットは管理書のS17-1〜S17-3を進め、結果を本書の新しい節へ追記する。
新管理書は未実装の完了報告ではない。実機メタ監査・coverage登録・本診断はS17-4以降の条件に従う。
本更新では実装コード・実データ・テストを実行していない。文書は未コミットである。

---

## 5. S17-1 幾何コアの実装報告（2026-09-28）

担当: S5-17実装チャット。状態: **S17-1の実装を完了。コア分の合成テストは作成済みで、実機での実行待ち（未実行を合格としない）。**
根拠: 4章の着手承認、管理書6章S17-1。ユーザー指示により、合成テストはStep 0と同様に実機で実行する。
実データ・`/mnt`・封印物は読んでいない。既存の学習・評価コード、pins、Stage2to4、`split_contract`・`split_identity`は変更していない。

### 5.1 追加したファイル

| ファイル | 内容 |
| --- | --- |
| `Stage5/stage5/geometry/__init__.py` | パッケージ。numpyのみ（torch・h5py・`checks/`をimportしない） |
| `types.py` | `CoordSpace`・`Unit`・`Points2D`（座標空間付き配列）、`InstanceGeometry`（`centroid`と`center`を分離）、`FrameGeometry`、`VideoFLEstimate`。3種の「答えなし」を分離: `GeometryContractError`（停止）、`InputUnevaluable`（入力側の評価不能、理由付き）、`FailureReason`（手法側の失敗） |
| `transform.py` | `CropTransform.from_attrs`（`local_preprocess_effective`で分岐。fallback resize・非有限scale・field欠落・不明モードは`InputUnevaluable`）、local↔元frame、元frame↔正規化、範囲検査。`PixelToMM`（x/y別、供給元hash必須、既定値なし、frame別表は他frameから借りない）。異方比による長さ変化の解析式 |
| `frame_geometry.py` | 連結成分（格子ハッシュ＋union-find、距離4 local px以下で連結、5点未満を除外、ラベル0が最大）、PCAによるinstance幾何、(a)(b)(c)の領域判定、frame・動画単位の構築 |
| `fl_estimate.py` | (a)〜(c)の動画q90、(d)のframe選択をGT・実用・oracleの**別関数**に分離、(e) pseudo-3D参考値（`candidate_eligible=False`） |
| `postprocess.py` | P0・P1・P2・P3・P1+P2と、train_coreから定数を作る`PostprocessConstants.from_train_core` |
| `matching.py` | Hungarian法（scipy非依存）とゲート付きM3 matching |
| `prior.py` | 元frame正規化座標でのprior構築、identityによる自己除外、prior-onlyのframe幾何と動画推定 |
| `metrics.py` | 未定義値の扱い、符号・端点交換に不変な誤差、frame存在判定（TN含む）、instance検出、FP点の空間分類（分母を名前に含む）、保持被覆率、摂動15通り、安定性規則、主候補選定、18件の内訳（`DenominatorBreakdown`） |
| `Stage5/checks/dummy/check_dummy_geometry_core.py`・`.sh` | コアの決定的な合成テスト（11群、117項目。ループ内の検査を含む） |

### 5.2 コード化した定数

管理書4.2〜4.4のとおりで、実データを読む前に固定している。validationで調整しない。

| 定数 | 値 | 所在 |
| --- | --- | --- |
| 連結距離・最小点数 | 4 local px・5点 | `frame_geometry.py` |
| 分位点・動画集約 | 0.02／0.98・q90 | `frame_geometry.py`・`fl_estimate.py` |
| 軸曖昧・crop端 | λ2/λ1>0.9・2 local px | `frame_geometry.py` |
| 数値上の下限（調整値ではない） | 固有値≤1e-18 raw px²を全点同一、幅≤1e-9×max(1,長さ)を共線 | `frame_geometry.py` |
| 逆変換後の元frame範囲の余裕 | 1 raw px | `transform.py` |
| P2・P3 | 前後2 frame・GT長中央値×0.5、K＝GT陽性点数/frameの中央値（四捨五入） | `postprocess.py` |
| M3ゲート・FP近傍 | GT長×0.5・GT長×0.1 | `matching.py`・`metrics.py` |
| 保持評価 | n≥10、A＝ceil(n/2)、seed 0/1/2、共通frame 3以上 | `metrics.py` |
| 摂動 | 間引き50/75%、FP 10/30%、揺らぎσ1 local px、各3 seed＝15通り | `metrics.py` |
| 安定性規則・主候補 | 中央値≤5%かつP90≤15%、P90の差1パーセントポイント以内は事前順位(c)(d)(b)(a) | `metrics.py`・`fl_estimate.py` |
| prior | 正規化時刻10区間、存在率>0.5 | `prior.py` |

### 5.3 実装上の解釈（管理確認を求める事項）

仕様の文言から実装へ落とす際に解釈を要した点。いずれも実データを見る前の判断である。

| # | 事項 | 実装 | 理由 |
| --- | --- | --- | --- |
| I-1 | P3の適用範囲 | 保存済み`pred_label`で陽性点が1点以上あるframeにだけ適用する | 全frameに上位K点を適用すると、全frameが「存在」になり、frame存在判定が意味を失うため |
| I-2 | (d)のframe選択（管理書4.2の「予測情報によるscore」） | 予測診断の実用処理の規定と読んだ。H17-1のGT表現ではGT点数で選び（保持評価では構築側Aの点数のみ）、oracleはGTで選んだframeを予測に適用する。3つは別関数 | GT表現には予測がないため |
| I-3 | M3の距離 | box中心ではなく重心間距離 | 重心は全instanceで定義され、座標がすべて同一の退化instanceでも使えるため |
| I-4 | (a)が使えない条件 | AABBの幅・高さのどちらかが0のframeでは(a)を使わない（未定義） | 報告書3.4の保持評価の条件を、長さ推定にもそろえた |
| I-5 | seedが揃わない場合 | 保持被覆率・安定性とも、全seedが定義された動画だけに値を与え、欠けた動画は未定義として件数を記録する。安定性規則は、未定義や失敗が1件でもあれば`pending_failure_policy`を返し、合格・不合格を出さない | 管理書5.1。通過条件はS17-3の報告で提案する |
| I-6 | prior-onlyの(d) | 全frameが同じ幾何なので、最初の存在frameの長さ | prior-onlyには選択scoreの元になる点も確率もないため |
| I-7 | 予測で陽性点が0の動画 | 幾何関数は`empty`を返す。`no_prediction`への振り分けはCLI（S17-2）で行う | コアは予測とGTを区別しない共通関数とするため |

### 5.4 合成テストの範囲

`check_dummy_geometry_core`は決定的な検査だけを行う。期待値は手計算の定数、または実装から独立した構成（例: グリッドの分位点を`np.quantile`で直接計算）から求め、実装の出力から再計算しない。

| 群 | 主な検査 |
| --- | --- |
| [1] crop逆変換 | 3モードの既知座標（報告書2.10.1の値）、Stage 4の順変換式との整合、往復≤1e-6、入力評価不能の理由5種、crop端の範囲外で停止、空間の取り違えで停止 |
| [2] PixelToMM | (100,200)→(10,26) mm、x/y取り違えの検出、往復、供給元なし・片軸欠落・frame別の借用の拒否 |
| [3] instance幾何 | 30°回転矩形の長さ96・幅10・角度・中心・端点、軸符号・端点交換の不変性、1°対179°が2°、共線・全点同一・等方円の状態 |
| [4] (b)(c)一致 | 副軸方向に非対称な分布で、box判定とband判定が点ごとに一致。重心線上のbandなら一致しないこと |
| [5] 連結成分 | S5-15の`component_sizes`とのサイズ一致、距離ちょうど4で連結・4.0001で非連結、ラベル順 |
| [6] frame・動画 | M1/M2、点不足・空、q90、(d)のGT・実用・oracleが別のframeを選ぶ設定で分離を確認、crop端frameの除外、空間違いで停止、(e)の単位と候補外 |
| [7] 後処理 | P1・P2・P3（0.5未満を含む上位K、予測のないframeは対象外）、定数の算出 |
| [8] matching | Hungarian法と総当たりの一致（20試行）、ゲート内外 |
| [9] 検出・未定義 | GT・予測とも無いframeがTN、0/0は未定義・0/1は0、FP点の分類と分母 |
| [10] 摂動・規則 | 15通り、FPは有効背景点からのみ、不足の記録、決定性、安定性規則の合格・保留・選定不能、主候補の同値処理 |
| [11] 保持・prior・分母 | 分割の条件と決定性、n<10と共通frame不足は未定義、identityによる自己除外、18＝入力2＋評価可能16（成功13＋失敗3） |

**本開発コンテナにはnumpyがなく、合成テストは実行していない。** ここで確認したのは、全ファイルの構文（`py_compile`）、`bash -n`、禁止依存（torch・h5py・scipy・`checks/`）がコアに無いこと、未使用import・未定義名の静的検査までである。

実機での実行コマンド:

```bash
bash checks/dummy/check_dummy_geometry_core.sh
```

期待する出力の末尾は`checks: 117, failures: 0`と`All geometry core checks passed.`。
失敗した場合は、`FAIL:`の行と末尾の一覧をそのまま共有してほしい（合成データのみで、private情報は含まれない）。

### 5.5 S17-3で追加するもの（本章では未実施）

- 既知形状4種×20/100/1000点での保持被覆率の**観測**（確率的な検査）。解析値92〜96%との差だけで実装不良とはしない（管理書5.2）。
- 一部失敗・seed不足の場合の候補通過条件の提案（管理書5.1）。
- CLI（S17-2）の合成テスト: 封印拒否、未知coverage、改変hash、許可リスト外、点対応不一致、完走してJSONを書く成功経路、共有出力の境界。

### 5.6 次の作業

1. 実機で`check_dummy_geometry_core.sh`を実行してもらい、結果を本書へ記録する。失敗があればコアを直す。
2. S17-2（CLI、入力監査、coverage登録コマンド、guard、private/shared出力）に着手する。
3. S17-3の残り（5.5）を進め、まとめて報告する。
4. FILES.md・data_construct.mdの同期は、S17-2のCLIと出力構成が決まった時点でまとめて行う。

コミットは行っていない。

---

## 6. S17-1合成テスト結果、5.3への管理判断、I-4の修正（2026-09-28）

担当: S5-17実装チャット。状態: **I-4を修正。修正後の合成テストは実機での再実行待ち。** ユーザー指示により、S17-2の前に修正するのはI-4だけとした。

### 6.1 合成テストの実機結果（I-4修正前のコード）

ユーザーが実機で`bash checks/dummy/check_dummy_geometry_core.sh`を実行した。

```text
checks: 117, failures: 0
All geometry core checks passed.
```

11群すべてが出力され、失敗は0件だった。**この結果は5章時点（I-4修正前）のコードに対するものである。**
6.3の修正後のコードでは再実行が必要であり、本結果をもって修正後のコードの合格とはしない。

### 6.2 5.3（I-1〜I-7）への管理判断

| # | 判断 | 付帯事項 |
| --- | --- | --- |
| I-1 P3の適用範囲 | **条件変更として承認。** 保存済み`pred_label`に陽性点があるframeだけを対象とする仕様で固定 | 理由の説明を改める: 全frameに適用した結果（全frame陽性）も評価はできる。本仕様の目的は「陽性候補frameの中での点選択を比較すること」とする。制約として、予測陽性の無いframeのFNはP3では回復できないことを結果に付記する。また陽性点が1点あっても、後処理後に最小5点の成分が成立するとは限らないため、「P3の対象frame」と「幾何的な存在frame」は別物として扱う |
| I-2 (d)のframe選択 | **承認** | 保持評価の選択にBを使わないこと、score同値時は最小`frame_order`を優先する規則を固定する（現実装どおり） |
| I-3 M3の距離 | **重心間距離で承認** | 矩形中心との違いを指標名・仕様に明記し、matching用の距離と報告する中心誤差を混同しない |
| I-4 AABBの幅・高さ0 | **修正を要求** | 水平・垂直の線分は面積0でも長辺長を定義できる。面積・保持被覆率の成立条件と長さ推定の成立条件を分け、軸に平行な線だけを不利に扱わない。→ 6.3で対応 |
| I-5 seed不足・安定性判定 | **暫定実装として承認** | 3 seedが揃わない動画は未定義、部分的な失敗は`pending_failure_policy`、全件未定義は選定不能。最終的な通過条件はS17-3で確定する |
| I-6 prior-onlyの(d) | **承認** | 存在frameがなければ失敗として扱う（現実装どおり） |
| I-7 空予測の分類 | **承認** | CLIで、後処理で消えた場合と、元から予測陽性が無い場合を区別する |

管理確認で見つかった別の仕様差分（未修正）:

- **priorの正規化:** `prior.py`は中心だけを正規化し、長さ・幅・角度は元frame pixelのまま保持している。管理書の「元frame正規化座標でpriorを構築」と一致しない。
  正規化する幾何量と逆変換の方法を揃え、画像寸法・縦横比の異なる合成ケースで確認することが求められた。

管理からの指示: 今回の判断と修正事項を本書へ記録し、確定仕様を管理書へ反映する。S17-2・S17-3は継続してよい。

### 6.3 I-4の修正

| 箇所 | 修正前 | 修正後 |
| --- | --- | --- |
| `fl_estimate.frame_length`（長さ推定） | (a)はAABBの幅・高さの両方が正のときだけ使用 | **長辺が正であれば使用。** 水平・垂直の線分も長辺長を持つ。全点が同一の場合（長辺0）だけ未定義 |
| `metrics.frame_holdout`（保持被覆率） | 面積0は理由`zero_width`で未定義 | 面積0のときの未定義は維持し、理由を専用の`zero_area_aabb`（`FailureReason.ZERO_AREA`）に分けた。長さの失敗とは別の理由になる |
| `metrics.relative_inflation`（相対膨張率） | 面積0・凸包が定義できない場合は未定義 | 変更なし（面積に基づく量のため） |

これにより、面積に基づく量（保持被覆率・相対膨張率）の成立条件と、長さ推定の成立条件が分かれた。
なお、(b)(c)(d)は線分に対して`zero_width`で使えない。これは傾いた線分でも同じであり、軸に平行な線だけを不利に扱うものではない。

追加した合成テスト（`check_dummy_geometry_core`の[3]、水平・垂直の線分それぞれ3項目、計6項目）:

- (a)の長さが長辺58になること
- 相対膨張率が未定義になること
- 保持被覆率の(a)が`zero_area_aabb`で未定義になること

本コンテナでの確認は構文チェックまで。**実機での再実行が必要**で、期待する末尾は`checks: 123, failures: 0`。

```bash
bash checks/dummy/check_dummy_geometry_core.sh
```

### 6.4 未実施の事項

ユーザー指示（S17-2前はI-4のみ）により、次は本章では行っていない。

| 事項 | 予定 |
| --- | --- |
| priorの正規化の修正（6.2）と、画像寸法・縦横比の異なる合成ケース | S17-2と並行して、ユーザーの指示を受けて着手 |
| I-1の付記（P3の目的の説明、FNを回復できない制約、P3の対象frameと存在frameの区別）の結果出力への反映 | S17-2（CLIの出力）で実装 |
| I-3の指標名の明記（`centroid_distance`はmatching用、`center_error`は矩形中心の誤差） | 現行コードの名前は分かれている（`MatchResult.pairs`の距離と`pair_errors`の`center_error`）。共有出力の列名・説明はS17-2で明記 |
| I-7の区別（後処理で消えた／元から予測なし） | S17-2（CLI）で実装 |
| 管理書への確定仕様の反映 | 未実施。反映の担当と時期は指示を待つ |

コミットは行っていない。

---

## 7. priorの正規化修正と管理書の同期（2026-09-28）

担当: S5-17実装チャット。状態: **修正と同期を完了。修正後の合成テストは実機での再実行待ち。**
根拠: 6.2の管理確認（priorの仕様差分）と、ユーザー指示（S17-2の前にpriorを修正し、文書を更新する）。

### 7.1 priorの修正

**修正前の問題:** `prior.py`は中心だけを正規化し、長さ・幅・角度を元frame pixelのまま平均・適用していた。
画像寸法や縦横比が違う動画を混ぜると、長さは解像度の平均で決まり、角度は元の動画の縦横比のまま対象動画に写される。

**修正後:** 幾何量をすべて元frame正規化座標で持ち、正規化と逆変換に同じ対角スケール`(W, H)`を使う。

| 段階 | 処理 |
| --- | --- |
| 動画要約 | 各frameのM1のGT点を`(x/W, y/H)`へ写し、**正規化空間で箱を当て直す**（PCA、2〜98分位）。pixelで当てた箱を`(W, H)`で割ると、`W≠H`のとき矩形ではなく平行四辺形になるため |
| 動画値 | 中心は箱中心の平均、角度は2倍角平均、長さは動画q90、幅は中央値。すべて正規化値 |
| prior | 動画間の平均（角度は2倍角平均） |
| 逆変換 | 正規化した箱の中心・端点・角を**対象動画の**`(W, H)`で写す。長さと角度は写像後の端点から、AABBは写像後の角から、幅は写像後の副軸の半ベクトルを写像後の軸に垂直な方向へ射影して求める |

変更したAPI:

- `summarize_video_for_prior(identity, gt_frames, raw_points, *, raw_wh)`: M1の点を読むため、元frame pixelの点配列を受け取るようにした（`raw_frame_px`以外は停止）。
- `VideoPriorSummary`・`Prior`: フィールドを`center_norm`・`axis_angle_norm`・`length_norm`・`width_norm`に改め、正規化値であることを名前で示した。
- `frame_geometry.instance_geometry`: prior用に限り`RAW_FRAME_NORM`を受け付ける。正規化長は等方でないため、長さとして報告しないことをdocstringに明記した。

限界として記録する事項: 正規化長はx/yの分母が異なるため等方ではない。解像度・視野の異なる動画を正規化空間で平均している。

### 7.2 追加・変更した合成テスト

`check_dummy_geometry_core`に[12]を追加した（8項目）。各ケースは、修正前の実装なら異なる値になるように設定した。

| ケース | 設定 | 期待値（構成から計算） |
| --- | --- | --- |
| (A) 寸法の違い | 同じ正規化箱（中心(0.5,0.4)、101×21格子）を640×480と960×540で描き、1000×600へ写す | 正規化長0.384・幅0.04・中心(0.5,0.4)。写像後は長さ384、幅24、中心(500,240)。修正前（pixel長の平均）なら307.2 |
| (B) 縦横比の違い | 正規化で45°の箱を1000×500で描き（pixel上26.57°）、500×1000へ写す | 正規化角45°。写像後の角度は`atan(1000/500)`＝63.43°（修正前なら26.57°）。長さは写像後の端点間距離、幅は写像後の軸に垂直な成分、中心(250,500) |
| 共通 | prior-onlyの推定値が写像後の幾何を使うこと、`raw_frame_px`以外の入力を停止すること | — |

[11]のprior呼出しは新しい引数に合わせて更新した。I-4の6項目（6.3）と合わせ、コア合成テストは**131項目**になった。

本コンテナでの確認は構文チェックまで。実機で再実行してほしい。期待する末尾は`checks: 131, failures: 0`。

```bash
bash checks/dummy/check_dummy_geometry_core.sh
```

### 7.3 管理書の同期（版2）

[管理書](stage5_s5_17_implementation_management.md)の現行仕様へ、6.2の管理判断と6.3・7.1の修正を反映した。管理書の運用（1章）に従い、変更履歴は管理書に書かず本節に記録する。

| 管理書の箇所 | 反映内容 |
| --- | --- |
| 冒頭 | 最終更新日2026-09-28、版2、状態 |
| 4.1 | M3は重心間距離。報告する中心誤差（矩形中心）と区別する（I-3） |
| 4.2 (a) | 長辺>0で成立。面積0の水平・垂直線分も長さを持つ（I-4） |
| 4.2 (d) | GT・実用・oracleの選択基準と別関数での分離、同値は最小`frame_order`、crop端frameは選択不可（I-2） |
| 4.2 P3 | 陽性点のあるframeだけが対象。目的は陽性候補frame内の点選択の比較。FNを回復できないこと、P3の対象frameと幾何的な存在frameが別であることを結果に付記（I-1） |
| 4.3 | 面積に基づく量は面積0を`zero_area_aabb`で未定義とし、長さの条件と分ける（I-4） |
| 4.4 prior | 正規化空間での構築と逆写像（7.1）、限界、prior-onlyの(d)と存在frameなしの失敗（I-6） |
| 5.1 | seed不足・部分失敗・全件未定義の暫定扱い（I-5）、`empty`の分類をCLIで行い「元から予測なし」と「後処理で消えた」を区別する（I-7） |
| 6章 S17-1 | 実装済み。117/117は修正前の結果で、修正後の再実行（131項目）の合格で完了 |

### 7.4 残る事項

- 修正後のコア合成テストの実機再実行（131項目）。合格をもってS17-1を完了とする。
- S17-2で実装するもの: I-1の結果付記、I-3の指標名の明記、I-7の区別。
- S17-3で行うもの: 保持被覆率の確率的な観測、一部失敗時の通過条件の提案、CLIの合成テスト。

コミットは行っていない。

---

## 8. S17-1完了とS17-2着手（2026-09-28）

担当: S5-17実装チャット。

### 8.1 修正後のコア合成テストの実機結果

ユーザーが実機で`bash checks/dummy/check_dummy_geometry_core.sh`を再実行した（I-4・priorの修正後のコード）。

```text
[1]〜[12] の全群を出力
checks: 132, failures: 0
All geometry core checks passed.
```

**件数の訂正:** 7.2では期待値を131項目としたが、正しくは132項目である。[12]の検査は8項目ではなく9項目だった（(A) 2項目、(B) 6項目、入力空間の拒否1項目）。
内訳は、5章の117項目＋I-4の6項目＋[12]の9項目＝132項目。テスト側の誤りではなく、報告での数え違いである。

管理書6章の完了条件（修正後の再実行で合格）を満たしたため、**S17-1を完了とする。**

### 8.2 管理書の同期（版3）

冒頭の状態、6章のS17-1（完了、132/132）とS17-2（着手中）を更新した。

### 8.3 S17-2の着手

管理書3章・6章に従い、CLI・入力監査・coverage登録・guard・private/shared出力を実装する。結果は次章以降に記録する。

---

## 9. S17-2 CLIの実装報告（2026-09-28）

担当: S5-17実装チャット。状態: **S17-2のコードを実装。CLIの合成テストは未作成・未実行（S17-3で作成し、実機で実行する）。実データは読んでいない。**
既存の学習・評価コード、pins、Stage2to4、`split_contract`・`split_identity`・`file_list_mode`・`h5_io`は変更していない。

### 9.1 追加・変更したファイル

| ファイル | 内容 |
| --- | --- |
| `Stage5/evaluate_stage5_geometry.py`（新規） | CLI。サブコマンド`register-coverage`・`audit`・`run`。契約解決、目的別許可リスト、封印guard、来歴・hash状態、private/shared出力、エラー出力の匿名化 |
| `Stage5/evaluate_stage5_geometry.sh`（新規） | launcher。`MODE`でサブコマンドを選ぶ。未確定の値（sanityリスト、出力先）は既定値を持たず、未設定なら停止する |
| `Stage5/stage5/utils/geometry_inputs.py`（新規） | CLI専用の読込関数（h5py・numpy）。生配列を検査してから変換する。契約検査は行わず、CLIがguardを通したパスだけを渡す |
| `stage5/geometry/types.py`（追記） | `FailureReason`に`removed_by_postprocess`・`gt_reference_undefined`を追加（I-7） |
| `stage5/geometry/metrics.py`（追記） | matchingで組になったペアの距離を`matching_centroid_distance`に改名し、矩形中心の`center_error`と区別（I-3） |

### 9.2 読込順と封印・許可リスト（管理書3.2）

全サブコマンドで次の順に処理する。

1. pins→registry→manifestを解決する（manifestはpin名で指定し、パスでは指定できない）。
2. `--train_core_list`・`--validation_list`の完全identity SHA-256をmanifestの値と照合する。`--train_sanity_list`は3件で、すべてtrain_coreに含まれることを確認する。
3. 目的別の許可リストを作る: `gt_prior`＝train_core、`prediction_diagnosis`＝validation、`in_sample_diagnosis`＝sanity3、`evaluation_run_artifact`＝validation∪sanity3。
4. 各パスを**開く前に**、封印拒否（`assert_paths_allowed`、registryが先に判定）と許可リスト照合を行う。
   - teacher H5は、属性を読む前に検査する。
   - 中間H5は、記録attrか完全一致basenameで解決し、ファイル名のidentityがteacherと完全一致することを確認してから、開く前に検査する。解決できなければ停止する（globは使わない）。
   - NPZは許可リストから`predictions/<split>/<safe_name(video_name)>.npz`を組み立て、名前のidentityが一致することを確認する。無ければ停止し、ディレクトリの列挙で代わりを探さない。
   - `summary.json`・`h5_metrics.csv`は、hashに紐づくcoverage記録を必須とする（9.3）。

### 9.3 複数動画を含む保存物のcoverage登録（管理書3.2、`register-coverage`）

本実行とは別のコマンドで、checkpointディレクトリ1つずつ（`best`、`last`）登録する。2ファイルの内容は解釈しない。

| 証拠 | 実装 |
| --- | --- |
| (i) ユーザーが指定したディレクトリ | `--evaluation_checkpoint_dir`の末尾名が`--checkpoint_name`（best/last）と一致すること |
| (ii) git管理下の記録 | `list_content_sha256(validationリスト)`が、S5-15 launcher（`checks/real_h5/run_stage5_s5_15_arm.sh`）の`EXPECTED_VAL_LIST_SHA256`と一致すること。launcherがgit管理下にあり、HEADから変更されていない場合に限り証拠として採用する |
| (iii) 予測ファイル名の集合 | `predictions/`の中身が`train_sanity/`と`validation/`だけで、それぞれのNPZ名のidentityがsanity3・validation18と過不足なく一致すること（名前だけを見て、内容は開かない） |

証拠がすべて成立した場合に限り、2ファイルのSHA-256と、対象動画の集合（許可リストから得た21件）を`stage5_artifact_coverage_v1`形式でprivateに記録する。
本実行は、このcoverageでhash照合・封印拒否・許可リスト照合を行ってから2ファイルを解析する。解析後には、全行・全パスのidentityと集合を再照合する（過不足があれば停止）。
残る限界（管理書3.2）: 来歴が誤っていれば、解析後の照合より前に誤った情報がメモリへ入り得る。

### 9.4 点対応の整合性検査と来歴（管理書3.3）

| 検査 | 実装 | 不一致時 |
| --- | --- | --- |
| NPZの形 | 4キーの存在、各配列長＝H5の点数、`point_indices`＝`arange(N)` | 停止 |
| 確率とlabel | `p1>0.5+1e-6`ならlabel1、`p1<0.5−1e-6`ならlabel0。境界帯は両方を許し、件数を記録 | 停止 |
| vote_count（追加） | window16/stride8/tailで再計算したwindow数と点ごとに完全一致。`summary.json`のwindow設定もこの値であることを確認 | 停止 |
| 混同行列 | 再計算したTP/FP/TN/FNが`h5_metrics.csv`の該当行と整数で一致 | 停止 |
| run・checkpoint | `summary.json`の`checkpoint`の名前が`best.pt`/`last.pt`、epochが6/50 | 停止 |
| hash状態 | checkpoint（`--checkpoint_sha256_*`）とteacher H5（`--h5_hash_record`）の記録値が与えられれば照合する。記録なし＝`unknown`、一致＝`match`、不一致＝停止（不明に落とさない）。評価時のrevisionは与えられれば`recorded`、なければ`unknown` | 不一致は停止 |

生配列の検査（`geometry_inputs.py`）: dtype・形状・行数・有限性・整数性・値域・int64の範囲を確認してから変換する。
`point_label∈{-1,0,1}`、`valid_mask`は真偽値または{0,1}で、**`valid_mask`＝`point_label≠-1`**（teacher preflightと同じ整合性）、`frame_order`は0以上かつ中間H5のlocal frame数未満、`pixel_xy`はlocal crop内、とする。

### 9.5 `audit`（S17-4用、メタ情報のみ）

teacher H5のdatasetの形状・dtypeと属性の有無、中間H5の解決方法、`local_preprocess_effective`、crop逆変換が可能か（入力評価不能の理由）、NPZのキー・形状（`.npy`ヘッダだけを読む）、coverageとhashの状態を件数で出力する。
GT値・予測値は読まない。動画別の詳細はprivateに、件数は共有JSONに出す。

### 9.6 `run`（S17-5用）

| 段階 | 内容 |
| --- | --- |
| H17-1（train_core） | 動画ごとにGT幾何、(a)〜(d)のM1推定とM2（H17-5）、(e)参考値、保持被覆率、相対膨張率、BBoxとの整合性（H17-1b、記述量）、摂動15通りでの相対変化。集計は表現ごとに分母の内訳、安定性5区分、候補規則、選定。候補規則が未定義や部分失敗を含む表現があれば`undecided`とし、主候補を出さない。入力側の評価可能件数が130件未満、またはモード条件に当たれば`suspended_input_insufficient` |
| 定数・prior | train_coreからP2距離・P3のKを1回算出し、priorを構築 |
| H17-2〜H17-5（validation） | best（主）とlast（補助）のそれぞれで、5種の後処理ごとに、frame存在判定、instance検出、FP点の分類、表現ごとの推定と相対誤差（`τ_rel`は記述のみ、`T_FL`ではないと明記）、(d)のoracle（別欄）、P3の対象frame数と幾何的な存在frame数、失敗理由（`no_prediction`／`removed_by_postprocess`／`gt_reference_undefined`を区別） |
| prior-only | validationの各動画でGTと比較（記述）。sanity3では対象動画をidentityで除いたprior |
| sanity3 | train_sanityのNPZで同じ解析を行い、別表（`sanity_in_sample`）に出す |
| 出力 | privateに動画別の記録とalias対応表。共有JSONに集計・状態・注記（P3、指標名、最良後処理はvalidationで選んだ開発上の選択であること）・限界 |

`--coordinate_mode mm`は、mmの供給元・キー・粒度が確定するまで停止する（管理書5.4）。pixelモードではmmの結果も`T_FL`判定も出さない。

### 9.7 出力境界

- 共有JSONは、書き出す前に、両命名規則の動画identityと絶対パスが含まれていないことを検査する。検査に通らなければ何も書かない。
- 停止時のメッセージは、identityをマスクした形に、パスを`<path>`に置き換えてから出す。想定外の例外のtracebackはprivateディレクトリにだけ書く。
- privateディレクトリは0700、ファイルは0600で作る。

### 9.8 実装上の判断（管理確認を求める事項）

| # | 事項 | 実装 | 理由 |
| --- | --- | --- | --- |
| J-1 | 読込関数の置き場所 | `stage5/utils/geometry_inputs.py`に分離した（h5py依存、CLIからだけ使う） | 幾何コア（numpyのみ）とCLIの間で、生配列の検査を単体で試験できるようにするため。管理書4.1の「I/O・契約検査はCLI側」の範囲と解釈した |
| J-2 | sanity3の指定 | `--train_sanity_list`を必須とし、3件かつtrain_coreに含まれることを検査する。評価出力の`selected_train_files.txt`からは取らない | 検証される側の保存物を根拠にしないため |
| J-3 | H17-2で使う表現 | H17-1で主候補・副候補が決まればそれを使う。決まらない場合（未定義や部分失敗で`undecided`）は、`--h17_2_representations`で明示的に3つ以内を指定したときだけ計算し、指定がなければH17-2は計算しない | 失敗時の通過条件はS17-3の提案を経て実データ前に固定される（管理書5.1）。本実行は1回なので、決まらなかった場合の扱いはS17-5の承認時に決めてほしい |
| J-4 | vote_countの再計算 | 整合性検査に追加した | 点順・window構成の不一致を検出できる、実装コストの小さい検査のため |
| J-5 | P2・P3の定数の算出単位 | P2＝train_coreの全存在frameにおけるM1の(b)長さの中央値×0.5。K＝存在frameあたりのGT陽性点数（frame内の全GT陽性点）の中央値 | 管理書4.2の「GT長中央値」「GT陽性frameあたり点数中央値」をframe単位と解釈した |
| J-6 | 中間H5が解決できない場合 | 停止する（入力評価不能として数えない） | 管理書3.2の「曖昧解決は停止」に合わせ、所在の問題を件数で流さないため |
| J-7 | H17-1b（BBoxとの整合性） | frame内の全BBoxを合わせた外接矩形を元frameへ写し、点GTのM2のAABBとIoU・長辺の相対差を比べる（記述量） | 報告書1.5.2の記述量。採否には使わない |
| J-8 | hashの記録が無い場合 | `unknown`として記録し、実行は続ける | 管理書5.3により、扱いは実データ診断の承認時に確定する |

### 9.9 本コンテナでの確認

構文チェック（`py_compile`、`bash -n`）、未使用import・未定義名の静的検査まで。**実行は一度もしていない。**

### 9.10 次の作業（S17-3）

1. CLIの合成テスト（`check_dummy_geometry_cli_guard`）を作る。合成の契約（pins・registry・manifest）、合成teacher・中間H5、NPZ、`summary.json`・`h5_metrics.csv`、合成launcherを使う。検査項目は次のとおり。
   - 封印動画が存在しないパスでも、開く前に封印例外になること
   - 許可リスト外、list identityの不一致、coverage欠落、登録後のhash改変、名前集合の過不足を拒否すること
   - 前方一致の中間H5を拾わないこと、期待NPZが無ければ代わりを探さないこと
   - 点数・確率・vote_count・混同行列の不一致で停止すること
   - `register-coverage`→`audit`→`run`が完走してJSONを書くこと
   - 共有JSON・stdout・例外に動画ID（両命名規則・第4セグメント）と絶対パスが出ないこと
2. 既知形状での保持被覆率の観測（確率的な検査）。
3. 一部失敗・seed不足の場合の通過条件の提案。
4. FILES.md・data_construct.mdの同期（CLIと出力構成）。

コミットは行っていない。

---

## 10. S17-3: 合成テスト・被覆率観測・失敗時の通過条件の提案（2026-09-28）

担当: S5-17実装チャット。状態: **テストと観測スクリプトを作成。実機での実行待ち。失敗時の通過条件は提案で、管理判断待ち。**
実データは読んでいない。本コンテナでは構文チェックと静的検査だけを行った（numpyが無いため）。

### 10.1 作成したもの

| ファイル | 種類 | 内容 |
| --- | --- | --- |
| `Stage5/checks/dummy/check_dummy_geometry_cli_guard.py`・`.sh` | 合否判定あり | CLIの合成テスト（10.2） |
| `Stage5/checks/dummy/observe_geometry_holdout_coverage.py`・`.sh` | **観測のみ（合否判定なし）** | 既知形状での保持被覆率の観測（10.3） |
| `Stage5/evaluate_stage5_geometry.py`（追記） | — | `run`に入力件数の下限（既定130／16）を上書きする引数を追加した。合成テストで小さな集合を完走させるためだけのもので、launcherには出さず、helpにも表示しない。使った値と、既定値から変えたかどうかは共有JSONに`input_minimums`として必ず記録する |
| `docs/stage5/FILES.md`・`data_construct.md` | 文書 | CLI・幾何コア・読込関数・テストの説明、S5-17の出力構成と共有境界を追記した。FILES.mdの封印節では、封印契約に接続された経路に`evaluate_stage5_geometry.py`を加えた |

### 10.2 CLIの合成テスト（`check_dummy_geometry_cli_guard`）

一時ディレクトリに合成の環境を作る。合成の契約（pins・registry・manifest）、teacher H5・中間H5（train_core 6件、sanity3、validation 3件）、best/lastの評価出力（NPZ、`summary.json`、`h5_metrics.csv`、checkpointファイル）、git管理下の合成launcherからなる。
動画IDは、timestamp形式（第4セグメントあり・なし）とcase形式を混ぜ、`..._1`と`..._1_02`を前方一致の罠として共存させた。train_coreの1件は`offset_crop_fallback_resize`で、入力評価不能になる。

| 群 | 主な検査 |
| --- | --- |
| [1] 読込関数 | 非整数・非有限・int64範囲外の拒否、`point_label`の値域、`valid_mask`と`point_label≠-1`の不一致で停止、確率の境界帯、`..._1`が`..._1_02`へ解決されないこと、別identityの記録attrで停止、NPZヘッダの読取り |
| [2] 出力境界 | 共有検査が両命名規則のIDと絶対パスを拒否すること、エラーメッセージの匿名化 |
| [3] 許可リスト | validation動画はGT・prior用に使えない、sanity以外のtrain_coreはin-sample診断に使えない、train_coreは予測診断に使えない |
| [4] coverage登録 | best・lastの登録（4記録、各6動画、0600）。予測ファイル名に余分なものがあれば拒否、launcherが未コミットなら拒否、ディレクトリ名がcheckpoint名と違えば拒否 |
| [5] audit | 入力評価不能・解決方法・NPZキーの件数、checkpointのhash状態（一致／不明）、記録hashの不一致で停止、coverageなしで停止 |
| [6] 封印 | 存在しないパスを与えた封印動画が「ファイルが無い」ではなく封印として拒否されること、list順の違い・sanityにvalidation動画が混ざった場合の停止 |
| [7] run（成功経路） | 完走してprivate（0600、ディレクトリ0700）と共有JSONを書く。H17-1の分母（6＝入力評価不能1＋評価可能5）、除外が1つのモードに集中した場合の判断保留、H17-2の表現が明示指定によるものであること、validationの分母・成功数、`τ_rel`の注記、FP・ignoreの集計、P3の注記、最良後処理がbestでだけ選ばれること、sanityの別表 |
| [8] runの拒否 | coverageなし、登録後の`h5_metrics.csv`改変、期待NPZの欠落（前方一致の別名NPZを拾わない）、点数・確率とlabel・window数・混同行列の不一致、mmモード。停止時に共有JSONを書かないこと |

すべての成功・停止について、標準出力・エラーに動画ID・絶対パス・tracebackが出ないことも検査する。
検査項目数は**137**の見込み（ループ内を含む。実行時の表示を正とする）。

### 10.3 保持被覆率の観測（`observe_geometry_holdout_coverage`、合否判定なし）

管理書5.2に従い、既知形状4種（一様な矩形100×20、ガウス楕円5:1、曲がった帯、2領域）×点数20/100/1000×50 frame×seed 0/1/2で、(a)(b)(c)の保持被覆率の中央値・四分位と、未定義の件数・理由を表にする。
2領域だけは通常の連結（4 px）を使い、M1（大きい方の領域）の挙動を見る。他の3形状は全点を1つのinstanceとして箱の幾何だけを観測する。
参考値として、(b)の理想化した範囲（0.92〜0.96）と、軸に平行な一様矩形での(a)の式`((m−1)/(m+1))²`（m＝構築側の点数）を並べる。形状は25°回転させているので、後者は目安にとどまる。
**解析値との差だけで実装不良とはしない。** 結果を管理へ報告し、実データを読む前に被覆率の扱い（記述量のまま、または表現ごとの期待値からの乖離）を決めてもらう。

### 10.4 一部失敗・seed不足の場合の通過条件（管理書5.1、提案）

現在の暫定実装（I-5）では、未定義や失敗が1件でもある表現は`pending_failure_policy`を返し、合格・不合格を出さない。
実データでは、144件の中に幾何が得られない動画が数件出ることは十分あり得る。そのため、実データ前に次のどれかを固定する必要がある。

| 案 | 内容 | 評価 |
| --- | --- | --- |
| **F-A（推奨）** | 手法側の失敗（GTの基準長が未定義、摂動後が未定義、3 seedのどれかが未定義）を、その区分で**相対変化＝∞（最悪値）**として数える。中央値・P90は、評価可能な全動画を分母として計算する。入力評価不能は分母から除く（2.9・3.7のとおり） | 成功例だけで合格させない（管理書5.1）ことが、規則の形そのもので保証される。失敗が評価可能件数の10%を超えるとP90が∞になり、自動的に不合格になる。規則が1本で済む |
| F-B | 失敗率の上限（例: 評価可能件数の5%以下）を別条件とし、成功例の中央値・P90と両方を満たせば合格 | 失敗率と安定性を分けて読める。ただし上限値をもう1つ決める必要があり、成功例だけの統計が判定に入る |
| F-C | 現状どおり、1件でも失敗があれば選定不能 | 最も厳しい。144件では、ほぼ確実に選定不能になる見込み［推論］ |

seedについては、F-Aでは3 seedのどれかが未定義なら∞とする（seedを選ばない）。
F-Bを選ぶ場合は「3 seed中2つ以上が定義されていれば定義されたseedの中央値」という代案がある。ただし、都合のよいseedだけで集計しないという管理書4.3の趣旨から、F-Aを推奨する。

保持被覆率（記述量）は選定に使わないため、seedが揃わない動画は未定義として件数を報告するだけとする（現実装どおり）。

F-Aが承認されれば、`metrics.stability_cell_summary`・`candidate_rule`を変更し、コアの合成テストに∞扱いの検査を加える。**実装は承認後に行う。**

### 10.5 実機で実行してほしいコマンド

```bash
bash checks/dummy/check_dummy_geometry_cli_guard.sh
bash checks/dummy/observe_geometry_holdout_coverage.sh
bash checks/dummy/check_dummy_geometry_core.sh     # 回帰確認（コアは本章で未変更）
```

- `check_dummy_geometry_cli_guard`: 期待する末尾は`checks: 137, failures: 0`と`All geometry CLI checks passed.`。gitが必要。
- `observe_geometry_holdout_coverage`: 表をそのまま共有してほしい（合成データのみで、private情報は含まれない）。
- 失敗した場合は、`FAIL:`の行と末尾の一覧を共有してほしい。

### 10.6 管理判断を求める事項

| # | 事項 |
| --- | --- |
| S3-1 | 失敗時の通過条件（10.4）。推奨F-A |
| S3-2 | 9.8のJ-1〜J-8。特にJ-3（H17-1が決まらない場合にH17-2で計算する表現）はS17-5の承認時までに必要 |
| S3-3 | 被覆率の扱い（10.3の観測結果を受けて） |

コミットは行っていない。

---

## 11. S17-3の実機結果、管理判断、F-A・J項目の反映（2026-09-28）

担当: S5-17実装チャット。状態: **F-A等を反映。修正後の合成テストは実機での再実行待ち。**

### 11.1 実機結果（10章時点のコード）

| 実行 | 結果 |
| --- | --- |
| `check_dummy_geometry_cli_guard.sh` | [1]〜[8]の全群を出力、`checks: 137, failures: 0`（10.2の見込みどおり） |
| `check_dummy_geometry_core.sh`（回帰） | `checks: 132, failures: 0` |
| `observe_geometry_holdout_coverage.sh` | 下表（観測。合否判定なし） |

保持被覆率の観測（中央値。50 frame×3 seed＝150件のうち定義された件数で集計）:

| 形状 | n | (a) | (b) | (c) | 未定義 |
| --- | ---: | ---: | ---: | ---: | --- |
| 一様な矩形 | 20 | 0.700 | 0.600 | 0.600 | なし |
| 一様な矩形 | 100 | 0.940 | 0.820 | 0.820 | なし |
| 一様な矩形 | 1000 | 0.992 | 0.910 | 0.910 | なし |
| ガウス楕円5:1 | 20 | 0.800 | 0.600 | 0.600 | なし |
| ガウス楕円5:1 | 100 | 0.960 | 0.840 | 0.840 | なし |
| ガウス楕円5:1 | 1000 | 0.994 | 0.914 | 0.914 | なし |
| 曲がった帯 | 20 | 0.800 | 0.600 | 0.600 | なし |
| 曲がった帯 | 100 | 0.940 | 0.860 | 0.860 | なし |
| 曲がった帯 | 1000 | 0.994 | 0.916 | 0.916 | なし |
| 2領域 | 20 | — | — | — | 全件未定義（`holdout_too_small` 3件。ほかのframeはinstanceが成立せず） |
| 2領域 | 100 | 0.667 | 0.583 | 0.583 | なし |
| 2領域 | 1000 | 0.984 | 0.904 | 0.904 | なし |

観測から読み取れること（記述。判定には使わない）:

- (b)と(c)は全ケースで同じ値だった。3.5の領域一致の定義どおりである。
- 被覆率は点数に強く依存する。n=20では保持側が10点なので、値は0.1刻みになる。表現間の比較は同じ点数どうしでしか意味を持たない。
- 一様矩形の(a)は、n=1000で0.992。参考式`((m−1)/(m+1))²`＝0.992（m=500）と一致した。n=100・20では式よりやや高い（0.940対0.923、0.700対0.669）。形状を25°回転させているため、式は目安にとどまる。
- (b)はn=1000で0.90〜0.92。理想化した範囲（0.92〜0.96）の下限付近かやや下である。有限標本、推定したPCA軸、保持側での評価による差と考えられ、実装の不自然な挙動は見られなかった（管理書5.2のとおり、この差だけで不良とはしない）。
- 2領域では、点が疎だと連結成分が細かく割れる。n=20ではinstanceがほとんど成立せず、n=100でも被覆率が低い。点密度がM1と被覆率に影響することを示す。実データのframeあたりGT点数との関係は、実データの記述統計で確認する。
- **観測対象は(a)〜(c)であり、(d)のframe選択を含む挙動は検証していない。**

### 11.2 管理判断（10.6への回答）

| # | 判断 |
| --- | --- |
| S3-1 失敗時の通過条件 | **F-Aを採用する方向。** 実装前に次を固定する: 最悪値（∞）は判定上の扱いにとどめ、共有JSONにInfinityを書かず`null`と状態・失敗件数を別項目にする／分位点の計算法を明記する（∞を含む線形補間はNaNを生み得るため、補間しない順位方式等）／「失敗率10%超でP90が∞」の境界は方式と件数に依存するため、境界ケースを合成テストする／全件未定義は引き続き「選定不能」とし、部分失敗を最悪扱いにした不合格と区別する／記述用の成功例統計と失敗を含む選定用統計を分ける |
| J-1 | 承認（I/Oをutilsへ分離し、幾何コアから依存させない） |
| J-2 | 承認。ただし任意のtrain_core内3件ではなく、Step 0で固定したsanity3と一致する根拠を保持する |
| J-3 | 承認。fallbackの表現は実データを見る前に指定し、「診断対象であって採用候補ではない」と出力する。結果を見て指定し直さない。具体的な3表現以内の指定はS17-5承認時までに確定する |
| J-4 | 承認（保存時のwindow構成・tail処理に合わせて検証する） |
| J-5 | 概ね承認。P2は長さが定義できるM1 frameを対象とし、未定義の件数を記録する。P3の「陽性frame」は有効GT陽性点が1点以上あるframeとし、最小5点のinstance成立条件と混同しない |
| J-6 | 承認（所在・対応の不明を単なる除外として流さない） |
| J-7 | 記述量として承認。全BBoxを包む領域どうしの比較であり、個々のinstance対応の精度とは区別する |
| J-8 | 実データ実行時の扱いは保留。R6の確認結果を見て決める。必須coverageの欠落・hash不一致は停止のまま |
| S3-3 被覆率 | **記述量のまま維持する**（管理書で確定済みの方針）。観測は表現・点数・形状による挙動、未定義の発生、不自然な挙動の確認に使う。(d)のframe選択を含む挙動は検証済みとしない |

### 11.3 F-Aの実装（`stage5/geometry/metrics.py`）

| 項目 | 実装 |
| --- | --- |
| 最悪値の扱い | GTの基準長が未定義、摂動後が未定義、3 seedのどれかが未定義の動画は、その区分で最悪値とし、分母に残す。入力評価不能は分母に入らない（CLI側で除外済み） |
| 分位点 | **最近順位法、補間なし**（`SELECTION_QUANTILE_METHOD = "nearest_rank_no_interpolation"`）。1始まりの順位`ceil(q·n)`の値を取る。P90が最悪になるのは、最悪扱いの動画数が`n − ceil(0.9n)`を超えるとき（例: n=10なら2件以上、n=20なら3件以上、n=143なら15件以上）。中央値も同じ方式（n=10なら6件以上で最悪） |
| 出力 | 最悪値は内部の判定用マーカーだけに使う。選定統計の`median`・`p90`は最悪なら`null`とし、`median_is_worst`・`p90_is_worst`と、`n_worst`（内訳: `n_seed_incomplete`・`n_all_seeds_undefined`）を別項目で出す。共有・privateのJSONは`allow_nan=False`で書き、非有限値が混ざれば書かずに停止する |
| 判定 | 各区分の選定用中央値≤5%かつP90≤15%で合格。最悪値を含む区分は不合格とし、`failing_cells`に記録する。全動画が全区分で未定義なら`not_selectable`（不合格と区別） |
| 記述統計 | `descriptive_success_only`（成功例のみ、線形補間の中央値・四分位・P90）を選定統計とは別に出す。判定には使わない |
| 暫定状態の廃止 | `pending_failure_policy`（I-5の暫定実装）は廃止した |
| 選定結果（CLI） | 合格した表現があれば主・副候補を選ぶ（選定不能の表現は候補から外し、`not_selectable_representations`に記録する）。合格がなく、全表現が選定不能なら`not_selectable`、それ以外は`no_candidate` |

最後の行の「選定不能の表現を候補から外して残りで選ぶ」は、管理書の「全件未定義は選定不能」を表現ごとに適用した解釈である。全表現の判断を止めるべきであれば指示してほしい。

### 11.4 J項目の反映

| # | 実装 |
| --- | --- |
| J-2 | `--train_sanity_list`のファイルSHA-256が、pin済みmanifestの`input_sha256.sanity_list`（Step 0の分割builderがS0-1監査のhashで照合した入力）と一致しなければ停止する。manifestは読む直前に再度hashを取り、pinで検証した内容と同じであることを確認する。同じ3件でも、Step 0とは別のファイル（並べ替え等）は受け付けない |
| J-3 | 明示指定のときは共有JSONに`role: diagnostic_target_not_adoption_candidate`と「実データ前に固定し、結果を見て指定し直さない」旨の注記を出す。H17-1の候補を使うときは`role: h17_1_candidates` |
| J-5 | P2は長さが定義できるM1 frameだけを使い、長さ未定義のframe数（`p2_frames_length_undefined`）を記録する。P3は有効GT陽性点が1点以上あるframeの点数を使う（以前は幾何的な存在frameに限っていたので修正した）。共有JSONに定義文と対象frame数を出す |
| J-7 | BBox整合性の出力に「全BBoxを包む領域と全GT instanceを包む領域の比較であり、instance単位の一致ではない」旨の注記を付けた |
| J-8 | 保留（コードは現状のまま。hashの記録がなければ`unknown`、不一致は停止） |

### 11.5 合成テストの変更

| テスト | 変更 | 期待する項目数 |
| --- | --- | --- |
| `check_dummy_geometry_core` [10] | 暫定の保留状態の検査を、F-Aの検査に置き換えた。失敗した動画が最悪値になり不合格になること（選定不能とは別）、seedが1つ欠けた動画が最悪になること、全件未定義は選定不能であること。最近順位法の境界（n=10で1件／2件、n=20で2件／3件、n=143で14件／15件、中央値はn=10で5件／6件）。記述統計と選定統計の分離、補間しないこと、出力にInfinity・NaNが無いこと | **142**（132−12＋22） |
| `check_dummy_geometry_cli_guard` | 合成manifestにStep 0のsanityリストhashを記録した。sanityにvalidation動画が混ざった場合と、同じ3件を並べ替えた別ファイルの場合に、Step 0のsanityリストでないとして停止すること。fallback表現が診断対象と表示されること。共有JSONにInfinity・NaNが無いこと | **143**（137＋6。fallback表示の確認は既存項目に統合） |

`FILES.md`・`data_construct.md`の記述は、今回の変更で内容が変わる箇所がないため更新していない。

### 11.6 実機で実行してほしいコマンド

```bash
bash checks/dummy/check_dummy_geometry_core.sh        # 期待: checks: 142, failures: 0
bash checks/dummy/check_dummy_geometry_cli_guard.sh   # 期待: checks: 143, failures: 0
```

観測スクリプトは今回変更していないため、再実行は不要。

### 11.7 次の段階

上記の合格をもって、S17-3の完了報告（成果物、合成テストの結果、観測、残る限界）をまとめる。
そのうえで、S17-4（実機の入力fail-fastとcoverage登録）の承認を依頼する。依頼には次を含める。

- 所在確認R1〜R6の結果（ユーザーへ依頼）
- J-3のfallback表現の指定（S17-5承認時まで）
- J-8の扱い（R6の結果による）

コミットは行っていない。

---

## 12. S17-3完了報告とS17-4の承認依頼（2026-09-28）

担当: S5-17実装チャット。状態: **S17-3の完了報告。管理確認とS17-4の承認を依頼する。**
ユーザーの判断（完了報告に移ってよい）を受けて作成した。本章までに、実データ・`/mnt`・封印物は一度も読んでいない。

### 12.1 修正後の合成テストの実機結果

| テスト | 結果 |
| --- | --- |
| `check_dummy_geometry_core.sh` | [1]〜[12]、`checks: 142, failures: 0`（11.5の見込みどおり） |
| `check_dummy_geometry_cli_guard.sh` | [1]〜[8]、`checks: 143, failures: 0`（11.5の訂正後の見込みどおり） |

これにより、S17-2（CLI）は管理書の完了条件（CLI合成テストの合格）を満たした。

### 12.2 S17-1〜S17-3の成果物

| 区分 | ファイル | 状態 |
| --- | --- | --- |
| 幾何コア（numpyのみ） | `Stage5/stage5/geometry/{types,transform,frame_geometry,fl_estimate,postprocess,matching,prior,metrics}.py` | コア合成テスト142/142 |
| 読込関数（h5py） | `Stage5/stage5/utils/geometry_inputs.py` | CLI合成テスト[1]で検査 |
| CLI・launcher | `Stage5/evaluate_stage5_geometry.py`・`.sh`（`register-coverage`・`audit`・`run`） | CLI合成テスト143/143 |
| 合成テスト | `Stage5/checks/dummy/check_dummy_geometry_core.*`、`check_dummy_geometry_cli_guard.*` | 実機で合格 |
| 観測 | `Stage5/checks/dummy/observe_geometry_holdout_coverage.*` | 実機で実行（11.1）。合否判定なし |
| 文書 | 本報告書、管理書（版5）、`FILES.md`、`data_construct.md` | 同期済み |

既存の学習・評価・推論コード、pins、Stage2to4、`split_contract`・`split_identity`・`file_list_mode`・`h5_io`は変更していない。

### 12.3 合成テストで確認したこと・していないこと

確認したこと（合成データ上）:

- 座標変換は、手計算の既知座標とStage 4の順変換式に一致する。往復誤差、異方的なmm換算も確認した。
- 幾何の定義（(b)と(c)の領域一致、符号・端点交換への不変性、退化の区別）、未定義と0の区別、分母の内訳を確認した。
- F-Aの判定、最近順位法の境界、選定不能と不合格の区別、出力にInfinity・NaNが無いことを確認した。
- priorは正規化空間で構築・逆写像され、画像寸法・縦横比が違っても正しく写される。
- 封印は、存在しないパスを与えても開く前に拒否される。許可リスト、coverageのhash、名前集合、中間H5の完全一致、NPZを代わりに探さないことも確認した。
- 点・確率・window数・混同行列の整合性検査で停止すること、mmモードで停止すること、出力境界（ID・パス・tracebackが出ない、0600/0700）を確認した。
- `register-coverage`→`audit`→`run`が完走すること。

していないこと（限界）:

- **実データでの動作。** 実H5の形式の揺れ、中間H5の実際の解決方法、NPZの所在は未確認である（R1〜R4）。
- 実データでの計算時間・メモリ。
- (d)のframe選択を含む被覆率の挙動（観測は(a)〜(c)のみ）。
- mmでの評価（mmの契約が未確定のため、mmモードは停止する）。
- 合成テストは本コンテナでは実行できず、実機の結果だけに基づく。

### 12.4 残る限界（結果に付ける事項。本章で変わらないもの）

1.5.6・LIMITSのとおり。GT由来量どうしの比較であり、臨床FL精度ではない。長さはx/y等方を仮定した元frame pixelである。
ファイル単位で、同一検査の兄弟ファイルを含む。validationは開発に使った集合で、最良後処理もvalidationで選ぶ。
旧train／validationにまたがる同一検査は4グループ（両集合で計17ファイル）で、validation側の件数は未確認。学習runもseedも1つだけである。

### 12.5 S17-4の承認依頼

#### 12.5.1 実施内容

実機で次の3コマンドをこの順に1回ずつ実行する（ユーザーが実行する）。**GT・予測の配列は読まない。** `run`（S17-5）は含めない。

| # | コマンド | 読むもの | 出力 |
| --- | --- | --- | --- |
| 1 | `MODE=register-coverage CHECKPOINT_NAME=best` | 契約とリスト、S5-15 launcherのhash行、`best/predictions/{train_sanity,validation}/`のファイル名（21件）、`best/summary.json`・`best/h5_metrics.csv`のバイト列（hash計算のみ、内容は解釈しない） | private: coverage記録。shared: 件数・hash先頭16桁・証拠の成否 |
| 2 | `MODE=register-coverage CHECKPOINT_NAME=last` | 同上（`last/`） | 同上（同じcoverageファイルに追記） |
| 3 | `MODE=audit` | teacher H5 162件の属性とdatasetの形状・dtype（配列値は読まない）、中間H5 162件の属性、NPZ 21件×2のヘッダ、`summary.json`・`h5_metrics.csv`（coverage照合後に解析）、与えられた場合はcheckpointファイルのバイト列（hash計算） | private: 動画別のメタ情報。shared: 件数（解決方法、crop mode、入力評価不能の理由、datasetの揃い、NPZキー、hash状態） |

実施量: 各コマンド1回。不具合を修正した後の再実行は1回まで（管理書6章）。結果を見て条件を変えて繰り返さない。
S17-4の読取りとS17-5の本実行は重複する。本実行は冒頭でfail-fastの検査をすべてやり直すが、重なるのはメタ情報とcoverage照合だけである。

#### 12.5.2 実行前にユーザーへ確認したいこと（個人情報を含まない形で）

| # | 事項 | 回答の形 |
| --- | --- | --- |
| R1 | S5-15長期runの評価出力root（launcherの既定は`/mnt/data/3d_projects/stage5_evaluations/260919/<実験名>`）に、`best/`・`last/`の`predictions/validation/`・`predictions/train_sanity/`があるか | 存否とNPZ件数（例: 18／3） |
| R2 | 同じディレクトリに`summary.json`・`h5_metrics.csv`があるか | 存否 |
| R3 | Step 0のmm監査のprivate記録にある、train_core中間H5の解決方法の内訳 | 例: recorded_source_attr 144／glob 0。globが1件以上あれば、S17-4の`audit`は該当動画で停止する見込み |
| R4 | validation18の中間H5が、train_coreと同じroot（launcherの既定`pseudo3d_outputs/260711`、suffix`_ts448_oym96_corr.h5`）にあるか | はい／いいえ |
| R5 | mm/pixelの供給元（報告書1.7.1） | S17-4・S17-5のpixelモードには不要。mm評価を行う場合にだけ必要 |
| R6 | S5-15評価時点のgit revision、checkpointのSHA-256、teacher H5の動画別SHA-256の記録の有無 | 記録の有無と種類。ある場合は`CHECKPOINT_SHA256_BEST`等で与える |
| R7（追加） | Step 0の分割builderに渡したsanityリストのファイル（`--sanity_list`、元の評価の`selected_train_files.txt`と想定）のパス | ユーザー側で`TRAIN_SANITY_LIST`に設定する（共有しない）。manifestの記録hashと一致しなければ停止する |
| R8（追加） | private出力の置き場所 | 提案: `/mnt/data/3d_projects/stage5_private_work/s5_17_geometry/`（領域A）。共有JSONも同じ場所の`shared/`へ出し、確認後に共有する |

#### 12.5.3 停止条件（S17-4で想定するもの）

- 契約: pinとhashの不一致、リストのidentityがmanifestと違う、sanityリストのhashがStep 0の記録と違う、封印動画を含む、許可リスト外。
- 来歴: launcherが未コミット、validationリストのhashがlauncherと違う、予測ファイル名の集合がsanity3＋validation18と一致しない、`predictions/`に余分なエントリがある。
- 入力: teacher H5・中間H5の欠落、中間H5が記録attrでも完全一致basenameでも解決できない（globは使わない）、`video_name`がidentityと一致しない、local_input_shapeが読めない。
- 評価run: coverageの欠落、登録後のhash変化、`summary.json`・`h5_metrics.csv`の動画集合の過不足、checkpoint名・epoch（6／50）・window（16／8／tail）の不一致、記録されたcheckpoint hashとの不一致。
- 停止時の出力は、IDとパスを伏せたメッセージだけ。tracebackはprivate側に書く。

#### 12.5.4 S17-4の結果として報告するもの

共有JSON（`coverage_best.json`・`coverage_last.json`・`audit.json`）の内容と、停止した場合はその停止メッセージ。
これをもとに、入力評価不能の件数（特に`offset_crop_fallback_resize`、F4）、中間H5の解決方法の内訳（F2）、有限scaleの件数（F3）を報告する。
Stage 4側のBBox変換（F4）に該当する動画があれば、件数を報告し、扱いは管理の判断に委ねる。

### 12.6 管理判断を求める事項

| # | 事項 | 必要な時期 |
| --- | --- | --- |
| S4-1 | S17-3の完了確認と、S17-4の実施承認（12.5の内容・実施量） | S17-4の前 |
| S4-2 | 11.3の解釈（選定不能の表現は候補から外し、残りで主候補を選ぶ） | S17-5の前 |
| S4-3 | J-3のfallback表現（H17-1で候補が決まらない場合に計算する3表現以内） | S17-5の承認時まで |
| S4-4 | J-8（過去hashが不明な場合に続行するか） | R6の回答後、S17-5の前 |

コミットは行っていない。コード・文書のコミットが必要になった時点で、差分を確認してコマンドを示す（実行はユーザー）。

### 12.7 総括管理からの回答: S3-1〜S3-3（2026-09-29記録）

本節は、10.6への管理チャット回答を転記した決定記録である。
回答時点では10章のスクリプト作成報告を確認しており、合成テストの実行結果は未確認だった。
その後に追記された11〜12章の実装・実行結果は別の証拠であり、
**本節の追記自体を12.6のS4-1（S17-3完了確認・S17-4実施承認）への回答とは扱わない。**

#### 12.7.1 S3-1: 失敗時の通過条件

**F-Aを採用する方向で進める。** 手法側の失敗を分母から落とさず、3 seedの一つでも未定義なら、
その動画・摂動区分を最悪扱いにすることは管理書5.1と整合する。
実装に際して次を固定する。

1. **∞は判定上の扱い**とする。共有JSONへ`Infinity`を直接書かず、値は`null`、状態と失敗件数は別項目で出す。
2. 分位点の計算法を明記する。∞を含む配列への通常の線形補間ではNaNが生じ得るため、
   補間しない順位方式などを明示する。
3. 「失敗率10%超でP90が∞」の境界は、分位点方式と件数に依存する。境界ケースを合成テストする。
4. **全件未定義は引き続き選定不能**とする。部分失敗を最悪扱いにした不合格と区別する。
5. 記述用の成功例統計と、失敗を含む選定用統計を分ける。

この回答はF-Aの計算・出力規約を具体化して合成検証へ進める判断であり、
実データを見た後に通過条件を変更する許可ではない。

#### 12.7.2 S3-2: J-1〜J-8への回答

| 項目 | 管理判断 |
| --- | --- |
| J-1: I/Oをutilsへ分離 | **承認。** 幾何コアから依存させない構成でよい |
| J-2: sanityリストを必須化 | **承認。** ただし任意のtrain_core内3件ではなく、Step 0で固定したsanity3と一致する根拠も保持する |
| J-3: 候補未決時の明示指定 | **承認。** fallback表現は実データを見る前に指定し、診断対象であって採用候補ではないと出力する。結果を見て指定し直さない。具体的な3表現以内の指定はS17-5承認時までに確定する |
| J-4: vote_count再計算 | **承認。** 保存時のwindow構成・tail処理に合わせて検証する |
| J-5: P2・P3定数の集約 | **概ね承認。** P2は長さが定義できるM1 frameを対象とし、未定義件数を記録する。P3の陽性frameは有効GT陽性点が1点以上あるframeとし、最小5点のinstance成立条件と混同しない |
| J-6: 中間H5未解決で停止 | **承認。** 所在・対応の不明を単なる除外として流さない |
| J-7: BBox整合性 | **記述量として承認。** 全BBoxを包む領域同士の比較であり、個々のinstance対応の精度とは区別する |
| J-8: 過去hash不明でも続行 | **実データ実行時の判断は保留。** R6の確認結果を見て決める。必須coverageの欠落・hash不一致は停止のまま |

#### 12.7.3 S3-3: 被覆率の扱い

**被覆率は記述量のまま維持する。** これは管理書で確定した方針であり、
合成観測の結果を受けて足切りへ戻す必要はない。
観測結果は、表現・点数・形状による挙動、未定義の発生、実装の不自然な挙動の確認に用いる。
今回の観測対象は(a)〜(c)であり、(d)のframe選択を含む挙動まで検証済みとは扱わない。

#### 12.7.4 回答後の進行範囲

上記回答で認めた次の作業は、F-Aの計算・出力規約の具体化、合成テストと観測の実行・報告までである。
11〜12章の結果は、その実施報告として管理側が別途確認する。
**実データ監査・本診断の開始は本回答では承認していない。**
12.6のS4-1〜S4-4は別の判断依頼として扱う。

本追記は報告書への回答記録のみであり、実装・管理書の変更、テストや実データ処理、コミットは行っていない。

### 12.8 12.7の確認と、R7・R8へのユーザー回答（2026-09-29）

担当: S5-17実装チャット。状態: **記録のみ。実装・テスト・実データ処理は行っていない。**

#### 12.8.1 12.7の確認

12.7（S3-1〜S3-3への回答）の内容は、11章で反映済みの実装と一致している。追加で必要な変更はない。

| 12.7の項目 | 対応 |
| --- | --- |
| 12.7.1の1〜5（∞の扱い、分位点方式、境界の合成テスト、選定不能と不合格の区別、記述統計と選定統計の分離） | 11.3・11.5で実装し、12.1で合成テストに合格（コア142/142） |
| 12.7.2 J-1〜J-7 | 11.4で反映済み。J-8は保留のまま（hashの記録がなければ`unknown`、不一致は停止） |
| 12.7.3 被覆率は記述量 | 実装も記述量のみ（選定に使わない）。(d)を含む挙動は未検証として扱う（11.1・12.3） |
| 12.7.4 進行範囲 | 了解した。12.7はS4-1への回答ではないため、**S17-4（実データ監査）には着手しない。** S4-1〜S4-4の回答を待つ |

#### 12.8.2 R7・R8へのユーザー回答

| # | 回答 | 扱い |
| --- | --- | --- |
| R7 sanityリスト | S5-15長期run評価出力の`evaluation_data/selected_train_files.txt`（評価出力rootはlauncherの既定`EVALUATION_OUTPUT`と同じ） | `TRAIN_SANITY_LIST`に設定する。このファイルが根拠になるのは置き場所のためではなく、**SHA-256がpin済みmanifestの`input_sha256.sanity_list`（Step 0の記録）と一致すること**による。一致しなければ、S17-4の最初のコマンドで停止する（そのときは、Step 0が実際に使ったファイルの確認を依頼する） |
| R8 private出力先 | `/mnt/data/3d_projects/stage5_private_work/s5_17_geometry/`で問題ない | `PRIVATE_OUT_DIR`に設定する。共有JSONは`${PRIVATE_OUT_DIR}/shared/`に出し、内容を確認してから共有する |

未回答: R1〜R4（評価出力・予測・中間H5の所在）、R5（mm。pixelモードでは不要）、R6（過去のrevision・hashの記録）。
R1〜R4は、S17-4の`register-coverage`・`audit`自体が機械的に検査する項目である。承認時に「事前回答なしでS17-4の検査に委ねる」と判断することもできる。

#### 12.8.3 S17-4承認後に実行するコマンド（承認前は実行しない）

```bash
cd /mnt/data/3d_projects/models/Stage5
export TRAIN_SANITY_LIST=/mnt/data/3d_projects/stage5_evaluations/260919/pointnext_s_EX260919_s5_15_r0long50_none_gn8_cwfixed_lr1e3_ep50_bs1_acc8_nopad/evaluation_data/selected_train_files.txt
export PRIVATE_OUT_DIR=/mnt/data/3d_projects/stage5_private_work/s5_17_geometry

MODE=register-coverage CHECKPOINT_NAME=best SHARED_JSON="${PRIVATE_OUT_DIR}/shared/coverage_best.json" \
  bash evaluate_stage5_geometry.sh
MODE=register-coverage CHECKPOINT_NAME=last SHARED_JSON="${PRIVATE_OUT_DIR}/shared/coverage_last.json" \
  bash evaluate_stage5_geometry.sh
MODE=audit SHARED_JSON="${PRIVATE_OUT_DIR}/shared/audit.json" \
  bash evaluate_stage5_geometry.sh
```

- 順に1回ずつ実行する。途中で停止した場合は、次のコマンドへ進まずに停止メッセージを共有する（メッセージにIDとパスは出ない）。
- R6でcheckpointのSHA-256の記録が見つかった場合に限り、`CHECKPOINT_SHA256_BEST`・`CHECKPOINT_SHA256_LAST`を追加する。
- 共有してよいのは`${PRIVATE_OUT_DIR}/shared/`の3ファイルと、標準出力・エラーの表示だけである。`${PRIVATE_OUT_DIR}`直下のファイル（coverage記録、動画別のメタ情報）は共有しない。

コミットは行っていない。


### 12.9 総括管理からの回答: S17-3受入・S17-4承認と未回答事項の扱い（2026-09-29）

担当: 総括管理チャット。状態: **ユーザー承認を確認し、S17-3を受け入れ、S17-4の実施を承認する。S17-5の本計算は未承認。**
本節を12.6のS4-1〜S4-4および12.8で残った所在・来歴確認事項への管理回答とする。

#### 12.9.1 S4-1: S17-3完了確認とS17-4実施承認

11章の修正内容と12.1の実機合成テスト結果（コア142/142、CLI143/143）を根拠に、S17-3の完了を受け入れる。
管理側で今回テストを再実行したという意味ではなく、報告された結果に基づく受入である。実データでの適合性はS17-4以降で確認する。

ユーザー・管理側の承認が揃ったため、12.5・12.8に従い、ユーザーの実機で次を順に実施してよい。

1. bestの`register-coverage`
2. lastの`register-coverage`
3. `audit`によるメタ情報監査

読取り対象・出力・停止条件は12.5を維持する。GT・予測の配列値は読まず、各コマンド1回、不具合修正後の再実行は規定どおり1回までとする。
途中で停止した場合は次のコマンドへ進まず、原因を報告する。結果を見て条件を変えて繰り返さない。
今回の承認に`run`（S17-5）やmm評価は含めない。

#### 12.9.2 S4-2〜S4-4への回答

| 事項 | 判断と条件 |
| --- | --- |
| S4-2: 選定不能の表現を除外して残りから選ぶ | **承認。** 選定不能の表現を候補から外し、残る合格表現から主・副候補を選ぶ。除外した表現と理由を記録する。入力件数不足など全体の停止・保留条件は、この処理で回避しない。全表現が選定不能なら`not_selectable`、合格がなくそれ以外の場合は`no_candidate`とする。 |
| S4-3: H17-1で候補が決まらない場合のfallback | **(a) AABB、(c) PCA中心線端点、(d) best-frame端点の3表現に固定する。** (b)と(c)は長さ・領域が共通なので重複を避け、異なる構成を比較する。fallbackは診断用であり、採用候補として扱わない。実データの結果を見て入れ替えない。この固定自体はS17-5の実施承認ではない。 |
| S4-4: 過去hash不明時の続行 | **S17-4では`unknown`として記録してよい。S17-5での続行可否はR6と監査結果を確認して判断する。** `unknown`を一致確認済みと扱わない。必須coverageの欠落・hash不一致は停止のままとする。 |

#### 12.9.3 R1〜R6の扱い

以下は確認方法と進行条件の決定であり、実機の所在や来歴を確認済みとする回答ではない。

| 事項 | 管理側の回答 |
| --- | --- |
| R1: 予測出力の所在・件数 | **事前の手動回答は不要。** 実機側で対象rootを設定し、coverage登録・監査でbest/lastそれぞれvalidation18＋sanity3との一致を確認する。既定パスを実在確認済みと扱わない。 |
| R2: summary・CSVの存否 | **事前回答を待たず機械検査へ委ねる。** 欠落時は停止し、coverage照合後に内容を検査する。 |
| R3: Step 0当時の中間H5解決方法 | Step 0のprivate記録から方法別件数を確認できれば報告する。ただし、その確認をS17-4着手の必須条件にはしない。過去にglobを使ったことだけで現在の監査失敗とは断定しない。今回の許可された解決方法（記録attrまたは完全一致basename）での結果を判断材料とする。 |
| R4: validationの中間H5所在 | **事前回答は不要。** 必要なら実機側でrootを設定し、監査で解決できることを確認する。解決不能時に探索範囲を勝手に広げない。 |
| R5: mm/pixelの供給元 | 引き続き未確認として保持する。S17-4・S17-5のpixelモードには不要であり、その着手を妨げない。mm評価には別途契約の確定が必要。 |
| R6: 過去revision・hash | **S17-4と並行して既存記録を確認し、監査結果と一緒に報告する。** 評価時点のgit revision、checkpoint SHA-256、teacher H5の動画別SHA-256を個別に「記録あり／記録なし／未確認」とし、記録がある場合は種類と照合結果を示す。今計算したhashを評価当時の証拠として扱わない。記録されたhashとの不一致は停止する。 |

R7・R8は12.8.2の回答を維持する。共有するのは個人情報を含まない件数・状態・許可された共有出力とし、private記録や動画別のID・パスは共有しない。

#### 12.9.4 判定文言の明確化とS17-5への移行条件

11.3および管理書5.1にある「最悪値を含む区分は不合格」は、次の意味に明確化する。

> 失敗を最悪値として分母に含め、選定用中央値またはP90が最悪値、あるいは対応する閾値を超えた場合に、その区分を不合格とする。

失敗が1件でもあれば直ちに不合格、という意味ではない。中央値≤5%・P90≤15%、補間しない順位方式、全件未定義と不合格の区別は維持する。
これは実装に合わせた文書の明確化であり、S17-4着手を妨げない。過去の報告節は履歴として保持し、本節を補足回答とする。

`audit`が正常終了しても、それだけで入力の受入完了とはしない。欠落・入力評価不能の件数、解決方法、crop mode、来歴状態を確認し、R6と合わせてS17-5の可否を管理側で判断する。
実装チャットでは本節の確定事項を管理書の現行方針・状態・メタ情報へ同期し、同期内容や残件は報告書に記録する。

本追記では報告書とその冒頭メタ情報のみを更新した。管理書・実装の変更、テスト、実データ処理、コミットは行っていない。

### 12.10 12.9の同期、run経路の固定、R6のrepo内確認、S17-4の実行手順（2026-09-29）

担当: S5-17実装チャット。状態: **管理書を同期し、`run`経路のコードを修正。CLI合成テストの再実行待ち。実データは読んでいない。**

#### 12.10.1 管理書の同期（版6）

| 管理書の箇所 | 同期内容 |
| --- | --- |
| 冒頭 | 最終更新日2026-09-29、版6、状態（S17-1〜S17-3完了、S17-4承認済み・実施待ち、S17-5未承認） |
| 3.1 | R1〜R4は事前回答を必須とせずS17-4の機械検査で確認。R6は既存記録を項目別（記録あり／記録なし／未確認）に報告し、今計算したhashを評価当時の証拠としない。R5は未確認のまま保持。R7・R8の回答 |
| 4.2 | H17-1で候補が決まらない場合（判断保留を含む）のH17-2は(a)(c)(d)に固定。診断対象であり採用候補ではない。S17-5の実施承認ではない（S4-3） |
| 5.1 | 「最悪値を含む区分は不合格」を12.9.4の文言に明確化。選定不能・不合格の表現は候補から外し、除外した表現と理由を記録。入力不足による保留はこの処理で回避しない（S4-2） |
| 5.3 | S17-4では過去hashの不明を`unknown`として記録。S17-5の続行可否はR6と監査結果で判断し、`unknown`を一致確認済みと扱わない（S4-4） |
| 6章 | S17-3完了（管理受入）。S17-4承認済み（実施内容・停止時の扱い・報告事項を記載）。S17-5はS17-4の結果とR6を管理が確認した後の承認とし、`audit`の正常終了だけでは入力受入としない |

#### 12.10.2 `run`経路のコード修正（S4-2・S4-3）

実データを読む前にコードで固定するため、次を修正した。**S17-4で使う`register-coverage`・`audit`の経路は変更していない。**

| 修正 | 内容 |
| --- | --- |
| fallback表現の固定（S4-3） | `--h17_2_representations`引数を廃止し、H17-1で候補が決まらない場合は定数`FALLBACK_H17_2_REPRESENTATIONS`＝(a)(c)(d)を計算する。出力の`basis`は`fixed_fallback_while_h17_1_undecided`、`role`は`diagnostic_target_not_adoption_candidate`。launcherの`H17_2_REPRESENTATIONS`も削除した |
| 除外の記録（S4-2） | 選定結果に`excluded_representations`（表現ごとの状態と不合格区分）を追加した |
| 保留時の記録 | H17-1が入力不足で判断保留になった場合、これまでは規則の結果を上書きしていた。修正後は、保留の状態を維持したまま、規則の結果を`rule_result_not_used_while_suspended`として記録だけする。保留中は、この結果を候補として使わず、H17-2は固定fallbackで計算する |
| CLI合成テスト | fallbackが(a)(c)(d)で診断対象と表示されること、保留時に除外情報付きの規則結果が記録されることを確認するよう変更した（**144項目**の見込み） |

#### 12.10.3 R6: リポジトリ内で確認できた範囲

リポジトリ内の文書・共有済み成果物だけを確認した（実機の記録は見ていない）。

| 項目 | repo内の状況 | 扱い |
| --- | --- | --- |
| 評価時点のgit revision | **未確認。** S5-15報告8.12.5に、長期runの**学習起動時**のgit HEAD（`d0e3752…`、作業ツリーclean。manifestは実機の`work_dirs/_s5_15_longrun_manifests/`）が記録されている。ただし評価（`evaluate_stage5`）を実行した時点のrevisionの記録は見当たらない。`summary.json`にもrevisionのfieldはない | 学習時のrevisionを評価時のrevisionとは扱わない。実機側の記録（評価launcherのログ等）の確認を依頼する |
| best・lastのcheckpoint SHA-256 | **未確認。** repo内に長期runのbest/lastのhash記録は見当たらない。S5-15にはcheckpoint同一性監査（`check_stage5_checkpoint_identity`）があり、長期runに適用された出力が実機にあれば記録になり得る | 実機側の確認を依頼する。記録があれば`CHECKPOINT_SHA256_BEST`／`_LAST`で与え、不一致なら停止する |
| teacher H5の動画別SHA-256 | **repo内には記録なし。** Step 0のteacher preflightは集合単位の期待値で、動画別hashではない。S5-15のframe可視化（`export_stage5_prediction_frames.sh`）は`HASH_H5=1`でH5のhashを記録できるが、実行時に有効だったかは不明 | 実機側の確認を依頼する（可視化出力のmanifestにH5のhashがあるか） |

ユーザーに確認してほしいこと（S17-4と並行、個人情報を含まない回答でよい）:

1. 長期runの**評価**を実行したときのgit revisionの記録の有無（評価ログ等）。
2. 長期runのbest.pt・last.ptのSHA-256の記録の有無（checkpoint同一性監査の出力等）。
3. S5-15のframe可視化の出力に、teacher H5の動画別hashの記録があるか（`HASH_H5=1`で実行したか）。

#### 12.10.4 実行順序（S17-4）

1. **CLI合成テストの再実行**（`run`経路を修正したため）。`bash checks/dummy/check_dummy_geometry_cli_guard.sh`。期待は`checks: 144, failures: 0`。コアは変更していないので再実行は不要。
2. （推奨）**S17-4の前に、コードと文書をコミットする。** CLIは出力に`git`の状態（dirtyかどうか）を記録する。未コミットの新規ファイルがあるとdirtyとして記録され、監査結果とコードの版の対応が弱くなるためである。コミットする場合は、差分を確認したうえでコマンドを用意する（実行はユーザー）。
3. 12.8.3の3コマンド（best・lastの`register-coverage`、`audit`）を順に1回ずつ実行する。途中で停止したら次へ進まず、停止メッセージを共有する。
4. `${PRIVATE_OUT_DIR}/shared/`の3ファイルと、12.10.3の確認結果を共有する。

コミットは行っていない。
