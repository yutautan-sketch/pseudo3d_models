# Stage 5 S5-16残務・Step 0 報告と進捗管理

作成日: 2026-09-22  
最終更新日: 2026-09-27（38章に同一検査重複件数の訂正を記録）
対象stage: S5-16の文書反映残務とStep 0（S5-17開始前の準備）  
担当: S5-16〜S5-20総括管理チャット  
状態: **S0-1〜S0-7・teacher期待値算出は完了、Step 0クローズ済み。受入・同期はf31cdd8でコミット済み。mm/pixelはUNCONFIRMEDのままS5-17へ持ち越す。次はS5-17実装依頼書の作成と実行条件の確定。**
現行の管理返信: **受入は35章、同期完了は36章、最終確認は37章**。各回の提案・実装報告・未実施表記は当時の履歴として保持する。

## 1. 参照仕様・承認範囲

- [S5-16ハンドオフ v3](stage5_s5_16_implementation_handoff.md)9〜12章、特に10.0節。
- [固定の引き継ぎ案内](stage5_s5_16_to_s5_20_policy_chat_transfer.md)。本作業では変更しない。
- [管理記録](../stage5_revision_management_record.md)8章の更新ルール。
- [S5-15終了報告](../s5-15/stage5_s5_15_report_to_policy_chat.md)8.14節。

2026-09-22、ユーザーは、既に決定されたS5-15終了・S5-16方針・B′→A方針の文書反映と、
新しい残務整理記録の作成について「はい、それらの作業に取り掛かってください」と承認した。
また「Step 0を含むS5-16の作業が完了次第、S5-17の専用管理チャットおよびそこへの伝達文書の作成に移ります」
と指定した。これは当初の運用案として保持する。**同日の後続判断で、管理チャット1つ＋各ステップの
実装チャットへ変更した（7章・D-038）。** 現在はS5-17専用管理チャットを設けず、Step 0完了後に
本総括管理でS5-17実装チャットへの依頼書を作成する。

この承認に基づく文書同期・Step 0実装依頼書の準備と、実機のsplit生成・封印・weight算出を区別する。
実装担当は[Step 0依頼書](stage5_step0_implementation_handoff.md)に計画・実施量を対応づけて報告する。
S5-17の診断、S5-18以降の学習、production変更、コミット実行の承認を含まない。

## 2. 現状確認と実行環境

- `/workspace`内の文書・コード・git履歴・匿名化済みCSVを参照した。
- 開始時HEADは`2f73520`（文書・research成果物整理）。前回の状況把握時の多数のステージ済み変更は
  既にコミットされ、今回開始時の既存差分は`docs/stage5/FILES.md`と`docs/stage5/data_construct.md`の
  未ステージ変更だった。これらを保持して編集する。旧git statusの件数やコミットコマンドは流用しない。
- 実データ・runはユーザー実機の`/mnt/data/3d_projects/`以下。本チャットではその存在・内容・CUDAを検証していない。
- 2026-09-22の確認に対し、ユーザーはB′分割・封印を**「未実施」**と回答した。
- Step 0専用コード・完了記録は開始時のリポジトリ検索では確認できなかった。
  新weightは未確定。実機で算出済みと推定しない。

## 3. 今回の文書反映内容

| 対象 | 内容 | 状態 |
| --- | --- | --- |
| 管理記録0・3章 | S5-15終了／S5-16方針完了／Step 0未完了を現在状態へ反映。v6基準runの旧表は履歴に区分 | 完了 |
| 管理記録4〜7章 | timeline、段階別要約、現行フロー、D-035〜D-037を追加 | 完了 |
| 管理原則・Stage 5/6契約 | 原則6／D-004はS5-18前、責務境界はS5-20a前に改訂する予定を明記。今は適用範囲変更済みとしない | 完了 |
| 評価レポート9.10 | 50 epoch公式評価、可視化・追加解析、終了判断と8.14の留保を反映 | 完了 |
| 評価レポート9.11・結論 | 新方針・評価境界と現在の結論を追加、旧結論を履歴として保持 | 完了 |
| S5-15報告書9章 | 過去ログを変えず、文書同期済みと移管先を更新 | 完了 |
| 本記録・Step 0依頼書・索引 | 残務・開始条件・完了条件と新規文書の導線を整備 | 完了 |

管理側で追加実装・学習・GPU推論・実データCPU診断は行っていない。
匿名化済みCSVの既報値照合、Markdownリンクと差分の検査だけを実施した。
元の固定引き継ぎ文書、S5-16 v3、S5-15報告8.1〜8.14の履歴は変更しない。

## 4. Step 0残務と完了条件

| ID | 作業・受入証拠 | 状態 |
| --- | --- | --- |
| S0-1 | 保存済みtrain162／validation18／sanity3、teacher v7、層化情報・臨床FLの実機所在と対象対応を監査 | 未実施 |
| S0-2 | 層化の定義・欠損／同値／小層処理・seed・実施量・出力先・封印方式を実行前に固定 | 依頼書へ確認項目を記載、実装担当の計画待ち |
| S0-3 | train_core144／internal_test18／validation18を生成。重複なし、旧集合との一致、sanity保持、リストhashを確認 | 未実施 |
| S0-4 | internal_testを方式選択・通常評価から隔離。旧checkpoint拒否・無断アクセス拒否を合成テストし、実機の封印証拠を残す | 未実施 |
| S0-5 | train_coreだけの新weightを1回算出。対象リストhash、算出コードrevision/hash、設定・数値精度を記録し固定 | 未実施 |
| S0-6 | x/y別mmスケールの所在・単位・動画／frame粒度、crop逆変換情報の取得可否をtrain_coreで確認 | 未実施。実値は共有しない |
| S0-7 | 実装・実機結果・匿名化確認・逸脱・未決事項を本書へ追記し、総括管理で受入、管理記録と評価レポートを同期 | 未実施 |

S0-1〜S0-7を実施・受入してStep 0を閉じる。mm情報に不足がある場合は、除外／対応方針を明記して
ユーザー判断を残し、未解決のまま「確認済み」としない。`T_FL`はS5-17実行前のユーザー設定事項であり、
Step 0で臨床的な数値を推定しない。未確定のまま引き継ぐ場合はS5-17開始条件として明示する。
コード・文書のコミット対象はユーザーの依頼時に現差分から提示し、実行はユーザーが行う。

## 5. 未決事項・次のアクション

次はStep 0依頼書を実装担当へ渡し、S0-1／S0-2の監査・具体計画を受ける。
実機の臨床FLや実video ID対応表は本チャットへ貼り付けず、所在と形式・取得可否だけを報告する。
層化情報の不足を、モデル予測やvalidation臨床FLによる補完で解消しない。
計画・申請・管理返信・実装結果は本書の新しい節として日付・承認範囲付きで追記する。

S5-16方針策定済みという状態は維持するが、S5-16残務／Step 0完了とはまだ記録しない。
完了受入後も本総括管理を継続し、S5-17実装チャットへの依頼書を作成する。

## 6. 文書同期の確認結果（2026-09-22）

- 長期runの共有CSVでpooled指標を照合し、動画別CSVからvalidationのbest／last各18件、
  TP0 1／10件、F1中央値7.75%／0.00%、last−bestの改善4／悪化13／同値1を確認した。
  保存済み匿名化metricsの文書照合であり、モデル評価の再実行ではない。
- 編集・新設した7文書内の相対Markdownリンクは存在確認に合格した。
- S5-15報告書8章全体とS5-16 v3／固定引き継ぎ案内がHEADから不変であることを確認した。
- `git diff --check`に合格。既存`FILES.md`の変更を保持して索引2件だけを追加し、
  `data_construct.md`は今回編集していない。Step 0の実装・出力構造が確定した時点で更新する。
- コード変更、実データ処理、GPU実行、コミットは行っていない。

## 7. 管理体制の変更とStep 0依頼書の確定（2026-09-22）

ユーザーは「S5-16〜S5-17の管理は全てこのチャットに集約」と決定し、続いて
「管理チャット1つ＋各ステップの実装チャット」という方針の同期とStep 0依頼文書の作成を指示した。
1章の当初案とD-036の専用管理チャット移行予定は、この後続判断（D-038）で変更する。
実験の順序・固定条件・実施量の承認範囲を変更する判断ではない。

- 総括管理: 方針、優先順位、仕様・実施範囲、結果の解釈、採否、完了受入と記録同期を担当。
- 各実装チャット: 担当stepの監査・具体設計・実装・検証・報告を担当。小さな修正ごとに分割しない。
- Step 0に別の管理チャットは不要。既存の実装依頼書を更新し、担当の説明、最小限の参照範囲、
  最初の作業、初回報告の形式を追加した。重複する新しい依頼書は作成しない。
- Step 0の残作業・完了条件は4章のS0-1〜S0-7を維持する。分割・封印・weight算出は未実施のまま。
- Step 0完了後、S5-17の管理も本チャットで継続し、S5-17実装依頼書を用意する。

管理記録の現在状態・現行フロー・D-038、S5-15報告9章、文書索引にも現行体制を反映する。
固定引き継ぎ案内、S5-16 v3、過去のdecision record本文は保持する。

## 8. Step 0実装チャットの初回報告: 入力監査と具体計画（2026-09-22）

担当: Step 0実装チャット。状態: **提案（計画）。実装・実機実行・コミットは未実施。**
本節は[Step 0依頼書](stage5_step0_implementation_handoff.md)7章の初回報告に対応する。
承認範囲の変更を求めるものではなく、S0-1の監査結果とS0-2〜S0-6の具体案を総括管理へ返す。

| 項目 | 報告内容 |
| --- | --- |
| 現状・根拠 | 8.1。`/workspace`のコード・文書・匿名化済みCSVのみを読んだ。実データ・実機runは未参照 |
| S0-1入力監査 | 8.2。既存コードで取得できる入力と、実機でしか確認できない入力を分離した |
| S0-2仕様案 | 8.3。層化変数・三分位・配分規則・seed・封印方式の具体案 |
| 変更範囲 | 8.4。再利用箇所、追加／変更ファイル、既存経路への影響 |
| 検証と実施量 | 8.5。合成テストと、実機CPU処理5回の対象・出力・停止条件 |
| 判断が必要な事項 | 8.6。初回提案5件と管理側の追加修正3件、計8件の判断を記載。返答・条件は8.6.1 |

### 8.1 現状・根拠

**読んだコード（確認済みの事実）**

- `Stage5/stage5/utils/file_list_mode.py`: 固定リスト検証、`list_content_sha256`（パス表記込み）と
  `list_identity_sha256`（ファイル名のみ）の2種hash、`compare_file_lists`による内容差と表記差の分離。
  依頼書4章が指定する「パス表現差を理由に実対象の相違まで許容しない」機構は既に存在する。
- `Stage5/train_stage5.py`: `compute_class_counts_from_h5()`（`annotation/point_label`と`valid_mask`を
  元H5から動画1回読み、`valid & label != ignore_index`のみ計数）、
  `pointnext_class_weights_from_counts()`（`1/(freq+eps)`、平均1へ正規化、float32化）、
  `resolve_class_weight()`。`resolve_train_val_paths()`の入力解決とdirectory-mode。
- `Stage5/train_stage5.sh`の`input_source_args()`: `TRAIN_LIST`/`VAL_LIST`が両方指定されたときだけ
  固定リストmodeになり、片側だけならエラー、未指定なら`--train_dir`＋`--val_fraction`のdirectory-modeへ戻る。
- `Stage5/evaluate_stage5.py`: 入力はリストのみ（directory-mode無し）。`select_train_paths()`は
  固定1動画＋`random.Random(seed).sample(candidates, num_random)`。
- `Stage5/infer_stage5.py`: `--input_h5`で単一H5を直接受ける。リスト・許可集合の概念が無い。
- `Stage5/checks/real_h5/check_stage5_gt_component_count.py`: frameごとのGT陽性点を`pixel_xy`上で
  単連結クラスタリングし、link距離を掃引して`multi_region_*`を判定する。
- `Stage5/checks/real_h5/check_stage5_xy_coordinate_provenance.py`: 最終H5の`source_pseudo3d_h5`属性から
  中間pseudo-3D H5を解決し、`raw_width/raw_height`・`local_crop_top/left`・`local_resize_scale`・
  `local_input_shape`を読む既存経路。
- `Stage2to4/src/utils/pseudo3d_processing.py`と`pseudo3d/inference/*_infer_video_to_pseudo3d_h5.py`:
  中間H5の`spacing`データセットの由来。

**確認済みの事実（実データを読まずに判明したもの）**

1. **train sanity 3動画はリスト順に依存して選ばれている。** `evaluate_stage5.py:36`の固定動画
   `20250626_124212_7300`に、`random.Random(42).sample(candidates, 2)`で2動画を加える。
   `candidates`は**train162リストの順序**である。したがってsanity 3の同一性はtrain162リスト順にのみ定義され、
   train_core144へ縮めた後に同じ規則を再実行すると別の動画が選ばれる。
   **sanity 3は再導出せず、S5-15 runが保存した`selected_train_files.txt`から明示リストとして固定する**。
2. **旧W-A class weightは、S5-15 runが記録した162動画のlabel統計から厳密には再現しない。**
   匿名化済み`training_config_anonymized.json`の`label_policy_diagnostics.train.totals`
   （positive 688,292／background 61,192,000）へ現行式（eps=0.02、正規化有効）を適用すると
   `[0.05985185, 1.94014815]`となり、記録値`[0.05963856, 1.94036150]`とbackground側で相対0.36%ずれる。
   逆算するとW-Aの算出元は陽性率約1.1012%、S5-15の162動画は約1.1123%で、母集団が一致しない。
   同runの`class_weight_info.mode`は`manual`（`requested: "0.05963856,1.94036150"`）であり、
   この値はrun内で算出されたものではなく外部から与えられている。
   → 旧W-Aの算出元記録は実機側にしか無い。**新値の検証に旧値との一致を使えない**ことを前提に設計する。
3. **臨床FLを扱うコードはリポジトリ内に存在しない。** `臨床FL`／`clinical_fl`の実装・スキーマ・
   読込経路のいずれも無い。形式・所在・単位は完全に実機確認事項である。
4. **mmスケールの実値はリポジトリ内のどの経路にも無い。** 中間pseudo-3D H5の`spacing`データセットは
   `--spacing_x/--spacing_y`（既定1.0、helpは "Temporary pixel spacing ... for visualization geometry"）が
   そのまま保存されたものである。さらに`make_pixel_to_image_matrix()`へ渡される`width/height`は
   **local crop後のplane寸法**であり、`dimensions`も`[plane_w, plane_h, 1]`である。
   つまり`spacing`は仮に実値が与えられていても「元frameのpixel」ではなく「crop/resize後のpixel」に対する
   スケールになる。pseudo-3D Z方向のmm換算が未確立であることはS5-16 9.1と整合する。
   → 8.6の判断事項4として総括管理へ返す。**Step 0で代理値を作らない。**
5. **匿名alias体系は21動画しか覆っていない。** `export_anonymized_stage5_metrics.py`が作る
   `video_id_map_DO_NOT_SHARE.csv`のaliasは`train_sanity_fixed_NNN`／`train_sanity_random_NNN`／
   `validation_NNN`のみで、internal_test候補159動画にaliasが無い。
   `check_stage5_gt_component_count.py`はこのCSV（`anonymous_id`/`split`/`original_h5_path`列）を必須入力にするため、
   **既存GT成分出力が159候補を網羅していないだけでなく、現状の入力では網羅できない**。依頼書2章の
   「既存GT成分出力が159候補を網羅するとは仮定しない」はこの形で具体化した。
6. **封印を強制する機構は現在どこにも無い。** `train_stage5.py`のdirectory-mode
   （`--train_dir` + glob + `val_fraction`による再分割）は、internal_testのH5が同一ディレクトリにある限り
   無言で取り込む。`infer_stage5.py`は任意の単一H5を受ける。`evaluate_stage5.py`はリスト専用だが
   リスト内容を検査しない。依頼書4章の「既存のdirectory-modeへ暗黙fallbackして封印集合を混ぜない」は
   新規のguardを追加しない限り満たせない。
7. **規模の見積り（匿名化済み統計からの推定、実測ではない）:** 162動画で全点62,118,669、
   715 window。1動画あたり約383,000点・約35 frame・約10,900点/frame、GT陽性は約4,250点/動画
   （GT frameあたり最大約120点）。GT成分クラスタリングはframeあたり約120点のN×N距離行列で済み、
   `MAX_POINTS_PER_FRAME=20000`には遠く及ばない。**計算量ではなくH5読込量（159動画で概算1.5〜2GB）が支配的。**

**実機確認待ち（本チャットで検証していない）**

`/mnt/data/3d_projects/`配下の実体、保存済みtrain162／validation18リストと
S5-15 runの`selected_train_files.txt`、v7 H5の実在と件数、臨床FLの所在・形式・単位、
W-A class weightの算出元記録、mmスケールの実際の供給元、CUDA環境。

### 8.2 S0-1 入力監査

| 入力 | 所在・形式 | 取得可否 | 不足時の確認手順 |
| --- | --- | --- | --- |
| train162リスト | 実機private。1行1 H5絶対パスのtxt。`file_list_mode.read_file_list`で読める | 実機のみ | パスをユーザーに確認。`list_content_sha256`と`list_identity_sha256`の両方を記録し、S5-15 run manifestの記録値と突合 |
| validation18リスト | 同上 | 実機のみ | 同上。**内容・順序を一切変更しない** |
| train sanity 3 | S5-15評価出力の`selected_train_files.txt`（3行） | 実機のみ | 見つからない場合のみ、train162リスト順に`select_train_paths(fixed="20250626_124212_7300", num_random=2, seed=42)`を**再現専用として**1回実行し、S5-15の匿名metricsのalias 3件と件数一致を確認。再導出したことを報告に明記する |
| v7 H5 | `*_..._bboxrank_v7_cvat_authoritative_crop_quality_v1.h5` | 実機のみ | `file_list_mode.validate_fixed_lists(h5_pattern=...)`で実在とteacher patternを一括検査 |
| GT multi-region情報 | **既存出力は存在しないか、あってもvalidation 18件のみ。** 159候補分は未算出 | **新規算出が必要** | 8.3-1の設定でcheckerを159候補へ1回実行。入力の`video_id_map`を159候補まで拡張する必要がある（8.4） |
| 臨床FL | リポジトリ内に一切無い。所在・形式・単位・動画対応キーすべて不明 | **実機・ユーザー回答が必要** | ファイル形式（CSV/Excel）、動画識別子の列と表記（H5 basenameか、`YYYYMMDD_HHMMSS_N`か）、単位（mm/cm）、欠損表現、1動画1値か複数計測値かを回答してもらう。**実値は共有側へ持ち込まない** |
| x/y別mmスケール | 中間H5の`spacing`は`--spacing_x/y`既定1.0の「visualization用の暫定値」で、かつcrop後pixel基準 | **実質取得不可の可能性** | 8.6-4。実際の供給元（装置出力、DICOM、計測ソフト、手動記録）をユーザーに確認 |
| crop/resize逆変換メタ | 中間pseudo-3D H5のattr `raw_width`/`raw_height`/`local_crop_top`/`local_crop_left`/`local_resize_scale`/`local_input_shape`。最終H5の`source_pseudo3d_h5`から解決 | **既存経路で取得可能** | `check_stage5_xy_coordinate_provenance.py`の読取り関数を再利用し、train_core144で欠落・解決不能を件数で報告 |
| W-A算出元記録 | 実機のrun config／ログ | 実機のみ | 8.1-2のずれを踏まえ、算出元の動画集合とteacher版を確認。**一致しなくても新値を調整しない**（依頼書5章） |

### 8.3 S0-2 仕様案（結果を見る前に1仕様へ固定する）

#### 1. GT multi-regionの定義

- 使用checker: `Stage5/checks/real_h5/check_stage5_gt_component_count.py`。版はgit revisionとファイルのSHA-256で固定。
- 出力field: `classification.multi_region_any_frame_all_radii` を**唯一の層化変数**にする。
- 設定: `link_distances = 2,3,4,6,8,12`（既定）、`min_component_points = 5`（既定）。
- 選定理由: 依頼書3章の「複数領域frameが存在する」と「割合が閾値以上」の混同を避けるため、
  **frame割合の閾値（`frame_fraction_threshold`）に依存しない指標**を選ぶ。checker自身のコメントが
  S5-15データは0%か11%以上に二分されると記録しており、割合閾値の置き場所が結果を決めてしまう。
  `_all_radii`を選ぶのは、link距離1つだけで成立する分類を採らないという同checkerの設計方針に従うため。
- 対象・CPU量: 159候補動画のみ（validation 18とsanity 3は算出しない）。読込は`pixel_xy`・`frame_order`・
  `point_label`・`valid_mask`の4配列のみ、1回のpassで6半径を評価する。8.1-7の見積りでCPU数分〜十数分、
  追加メモリはframeあたり約120点の距離行列で無視できる。GPU不要。

#### 2. 臨床FL三分位

- 母集団: **159候補のみ**。validation 18とsanity 3は読まない（依頼書2・6章、S5-16 10.0「参照FL」）。
- 方式: 値そのものの分位点ではなく**順位ベースの三分割**にする。159件を`(FL値, train162リスト内index)`で
  昇順整列し、順位で3群（53/53/53）へ切る。
  - 選定理由: 境界値の同値問題が原理的に発生せず、丸め・浮動小数の実装差にも依存しない。
    件数が層で揃うため8.3-3の配分も安定する。境界のFL実値はprivate側にのみ記録する。
- 欠損: **独立した第4層`FL_missing`として扱う**（推奨、8.6-2）。推定補完しない。除外もしない。
- 単位不整合: FL値の単位をユーザー申告と突合し、**単位が確定できない、または動画内で混在する場合は停止**する。
  cm表記が混ざっている等を自動判定で正規化しない。
- 対応付け: 動画識別子の突合は完全一致のみ。前方一致・正規化・手動修正を行わない。
  対応しない行・重複行が1件でもあれば停止して件数を報告する。

#### 3. 層から18件を配分する規則

- 層 = multi-region 2値 × FL群 3（＋欠損があれば4） = 6または8セル。セルは`(multi_region, fl_group)`の
  昇順キーで固定順に並べる。
- 配分: 最大剰余法。`quota_k = 18 * n_k / 159` を計算し、整数部を配分、残りを小数部の大きい順に1件ずつ配る。
  **小数部が同値の場合はセルキーの昇順で先に配る**（乱数を使わない）。
- 小さい層・空の層: `alloc_k <= n_k` を強制する。空の層は0件。上限に当たって余った枠は、
  余力のあるセルへ同じセルキー昇順で再配分する。再配分が完結しない（総数18に満たない）場合は停止する。
- 入力順序: すべてtrain162リストの記載順を正本にする。ディレクトリ走査順・ファイルシステム順を使わない。
- 乱数器・seed: `numpy.random.default_rng(seed)`を1個だけ作り、セルキー昇順にループして
  `rng.choice(len(members), size=alloc_k, replace=False)`で選ぶ（`members`はtrain162順で整列済み）。
  **seedは42を提案する**が、S5-16でsplit seedの具体値が決定済みだとは記録しない（8.6-1）。
- 探索の禁止: 生成は1回。別seedでの成績比較・見栄え比較は行わない。異常停止後の再実行は理由を報告し、
  同じseedで再実行する。

#### 4. 実行・封印

- 新規private出力先（実機）: `/mnt/data/3d_projects/stage5_splits/s5_16_step0_bprime/`
  - `train_core_144.txt` / `validation_18.txt`（旧リストからの写し、内容・順序不変）
  - `split_manifest.json`: 3リストの`content_sha256`と`identity_sha256`、teacher版、入力リストhash、
    層化仕様（checker版・設定・FL群定義・配分規則・seed）、コードrevision/hash、生成日時、`sealed_until: "S5-20c"`
  - `sealed/internal_test_18.txt` と `sealed/stratum_stats_DO_NOT_SHARE.json`（層別GT統計・FL分布）
    は下位ディレクトリへ分離し、実機でmode 0500/0400・所有者限定にする（chmod/chownはユーザー実行）
- 共有側へ出すのは**件数・hash・検査成否のみ**。internal_testの層別GT統計・FL分布・動画別値は
  確定後に人間が読むレポートへ出さない（依頼書3章末）。stdout・例外メッセージ・一時ファイルにも同じ境界を適用し、
  例外は動画名でなくalias・index・件数で報告する。
- 実施上限: 正常なsplit確定は1回。異常停止時は原因と停止段階を報告してから、同一seed・同一入力hashで再実行する。
- 旧リスト・H5は一切変更しない（読取り専用でオープンする）。
- 契約検査（生成直後に実行し、失敗したら出力を確定しない）:
  1. 3集合が相互に素
  2. 和集合の`identity`が旧180動画と一致
  3. `train_core ∪ internal_test`の`identity`が旧train162と一致
  4. sanity 3が全てtrain_coreに含まれる
  5. 件数が144/18/18
  6. validation18の内容と**順序**が旧リストと完全一致
  - 照合は`list_identity_sha256`（動画同一性）と`list_content_sha256`（パス表記）を併記し、
    `compare_file_lists`の`representation_differs_only`で表記差と内容差を分けて報告する。
- 旧checkpoint契約: manifestに「学習集合の`identity_sha256`が`train_core_144`と一致しないcheckpointは
  internal_testでの評価に使えない」という規則をS5-20cまで有効なものとして記録する。
  旧R0 50 epoch・W-Aはこれに該当し恒久的に不可であることを明記する。20cの解除・評価処理は実装しない。

### 8.4 変更範囲

**再利用（変更しない）**

- `stage5/utils/file_list_mode.py`のhash・比較・検証関数
- `checks/real_h5/check_stage5_gt_component_count.py`の成分計数ロジック（**コード変更なしで使う**）
- `checks/real_h5/check_stage5_xy_coordinate_provenance.py`の中間H5解決・attr読取り
- `checks/real_h5/write_stage5_s5_15_run_manifest.py`のgit状態収集・privacy self-check様式

**追加ファイル（案）**

| ファイル | 内容 |
| --- | --- |
| `Stage5/stage5/utils/class_weights.py` | `compute_class_counts_from_h5()`と`pointnext_class_weights_from_counts()`を`train_stage5.py`から**そのまま移設**し、`train_stage5.py`はここからimportする。式・既定値・戻り値型は不変 |
| `Stage5/stage5/utils/split_contract.py` | `split_manifest.json`の読込、8.3-4の契約検査、`assert_paths_allowed(paths, allowed)`。**H5を開く前に**例外を投げる。stdlibのみ |
| `Stage5/checks/real_h5/build_stage5_bprime_split.py` | split生成CLI。層化変数の読込・三分位・配分・契約検査・manifest書き出し・封印ディレクトリ作成。**層化変数へアクセスする唯一の処理** |
| `Stage5/checks/real_h5/build_stage5_train_pool_video_id_map.py` | 159候補のalias付きCSV（`anonymous_id`/`split`/`original_h5_path`）を生成。alias は`train_pool_NNN`（train162リスト順の連番）。既存の`validation_NNN`等の体系を変更せず、新しい名前空間を1つ足すだけにする |
| `Stage5/checks/real_h5/compute_stage5_train_core_class_weight.py` | train_core144リストを明示入力に1回だけ算出。torch非依存（`class_weights.py`とh5py/numpyのみ） |
| `Stage5/checks/real_h5/audit_stage5_train_core_mm_metadata.py` | S0-6。train_core144の中間H5メタを集計し、欠落・解決不能を件数で報告 |
| `Stage5/checks/dummy/check_dummy_split_contract.{py,sh}` | 8.5の合成テスト |
| `Stage5/checks/dummy/check_dummy_train_core_class_weight.{py,sh}` | 純粋関数の合成検証 |

**変更ファイル（最小）**

| ファイル | 変更 |
| --- | --- |
| `train_stage5.py` | (a) class weight 2関数を`stage5/utils/class_weights.py`からimportする形へ置換（振る舞い不変）。(b) `resolve_train_val_paths()`の直後・dataset構築とclass weight計数の**前**に`assert_paths_allowed()`を挿入。(c) manifestが封印有効を宣言している場合、`--train_dir`/`--val_dir`によるdirectory-modeを拒否する |
| `evaluate_stage5.py` | リスト読込直後に`assert_paths_allowed()`を挿入 |
| `infer_stage5.py` | `--input_h5`をH5オープン前に`assert_paths_allowed()`へ通す |
| `docs/stage5/FILES.md` | 追加ファイルの索引。**実装確定後に更新** |
| `docs/stage5/data_construct.md` | 新private成果物構造。**実装確定後に更新** |

**既存経路への影響**

- `split_manifest.json`が指定されていない場合、`assert_paths_allowed()`は何も拒否せず既存の振る舞いのままとする。
  これによりS5-15以前の再現性と既存checkの回帰が保たれる。封印を有効にするのは manifest パスを明示したときだけ。
- class weight関数の移設は同一モジュール内の再配置であり、`resolve_class_weight()`の呼び出し規約・
  `class_weight_info`のフィールド・既定値`auto_class_weight_epsilon=0.02`・正規化既定Trueを変更しない。
- 封印はフラグ保存だけで完了とせず、(1) 実機のディレクトリ分離と権限、(2) `assert_paths_allowed()`による
  読込前拒否、(3) manifestの契約記録、の3点を揃えて初めて成立とする（依頼書4章）。

### 8.5 検証と実施量

**合成テスト（実データを使わない。CPU。すべて`checks/dummy/`）**

split契約側: 重複エントリ／欠落ファイル／件数違い（143や19）／sanity動画がtrain_coreから外れる／
不正hash（content・identityそれぞれ）／封印集合のパス要求／旧checkpoint（学習集合hash不一致）の要求／
片側リストのみの指定／directory-modeへのfallback試行 — **以上すべてが例外で停止すること**を確認する。
正常系: 同一入力・同一seedで2回生成したリストがbyte一致すること、配分が18件・層ごとの上限を超えないこと、
空の層と剰余同値で乱数を使わずに決着すること、manifest無しなら既存経路が不変であること。
`assert_paths_allowed()`が**H5を開く前に**投げることは、存在しないパスを封印集合として与え
`FileNotFoundError`ではなく封印例外が出ることで示す。**実internal_testを読み出して拒否動作を試さない。**

class weight側: 既知counts→期待weightの一致（`1/(freq+eps)`・平均1正規化・float32）、
`valid_mask=False`とignore点が計数から外れること、対象外の動画を渡すと停止すること、
空集合・全ignoreで`ValueError`になること、overlap windowを経由せず元H5から動画1回だけ数えること。

**実機CPU処理（read-only。GPU不要。新規書き込みは新private出力先のみ）**

| # | 対象 | 回数 | 出力 | 停止条件 |
| --- | --- | --- | --- | --- |
| 1 | 159候補のalias map生成 | 1 | private CSV | train162リストとsanity 3の差集合が159件にならない／既存aliasと衝突 |
| 2 | 159候補のGT成分計数 | 1（修正後の再実行1回まで） | private JSON＋共有用集計JSON | H5欠落、`frame_order`とpixel_xyの行数不一致、link距離間で分類が不安定な動画が出た場合は停止せず件数を報告 |
| 3 | 臨床FL読込＋三分位＋split生成＋契約検査＋封印 | 1 | 8.3-4の出力先一式 | 動画対応の欠落・重複、単位不確定、配分不成立、契約検査6項目のいずれか不合格 |
| 4 | train_core144のclass weight算出 | 1 | private JSON（counts・weight・対象hash・code hash・設定） | 対象リストhashがmanifestと不一致、valid点0、範囲外label |
| 5 | train_core144のmmメタ監査 | 1 | private JSON＋共有用件数集計 | 中間H5が解決できない、`local_input_shape`が不正／非正方、**mmスケールの実体が確認できない場合は「取得不能」として報告し、代理値を作らない** |

合計5回（#2の再実行を含めて最大6回）。いずれも読取り専用で、既存のH5・リスト・run成果物を変更しない。
処理順は 1 → 2 → 3 → 4 → 5 で、#3の契約検査に合格するまで#4・#5へ進まない。

### 8.6 判断が必要な事項

| # | 事項 | 推奨案と理由（初回提案／追加修正案） | 判断 |
| --- | --- | --- | --- |
| 1 | split seedの具体値 | **42を採用**。プロジェクト全体の固定seedと一致し（S5-16 10.0）、新しい任意値を持ち込まない。ただしS5-16文書には未記録のため、総括管理で明示的に確定してほしい。確定後は別seedを試さない | 採用。split seedを42に固定する。入力順序・乱数器・配分規則・生成リストhashも固定し、seed探索は行わない。 |
| 2 | 臨床FL欠損動画の扱い | **独立した第4層`FL_missing`として層化に含める**。除外すると159件が崩れて旧train162との一致検査が成立せず、依頼書2章の「都合よく動画を除外して件数を合わせない」にも反する。推定補完はS5-16 10.1のfail-fast方針に反する | 条件付き採用。非欠損だけで3群を作り、欠損は別群とする。値の欠損と動画対応不明を区別し、FL層化が成立しない場合は停止・報告する。 |
| 3 | GT成分checkerの入力拡張方法 | **checkerを変更せず、alias mapを159候補へ拡張する**（8.4の`build_stage5_train_pool_video_id_map.py`）。監査済みcheckerへ`--h5_list`を足すより既存経路への影響が小さく、`check_stage5_brightness_position_vs_prediction.py`が`read_video_id_map`を共有している点にも触れずに済む | 条件付き採用。既存ロジックを再利用し、159候補の計数はsplit構築の内部処理へ組み込む。候補GT統計の共有出力は認めない。 |
| 4 | **mmスケールの実体（S5-17の開始条件に直結）** | 8.1-4のとおり、リポジトリ内の`spacing`は「visualization用の暫定値（既定1.0）」であり、しかもcrop後pixel基準である。**実mmスケールの供給元をユーザーに確認する必要がある**。確認できるまでStep 0では「取得不能」と記録し、代理FLも`T_FL`も推定しない（依頼書6章）。S5-17の未決事項として引き継ぐことを提案する | 供給元の確認待ち。現時点は「取得不能」ではなく「未確認」。分割等の準備は進められるが、mm評価の開始条件は未充足。 |
| 5 | 旧W-A算出元の不一致（8.1-2） | 実機の算出元記録との照合を#4の実施時に行い、**一致しなくても新weightを調整しない**（依頼書5章）。差は履歴として報告するだけにする。照合の結果「算出元記録が実機にも無い」場合は、その事実を報告してStep 0を進めることを提案する | 条件付き採用。由来を監査し、新weightは旧値へ合わせない。原因は未確定。記録不在時は確認済み算出仕様を明示して管理へ返す。 |
| 6 | manifest省略による封印回避（管理側追加） | 8.4はmanifest未指定時にguardを無効化するため、指定漏れで旧directory-mode等から封印集合を取り込める。今後の開発経路でmanifestを必須にする。 | 要修正。通常のStep 0以降の開発・学習・評価経路はmanifest未指定／不正時に停止する。旧経路の互換性と封印の迂回を分離して設計する。 |
| 7 | 実機権限と封印の成立条件（管理側追加） | リストのディレクトリ分離と0500/0400だけでは所有者による読取りや元H5へのアクセスを防げない。実行経路・アクセス制限・閲覧運用を合わせて定義する。 | 要修正。何を誰から制限するか、元H5・中間出力・ログの扱い、実行主体と権限を具体化する。権限値やsealedフラグだけで封印完了としない。 |
| 8 | train sanity3の復元証拠（管理側追加） | 保存済みselected_train_files.txtを優先する。代替再導出時のalias3件・件数一致だけでは同一動画を証明できない。 | 要修正。旧リスト順・設定による再現に加え、旧評価のprivate対応記録等と動画identityを照合する。証拠不足なら分割を確定せず報告する。 |

初回の推奨案は履歴として保持する。採用条件・訂正・実施境界は次の管理返信を優先する。

#### 8.6.1 判断事項への返答（総括管理チャット、2026-09-22）

作成元: S5-16〜S5-20総括管理チャット。対象: 8.6の8事項。  
状態: **管理判断・修正条件の記録。実装計画の一括承認ではない。**  
記録根拠: ユーザーの依頼により初回報告を精査し、その返答と追加修正事項を本節へ記録する。
8.1〜8.5の原文は提案履歴として保持する。以下と相違する説明・仕様は修正版計画で訂正する。

**1. split seedは42を採用する。**

42に統計的な優位性があるという判断ではなく、結果を見る前に1つへ固定してsplit探索を避けるための決定である。
学習seedとは別設定として、候補リストの順序、セル順、乱数器・ライブラリ版、配分規則を記録する。
生成したfile listとhashを正本とし、別seedを試して成績や見栄えで選び直さない。

**2. FL_missingは条件付きで採用する。**

- 明示的なFL欠損を推定補完・除外せず、独立群として層化する。
- 53/53/53の順位分割は159件すべてにFLがある場合だけ成立する。欠損がある場合は、非欠損動画だけを
  ほぼ同数の3群へ分け、端数の割当規則を事前固定する。欠損群を含む全候補159件から18件を配分する。
- 同じFL値が異なる群に入る場合があるため、「同値問題が原理的に発生しない」という説明は訂正する。
  同値は固定のtrainリスト順で振り分ける。同値の解決と、値の精度・単位・浮動小数処理の問題を区別する。
- 動画との対応が確定していて値だけ欠損する場合と、識別子不一致・重複・行欠落で対応を確定できない場合を分ける。
  対応不明を無条件にFL_missingへ落とさず、入力不備として停止する。明示的欠損の入力表現を定義する。
- 全件欠損や非欠損が少なく3群を作れない場合は、別方式へ自動fallbackせず停止・報告する。
  欠損が多い場合のS5-17／20cの評価可能件数への影響も示し、必要な判断を返す。
  この確認で封印後のinternal_testのFL分布や動画別値を共有しない。

**3. GT成分checkerの既存ロジック再利用を条件付きで採用する。**

159候補用のalias mapはprivate入力として作成する。既存checkerを変更しない方針は維持できるが、
計数を独立した探索診断にせず、split構築処理から呼び出す内部工程にする。
159候補には将来のinternal_testが含まれるため、8.5の「共有用集計JSON」をそのまま出す案は採用しない。
人間が候補動画のGT統計を見て分割を選ばず、split確定後は候補全体の中間出力・stdout・ログからも
internal_test情報を閲覧できない扱いにする。共有は許可された件数・hash・検査成否に限定する。
既存checkerの匿名化チェック合格は、封印情報の共有許可とは異なる。呼出方法と出力抑制・保管方法を
修正版で示す。匿名化と封印は別の要件として検証する。

**4. mmスケールは供給元の確認待ちとする。**

コード上のspacingが可視化用の既定値を持つことは確認できるが、実機の供給元が存在しない証拠ではない。
S5-16での「縦・横の実mmスケールを取得できる」というユーザー回答を維持し、現状を「未確認」と記録する。
確認対象は、供給元・保存形式、mm/pixelか画像全体の実寸か、元frameかcrop/resize後か、
x/y別の値、動画単位かframe単位か、動画・frameへの対応方法である。確認後に取得不能と分かったものだけ
理由を付けて「取得不能」とする。spacing=1.0を実測値とみなさず、代理FLやT_FLを推定しない。
分割・封印・weight算出の実装準備は並行して進められる。mm評価を伴うS5-17の実行前には解決が必要である。
Step 0で未解決なら、対応方針とS5-17開始条件を明示して受入判断を返し、確認完了とはしない。

**5. 旧W-Aの由来を監査し、新weightを旧値へ合わせない。**

確認されたのは「共有された162動画の統計と現行式から旧固定値を厳密に再現できない」という事実である。
母集団の違いは候補であり、teacher版、label計数方法、設定、転記経緯等も未確認のため、
8.1-2の「母集団が一致しない」という原因の断定は留保する。
旧値の由来の監査と、新しいtrain_core用の算出仕様の固定を分ける。
旧記録が見つからない場合も、確認できた式・valid/ignore処理・epsilon・正規化・数値精度を明示し、
新baselineとして固定する案を管理へ返せる。ただし「W-Aと同一ロジックを完全確認した」とは記録しない。
由来不明の事実だけを理由に無期限に止めず、確認済み仕様と未確認範囲を明示して継続可否を判断する。
旧値との差を縮めるためのweight調整・探索は認めない。

**6. manifest省略時に封印を迂回できる案は修正する。**

8.4の「manifest未指定なら従来どおり何も拒否しない」は、今後の通常開発経路の仕様として採用しない。
Step 0以降の学習・評価・推論・関連checkerでは、必要なmanifestが未指定・不正ならH5読込前に停止する。
旧経路の再現性を保つ必要があっても、通常の起動方法や指定漏れから封印を迂回できない構成を示す。
CLIだけでなくbash／preflight等の先行読込も確認し、guardの適用箇所と境界を明示する。
合成テストには、manifest省略・不正・directory-modeからの混入・封印集合への直接指定を含める。

**7. 封印の成立条件を具体化する。**

0500/0400は所有者の読取りを許す。リストを別ディレクトリに置いても元H5へのアクセスは残る。
したがって、権限値・sealedフラグ・manifestの保存だけで封印完了とはしない。
何を誰から制限するか、実行ユーザー／プロセスの権限、元H5への経路、候補全体の中間統計・ログ、
人間による閲覧を避ける運用を組み合わせ、担保する範囲と残る制限を報告する。
実機の権限変更を無断で実行せず、旧成果物を破壊しない具体策を提示する。
20cでの解除・評価そのものの先行実装は不要だが、旧学習済みcheckpointの評価禁止は解除後も維持する。

**8. train sanity3の同一性を動画単位で証明する。**

S5-15評価時の保存済みselected_train_files.txtを第一選択とする。再導出が必要な場合は、
旧train162リストの内容・順序・seed・固定動画指定を再現し、旧評価のprivate対応表・保存リスト等で
動画identityを照合する。aliasが3件存在することや件数一致だけを証明にしない。
照合証拠が不足する場合はsplitを確定せず、確認できた範囲と不足を報告する。実IDは共有文書へ出さない。

**実装担当への次の依頼と実施境界**

以上を反映した修正版計画を、本報告書の新しい節として追記する。少なくともFL欠損の分岐、
GT計数から封印までの処理・出力境界、manifest必須化の範囲、実機封印方法、sanity照合証拠を示す。
処理の統合による変更ファイル・合成テスト・実機CPU回数も更新し、旧提案の5回／最大6回を
確定した承認量として流用しない。実機情報が未取得のものは未確認のまま記録する。
本返信で確定した事項について再度同じ承認を求める必要はないが、計画全体の実装・実機実行の
一括承認とはしない。修正条件を満たす計画を確認してから実装へ進む。

### 8.7 本節の位置づけ

8.1〜8.5は**初回提案の履歴**であり、実装・実機実行・コミットは行っていない。
8.6の8件への現行回答は8.6.1とする。次は修正版計画を新しい節へ追記し、管理確認後に
承認範囲内の実装・合成テストへ進む。旧8.4・8.5をそのまま承認済み仕様として扱わない。
実機CPU処理の結果（日付・承認範囲・コードrevision・環境・入力／出力hash・成否・逸脱・未決事項）は
本報告書のさらに新しい節へ追記し、提案・承認・実装・実行完了を区別する。
`FILES.md`と`data_construct.md`の更新は実装配置が確定した時点で行う。

## 9. Step 0実装チャットの修正版計画（2026-09-22）

担当: Step 0実装チャット。状態: **修正版計画（提案）。実装・実機実行・コミットは未実施。**
対象: 8.6.1「判断事項への返答」の8項目と、同節末尾の「実装担当への次の依頼と実施境界」。
8.1〜8.5は初回提案の履歴として保持し、本節と相違する箇所は本節を優先する。
8.6.1で確定した事項について再度の承認は求めない。9.8〜9.10の実施量と、9.11の新規事項について確認を求める。

### 9.1 初回報告の訂正

管理返答を受けて、初回報告の次の記述を訂正する。訂正内容は以後の計画に反映済みである。

| 箇所 | 初回の記述 | 訂正 |
| --- | --- | --- |
| 8.3-2 | 順位ベース三分割は「境界値の同値問題が原理的に発生しない」 | **誤り。** 同一FL値が異なる群へ分かれる場合がある。同値はtrain162リスト順で振り分ける規則を別途固定する（9.2）。順位分割が回避するのは分位点の計算方式・丸め・浮動小数表現への依存であって、同値そのものではない |
| 8.3-2 | 159件を53/53/53へ切る | **欠損がある場合は成立しない。** 非欠損m件を3群へ分け、欠損を第4群とする（9.2） |
| 8.1-2 | W-Aの算出元は「母集団が一致しない」 | **原因の断定を撤回。** 確認された事実は「共有された162動画統計と現行式から旧固定値を厳密に再現できない」ことのみ。母集団差・teacher版・計数方法・設定・転記経緯のいずれも未確認である（9.7-3） |
| 8.1-4／8.6-4 | mmスケールは「実質取得不可の可能性」 | **「未確認」に改める。** コード上のspacingが可視化用既定値であることは、実機に供給元が無い証拠ではない。S5-16 9.1のユーザー回答を維持する（9.7-2） |
| 8.4 | manifest未指定時はguardを無効化し既存の振る舞いを保つ | **採用しない。** 通常経路はmanifest未指定・不正で停止する。旧run再現はguardを外すのではなくlegacy manifestを通す（9.4） |
| 8.5 | 159候補GT成分計数の「共有用集計JSON」を出す | **採用しない。** 候補GT統計の共有出力を行わない（9.3） |
| 8.3-4 | ディレクトリ分離と0500/0400で封印する | **不十分。** 所有者の読取りと元H5への経路が残る。担保範囲と残存リスクを分けて記述する（9.5） |
| 8.4 | 追加ファイルに`build_stage5_train_pool_video_id_map.py` | **廃止。** alias付与とGT計数をsplit構築処理へ統合する（9.3、9.8） |

### 9.2 臨床FL層化の分岐（管理2）

#### 入力の受入と停止条件

FL入力は「動画識別子」と「FL値」を持つ表（形式は実機確認、9.7-1）とする。処理は次の順で行い、
**分類できない行が1件でもあれば層化を行わず停止する**。自動補正・前方一致・正規化は行わない。

| 区分 | 定義 | 扱い |
| --- | --- | --- |
| 正常 | 候補159件のいずれかと識別子が完全一致する行が**ちょうど1行**あり、値が数値として解釈できる | FL値を採用 |
| 明示的欠損 | 対応行がちょうど1行あり、値のセルが空白のみ、または入力監査時に確定した欠損トークン集合に一致する | `FL_missing`群 |
| 対応不確定 | 候補に対応行が無い／2行以上ある／候補外の識別子が現れる | **停止。** `FL_missing`へ落とさない |
| 値の不正 | 対応行はあるが、値が欠損トークンでも数値でもない | **停止。** 欠損とみなさない |
| 単位不確定 | ユーザー申告の単位と入力の単位が一致しない、単位列が混在する、単位が判別できない | **停止。** cm→mm等の自動換算を行わない |

欠損トークン集合は入力監査（9.10 #1）で実ファイルを見て確定し、manifestへ記録する。
**split構築の実行時に新しいトークンを追加しない。** 追加が必要と判明した場合は停止して管理へ返す。

#### 非欠損m件の3分割

- `q = m // 3`、`r = m % 3`。群サイズは低位側から `[q+1] * r + [q] * (3 - r)`。
  端数は**低いFL群へ先に配る**。この規則を結果を見る前に固定する。
- 整列キーは `(FL値, train162リスト内のindex)` の昇順。同値は**train162リスト順**で先の動画が低い群へ入る。
  乱数を使わない。
- 成立条件: `m >= 3` かつ全群サイズ >= 1。満たさない場合は**停止・報告**し、
  別方式（二分割、FL層化の省略、閾値変更等）へ自動fallbackしない。
- 境界となったFL実値はprivate記録にのみ残す。

#### 層と配分

- 層 = `multi_region`（2値）×`fl_group`（非欠損3群＋欠損があれば`FL_missing`）＝最大8セル。
- 配分母集団は**候補159件全体**（欠損群を含む）。18件を8.3-3の最大剰余法・セルキー昇順tie-break・
  上限`alloc_k <= n_k`・余力セルへの再配分で割り当てる。総数18に満たない場合は停止する。

#### 欠損が多い場合の影響報告

FL欠損は、S5-17の代理FL定義選定（train_coreのみ）とS5-20cの臨床FL照合（internal_test）の
**評価可能件数**を直接減らす。したがって層化を確定する前に、9.10 #2aで次だけを管理へ返す。

- 候補159件中のFL欠損件数、非欠損件数m、3群サイズ、3群が構成可能か
- train_core側／internal_test側の内訳は**返さない**

理由: 層別の配分は層サイズから決定的に定まるため、`multi_region`との**クロス表**やセル別配分を共有すると、
確定後にinternal_testの層別GT統計・FL分布が復元できてしまう（依頼書3章が禁じている内容そのものになる）。
FL側の周辺度数のみでは、`multi_region`の周辺度数を共有しない限りクロス表は定まらない。
そこで**FLの周辺度数だけをsplit確定前に1回共有し、`multi_region`の周辺度数・クロス表・セル別配分は
共有しない**。この共有はsplit確定前の1回に限り、確定後に繰り返さない。

### 9.3 GT計数から封印までの処理・出力境界（管理3）

#### 処理の統合

`check_stage5_gt_component_count.py`は**変更しない**。split構築処理が同モジュールから
`read_gt_points` / `component_sizes` / `analyze_video` / `classify_video` を import して呼ぶ
（同ファイルは`if __name__ == "__main__"`で保護されているためimportで副作用は起きない）。
同checkerの`main()`とCLIは使わない。したがって`--video_id_map`形式のCSVを作る必要が無くなり、
初回提案の`build_stage5_train_pool_video_id_map.py`は廃止する。
aliasは`train_pool_NNN`（train162リスト順の連番）としてsplit構築処理が内部で付与し、
private記録と例外メッセージにのみ使う。既存の`validation_NNN`等の名前空間は変更しない。

設定は8.3-1のまま（`link_distances = 2,3,4,6,8,12`、`min_component_points = 5`、
層化変数は`multi_region_any_frame_all_radii`）。checkerのgit revisionとファイルSHA-256をmanifestへ記録する。

#### 出力境界

| 出力 | 保存先 | 共有 |
| --- | --- | --- |
| 候補159件の動画別GT成分計数・per-frame詳細 | 封印ディレクトリ内のみ | **不可** |
| 候補159件のFL値・三分位境界・セル別配分表 | 封印ディレクトリ内のみ | **不可** |
| FLの周辺度数（欠損件数・m・3群サイズ） | 封印ディレクトリ内＋split確定前の管理報告 | 確定前の1回のみ可（9.2） |
| 3リストの`content_sha256`／`identity_sha256`、件数、teacher版、仕様、code hash、契約検査6項目の成否 | manifest | 可 |
| internal_testの層別GT統計・FL分布・動画別値 | 封印ディレクトリ内のみ | **不可（S5-20cまで）** |

- 中間生成物は最初から封印ディレクトリ内へ書く。`/tmp`・作業ディレクトリ・`TMPDIR`へ一旦出して移動しない。
- stdoutは工程名・件数・hash・検査成否のみ。動画別のGT統計・FL値・実IDを出さない。
- 例外メッセージはalias・index・件数で記述する。実パス・実IDを含めない。
- ログファイルを作る場合も封印ディレクトリ内へ置く。
- 既存checkerの匿名化self-checkに合格することは、封印情報の共有許可とは別要件である。
  **匿名化（実IDを出さない）と封印（統計そのものを出さない）を別々に検証する**（9.9）。

### 9.4 manifest必須化の範囲（管理6）

#### 方針: guardを外す経路を作らず、すべてmanifestを通す

`split_manifest.json`の解決順は (1) CLIの`--split_manifest`、(2) 環境変数`STAGE5_SPLIT_MANIFEST`。
**どちらでも解決できない、または署名（manifest自身のSHA-256）・契約検査が通らない場合は、
H5を1つも開かずに停止する。** guardを無効化するフラグは設けない。

manifestの主要field（案）:

| field | 用途 |
| --- | --- |
| `schema` | `stage5_split_contract_v1` |
| `lists` | `train_core` / `validation` / `internal_test` それぞれの`content_sha256`・`identity_sha256`・件数・パス |
| `sealed` | 封印対象split名の配列。空配列は「封印対象なし」を**明示**する |
| `sealed_until` | `"S5-20c"`。解除条件の記録であり、解除処理は実装しない |
| `allows_new_training` | 新規学習でこのmanifestを使ってよいか |
| `allows_directory_mode` | directory-modeを許すか。B′では常に`false` |
| `stratification` | checker版・設定・FL群定義・配分規則・seed・入力hash |
| `prohibited_checkpoint_rule` | 学習集合の`identity_sha256`が`train_core`と不一致のcheckpointはinternal_test評価に使えない |

#### 旧経路の再現性と封印の迂回の分離

旧runの再現は、guardを外すのではなく**legacy manifest**を通す。
`split_contract.py`のCLIが、旧train162／validation18リストから
`sealed: []`・`allows_new_training: false`・`allows_directory_mode: false`のmanifestを生成する。

- 再現実行はguardを通過するが、`allows_new_training: false`のため**新規学習には使えない**。
  S5-17以降の新規学習でlegacy manifestを指定すると停止する。
- これにより「guardを通らない経路」が1つも残らない。指定漏れは無効化ではなく停止になる。
- `checks/real_h5/run_stage5_s5_15_arm.sh`・`run_stage5_s5_15_r0_longrun.sh`は旧条件の再現用launcherであり、
  legacy manifestを指定するよう更新する（S5-15条件そのものは変更しない）。

#### guardの適用箇所（H5オープン前であることを個別に確認する）

| 経路 | 挿入位置 | 備考 |
| --- | --- | --- |
| `train_stage5.sh` | `file_list_mode.py`検証の直後、**teacher preflightの前** | preflightは`LIST_MANIFEST`の全H5を開く。ここを通さないとguardより先にH5が開かれる |
| `train_stage5.sh`の`LIST_MANIFEST` | `mktemp`先を`${TMPDIR:-/tmp}`から封印運用下のディレクトリへ変更 | 現状は実H5絶対パス一覧を`/tmp`へ書く。依頼書3章「一時ファイルも同じ境界を守る」に抵触する |
| `train_stage5.py` | `resolve_train_val_paths()`直後、dataset構築とclass weight計数の**前** | `--train_dir`/`--val_dir`は`allows_directory_mode: false`なら拒否 |
| `evaluate_stage5.sh` / `evaluate_stage5.py` | リスト読込直後、`predict_h5`の前 | `RUN_DIR/train_files.txt`等も検査対象にする |
| `infer_stage5.sh` / `infer_stage5.py` | `--input_h5`のオープン前 | 単一H5指定の直接経路 |
| `checks/real_h5/`の実データcheck | 各スクリプトのH5オープン前 | 対象を一覧化して個別に確認する（9.11-3） |

### 9.5 実機封印の方法と担保範囲（管理7）

封印は**技術的guard・出力境界・運用**の3層で構成し、各層が何を誰から守るかと、残る制限を分けて記述する。

| # | 守る対象 | 相手 | 手段 | 担保 |
| --- | --- | --- | --- | --- |
| 1 | internal_testのH5点データ | 学習・評価・推論・checkerのコード | `split_contract`によるH5オープン前拒否（9.4） | **技術的に担保。** 合成テストで検証（9.9） |
| 2 | 同上 | directory scan・glob | manifest有効時のdirectory-mode拒否、`allows_directory_mode: false` | 技術的に担保 |
| 3 | internal_testのGT統計・FL値・動画別指標 | 開発者（人間）の閲覧 | 封印ディレクトリへの隔離、レポート・stdout・ログ・一時ファイルからの除外（9.3） | **技術と運用の併用。** 所有者権限では技術的に防げない（後述） |
| 4 | 同上 | 集計値からの復元 | 周辺度数のみ共有、クロス表・セル別配分を共有しない（9.2） | 設計で担保 |
| 5 | internal_testでの旧checkpoint評価 | 20c解除後も含む | manifestの`prohibited_checkpoint_rule`と、学習集合hash照合による拒否 | 契約として記録。20cの解除・評価は実装しない |

**残る制限（担保できないこと）を明示する。**

- 実機は単一ユーザー環境であり、**所有者は封印ディレクトリを読める**。0500/0400は誤操作と
  他プロセスからの偶発的アクセスを減らすが、意図的な閲覧は防げない。
- **元H5は元の場所に残す。** 移動・改名・権限変更を行わない（既存成果物を壊さないため）。
  したがってファイルシステム上はinternal_testの元データへ到達可能である。封印は
  「OS権限による物理隔離」ではなく「コードguard＋出力境界＋運用」であると記録する。
- 中間pseudo-3D H5、Stage2to4の出力、既存のrun成果物には手を加えない。
- **実機の権限変更・ディレクトリ作成は提案のみを行い、コマンドはユーザーが実行する。**
  新規に作る封印ディレクトリ以外へchmod/chownを行わない。

### 9.6 train sanity 3の照合証拠（管理8）

| 段階 | 手順 | 合格条件 |
| --- | --- | --- |
| 第一選択 | S5-15評価runの`selected_train_files.txt`（3行）を読む | 3件すべてがtrain162リストに**動画identityとして**含まれ、実ファイルが存在する |
| 照合 | 旧評価のprivate `video_id_map_DO_NOT_SHARE.csv`の`split == "train_sanity"`行の`original_h5_path`と、**動画identityで**突合 | 3件が1対1で一致する。**aliasが3件あること・件数が3であることは証拠にしない** |
| 代替（第一選択が無い場合のみ） | 旧train162リストの`content_sha256`がS5-15 run manifestの記録値と一致することを先に確認し、その順序で`select_train_paths(fixed_videos=["20250626_124212_7300"], num_random=2, seed=42)`を再現 | 再現結果が上記private対応表と動画identityで一致する。再導出したことを報告に明記する |
| 不足時 | 照合証拠が揃わない | **splitを確定しない。** 確認できた範囲と不足を報告して管理判断を求める |

実IDは共有文書へ出さない。報告は「照合方法・照合できた件数・不足の種類」に限る。
sanity 3を再導出で決め打ちしない理由は8.1-1のとおりで、選択がtrain162リスト**順序**に依存するためである。

### 9.7 実機確認事項（管理4・5の具体化）

1. **臨床FLの入力仕様（9.2の前提）:** 供給元・ファイル形式、動画識別子の列名と表記（H5 basename／`YYYYMMDD_HHMMSS_N`／その他）、
   FL値の列と単位、欠損の表現（空欄・記号・別列のフラグ）、1動画1値か複数計測値か（複数なら代表値の決め方）、
   候補外動画の行が含まれるか。**実値は共有側へ持ち込まない。**
2. **mmスケールの供給元（管理4、S5-17開始条件）:** 供給元・保存形式、mm/pixelか画像全体の実寸か、
   元frame基準かcrop/resize後基準か、x/y別の値の有無、動画単位かframe単位か、動画・frameへの対応方法。
   確認後に取得不能と分かったものだけ理由を付けて「取得不能」とする。`spacing = 1.0`を実測値とみなさない。
   Step 0で未解決なら、対応方針とS5-17開始条件を明示して受入判断を返し、確認完了としない。
   代理FL・`T_FL`は推定しない。分割・封印・weight算出はこの確認と並行して進められる。
3. **旧W-Aの由来監査（管理5）:** 実機のrun config・ログ・算出時の対象リストを探し、
   算出元の動画集合・teacher版・label計数方法・epsilon・正規化・数値精度・転記経緯を確認する。
   **旧値との一致を新値の検証条件にしない。差を縮めるための調整・探索は行わない。**
   記録が見つからない場合も、確認できた式・valid/ignore処理・設定・精度を明示して新baselineとして固定する案を返す。
   その場合「W-Aと同一ロジックを完全確認した」とは記録しない。由来不明だけを理由に無期限に停止しない。

### 9.8 変更範囲（更新）

**再利用（変更しない）**

`stage5/utils/file_list_mode.py`のhash・比較・検証関数、
`checks/real_h5/check_stage5_gt_component_count.py`の成分計数関数群（importのみ）、
`checks/real_h5/check_stage5_xy_coordinate_provenance.py`の中間H5解決・attr読取り、
`checks/real_h5/write_stage5_s5_15_run_manifest.py`のgit状態収集とprivacy self-check様式。

**追加ファイル**

| ファイル | 内容 | 初回提案からの差分 |
| --- | --- | --- |
| `stage5/utils/class_weights.py` | class weight 2関数の移設（式・既定値・戻り値型は不変） | 変更なし |
| `stage5/utils/split_contract.py` | manifest schema・読込・署名検証・契約検査6項目・`assert_paths_allowed()`・legacy manifest生成CLI・`--assert`サブコマンド（bashから呼ぶ） | legacy manifest生成と`--assert`を追加 |
| `checks/real_h5/build_stage5_bprime_split.py` | alias付与＋GT計数＋FL層化＋配分＋契約検査＋manifest＋封印出力の統合。`--dry_run_stratification` / `--confirm`の2段構え | alias map生成とGT計数を統合。2段構えを追加 |
| `checks/real_h5/compute_stage5_train_core_class_weight.py` | train_core144のみ1回算出（torch非依存） | 変更なし |
| `checks/real_h5/audit_stage5_train_core_mm_metadata.py` | train_core144の中間H5メタ監査 | 変更なし |
| `checks/dummy/check_dummy_split_contract.{py,sh}` | guardとmanifest必須化の合成テスト | 範囲拡大（9.9） |
| `checks/dummy/check_dummy_bprime_split_builder.{py,sh}` | FL分岐・配分・同値・停止条件・出力境界の合成テスト | **新規**（初回提案では分離していなかった） |
| `checks/dummy/check_dummy_train_core_class_weight.{py,sh}` | 純粋関数の合成検証 | 変更なし |

廃止: `checks/real_h5/build_stage5_train_pool_video_id_map.py`（9.3で統合）。

**変更ファイル**

| ファイル | 変更 |
| --- | --- |
| `train_stage5.py` | class weight 2関数のimport化（振る舞い不変）、guard挿入、directory-mode拒否 |
| `train_stage5.sh` | preflight**前**のguard呼び出し、`LIST_MANIFEST`の一時ファイル出力先変更、`EXPECTED_INPUT_FILES`と teacher preflight期待値定数の扱い（9.11-1） |
| `evaluate_stage5.py` / `evaluate_stage5.sh` | guard挿入 |
| `infer_stage5.py` / `infer_stage5.sh` | guard挿入 |
| `checks/real_h5/run_stage5_s5_15_arm.sh` / `run_stage5_s5_15_r0_longrun.sh` | legacy manifestの指定（S5-15の実験条件自体は変更しない） |
| `checks/real_h5/`の実データcheck群 | guard挿入。対象の一覧化は9.11-3 |
| `docs/stage5/FILES.md` / `docs/stage5/data_construct.md` | 実装配置確定後に更新 |

### 9.9 検証（合成テスト、更新）

実データを使わない。CPUのみ。dummy側は合成H5と**合成manifest**を使うため、guardを迂回する経路を作らない。

**split contract / guard**

manifest未指定で停止、manifest不正（署名不一致・schema不一致・field欠落）で停止、
directory-modeからの混入が`allows_directory_mode: false`で停止、封印集合の直接指定で停止、
片側リストのみの指定で停止、legacy manifestでの新規学習が`allows_new_training: false`で停止、
legacy manifestでの旧条件再現は通過、
`assert_paths_allowed()`が**H5を開く前に**投げること（存在しないパスを封印集合として渡し、
`FileNotFoundError`ではなく封印例外になることで示す）、
bash側のguardが**teacher preflightより前**に発火すること（preflightが到達しないことをexit codeと出力で確認）。

**契約検査6項目**

重複、欠落、件数違い（143／19）、sanityがtrain_coreから外れる、旧train162との不一致、
validation順序の変更、content hash不一致とidentity hash不一致の**区別**。

**split builder（新規）**

FL分岐: 明示的欠損→`FL_missing`、対応行なし／重複／候補外→停止、値不正→停止、単位不確定→停止、
欠損トークンの実行時追加→停止。
3分割: `m % 3`の各剰余で群サイズが`[q+1]*r + [q]*(3-r)`になること、同値がtrain162順で振り分けられること、
`m < 3`や空群で**停止し別方式へfallbackしないこと**。
配分: 最大剰余法の結果、`alloc_k <= n_k`、空セル0件、剰余同値がセルキー昇順で決着（乱数不使用）、
余力セルへの再配分、総数18に満たない場合の停止。
再現性: 同一入力・同一seedで2回生成したリストがbyte一致すること。
**出力境界**: stdoutと例外メッセージに動画別GT統計・FL値・実IDが現れないこと、
中間生成物が封印ディレクトリ外へ書かれないこと、共有用出力に`multi_region`の周辺度数・
クロス表・セル別配分が含まれないこと。**匿名化チェックとは別のテストとして書く。**

**class weight**

既知counts→期待weight（`1/(freq+eps)`・平均1正規化・float32）、`valid_mask=False`とignoreの除外、
対象外動画で停止、空集合・全ignoreで`ValueError`、元H5から動画1回だけ数えること（overlap window非経由）、
移設後の`resolve_class_weight()`の戻り値と`class_weight_info`が移設前と同一であること。

**既存経路の回帰**

`check_dummy_fixed_list_mode`のdirectory-mode回帰、`input_source_args()`の挙動、
既存dummy checkがすべて合格すること。

### 9.10 実機CPU実施量（再算定）

初回提案の「5回／最大6回」は流用せず、処理統合と2段構えを反映して算定し直した。
すべて読取り専用で、新規書き込みは新しい封印ディレクトリのみ。GPU不要。

| # | 工程 | H5読込 | 回数 | 出力 | 停止条件 |
| --- | --- | --- | --- | --- | --- |
| 1 | 入力監査（リスト読込・両hash・実在・teacher pattern・sanity照合・FL入力仕様の確認） | **なし**（実在確認のみ） | 1 | private監査JSON、共有用は件数・hash・成否 | リスト不整合、sanity照合証拠不足（9.6）、FL入力仕様が確定できない |
| 2a | split層化のdry run（alias付与→GT計数→FL分類→層構成）。**乱数を引かずsplitを確定しない** | 159件×1 pass | 1 | 封印ディレクトリへ中間結果（hash付き）、管理へFLの周辺度数のみ | 9.2の停止条件、GT計数の入力不整合 |
| 2b | split確定（2aの中間結果をhash照合して再利用→配分→契約検査→manifest→封印） | **なし**（再利用） | 1 | 3リスト、manifest、封印ディレクトリ | 契約検査6項目のいずれか不合格、中間結果hash不一致 |
| 3 | train_core144のclass weight算出 | 144件×1 pass | 1 | private JSON（counts・weight・対象hash・code hash・設定・精度） | 対象リストhashがmanifest不一致、valid点0、範囲外label |
| 4 | train_core144のmmメタ監査 | 最終H5はattrのみ、中間H5はattrのみ | 1 | private JSON、共有用は件数集計 | 中間H5が解決できない、`local_input_shape`が不正／非正方。mmスケール実体が確認できない場合は**「未確認」と記録して継続**（停止しない） |

- 合計5工程。**元H5を全点読むのは#2aと#3の2 passのみ**（初回提案では最大3 pass相当だった）。
- #2aと#2bを分けるのは、FLの周辺度数を管理へ返して層化の成立を確認してから確定するためであり、
  GT計数を2回実行しないため中間結果をhash照合で再利用する。
- 異常停止後の再実行は各工程1回まで、理由・停止段階・同一seed／同一入力hashであることを報告する。
  **正常なsplit確定（#2b）は1回のみ。** 別seedでの再生成・比較は行わない。
- 順序は 1 → 2a →（管理確認）→ 2b → 3 → 4。#2bの契約検査に合格するまで#3・#4へ進まない。
- 本節は実施量の申請であり、実行承認を得た状態ではない。

### 9.11 新規に判明した事項・判断が必要な事項

| # | 事項 | 内容と推奨案（提案履歴／追加修正案） | 判断 |
| --- | --- | --- | --- |
| 1 | **teacher preflightの期待値定数がtrain_core構成で成立しない（新規発見）** | `train_stage5.sh`の`EXPECTED_INPUT_FILES`既定は180で、teacher preflightは`cvat_videos`・`cvat_frames`・`invalidated_frames`・`removed_positive`・crop系の各合計を**対象ファイル全体で積算して定数と比較**する。train_core144＋validation18＝162では件数も各合計も一致せず、S5-18以降の学習が起動段階で停止する。**推奨: 162構成用の期待値定数を#2b確定後に1回だけ算出して固定する。** ただし「180の値」と「162の値」を両方共有すると差分からinternal_test 18件のteacher QC統計が復元されるため、**共有はpreflightの成否のみとし、両方の定数値を共有文書へ出さない**ことを併せて提案する | 条件付き採用。新集合の期待値固定とteacher品質検査を分離する。既存の整合性検査を維持し、新定数・詳細ログはprivateに保持。算出工程を9.10の実施量へ加える。 |
| 2 | `LIST_MANIFEST`の一時ファイル | 現状は実H5絶対パス一覧を`${TMPDIR:-/tmp}`へ書く。封印運用下のディレクトリへ変更することを推奨する（9.4）。既存runの再現性には影響しない | 採用。通常処理用のprivate一時領域へ変更する。封印成果物の領域と分離してよい。権限・終了時の後始末・ログ露出を確認する。 |
| 3 | guard挿入対象のcheck群の確定 | `checks/real_h5/`の実データcheckは多数あり、どこまでguardを必須にするかで変更量が変わる。**推奨: 元H5を開くものすべてを対象とし、対象一覧を実装時に提示して確認を受ける。** 一覧提示を#1の入力監査報告と同時に行う | 原則採用、範囲を修正。実装前に利用経路と読込成果物を一覧化し、H5だけでなく予測・GT統計等も対象にする。未対応経路は封印期間中の使用対象から外す。 |
| 4 | FL周辺度数の共有（9.2） | split確定**前**に1回だけ共有する設計を採る。`multi_region`の周辺度数・クロス表・セル別配分は共有しない。**この共有範囲で差し支えないか確認を求める。** 認められない場合は「3群が構成可能か」の真偽のみを返す代替案とするが、欠損が多い場合の判断材料は失われる | 条件付き採用。候補全体の欠損／非欠損件数・3群サイズ・成立可否を確定前に1回だけ共有する。推測や復元を完全に防ぐ保証とはせず、分割の選び直しに使わない。 |
| 5 | S5-17へ引き継ぐ未決事項 | mmスケール（9.7-2）が未確認のままStep 0を閉じる場合、S5-17の開始条件として明示する。`T_FL`もS5-17実行前のユーザー設定事項として未決のまま引き継ぐ。Step 0では代理FLを計算・選択しない | 従来判断を維持。未解決事項とS5-17実行条件を明示し、Step 0終了時に総括管理で受入判断する。未確認を記録するだけで自動完了とはしない。 |
| 6 | legacy manifestによる封印の迂回（管理側追加） | 旧train162に対するsealed空配列は、新規学習を禁止しても評価・checkerからinternal_testを読める経路を残す。現在の封印対象の拒否をlegacyにも適用する。 | 要修正。現在の封印拒否を各manifestの許可より優先する。封印集合と重なる旧データ処理は停止する。legacy指定で現在の封印を解除できない設計・テストを示す。 |
| 7 | manifest自身のSHA-256を署名と呼んでいる（管理側追加） | hashは整合性確認に使えるが、manifestとhashを一緒に差し替えられる場合、承認済み内容であることを保証しない。承認済み期待hashの固定先と照合方法を定義する。 | 要修正。「署名」を「hashによる整合性照合」に訂正し、期待hashの保存・承認・更新方法を明示する。新しい署名基盤の導入は求めない。 |

提案欄は履歴として保持する。現行の条件・訂正と次の作業は9.11.1を優先する。

#### 9.11.1 判断事項への返答（総括管理チャット、2026-09-22）

作成元: S5-16〜S5-20総括管理チャット。対象: 9.11の7事項と9.8〜9.10の変更・実施量。  
状態: **管理判断・修正条件の記録。CPU実装・実機実行の一括承認ではない。**  
記録根拠: ユーザーの確認依頼を受けて修正版計画と既存コードを照合し、続く追記・修正依頼により本節へ記録する。
FL欠損の分岐、GT計数の内部工程化、sanity3の照合方法は具体化されたものとして受け止める。
8.6.1の確定事項を再審議せず、以下の追加条件を満たしてからCPU実装・合成テストの着手判断へ進む。

**1. teacher preflightの期待値は新しい対象集合に合わせて固定する。**

現行train_stage5.shは180件を期待し、CVAT frame数・修正点数等を対象H5全体で積算している。
train_core144＋validation18では件数検査は確実に不一致となる。各QC合計も新しい集合に対応させる必要があるが、
各項目すべての合計が必ず旧値と異なるとは限らないため、9.11-1の表現はこの範囲に限定する。

- teacher版、label／valid_mask整合、修正後状態、由来情報等の既存検査は維持する。
- 新しい集合の観測値を集計して期待値へ保存することは、以後の変化検知の基準作成である。
  それだけで初期状態のteacher品質を検証したとはしない。検査の合格と期待値固定を別々に記録する。
- 新期待値を対象リストhash・teacher由来情報・算出コードhashと結び付け、以後は自動再計算で検査を通さない。
- 旧180件の定数は既にコードに存在する。新162件の定数を公開すると差分が得られるため、
  新定数と詳細ログはprivateに保持し、共有はpreflightの成否等の許可された情報に限る。
  bash／checkerが既定で出す集計値・例外ログも確認し、新値を共有コードへ直書きしない。
- 期待値算出・品質検査の工程は9.10に未計上である。対象H5と読込配列、既存工程との再利用、
  追加pass・実行回数・出力先を具体化して実施量を更新する。internal_testを再読込して差分を検算しない。

**2. LIST_MANIFESTはprivateな一時領域へ変更する。**

通常のtrain_core／validation処理に使うパス一覧は、private作業ディレクトリに置く。
internal_testの情報を保持する封印領域と同じ場所にする必要はない。用途とアクセス権を分け、
正常・異常終了時の後始末とログへの実パス露出を確認する。9.10の「新規書込みは封印ディレクトリのみ」は、
通常処理用private領域と封印成果物領域を区別した記述へ更新する。

**3. guard対象を成果物と使用経路から確定する。**

元H5を開くcheckだけでなく、保存済み予測NPZ、GT成分JSON、臨床FL表、中間H5等にも
internal_test情報への経路がある。今後使う経路について、読込成果物、動画identityの判定方法、
読込前の拒否位置、通常開発用か過去の専用診断かを、実装前に一覧化する。
全checkerへ機械的に改修を広げることを目的にせず、使用する経路を確定し、未対応経路は
封印期間中の使用対象から外す。その制約をlauncher・運用文書で明示し、未対応のまま全面的な技術保証をしない。
CLIとbash／preflightの先行読込を含め、必要な経路を対象とした合成テストを示す。

**4. split確定前のFL周辺度数共有を限定的に認める。**

候補159件の欠損件数・非欠損件数m・3群サイズ・構成可能か、に限って確定前に1回共有してよい。
後続の評価可能性を判断するための情報であり、FL実値・境界値・動画別値・multi-region度数・
クロス表・セル別配分・確定後のsplit別内訳は共有しない。seedや分割の選び直しに使わない。
クロス表が一意に決まらないことは、構成の推測や範囲の絞り込みが不可能であることを意味しない。
9.2・9.5の説明は「情報を限定する設計」であり、推測・復元を完全に防ぐ保証ではないと訂正する。
層化不成立・欠損の多さから計画変更が必要なら、splitを確定せず判断を返す。

**5. mmスケール・T_FLの持越しは従来の条件を維持する。**

mm供給元の未確認だけで分割・封印・weight算出の準備を一律に止めない。
未解決の場合は対応方針とS5-17の実行条件を明示し、Step 0終了時に総括管理で受け入れるか判断する。
S5-17のmm評価・臨床FL照合を含む処理は、実スケールと必要な判定条件を確定してから実行する。
T_FLはS5-17実行前のユーザー設定事項として保持し、代理値を推定しない。未確認の記録だけではStep 0完了にならない。

**6. legacy manifestにも現在の封印拒否を適用する。**

9.4のsealed空配列・新規学習禁止だけでは、旧train162への評価・checkerが現在のinternal_testを読める。
この設計は採用しない。現在の封印集合の拒否を、legacyを含む各manifestの許可より優先する。
封印集合と重なる旧データ処理は停止し、過去条件の完全再現のために封印を解除しない。
legacy manifestは現在の封印集合を消す／置換する権限を持たないものとし、現在の封印契約の
参照・検証方法を具体化する。実装はStep 0の範囲にとどめ、20cの解除処理は先行実装しない。
9.9の「legacy manifestでの旧条件再現は通過」は、現在の封印集合と交差しない許可済み対象に限定する。
legacyで封印動画を要求した場合、新規学習以外の評価・推論・checkerでも読込前に停止する合成テストを加える。

**7. manifestのhashと承認済み内容の照合を分ける。**

9.4・9.8・9.9のSHA-256を指す「署名／署名検証」は「hashによる整合性照合」に訂正する。
manifestとhashを一緒に差し替えられるなら、両者の一致だけでは承認済み内容であることを確認できない。
承認済みmanifestの期待hashを固定する場所、誰がどの手順で承認・更新するか、
実行時にどの値と照合するかを示す。実行対象manifestからその場で計算したhashを、
独立した期待hashの代わりに使わない。同一ユーザーが全ファイルを変更できる制限は明示する。
新しい署名基盤や過剰な権限管理システムの導入は要求しない。誤指定・未承認の差替えを防ぐ
固定参照と照合の仕組みを、既存の実行・承認運用に合わせて最小範囲で設計する。

**実装担当への次の依頼・実施境界**

上記条件を満たす差分を、本報告書の新しい節へ短く追記する。計画全文の再掲は不要。
少なくともlegacyの迂回防止、期待hashの固定・照合、guard対象一覧、teacher検査／新期待値算出、
private一時領域と封印領域の区別、更新後のCPU実施量と必要な合成テストを示す。
既決のseed・FL欠損方針等の再承認は不要。今回の条件修正を確認してCPU実装・合成テストの着手を判断する。
実機CPU実行は合成テスト結果と更新後の対象・回数を確認してから扱い、9.10の5工程や
各工程1回の再実行枠を今回の返信で自動承認したものとはしない。新規学習・GPU推論・コミットは実施しない。

### 9.12 本節の位置づけ

9.1〜9.10は8.6.1を受けた**修正版計画の提案履歴**として保持する。実装・実機実行・コミットは未実施。
9.11の7件への現行の管理返信は9.11.1とする。次は追加条件に対応する差分計画を新しい節へ追記し、
管理確認後に承認範囲内のCPU実装・合成テストへ進む。旧9.4・9.8〜9.10をそのまま承認済みとは扱わない。
実機CPU工程は、合成テスト結果と更新後の実施量を確認してから進める。
実機結果（日付・承認範囲・コードrevision・環境・入力／出力hash・成否・逸脱・未決事項）は
本報告書のさらに新しい節へ追記し、提案・承認・実装・実行完了を区別する。
`FILES.md`と`data_construct.md`の更新は実装配置が確定した時点で行う。

## 10. 9.11.1の修正条件への差分（2026-09-22）

担当: Step 0実装チャット。状態: **計画差分（提案）。実装・実機実行・コミットは未実施。**
9章の計画全文は再掲せず、9.11.1の7条件に対する変更点のみを示す。
既決事項（seed 42、FL欠損方針、GT計数の内部工程化、sanity3の照合方法）は再審議しない。

### 10.1 訂正

| 箇所 | 訂正 |
| --- | --- |
| 9.4・9.8・9.9の「署名／署名検証」 | **「hashによる整合性照合」に訂正。** manifestとhashは同時に差し替えられるため、manifest自身が持つhashとの一致は承認済み内容の確認にならない（10.3） |
| 9.4の legacy manifest（`sealed: []`） | **設計欠陥として撤回。** 旧train162への評価・checkerが現在のinternal_testを読めてしまう。封印集合の拒否をmanifestの許可より優先する構成へ改める（10.2） |
| 9.11-1「各合計も一致せず」 | **過大な断定を訂正。** 確実に不一致なのは件数検査のみ。各QC合計は対象集合が変われば変わりうるが、全項目が必ず旧値と異なるとは限らない |
| 9.2・9.5の漏洩に関する説明 | **「情報を限定する設計」であり、推測・復元を完全に防ぐ保証ではない**と訂正する。クロス表が一意に定まらないことは、構成の推測や範囲の絞り込みが不可能であることを意味しない |
| 9.10「新規書込みは封印ディレクトリのみ」 | 通常処理用private領域と封印成果物領域を区別した記述へ更新（10.5） |

### 10.2 封印集合の優先と legacy の迂回防止（条件6）

**封印契約をmanifestから分離する。** `sealed_registry.json`（現在の封印集合の正本）を、どのmanifestを
使う場合でも独立に解決する。解決できない・照合に失敗する場合は**許可ではなく停止**とする。

- 判定順序は `deny → allow`。`assert_paths_allowed()`はまずregistryの封印identityと突き合わせ、
  一致すれば**いかなるmanifestの許可よりも先に**H5オープン前で停止する。
- manifestはregistryの封印集合を**消去・置換・縮小できない**。registryと矛盾する`sealed`を宣言した
  manifestは読込時点で拒否する。
- 判定単位は**動画identity**（teacher H5 basenameから固定regexで抽出する`YYYYMMDD_HHMMSS_N`）とする。
  これによりpath表記差だけでなく、中間pseudo-3D H5・予測NPZ・GT成分JSON等、
  拡張子や接尾辞の異なる成果物にも同じ拒否が効く。
- **legacy manifestは「封印集合と交差しない対象に限り通過する」ものに限定する。**
  旧train162は現在のinternal_test18を含むため、legacyでもそれら18件を要求した時点で停止する。
  学習だけでなく評価・推論・checkerでも停止する。**過去条件の完全再現のために封印を解除しない**ことを
  制約として受け入れ、manifestと運用文書に明記する。
- 20cの解除処理は先行実装しない。旧checkpointのinternal_test評価禁止は解除後も維持する契約として残す。

### 10.3 期待hashの固定参照と照合（条件7）

実行対象からその場で計算したhashを期待値の代わりに使わない。**期待hashはリポジトリ側へ固定する。**

- 追加する`Stage5/stage5/config/split_contract_pins.json`（git管理）に、承認済みmanifestと
  `sealed_registry.json`の`expected_sha256`、承認日、承認範囲を記録する。
- 実行時は、解決したファイルのSHA-256を**pinの値**と照合する。ファイル内の自己記載hashとは照合しない。
  pinが無い・一致しない場合は停止する。
- 承認・更新手順: pinの変更はgit差分としてユーザーが確認し、ユーザーがコミットする
  （本プロジェクトの既存のコミット運用と同じ）。新しい署名基盤・権限管理は導入しない。
- **残る制限を明示する:** 同一ユーザーがpinとmanifestの両方を変更できる。本仕組みが防ぐのは
  誤指定と未承認の差替えであり、意図的な同時改変は防げない。

### 10.4 guard対象経路の一覧（条件3）

全checkerへ機械的に広げず、**封印期間中に使う経路を確定**し、残りは使用対象外として宣言する。

**対象（guard必須。実装時に最終確認を受ける）**

| 経路 | 読込成果物 | 拒否位置 |
| --- | --- | --- |
| `train_stage5.sh` / `train_stage5.py` | 元H5、file list | fixed-list検証直後・**teacher preflightの前**／`resolve_train_val_paths()`直後 |
| `evaluate_stage5.sh` / `evaluate_stage5.py` | 元H5、`RUN_DIR/train_files.txt`等 | リスト読込直後 |
| `infer_stage5.sh` / `infer_stage5.py` | 元H5 | `--input_h5`オープン前 |
| `prepare_stage5_evaluation_data.py` | 元H5、参照PLY | リスト読込直後 |
| `export_anonymized_stage5_metrics.py` | metrics CSV、`video_id_map` | 入力読込前 |
| `checks/real_h5/write_stage5_s5_15_run_manifest.py` | file list | リスト読込直後 |
| `checks/real_h5/run_stage5_s5_15_arm.sh` / `run_stage5_s5_15_r0_longrun.sh` | legacy経路 | launcher冒頭（10.2の交差拒否） |
| Step 0新規3本（split構築／class weight／mmメタ監査） | 元H5、中間H5、臨床FL表 | 各入力のオープン前 |
| `checks/real_h5/check_stage5_s5_15_gpu_preflight.py` | 元H5、file list | `--train_list`/`--val_list`読込直後。S5-18／S5-19のGPU実行前に使う |
| `checks/real_h5/check_stage5_effective_run_config.py` | run_dirのconfigと解決済みリスト | 記録されたリストのidentity hash照合（H5は開かない）。S5-18のconfig突合に使う |
| `checks/real_h5/check_stage5_s5_15_p1_manifest_audit.py` | run_dirの成果物 | 同上。`--expected_train_files`の既定162は新規runでは144を指定する（10.6と同種の集合依存値） |

**封印期間中の使用対象外（guard未対応のまま使用しないと宣言する）**

`checks/real_h5/`のS5-08〜S5-14専用診断: `check_stage5_batchnorm_mode_parity` / `batchnorm_recalibration` /
`label_policy_ablation_eval` / `label_policy_bbox_preflight` / `label_policy_dataset_parity` /
`overlap_aggregation` / `padding_parity` / `coordinate_transform_diagnostics` / `coordinate_transform_reconciliation` /
`frame_xy_diagnostics` / `frame_spatial_overlap_diagnostics` / `brightness_position_vs_prediction` /
`frame_failure_vs_gt_regions` / `class_weight_ablation` / `class_weight_threshold_free` /
`rotation_augmentation_ablation` / `s5_14_summary_export` / `s5_14_supplement` /
`checkpoint_identity` / `batch_integrity` / `xy_coordinate_provenance`（CLI）/ `gt_component_count`（CLI）/
`check_real_h5_pointnext_s_transfer_forward`。
`checks/real_h5/check_stage4_combined_v2_loader_compatibility.py`は`--input_dir`をglob走査するため、
封印期間中は使用対象外とし、**directory走査を行う経路として個別に名指しで禁止する**。

- `gt_component_count`と`xy_coordinate_provenance`は**関数importのみ**Step 0で使い、CLI経路は使用対象外とする。
- この制約をlauncherの起動時メッセージと運用文書（`FILES.md`）へ明記する。
  **未対応経路について技術的保証をしたとは記録しない。**
- 臨床FL表とGT成分中間結果は、**split構築処理以外の入力にしない**。表形式入力は経路単位で
  読込主体を限定し、path判定だけに依存しない。

### 10.5 private一時領域と封印領域の区別（条件2・5）

| 領域 | 用途 | 内容 | 共有 |
| --- | --- | --- | --- |
| A: private作業領域 | 通常処理の一時ファイル・ログ。`LIST_MANIFEST`の`mktemp`先 | train_core／validationのパス一覧 | 不可（実パスのため） |
| B: split成果物領域 | 3リストのうち非封印分、manifest、pin参照、teacher新期待値 | hash・件数・仕様・新定数 | hash／件数／成否のみ可。**新定数は不可** |
| C: 封印領域（B配下） | internal_test情報を含む成果物 | internal_testリスト、候補GT統計、FL値、セル別配分 | **不可（S5-20cまで）** |

- Aは封印領域と同じ場所に置かない。用途とアクセス権を分ける。
- `LIST_MANIFEST`は`${TMPDIR:-/tmp}`からAへ変更し、既存の`trap ... EXIT`による後始末を維持して
  正常終了・異常終了の双方で残らないことを合成テストで確認する。
- teacher preflightの標準出力は件数のみで実パスを出さないことを確認済み。例外時に実パスが出るのは
  既存挙動であり、封印集合はguardで到達しないため新たな露出は生じないが、例外ログを共有文書へ貼らない運用とする。
- mmスケール・`T_FL`の扱いは9.7-2・9.11-5のまま変更しない。未確認の記録だけではStep 0完了としない。

### 10.6 teacher検査と新期待値の算出（条件1・4）

- **既存検査は維持する。** teacher版、`label_mode`／`contour_teacher_schema`／`label_source`、
  `valid_mask`と`point_label != -1`の整合、修正後状態、由来情報、必須dataset存在の各検査は変更しない。
  これらは対象集合に依存しない不変条件である。
- **変更するのは集合依存の期待値のみ:** `EXPECTED_INPUT_FILES`（180→162）と、
  `cvat_videos` / `cvat_frames` / `invalidated_frames` / `invalidated_videos` / `removed_positive` /
  `removed_bbox_rows` およびcrop系5項目の各合計。
- **検査の合格と期待値の固定を別々に記録する。** 新しい集合の観測値を期待値として保存することは
  以後の変化検知の基準作成であり、それだけで初期状態のteacher品質を検証したとは記録しない。
- 新期待値は、対象リストhash・teacher由来情報・算出コードhashと結び付けてB領域へ保存し、
  `train_stage5.sh`はそのファイルを参照する（10.3のpinで照合）。
  **実行時の自動再計算で検査を通さない。共有コードへ新定数を直書きしない。**
- 算出は既存preflightの積算ロジックへ観測値出力modeを追加して行い、計数処理を別実装へ分岐させない。
- **internal_testを再読込して差分で検算しない。** 新定数は162件の直接観測のみから作る。

### 10.7 CPU実施量の更新（条件1末尾）

9.10の5工程に工程5を追加し、6工程とする。9.10の枠を承認済みとして流用しない。

| # | 工程 | 元H5の全点読込 | 回数 |
| --- | --- | --- | --- |
| 1 | 入力監査（リスト・hash・実在・sanity照合・FL入力仕様確認） | なし | 1 |
| 2a | split層化のdry run（乱数を引かず確定しない） | 159件×1 pass | 1 |
| 2b | split確定（2aの中間結果をhash照合して再利用） | なし | 1 |
| 3 | train_core144のclass weight算出 | 144件×1 pass | 1 |
| 4 | train_core144のmmメタ監査 | attrのみ | 1 |
| **5（新規）** | **teacher preflight新期待値の算出（train_core144＋validation18＝162件）** | **162件×1 pass**（annotation以外のQC group含む） | **1** |

- 元H5を全点読むのは**3 pass**（2a／3／5）。工程3と5は対象集合（144対144+18）も読込配列も異なるため統合せず、
  「train_core144のリストを明示入力として1回算出する」という依頼書5章の条件を保つ。
- 順序は 1 → 2a →（管理確認）→ 2b → 3 → 4 → 5。工程5は2bのtrain_coreリスト確定後にのみ実行できる。
- 異常停止後の再実行は各工程1回まで、理由・停止段階・同一seed／同一入力hashであることを報告する。
- 本節は実施量の**申請**であり、実行承認を得た状態としない。実機CPU実行は合成テスト結果を
  報告してから改めて扱う。新規学習・GPU推論・コミットは行わない。

### 10.8 追加する合成テスト

9.9に次を追加する（実データ・実manifestを使わず、合成registry／合成pinで行う）。

- **封印優先:** registry不在・不正で**停止すること（許可にならない）**。manifestが`sealed`を空にする／
  registryと矛盾する宣言をした場合に拒否されること。deny判定がallow判定より先に評価されること。
- **legacy迂回:** legacy manifestで封印動画を要求した場合、**学習以外の評価・推論・checkerでも
  読込前に停止すること**。封印集合と交差しない対象でのみ通過すること。
- **identity判定:** 元H5以外（中間H5・NPZ・JSON風の名前）でも封印identityで拒否されること。
  path表記差では回避できないこと。
- **hash照合:** pin不一致で停止。pin不在で停止。**自己記載hashを正しく書き換えた改竄manifestが
  pin不一致で停止すること**（自己記載hashを期待値に使っていないことの証明）。
- **一時領域:** `LIST_MANIFEST`がA領域に作られ、正常終了・異常終了の双方で残らないこと。
- **teacher期待値:** 新定数が対象リストhash不一致で拒否されること。実行時の自動再計算では
  検査が通らないこと。既存の集合非依存検査が変更されていないこと。

### 10.9 本節の位置づけ

本節は9.11.1の7条件に対する**計画差分の提案**であり、実装・実機実行・コミットは未実施である。
差分への管理側の確認結果と補足条件は10.10に記録する。CPU実装と合成テストの着手判断は、その条件を含めて扱う。
合成テストの結果を報告してから、10.7の実機CPU工程を改めて扱う。
`FILES.md`（10.4の使用対象外宣言を含む）と`data_construct.md`（10.5の3領域）の更新は、
実装配置が確定した時点で行う。

### 10.10 差分確認と補足条件（総括管理チャット、2026-09-22）

担当: S5-16〜S5-20総括管理チャット。対象: 10.1〜10.9の差分計画。  
状態: **主要修正の反映を確認。以下3点を実装条件へ追加。実装・実機実行の完了や一括承認ではない。**

独立した封印registryによるdeny優先、別pinによる期待hash固定、guard対象と使用対象外の明示、
private一時領域の分離、teacher既存検査の維持と期待値算出工程の追加は、前回の修正条件に対応している。
大きな方針の再検討は不要とし、ユーザーの補足反映依頼に基づき次を追記する。

| # | 補足対象 | 判断・条件 |
| --- | --- | --- |
| 1 | 初回のregistry・pin生成 | 通常処理での必須化を維持しつつ、初回split構築専用の条件と通常運用への切替順を明示する。registry不在を通常経路の許可理由にしない |
| 2 | identity不明の成果物 | ファイル名だけで判定できなければ許可しない。検証済みの対応情報で全対象動画を確認するか、読込前に停止する |
| 3 | teacher preflightの出力 | 現コードは成功時にもQC合計を出す。新期待値・観測値が共有ログへ出ないよう、成功・失敗の双方で出力を制御する |

#### 10.10.1 初回構築と通常運用の切替

10.2〜10.4の「registry・pin不在なら入力を開かず停止」を初回split構築にもそのまま適用すると、
生成に必要な入力を読めず循環する。初回構築だけの限定経路を設け、通常経路とは区別する。

- 初回構築の入力は、事前監査した旧train162／validation18／sanity3のリストとhash、承認範囲内の
  層化入力に限定する。任意ディレクトリの走査や任意の入力を許すguard無効化フラグにはしない。
- 初回専用経路を使える状態・出力先・既存契約の有無を検査する。既に封印を確定した出力がある場合は、
  初回経路で上書き・縮小・再生成しない。途中停止からの再開は記録した入力hash・状態を照合する。
- 2aの中間結果確認、2bのリスト／registry／manifest生成、生成物hashの確認とpin固定、通常運用への切替を
  順に明示する。pinが未固定の間、通常の学習・評価・推論・checkerを起動できる状態にはしない。
- 工程3〜5の読込は確定した封印契約に従う。新teacher期待値は工程5で初めて生成されるため、
  その生成と期待値ファイルのpin固定を通常のpreflight検査から分ける。期待値がまだないことを理由に
  通常preflightを無効化せず、承認された生成工程だけが作成し、以後の検査は固定値を参照する。
- 合成テストには、初回正常系、通常経路のregistry／pin欠落拒否、封印確定後の初回経路再使用拒否、
  入力hash不一致、切替途中の通常処理拒否を含める。実機の初回処理を本追記だけで開始しない。

#### 10.10.2 ファイル名でidentityを確定できない場合

10.2のregexは動画IDを含む既知の命名規則に対して使う。regexに一致しないことは、封印対象でない証明ではない。
共通名のJSON／CSV、複数動画を含む集計、aliasだけの成果物では、ファイル名のみの判定は使えない。

- 検証済みの対応表・出力manifest等で、成果物とその全対象動画を読込前に結び付ける。
  対応情報は対象成果物のhash等と照合し、別成果物の対応表を流用できないようにする。
- 対応が不明・曖昧・欠落・不一致なら停止する。複数動画の成果物に封印動画が1件でも含まれる場合も拒否する。
  対象集合を調べるために先にGT統計や予測内容を読む経路にはしない。
- 初回split構築の層化表は10.10.1の限定入力として扱い、通常の成果物読込に対する例外を広げない。
- 合成テストには、IDなしの名前、未対応alias、複数動画中の封印動画混入、対応表と成果物の不一致を追加する。
  対応関係を確立できない経路は、10.4の使用対象外として明示する。

#### 10.10.3 preflight成功・失敗時の出力境界

10.5の「標準出力は件数のみで実パスを出さない」という確認では不十分である。
現行train_stage5.shは成功時にCVAT frame数・修正点数等を出し、不一致時にもactual／expectedを出す。
旧180件の定数が既知なので、新162件の値が共有ログへ出ると差分から封印集合のQC統計を得られる。

- 成功時の共有可能な出力は、工程名・許可された対象件数・契約hash・成否に限定する。
  新期待値・観測されたQC合計・その差分・動画別値は出さない。
- 失敗時も標準出力／標準エラー、例外、bashの表示を含め同じ境界を守る。共有向けには数値を含まない
  エラー種別・成否を示し、診断に必要な詳細はprivateログへ保持する。
- 実パスが出ないことと、統計が漏れないことを別々に検証する。単に「例外ログを貼らない」という運用だけに依存しない。
- 合成テストでは成功・QC不一致・入力不備の各経路を通し、捕捉したstdout／stderrに非共有の期待値・観測値・
  差分・実IDが現れず、必要な詳細だけがprivate出力へ記録されることを確認する。

#### 10.10.4 次の作業・実施境界

実装担当は、この3条件を具体的な処理順・guard・合成テストへ反映する。計画全文の再提出は不要とし、
対応箇所と、解決できない点や実施量が変わる点があれば差分だけを報告する。
既決事項の再承認は不要。条件を含めたCPU実装・合成テストの着手判断と、実機CPU6工程の実行判断は分ける。
実機処理は合成テスト結果と対象・回数の確認後に扱う。Step 0未完了、学習・GPU推論・コミット未実施の状態を維持する。

## 11. CPU実装と合成テストの結果（2026-09-22）

担当: Step 0実装チャット。状態: **CPU実装完了。合成テストは実行できた範囲で合格
（class weightのnumpy/h5py依存テストは本環境で未実行、11.4-1）。実機CPU実行・コミットは未実施。**
※本節の内容は11.7.1・11.7.3の管理判断を受けて12章で修正した。相違する箇所は12章を優先する。
対象: 10.10.4の着手条件（legacy迂回防止、期待hash固定・照合、guard対象一覧、teacher検査／新期待値、
private一時領域と封印領域の区別、更新後のCPU実施量と合成テスト）。
計画全文は再掲せず、対応箇所・逸脱・新規判明事項・残作業のみを示す。

### 11.1 実装した範囲と合成テスト結果

新規8ファイル・変更11ファイル。合成テストは**新規157項目＋既存回帰11項目がすべて合格**した。

| 条件 | 実装 | 主な検証項目 |
| --- | --- | --- |
| 10.10.1 初回構築と通常運用の切替 | `split_contract.py`の`assert_bootstrap_allowed` / `verify_bootstrap_inputs` / `assert_resume_consistent`、builderの`--bootstrap`必須化 | 初回正常系、registry既存・pin承認済みでの初回経路拒否、入力hash不一致、再開時の入力変更拒否 |
| 10.10.2 identity不明の成果物 | `split_identity.py`。名前から`YYYYMMDD_HHMMSS_N`を抽出、取れない場合はhash照合済み`ArtifactCoverage`必須 | 共通名JSONの無条件拒否、複数動画集計への封印動画混入拒否、hash不一致の対応表拒否、空の対応表を「異常なし」としない |
| 10.10.3 preflight出力境界 | `train_stage5.sh`のpreflightを成功・失敗とも数値非開示に変更、詳細は`PREFLIGHT_PRIVATE_LOG`へ | 埋め込みpythonをASTで解析し、**成功・失敗いずれの共有メッセージにもQC合計が補間されていない**ことを確認。実パス非露出とは別項目として検証 |
| 10.2 legacy迂回防止 | registryをmanifestから分離し`deny→allow`順で評価 | legacyでの新規学習・directory-mode拒否、**評価・推論・checkerでも封印動画を拒否**、非交差対象のみ通過 |
| 10.3 期待hash | git管理の`stage5/config/split_contract_pins.json`。`--split_manifest`はpin名 | **自己記載hashを正しく書き換えた改竄manifestがpin不一致で停止**、pin不在・不一致で停止 |
| 10.4 guard対象 | 10経路へ挿入。対象外を`FILES.md`へ明示 | guardがpreflightより前に呼ばれること（静的順序＋実行での停止）、contract未承認でexit 2 |
| 10.5 領域分離 | `LIST_MANIFEST`を`PRIVATE_WORK_DIR`へ、封印物は`sealed/`（0700/0600）へ直接書き出し | `/tmp`経路の消滅、`trap ... EXIT`の維持、作業領域と封印領域の分離 |
| 10.6 teacher期待値 | 集合依存の期待値のみ`TEACHER_EXPECTED_JSON`から読み、`EMIT_TEACHER_EXPECTED`で1回生成 | emitとcompareが排他分岐であること、**emit側は判定を行わない**こと、180件の既定が不変であること |
| 8.3-4 契約検査6項目 | `verify_split_partition()` | 重複・件数違い・sanity流出・旧リスト不一致・validation並べ替え、パス表記差のみの変更を「内容差」と区別して報告 |
| 9.2 FL分岐・配分 | `split_allocation.py` | 対応行なし／重複／候補外／未宣言トークン／単位付き値の停止、剰余規則、同値のtrain162順解決、上限・空層・再配分、乱数非使用 |
| 5章 class weight | `class_weights.py`へ**byte一致で移設**（git照合済み） | 式・ignore／valid_mask・動画1回計数・空集合停止・対象制限 |

合成テスト内訳: `check_dummy_split_contract` 68項目、`check_dummy_bprime_split_allocation` 53項目、
`check_dummy_preflight_guard_and_output_boundary` 36項目。既存`check_dummy_fixed_list_mode`（11 test、
bash半分を含む）も合格し、directory-mode回帰と`input_source_args()`の挙動は不変である。
全`.sh`の構文検査も合格した。

**封印前拒否の証明方法:** 合成テストは実internal_testを読まない。代わりに
**存在しないパスを封印集合として渡し、`FileNotFoundError`ではなく封印例外が出ること**を確認する。
これによりguardが「読んで失敗した」のではなく「読む前に止めた」ことを区別している。

### 11.2 仕様からの逸脱・変更（要確認）

| # | 事項 | 内容と理由 |
| --- | --- | --- |
| 1 | **層内抽出の乱数器** | 8.3-3・9.2は`numpy.random.default_rng(seed)`としていたが、`sha256(seed\|動画identity)`による鍵付きソートへ変更した。理由: `random.sample`もnumpyのGeneratorもメソッドの乱数列を版間で保証しておらず、封印が続く限り再現できる必要がある。本方式はPython版・ライブラリ版に依存せず、本リポジトリが既にaugmentation角度の導出で使っている方式と同じである。seed42・入力順・配分規則は変更しない |
| 2 | **manifestの指定方法** | `--split_manifest`／`STAGE5_SPLIT_MANIFEST`は**パスではなくpin名**とした。計画ではパスとしていた。任意のファイルを指せなくなるため誤指定防止が強まり、未承認manifestは指定する名前自体が存在しない |
| 3 | **legacy manifestの`sealed`** | 10.2では`sealed: []`と書いたが、registryが非空のときに`sealed: []`を宣言するmanifestは読込時に拒否する設計にしたため、legacyは`sealed: ["train_core"]`を宣言する。封印の実効はregistryが決めるので動作は同じだが、「何も封印されていない」と主張するmanifestが黙って無効化されるのではなく明示的に拒否される |
| 4 | **ファイル構成** | 10.8の予定から、`split_identity.py`（10.10.2用）と`split_allocation.py`（stdlib化のため分離）を追加、`check_dummy_preflight_guard_and_output_boundary.*`を追加、`build_stage5_train_pool_video_id_map.py`を廃止（10.3どおり統合）。合成テストは計画の3本から4本になった |
| 5 | **stdlib限定の徹底** | guard・identity・配分の3モジュールをnumpy/h5py非依存にした。結果として**実機に入る前に157項目を実行できた**（本実装環境にはnumpy/h5py/torchが無い）。class weightのみnumpy/h5pyを要するため、`check_dummy_train_core_class_weight.py`は**本環境では未実行**である（11.4-1） |

### 11.3 新規に判明した事項

1. **`check_stage5_s5_15_p1_manifest_audit.py`の`--expected_train_files`既定が162。** 新splitでは144を
   指定する必要がある。`EXPECTED_INPUT_FILES`（180→162）と同種の集合依存値であり、`FILES.md`へ記載した。
   `--expected_val_files`（既定18）は変更不要。
2. **`train_stage5.sh`には`<<'PY'`が2箇所ある**（先頭のtorch-lib探索とteacher preflight）。
   preflightを内容で選ばないと別のブロックを取得する。合成テスト側で対処済みで、コードへの影響はない。
3. **`--confirm`の後、pinを追加してコミットするまで通常経路はすべて停止する。** これは10.10.1の
   「pin未固定の間、通常の学習・評価・推論・checkerを起動できる状態にしない」を満たす意図的な設計だが、
   **実機作業の手順として明示が必要**である（11.5）。builderは確定時に追加すべきpath・hashを表示する。

### 11.4 未実施・残作業

| # | 事項 | 状態 |
| --- | --- | --- |
| 1 | `check_dummy_train_core_class_weight.py`の実行 | 実装済み・構文検査済みだが、本実装環境にnumpy/h5pyが無いため**未実行**。実機の工程3の前に実行し、結果を報告する |
| 2 | guard未挿入の3経路 | `export_anonymized_stage5_metrics.py`、`check_stage5_effective_run_config.py`、`check_stage5_s5_15_p1_manifest_audit.py`。いずれも**H5リストではなくrun成果物やCSVを読む**ため、identityの確立を`ArtifactCoverage`の対応表で行う必要があり、対応表を誰がどの時点で作るかの設計判断が残る。10.4で対象と宣言した経路なので、**放置せず次の差分で扱う**。それまではこの3本を封印期間中の使用対象から外すか、対応表方式を確定するかの判断を求める |
| 3 | 162件用teacher期待値の算出 | 工程5で実施。未実行 |
| 4 | 実機CPU6工程 | すべて未実行 |
| 5 | コミット | 未実施。差分はユーザーの依頼時に提示する |

### 11.5 実機CPU実施量（更新）と手順

10.7の6工程から**工程数は変わらないが、工程2が1ツールの2回起動になり、工程2と工程3の間に
人手のpin承認が入る**。元H5の全点読込は3 pass（工程2a・3・5）のままである。

| 順 | 工程 | 起動 | H5全点読込 |
| --- | --- | --- | --- |
| 1 | 入力監査（リスト・hash・実在・sanity照合・FL入力仕様とトークン確定） | 1 | なし |
| 2a | `build_stage5_bprime_split.py --bootstrap --dry_run_stratification` | 1 | 159件 |
| — | **管理へFL周辺度数のみ返し、層化の成立を確認** | — | — |
| 2b | 同ツール `--bootstrap --confirm` | 1 | なし（2aの封印中間結果をhash照合して再利用） |
| — | **ユーザーがpinを`split_contract_pins.json`へ追加し、git差分を確認してコミット** | — | — |
| 3 | `compute_stage5_train_core_class_weight.py` | 1 | 144件 |
| 4 | `audit_stage5_train_core_mm_metadata.py` | 1 | なし（attrのみ） |
| 5 | `train_stage5.sh PREFLIGHT_ONLY=1 EMIT_TEACHER_EXPECTED=...` | 1 | 162件（QC group含む） |

- 工程2bの契約検査6項目に合格するまで工程3以降へ進まない。pin承認前は工程3〜5も起動できない。
- 異常停止後の再実行は各工程1回まで、理由・停止段階・同一seed／同一入力hashであることを報告する。
- **正常なsplit確定（2b）は1回のみ。** 別seedでの再生成・比較は行わない。
- 本節は実施量の申請であり、**実行承認を得た状態としない**。10.10.4のとおり、合成テスト結果と
  対象・回数の確認後に実機CPU実行を扱う。新規学習・GPU推論・コミットは行わない。

### 11.6 本節の位置づけ

CPU実装と合成テストは完了し、Step 0はS0-1〜S0-7のいずれも**未完了**のままである
（分割・封印・weight算出・mm確認はすべて実機未実行）。
確認を求めるのは11.2の逸脱5件と11.4-2（guard未挿入3経路の扱い）である。
11.4-1（class weight合成テストの実機実行）は実機作業の冒頭で実施して結果を報告する。
実機結果は本報告書のさらに新しい節へ、日付・承認範囲・コードrevision・環境・入力／出力hash・
成否・逸脱・未決事項を付けて追記する。


### 11.7 実装逸脱・未対応経路への管理判断（2026-09-22）

担当: S5-16〜S5-20総括管理チャット。対象: 11.2の5件、11.4-2、および実機処理前の開始条件。
ユーザーの判断依頼に基づき実装コードと報告を照合した。今回、管理側ではテストを再実行していない。
11.1の合格数は実装担当の報告として扱い、class weightの未実行と区別する。

#### 11.7.1 11.2の逸脱5件への判断

| # | 事項 | 判断・条件 |
| --- | --- | --- |
| 1 | SHA-256による層内抽出 | 条件付き受入。実データの分割前なので、抽出アルゴリズムの変更を正式に記録して固定する。seed42、UTF-8での文字列化、`seed`とidentityの区切り、digest昇順、同値時identity順、出力の候補順保持を仕様化し、manifestへ方式名を記録する。numpy方式と同じ抽出結果とは扱わず、結果比較で方式を選び直さない |
| 2 | manifestをパスでなくpin名で指定 | 受入。CLI・環境変数・bash・手順書をpin名で統一する。実際の信頼根拠はpin内容の承認・hash照合であり、名前指定だけで保証されたとはしない |
| 3 | legacyの`sealed: ["train_core"]` | 現状の表現は不採用。旧train162全体と、そのうち封印された18件を同一視している。現在のregistryによるdeny優先は維持し、manifestのsplit名と実際の封印集合の関係を正確に表現する。空配列拒否を通すためだけにtrain_coreを封印扱いにしない |
| 4 | モジュール・テスト構成の変更 | 受入。identity・配分の分離は責務に沿う。旧計画からの追加・廃止と実ファイル一覧を対応させ、テスト本数の変更を実験量の追加と混同しない |
| 5 | stdlib化とclass weightテスト未実行 | stdlib化は受入。ただし「合成テスト完了」は実行済み範囲に限定する。numpy/h5pyを使うclass weight合成テストは実機データ処理に先立って実行し、合格・環境を報告する |

抽出方式変更の理由は版依存を減らす実装上の選択として受け入れる。乱数列の版保証に関する説明を、
従来方式が不正だった根拠としない。既存augmentationはhashをseed導出に利用するもので、層内hashソートと
同一アルゴリズムとは記録しない。再現性の正本は確定したfile list・hashである。

legacyは「現在の封印registryを必ず適用する」という参照と、legacy自身の対象リストを分けて表現する案を推奨する。
実効denyは常にregistryが決める。split名だけの宣言で封印集合の整合性を検査したとはしない。
legacyの学習禁止、評価・推論・checkerでの封印対象拒否、非交差対象の通過を修正後の合成テストで確認する。

#### 11.7.2 guard未挿入3経路の扱い

**Step 0では次の3経路を一時的に使用対象外とする。ArtifactCoverage方式の完成をStep 0のために先行拡大しない。**

- `export_anonymized_stage5_metrics.py`
- `check_stage5_effective_run_config.py`
- `check_stage5_s5_15_p1_manifest_audit.py`

次の条件を満たして運用する。

1. `FILES.md`の「3経路もguard接続済み」という記述を訂正し、未対応・使用禁止を明記する。
2. 直接起動だけでなくlauncherからの間接起動も対象とする。現行evaluate_stage5.shは匿名化exporterを
   条件付きで自動呼出しするため、封印運用では未対応exportを実行できないようにする。
   起動時に使用不可を明示し、設定の既定値だけに依存しない。旧launcherの案内コマンドも訂正する。
3. S0-1の入力監査はStep 0専用手順で実施する。上記の旧auditを無断で代用しない。
   Step 0の共有出力は専用ツールの許可された件数・hash・成否に限定する。
4. coverageを整備する際は、対象成果物の生成時にproducerが入力契約と対象集合から対応情報を記録する方式を基本とする。
   既存成果物は由来を検証してから登録し、内容を先に読んで後から安全と宣言する方式にはしない。
5. S5-18等で実効config突合や匿名化exportが必要になる前に再開条件を満たす。
   対応実装と読込前拒否の合成テストを受け入れるまでは使用しない。

#### 11.7.3 実機分割の前に修正する2件

確認範囲のコードから、報告の意図と実装に次の差を認めた。新しい研究方針ではなく、既定のhash照合条件の修正である。

| 箇所 | 確認した実装 | 必要な修正 |
| --- | --- | --- |
| builderの`declared_inputs` | 入力ファイルからその場で計算したhashを期待hashとして`verify_bootstrap_inputs()`へ渡している | S0-1で監査・固定した期待hashを別の監査記録等から受け取り、現在の入力hashと照合する。自己計算値同士の一致を承認済み入力の検証としない |
| builderの`--confirm` | 中間JSONのschema・入力hash・候補・seedは確認するが、中間ファイル自体のhashを2a時の固定値と照合していない | 2aで固定した中間結果hashを独立に受け取り、2bでファイル自体を照合してから利用する。GT分類・FL群・仕様が変わった中間結果を拒否する |

通常の入力改変、入力リストは同じまま中間結果のGT分類／FL群だけを改変した場合、期待hash未指定を
合成テストへ加える。実データを用いて拒否動作を試さない。正常なsplit確定は1回の条件を維持する。

#### 11.7.4 S0-1〜S0-7への開始条件

- S0-1の所在・形式・保存済みリスト・sanity証拠等の入力監査、および未実行のclass weight合成テストは進めてよい。
  臨床FL値・GT層化統計を人間が閲覧することは含めず、候補データの計数は承認されたsplit構築工程に限る。
- 11.7.1のlegacy修正、11.7.2の未対応経路停止と文書訂正、11.7.3の照合修正、およびそれらの合成テストを行う。
  同じ方針について再承認を求めず、修正結果を報告する。
- 実機の2a・2b以降は上記の修正結果、class weight合成テスト、入力監査結果と実施条件を確認して開始する。
  本返信だけでS0-1〜S0-7の全工程を一括実行承認したとは扱わない。
- S0-2の仕様固定、S0-3〜S0-6の実機結果、S0-7の管理受入はそれぞれ未完了として追跡する。
  報告済み157項目＋既存11項目の合格を、実機6工程の合格やStep 0完了へ読み替えない。

## 12. 11.7の修正条件への対応（2026-09-22）

担当: Step 0実装チャット。状態: **修正実装・合成テスト完了（実行できた範囲）。実機CPU実行・コミットは未実施。**
対象: 11.7.1の5件、11.7.2、11.7.3の2件。11章と相違する箇所は本章を優先する。
同じ方針の再承認は求めず、修正結果のみを報告する。

合成テストは**新規187項目**（split contract 83、split allocation 61、preflight/境界 43）に増え、
既存回帰11項目とあわせてすべて合格した。全`.sh`の構文検査も合格。
class weightのnumpy/h5py依存テストは本環境に両ライブラリが無いため**引き続き未実行**である。

### 12.1 11.7.3 — 実機分割前に必要な2件の照合修正（最優先）

管理の指摘どおり、いずれも「自分自身と照合していて何も検証していない」欠陥だった。修正済み。

| 箇所 | 修正前 | 修正後 |
| --- | --- | --- |
| 入力hash | builderが入力ファイルからその場でhashを計算し、それを期待値として`verify_bootstrap_inputs()`へ渡していた。**必ず一致するため検証になっていない** | `--input_audit_json`（必須）でS0-1が固定した期待hashを受け取る`load_input_audit()`を追加。期待hashが無い／監査対象外の入力がある場合は**停止**する。`verify_bootstrap_inputs(inputs, expected_sha256=...)`へ署名変更 |
| 中間結果 | `--confirm`はschema・入力hash・候補・seedは確認するが、**中間ファイル自体のhashは未照合**。入力リストが同じまま中間結果のGT分類／FL群だけを改変すると全ての内容検査を通過する | `--expected_intermediate_sha256`（2aが出力する値）を受け取る`verify_intermediate_fingerprint()`を追加。未指定・不一致・ファイル欠落で停止 |

合成テストに追加（[12]）: 期待fingerprint未指定での停止、中間ファイル欠落、
**入力リスト・候補集合・seedを一切変えずGT分類とFL群だけを改変した中間結果**が、
内容検査をすべて通過したうえでfingerprintで拒否されること。
入力側も、監査hash不一致、監査記録なし、監査対象外の入力、監査記録の欠落を追加した。
いずれも合成データのみで、実データによる拒否動作の試行は行っていない。正常なsplit確定1回の条件は維持。

### 12.2 11.7.1 — 逸脱5件への対応

**1. SHA-256抽出方式の仕様化（条件付き受入）**

`split_allocation.py`に`SELECTION_METHOD`定数を追加し、manifestへ記録する。
固定した項目: 方式名`sha256_keyed_sort_v1`、鍵`sha256(f"{seed}|{identity}").hexdigest()`、
UTF-8、区切り`|`、seedは10進整数の文字列化、digest昇順・同値時はidentity昇順、
先頭`alloc_k`件を採用、**出力は候補入力順を保持**。seed42は変更していない。

説明の訂正も反映した。コード内コメント・`FILES.md`から「numpy/randomの乱数列が版間で保証されない」を
**従来方式が不正だった根拠として使う書き方を削除**し、「版依存を減らすための実装上の選択であり、
実データ分割前に固定したもの。RNG方式と等価とは扱わず、結果比較で選び直さない」と記述し直した。
既存augmentationは**hashをseed導出に使うもの**で、層内hashソートとは別構成であり同一アルゴリズムとは
記録しない旨も明記した。再現性の正本は確定したfile listとhashである。

合成テスト（[6]）: 仕様文の各条項だけから鍵を再構成して実装と一致すること、
「昇順・同値identity順・先頭N件」の成立、出力が候補入力順であること、
メンバーの到着順に依存しないこと、注記に非等価の記述があること。

**2. pin名指定（受入）** — CLI・環境変数・bash・文書をpin名で統一済み。
`FILES.md`に「信頼の根拠はpin内容の承認とhash照合であり、名前指定だけで保証されるものではない」旨を記載。

**3. legacyの`sealed`表現（不採用→修正）**

`sealed: ["train_core"]`は旧train162全体と封印18件を同一視していた。manifestの項目を次のとおり分離した。

- `sealed_splits`: **そのmanifest自身が封印集合として定義するsplit**。legacyは`[]`（自身は封印集合を定義しない）。
- `seal_registry_required`: **現在のregistryが適用されることの明示的宣言**。全manifestで必須。
  未設定・欠落は「何も封印されていない」ではなく**拒否**する。空配列を通すためにtrain_coreを封印扱いにしない。
- registry側に`sealed_list_identity_sha256`を追加。封印splitを**保持すると主張する**manifestは、
  この指紋と一致しなければ拒否する。**split名の宣言だけを整合性検査としない。**

実効denyは従来どおり常にregistryが決める（`deny→allow`順）。
合成テスト（[3]・[6]）: `seal_registry_required`未設定の拒否、封印split非定義manifestが
正当に読み込まれたうえで**registryにより封印動画を拒否する**こと、未記載splitの封印宣言の拒否、
指紋不一致の拒否、registryに指紋記録が無い状態での保持主張の拒否、
legacyの学習禁止・directory-mode禁止・評価／推論／checkerでの封印対象拒否・非交差対象の通過。

**4. モジュール・テスト構成（受入）** — 11.2-4の対応表のとおり。
テスト項目数の増加（157→187）は検証の細分化であり、**実機実施量の追加ではない**（実機は6工程のまま）。

**5. stdlib化と未実行テスト（受入・限定）** — 11章冒頭の状態表記を
「CPU実装完了。合成テストは実行できた範囲で合格」に訂正した。
class weight合成テストは実機データ処理に先立って実行し、合格と環境を報告する。

### 12.3 11.7.2 — guard未挿入3経路の一時使用停止

`export_anonymized_stage5_metrics.py`、`check_stage5_effective_run_config.py`、
`check_stage5_s5_15_p1_manifest_audit.py`をStep 0では使用対象外とし、ArtifactCoverage方式の
先行拡大は行わない。条件1〜5への対応:

1. **`FILES.md`の訂正済み。** 旧記述は3経路を接続済みとして列挙していた。
   「contract未接続・封印期間中は使用禁止」と、その理由（H5リストではなくCSV・run成果物を読むため
   identityをファイル名から判定できない）を明記した。
2. **間接起動も遮断済み。** `evaluate_stage5.sh`は`EXPORT_ANONYMIZED_METRICS=1`かつ
   split manifestが有効なとき**exit 2で拒否**する。既定値0に頼らず明示的に停止する
   （既定値は呼び出し側が元に戻せるため）。非封印時の既定`1`は変更していない。
   `run_stage5_s5_15_r0_longrun.sh`の案内コマンドも削除し、撤回理由を表示するよう訂正した。
   合成テスト（[6]）で、拒否が起動より前にあること・`exit 2`であること・
   条件が既定値でなくmanifest有効性であること・案内コマンドが消えていることを確認した。
3. **S0-1はStep 0専用手順で実施する。** 旧`p1_manifest_audit`を代用しない旨を`FILES.md`へ記載。
   共有出力は専用ツールの許可された件数・hash・成否に限る。
4. **coverage整備方針を記録した。** producerが生成時に入力契約と対象集合から対応情報を記録する方式を基本とし、
   既存成果物は由来を検証してから登録する。**内容を先に読んで後から安全と宣言する順序を採らない**旨を`FILES.md`へ明記。
5. **再開条件:** 対応実装と読込前拒否の合成テストが受け入れられるまで使用しない。
   S5-18の実効config突合・匿名化exportが必要になる前に満たす必要がある。

### 12.4 変更ファイル（12章分）

| ファイル | 変更 |
| --- | --- |
| `stage5/utils/split_contract.py` | `sealed_splits`/`seal_registry_required`/`sealed_list_identity_sha256`、`load_input_audit()`、`verify_intermediate_fingerprint()`、`verify_bootstrap_inputs()`署名変更、legacy manifest表現、module header訂正 |
| `stage5/utils/split_allocation.py` | `SELECTION_METHOD`追加、抽出方式の説明訂正 |
| `checks/real_h5/build_stage5_bprime_split.py` | `--input_audit_json`必須化、`--expected_intermediate_sha256`、registryへの指紋記録、manifestへ`selection_method`・`intermediate_sha256`記録 |
| `evaluate_stage5.sh` | 未接続exporterの拒否 |
| `checks/real_h5/run_stage5_s5_15_r0_longrun.sh` | 案内コマンドの訂正 |
| `checks/dummy/check_dummy_split_contract.py` | [3]差し替え、[12]追加、bootstrap・legacyテスト更新（68→83項目） |
| `checks/dummy/check_dummy_bprime_split_allocation.py` | [6]追加（53→61項目） |
| `checks/dummy/check_dummy_preflight_guard_and_output_boundary.py` | [6]追加（36→43項目） |
| `docs/stage5/FILES.md` | 3経路の訂正、coverage整備方針、抽出方式・seal表現・照合修正の記述更新 |

### 12.5 実機CPU手順の更新

11.5の6工程・3 passは変わらない。工程1と2の間の受け渡しが増えた。

- **工程1（S0-1入力監査）**は、4入力の期待hashを`input_audit.json`
  （`schema: stage5_step0_input_audit_v1`）として固定・出力する。これが工程2aの必須入力になる。
- **工程2a**は中間結果のfingerprintを表示する。これを**工程2bへ`--expected_intermediate_sha256`として渡す**。
- 工程2bは、入力hash（監査記録と照合）と中間fingerprint（2aの値と照合）の**両方**が一致しないと確定しない。

### 12.6 本節の位置づけ

11.7.1・11.7.2・11.7.3の修正と、その合成テストを完了した（実行できた範囲）。
Step 0はS0-1〜S0-7のいずれも**未完了**である。分割・封印・weight算出・mm確認はすべて実機未実行であり、
報告済みの187項目＋既存11項目の合格を実機6工程の合格やStep 0完了へ読み替えない。
次は、11.7.4のとおり (a) class weight合成テストの実行、(b) S0-1入力監査（臨床FL値・GT層化統計の
人間による閲覧を含まない）を行い、結果と実施条件を確認したうえで実機2a以降を扱う。
新規学習・GPU推論・コミットは実施しない。

## 13. 手順1〜3の合成テスト結果とS0-1監査ツールの実装（2026-09-26）

担当: Step 0実装チャット。状態: **実機での合成テスト全合格。S0-1監査ツール実装・合成テスト完了。
実機のS0-1実行・コミットは未実施。**

### 13.1 実機での合成テスト結果（ユーザー実行、2026-09-26）

| テスト | 結果 |
| --- | --- |
| `check_dummy_split_contract` | 83項目 / 失敗0 |
| `check_dummy_bprime_split_allocation` | 61項目 / 失敗0 |
| `check_dummy_preflight_guard_and_output_boundary` | 43項目 / 失敗0 |
| `check_dummy_fixed_list_mode`（既存回帰、bash半分含む） | 11 test 合格 |
| **`check_dummy_train_core_class_weight`（11.4-1の未実行分）** | **20項目 / 失敗0（実機で初回実行）** |

実行環境は`/home/kodaira/anaconda3/envs/dualtrack311/bin/python`。
stdlib系3本は本実装環境と同一件数で合格し、既存directory-mode経路の回帰も確認できた。
**11.4-1の残件は解消した。** ただし11.7.1-5が求める環境記録のうち、
**numpy／h5pyの版は未取得**である。S0-1実行時にあわせて記録する。

### 13.2 S0-1監査ツールの実装（12.5の積み残し）

12.5で「工程1が`input_audit.json`を出力し工程2aの必須入力になる」と記載しながら、
**それを生成するツールを実装していなかった。** 本節で解消した。

`checks/real_h5/audit_stage5_step0_inputs.py`（stdlib限定、H5を一切開かない）。

- 旧train162／validation18リストの検査（実在・teacher v7 pattern・重複なし・相互排他・件数）と両hash算出。
- train sanity 3を**動画identityで確定**し、旧評価の`video_id_map_DO_NOT_SHARE.csv`の
  `split == "train_sanity"`行と1対1照合する。**alias 3件・件数3は証拠として採用しない**（9.6）。
  再導出は`--allow_sanity_rederivation`かつ旧trainリストのcontent hash固定を要求する
  （選択がリスト順に依存するため）。
- 候補159件の算出、臨床FL表の構造確認（列・1候補1行・候補外行なし・宣言済みトークンで全値が分類可能）。
- **`input_audit.json`（`schema: stage5_step0_input_audit_v1`）を出力。** builderが参照する
  4ラベル（`old_train_list` / `old_val_list` / `sanity_list` / `clinical_fl_csv`）で期待hashを固定する。
- **確認に失敗した場合は監査記録を書かずに停止する。** 未確認の入力が固定hashを帯びてbuilderへ渡ることを防ぐ。
- 出力境界: 臨床FL値・per-tokenの件数（＝欠損件数）を**一切収集・出力しない**。
  非数値トークンの文字列のみをprivate記録へ残す（トークン集合の宣言に必要であり、測定値ではないため）。
  stdout・共有JSONに動画IDは出さない。欠損件数の公開は従来どおり2aの1回に限る。
- 封印registryが承認済みの場合は、旧train162が封印動画を含むため契約により拒否される（S0-1は封印前の1回限り）。

あわせて`stage5/utils/train_sanity_selection.py`（stdlib）を追加した。
`evaluate_stage5.py`・`prepare_stage5_evaluation_data.py`の既存コピーは**変更していない**
（評価経路の挙動変更はStep 0の範囲外）。合成テストが`evaluate_stage5.py`のコピーを
テキスト抽出して両者の一致を確認するため、乖離は検出される。

合成テスト`check_dummy_step0_input_audit`（40項目、全合格）:
正常系のhash固定、**全ての失敗経路が記録を書かずに停止すること**（H5欠落・件数違い・
対応表なし・**3件の別動画を指すおとり対応表**・fingerprint未固定の再導出）、
FL構造の各停止条件、出力境界、`evaluate_stage5.py`コピーとの一致、
およびリストから1件除くと選択が変わること（＝保存済み選択を優先する理由）。

### 13.3 合成テスト合計

新規5本・**227項目**（split contract 83、allocation 61、preflight/境界 43、input audit 40）＋
class weight 20、既存回帰11 test。本実装環境では227項目中187項目を実行可能で全合格、
class weight 20項目は実機で合格（13.1）。input audit 40項目は実機で再実行する。

### 13.4 本節の位置づけ

S0-1の**実行はまだ行っていない**。Step 0はS0-1〜S0-7のいずれも未完了である。
次は実機でS0-1監査ツールの合成テストを実行し、続いてS0-1本実行（実リスト・実FL表を入力）へ進む。
実機の2a以降は、S0-1の結果と実施条件を確認してから扱う。新規学習・GPU推論・コミットは実施しない。

## 14. FL表の準備待ちに先行できる作業（2026-09-26）

担当: Step 0実装チャット。状態: **CPU実装完了。実機の合成テスト3本と部分S0-1が未実行。**
臨床FL表の準備に時間を要するとのユーザー申告を受け、FL表に依存しない作業を先行させた。
実施量の申請であり、工程2a以降の実行承認を求めるものではない。

### 14.1 依存関係の整理

| 工程 | FL表への依存 | 状態 |
| --- | --- | --- |
| S0-1のリスト検査・train sanity 3の確定 | **なし** | 先行可能（14.2で分割対応） |
| S0-1の臨床FL表の構造確認 | あり | 待ち |
| 工程2a・2b（split構築・確定） | あり | 待ち |
| 工程3（class weight）・工程4（mmメタ）・工程5（teacher期待値） | 間接（train_core144が2bの出力） | 待ち |
| mmスケール供給元の確認（9.7-2） | **なし。ユーザー回答のみ** | 先行可能。S5-17の開始条件 |

工程4のmmメタ監査は、依頼書6章が対象をtrain_coreに限定しているため、162件で先行実施はしない。

### 14.2 S0-1の分割（`--defer_clinical_fl`）

S0-1のうちリスト検査とsanity照合はFL表に依存しないため、2パスに分けられるようにした。
**sanity照合の証拠不足は9.6で「splitを確定せず報告する」事項であり、早期に判明する価値が大きい。**

- `--defer_clinical_fl`でFL表を省略して実行できる。省略のみ（フラグなし）はエラーで、黙って飛ばされない。
- 書き出される`input_audit.json`は`complete: false`で、**FL表の期待hashを一切含まない**。
  builderは`clinical_fl_csv`の期待値が無いことを検出して停止するため、
  **部分監査を完了済みS0-1と取り違えることができない**（合成テストで確認済み）。
- 工程1の実行回数は1回から2回になる。実データのH5読込は依然ゼロ。

### 14.3 未整備だった合成テスト2本（先行して作成）

監査の結果、**実際に分割を行うツールに専用の合成テストが無い**ことが判明した。先行して作成した。

| 追加 | 内容 |
| --- | --- |
| `check_dummy_bprime_split_builder` | builderのend-to-end。dry runがFL周辺度数**のみ**を出すこと（multi-region周辺度数・クロス表・セル別配分を出さないこと）、封印物が0700/0600の封印ディレクトリへ直接書かれること、fingerprint未指定・改竄中間結果・監査記録と不一致の入力の拒否、6項目の契約検査、**pin後に通常guardが封印動画を拒否するところまでの一巡**、初回経路の再実行拒否 |
| `check_dummy_train_core_mm_metadata` | mm監査の抑制。spacing 1.0を既定placeholderとして扱い測定値としないこと、x/yを個別に比較すること、結論が「取得不能」ではなく**UNCONFIRMED**であること、解決不能動画を他動画の値で埋めないこと、代理FL・`T_FL`を算出しないこと |

いずれもnumpy/h5pyを要するため本実装環境では未実行（構文検査のみ）。実機で実行する。

### 14.4 合成テストの現状

| テスト | 項目数 | 実行状況 |
| --- | --- | --- |
| `check_dummy_split_contract` | 83 | 実機合格（13.1） |
| `check_dummy_bprime_split_allocation` | 61 | 実機合格（13.1） |
| `check_dummy_preflight_guard_and_output_boundary` | 43 | 実機合格（13.1） |
| `check_dummy_train_core_class_weight` | 20 | 実機合格（13.1） |
| `check_dummy_step0_input_audit` | **53**（43→53、14.2の分） | 本環境合格。実機未実行 |
| `check_dummy_bprime_split_builder` | 新規 | **未実行** |
| `check_dummy_train_core_mm_metadata` | 新規 | **未実行** |
| `check_dummy_fixed_list_mode`（既存回帰） | 11 test | 実機合格（13.1） |

環境記録（11.7.1-5）: Python 3.11.15 / numpy 2.2.2 / h5py 3.13.0。

### 14.5 小修正

臨床FL表の読込を`utf-8`から`utf-8-sig`へ変更した（S0-1監査ツールとbuilderの**両方**）。
表計算ソフトの「CSV UTF-8」出力はBOMを含み、1列目の列名が一致せず
「その列が存在しない」という誤解を招くエラーになることを確認したため。
片方だけ変更すると、hashは一致するのに解釈が異なる状態になるので同時に変更した。BOMの有無どちらでも読める。

### 14.6 本節の位置づけ

Step 0はS0-1〜S0-7のいずれも未完了である。次は実機で14.4の未実行3本を実行し、
続いて部分S0-1（`--defer_clinical_fl`）を実行する。FL表が整い次第、完全なS0-1を再実行して
`input_audit.json`を完成させ、その後に工程2aを扱う。新規学習・GPU推論・コミットは実施しない。

## 15. FL参照値の由来に関する前提の訂正（2026-09-26）

担当: Step 0実装チャット。状態: **前提の相違を報告。総括管理の判断待ち。実装・実機実行は行っていない。**
ユーザー回答（mmスケールの供給元、FL値の由来）と`research/stage5/s5-16/step0/reference_scripts/mk_legcase_json.py`の確認結果を記録する。
**S5-16 9.1に記録された前提と相違があり、S5-17の仮説H17-1に影響する。**

### 15.1 mmスケール（9.7-2・管理判断4への回答）

ユーザー回答: **供給元はmm/pixel。動画ごとに異なる可能性がある。**

- 動画単位の可変性は既存実装で対応済み（`audit_stage5_train_core_mm_metadata.py`は動画ごとに、
  x/yを個別に読み取る）。
- **残る未確認事項:** mm/pixelが**元frame基準かcrop/resize後基準か**。`pixel_xy`はlocal crop座標系のため、
  元frame基準なら`local_resize_scale`・`local_crop_left/top`の逆変換を挟む必要がある。
  S5-16 10.0の換算経路（`pixel_xy`→crop逆変換→元frame→mm）はこの前提で書かれており、確認が要る。
- 中間H5の`spacing`が可視化用既定値である件（9.7-2）とは別に、実mm/pixelの保存場所・形式の確認は続く。

### 15.2 FL値の由来（**前提の相違**）

S5-16 9.1はユーザー回答として「FL参照値: **臨床計測値**が動画ごとに存在する」と記録している。
今回の回答と[`mk_legcase_json.py`](../../../research/stage5/s5-16/step0/reference_scripts/mk_legcase_json.py)の確認により、実際は次のとおりであることが判明した。

| 区分 | 実態 |
| --- | --- |
| アノテーション済み180動画のFL値 | **BBoxアノテーション由来の導出値**。臨床計測値ではない |
| 真の実測FL値を持つデータ | 存在するが、**BBoxを含むアノテーションが未付与**。かつ実測は動画取得とは別に行われており、**動画に映る大腿骨長と一致しない可能性がある**（ユーザー申告） |

[`mk_legcase_json.py`](../../../research/stage5/s5-16/step0/reference_scripts/mk_legcase_json.py)の`femur_traj_len`が実際に計算している値:

1. 各frameで`leg` BBoxを切り出し、median blur背景差分＋CLAHE＋Otsu二値化を行い、
   **最大輪郭の重心**を求める（輪郭が取れない場合はBBox中心へフォールバック）。
2. 重心を`(x/width, y/height)`で正規化し、最近傍＋時系列単調制約でtrackへ連結する。
3. **連続frame間のユークリッド距離を積算**した軌跡長が`femur_traj_len`。

したがってこの値は、大腿骨の解剖学的長さそのものではなく、
**sweep中に大腿骨断面の重心が画像平面上を移動した軌跡長**である。留意点:

- **単位は正規化座標**で、xはwidth、yはheightという**異なる分母**で割られている。等方的な距離ではなく、mmでもない。
- **画像平面内の移動のみ**を積算しており、pseudo-3Dのtrackingが持つ面外方向の運動を含まない。
- track範囲は`start`/`end`という**別系統のXMLアノテーション**（`femur_end_anno`）に依存する。
- スクリプトのパスは別データセット（`251117_検証用`）を指しており、180動画を覆うかは未確認。

### 15.3 S5-17の仮説H17-1への影響

S5-16 10.1のH17-1は「GTから導くどの幾何表現・代理FL定義が、**臨床FLと最もよく一致するか**」であり、
採否規則は「train_coreで**臨床FLとの誤差が最小**のものを事前規則で1つ選んで固定する」である。

参照側のFLがGTアノテーション由来であるなら、**この比較は同一ラベルの2通りの導出同士の比較になり、
臨床的な正しさを示すものにはならない**（［推論］）。GT由来の代理FL定義を、別のGT由来導出（軌跡長）との
一致度で選ぶことになり、選ばれるのは「既存のアノテーション処理系列に最も近い定義」である。
診断として無意味ではないが、H17-1の目的（臨床値との一致の確認）とは異なる。

真に独立した実測FLは未アノテーション動画側にのみ存在し、しかも動画内大腿骨長との一致が
保証されない旨のユーザー申告がある。これはA案（外部test）の参照値の妥当性に関わるため、
**A案の設計時に持ち越すべき留保として今記録する。**

### 15.4 「データセットから導出する」案の評価

ユーザー提案「FL値のまとめ表を作るより、pseudo-3Dのセグメンテーションマスクから端点を自動取得するほうが早い」
について、Step 0の範囲で実施できるかを検討した。**Step 0では実施すべきでない。**

1. **teacher v7 H5に端点情報が無い。** 必須datasetは`point_cloud/frame_order`、`annotation/point_label`、
   `annotation/valid_mask`、`frame_annotation/*`、各invalidation groupであり、
   [`mk_legcase_json.py`](../../../research/stage5/s5-16/step0/reference_scripts/mk_legcase_json.py)が使う`start`/`end`マーカーに相当するものは含まれない（確認済み）。
2. **マスクから端点を定義して代理FLを作る作業は、S5-17そのものである。**
   S5-16 10.1が「(a)〜(e)の幾何表現を比較し、事前規則で1つを選んで固定する」と定めた対象であり、
   Step 0依頼書6章は「ここでは代理FLを計算・選択せず、`T_FL`も推定しない」と明記している。
   Step 0で1案を実装すれば、S5-17の比較対象を先に既成事実化することになる。

### 15.5 Step 0の層化変数について判断を求める

D-037・9.2は層化の第2軸を「臨床FLの三分位」としているが、15.2によりこの名称と前提が成立しない。
実装は未着手で、いずれの案でも実データ処理は増えない。**総括管理の判断を求める。**

| 案 | 内容 | 評価 |
| --- | --- | --- |
| A | 既存の`femur_traj_len`等を159件分の表として用意し、現行仕様のまま層化する | 表の作成コストが残る。名称を「臨床FL」から改める必要がある。180動画を覆うかも未確認 |
| **B（推奨）** | 層化の第2軸を、split構築処理内でGTから直接算出する**層化専用のサイズ代理量**に置き換える（例: GT陽性frame数、主軸方向のGT広がり）。**代理FL定義ではないこと、S5-17へ引き継がないことを明記** | 表が不要。追加I/Oなし（builderは既にGT点を読む）。S5-17の選定を先取りしない。層化の目的（大きさ方向の均衡）は保たれる |
| C | FL軸を落とし、multi-regionのみの2層で層化する | 最小変更だが、大きさ方向の均衡が失われる |

**B案を推奨する理由:** 層化の目的は「大きさ／範囲の偏りを抑える」ことであり、
その目的にはGT由来の代理量で足りる。臨床的な正しさを要求されるのはS5-17以降の評価であって、
split構築時の層化ではない。またB案では臨床値を一切読まないため、
9.2で設計した「FL周辺度数をsplit確定前に1回だけ共有する」手続き自体が不要になる。
ただし11.7.2で「候補GT統計の共有出力は認めない」とされているため、
**2aの共有出力は「層が構成可能か」の真偽と候補件数のみ**となる。これは漏洩面ではより安全である。

B案採用時に必要な変更: D-037・9.2・10.x・`FILES.md`の「臨床FL三分位」記述の改訂、
`split_allocation.py`のFL intake経路の置換（配分・抽出・契約検査は不変）、合成テストの差し替え。
`--defer_clinical_fl`を設けた14.2の部分S0-1は、B案でもそのまま有効である。

### 15.6 追加のユーザー回答: 実測FL付きデータのアノテーション作成

2026-09-26の追加回答: **実測FL付きデータのアノテーション作成は可能だが時間がかかる。
実測FLを持つデータは数例のみ。それでも意味があるなら、S5-18・S5-19の学習のタイミングで進めたい。**

S5-16 11.5は「A案の準備はS5-17〜S5-20と並行して進めてよい。ただし方式選択には一切使わない」と
しており、**S5-18・S5-19と並行して進めること自体は既存方針の範囲内**である。

数例でできないこと・できることを分けて評価した。

- **できない:** S5-16 9.2がA案に期待するend-to-end FL testとしての統計的評価。
  10.4の採否規則（MAE、失敗率、動画別で過半が改善）は数例では判定できない。
  **数例の結果をproduction判断やtask採否の根拠にしない。** A案の完成も意味しない。
- **できる可能性:** (1) 同一動画について「アノテーション由来FL」と「実測FL」の両方が得られる
  **ペア比較**。この対応は現在どこにも存在せず、15.3の循環性に対する唯一の外部的な足がかりになる
  （［推論］）。件数が0から数例へ増えることは、100から110へ増えることとは質的に異なる。
  (2) `pixel_xy`→crop逆変換→元frame→mmの換算チェーンの健全性確認。数例で足り、15.1の未確認事項の
  検証にも使える。
- **前提となる制約:** 実測は動画取得とは別に行われており、不一致が出ても
  「パイプラインの誤り」か「参照値が動画に対応していない」かを区別できない。
  この曖昧さが残るかぎり、件数を増やしても解釈可能な結論は得られない。

**推奨: 条件付きで実施。** 先に実測値と動画の対応可能性を確認し（アノテーション作業を伴わない）、
確認できた場合に限りS5-18・S5-19と並行して作成する。用途は
「導出FLの実測に対する較正とmm換算チェーンの検証」に限定し、採否判断には使わない。
確認できない場合は着手せず、その事実をA案の参照値に関する留保として記録する。

本件は**Step 0の進行を妨げない。S5-18開始前までに決まればよい。**

### 15.7 本節の位置づけ

前提の相違の報告であり、実装・実機実行は行っていない。判断を求めるのは2件である。

| # | 判断事項 | Step 0への影響 |
| --- | --- | --- |
| 1 | 15.5の層化変数（推奨B案） | **工程2aが停止中。これが解ければ再開できる** |
| 2 | 15.6の実測FL付きデータのアノテーション作成 | なし。S5-18開始前までに決めればよい |

15.3（H17-1の循環）は**S5-17開始前に総括管理で扱うべき事項**として提起し、Step 0では解決しない。
15.1のmm/pixelは回答を得たが、元frame基準かcrop後基準かは未確認のまま残る。

本節と同内容を、管理チャットへ渡す判断依頼書として`.tmp/`へ別途まとめた
（`stage5_s5_16_step0_fl_premise_decision_request.md`、gitignore対象）。

## 16. D-039実装と合成テスト結果（2026-09-27）

担当: Step 0実装チャット。状態: **実装完了。本環境で実行できた合成テストは全合格。
numpy/h5py依存の3本は実機未実行。実機のS0-1・工程2a以降とコミットは未実施。**
対象: 管理チャットが確定した層化量`gt_positive_frame_count`と中央値の成立条件、
および軽微な補足・訂正4件（`docs/stage5/s5-16/stage5_s5_16_step0_stratification_spec_proposal.md`版3）。

### 16.1 進行順の1（管理記録への決定記載）

管理記録へ**D-039**（層化第2軸の変更）と**D-040**（実測FL 5例のアノテーション）を追加した。
3.1の現在状態表にも反映した。過去のD-037本文は履歴として保持し、変更はD-039で明示している。

### 16.2 実装内容

| ファイル | 内容 |
| --- | --- |
| `stage5/utils/split_allocation.py` | FL intake（`classify_clinical_fl`／`FLIntake`／`assign_fl_groups`／`FL_GROUP_*`）を削除。`STRATIFICATION_QUANTITY`／`STRATIFICATION_CONDITION`／`STRATUM_GROUP_*`と、`assign_stratification_groups`／`stratification_group_medians`／`assert_stratification_is_constructible`／`count_boundary_tie_members`／`assert_supported_versions`を追加。**三分割の順位規則・最大剰余配分・seed42・`SELECTION_METHOD`・契約検査6項目は不変** |
| `stage5/utils/split_contract.py` | `INPUT_AUDIT_SCHEMA`をv2へ。v1は「再実行せよ」と明示して拒否 |
| `checks/real_h5/audit_stage5_step0_inputs.py` | 臨床FL関連の引数・検査・`--defer_clinical_fl`を全廃。**期待hashは3入力**。分割パスが無くなったため記録は常に`complete: true` |
| `checks/real_h5/build_stage5_bprime_split.py` | **`read_gt_points`の使用を停止し、`read_and_validate_gt_points`を自前実装**（16.3）。両軸を同一passで算出。中間schemaをv2。設定の固定と確定時照合を追加。共有出力を縮小 |

### 16.3 変換前検査（レビュー指摘2への対応）

既存`read_gt_points`が`astype`を検査前に行うことを実機動作で確認した
（`3.7→3`、`2→True`、`NaN→True`、`shape[0]`のみ照合）。
このためbuilderは同関数を使わず、生配列に対して
dtype・次元・行数対応・有限性・**整数値浮動小数のint64表現範囲**・ラベル値域・
`valid_mask == (point_label != -1)`を検査してから変換する。
検査済みdictを**監査済みの`analyze_video`／`classify_video`へそのまま渡す**ため、
クラスタリングと分類のロジックは変更しておらず、読み込みも1動画1回のままである。

### 16.4 共有出力（レビュー指摘3への対応）

工程2aの共有出力を**「GT集計」と「照合・再現情報」に分離**した。

- GT集計として共有するのは**候補件数と層構成可能性の真偽のみ**。
  群サイズ・中央値・multi-region周辺度数・動画別値・セル別配分は封印側。
- `intermediate_sha256`とschema版・層化量版・成立条件版・抽出方式名は共有側に残す。
  確定処理がこれらを必要とし、かつGT値を含まないため。

### 16.5 設定照合（レビュー指摘4への対応）

中間生成物へ層化量の仕様版・成立条件・multi-region設定・抽出設定・seed・
`internal_test_size`・入力hashを固定し、確定時は**中間生成物を唯一の基準**とする。
CLI引数が与えられて不一致なら**いずれも黙って優先せず停止**する。
さらに**CLI引数が省略されていても、実装が対応していない版は拒否する**。

### 16.6 合成テスト結果

| テスト | 項目数 | 状況 |
| --- | --- | --- |
| `check_dummy_split_contract` | **84**（83→84。v1監査記録の拒否を追加） | 本環境合格 |
| `check_dummy_bprime_split_allocation` | **67**（61→67。層化量の群割当・成立条件・境界同値・版拒否へ差し替え） | 本環境合格 |
| `check_dummy_step0_input_audit` | **38**（53→38。FL構造・遅延パス廃止、v2発行とv1拒否を追加） | 本環境合格 |
| `check_dummy_preflight_guard_and_output_boundary` | 43 | 本環境合格 |
| `check_dummy_fixed_list_mode`（既存回帰） | 11 test | 本環境合格 |
| `check_dummy_bprime_split_builder` | 更新（FL fixture廃止、**変換前不正値検出12件**、設定不一致、v1中間生成物拒否、共有出力縮小を追加） | **未実行**（numpy/h5py） |
| `check_dummy_train_core_mm_metadata` | 新規（未実行のまま） | **未実行** |
| `check_dummy_train_core_class_weight` | 20 | 実機合格済み（13.1）。再実行で回帰確認 |

**注意:** 本環境で実行できた232項目の合格は、実機確認・S0-1〜S0-7の受入完了を意味しない。

### 16.7 次の手順

管理チャット指示の進行順に従う（1は16.1で完了）。

2. 実機で合成テストを実行（`check_dummy_bprime_split_builder`と
   `check_dummy_train_core_mm_metadata`は**初回実行**）。
3. 結果報告。
4. 改訂S0-1（v2監査記録の発行）。**旧v1の部分監査記録は手動で完了扱いにしない。**
5. 工程2a（dry run）。共有されるGT集計は候補件数と構成可能性の真偽のみ。
6. **工程2bの分割確定・封印は、dry run結果の管理確認後。**

新規学習・GPU推論・コミットは実施していない。

### 16.8 実機初回実行での不具合と修正（2026-09-27）

`check_dummy_bprime_split_builder`の実機初回実行が失敗した。**原因は合成fixtureの不備であり、
実装は正しく動作していた。**

- 症状: dry runが`medians 6.0, 6.0, 6.0 are not strictly increasing`で停止。
- 原因: fixtureが全24動画をGT陽性frame数6で生成していた。
  `gt_positive_frame_count`は**まさにこの値**であるため、全件同値＝退化した三分割になり、
  3.3の成立条件が正しく拒否した。**検出されるべきものが検出された。**
- 修正: fixtureのGT陽性frame数を動画ごとに変化させた（`3 + (i % 9)`、背景点は全12 frameに配置）。
  候補21件で中央値4／7／10となり狭義単調増加を満たす。セル数6、配分4件、境界同値6件。
- 併せて、失敗時に共有ファイルの不在でtracebackしていた箇所を、
  dry run失敗時は空のfingerprintを返して報告する形に改めた。
- **この偶発から、退化検出のend-to-endテストが無いことが判明したため追加した**（テスト[8]）。
  全件同値のfixtureで、(1) dry runがexit 2で停止、(2) 「fall backしない」と報告、
  (3) **共有出力も中間生成物も書かれない**（成立条件が保存より前に評価される）、
  (4) 拒否メッセージに動画IDが出ない、を確認する。
- 旧FL時代の関数名`test_dry_run_releases_only_the_fl_marginal`も併せて改名した。

本環境で実行できる5本（84／67／43／38／11 test）は修正後も全合格である。
`check_dummy_bprime_split_builder`・`check_dummy_train_core_mm_metadata`は引き続き実機での確認が必要。

### 16.9 実機2回目の実行結果と修正（2026-09-27）

| テスト | 結果 |
| --- | --- |
| `check_dummy_train_core_mm_metadata` | **23項目 / 失敗0（初回実行、合格）** |
| `check_dummy_bprime_split_builder` | 88項目中85合格、**3件失敗** |

失敗3件はいずれも**テスト側の判定方法の誤りであり、実装からの漏洩ではない。**
共有ファイルの内容を確認したところ、released keyは
`schema` / `num_candidates` / `strata_constructible` / `intermediate_sha256` /
`schema_versions` / `withheld` のみで、GT集計は候補件数と真偽だけである。

原因は、テストが共有ファイルの**文字列**に対して単語検索していたことである。
`withheld`欄は「group sizes, medians, ... per-cell allocation」と
**何を出さないかを明記している**ため、"allocation" や "median" が文字列として一致してしまう。
stdoutも「Group sizes, medians, multi-region counts ... are sealed」と述べるため同様に一致する。
すなわち、**出さないと明記していること自体を漏洩と誤判定していた。**

修正は、単語検索をやめて**構造と実値**で判定する形に置き換えた。

1. 共有ファイルのtop-level keyが許可集合と**完全一致**すること。
2. 数値を持つfieldが`num_candidates`**だけ**であること（boolは除外）。
3. 封印側の中間生成物から**実際に算出された中央値・群サイズ・動画別値**を読み、
   それらが共有ファイルのデータfieldとstdoutに現れないことを確認する。
   schema文字列は"stage5"等の数字を含み誤一致するため、比較対象から除外する。
   指紋はhexのため、stdoutの該当行を除外する。
4. `withheld`欄が**何を withhold するかを明記していること**を逆に確認する。
5. 動画IDについては、注記を含むファイル全体に対して従来どおり検査する。

これにより、`group_sizes`等を共有側へ追加すれば1と2の両方が発火する。
併せて、テスト修正時に`text`変数の未定義参照を作り込んでいたため修正し、
全関数に対する未定義名検査を実施して0件であることを確認した。

本環境で実行できる5本（84／67／43／38／11 test）は修正後も全合格である。
`check_dummy_bprime_split_builder`の再実行が必要である。

## 17. D-039実装の完了報告（進行順3、2026-09-27）

担当: Step 0実装チャット。状態: **実装・合成テスト完了（進行順1〜2）。
実機のS0-1（進行順4）以降とコミットは未実施。**

### 17.1 合成テスト結果（全合格）

| テスト | 項目数 | 実行環境 |
| --- | --- | --- |
| `check_dummy_bprime_split_builder` | **96** | 実機（3回目で全合格） |
| `check_dummy_split_contract` | 84 | 実装環境 |
| `check_dummy_bprime_split_allocation` | 67 | 実装環境 |
| `check_dummy_preflight_guard_and_output_boundary` | 43 | 実装環境 |
| `check_dummy_step0_input_audit` | 38 | 実装環境 |
| `check_dummy_train_core_mm_metadata` | **23** | 実機（初回、合格） |
| `check_dummy_train_core_class_weight` | 20 | 実機（13.1で合格。D-039の変更対象外だが未再実行） |
| 合計 | **371項目** | |
| `check_dummy_fixed_list_mode`（既存回帰） | 11 test | 実装環境で合格 |

環境: Python 3.11.15 / numpy 2.2.2 / h5py 3.13.0。

**この371項目の合格は合成データに対するものであり、実機のS0-1〜S0-7の確認・受入完了を意味しない**（9.2.6）。

### 17.2 実機実行で見つかり修正した3件

いずれも**テスト側の不備であり、実装の欠陥ではなかった**。ただし2件は
「テストが無かった／判定が誤っていた」ことの発見であり、記録に残す。

| # | 症状 | 実態と対応 |
| --- | --- | --- |
| 1 | dry runが`medians 6.0, 6.0, 6.0`で停止 | fixtureが全動画をGT陽性frame数6で生成していた。層化量がまさにこの値のため全件同値＝退化であり、**成立条件が正しく拒否した**。fixtureを可変化（中央値4／7／10）。併せて**退化検出のend-to-endテストが無いことが判明したため追加**（テスト[8]） |
| 2 | 共有出力の漏洩判定3件が失敗 | テストが文字列を単語検索しており、`withheld`欄が「何を出さないか」を明記していること自体を漏洩と誤判定していた。**構造と実値による判定へ置換**（key集合の完全一致、数値fieldは`num_candidates`のみ、封印側の実算出値がデータfieldとstdoutに現れないこと） |
| 3 | `text`の未定義参照 | 2の修正時に作り込み。修正し、全関数に対する未定義名検査を実施して0件を確認 |

1と2は、**合成テストが実データに触れる前に設計の穴を検出した例**である。
特に1は、9.2.4が求めた「3群へ件数を割り当てられることを構成可能性と同一視しない」という
要件が実装で機能していることを、偶発的にではあるが実証した。

### 17.3 確定した仕様の実装状況

| 仕様 | 実装 | 検証 |
| --- | --- | --- |
| 層化量`gt_positive_frame_count` | `split_allocation.py`＋builderの同一GT pass | テスト[1][8]、allocation 67項目 |
| 成立条件（中央値の狭義単調増加） | `assert_stratification_is_constructible` | 退化fixtureで停止・保存前に評価（テスト[8]） |
| 変換前の生配列検査 | `read_and_validate_gt_points`（`read_gt_points`不使用） | **12ケース**（非整数frame／Inf／NaN／int64範囲外／負値／mask=2／maskがNaN／pixel_xyが(N,3)／pixel_xyにNaN／範囲外ラベル／mask不整合／空配列）＋正常ゼロが0を返すこと |
| 設定の固定と確定時照合 | 中間生成物を唯一の基準、不一致は停止 | テスト[3]（seed／`min_component_points`／`internal_test_size`／link距離／未対応版／v1中間生成物） |
| 共有出力の縮小 | GT集計は候補件数と真偽のみ。指紋・仕様版は照合情報として別扱い | テスト[1] |
| v2 schemaと旧仕様拒否 | 監査記録・中間生成物とも | 監査38項目、contract 84項目 |
| 三分割・配分・seed42・SHA-256抽出・契約検査6項目 | **不変** | allocation 67項目、テスト[4] |
| pin後に通常guardが封印動画を拒否 | 既存 | テスト[5]（一巡を確認） |

### 17.4 次（進行順4以降）

4. **改訂S0-1の実行**（v2監査記録の発行）。旧v1の部分監査記録は手動で完了扱いにしない。
   なお部分S0-1は結局一度も実行していないため、今回が初回である。
5. 工程2a（dry run）。共有されるGT集計は候補件数と層構成可能性の真偽のみ。
6. **工程2bの分割確定・封印は、dry run結果の管理確認後。**

`FILES.md`・`data_construct.md`・管理記録（D-039／D-040）は同期済み。
コミットはユーザー依頼時に差分を提示する。新規学習・GPU推論は実施していない。

## 18. 教師データに2系統の命名規則が存在することの判明（2026-09-27）

担当: Step 0実装チャット。状態: **実装・合成テスト完了。実機のS0-1再実行は未実施。**
進行順4（改訂S0-1）の初回実行が失敗し、その原因調査で判明した事実を記録する。

### 18.1 判明した事実

S0-1が`a listed file name yields 0 video identities`で停止した。
実IDを出さない形状調査（数字→`#`、英字→`a`）をユーザーに実行してもらった結果、
**180動画の教師データには2系統の命名規則が存在する。**

| 系統 | 形式 | 例 | 件数 |
| --- | --- | --- | --- |
| timestamp | `YYYYMMDD_HHMMSS_N` | `20250403_103433_091` | 170 |
| **case** | `D-D_NN[_NN]` | `1-2_34_56`、`1-2_34` | **10**（train 9、validation 1） |

train sanity 3動画のうち**1件がcase系**である。

### 18.2 影響

**Step 0への影響（対応済み）**

封印契約の判定単位は動画identityであり、timestamp形式しか認識しない実装では
**180動画のうち10動画を識別できない**。これは封印が名指しできなければならない当の集合である。
`split_identity.py`の`VIDEO_ID_PATTERN`を2系統の**列挙**へ拡張した。
2形式は相互にcross-matchしない（timestamp形式にハイフンは無く、case形式の後続群は1〜2桁に限るため
8桁の日付を飲み込めない）ことをコードで確認した。
**列挙であって寛容なパターンではない。** どちらにも一致しない名前は従来どおり拒否し、
形状を添えて報告する。

**既存ツールへの影響（Step 0の範囲外、要記録）**

Stage 5の既存ツールはすべて`[0-9]{8}_[0-9]{6}_[0-9]+`のみで判定している
（`export_anonymized_stage5_metrics.py`、`check_stage5_gt_component_count.py`、
`write_stage5_s5_15_run_manifest.py`、`check_stage5_s5_15_p1_manifest_audit.py`ほか）。
したがって**これらのprivacy self-checkはcase系の名前が混入しても検出しない。**

ただし`assert_share_bundle_anonymous`は2重の検査を持ち、
`forbidden`（private rowsの`original_*`全field）による**完全一致検査がcase系も捕捉する**ため、
**既存の共有成果物から実IDが漏洩した形跡は無い。** 破られているのは
パターン照合という二次的な網であり、一次の網は機能している。
S5-16では既存checkerを改修しないが、**S5-18以降でこれらを使う際に修正すべき事項として記録する。**

なお`export_anonymized_stage5_metrics.py`の`video_token_from_path()`はcase系で`None`を返す。
alias付与への影響は本節では調査していない。同exporterは11.7.2で封印期間中の使用対象外である。

### 18.3 自分の検査にも同じ盲点があったこと

本チャットが書いた合成テストの「動画IDが出力に現れないこと」の検査も、
同じくtimestamp形式のみを見ていた。**case系の名前が漏れても合格していた。**
`contains_video_identity()`を追加し、**両系統を見る**よう11箇所すべてを置き換えた。
privacyの検査を片方の規則だけで書くと、もう片方の規則の漏洩を静かに通してしまう。

### 18.4 併せて修正した診断

エラーが「どのリストの何番目か」を示していなかった。`masked_shape()`を追加し、
リスト名・位置・**名前の文字形状**を報告するようにした。実IDは出さないため共有ログへ貼れる。

```
FAIL: train entry #57 yields 0 video identities; exactly one is required.
  expected shape: ########_######_# ... (YYYYMMDD_HHMMSS_N)
  actual shape  : #-#_##_##_aaaaaaaaaa_...
```

### 18.5 合成テスト

`check_dummy_step0_input_audit`を45→**56項目**へ拡張し全合格。追加内容:

- 2系統それぞれの解決、多桁case番号、**cross-matchしないこと**、
  case形式が8桁日付を飲み込まないこと、未知の形式は依然として拒否されること。
- **両系統が混在するリストのend-to-end監査**と、その出力に両系統いずれのIDも現れないこと。
- 命名不一致時に、リスト名・位置・形状を示し実IDを出さないこと。

他の runnable 4本（84／67／43／11 test）も全合格。

### 18.6 未確認事項

case系10動画が**なぜ別系統なのか**（別データソース、別取得系、手動命名等）は確認していない。
S0-1の再実行は可能だが、この10動画が180動画セットの正規の一部であることは
ユーザー・管理チャットの確認事項として残る。

## 19. 合成テストの最終集計（2026-09-27）

18章の変更後、**Step 0新規の7本が合格**した。**17.1の集計（371項目）は本節で更新する**
（`check_dummy_step0_input_audit`が38→56項目に増えたため）。

**集計の区別（35.4）:** 「7本」は**Step 0で新規に作成したテスト**であり、
`check_dummy_fixed_list_mode`（既存回帰、11 test）は**別枠**である。項目数の合計に含めない。

| テスト | 項目数 | 実行環境 |
| --- | --- | --- |
| `check_dummy_bprime_split_builder` | 96 | 実機 |
| `check_dummy_split_contract` | 84 | 実装環境 |
| `check_dummy_bprime_split_allocation` | 67 | 実装環境 |
| `check_dummy_step0_input_audit` | **56** | 実機 |
| `check_dummy_preflight_guard_and_output_boundary` | 43 | 実装環境 |
| `check_dummy_train_core_mm_metadata` | 23 | 実機 |
| `check_dummy_train_core_class_weight` | 20 | 実機（13.1。D-039の変更対象外、未再実行） |
| **新規7本の合計** | **389** | |
| `check_dummy_fixed_list_mode`（**既存回帰。別枠**） | 11 test | 実装環境 |

環境: Python 3.11.15 / numpy 2.2.2 / h5py 3.13.0。

**389項目の合格は合成データに対するものであり、S0-1〜S0-7の実機確認・受入完了を意味しない**（9.2.6）。
進行順2（実装・合成テスト）と3（結果報告）は完了し、次は4（改訂S0-1の実行）である。

18章で判明した2系統命名への対応を含め、封印判定・出力境界・入力検査の各テストは
両系統を対象とするよう更新済みである。

## 20. case系命名パターンの欠陥修正と重複診断（2026-09-27）

担当: Step 0実装チャット。状態: **修正・合成テスト完了。S0-1の再実行は未実施。**

### 20.1 18章で導入したパターンに切り捨ての欠陥があった

S0-1が`train list names the same video more than once`で停止した。調査の結果、
**18章で追加したcase系パターン自体に欠陥があった。**

18章では後続群を`(?:_[0-9]{1,2})+`（1〜2桁）と定義した。3桁以上の群があると
backtrackで**途中までしか一致せず、識別子が切り捨てられる**。

```
1-2_34_567  ->  1-2_34      (切り捨て)
1-2_34_891  ->  1-2_34      (切り捨て)
```

**異なる2動画が同一identityに潰れる。** 封印契約においてこれは最悪の故障であり、
ある動画が別の動画の身代わりになることを許す。18章の
「2形式はcross-matchしない」という確認は行ったが、**同一形式内での衝突は検査していなかった。**

修正: 後続群を`(?:_[0-9]+)+`とし、**桁数制限を撤廃**した。パターン自体が衝突を生むことはなくなる。

### 20.2 撤廃に伴い受け入れたトレードオフ

桁数制限は、case形式が8桁の日付を飲み込むのを防ぐ目的も兼ねていた。撤廃により、
`1-2_20250403_103433_091`のような**両形式を含む名前**はtimestamp部分だけでなく全体が
1つのidentityになる。

- この形式の名前は180動画セットに**存在しない**（2026-09-27確認）。
- 結果は依然として**単一の安定したidentity**であり、衝突ではない。
- 切り捨てによる衝突は実在し得る故障で、こちらは仮定上の名前に関する挙動である。
  **衝突を防ぐ側を優先した。**

合成テストは、この挙動変化を**削除せず明示的な検査へ書き換えた**。
桁数の異なる後続群が別identityのままであること、両形式を含む名前が
1つのidentityになりそれが切り捨てでないことを確認する。

### 20.3 重複エラーの診断を改善

`train list names the same video more than once`は、**性質の異なる2つの問題**を
同じ文言で報告していた。

| 症状 | 意味 |
| --- | --- |
| 同一ファイル名が2回記載されている | 入力の不備 |
| **異なるファイル名が1つのidentityへ解決される** | **identity規則の欠陥**（20.1がこれ） |

後者は封印の前提を壊すため、区別して報告する必要がある。
衝突したidentityの件数、該当エントリの**位置**、名前が同一か異なるか、
および**マスクした形状**を出すようにした。実IDは出さない。

### 20.4 合成テスト

`check_dummy_step0_input_audit`を56→**65項目**へ拡張し全合格。追加:

- 3桁以上の後続群が切り捨てられないこと、桁数の異なる群が別identityのままであること。
- 両形式を含む名前が単一identityになり、それが切り捨てでないこと。
- 重複時に、**どちらの種類か**・位置・形状を報告し、実IDを出さないこと。

他の runnable 4本（84／67／43／11 test）も全合格。

### 20.5 S0-1の停止原因は未確定

20.1の修正により**パターン由来の衝突は解消した**が、
実機の重複が(a)パターンの切り捨てによるものだったのか、
(b)リストに実在する重複なのかは**まだ確認できていない**。
再実行時に(b)であれば、20.3の診断が「同一ファイル名が複数回」と明示する。
その場合は入力の不備であり、**splitを確定せず報告する**（9.6と同じ扱い）。

## 21. timestamp形式に第4セグメントが存在する — 分割単位の前提に関わる（2026-09-27）

担当: Step 0実装チャット。状態: **identity規則は修正・合成テスト完了。
ただし分割単位についての前提確認が必要で、S0-1は再実行していない。**

### 21.1 判明した事実

20.3で追加した診断により、S0-1の停止原因が特定できた。
**train 162件のうち14グループが、異なるファイルでありながら同一identityへ解決されていた。**
形状は次のとおり。

```
########_######_###_##_aaaaaaaaaa_...     ← YYYYMMDD_HHMMSS_NNN_NN_pointcloud_...
```

すなわち**timestamp形式には省略可能な第4セグメント`_NN`が存在する**。
既存のStage 5ツールが一様に使う`[0-9]{8}_[0-9]{6}_[0-9]+`は第4セグメントの手前で停止するため、

```
20250403_103433_091_01  ┐
20250403_103433_091_02  ├→ すべて 20250403_103433_091
20250403_103433_091_03  ┘
```

と潰れる。診断が報告したグループは3件が1組、残りは2件ずつで、計14 identityが重複していた。

### 21.2 identity規則の修正（完了）

trailing の数値セグメントをすべてidentityへ含めるよう修正した
（`[0-9]{8}_[0-9]{6}_[0-9]+(?:_[0-9]+)*`）。
これにより各**ファイル**が一意で曖昧さのないidentityを持つ。3セグメント形式・case形式は影響を受けない。

合成テストを65→**68項目**へ拡張し全合格。第4セグメントがidentityに含まれること、
`_01`／`_02`／`_03`が3つの別identityになること、3セグメント形式と衝突しないこと、
第4セグメント付きの名前を含む混在リストのend-to-end監査を追加した。
他の runnable 4本（84／67／43／11 test）も全合格。

### 21.3 **前提確認が必要: `_NN`は独立した動画か、1回の検査の分割か**

**ここから先は実装判断の範囲を超えるため、確認を求める。**

S5-16 9.1はユーザー回答として「**同一患者・同一検査の重複は180動画内には存在しない**」を記録し、
これを根拠に「**分割単位は動画でよい**」としている（D-037も同じ前提に立つ）。

`_NN`の意味により、結論が正反対になる。

| 解釈 | 帰結 |
| --- | --- |
| **(a) 独立した別動画**（別被検者・別検査がたまたま近い時刻） | 9.1の前提は成立。ファイル単位の分割でよく、21.2の修正で完了 |
| **(b) 1回の取得を分割したclip**（同一被検者・同一検査） | **9.1の前提が成立しない。** ファイル単位で分割すると、同一検査のclipがtrain_coreとinternal_testへ分かれ、**封印が防ぐはずのleakageがsplit生成時点で発生する** |

(b)の場合、抽出の単位は3セグメントのグループでなければならない。
これは層化・配分・契約検査すべてに関わる仕様変更であり、**実装側で決めない。**

判断に必要な情報を、実IDを出さずに取得できる。次を実行していただきたい。

```bash
cd /mnt/data/3d_projects/models/Stage5
/home/kodaira/anaconda3/envs/dualtrack311/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
import re
from collections import Counter, defaultdict
from pathlib import Path

WA = "<W-A run dir>"
ts = re.compile(r"^([0-9]{8}_[0-9]{6}_[0-9]+)(?:_([0-9]+))?_pointcloud")
groups = defaultdict(list)
plain = 0
for label in ("train_files.txt", "val_files.txt"):
    for line in Path(f"{WA}/{label}").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = ts.match(Path(line).name)
        if not m:
            continue
        if m.group(2) is None:
            plain += 1
        else:
            groups[m.group(1)].append(m.group(2))
print(f"3-segment-only files : {plain}")
print(f"groups with a 4th segment: {len(groups)}")
print("group sizes:", dict(Counter(len(v) for v in groups.values())))
print("4th-segment values seen:", sorted({s for v in groups.values() for s in v})[:12])
print("are they contiguous from 01?:",
      all(sorted(v) == [f"{i:02d}" for i in range(1, len(v) + 1)] for v in groups.values()))
PY
```

第4セグメントが`01, 02, 03...`と連番であれば(b)（同一取得の分割）である可能性が高い。
出力に実IDは含まれない。

### 21.4 併せて確認すべき既存への影響

第4セグメントを見落としているのは本チャットの実装だけではない。
`export_anonymized_stage5_metrics.py`の`video_token_from_path()`も3セグメント形式で
tokenを取るため、**第4セグメントだけが異なるファイルは同じtokenになる。**
S5-15の動画別metricsのalias付与がこれらをどう扱ったかは未調査である。
18.2と同様、S5-16では既存checkerを改修しないが、**記録して引き継ぐ。**

評価レポートやS5-15報告に記載された「動画別」の件数・指標が、
このtoken衝突の影響を受けていないかは、別途確認が要る可能性がある。

### 21.5 現在の状態

- identity規則の修正と合成テストは完了している。
- **21.3の(a)/(b)が確定するまでS0-1を先へ進めない。**
  (b)であれば分割単位の仕様変更が必要で、層化・配分・契約検査に波及する。
- Step 0はS0-1〜S0-7のいずれも未完了。分割・封印・weight算出・mm確認はすべて実機未実行。

## 22. 第4セグメントの意味の確認結果（2026-09-27）

担当: Step 0実装チャット。状態: **S0-1のブロッカーは解消。ただし残る確認事項が1件ある。**

### 22.1 確認結果

21.3で提起した(a)/(b)について、中間pseudo-3D H5の`source_video`属性で判別した。
複数メンバーを持つ17グループ（39ファイル）すべてについて、**メンバーは互いに異なる元動画ファイル
から生成されている**（`all different` 17/17、中間H5の解決失敗0件）。

| 解釈 | 判定 |
| --- | --- |
| (b) 1回の取得を機械的に分割したclip | **否定された。** clipであれば同一の`source_video`を共有するはずである |
| (a) それぞれ独立したファイル由来の動画 | **支持される** |

したがって**21.3で懸念した「同一取得のclipがsplitで分断される」形のleakageは生じない。**
S5-16 9.1の前提とD-037の「分割単位は動画」は、この点について覆らない。
21.2のidentity修正（trailing数値セグメントをすべて含める）で十分であり、
**分割単位の仕様変更は不要である。**

数字の内訳も整合した。3セグメントのみ92、第4セグメント付き78（3件×5、2件×12、1件×39）、
timestamp計170、case 10、合計180。

### 22.2 残る確認事項（S0-1は妨げない）

**`source_video`が異なることは、元動画ファイルが異なることを示すのみで、
被検者・検査が異なることを直接示すものではない。**

17グループのメンバーは`YYYYMMDD_HHMMSS_NNN`まで一致している。
秒単位まで一致する接頭辞を共有することは、**同一セッション内で取得された複数動画**である
可能性を示唆する（［推論］）。その場合、元動画ファイルは別でも被検者・検査は同一であり得る。

S5-16 9.1は「同一患者・同一検査の重複は180動画内には存在しない」というユーザー回答を記録しているが、
この回答が本構造（17グループ・39ファイルが秒単位一致の接頭辞を共有する）を踏まえたものかは確認していない。

- 同一セッション・同一被検者であれば、ファイル単位の抽出は依然としてleakage経路になり得る。
- 判別にコードは不要である。**取得プロトコル上、3セグメント接頭辞を共有する動画が
  同一被検者・同一検査のものかをユーザーへ確認すれば足りる。**
- リスクは非対称である。仮に独立でもグループ単位で引けば抽出の自由度が17グループ分落ちるだけだが、
  同一検査でファイル単位に引けば封印が無効になる。

**この確認は工程2a（抽選）の前に必要であり、S0-1（入力監査・hash固定）は妨げない。**
S0-1は候補集合の導出と入力hashの固定のみを行い、抽選単位を確定しない。

### 22.3 現在の状態

- identity規則は確定（21.2）。重複衝突は解消し、S0-1は実行可能である。
- 22.2の確認が済むまで**工程2aへは進まない。**
- 中間H5が39件すべて解決できたことは、工程4（mmメタ監査）の前提としても良い材料である。

## 23. S0-1完了と、同一被検者・同一検査の確認による前提訂正（2026-09-27）

担当: Step 0実装チャット。状態: **S0-1（工程1）完了。工程2aは着手しない。
前提の訂正により、抽選単位の仕様変更が必要である。総括管理の判断を求める。**

### 23.1 S0-1の実行結果（進行順4、完了）

| 項目 | 結果 |
| --- | --- |
| 封印状態 | pins存在・registry未承認（pre-seal。想定どおり） |
| train list | 162件 identity `74cd6030cd07c852` |
| validation list | 18件 identity `de3f3d513fc0bf41` |
| train sanity | **3件確定。保存済み`selected_train_files.txt`を第一選択とし、旧評価ID mapで動画identity照合済み** |
| 候補 | 159件 |
| 監査記録 | `input_audit.json`（schema `stage5_step0_input_audit_v2`）を発行 |

9.6が求めた「alias 3件・件数一致を証拠にしない」照合は満たされた。
S0-1はH5を1つも開かず、入力の確定と3入力の期待hash固定のみを行った。

### 23.2 前提の訂正: 接頭辞を共有する動画は同一被検者・同一検査である

22.2の確認に対し、ユーザーは**「これらは基本的に同一被験者・同一検査のものです」**と回答した。

**S5-16 9.1に記録された「同一患者・同一検査の重複は180動画内には存在しない」という前提は、
この構造については成立しない。** 同節はこの回答を根拠に「分割単位は動画でよい」としており、
D-037もその前提に立つ。したがって**抽選単位の見直しが必要である。**

- 17グループ・39ファイルが、同一被検者・同一検査の複数動画である。
- **ファイル単位で抽選すると、同一検査の動画がtrain_coreとinternal_testへ分かれる。**
  これは封印が防ごうとしているleakageそのものであり、split生成の時点で発生する。
- 22.1で否定されたのは「1回の取得を機械的に分割したclip」という形態のみであった。
  `source_video`が異なることは被検者・検査が異なることを意味しない、と22.2で留保したとおりである。

### 23.3 先に確認すべき事項: 既存splitへの影響

17グループはtrainとvalidationを合算して検出した。**グループがtrainとvalidationにまたがっていれば、
既存のvalidation 18動画による評価が同一検査のleakageを含むことになる。**
これはStep 0の範囲を超え、**S5-15の評価結果とS5-16の判断根拠に関わる。**

確認が済むまで、この点について結論を述べない。次を実行して判別する（実IDは出ない）。

```bash
cd /mnt/data/3d_projects/models/Stage5
WA="$(sed -n 's/^WA_RUN_DIR="\(.*\)"$/\1/p' checks/real_h5/run_stage5_s5_15_arm.sh)"
EV=/mnt/data/3d_projects/stage5_evaluations/260919/pointnext_s_EX260919_s5_15_r0long50_none_gn8_cwfixed_lr1e3_ep50_bs1_acc8_nopad

/home/kodaira/anaconda3/envs/dualtrack311/bin/python - "${WA}" "${EV}" <<'PY'
import sys, re
from collections import Counter, defaultdict
from pathlib import Path

WA, EV = sys.argv[1], sys.argv[2]
ts = re.compile(r"^([0-9]{8}_[0-9]{6}_[0-9]+)(?:_[0-9]+)?_pointcloud")
case = re.compile(r"^([0-9]+-[0-9]+)(?:_[0-9]+)+_pointcloud")

def key(name):
    m = ts.match(name) or case.match(name)
    return m.group(1) if m else name

def load(path):
    return [Path(l.strip()) for l in Path(path).read_text().splitlines() if l.strip()]

train = load(f"{WA}/train_files.txt")
val = load(f"{WA}/val_files.txt")
sanity = load(f"{EV}/evaluation_data/selected_train_files.txt")

side = defaultdict(lambda: Counter())
for p in train: side[key(p.name)]["train"] += 1
for p in val:   side[key(p.name)]["val"] += 1

straddling = [k for k, c in side.items() if c["train"] and c["val"]]
print("exam groups total        :", len(side))
print("groups spanning train+val:", len(straddling), "  <- leakage in the EXISTING split if > 0")
print("  files involved         :", sum(sum(side[k].values()) for k in straddling))

sanity_keys = {key(p.name) for p in sanity}
print("sanity videos            :", len(sanity), "in", len(sanity_keys), "group(s)")
print("extra train files tied to a sanity group:",
      sum(side[k]["train"] for k in sanity_keys) - len(sanity))

cand = [p for p in train if key(p.name) not in sanity_keys]
cg = Counter()
for p in cand: cg[key(p.name)] += 1
print("candidate files          :", len(cand), "in", len(cg), "groups")
print("candidate group sizes    :", dict(Counter(cg.values())))
PY
```

### 23.4 抽選単位の仕様変更について判断を求める

D-037は144／18／18を**ファイル数**として定めている。グループを分割できないなら、
**18ファイルちょうどを引けるとは限らない**（引いたグループのファイル数合計が18になる保証がない）。
180ファイルは141の単独ファイル＋17グループ＝158の検査単位に対応する。

想定される選択肢と、それぞれの帰結は23.3の結果を見てから整理する。
件数の扱い（18ファイルか18単位か）、層化量のグループへの集約方法、契約検査6項目の件数条件、
sanity 3を含むグループの扱いが、いずれも影響を受ける。

**実装側では決めない。** 23.3の確認結果とあわせて総括管理へ諮る。

### 23.5 現在の状態

- **S0-1（工程1）は完了。** 監査記録は発行済みで、旧v1記録の手動昇格は行っていない。
- **工程2a以降は、23.3の確認と23.4の判断が済むまで着手しない。**
- 21.2のidentity修正（ファイル単位で一意）は引き続き有効で、封印の判定単位としては正しい。
  変更が必要なのは**抽選の単位**であって、identityの定義ではない。

## 24. 同一検査重複の影響範囲と、既存validationの汚染（2026-09-27）

担当: Step 0実装チャット。状態: **事実の報告と判断依頼。工程2aは着手しない。**

### 24.1 なぜ同一検査の重複が問題になるのか

評価指標は「**未知の検査・未知の被検者に対してどれだけ当たるか**」を推定するために測る。
同一検査の動画が学習側と評価側の両方にあると、この推定が壊れる。

- モデルは評価対象と**同じ大腿骨・同じ被検者・同じプローブ設定・同じ術者・同じ session**を
  学習時に見ている。超音波sweepでは骨の大きさ・輝度・深度・gainがほぼ同一であり、
  frame自体が酷似することもある。別被検者の動画同士より**桁違いに似ている**。
- したがって指標は、汎化ではなく**記憶**を部分的に測る。値は**楽観側へ偏る**。
- 偏りの大きさは事前に見積もれない。「少し高めに出る」程度で済む保証はない。

Step 0のinternal_testについては、これは設計目的そのものを無効化する。
B′の目的は「今後の開発から隔離された評価集合」を作ることだが、
internal_testの動画がtrain_coreの動画と同一検査なら、**隔離は成立していない。**
しかもファイル名が異なるため、**封印フラグもhash照合も異常を検出しない。静かに壊れる。**

### 24.2 判明した構造

| 項目 | 値 |
| --- | --- |
| 180ファイルが対応する検査単位 | **149グループ**（31ファイルが追加メンバー） |
| **train/validationにまたがるグループ** | **4グループ、旧train／validation合計17ファイル** |
| sanity 3動画が属するグループ | 3グループ。**同一検査の兄弟9ファイルがtrainにある** |
| 候補（train162 − sanityグループ12ファイル） | **150ファイル / 133グループ**（サイズ 1:120、2:9、3:4） |

### 24.3 既存のvalidation 18動画は汚染されている

**4グループがtrainとvalidationにまたがる。** すなわち、
**validation 18動画のうち少なくとも4件、最大13件が、train動画と同一被検者・同一検査である。**

これはStep 0の範囲を超え、**S5-15の評価結果とS5-16の判断根拠に関わる。**

**ただし、S5-16の方針判断を覆すものではない（［推論］）。** 汚染は指標を楽観側へ偏らせる。
S5-15はvalidation成績が**低い**と結論した。汚染があるなら真の汎化性能は
**観測値以下**であり、結論の方向は変わらず、むしろ強まる。
train sanityとvalidationの乖離についても、汚染はvalidationをtrain側へ近づけるため、
**乖離は過小評価されていた**ことになる。D-035（S5-15終了）・D-036（S5-16方針）の
判断は維持できると考えるが、**この解釈の採否は総括管理の判断事項**である。

正確な影響件数（validation 18のうち何件か）は未取得である。24.5の確認で得られる。

### 24.4 D-037の件数は維持できる

グループを分割しない場合でも、**144／18／18というファイル数は達成可能である。**

- 候補のグループサイズは1・2・3であり、合計18ファイルになる組み合わせは存在する。
- train_core = 162 − 18 = 144。
- 変わるのは**抽選の単位**（ファイル→検査グループ）と、
  配分がグループの**ファイル数で重み付け**される点である。
- sanityグループの12ファイルはtrain_coreに固定される。候補は159→**150ファイル／133グループ**になる。

すなわち、**D-037の件数条件と契約検査6項目はそのまま維持でき、変更は抽選ロジックに限られる。**

### 24.5 判断を求める事項

| # | 事項 | 実装側の見解 |
| --- | --- | --- |
| 1 | 抽選単位を検査グループへ変更し、ファイル数で重み付けする | **必要。** これをしないと封印が静かに無効化される |
| 2 | 層化量のグループへの集約方法 | グループ内のファイルの`gt_positive_frame_count`をどう集約するか（合計／最大／代表）を決める必要がある。**未決** |
| 3 | 汚染したvalidationの扱い | D-037は「validation18の内容・順序を維持」としている。(A) 現状維持＋限界として記録、(B) 汚染分を除いた部分集合を主指標にする、等。**実装側では決めない** |
| 4 | S5-15評価結果の再解釈 | 24.3の整理（結論の方向は変わらない）の採否 |
| 5 | 9.1・D-037の前提記述の訂正 | 「同一患者・同一検査の重複は存在しない」を新decision IDで訂正する必要がある |

validation側の正確な汚染件数は、次で取得できる（実IDは出ない）。

```bash
cd /mnt/data/3d_projects/models/Stage5
WA="$(sed -n 's/^WA_RUN_DIR="\(.*\)"$/\1/p' checks/real_h5/run_stage5_s5_15_arm.sh)"
/home/kodaira/anaconda3/envs/dualtrack311/bin/python - "${WA}" <<'PY'
import sys, re
from collections import Counter, defaultdict
from pathlib import Path
WA = sys.argv[1]
ts = re.compile(r"^([0-9]{8}_[0-9]{6}_[0-9]+)(?:_[0-9]+)?_pointcloud")
case = re.compile(r"^([0-9]+-[0-9]+)(?:_[0-9]+)+_pointcloud")
key = lambda n: (ts.match(n) or case.match(n)).group(1) if (ts.match(n) or case.match(n)) else n
load = lambda p: [Path(l.strip()) for l in Path(p).read_text().splitlines() if l.strip()]
side = defaultdict(Counter)
for p in load(f"{WA}/train_files.txt"): side[key(p.name)]["train"] += 1
for p in load(f"{WA}/val_files.txt"):   side[key(p.name)]["val"] += 1
bad = {k: c for k, c in side.items() if c["train"] and c["val"]}
print("validation files sharing an exam with train:", sum(c["val"] for c in bad.values()), "/ 18")
print("train files in those exams               :", sum(c["train"] for c in bad.values()))
print("per-group (train,val) sizes              :", sorted((c["train"], c["val"]) for c in bad.values()))
PY
```

### 24.6 現在の状態

S0-1は完了している（23.1）。**工程2a以降は24.5の判断が済むまで着手しない。**
identity規則（ファイル単位で一意）は変更不要で、変わるのは抽選単位である。

## 25. 同一検査動画の独立性についてのユーザー説明と、24章の評価の訂正（2026-09-27）

担当: Step 0実装チャット。状態: **24章の評価を訂正。判断依頼を縮小する。**

### 25.1 ユーザー説明

2026-09-27、24章に対しユーザーは次を説明した。

> 同一検査における別動画は、基本的に全く異なる映像構成を持つ。これはsweepの角度や方向の違いによる。
> sweep構成の違いにより、映り込む他オブジェクト（他の体の部位など）やスペックルノイズも全く違う映像になる。
> したがって（ほぼ唯一の共通事項と言える、別計測のFL実測値を除けば）十分に別データとして扱える差異を
> 保持しており、汎化を妨げない。

### 25.2 24.1の評価を訂正する

**24.1で述べた機構の主要部分は成立しない。** 撮像条件の説明を受けて訂正する。

sweep角度・方向が異なり、写り込む他部位とスペックル実現値まで異なるなら、
「評価対象と酷似したframeを学習時に見ている」という状態にはならない。
**点単位segmentationの指標（IoU／F1／recall／FPR）について、
同一検査由来であることによる楽観バイアスは小さいと考えられる**（［推論］、
撮像条件に関するユーザー説明に依拠する）。

24.1で「別被検者の動画同士より桁違いに似ている」「偏りの大きさは見積もれない」と書いたのは、
**撮像条件を確認せずに超音波一般の想定を当てたものであり、過大であった。**

### 25.3 それでも残る2点

ユーザー説明は**外観の独立性**についてのものである。外観が独立でも、次は共有される。

**(1) 目的量そのものが共有される**

ユーザー自身が例外として挙げたとおり、FL実測値は同一検査で同一である。
これは付随的な共通項ではなく、**プロジェクトの最終評価対象**である。

- S5-20cはinternal_testでFLを評価する。S5-20の修正案3ではFL・幾何がモデル出力になり得る。
- train_coreとinternal_testに同一検査の動画があると、そのFL値は**設計上一致している**。
- 外観から患者を同定できなくても、目的量の一致は指標に効く経路が別に残る（［推論］）。

なお、GT由来の代理FLは、trajectory由来の導出であればsweep幾何に依存して動画ごとに異なり得る（［推論］）。
共有されるのは**解剖学的な真の長さ**であり、GT由来代理値そのものではない可能性がある。
この点は本チャットで確認していない。

**(2) 実効標本サイズがファイル数より小さい**

これは外観の独立性とは別の統計的な論点である。
18ファイルが18検査から来る場合と14検査から来る場合では、得られる情報量が異なる。
動画別の中央値・勝敗数・MAEといったS5-16が採否に使う統計量は、
**ばらつきを過小評価する**。偏りの有無とは独立に成立する。

### 25.4 訂正後の推奨

**推奨は変わらないが、理由と緊急度が変わる。**

- 24.1の「封印が静かに無効化される」という表現は**撤回する。** segmentation評価の妥当性が
  損なわれているという主張は、撮像条件の説明により支持されない。
- それでも**抽選を検査グループ単位にすることを推奨する。** 理由は25.3の(1)(2)であり、
  「correctness上の緊急事態」ではなく「**費用がほぼ無いのに、目的量の共有と
  実効標本サイズの過小評価という2つの経路を閉じられるため**」である。
- 費用は小さい。24.4のとおり**144／18／18は維持でき**、契約検査6項目も不変で、
  変更は抽選ロジックに限られる。抽選の自由度が133グループ分に落ちるだけである。

### 25.5 既存validationの扱い（24.3の訂正）

**24.3の「validationは汚染されている」という評価も弱める。**
segmentation指標については25.2のとおり影響は小さいと考えられ、
**S5-15の評価結果とD-035・D-036の判断は維持できる。**

ただし**validationでFLや幾何を指標にする場合は、4グループ・旧train／validation合計17ファイルの構造を考慮する必要がある。**
S5-16 10.0はvalidationの臨床FLを開発中に参照しないと定めており、
S5-17のH17-2も臨床FLではなくGT由来幾何との比較であるため、**現時点で抵触しない。**
S5-20以降でvalidation上のFL指標を使う場合に再確認する事項として記録する。

### 25.6 判断依頼の縮小

24.5の5件を次へ改める。

| # | 事項 | 状態 |
| --- | --- | --- |
| 1 | 抽選単位を検査グループとし、ファイル数で重み付けする | **推奨（25.4）。** 判断を求める |
| 2 | 層化量のグループへの集約方法（合計／最大／代表） | 1を採る場合に必要。**未決** |
| 3 | 汚染validationの扱い | **取り下げる。** 現状維持でよく、25.5を限界として記録する |
| 4 | S5-15評価結果の再解釈 | **取り下げる。** 判断は維持できる（25.5） |
| 5 | 9.1・D-037の前提記述の訂正 | **必要。** 「同一患者・同一検査の重複は存在しない」は、ファイル単位では成立しない。ただし25.1の撮像条件による独立性とあわせて記述する |

1を採らない場合、S0-1で確定した候補159ファイルのまま従来のファイル単位で抽選でき、
実装変更は不要である。その場合は25.3の(1)(2)を**既知の限界として記録して進む**ことになる。

### 25.7 現在の状態

S0-1は完了（23.1）。工程2aは25.6の1・2が確定してから着手する。
24章の記述は履歴として残し、評価の訂正は本章を優先する。

## 26. 抽選単位の判断（ユーザー決定、2026-09-27）

担当: Step 0実装チャット。状態: **判断確定。工程2aへ着手できる。**

### 26.1 決定

**25.6-1（抽選単位を検査グループへ変更する案）は採らない。従来どおりファイル単位で抽選する。**

ユーザーの判断理由:

- **今回の180例データにはFLの真の実測値が用意されていない。**
  したがって25.3-(1)「目的量そのものが共有される」という懸念は、この集合では成立しない。
- 映像面では25.1のとおり同一検査であることによる問題が現れにくい。
- 以上より、現時点で問題になりにくい。

実装側の評価: **この論拠は25.3-(1)の前提を実際に外している。**
同節の懸念は「共有される目的量が存在すること」を前提にしていたが、
15.2のとおり180動画に臨床実測FLは存在せず、手元のFL値はアノテーション由来の導出値である。
さらに25.3で触れたとおり、GT由来の代理FLはsweep幾何に依存して動画ごとに異なり得る（［推論］）。

### 26.2 記録する限界

決定に伴い、次を既知の限界として記録する。**解消はしない。**

1. **実効標本サイズがファイル数より小さい**（25.3-(2)）。
   候補159ファイルは133グループ＋sanityグループに対応する。
   動画別の中央値・勝敗数・MAEは、検査単位で見たばらつきを過小評価する。
   S5-16が採否に使う「動画別で過半が改善」等の統計量にこの限界が付く。
2. **旧train162とvalidation18にまたがる同一検査は4グループで、所属ファイルは両集合の合計17件。validation側の正確な件数は未確認（既存記録上の範囲は4〜13件）**（24.2〜24.3、25.5。件数の誤転記は38章で訂正）。
   segmentation指標への影響は小さいと**考えられる**が、これは**撮像条件に関するユーザー説明に基づく
   推論であり、影響が小さいことも検査単位の独立性も実証されていない**（［推論］）。
   **validationでFL・幾何を指標にする場合は再確認を要する。**
3. 9.1の「同一患者・同一検査の重複は180動画内には存在しない」は、
   **ファイル単位では成立しない。** 25.1の撮像条件による独立性とあわせて訂正する（D-041）。

### 26.3 アノテーション待ち5例への引き継ぎ（重要）

ユーザーは「アノテーション未完了の5例については、**各例に複数の動画が対応する**ため、
あらためて引き合いに出す価値がある」と指摘した。**これはD-040の設計に直接効く。**

5例は**実測FLを持つ唯一のデータ**であり、25.3-(1)の懸念が成立する唯一の集合である。

| 論点 | 内容 |
| --- | --- |
| 比較設計 | 1つの実測値に複数動画が対応する。**各動画を独立した比較標本として数えない。** 比較の単位（例単位か動画単位か）を9.3.3の「探索的比較」の手順固定時に決める |
| 実効標本 | 5例であり、動画数ではない。9.3.2・9.3.3の「数例」という限定はこの意味で読む |
| **副産物として得られる検査** | 同一例の複数動画からそれぞれ導出FLを算出すれば、**導出方法の検査内一貫性**を測れる。実測値を使わずに成立する再現性検査であり、25.3で未確認とした「GT由来代理FLがsweep幾何に依存して動画ごとに異なるか」に直接答える |
| **用途制限（35.3-5）** | **この検査内一貫性も、S5-17の方式選択・較正へ流用しない。** 当初「S5-17の判定材料になり得る」と書いたが、これは9.3.3の用途限定（探索的比較とmm換算の健全性確認の補助材料に限る）に反するため**撤回する。** 比較単位は例単位を踏まえて別途固定する |

この引き継ぎ内容はD-040の補足としてS5-18開始前に確定する。本チャットでは実施しない。

### 26.4 S0-1の結果は有効である

ファイル単位を維持するため、**S0-1が確定した候補159ファイルと入力hashはそのまま有効で、再実行は不要。**
`input_audit.json`（v2）をそのまま工程2aへ渡す。

### 26.5 現在の状態

- 25.6の判断依頼1は決定、2（グループ集約方法）は**1を採らないため不要となった**。
- 3・4は25.6で取り下げ済み。5（前提記述の訂正）はD-041として記録する。
- **工程2a（dry run）へ着手できる。** 工程2bは従来どおりdry run結果の管理確認後。

## 27. 工程2a（dry run）完了報告と、工程2b実施の確認依頼（2026-09-27）

担当: Step 0実装チャット。状態: **工程2a完了。工程2bは管理確認待ちで未実施。**

### 27.1 実行結果（共有可能な範囲）

回帰確認（`check_dummy_bprime_split_builder` 96項目、`check_dummy_step0_input_audit` 68項目）
に成功した後、dry runを実行した。

| 項目 | 値 |
| --- | --- |
| 候補 | **159件** |
| 層構成可能性 | **True**（3群の中央値が狭義単調増加） |
| 中間結果 fingerprint | `f25ee988b48b5c86145c2652aa37613e2fa36498fb7f918b91858782bc25349e` |
| 抽選 | **実施していない。splitは確定していない** |

4.4の出力境界どおり、共有されたGT集計は**候補件数と構成可能性の真偽のみ**である。
群サイズ・中央値・multi-region周辺度数・動画別の層化量・セル別配分は封印側にある。
fingerprintと各仕様版は照合・再現情報であり、GT値を含まない。

これが候補159件のH5を点データまで読んだ最初の工程である（1 pass）。
共有用出力は`stratification_feasibility_SHARE.json`にある。

### 27.2 工程2bが行うこと（確認のため明示する）

管理確認が得られた場合、`--confirm`は次を実行する。

1. 中間結果をfingerprint（27.1の値）とschema・設定で照合し、一致しなければ停止する。
2. 確定した仕様で18件を抽選する（seed 42、SHA-256鍵付きソート、層別最大剰余配分）。
   **dry runで固定した中間結果を使うため、H5は再読込しない。**
3. 契約検査6項目（相互排他、144／18／18、旧train162の再構成、旧180の網羅、
   sanity保持、validationの内容・順序不変）を実行し、1つでも不合格なら確定しない。
4. 非封印側へ`train_core_144.txt`・`validation_18.txt`・`s5_16_bprime.json`・`sealed_registry.json`、
   封印側へ`internal_test_18.txt`・セル別配分を書き出す。
5. pinへ追加すべきpathとhashを表示する。

### 27.3 不可逆性について

**工程2bの後、初回構築経路は恒久的に閉じる。**
`assert_bootstrap_allowed`はregistryが存在するか承認済みであれば停止するため、
dry runも確定もやり直せない。これは「正常なsplit確定は1回」という要件（依頼書3章）の実装である。

またpinを追加してコミットするまで、通常の学習・評価・推論・checkerは起動できない（10.10.1）。

### 27.4 確認を依頼する事項

1. **27.1の結果をもって工程2bへ進んでよいか。**
2. 27.1で共有した範囲（候補件数・真偽・fingerprint・仕様版）が4.4の境界に適合しているか。

### 27.5 工程2b以降の予定

2b（確定・封印）→ ユーザーによるpin追加とコミット → 工程3（train_core専用class weight）
→ 工程4（mmメタ監査）→ 工程5（teacher期待値の算出）。
その後S0-7として総括管理の受入を受け、Step 0を閉じる。

## 28. 工程2bの実施承認（総括管理チャット、2026-09-27）

担当: Step 0実装チャット（管理回答の同期）。
状態: **承認の記録である。工程2bは未実施であり、完了・pin登録・Step 0受入の記録ではない。**
原文: `docs/stage5/s5-16/stage5_s5_16_step0_2b_confirmation_request.md`8章。

### 28.1 承認内容

**27.1で報告した中間結果を用いた工程2b（分割確定・封印）への移行が承認された。**
同じ範囲について再度の着手確認は不要とされた。

承認対象の中間結果fingerprint:

```text
f25ee988b48b5c86145c2652aa37613e2fa36498fb7f918b91858782bc25349e
```

共有範囲（27.1および依頼書3章）は承認済みの出力境界に適合すると判断された。
候補件数と層構成可能性の真偽は共有可能な集計であり、fingerprint・仕様版は照合・再現情報として共有できる。
群サイズ・中央値・multi-region周辺度数・動画別値・セル別配分は引き続き封印側に保持する。

ただしこの判断は**依頼書・報告書に掲載された内容に対する確認**であり、
管理チャットが実機の共有JSON・stdout・例外ログ・封印物を直接検査したという意味ではない。

### 28.2 承認に付された条件

| # | 条件 |
| --- | --- |
| 1 | fingerprintを`--expected_intermediate_sha256`へ指定し、固定済みの中間結果・仕様・設定を用いる |
| 2 | 候補159件、層構成可能性True、**ファイル単位の抽選（D-041）**、seed 42、SHA-256鍵付きソート、層別最大剰余配分を維持する |
| 3 | **H5の再読込み、設定変更、再抽選は承認されていない** |
| 4 | fingerprint・schema・設定・入力の照合と契約検査6項目の停止条件を維持する |
| 5 | 不一致や異常があれば**停止して報告**する。**registryの削除やguardの迂回により再実行しない** |
| 6 | 正常な分割確定後は初回構築経路を閉じ、封印対象とそのGT統計を開発へ持ち込まない |

### 28.3 明確化された留保

- **D-041のファイル単位維持は変更しない。**
- ただし26.2-2の「segmentation指標への影響は小さい」は**撮影条件の説明に基づく推論**であり、
  **影響の小ささも検査単位の独立性も実証されたという意味ではない。**
  本報告書26.2の記述を、この趣旨が明確になるよう改めた。
- **封印は、過去の学習利用歴や集合間の同一検査重複を解消するものではない。**
  封印が行うのはinternal_testを今後の開発から隔離することだけである。

### 28.4 実施後に報告する内容

工程2b実施後、**共有可能な範囲で**次を報告する。

1. 契約検査6項目の成否
2. 分割件数 144／18／18
3. 封印処理の成否
4. pin登録に必要なpathとhash
5. 異常の有無

封印内容は開示せず、既存の出力境界（4.4）を守る。

### 28.5 現在の状態

- 本節は**承認の記録**である。工程2bは未実施。
- 実施後は6章の順序に従い、ユーザーによるpin追加とコミット、工程3〜5、S0-7受入へ進む。
- 承認と実施結果は別の節として区別して記録する。

## 29. 工程2b（分割確定・封印）実施結果（2026-09-27）

担当: Step 0実装チャット。状態: **工程2b完了。pin登録とコミットは未実施（ユーザー実行）。**
承認: 28章。28.4が求めた5項目を共有可能な範囲で報告する。

### 29.1 実施結果

| # | 項目 | 結果 |
| --- | --- | --- |
| 1 | **契約検査6項目** | **全項目合格** |
| 2 | **分割件数** | **train_core 144 ／ validation 18 ／ internal_test 18** |
| 3 | **封印処理** | **成功。** internal_testリストとセル別配分・GT統計は封印側に格納 |
| 4 | pin登録に必要な情報 | 29.3 |
| 5 | **異常** | **なし。** 停止条件に触れた項目なし |

identity fingerprint（照合・再現情報。GT値を含まない）:

| split | 件数 | identity |
| --- | --- | --- |
| train_core | 144 | `fa8429e30b7c8724` |
| validation | 18 | `de3f3d513fc0bf41` |
| internal_test | 18 | `1247e6ae901f2012` |

### 29.2 validationの不変性が独立に裏付けられた

**validationのidentityはS0-1（23.1）が報告した`de3f3d513fc0bf41`と完全に一致する。**
`validation path spelling unchanged: True`もあわせて出力された。

S0-1は旧validationリストを、工程2bは確定後のvalidationリストを、
別々に読んで別々にfingerprintを計算している。両者の一致は、
契約検査6項目のうち「validationの内容・順序不変」を独立に裏付ける。
D-037の「validation18の内容・順序は維持する」は満たされた。

### 29.3 pin登録に必要な情報

| pin名 | path | sha256 |
| --- | --- | --- |
| `seal_registry` | `<OUT>/sealed_registry.json` | `882305d566ac0bf0db33923776821c70f50bdca13853999c30bcc4258e8b13e7` |
| `s5_16_bprime` | `<OUT>/s5_16_bprime.json` | `2a0c07433c8a6526b66205a0cf3d5f428f3b1299df2699cb45a070193623744b` |

`<OUT>` = `/mnt/data/3d_projects/stage5_splits/s5_16_step0_bprime`

実装チャットは`Stage5/stage5/config/split_contract_pins.json`へ上記2件を追記した。
**承認はユーザーによる差分確認とコミットである**（10.3）。コミットは未実施。

### 29.4 現在の状態と、次に起きること

- **初回構築経路は閉じた。** registryが存在するため、dry runも確定も再実行できない。
- **pinをコミットするまで、通常の学習・評価・推論・checkerは起動できない**（10.10.1の意図した動作）。
  現在は「splitは確定したが、通常運用はまだ有効化されていない」状態である。
- pin有効化後、通常経路のguardがinternal_testの18動画を読込前に拒否するようになる。

### 29.5 補足: seal registryの所在について

`sealed_registry.json`は封印ディレクトリの外（非封印側）にある。
これは**guardが毎回読む必要があるため設計上そうしている。**
同ファイルは封印対象18動画のidentityを保持するが、
封印の対象は依頼書3章のとおり**internal_testのGT統計・metrics・臨床FL**であって、
どの動画が封印対象かという事実そのものではない
（「通常の記録にはfile listとhashのみを残す」）。
GT統計・層化量・セル別配分は`sealed/`配下に0600で格納されている。

### 29.6 Step 0の残り

S0-3（分割生成）とS0-4（隔離・封印）の実機実施は完了した。残るのは次のとおり。

| ID | 内容 | 状態 |
| --- | --- | --- |
| S0-5 | train_core専用class weightの算出（工程3） | 未実施。pin有効化後 |
| S0-6 | mmメタ情報の確認（工程4） | 未実施 |
| — | teacher期待値の算出（工程5） | 未実施 |
| S0-7 | 総括管理による受入と記録同期 | 未実施 |

工程5（teacher期待値）は、確定したtrain_core 144＋validation 18＝162件に対する
集合依存の期待値を1回算出するものである（10.6）。

## 30. 工程3（train_core専用class weight）実施結果 — S0-5（2026-09-27）

担当: Step 0実装チャット。状態: **S0-5完了。値を固定する。**

### 30.1 算出結果

| 項目 | 値 |
| --- | --- |
| 対象 | train_core **144件**（`train_core_144.txt`を明示入力） |
| 対象 identity | `fa8429e30b7c8724`（29.1のtrain_coreと一致） |
| 設定 | `epsilon=0.02`、正規化有効、`num_classes=2`、`ignore_index=-1` |
| **class weight** | **`[0.06023312732577324, 1.9397668838500977]`**（background, positive） |

**この値をS5-18〜S5-20のsegmentation lossで固定して使う。stageごとに再算出しない**（S5-16 10.0）。
クラス別点数はprivate JSONにのみ記録した。

### 30.2 値の健全性確認

共有された値だけから次を確認した。**GT統計を参照していない。**

| 検査 | 結果 |
| --- | --- |
| 平均1への正規化 | 2値の平均 = 1.0000000056。**成立** |
| float32表現 | 両値とも float32 で厳密表現可能。**成立**（実装は float32 へキャストする） |
| 対象集合の一致 | identity `fa8429e30b7c8724` が29.1のtrain_coreと一致。**成立** |

対象リストのidentityがmanifestの`train_core`と一致しなければ計算せず停止する設計であり、
これが**pin有効化後の最初の実運用であった。** guardは正常に機能した。

### 30.3 旧W-Aとの差（履歴。調整はしない）

| | background | positive |
| --- | --- | --- |
| 新（train_core 144） | 0.06023313 | 1.93976688 |
| 旧W-A（履歴値） | 0.05963856 | 1.94036150 |
| 相対差 | **+0.997 %** | −0.031 % |

比から逆算した陽性率（［推論］、共有値のみから導出）:

| 集合 | 陽性率 |
| --- | --- |
| train_core 144 | 1.1321 % |
| S5-15 train162 | 1.1123 % |
| W-A算出元（履歴） | 1.1012 % |

train_core 144はtrain162より陽性率がやや高い。internal_test 18件を除いた結果として整合する。
**依頼書5章のとおり、この差を見てweightを調整・再探索しない。**
8.1-2で報告したとおり旧W-Aは共有された162動画統計から厳密に再現できないため、
**旧値との一致を新値の検証条件にしていない。** 30.2の3検査が検証である。

### 30.4 Step 0の残り

| ID | 内容 | 状態 |
| --- | --- | --- |
| S0-1〜S0-5 | 入力監査／分割／封印／class weight | **完了** |
| S0-6 | mmメタ情報の確認（工程4） | 未実施 |
| — | teacher期待値の算出（工程5） | 未実施 |
| S0-7 | 総括管理による受入と記録同期 | 未実施 |

## 31. 工程4の初回実行失敗と、2件の欠陥修正（2026-09-27）

担当: Step 0実装チャット。状態: **修正完了。工程4は未実施（再実行が必要）。**

### 31.1 症状

工程4（mmメタ監査）の初回実行が、**監査を最後まで完了したうえでJSON書き出しで失敗**した。

```
TypeError: Object of type int64 is not JSON serializable
```

### 31.2 欠陥1: numpy型の書き出し（実装の欠陥）

h5pyは属性をnumpyスカラーで返す。`raw_width`は`int`ではなく`int64`であり、
`json.dumps`がこれを拒否する。`read_spacing`は明示的に`float()`していたが、
S5-14のcheckerから再利用している`read_intermediate_local_dimensions`は
生の属性値を返すため、そこを通る値が未変換だった。

修正: `_json_default`を追加し、numpyスカラー・配列・bytesをJSON可能な型へ変換する。
private・shareableの両方の書き出しへ適用した。

### 31.3 これが通った理由 — テストが書き出し経路を一度も通っていなかった

**本質的な問題はこちらである。** `check_dummy_train_core_mm_metadata`の
end-to-endケースは`test_requires_a_contract`だけで、
これは「契約が無いので**拒否される**」ことを確認するものだった。
すなわち**成功経路でJSONを書く箇所に一度も到達していなかった。**
残りの検査は`read_spacing`の直接呼び出しか、ソース文字列の検査である。

修正: **完走してJSONを書くend-to-endテストを追加した**（テスト[5]）。
合成の契約（pins・registry・manifest）と合成H5を用意してツールを通し、
戻り値0、両ファイルの生成、**`raw_width`が数値として書かれていること**（31.2の回帰）、
placeholder spacingの認識、共有側が`UNCONFIRMED`かつ動画別詳細と動画IDを含まないことを確認する。

### 31.4 欠陥2: 合成テストが本番pinsファイルを読んでいた

**29.3でpinを登録した結果、無関係の合成テスト`check_dummy_step0_input_audit`が失敗した。**

原因は、合成テストが`--split_contract_pins`を指定せず、
リポジトリの実pinsファイルへfallbackしていたことである。
pin登録前は`"pins": {}`だったため偶然動いていたが、
登録後は`assert_pre_seal`が実契約を解決しようとして停止する。

**合成テストが本番の設定状態に依存していた。** これは
「合成テストの成功と実機確認を混同しない」（9.2.6）以前の問題で、
テストの結果が実機の運用状態で変わってしまう。

修正:

- `build_stage5_bprime_split.py`へ`--split_contract_pins`を追加し、
  初回構築判定が既定のpinsファイルに固定されないようにした。
- `check_dummy_step0_input_audit`と`check_dummy_bprime_split_builder`が、
  **自前の空pinsファイル**を使うようにした（`fx.run()`経由と手組みの呼び出しの両方）。

なお`assert_pre_seal`が実pinsで停止した動作自体は**設計どおりである。**
S0-1は封印前の1回限りの工程であり（4.2）、registry承認後は拒否される。
実機でS0-1を再実行できないことは意図した仕様である。

### 31.5 検証

本実装環境で実行できる5本は全合格。**本番pinsは変更していない。**

| テスト | 項目数 |
| --- | --- |
| `check_dummy_split_contract` | 84 |
| `check_dummy_step0_input_audit` | **68**（pins依存を解消） |
| `check_dummy_bprime_split_allocation` | 67 |
| `check_dummy_preflight_guard_and_output_boundary` | 43 |
| `check_dummy_fixed_list_mode`（既存回帰） | 11 test |

`check_dummy_train_core_mm_metadata`（テスト[5]追加）と
`check_dummy_bprime_split_builder`（空pins化）は**実機での再実行が必要。**

## 32. 工程4（mmメタ情報監査）実施結果 — S0-6（2026-09-27）

担当: Step 0実装チャット。状態: **S0-6完了。ただし「mmスケールを取得した」ではない。**

### 32.1 結果

| 項目 | 144件中 |
| --- | --- |
| 中間pseudo-3D H5の解決 | **144**（未解決0、最終H5欠落0） |
| crop逆変換メタの読取り | **144** |
| **crop逆変換メタの全field完備** | **144** |
| `spacing`データセットの読取り | 144 |
| **`spacing`が既定placeholder（1.0/1.0）** | **144** |
| `spacing`のx≠y | 0 |
| mmスケールの結論 | **UNCONFIRMED** |

代理FLも`T_FL`も算出していない。

### 32.2 判明したこと

**(1) crop逆変換の経路は完全に揃っている。**
`raw_width`・`raw_height`・`local_crop_top`・`local_crop_left`・`local_resize_scale`が
**144件すべてで欠落なく読める。** すなわち
`pixel_xy`（local crop座標）→ 元frame座標 の変換は、全train_coreについて実行可能である。
S5-16 10.0の換算経路のうち、**mmの手前までは成立している。**

**(2) 監査した保存項目から実mm/pixelを確証できない。**
`spacing`は**144件すべてが既定値1.0/1.0**である。9.7-2で指摘したとおり
`spacing`は`--spacing_x`/`--spacing_y`（既定1.0、helpは可視化用の暫定値）が
そのまま保存されたものである。

**［35.3-1による補正］** これは「実mm値がどのH5のどの項目にも存在しない」ことや
「生成時に実値を指定しなかった」ことの証明ではない。**確認したのは監査対象の保存項目に限られる。**
結論は「**確認した保存項目から実mm/pixelを確証できない。外部供給元と対応の確認が必要**」である。
15.1のユーザー回答（供給元はmm/pixel、元frame基準、動画ごとに異なる可能性）は維持される。

### 32.3 S0-6の位置づけ

依頼書6章が求めたのは**取得可否の確認**であり、mmスケールの取得ではない。
その意味でS0-6は完了している。結論は次のとおり。

| 対象 | 状態 |
| --- | --- |
| crop/resize逆変換情報 | **fieldが144/144で読める。** ただし逆変換の数値的な正しさやmm換算の実証とは別である（35.3-1） |
| 元frameの寸法 | **取得可能**（`raw_width`/`raw_height`） |
| x/y別のmm/pixel | **監査した保存項目からは確証できない。外部供給元の確認が必要** |
| 動画内で一定かframe別か | **未確認。** `spacing`は動画単位で1組しか持たないため、frame別の値が要るなら外部側で解決する |

32.2-(2)は「取得不能」ではない。9.1のユーザー回答とmm/pixelという形式は維持されており、
**供給元・形式・動画対応の確認が未了**という状態である（管理判断4、9.1の整理どおり）。

### 32.4 S5-17の開始条件として引き継ぐ

S5-17でmm評価・臨床FL照合を行う前に、次を確定する必要がある。

1. mm/pixelの**供給元と保存形式**（ファイル、列、単位）。
2. **動画への突合キー**。180動画には2系統の命名規則があり（18章）、
   timestamp形式には第4セグメントがある（21章）。突合は完全一致で行う。
3. **動画単位かframe単位か**。frame単位なら、frame識別子との対応方法。
4. 元frame基準であること（15.1で回答済み）を前提に、
   `pixel_xy`→crop逆変換→元frame→mmの往復誤差をS5-17のfail-fastで検査する。

`T_FL`はS5-17実行前のユーザー設定事項のまま未決である。Step 0では推定しない。

### 32.5 Step 0の残り

| ID | 内容 | 状態 |
| --- | --- | --- |
| S0-1〜S0-6 | 入力監査／分割／封印／class weight／mmメタ | **完了** |
| — | teacher期待値の算出（工程5） | 未実施 |
| S0-7 | 総括管理による受入と記録同期 | 未実施 |

## 33. 工程5（teacher期待値の算出）実施結果（2026-09-27）

担当: Step 0実装チャット。状態: **完了。Step 0の実機工程はこれで全て終了。**

### 33.1 結果

| 項目 | 結果 |
| --- | --- |
| 固定リスト検証 | train 144 ／ val 18 ／ 計 162 |
| **seal guard** | **通過。checked_inputs=162、sealed_videos=18** |
| manifest / registry | `2a0c07433c8a6526` ／ `882305d566ac0bf0`（29.3のpinと一致） |
| teacher期待値 | **162件分の観測値を書き出し。値は共有出力から伏せられた** |
| 学習 | **開始していない**（`PREFLIGHT_ONLY=1`） |

出力は`teacher_preflight_expected_162.json`（領域B、DO_NOT_SHARE）。
詳細ログは封印側の`teacher_preflight_log_DO_NOT_SHARE.json`。

### 33.2 guardがpreflightより前に走ることが実データで確認された

出力の順序が設計どおりであった。

```
1. Fixed-list inputs validated: train=144 val=18 total=162
2. Split contract satisfied: ... checked_inputs=162 sealed_videos=18
3. Stage 5 teacher v7 preflight: observed totals written ...
```

seal guardが**teacher preflightより前**に162件すべてを検査している。
preflightは全H5を開くため、guardがその後にあれば
「封印動画を読んでよいか判断するために封印動画を読む」ことになる（10.4）。
合成テストで確認していたこの順序が、**実データで初めて成立を確認できた。**

**［35.3-3による補正］** これは**許可された162件についてguardが先に実行された記録**であり、
**封印18件を実際に読み込ませて拒否を確認した試験ではない。**
拒否動作の証拠は合成テスト（存在しないパスを封印集合として渡し、
`FileNotFoundError`ではなく封印例外になることを確認）とコード確認と組み合わせて扱う。

### 33.3 validationの不変性が3通り目の方法で確認された

工程5が報告した validation の**content sha256 `0c251380e40f0def`** は、
`checks/real_h5/run_stage5_s5_15_arm.sh`に固定されている
`EXPECTED_VAL_LIST_SHA256="0c251380e40f0def3a76..."`と**先頭16桁が一致する。**

これは`list_content_sha256`であり、**動画同一性だけでなくパス表記と順序を含む。**
すなわち新しい`validation_18.txt`は、S5-15が使ったvalidationリストと
**内容・順序・パス表記まで同一**である。

確認は3通り、うち3つ目は**Step 0開始以前からリポジトリに固定されていた外部の基準値**との照合である。

| # | 方法 | 値 |
| --- | --- | --- |
| 1 | S0-1が旧リストから算出したidentity | `de3f3d513fc0bf41` |
| 2 | 工程2bが確定後リストから算出したidentity＋`path spelling unchanged: True` | `de3f3d513fc0bf41` |
| 3 | **工程5のcontent sha256 対 S5-15 launcherの固定値** | **`0c251380e40f0def`** |

D-037の「validation18の内容・順序は維持する」は、独立した3通りの方法で満たされた。

**［35.3-2による補正］** 上表の値はいずれも**先頭16桁の抜粋**であり、
完全なSHA-256照合を総括管理が行ったという記録ではない。
内容・順序不変の**主たる受入根拠は、工程2bの契約検査結果と
`validation path spelling unchanged: True`の報告**である。
3の照合はそれを補強する補助証拠として扱う。

train側の`8393a22a33df5872`はtrain_core 144の新しい値であり、
旧train162の`582579833f345b77`と一致しないのが正しい。

### 33.4 期待値の位置づけ（10.6の再掲）

この期待値の固定は**以後の変化検知の基準作成**であり、
**teacher品質の検証ではない。** 検査の合格と期待値の固定は別の事項として記録する。
集合非依存の検査（teacher版、`valid_mask`と`point_label != -1`の整合、由来情報、
必須dataset存在）は変更していない。

新定数は共有しない。既にgitにある180件の定数との差から、
封印18動画のQC統計が得られるためである。stdoutも数値を伏せて出力された。

## 34. Step 0 実機工程の完了サマリ（2026-09-27）

### 34.1 S0-1〜S0-7の状態

| ID | 内容 | 状態 |
| --- | --- | --- |
| S0-1 | 入力監査（リスト・sanity照合・候補導出・hash固定） | **完了**（23.1） |
| S0-2 | 層化・seed・封印方式の事前固定 | **完了**（D-039、仕様版3） |
| S0-3 | train_core144／internal_test18／validation18の生成 | **完了**（29.1、契約検査6項目合格） |
| S0-4 | internal_testの隔離・封印 | **完了**（29.1、pin登録済み） |
| S0-5 | train_core専用class weightの算出 | **完了**（30.1） |
| S0-6 | mmメタ情報の確認 | **完了**（32.1。mmスケール自体はUNCONFIRMED） |
| — | teacher期待値の算出 | **完了**（33.1） |
| **S0-7** | **総括管理による受入と記録同期** | **未実施** |

### 34.2 確定した成果物

| 成果物 | 値 |
| --- | --- |
| train_core | 144件 identity `fa8429e30b7c8724` |
| validation | 18件 identity `de3f3d513fc0bf41`（旧と同一） |
| internal_test | 18件 identity `1247e6ae901f2012`（封印、S5-20cまで） |
| **class weight** | **`[0.06023312732577324, 1.9397668838500977]`** |
| 層化量 | `gt_positive_frame_count`（D-039、仕様版`gt_positive_frame_count_v1`） |
| 抽選 | seed 42、`sha256_keyed_sort_v1`、層別最大剰余配分 |

### 34.3 S5-17へ引き継ぐ未決事項

1. **mm/pixelの供給元・形式・動画突合キー・動画単位かframe単位か**（32.4）。
   crop逆変換は144/144完備で、不足しているのはmm値そのものである。
2. **`T_FL`**（臨床上の許容誤差）はS5-17実行前のユーザー設定事項。
3. **H17-1の循環性**（15.3）。180動画に臨床実測FLが無いため、
   「臨床FLとの一致で代理FL定義を選ぶ」という設計はそのままでは成立しない。
4. **5例のアノテーション**（D-040）の比較単位（26.3）。
   **検査内一貫性を含め、S5-17の方式選択・較正へ流用しない**（35.3-5）。

### 34.4 残る作業

- 31章の修正のコミット（ユーザー実行）。
- **S0-7: 総括管理への受入依頼と、受入後の管理記録・評価レポートの同期。**

## 35. 総括管理による完了受入判断（2026-09-27）

### 35.1 判断と確認範囲

**S0-1〜S0-6およびteacher期待値算出の完了を受け入れる。**
Step 0依頼書・本報告の完了条件、D-039〜D-041、仕様書、受入依頼書を照合し、
split契約・封印guard、class weight算出、mm監査と修正テスト、FILES.mdの使用禁止経路を確認した。
コミット`392b0ab`（実装・pin）と`b10dd87`（監査修正・実施報告）を確認し、レビュー開始時の作業ツリーはcleanであった。

実機での合成テスト・CPU工程の結果は23〜34章の実施記録に基づく。
管理チャットは実機H5・封印物・private JSONへアクセスしておらず、実機結果の独立再検査とは位置付けない。
合格済みテストや実機工程の再実行、再抽選、weight再算出を受入のために要求しない。

### 35.2 受入内容

| 対象 | 判断 |
| --- | --- |
| S0-1・S0-2 | 入力監査と層化仕様の事前固定を受入。臨床FL所在監査はD-039で対象外 |
| S0-3・S0-4 | 144／18／18、契約検査6項目合格、封印・pin登録を受入。過去利用歴・検査重複の限界は維持 |
| S0-5 | train_core144から1回算出した`[0.06023312732577324, 1.9397668838500977]`を固定。旧W-Aとの一致や差による調整を要求しない |
| S0-6 | 取得可否監査の完了を受入。crop逆変換メタ144/144の取得とmmスケールUNCONFIRMEDを区別する |
| teacher期待値 | 162件の変化検知基準の作成を受入。teacher品質の新たな証明とはしない。数値は共有しない |
| S0-7 | 本章で受入判断は完了。関連文書の最終同期・同期結果の記録は残る |

### 35.3 結果の解釈に関する補正と持越し

1. **mmの不存在は監査範囲に限定する。** 既定値と一致するspacingだけから、実mm値が
   全H5の全項目に存在しないことや、生成時に実値を指定しなかったことまで証明したとは扱わない。
   32章の結論は「確認した保存項目から実mm/pixelを確証できない。外部供給元・対応の確認が必要」として引き継ぐ。
   cropメタの存在も、逆変換の数値的正しさやmm換算の実証とは別である。
2. **validationの短縮hashは補助証拠。** 29.2・33.3の掲載hashは先頭16桁であり、
   完全なSHA-256照合を管理側が行ったとは記録しない。内容・順序不変の主たる受入根拠は
   工程2bの契約検査結果と`path spelling unchanged: True`の報告である。
3. **guardの実機証拠の範囲。** 33.2は許可された162件についてguardが先に実行された記録である。
   封印18件の実読込みによる拒否試験ではない。拒否動作の証拠は合成テスト・コード確認と組み合わせて扱う。
4. **S5-17の実行条件は別途確定する。** mm/pixelの供給元・形式・完全一致キー・粒度、
   換算検証、`T_FL`、H17-1の目的と採否規則の再設計を引き継ぐ。
   Step 0受入は、未確定のmm評価や臨床FLとの一致による選定を開始する承認ではない。
5. **5例の用途制限は維持。** S5-18学習中にアノテーションし、固定した方法による探索的比較に使う。
   検査内一貫性も、5例を使ったS5-17の方式選択・較正へ流用しない。比較単位は例単位を踏まえ別途固定する。
6. **使用禁止経路とprivacy self-checkの不足は残る。** FILES.mdにある未guard経路の使用禁止を維持し、
   命名2系統・第4セグメントに対応していない既存privacy self-checkは、当該経路を使う前に修正・検証する。
   本受入を全既存ツールの封印対応完了とは読み替えない。

これらは既知の制約と報告表現の補正であり、S0-1〜S0-6をやり直す条件ではない。

### 35.4 受入後の実装チャットの作業

受入依頼書8章に従い、管理記録の現在状態・現行フロー、評価レポート、S5-15報告、
FILES.md・data_construct.mdを同期し、35.3の補正とS5-17への持越しを反映する。
過去のログ・凍結文書は上書きせず、現在状態と過去時点の状態を区別する。
合成テストの「7本」はStep 0新規分で、既存回帰fixed_list_modeは別枠であることも集計に明記する。
同期結果を追記した時点でS0-7の記録同期を完了とし、Step 0全体を閉じる。
この通常の文書同期に再承認は不要。新しい仕様変更や不整合を発見した場合は報告する。
その後、本総括管理でS5-17実装依頼書を作成する。専用管理チャットは設けない。

本判断の追記は未コミットであり、`b10dd87`に含まれるとは扱わない。

## 36. S0-7 記録同期の実施結果（2026-09-27）

担当: Step 0実装チャット。状態: **同期完了。これをもってS0-7を完了とし、Step 0を閉じる。**
根拠: 35.4の指示。通常の文書同期であり再承認は要さない。

### 36.1 35.3の補正を本報告へ反映した

| # | 補正 | 反映先 |
| --- | --- | --- |
| 1 | mmの不存在は監査範囲に限定。cropメタの存在は逆変換の数値的正しさやmm換算の実証とは別 | 32.2-(2)、32.3 |
| 2 | validationの掲載hashは**先頭16桁の抜粋**。主たる受入根拠は工程2bの契約検査と`path spelling unchanged: True` | 33.3 |
| 3 | guardの実機証拠は**許可された162件**についての順序記録。封印18件の拒否試験ではない | 33.2 |
| 5 | 5例の**検査内一貫性もS5-17の方式選択・較正へ流用しない。** 26.3の「S5-17の判定材料になり得る」を撤回 | 26.3、34.3-4 |
| — | 合成テスト「7本」はStep 0新規分。既存回帰`fixed_list_mode`は別枠 | 19章 |

35.3-4（S5-17の実行条件）と35.3-6（使用禁止経路とprivacy self-check）は
持越し事項であり、36.2の各文書へ反映した。

### 36.2 同期した文書

| 文書 | 内容 |
| --- | --- |
| 管理記録 0章・3.1節 | 現在の段階をStep 0完了・受入済みへ更新。split・class weight・S5-16の各行を確定値へ |
| 管理記録 6章 現行フロー | 項目1をStep 0完了へ。S5-17へ持ち越す未決事項4件を明記 |
| 評価レポート 9.12（新設） | 確定した評価境界、既存validationの同一検査重複と解釈（結論の方向は不変）、動画別統計の限界 |
| S5-15報告 10章（新設） | 8章までの記録は変更せず、validationの重複、train集合の変化、旧class weightの再現不能、命名規則とprivacy self-checkの盲点を追記 |
| `FILES.md` | 封印が発効中であることと、使用禁止経路のprivacy self-checkの盲点を明記 |
| `data_construct.md` | 3領域が実体化したことと、各領域の実際の内容を明記 |

**過去のログ・凍結文書は上書きしていない。** S5-15報告8章までの記録・数値・判断、
評価レポートの既存章、固定引き継ぎ文書、S5-16 v3は変更していない。
現在状態と過去時点の状態は各文書内で区別した。

### 36.3 Step 0のクローズ

| ID | 状態 |
| --- | --- |
| S0-1〜S0-6、teacher期待値算出 | **完了**（35.2で受入） |
| S0-7 | **完了**（本節） |

**Step 0を閉じる。**

### 36.4 残る作業と次段階

- **未コミット差分の提示。** 本節までの文書同期（管理記録、評価レポート、S5-15報告、
  `FILES.md`、`data_construct.md`、本報告23〜36章）。**コミットはユーザーが実行する。**
- 総括管理でS5-17実装チャットへの依頼書を作成する（D-038。専用管理チャットは設けない）。
  依頼書には34.3／35.3-4の持越し事項、特に**H17-1の目的と採否規則の再設計**を含める必要がある。
  180動画に臨床実測FLが無いため、「臨床FLとの一致で代理FL定義を選ぶ」という現行設計は
  そのままでは成立しない（15.3）。

## 37. クローズ後・S5-17移行前の最終確認（2026-09-27）

総括管理でコミット`f31cdd8`と36章の同期記録、管理記録の現行フロー、評価レポート9.12、
S5-15報告10章、FILES.md、data_construct.md、承認pinを確認した。
レビュー開始時の作業ツリーはclean。直近の同期コミットは文書のみで、実装・pinは変更されていない。
**S0-7の完了とStep 0のクローズを確認し、S5-17実装依頼書の作成へ進んでよいと判断する。**

固定分割144／18／18、class weight `[0.06023312732577324, 1.9397668838500977]`、
S5-20cまでの封印、未guard経路の使用禁止、5例の用途制限は維持されている。
35.3の持越し、特にmm/pixel・換算検証・`T_FL`・H17-1の再設計は、S5-17依頼書で明示する。
今回の確認はS5-17の未確定な評価・実験を開始する承認ではない。

最終確認で、管理記録0章・評価レポート冒頭・本報告冒頭に古い未完了／同期待ちの記述が残っていたため、
現在状態と更新情報を訂正した。過去日付の実施記録、凍結文書、35章・36章当時のコミット待ち記録は保持する。
本節と冒頭表示の訂正は`f31cdd8`後の未コミット文書差分であり、Step 0再開を意味しない。
実データ・封印物の再読込み、テスト再実行、再抽選、weight再算出、学習は行っていない。

## 38. 同一検査重複件数の誤転記の訂正（2026-09-27）

S5-17初回報告F10を受け、24.2〜24.3の元記録と照合した。
正しい記録は「旧train162／validation18にまたがる4グループの所属ファイルが両集合合計17件」であり、
「validation18のうち17件」ではない。validation側の正確な件数は未確認で、記録上の範囲は4〜13件である。
これは旧分割についての記録であり、新train_coreとの重複件数を確定したものでもない。

26.2および後続のS5-15報告10.1・評価レポート9.12・S5-17依頼書4.3に生じた誤転記を訂正した。
D-041の合計の意味も明確化し、過去の工程2b確認依頼・受入依頼にも訂正を明記した。
変更理由を本節に残し、正確な件数を推測で補完しない。

訂正は文書上の分母の取り違えを直すものであり、D-041のファイル単位維持、封印、Step 0受入・クローズ、
既存のS5-15終了判断は変更しない。分割再構築・実データの再読込みは行っていない。
validation側の正確な件数を取得する場合は、既存記録や許可されたリストの確認で実施範囲を定め、封印物を読まない。
S5-17初回報告本体の改訂は実装チャットが担当し、管理側では同報告書を変更していない。
