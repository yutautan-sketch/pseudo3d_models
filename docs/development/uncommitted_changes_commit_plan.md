# 未コミット変更の確認・コミット分割管理記録

> 配置注記（2026-09-22）：具体的な成果物参照は現在の保存場所へ更新済みです。本文のGit状態・実行結果は当時の記録です。保存バンドルは履歴資料であり、移動先での再実行を保証しません。

初版: 2026-09-12
現在の段階: ファイル名・Git状態・docs文書による概略分類まで。ソース差分の詳細確認前。

## 1. 目的と運用

- 蓄積した未コミット変更を把握し、保守可能な単位に分割して各段階を再現できる履歴にする。
- 最終的にGitの管理状態をクリーンにする。仮想環境などの生成物は追跡対象と区別する。
- 今後の確認結果、判断根拠、方針変更、未解決事項、ユーザー実行結果はこの文書に追記する。
- 過去の判断を黙って上書きしない。変更時は日付、旧判断、新判断、理由を記録する。
- 詳細調査で依存関係が判明したら、概略区分とコミット数を調整する。

## 2. 実行環境とユーザーとの役割分担

2026-09-12 ユーザー指定:

- このワークスペースはDockerコンテナ上にあり、実際のGitHubリポジトリとは接続していない。
- ユーザーはこの環境でのGitコマンドが失敗すると想定している。
- 実際のコミットおよびプッシュは、アシスタントがコマンドを提示し、ユーザーが実環境で実行する。
- アシスタントはこのワークスペースでコミット・プッシュを実行しない。

観測事実との区別:

- 初回調査のローカル `git status` と `git ls-files` は成功した。
- ローカルの読み取り成功は、リモート接続、Gitへの書き込み、実環境との同期を保証しない。
- リモート接続・書き込み操作の可否は未確認。ユーザー指定の役割分担を優先する。
- 今後のローカルGit読み取りも成功を前提にしない。失敗時はファイル調査またはユーザーからの実行結果で補う。
- 実行用コマンドは実環境の状態に合わせて提示する。コンテナ上の状態だけから、ユーザー側のindex・branch・remoteが同一と断定しない。

## 3. ここまでの確認範囲

- 未コミットのファイル名と状態を列挙し、ディレクトリ別に集計した。
- リネームの移動先を基準に集計し、リネーム1件を1エントリとして数えた。
- `docs/` の24文書について、構成・見出し・主要な仕様・進捗記録を確認した。全行の精読やリンク先の検証までは行っていない。
- ソースコード・設定ファイルの内容や差分はまだ詳細確認していない。
- ステージ操作、コミット、プッシュ、仮想環境の除外設定変更は行っていない。
- 文書に記された過去の検査結果は文書上の報告として扱う。今回の調査で再実行した結果ではない。

## 4. 初回調査時の件数

この記録ファイル作成前のスナップショット。以後の作業で変化し得る。

| 移動先基準の対象 | エントリ数 |
| --- | ---: |
| `.venvs/cvat273/` | 1,680 |
| `Stage2to4/` | 64 |
| `Stage5/` | 28 |
| `docs/` | 24 |
| `.gitignore` | 1 |
| ルート `Dockerfile.codex` | 1 |
| 合計 | 1,798 |
| 仮想環境を除く合計 | 118 |

| Git状態 | 件数 |
| --- | ---: |
| 未ステージ変更 ` M` | 16 |
| 未ステージ削除 ` D` | 1 |
| ステージ済みリネーム＋未ステージ変更 `RM` | 5 |
| ステージ済みリネーム `R ` | 10 |
| ステージ済み追加＋未ステージ変更 `AM` | 1 |
| 未追跡 `??` | 1,765 |

文書の移動と本文編集がindex・作業ツリーに混在している。既存ステージ内容をそのまま1コミットにする前提は置かない。

## 5. 初期判断

1. 大分類は18区分。実際のコミット数は35〜55件程度を初期目安にする。件数は確定値ではない。
2. `.venvs/cvat273/` はファイル名からCVAT SDK、pip、setuptools等を含むローカルPython環境と判断した。依存関係・構築手順を管理し、環境本体をignoreする案を推奨する。採用・実施はまだしていない。
3. 1,700件超の大半が仮想環境なので、件数だけを根拠に百数十コミットへ分割する必要性は薄い。
4. 各段階に必要な実装・設定・検査・文書を揃える。ファイル種別ごとの機械的分割で中間状態を壊さない。
5. 同一ファイルに複数機能の変更がある場合は差分単位の分割を検討する。
6. 歴史上の開発順と今回作るコミット順を区別する。文書に過去段階が存在するだけで、当時のコードを復元できるとは限らない。
7. 既にHEADに含まれる機能を、新規実装として再度コミット区分へ割り当てない。詳細確認時に判別する。

## 6. 概略の分割区分と順序

各区分は複数コミットを含み得る。依存順はソース差分確認後に確定する。

| ID | 区分 | 主な対象・分割の軸 |
| --- | --- | --- |
| 01 | 開発環境・Git管理 | `.gitignore`、Dockerfileの移設候補、CVAT環境の再現方法 |
| 02 | 文書の配置整理 | `docs/`への移動、目次・参照先。本文の機能追加記録とは分離 |
| 03 | Stage 4既存契約の検査更新 | BBoxラベル方針、sampling sweep manifest、既存可視化pipeline |
| 04 | CVATマスク形式の変換 | `convert_masks_to_cvat_segmentation_mask_1_1.py`、形式検査 |
| 05 | 自動輪郭改善の試作 | `contour_teacher_refinement.py`、prototype、Phase 3設定・検査 |
| 06 | 人手レビューの基本往復 | review export、修正mask import、設定、round-trip検査 |
| 07 | teacher v3自動補正の反映 | batch適用、production設定、v3構築pipeline |
| 08 | CVATレビュー対象の展開 | Phase 3全ケースの出力、Task作成、関連pipeline |
| 09 | full-videoレビュー | 全フレームpackage、事前検査、Task作成、文字なし画像への対応 |
| 10 | snapshot・除外管理・teacher v4 | snapshot保存、動画除外manifest、最終import、v4構築 |
| 11 | 保存済みpoint labelの可視化 | 描画本体、batch出力、v4向けpipeline・検査 |
| 12 | CVATを正本とするteacher v5 | 全Task frameへの反映、空mask、provenance、preflight・受入検査 |
| 13 | XML削除を反映するteacher v6 | 欠落監査、無効化manifest、H5補正、batch・可視化・受入検査 |
| 14 | Stage 5データ経路の診断 | batch integrity、padding parityのchecker |
| 15 | paddingなし学習・評価経路 | loss、勾配蓄積、学習・推論・評価整合、dummy検査 |
| 16 | Stage 5のteacher v6移行 | 学習・推論・評価shell、入力検証、匿名化metrics関連変更 |
| 17 | overlap・BatchNorm診断 | overlap aggregation、mode parity、recalibrationを独立させる |
| 18 | GroupNorm比較 | 正規化層、decoder修正、CLI・checkpoint、構造・学習・転移検査 |

## 7. 文書から把握した仕様・進捗の境界

### Stage 4

- 自動輪郭改善のPhase 3試作と、teacher v3への実適用は別段階。
- v4はfull-videoレビュー結果の一部対象への適用という過去の段階。context frameの修正が反映されない不整合が記録されている。
- v5はsnapshot対象動画の全Task frameでCVAT maskをpositiveの唯一の根拠とする。target/context、空mask、BBox外maskの扱いを含む。
- snapshot対象59動画と非対象122動画を区別する。181動画すべてをCVAT由来と表現しない。
- v6は明示的XML無効化manifestをCVAT snapshotより優先する。単なるパス上のXML欠落を自動無効化条件にしない。
- BBox境界接触の見直しは文書上「未実装」。現時点では計画文書として扱い、実装済み機能と混ぜない。
- 古い文書のtarget-onlyなどの記述は履歴として扱い、最新のlabel authorityと混同しない。

主な根拠:

- `docs/stage2to4/stage4/stage4_contour_teacher_improvement_plan.md`
- `docs/stage2to4/stage4/stage4_phase5_fullvideo_cvat_review_implementation.md`
- `docs/stage2to4/stage4/stage4_cvat_snapshot_authoritative_label_revision_plan.md`
- `docs/stage2to4/stage4/stage4_deleted_xml_annotation_invalidation_plan.md`
- `docs/stage2to4/stage4/stage4_v4_point_label_visualization_plan.md`
- `docs/stage2to4/stage4/stage4_bbox_ranked_border_contact_revision_plan.md`

### Stage 5

- 現在状態・優先順の正本は `../stage5/stage5_revision_management_record.md`。初期改善計画より優先する。
- batch integrity、padding parity、paddingなし学習、teacher v6移行、overlap、BatchNorm mode parity、recalibration、GroupNorm比較を別の段階として扱う。
- paddingなし学習はphysical batch 1と8 windowの勾配蓄積、class-weighted有効分母による正規化を含む。
- overlapの代替集約、train-mode推論、recalibrated checkpointは診断とproduction採用を区別する。
- GroupNormは文書上、decoderへの正規化設定伝播修正を含み、1 epoch smokeまで確認済み。5 epoch pilotの評価は未実行と記録されている。
- GroupNorm比較が存在することをもって、既定normalizationの切り替えやproduction受入済みと解釈しない。

主な根拠:

- `docs/stage5/stage5_revision_management_record.md`
- `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`
- `docs/stage5/s5-08-09/stage5_overlap_aggregation_handoff_prompt.md`
- `docs/stage5/s5-08-09/stage5_overlap_aggregation_implementation_policy.md`
- `docs/stage5/FILES.md`

## 8. 次の詳細確認で解決する事項

- HEAD・index・作業ツリーの差分と全118エントリの区分対応。
- リネームだけの変更と、移動後の本文変更の分離可否。
- Dockerfileの削除・追加が単純移設か、内容変更を伴うか。
- CVAT importer、可視化、Stage 5 CLI等の共有ファイルに重なる複数段階の変更。
- v3/v4などの中間状態を現存差分から復元できるか。復元できない状態を捏造しない。
- コミット間の依存関係と各段階で必要な検査・環境・外部入力。
- 文書の過去の実行結果と、今回の分割後に検証できた結果の区別。
- `.venvs/`の除外案と依存関係・構築手順の管理方法。
- 実環境でユーザーが安全に適用できるステージ・コミット・検証コマンドの具体化。

## 9. 確認・判断の追記履歴

### 2026-09-12 初期記録

- ファイル名・Git状態・docsの主要内容による概略分類を記録した。
- 18区分、35〜55コミット程度を暫定案とした。
- ユーザー指定によりコミット・プッシュはユーザー実行とする役割分担を確定した。
- この文書作成時点では追加のソース詳細調査へ進んでいない。

今後の追記には、確認対象、観測事実、判断と根拠、未解決事項、検証またはユーザー実行結果を記載する。

### 2026-09-12 詳細確認: 差分・依存関係・分割境界

#### 確認範囲と証拠

- ローカルHEADは `5e72cc93d679acae2cb16b582f27bfe44886d2d0`。今回もGitの読み取りは成功した。
- 仮想環境を除く118エントリを列挙し、付録Aに全件の区分割当を記録した。
- 既存コードのHEAD差分、文書移動のHEAD/index/worktree比較、新規Pythonの構文木・import・関数構成、新規設定5件の内容、shellの呼び出し先と実行条件を調査した。
- 分割判断に影響する共有実装（v4/v5 importer、Stage 5学習・モデル・CLI等）を追加で読んだ。全37,159行の逐行レビューやアルゴリズムの正当性監査が完了したという意味ではない。
- 対象Python/shellの現行ファイル総行数は37,159行。そのうち未追跡Python/shellは32,025行。既存ファイルの行数は変更行数ではない。
- `research/development/commit-preparation/commit_inventory.json` に状態・移動元・行数・import・トップレベル関数/クラス・区分を保存した。
- `research/development/commit-preparation/commit_dependency_edges.json` に静的に取得したローカルimport辺を保存した。動的import、shell埋込みPython、全相対importの完全な依存解析ではない。

#### 初期案から変更する判断

1. **区分03を独立コミット群にしない。** `check_stage4_sampling_sweep_manifest.py` の変更は動画除外機能への依存追加なので区分10へ、`check_stage4_bbox_ranked_label_policy.py` はv5 provenanceを条件にno-BBox positiveを許す変更なので区分12へ移す。既存PLY出力shellはv6 collected入力と最終label色への切替なので区分13/16へ移す。
2. **Dockerfileは単純移設ではない。** ルート版は旧 `Stage2to4/Dockerfile.codex` にClaude Codeのnpmインストール行を加え、末尾改行も変更している。移設とツール追加を別候補にする。
3. **現在のignore変更には `.venvs/` がない。** HEADから追加されたのは `.tmp/` と `CLAUDE.md`。仮想環境除外は未実施の提案であり、クリーン化時に別途扱う必要がある。
4. **文書移動は分離可能。** 15件すべてで移動元HEADと移動先indexの内容がバイト一致する。10件は作業ツリーも一致し、5件は移動後の追加編集がある。ステージ済み追加文書1件（overlap handoff）は移動群から除く。
5. **基盤を新規コミットとして再現しない。** Stage 4監査本体、既存annotation本体、Stage 5 Dataset・model factory等はHEADに存在する。新規コードから静的に参照する変更対象外のローカル依存17ファイルもHEADに存在する。
6. **新規ファイルでも機能単位の分割が必要。** 未追跡というだけでは一括追加できない。特にreview exporter、CVAT Task作成、fullvideo importer、point label可視化、roundtrip checkerに複数段階が同居する。

#### 共有ファイルの分割境界

| ファイル | 確認した境界・依存 |
| --- | --- |
| `Stage5/train_stage5.py` | `run_one_epoch`・loss集計・勾配蓄積引数と、GroupNorm import・model kwargs・CLIを分離可能な候補とする |
| `Stage5/train_stage5.sh` | batch 1/accumulation 8、v6入力と約200行のpreflight、GroupNorm knob・初期checkpoint上書きの3系統。run名の同一行に複数系統が混ざるため機械的hunk選択だけでは足りない |
| `Stage5/checks/dummy/check_dummy_pointnext_s_training.py/.sh` | accumulation件数検査とnorm選択・checkpoint reload・出力先の分割が必要 |
| `Stage5/evaluate_stage5.py`・`infer_stage5.py` | 現在のPython差分はGroupNorm設定の受渡し。teacher v6の移行は主にshell側。Pythonまで区分16へ一括追加しない |
| `Stage5/stage5/training/losses.py` | `loss_sum`・`loss_normalizer`を返すAPI追加。これを先に成立させ、勾配蓄積の利用側を後に置く |
| `batch_export_stage4_manual_review_cvat.py` | 基本review、全Phase 3ケース、full-video描画/partitionが同居。v3 batchも `_phase3_contract`・`_read_proposal` をimportするため、exporter基盤がv3適用より先 |
| `check_stage4_cvat_manual_roundtrip.py` | 初期roundtripに加えてfullvideo描画・textfree移行をimport/検査。初期段階へ現行ファイル全量を追加すると後続依存が欠ける |
| `batch_create_stage4_phase5_cvat_tasks.py` | selected/master Taskとfullvideo standalone Task、修正snapshot再利用・除外対応が同居。CVAT接続は今回実行しない |
| `batch_import_stage4_phase5_fullvideo_cvat.py` | v4とv5を `output_teacher_token` で分岐。v4処理は残るが `_package_contract` は既にv5対応readerを使う。過去のv4コードそのものが保存されているとは言えない |
| `check_stage4_phase5_fullvideo_final_import.py` | 現行検査はauthoritative oracle/readerをimport。v4段階へそのまま追加せず段階別fixture・assertを選別する |
| `export_stage4_point_label_visualization.py` とbatch/checker | v4保存ラベル、v5全frame CVAT、v6 XML無効化表示を共通実装。CLI・schema受付・resume確認も同時に分ける |

#### 再現できる状態の考え方

- 「過去の日付時点のコードを完全復元」と「今回のコミットをcheckoutして段階の機能を再現」を区別する。
- v4/v5の動作分岐は現在も残っている。ただし共通reader等が後から更新されているため、当時のバイト列や生成物の完全一致は未保証。
- Stage 4の実データ依存は単純な直列ではない。v4とv5はいずれもsource v3とreview/snapshotを入力とし、v6はv5を入力にする。v5はv4 H5の差分補正ではない。
- 分割用の中間ファイルを作る際は、先行コミットの全import・CLI・設定・検査が揃うことを確認する。後続機能のstubや、既知のGroupNorm decoder不具合を意図的に再導入してコミット数を増やさない。
- 最終ソースが現在の作業ツリーと一致することを検証する。新たな修正を加える場合は既存変更の分割と区別し、理由を記録する。
- コードのcheckoutだけでは外部H5、CVAT ZIP/backup、学習重みや実行環境は復元しない。段階ごとに入力manifest・hash・設定・実行コマンド・確認結果の所在を記録する必要がある。

#### 次段階で具体化する43コミット候補

初期目安35〜55件の範囲内で、43候補まで具体化した。これは実行確定済みのコマンド列ではない。共有ファイルの段階別パッチと検査の成立を確認してから確定する。各機能の仕様・検査・結果記録はその機能の候補へ含め、最後の文書コミットへすべて先送りしない。

| 候補 | 内容 | 主な先行条件・分割注意 |
| --- | --- | --- |
| C01 | ignoreとローカル環境管理 | `.venvs/`除外・依存再現は追加提案として明示 |
| C02 | Dockerfileのルート移設 | 旧内容を維持した移設 |
| C03 | Docker環境へのClaude Code追加 | C02 |
| C04 | 文書移動と参照基準の整備 | 既存indexの移動部分を利用。新規報告本文を混ぜない |
| C05 | CVAT mask converterと形式検査 | context-only欠落・RGB等を含め、内部で成立する単位にする |
| C06 | 輪郭refinement核心と設定 | screening/production双方を参照する検査を考慮 |
| C07 | read-only prototypeと統合検査 | C06、HEADの監査基盤 |
| C08 | manual review共通契約・設定 | 3値ラベル、manifest、mask共通処理 |
| C09 | 基本review exporter | C05/C07/C08、後続機能のimportを残さない |
| C10 | manual correction importerと基本roundtrip | C09、checkerのfullvideo/textfree部分を後続へ |
| C11 | teacher v3のbatch適用・構築 | C07/C09、production設定、適用検査 |
| C12 | 全Phase 3ケースのreview出力 | C09/C11、all_phase3・動画別partition |
| C13 | 可逆な動画除外manifest | loader/builder/固定CSVと既存sampling検査変更 |
| C14 | selected/master CVAT Task作成 | C05/C12、fullvideo依存をまだ持ち込まない |
| C15 | fullvideo export・crop metric・描画 | C12、関連roundtrip検査の該当部分 |
| C16 | fullvideo preflight・package検証 | C15、export shellがvalidatorを呼ぶ順序に注意 |
| C17 | fullvideo Task作成・再開・修正snapshot再利用 | C13/C14/C16、fake client検査 |
| C18 | textfree package移行・入口 | C15/C16/C17、roundtrip検査の残り |
| C19 | CVAT Task snapshot保存・検査 | C17、checksum・backup・resume |
| C20 | teacher v4 importer・構築 | C10/C13/C18/C19、v4用の成立する検査に分ける |
| C21 | v4保存point labelの可視化 | C05/C20、batchと検査、既知不整合の観測記録 |
| C22 | v5の全Task frame reader・3値契約 | C20、共通readerの置換範囲を検証 |
| C23 | v5 authoritative適用・ラベル監査 | C22、既存no-BBox checker変更もここ |
| C24 | v5 read-only preflight | C23、入力不変性とprojection検査 |
| C25 | v5可視化・受入検査・構築pipeline | C21/C23/C24 |
| C26 | 削除XMLのread-only監査 | C13/C25、検査とshell |
| C27 | 明示的XML無効化manifest | C26、固定7行CSVと検証 |
| C28 | 単一H5へのXML無効化適用 | C27、provenance・不変条件・resume検査 |
| C29 | v6 batch構築・収集 | C28、batch検査と構築shell |
| C30 | v6無効化可視化とPLY切替 | C21/C25/C29、可視化検査・pipeline |
| C31 | BBox境界接触の未実装計画保存 | 実装を追加しない。最新authorityとの関係を記録 |
| C32 | Stage 5 batch integrity診断 | HEADのDataset、checkerとshell |
| C33 | padding parity診断 | 旧baseline対象の診断として保持 |
| C34 | weighted CEの和と分母API | loss変更とdummy検査中のloss単体契約 |
| C35 | point-weighted勾配蓄積とpadding-free設定 | C34、学習本体・dummy・shellの該当差分 |
| C36 | Stage 5入力・評価・匿名化run参照のv6移行 | C29/C30/C35、train shellのv6 preflightを含む |
| C37 | overlap aggregation診断 | 現行mean baseline比較・出力・self-test |
| C38 | BatchNorm mode parity診断 | C33/C37のhelperをimport |
| C39 | BatchNorm recalibration診断 | C32/C33/C37/C38のhelperをimport |
| C40 | GroupNorm adapter・decoder対応・構造検査 | decoder修正を含めてnorm経路全体を成立させる |
| C41 | GroupNormの学習/推論/評価CLI・checkpoint対応 | C40、古いcheckpointはbatchnormへfallback |
| C42 | GroupNorm dummy学習・BN→GN転移 | C35/C41、shell knobs・checkpoint上書き設定 |
| C43 | 現在状態・文書目次・管理記録の整合 | 各機能で追加済みの記録を統合。5 epoch評価待ちを維持 |

#### 分割前に対応する要確認箇所

- `../stage2to4/stage4/stage4_phase7_report.md` のコマンド内Stage 5パスが `../Stage5/...` から `../../../Stage5/...` に変更されている。同ブロックのStage 4検査パスはStage2to4ルートからの実行を前提にしており、文書移動はshellのcwdを変えない。実行コマンドと文書相対リンクを区別して修正候補にする。現時点ではコード・文書本文を修正していない。
- 新旧文書内に `docs/stage4/...` 形式の参照が残る。履歴のパス表記と現行案内を区別し、現行参照の整備をC04/C43で検討する。
- GroupNormの`--self_test`はGPU不要でも、トップレベルimportにtorchやH5関連の依存がある。overlapのself-testもトップレベルでtorch/h5pyをimportするため、「numpyだけでそのまま起動できる」とは扱わない。
- 学習shellの既定値はEPOCHS=200のまま。記録上の5 epoch診断と区別し、後でユーザーへ検証コマンドを提示する際はepoch数・出力先を明示する。今回は学習を起動しない。
- v4/v5 reader分離と可視化の段階別パッチはまだ作っていない。過去挙動の再現・中間コミットの受入可否は未確定。

#### 今回の検証結果と限界

- 変更対象Python 55本をAST parseし、構文エラー0件。
- 変更対象shell 31本を `bash -n` で確認し、エラー0件。埋込みPythonの実行や実データ処理は含まない。
- 調査対象の絶対ローカルimportで参照先不在0件。import実行、symbol解決、動的依存まで保証するものではない。
- 現行 `python3` ではnumpy、torch、h5py、cv2、yamlが見つからない。依存インストールやGPU/実H5/CVAT検査は実施していない。
- `research/development/commit-preparation/commit_static_checks.json` に静的検査件数と移動比較結果を保存した。
- ステージ・コミット・プッシュは実行していない。変更したのは `.tmp/` の調査記録・補助資料のみ。

#### 次の作業境界

今回で全件の区分割当と主要な差分・依存関係の確認を記録した。次はC01〜C43の段階別パッチ/ファイル内容を具体化し、適用後のimport・設定・検査を成立させる。ユーザー実環境の状態確認用コマンドと、レビュー可能な段階別のステージ・コミットコマンドはその後に提示する。Git書き込みとプッシュは引き続きユーザーが実行する。

## 付録A. 全118エントリの区分割当（2026-09-12詳細確認）

状態は初回inventoryと同じ。区分IDは第6節に対応し、複数IDは差分/節の分割が必要。
区分03は実装内容確認により独立区分から解除し、10・12・13へ再配属した。

| 状態 | 対象（移動先） | 区分 | 判断・分割上の注意 |
| --- | --- | --- | --- |
| ` M` | `.gitignore` | 01 | 環境設定。Dockerfileは移設＋Claude Code導入、別差分として扱う |
| ` D` | `Stage2to4/Dockerfile.codex` | 01 | 環境設定。Dockerfileは移設＋Claude Code導入、別差分として扱う |
| ` M` | `Stage2to4/checks/stage4/check_stage4_bbox_ranked_label_policy.py` | 12 | no-BBox positiveをv5 provenance付きで許可 |
| ` M` | `Stage2to4/checks/stage4/check_stage4_sampling_sweep_manifest.py` | 10 | 動画除外契約とその検査 |
| ` M` | `Stage2to4/pseudo3d/pipelines/export_stage4_bbox_ranked_pointcloud_visualizations.sh` | 13, 16 | v6 collected H5参照と最終label色への切替 |
| ` M` | `Stage5/checks/dummy/check_dummy_pointnext_s_training.py` | 15, 18 | 勾配蓄積の検査とGroupNorm選択・reloadを分割 |
| ` M` | `Stage5/checks/dummy/check_dummy_pointnext_s_training.sh` | 15, 18 | 勾配蓄積の検査とGroupNorm選択・reloadを分割 |
| ` M` | `Stage5/checks/dummy/check_dummy_training.py` | 15 | loss_sum/normalizerと勾配蓄積の契約 |
| ` M` | `Stage5/evaluate_stage5.py` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| ` M` | `Stage5/evaluate_stage5.sh` | 16 | v6入力/run参照への移行。評価checkpoint既定値の変更を含む |
| ` M` | `Stage5/export_anonymized_stage5_metrics.sh` | 16 | v6入力/run参照への移行。評価checkpoint既定値の変更を含む |
| ` M` | `Stage5/infer_stage5.py` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| ` M` | `Stage5/infer_stage5.sh` | 16 | v6入力/run参照への移行。評価checkpoint既定値の変更を含む |
| ` M` | `Stage5/stage5/models/pointnext_s_segmentor.py` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| ` M` | `Stage5/stage5/training/losses.py` | 15 | loss_sum/normalizerと勾配蓄積の契約 |
| ` M` | `Stage5/train_stage5.py` | 15, 18 | 勾配蓄積とGroupNormのhunk分割 |
| ` M` | `Stage5/train_stage5.sh` | 15, 16, 18 | 勾配蓄積・v6 preflight・GroupNorm・run名が交錯 |
| `RM` | `docs/README.md` | 02, 各機能の目次 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage2to4/docs/README.md`。 |
| `R ` | `docs/stage2to4/legacy/dualtrack_legacy.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage2to4/docs/legacy/dualtrack_legacy.md`。 |
| `R ` | `docs/stage2to4/stage4/cvat_segmentation_mask_1_1_import_spec.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage2to4/docs/stage4/cvat_segmentation_mask_1_1_import_spec.md`。 |
| `R ` | `docs/stage2to4/stage4/edit_prompt.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage2to4/docs/stage4/edit_prompt.md`。 |
| `R ` | `docs/stage2to4/stage4/stage4_bbox_independent_sampling_edit_prompt.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `stage4_edit_prompt.md`。 |
| `RM` | `docs/stage2to4/stage4/stage4_contour_teacher_improvement_plan.md` | 02, 04, 05, 06, 07, 08, 09, 10 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage2to4/docs/stage4/stage4_contour_teacher_improvement_plan.md`。 |
| `RM` | `docs/stage2to4/stage4/stage4_phase7_report.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage2to4/docs/stage4/stage4_phase7_report.md`。 |
| `R ` | `docs/stage2to4/stage4/stage4_sampling_investigation_progress.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage2to4/docs/stage4/stage4_sampling_investigation_progress.md`。 |
| `R ` | `docs/stage2to4/stage4/stage4_sampling_sweep_investigation_edit_prompt.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `stage4_invest_edit_prompt.md`。 |
| `R ` | `docs/stage2to4/stage4/stage4_sampling_sweep_investigation_rule.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `stage4_invest_rule.md`。 |
| `RM` | `docs/stage2to4/submission/README.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage2to4/docs/submission/README.md`。 |
| `RM` | `docs/stage5/FILES.md` | 02, 14, 15, 16, 17, 18 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage5/FILES.md`。 |
| `R ` | `docs/stage5/TRAINING_IMPROVEMENT_PLAN.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `Stage5/TRAINING_IMPROVEMENT_PLAN.md`。 |
| `R ` | `docs/stage5/data_construct.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `../stage5/data_construct.md`。 |
| `R ` | `docs/stage5/stage5_edit_prompt.md` | 02 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う 移動元: `../stage5/stage5_edit_prompt.md`。 |
| `AM` | `docs/stage5/s5-08-09/stage5_overlap_aggregation_handoff_prompt.md` | 17 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う |
| `??` | `Dockerfile.codex` | 01 | 環境設定。Dockerfileは移設＋Claude Code導入、別差分として扱う |
| `??` | `Stage2to4/checks/stage4/check_stage4_contour_auto_refine.py` | 05 | checkerはscreening/production両configを参照 |
| `??` | `Stage2to4/checks/stage4/check_stage4_contour_teacher_phase5_apply.py` | 07 | review共通処理とexporter helperが先行依存 |
| `??` | `Stage2to4/checks/stage4/check_stage4_cvat_authoritative_preflight.py` | 12 | v5検査・preflight・pipeline |
| `??` | `Stage2to4/checks/stage4/check_stage4_cvat_manual_roundtrip.py` | 06, 09 | full-video描画・textfree移行もimport/検査。全量を初期段階へ追加しない |
| `??` | `Stage2to4/checks/stage4/check_stage4_cvat_segmentation_mask_export.py` | 04 | context-only欠落許可等も含む現行converter契約 |
| `??` | `Stage2to4/checks/stage4/check_stage4_deleted_xml_annotation_audit.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/checks/stage4/check_stage4_deleted_xml_invalidation_apply.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/checks/stage4/check_stage4_deleted_xml_invalidation_batch.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/checks/stage4/check_stage4_deleted_xml_invalidation_manifest.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/checks/stage4/check_stage4_deleted_xml_invalidation_visualization.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/checks/stage4/check_stage4_phase5_cvat_task_snapshots.py` | 10 | snapshot保存・v4構築 |
| `??` | `Stage2to4/checks/stage4/check_stage4_phase5_fullvideo_cvat_tasks.py` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage2to4/checks/stage4/check_stage4_phase5_fullvideo_final_import.py` | 10, 12 | v4/v5同居。共有readerもv5対応済み |
| `??` | `Stage2to4/checks/stage4/check_stage4_point_label_visualization.py` | 11, 12, 13 | 保存label描画とCVAT全frame・XML無効化対応が同居 |
| `??` | `Stage2to4/checks/stage4/check_stage4_v5_cvat_authoritative_acceptance.py` | 12 | v5検査・preflight・pipeline |
| `??` | `Stage2to4/pseudo3d/analysis/audit_stage4_deleted_xml_annotations.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/analysis/build_stage4_deleted_xml_invalidation_manifest.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/analysis/build_stage4_exclusion_manifest.py` | 10 | 動画除外契約とその検査 |
| `??` | `Stage2to4/pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3.yaml` | 05 | checkerはscreening/production両configを参照 |
| `??` | `Stage2to4/pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3_production_v1.yaml` | 05 | checkerはscreening/production両configを参照 |
| `??` | `Stage2to4/pseudo3d/analysis/configs/stage4_deleted_xml_invalidations_v1.csv` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/analysis/configs/stage4_manual_review_cvat_v1.yaml` | 06 | importerからreview exporterのhelper参照あり |
| `??` | `Stage2to4/pseudo3d/analysis/configs/stage4_video_exclusions_v1.csv` | 10 | 動画除外契約とその検査 |
| `??` | `Stage2to4/pseudo3d/analysis/preflight_stage4_cvat_authoritative_labels.py` | 12 | v5検査・preflight・pipeline |
| `??` | `Stage2to4/pseudo3d/analysis/preflight_stage4_phase5_fullvideo_cvat_review.py` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage2to4/pseudo3d/analysis/prototype_stage4_contour_auto_refine.py` | 05 | checkerはscreening/production両configを参照 |
| `??` | `Stage2to4/pseudo3d/analysis/validate_stage4_phase5_fullvideo_cvat_package.py` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage2to4/pseudo3d/annotation/apply_deleted_xml_invalidations.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/annotation/contour_teacher_refinement.py` | 05 | checkerはscreening/production両configを参照 |
| `??` | `Stage2to4/pseudo3d/annotation/import_cvat_segmentation_mask_corrections.py` | 06 | importerからreview exporterのhelper参照あり |
| `??` | `Stage2to4/pseudo3d/annotation/stage4_manual_review.py` | 06 | importerからreview exporterのhelper参照あり |
| `??` | `Stage2to4/pseudo3d/batch/annotation/batch_apply_stage4_contour_refinement.py` | 07 | review共通処理とexporter helperが先行依存 |
| `??` | `Stage2to4/pseudo3d/batch/annotation/batch_apply_stage4_deleted_xml_invalidations.py` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/batch/annotation/batch_import_cvat_segmentation_mask_corrections.py` | 06 | importerからreview exporterのhelper参照あり |
| `??` | `Stage2to4/pseudo3d/batch/annotation/batch_import_stage4_phase5_fullvideo_cvat.py` | 10, 12 | v4/v5同居。共有readerもv5対応済み |
| `??` | `Stage2to4/pseudo3d/batch/export/batch_create_stage4_phase5_cvat_tasks.py` | 08, 09 | selected/master/fullvideo/修正snapshot再利用が同居 |
| `??` | `Stage2to4/pseudo3d/batch/export/batch_export_stage4_manual_review_cvat.py` | 06, 07, 08, 09 | 基本export・v3で使うhelper・全ケース・full-videoが同居 |
| `??` | `Stage2to4/pseudo3d/batch/export/batch_export_stage4_phase5_cvat_task_snapshots.py` | 10 | snapshot保存・v4構築 |
| `??` | `Stage2to4/pseudo3d/batch/export/batch_export_stage4_point_label_visualization.py` | 11, 12, 13 | 保存label描画とCVAT全frame・XML無効化対応が同居 |
| `??` | `Stage2to4/pseudo3d/batch/export/rebuild_stage4_phase5_textfree_review_package.py` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage2to4/pseudo3d/export/convert_masks_to_cvat_segmentation_mask_1_1.py` | 04 | context-only欠落許可等も含む現行converter契約 |
| `??` | `Stage2to4/pseudo3d/export/export_stage4_point_label_visualization.py` | 11, 12, 13 | 保存label描画とCVAT全frame・XML無効化対応が同居 |
| `??` | `Stage2to4/pseudo3d/pipelines/audit_stage4_v5_deleted_xml_annotations.sh` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/pipelines/build_stage4_bbox_ranked_v3_refined_auto.sh` | 07 | review共通処理とexporter helperが先行依存 |
| `??` | `Stage2to4/pseudo3d/pipelines/build_stage4_bbox_ranked_v4_manual_fullvideo.sh` | 10 | snapshot保存・v4構築 |
| `??` | `Stage2to4/pseudo3d/pipelines/build_stage4_bbox_ranked_v5_cvat_authoritative.sh` | 12 | v5検査・preflight・pipeline |
| `??` | `Stage2to4/pseudo3d/pipelines/build_stage4_bbox_ranked_v6_xml_invalidation.sh` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/pipelines/build_stage4_deleted_xml_invalidation_manifest.sh` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/pipelines/create_stage4_phase5_cvat_tasks.sh` | 08 | 全ケースレビューの入口 |
| `??` | `Stage2to4/pseudo3d/pipelines/create_stage4_phase5_fullvideo_cvat_tasks.sh` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage2to4/pseudo3d/pipelines/create_stage4_phase5_fullvideo_textfree_cvat_tasks.sh` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage2to4/pseudo3d/pipelines/export_stage4_phase5_cvat_review_cases.sh` | 08 | 全ケースレビューの入口 |
| `??` | `Stage2to4/pseudo3d/pipelines/export_stage4_phase5_fullvideo_cvat_review.sh` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage2to4/pseudo3d/pipelines/export_stage4_phase5_fullvideo_cvat_task_snapshots.sh` | 10 | snapshot保存・v4構築 |
| `??` | `Stage2to4/pseudo3d/pipelines/export_stage4_v4_point_label_visualizations.sh` | 11 | v4可視化入口 |
| `??` | `Stage2to4/pseudo3d/pipelines/export_stage4_v5_cvat_authoritative_point_label_visualizations.sh` | 12 | v5検査・preflight・pipeline |
| `??` | `Stage2to4/pseudo3d/pipelines/export_stage4_v6_xml_invalidation_point_label_visualizations.sh` | 13 | 監査→manifest→適用→batch→可視化の順 |
| `??` | `Stage2to4/pseudo3d/pipelines/preflight_stage4_phase5_fullvideo_cvat_review.sh` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage2to4/pseudo3d/pipelines/preflight_stage4_v5_cvat_authoritative_labels.sh` | 12 | v5検査・preflight・pipeline |
| `??` | `Stage2to4/pseudo3d/pipelines/rebuild_stage4_phase5_fullvideo_textfree_review.sh` | 09 | full-video検査・package・Task・textfree移行 |
| `??` | `Stage5/checks/dummy/check_dummy_pointnext_s_groupnorm.py` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| `??` | `Stage5/checks/dummy/check_dummy_pointnext_s_groupnorm.sh` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| `??` | `Stage5/checks/real_h5/check_stage5_batch_integrity.py` | 14 | 診断checkerとshellを組にする |
| `??` | `Stage5/checks/real_h5/check_stage5_batch_integrity.sh` | 14 | 診断checkerとshellを組にする |
| `??` | `Stage5/checks/real_h5/check_stage5_batchnorm_mode_parity.py` | 17 | overlap→BN parity→recalibrationのimport順を守る |
| `??` | `Stage5/checks/real_h5/check_stage5_batchnorm_mode_parity.sh` | 17 | overlap→BN parity→recalibrationのimport順を守る |
| `??` | `Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.py` | 17 | overlap→BN parity→recalibrationのimport順を守る |
| `??` | `Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.sh` | 17 | overlap→BN parity→recalibrationのimport順を守る |
| `??` | `Stage5/checks/real_h5/check_stage5_overlap_aggregation.py` | 17 | overlap→BN parity→recalibrationのimport順を守る |
| `??` | `Stage5/checks/real_h5/check_stage5_overlap_aggregation.sh` | 17 | overlap→BN parity→recalibrationのimport順を守る |
| `??` | `Stage5/checks/real_h5/check_stage5_padding_parity.py` | 14 | 診断checkerとshellを組にする |
| `??` | `Stage5/checks/real_h5/check_stage5_padding_parity.sh` | 14 | 診断checkerとshellを組にする |
| `??` | `Stage5/checks/transfer/check_stage5_batchnorm_to_groupnorm_transfer.py` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| `??` | `Stage5/checks/transfer/check_stage5_batchnorm_to_groupnorm_transfer.sh` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| `??` | `Stage5/stage5/models/norm_layers.py` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| `??` | `Stage5/stage5/models/pointnext_decoder_patch.py` | 18 | GroupNorm本体・decoder・CLI/checkpoint・構造/転移検査 |
| `??` | `docs/stage2to4/stage4/stage4_bbox_ranked_border_contact_revision_plan.md` | 計画のみ | 未実装計画。機能実装コミットにはしない |
| `??` | `docs/stage2to4/stage4/stage4_cvat_snapshot_authoritative_label_revision_plan.md` | 12, 13 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う |
| `??` | `docs/stage2to4/stage4/stage4_deleted_xml_annotation_invalidation_plan.md` | 13 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う |
| `??` | `docs/stage2to4/stage4/stage4_phase5_fullvideo_cvat_review_implementation.md` | 09, 10, 12, 13 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う |
| `??` | `docs/stage2to4/stage4/stage4_v4_point_label_visualization_plan.md` | 11, 12 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う |
| `??` | `docs/stage5/s5-08-09/stage5_overlap_aggregation_implementation_policy.md` | 17 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う |
| `??` | `docs/stage5/stage5_pointnext_s_training_evaluation_report.md` | 14, 15, 16, 17, 18 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う |
| `??` | `docs/stage5/stage5_revision_management_record.md` | 14, 15, 16, 17, 18 | 移動があるものは移動と本文を分離。複数段階の報告は節単位で扱う |

## 10. 2026-09-12 段階別パッチ・ユーザー実行手順の具体化

### 成果物

- `research/development/commit-preparation/archive/commit-bundle/README.md`: ユーザー実環境での事前照合、別worktree準備、段階別コミット、元branchへの反映、追加環境整理、pushの手順。
- `research/development/commit-preparation/archive/commit-bundle/patches/001.patch`〜`037.patch`: 現在の未コミット変更を再構成する37本の実パッチ。
- `research/development/commit-preparation/archive/commit-bundle/manifest.json`: 基準HEAD、段階別タイトル・before/after SHA-256・mode、元作業ツリーの期待値、既存indexの許容内容。
- `research/development/commit-preparation/archive/commit-bundle/series.txt`: 実パッチの順序とコミット名。
- `research/development/commit-preparation/archive/commit-bundle/user_commit.py`: ユーザー実行用。preflightのみ読み取り専用。prepare/step/finalizeはユーザー環境でGitを書き換える。
- `research/development/commit-preparation/archive/commit-bundle/verify_replay.py`・`verification.json`: Git管理外ツリーでの再生検証と結果。
- `research/development/commit-preparation/archive/commit-bundle/test_user_commit.py`: 実行補助スクリプトの停止条件検査。Git書き込みはモックし、実行していない。
- `research/development/commit-preparation/archive/commit-bundle/VALIDATION.md`: 実環境での動作検証候補と未検証事項。
- `research/development/commit-preparation/archive/commit-bundle/optional-environment.patch`: 追加提案の38件目。既存変更の再構成とは区別する。
- `research/development/commit-preparation/archive/commit-bundle.zip`: ユーザー環境への移動用バンドル。生成スクリプトは元inventoryを参照するため、通常の実行手順で再生成しない。

### 43候補から37実パッチへ変更した理由

実ファイルを組み立てた結果、v4/v5の共通reader、全ケース/fullvideo exporter、CVAT Task作成、複数schemaの可視化、GroupNormのdecoder/CLI/checkpointは相互依存が強いと判断した。旧コードを推測して復元したり、既知不具合を再導入したりせず、共通実装と検査を同じ段階にまとめる。

- 旧C09/C12/C15のexporter能力は実パッチ009で共通実装として導入。運用入口・validator・textfreeは010〜013へ分離。
- 旧C14/C17のTask作成は015で統合し、fake client検査も同時に追加。
- 旧C20/C22/C23のv4/v5 importerとlabel checkerは017で統合。v4入口は018、v5 preflight・受入・構築入口は020/021へ分離。
- 旧C21/C25/C30のrendererは019で現行schema readerをまとめて追加。v6無効化の統合検査・入口は026で追加。
- 旧C40〜C42のGroupNormは036で統合。decoder・モデル・CLI・checkpoint・検査を一緒に成立させる。
- 文書は移設004と、Stage 4の既存実験を記す回顧的報告027、Stage 5の報告/全体目次037に分ける。過去の報告を細切れにして「今回の各中間コミットで実行済み」と誤認させる形にはしない。個別checker設計文書は033へ含める。
- 勾配蓄積とGroupNormは実際の中間ファイルを作成した。031の学習Pythonと032の学習shellはnorm関連部分だけを保留し、036で元の最終内容に完全一致させる。

過去の43候補は検討履歴として残す。実行する順序・ファイル範囲は `series.txt` と `manifest.json` を正本とする。

### 37件目と38件目の違い

37件目の状態は、調査時の仮想環境以外118エントリを反映した作業ツリーとバイト・mode単位で一致する。移動元の削除も含む。新たな機能修正や文書パス修正は混ぜていない。

38件目は既存変更にはなかった追加提案で、次だけを変更する。

1. `.gitignore` に `.venvs/` を追加。
2. `requirements-cvat-review.txt` に既存CVAT環境の12パッケージの観測バージョンを記録。
3. `docs/development/cvat_environment.md` にPython 3.11.15、CVAT SDK 2.73.0、再作成方法と検証の限界を記録。

既存 `.venvs/` 内のMETADATAとpyvenv.cfgを読み取って作成した。パッケージ導入・環境再作成は未実行で、hash付きlockfileではない。Stage 4/5用のPyTorch/CUDA環境まで再現するものではない。環境本体1,680件をGitへ追加しない。38件目を採用すれば、調査時に見えていた未追跡仮想環境を除外してクリーン化できる。採用しない場合は別の除外方針が必要。

### 実行の保全方針

- source HEADと全対象ファイル、既存index内容、追加の未追跡ファイル、submodule状態を照合。不一致は強制上書きせず停止する。
- 別branch `split/uncommitted-20260912` と別worktreeでコミットを作る。元の作業ツリー・indexは最後まで維持する。
- ユーザー実行のprepare時に元branch/HEAD・staged/unstaged差分を `user-backup/` へ保存する。既存バックアップ内容が異なる場合は上書きしない。
- 各stepは順序、パッチhash、期待ファイル内容を確認してコミットする。既に完了したstepやコミット直前の中断から再開可能。予期しない差分は停止する。
- finalizeは全履歴と元作業ツリーを再照合後、ユーザーが `git reset --mixed <完成コミット>` 相当を実行する。元branchとindexを揃える操作であり、`--hard`、作業ファイルのcheckout・削除は使わない。
- pushは自動化しない。ユーザー環境でremote/branchを確認後、通常のpushを行う。force pushは手順に含めない。
- 38件目はfinalize後の元branchへ追加する。split branchは37件目に留まるため、push対象を混同しない。

### 今回の検証結果

- Git metadataを持たない一時ツリーへ全37パッチを順番に適用し、全段階のbefore/after SHA-256・modeが一致した。
- 各段階の累積変更PythonをAST parse、shellを `bash -n`、`PY` heredocをAST parseした。最終段階はPython 55本、shell 31本、heredoc 12個が成功。
- 各段階の静的に判別できるローカルimport先が存在することを確認した。symbolの動的解決や実行の成功を保証するものではない。
- 37件目が元作業ツリーの全期待ファイル内容・modeと一致。追加環境パッチもその状態へ適用でき、期待hashが一致した。
- 実行補助スクリプトの停止条件6件がモック/一時ファイルの検査で成功。
- 実ワークスペースで `user_commit.py preflight --repo /workspace` が成功。これは読み取りのみ。
- prepare/step/finalize、Git indexへの適用、commit、push、CVAT接続、学習、依存インストールは実行していない。
- 変更したのは `.tmp/` の成果物だけ。ユーザー作業ファイル・元index・HEADを変更していない。

### 次の実行

ユーザー環境へバンドルを配置し、READMEのpreflightから進める。不一致があればその出力をもとにこの文書へ確認結果と判断を追記し、パッチの再生成要否を決める。動作検証の結果も引き続きこの文書に記録する。

## 11. ユーザー実環境での完了報告

ユーザーからの報告と提示ログに基づく記録。コンテナのGit状態やリモートを再照会した結果ではない。

- ユーザーより実行手順「4. 仮想環境の整理」まで進めたとの報告を受領。
- `git remote -v` と `git branch --show-current` の提示結果で、remote名 `origin`、現在branch `main` を確認。
- ユーザーが `git -C "$REPO" push -u origin main` を実行し、成功ログを提示。
- push先: `github.com:yutautan-sketch/pseudo3d_models.git`。
- リモートmainの更新範囲: `481d8c8..c7c8cf5`。更新先は `c7c8cf5`。
- `main -> main` と `branch 'main' set up to track 'origin/main'` により、push成功とupstream設定完了を確認。
- ログの開始SHA `481d8c8` はリモートmainの更新前の先端を表す。パッチ構築基準HEAD `5e72cc9...` と異なることだけを不整合とは判断しない。
- 分割コミット・環境整理はユーザー報告上完了し、リモートへの反映は提示ログで確認できた。実際のコミット件数と最終作業ツリーのクリーン状態は、このpushログだけでは確認できない。
- 最終確認としてユーザー環境の `git -C "$REPO" status --short --branch` を使用する。`## main...origin/main` のみで変更行やahead/behind表示がなければ、作業ツリーのクリーン状態とローカルの追跡情報上の同期を確認できる。

配布済みZIPとそのhashは再生成せず維持する。この完了報告は本管理文書への追記であり、配布時点の同梱記録とは更新時点が異なる。
