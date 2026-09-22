# .tmp 配下の調査成果物・管理文書の移動方針

作成日: 2026-09-22
状態: 初回移動241件・Stage5ステップ別再配置25件・文書内の参照パス修正・索引更新を完了。下記の旧パス・未実施表記は各段階の履歴として保持。

## 調査結果

- `.tmp/` は242ファイル、8,761,801 bytes（約8.36 MiB）。CSV 87、JSON 51、JSONL 13、Markdown 30、その他61。シンボリックリンクなし。
- Markdownは直下27件と`commit-bundle/`内3件。Stage5の依頼・報告・判断記録が中心で、S5-16〜S5-20への現行引き継ぎも含む。一括して「過去資料」にしない。
- 調査データはpadding/no-padding、overlap aggregation、BatchNorm、GroupNorm、label policy、class weight、構造診断、回転augmentationのまとまり。共有用ディレクトリにはCSVだけでなくmanifest、匿名化設定、JSONL、共有範囲の説明TXTがあるため、まとまりごと保持する。
- `260917/`はS5-15 R0/R1短期比較、`260919/`はR0長期実験の結果。元の実験名・日付・ディレクトリ階層を保持する。
- 既存docsの`.tmp/`参照は4文書・40行で確認：Stage5管理正本26行、overlap引き継ぎ4行、評価報告9行、Stage4 crop品質計画1行。移動対象文書同士にも参照があり、S5-16引き継ぎには相対Markdownリンクがある。
- 調べたStage5/Stage2to4のPython・shell・JSONには`.tmp/`参照なし。一方、コミットバンドルのbuild scriptは`.tmp/commit_inventory.json`に依存し、実行補助は`.tmp/`を未追跡検出から除外する。単なる移動後の再実行は保証できない。
- `uncommitted_changes_commit_plan.md`とバンドル内`management-record.md`は同一内容ではない。統合・削除せず、それぞれの履歴を保持する。
- 開始時から`docs/stage5/FILES.md`と`docs/stage5/data_construct.md`に未コミット変更あり。移動作業で上書き・巻き戻しをしない。

## 配置方針

調査成果物の新しいルートを`research/`とする。実装コードやdocs本文から分離し、`research/stage5/`の調査単位別に置く。ファイル名は維持し、共有パッケージの内部階層も維持する。研究データの数値、実験設定、元の実行環境パスはこの整理で変更しない。

管理文書は既存`docs/README.md`の「Markdownをdocsに集約する」方針に合わせる。Stage5文書は`docs/stage5/`へ直接配置し、既存の管理正本・評価正本を維持する。Stage4へのcrop品質依頼と修正報告の2件は`docs/stage2to4/stage4/`へ配置する。ファイル名のStage5接頭辞は参照の追跡のため保持する。

コミット作業資料は`docs/development/`と`research/development/commit-preparation/`に分ける。再現用バンドルは`research/development/commit-preparation/archive/`に一式保存し、内部Markdown・patch・checksumを分離・書換えしない。この内部Markdownは独立した運用文書ではなく、配布物の構成要素として例外扱いにする。移動先のREADMEで履歴資料であることと、再利用には別途検証が必要なことを示す。

## 調査データ・作業資料の移動対応

以下の移動元はすべて`.tmp/`相対。移動先はリポジトリルート相対。表の移動先ディレクトリに元の名前で格納する。

| 移動元 | 移動先ディレクトリ |
| --- | --- |
| `stage5_padding_parity_metrics.tar.gz` | `research/stage5/padding/` |
| `stage5_anonymized_metrics_SHARE_THIS.tar.gz` | `research/stage5/baseline/` |
| `stage5_nopad_smoke/`, `stage5_v6_nopad_smoke/`, `stage5_v6_nopad_pilot_ep5/`, `stage5_v6_nopad_ep5_eval_metrics_extracted/`, `stage5_v6_nopad_ep5_eval_metrics.tar.gz` | `research/stage5/nopad/` |
| `overlap_aggregation_share/` | `research/stage5/s5-08/` |
| `batchnorm_mode_parity_share/` | `research/stage5/s5-09/` |
| `batchnorm_recalibration_share/` | `research/stage5/s5-10/` |
| `S5-11_anonymized_metrics_SHARE_THIS/`, `S5-11_ep5_anonymized_metrics_SHARE_THIS/` | `research/stage5/s5-11/` |
| `label_policy_bbox_preflight_share_metrics/`, `label_policy_ablation_RunA_share_metrics/`, `label_policy_ablation_RunB_share_metrics/` | `research/stage5/s5-12/` |
| `s5_13_wa_share_metrics/`, `s5_13_wb_share_metrics/`, `s5_13_wc_share_metrics/`, `s5_13_supplement_wa_threshold_free.json`, `s5_13_supplement_wb_threshold_free.json`, `s5_13_supplement_wc_threshold_free.json` | `research/stage5/s5-13/` |
| `s5_14_*.csv`（現存7件） | `research/stage5/s5-14/` |
| `260917/`, `260919/` | `research/stage5/s5-15/` |
| `commit_inventory.json`, `commit_dependency_edges.json`, `commit_static_checks.json` | `research/development/commit-preparation/` |
| `commit-bundle/`, `commit-bundle.zip` | `research/development/commit-preparation/archive/` |

圧縮ファイルはそのまま移動する。tarの構成一覧は確認済みだが、展開済みデータとの完全一致は未検証なので重複排除はしない。`baseline/`は暫定的な分類であり、アーカイブ名だけから特定のS5番号を断定しない。

## Markdownの移動対応

以下の移動元は`.tmp/`相対。すべて元のファイル名を維持する。

| 移動元 | 移動先ディレクトリ |
| --- | --- |
| `commit_inventory_appendix.md` | `docs/development/` |
| `stage5_message_to_implement_chat.md` | `docs/stage5/` |
| `stage5_prediction_frame_visualization_implementation_handoff.md` | `docs/stage5/` |
| `stage5_report_to_policy_chat.md` | `docs/stage5/` |
| `stage5_s5_10_recovery_handoff_prompt.md` | `docs/stage5/` |
| `stage5_s5_10_report_to_policy_chat.md` | `docs/stage5/` |
| `stage5_s5_11_groupnorm_implementation_handoff_prompt.md` | `docs/stage5/` |
| `stage5_s5_11_report_to_policy_chat.md` | `docs/stage5/` |
| `stage5_s5_12_label_policy_implementation_handoff_prompt.md` | `docs/stage5/` |
| `stage5_s5_12_report_to_policy_chat.md` | `docs/stage5/` |
| `stage5_s5_12_stage4_crop_quality_correction_report.md` | `docs/stage2to4/stage4/` |
| `stage5_s5_12_stage4_crop_quality_investigation_request.md` | `docs/stage2to4/stage4/` |
| `stage5_s5_13_class_weight_ablation_implementation_request.md` | `docs/stage5/` |
| `stage5_s5_13_report_to_policy_chat.md` | `docs/stage5/` |
| `stage5_s5_13_supplement_implementation_handoff.md` | `docs/stage5/` |
| `stage5_s5_13_supplement_report_to_policy_chat.md` | `docs/stage5/` |
| `stage5_s5_14_report_to_policy_chat.md` | `docs/stage5/` |
| `stage5_s5_14_step_h4_parity_boundary_case_decision_request.md` | `docs/stage5/` |
| `stage5_s5_14_structural_diagnostics_implementation_handoff.md` | `docs/stage5/` |
| `stage5_s5_14_supplement2_report_to_policy_chat.md` | `docs/stage5/` |
| `stage5_s5_14_supplement_implementation_handoff.md` | `docs/stage5/` |
| `stage5_s5_14_supplement_report_to_policy_chat.md` | `docs/stage5/` |
| `stage5_s5_15_report_to_policy_chat.md` | `docs/stage5/` |
| `stage5_s5_15_rotation_augmentation_implementation_handoff.md` | `docs/stage5/` |
| `stage5_s5_16_implementation_handoff.md` | `docs/stage5/` |
| `stage5_s5_16_to_s5_20_policy_chat_transfer.md` | `docs/stage5/` |
| `uncommitted_changes_commit_plan.md` | `docs/development/` |

## .tmp に残すもの

- `.tmp/log.txt`：一時ログとして残す。ルートの`log.txt`も今回の対象外。
- 今回の移動計画・実績ログと検証記録。バンドル内のcacheは一式保存に含め、今回の整理で削除しない。
- 次回以降の一時生成物。`.gitignore`の`.tmp/`設定は維持する。

## 実施順序と移動ログ

**必ず「内容を変更せず移動 → 移動完了の照合 → 文書内のパス修正」の順で行う。** 移動と本文修正を同時に行わない。

1. 実行時点で再棚卸しし、対象ごとの旧パス・新パス・サイズ・SHA-256を記録する。表のワイルドカードとディレクトリを実ファイル単位に展開する。存在確認・移動先衝突・対象漏れを検査し、既存ファイルは上書きしない。
2. `.tmp/relocation/<UTC日時>/plan.json`に確定対応表を保存する。これは予定表であり、移動済みを意味しない。今回の方針文書自体は移動ログの代用としない。
3. `.tmp/relocation/<UTC日時>/moves.jsonl`へ各移動の開始・完了・失敗を追記し、書き出しを確定させる。各行に`run_id`, `event`, `timestamp_utc`, `old_path`, `new_path`, `size_bytes`, `sha256_before`を持たせ、完了行に`sha256_after`、失敗行に`error`を加える。パスはルート相対。開始行と完了行を分けることで中断も追跡する。
4. 全対象について移動先の存在・サイズ・hash一致、旧パスの不在、対象件数一致を検証し、`verification.json`に保存する。中断時は現物とhashを照合して再開し、未検証のままパス修正へ進まない。
5. 完了ログを根拠に、移動した文書と既存docsの参照を修正する。`.tmp/`全体の一括置換は禁止。旧→新の対応表で完全なパスを照合し、ルート相対・絶対パスの表記・相対Markdownリンク・コードブロック内の手順を個別に確認する。移動前には本文を直さない。
6. `docs/README.md`、`docs/stage5/FILES.md`の索引を更新し、`research/README.md`と各調査の案内を追加する。調査ID・結果の説明・対応する報告書・元の実験名を結び付ける。
7. `reference_updates.jsonl`に修正文書・旧参照・新参照・判断を記録する。旧パスの残存検索と相対リンクの解決確認を行い、意図的に残す履歴参照は`verification.json`に理由付きで列挙する。

移動直後のhashを元データ保持の証拠とする。文書修正後はhashが変わるため、その変更は別段階の記録にする。移動ログ自体の旧パスを置換しない。

## 参照修正の境界と完了条件

- 実在するリポジトリ内成果物を案内するパスは新パスへ更新する。過去の実行事実を表すパスは必要に応じて当時の場所として残し、現在の場所を併記する。
- JSON内の`evaluation_dir`、configの入力・出力・checkpoint等は実験provenanceであり、移動先へ機械的に変更しない。外部実験環境のパスとリポジトリ内参照を区別する。
- アーカイブ内の文書、patch、manifest、checksumは過去時点の一式として保持し、参照修正対象から除外する。バンドル用スクリプトの改修・再生成・実行は今回の整理に含めない。
- `.tmp/`は現在Git無視対象であり、移動先は原則無視されない。移動とGit登録は別扱いとする。`260917/`の未匿名化configや文書には実環境パス・識別子があり、ディレクトリ名だけで全データを共有可能とみなさない。今回の移動で自動stage・commit・pushはしない。
- 対象の全ファイルに移動実績があり、本文修正前のhashが一致し、意図しない参照切れがなく、開始時のユーザー差分が保持されていれば完了。数値再計算や学習テストは不要。

## 2026-09-22 実施記録・Stage5ステップ別配置への改訂

ユーザーの追加指定により、上記の「Stage5文書を直下へ配置」は初回移動の履歴とし、最終配置を以下に改訂する。初回対応表・移動ログは当時の記録として保持する。

- 初回移動: 241ファイル、8,752,674 bytes。全件hash一致・旧パス不在を確認済み。記録: `.tmp/relocation/20260922T093047685249Z/{plan.json,moves.jsonl,verification.json}`。
- 今回の再配置: 25文書。内容・ファイル名は維持。S5-08/09の一連の仕様・設計・完了報告は`docs/stage5/s5-08-09/`にまとめる。
- S5-16〜S5-20への引き継ぎ案内はS5-16を起点とするため`s5-16/`に置く。未作成のS5-17〜S5-20用の空ディレクトリは作らない。
- 今回の記録: `.tmp/relocation/20260922T093725045342Z/{plan.json,moves.jsonl,verification.json}`。
- 統合対応表: `.tmp/relocation/20260922T093725045342Z/current_path_map.json`。初回の旧パス・中間パス・現在パスを243ファイル分記録（初回241件＋既存docsから移動した2件）。次の参照修正では旧パスと中間パスの両方を現在パスへ対応させる。

以下の移動元・移動先は`docs/stage5/`相対。

| 今回の移動元 | 最終配置 |
| --- | --- |
| `stage5_message_to_implement_chat.md` | `s5-10/stage5_message_to_implement_chat.md` |
| `stage5_overlap_aggregation_handoff_prompt.md` | `s5-08-09/stage5_overlap_aggregation_handoff_prompt.md` |
| `stage5_overlap_aggregation_implementation_policy.md` | `s5-08-09/stage5_overlap_aggregation_implementation_policy.md` |
| `stage5_prediction_frame_visualization_implementation_handoff.md` | `s5-15/stage5_prediction_frame_visualization_implementation_handoff.md` |
| `stage5_report_to_policy_chat.md` | `s5-08-09/stage5_report_to_policy_chat.md` |
| `stage5_s5_10_recovery_handoff_prompt.md` | `s5-10/stage5_s5_10_recovery_handoff_prompt.md` |
| `stage5_s5_10_report_to_policy_chat.md` | `s5-10/stage5_s5_10_report_to_policy_chat.md` |
| `stage5_s5_11_groupnorm_implementation_handoff_prompt.md` | `s5-11/stage5_s5_11_groupnorm_implementation_handoff_prompt.md` |
| `stage5_s5_11_report_to_policy_chat.md` | `s5-11/stage5_s5_11_report_to_policy_chat.md` |
| `stage5_s5_12_label_policy_implementation_handoff_prompt.md` | `s5-12/stage5_s5_12_label_policy_implementation_handoff_prompt.md` |
| `stage5_s5_12_report_to_policy_chat.md` | `s5-12/stage5_s5_12_report_to_policy_chat.md` |
| `stage5_s5_13_class_weight_ablation_implementation_request.md` | `s5-13/stage5_s5_13_class_weight_ablation_implementation_request.md` |
| `stage5_s5_13_report_to_policy_chat.md` | `s5-13/stage5_s5_13_report_to_policy_chat.md` |
| `stage5_s5_13_supplement_implementation_handoff.md` | `s5-13/stage5_s5_13_supplement_implementation_handoff.md` |
| `stage5_s5_13_supplement_report_to_policy_chat.md` | `s5-13/stage5_s5_13_supplement_report_to_policy_chat.md` |
| `stage5_s5_14_report_to_policy_chat.md` | `s5-14/stage5_s5_14_report_to_policy_chat.md` |
| `stage5_s5_14_step_h4_parity_boundary_case_decision_request.md` | `s5-14/stage5_s5_14_step_h4_parity_boundary_case_decision_request.md` |
| `stage5_s5_14_structural_diagnostics_implementation_handoff.md` | `s5-14/stage5_s5_14_structural_diagnostics_implementation_handoff.md` |
| `stage5_s5_14_supplement2_report_to_policy_chat.md` | `s5-14/stage5_s5_14_supplement2_report_to_policy_chat.md` |
| `stage5_s5_14_supplement_implementation_handoff.md` | `s5-14/stage5_s5_14_supplement_implementation_handoff.md` |
| `stage5_s5_14_supplement_report_to_policy_chat.md` | `s5-14/stage5_s5_14_supplement_report_to_policy_chat.md` |
| `stage5_s5_15_report_to_policy_chat.md` | `s5-15/stage5_s5_15_report_to_policy_chat.md` |
| `stage5_s5_15_rotation_augmentation_implementation_handoff.md` | `s5-15/stage5_s5_15_rotation_augmentation_implementation_handoff.md` |
| `stage5_s5_16_implementation_handoff.md` | `s5-16/stage5_s5_16_implementation_handoff.md` |
| `stage5_s5_16_to_s5_20_policy_chat_transfer.md` | `s5-16/stage5_s5_16_to_s5_20_policy_chat_transfer.md` |

直下に残す全体管理・共通文書（6件）:

- `FILES.md`
- `TRAINING_IMPROVEMENT_PLAN.md`
- `data_construct.md`
- `stage5_edit_prompt.md`
- `stage5_pointnext_s_training_evaluation_report.md`
- `stage5_revision_management_record.md`

今回の範囲は移動と記録更新まで。文書本文の参照、`docs/README.md`・`FILES.md`等の索引は次工程で更新する。初回のverificationは初回終了時点の証拠として維持し、現在の配置は今回のverificationと統合対応表で確認する。

## 2026-09-22 参照パス・索引更新の実施記録

- 統合対応表を用い、初回の旧パスとStage5再配置前のパスを現在位置へ更新。文書間の相対リンクも修正した。
- `docs/README.md`・`docs/stage5/FILES.md`を更新し、`docs/stage5/README.md`に全ステップ文書の索引を追加した。
- `research/README.md`・`research/stage5/README.md`・各調査ディレクトリのREADME、およびコミット準備資料・保存バンドルの案内を追加した。
- S5-16の将来文書の配置例をステップ配下へ変更。未作成の文書は命名例であることを維持し、空文書は作成しない。
- Stage4文書に残っていた旧`docs/stage4/`参照も実在する配置へ修正した。
- 実験データ、保存バンドル、過去の移動ログは変更しない。旧Git状態・過去の配置を示す履歴記述は保持する。
- 更新差分・検証記録: `.tmp/relocation/20260922T094141034118Z-references/reference_updates.jsonl`、`.tmp/relocation/20260922T094141034118Z-references/verification.json`。修正前文書は同ディレクトリの`before/`に保存。
