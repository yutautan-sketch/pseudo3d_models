# .tmp の追加整理方針（2026-09-29）

状態: 11ファイルの移動・hash照合、参照パス・索引更新を完了。以下の移動元と手順は計画時の記録として保持。

## 根拠と対象

既存の[移動方針](tmp_relocation_plan.md)、[Stage5文書索引](../stage5/README.md)、[調査索引](../../research/stage5/README.md)に従い、管理文書はdocsのステップ別、調査成果物はresearchのステップ別に置く。

今回の対象は`.tmp/shared/`内のJSON6件、`.tmp/mk_legcase_json.py`、`.tmp/`直下のMarkdown4件の計11ファイル。過去の移動記録に含まれるMarkdownのバックアップは今回の対象ではない。

## 推奨する移動先

パスはリポジトリルート相対。ファイル名・内容は維持する。

| 移動元 | 移動先 |
| --- | --- |
| `.tmp/shared/audit.json` | `research/stage5/s5-17/shared/audit.json` |
| `.tmp/shared/coverage_best.json` | `research/stage5/s5-17/shared/coverage_best.json` |
| `.tmp/shared/coverage_last.json` | `research/stage5/s5-17/shared/coverage_last.json` |
| `.tmp/shared/run.json` | `research/stage5/s5-17/shared/run.json` |
| `.tmp/shared/supplement.json` | `research/stage5/s5-17/shared/supplement.json` |
| `.tmp/shared/supplement_registration.json` | `research/stage5/s5-17/shared/supplement_registration.json` |
| `.tmp/mk_legcase_json.py` | `research/stage5/s5-16/step0/reference_scripts/mk_legcase_json.py` |
| `.tmp/stage5_s5_16_step0_2b_confirmation_request.md` | `docs/stage5/s5-16/stage5_s5_16_step0_2b_confirmation_request.md` |
| `.tmp/stage5_s5_16_step0_acceptance_request.md` | `docs/stage5/s5-16/stage5_s5_16_step0_acceptance_request.md` |
| `.tmp/stage5_s5_16_step0_fl_premise_decision_request.md` | `docs/stage5/s5-16/stage5_s5_16_step0_fl_premise_decision_request.md` |
| `.tmp/stage5_s5_16_step0_stratification_spec_proposal.md` | `docs/stage5/s5-16/stage5_s5_16_step0_stratification_spec_proposal.md` |

### shared: S5-17の診断結果として一式保存

[総括報告書10.2](../stage5/s5-17/stage5_s5_17_summary_report.md)に6ファイルの内容と完全SHA-256がある。現物のJSONを読み取り、schemaを確認し、6件すべてのSHA-256が同表と一致することを照合した。

- `coverage_best.json` / `coverage_last.json`: S17-4のbest/last評価run登録。
- `audit.json`: S17-4のメタデータ・入力監査。
- `run.json`: S17-5の幾何表現・後処理比較の集計結果。
- `supplement_registration.json`: 補完集計に使うprivate記録の登録結果。
- `supplement.json`: pseudo-3D参考値、BBox整合性、複数instance、matching誤差の補完集計。

`shared/`を分割せず、`research/stage5/s5-17/shared/`へまとめる。実機の`${PRIVATE_OUT_DIR}/shared/`とは別の、開発コンテナに共有された写しである。実機の生成先、private記録、JSON内部の来歴、固定hashは変更しない。ここでの確認は内容・schema・hashの照合であり、privacy検査の再実施ではない。

### mk_legcase_json.py: Step 0の前提確認に使った参照コード

[Step 0報告書15章](../stage5/s5-16/stage5_step0_report_to_policy_chat.md)が、FL参照値の前提訂正の根拠としてこのスクリプトを挙げている。コードはXMLのstart/end・leg BBoxと画像から重心軌跡を作り、正規化座標の軌跡長`femur_traj_len`を出力する。臨床FLや現在のS5-17評価器として位置付けられたものではない。

したがって、`research/stage5/s5-16/step0/reference_scripts/`に原本を保存する。`Stage5/`の実装や`Stage2to4/scripts/data/`の運用ツールへ組み込む根拠はない。末尾には実環境の絶対パスとトップレベルの処理・JSON書き出しがあるため、調査ではimport・実行していない。移動時も内容修正・CLI化・パス書換えは行わない。将来運用ツール化する場合は別作業とする。

### Markdown4件: S5-16 Step 0の管理記録

4件はFL前提の判断依頼、層化仕様確定、工程2bの確認、完了受入の記録で、管理回答も含む。既存のStep 0依頼書・報告書と同じ`docs/stage5/s5-16/`に置き、統合・削除せず保持する。S5-17への影響があっても作成対象はStep 0であるため、S5-17側には移さない。過去時点の状態を現在の実施状況に書き換えない。

## 後続の参照・索引更新箇所

移動完了後、ログに従って以下を更新する。

- S5-17総括報告書10.2、S5-17報告書の`.tmp/shared/`・`.tmp/shared/run.json`等の開発コンテナ側参照。実機の生成先やコマンド中の`${PRIVATE_OUT_DIR}/shared/`はそのまま保持する。
- Step 0報告書のスクリプト・判断書・仕様書・工程2b確認書参照、全体管理記録D-039の層化仕様書参照。
- 移動する4文書内の相互リンクとStep 0報告書へのリンク。現状の`../docs/stage5/s5-16/...`は移動後には不適切なので、同じディレクトリのファイル名に直す。`FILES.md`等の親ディレクトリ参照も確認する。
- `docs/stage5/README.md`・`docs/stage5/FILES.md`に4文書を追加。`research/stage5/README.md`にS5-16・S5-17を追加し、各ステップのREADMEから成果物・関連報告書を案内する。
- `research/stage5/s5-16/step0/reference_scripts/README.md`を追加し、スクリプトの用途・由来・原本保存であることを記す。

## 実行手順と対象外

1. 実行時に再棚卸しし、移動先衝突・内容変更を確認する（今回の調査時点では全11件の移動先に同名ファイルなし）。
2. `.tmp/relocation/<UTC日時>/plan.json`にファイル単位の旧新パス・サイズ・SHA-256を記録する。
3. 内容を変更せず移動し、`moves.jsonl`に開始・完了・失敗を追記する。移動後のhash一致・件数・旧パス不在を`verification.json`に記録する。
4. 全件照合後に参照・索引を更新し、`reference_updates.jsonl`へ変更を記録する。過去の移動ログ・旧パス記録は改変しない。

`.tmp/log.txt`、`.tmp/reference-update-tools/`、`.tmp/relocation/`は残す。既存の移動ログ・文書バックアップを再分配しない。

調査開始時点で`docs/stage5/FILES.md`、`README.md`、S5-17管理書・報告書、評価レポート、全体管理記録に未コミット変更があり、S5-17総括報告書は未追跡だった。これらの内容は保持する。移動や索引更新とGit登録・コミットは別工程とする。

## 調査時点の対象一覧

この一覧は移動実績ではない。実行時のログは別途作成する。

| 移動元 | bytes | SHA-256 |
| --- | ---: | --- |
| `.tmp/shared/audit.json` | 1040 | `ddc38a83f13a4b5ca4520a8a5523d9cfacd3f90957ecdba12faa571d7359f62e` |
| `.tmp/shared/coverage_best.json` | 544 | `b36266f769c869518df24f12326ccacdc3a57fe20b9fe8d192c4c0aea66f902f` |
| `.tmp/shared/coverage_last.json` | 544 | `9846c411cebf99b6b03e22005d48e292344c8a0f5fad09e6ebd02ff2fc224540` |
| `.tmp/shared/run.json` | 155205 | `3348f49823967676d2ded96504725b91e4dda8659def089a1094ecbfcfd9035d` |
| `.tmp/shared/supplement.json` | 45656 | `0141d323f497245985709b0cadc3e72dcf241035bfb0ea4d3f4760d8db60c3f4` |
| `.tmp/shared/supplement_registration.json` | 929 | `30e40fdac9d095ab1efb579d0ed80081d8605f5357625e120cf53249f7c63af9` |
| `.tmp/mk_legcase_json.py` | 11532 | `70c89e00875b8c4edb9ce069a43ce82495333a12e50d6381f18269b2caf2fc72` |
| `.tmp/stage5_s5_16_step0_2b_confirmation_request.md` | 8068 | `1ee7dfc643f5cf38cd92bbe9d54dc127cc5f1caeaadfbe807386315c06212e33` |
| `.tmp/stage5_s5_16_step0_acceptance_request.md` | 10926 | `793b9e96e45ee643a6b82b8c87315a7ca8e31c5b11c72487d54f2af56bd72797` |
| `.tmp/stage5_s5_16_step0_fl_premise_decision_request.md` | 38538 | `043a727331b4ed2a9f8d34ac1178147b90345611feabd8451df73864df871643` |
| `.tmp/stage5_s5_16_step0_stratification_spec_proposal.md` | 37817 | `a3d92af091989ab63098e4f634bedeb24aa19da2572474bedd116ca07e6b0908` |

合計: 11ファイル、310,799 bytes。

## 実施記録

- 移動実績: `.tmp/relocation/20260929T080704118710Z/moves.jsonl`、同ディレクトリの`verification.json`。11件・310,799 bytesを照合済み。
- 参照・索引更新: `.tmp/relocation/20260929T080859240177Z-references/reference_updates.jsonl`、同ディレクトリの`verification.json`。移動済み4文書の相対リンクを修正し、文書索引・調査索引とS5-16/S5-17の案内を追加。
- 実機側の生成先・固定hash・共有JSON・原本スクリプトは変更しない。移動ログは当時の記録のまま保持。
