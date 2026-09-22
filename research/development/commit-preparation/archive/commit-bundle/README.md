# 未コミット変更の分割コミット実行手順

このバンドルはユーザー環境で実行するためのものです。コンテナではコミット・プッシュを実行していません。

元の43候補を依存関係に合わせて37コミットへ整理しました。`series.txt` が確定した順序、
`patches/001.patch`〜`037.patch` が実際の差分です。各パッチは直前の段階へ適用します。
追加提案の環境整理を採用する場合は最後に38件目を作成します。

## 内容と確認済み範囲

- 基準HEAD: `5e72cc93d679acae2cb16b582f27bfe44886d2d0`
- 37件目の状態は、調査時の仮想環境以外118エントリを反映した作業ツリーとファイル内容・モードが一致します。
- 文書の旧版→新版移動、Dockerfileの移設→ツール追加、Stage 5の勾配蓄積→GroupNorm対応を実際に分割しています。
- v4/v5 importer、複数世代の可視化、GroupNorm decoder/CLI/checkpointは共通実装を保持した単位です。過去の不具合を再導入して段階数を増やしていません。
- 全37段階をGit管理外の一時ツリーへ順に `git apply` し、before/after SHA-256とモード、Python AST、shell構文、shell埋込みPython、ローカルimport先の存在を確認しました。
- 実際のGit indexへの適用、commit hook、署名、worktree作成、コミット、ブランチ更新はユーザー環境での実行事項です。コンテナで実行成功を確認したと偽ってはいません。
- GPU・実H5・CVAT接続・モデルの動作検証は未実施。依存環境付きの検証候補は `VALIDATION.md` に記載しています。
- 既存の文書参照パス問題などを勝手に修正していません。37件は現在の変更の再構成です。

## 0. 配置と変数

バンドル全体をユーザー環境のリポジトリの `.tmp/commit-bundle/` に配置してください。
Python 3.10以上、Git、bashのあるLinux/macOSを想定します。作業ツリーの実行ビットも照合します。
以下の `REPO` は実際のリポジトリ絶対パスへ置き換えてください。

```bash
REPO=/absolute/path/to/models
BUNDLE="$REPO/.tmp/commit-bundle"
WORKTREE="${REPO}-commit-split"
python3 "$BUNDLE/user_commit.py" preflight --repo "$REPO"
```

事前照合は読み取り専用です。HEAD、全対象ファイル、既存index、追加の未追跡ファイル、
submoduleの状態、パッチhashを確認します。差異があれば停止します。
**不一致を通すためにresetやforceを実行しないでください。差異を調査してバンドルを更新します。**
`.venvs/` と `.tmp/` はローカル生成物として事前照合の未追跡検出から除外します。
リモートへは接続しません。

## 1. 分割専用worktreeを準備

```bash
python3 "$BUNDLE/user_commit.py" prepare --repo "$REPO" --worktree "$WORKTREE"
```

`split/uncommitted-20260912` ブランチを基準HEADから作り、別のworktreeへ展開します。
既存の同名branch・worktreeは上書きしません。元の作業ファイル・indexは維持します。
`user-backup/` に元のbranch/HEAD、staged/unstaged差分を保存します。
未追跡のプロジェクトファイルはこのバンドルのパッチに含まれます。仮想環境はバックアップ対象外です。

prepare後に中断した場合はworktreeの状態を確認してください。既に作成済みならprepareを繰り返さず、stepへ進みます。

## 2. 各段階を確認してコミット

まず1件目:

```bash
python3 "$BUNDLE/user_commit.py" step --repo "$REPO" --worktree "$WORKTREE" --number 1
git -C "$WORKTREE" show --stat --oneline HEAD
```

以後は `--number 2`、`--number 3` のように順番に実行します。
各コミットは、パッチ適用・内容照合・変更ファイルの構文確認を経て作成します。
先にパッチを読む場合は `less "$BUNDLE/patches/002.patch"` のように確認できます。

37件を順次実行する場合:

```bash
for n in $(seq 1 37); do
  python3 "$BUNDLE/user_commit.py" step --repo "$REPO" --worktree "$WORKTREE" --number "$n" || break
done
```

既に完了した番号はコミット内容を検証してスキップします。commit hookや署名で停止した場合、
同じ番号を再実行すると、staged内容が期待どおりの場合だけコミットを再試行します。
予期しない変更があれば停止するため、作業内容を捨てずに確認してください。
コミット履歴の照合が通らなければfinalizeできません。

このループは動作検証を自動実行しません。必要な段階で停止し、`VALIDATION.md` の検証を実環境で実施してください。
各段階には新しい実装のimport先が存在しますが、外部データやCUDA extensionまでバンドルされるわけではありません。

## 3. 完成した履歴を元の作業branchへ反映

```bash
git -C "$WORKTREE" log --oneline --reverse 5e72cc93d679acae2cb16b582f27bfe44886d2d0..HEAD
python3 "$BUNDLE/user_commit.py" finalize --repo "$REPO" --worktree "$WORKTREE"
git -C "$REPO" status --short
```

finalizeは全37コミット、専用worktreeのクリーン状態、元のファイル/HEAD/branch/indexを再照合します。
その後 **`git reset --mixed <完成コミット>`** で元のbranchとindexを完成履歴へ揃えます。
`--hard` は使用せず、作業ファイルをcheckout・削除しません。これは通常のmergeではありません。
元のHEADとindex差分は `user-backup/`、完成履歴は専用branchに残ります。

この時点で既存の118件はすべてコミットされていますが、`.venvs/` の未追跡状態は残ります。

## 4. 仮想環境の整理（追加提案・38件目）

環境本体をコミットせずクリーンにするための追加パッチです。元の変更には存在しなかった以下を加えます。

- `.gitignore` の `.venvs/` 除外。
- 既存CVAT環境の12パッケージのバージョン記録 `requirements-cvat-review.txt`。
- Python 3.11.15での環境情報と再作成手順 `docs/development/cvat_environment.md`。

これは観測した環境の記録で、hash付きlockfileや移植可能性を検証済みの環境定義ではありません。
インストール・環境再作成はコミット手順では実行しません。既存 `.venvs/` はそのまま残します。

37件目をfinalizeした元のリポジトリで、追加案を採用する場合:

```bash
git -C "$REPO" diff --exit-code
git -C "$REPO" diff --cached --exit-code
git -C "$REPO" apply --check --index "$BUNDLE/optional-environment.patch"
git -C "$REPO" apply --index "$BUNDLE/optional-environment.patch"
git -C "$REPO" diff --cached --stat
git -C "$REPO" commit -m "chore: exclude local virtual environments and record CVAT dependencies"
git -C "$REPO" status --porcelain=v1 --untracked-files=all
```

各コマンドが成功したことを確認してから次へ進みます。最後の出力が空であればクリーンです。
追加案を採用しない場合は `.venvs/` の扱いを別途決める必要があり、この37件だけではクリーンになりません。

## 5. Push

pushは自動化していません。実環境でremoteとbranchを確認してから実行します。

```bash
git -C "$REPO" remote -v
git -C "$REPO" branch --show-current
```

確認した値を使い `git -C "$REPO" push -u <remote名> <branch名>` を実行します。
force pushはこの手順に含みません。リモート側が進んでいて拒否された場合は、その差分を確認してから対応します。
38件目を作った場合は元のbranchをpushします（専用split branchは37件目で止まっています）。

## 付属資料

- `manifest.json`: 各段階のbefore/after hash・mode・コミット名・許容index内容。
- `series.txt`: コミット順。
- `verification.json`: コンテナでのパッチ再生・構文確認結果。
- `verify_replay.py`: Git管理外ツリーで再生する読み取り/ローカルファイル検証。Git commitは作りません。
- `build_bundle.py`: 生成元。元のコンテナ配置とinventoryを前提とするため、ユーザー実行手順では再実行しません。
- `optional-environment.json`: 追加パッチのhash・変更範囲。
- `VALIDATION.md`: 実環境での動作検証候補と未検証事項。
