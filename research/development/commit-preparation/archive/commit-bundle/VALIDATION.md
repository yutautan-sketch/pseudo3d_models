# 検証記録と実環境での確認候補

## コンテナで確認済み

全37パッチの順次適用、各段階の変更ファイルのSHA-256・mode、累積の変更対象PythonのAST、
shellの `bash -n`、`PY` heredocのPython AST、静的に判別できるローカルimport先の存在。
最終段階ではPython 55本・shell 31本・Python heredoc 12個を確認した。
追加の環境整理パッチも37件目へ適用でき、変更後hashが一致することを確認した。
実行補助スクリプトの停止条件6件（内容不一致、削除予定パスの再出現、パッチ改変、HEAD不一致、
未記録index内容、専用branch以外での実行）をモック/一時ファイルで検査し、すべて成功した。
実ワークスペースに対する読み取り専用preflightも成功した。Git書き込みサブコマンドは実行していない。

## 未確認

- 全importの実行、関数・クラス・数値挙動の正当性、動的importの完全解決。
- Git worktree/index/commit/signing/hookの実環境動作（実際のGit書き込みはユーザー実行）。
- GPU、OpenPoints CUDA extension、実H5・学習重み、CVAT server/API。
- 環境snapshotの再インストール。
- 過去のv4生成物とバイト単位で同一の再生成。v4/v5共通readerには後続改修が含まれる。

## 動作検証の入口

以下は自動実行されない検証候補。`PYTHON` はnumpy、torch、h5py、OpenCV、PyYAML等が必要に応じて
導入済みの既存環境のPython絶対パスを指定する。CVAT用venvだけではStage 4/5全体を検査できない。
実行した番号・コマンド・結果・環境を管理文書へ追記する。

| 段階 | worktree内の検査入口 | 条件 |
| --- | --- | --- |
| 005 | `Stage2to4/checks/stage4/check_stage4_cvat_segmentation_mask_export.py` | 合成mask、CVAT接続不要 |
| 007 | `Stage2to4/checks/stage4/check_stage4_contour_auto_refine.py` | 合成fixture、Stage 4依存環境 |
| 011 | `Stage2to4/checks/stage4/check_stage4_cvat_manual_roundtrip.py` | 合成roundtrip・textfree |
| 012 | `Stage2to4/checks/stage4/check_stage4_contour_teacher_phase5_apply.py` | 合成H5への適用 |
| 014 | `Stage2to4/checks/stage4/check_stage4_sampling_sweep_manifest.py` | manifest・除外契約 |
| 015 | `Stage2to4/checks/stage4/check_stage4_phase5_fullvideo_cvat_tasks.py` | fake client、実Taskを作らない |
| 016 | `Stage2to4/checks/stage4/check_stage4_phase5_cvat_task_snapshots.py` | fake client |
| 017 | `Stage2to4/checks/stage4/check_stage4_phase5_fullvideo_final_import.py` | 合成v4/v5契約 |
| 019 | `Stage2to4/checks/stage4/check_stage4_point_label_visualization.py` | 合成H5・可視化 |
| 020 | `Stage2to4/checks/stage4/check_stage4_cvat_authoritative_preflight.py` | projection契約 |
| 022〜026 | `Stage2to4/checks/stage4/check_stage4_deleted_xml_*.py` | audit/manifest/apply/batch/visualizationを各導入段階で実行 |
| 028〜029 | `Stage5/checks/real_h5/check_stage5_batch_integrity.py`、`check_stage5_padding_parity.py` | 実H5、後者はモデル・checkpoint・GPUも必要 |
| 031 | `Stage5/checks/dummy/check_dummy_training.py --device cpu` | 合成データ、torch/H5環境 |
| 033 | `Stage5/checks/real_h5/check_stage5_overlap_aggregation.py --self_test` | GPU不要。ただしtorch/h5py等のimportあり |
| 034 | `Stage5/checks/real_h5/check_stage5_batchnorm_mode_parity.py --self_test` | CPU、torch/H5依存 |
| 035 | `Stage5/checks/real_h5/check_stage5_batchnorm_recalibration.py --self_test` | CPU、torch/H5依存 |
| 036 | `Stage5/checks/dummy/check_dummy_pointnext_s_groupnorm.py --self_test` | CPU、依存環境 |
| 036 | `Stage5/checks/transfer/check_stage5_batchnorm_to_groupnorm_transfer.py --self_test` | CPU、依存環境 |
| 036 | GroupNorm full check・dummy PointNeXt training・重み転移 | GPU/CUDA extension、必要に応じcheckpoint |

Stage 4合成検査の例（該当段階のコミット後）:

```bash
PYTHON=/absolute/path/to/existing/environment/bin/python
cd "$WORKTREE/Stage2to4"
"$PYTHON" checks/stage4/check_stage4_cvat_segmentation_mask_export.py
```

Stage 5のCPU self-test例（033以降）:

```bash
cd "$WORKTREE/Stage5"
"$PYTHON" checks/real_h5/check_stage5_overlap_aggregation.py --self_test
```

実H5のpreflight/受入検査は各CLIの `--help` と既存文書で入力を確認する。
CVAT Task作成shell、teacher再生成shell、学習shellをコミット確認のために無条件で起動しない。
学習shellの既定EPOCHS=200と、文書上の5 epoch診断を区別する。
