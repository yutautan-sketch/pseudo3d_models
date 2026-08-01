# Stage 4 Phase 7 実装確認・実データ受け入れ報告

作成日: 2026-07-22  
対象プロジェクト: `Stage2to4`  
対象機能: BBox非依存の `combined_v2` pseudo3D sampling  
判定: **Phase 7 合格（後続parameter sweepへ移行可能）**

## 1. 目的と確認範囲

Phase 1〜6で追加したStage 4のBBox非依存samplingについて、以下の一連の経路を確認した。

1. Base Grid Context
2. Local Percentile Evidence
3. Top-hat Evidence
4. source flagsによる重複なしの統合
5. 2D pixelから3D point cloudへの投影
6. H5 schemaおよびmetadata保存
7. annotation H5への伝播
8. collect処理およびStage 5 loaderへの伝播
9. raw/annotated PLY出力
10. annotation mask visualization
11. single/batch/shell/pipelineの整合性
12. legacy互換性

Phase 7の途中で、実データ可視化からlocal-percentileおよびtop-hat maskの孤立ノイズが確認された。そのため、global foregroundと同系統の後処理を両sourceへ独立して追加し、synthetic test、3件の実データ再export、source別可視化で反映を確認した。

本確認ではVOC BBoxおよび教師labelをsampling mask生成に使用していない。これらは候補選定と出力評価のみに使用した。

## 2. 実行環境とcommit

### 2.1 実行環境

実データ検証環境:

```text
Python executable: /home/kodaira/anaconda3/envs/dualtrack311/bin/python
numpy: 2.2.2
opencv: 4.13.0
scipy: 1.15.1
h5py: 3.13.0
Stage 4 dependency imports: OK
```

作業ディレクトリ:

```text
/mnt/data/3d_projects/models/Stage2to4
```

### 2.2 commit

```text
現在のHEAD                 : 387169a パイプラインのアップデート
Phase 6                    : 37288b1 Stage4更新Phase6
Phase 4〜5                 : 33a0bb9 Stage4更新Phase4-5
Phase 3                    : 722d734 Stage4編集Phase3
Phase 1〜2                 : 75fa665 Stage4更新Phase1-2
legacy比較に使用したbaseline: 65daf6d
```

Phase 7で追加・修正した確認コード、strict XML対応、Stage 5 loader対応、evidence cleanup等は、報告書作成時点ではHEAD以降の作業ツリー変更を含む。

## 3. Stage 4 samplingの最終確認構造

### 3.1 sampling mode

| mode | 処理 |
|---|---|
| `legacy` | 従来のglobal/foregroundまたはgrid/dense選択を維持 |
| `combined_v2` | global、local-percentile、top-hat、context gridをpixel単位でunion |

`combined_v2`は今回 `point_mode=foreground` で確認した。legacyは追加オプションを渡してもStep 1〜3およびevidence cleanupを有効にしない。

### 3.2 source flags

| bit | 値 | source |
|---:|---:|---|
| 0 | 1 | global evidence |
| 1 | 2 | local percentile |
| 2 | 4 | top-hat |
| 3 | 8 | context grid |

同一frame・同一pixelが複数sourceに選ばれた場合も3D点は1点だけ生成され、`source_flags` に該当bitがORされる。代表値 `source_type` の優先順位はglobal、local、top-hat、contextの順である。

### 3.3 暫定パラメータ

```text
sample_stride                    = 2
context_grid_stride              = 4
context_grid_phase               = origin
local_window_size                = 41
local_percentile                 = 80
local_min_contrast               = 4
tophat_kernel_size               = 21
tophat_percentile                = 85
tophat_min_response              = 2
tophat_morph_shape               = ellipse
evidence_open_ksize              = 3
evidence_close_ksize             = 5
evidence_morph_shape             = ellipse
evidence_min_component_area      = 100
```

これらはPhase 7の動作確認用暫定値であり、最終推奨値ではない。

### 3.4 evidence cleanup

local-percentile maskとtop-hat maskに対し、sourceごとに以下を独立して適用する。

```text
raw evidence mask
  -> morphological opening
  -> morphological closing
  -> connected component area filtering
  -> source_flagsへの統合
```

global maskとcontext gridには適用しない。cleanup前のraw maskとmorphology後maskはdebug mapとして取得できる。

新規CLI引数:

```text
--enable-evidence-cleanup / --no-enable-evidence-cleanup
--evidence_open_ksize
--evidence_close_ksize
--evidence_morph_shape rect|ellipse|cross
--evidence_min_component_area
```

`combined_v2`では既定で有効、`legacy`では無効である。

## 4. 実行した主要確認コマンド

### 4.1 dependency、compile、CLI、shell

```bash
/home/kodaira/anaconda3/envs/dualtrack311/bin/python -m py_compile <対象Python群>
/home/kodaira/anaconda3/envs/dualtrack311/bin/python pseudo3d/export/export_pseudo3d_point_cloud.py --help
/home/kodaira/anaconda3/envs/dualtrack311/bin/python pseudo3d/batch/export/batch_export_pseudo3d_point_cloud.py --help
bash -n <対象shell群>
```

### 4.2 synthetic sampling

```bash
/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  check_pseudo3d_sampling_synthetic.py
```

### 4.3 projection、H5、CLI

```bash
/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  check_pseudo3d_projection_h5.py
```

### 4.4 legacy互換性

```bash
/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  check_stage4_legacy_compatibility.py
```

### 4.5 実データ候補監査

```bash
/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  check_stage4_realdata_candidates.py
```

候補選定では、annotation contour positiveではなく「有効なBBoxがあり、frame上には点が存在するが、legacy点がBBox内に0点」の条件をprimaryとした。

### 4.6 combined_v2 batch export

```bash
/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  pseudo3d/batch/export/batch_export_pseudo3d_point_cloud.py \
  --h5_list /mnt/data/3d_projects/pseudo3d_dataset/stage4_rebuild/260722/candidates.txt \
  --output_root /mnt/data/3d_projects/pseudo3d_dataset/stage4_rebuild/260722/combined_v2_default_corr \
  --summary_csv /mnt/data/3d_projects/pseudo3d_dataset/stage4_rebuild/260722/combined_v2_default_corr/summary.csv \
  --video_suffix_to_strip _ts448_oym96_corr \
  --output_subdir_template '{video_name}' \
  --output_h5_filename_template '{video_name}_pointcloud_foreground_combined_v2.h5' \
  --output_ply_filename_template '{video_name}_pointcloud_foreground_combined_v2.ply' \
  --geometry_key corr \
  --preset percentile85_area100 \
  --point_mode foreground \
  --sample_stride 2 \
  --sampling_mode combined_v2 \
  --include_context_grid \
  --context_grid_stride 4 \
  --context_grid_phase origin \
  --enable_local_percentile \
  --local_window_size 41 \
  --local_percentile 80 \
  --local_min_contrast 4 \
  --enable_tophat \
  --tophat_kernel_size 21 \
  --tophat_percentile 85 \
  --tophat_min_response 2 \
  --tophat_morph_shape ellipse \
  --enable_evidence_cleanup \
  --evidence_open_ksize 3 \
  --evidence_close_ksize 5 \
  --evidence_morph_shape ellipse \
  --evidence_min_component_area 100 \
  --texture_percentile 85 \
  --texture_min_component_area 100 \
  --continue_on_error
```

cleanup導入前および導入後の双方で3件の処理が完了し、`processed=3`、`failed=0` を確認した。

### 4.7 real-data smoke check

```bash
/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  check_stage4_combined_v2_realdata_smoke.py \
  --candidates_list /mnt/data/3d_projects/pseudo3d_dataset/stage4_rebuild/260722/candidates.txt \
  --combined_root /mnt/data/3d_projects/pseudo3d_dataset/stage4_rebuild/260722/combined_v2_default_corr
```

### 4.8 schema、collect、Stage 5、PLY

```bash
/home/kodaira/anaconda3/envs/dualtrack311/bin/python check_stage4_annotation_schema_propagation.py
/home/kodaira/anaconda3/envs/dualtrack311/bin/python check_stage4_collect_schema_propagation.py
/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  ../Stage5/checks/real_h5/check_stage4_combined_v2_loader_compatibility.py
/home/kodaira/anaconda3/envs/dualtrack311/bin/python check_stage4_combined_v2_annotated_ply.py
```

### 4.9 batch/pipeline integration

```bash
/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  check_stage4_batch_pipeline_integration.py
```

### 4.10 cleanup後のsource別可視化更新

既存のstrict visualization manifestを用い、`--skip_existing`を付けずに再生成した。

```bash
/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  pseudo3d/batch/export/batch_export_annotation_mask_visualization.py \
  --h5_list /mnt/data/3d_projects/pseudo3d_dataset/stage4_rebuild/260722/combined_v2_default_corr_visualization_strict/visualization_manifest.csv \
  --texture_dir_name_template combined_v2_annotation_textures \
  --summary_csv /mnt/data/3d_projects/pseudo3d_dataset/stage4_rebuild/260722/combined_v2_default_corr_visualization_strict/visualization_summary.csv \
  --geometry_key corr \
  --xml_frame_id_source frame_index \
  --xml_frame_number_offsets 1 \
  --xml_annotation_dir_name annotations_renamed \
  --strict_xml_annotation_dir \
  --contour_preset percentile85_area100 \
  --contour_percentile 85 \
  --contour_min_component_area 100 \
  --contour_min_area 20 \
  --contour_min_alpha 1 \
  --contour_open_ksize 3 \
  --contour_close_ksize 5 \
  --draw_point_cloud_samples \
  --sample_point_radius 1 \
  --sample_point_alpha 0.75 \
  --global_sample_color 255,255,255 \
  --local_percentile_sample_color 64,200,255 \
  --tophat_sample_color 255,200,64 \
  --context_grid_sample_color 120,120,120 \
  --continue_on_error
```

## 5. 確認結果一覧

| 区分 | 結果 | 主な確認内容 |
|---|---|---|
| dependency import | 合格 | NumPy、OpenCV、SciPy、h5py |
| static/CLI/shell | 合格 | compile、help、shell syntax、引数伝播 |
| Base Grid synthetic | 合格 | phase、stride、境界、重複なしunion、validation |
| Local Percentile synthetic | 合格 | 一様画像抑制、各形状、debug map、determinism |
| Top-hat synthetic | 合格 | positive response percentile、各形状、morph shape、validation |
| evidence cleanup synthetic | 合格 | opening/closing、極小成分除去、source独立性、debug map |
| source merge/confidence | 合格 | bit OR、座標一意性、優先順位、統計、配列alignment |
| 3D projection/H5 | 合格 | prior/corr、独立再投影、schema、dtype、compression、CLI一致 |
| subsampling | 合格 | frame単位上限、seed再現性、全配列・統計alignment |
| legacy compatibility | 合格 | H5、PLY、annotation、old-schema fallback |
| combined_v2 real-data smoke | 合格 | 対象51/51 frameをrecovery、NaN/Inf・shape不整合なし |
| annotation propagation | 合格 | 新dataset/metadata、point order、strict XML path |
| collect propagation | 合格 | byte/hashを含む完全保持 |
| Stage 5 loader | 合格 | 3件、11 windows、未知datasetを含む読込 |
| annotated PLY | 合格 | vertex/property/bodyとH5の一致 |
| batch/pipeline | 合格 | single/batch同一、tag分離、skip、stop/continue |
| strict visualization | 合格 | BBox対応、source別表示、cleanup反映を目視確認 |

## 6. Syntheticおよびprojectionの詳細

最終的な `check_pseudo3d_sampling_synthetic.py` では以下の全テストが成功した。

```text
test_context_grid_counts
test_context_grid_boundaries_and_validation
test_local_percentile_evidence
test_local_percentile_debug_boundaries_and_validation
test_tophat_evidence
test_tophat_debug_shapes_and_validation
test_evidence_cleanup_and_source_isolation
test_source_flag_merge_and_confidence
test_source_type_merge_and_statistics
test_sampling_confidence_alignment_and_validation
test_build_frame_point_cloud_legacy_compatibility
test_build_frame_point_cloud_combined_v2
test_projection_schema_empty_and_grid_modes
test_subsampling_alignment_and_determinism
```

projection/H5 testでは以下も確認した。

* prior/corrected geometryの読込と投影
* prior/corrでsampling配列が同一で、3D座標差がtracking差と一致
* non-contiguous frame indexとempty frame
* single CLIと直接関数呼出しの完全一致
* H5 gzip compressionとmetadata round trip
* frame count、frame index、行列shape不正時の失敗
* CLI失敗時に不完全な出力H5を作らない

## 7. Legacy互換性

baseline commit `65daf6d` と現行 `--sampling_mode legacy` を比較した。

確認済みケース:

* original foreground
* threshold-alpha foreground
* foreground stride変更
* random subsampling
* grid
* dense
* prior geometry
* combined_v2用enable optionをlegacyへ渡した場合
* raw H5
* raw PLY
* annotation H5
* annotated PLY
* old-schema loader fallback

共通H5 datasetとmetadataは一致し、比較対象PLYはbyte-identicalであった。旧H5に新fieldが存在しない場合のみfallbackが動作し、現行H5のfieldを推定値として扱わないことも確認した。

## 8. 実データ候補と選定理由

以下の3件を採用した。

| video | 選定理由 |
|---|---|
| `20250627_092449_3280` | 有効BBox 19 frameすべてでlegacy BBox内点が0。frame自体には点が存在 |
| `20250624_132811_8530` | 有効BBox 16 frameすべてでlegacy BBox内点が0。frame自体には点が存在 |
| `20250610_155045_7420` | 有効BBox 42 frame中16 frameでlegacy BBox内点が0。他のBBox frameには点があり比較可能 |

これらは「annotation contour positiveが0」という条件ではなく、BBox geometryに対してlegacy sampling点が0という条件で選定した。したがって、Stage 4 samplingの失敗例として直接評価できる。

## 9. legacy対combined_v2の実データ基準値

以下は **evidence cleanup導入前** のsmoke test基準値である。cleanup後は同じ3件を再exportして可視化を更新したが、報告時点では同一形式の全数値比較を再採取していないため、前後の数値を混在させない。

| video | frames | legacy points | combined points | multiplier | target/recovered | BBox points min/median/max |
|---|---:|---:|---:|---:|---:|---:|
| `20250627_092449_3280` | 30 | 74,458 | 498,686 | 6.698 | 19/19 | 18 / 189 / 431 |
| `20250624_132811_8530` | 28 | 69,128 | 458,587 | 6.634 | 16/16 | 27 / 314 / 573 |
| `20250610_155045_7420` | 72 | 179,715 | 1,174,760 | 6.537 | 16/16 | 469 / 625 / 739 |

合計target frameは51、recovered frameは51、failureは0であった。

### 9.1 BBox内source別参考値

同じくcleanup導入前の集計値である。source件数はbitごとの件数であり、総点数との単純加算関係にはない。

| video | global | local | top-hat | context | context-only |
|---|---:|---:|---:|---:|---:|
| `20250627_092449_3280` | 0 | 2,753 | 3,421 | 495 | 272 |
| `20250624_132811_8530` | 0 | 3,898 | 4,689 | 552 | 261 |
| `20250610_155045_7420` | 0 | 7,747 | 8,677 | 1,286 | 707 |

3件の目視確認から、今回の暫定条件では以下が観察された。

* context gridおよびtop-hatの大腿骨拾得への寄与は限定的に見える
* local-percentileとtop-hatのraw結果には多数の孤立ノイズが含まれる
* evidence cleanup追加後のstrict source visualizationで、青色localおよび橙色top-hatに対する後処理の反映を確認した
* global点とcontext grid点はcleanup対象外として維持された

寄与度に関する観察は3件だけの参考所見であり、機能削除や最終パラメータ決定の根拠にはしない。

## 10. Annotation、schema、collect、Stage 5結果

### 10.1 annotation H5

cleanup導入前のannotated H5検証値:

| video | points | BBox rows | valid contours | annotation positive |
|---|---:|---:|---:|---:|
| `20250627_092449_3280` | 498,686 | 15 | 0 | 0 |
| `20250624_132811_8530` | 458,587 | 20 | 0 | 0 |
| `20250610_155045_7420` | 1,174,760 | 79 | 28 | 3,993 |

schema propagation check:

```text
videos_checked       : 3/3
point_cloud_datasets : 54
points_preserved     : 2,132,033
strict_xml_paths     : 114
annotation_positive  : 3,993
failures             : 0
```

最初の2件でannotation positiveが0なのは、strict annotation処理のvalid contourが0であるためであり、combined_v2のBBox内sampling点が0であることを意味しない。sampling rescueはraw BBox点のsmoke checkで別に確認した。

### 10.2 collect

```text
videos_checked      : 3/3
bytes_preserved     : 26,692,619
groups_preserved    : 15
datasets_preserved  : 141
points_preserved    : 2,132,033
annotation_positive : 3,993
failures            : 0
```

source/destination間でfile size、SHA-256、group、dataset、point数およびmetadataを照合した。

### 10.3 Stage 5 loader

```text
videos_checked      : 3/3
points_loaded       : 2,132,033
valid_points        : 1,124,003
annotation_positive : 3,993
overlap_windows     : 11
failures            : 0
```

Stage 5の既存feature/label読込を維持しつつ、必要な場合に `source_flags` と `sampling_confidence` を取得できる。今回これらを学習特徴へ強制追加していない。

## 11. PLYと可視化

annotated PLY check:

```text
videos_checked      : 3/3
full_vertices       : 2,132,033
annotation_vertices : 3,993
PLY bytes checked   : 152,582,048
failures            : 0
```

以下を確認した。

* H5 point数とPLY vertex数の一致
* PLY header、property、bodyの整合
* source flagsとsampling confidenceの出力
* annotation-only PLYとfull PLYの分離
* old-schema fallbackからのannotated PLY出力
* source別色表示
* `pixel_xy` とframe visualization上の描画位置
* strict XML適用後、意図しないBBox描画が解消
* evidence cleanup後の青色・橙色source表示を再生成して目視確認

## 12. Batch、shell、pipeline結果

```text
real videos compared    : 1
batch summaries checked : 3
sampling modes isolated : legacy, combined_v2
failure modes checked   : stop, continue
persistent files written: 0（checker自身）
```

確認内容:

* single/batch H5 datasetとmetadataの一致
* single/batch raw PLYのbyte identity
* sampling modeを含むfilename/tagによる出力分離
* `--skip_existing` がlegacyとcombined_v2を混同しない
* stop-on-errorとcontinue-on-error
* summary CSVの成功状態、統計、failure reason
* shellからsampling引数とfilenameへの伝播
* strict XML optionのpipeline伝播
* pipeline defaultがlegacyのままであること

## 13. Phase 7中に修正した問題

### 13.1 synthetic test拡張

Base Grid、Local Percentile、Top-hat、source merge、confidence、projection、subsamplingについて、境界、validation、dtype、determinismの確認を追加した。

### 13.2 local frame layout

実データ候補監査で `(1, 256, 256)` layoutが未対応だったため、サポート対象layoutとして正規化した。layout変換のfloat差については実データ精度に合わせてtoleranceを調整した。

### 13.3 zero-positive候補の定義

当初のcontour eligibilityベースの候補抽出では、Stage 4 samplingのzero-positiveを直接表していなかった。候補条件を「有効BBox内のlegacy pointが0、かつframe全体にはpointが存在」に修正した。

### 13.4 annotation XML directory

複数のannotation directory候補から意図しないXMLを拾う可能性があった。以下を追加・伝播した。

```text
--xml_annotation_dir_name annotations_renamed
--strict_xml_annotation_dir
```

pseudo3D H5を再構築し、strict visualizationでアノテーションが存在しない対象frameへBBoxが描かれないことを確認した。

### 13.5 old-schema fallback

旧H5に `source_flags` や `sampling_confidence` がない場合のfallbackを追加・検証し、推定の有無と理由をmetadataへ記録した。

### 13.6 Stage 5 loader

combined_v2の追加datasetを保持したH5を既存Stage 5 loaderで読み込めるようにし、既存feature/label semanticsを維持した。

### 13.7 evidence cleanup

local-percentileおよびtop-hatの孤立ノイズに対し、global foregroundと同系統のmorphologyおよび極小成分除去を後段化した。

追加確認:

```text
[OK] test_evidence_cleanup_and_source_isolation
combined_v2 batch export: processed=3, failed=0
strict source visualization: 更新完了、目視確認完了
```

## 14. 主な変更ファイル

### 14.1 samplingとpoint cloud

* `src/utils/pseudo3d_sampling.py`
  * `PointSamplingConfig`
  * Base Grid、Local Percentile、Top-hat
  * source flags、confidence、統計
  * evidence cleanup
* `pseudo3d/export/export_pseudo3d_point_cloud.py`
  * single CLI、投影、H5/PLY schema、metadata
* `pseudo3d/batch/export/batch_export_pseudo3d_point_cloud.py`
  * batch CLI、summary、設定伝播
* 関連single/batch/pipeline shell

### 14.2 annotationとvisualization

* `pseudo3d/annotation/annotate_pseudo3d_point_cloud.py`
* `pseudo3d/batch/annotation/batch_annotate_pseudo3d_point_cloud.py`
* `pseudo3d/export/export_annotation_mask_visualization.py`
* `pseudo3d/batch/export/batch_export_annotation_mask_visualization.py`
* 関連shell

### 14.3 Stage 5

* `../Stage5/stage5/utils/h5_io.py`
* `../Stage5/train_stage5.sh`
* `../Stage5/infer_stage5.sh`
* `../Stage5/checks/real_h5/check_stage4_combined_v2_loader_compatibility.py`

### 14.4 Phase 7 checker

* `check_pseudo3d_sampling_synthetic.py`
* `check_pseudo3d_projection_h5.py`
* `check_stage4_realdata_candidates.py`
* `check_stage4_legacy_compatibility.py`
* `check_stage4_combined_v2_realdata_smoke.py`
* `check_stage4_annotation_schema_propagation.py`
* `check_stage4_collect_schema_propagation.py`
* `check_stage4_combined_v2_annotated_ply.py`
* `check_stage4_batch_pipeline_integration.py`
* `export_stage4_combined_v2_smoke_visualizations.py`

## 15. H5 schema差分

### 15.1 point-level dataset

```text
source_flags         [K] uint16
sampling_confidence  [K] float32
source_type          [K] uint8
```

既存の `points`、`intensity`、`alpha`、`frame_index`、`frame_order`、`pixel_xy`、`confidence` 等は維持する。

### 15.2 frame-level dataset

```text
per_frame_counts
per_frame_global_counts
per_frame_local_percentile_counts
per_frame_tophat_counts
per_frame_context_grid_counts
per_frame_context_only_counts
per_frame_evidence_counts
per_frame_overlap_counts
```

### 15.3 metadata

* sampling modeと全sampling parameter
* evidence cleanup parameter
* source flag bit定義
* source別、context-only、evidence、overlap統計
* points/frame min、mean、median、max
* legacy point count multiplier
* old-schema fallbackの有無と理由
* strict XML pathおよびannotation設定

## 16. 合格・未実施・既知の制約

### 16.1 合格

* dependency、compile、import、CLI、shell
* syntheticと境界条件
* legacy H5/PLY/annotation互換性
* prior/corr 3D投影
* combined_v2の4 source、重複排除、confidence、統計
* 3件51 target frameのsampling recovery
* H5からannotation、collect、Stage 5、PLYへのschema伝播
* single/batch/pipeline整合性
* strict XML frame/BBox対応
* evidence cleanupのsynthetic、実データexport、可視化反映

### 16.2 Phase 7後も未実施

* 全データを対象にしたparameter sweep
* final sampling parameterの決定
* source confidenceのcalibration
* Stage 5モデル・lossへのsource情報の本格統合
* BBox Rescue、frame fallback、line/vesselness sampling
* cleanup導入前後の全動画・全source統計の再集計

### 16.3 回帰確認上の注記

annotation、collect、Stage 5、annotated PLY、最終batch integrationの詳細数値はevidence cleanup導入前のcombined_v2出力で確認した。cleanup後はsampling schemaを変更せずpoint selectionだけを変更し、以下を確認した。

* cleanup専用synthetic test
* Python/shell静的整合性
* 3件のraw combined_v2再export
* strict source visualizationの再生成と目視確認

正式なcommitまたは大規模生成の直前には、cleanup後のH5を用いてannotation以降のcheckerを一括再実行することが望ましい。

### 16.4 その他の既知事項

* 暫定source confidenceはvalidationされておらず、NaNや1超の値を直接設定すると伝播し得る
* 不正なtracking matrix shapeの一部はNumPy matrix multiplicationが最終的に拒否する
* context gridはpositiveを保証する機能ではなく、context pointを保持する機能である
* source別件数はbit件数なので、overlapがある場合は総点数へ単純加算できない
* pre-cleanupではpoint count multiplierが約6.5〜6.7であり、計算量と容量の評価が必要
* strict contour annotationが0でも、raw BBox内にsampling点が存在する場合がある

## 17. Phase 8以降への持ち越し

次段階では、今回の3件の所見だけで機能を削除せず、教師を評価にのみ使用するparameter sweepを実装する。

優先評価項目:

1. positive frame rescue率
2. zero-positive frame数
3. positive pixel recall
4. BBox coverageおよびBBox内point数
5. source別のBBox内point数とsource単独寄与
6. context-only、evidence、overlap件数
7. points/frameとlegacy multiplier
8. 動画単位のworst case
9. cleanup前後のノイズ量とpositive保持率
10. 処理時間、H5容量、Stage 5メモリ使用量

特に調査すべきパラメータ:

* context stride: 4、6、8、12
* context phase: origin、centered、dual
* local window、percentile、min contrast
* top-hat kernel、percentile、min response
* evidence open/close kernel、morph shape、min component area
* `max_points_per_frame`

3件の目視ではcontext gridとtop-hatの大腿骨拾得への寄与が低く見えたが、top-hatの有効性、context密度、cleanupによるpositive損失は全対象または代表性のあるデータ集合で評価してから判断する。

## 18. 最終判定

Phase 7で要求されたstatic、synthetic、legacy、projection、H5 schema、annotation、collect、Stage 5、PLY、visualization、batch/pipeline確認は完了した。実データ3件では、legacy BBox内点が0だった51 frameすべてでcombined_v2点を確認した。

追加したevidence cleanupについてもsynthetic test、3件の再export、strict source visualizationで動作を確認した。BBoxまたは教師labelへの本番sampling依存は導入されていない。

したがって、Stage 4 Phase 7を **合格** とし、次のparameter sweep設計・実装へ進める状態と判断する。
