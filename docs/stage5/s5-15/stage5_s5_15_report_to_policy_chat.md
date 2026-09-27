# Stage 5 S5-15: 回転augmentation短期比較・R0 50 epoch評価 終了報告書

作成日: 2026-09-15
最終更新日: 2026-09-22（9章の文書同期状況のみ更新。終了判断は09-20）
作成元: Stage 5実装チャット
状態: **S5-15終了。R0/R1各5 epoch比較、R0新規50 epoch学習・固定21動画評価・可視化・
追加解析の結果を記録。現設定の100〜200 epoch延長・production採用には進まない。
改善方針の考察と未承認の実験候補をS5-16へ移管する。**

管理チャット最新返信（2026-09-20）: **8.14節**を現行判断とする。
8.1〜8.13はログとして改変せず保持し、強い断定や途中の仮説は8.14の訂正を併せて読む。
3〜7章は当初の条件・設計・検証計画を参照用に残したもので、未来の実行指示ではない。
終了はproduction候補確定まで成功したことを意味しない。未実施工程・未検証項目は未実施のまま残す。

本書は`docs/stage5/s5-15/stage5_s5_15_rotation_augmentation_implementation_handoff.md`（実装依頼、正本）と
`docs/stage5/stage5_revision_management_record.md` 6章「S5-15 長期学習とproduction候補確定」
（P1〜P6、Decision record D-034）を受けて作成し、実装・監査・実行・判断を蓄積した終了記録である。依頼書9章が
指定する報告ファイル名`stage5_s5_15_report_to_policy_chat.md`と一致させてある。

## 1. 経緯

S5-14 core・補足1〜3はユーザー判断により完了・コミット済み（D-034）。補足1〜3では分母補正・
train sanity自己除外を反映したうえで、平行移動には低感度・回転には明確な感度があることを
validation単独でも確認したが、座標暗記や不具合の確定、augmentation採用の判断はしていない。
＋15度推論が有利だった事実だけを根拠に片方向回転や推論時回転は採用しない。D-034はこれを受けて
追加のGPU座標診断を終了し、次候補として「回転augmentation単独5 epoch比較」を挙げた。

管理記録6章は、この比較をP1（固定条件・manifest監査）→P2（training-only回転augmentationの実装・
CPU test・限定GPU preflight）→P3（R0/R1 5 epoch比較・固定21動画評価・事前選定基準）の順に段階承認で
進めた。その後の段階承認により10 epoch pilotを省略し、R0を初期重みから50 epoch学習・評価した。
当初計画と実施内容には保存仕様逸脱等の差があり、8章に履歴を残した。100〜200 epoch・production
確定には進まず、この後の介入選択をS5-16で改めて扱う。

## 2. 目的

学習時のランダムZ軸回転augmentation（training-onlyで、評価・推論には一切適用しない）が、W-A固定
条件（teacher v7、GroupNorm8、`bbox_noncontour_ignore`、固定class weight、window16/stride8）のもとで
5 epochという短期間でvalidation・train sanityの指標を悪化させずに改善方向へ働くかを、単一seedの
暫定比較として確認した。さらにR0を50 epoch学習し、期間延長で汎化性能が改善するかを確認した。
augmentationの長期効果や原因機序の確定は本段階の成果に含めない。

### 2.1 終了時の結果要約（2026-09-20）

| validation pooled指標 | P3 R0 epoch5 | R0長期best epoch6 | R0長期last epoch50 |
| --- | ---: | ---: | ---: |
| recall | 45.08% | 38.95% | 6.97% |
| precision | 3.66% | 5.27% | 3.90% |
| F1 | 6.77% | 9.28% | 5.00% |
| IoU | 3.51% | 4.86% | 2.57% |
| TP0動画数 | 1/18 | 1/18 | 10/18 |

P3のR1はFPを減らしたがrecall低下と動画別F1改善/悪化9/9が残り、R0を維持した。
50 epochではtrain sanity recall 97.36%に対してvalidation 6.97%となり、強い過学習・汎化不足と
整合する。単一seed・調整に使用済みvalidation18動画での結果であり、記憶の機序や全般的な
汎化限界を証明したわけではない。最良領域はこのrunでは概ねepoch6〜10で、epoch50ではない。

画像可視化では、足・頭蓋骨・腹部の輪郭、細長いアーティファクト等のFPが観察された。
これはユーザーの目視所見であり、モデルが解剖構造をどう表現しているかの確定診断ではない。
「フレーム内で相対的に明るい点を拾う」という初期仮説は、その後の解析で一般化できないと分かった。
高輝度点で位置と予測の関連が強まる傾向は両checkpointで残ったが、因果効果とは呼ばない。

2領域が現れる動画群の低性能は探索的な関連であり、フレーム内の領域数を失敗原因とする根拠は
得られていない。輝度・位置の標準化後にも差は残るが、32.5%という説明割合は特定のbin/調整軸に
基づく暫定値で、説明可能量の上限ではない。詳細な修正と提案の扱いは8.14節を参照する。

## 3. P1: 固定条件（依頼書4章 / 管理記録6章P1表）

### 3.1 最新確定版: P1固定条件の確定事項（最終更新 2026-09-16、8.6節の監査結果と8.7節の判断を反映）

**P1監査（8.6節）とその判断（8.7節）を受けた確定事項:**

- 初期checkpointは実機で特定済み。SHA-256 `55ec6e6bcb39d58f398719b33826e80715a94bc6e7670d5b88623cd7c668438b`
  （`stage5_pointnext_s_s3dis_partial_init_groupnorm.pt`、3,199,662 bytes）。両armともこのファイルから開始する。
- splitはW-A保存済みリストを固定リストモード（5.3.1節）で渡す。content SHA-256は
  train `582579833f345b77d111f9f02a0606a7423d4d5161dd0cb6436c994fcf353ab0` /
  val `0c251380e40f0def3a76bdedd74572ac0ba22e799bf0287c8fee3f14ca59e83f`。
- **R0は旧W-Aを再利用せず新規学習した（8.7.2）。** 要求は`SAVE_EVERY=1`だったが、P3実行時は
  bashの上書きにより実効値10となり、中間重みは両armとも未保存。epoch5主比較は成立し、
  欠落を理由とした再学習は行わなかった。後続50 epoch runでは修正・保存確認を実施した。

### 3.2 固定条件表（依頼書4章 / 管理記録6章P1表、初版 2026-09-15）

| 項目 | 条件 |
| --- | --- |
| teacher | v7 `bboxrank_v7_cvat_authoritative_crop_quality_v1`、180 H5 |
| split | W-A保存済みtrain162 / validation18 file list。再分割禁止 |
| 評価対象 | 固定train sanity3動画＋全validation18動画 |
| モデル | PointNeXt-S、GroupNorm8 groups、W-Aと同一width/radius/nsample等 |
| 初期重み | W-Aの学習開始に使ったGroupNorm転移初期checkpoint（両armでhash一致） |
| features | intensity,confidence |
| label policy | bbox_noncontour_ignore |
| loss / weight | CE、smoothing0、固定weight [0.05963856, 1.94036150]（auto再計算しない） |
| optimizer | AdamW、lr1e-3、weight_decay1e-4、grad_clip_norm10 |
| window | size16 / stride8 / tailあり |
| batch | physical1 / accumulation8、paddingなし |
| sampling | overlapの実点を維持。random point removalなし |
| 評価 | augmentationなし、eval mode、mean probability、既存2クラスargmax・同値background |
| seed | W-A実configと同一のseed。augmentation用乱数は独立 |
| 保存 | 各epoch checkpoint、best/last、config、metrics、file lists、manifest |

W-Aのepoch5学習済みcheckpointからは開始しない。R0/R1とも同じ「学習開始前の初期checkpoint」から
開始し、hashを照合する。

## 4. 実装前に読んだ既存コード（依頼書5.1節の指示に基づく監査）

### 4.0 P1ローカル監査（8.2回答1を受けて、ローカル資料の範囲で確認できたこと）

回答1で示された参照先`/mnt/data/3d_projects/stage5_runs/260914/pointnext_s_EX260914_260711_
w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad`について、
本環境からアクセス可能な2つのローカル資料を確認した。**実機での存在・hash確認は未実施であり、
以下はいずれも過去に共有された記録の再確認にとどまる。** この未実施表記はP1ローカル監査時点の
範囲を示すものであり、後の実機監査結果は8.6節を参照する。

1. `docs/stage5/s5-13/stage5_s5_13_report_to_policy_chat.md`「Step F4残り: W-A候補(EX260914)の個別確認
   （完了、2026-09-14）」: このrun dirのconfig.jsonを直接確認した記録として、
   `checkpoint = stage5_pointnext_s_s3dis_partial_init_groupnorm.pt`（basenameのみ、絶対パス・
   hashの記載はなし）、`pointnext_norm=groupnorm`（8 groups）、`label_policy=bbox_noncontour_ignore`、
   `seed=42`、`window=16/8`、`batch=1`、`grad_accum=8`、`lr=1e-3`、`weight_decay=1e-4`、
   `dropout=0.0`、`class_weight_info.resolved=[0.05963856, 1.9403615]`が記載されている。
2. `research/stage5/s5-13/s5_13_wa_share_metrics/training_history/training_config_anonymized.json`
   （同じEX260914 runの匿名化済みconfig、`checkpoint`/`train_dir`/`output_dir`は`REDACTED_PATH`）:
   `class_weight_info.mode = "manual"`、`requested = "0.05963856,1.94036150"`、
   `resolved = [0.05963856, 1.9403615]`。`pointnext_norm=groupnorm`、`pointnext_norm_groups=8`、
   `label_policy=bbox_noncontour_ignore`、`window_mode=overlap`・`window_size_frames=16`・
   `window_stride_frames=8`・`include_tail_window=true`、`seed=42`、`num_train_files=162`・
   `num_val_files=18`、`batch_size=1`・`gradient_accumulation_steps=8`、`lr=0.001`・
   `weight_decay=0.0001`・`grad_clip_norm=10.0`が確認できた。依頼書4章の期待値と一致する。

**回答1で指摘された「auto指定と手動固定指定を文字列だけで不一致としない」という点がまさに
該当する事例がここにある。** run dir名には`auto_weight`という文字列が含まれるが、実際に記録
されている`class_weight_info.mode`は`"manual"`であり、`requested`値もW-Aの固定weightそのもの
（`[0.05963856, 1.94036150]`）である。したがってrun dir名の`auto_weight`という文字列だけをもって
「W-Aはauto class weightで学習された」と判断しない。ディレクトリ名の由来（命名生成時点の
`train_stage5.sh`の挙動が現行版と異なっていた可能性、あるいは`EXPERIMENT_NAME`が手動指定された
可能性）は未確認であり、実機でのconfig.json直接確認時に矛盾がないか改めて見る。

**ローカル資料だけでは確認できないこと（実機監査が必要）:**

- 初期checkpointの実パス・SHA-256 hash（S5-13記録はbasenameのみで、hashは未記録）。
- 上記run dirが現在も実機に存在するかどうか。
- `train_files.txt`/`val_files.txt`の内容が、S5-15で使う保存済みfile listと完全一致するか。
- `checkpoint`/`train_dir`/`output_dir`の匿名化前の実際の値。

既存`checks/real_h5/check_stage5_class_weight_ablation.py`（S5-13、W-A/W-B比較で使用）は
「initialization-checkpoint path+SHA-256 match」を含むconfig parity検証を既に実装している
（`docs/stage5/FILES.md`記載）。回答3で了承された新規`check_stage5_rotation_augmentation_ablation.*`
は、このcheckerの検証項目（file list行一致、resolved weight一致、history.jsonのepoch到達・finite値、
初期checkpoint hash一致）をR0/R1にも適用し、加えてEX260914（W-A）とR0のhash一致検証にも使える設計
にする。新規に一から監査ロジックを組むのではなく、この既存checkerの検証パターンを踏襲する。

### 4.1 `train_stage5.sh`の既定値とW-Aとの乖離リスク

- `INIT_CHECKPOINT`の既定値（149行目）は**BatchNorm用**の
  `stage5_pointnext_s_s3dis_partial_init.pt`であり、`POINTNEXT_NORM`の既定値（162行目）も
  `batchnorm`である。GroupNorm転移initを使うW-A相当の学習には、`POINTNEXT_NORM=groupnorm`と
  `INIT_CHECKPOINT=.../stage5_pointnext_s_s3dis_partial_init_groupnorm.pt`（S5-11
  `check_stage5_batchnorm_to_groupnorm_transfer.sh`の生成物）を**両arm・明示的に**指定する必要がある。
  これを指定し忘れるとR0/R1ともBatchNormから学習してしまい、W-Aと別条件になる。
- `CLASS_WEIGHT`の既定値（90行目）は`auto`であり、固定weight`[0.05963856, 1.94036150]`を使うには
  `CLASS_WEIGHT`を明示的に指定する必要がある（auto再計算を防ぐため）。
- `SEED=42`（135行目）、`FEATURES="intensity,confidence"`（126行目）、`LABEL_POLICY`既定値
  `bbox_noncontour_ignore`（168行目）はW-Aと一致する既定値だが、いずれもW-A実configとの突合で
  確認する（依頼書が指定するmanifest監査の対象）。
- 依頼書8章の指示どおり、`train_stage5.sh`の一般利用向け既定値は変更しない。R0/R1はこのスクリプトを
  env var override付きで呼び出す形にする（既存W-B/W-C class weight ablationと同じ運用パターン、
  `checks/real_h5/check_stage5_class_weight_ablation.*`が事後にconfig/checkpoint hash parityを
  検証する構成を踏襲する）。**ただし呼び出し元は手入力のenv overrideではなく、内部定数を持つ
  実験用bashとする（8.3-3、5.4節）。また固定リストモードの追加だけは`train_stage5.sh`本体への
  限定変更が必要で、8.7.1で了承済み（5.3.1節）。**

### 4.2 augmentation挿入位置

- `Stage5/stage5/datasets/pseudo3d_pointcloud_dataset.py`の`Pseudo3DPointCloudDataset.__getitem__`
  （251〜304行目）: `points = data["points"]` → `normalize_xyz(points)`（279〜280行目）の直後、
  `selected_indices`によるwindow切り出し（282行目）より前が「全動画正規化後・window抽出前」という
  依頼書5.1節の契約に一致する挿入点。`normalize_xyz()`は動画ごとの重心中心化・最大norm scalingで
  あり（`stage5/utils/feature_normalization.py:27-39`）、centered後の重心は原点(0,0,0)に一致するため、
  「動画重心中心の回転」は「正規化後の原点中心の回転」と同じ操作になる（S5-14補足2の座標診断と
  同一の事実）。
- window所属（`generate_frame_order_windows`）は`frame_order`のみに依存し`points`の値に依存しない
  （`stage5/utils/frame_windows.py`）。回転を挿入してもwindow構成・vote対応は変わらない。
- `data["points"]`自体はin-placeで書き換えず、`points = data["points"].astype(np.float32)`で
  新しい配列を作ってから変換するため、`cache_data=True`時のキャッシュ済み原点群への回転累積は
  発生しない（現状の`normalize_xyz`適用と同じ安全性）。

### 4.3 epoch/workerへの伝達（8.3-1を反映した修正版）

初稿では「毎epoch非persistent workerがforkされ直す」という現状の挙動（`persistent_workers`
未指定＝既定`False`、`train_loader`はループ外で1回だけ構築）に依存し、`set_epoch()`を
`run_one_epoch()`呼び出し直前に呼べばfork時点で伝わる、という設計にしていた。8.3-1の指摘により、
これは`persistent_workers=True`にした場合に壊れる設計（forkは初回iterでしか起きず、以後は
worker側の古いコピーのままメインプロセスの属性更新が届かない）であり、依頼書が要求する
persistent worker込みの再現性検証を満たさないため、次のように修正する。

- `Pseudo3DPointCloudDataset.__init__`で`multiprocessing.Value("i", 0)`（共有メモリ、ロック付き）を
  `self._epoch_shared`として保持する。`multiprocessing.Value`はDataLoader workerへ渡される際に
  真の共有メモリを指すため、`persistent_workers=True`で一度forkされたworkerでも、メインプロセス側の
  `set_epoch(epoch)`（`self._epoch_shared.value = int(epoch)`を書き込むだけ）が**後から**worker側へ
  伝わる。`persistent_workers=False`（既存の既定値）でも同じ経路で問題なく動作する。
  `__getitem__`は毎回`int(self._epoch_shared.value)`を読み直すため、fork/spawnのタイミングにも
  worker再利用の有無にも依存しない。
- `make_loader()`の既定値`persistent_workers`未指定（=`False`）はproduction既定として維持するが、
  実験側で`persistent_workers=True`を試す場合でも同じ`set_epoch()`呼び出しだけで正しく動作する
  設計にする（P2実装時、CPU syntheticで両方を検証する）。
- 起動環境の実際のmultiprocessing start method（Linuxなら通常`fork`だが仮定せず）をP1報告・
  manifestへ記録する。`torch.multiprocessing.get_start_method()`または標準`multiprocessing`の
  相当関数で確認する。
- 角度自体はPythonの`hash()`（プロセスごとにrandomize設定されうる）やworkerの取得順に依存させず、
  `hashlib`ベースの決定的関数`(base_seed, epoch, stable_video_id) -> np.random.default_rng(...).uniform(...)`
  で生成する。`stable_video_id`の定義は8.3-2を受けて4.3.1節で別途修正する（絶対パスは使わない）。
  これにより同一動画・同一epochであればworker数・取得順に関係なく同じ角度になる。

### 4.3.1 stable_video_idの定義（8.3-2を反映）

初稿の「H5ファイルパス文字列」はmount prefixや絶対/相対指定の変更で値が変わってしまうため、
次のように修正する。

- `stable_video_id = Path(h5_path).name`（ディレクトリ部分を含まないファイル名。拡張子込み）を
  角度生成のIDとして使う。同じデータを異なるmount prefix・絶対/相対パスで読んでも、ファイル名
  自体は不変なので同じIDになる。
- Dataset構築時（`_build_sample_index()`と同じタイミング）に、`self.h5_paths`のbasenameの重複を
  検査し、重複があれば`ValueError`で停止する（同名ファイルが2箇所から重複して指定されている場合、
  異なる物理動画に同じ角度が割り当たってしまう不具合を未然に防ぐ）。file listに存在しない
  「欠損」ケースはID生成の対象外（h5_pathsに含まれる動画は必ずbasenameを持つ）。
- ファイル名は`[0-9]{8}_[0-9]{6}_[0-9]+`形式のtimestamp状動画IDを含みうる機微情報のため、
  この`stable_video_id`自体は実行時のprivate manifest（実機ローカル）にのみ記録し、方針管理チャット
  へ共有する集計・報告には含めない。共有時は既存の`video_alias`匿名化規約を使う。

### 4.4 augmentation用RNGの分離

- 現行`Pseudo3DPointCloudDataset._rng_for_index()`は`window_mode="none"`のsampling専用であり、
  S5-15が使う`window_mode="overlap"`では`selected_indices`は`frame_order`から決定的に得られるため
  乱数を一切消費しない。DataLoaderの`shuffle=True`はPyTorchの`torch.Generator`（Dataset非依存）を
  使う。したがって新設するaugmentation用RNG（`hashlib`＋`np.random.default_rng`）はshuffle・
  point sampling・model初期化のいずれの乱数列とも独立であり、`augmentation="none"`時は一切呼ばれない
  （無効時に乱数列を消費しない、という依頼書5.2節の要求を自然に満たす設計）。

### 4.5 `pointnext_s_segmentor.py` / `evaluate_stage5.py`への影響

- `_prepare_openpoints_batch()`は`points`を無加工で`pos`へ渡す（S5-14補足2で確認済みの経路と同一）。
  training側で回転を挿入してもモデル呼び出し経路自体は変更不要。
- `evaluate_stage5.py`の`predict_h5()`は今回のaugmentation経路を一切通らない別スクリプトであり、
  変更しない。config読み込み経路を持たせる場合でも、augmentationフラグをeval側へ伝播させない
  （依頼書5.1節「評価CLIが学習configを読む場合でもaugmentationを評価へ適用してはいけない」）。

## 5. P2: 実装方針（依頼書5章 / 管理記録6章P2）

### 5.0 最新確定版: P2の完了状態（最終更新 2026-09-16、GPU preflight完了時点）

以下はP2当時の検証履歴です。現在はP3の学習・評価も完了しています。
ここに残る未実施項目を終了後の追加条件として復活させない。50 epochの結果は8.12節に記録済み。

**preflightの進捗（2026-09-16時点）:**

| Stage | 内容 | 状態 |
| --- | --- | --- |
| A | 固定リスト経路の検証（GPU不使用） | **実機実行済み・合格（2026-09-16）**。結果は上記 |
| B | dummy forward/backward（GPU） | 1回目失敗（スクリプト不具合）→修正→**2回目実機合格（2026-09-16）** |
| C | 実H5少数step（GPU） | **実機合格（2026-09-16、none / random_z_rotationの2回とも violations 0件）** |
| — | 8.7.3 旧W-A best/last監査（CPU） | 実装・CPU検証済み。**実機未実施** |

**GPU preflightは3 stageすべて合格し、完了した。P3実行申請は上記「P3実行申請」節。**

CPU検証の累計: **Python 96件＋bash 14件、すべて合格**（preflightのCPU側15件を含む）。
GPU preflight実施量の累計: **計4回**（Stage B 2回＝失敗1・合格1、Stage C 2回）。
当初申請は「Stage B 1回＋Stage C 2回＝3回」であり、
**超過分はStage Bの再実行1回**（承認済み）。Stage A（GPU不使用）1回は合格。
実学習相当の計算量は**optimizer step 計9回**（Stage B 1＋Stage C 4×2）。

### 5.1 新規モジュール

`Stage5/stage5/utils/rotation_augmentation.py`（新規、CPU-only・torch非依存のpure numpy）を追加する。

- `rotation_matrix_z(degrees: float) -> np.ndarray`: 3x3回転行列（Z軸、右手系）。
- `derive_rotation_angle_degrees(*, base_seed: int, epoch: int, video_id: str, max_abs_degrees: float) -> float`:
  `hashlib.sha256(f"{base_seed}|{epoch}|{video_id}".encode()).digest()`から64bit整数seedを作り、
  `np.random.default_rng(seed).uniform(-max_abs_degrees, max_abs_degrees)`で角度を返す。
  Pythonの`hash()`は使わない（プロセス間のPYTHONHASHSEED依存を避けるため）。
- `apply_z_rotation(points: np.ndarray, angle_degrees: float) -> np.ndarray`: 原点中心のZ軸回転。
- `AugmentationConfig`（mode: `"none"|"random_z_rotation"`、max_abs_degrees既定15.0、base_seed）を
  dataclassとして定義し、`train_stage5.py`のCLI引数・config.json・checkpointへ保存する値の単一の
  ソースにする。

S5-14補足2の診断checker（`checks/real_h5/check_stage5_coordinate_transform_diagnostics.py`）が持つ
`rotation_matrix_z`と数式は同じだが、診断checkerは確定済みのS5-14成果物でありimportして流用せず、
productionコード側に独立して実装する（診断側は変更しない）。

### 5.2 Dataset変更（8.3-1/8.3-2を反映した修正版）

`Pseudo3DPointCloudDataset.__init__`に`augmentation_mode: str = "none"`、
`augmentation_rotation_degrees: float = 15.0`、`augmentation_seed: int | None = None`を追加し、
`self._epoch_shared = multiprocessing.Value("i", 0)`（4.3節の共有メモリ設計）と
`set_epoch(self, epoch: int) -> None`（`self._epoch_shared.value = int(epoch)`）を実装する。
`augmentation_mode == "none"`のときは`_epoch_shared`を作らずコストを避けてもよいかは実装時に判断する
（作成自体は軽量なため、判断保留で常時作成する案を基本とする）。`__getitem__`内、
`normalize_xyz(points)`の直後・`selected_indices`によるwindow切り出しの直前に、
`augmentation_mode == "random_z_rotation"`のときだけ4.3/4.3.1/4.4節の手順
（共有epoch値の読み出し→`stable_video_id`（ファイル名basename）→`hashlib`決定的角度生成）で
回転を適用する。`augmentation_mode == "none"`（既定）のときは新規コード経路を一切通らず、
既存出力と完全一致させる。`val_dataset`・train sanity評価用datasetの構築時は
`augmentation_mode`を渡さない（常に`"none"`のまま、8.3-5でも明示された不変条件）。

### 5.3 CLI / config / checkpoint

`train_stage5.py`に`--augmentation {none,random_z_rotation}`（既定`none`）、
`--augmentation_rotation_degrees`（既定15.0）、`--augmentation_seed`（既定None→未指定時のみ
`args.seed + 500000`を解決済みseedとして使う、回答2で了承済みのfallback規則）を追加し、
train_dataset構築時のみ渡す（val_dataset・train sanity評価用datasetには渡さない＝常にnone）。
`build_config()`へ`augmentation`セクション（mode/degrees/**解決済みseed**。Noneではなく実際に
使われた整数値と派生規則の別を記録する、回答2の要求）を追加し、checkpoint保存時にもconfig全体を
含める既存経路でそのまま保存されるようにする。学習ループでは`train_dataset.set_epoch(epoch)`を
`run_one_epoch(loader=train_loader, ...)`呼び出し直前に追加する（val側は呼ばない＝
augmentation自体が常に無効のため epoch値も無関係）。

### 5.3.1 固定リストモード（8.7.1で了承された限定変更）

8.6.4で判明したとおり`train_stage5.py`は`--train_list`/`--val_list`に既に対応しており、
不足しているのは`train_stage5.sh`側の引数受け渡しと事前検査の入力経路である。ここへ
**両リスト指定時のみ有効になる固定リストモード**を追加する。

- `TRAIN_LIST`と`VAL_LIST`が**両方**与えられたとき、`--train_list`/`--val_list`を渡し
  `--train_dir`を渡さない（両者は`resolve_train_val_paths()`で排他のためエラー停止する）。
  `--val_dir`も割合分割も使わない。`split_train_val_paths()`は呼ばれない経路になる。
- **片方だけの指定・読込不能・空リストは停止**する。ディレクトリ＋割合分割への自動フォールバックは
  実装しない（サイレントに別のsplitで学習が走る事故を防ぐため）。
- 検査項目: train/val間の重複、各リスト内の重複、各ファイルの実在、全行のteacher v7 suffix。
- S5-15では`max_train_files`/`max_val_files`による切り詰めを無効にし、**リストの入力順序を維持**する
  （`resolve_train_val_paths()`は`--train_list`使用時に`max_*`>0なら先頭切り詰めを行うため、
  実験用bashからは0を明示する）。
- **`train_stage5.sh`のteacher preflightの対象を、ディレクトリ走査結果ではなく実際のリスト内H5に
  切り替える。** 走査対象と学習対象が別物になる構成を残さない。リスト外にファイルを追加しても
  学習対象が変わらないことをCPUテストで検証する。
- 一般利用向けの既定動作（`TRAIN_LIST`/`VAL_LIST`未設定時のディレクトリ＋`VAL_FRACTION=0.1`）は
  変更しない。固定リストモードの追加に必要な変更以外は触らない。
- 学習後に`train_stage5.py`が保存する`train_files.txt`/`val_files.txt`の内容・順序・
  content SHA-256を8.6.1のfingerprint（train `582579833f...`、val `0c251380e4...`）と照合する。
  **不一致時に基準を緩めず、内容差（別のファイル集合・順序）と表現差（改行コード・絶対/相対パス
  表記）を分けて報告する。**

### 5.4 実験用bash（8.3-3を反映、初稿5.3から分離）

初稿では「`train_stage5.sh`への手入力env var override」のみで足りるとしていたが、8.3-3の指摘に
従い、内部定数で入出力パス・条件を固定した薄い実験用bashを新設する
（`Stage5/checks/real_h5/`配下、依頼書8章の配置方針に従う）。この実験用bashは：

- 初期重み（`INIT_CHECKPOINT`、**SHA-256 `55ec6e6bcb39d58f398719b33826e80715a94bc6e7670d5b88623cd7c668438b`**）、
  `POINTNEXT_NORM=groupnorm`、固定weight`CLASS_WEIGHT="0.05963856,1.94036150"`、
  保存済みtrain/val file list（5.3.1節の固定リストモード）、`EPOCHS=5`、
  `SAVE_EVERY=1`（各epoch checkpointを保存する要求に対応）、arm別の独立出力先を内部定数として
  明示し、`train_stage5.sh`をenv var override付きで呼び出す。
  **Pythonを直接呼ぶ別の学習経路は作らない**（8.7.1の明示的な指示。既存の検査・引数設定の
  二重実装を避けるため）。
- R0用・R1用を1つのスクリプト内で`ARM=r0|r1`のような切替にするか2ファイルに分けるかは
  P2実装時に決める。一般利用向け`train_stage5.sh`自体の既定値は変更しない。
- 比較・parity検証用の`check_stage5_rotation_augmentation_ablation.*`（回答3）と統合する場合も、
  「学習を実行するモード」と「既存run出力を集計・検証するモード」を明確に分け、
  checkerを起動しただけで未承認の学習が始まらないようにする。
- 実行そのもの（GPU上でこのbashを起動すること）はP2実装・CPU test完了後、GPU preflight承認を
  経てからのみ行う。

## 6. P3: 比較・評価の実装方針（依頼書6章 / 管理記録6章P3、実行はP1/P2承認後）

### 6.0 最新確定版: P3完了報告（最終更新 2026-09-19）

依頼書9章・P3承認の末尾が求める報告項目に対する回答。

| 報告項目 | 内容 |
| --- | --- |
| 主比較 | 両armのepoch5（`last.pt`）。固定21動画、無変換・mean aggregation |
| 副次比較（best） | **実施不要と判定。** best/lastは両armとも評価モデルとして同一（epoch 5）であり、依頼書6.2に従い重複評価を回避。その事実と根拠（state_dict 63キー完全一致）を記録済み |
| split別指標 | 上記「pooled」「動画別」の各表 |
| PLY所見 | **未実施。** パイプラインはPLYを出力したが目視確認していない |
| 実施量 | 学習: 2 run×5 epoch、optimizer更新 **900**（予定値と一致）。GPU preflight: 4回（Stage B 2回＝失敗1・合格1、Stage C 2回）。評価: **42動画条件**（承認上限84の半分）。CPU: best/last同一性確認 各1回、比較checker 1回 |
| 失敗・再実行履歴 | GPU preflight Stage B 1回失敗（preflightスクリプトの実装不具合、学習コードとは無関係）→修正後に再実行1回で合格。**学習runの失敗・再実行は0回** |
| manifest | 各armのdry run／training計4件。git HEAD `646cb52b...`、起動時**clean**、実行環境、初期checkpoint hash、list fingerprint、解決済みaugmentation seed 500042を記録 |
| 承認範囲との差分 | ①per-epoch checkpoint未保存（`SAVE_EVERY`がbashに無視された。P1固定条件からの逸脱。**解消せず記録として保持**）。②~~manifest突合は配線省略によりUNKNOWN~~ → **2026-09-19に実施済み（8.12.4）。既知の`save_every`以外に不一致なし**。③`EX_DATE=260917`とdry run実施日(09-18)のラベル1日ずれ。いずれも比較の妥当性には影響しない |
| **判断** | 実装チャットは**R0維持を提案**し、管理チャットが**R0維持・R1不採用を正式決定**した（8.11.11）。R1は比較履歴として保持する |
| 次段階の判断依頼 → **回答済み（8.11.11・8.12.1）** | (a) **R0維持・R1不採用**、(b) **再学習しない**（逸脱は記録として保持）、(c) **P4へ移行**し、工程を絞ってR0の新規50 epochへ、(d) 別コミット（`cd3b886`）。なお対象は~~7ファイル~~ **変更4＋新規1＝5ファイル**（訂正） |

**未確定事項として残すもの:** 本比較は単一seed・5 epochの暫定結果である。F1約7%・IoU約3.5%という
初期段階のモデル同士の比較であり、50〜200 epochでの優劣を予測するものではない。
validation18動画は既に多数の方式選択に使用済みで独立testではない（管理記録P6の記載どおり）。
回転augmentationの有効性は「確定していない」。

### 6.1 比較・評価の実装方針（依頼書6章 / 管理記録6章P3、最終更新 2026-09-16、8.3・8.7を反映）

- **R0は新規学習する（8.7.2で決定、選択肢B）。** 旧W-A（EX260914）はR0として再利用せず、
  診断履歴として保持する（上書き・削除しない）。R0（`augmentation=none`）とR1
  （`random_z_rotation`, ±15度）を、改修後の同一コード・同一環境・同一保存設定
  （`SAVE_EVERY=1`、arm別の独立run dir）で、初期checkpoint SHA-256 `55ec6e6b...438b`から
  各5 epoch学習する（計10 training epochs、管理記録P3の通常上限内）。主比較は両armのepoch5。
- **R0を新規学習してもnone時の回帰・同値性テストは省略しない（8.7.2）。** 同じコードを使うことは
  学習経路の正しさを保証しないため、7.1節のnone経路テストは維持する。
- **保存済みfile listの経路監査（8.3-4 / 8.7.1）**: 「同じseedで学習すれば同じ分割になる」ことを
  再分割の代わりにしない。5.3.1節の固定リストモードにより、bash→CLI（`--train_list`/`--val_list`）→
  `Pseudo3DPointCloudDataset`の`h5_paths`まで保存済みリストがそのまま渡ることを保証し、
  学習後に保存される`train_files.txt`/`val_files.txt`のcontent SHA-256を8.6.1のfingerprintと
  照合する（不一致時は内容差と表現差を分けて報告し、基準を緩めない）。
- 比較は固定21動画（train sanity3＋validation18）のepoch5チェックポイントに対する無変換評価。
  既存`evaluate_stage5.py`のmean probability・2クラスargmax・同値backgroundをそのまま使う。
  pooled（TP/FP/FN/TN合算）指標と動画別F1/IoU中央値・pairedR1-R0差分中央値・中央値同士の差
  （S5-14補足3で確立した「diff-of-medians」と区別した命名）を併記する。TP0動画数、
  ignore上のpositive率、recall/FPRの変化も同じ枠組みで出す。
  この集計ロジックは`checks/real_h5/check_stage5_coordinate_transform_reconciliation.py`の
  pooled/median-diff関数と同じ設計を再利用できるかを実装時に検討する（重複実装を避ける）。
- 事前選定基準（依頼書6.3節）はコード実装ではなく報告時の判定ロジックとして、CSV/JSON集計結果から
  機械的に判定できる形（4条件の充足/非充足を表で出す）にする。

## 7. テスト計画

### 7.1 CPU synthetic（`checks/dummy/check_dummy_rotation_augmentation.py`、実装・実行は承認後）

- 角度が`[-max_abs_degrees, max_abs_degrees]`の範囲内であること。
- 回転行列適用後も原点からの距離・点同士の相対距離が保存されること（直交行列の性質）。
- 同一`(base_seed, epoch, video_id)`なら複数回計算しても同じ角度になること（同一動画の複数window、
  複数workerプロセスからの呼び出しを模擬）。
- `epoch`が変わると角度が変わること（毎epoch固定角度になる不具合の検出）。
- `video_id`が違えば角度が独立して変わること（同一epoch内で動画ごとに異なる角度）。
- `augmentation_mode="none"`のとき、既存出力（回転なし）と完全一致すること。
- 元の`points`配列がin-placeで書き換わらないこと（`data["points"]`の同一性チェック）。
- `set_epoch()`を呼ばない場合の既定epoch（0）での動作、複数回`set_epoch()`を呼んだ場合の状態遷移。
- 距離・点順・intensity/confidence・GT・valid_mask・frame_order・point_indicesが回転前後で
  変化しないこと（回転はXYZのみに作用し、他フィールドはインデックス選択のみで不変であることの
  確認）。
- **実DataLoaderでのepoch/worker検証を必須化（8.3-1）**: 初稿の「可能であれば」を削除し、
  実際に`torch.utils.data.DataLoader`（ダミーH5 fixture、`num_workers`と`persistent_workers`の
  組み合わせ: `num_workers=0`、`num_workers=2 persistent_workers=False`、
  `num_workers=2 persistent_workers=True`の3パターン最低限）で複数epochにわたり同一動画の複数
  windowを取得し、(a) 同一epoch内では同一動画の全windowが同じ角度になること、(b) `set_epoch()`で
  epochを進めると3パターンいずれでも新しい角度が反映されること、(c) 3パターン間で同一
  `(base_seed, epoch, video_id)`の角度が一致することを検証する。テスト実行環境の実際の
  multiprocessing start methodをテスト出力にも記録する。

### 7.1.1 固定リストモードのCPUテスト（8.7.1が指定した項目）

`checks/dummy/`配下に、5.3.1節の固定リストモード用のsyntheticテストを追加する。

- **モード排他**: 両リスト指定時に`--train_dir`が渡らないこと、`split_train_val_paths()`
  （seedベースの再分割）が**呼ばれない**こと（呼び出し検出を含む）。
- **片側リスト拒否**: `TRAIN_LIST`だけ／`VAL_LIST`だけの指定、読込不能ファイル、空リストで
  停止すること。**ディレクトリ分割へフォールバックしない**ことを明示的に検査する。
- **リスト順序・件数の保持**: 入力順がそのまま`h5_paths`に反映され、`max_train_files`/
  `max_val_files`による切り詰めが起きないこと。
- **重複・実在・teacher条件**: train/val間の重複、リスト内重複、存在しないパス、teacher v7以外の
  suffixをそれぞれ検出して停止すること。
- **teacher preflightの対象**: 検査対象がディレクトリ走査結果ではなくリスト内H5であること。
  **リスト外に余分なH5を置いても学習対象・検査対象が変わらない**ことを検証する。
- **ディレクトリモードの回帰**: `TRAIN_LIST`/`VAL_LIST`未設定時に、従来どおり
  `--train_dir`＋`--val_fraction`が渡り、既定動作が変わっていないこと。
- **fingerprint照合**: 保存された`train_files.txt`/`val_files.txt`のcontent SHA-256計算が、
  内容差と表現差（改行・絶対/相対パス）を区別して報告できること。

### 7.2 GPU preflight（実施対象・回数は別途提示し、実行前に承認を得る）

依頼書5.3節・管理記録P2に従い、対象動画・step数・実行回数・停止条件を別途提示する。想定内容は
dummy forward/backward（finite loss/gradient）、実H5少数step（train sanity1動画＋validation1動画
程度）での点対応・角度ログの確認の確認。フル1 epoch smokeは今回追加しない。NaN/Inf・GT対応不一致・
none時の回帰が出た場合は停止し、再実行は理由・対象・回数を示して別途承認を得る。

**評価無変換の確認範囲（8.3-5）**: 独立`evaluate_stage5.py`側の無変換だけでなく、`train_stage5.py`
内部で構築される`val_dataset`自体が`augmentation_mode`を一切渡されず常に`"none"`であることを
コードレベルで確認し、GPU preflightの少数stepでも（train側の`val_dataset`経由の出力が）無変換
であることを実データで確認する。bestチェックポイントはepoch5と同じ選択規則で保存したうえで副次
評価し、主比較はepoch5を維持する（epoch5とbestが同一なら重複評価を避け、その旨を記録する）。
「角度やepochが違えば必ず異なる乱数値になる」という一般的な保証は主張せず、固定fixture
（既知のbase_seed・epoch・video_id列）での実際の生成値を記録して再現性を確認する。

## 8. 未確定事項・確認したいこと（依頼書10章）

1. **R0のW-A再利用可否**: 4.1節の監査どおり、W-Aの実際の起動コマンド・config.json（特に
   `POINTNEXT_NORM`/`INIT_CHECKPOINT`/`CLASS_WEIGHT`の実際の値とhash）を確認できる場所
   （`/mnt/data/3d_projects/models/Stage5/work_dirs/`配下の実際のrun dir）をご提示いただくか、
   実機で該当config.jsonを確認いただく必要がある。本環境にはW-Aの実run出力がない。
2. **augmentation seedの具体的な派生規則**: 5.3節で`args.seed + 500000`という案を仮に置いたが、
   既存の`val_dataset`が`args.seed + 100000`を使っている（`train_stage5.py:839`）ため、桁の
   衝突を避けるオフセットとして問題ないか確認したい。固定案で良ければP2実装時に確定する。
3. **`checks/real_h5/check_stage5_rotation_augmentation_ablation.*`（新規、比較・parity検証用）の
   要否**: `check_stage5_class_weight_ablation.py`と同様に、config/checkpoint hash parityと
   固定21動画のpooled/median指標比較を1つのcheckerにまとめる想定だが、6章で述べた
   reconciliation関数の再利用可否を含め、P2実装時に具体案を提示してよいか。
4. 上記1・2が確認でき次第、P1報告（差分監査結果、R0再利用可否の結論案）を先に提示し、その後
   P2のコード実装・CPU testへ進む。GPU preflightの対象・回数は、コード実装完了後にあらためて
   提示し、実行承認を得てから着手する。

### 8.1 管理チャットによる照合結果

返信日: 2026-09-15。目的・固定条件・R0/R1の一要因比較・評価時無変換・段階承認の大枠は
S5-15管理方針と一致しています。ただし「内容の齟齬はない」という1章の記述は現時点では
留保し、以下8.3の修正を反映してください。本回答はP1監査継続の了承であり、R0再利用、
GPU preflight、5 epoch学習、P4以降の実行承認ではありません。

### 8.2 4事項への正式回答

**回答1: W-Aの参照先を既存記録から特定し、実機監査後にR0再利用を判断する。**

W-Aのrun出力は`models/Stage5/work_dirs/`ではありません。S5-13報告に記録された候補は以下です。

```text
/mnt/data/3d_projects/stage5_runs/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad
```

これは履歴に基づく参照先であり、現時点で実機の存在確認・hash検証を管理チャットが実施した
という意味ではありません。ここにあるconfig.json、train/val file lists、epoch5 checkpoint等を
監査してください。**初期checkpointの実パスは当該config等から取得し、basenameから推測しない**で
ください。`work_dirs/`は転移初期重み等の作業成果物の候補であり、run出力と区別します。

ローカルで先行確認できる資料は次です。

- `docs/stage5/s5-13/stage5_s5_13_report_to_policy_chat.md`の「W-A候補(EX260914)の個別確認」とF4本比較。
- `research/stage5/s5-13/s5_13_wa_share_metrics/training_history/training_config_anonymized.json`。
- 同ディレクトリのtraining metricsと匿名化済み評価結果。

匿名化configだけでは初期重み・GTの同一性まで確定できません。実機監査用のCPUスクリプトまたは
コマンドを用意し、config・実効class weight・初期重みhash・file list/GT・環境・更新条件を確認して
P1報告へまとめてください。機微なパスや動画IDは共有資料から除きます。
R0の再利用可否はその報告後に判断します。旧W-Aの`auto`指定と新R0の手動固定指定は文字列だけで
不一致とせず、**実際に使用されたweightと関連するloss処理**を比較してください。

**回答2: augmentation seedの既定派生を`args.seed + 500000`とする案を了承する。**

明示指定があればそれを優先し、未指定時だけ上記規則を使います。config/checkpoint/manifestには
Noneではなく解決済みseedと派生規則を保存してください。オフセットの大小やval seedとの差が
乱数独立性を保証するわけではなく、専用RNGとSHA-256による決定的派生、global RNGを消費しない
実装が本質です。入力の整数範囲・型を検証し、digestの整数化byte orderと動画ID規約を固定します。
none時には角度生成も乱数消費も行わないことをテストしてください。

**回答3: 比較・parity検証用checkerの追加を了承する。具体案をP2前に提示する。**

`checks/real_h5/check_stage5_rotation_augmentation_ablation.py/.sh`を追加する案で構いません。
初期checkpoint hash、固定条件、file lists、実効weight、評価対象の対応を確認し、R0/R1の
pooled/動画別指標を比較してください。**学習後のR0/R1 checkpoint同士のhash一致は要求しません。**
epoch5の対応と、augmentation以外の差分を検査します。意図した差分（run dir、augmentation設定等）は
allowlistで示し、広いキー除外で実際の設定差を隠さないでください。

reconciliation関数は入力schema・指標定義・identity依存などを確認して、自然に再利用できる部分だけ
利用してください。S5-14の結果や既存checkerの挙動を変える大きな共通化は不要です。
数値基準は機械的に集計してよい一方、微差・recall/FPRのトレードオフやPLY所見まで自動的な
採用判定へ落とし込まず、充足/非充足/要判断を分けて管理チャットへ返してください。

**回答4: P1報告を先に提示し、P2実装・CPU test、GPU preflight、P3の順で進める案を了承する。**

まず回答1の監査結果、R0再利用の結論案、8.3を反映したepoch伝達・動画ID・bash構成を提示して
ください。P1監査に必要なCPUスクリプト作成・CPU確認は進めて構いません。P2開始の了承後に
コード実装・CPU testを行い、結果とGPU対象動画/step数/実行回数/出力先を提示します。
GPU preflightはその実行承認後、P3学習・固定21動画評価はpreflight結果確認と別の実行承認後です。
承認が得られる前にrunを起動したり、フル1 epoch smokeを追加したりしないでください。

### 8.3 計画書へ反映が必要な修正条件

1. **persistent workerの検証を省略しない。** 4.3節の「persistent_workersを追加しない」という
   条件だけでは、依頼書が要求するpersistent workerを含むepoch伝達・再現性検証を満たしません。
   production loaderの既定値Falseは維持して構いませんが、epoch共有状態等の最小限の方法を
   検討し、Trueでもepoch更新が届く設計・テストを提示してください。fork前提だけに依存せず、
   実際の環境のstart methodを記録します。方式を限定したい場合は代替案として再相談してください。
   7.1の実DataLoaderテストを「可能であれば」から必須へ変更し、num_workers=0と複数worker、
   persistent=False/Trueで複数epoch・同一動画の複数windowを検証してください。
2. **絶対H5パスをstable_video_idにしない。** 同じデータでもmount prefixや相対/絶対指定の変更で
   角度が変わります。H5の既存動画識別子、または固定inventory内で検証した一意な相対識別子等を
   使い、重複・欠損時の扱いを定義してください。同一動画を別mount prefixで読んでも同じIDと角度に
   なることを確認し、IDはprivate manifestに記録、共有時は匿名化します。
3. **パス・条件を内部定数で指定する実験用bashを用意する。** 一般用train_stage5.shへ既定noneの
   env/CLI受け渡しを追加する方針は妥当ですが、手入力env overrideだけの運用は依頼書8章の
   意図と異なります。実験用の薄いbashから既存train_stage5.shを呼び、W-A初期重み・GroupNorm・
   固定weight・file lists・arm・5 epochs・SAVE_EVERY=1・独立出力先を明示してください。
   比較checkerのbashと統合する場合も、学習と集計の実行モードを明示し、checker起動だけで
   未承認学習が始まらない構成にします。一般利用向け既定値は変更しません。
4. **実際の固定split経路を監査する。** 同じseedでディレクトリを再分割することは保存済み
   file listの再利用の代わりにはなりません。bash→CLI→Datasetで既存train/val listがそのまま
   使用されることを確認し、不足する引数受け渡しがあれば限定的に対応してください。
5. **評価無変換・保存・数値比較を明示する。** train CLI内のvalidation Datasetも検査対象です。
   独立evaluateスクリプトが無変換であるだけでは足りません。bestは同一選択規則で副次評価し、
   主比較はepoch5を維持します。角度やepochが違えば必ず異なる乱数値になるという一般保証は
   置かず、固定fixtureでの更新と再現性を確認してください。

以上を反映すれば、S5-15の目的・内容に沿ったP1/P2計画になります。S5-14の診断再開、
resume・schedulerの先行実装、P4以降の自動実行は今回の回答に含めません。

### 8.4 実装チャットの確認（2026-09-15）

8.1〜8.3を確認しました。理解した内容は次のとおりです。齟齬があればご指摘ください。

- **8.1**: 今回の承認範囲はP1監査の継続のみです。R0再利用の可否、GPU preflight、5 epoch学習、
  P4以降の実行は含まれません。1章の「内容の齟齬はない」という記述は誤りで、8.3の5項目を
  反映する必要があるという指摘として受け取りました。
- **回答1**: `/mnt/data/3d_projects/stage5_runs/260914/pointnext_s_EX260914_...`は履歴上の参照先
  であり、実機での存在・hash確認は未実施という前提を保持します。初期checkpointの実パスは
  当該run dirの実config（basenameの推測ではなく）から取得します。「旧W-Aのauto指定」という
  表現も文字列一致では判断しないという点、了解しました（4章に確認済みの局所監査結果を追記します）。
- **回答2**: `args.seed + 500000`は明示指定が無いときのみのfallbackとし、解決済みseedと派生規則
  自体をconfig/checkpoint/manifestへ記録します。SHA-256 digestの整数化方法（byte order）と
  動画ID規約を実装時に固定し、noneモードでは角度生成・乱数消費を一切行わないことをテストで
  確認します。
- **回答3**: `checks/real_h5/check_stage5_rotation_augmentation_ablation.py/.sh`を追加します。
  R0/R1のtrained checkpoint同士のhash一致は要求せず、epoch5の対応関係とaugmentation以外の
  config差分の検査（allowlist方式）に絞ります。reconciliationの関数は自然に再利用できる範囲に
  限定し、S5-14既存checkerの挙動は変更しません。採否判定は「充足/非充足/要判断」の3区分で返し、
  自動的な単一判定へ丸めません。
- **回答4**: P1報告（監査結果・R0再利用結論案・8.3反映後のepoch伝達/動画ID/bash設計）→
  P2承認→P2実装・CPU test→GPU preflight対象/回数の提示・承認→GPU preflight実行→P3承認→
  5 epoch学習、という順序を維持します。承認前のrun起動、フル1 epoch smokeの追加は行いません。
- **8.3-1（persistent worker）**: 「`persistent_workers`を追加しない」という設計だけでは
  不十分という指摘を理解しました。productionの既定値`False`は維持しつつ、`persistent_workers=True`
  でもepoch更新が届く経路（例: `multiprocessing.Value`等の共有状態を`__getitem__`側で毎回読み直す
  設計）を検討し、実際の起動環境のstart method（fork/spawn/forkserver）を記録します。
  7.1のテストは「可能であれば」ではなく必須とし、`num_workers=0`と複数worker、
  `persistent_workers=False/True`の組み合わせで複数epoch・同一動画の複数windowを検証します。
- **8.3-2（stable_video_id）**: 絶対/相対H5パスやmount prefixに依存しないID（ファイル名のstem等、
  H5内容に対して安定な識別子）を使う設計へ修正します。重複・欠損時の扱いを定義し、IDは
  private manifestにのみ記録し、共有資料では匿名化します。
- **8.3-3（実験用bash）**: 手入力env overrideだけに頼らず、内部定数でW-A初期重み・GroupNorm・
  固定weight・file lists・arm・5 epochs・`SAVE_EVERY=1`・独立出力先を明示する薄い実験用bash
  （`train_stage5.sh`を呼び出す形）を用意します。比較checkerと統合する場合も学習/集計モードを
  明示し、checker起動だけで未承認学習が始まらない構成にします。
- **8.3-4（split経路の監査）**: 保存済みfile listがbash→CLI→Datasetでそのまま使われること
  （同一seedでの再分割で代用しないこと）を実装・監査の両方で確認します。
- **8.3-5（評価無変換の明示）**: 独立`evaluate_stage5.py`だけでなく、train CLI内部の
  `val_dataset`側にも`augmentation`を渡さない（常にnone）ことを検査対象にします。bestは同一選択
  規則で副次評価し主比較はepoch5を維持すること、角度・epochの違いが常に異なる乱数値を生む
  という一般論ではなく固定fixtureでの実際の値を確認することを実装方針に反映します。

### 8.5 P1実機監査スクリプト（2026-09-16、作成・CPU検証済み・実機実行待ち）

回答4「P1監査に必要なCPUスクリプト作成・CPU確認は進めて構いません」の範囲で、4.0節が挙げた
実機監査必須の4項目を1回の実行で確認するCPU専用スクリプトを作成した。**学習は一切起動せず、
監査対象run dir内のファイルも変更しない。**

- `Stage5/checks/real_h5/check_stage5_s5_15_p1_manifest_audit.py` / `.sh`（新規）
- `Stage5/checks/dummy/check_dummy_s5_15_p1_manifest_audit.py` / `.sh`（新規、synthetic test 20件）

**検査内容:**

| 区分 | 内容 |
| --- | --- |
| config | S5-15 P1固定条件37項目（`pointnext_norm`/`norm_groups`/`label_policy`/`features`/window/batch/optimizer/seed等）をconfig.jsonと逐一照合 |
| 同一性 | `num_train_samples=715`・`num_val_samples=87`（S5-13がW-Aについて記録した値）と一致するか＝監査対象がS5-13のW-Aと同一runか |
| class weight | `class_weight_info.resolved`を期待値`[0.05963856, 1.94036150]`（tol 1e-6）と照合。`mode`は**判定材料ではなく記録**とし、`manual`以外なら`JUDGE`（回答1「文字列だけで不一致としない」に対応） |
| 初期checkpoint | **config.jsonに記録された実パス**（basenameからの推測ではない）の存在確認・SHA-256・サイズ |
| file list | `train_files.txt`/`val_files.txt`の行数、内容のSHA-256 fingerprint、全行のteacher v7 suffix、**basename重複の有無**（4.3.1節の`stable_video_id`設計の前提条件） |
| checkpoint | `best.pt`/`last.pt`の有無、`checkpoint_epoch_*.pt`の有無と`save_every`の記録値 |
| history | epoch数が5か、全metricsがfiniteか |
| 環境 | **実際のmultiprocessing start method**（4.3節がforkを仮定しないための記録）、Python/numpy/torch/CUDAバージョン |

**判定区分は`PASS`/`FAIL`/`JUDGE`/`UNKNOWN`の4値**とし、最初の不一致で停止せず全項目を記録する
（実機との往復を1回で済ませるため）。`JUDGE`は「不具合ではなく人の判断が必要」、`UNKNOWN`は
「このホストでは検証不能」を意味し、`FAIL`が0件なら`status: passed`・exit 0となる。

**出力は2ファイル:** private JSON（実パス・実ファイル名を含む。実機に置いたまま共有しない）と
shareable JSON（件数・hash・判定のみ。実パスとfile list本体を除去）。shareable側には既存規約の
privacy self-check（timestamp状動画ID・絶対hostパスの正規表現スキャン）を同梱してある。

**ローカルCPU検証済み（2026-09-16）:** `py_compile`・`bash -n`合格、synthetic test 20件合格、
合成run dirに対するend-to-end実行でshareable出力に上記2パターンの検出が0件であることを確認した。
実機データに対してはまだ実行していない。

**実機で実行していただきたいコマンド（GPU不要、学習は起動しない）:**

```bash
cd /mnt/data/3d_projects/models/Stage5
RUN_DIR=/mnt/data/3d_projects/stage5_runs/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad \
  bash checks/real_h5/check_stage5_s5_15_p1_manifest_audit.sh
```

出力は`work_dirs/_s5_15_p1_audit/`配下に`p1_manifest_audit_private.json`（非共有）と
`p1_manifest_audit_shareable.json`（共有可）として保存される。**共有いただきたいのは
shareable側とコンソール出力のみ**。run dirが実機に存在しない場合はexit 2で停止するので、
その場合は正しいパスをご教示いただきたい。

なお`bash checks/dummy/check_dummy_s5_15_p1_manifest_audit.sh`も実機で実行可能だが、
本開発環境で既に20件合格を確認済みのため必須ではない。

### 8.6 P1監査結果（2026-09-16、実機実行済み）

8.5節のスクリプトをユーザーが実機で実行。**`status: passed`、PASS 60 / FAIL 0 / JUDGE 1 / UNKNOWN 0。**
shareable出力のprivacy self-checkは2パターンとも検出0件。

#### 8.6.1 確認できたこと

| 項目 | 結果 |
| --- | --- |
| 固定条件37項目 | 全PASS（`groupnorm`/8 groups、`bbox_noncontour_ignore`、`intensity,confidence`、window16/8/tail、batch1/accum8、lr1e-3/wd1e-4/clip10、seed42、epochs5 ほか） |
| 同一性 | `num_train_samples=715`・`num_val_samples=87`ともS5-13記録と一致＝**監査対象はS5-13のW-Aと同一run** |
| 実効class weight | `mode="manual"`、`requested="0.05963856,1.94036150"`、`resolved=[0.05963856, 1.9403615]` |
| 初期checkpoint | `stage5_pointnext_s_s3dis_partial_init_groupnorm.pt`、3,199,662 bytes、**SHA-256 `55ec6e6bcb39d58f398719b33826e80715a94bc6e7670d5b88623cd7c668438b`** |
| file list | train 162 / val 18行、全行teacher v7 suffix、**basename全一意**（4.3.1節`stable_video_id`設計の前提成立）、180ファイル全て実在 |
| file list fingerprint | train `582579833f345b77d111f9f02a0606a7423d4d5161dd0cb6436c994fcf353ab0` / val `0c251380e40f0def3a76bdedd74572ac0ba22e799bf0287c8fee3f14ca59e83f` |
| history | 5 epoch記録、全metrics finite |
| checkpoint | `best.pt`・`last.pt`のみ |
| 環境（現時点） | Python 3.11.15、torch 2.7.1+cu128、CUDA 12.8、numpy 2.2.2、GPU 1基、**start method `fork`**（利用可能: fork/spawn/forkserver） |

#### 8.6.2 `auto_weight`というrun dir名の由来が確定した

4.0節で「未確認」としていたディレクトリ名の矛盾が、ローカルのgit履歴で確定した。
commit `afe76ee`（2026-09-15 03:41、S5-13のclass weight ablation）**以前**の`train_stage5.sh`では、
PREFIXが次のようにリテラル固定されていた。

```text
PREFIX="w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep${EPOCHS}_..."
```

つまり`auto_weight`は**CLASS_WEIGHTの値と無関係にハードコードされていた文字列**であり、当時の
run dir名はclass weightの情報を一切持っていなかった。`class_weight_tag()`による正しい命名は
`afe76ee`で導入され、S5-13のW-B（EX260916）が最初に`cw_manual_0p5_1p5`タグを得たrunである。
**W-Aは手動固定weightで学習されており、run dir名は当時の命名仕様の遺物**である、と確定できる。
回答1の「文字列だけで不一致としない」という指示どおりに扱った結果、実体は固定weightであった。

#### 8.6.3 コードrevisionの追跡（監査スクリプトでは確認できないためローカルgitで実施）

- 学習経路のPythonコード（`train_stage5.py`・`stage5/datasets/`・`stage5/utils/`・`stage5/models/`）を
  変更した最後のcommitは`4b55e55`（2026-09-14 19:16、label policy ablation）。
- W-Aのconfig.jsonは`label_policy`・`label_policy_diagnostics`を持つが、**これらのフィールドは
  `4b55e55`が導入したもの**である。したがってW-Aは`4b55e55`以降のコードで学習された。
- `4b55e55`より後に学習経路のPythonを変更したcommitは存在しない（`afe76ee`は`train_stage5.sh`のみ、
  `c68ae81`・`a6b7c7f`は学習経路に触れていない）。
- `afe76ee`の`train_stage5.sh`差分は、命名（`class_weight_tag()`）・PREFIX/EXPERIMENT_NAME/OUTPUT_DIRの
  env var化・出力先の既存検出ガード・ログ行の追加であり、**train_stage5.pyへ渡す学習上の引数は
  変更していない**。
- **現在の**作業ツリーに学習経路の未コミット差分はない。

**結論（8.7.2の指摘を反映して限定した表現）: 記録されたcommit上、W-A実行時点と現在のHEADとの間に
学習経路のPythonコード差分は存在しない。** ただしこれはgit履歴から言えることの上限であり、
**W-A実行時に未コミットの作業差分が存在しなかったことや、実行時のライブラリ・CUDA環境が
現在と同一であることまでは証明していない**（W-A実行時の環境は8.6.5-3のとおり記録が残っていない）。
このため本結論は「B案（R0新規学習）を選べばこれらの不確定要素が設計上不要になる」という文脈での
補強材料として扱い、旧W-Aの再利用可能性を単独で根拠づけるものとしては使わない。

#### 8.6.4 新たに判明した実装上の必須対応（8.3-4関連）

`train_stage5.sh`は`--train_list`/`--val_list`を**一切渡していない**（`--train_dir`と
`--val_fraction 0.1`のみ、619行目）。一方`train_stage5.py`側は両方の引数を持ち、
`resolve_train_val_paths()`（248〜285行目）は次の挙動である。

- `--train_list`と`--train_dir`は**排他**（両方指定するとValueErrorで停止）。
- `--val_list`が与えられた場合、seedベースの再分割`split_train_val_paths()`は**呼ばれない**。

つまりW-A自身は`--train_dir`＋`val_fraction=0.1`による実行時分割でsplitを作り、その結果を
`train_files.txt`/`val_files.txt`へ保存したrunである（W-Aがこのsplitの起点）。
**現状の`train_stage5.sh`をそのまま使うとR1でもディレクトリ再走査・再分割が起き、8.3-4が禁じた
「同じseedでの再分割」になってしまう。** したがって5.4節の実験用bashでは、保存済みlistを
`--train_list`/`--val_list`で渡し、かつ`--train_dir`を**外す**プラミングが必須である
（8.3-4「不足する引数受け渡しがあれば限定的に対応してください」に該当する限定変更）。
受け入れ基準は明確で、R1が保存する`train_files.txt`/`val_files.txt`の内容SHA-256が
8.6.1のfingerprintと完全一致することである。

#### 8.6.5 残る留保（FAILではないが未解決）

1. **per-epoch checkpointの欠如（唯一のJUDGE）**: W-Aは`save_every=10`で実行されたため、
   5 epoch runに`checkpoint_epoch_*.pt`が存在しない。主比較対象のepoch5は`last.pt`として存在するため
   主比較自体は可能だが、R1を`SAVE_EVERY=1`で実行すると**両armの保存物が非対称**になり、
   中間epochの副次分析はR0側でできない。
2. **`augmentation=none`経路の同値性**: W-Aはaugmentation実装**以前**のコードで学習されている。
   R1は改修後のコードで学習するため、「R0とR1の差がaugmentationのみである」ことは、
   P2実装後にnone経路の同値性を実証して初めて主張できる。**現時点では原理的に確認不能。**
3. **W-A実行時の環境**: config.jsonにライブラリ・CUDAバージョンの記録がなく、8.6.1の環境情報は
   **監査を実行した2026-09-16時点の環境**である。W-A実行時（2026-09-14）の環境が同一である確証はない
   （同一マシン・同一conda環境である可能性は高いが、記録がないため断定しない）。
4. **`best.pt`と`last.pt`が同一checkpointか未確認**: 依頼書6.2節は「epoch5とbestが同一なら重複評価を
   避け、その事実を記録する」ことを求めている。今回の監査は両ファイルの存在のみ確認しており、
   同一性（SHA-256一致）は見ていない。

#### 8.6.6 R0再利用可否の判断案（管理チャットのご判断を仰ぎます）

**監査の範囲では、W-AをR0として再利用することを否定する材料は見つからなかった。**
固定条件・データ・初期重み・実効weight・学習コードのいずれもS5-15の要求と一致している。

一方で、単一要因性の保証の「質」に差がある点を明示したい。

| 選択肢 | 単一要因性 | 追加計算 | 残る留保 |
| --- | --- | --- | --- |
| A. W-AをR0として再利用 | none経路の同値性**論証**に依存 | R1の5 epochのみ | 8.6.5の1〜4すべて |
| B. R0を新規学習 | 同一コード・同一環境・同一保存設定のため**構成上**成立 | R0+R1で10 epochs（管理記録P3の通常上限そのもの） | 実質的に解消 |

選択肢Bでは、R0もR1と同じ改修後コード・同じ`SAVE_EVERY=1`・同じ環境で学習するため、
8.6.5の1（保存の非対称）・2（none経路の同値性）・3（環境差）が**設計上**解消し、
「差はaugmentationのみ」が事後論証ではなく構成によって担保される。

**実装チャットとしてはBを推奨する。** 理由は、8.6.5-2が事後検証でしか埋められない性質の留保であり、
本比較の目的（一要因比較）の中核に関わるためである。ただしこれは計算コストとのトレードオフであり、
Aを選ぶ判断も監査結果とは矛盾しない。**どちらを採るかは管理チャットのご判断に委ねる。**
Aを選ぶ場合は8.6.5の1〜4を留保として記録したうえで進める。

なお、Bを選ぶ場合でも**旧W-Aは履歴として保持**し（管理記録P3の記述どおり）、上書きしない。
どちらの場合も、R0/R1は8.6.1のSHA-256 `55ec6e6b...438b`の初期checkpointから開始する。

#### 8.6.7 追加で実施したい軽微な監査（承認をお願いします）

8.6.5-4を埋めるため、監査スクリプトに`best.pt`と`last.pt`のSHA-256比較を追加し、
**同一run dirに対してもう1回だけ**実行させていただきたい。CPU専用・read-only・数秒で、
学習もGPUも伴わない。同じコマンドの再実行も計算量の追加であるという依頼書1章の原則に従い、
理由（依頼書6.2節が求める重複評価回避の判断材料）・対象（W-A run dir 1件）・回数（1回）を
明示して承認をお願いする。不要であれば、この項目は「未確認」のまま進める。

### 8.7 管理チャット返信: P1監査の受入とP2への判断

返信日: 2026-09-16。8.6節と現在の`train_stage5.py` / `train_stage5.sh`を確認しました。
P1監査結果を受け入れ、以下を実装チャットへの正式判断とします。
本節はP2の限定実装・CPUテストへの着手を了承するものであり、GPU preflightや学習の実行承認ではありません。

#### 8.7.1 固定リスト入力をbashへ追加する限定変更を了承する

Python側は既に`--train_list` / `--val_list`に対応しています。不足は主にbashの引数受け渡しと
事前検査の入力経路です。既存のディレクトリ＋割合分割を一般利用向けに維持しつつ、
**S5-15では保存済みtrain/valリストを両方渡す固定リストモード**を追加してください。

- 両リストが指定されたら`--train_dir`を渡さず、`--val_dir`や割合による再分割も使用しない。
- 片方だけの指定、読込不能、空リスト等は停止し、ディレクトリ分割へ自動フォールバックしない。
- train/valの重複、各リスト内の重複、ファイル実在とteacher v7条件を検査する。
- `max_train_files` / `max_val_files`等による切り詰めをS5-15では無効にし、入力順序を維持する。
- bashのteacher preflightも**実際のリストに含まれるH5**を対象にする。ディレクトリ走査結果と
  学習対象が別になる構成を残さない。リスト外ファイルの追加で今回の対象が変わらないことを検証する。
- 出力`train_files.txt` / `val_files.txt`の内容・順序とfingerprintをW-A保存済みリストと照合する。
  今回の実機パスでは8.6.1のhash一致を基準とし、改行やパス表現の差で不一致が出た場合も
  黙って基準を緩めず、内容差と表現差を分けて報告する。

S5-15実験用の薄いbashから既存`train_stage5.sh`を呼ぶ構成を維持してください。
Pythonを直接呼ぶ別の学習経路を増やして既存検査・引数設定を重複させる案は今回は採用しません。
リストモード追加に必要な変更以外の一般向け既定値は維持します。
CPUテストにはモード排他、片側リスト拒否、リスト順序・件数保持、再分割を呼ばないこと、
ディレクトリモードの回帰を含めてください。

#### 8.7.2 選択肢B: R0/R1を新規学習する方針を採用する

旧W-AをR0として再利用せず、**改修後の同じコード・環境・保存設定からR0/R1各5 epoch**を
比較する方針とします。通常上限の計10 training epochs内であり、一要因比較の条件を揃える利点を
計算量削減より優先します。旧W-Aは診断履歴として保持し、上書き・削除しません。

両armは8.6.1で確認した同一GroupNorm転移初期checkpointから開始します。
初期重みSHA-256は`55ec6e6bcb39d58f398719b33826e80715a94bc6e7670d5b88623cd7c668438b`です。
保存済みtrain162 / val18リスト、実効固定weight、seed、その他の条件を一致させ、
`SAVE_EVERY=1`と独立run dirを明示してください。主比較は両armのepoch5です。

新規R0を作っても**augmentation=none時の回帰・同値性テストは省略しません**。
同じコードを使うことだけで学習経路の正しさを保証できるわけではありません。
また8.6.3のGit追跡は「記録されたcommit上の学習コード差分がない」根拠として扱い、
過去実行時の未コミット変更や環境まで完全同一と断定しないでください。

この判断は比較設計の選択です。実際の学習開始はP2結果とGPU preflightを確認した後に、
コマンド・manifest・出力先・実施量を提示して別途承認を得てください。

#### 8.7.3 旧W-Aのbest/last追加CPU監査を1回承認する

8.6.7の申請について、**W-A run dir 1件に対するCPU専用・checkpoint読取専用監査を1回**
承認します。監査結果の新規保存は構いませんが、元run・checkpointは変更しません。
GPU・学習・別runへの探索は含みません。

まずSHA-256を比較し、一致ならファイルとして同一と判断します。不一致の場合でも
serializationやメタ情報だけが異なる可能性があるため、不一致だけで異なるモデルと結論しません。
必要に応じて同じ1回の監査内でCPU上に読み込み、epoch・モデル設定・state_dictのキー/shape/dtype/
値を確認し、ファイル同一性と評価モデル同一性を区別してください。
読込は既存の信頼済みcheckpointの取扱いに従い、比較結果を追記します。

今回Bを選ぶため、この監査は旧W-Aの履歴整理であり、P2開始を止める必須条件ではありません。
未実施なら未実施と記録し、失敗後の再実行は理由を示して追加承認を求めてください。

#### 8.7.4 次の進め方と実施境界

P1の判断待ち事項は上記で回答済みです。4〜7章・9章の計画を整合させ、8.3の修正条件を保持した
P2（rotation helper、Dataset epoch/worker伝達、CLI/config、固定リスト対応、実験用bash、
比較checker）の実装・CPUテストへ進んでください。persistent workerと安定動画IDの検証、
評価時無変換、元cache不変、RNG分離も省略しません。

実装後は変更箇所、CPU検証結果、none時の回帰確認、固定splitの検算を報告し、
GPU preflightの対象動画・step数・回数・停止条件・出力先を申請してください。
承認済みGPU preflight完了後にP3実行を別途申請します。追加seed/角度探索、フル1 epoch smoke、
P4以降、production設定変更は今回の承認に含めません。

### 8.8 実装チャットの確認（2026-09-16、8.7に対して）

8.7を確認しました。理解した内容と、計画各章へ反映した箇所は次のとおりです。

**8.7.1（固定リストモード）— 了承事項として理解:**

- Python側は既に対応済みで、不足はbashの引数受け渡しと事前検査の入力経路である、という整理に
  同意する（8.6.4の調査結果と一致）。
- 追加するのは「両リスト指定時の固定リストモード」であり、一般利用のディレクトリ＋割合分割は
  維持する。**片側だけの指定・読込不能・空リストは停止**し、ディレクトリ分割へ自動フォールバック
  しない（サイレントな代替経路を作らない）。
- 検査項目: train/val間の重複、各リスト内の重複、ファイル実在、teacher v7条件。
- S5-15では`max_train_files`/`max_val_files`による切り詰めを無効にし、**入力順序を維持**する。
- **bashのteacher preflightの対象を実際のリスト内H5に変更する。** ディレクトリ走査結果と学習対象が
  別物になる構成を残さない。リスト外にファイルを足しても対象が変わらないことを検証する。
- 出力`train_files.txt`/`val_files.txt`の内容・順序・fingerprintを8.6.1のhashと照合し、
  **不一致時に基準を緩めない。内容差と表現差（改行・パス表現）を分けて報告する。**
- 薄い実験用bash→既存`train_stage5.sh`という構成を維持し、**Pythonを直接呼ぶ別の学習経路は作らない**。
- CPUテストに「モード排他」「片側リスト拒否」「リスト順序・件数保持」「再分割を呼ばないこと」
  「**ディレクトリモードの回帰**」を含める。
- → 5.3節（固定リストモードのCLI/bash仕様）、5.4節（実験用bash）、7.1節（CPUテスト項目）へ反映。

**8.7.2（選択肢B: R0/R1新規学習）— 了承事項として理解:**

- 旧W-Aを再利用せず、改修後の同じコード・環境・保存設定でR0/R1各5 epoch（計10 epochs、通常上限内）。
  旧W-Aは診断履歴として保持し、上書き・削除しない。
- 両armとも初期checkpoint SHA-256 `55ec6e6b...438b`から開始。保存済みtrain162/val18リスト、
  実効固定weight、seedを一致させ、`SAVE_EVERY=1`・独立run dirを明示する。**主比較は両armのepoch5。**
- **新規R0にしてもnone時の回帰・同値性テストは省略しない。** 「同じコードを使うこと」自体は
  学習経路の正しさを保証しないという指摘を理解した。7.1節の該当テストは維持・強化する。
- 8.6.3のgit追跡は「記録されたcommit上の差分がない」根拠に限定し、過去実行時の未コミット変更や
  環境の同一性まで断定しない。→ **8.6.3の結論文を上記のとおり限定表現へ修正済み。**
- 学習開始自体はP2結果・GPU preflight確認後の別承認である（本節は比較設計の決定）。
- → 3章・6章（R0は新規学習前提へ）、9章へ反映。

**8.7.3（旧W-A best/last監査、1回承認）— 承認範囲として理解:**

- W-A run dir 1件・CPU専用・checkpoint読取専用・**1回**。監査結果の新規保存は可、元run・
  checkpointは変更しない。GPU・学習・別runへの探索は含まない。
- 手順: まずSHA-256比較。一致ならファイルとして同一。**不一致でも「異なるモデル」と結論しない**
  （serialization・メタ情報だけが異なる可能性）。必要なら同じ1回の監査内でCPU読込し、
  epoch・モデル設定・state_dictのキー/shape/dtype/値を確認して、
  **ファイル同一性と評価モデル同一性を区別**する。
- B採用のため本監査は旧W-Aの履歴整理であり、**P2開始の必須条件ではない**。未実施なら未実施と記録。
  失敗後の再実行は理由を示して追加承認を求める。

**8.7.4（P2着手と実施境界）— 理解:**

- P2の実装範囲: rotation helper、Dataset epoch/worker伝達、CLI/config、固定リスト対応、
  実験用bash、比較checker。8.3の5修正条件は保持する。
- 省略しない検証: persistent worker、安定動画ID、評価時無変換、元cache不変、RNG分離。
- 実装後の報告内容: 変更箇所、CPU検証結果、none時の回帰確認、固定splitの検算。
  そのうえでGPU preflightの対象動画・step数・回数・停止条件・出力先を申請する。
- 今回の承認に**含まれない**もの: 追加seed/角度探索、フル1 epoch smoke、P4以降、production設定変更。
  P3実行は承認済みGPU preflight完了後に別途申請する。

**留意点（実装時に守る）:** 8.7.3の監査はP2と独立に実施可能だが、こちらも「1回」の範囲を厳守し、
実行前に本書へ実施予定を記録してから行う。

#### 8.8.1 P2実装の5ステップ（着手順、いずれもCPUのみ・学習は起動しない）

8.7.4で着手が了承された範囲を、依存関係の順に次の5ステップで実装する。各ステップ完了時に
`py_compile`・`bash -n`・`git diff --check`と該当CPUテストを通し、ステップ5でまとめて報告する。
**この5ステップにGPU実行・学習は一切含まない。**

##### 8.8.1.1 Step 1: rotation helper（純numpy、torch非依存）

- 新規`Stage5/stage5/utils/rotation_augmentation.py`（5.1節の設計）:
  `rotation_matrix_z()`、`derive_rotation_angle_degrees()`（`hashlib.sha256`ベース、
  Pythonの`hash()`を使わない）、`apply_z_rotation()`、`AugmentationConfig`。
- digestの整数化byte order、入力の整数範囲・型検証、角度範囲を実装時に固定する（回答2）。
- 新規`checks/dummy/check_dummy_rotation_augmentation.py`/`.sh`: 7.1節の角度範囲・距離保存・
  同一`(base_seed, epoch, video_id)`の一致・epoch更新・動画ID独立性・固定fixtureでの実値記録
  （8.3-5の「一般保証を主張しない」に対応）。
- S5-14補足2の診断checkerは**変更しない**（同じ数式だが独立実装、5.1節）。

##### 8.8.1.2 Step 1 完了報告（2026-09-16、CPU検証済み）

新規ファイル2種（+bash 1）。実装は上記方針どおりで、逸脱はない。

- `Stage5/stage5/utils/rotation_augmentation.py`（新規、純numpy・torch非依存）
- `Stage5/checks/dummy/check_dummy_rotation_augmentation.py` / `.sh`（新規、synthetic test **19件全合格**）

確定した実装仕様（回答2「digestの整数化byte orderと動画ID規約を固定します」への対応）:

| 項目 | 確定値 |
| --- | --- |
| hash対象キー | `f"{base_seed}\|{epoch}\|{video_id}"`（UTF-8）。base_seedとepochは整数で`\|`を含まないため、末尾の`video_id`が他の三つ組のキーを詐称できない＝単射 |
| digest→整数 | SHA-256 digestの**先頭8バイト・big-endian**を`np.random.default_rng()`へ与える |
| 角度 | `default_rng(seed).uniform(-max_abs_degrees, +max_abs_degrees)`、既定`max_abs_degrees=15.0` |
| `stable_video_id` | `Path(h5_path).name`（ファイル名のみ。mount prefix・相対/絶対の違いで変わらない） |
| seed解決 | 明示指定を優先。未指定時のみ`train_seed + 500000`。`seed_source`として派生規則も記録 |
| 回転行列 | `[[c,-s,0],[s,c,0],[0,0,1]]`（右手系CCW、float64計算→float32出力） |

特筆すべき検証結果:

- **S5-14補足2の診断checkerの回転行列とbit単位で一致**（差分厳密に0.0、`np.array_equal`で検証）。
  ±15度がS5-14診断と同じ意味を持つことを担保した。診断checkerはimportしての比較のみで、**変更していない**。
- **`PYTHONHASHSEED`非依存を実証**: 0 / 12345 / random の3通りの環境変数で別プロセスを起動し、
  同一`(base_seed, epoch, video_id)`から**同一の角度**が得られることを確認した
  （Pythonの`hash()`を使わない理由そのものの検証）。
- **numpyのグローバルRNGを消費しない**: 角度生成19回の前後で`np.random.get_state()`が不変。
  shuffle・model初期化の乱数列と分離されていること（4.4節）を実測で確認した。
- **none時は入力配列そのものを返し、角度生成も乱数消費も行わない**（`rotated is points`で検証）。
- **1動画1角度の契約**: 動画全体を回転してからwindowを切り出した結果と、window単体を同じ角度で
  回転した結果が完全一致。overlap領域の点も一致することを確認した。
- 固定fixtureの**実際の角度値6件を表としてテストに埋め込み**、キー形式・byte order・生成器の
  いずれかが変わればテストが落ちるようにした（8.3-5「一般保証を主張せず固定fixtureで確認」への対応）。
- 入力検証14件（bool/float/strのseed・epoch、負のepoch、空/非strのvideo_id、
  `max_abs_degrees<=0`・NaN、未知のmode、`[N,2]`や1次元のpoints）が拒否されることを確認。

静的チェック: `py_compile`・`bash -n`・`git diff --check`いずれも合格。
テストは本開発環境（numpy 2.5.3のvenv）で実行した。torch・CUDA・H5は不要。

##### 8.8.1.3 Step 2: Dataset（epoch共有・回転適用・stable_video_id）

- `Pseudo3DPointCloudDataset`に`augmentation_mode`/`augmentation_rotation_degrees`/
  `augmentation_seed`、`multiprocessing.Value`による`_epoch_shared`と`set_epoch()`を追加（4.3節）。
- 回転は`normalize_xyz()`直後・window切り出し直前に適用（4.2節）。`stable_video_id`は
  ファイル名basename、構築時にbasename重複を検査して停止（4.3.1節。実データでの一意性は
  8.6.1で確認済み）。
- 元`data["points"]`をin-placeで書き換えない（cache不変、8.7.4）。
- `checks/dummy/`に**実`torch.utils.data.DataLoader`テスト**を追加（7.1節、必須化済み）:
  `num_workers=0` / `num_workers=2 persistent_workers=False` / `num_workers=2
  persistent_workers=True`の3パターン×複数epoch×同一動画の複数window。実際のstart methodを記録。
- `augmentation_mode="none"`時に新規経路を通らず既存出力と完全一致すること、RNGを消費しないこと。

##### 8.8.1.4 Step 2 完了報告（2026-09-16、CPU検証済み）

- `Stage5/stage5/datasets/pseudo3d_pointcloud_dataset.py`（**既存ファイルへの追加のみ**、
  既定値は`augmentation_mode="none"`で既存挙動は不変）
- `Stage5/checks/dummy/check_dummy_rotation_augmentation_dataset.py` / `.sh`（新規、**11件全合格**）

実装内容:

- `augmentation_mode` / `augmentation_rotation_degrees` / `augmentation_seed`を追加。
  seed解決はStep 1の`AugmentationConfig.resolve()`に委譲し、Dataset側に重複実装を作らない。
  解決済みconfigは`dataset.augmentation`として公開し、Step 3のconfig.json記録で使う。
- 回転は`normalize_xyz()`直後・window切り出し直前（契約どおりの位置）。
  `astype`→`normalize_xyz`→回転がそれぞれ新しい配列を確保するため、`cache_data=True`でも
  キャッシュ元配列は書き換わらず、回転は累積しない。
- `meta["rotation_angle_degrees"]`に実際の角度を載せた（none時は`None`）。
  既存collateは`meta`をlistのまま透過するため**collate変更は不要**で、既存の`meta`利用箇所は
  すべてキー名指定の読み取りのため影響がないことを確認済み。GPU preflightでの角度ログに使える。
- 評価側の不変条件: `val_dataset`等は`augmentation_mode`を渡さなければ既定の`"none"`のまま。
  加えて**augmentation有効時は`normalize_points=True`を必須**とした（原点回転が動画重心回転と
  一致するのは正規化後のみのため、不整合な組合せを構築時に停止させる）。

**設計変更（重要、8.3-1関連）: 共有epoch値を`lock=False`に変更した。**
当初`multiprocessing.Value("i", 0)`（既定`lock=True`）で実装しStep 2のテストはfork環境で全合格したが、
start method非依存性を実測で確認したところ、**spawn文脈のworkerでは既定のlock付きValueが
`RuntimeError: A SemLock created in a fork context is being shared with a process in a spawn context`
で失敗する**ことが判明した。`lock=False`（共有メモリのみ、ロックなし）へ変更したところ
**fork・spawnの両方で成功**した。ロックは不要である（書き込むのはepoch境界の親プロセスのみ、
workerは読むだけで、単一のC intは分割されずに読み書きされる）。
これにより「forkを前提にしない」という8.3-1の要求を、仮定ではなく実測で満たしている。

CPU検証結果（11件）:

| 検証 | 結果 |
| --- | --- |
| none時の既存経路との同値性 | `normalize_xyz()`+window切り出しの結果と**bit単位で一致**。角度は`None` |
| none時の乱数・キャッシュ | グローバルRNG不変、キャッシュ元`points`がディスク上の値と一致 |
| 有効時の角度適用 | 導出角度で回転した参照と完全一致 |
| 1動画1角度 | 2動画10 windowで**動画ごとに角度が厳密に1種類**、動画間では異なる |
| 回転の非累積 | `cache_data=True`でepoch 1→2→3と進めて3種の角度、**epoch 1へ戻すと初回と完全一致** |
| XYZ以外の不変性 | features/labels/valid_mask/frame_order/point_indices/window境界が全サンプルで一致 |
| basename重複 | augmentation有効時のみ`ValueError`。無効時は従来どおり許容（一般利用の挙動を変えない） |
| 前提条件 | `normalize_points=False`との併用、seed無しでの有効化をいずれも構築時に停止 |
| `set_epoch()`検証 | 既定0、往復、bool/float/負値/C int超過を拒否 |
| **worker/epoch行列（必須）** | `num_workers=0` / 2 worker非persistent / 2 worker persistent / **明示spawn文脈**の4構成 × 3 epoch × 2動画で**すべて厳密一致**。ホスト既定のstart methodは`fork`と記録 |
| none時のDataLoader経由 | 2 worker・複数epochで全サンプル角度`None` |

静的チェック（`py_compile`・`bash -n`・`git diff --check`）合格。Step 1のテスト19件、
P1監査テスト20件も再実行して回帰がないことを確認した。

**テスト実行環境について:** この必須テストはtorchを要するため、本開発環境の検証用venvへ
**CPU版torch 2.7.1+cpu**（実機の`2.7.1+cu128`と同一バージョン系列）を導入して実行した。
CUDAは使用しておらず（`torch.cuda.is_available()`は`False`）、学習・GPU実行は行っていない。
実機での再実行も`bash checks/dummy/check_dummy_rotation_augmentation_dataset.sh`で可能。

##### 8.8.1.5 Step 3: CLI/config ＋ 固定リストモード

- `train_stage5.py`: `--augmentation`/`--augmentation_rotation_degrees`/`--augmentation_seed`
  （未指定時のみ`args.seed + 500000`）を追加し、**train_datasetにのみ渡す**。`build_config()`へ
  **解決済みseedと派生規則**を保存（回答2）。学習ループに`train_dataset.set_epoch(epoch)`を追加。
  `val_dataset`側は常に`"none"`であることをコードで固定する（8.3-5）。
- `train_stage5.sh`: 5.3.1節の固定リストモード（両リスト指定時のみ有効、片側指定・読込不能・
  空リストは停止、フォールバックなし、`max_*`切り詰め無効、**teacher preflightの対象を
  リスト内H5へ切替**）。一般利用向け既定値は変更しない。
- `checks/dummy/`に7.1.1節の7項目（モード排他・片側拒否・順序保持・重複/実在/teacher検出・
  preflight対象・**ディレクトリモード回帰**・fingerprint照合の内容差/表現差分離）。

##### 8.8.1.6 Step 3 完了報告（2026-09-16、CPU検証済み）

- `Stage5/stage5/utils/file_list_mode.py`（新規、stdlibのみ）
- `Stage5/train_stage5.py`（既存、CLI 3引数・config記録・`set_epoch()`呼び出しを追加）
- `Stage5/train_stage5.sh`（既存、固定リストモードとaugmentation env varを追加）
- `Stage5/checks/dummy/check_dummy_fixed_list_mode.py` / `.sh`（新規、**bash 6件＋Python 10件合格**）
- `check_dummy_rotation_augmentation_dataset.py`にtrain CLI配線の検証を1件追加（**計12件合格**）

**CLI/config:**

- `--augmentation {none,random_z_rotation}`（既定`none`）、`--augmentation_rotation_degrees`（既定15.0）、
  `--augmentation_seed`（未指定時は`--seed + 500000`）を追加。
- **`augmentation_mode`はtrain datasetにのみ渡す。** val dataset構築時は引数自体を渡さない
  （既定の`"none"`のまま）ことをコードとテストの両方で固定した（8.3-5）。
- `build_config()`に`augmentation_info`を追加。**生のCLI値ではなく解決済みconfig**を記録するため、
  `--augmentation_seed`未指定でも`base_seed=500042`・`seed_source="train_seed_plus_500000"`・
  角度導出式が残る（回答2の「Noneではなく解決済みseedと派生規則を保存」）。
- 学習ループの各epoch先頭に`train_dataset.set_epoch(epoch)`を追加（loaderのiteration開始前）。

**固定リストモード（8.7.1の要求を実装）:**

`TRAIN_LIST`と`VAL_LIST`の**両方**が設定されたときのみ有効。実装は`train_stage5.sh`内の
抽出可能な関数`input_source_args()`と、検証用モジュール`file_list_mode.py`に分けた。

| 要求（8.7.1） | 実装 |
| --- | --- |
| 両リスト時は`--train_dir`を渡さない | `input_source_args()`が`--train_list/--val_list`のみを出力。`train_stage5.py`は両者を排他として拒否するため、構造的に両立しない |
| 割合・`--val_dir`による再分割を使わない | `--val_fraction`を渡さないため`resolve_train_val_paths()`の`split_train_val_paths()`分岐に到達しない |
| 片側指定・読込不能・空リストは停止 | bash側で片側指定を拒否、Python側で不在・空・コメントのみを`FixedListError`→exit 2。**フォールバック経路は実装しない** |
| train/val重複・リスト内重複・実在・teacher v7 | `validate_fixed_lists()`が全て検査。**別mountの同名ファイル**も拒否（動画名が角度と評価の識別子のため） |
| `max_*`切り詰め無効・順序維持 | 固定リストモードでは`--max_train_files 0 --max_val_files 0`を強制。ソートせずリスト順を保持 |
| **teacher preflightの対象をリスト内H5へ** | 検証済みリストからmanifestを生成し、preflightのPythonがディレクトリglobではなくmanifestを読む。固定リストモードでは`INPUT_DIR`を走査しない |
| リスト外ファイルを足しても対象が不変 | テストで検証（下記） |
| fingerprint照合で内容差と表現差を分離 | `compare_file_lists()`が`content_sha256`（パス表記込み）と`identity_sha256`（動画名のみ）を別々に返し、`representation_differs_only`を明示 |

**一般利用向けの既定動作は不変**: `TRAIN_LIST`/`VAL_LIST`未設定時は従来どおり
`--train_dir`＋`--val_fraction`＋`--max_*`を渡す。これをテストで回帰検証している。

CPU検証結果:

- **bash 6件**（`input_source_args()`を`sed`で抽出して実行。既存`check_dummy_class_weight_tag.sh`と
  同じ手法のため実装から乖離しない）: 固定リスト時の引数構成、`MAX_*_FILES`が非0でも切り詰めない、
  **ディレクトリモードの回帰**2件、片側指定の拒否2件。
- **Python 10件**: 順序保持（ソートし直さない）、不在/空/コメントのみの拒否、重複・overlap・同名別mount、
  実在しないパス・teacher v6の拒否、総数検証、**ディレクトリへ未掲載H5を3件追加しても解決結果が
  byte一致**、manifestのtrain→val順序、fingerprintの内容差/表現差の分離（prefix差・並べ替え・
  動画差を区別）、コメント・空行・空白の扱い、CLI成功時のmanifest生成と失敗時のexit 2
  （**失敗時はmanifestを残さない**）。
- **train CLI配線1件**: train datasetのみaugmentation有効・val dataset無効（`meta`の角度も`None`）、
  `augmentation_info`に解決済みseedが記録される、**`set_epoch(epoch)`が`run_one_epoch()`より前に
  呼ばれる**ことをソース上で確認（8.3-1が警告する「角度が全epoch固定になる不具合」への回帰ガード）。

静的チェック（`py_compile`・`bash -n`・`git diff --check`）合格。
既存の`check_dummy_class_weight_tag.sh`（`train_stage5.sh`の別関数を検証）も再実行して合格を確認し、
Step 1（19件）・Step 2（12件）・P1監査（20件）にも回帰がないことを確認した。
`train_stage5.py --help`で新規3引数が正しく表示されることも確認した。

**未実施（設計上ここでは確認できないもの）:** 実H5・実file listでの`train_stage5.sh`の
end-to-end実行は、teacher preflightが実データ180件を要するため本環境では行えない。
GPU preflight申請時（Step 5）に、まず`PREFLIGHT_ONLY=1`での固定リストモード確認を含める予定。

##### 8.8.1.7 Step 4: 実験用bash ＋ 比較checker

- 5.4節の薄い実験用bash（内部定数で初期重みSHA-256 `55ec6e6b...438b`・`groupnorm`・固定weight・
  保存済みリスト・`EPOCHS=5`・`SAVE_EVERY=1`・arm別独立出力先を明示し、`train_stage5.sh`を呼ぶ）。
  **Pythonを直接呼ぶ別経路は作らない**（8.7.1）。
- `checks/real_h5/check_stage5_rotation_augmentation_ablation.py`/`.sh`（回答3）: 初期checkpoint
  hash一致、固定条件、file lists、実効weight、評価対象の対応、R0/R1のpooled/動画別指標比較。
  **学習済みcheckpoint同士のhash一致は要求しない。** 意図した差分はallowlistで明示し、広いキー除外を
  しない。reconciliation関数は自然に再利用できる範囲のみ利用し、S5-14既存checkerは変更しない。
- 「学習実行モード」と「集計・検証モード」を分離し、**checker起動だけで未承認学習が始まらない**
  構成にする（8.3-3）。

##### 8.8.1.8 Step 4 完了報告（2026-09-16、CPU検証済み）

- `Stage5/checks/real_h5/run_stage5_s5_15_arm.sh`（新規、arm起動用）
- `Stage5/checks/real_h5/check_stage5_rotation_augmentation_ablation.py` / `.sh`（新規、比較・検証用）
- `Stage5/checks/dummy/check_dummy_rotation_augmentation_ablation.py` / `.sh`（新規、**13件合格**）

**学習実行モードと集計モードの分離（8.3-3・8.7.1）:**

- 起動スクリプトだけ`run_`接頭辞にした（`check_`ではない）。checks配下で学習を起動しうる唯一の
  スクリプトであることを名前で区別するため。
- **既定はdry run。** 全条件を検証して解決済みコマンドを表示し、**学習せずexit 0**する。
  実際の起動は`CONFIRM_TRAINING=1`のときのみ。承認前に誤って学習が始まらない構成にした。
  これは依頼書6.1「学習開始前に両armのコマンド、manifest、出力先を提示」にもそのまま使える。
- 比較checker側には`train_stage5.sh`を起動する経路が存在しないことをテストで固定した。
- 実験条件（初期checkpoint・W-A保存済みリスト・groupnorm・固定weight・5 epoch・`SAVE_EVERY=1`・
  arm別出力先）はすべて**内部定数**。呼び出し側のenv打ち間違いで両armがずれない。
  起動前に初期checkpointのSHA-256照合、リスト検証（Step 3のモジュールを再利用）、
  出力先の既存検出を行い、いずれか失敗すれば学習前に停止する。
- **Pythonを直接呼ぶ別の学習経路は作っていない**（`train_stage5.sh`を呼ぶだけ）。

**比較checkerの設計（回答3を反映）:**

| 回答3の指示 | 実装 |
| --- | --- |
| 学習後checkpoint同士のhash一致は要求しない | 要求しない。むしろ**byte一致の場合に`JUDGE`**（augmentationが学習に届いていない疑い）として報告 |
| 意図した差分はallowlistで示す | allowlistは`output_dir`/`augmentation`/`augmentation_info`の**3キーのみ**。seed・class weight・norm・`train_list`/`val_list`・epochs・回転範囲などの差は全て検出される |
| 広いキー除外で設定差を隠さない | S5-13のchecker（`train_list`等を広く除外）とは対照的に、file list引数も一致必須にした |
| reconciliation関数は自然に再利用できる部分だけ | `pooled_confusion_metrics()`のみ**無変更で再利用**（`h5_metrics.csv`の列名が既に一致）。condition対identityで鍵付けされた中央値関数は構造が合わないため**再利用せず**本checker内に実装した。S5-14既存checkerは変更していない |
| 採用判定へ自動的に落とし込まない | 4基準を`satisfied`/`not_satisfied`/`requires_judgment`で報告。recall/FPRトレードオフの基準3は**常に`requires_judgment`**、総合は**常に`requires_policy_chat_judgment`**。採用の判定は出力しない |

集計内容は、split別pooled指標（合算TP/FP/TN/FNから導出）、動画別pairedの
`median_of_per_video_diffs`と`diff_of_split_medians`（S5-14補足3で分離した2統計を別名で併記）、
改善/悪化/同値数、arm別TP0動画数、**ignore領域のpositive率（valid GT上のFPとは別集計）**。
出力はprivate JSONと匿名化shareable JSON（privacy self-check同梱）。

CPU検証（13件合格）:

- allowlist外の設定差6種（seed・class weight・norm・`train_list`・epochs・回転範囲）を全て検出。
- arm入れ替え・回転範囲不一致の検出。
- file listのprefix差は`JUDGE`、動画集合の差は`FAIL`（内容差と表現差の区別）。
- 学習後checkpointが異なれば`PASS`、byte一致なら`JUDGE`、per-epoch checkpoint不足は`JUDGE`。
- pooled指標が合算カウント由来であること、2つの中央値統計が別々に出ること、TP0のarm別集計。
- 評価データが無い場合、計算可能な基準は`UNKNOWN`となり**「満たした」と見なさない**。
- 微差（+0.04pt）は規定どおり`satisfied`としつつ`marginal`フラグを立てる。
- shareable出力からrun path・動画識別子が除かれ、privacy 2パターンとも検出0件。
- **launcherが`CONFIRM_TRAINING=1`なしに学習しないこと、比較checkerが学習を起動できないこと**を
  構造検査で固定。

**実機なしでの動作確認:** 合成run dir・合成`h5_metrics.csv`に対して比較checkerを
end-to-end実行し、`status: passed`（PASS 22 / FAIL 0 / UNKNOWN 1）、pooled・動画別・4基準の
出力とprivacy self-check合格を確認した。launcherもdry runで完走し、
条件検証→コマンド提示→**学習せずexit 0**を実測した。

静的チェック（`py_compile`・`bash -n`・`git diff --check`）合格。Step 1〜3の全テスト
（19+12+10件）とP1監査（20件）にも回帰なし。

##### 8.8.1.9 Step 5: CPU検証のとりまとめとGPU preflight申請

- 全CPUテストの実行結果、変更ファイル一覧、none時の回帰確認、固定splitの検算
  （保存リストのfingerprintが8.6.1と一致する経路の確認）を本書へ追記する。
- そのうえでGPU preflightの**対象動画・step数・実行回数・停止条件・出力先**を申請する（8.7.4）。
  フル1 epoch smokeは申請しない。承認を得るまでGPUでの実行は行わない。
- 8.7.3の旧W-A best/last監査（CPU・1回）もこの区切りで実施し、結果を併記する。未実施なら
  未実施と記録する。

##### 8.8.1.10 Step 5 完了報告（2026-09-16）

### 8.9 P2実装の完了報告（2026-09-16、8.8.1 Step 5報告の内訳）

#### 8.9.1 変更・追加ファイル一覧（P2全体）

| 区分 | ファイル | 種別 |
| --- | --- | --- |
| production | `stage5/utils/rotation_augmentation.py` | 新規 |
| production | `stage5/utils/file_list_mode.py` | 新規 |
| production | `stage5/datasets/pseudo3d_pointcloud_dataset.py` | 変更（+107行、削除0） |
| production | `train_stage5.py` | 変更（+62/-6行） |
| production | `train_stage5.sh` | 変更（+156/-22行） |
| 実験用 | `checks/real_h5/run_stage5_s5_15_arm.sh` | 新規（学習起動、既定dry run） |
| 検証用 | `checks/real_h5/check_stage5_rotation_augmentation_ablation.py` / `.sh` | 新規 |
| 検証用 | `checks/real_h5/check_stage5_s5_15_p1_manifest_audit.py` / `.sh` | 新規（P1、実行済み） |
| 検証用 | `checks/real_h5/check_stage5_checkpoint_identity.py` / `.sh` | 新規（8.7.3用） |
| CPU test | `checks/dummy/check_dummy_rotation_augmentation.*` | 新規 |
| CPU test | `checks/dummy/check_dummy_rotation_augmentation_dataset.*` | 新規 |
| CPU test | `checks/dummy/check_dummy_fixed_list_mode.*` | 新規 |
| CPU test | `checks/dummy/check_dummy_rotation_augmentation_ablation.*` | 新規 |
| CPU test | `checks/dummy/check_dummy_checkpoint_identity.*` | 新規 |
| CPU test | `checks/dummy/check_dummy_s5_15_p1_manifest_audit.*` | 新規 |
| 記録 | `docs/stage5/FILES.md` | 追記 |

**既存コードの削除・挙動変更はない。** productionの既定値は`augmentation=none`・ディレクトリ分割の
ままで、`TRAIN_LIST`/`VAL_LIST`未設定時の`train_stage5.sh`の動作は従来と同一である。
外部PointNeXt本体、S5-14の確定済みchecker、`evaluate_stage5.py`、`infer_stage5.py`は変更していない。

#### 8.9.2 CPU検証結果の総括（2026-09-16、全件合格）

| suite | 件数 | 主対象 |
| --- | ---: | --- |
| `check_dummy_rotation_augmentation` | 19 | 角度導出・回転・seed解決（Step 1） |
| `check_dummy_rotation_augmentation_dataset` | 12 | Dataset・epoch/worker伝達・train CLI配線（Step 2/3） |
| `check_dummy_fixed_list_mode` | 10 + bash 6 | 固定リストモード（Step 3） |
| `check_dummy_rotation_augmentation_ablation` | 13 | R0/R1比較checker・launcher安全性（Step 4） |
| `check_dummy_checkpoint_identity` | 7 | best/last同一性監査（8.7.3） |
| `check_dummy_s5_15_p1_manifest_audit` | 20 | P1監査（実機実行済み） |
| `check_dummy_class_weight_tag`（既存） | bash 8 | `train_stage5.sh`変更による回帰確認 |

**合計: Python 81件 + bash 14件、すべて合格。** 静的チェック（`py_compile`・`bash -n`・
`git diff --check`）も全ファイルで合格。

#### 8.9.3 none時の回帰確認（8.7.2が省略を認めなかった項目）

新規R0を採用してもこの検証は省略していない。確認済みの内容は次のとおり。

1. **Dataset出力のbit一致**: `augmentation_mode="none"`のDataset出力が、
   `normalize_xyz()`+window切り出しのみで計算した参照と**完全一致**（`np.array_equal`）。
2. **配列の同一性**: none時は`rotate_video_points()`が入力配列**そのもの**を返す
   （`rotated is points`）。コピーも回転も行わない。
3. **乱数を消費しない**: none経路の全サンプル取得前後で`np.random.get_state()`が不変。
4. **DataLoader経由**: 2 worker・複数epochで全サンプルの`rotation_angle_degrees`が`None`。
5. **評価側**: train CLI内部の`val_dataset`に`augmentation_mode`を渡さないことをコードで固定し、
   実際に`val_dataset[0]`の角度が`None`であることをテストで確認。

**未確認（実機でのみ可能）**: 同一条件・同一seedでの「改修前コード」と「改修後コードのnone」の
学習結果の一致は、GPU学習を伴うため本環境では確認できない。上記1〜5はDataset出力レベルの
同値性であり、学習全体の同値性を主張するものではない。

#### 8.9.4 固定splitの検算

| 確認事項 | 状態 |
| --- | --- |
| bash→CLI→Datasetの引数経路で保存済みリストがそのまま渡る | **CPU検証済み**（`input_source_args()`とDataset構築の両方） |
| `--train_dir`が同時に渡らない（Python側が排他で拒否） | **CPU検証済み** |
| seedベース再分割が呼ばれない | **CPU検証済み**（`--val_fraction`を渡さない経路） |
| リスト外のH5を追加しても対象が変わらない | **CPU検証済み**（3件追加してbyte一致） |
| 実リスト（162/18）のfingerprintが8.6.1と一致 | **未確認**。実際の学習run後に`train_files.txt`/`val_files.txt`を比較checkerで照合する（内容差と表現差を分離） |

#### 8.9.5 旧W-A best/last監査: 実装完了・実機実行待ち（8.7.3）

承認された「W-A run dir 1件・CPU専用・読取専用・1回」の監査を実装し、合成checkpointで
7件のCPUテストに合格した。**実機では未実施**（W-A run dirへのアクセスが必要なため）。

手順は8.7.3の指示どおり: まずSHA-256を比較し、一致すればファイルとして同一と判断して
**読み込みを行わない**。不一致の場合のみCPU上へ読み込み、epoch・config・state_dictの
キー/shape/dtype/値を比較して、**ファイル同一性と評価モデル同一性を別々に報告**する
（結論は`identical_file` / `different_file_same_evaluation_model` / `different_evaluation_model`）。
run dirは変更せず、出力JSONのみ別ディレクトリへ書く。

```bash
cd /mnt/data/3d_projects/models/Stage5
RUN_DIR=/mnt/data/3d_projects/stage5_runs/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad \
  bash checks/real_h5/check_stage5_checkpoint_identity.sh
```

### 8.10 GPU preflight（2026-09-16、申請・手順改訂・実機3 stage）

#### 8.10.1 GPU preflight申請（8.7.4、実行承認をお願いします）

> **2026-09-16 改訂**: 本節の当初申請は、実施方法に実現不能な点があったため
> **次節「GPU preflight実施手順（改訂版）」で置き換える**。以下は経緯として残す。
> 変更点は「少数動画で`train_stage5.sh`を動かす」という実施方法であり、
> 確認項目・実施量・停止条件・出力先の方針は変えていない。

依頼書5.3節・管理記録P2の範囲に限定した申請である。**フル1 epoch smokeは申請しない。**

**目的:** 実機GPU上で、(a) 有限のloss/gradientとoptimizer更新、(b) epochごとの角度更新が
worker越しに届くこと、(c) 点対応・GT対応が保たれること、(d) 評価経路が無変換であること、
(e) none時に従来経路から回帰していないこと、を確認する。

| 項目 | 申請内容 |
| --- | --- |
| 段階0 | 合成テンソルによるdummy forward/backward（H5不要）。finite loss/gradientとoptimizer更新を確認 |
| 対象動画 | **train 4動画＋validation 1動画**。W-A保存済みリストの**先頭から固定順**で選び、恣意的な選択をしない（動画IDは本書に記載せず、private manifestにのみ残す） |
| step数 | 上記5動画から生成されるwindowのみ。W-Aの実績（162動画→715 window）から**約18 window/epoch**、accumulation 8で**約2 optimizer step/epoch**、2 epochで**計約4〜5 optimizer step**（実測値は実行後に報告） |
| epoch数 | **2 epoch**。1 epochでは「epochごとに角度が変わる」ことを実機で確認できないため |
| 実行回数 | **2 run**（`augmentation=none`で1回、`random_z_rotation`で1回）＋段階0を1回。**計3回** |
| 出力先 | `work_dirs/_s5_15_gpu_preflight/`配下の専用ディレクトリ。**`stage5_runs/`配下の実runディレクトリには一切書かない**。既存成果物を検出したら停止する |
| 保存物 | preflight用config・metrics・角度ログ・検証結果JSON。checkpointは保存するが**P3の比較には使用しない**（別ディレクトリ・別条件のため） |
| 停止条件 | NaN/Infのloss・gradient、GT/点対応の不一致、none時の回帰、**同一epoch内で同一動画に複数の角度**が観測された場合、**epochを進めても角度が変わらない**場合、validation側で角度が`None`以外になった場合。いずれも即停止して報告し、**修正後の再実行は理由・対象・回数を示して別途承認を得る** |
| 実装 | 上記を実行するpreflightスクリプトは**本申請の承認後に実装**する（現時点では未実装）。実装後、CPUで可能な部分を検証してから実機へ渡す |

**申請に含まれないもの:** フル1 epoch smoke、R0/R1の本学習、追加seed・角度探索、
`stage5_runs/`配下への書き込み、production既定値の変更。これらはP3以降の別承認とする。

#### 8.10.2 GPU preflight実施手順（改訂版、2026-09-16）

##### 8.10.2.1 改訂の理由: 少数動画で`train_stage5.sh`を動かすことはできない

実装を詰める過程で、当初申請の実施方法が成立しないことが判明した。`train_stage5.sh`の
teacher preflightは、CVAT動画数58・CVAT frame数2960・XML/crop無効化件数などの**集計値を
対象ファイル集合全体に対して無条件に検証する**（`train_stage5.sh`の`expected_counts`ブロック）。
4〜5動画のサブセットではこれらの集計値が一致せず、学習に到達する前に停止する。

`EXPECTED_CVAT_VIDEOS`等をサブセット用の値へ上書きすれば通過させられるが、それは
**データ健全性を守るための検査そのものを弱めることになる**ため採らない。依頼書8章の
「既存`train_stage5.sh`の一般利用向け設定を実験値で上書きしない」という指示にも反する。

したがって検証を次の3段階に分け、**bash/リスト経路の検証**と**GPU上の少数step検証**を
別の手段で行う。確認項目・実施量・停止条件・出力先の方針は当初申請から変えていない。

##### 8.10.2.2 Stage A: bash/固定リスト経路の検証（GPU不使用・学習なし）

`PREFLIGHT_ONLY=1`と固定リストモードを併用し、**実際の180ファイル**（W-A保存済みリスト）に
対して実行する。集計値がすべて成立する唯一の構成であり、`train_stage5.sh`の実経路を
検証できる唯一の方法でもある。

- 確認: 固定リストモードがteacher preflightを通過すること、preflightの対象が
  **ディレクトリ走査結果ではなくリスト内H5**であること、`INPUT_DIR`を走査しないこと、
  学習を開始せずexit 0すること、起動ログが`input source : fixed lists`を示すこと。
- コスト: 180ファイルの属性読み取り1回。**GPU学習は発生しない。**

##### 8.10.2.3 Stage B: dummy forward/backward（GPU・合成テンソル・H5不要）

監査済み初期checkpoint（SHA-256 `55ec6e6b...438b`）からPointNeXt-S（GroupNorm 8 groups）を
構築し、合成点群で forward → CE loss（固定weight `[0.05963856, 1.94036150]`）→ backward →
optimizer更新を行う。

- 確認: 有限のloss、有限のgradient、optimizer更新後にパラメータが実際に変化すること。
- コスト: 数秒、数stepのみ。

##### 8.10.2.4 Stage C: 実H5少数step（GPU）

| 項目 | 内容 |
| --- | --- |
| 対象動画 | **train 4動画＋validation 1動画**。W-A保存済みリストの**先頭から固定順**で選択し、恣意的に選ばない。動画IDは本書へ記載せずprivate manifestにのみ残す |
| step数 | 末尾の端数accumulationも必ずflushされる実装のため（`train_stage5.py`の`should_step`は`batch_index + 1 == len(loader)`を含む）、1 epochあたり`ceil(window数 / 8)`。W-Aの実績（162動画→715 window、平均4.41 window/動画）から4動画で**約18 window**、すなわち**約3 optimizer step/epoch**。実測値は実行後に報告する |
| epoch数 | **2 epoch**。1 epochでは「epochごとに角度が変わる」ことを実機で確認できない |
| 実行回数 | **2 run**（`augmentation=none`で1回、`random_z_rotation`で1回）。実学習相当は**計約12 optimizer step** |
| 確認項目 | ①有限のloss/gradient ②同一epoch内で1動画に角度が1つ、epoch間では角度が変わる ③none時は角度が記録されない ④点対応・GT・`frame_order`が無変換時と一致 ⑤**validation側が常に無変換** ⑥実データ上でも回転が点間距離を保存する |

Stage B・CはS5-14の診断checker（`check_stage5_coordinate_transform_diagnostics.py`等）と同様に
**専用checkerがモデルを直接呼ぶ**形にする。上記の制約により`train_stage5.sh`経由にはできないため
であり、R0/R1の本学習は従来どおり`train_stage5.sh`経由（`run_stage5_s5_15_arm.sh`から）を維持する。
**学習経路を二重化するものではない**（8.7.1の「Pythonを直接呼ぶ別の学習経路を増やさない」は
本学習の経路に関する指示であり、診断checkerはS5-14で確立した既存の形式に従う）。

##### 8.10.2.5 実施量・出力先・停止条件（当初申請から不変）

| 項目 | 内容 |
| --- | --- |
| GPU実行 | **3回**（Stage B 1回、Stage C 2回）。実学習相当は計約12 optimizer step |
| CPU実行 | Stage A 1回（GPU不使用） |
| 出力先 | `work_dirs/_s5_15_gpu_preflight/`配下のみ。**`stage5_runs/`配下には一切書かない**。既存成果物を検出したら停止する |
| 保存物 | 角度ログ、検証結果JSON、実測step数。Stage B/Cのcheckpointは保存せず、**P3の比較には一切使用しない** |
| 停止条件 | ①NaN/Infのloss・gradient ②点・GT対応の不一致 ③none時の回帰 ④同一epoch内で同一動画に複数の角度 ⑤epochを進めても角度が変わらない ⑥validation側で角度が`None`以外。いずれも即停止して報告し、**修正後の再実行は理由・対象・回数を示して別途承認を得る** |

**含まないもの（当初申請と同じ）:** フル1 epoch smoke、R0/R1の本学習、`stage5_runs/`配下への
書き込み、追加seed・角度探索、production既定値の変更。

##### 8.10.2.6 実行までの手順

1. 本改訂手順のご確認（実施方法の変更点が主）。→ **2026-09-16に了承いただいた。**
2. preflightスクリプトの実装（CPU作業）。→ **完了（下記）。**
3. 実機で実行いただくコマンドの提示。→ **下記。Stage AはGPU不要**のため独立に先行実施できる。

##### 8.10.2.7 preflightスクリプトの実装完了（2026-09-16）

- `Stage5/checks/real_h5/check_stage5_s5_15_gpu_preflight.py` / `.sh`（新規）
- `Stage5/checks/dummy/check_dummy_s5_15_gpu_preflight.py` / `.sh`（新規、**CPU側14件合格**）

実装上の要点:

- **停止条件6項目の判定ロジックをtorch非依存の関数に分離**した（`angle_violations()`・
  `correspondence_violations()`・`none_mode_regression_violations()`・`finite_violations()`）。
  これにより、実機で初めて動かすのではなく、**6条件すべてを合成レコードでCPU検証済み**である。
- **Stage Cのデータ経路は本環境で実際に完走させた**。合成H5 5本（train 4＋validation 1）に対し、
  実Dataset・実DataLoader（worker使用）で2 epoch×2モードを`--skip_optimization`付きで実行し、
  violation 0件、epoch間で角度が変化、動画ごとに異なる角度、none時は全て`None`、
  validation側も`None`であることを確認した。**GPUを要するのはモデルのforward/backwardのみ**である。
- 出力先の安全装置: `stage5_runs/`配下への書き込みと、非空ディレクトリへの書き込みを
  **処理開始前に**拒否する（exit 2）。CPUテストで両方を検証済み。
- 出力はprivate（部分集合の実動画名を含む）とshareable（匿名化・privacy self-check同梱）の2本。
  shareable側に実動画名・絶対パスが出ないことをCPUテストで確認した。

##### 8.10.2.8 実機で実行いただくコマンド

**Stage A（GPU不要・学習なし）**

```bash
cd /mnt/data/3d_projects/models/Stage5
STAGE=a bash checks/real_h5/check_stage5_s5_15_gpu_preflight.sh
```

`train_stage5.sh`が`PREFLIGHT_ONLY=1`で停止し、`input source : fixed lists`と
`matched files : 180`が表示されれば合格。**学習は開始されない。**

**Stage B（GPU、数秒）**

```bash
STAGE=b bash checks/real_h5/check_stage5_s5_15_gpu_preflight.sh
```

**Stage C（GPU、none→random_z_rotationの2回を自動で連続実行）**

```bash
STAGE=c bash checks/real_h5/check_stage5_s5_15_gpu_preflight.sh
```

出力は`work_dirs/_s5_15_gpu_preflight/`配下の`stage_b_dummy_forward_backward/`・
`stage_c_none/`・`stage_c_random_z_rotation/`に、それぞれ
`preflight_private.json`と`preflight_shareable.json`として保存される。
**共有いただきたいのはshareable側とコンソール出力のみ**。
いずれのstageも`status: failed`なら停止して報告し、修正後の再実行は別途承認をいただく。

#### 8.10.3 Stage A 実行結果（2026-09-16、実機、合格）

ユーザーが実機で実行。**合格。学習は開始されていない**（`training was not started`）。

| 確認項目 | 結果 |
| --- | --- |
| 固定リストモードが選択されたこと | `input source : fixed lists (INPUT_DIR not scanned)` |
| リストの件数 | `train=162 val=18 total=180` |
| **リストのfingerprint** | `train_sha256=582579833f345b77` / `val_sha256=0c251380e40f0def`。**8.6.1のP1監査値の先頭16桁と一致** |
| teacher preflightが**リスト内H5**を対象にしたこと | `files=180, videos=180`。ディレクトリ走査ではなくmanifest経由 |
| teacher v7の集計値 | `cvat_videos=58, cvat_frames=2960, invalidated_videos=2, invalidated_frames=7, removed_positive=2124, removed_bbox_rows=7, crop_invalidated_videos=1, crop_invalidated_frames=1, crop_removed_ignore=6, crop_removed_bbox_rows=1`。**すべて既定の期待値と一致** |
| 学習が開始されないこと | `PREFLIGHT_ONLY=1`で停止、`No arm training was started` |
| 初期checkpoint | SHA-256が監査値`55ec6e6b...438b`と一致することを起動時に確認 |

**この結果から言えること:**

1. 固定リストモードが実データで機能し、`INPUT_DIR`を走査せずにリストの内容だけで学習対象が
   決まることを実機で確認した（8.7.1の要求）。
2. teacher preflightが**リスト内の180ファイルを対象として**全集計値を満たした。これは
   「リストの180件＝teacher v7完全データセット」であることの実機確認でもある。
   `videos=180`（重複なし）は、4.3.1節の`stable_video_id`設計の前提が実データでも
   成り立つことを重ねて裏づける。
3. リストのfingerprintがP1監査値と一致した。**ただし照合したのは先頭16桁（64bit）**であり、
   全64桁の照合は学習後に比較checkerが`train_files.txt`/`val_files.txt`に対して行う
   （内容差と表現差を分離した比較）。
4. 改訂手順の前提（「サブセットではteacher preflightの集計値を満たせない」）とは逆に、
   **全180件であれば固定リストモードでも問題なく通過する**ことが実証された。改訂の判断は妥当だった。

**留保:** Stage Aは学習を開始しないため、`train_stage5.py`側の引数受け取り
（`--train_list`/`--val_list`がDatasetまで届くこと）は本stageでは実行されていない。
これはCPUテストで検証済みであり、実機ではStage C（Datasetを直接構築）と、
P3の本学習が保存する`train_files.txt`/`val_files.txt`で確認される。

#### 8.10.4 Stage B 1回目: 失敗（2026-09-16、実機）— preflightスクリプトの実装不具合

**結果: 失敗。preflightスクリプト側のバグであり、学習コード・モデル・データの問題ではない。**

```text
AttributeError: 'dict' object has no attribute 'reshape'
  check_stage5_s5_15_gpu_preflight.py line 211, in run_stage_b
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
```

**根本原因:** preflightスクリプトがモデル出力を生のテンソルと誤って仮定していた。実際には
Stage5のsegmentorは`{"logits": [B, N, C]}`という**dictを返す**
（`pointnext_s_segmentor.py:205`、`mlp_baseline_segmentor.py:149`）。また損失も
`torch.nn.CrossEntropyLoss`を直接呼ぶのではなく、`build_loss()`が返す損失を
`loss_fn(output, batch)`の形で呼び、`loss`/`loss_sum`/`loss_normalizer`を含むdictを受け取るのが
`train_stage5.py`の実経路である（`train_stage5.py:663-664`）。

**なぜCPUテストで捕まらなかったか:** Stage CのCPUテストは`--skip_optimization`で実行しており、
**モデルと損失を呼ぶ経路を一度も通っていなかった**。Stage Bにはそもそもテストが無かった。
「GPUが必要だから実機でしか確認できない」と判断した範囲が広すぎた。

**修正内容:**

1. `build_model_and_loss()`が`build_loss()`を使うようにし、**本番と同一の損失経路**を通すようにした
   （自前のCrossEntropyLossをやめた）。これは不具合修正であると同時に、preflightが検証する
   対象を実経路へ近づける改善でもある。
2. Stage B・Stage Cの両方で`output = model(batch)` → `loss_dict = loss_fn(output, batch)` →
   `loss_dict.get("loss_sum", loss).backward()`という実経路と同じ呼び方に統一した。
3. Stage Bのbatchに`valid_mask`を追加した（損失が`valid_mask`でignoreを反映するため）。
4. **CPUテストを1件追加**（計15件）。CUDA不要の`mlp_baseline`と**実際の**`build_loss()`を使い、
   「segmentorはdictを返す」「`loss_fn(output, batch)`はdictを返す」「その呼び方でbackwardが
   有限のgradientを生む」という契約を固定した。**修正前のコードがこのテストで実際に
   `AttributeError`を出すことも確認済み**であり、同種の取り違えは今後CPU側で検出される。

**実施量の記録:** Stage B 1回を消費した（モデル構築とforwardまで到達し、損失計算で失敗）。
出力ディレクトリ`work_dirs/_s5_15_gpu_preflight/stage_b_dummy_forward_backward/`は
作成されたが**空のまま**であり、既存成果物の検出ガードは空ディレクトリを許容するため、
**再実行前に削除等の操作は不要**である。

#### 8.10.5 Stage B 再実行の承認依頼

| 項目 | 内容 |
| --- | --- |
| 理由 | 上記のpreflightスクリプト実装不具合の修正後確認。学習条件・モデル・データは一切変更していない |
| 対象 | Stage Bのみ（合成テンソル、H5不要） |
| 回数 | **1回** |
| 実施量 | 数秒、optimizer step 1回 |
| 出力先 | 変更なし（`work_dirs/_s5_15_gpu_preflight/stage_b_dummy_forward_backward/`） |

Stage Cは未実行であり、当初申請の2回のまま変更はない。Stage Bの再実行をご承認いただければ、
続けてStage Cへ進みたい。

#### 8.10.6 Stage B 2回目: 合格（2026-09-16、実機）

承認を得て再実行し、**合格**した。

| 項目 | 値 | 評価 |
| --- | --- | --- |
| status | `passed` | violation 0件 |
| loss | `0.6809854507446289` | **有限**。2クラスのchance level `ln(2)=0.693`に近い水準であり、Stage5データを学習していない転移初期重みとして妥当。モデルの良否を示すものではない |
| gradient norm | `240.51934086042826` | **有限**かつ非ゼロ。これは`clip_grad_norm_`適用**前**の値であり、本番の`grad_clip_norm=10`ではclipが効く状態。学習開始時としては想定内 |
| params changed | `63` | optimizer更新で63個のパラメータテンソルが実際に変化した。「更新が素通りしていない」ことの確認 |
| privacy self-check | 2パターンとも検出0件 | 合成テンソルのみのstageのため当然だが、規約どおり確認 |
| 学習非開始 | `No arm training was started` | — |

**この結果で確認できたこと:** 監査済み初期checkpointからGroupNorm 8 groupsのPointNeXt-Sが
実機GPU上で構築でき、**本番と同一の損失経路**（`build_loss()`→`loss_fn(output, batch)`）で
有限のloss・gradientが得られ、optimizer更新がパラメータを実際に動かす。
これはStage Cの前提条件であり、S5-15の改修がモデル計算そのものを壊していないことも示す。

**留保:** Stage Bは合成テンソルのみで、H5・Dataset・DataLoader・augmentationを一切通らない。
角度の検証、点対応、評価無変換の確認はStage Cで行う。

**実施量の累計:** Stage B 2回（1回目失敗・2回目合格）。Stage Cは未実行。

**追記（shareable JSON確認後）:** `num_parameters_with_gradient = 63`であり、
`num_parameters_changed_by_step = 63`と**完全に一致**した。すなわち
**勾配を持ったパラメータはすべて更新された**（更新漏れのあるテンソルは存在しない）。

#### 8.10.7 Stage C: 合格（2026-09-16、実機、none / random_z_rotation の2回）

**両モードとも`status: passed`、`violations: []`。学習は開始されていない。**

| 項目 | none | random_z_rotation |
| --- | --- | --- |
| status | `passed` | `passed` |
| violations | 0件 | 0件 |
| windows/epoch | 16（4動画） | 16（4動画） |
| validation windows | 15（1動画） | 15（1動画） |
| optimizer steps | 4 | 4 |
| forward/backward回数 | 32（16 window×2 epoch） | 32 |
| loss範囲 | 0.6345〜0.7979（全て有限） | 0.6458〜0.7988（全て有限） |
| `skip_optimization` | `false` | `false` |
| 角度 | 全epoch・全動画で`null` | 下表 |

**実測step数と事前見積りの対比:** 事前見積りは「4動画で約18 window→約3 optimizer step/epoch、
2 epochで約6」だったが、**実測は16 window→2 step/epoch、2 epochで4**であった
（`ceil(16/8)=2`）。見積りは平均window数（W-A実績4.41 window/動画）からの概算であり、
実際のこの4動画は4.0 window/動画だった。**申請した実施量の範囲内**である。

**角度の実測値（random_z_rotation、base_seed=500042）:**

| 動画 | epoch 1 | epoch 2 |
| --- | ---: | ---: |
| train_000 | 7.5937 | 1.1971 |
| train_001 | 9.3388 | 1.9204 |
| train_002 | 6.2439 | -14.9730 |
| train_003 | 14.5206 | 4.9637 |

検算した結果:

- **範囲**: 最小`-14.9730`・最大`14.5206`。**全8値が`[-15, +15]`の内側**。
- **動画間の独立性**: 各epoch内で4動画の角度がすべて異なる。
- **epoch更新**: 4動画すべてでepoch 1と2の角度が異なる。**実機・worker 2並列でも
  `set_epoch()`がworkerへ届いている**ことの確認であり、8.3-1が要求した検証項目そのものである。

**violations 0件が意味すること（停止条件6項目すべてを実データで通過）:**

1. loss・gradientが全32 stepで有限（条件①）。
2. 回転後も点対応・GT・`frame_order`・`features`・window境界が無変換時と一致し、
   点間距離が保存された（条件②、最終epochの全16 windowで照合）。
3. none時は**無変換の参照とbit単位で一致**し、角度も記録されなかった（条件③）。
4. 同一epoch内で1動画に複数の角度が観測されなかった（条件④）。
5. epochを進めて角度が固定されたままの動画は無かった（条件⑤）。
6. **validation datasetの全15 windowで角度が`null`**。rotation側の実行でも同様（条件⑥）。

**過度に読み取らないこと:** 両モードのloss範囲はほぼ同等だが、これは**4 optimizer stepの結果**であり、
augmentationの有効性について何も示さない。有効性の判断はP3の5 epoch比較で行う。
またStage Cのcheckpointは保存しておらず、P3の比較には一切使用しない。

**privacy:** 3つのshareable JSONすべてでtimestamp状動画ID・絶対hostパスの検出0件。
実動画名を含むprivate JSONは実機に残され、共有されたのはshareable側のみであることを確認した。

### 8.11 P3の申請・承認・実行・結果（2026-09-17〜2026-09-19）

#### 8.11.1 P3実行申請（2026-09-17、実行承認をお願いします）

GPU preflightが3 stageすべて合格したため、依頼書6.1節「学習開始前に両armのコマンド、manifest、
出力先と予定実施量を提示」に従いP3の実行を申請する。

##### 8.11.1.1 実施量

| 段階 | 内容 | 実施量 |
| --- | --- | --- |
| 学習 | R0（none）・R1（random_z_rotation）を各5 epoch | 715 window/epoch、accumulation 8（tail flush込み）で`ceil(715/8)=90` optimizer step/epoch → **450 step/run、2 runで計900 optimizer step**。~~forward/backwardは7,150回/run~~ **2026-09-17訂正: 正しくは3,575 forward/backward組/run、両arm合計7,150組**（forwardとbackwardを個別に数えれば7,150回/runだが、組数と混同しない）。validation・学習後評価のforwardは含まない |
| 評価 | 固定21動画（train sanity 3＋validation 18）× 2 arm | epoch5（`last.pt`）が主比較。`best.pt`は同一選択規則で副次評価 |
| 比較 | `check_stage5_rotation_augmentation_ablation.sh` | CPUのみ、学習なし |

**重複評価の回避（依頼書6.2）:** 評価に入る前に、各armへ
`check_stage5_checkpoint_identity.sh`（8.7.3で実装済み、CPU・読取専用）を適用して
`best.pt`と`last.pt`の同一性を確認する。**同一なら評価は1 checkpoint/armに絞り、その事実を記録**する。
同一でなければ2 checkpoint/armを評価する。したがって評価対象は
**42動画分（同一の場合）または84動画分（異なる場合）**である。

##### 8.11.1.2 起動コマンド

学習条件はすべて`run_stage5_s5_15_arm.sh`の内部定数であり、呼び出し側で指定するのは
armと実行日だけである。**まずdry runで条件とコマンドを確定し**、その内容をご確認いただいてから
`CONFIRM_TRAINING=1`で起動する。

```bash
cd /mnt/data/3d_projects/models/Stage5

# 1) dry run（学習は起動しない。条件検証とコマンド提示のみ）
EX_DATE=260917 ARM=r0 bash checks/real_h5/run_stage5_s5_15_arm.sh
EX_DATE=260917 ARM=r1 bash checks/real_h5/run_stage5_s5_15_arm.sh

# 2) 承認後の本実行
EX_DATE=260917 ARM=r0 CONFIRM_TRAINING=1 bash checks/real_h5/run_stage5_s5_15_arm.sh
EX_DATE=260917 ARM=r1 CONFIRM_TRAINING=1 bash checks/real_h5/run_stage5_s5_15_arm.sh
```

`EX_DATE`は出力ディレクトリ名に入るだけで、学習条件には影響しない。実行日に合わせて指定する。

**出力先:**

```text
/mnt/data/3d_projects/stage5_runs/260917/pointnext_s_EX260917_s5_15_r0_none_gn8_cwfixed_lr1e3_ep5_bs1_acc8_nopad
/mnt/data/3d_projects/stage5_runs/260917/pointnext_s_EX260917_s5_15_r1_random_z_rotation_gn8_cwfixed_lr1e3_ep5_bs1_acc8_nopad
```

arm別に分かれており、既存run dirは一切上書きしない。非空ディレクトリを検出した場合は
起動前に停止する。旧W-A（EX260914）は履歴として保持し、触れない。

##### 8.11.1.3 manifestとして記録される内容

起動時のdry run出力（条件一覧＋解決済みコマンド）と、各runの`config.json`が実質のmanifestとなる。
加えて比較checkerが初期checkpoint hash・file list fingerprint・実効weightを照合して記録する。

**ただし依頼書4章が求める「コードrevisionと未コミット差分の有無」について、現状を明示しておく。**

| 項目 | 現状 |
| --- | --- |
| HEAD | `a6b7c7f`（S5-14完了時のコミット） |
| 作業ツリー | **S5-15の実装は全て未コミット**（~~変更5ファイル＋新規20ファイル~~ **2026-09-17訂正: 変更5＋新規25**、計30エントリ） |

つまり**このままP3を実行すると、「HEADからの未コミット差分で学習した run」になる**。
依頼書は「未コミット差分の有無」の記録を求めており、有無を正直に記録すれば要件は満たせる。
一方、採用候補となりうる比較の再現性を考えると、**P3開始前にS5-15実装をコミットしておく方が
run と code revision が1対1で対応し、後から「どのコードで学習したか」を確定できる**。

**ご判断をお願いしたい:**

- (A) **P3開始前にS5-15実装をコミットする**（推奨）。commit hashがmanifestの`code revision`となる。
  コミット対象はコードと文書のみで、`.tmp/`・生データ・checkpoint・PLYは含めない（依頼書8章）。
- (B) 未コミットのまま実行し、manifestに「HEAD `a6b7c7f` ＋未コミット差分あり」と記録する。

いずれの場合も、コミットは**ご指示があった場合のみ**行う。

##### 8.11.1.4 停止条件と実施境界

| 項目 | 内容 |
| --- | --- |
| 停止条件 | NaN/Inf、点対応・GT対応の破損、出力先の既存成果物検出。いずれも即停止・報告し、再実行は理由・対象・回数を示して別途承認を得る |
| 今回に含まれないもの | 追加seed・角度探索・3本目のrun、P4以降の延長、resume経路の実装、threshold sweep、aggregation変更、production既定値変更 |
| 採否判断 | 比較checkerは4基準を`satisfied`/`not_satisfied`/`requires_judgment`で返すのみで、**採用判定は出力しない**。最終判断は管理チャットに委ねる |
| PLY所見 | 評価パイプラインがPLYを出力する。**目視確認を実施しない場合は「未実施」と記録**する（依頼書6.2） |

#### 8.11.2 管理チャット返信: P3実行承認（2026-09-17）

**コミット確認:** workspaceで次のcommitを確認しました。

```text
7abd3693561de8fead05f06e22bfe31e5fa79ea6
Add S5-15 rotation augmentation and fixed-split comparison tooling
```

`git status --short`は空で、確認時点の非ignored作業ツリーに変更はありません。
対象はコードと文書の30ファイルです。当時、本書はignoredな`.tmp/`内の管理記録であり、コミットには
含まれません。実機で使用するcheckoutも別途確認し、workspaceの確認を実機の確認と代用しないでください。

**承認判断:** P2・GPU preflightの合格報告を受け入れ、申請された**新規R0/R1各5 epochの学習と、
固定21動画の評価・CPU比較を承認します**。以下の開始条件と実施上限を守ってください。
この返信時点で管理チャットが学習を実行したわけではありません。

##### 8.11.2.1 申請の訂正と実施量

1. コミット直前の変更数は**変更5＋新規25＝30ファイル**です。「新規20」の記載を訂正します。
2. train715 window×5 epochは**3,575 forward/backward組/run**、両arm合計7,150組です。
   forwardとbackwardを個別に数えるなら7,150回/runですが、組数と混同しないでください。
   accumulation8・各epoch末tail flushを前提とする予定optimizer更新は90/epoch、450/run、
   両arm900です。skip等が生じた場合の実更新数は別途報告し、予定値と一致したとみなさないでください。
   これらにvalidationや学習後評価のforwardは含みません。
3. 学習後評価は固定21動画×両armのlastを主比較とし、bestはモデル同一性を確認して重複を避けます。
   両armともbest/last同一なら42動画条件、片armだけ異なれば63、両armとも異なれば84です。
   **最大84動画条件**を承認範囲とします（各動画のwindow数だけforwardが必要）。
4. 各armのbest/last同一性確認はCPU読取専用で各1回、比較checkerはCPUで1回を基本とします。
   ファイルhash不一致だけで別モデルとせず、epoch・モデル設定・state_dict等も必要に応じ確認します。

##### 8.11.2.2 学習開始前の必須確認

- 実機で上記commitと作業ツリー状態を確認し、R0/R1の途中でコード・環境を変更しない。
- 両armをまずdry runし、同一初期重みhash、固定train/val listの内容・順序・hash、固定weight、
  GroupNorm8、batch1/accum8、window16/8、seed42、5 epochs、SAVE_EVERY=1を確認する。
- R0はnone、R1はrandom_z_rotation・±15度・解決済みseed500042とし、意図しない環境変数の
  上書きがないことを確認する。validation・評価は無変換、sampling/paddingなしを維持する。
- 新規のarm別出力先で旧runとの非混在を確認する。dry run出力を保存し、実効条件が申請と一致した
  場合に限り`CONFIRM_TRAINING=1`で実行してよい。相違があれば停止・相談する。

dry runが条件に一致すれば、この承認の範囲内で本実行へ進めます。形式的に同じ承認を再申請する
必要はありません。失敗後の再実行・条件変更は別承認です。

##### 8.11.2.3 各runへ保存するmanifest

**コミットだけではrevisionは自動記録されません。** R0/R1それぞれの実行開始時に、次をprivateな
run記録へ保存してください。config.jsonだけに存在しない項目も補います。

- 実際の`git rev-parse HEAD`、`git status --short`、記録時刻、起動コマンド・dry run出力。
- Python実行ファイル、Python/torch/numpy/CUDA runtime・GPU/driver等の実行環境。
- 初期checkpoint hash、入力file list fingerprint、解決済み設定・augmentation seed、出力先。
- 実施したepoch・optimizer更新・評価checkpoint数と、承認範囲との差分。

非ignoredな差分が発生していれば「clean」と記録せず、学習関連差分を解消または相談してから実行します。
manifest作成で新規run dirを非空にすると既存保護により起動が拒否されるため、開始時の記録は
別のprivate記録先へ保存し、launcherが正常に作成したrun dirへ後で関連付ける等の方法を使ってください。
このために出力先保護を無効化しないでください。共有用にはパス・動画ID等を匿名化します。

##### 8.11.2.4 停止・報告・非対象

NaN/Inf、点/GT対応破損、split/初期重み不一致、既存出力混在は停止・報告します。
中断・失敗後の再実行は理由・対象・回数を示して別承認を得てください。
追加seed・角度探索・3本目の学習・P4以降・resume・threshold tuning・production変更は未承認です。
P3終了後はepoch5を主比較、bestを副次比較としてsplit別指標・PLY所見（未確認なら未確認）・
実施量・manifestを報告し、R0維持/R1暫定採用と次段階の判断を管理チャットへ返してください。

#### 8.11.3 承認条件への対応（2026-09-17）

P3承認の条件を反映した。訂正2件と、実装追加2件、そして**実行前に解消が必要な論点1件**がある。

##### 8.11.3.1 報告の訂正（ご指摘どおり、該当箇所に訂正注記を入れた）

1. コミット直前の変更数は**変更5＋新規25＝30ファイル**。「新規20」は誤りであった。
2. train 715 window×5 epochは**3,575 forward/backward組/run、両arm合計7,150組**。
   「7,150回/run」は組数と回数を混同した記載であった。optimizer更新の予定値
   （90/epoch、450/run、両arm900）は変更なく、実更新数は実行後に別途報告し予定値と
   一致したとはみなさない。validation・学習後評価のforwardは含まない。

##### 8.11.3.2 実装追加1: dry runでのlist fingerprint「照合」（従来は表示のみ）

承認条件は「dry runで固定train/val listの内容・順序・hashを確認する」ことを求めている。
従来の`run_stage5_s5_15_arm.sh`は期待値を**表示するだけで照合していなかった**ため、
実際に検証するよう修正した。

- `stage5/utils/file_list_mode.py`のCLIに`--expected_train_sha256`/`--expected_val_sha256`を追加。
  不一致なら**manifestを書かずにexit 2**で停止する。
- 不一致時は**content hash（パス表記込み）とidentity hash（ファイル名のみ）を併記**し、
  「動画・順序が違う」のか「パス表記だけが違う」のかを切り分けられるメッセージを出す。
- launcherが両リストの期待値を渡すようにした。実データでの合成テストで、
  一致時は通過し、不一致時はexit 2で停止することを確認済み。

##### 8.11.3.3 実装追加2: 各runのmanifest（新規、承認条件の必須項目）

「コミットだけではrevisionは自動記録されない」というご指摘に対応し、
`checks/real_h5/write_stage5_s5_15_run_manifest.py`を新設した。launcherが
**dry runと本実行の両方で**自動的に呼ぶ。

記録内容は承認条件の列挙に対応する:

| 承認条件の項目 | 記録内容 |
| --- | --- |
| `git rev-parse HEAD`、`git status --short`、記録時刻 | そのまま記録。**dirty時に`clean`と記録しない**。git情報が読めない場合は`worktree_clean: null`とし、cleanへ丸めない |
| 起動コマンド・dry run出力 | 解決済みコマンド全文を記録 |
| Python実行ファイル、Python/torch/numpy/CUDA runtime・GPU/driver | `python_executable`、各バージョン、cuDNN、GPU名・capability、`nvidia-smi`のdriver情報 |
| 初期checkpoint hash、file list fingerprint、解決済み設定・augmentation seed、出力先 | 初期checkpointのSHA-256・サイズ、両リストのcontent/identity hashと件数、`resolved_augmentation_seed=500042`を含む設定一式、出力先パス |
| 実施epoch・optimizer更新・評価checkpoint数と承認範囲との差分 | **予定値**（450 step/run）を`planned`として記録し、「実測値と一致したとみなさない」旨を明記。実測値は実行後の報告で扱う |

**出力先保護との両立（ご指摘の制約）:** manifestは**run dirの外**
（`work_dirs/_s5_15_p3_manifests/`）へ書く。run dirを非空にして既存保護に弾かれることを避けるためで、
**保護の無効化は行っていない**。manifestにはrun dirのパスを記録するので後から対応づけられる。
dry run分と本実行分はタイムスタンプ付きの別ファイルとして共存する。
private版とshareable版（パス・コマンド・git status本文を除去、hashとdirtyフラグは保持）を出力し、
shareable版にはprivacy self-checkを同梱する。

CPUテスト11件を追加し、全件合格（dirty/clean/非repoの3状態、run dir非接触、
dry_run/trainingの共存、匿名化と非破壊性を含む）。

##### 8.11.3.4 追加コミット完了（2026-09-17）

下記の論点は解消済み。ユーザーが追加コミットを実施した。

```text
646cb52b3507e174c996c11d6889d6d0f549dfe5
Add S5-15 run manifests and refine fixed-list launch checks
  6 files changed, 738 insertions(+), 5 deletions(-)
```

workspaceでの確認結果:

- `git rev-parse HEAD` = `646cb52b3507e174c996c11d6889d6d0f549dfe5`
- `git status --short`は**空**（非ignoredな差分なし）
- コミット対象は6ファイル（コード5＋`docs/stage5/FILES.md`）。`.tmp/`・生データ・checkpoint・PLYは含まない
- この状態で全CPUスイート再実行（**Python 108件＋bash 14件、全件合格**）

**留保:** 上記はworkspaceでの確認であり、**実機checkoutの状態確認の代用にはならない**
（承認条件の明示的な指示）。実機側のHEADと作業ツリーは、dry run時にmanifestが
自動的に記録する（`worktree_clean`がfalseまたはnullなら、そこで停止して相談する）。

##### 8.11.3.5 （解消済み）実行前に解消が必要だった論点: 追加コミット

上記の実装追加（list照合・manifest・launcher変更）は**学習経路に関わる変更**であり、
現時点で未コミットである。実際にlauncherのdry runを動かすと、manifestが
`worktree: 5 non-ignored change(s) -- NOT clean`と正しく記録する。

承認条件は「非ignoredな差分が発生していれば『clean』と記録せず、**学習関連差分を解消または相談してから
実行します**」としている。したがって**P3実行の前にこれらをコミットする必要がある**。
コミットはご指示があった場合のみ行う。

対象は次の5ファイル（いずれもコード。`.tmp/`・生データ・checkpoint・PLYは含まない）:

- `Stage5/stage5/utils/file_list_mode.py`（fingerprint照合オプション追加）
- `Stage5/checks/real_h5/run_stage5_s5_15_arm.sh`（照合の実行、manifest呼び出し、解決済みseed表示）
- `Stage5/checks/real_h5/write_stage5_s5_15_run_manifest.py`（新規）
- `Stage5/checks/dummy/check_dummy_s5_15_run_manifest.py`／`.sh`（新規）
- `Stage5/checks/dummy/check_dummy_fixed_list_mode.py`（照合のテスト追加）
- `docs/stage5/FILES.md`（記載追加）

#### 8.11.4 dry run実行結果（2026-09-18、実機、両arm）

両armをdry runし、**学習は起動していない**（`Dry run: training was NOT started.`）。
承認条件「学習開始前の必須確認」の各項目を実ログと照合した結果は次のとおり。

| 承認条件の確認項目 | R0 | R1 | 判定 |
| --- | --- | --- | --- |
| 実機のcommit | `646cb52b3507e174c996c11d6889d6d0f549dfe5` | 同左 | 一致 |
| 実機の作業ツリー | `clean (no non-ignored changes)` | 同左 | **実機で直接確認**（workspace確認の代用ではない） |
| 同一初期重みhash | `55ec6e6b...438b` | 同左 | 監査値と一致、両arm同一 |
| 固定train/val listの内容・順序・hash | `train=162 val=18 total=180`、`fingerprints verified against the expected values` | 同左 | **照合が実行され通過**。両armとも同一のlistパスを使用 |
| 固定weight | `0.05963856,1.94036150` | 同左 | 一致 |
| GroupNorm8 | `groupnorm (8 groups)` | 同左 | 一致 |
| batch1/accum8・window16/8 | `train_stage5.sh`の既定値（dry runの表示対象外） | 同左 | 実配線はP2でCPU検証済み。config.jsonで事後確認する |
| seed42 | `42` | 同左 | 一致 |
| 5 epochs / SAVE_EVERY=1 | `epochs: 5 (save_every=1, every epoch kept)` | 同左 | 一致 |
| augmentation | `none` | `random_z_rotation` / `±15.0` | **意図どおり。armの唯一の差** |
| 解決済みaugmentation seed | `500042` | `500042` | 一致（R0では使用されない） |
| 意図しない環境変数の上書き | コマンド中のenv varは想定した15項目のみ | 同左 | 余分な上書きなし |
| arm別出力先・旧runとの非混在 | `.../260917/..._r0_none_...` | `.../260917/..._r1_random_z_rotation_...` | 別ディレクトリ。旧W-A（260914）とも別 |
| manifest | `manifest_r0_dry_run_20260918T230915.json` | `manifest_r1_dry_run_20260918T230916.json` | run dirの外へ出力。private/shareableの2本 |

**「validation・評価は無変換、sampling/paddingなし」について:** これはdry runの表示項目ではなく、
構造的に担保している。`train_stage5.py`は`val_dataset`構築時に`augmentation_mode`を渡さない
（既定`none`のまま）ため、設定で有効化する余地がない。CPUテストとGPU preflight Stage C
（validation全15 windowで角度`null`）で実データ確認済み。

**記録しておく軽微な不一致:** `EX_DATE=260917`を使用したが、dry runの実施は2026-09-18
（manifestのタイムスタンプ`20260918T230915`）である。したがって**出力ディレクトリ名の日付
（260917）と実際の実行日（260918）が1日ずれる**。`EX_DATE`はディレクトリ名のラベルにのみ影響し、
学習条件には一切影響しない。本実行でも**dry runで検証した条件と完全に一致させるため
`EX_DATE=260917`を維持する**方針とし、実際の実行時刻はmanifestとconfig.jsonが記録する。
日付ラベルを揃える方を優先する場合は`EX_DATE=260918`へ変更できるが、その場合は出力先が
dry run検証時と変わるため、変更後に再度dry runを行ってから本実行する。

#### 8.11.5 P3学習完了とhistory.json解析（2026-09-18、実機）

両armとも5 epochを完走した。**以下はすべて`train_stage5.py`学習ループのwindow単位running metrics
であり、公式比較に使う`evaluate_stage5.py`のmean-probability H5単位評価ではない**（S5-13で確立した
区別）。採否の判断はここではできない。

##### 8.11.5.1 実施量と条件の同一性（config.json・history.jsonで確認）

- **optimizer更新: 両arm 90/epoch×5＝450、合計900。申請した予定値と完全一致**（差分ゼロ）。
- 715 sample/epoch、padding 0、`class_weight_mode=manual`・`[0.05963856, 1.9403615]`、
  accumulation 8、label smoothing 0 — **debug固定条件が両armで完全一致**。
- config.jsonで確認: `epochs=5`、`groupnorm`/8 groups、`window_mode=overlap` 16/8・tailあり、
  `batch_size=1`、`gradient_accumulation_steps=8`、`seed=42`、`label_policy=bbox_noncontour_ignore`、
  `num_train_files=162`/`num_val_files=18`/`num_train_samples=715`、
  **`train_list`設定済み・`train_dir`未設定・`val_fraction=0.0`・`max_*=0`**（固定リストモードが
  実学習でも機能した）。
- `augmentation_info`: R0は`mode=none`、R1は`mode=random_z_rotation`。**両arm共通で
  `base_seed=500042`・`seed_source=train_seed_plus_500000`・角度導出式が記録された**（回答2の要求）。
- 全metricsが有限。**TP=0のepochは両armとも皆無**（S5-13のW-Bで起きたpositive予測の完全崩壊は
  再現していない）。

##### 8.11.5.2 window単位valの推移

| ep | loss R0/R1 | F1 R0/R1 | IoU R0/R1 | recall R0/R1 | FP R0/R1 |
| ---: | --- | --- | --- | --- | --- |
| 1 | 0.559 / 0.551 | 0.040 / 0.040 | 0.020 / 0.021 | 0.064 / 0.237 | 0.30M / 1.50M |
| 2 | 0.550 / 0.526 | 0.046 / 0.055 | 0.024 / 0.029 | 0.432 / 0.352 | 2.47M / 1.61M |
| 3 | 0.512 / 0.511 | 0.061 / 0.058 | 0.031 / 0.030 | 0.361 / 0.344 | 1.50M / 1.49M |
| 4 | 0.504 / 0.504 | 0.059 / 0.059 | 0.031 / 0.030 | 0.494 / 0.477 | 2.16M / 2.11M |
| 5 | 0.492 / 0.495 | 0.069 / 0.074 | 0.036 / 0.038 | 0.449 / 0.338 | 1.64M / 1.11M |

両armともtrain/val lossは単調減少。epoch 5ではR1が**FP 32%減・precision/F1/IoUがやや高い一方、
recallは低い**という形になっており、S5-14の回転診断で観測したトレードオフと同じ向きである。

**しかしこの差は現時点で解釈できない。** epoch間の振れ幅と比較すると、
**armの差はすべて各armが1 epochで動く幅より小さい**。

| 指標 | epoch5のarm差 | arm内の最大epoch変動 |
| --- | ---: | ---: |
| F1 | 0.0045 | 0.0152 |
| IoU | 0.0024 | 0.0081 |
| recall | 0.111 | 0.368 |
| FPR | 0.043 | 0.178 |

単一seed・5 epochでこれだけ振れる指標から方向性を主張しない。判断は固定21動画の公式評価で行う。

##### 8.11.5.3 best checkpointの予測

`best_metric=iou_femur`のval値は**両armともepoch 5が最大**（R0: 0.0203→0.0235→0.0312→0.0305→
0.0358、R1: 0.0205→0.0285→0.0299→0.0301→0.0382）。`last.pt`は毎epoch上書き、`best.pt`は
`score >= best_score`で上書きされる実装のため、**両armとも`best.pt`と`last.pt`はepoch 5の保存**
になっているはずである。確認は`check_stage5_checkpoint_identity.sh`で行う。

#### 8.11.6 不具合報告: `SAVE_EVERY=1`が`train_stage5.sh`に無視されていた

**各epoch checkpointが保存されていない。**実機の`ls -1 *.pt`は両armとも`best.pt`と`last.pt`のみ。

**根本原因:** `train_stage5.sh`の変数定義のうち、環境変数の上書きを受け付けるのは
`VAR="${VAR:-default}"`形式で書かれたものだけである。`SAVE_EVERY`は**`SAVE_EVERY=10`という
素の代入**（176行目）であり、launcherが渡した`SAVE_EVERY=1`を**上書きして捨てていた**。
config.jsonにも`save_every: 10`と記録されている。`epochs=5`で`epoch % 10 == 0`が成立しないため、
`checkpoint_epoch_XXXX.pt`は一度も保存されなかった。

同じ形式の変数は他に`SEED=42`（174行目）、`GRAD_CLIP_NORM=10`（175行目）、`NUM_WORKERS=4`
（172行目）がある。**今回は要求値とたまたま一致していたため実害は出ていない**が、
`SEED`が無視される構造は今後seedを変える場合に静かに誤る危険がある。
`EPOCHS`・`CLASS_WEIGHT`・`POINTNEXT_NORM`・`LABEL_POLICY`・`AUGMENTATION`等は
overridable形式であり、config.jsonのとおり正しく反映されている。

**これは私のlauncher設計の確認漏れである。** 追加した`AUGMENTATION`系は
overridable形式にしたが、既存変数が同じ形式かを確認していなかった。P2のCPUテストは
`input_source_args()`の出力（launcher側）までしか検証しておらず、
`train_stage5.sh`が env を実際に採用するかを検証していなかった。

##### 8.11.6.1 影響評価

| 項目 | 影響 |
| --- | --- |
| **主比較（epoch 5）** | **影響なし。** `last.pt`は両armともepoch 5であり、比較に必要なcheckpointは揃っている |
| 副次比較（best） | 影響なし。上記のとおりbestもepoch 5 |
| epoch別の**指標** | 影響なし。history.jsonに5 epoch分すべて残っている |
| epoch別の**重み** | **失われた。** 中間epochのcheckpointから再評価・分岐することはできない |
| arm間の公平性 | **影響なし。両armに同一に作用**しており、比較にバイアスは入らない |
| P1固定条件との差 | 「保存: 各epoch checkpoint」を**満たしていない**。条件からの逸脱として記録する |

**実装チャットの見解（ご判断をお願いします）:** P3の主比較・副次比較はいずれも成立するため、
**このまま評価へ進むことを提案する**。中間epochの重みは、P3の判定にも依頼書6.2の報告項目にも
使用しない。再学習には900 optimizer stepの追加が必要で、得られるのは「P3の判断に使わない
中間重み」である。ただしこれは条件逸脱であり、再実行のご判断は管理チャットに委ねる。

**再発防止（実施はご指示後）:** `train_stage5.sh`の`SAVE_EVERY`・`SEED`・`GRAD_CLIP_NORM`・
`NUM_WORKERS`を`"${VAR:-default}"`形式へ変更し、既定値は現行のまま維持する（一般利用の挙動は
不変）。あわせて、launcherが渡した各env varが`train_stage5.py`へ実際に到達することを
検証するCPUテストを追加する（今回欠けていた検証である）。

#### 8.11.7 manifest確認結果（2026-09-18/19、実機、4件すべて）

先に「training manifestが出力されていない」と報告したが、**私の探索場所の問題であり出力されていた**。
実機の`work_dirs/_s5_15_p3_manifests/`に dry_run×2・training×2 の計4件（private/shareable各々）が
存在する。共有されたshareable 4件を照合した結果:

| 確認項目 | 結果 |
| --- | --- |
| git HEAD（4件すべて） | `646cb52b3507e174c996c11d6889d6d0f549dfe5`。**学習起動時点で実機のcommitが確定** |
| 作業ツリー（4件すべて） | `worktree_clean: true`、`num_dirty_entries: 0`。**学習起動時点でclean** |
| 初期checkpoint | `55ec6e6b...438b`、3,199,662 bytes。**両arm同一** |
| train list | count 162、content `582579833f...`。**両arm同一かつ8.6.1のP1監査値と一致（全64桁）** |
| val list | count 18、content `0c251380e4...`。**同上** |
| seed / epochs / class weight / groupnorm 8 | 両arm同一 |
| 解決済みaugmentation seed | 両arm`500042` |
| **armで異なるべき唯一の項目** | `augmentation`: R0=`none` / R1=`random_z_rotation`。**ここだけが異なる** |
| 実行環境（両arm同一） | Python 3.11.15、torch 2.7.1+cu128、CUDA 12.8、cuDNN 90701、numpy 2.2.2、RTX 5090、driver 580.173.02 |
| privacy self-check | 4件すべて検出0件 |

学習の所要時間は、R0がtraining manifest 23:13:31、R1が翌00:30:49に記録されていることから
**R0が約77分**であったと読み取れる。

**private版は不要である。** shareable版から落ちているのは`output_dir`・`command`・
`git.status_short`・各パス・`python_executable`だが、前2者はdry runログで既知、
`status_short`は`worktree_clean: true`／`num_dirty_entries: 0`で内容が確定している（cleanなので空）、
パス類は照合に不要である。

##### 8.11.7.1 新たに判明した記録上の欠陥: manifestは「意図値」であって「実効値」ではない

manifestの`settings.save_every`は**`1`**と記録されているが、`config.json`の実効値は**`10`**である。
manifestはlauncherが渡そうとした値を記録しており、`train_stage5.sh`が実際に採用した値ではない。

**つまりmanifestは、前節のSAVE_EVERY不具合を検出できないどころか、
「per-epoch checkpointが有効だった」と誤って裏づける記録になっていた。**
config.jsonと突き合わせて初めて食い違いが判明した。manifestの他の項目
（seed・epochs・class weight・norm）は今回たまたま実効値と一致しているが、
**同じ構造的リスクがある**（overridable形式でない変数が今後増えれば同様に食い違う）。

**再発防止に追加すべき項目（実施はご指示後）:** 前節の`train_stage5.sh`修正に加えて、
**学習後にmanifestの意図値とrunの`config.json`の実効値を突き合わせ、
不一致を`FAIL`として報告する検査**を比較checkerへ追加する。今回のような
「渡したつもりが届いていない」種類の逸脱は、この突合でしか捕まらない。

#### 8.11.8 再発防止の実施（2026-09-19、ユーザー指示により実施）

P3の再学習は行わず評価へ進む方針のもと、不具合の再発防止のみ先に実施した。
**R0/R1の学習は完了済みで、評価は`evaluate_stage5.sh`を使うため、`train_stage5.sh`の修正は
残るP3工程に影響しない。** 学習を生成したcommitは各manifestに`646cb52b...`として固定済みである。

##### 8.11.8.1 修正1: `train_stage5.sh`の変数をoverridable形式へ

`SAVE_EVERY`・`SEED`・`GRAD_CLIP_NORM`・`NUM_WORKERS`、および**新たに判明した`PYTHON`**を
`VAR="${VAR:-default}"`形式へ変更した。**既定値は現行のまま**であり、一般利用の挙動は変わらない。

`PYTHON`は当初の調査で見落としていたもので、下記の新規テストが検出した。今回は実機の値が
launcherの値と同一だったため実害はない。

##### 8.11.8.2 修正2: env passthrough検査（新規、今回の不具合そのものを固定）

`checks/dummy/check_dummy_launcher_env_passthrough.py`（CPU、shellのテキスト解析のみ）を追加。
**「launcherがexportする全env varは`train_stage5.sh`に採用されねばならない」**という不変条件を
検査する。現在16個すべてがoverridableであることを確認した。

このテストは次も検証する: 修復した4変数が**元の既定値を保っている**こと（一般利用の非変更）、
分類器自体が2つの代入形式を区別できること（全部overridableと誤判定する分類器なら壊れた
スクリプトも通してしまうため）、そして**旧`SAVE_EVERY=10`形式に戻すとこの検査が落ちる**こと。

##### 8.11.8.3 修正3: manifest意図値 vs config実効値の突合（比較checkerへ追加）

`check_stage5_rotation_augmentation_ablation.py`に`--manifest_r0`/`--manifest_r1`を追加し、
manifestの`settings`とrunの`config.json`を突き合わせる。**不一致の重大度を項目で分ける**:

| 区分 | 対象 | 判定 |
| --- | --- | --- |
| 比較可能性に影響する | `augmentation`・`augmentation_rotation_degrees`・`seed`・`epochs`・`pointnext_norm`・`pointnext_norm_groups` | **FAIL** |
| 生成物のみに影響する | `save_every` | **JUDGE**（人の判断に委ねる） |

今回の`save_every` 意図1／実効10はJUDGEとして機械的に記録される。比較そのものは無効化しない、
という評価と一致する。manifestの`worktree_clean`も検査し、`false`はJUDGE、`null`はUNKNOWNとして
**決してPASSへ丸めない**。

CPUテスト4件を追加（計17件）。今回の実ケース（意図1/実効10）をJUDGEとして検出すること、
seed・epochs・augmentation等の不一致はFAILとなることを含む。

**検証結果:** 全スイート再実行で合格（**Python 116件＋bash 14件**）。既存の
`check_dummy_class_weight_tag.sh`（`train_stage5.sh`の別関数）と`input_source_args`の
bashテストも再実行し、`train_stage5.sh`修正による回帰がないことを確認した。

#### 8.11.9 best/last同一性確認の結果（2026-09-19、実機、CPU各1回）

承認された「各armのbest/last同一性確認をCPU読取専用で各1回」を実施した。
**結果は8.7.3が警告したケースそのものであった。**

| 項目 | R0 | R1 |
| --- | --- | --- |
| `file_identical`（SHA-256） | **false** | **false** |
| ファイルサイズ | 9,578,177 bytes（両ファイル同一） | 9,578,241 bytes（両ファイル同一） |
| epoch | 5 / 5 | 5 / 5 |
| `best_score` | 0.03579247370362282（一致） | 0.038233496248722076（一致） |
| config | 一致 | 一致 |
| state_dict | 63キー、**値・shape・dtypeの不一致0件** | 同左 |
| `evaluation_model_identical` | **true** | **true** |
| 結論 | `different_file_same_evaluation_model` | 同左 |

**hashだけで判断していたら「別モデル」と誤り、評価を84動画条件へ倍増させていた。**
8.7.3の「ファイルhash不一致だけで別モデルとせず、state_dict等も必要に応じ確認する」という
指示がそのまま効いた形である。

##### 8.11.9.1 バイト差の原因（本環境で再現実験して確定）

`torch.save`は**保存先ファイル名をZIPアーカイブ内部の名前として埋め込む**。
同一オブジェクトを`best.pt`と`last.pt`へ保存すると、アーカイブ内のエントリが
`best/data.pkl`と`last/data.pkl`になり、**サイズは同一のままバイト列だけが変わる**。
本環境のtorch 2.7.1で再現実験し、同一オブジェクト→同サイズ・異なるhashとなることを確認した。
実機のcheckpointが「サイズ一致・hash不一致・テンソル完全一致」であることと整合する。

これはStage5固有の問題ではなく`torch.save`の一般的な性質であり、
**今後もcheckpointのhash比較だけで同一性を論じてはならない**という一般則として記録する。

なお、R0とR1のファイルサイズが64 bytes異なるのは、config内の文字列長の差
（`augmentation`が`none`と`random_z_rotation`、`augmentation_info.mode`、出力先名）によるものと
考えられる。実害はない。

##### 8.11.9.2 評価実施量の確定

両armとも`best.pt`と`last.pt`は**評価モデルとして同一のepoch 5**である。したがって
依頼書6.2「epoch5とbestが同一なら重複評価を避け、その事実を記録する」に従い、
**評価は各arm 1 checkpoint（`last.pt`）に絞る**。

**実施量: 21動画 × 2 arm × 1 checkpoint = 42動画条件。承認上限84の半分。**

#### 8.11.10 固定21動画評価の結果（2026-09-19、実機、42動画条件）

両armとも`evaluate_stage5.sh`が`Stage5 evaluation passed`で完了。train sanity 3・validation 18、
window metrics 93、checkpointは`last`のみ（best/last同一のため）。

##### 8.11.10.1 入力の健全性（解析前の確認）

- 共有された2つの`anonymized_metrics_SHARE_THIS`は**全ファイルが相違**しており、
  同一ディレクトリの取り違えではない（ご懸念への回答）。
- `training_config_anonymized.json`で**armのラベルが正しい**ことを確認
  （R0=`none`、R1=`random_z_rotation`、ともにepochs 5・seed 42・groupnorm）。
- 評価対象は両arm**同一の21 (split, alias) 組**。評価checkpointはどちらも`last`のみ。
- **GT・点集合が両armで完全一致**（`valid_positive_count`・`valid_background_count`・
  `total_point_count`・`ignore_point_count`・`num_windows`）。同一データ上の比較であることを確認した。
- 共有ディレクトリへのprivacy scanで、timestamp状動画ID・絶対hostパスの検出は0件。

##### 8.11.10.2 pooled（点数加重、合算TP/FP/TN/FNから導出）

| split | 指標 | R0 | R1 | 差分 |
| --- | --- | ---: | ---: | ---: |
| validation | precision | 3.66% | 4.15% | **+0.49pt** |
| validation | **recall** | 45.08% | 32.41% | **-12.67pt** |
| validation | FPR | 12.53% | 7.91% | **-4.63pt** |
| validation | F1 | 6.77% | 7.36% | +0.59pt |
| validation | IoU | 3.51% | 3.82% | +0.32pt |
| train sanity | precision | 6.94% | 9.84% | +2.89pt |
| train sanity | **recall** | 70.30% | 60.92% | **-9.38pt** |
| train sanity | FPR | 14.88% | 8.82% | -6.06pt |
| train sanity | F1 | 12.64% | 16.94% | +4.30pt |
| train sanity | IoU | 6.75% | 9.25% | +2.51pt |

**ignore領域のpositive率（valid GT上のFPとは別集計）**: validation 40.08%→28.62%、
train sanity 66.94%→21.99%。いずれもR1で低下した。

##### 8.11.10.3 動画別（動画等重み）

| split | 指標 | median-of-diffs | diff-of-medians |
| --- | --- | ---: | ---: |
| validation | F1 | **+0.03pt** | +1.04pt |
| validation | IoU | +0.02pt | +0.55pt |
| validation | recall | -6.17pt | **-24.25pt** |
| validation | FPR | -4.54pt | -4.29pt |
| train sanity | F1 | +1.66pt | +1.66pt |
| train sanity | recall | -8.15pt | +1.96pt |

- **validationのF1改善/悪化: 9/9**（同値0）。**動画単位では完全に拮抗している。**
- train sanityのF1改善: 3/3（ただしn=3で代表性なし）。
- **TP0動画数: validation R0=1 → R1=0**、train sanityは両arm0。
- recallで**median-of-diffsとdiff-of-mediansが大きく乖離**（-6.17pt対-24.25pt）。
  動画ごとのrecall変化が不均一であることを示し、S5-14補足3で分離した2統計の意義が再び現れた。

##### 8.11.10.4 事前選定基準の機械判定

| 基準 | 判定 |
| --- | --- |
| 1. validation paired F1差分中央値が正 | **satisfied**（ただし`marginal=True`。+0.03ptは閾値0.5pt未満） |
| 2. validation split median IoU・pooled F1がR0以上、TP0が増えない | satisfied |
| 3. recall低下やFP増加だけで説明される悪化がない／トレードオフを明示 | **requires_judgment** |
| 4. train sanityのmedian F1/IoU・TP0が悪化しない | satisfied |
| **総合** | **requires_policy_chat_judgment** |

##### 8.11.10.5 実装チャットの所見と提案

**一貫して大きい効果はFPの削減とrecallの低下**である。R1はpooled FPRを4.63pt下げ、
ignore領域のpositive率も下げる一方、pooled recallを12.67pt失っている。
F1・IoUの「改善」はこのトレードオフの差し引きであり、**validationの動画別では+0.03pt・
改善9/悪化9と、実質的に拮抗している**。

依頼書6.3は「全条件を満たしても単一seedでの暫定判断」であり、
「**微差・トレードオフ・train sanityとの不一致が残る場合はR0を維持する提案とし、
管理チャットへ返す**」と定めている。本結果はまさに微差とトレードオフが残る場合に該当するため、
**実装チャットとしてはR0維持を提案する**。R1暫定採用を提案しない理由は次のとおり。

1. 基準1はchecker自身が`marginal`と印を付ける水準（+0.03pt）でしか満たしていない。
2. 動画別の改善/悪化が9/9で、方向性がない。
3. recallの低下（pooled -12.67pt）は大きく、Stage6入力としての意味は未評価である。
4. 単一seed・5 epochであり、F1が約7%・IoUが約3.5%という初期段階のモデル同士の比較である。
   この段階の差が50〜200 epochでの優劣を予測する保証はない。

ただし**「R1に効果がない」と結論するものでもない**。FP削減とignore領域positive率の低下は
両splitで一貫しており、S5-14の回転診断が示した感度と方向が一致する。
threshold調整やStage6のFP許容基準が定まった段階では評価が変わりうる。

**PLY所見: 未実施。** 評価パイプラインはPLYを出力しているが、目視確認は行っていない。
依頼書6.2の指示に従い「未実施」として記録する。

##### 8.11.10.6 比較checkerの実機実行結果（2026-09-19、実機、CPU 1回）

**`status: passed`、PASS 23 / FAIL 0 / JUDGE 2 / UNKNOWN 2。**

| 判定 | 内容 |
| --- | --- |
| **FAIL 0** | config parity（allowlist外の差分なし）、両armの固定リストモード、augmentationモード、初期checkpointの同一パス＋期待hash一致、**両arm間のfile list一致**、**W-A参照リストとの一致**、history epoch到達・finite、best/last存在、学習後checkpointが相違、評価対象の一致——**すべて合格** |
| JUDGE 2 | `r0/r1.checkpoint.per_epoch`: per-epoch checkpoint 0件（既報のSAVE_EVERY不具合） |
| UNKNOWN 2 | `r0/r1.manifest.settings_match_config`: manifest未指定のため突合せず（配線省略の判断による） |
| privacy | 検出0件 |

**W-A保存済みリストとの全64桁照合が完了した。** UNKNOWNはmanifest突合の2件のみであり、
`file_list.*.vs_reference`はPASS群に含まれる。dry run時は先頭16桁の照合だったが、
学習後のrunが自ら保存したリストが参照リストと**内容・順序ともに一致**することが確認された。
これで「固定splitの検算」の未確認項目が解消した。

**数値は本環境での事前計算と完全一致した**（pooled validation F1差分 0.005885493430879907、
recall差分 -0.12673090506843654、動画別F1 median-of-diffs 0.00031922084396403163、
diff-of-medians 0.010359602276483332、TP0 1→0）。事前基準の判定も同一で、
総合は`requires_policy_chat_judgment`。

##### 8.11.10.7 比較checkerの複数CSV対応

匿名化exportがsplitごとにCSVを分けて出力するため、checkerが**1 armあたり複数CSV**を
受け取れるようcomma区切りに対応させた（CPUテスト1件追加、計18件）。

#### 8.11.11 管理チャット返信: P3の4判断とP4への移行（2026-09-19）

ユーザーが管理チャットの4判断案を採用したことを受け、以下を正式判断として返信します。
返信日はユーザー確認済みの2026-09-19です。同日付の実機記録に関する日付確認は不要とし、
`EX_DATE`のrunラベルと実行日時の違いは別の履歴情報として保持してください。

##### 8.11.11.1 判断1: R0を維持し、R1を今回は採用しない

新規R0（augmentation=none）をP4のbaselineとして維持します。旧W-Aと新規R0を混同せず、
R1は比較履歴として保持してください。追加の角度・seed探索は行いません。

R1のvalidation pooled FPRは12.53%→7.91%へ低下する一方、recallは45.08%→32.41%へ低下しました。
pooled F1は6.77%→7.36%、TP0は1→0と改善しますが、動画別F1は改善9/悪化9、paired差分中央値は
約+0.03ptです。微差と大きなrecall/FPトレードオフが残るため、事前方針どおりR0維持とします。
この判断はcheckerの`marginal`閾値だけに依存せず、上記の動画別結果とトレードオフに基づきます。

「R1に効果がない」「回転augmentationが一般に無効」とは結論しません。FP削減効果は観測されていますが、
今回の単一seed・5 epoch比較で採用する根拠は不足しています。threshold調整で今回の採否を
後から覆そうとせず、production設定も変更しません。

##### 8.11.11.2 判断2: per-epoch checkpoint欠落を理由とする再学習は行わない

両armのepoch5 `last.pt`が揃い、bestも同じ評価モデルであり、5 epoch分の指標も残っています。
中間epochの重みがないことは主比較の成立を妨げないため、900 optimizer stepを再実行して
欠落重みを作り直すことは承認しません。

ただし「各epoch保存」の仕様逸脱は解消済みと偽らず、**意図SAVE_EVERY=1／実効save_every=10、
中間重み欠落、復元不能**をrun記録に保持します。既存manifest・config・重みを修正して
当初から正しく保存されていたように見せないでください。

##### 8.11.11.3 判断3: P4へ進む。ただし、まず既存成果物の確認を行う

新規R0の5 epoch結果をP4最終pilotの前半として再利用します。P4への移行は追加学習開始の
承認ではありません。以下を先に行ってください。

1. **manifest突合:** 保存済みtraining manifestを両armとも比較checkerへ渡し、CPU読取専用で
   1回再検査することを承認します。既存のUNKNOWN 2件を解消し、意図値と実効configの差が
   既知のsave_every逸脱に限定されるか確認してください。既知の逸脱はJUDGEのまま保持し、
   PASSへ丸めません。他の差や未検査キーがあれば明記し、比較可能性に関わるFAILならP4を停止して
   報告します。対象キー外も含めた全設定検証を実施したとは表現しないでください。
2. **R0履歴確認:** 保存済み5 epoch metricsからloss・FP/FN・recall/FPRの推移とepoch5の動画別
   指標を整理してください。中間重みがないため、epoch別の動画単位再評価は実施できないことを
   明記します。既存CSV/JSONのCPU解析のみで行います。
3. **PLY確認:** 既に出力済みのR0 PLYについて、ユーザーへ確認対象と観点を案内してください。
   train sanityの大腿骨欠損、背景FP、時間方向の反復、およびvalidationの検出状況を確認し、
   R1も比較資料として参照できます。目視未実施なら未実施のまま記録し、実施済みとしません。

これらをP4準備報告として管理チャットへ返してください。必要であればR0だけを累計10 epochまで
延長する計画を次に審議します。既存`--checkpoint`は完全resumeではないため、延長には
optimizer・epoch・必要なRNG等を復元する実装/検証案、または再開始の別計画を提示する必要があります。
今回の返信ではresume改修・追加学習・50 epochへの移行・GPU再評価を承認していません。

##### 8.11.11.4 判断4: 未コミット5ファイルは再発防止修正として別コミットする

確認した対象は次の**変更4＋新規1＝5ファイル**です。本書の「7ファイル」は訂正します。

```text
Stage5/checks/dummy/check_dummy_rotation_augmentation_ablation.py
Stage5/checks/real_h5/check_stage5_rotation_augmentation_ablation.py
Stage5/train_stage5.sh
docs/stage5/FILES.md
Stage5/checks/dummy/check_dummy_launcher_env_passthrough.py
```

再発防止と複数CSV対応を独立したコミットへまとめる方針を了承します。コミット対象・差分・
CPU検証結果を確認し、実装チャットはユーザーが実行するコマンドを提示してください。
無関係な変更や`.tmp/`・metrics・checkpoint・PLYは含めません。

P3学習を生成したrevisionは`646cb52b3507e174c996c11d6889d6d0f549dfe5`のままです。
修正後commitは再発防止・比較検査用のrevisionとして別に記録し、過去runへ付け替えないでください。
比較checkerの再検査にも使用revision・実行日時を残します。

env passthroughテストはshellの**静的解析**であり、Pythonへ実効値が届くことの実行検証とは
区別してください。次回学習では実効configと初回保存予定epochのcheckpoint存在を早期に確認し、
再び5 epoch終了まで逸脱を見逃さない運用を計画へ含めます。manifest突合機能は追加だけで終えず、
上記の保存済み成果物へ適用してください。

##### 8.11.11.5 記録更新と停止点

管理記録・評価レポートにもR0維持、再学習不要、保存仕様逸脱、P4準備中を反映してください。
P3の比較結果は受け入れますが、manifest未突合という確認残りは上記CPU検査が終わるまで残します。
本書の古い「未実施」「判断待ち」や仕様逸脱を無条件に無影響とする表現は、履歴と現行状態が
分かるよう整理します。P4準備報告後、追加学習の必要性・方法・実施量を別途判断します。

### 8.12 段階P4への移行: R0の50 epochテスト（2026-09-19）

#### 8.12.1 最新確定方針 — 方針更新: 確認工程を絞りR0の50 epochテストへ進む（2026-09-19）

ユーザー了承により、直前の「判断3」とP4準備の停止点を以下で更新します。
構造・入力・学習経路の検証を重ねるより、次は学習期間による変化を確認します。
R0維持、保存欠落だけを理由としたP3再学習不要、再発防止修正の別コミットという判断は維持します。

1. **P3の新規R0の5 epoch結果を最終pilotの5 epoch部分として受け入れる。**
   5 epoch PLY確認は任意に変更し、開始条件から外します。追加の10 epoch pilotは省略します。
   既存metricsの整理は行って構いませんが、反復的な厳密検証やGPU preflightを追加しません。
2. **開始前の必須確認は、既存manifestと実効configのCPU突合。**
   前節で承認した両armのCPU検査1回を実施し、既知のsave_every以外に比較可能性を損なう不一致が
   ないことを確認します。既知の逸脱は記録したまま受け入れ、新しい重大差があれば停止・相談します。
3. **同じGroupNorm転移初期重みから、R0条件で新規50 epochを1 run実行する。**
   P3のlast.ptからモデル重みだけを読み込んで継続しません。resume実装は今回は不要です。
   初期5 epoch相当の計算重複は、再開実装・検証を増やさず一貫した学習履歴を得るため受け入れます。
   これは欠落checkpointを再生成する目的のP3再実行ではありません。
4. **条件はR0のまま固定。** teacher v7、保存済みtrain162/val18、初期重みhash、固定weight、
   CE/smoothing0、GroupNorm8、seed42、lr1e-3/weight_decay1e-4、window16/8、batch1/accum8、
   augmentationなし、mean aggregationを維持します。scheduler・threshold・samplingも変えません。
5. **確認は開始時・25 epoch・50 epochへ集約。**
   開始時に実効configを照合し、最初の保存予定epochでcheckpoint生成を確認します。
   25 epochは通常のvalidation指標・loss/FP/FN等の推移で中間確認し、PLY・固定21動画の別推論は
   必須にしません。50 epochで固定21動画の評価とPLY確認をまとめて行います。

実装チャットは再発防止修正を別コミットし、50 epoch用の起動設定・独立出力先・保存間隔・
最初の保存確認epoch・manifest記録方法を提示してください。元の5 epoch比較launcherを黙って
50 epochへ変更せず、比較履歴と新規長期runを明確に分けます。少なくともbest/lastを保持し、
定期保存の間隔とepoch25/50の保存方法を起動前に明示します。

本決定は上記の50 epochテストへ進む条件付き承認です。CPU突合・最新コードのdry run・
実効設定確認が申請条件と一致すれば進めてよく、同内容の形式的な再審議は不要です。
715 window/epochなら予定は35,750 forward/backward組、90 optimizer更新/epoch、計4,500更新です。
validation forwardは別計数とし、実際のwindow数・skip・更新数を記録します。

50 epoch評価はlastを主とし、bestが同じ評価モデルなら重複を避けます。異なる場合は副次評価として
両方を固定21動画で評価し、最大42動画条件とします。新しい動画探索は行いません。
NaN/Inf、点対応破損、設定不一致、保存失敗は停止・報告します。一時的な精度変動だけでは止めず、
recall崩壊、TP0増加、FP/FPR増大等の明確な悪化が継続する場合に相談します。
100〜200 epoch延長・追加seed・R1長期比較・threshold tuning・production変更は未承認です。
50 epoch結果を返してから次を判断します。

#### 8.12.2 方針更新への対応（2026-09-19）

管理チャットの4判断と方針更新を確認し、次を実施した。**学習・GPU実行は行っていない。**

##### 8.12.2.1 訂正の受け入れ

未コミットファイル数は**変更4＋新規1＝5**であり、本書の「7ファイル」は誤りであった。
ユーザーがcommit `cd3b886`として別コミット済み。P3学習を生成したrevisionは
`646cb52b...`のままであり、付け替えていない。

##### 8.12.2.2 比較checkerのJSON確認（27行すべて）

`rotation_ablation_shareable.json`を照合し、PASS 23の内訳を確認した。
特に`file_list.train_files.txt.vs_reference`・`file_list.val_files.txt.vs_reference`が
ともにPASS（`reference_content_identical: true`）であり、**両armが自ら保存したリストが
W-A保存済みリストと内容・順序ともに全64桁一致**することが確定した。
dry run時点の先頭16桁照合から、学習後の完全照合まで到達した。

##### 8.12.2.3 実施した作業

1. **manifest突合の配線**（`check_stage5_rotation_augmentation_ablation.sh`に
   `MANIFEST_R0`/`MANIFEST_R1`を追加）。既存のUNKNOWN 2件を解消するための実行に使う。
   既知の`save_every`逸脱はJUDGEのまま保持され、PASSへ丸められない実装である。
2. **50 epoch用launcherの新設**（`checks/real_h5/run_stage5_s5_15_r0_longrun.sh`）。
   **5 epoch比較用launcherは変更していない**（比較履歴と長期runを分離するため）。
   独立した実験名・出力先・manifest出力先を持ち、既定はdry run。
3. **実効config早期確認checkerの新設**（`check_stage5_effective_run_config.py`/`.sh`、
   CPUテスト7件）。env passthroughテストがshellの静的解析であるのに対し、
   こちらは`train_stage5.py`が実際に書いた`config.json`を読む**実行時検証**である。
   P3の逸脱（実効`save_every=10`）を検出できることをテストで固定した。
4. **記録更新**: 管理記録（現在の段階、S5-15結果・R0維持・保存仕様逸脱・次段階）、
   評価レポート9.9節（数値・判断・実施量・逸脱）、`../FILES.md`（新規3種）。

全CPUスイート合格（**Python 124件＋bash 14件**）。

##### 8.12.2.4 50 epoch run の提示内容

| 項目 | 内容 |
| --- | --- |
| 起動 | `checks/real_h5/run_stage5_s5_15_r0_longrun.sh`（新規・独立） |
| 位置づけ | 監査済み初期重みからの**新規run**。P3 `last.pt`からの継続ではなく、resume実装も行わない。**欠落checkpoint再生成を目的としたP3再実行でもない** |
| 条件 | R0のまま固定（teacher v7、保存済みlist、初期重みhash `55ec6e6b...438b`、固定weight、GroupNorm8、seed42、lr1e-3/wd1e-4、window16/8、batch1/accum8、augmentationなし）。scheduler・threshold・samplingは変更なし |
| epochs | 50 |
| **保存間隔** | `SAVE_EVERY=5` → epoch 5,10,15,20,25,30,35,40,45,50＋`best.pt`/`last.pt`。管理記録P5の「10 epochごと」を満たし、かつ**epoch25と50を明示的に保存**する |
| **最初の保存確認epoch** | **epoch 5**（`checkpoint_epoch_0005.pt`）。完了時点で実効config照合とあわせて確認する |
| 出力先 | `stage5_runs/<EX_DATE>/pointnext_s_EX<date>_s5_15_r0long50_none_gn8_cwfixed_lr1e3_ep50_bs1_acc8_nopad`。既存runとは別 |
| manifest | `work_dirs/_s5_15_longrun_manifests/`（run dirの外）。dry run分と本実行分を別ファイルで保存 |
| 予定実施量 | 35,750 forward/backward組、90 optimizer更新/epoch、**計4,500更新**。validation forwardは別計数。実際のwindow数・skip・更新数は実行後に記録する |
| 起動前検証 | 初期checkpoint SHA-256照合、両listのfingerprint全64桁照合、出力先の既存検出。いずれか失敗で停止 |
| 既定動作 | **dry run**。`CONFIRM_TRAINING=1`のときのみ学習を開始する |

50 epoch評価は`last.pt`を主とし、bestが同一評価モデルなら重複を避ける（異なる場合は副次評価として
最大42動画条件）。NaN/Inf・点対応破損・設定不一致・保存失敗は停止・報告する。

#### 8.12.4 manifest突合の実行結果（2026-09-19、実機、CPU 1回）

承認された「両armのCPU検査1回」を実施し、**UNKNOWN 2件が解消した**。

`status: passed`、**PASS 25 / FAIL 0 / JUDGE 4 / UNKNOWN 0**。

| arm | git HEAD | 起動時の作業ツリー | 検出された不一致 |
| --- | --- | --- | --- |
| R0 | `646cb52b...` | `clean` | `save_every`: 意図1／実効10（critical=false）のみ |
| R1 | `646cb52b...` | `clean` | 同上 |

**既知の`save_every`逸脱以外に不一致は検出されなかった。** JUDGE 4件の内訳は
`save_every`突合2件と`per-epoch checkpoint`欠落2件で、いずれも既報の同一事象に由来する。
**JUDGEのまま保持されておりPASSへ丸めていない。** 比較可能性に関わるFAILは0件のため、
P4（50 epoch）を停止する事由はない。

**この検査が比較した範囲（限定を明示する）:** 突合対象は次の**7キーのみ**である。

| キー | 重大度 | 結果 |
| --- | --- | --- |
| `augmentation` / `augmentation_rotation_degrees` / `seed` / `epochs` / `pointnext_norm` / `pointnext_norm_groups` | critical（不一致ならFAIL） | 両arm一致 |
| `save_every` | 非critical（不一致はJUDGE） | **両arm不一致（意図1／実効10）** |

**「対象キー外も含めた全設定を検証した」とは表現しない。** manifestが記録していない設定
（lr、weight_decay、window、batch、accumulation、label_policy、class_weight等）はこの突合の
対象外である。ただしそれらは比較checkerの**config parity検査**（arm間の完全一致、
allowlist 3キー以外）で別途FAIL 0が確認されており、P1監査・dry run・実効config記録とも
整合している。両者は目的の異なる検査であり、片方をもう片方の代替としない。

なお、manifestは`worktree_clean: true`・git HEAD `646cb52b...`を記録しており、
**P3学習が clean なツリーから起動されたこと**が両armで確認できた。

#### 8.12.5 50 epoch run のdry run結果（2026-09-19、実機、学習未起動）

新規launcherをdry runし、**学習は起動していない**（`Dry run: training was NOT started.`）。
申請条件との照合結果は次のとおりで、**全項目が一致した**。

| 申請条件 | dry runの実値 | 判定 |
| --- | --- | --- |
| 初期重み | `55ec6e6bcb39d58f398719b33826e80715a94bc6e7670d5b88623cd7c668438b` | 監査値と一致 |
| train list | 162件、content `582579833f345b77…`、identity `74cd6030cd07c852…` | P3・W-Aと同一。全64桁照合を通過 |
| val list | 18件、content `0c251380e40f0def…`、identity `de3f3d513fc0bf41…` | 同上 |
| augmentation | `none` | R0条件、P3から不変 |
| seed / norm / weight / label policy | 42 / `groupnorm` 8 groups / `0.05963856,1.94036150` / `bbox_noncontour_ignore` | 一致 |
| epochs | 50 | 一致 |
| **保存間隔** | `save_every=5` → epoch 5,10,15,20,25,30,35,40,45,50＋best/last | epoch25・50を含む |
| **最初の保存確認** | `checkpoint_epoch_0005.pt` | epoch 5完了時に確認 |
| 予定実施量 | optimizer更新 **4,500**（90/epoch×50） | 方針更新の予定値と一致 |
| 出力先 | `stage5_runs/260919/pointnext_s_EX260919_s5_15_r0long50_none_gn8_cwfixed_lr1e3_ep50_bs1_acc8_nopad` | **P3（260917）・W-A（260914）と別**。既存runと非混在 |
| git HEAD | `d0e37526a669b686d3e1adf80a5bd76e28bd6c90`、作業ツリー`clean` | 新規revision。**P3の`646cb52b…`は付け替えていない** |
| 実行環境 | torch 2.7.1+cu128、RTX 5090、driver 580.173.02 | P3と同一 |
| privacy self-check | 検出0件 | 合格 |

`EX_DATE=260919`と実施日（2026-09-19）が一致しており、P3で生じた1日ずれは解消している。

**記録上の小さな差異（実害なし、記録のみ）:** manifestの`resolved_augmentation_seed`が
P3では`500042`だったのに対し、本runでは`None`である。長期run用launcherが当該引数を渡していない
ためで、`augmentation=none`では角度生成に使われないため**「seedが使われていない」という記録として
むしろ正確**である。manifest突合の対象7キーに含まれないため、突合で誤検出も起きない。

**次の確認点:** manifestの`save_every=5`はあくまで**意図値**である。実効値は起動後に
`config.json`で確認する必要があり、epoch 5完了時点で
`check_stage5_effective_run_config.sh`により実効config照合と
`checkpoint_epoch_0005.pt`の存在確認を行う。P3の逸脱を5 epoch終了まで見逃した経緯への直接の対応である。

**進行条件の充足:** 方針更新が定めた条件付き承認の前提（CPU突合・最新コードのdry run・
申請条件との一致）のうち、CPU突合は8.12.4で、dry runは本節で満たした。
残る実効設定確認は起動後の工程であるため、**本実行へ進んでよい状態である**。

#### 8.12.6 50 epoch run 起動と段階1確認（2026-09-19、実機、学習継続中）

50 epoch runを起動し、epoch 1の途中（約60%）で段階1（実効config確認）を実施した。

**最重要の確認: `config.save_every` は expected=5 / effective=5 で一致した。**
P3で失われた設定が今回はPythonへ正しく届いており、`train_stage5.sh`の修正
（commit `cd3b886`）が実機で機能したことが確認された。他の10項目
（epochs 50、augmentation none、seed 42、groupnorm 8 groups、label policy、
batch 1、accumulation 8、window 16/8）もすべてPASS。**学習条件に問題はない。**

#### 8.12.7 不具合報告: `EXPECT_CHECKPOINT=""`が効かず誤ってFAILを出した

段階1の実行で`checkpoint.first_periodic`が**FAIL**となったが、これは
**確認スクリプト側の不具合による誤検出**であり、学習の異常ではない。

**根本原因:** `check_stage5_effective_run_config.sh`が
`EXPECT_CHECKPOINT="${EXPECT_CHECKPOINT:-checkpoint_epoch_0005.pt}"`と書かれていた。
`:-`は**未設定のときだけでなく空文字のときも**既定値へ置き換えるため、
段階1で意図的に渡した`EXPECT_CHECKPOINT=""`が既定値へ戻り、
epoch 5に到達していない段階で`checkpoint_epoch_0005.pt`を要求してFAILとなった。

**これは実装チャットの確認漏れである。** Python側は空文字をUNKNOWNとして扱う実装で、
その挙動はCPUテストで確認していたが、**シェルラッパー経由の挙動を検証していなかった**。
`SAVE_EVERY`不具合（bashが環境変数を捨てていた）と同じ「層をまたぐ検証の欠落」であり、
一度同種の誤りを経験した後にもかかわらず繰り返した。

**修正:** `${EXPECT_CHECKPOINT-checkpoint_epoch_0005.pt}`（コロンなし）へ変更し、
明示的な空文字は「確認をスキップする」意味になるようにした。

**追加した検証:** `check_dummy_effective_run_config.sh`に**シェルラッパー経由の3件**を追加した。
実際のラッパーの`SCRIPT_DIR`を当該checkoutへ書き換えたコピーを用意して実行する方式で、
Python単体ではなくラッパーを通した動作を検証する。

1. `EXPECT_CHECKPOINT=""`でcheckpoint確認がスキップされ、初回保存前のrunが通ること。
2. 未設定なら従来どおり`checkpoint_epoch_0005.pt`を要求すること。
3. **確認をスキップしても実効configの逸脱（`save_every=10`）は依然FAILになること**
   （スキップが検査全体を無効化しないことの確認）。

修正前のコードでは検証1が落ちることを確認済みである。

**学習への影響: なし。** 段階1のFAILは誤検出であり、学習は停止させていない。
run dirへの書き込みも行っていない（checkerは読取専用）。

#### 8.12.8 段階2確認: 合格（2026-09-19、実機、epoch 5完了後）

修正後のスクリプトでepoch 5完了後に再実行し、**12項目すべてPASS（FAIL 0 / UNKNOWN 0）**。

| 確認 | 結果 |
| --- | --- |
| 実効config 11項目 | epochs 50、**save_every 5**、augmentation none、seed 42、groupnorm 8、label policy、batch 1、accumulation 8、window 16/8 — すべて一致 |
| **定期保存の実動作** | `checkpoint_epoch_0005.pt` が存在（9,594,531 bytes）。**periodic saving is working** |

**P3の不具合はこれで完全に解消が確認された。** P3では意図`SAVE_EVERY=1`が
`train_stage5.sh`に捨てられ、実効`save_every=10`で**per-epoch checkpointが1件も保存されず**、
それが5 epoch終了まで発覚しなかった。今回は (a) 修正により設定がPythonへ届き、
(b) 起動直後の段階1で実効値を確認し、(c) epoch 5完了時の段階2で**保存が実際に行われたこと**を
確認した。「設定が届いたか」と「保存が動いたか」を別々に検証する運用が機能している。

なお checkpoint のサイズがP3の`last.pt`（9,578,177 bytes）より約16KB大きいのは、
埋め込まれるconfigの文字列長（experiment名・出力先・epochs等）と、
`torch.save`がアーカイブ名として埋め込むファイル名の長さ（`checkpoint_epoch_0005` と `last`）の
差によるもので、8.11.9で確認した性質と整合する。異常ではない。

#### 8.12.9 中間確認（epoch 25）で参照するファイル

`train_stage5.py`は**`history.json`を学習終了時にのみ書き出す**（epoch loopの後）。
一方`metrics.jsonl`は**毎epoch追記**される（`record`＝`{"epoch", "train", "val"}`を1行1 JSON）。
したがって**実行中の中間確認では`metrics.jsonl`を参照する**。構造はhistory.jsonの各要素と同一のため、
既存のCPU解析はそのまま適用できる。

epoch 25時点の中間確認は方針更新のとおり、通常のvalidation指標とloss/FP/FN等の推移のみとし、
PLY・固定21動画の別推論は行わない。CPU解析のみで完結する。

#### 8.12.10 50 epoch run 完了と学習推移の解析（2026-09-19、実機）

**実行の健全性（すべて合格）**

| 項目 | 結果 |
| --- | --- |
| epoch | 1〜50が欠けなく完走 |
| non-finite値 | **0件** |
| optimizer更新 | **4,500**（90/epoch×50）。**予定値と完全一致** |
| val TP=0のepoch | **皆無** |
| 保存物 | `best.pt`・`last.pt`＋`checkpoint_epoch_0005〜0050.pt`**10件**（`SAVE_EVERY=5`どおり） |

**P3の保存不具合はこれで完全に解消した。** 意図した保存間隔が実効値として反映され、
10件の定期checkpointが実際に生成された。

**明確な過学習（本runの主要な所見）**

| 指標 | 推移 |
| --- | --- |
| val loss | 最小`0.4619`（epoch 8）→ 最終`1.6901`（epoch 50）＝**3.66倍** |
| train loss | `0.5794` → `0.0998`（単調に近い減少） |
| train F1 | `0.0333` → `0.3886`（約12倍） |
| val F1 | `0.0333` → `0.0739`（約2.2倍、epoch 10以降は0.07〜0.086で振動） |
| val recall | `0.40`（epoch 6）→ `0.139`（epoch 50） |
| val FPR | `0.13` → `0.031` |

train側だけが改善し続け、val lossが8 epoch目以降ほぼ一貫して悪化する典型的な過学習である。
recallの低下とFPRの低下が同時に進み、モデルは「positiveを出さない」方向へ寄っている。
ただし**TP=0には至っていない**。

**`best_metric`（val iou_femur）は epoch 6 がピーク**（0.0483）。2位は epoch 41（0.0449）で、
**epoch 6以降の44 epochは一度もピークを更新しなかった。**

**主要な比較（window単位、公式H5評価ではない）**

| checkpoint | val F1 | val IoU | val recall |
| --- | ---: | ---: | ---: |
| P3 R0 epoch 5（P3の`last.pt`） | 0.0691 | 0.0358 | 0.4488 |
| 長期run epoch 6（本runの`best.pt`） | 0.0921 | 0.0483 | 0.4017 |
| 長期run epoch 50（本runの`last.pt`） | 0.0739 | 0.0384 | **0.1390** |

**「学習期間を延ばすと良くなるか」への暫定的な答えは「ならない」である。**
50 epoch時点はbest_metricでepoch 6を下回り、recallは約1/3に落ちている。
ただしこれはwindow単位のrunning metricsであり、**判断は固定21動画の公式評価で行う。**

**評価への影響:** `best.pt`（epoch 6）と`last.pt`（epoch 50）は**別のcheckpoint**である。
P3のような重複回避はできないため、評価は**両方×21動画＝42動画条件**となる
（方針更新が承認した上限どおり）。

なお中間checkpointが10件残っているため、将来epoch別の推移を動画単位で追う余地はあるが、
**今回の承認範囲外**であり実施しない。

#### 8.12.11 「過学習」解釈への2つの反証仮説の検証（2026-09-19、CPU解析）

ユーザーから、8.12.10の過学習解釈に対して2つの反証仮説が提起された。いずれも
**history.jsonのdebug指標で検証可能**であったため実施した。結果として**どちらも支持されなかった**が、
解釈を精密化する材料が得られた。

**仮説1: 「序盤が優勢なのは、モデルがpositiveに消極的でnegative判定で稼いでいるため」**

50 epoch全体での相関を取ると、**逆の結果**であった。

| 相関 | 値 |
| --- | ---: |
| corr(IoU, **precision**) | **+0.871** |
| corr(IoU, recall) | -0.071 |
| corr(IoU, 予測positive比率) | **-0.367** |

`best_metric`（IoU_femur）は**precisionと強く連動**しており、recallとはほぼ無相関である。
**positiveを多く出すepochほど低得点**になっており（IoU下位5 epochの予測positive比率は
0.197/0.167/0.099/0.096、上位5は0.089/0.044/0.043/0.045/0.031）、
「消極的だから高得点」という機序は成立しない。

なお epoch 6 は、予測positive比率 0.089 でrecall 0.40 を得ている。epoch 8 は比率0.16で
recall 0.57だがIoUは低い（0.0397）。**epoch 6はpositiveを撒かずにrecallを得ている点で
効率のよい動作点**であり、単なる出し過ぎではない。

**仮説2: 「7 epoch付近で過学習は考えにくい。評価データ18件では判断困難」**

閾値に依存しない指標として、**GT-positive点とGT-background点に対する平均予測確率の差**
（`debug_mean_prob_positive_on_gt_positive` − `..._on_gt_background`）を用いた。
これは0.5という閾値の取り方に左右されない識別能力の指標である。

| epoch | P(pos\|GT pos) | P(pos\|GT bg) | **gap** | train側のgap | train−val |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.379 | 0.295 | 0.084 | 0.061 | -0.022 |
| 6 | 0.401 | 0.163 | 0.238 | 0.327 | +0.090 |
| **8** | 0.506 | 0.206 | **0.300（最大）** | — | — |
| 10 | 0.461 | 0.171 | 0.290 | 0.423 | +0.133 |
| 20 | 0.303 | 0.099 | 0.205 | 0.644 | +0.439 |
| 30 | 0.259 | 0.077 | 0.182 | 0.789 | +0.607 |
| 50 | 0.151 | 0.036 | **0.114** | **0.860** | **+0.745** |

**validation側の識別gapは epoch 8 の 0.300 をピークに、epoch 50 では 0.114 へ低下する**
（corr(epoch, gap) = -0.631）。一方**train側のgapは 0.061→0.860 と単調に拡大**し続け、
train−valの乖離は -0.022 → **+0.745** へ広がる。

**これは閾値の取り方では説明できない。** 訓練データの分離は改善し続け、未知データの分離は
悪化し続けるという、過学習の定義そのものの挙動である。したがって
「閾値0.5が不適切なだけで、threshold sweepで回復する」という説明も成立しない。

**検証で得られた修正点:** ピークは epoch 6 ちょうどではなく **epoch 6〜10 の領域**である。
IoU@0.5は epoch 6、閾値非依存gapと val loss最小は **epoch 8**。
8.12.10で「epoch 6がピーク」と書いたのは`best_metric`に限った話であり、
**より頑健には「最良領域は epoch 6〜10、明確に epoch 50 ではない」**と述べるべきである。

**残る留保（ユーザー指摘の後半は依然有効）:** validation 18動画という規模の小ささは、
本解析でも解消していない。ただしtrain/val乖離の大きさと単調性（+0.745）は、
小標本の揺らぎでは説明しにくい。動画単位の分散は固定21動画評価の結果で確認する。

**主比較の決定はPLYの定性評価を待つ。** 本解析は数値面の材料であり、置き換えるものではない。
定量的には次が予測される: epoch 50 は予測positive比率が 0.032（epoch 6 は 0.089）であるため、
**PLY上では大腿骨の検出が疎になり欠損が目立つ**と見込まれる。
もしPLYで epoch 50 のほうが「大腿骨を捉えたうえで背景FPが少ない」と見えるなら、
本解析の解釈と矛盾するため、その不一致自体を報告対象とする。

#### 8.12.12 固定21動画の公式評価結果（2026-09-19、実機、42動画条件）

`best.pt`（epoch 6）と`last.pt`（epoch 50）の両方を固定21動画で評価した。
両checkpointでGT・点集合が完全一致し、評価対象も同一であることを確認済み。

**validation 18動画（pooled、3 checkpointの比較）**

| checkpoint | recall | precision | FPR | F1 | IoU | **TP0動画** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P3 R0 epoch 5 | 45.08% | 3.66% | 12.53% | 6.77% | 3.51% | 1/18 |
| 長期run **best = epoch 6** | 38.95% | 5.27% | 7.41% | **9.28%** | **4.86%** | 1/18 |
| 長期run **last = epoch 50** | **6.97%** | 3.90% | 1.81% | 5.00% | 2.57% | **10/18** |

**train sanity 3動画（学習に使った動画、pooled recall）**

| checkpoint | pooled recall |
| --- | ---: |
| P3 R0 epoch 5 | 70.30% |
| 長期run best = epoch 6 | 62.33% |
| 長期run **last = epoch 50** | **97.36%** |

**動画別F1中央値（validation）**: P3 epoch5 `5.95%` / 長期best epoch6 `7.75%` /
**長期last epoch50 `0.00%`**。epoch 50では**18動画中10動画でTPがゼロ**、
中央値の動画はrecall 0.00%である（改善4／悪化13／同値1）。

#### 記憶（memorization）の確定

同一checkpoint（epoch 50）が、**学習に使った3動画ではrecall 97.36%、未知の18動画では
recall 6.97%**を示す。この乖離は閾値の取り方でも、validationが18動画と少ないことでも
説明できない。**10動画でTPが厳密にゼロ**というのは標本揺らぎではない。
epoch 50のモデルは学習動画を記憶し、未知動画に対しては positive をほぼ出さなくなっている。

#### ユーザー提起の2仮説に対する最終回答

**仮説1（序盤はpositiveに消極的だから高得点）— 明確に否定された。**
消極的なのは**epoch 50のほう**であり、しかも**未知データに対してのみ**である
（validation recall 6.97%／FPR 1.81% に対し、train sanity recall 97.36%）。
挙動が「学習で見た動画かどうか」で分かれており、これは消極性ではなく記憶である。

**仮説2（7 epochでの過学習は考えにくい／18動画では判断困難）— 妥当な懸念だったが、
公式評価で否定された。** 同一checkpointでのtrain 97% 対 validation 7% という乖離は
validationの標本サイズでは説明できない。ただしこの指摘は有益であった。
これを受けて閾値非依存の識別指標（8.12.11）と動画単位の公式評価を確認した結果、
当初の「val lossが上がっている」よりはるかに強い証拠が得られた。

#### 学習期間に関する結論（暫定）

**「学習期間を延ばすと良くなるか」への答えは「わずかに、epoch 6〜10まで。それ以降は有害」。**
長期runのbest（epoch 6）はP3のR0（epoch 5）を validation F1 で 6.77%→9.28%、
IoU で 3.51%→4.86%、動画別F1中央値で 5.95%→7.75% と上回っており、
**5 epochより6〜8 epoch程度のほうが良い**。一方 epoch 50 は全指標で明確に劣る。

#### PLY定性評価への重要な申し送り

**train sanityのPLYだけで比較すると、結論が逆に見える。**
epoch 50はtrain sanityでrecall 97.36%であり、PLY上では大腿骨がほぼ完全に描かれるはずである。
一方validationでは10/18動画でTPゼロ、すなわち**ほぼ何も描かれない**。

したがって**定性評価はvalidation動画を主対象として行う必要がある**。
train sanityのPLYは「記憶できているか」の確認にはなるが、汎化の判断材料にはならない。
8.12.11で予測した「epoch 50は検出が疎になる」は、validationについては予測以上に強く現れた。

#### 残る留保

単一seedの結果であり、validation18動画は既に多数の方式選択に使用済みで独立testではない。
ただし記憶の兆候（train 97% 対 validation 7%、TPゼロ10動画）は極めて明瞭であり、
これらの留保によって方向が変わる性質のものではない。

#### 8.12.13 PLY・フレーム画像による定性評価の結果（2026-09-20）

専用の実装チャットがフレーム画像可視化ツールを実装・実行し
（`docs/stage5/s5-15/stage5_prediction_frame_visualization_implementation_handoff.md` 11節）、
ユーザーがPLYと画像群を目視した。本節はその結果を定量データと突き合わせた記録である。

**定性観察は観察メモに近い体裁であり、表現の曖昧さを含むという留保付きで扱う。**

##### (a) 番号体系のずれ（記録上の注意）

3系統のaliasが併存していた。照合の結果、次が確定した。

| 系統 | 採番 |
| --- | --- |
| 匿名化CSV（`validation_h5_metrics.csv`）・ユーザーの観察メモ | **1始まり**（`validation_001`〜`018`）。両者は一致 |
| 実装チャットの11節 | **0始まり**。11節の`validation_NNN` ＝ CSVの`validation_{NNN+1}` |

off-by-oneは**3件の完全一致**で確定した（11.2(c)の`TP=0/FP=24,120/FN=5,397`はCSVの
`validation_006`、11.2(d)の`2038+10=2048`はCSVの`validation_018`、11.2(b)の
`TP=0/FP=9,531`はCSVの`validation_004`）。**11節の動画指定を引用する際は+1して読むこと。**

##### (b) 新しい所見: GTが2箇所に写る動画で系統的に性能が落ちる

観察メモの「大腿骨が2箇所に映ると完全に該当箇所を取得できず、ノイズのみになる」は、
定量的に裏づけられた。メモでGTが2箇所と記述された8動画と、1箇所の10動画を比較する（best、epoch 6）。

| 群 | recall中央値 | F1中央値 | GT positive数 中央値 |
| --- | ---: | ---: | ---: |
| GT 1箇所（10動画） | **56.52%** | 8.28% | 2,686 |
| GT 2箇所（8動画） | **16.65%** | 5.57% | 3,907 |

**recall下位5動画のうち4件が2箇所群**、**上位5動画には2箇所群が1件も入らない**。
GT規模による交絡ではない（2箇所群のほうがGT positive数はむしろ多い）。

これは実装チャットの11節には含まれていない発見であり、
11.6が「未回答」とした「FNに共通の特徴があるか」への具体的な候補となる。

##### (c) ただし失敗の形は「時間方向の欠落」ではなく「部分的な取りこぼし」

window単位metrics（16フレーム/stride 8）で「GTを含むがTPがゼロのwindow」を数えたところ、
**2箇所群と1箇所群で差は出なかった**（TP=0 windowの割合の中央値はいずれも0.0%）。
全windowがTP=0なのは`validation_006`のみ、部分的に存在するのが`validation_008`（1/3）と
`validation_010`（1/4）である。

つまり2箇所群の低recallは、**特定の時間区間がまるごと欠落しているのではなく、
各windowの中で取れている点が少ない**という形をとる。
「2箇所写ると取れない」は正しいが、その機序は時間的欠落ではない。

##### (d) PLYの見え方が定量値を誤らせた事例（方法論上の教訓）

観察メモは`val_010`と`val_011`をほぼ同一の表現（「GTの右側の1割程度しか重なっていない」）で
記述していたが、実際のrecallは**36.78%と0.06%**であった。

ユーザーの確認により、`val_010`の記述はPLYを上から見た際の見え方に由来する誤りと判明した。
フレーム画像で確認すると、GTが写る全14フレームのうち9フレームがほぼFNであり、
**その9フレームで断面が大きく移動していたため、上から投影すると9割方FNに見えていた**。

**これはフレーム画像可視化を導入した価値を示す事例である。**
PLYの上面視は、断面が動く動画では被覆率の印象を誤らせる。
以後、被覆率の定量的主張はPLYの見え方ではなく数値またはフレーム画像に基づくこと。

なお、この「フレーム単位ではGTが写る14フレーム中9フレームがほぼFN」という観察は、
window単位metrics（(c)）では粒度が粗すぎて確認できない。
確認するには新ツールが生成する`<video>_frame_summary.csv`（フレーム別のTP/FP/FN集計）が必要である。

##### (e) FPの性質（実装チャットの観察と整合）

メモの「大腿骨周辺の足の輪郭」「腰の骨や胴体の輪郭」「頭蓋骨・腹部の輪郭」は、
実装チャットの11.4が立てた仮説「モデルは大腿骨ではなくフレーム内で相対的に明るい
細長い構造を学習している可能性」と整合する。いずれも解剖学的には無関係だが、
**形状と輝度が似た構造**である。

メモはさらに、**「各フレーム画像の水平に中心、垂直に少し上寄り」**（データ内で大腿骨が
存在しやすかった位置）にFPが集中すると繰り返し記述している（val_004/005/006/010/011/013/015等）。
これは11.4の「位置が効いている」という仮説を支持し、
**モデルが画像内の絶対位置に対する事前分布を学習している**可能性を示す。
S5-14の座標診断（hot binのFPRが高い）とも方向が一致する。

##### (f) train sanityでの観察

メモは train sanity について「lastのほうがbestより良い」「FNは少ないがFPノイズが非常に多い」
「GT付近をほぼ全フレームで覆うようにFPが出る」と記述している。
これは8.12.12の定量結果（epoch 50はtrain sanity recall 97.36%）と整合し、
**記憶の視覚的裏づけ**となる。予測した非対称性（trainでは良く見え、validationでは空に見える）は
実際に観察された。

##### 現時点の解釈と留保

(b)(e)は**次の改修方針を検討する材料**になるが、いずれも18動画・単一seedの観察である。
(b)は定量的裏づけがあるが、「2箇所」の判定はメモの目視分類に依存しており、
GTの連結成分数として機械的に数え直す検証が望ましい。
production設定の変更や採否の判断は本節では行わない。

#### 8.12.14 「2箇所」判定の機械化（2026-09-20、実装・CPU検証済み・実機未実行）

8.12.13(b)の所見は、GTが2箇所かどうかを**目視で分類した**結果に依存していた。
これを機械的な数え直しへ置き換えるcheckerを実装した。

- `Stage5/checks/real_h5/check_stage5_gt_component_count.py` / `.sh`（新規）
- `Stage5/checks/dummy/check_dummy_gt_component_count.py` / `.sh`（新規、**9件合格**）

**定義:** フレームごとに、valid かつ label=1 のGT点を`pixel_xy`空間で単連結クラスタリングし、
連結成分数を「そのフレームに写るGT領域の数」とする。ignore点・background点・
`valid_mask=False`の点は数えない。これは描画済みフレーム画像で観察者が「blob」として
見ているものと同じ対象である。

**方法論上の要点:**

1. **連結距離（link distance）を1つに決めない。** 領域数はこの半径に依存するため、
   既定で6通り（2,3,4,6,8,12 px）を掃引し、**各動画のラベルが全半径で一致するか**を記録する。
   半径によってラベルが反転する動画は`stable_across_link_distances: false`として報告し、
   どちらか一方へ丸めない。**1つの半径でしか成立しない結果は所見として扱わない。**
2. **微小な塊を第2領域と数えない。** `min_component_points`（既定5点）未満の成分は除外するが、
   除外前の数（`num_regions_all`）も併せて記録する。
3. **新しい採番体系を作らない。** 評価が既に出力している
   `video_id_map_DO_NOT_SHARE.csv`（`anonymous_id`↔`original_h5_path`）を入力とする。
   8.12.13(a)のoff-by-one問題の再発を防ぐための設計判断である。
4. **因果を主張しない。** metrics CSVを渡すと機械ラベル別のrecall中央値を出すが、
   出力に「これは関連であり、因果の実証ではない」と明記する。片方の群が空なら差を出さない。

**このcheckerは予測を使わない。** GTの幾何だけを見るため、モデル・checkpoint・
中間H5・再推論のいずれも不要である。CPUのみ。

**副次的な修正: shell構文チェックの自動化**

本checkerの`.sh`作成時、`${VAR:?メッセージ}`のメッセージ中にアポストロフィを書き、
bashの構文解析が壊れた。**S5-14補足3でも同じ誤りを犯しており、2度目である。**
（二重引用符の内側でもbashはメッセージをシェルテキストとして解析するため、
アポストロフィが閉じない引用符を開き、エラーはファイル末尾を指すので原因が分かりにくい。）

記憶に頼るのをやめ、`checks/dummy/check_dummy_shell_syntax.sh`を追加した。
Stage5配下の全shellスクリプト（現在64本）に`bash -n`をかけ、さらに
`${VAR:?...}`内のアポストロフィを直接grepで検出する。
壊れた版を実際に検出できることを確認済みである。

#### 8.12.15 機械ラベルと目視分類の一致（2026-09-20、実機実行済み・CPU解析）

8.12.14のcheckerを実機で実行した（validation 18動画、半径6通り）。
結果は**目視分類と18/18で完全一致**した。ただし**私の既定閾値は誤っていた**ので先に訂正する。

**訂正: 既定の50%閾値は厳しすぎた。**

既定の`--multi_region_frame_fraction 0.5`（「GTが写るフレームの50%以上で2領域」）では、
`validation_006`(50%)と`validation_016`(89%)の**2件しか**multi-regionにならない。
目視で「2箇所」とされた8件のうち6件を取りこぼす。この既定値は私が根拠なく置いたものである。

**実際のデータが示したこと: 分布が二峰的で、間が空いている。**

| 群 | 動画 | multi-regionフレーム割合 |
|---|---|---|
| 0%群（10件） | 1, 3, 9, 10, 11, 12, 13, 14, 17, 18 | **全件ちょうど 0%** |
| >0%群（8件） | 2, 4, 5, 6, 7, 8, 15, 16 | 11, 29, 18, 50, 33, 43, 25, 89 % |

0%群の最大が0%、>0%群の最小が11%で、**その間に1件も存在しない**。
そして>0%群の8件は、目視で「2箇所」とされた集合と**完全に一致する**（18/18）。

**したがって「2箇所」は目視判断ではなくなった。** 「GTが2領域以上写るフレームが
1枚でもあるか」という機械的性質に対応しており、閾値を0%〜11%のどこに置いても同じ分類になる。

**この機械ラベルで群別recallを取り直すと:**

| 機械ラベル | n | recall中央値 | recall平均 |
|---|---|---|---|
| multi-region（>0%） | 8 | **16.65%** | 20.03% |
| single-region（0%） | 10 | **56.52%** | 56.77% |

差は中央値で**−39.87pt**。8.12.13で目視群別に手計算した値を正確に再現する。
**これは関連であり因果ではない。** 動画数18、群間で撮影条件・GT点数も揃っていない。

**checkerへの反映（1点）:** `classify_video()`に`multi_region_any_frame`を追加した
（既存ラベルは削除せず併記）。上表の分類を手計算ではなくcheckerが直接出せるようにするため。
dummy test 1件追加（**計10件合格**）。閾値の既定値は変えていない。

**留保として明示すべき点:**

1. **`>0`という基準はデータを見た後に選んだ。** 事後選択である。事後でない主張は
   閾値そのものではなく、**0%と11%の間が完全に空いている（分離が完全である）**という分布の形と、
   その分離が独立に作られた目視分類と18/18で一致したという事実である。
2. **割合が6半径（2〜12px）で完全に不変だった。** 「1つの半径でしか成立しない結果は
   所見としない」という要件は満たすが、**掃引が全く効かなかった**ことは記録しておく。
   GT領域が内部は密で相互には十分離れていれば妥当な挙動だが、
   この定義がストレスを受けていないことも同時に意味する。不安定判定は0件。
3. **`validation_002`は根拠が薄い。** 11% = GTフレーム9枚中およそ1枚での判定であり、
   この1件だけは分離の端にある。8件から外しても群差の向きは変わらない。
4. **checkerが測っているのは「フレーム内の同時併存」である。** 観察者がPLYで見た「2箇所」は
   **時間方向に別の場所へ移る**ことを指していた可能性があり、概念が完全には同一でない。
   動画全体でのクラスタリングという別実装で切り分けられるが、現時点では実施していない。
5. **命名の紛れ:** 出力中の`multi_region_any`は「いずれかの半径で」の意味であり
   「いずれかのフレームで」ではない。今回追加した`multi_region_any_frame`が後者である。

#### 8.12.16 残る2検証の実装（2026-09-20、実装・CPU検証済み・実機未実行）

8.12.13で提案し保留していた検証(2)(3)を実装した。**いずれも追加学習・再推論を伴わず、
既存の成果物（teacher H5・中間pseudo3d H5・保存済み`predictions/*.npz`）だけを読む。CPUのみ。**

##### 検証(2) 失敗はフレーム単位か、動画単位か

- `Stage5/checks/real_h5/check_stage5_frame_failure_vs_gt_regions.py` / `.sh`（新規）
- `Stage5/checks/dummy/check_dummy_frame_failure_vs_gt_regions.py` / `.sh`（新規、**9件合格**）

**動機:** 8.12.15は動画単位の比較であり、**動画レベルの交絡**（該当8動画がそもそも別種の撮影である
可能性）を排除できない。「2領域だから悪い」のか「この動画群がたまたま別の理由で難しい」のかを
分離できない。

**方法:** 比較を**動画の内側へ移す**。GTを含む各フレームに、そのフレーム自身の連結成分数と
そのフレーム自身のrecallを付与し、**同一動画内の1領域フレームと2領域以上フレームを直接比較**する。
これで被験者・撮影・crop・モデルが固定される。動画をまたいだ集計は、その動画内差分に対する
**正確二項符号検定**（stdlibのみ。scipy非依存、正規近似を使わない）で行う。

**設計上の判断:**

1. **片方のフレームしか持たない動画は「差0」ではなく除外する。** 動画内比較の証拠を
   何も持たないため、タイとして数えると結果を薄めるだけである。除外数は別途報告する。
2. **全フレームをプールした値も出すが、「交絡を制御していない」と明記する。** プールすると
   フレーム数の多い動画が支配し、動画レベル交絡が復活する。**主たる結論は対応のある比較の側**である。
3. **閾値非依存の値を併走させる。** 各フレームのGT positive上の平均`prob_femur`をrecallと
   並べて出すので、0.5の切り方に依存しない読み方ができる。
4. **採番は`video_id_map_DO_NOT_SHARE.csv`の1始まりaliasを使う**（8.12.13(a)の再発防止）。
5. プール値はpoint-weighted（点数重み）とframe-median（フレーム等重み）を**別々に**出す。

##### 検証(3) 輝度・位置と予測の関係

- `Stage5/checks/real_h5/check_stage5_brightness_position_vs_prediction.py` / `.sh`（新規）
- `Stage5/checks/dummy/check_dummy_brightness_position_vs_prediction.py` / `.sh`（新規、**13件合格**）

**動機:** ハンドオフ11.4の2つの仮説は、画像を見ただけでは決着しない。

- **(3)** モデルは**そのフレーム内での相対的な明るさ**を追っており、動画全体での絶対輝度ではない。
- **(5)** 輝度だけではFPの位置を説明できない（画面端の明るいアーティファクトにはFPが出ず、
  より暗い中央部の構造には出る）。

**方法:** 各点に、その点が乗る画素のグレー値を与え、**3通りで表現**する。
生値、**自フレーム内でのmid-rank percentile**、**動画全体でのmid-rank percentile**。
さらに**crop端からの距離**を与える。これらを結合テーブルへ集計し、
**もう一方の軸を固定したうえで**各仮説を検定する。**周辺（marginal）の関連はどちらの仮説の
証拠にもならない**ため、条件付きテーブルが本体である。

- 仮説(3)の検定: 動画percentileを固定してフレーム内percentileを掃いたときの平均確率の振れ幅と、
  その逆。**振れ幅が大きい側が、確率がより強く追っている尺度**である。
- 仮説(5)の検定: フレーム内輝度binを固定して端からの距離を掃いたときの予測positive率の振れ幅。
  ゼロでなければ、輝度だけではFPの位置を説明できない。
- **`prob_femur`を用いる**ので、どの結果も0.5閾値に依存しない。
- グレー変換はStage2to4の`image_to_uint8_gray`をそのまま使う（GT可視化・フレーム出力と同じ値）。

**併せて報告する重要な値: H5の`intensity`特徴と画像グレー値の相関（動画別）。**
`intensity`はモデルが実際に受け取る2特徴の一方であるため、
**これが高ければ輝度はモデルの入力そのものであり、低ければ輝度効果は幾何経由でしか届かない。**
どちらかで解釈が変わるので、他の数値より先に見るべき値である。

**留保として出力に明記するもの:**

1. ここで測る輝度は**モデルが受け取るものとは限らない**（モデルの入力は`intensity`と`confidence`、
   および平均中心化・最大値スケーリング済み座標である）。
2. 2つのpercentileは構成上互いに強く相関するため、条件付き振れ幅が比べているのは
   **増分の寄与**であって全体の寄与ではない。片方が小さくても「無関係」を意味しない。
3. **フレーム内の点は独立な観測ではない**ため、bin内の点数は統計的な標本数ではなく、
   本checkerはp値を出さない（検証(2)の符号検定も方向確認として読むこと）。
4. すべて1 split・1 runの関連であり、因果の同定でもproduction設定変更の根拠でもない。

##### 実行コマンド（実機、いずれもCPU・読み取りのみ）

```bash
# 検証(2) bestとlastの両方で実行する
VIDEO_ID_MAP=<評価出力>/…/video_id_map_DO_NOT_SHARE.csv \
EVALUATION_DIR=<評価出力>/best CHECKPOINT=best \
  bash checks/real_h5/check_stage5_frame_failure_vs_gt_regions.sh

# 検証(3) 中間pseudo3d H5の置き場が必要
VIDEO_ID_MAP=… EVALUATION_DIR=<評価出力>/best CHECKPOINT=best \
PSEUDO3D_OUTPUTS_ROOT=<中間pseudo3d出力root> \
  bash checks/real_h5/check_stage5_brightness_position_vs_prediction.sh
```

共有するのは`*_shareable.json`と`*_shareable.csv`のみでよい。
`*_private_DO_NOT_SHARE.json`は実video名を含むため共有しないこと。

**副次的な確認:** `check_dummy_shell_syntax.sh`の対象は64本から**68本**になり、全件`bash -n`合格。

#### 8.12.17 検証(2)の結果: 「2箇所」は**フレームの性質ではなく動画の性質**（2026-09-20、実機、best）

検証(2)を実機で実行した（best、validation 18動画、GTを含む287フレーム）。
**結論は8.12.15の関連を否定しないが、その因果的な読みを支持しない。**

##### (a) 動画内で見ると、2領域フレームは1領域フレームと差がない

同一動画内で両方のフレームを持つ8動画（うち1件は両側0.0でタイ、有効7対）。

| 動画 | (2領域フレームのrecall中央値) − (1領域フレームのrecall中央値) |
|---|---:|
| validation_002 | −0.0092 |
| validation_004 | −0.0430 |
| validation_005 | **+0.1706** |
| validation_006 | ±0.0000（両側0.0） |
| validation_007 | −0.2891 |
| validation_008 | +0.0003 |
| validation_015 | **+0.2766** |
| validation_016 | −0.1068 |

**悪化4件・改善3件・タイ1件。中央値 −0.0046。正確二項符号検定 p = 1.0。**
方向に一貫性がない。

##### (b) 決定的な所見: 該当動画は**1領域フレームでも**同じだけ悪い

| 群 | **1領域フレームのみ**でのrecall中央値 |
|---|---:|
| 「2箇所」動画（8件） | **0.1358** |
| 「1箇所」動画（10件） | **0.6656** |
| 差 | **−0.5298** |

**8.12.15の群差（−39.87pt）は、該当動画の「2領域フレーム」を1枚も使わなくても再現する。**
つまり不利はフレームがいくつの領域を持つかに付随しておらず、**動画全体に一様にかかっている**。

##### (c) したがって解釈を改める

**フレーム内にGTが2領域現れることは、不調な動画の「目印」であって、その機序ではない。**
該当8動画は別の共通性質（解剖・描出角度・断面移動など）を持ち、それが全フレームのrecallを
押し下げていると考えるほうが、観測と整合する。「2箇所写ると取れない」という定性所見は
**動画の識別としては正しく、フレーム単位の因果としては裏づけられない**。

なお、プール値（全287フレームを動画区別なく集計）では2領域フレームのほうが明確に悪い
（point-weighted recall 0.250 対 0.430、フレーム中央値 0.181 対 0.605）。
**しかしこれは(b)が示す動画レベル交絡そのものである**。2領域フレームは不調な動画に偏在するため、
プールすると動画の不利がフレームの不利に見える。checkerはこのプール値に
「交絡を制御していない」と明記して出力している。

##### (d) 留保

1. **有効な対は7件しかなく、符号検定の検出力は非常に低い。** p = 1.0は「差がない証明」ではない。
   小さい効果なら見逃す。**(b)の−0.53という大きさと併せて初めて結論になる**のであって、
   符号検定単独では何も言えない。
2. 2領域フレームは全287フレーム中**37枚**しかない。動画あたり数枚である。
3. これはbestのみの結果である。lastは未実行（実行時の指定ミスによる。下記(e)）。
4. 副次的所見として、**GTを含むフレームの19.86%（287枚中57枚）がTPをまったく持たない**。
   ゼロrecallフレームの割合には群差がほとんどない（中央値 0.111 対 0.098）。

##### (e) 実行上の不備と修正（私の設計の落とし穴）

`last`の実行が`ValueError: no rows for checkpoint 'best'`で停止した。
`EVALUATION_DIR`を`.../last`にしつつ`CHECKPOINT`が既定の`best`のままだったためである。

**これは利用者の指定ミスというより、私が`--checkpoint`を`--evaluation_dir`と独立に既定値付きで
用意したことが原因である。** evaluate_stage5.pyはcheckpointごとにディレクトリを分けており、
各ディレクトリは自分のcheckpointの行しか持たない。したがって2つを別々に指定させる設計自体が、
「一方を名指しして他方を指す」誤りを誘発する。

修正:

- **`--checkpoint`の既定値を`--evaluation_dir`のディレクトリ名から取る**ようにした。
- 明示指定がディレクトリ名と食い違う場合は、**どちらかを信用せず停止する**。
- shell側も`CHECKPOINT="${CHECKPOINT:-$(basename "${EVALUATION_DIR}")}"`に変更した。
- 検証(3)のcheckerにも同じ修正を入れた。合成テストを1件追加（**検証(2)は計10件合格**）。

**停止したこと自体は正しい挙動である**（誤ったcheckpointの行を黙って使わなかった）が、
そもそも誘発しない設計にすべきだった。

##### (f) 次

- **`last`で検証(2)を再実行する**（`CHECKPOINT`の指定は不要になった）。
  記憶が強いlastで(b)の非対称性がどう出るかは、過学習の解釈にも関わる。
- 検証(3)は未実行。(b)が示す「動画レベルの共通性質」が何かを詰める材料になりうる。
- **8.12.13(b)および8.12.15の記述は本節の(c)で更新される。** 群分けは有効なままだが、
  「2箇所写ると取れない」を機序として引用しないこと。

#### 8.12.18 検証(2)のlast結果: 床効果と、閾値非依存で見た群構造の消失（2026-09-20、実機）

`last`（epoch 50）でも検証(2)を実行した。**recallによる対応比較はlastでは成立しない。**
一方、閾値非依存の値に切り替えると、bestとlastの重要な違いが見える。

##### (a) lastではrecallが床に張り付き、対応比較が機能しない

| 指標 | best | last |
|---|---:|---:|
| TPが1点もないGTフレーム | 57/287 = **19.86%** | 215/287 = **74.91%** |
| プールrecall（1領域フレーム） | 0.4295 | **0.0812** |
| プールrecall（2領域フレーム） | 0.2499 | **0.0296** |
| 比較可能8動画のうちrecallがタイ | 1件 | **6件** |
| 符号検定の有効対 | 7 | **2** |

lastでは18動画中14動画が1領域フレームのrecall中央値0.0である。
**タイ6件は「差がない」ではなく「どちらも0で差を測れない」である。**
したがって**検証(2)はlastについて問いに答えていない**。これはデータの限界であり、所見ではない。

8.12.17(b)の群比較もlastでは両群とも中央値0.0となり、判別できない。

##### (b) 閾値非依存に切り替えると、bestにあった群差がlastでは消えている

GT positive点上の平均`prob_femur`を、**該当動画の1領域フレームのみ**に限って比較する
（8.12.17(b)と同じ切り口を閾値非依存にしたもの）。

| | 「2箇所」動画(8) | 「1箇所」動画(10) | 差 | 相対差 |
|---|---:|---:|---:|---:|
| **best** | 0.3428 | 0.5013 | **−0.1585** | **−32%** |
| **last** | 0.0737 | 0.0746 | −0.0009 | −1.2% |

**bestでは群差が閾値非依存でも明確に存在し（−0.159）、8.12.17(b)のrecall差を裏づける。**
**lastではその群差が消えている（−0.0009）。**

これは「lastのほうが公平になった」のではない。**lastはGT上の確率が全動画で0.07前後まで落ちており
（bestは0.34〜0.50）、不調な動画を選んで失敗しているのではなく一様に失敗している。**
8.12.12の「10/18動画でTP=0」「validation recall 6.97%」と同じ現象を、
フレーム粒度かつ閾値非依存で見た形である。

##### (c) フレーム単位の効果は、閾値非依存で見ても存在しない

8.12.17(a)はrecallによる判定だったので、閾値非依存でも確認した。
動画内での（2領域フレーム − 1領域フレーム）平均`prob_femur`差:

| | 中央値 | 悪化 | 改善 |
|---|---:|---:|---:|
| best | −0.0131 | 4 | 4 |
| last | +0.0015 | 3 | 5 |

**bestで4対4、lastで3対5。8.12.17(c)の結論は閾値非依存でも変わらない。**
「2箇所」はフレームの性質ではなく動画の性質である、という(c)はrecall特有の効果ではない。

##### (d) 留保

1. lastの平均確率は0.07前後と小さいため、**絶対差は圧縮される**。上表に相対差を併記したのは
   そのためである。それでも−32%対−1.2%という開きは圧縮だけでは説明しにくい。
2. (b)の比較はn=8対n=10であり、検定は行っていない。**大きさの提示であって有意性の主張ではない。**
3. `validation_016`（2領域フレーム割合89%）はlastで唯一ゼロrecallフレームが0件の動画である。
   n=1であり、深読みしない。
4. 全てbest/lastそれぞれ1 run・1 split・単一seedの観測である。

##### (e) 検証(2)の結論（best・last統合）

- **「GTが2箇所に写る」は不調な動画の目印であり、フレーム単位の失敗機序ではない**（best・last両方、
  recallでも閾値非依存でも一貫）。
- **動画レベルの差はbestには確かに存在する**（閾値非依存で−32%）。8.12.15の群分けは有効である。
- **lastではその動画レベルの差すら消える**。一様な崩壊であり、選択的な失敗ではない。
- 次に問うべきは「フレーム内に2領域あるか」ではなく、**該当8動画が共有する別の性質は何か**である。
  検証(3)（輝度・位置）はその候補の1つを測る。

#### 8.12.19 検証(3)の結果: 輝度はモデルの入力そのもの、位置は高輝度でのみ効く（2026-09-20、実機、best）

`best`で検証(3)を実行した（validation 18動画、有効点 約917万点）。
**ハンドオフ11.4の仮説(3)は否定され、仮説(5)は支持されたが機序が想定と異なる。**

##### (a) 最重要: `intensity`特徴は画像グレー値そのものである

| | 動画別の相関（`intensity` 対 画像グレー値） |
|---|---|
| 最小 / 中央値 / 最大 | **1.0000 / 1.0000 / 1.0000** |

18動画すべてで**厳密に1.0**。`intensity`はグレー値のアフィン変換であり、
**輝度はモデルが実際に受け取る2特徴の一方そのものである**（代理指標ではない）。
以降の輝度に関する数値は、すべて「モデルの入力に対する応答」として読んでよい。

**副次的だが重要:** この相関1.0は、**`pixel_xy`と`local_encoder_images`の対応が正しいことの
強い検証にもなっている**。座標空間や[y,x]の取り違えがあれば1.0にはならない。

##### (b) 仮説(3)は否定された: モデルが追うのは絶対輝度であって、フレーム内相対値ではない

もう一方を固定したときの平均`prob_femur`の振れ幅（GT background点のみ、各10層）。

| 掃く軸 | 固定する軸 | 中央値 | 最大 |
|---|---|---:|---:|
| フレーム内percentile | 動画percentile | **0.0373** | 0.1022 |
| 動画percentile | フレーム内percentile | **0.1093** | 0.2558 |

**動画（絶対）側の増分寄与がフレーム内相対値の約3倍である。** 仮説(3)とは逆向きの結果。
使用層数はどちらも10、有効bin数も両者で同程度（2〜8対2〜8）であり、
**片方の軸だけ動く余地が広いという偏りではない**。

(a)と整合する: `intensity`は絶対グレー値なので、**モデルはフレーム内順位という情報を
そもそも直接持っていない**。目視での「フレーム内で相対的に明るい所を拾う」という印象は、
絶対輝度に対するほぼ固定的な応答が、フレームごとに見え方を変えたものと解釈できる。

ただしフレーム内側の振れ幅も0ではない（中央値0.037）。**絶対輝度が支配的だが、
フレーム内相対値も無寄与ではない。**

##### (c) 仮説(5)は支持される。ただし加算効果ではなく**交互作用**である

輝度binを固定して端からの距離を掃いたときの振れ幅:

| 固定した輝度bin | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8** | **9** |
|---|---|---|---|---|---|---|---|---|---|---|
| 振れ幅 | – | .023 | .023 | .023 | .020 | .021 | .031 | .053 | **.148** | **.305** |

**暗い点では位置はほぼ効かず、明るい点では非常に強く効く。**
これは目視所見(5)「画面端の明るいアーティファクトにはFPが出ないが、
より暗い中央部の構造には出る」と正確に一致する。

**私の要約統計量が誤っていた。** 既定で出していた中央値0.023は
「位置はほとんど効かない」と読めてしまうが、これは何も起きていない暗い層が多数派であるために
中央値がそちらへ引かれた結果である。**checkerを修正し、最大値・最低層/最高層の値・
増加傾向フラグを中央値の隣に必ず出すようにした**（合成テスト1件追加、**計14件合格**）。

##### (d) 端からの距離の周辺分布: 境界から離れるほど上がり、中心手前で頭打ち

| 端からの距離bin（0=境界, 4=中心） | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 平均`prob_femur` | 0.064 | 0.128 | 0.224 | **0.273** | 0.271 |
| 予測positive率 | 0.011 | 0.044 | 0.117 | **0.165** | 0.145 |

**ピークはbin 3であり、最中心のbin 4ではわずかに下がる。**
目視所見の「水平に中心、垂直に少し上寄り」と方向は整合するが、
**本checkerの距離指標は上下左右を区別できないため、「少し上寄り」は検証できていない。**
2次元位置はS5-14のXYグリッド診断の担当範囲である。

##### (e) 輝度の周辺分布と、暗端の小さな異常

フレーム内輝度binに対する平均`prob_femur`は 0.037 → 0.232 へ単調増加、
予測positive率は 0.0019 → 0.148 へ増加する。**輝度の効果は大きい。**

ただし**最暗bin 0だけ予測positive率が0.0019と、bin 1〜4（0.0001〜0.0003）より1桁高い**
（動画percentile側でも bin 0 が0.0059と最も高い）。平均確率は最低なのに、
0.5を越える点の割合だけが高い。**n=1,037点と少なく、極端な予測を持つ少数点の可能性がある。**
現時点では異常として記録するにとどめ、解釈しない。

##### (f) 留保

1. **フレーム内の点は独立な観測ではない**（隣接フレームは内容がほぼ重複する）。
   bin内の点数は統計的標本数ではなく、本節にp値はない。
2. 2つのpercentileは構成上相関するため、(b)の振れ幅は**増分寄与**の比較である。
   フレーム内側が小さいことは「無関係」ではなく「絶対輝度に上乗せする情報が少ない」を意味する。
3. **すべて関連であり因果ではない。** 輝度が入力であること((a))は確定だが、
   「モデルが輝度を根拠に判定している」ことの証明ではない。
4. best・validation・1 run・単一seedの観測である。lastは未実行。

##### (g) 8.12.18(e)への接続

検証(2)は「該当8動画が共有する別の性質は何か」を次の問いとした。
検証(3)はその候補として輝度と位置を測り、**輝度がモデルの入力そのものであること**と、
**高輝度点では位置が強く効くこと**を確定させた。
ただし**本節は動画群の差を直接説明していない**。
該当8動画の輝度・位置分布が他と異なるかは、本checkerの出力を群別に集計すれば測れるが、
現時点では未実施である。

#### 8.12.20 検証(3)のlast結果: 位置×輝度の交互作用のみが両checkpointで残る（2026-09-20、実機）

`last`でも検証(3)を実行した。**3つの所見のうち、(5)の交互作用だけが両checkpointで一貫する。**

##### (a) `intensity`＝画像グレー値（相関1.0000）はlastでも再現

当然の結果である（これはデータの性質であってモデルの性質ではない）が、
**両実行で同一の値が出たことは、パイプラインの一貫性確認として意味がある。**

##### (b) 「絶対輝度が支配的」はbest固有であり、lastでは消える

振れ幅は全体に圧縮されるため、**輝度の周辺レンジで正規化して比較する**。

| | フレーム内を掃く | 動画（絶対）を掃く | 比 (動画/フレーム内) |
|---|---:|---:|---:|
| **best**（レンジ 0.1953） | 0.0373（rel 0.191） | 0.1093（rel **0.559**） | **2.93** |
| **last**（レンジ 0.0572） | 0.0134（rel 0.235） | 0.0118（rel 0.205） | **0.87** |

**bestでは絶対輝度が約3倍支配的だったが、lastではほぼ同等（むしろわずかにフレーム内側）になる。**
正規化後も逆転しているので、**確率の圧縮だけでは説明できない**。

8.12.18(b)（bestにあった動画群差がlastで消失）と同じ方向の現象である。
**lastでは輝度尺度に対する構造化された応答そのものが失われている。**
8.12.19(b)の「モデルが追うのは絶対輝度」という結論は、**bestに限定して述べるべきである。**

##### (c) 唯一両checkpointで残る所見: 位置は高輝度点でのみ効く

輝度binを固定して端からの距離を掃いた振れ幅:

| 固定輝度bin | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8** | **9** | 中央値 | 最大 | 最大/中央値 |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| **best** | .023 | .023 | .023 | .020 | .021 | .031 | .053 | **.148** | **.305** | .023 | .305 | **13.0** |
| **last** | .004 | .014 | .013 | .013 | .011 | .009 | .014 | **.062** | **.103** | .013 | .103 | **7.9** |

輝度レンジで正規化した最大振れ幅は **best 1.56 → last 1.80** と、むしろ相対的に強い。

**「暗い点では位置は効かず、明るい点では強く効く」は、best・lastの両方で成立する。**
検証(3)で最も頑健な所見である。8.12.19(c)で追加した最大値・傾向フラグが、
lastの実行でも自動的にこれを検出した。

##### (d) lastはbestより中心寄りに偏っている

端からの距離に対する平均`prob_femur`の周辺分布:

| | 境界(bin 0) | 1 | 2 | 3 | 中心(bin 4) | ピーク位置 | 境界/ピーク |
|---|---|---|---|---|---|---|---:|
| **best** | 0.064 | 0.128 | 0.224 | **0.273** | 0.271 | bin 3 | 0.235 |
| **last** | 0.007 | 0.022 | 0.052 | 0.078 | **0.082** | bin 4 | **0.089** |

**lastではピークが最中心へ移り、境界との比が0.235→0.089へ下がる。**
中心への偏りが強まっている。8.12.13(e)の「データ内で大腿骨が存在しやすかった位置にFPが集中」
および8.12.12の記憶（train sanity recall 97.36%）と方向が整合する。
**位置に対する事前分布が強まった、という解釈と両立する**が、本節はそれを証明しない。

##### (e) 8.12.19(e)の暗端異常は再現しなかった

| 予測positive率 | bin 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **best** | **0.001929** | 0.000290 | 0.000096 | 0.000075 | 0.000283 |
| **last** | **0.000000** | 0.000869 | 0.001044 | 0.001097 | 0.001728 |

lastでは最暗binの予測positive率が**0.0**であり、bin 0→4は単調増加する。
**8.12.19(e)の異常はbest固有であり、再現しない。** n=1,037点の小標本に対する
当時の「解釈しない」という扱いは妥当だった。所見として扱わない。

##### (f) 検証(3)の結論（best・last統合）

1. **確定:** `intensity`は画像グレー値そのものであり、**輝度はモデルの入力である**（両実行、相関1.0000）。
   同時に`pixel_xy`と画像の対応が正しいことの検証にもなっている。
2. **両checkpointで一貫:** **位置は高輝度点でのみ強く効く**（交互作用）。目視所見(5)を支持する。
3. **best限定:** 絶対輝度がフレーム内相対値より支配的（約3倍）。lastでは消える。
   **仮説(3)の否定はbestについての結論であり、モデル一般の性質としては述べられない。**
4. **付随所見:** lastはbestより中心寄りに偏る。暗端異常は再現しない。
5. **未回答:** 本節は8.12.18(e)の問い（該当8動画が共有する別の性質）に**まだ答えていない**。
   次節の群別集計で扱う。

#### 8.12.21 群別集計: 輝度・位置は動画群差の1/3しか説明しない（2026-09-20、実機、best）

8.12.18(e)の問い「該当8動画が共有する別の性質は何か」に対し、
**輝度と位置がその説明になるか**を直接標準化で検定した。群ラベルは8.12.15の機械ラベル
（`multi_region_any_frame`、8動画 対 10動画）である。

##### (a) 方法: 直接標準化

群差には2つの可能性がある。**曝露差**（該当動画のGTが単に暗い／周辺にある）と
**応答差**（同じ明るさ・同じ位置の点をモデルが違う扱いにしている）。
該当群のbin別平均を**対照群の分布で重み付け直す**ことで分離する。
**標準化後も残る差は曝露差ではない。**

**落とした重み（対照群のbinのうち該当群が1点も持たないもの）は0.0000であった。**
調整は全binに支えられており、少数binへの依存はない。

##### (b) 結果: GT positive点の差の2/3は輝度・位置で説明できない

| | 単純比較 | 輝度・位置を揃えた後 |
|---|---:|---:|
| 対照群（1箇所、10動画）平均`prob_femur` | 0.48402 | — |
| 該当群（2箇所、8動画）平均`prob_femur` | 0.31924 | **0.37278** |
| **差** | **−0.16478** | **−0.11125** |

**曝露で説明されるのは32.5%にとどまり、67.5%が残る。**
**同じ明るさ・同じ端からの距離の点であっても、該当8動画のGTは低い確率しか受け取らない。**

軸別の内訳（GT positive点）:

| 揃えた軸 | 残る差 | 説明割合 |
|---|---:|---:|
| フレーム内輝度のみ | −0.12057 | 26.8% |
| **動画（絶対）輝度のみ** | −0.09921 | **39.8%** |
| 端からの距離のみ | −0.15131 | 8.2% |
| フレーム内輝度＋距離（同時） | −0.11125 | 32.5% |

##### (c) 背景点には群差がない。これは全体的な校正のずれではない

| | 単純比較 | 標準化後 |
|---|---:|---:|
| 対照群のGT background平均 | 0.16434 | — |
| 該当群のGT background平均 | 0.15956 | 0.17381 |
| 差 | −0.00478 | +0.00946 |

**背景点ではほぼ差がない（いずれも絶対値0.01未満）。**
もし該当動画が「全体に暗いので全点の確率が下がる」だけなら、背景点にも差が出るはずである。
出ていない。**不利はGT点に特異的である。**

閾値非依存の識別力（GT positive平均 − GT background平均）で見ると明確になる:

| | 識別力 |
|---|---:|
| 対照群 | **+0.31968** |
| 該当群（単純） | **+0.15968**（対照群のちょうど半分） |
| 該当群（輝度・位置を揃えた後） | **+0.19897**（なお対照群より38%低い） |

##### (d) 曝露差は実在する（ただし説明力は1/3）

GT positive点の分布（各群のGT点が各binに占める割合）:

- **フレーム内輝度:** 対照群は**76.9%**が最上位binに集中するのに対し、該当群は**51.2%**。
  該当群のGTは中位bin（5〜8）へ広がる。**該当動画のGTはフレーム内で相対的に暗い。**
- **端からの距離:** 対照群のピークはbin 3（39.8%）だが、該当群はbin 1（34.9%）。
  **該当動画のGTはより境界寄りにある。**

いずれも予測を下げる向きであり、**曝露差の存在自体は確かである**。
それでも差の1/3しか説明しない。

##### (e) checkerの2点の修正（いずれも本実行で露見した私の不備）

1. **説明割合の比が発散していた。** 背景点で「explained 2.98」と出たのは、
   crude gapがほぼ0のときに割り算が壊れるためである（意味がない値）。
   **crude gapが対照群平均の5%未満のときは比をNoneにし、`ratio_suppressed_as_gap_too_small`を
   立てる**ようにした。生の2つの差を直接読ませる。
2. **同時標準化に動画（絶対）輝度軸が入っていなかった。** (b)のとおり単独では動画輝度が
   最も説明力が高い（39.8%）のに、同時版はフレーム内輝度×距離しか使っていなかった。
   **動画輝度×距離、および両輝度尺度の同時版を追加した。**
   したがって**本節(b)の32.5%は上限ではない**。再実行で確定する。

合成テスト2件追加、**計21件合格**。

##### (f) 留保

1. **(b)の32.5%は暫定値である。**(e)2の追加軸を含めた再実行が必要。
   ただし動画輝度単独でも39.8%であり、**過半が残るという結論は動きにくい**。
2. 18動画を10対8に分けた比較であり、**検定は行っていない**。大きさの提示である。
3. 標準化が揃えるのは**測定した軸だけ**である。残った差は
   「これらの軸では説明できない」ことを意味し、「何も説明できない」ことではない。
4. best・validation・1 run・単一seed。lastは未実行。
5. **因果ではない。** 該当群の低い確率が、GTの幾何そのものによるのか、
   撮影条件や解剖の別の相関量によるのかは、本節では区別できない。

##### (g) 現時点の整理

- **8.12.15の群差は実在し、輝度・位置では主に説明できない**（少なくとも2/3が残る）。
- **その不利はGT点に特異的**であり、動画全体の確率シフトではない（(c)）。
- **フレーム内の領域数はその機序ではない**（8.12.17〜18）。
- したがって該当8動画は、**輝度でも位置でもGT領域数でもない何か**を共有している。
  次の候補としては、GTの形状（細長さ・面積・断面の移動量）、
  隣接フレーム間のGT重心移動、GT点の密度などが、いずれも既存成果物から測定可能である。

#### 8.12.3 共有依頼の訂正: file listは共有不要

先の共有依頼で`train_files.txt`/`val_files.txt`を挙げたのは**私の誤り**である。これらは
実video名を含むため共有できない。**fingerprintの照合は実機上で比較checkerが行い、
出力されるshareable JSONにはhashと判定のみが載る**設計になっている。共有は不要。

### 8.13 P4判断: 実装チャットからの提案（2026-09-20）

50 epoch run・固定21動画評価・定性評価・検証(2)(3)がすべて完了したため、
章9項目8「100〜200 epoch延長・production採用等は50 epoch結果を受けて決める」に対する
**実装チャットとしての提案**をまとめる。**決定は方針管理チャットが行う。**

#### 8.13.1 提案A: 100〜200 epochへの延長は行わない（確度: 高）

**根拠は決定的である。**

- `best_metric`（val iou_femur）のピークは**epoch 6**であり、**続く44 epochは一度も更新しなかった**。
- val lossは epoch 8 の 0.4619 を最小として、epoch 50 に 1.6901（**3.66倍**）まで悪化。
- epoch 50は**記憶**である。同一checkpointでtrain sanity recall **97.36%** 対
  validation recall **6.97%**、**18動画中10動画でTP＝0**（8.12.12）。
- 検証(2)(3)により、epoch 50では動画群差(8.12.18)も輝度尺度への構造化された応答(8.12.20)も
  失われている。**選択的な失敗ではなく一様な崩壊**である。

50 epochで単調に悪化しているものを100〜200 epochへ延ばす根拠はない。
**延長は計算資源を消費するだけであり、実施すべきでない。**

#### 8.13.2 提案B: 現設定でのproduction採用は行わない（確度: 高）

最良点（epoch 6）でもvalidation pooledは
**recall 38.95% / precision 5.27% / F1 9.28% / IoU 4.86%**である。
**precision 5.27%は、TP 1点あたりFPが約18点**という意味であり、
Stage6入力として使える水準ではない。定性評価でも、大腿骨と無関係な
足の輪郭・腰・頭蓋骨・腹部の輪郭にFPが広く出ることが確認されている（8.12.13(e)）。

#### 8.13.3 提案C（本命）: R0対R1を、過学習が実際に起きる長さで比較し直す（確度: 中〜高）

**P3のR0/R1比較は5 epochで行われた。50 epoch runは過学習がepoch 6〜8から始まることを示した。
つまりR1は、過学習が発生する前の領域だけで評価されていた。**

augmentationの目的は過学習の抑制である。**過学習が起きていない区間で比較しても、
その効果は原理的に現れない。** 8.11.11.1がR0維持とした判断は当時の情報では妥当であり、
「R1に効果がない」と結論していない点も正しかった。しかし**比較の設計自体が、
測りたいものを測れない長さだった**ことが、50 epoch runによって事後的に判明した。

さらに、5 epochでR1が示した挙動——**pooled FPR −4.63pt、pooled recall −12.67pt**——は、
**まだ正則化が必要でない段階に正則化をかけたときの典型的な signature**である。
これは「R1は有害」ではなく「5 epochでは早すぎた」と読むほうが整合する。

**提案する実験:**

- R0とR1を**各25 epoch**で1 runずつ。ほかの条件はすべて固定（既存の固定条件どおり）。
- 25 epochとする理由: 過学習の発生（epoch 6〜8）を十分に跨ぎ、
  かつepoch 50の完全な崩壊まで行かない区間を両armで観測するため。
- **計算量は 25×2＝50 epoch分**であり、**今回完了した50 epoch runとほぼ同じ**である。
- 主要な見どころは最終性能ではなく、**val lossの最小値・その到達epoch・
  そこからの悪化の速さが両armで異なるか**である。
- `SAVE_EVERY=5`を継続し、両armで同じepochのcheckpointを残す。
- 評価は固定21動画。checkpointは各armのbestとlast。

**この実験が否定的な結果でも価値がある。** 「回転augmentationでは過学習を抑えられない」が
確定すれば、次に試すべき介入の範囲が絞られる。

#### 8.13.4 提案D: 以後の方式選択の前に、独立したtest splitを確保する（確度: 高）

現在のvalidation 18動画は、**label policy・class weight・正規化・R0/R1・checkpoint選択**という
多数の方式選択に既に使用されている。**独立したtestではない。**
8.12.12の「残る留保」でも記載済みだが、**提案Cを実行するなら、その前に扱いを決めるべきである。**

選択肢は「現180動画から新たにtest splitを切る（train/valがさらに減る）」か
「当面はvalidationを選択用と割り切り、production判断の直前に別途データを確保する」かである。
**これは方針判断であり、実装チャットとしては決めない。**

#### 8.13.5 提起したい論点: 現在の特徴量で課題が解けるのか

**これは提案ではなく、方針管理チャットに判断を仰ぎたい論点である。**

検証(3)で確定したこと:

- モデルが受け取る2特徴の一方`intensity`は、**画像グレー値そのもの**（相関1.0000）。
- 予測確率は**輝度に強く従い**、**高輝度点では画面端からの距離が強く効く**（両checkpoint一貫）。
- 定性評価のFPは「解剖学的に無関係だが、**形状と輝度が似た**構造」に集中する。

precision 5%という水準と併せると、**`intensity`と`confidence`という特徴量の組では、
大腿骨断面と「明るく細長い他の構造」を分離する情報が足りていない可能性**がある。
もしそうなら、学習長・augmentation・正則化のいずれを変えても頭打ちになる。

ただし**これは仮説であり、本チャットは証明していない**。検証するなら、
たとえば「特徴量を落としたablation（座標のみ／confidenceのみ）で性能がどれだけ落ちるか」
を測れば、各特徴の寄与を切り分けられる。CPU不可・追加学習が必要である。

#### 8.13.6 全提案に共通する留保

1. **すべて単一seedの結果である。** 提案A（延長しない）は効果量が極めて大きいため
   seedで覆らないと考えるが、**提案Cで比較する差はそれより小さい可能性が高く、
   単一seedでは判断しきれない場合がある**。その場合は「決められなかった」と報告する。
2. **F1約9%・precision約5%という水準では、1〜2ptの差の意味づけが難しい。**
   提案Cでも、最終指標の微差ではなく**過学習の起き方**を主要な観測対象にすべきである。
3. validation 18動画は独立testではない（提案D）。
4. 本節は実装チャットの提案であり、**承認なしに学習は開始しない**。

#### 8.13.7 未処理の実務事項（P4判断とは独立）

- **未コミット:** 新規10ファイル（検証(2)(3)のchecker群とそのテスト、GT連結成分checker、
  shell構文check）＋`docs/stage5/FILES.md`変更＋`check_dummy/check_stage5_effective_run_config.sh`の変更2件。
- **任意の再実行:** 8.12.21(e)2で同時標準化に動画輝度軸を追加した。
  再実行すれば説明割合の上限が確定するが、**過半が残るという結論は動きにくく、
  P4判断には影響しない**。lastの群別集計も同様に未実行。

### 8.14 管理チャット返信: 解釈の訂正・S5-15終了とS5-16への移管（2026-09-20）

#### 8.14.1 終了判断と記録の読み方

ユーザー判断によりS5-15をここで終了します。R0/R1の5 epoch比較、R0の新規50 epoch学習、
固定21動画のbest/last評価、可視化と追加解析を実施結果として保持します。
**現設定の100〜200 epoch延長・production採用には進まず、未実施の後続工程を終了し、
改善方針の再検討をS5-16へ移管します。** production候補確定まで成功したという終了ではありません。

8.1〜8.13はログとして一切書き換えず、本節に訂正と現行判断を追記します。ログ内の
「確定」「否定」「原因」「説明割合」等を引用するときは以下の留保を必ず反映してください。
旧仮説・未実施提案を既成事実として新しい要約へ持ち越さないでください。

#### 8.14.2 表現・解釈の訂正事項

1. **過学習の強い示唆と「記憶の確定」を区別する。**
   epoch50のtrain sanity recall97.36%対validation6.97%、TP0 10/18、validation loss悪化は、
   このrun/splitでの過学習・汎化不足と強く整合します。ただしtrain sanityは3動画であり、
   「記憶という機序が確定」「小標本や分布差では説明できない」「全動画で一様な崩壊」とはしません。
   動画ごとの例外・撮影条件差・GT特性の寄与を除外していません。
2. **平均予測確率gapは、閾値を使わない要約値であって識別力の完全な尺度ではない。**
   GT positive/background間の平均確率差は校正や確率の圧縮にも影響され、順位性能を直接測る
   AUROC/AUPRCとは異なります。gap低下だけで「threshold調整では回復不能」と断定しません。
   同様にIoUとのepoch間相関は消極性の機序を否定する因果証拠ではありません。
   現設定で延長しない判断は、loss・動画別指標・TP0・定性評価を総合したものです。
3. **最良epochと学習期間を一般化しない。**
   このrunではIoUのbestはepoch6、val loss最小等はepoch8付近で、良好な領域は概ね6〜10です。
   別runのP3 epoch5との比較だけで、1 epoch増加の因果効果や「以後の学習は常に有害」を断定しません。
   単一seedの結果を別条件・別seedの長期成績へ保証しません。
4. **定性観察は認識機序・解剖学的分類の証明ではない。**
   輪郭・細長いアーティファクト等のFPは観察事実として保持しますが、形状・輝度のどちらに反応したかは
   分離されていません。「相対的に明るい構造を学んだ」は初期仮説です。bestでは絶対輝度側との関連が
   強いという解析があっても、lastで同様ではなく、モデル一般の性質として記述しません。
   intensityと画像値の一致は調べたデータと変換経路での結果であり、相関1だけで任意データの
   値の完全一致を保証しません。
5. **位置×輝度は関連であり因果効果ではない。**
   高輝度点で位置による予測差が大きい傾向は両checkpointに見られますが、他の形状・撮影条件を
   固定した介入ではありません。画像位置とモデルXYZも同一視しません。
6. **2領域動画群の関連とフレーム内の失敗原因を区別する。**
   8対10動画の性能差は探索的な所見です。同一動画内で領域数と失敗の一貫した関連が得られないことは、
   「領域数が原因ではないと証明した」ことではありません。GT点数が多いだけで規模の交絡も除外できません。
   lastではゼロrecallが多く比較能力が低下していることを残します。目視との一致に用いた閾値の
   事後修正も、独立検証ではなく探索的分類の履歴として明記します。
7. **標準化の32.5%/67.5%は暫定値であり上限・下限ではない。**
   実施したフレーム内輝度×距離binで群間平均差の32.5%が縮小した、という結果に限定します。
   動画輝度を含む追加軸は未実行なので「少なくとも2/3が説明不能」「過半が残る結論は動かない」と
   予測しません。追加軸を実行しても全交絡に対する説明割合の上限は確定しません。
   共通binの欠落重み0はoverlapがあることを示すだけで、各binの十分な標本数や動画の代表性を
   保証しません。残差を未知の形状因子の因果効果と呼ばず、検定未実施の記述的解析とします。
8. **背景群差の小ささから全体的な校正差を排除しない。**
   対象解析ではGT positiveの平均確率差が背景より大きかった、という範囲にとどめます。
   クラス別の校正変化・非線形変換でも異なる見え方になり得るため、平均値だけで排除できません。
9. **augmentationの短期結果と長期効果を区別する。**
   「過学習前には原理的に効果が現れない」「P3は測りたいものを測れなかった」は採用しません。
   短期でも効果は現れ得ます。P3は5 epochでの比較として有効ですが、長期の汎化効果までは
   評価していません。FP/recallの同時低下を正則化の典型的証拠とせず、長期比較は未検証案とします。
10. **評価範囲とprivacy上の対応を明示する。**
    フレーム可視化文書の全21動画集計と公式評価のvalidation18動画集計は混ぜません。
    0始まり/1始まりaliasの照合履歴も保持し、数値一致だけで全index対応の完全証明としません。
    validation18動画は方式選択済みで独立testではなく、点数の多さを独立動画数の代用にしません。

#### 8.14.3 8.13の提案への扱いとS5-16への申し送り

| 提案・論点 | S5-15終了時の扱い |
| --- | --- |
| A: 100〜200 epoch延長をしない | 採用。現R0設定を単に延長する根拠は乏しいため、S5-15の延長は行わない |
| B: 現設定をproduction採用しない | 採用。精度・汎化・FPの結果を総合して見送る。Stage6の許容基準が確定したという意味ではない |
| C: R0/R1各25 epoch比較 | S5-16の未承認候補。長期効果は未検証だが、必須・本命と先に確定しない。既存R0のepoch25等を再利用できるかも監査し、両arm再学習を自動追加しない |
| D: 独立testの確保 | 必要性をS5-16で扱う。現在のvalidationを選択用として使うことと独立評価を区別する。使用済み180動画を事後に切り直すだけで、これまでの選択から独立したtestになるとはしない |
| 特徴表現不足 | 未証明の仮説として移管。XYZ/局所幾何もモデル入力であり、intensity/confidenceの2値だけで予測しているわけではない。特徴削除ablationは寄与を調べるが、情報不足や課題の不可能性を単独で証明しない |
| 追加の標準化・形状/移動量/密度解析 | 未承認候補として保持。測れるという理由だけで診断を追加せず、次の判断を変え得るかと費用を先に評価する |

S5-16は「汎化不足への改善方針策定と限定比較」として分離します。最初は結果を短く要約し、
候補介入について目的、切り分けられる要因、既存成果物の再利用、計算量、採否/停止基準を比較して、
最初の一手を選びます。**移管は実装・追加学習・GPU/CPU追加診断の実行承認ではありません。**
R0/R1・best/lastは履歴baselineとして保持し、R1不採用を一般的なaugmentation無効論へ広げません。

#### 8.14.4 終了後の文書・実務整理

本書8.1〜8.13と元metrics・manifest・checkpoint・可視化成果物は削除・上書きせず保持します。
短い終了要約とS5-16開始文書は別に用意し、詳細の節へ参照を付けます。管理記録・評価レポートへも
終了状態を反映する必要がありますが、本返信だけで他文書が更新済みとはしません。

未コミットのchecker/テスト/記録修正は実際のgit statusで対象を確認し、コード・文書のみを別途
コミットします。8.13.7の件数を最新状態と仮定しません。これは成果物整理であって新規診断の承認では
ありません。保存仕様逸脱、manifest突合の範囲、未実施の追加軸解析などは未解消の履歴として明示します。

## 9. 次のアクション（2026-09-22更新、S5-15終了・文書同期済み）

1. **完了:** P1/P2、GPU preflight、P3 R0/R1各5 epoch比較、manifest CPU突合。
   P3はR0維持。保存仕様逸脱を保持し、欠落checkpointの再生成は行わない。
2. **完了:** R0新規50 epoch、実効config／保存確認、固定21動画best6／last50評価、可視化と8.12の追加解析。
3. **終了判断維持:** 現R0の100〜200 epoch延長・production採用・S5-15内の追加探索は行わない。
   省略した10 epoch pilotや未実施の後続工程を完了扱いにしない。
4. **文書同期完了（09-22）:** 8.14の留保を適用し、管理記録D-035・現在状態・timelineと
   評価レポート9.10へ50 epoch結果・終了判断を反映した。8.1〜8.14の過去ログは変更していない。
5. **移管先更新:** S5-16は方針策定完了（v3、D-036／D-037）。後続は
   [Step 0報告・残務管理](../s5-16/stage5_step0_report_to_policy_chat.md)で扱う。
   B′分割・封印は09-22ユーザー回答で未実施。Step 0を含むS5-16完了後も本総括管理で管理し、S5-17実装チャットへ依頼する（D-038）。
6. **deferred:** R0/R1各25 epoch、特徴削除ablation、標準化の追加軸はS5-17〜S5-20の範囲外。
   実装済みcheckerの存在を実行承認と読み替えない。
7. **成果物整理:** 09-22の文書作業開始時HEADは`2f73520`。旧ログの未コミット件数は流用せず、
   今後のコミット依頼時に現差分を確認する。既存変更を保持し、`.tmp/`・患者画像・H5・重み・private対応表を含めない。

S5-15は終了記録として参照する。新たな実行申請・結果はStep 0以降の個別記録へ残す。
