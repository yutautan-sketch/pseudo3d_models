# Stage 4 crop品質・stray ignore調査計画

作成日: 2026-09-13  
最終更新日: 2026-09-14  
編集対象ディレクトリ: `/workspace/Stage2to4`（主対象）、`/workspace/Stage5`（入力契約・影響確認）  
状態: teacher v7構築・全180 H5の機械受入・保存ラベル可視化完了

関連文書:

- `docs/stage2to4/stage4/stage4_phase5_fullvideo_cvat_review_implementation.md`
- `docs/stage2to4/stage4/stage4_cvat_snapshot_authoritative_label_revision_plan.md`
- `docs/stage2to4/stage4/stage4_deleted_xml_annotation_invalidation_plan.md`
- `docs/stage2to4/stage4/stage4_contour_teacher_improvement_plan.md`
- `docs/stage2to4/stage4/stage5_s5_12_stage4_crop_quality_investigation_request.md`

## 1. 背景

Stage 5 S5-12の学習前preflightで、teacher v6の全181 H5のうち2 H5に、保存済みBBoxの
いずれにも属さない`point_label=-1`が見つかった。Stage 5の
`bbox_noncontour_background`は、ignoreを「BBox内かつpositive外」としてbackgroundへ変換するため、
この契約を満たさないH5をfail-fastで拒否している。

匿名化された検出結果と確定した実データの対応は次のとおりである。記載するframe番号はまず
`frame_order`として扱い、監査時にH5の`frame_index`と照合する。

| alias | split | video_name | 問題frame_order | ignore | BBox内ignore | stray ignore | stray率 |
|---|---|---|---|---:|---:|---:|---:|
| `train_068` | train | `20250626_090652_6340` | 38, 43, 44, 46 | 97 | 87 | 10 | 10.3% |
| `val_009` | validation | `20250625_161030_0550` | 51 | 843 | 837 | 6 | 0.7% |

全181 H5ではignore 262,347点に対してstrayは16点である。no-BBox frame上のignoreは0で、
schema、3値label、`valid_mask == (point_label != -1)`は正常である。

ユーザーの目視では、`train_068`は動画の大部分でBBoxがlocal crop外、`val_009`は時系列に沿って
BBoxが右端へ移動し、問題frame付近でcrop外へ出る。これは既存除外動画
`20250626_090758_8000`の`local_crop_tracking_drift`と同種の可能性がある。

## 2. 調査で分離する2つの問題

### 2.1 label幾何契約の不一致

crop端へクリップされたBBoxが幅0または高さ0になる場合の扱いが、処理間で一致していない可能性が
高い。

- `xml_bbox_to_local()`は`x2 > x1 and y2 > y1`を満たさないBBoxを`valid=False`にする
- Stage 4/5のlabel-policy監査は`right <= left`または`bottom <= top`を無効として除外する
- 修正前のCVAT-authoritative反映に使われた`bbox_bounds()`は、座標列に対して`right < left`または
  `bottom < top`だけを無効としていたため、幅0・高さ0のBBoxを1 pixel線として扱えた
- CVAT反映時に`LocalBBox.valid`が座標列への変換で失われると、その1 pixel線上へignoreが付く一方、
  Stage 5監査ではBBox外と判定される可能性がある

実データ監査により、この差が16点を全て説明することを確認した。

### 2.2 local crop品質

Stage 2のlocal cropは動画単位の固定offsetを全frameへ適用する。大腿骨が時間方向に移動すると、
XML BBoxが`fully_visible`から`partially_clipped`、`fully_outside_crop`へ遷移し得る。

label幾何を修正してstray ignoreを0にしても、対象がcrop外にあるframeの教師品質は回復しない。
したがって、実装境界条件の修正と、動画・frameの採否判断は別々に行う。

## 3. 現時点の重要な補足

既存のStage 4受入検査と今回の検出は矛盾しない。従来検査は主に次を確認していた。

- no-BBox frameのpointがbackground中心であり、ignoreがない
- 保存BBox内の非positive pointがignoreである
- CVAT snapshot対象frameのpositiveがCVAT mask外へ出ない

一方、「全ignore pointが少なくとも1つの有効な保存BBox内にある」という逆方向の包含条件は、
全181 H5の受入条件ではなかった。今後はこの条件をStage 4側にも追加する候補とする。

また、既存full-video preflightはPhase 3で選ばれた60動画・129 BBoxを前提とする。任意の動画に
存在する全XML BBoxを監査する用途には、そのままでは不足する可能性がある。

## 4. 実データ監査結果

### 4.1 label幾何契約

read-only監査は正常終了し、次を確認した。

- 2動画・5 frameに存在する16 stray ignoreの全点が、退化BBoxを無効とする厳密判定と、1 pixel線と
  する判定の差で説明された
- source v5とv6のstray数は一致した
  - `20250626_090652_6340`: v5=10、v6=10
  - `20250625_161030_0550`: v5=6、v6=6
- v6 deleted-XML invalidationはstrayの発生源ではなく、v5から既存labelを伝播しただけである
- 入力H5、XML、manifest、teacher configのchecksumは監査中に変化しなかった

`20250625_161030_0550`のframe 51では、BBoxが右crop外へ完全に移動し、保存BBoxが
`[255.0, 117.706, 255.0, 135.824]`の幅0へ退化している。`valid_contour=False`かつ
CVAT authoritative maskはemptyであるにもかかわらず、幅0 BBoxを1 pixel線として扱う経路が6点を
ignoreにした。したがって、境界条件不一致を根本原因として確定する。

### 4.2 crop可視性

| video_name | XML BBox | fully visible | partial | outside | 判定 |
|---|---:|---:|---:|---:|---|
| `20250626_090652_6340` | 10 | 0 | 4 | 6 | 動画全体がcrop品質不良 |
| `20250625_161030_0550` | 14 | 9 | 4 | 1 | 終端だけがcrop外へ遷移 |
| `20250626_090758_8000`（既存除外） | 13 | 0 | 4 | 9 | 動画全体がcrop品質不良 |

`20250625_161030_0550`は右端方向へ連続的に移動する。

| frame_order | crop status | visible fraction | 保存label状態 | 採否 |
|---:|---|---:|---|---|
| 47 | partially clipped | 0.7857 | CVAT authoritative、valid | 維持 |
| 48 | partially clipped | 0.7321 | CVAT authoritative、valid | 維持 |
| 49 | partially clipped | 0.6494 | CVAT authoritative、valid | 維持 |
| 50 | partially clipped | 0.3031 | CVAT authoritative、valid | 維持 |
| 51 | fully outside crop | 0.0000 | CVAT authoritative empty、invalid | 無効化 |

frame 47〜50はcrop内に実在する部分へ人力確認済みCVAT maskがあり、保存輪郭もvalidである。今回の
異常検出対象はframe 51だけであるため、visible fractionだけから新しい全体閾値を導入せず維持する。
frame 50を含む低可視率partial cropの一般的な採否基準は、別途全データ分布を監査する場合に決める。

### 4.3 対象別補正方針

- `20250626_090652_6340`: 既存除外例と同じ`local_crop_tracking_drift`として動画単位で除外する
- `20250625_161030_0550`: frame 51だけをcrop-quality invalidationする
- frame 51は全pointをbackgroundとし、active BBox rowを除去して元情報をprovenanceへ退避する
- frame 47〜50およびその他の非対象frameは変更しない
- 幅0・高さ0 BBoxを1 pixel線として扱わない境界条件修正を、今後の再生成経路へ追加する
- teacher v6は上書きせず、補正結果を新しいversioned teacher rootへ作成する

## 5. 調査中の保護条件

- teacher v6の`annotated`、`collected`、可視化を上書きしない
- source pseudo3D H5、VOC XML、返却CVAT snapshot、既存除外manifestを変更しない
- Stage 5のfail-fastを緩めず、stray pointを黙ってbackgroundへ変換しない
- 実video ID、frame番号、入力SHA-256を確定するまで除外・無効化を適用しない
- 調査結果と補正成果物は新しいversioned rootへ保存する
- crop品質問題を丸め誤差または16点だけの軽微な問題として処理しない

## 6. 調査手順

### Step 1: 対象IDと入力証拠の固定（完了）

ユーザーが保持する対応表から`train_068`、`val_009`を実video IDへ変換する。次をread-onlyで固定する。

- split、train/validation list上の位置
- teacher v6 annotated/collected H5の絶対pathとSHA-256
- Stage 5 stray detailのframe order、丸め後pixel座標、保存BBox座標
- source v5/v6 schema、label authority、CVAT provenanceの有無
- 対応するpseudo3D H5、VOC XML root、CVAT package/snapshotの有無

匿名aliasと実video IDの対応表は共有不要な場所へ維持し、調査出力には必要最小限の識別情報だけを
記録する。

### Step 2: H5内部のframe単位label/BBox監査（完了）

該当2 H5について、strayがあるframeごとに次を出力する。

- `frame_order`、`frame_index`
- point数とpositive/background/ignore数
- stray ignoreのpoint index、元`pixel_xy`、`np.rint`後の座標
- active `frame_annotation`の全BBox rowと`bbox_index`
- `bbox_local_xyxy`の幅、高さ、有限性、退化判定
- `bbox_raw_xyxy`、`bbox_xml_xyxy`、保存XML path
- `selected_contour_source`、annotation reason、valid contour
- `manual_review_fullvideo_frames`のauthority、CVAT mask status、mask SHA
- v5からv6への変更対象か、単純伝播frameか

Stage 4生成時、CVAT-authoritative反映時、Stage 4監査時、Stage 5監査時のBBox unionを同じpointへ
適用し、どの境界条件で判定が分かれるかを明示する。

### Step 3: 未クリップBBoxによるcrop可視性監査（完了）

既存の権威関数を再利用する。

- `_project_bbox_to_local_unclipped()`
- `_bbox_crop_metrics()`

Phase 3 targetだけでなく、対象2動画の全frame・全strict XML BBoxについて次を出力する。

- unclipped/projected BBoxとclipped BBox
- `visible_fraction`
- `fully_visible`、`partially_clipped`、`fully_outside_crop`
- `touches_left/right/top/bottom`
- frameごとのstatus遷移
- BBox frame数に対するoutside/partial割合

既存除外例`20250626_090758_8000`にも同じ計算を行い、同一指標で重篤度を比較する。

### Step 4: 根本原因の確定（完了）

次の件数が一致するか確認する。

1. 退化BBoxを1 pixel線として扱った場合だけ生じるignore点数
2. Stage 5が有効BBox外と判定した16点
3. 問題frameのcrop statusとedge方向
4. v5/v6のどの段階でignoreが初めて生成されたか

16点が全て一致すればlabel幾何契約不一致を確定する。一致しない場合は、複数BBox union、
CVAT mask適用、BBox row除去、float丸めを個別に追跡し、複数原因を混同しない。

### Step 5: データ採否の決定（完了）

調査結果から動画ごとに次のいずれかを選ぶ。

| 判定 | 適用候補 |
|---|---|
| cropは十分で、退化BBoxの境界条件だけが原因 | label幾何修正のみ |
| 動画の大部分で対象がcrop外 | 動画単位の`local_crop_tracking_drift`除外 |
| 一部の連続frameだけがcrop外 | frame単位のcrop-quality invalidation |
| crop外区間も教師として必要 | 将来の再crop対象として保留し、現teacherからは除外 |

比較結果に基づき、`train_068`は動画除外、`val_009`はframe 51だけの無効化とした。frame無効化は
削除XML用manifestへ意味を混在させず、crop品質専用のversioned manifestを作る。

## 7. 修正方針

確定した根本原因に対して最小限の変更を行う。

### 7.1 BBox境界条件

- BBox unionの有効条件を全処理で`right > left and bottom > top`へ統一する
- 可能なら`LocalBBox.valid`を失わずにCVAT反映へ渡す
- 退化BBoxを1 pixelのignore領域へ暗黙変換しない
- 幅0、高さ0、四辺それぞれのcrop外、複数BBoxをsynthetic testへ追加する

実装では`stage4_manual_review.bbox_bounds()`に連続座標上の正面積判定と、整数化後の
`right > left and bottom > top`判定を追加した。これによりCVAT-authoritative反映、manual import、
review packageのdrawable判定で同じBBox union契約が使われる。過去teacherの原因監査は修正後も
再現できるよう、旧1 pixel線判定を監査スクリプト内のread-only legacy計算として隔離した。

2026-09-13の回帰検査では次の3系統がすべてexit code 0で完了した。

- Stage 4 CVAT manual round-trip: 退化BBoxが1 pixel ignore線を作らず、既存のCVAT export/import、
  full-video、context-only、text-free package契約も維持された
- crop-quality/stray-ignore監査: 旧teacherでの退化BBox問題を引き続き再現・説明でき、read-only入力と
  決定的出力が維持された
- Stage 5 label-policy: BBox内判定とDataset適用の合成検査が通過した

これにより境界条件修正を完了とし、既存teacher v6の16 stray pointそのものは、次の明示的な動画
除外・frame invalidationで解消する。

### 7.2 teacher補正

- 動画除外またはframe無効化は明示manifestでのみ行う
- 無効化frameは全pointをbackgroundとし、active BBox rowを除去する
- 元BBox/CVAT情報はprovenanceとして退避する
- v6を上書きせず、仮の次版teacherへ適用する
- 対象外動画・frameのpoint cloudとlabelを変更しない

補正対象は、用途を混在させない2種類の固定manifestとして実装した。

- `stage4_video_exclusions_v2.csv`
  - v1の`20250626_090758_8000`行を同一内容で維持する
  - `20250626_090652_6340`を`local_crop_tracking_drift`として追加する
- `stage4_crop_quality_invalidations_v1.csv`
  - `20250625_161030_0550`のframe order/index 51だけを対象とする
  - 保存BBox 1、stray ignore 6、`fully_outside_crop`、visible fraction 0を固定証拠とする
  - actionは`invalidate_entire_frame`、reasonは`local_crop_fully_outside`とする

`validate_stage4_crop_quality_manifests.py`は、v1除外の欠落・変更、除外動画とframe無効化の重複、
監査CSVの件数・crop状態の変化、重複frame、非0可視率のoutside指定をfail-fastする。ここではH5を
書き換えず、検証summaryと動画除外適用済みtrain manifestだけを生成する。

実データmanifest生成は2026-09-14に完了した。182動画中180動画がenabled、2動画がdisabledであり、
frame invalidationは`20250625_161030_0550`のframe 51の1件、除去対象stray ignoreは6点で固定された。

teacher補正は次の3ファイルとして実装した。

- `pseudo3d/annotation/apply_crop_quality_invalidations.py`
  - teacher v6をread-only入力とし、対象frameの全pointをbackgroundへ変換する
  - active `frame_annotation`と対応するmanual-review BBox rowを除去・archiveする
  - CVAT frame provenanceの補正前rowをarchiveし、active rowのlabel authorityをcrop-quality manifestへ更新する
  - XMLファイルが現在も存在すること、保存BBoxが退化していること、補正前ignoreが固定6点であることを
    fail-fastで検証する
  - v6の`xml_annotation_invalidation`履歴と非対象データを保持し、verified resumeに対応する
- `pseudo3d/batch/annotation/batch_apply_stage4_crop_quality_invalidations.py`
  - `train_manifest_cropclean_v2.csv`のenabled 180動画だけを処理する
  - source v6の181動画を上書きせず、除外2動画を次版出力へ含めない
  - annotated/collectedをSHA-256一致で収集し、固定件数をsummaryへ記録する
- `pseudo3d/pipelines/build_stage4_bbox_ranked_v7_crop_quality.sh`
  - 上記契約を固定値付きで実行し、次版teacherを独立rootへ構築する

次版schema/tokenは`bboxrank_v7_cvat_authoritative_crop_quality_v1`、既定出力rootは
`global_local_l75_w31_c12_area15_bboxrank_v7_cvat_authoritative_crop_quality_v1`とする。

再cropはforeground点、座標、CVAT mask対応まで変えるため、局所補正で安全に解決できない場合だけ
別計画として扱う。

## 8. 検証・受入条件

### 8.1 Synthetic

- 退化BBoxをStage 4/5の全経路で一貫して無効化できる
- 有効な1 pixel幅BBoxを仕様上許可するか拒否するかが明示されている
- partial/outside cropの四方向を正しく分類できる
- 複数BBox frameでunion判定が一致する
- frame invalidationと動画除外が非対象データへ波及しない

### 8.2 実データ

- 全学習・validation H5で`stray_ignore_outside_bbox_count=0`
- no-BBox frameのignoreが0
- `valid_mask == (point_label != -1)`
- 問題2動画のcrop status、採否、変更point数をframe単位で説明できる
- 非対象H5はbyte identity、またはversion metadataを除くsemantic identityを満たす
- annotated/collectedが対応し、収集時にschemaとchecksumが保たれる
- 問題frameと遷移前後frameの保存label可視化を目視確認する

## 9. Stage 5への影響整理

S5-12 Run A/Bは本調査と補正後の入力listが固定されるまで保留する。

- train動画を除外する場合、train file数とsampling分布の変更を記録する
- validation動画またはframeを補正する場合、既存validation指標との比較条件が変わることを記録する
- S5-07〜S5-11は旧v6入力による既存結果として保持し、直ちに全再学習とはしない
- 補正対象の点数・frame数・動画数を確定後、結論への影響可能性から再学習要否を判断する
- Stage 5側のfail-fastは最終teacherで再実行し、通過ログを受入証拠にする

## 10. 完了した監査実行

次は実video IDと問題frameを固定したread-only監査を実行する。実装は次の3ファイルである。

- `pseudo3d/analysis/audit_stage4_crop_quality_stray_ignore.py`
- `checks/stage4/check_stage4_crop_quality_stray_ignore.py`
- `pseudo3d/pipelines/audit_stage4_crop_quality_stray_ignore.sh`

合成検査を実行する。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_crop_quality_stray_ignore.py

echo "crop_quality_stray_ignore_synthetic_exit_code=$?"
```

通過後、teacher v6、pseudo3D H5、strict XML、source v5を変更しない実データ監査を実行する。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

OVERWRITE=0 \
bash pseudo3d/pipelines/audit_stage4_crop_quality_stray_ignore.sh

echo "crop_quality_stray_ignore_realdata_exit_code=$?"
```

監査は対象2動画の16 stray pointが指定した5 frameだけに存在することをfail-fastで照合し、全XML
BBoxの未クリップcrop可視性を既存除外例`20250626_090758_8000`と同じ指標で出力する。結果の
`audit_summary.json`、`stray_points.csv`、`frame_metrics.csv`、`bbox_geometry.csv`、
`video_summary.csv`を比較し、動画除外・frame無効化・境界条件修正の組み合わせを確定した。

## 11. 次の実装順序

1. ~~`bbox_bounds()`を含むBBox union経路で、幅0・高さ0のBBoxを一貫して無効化する合成検査と修正を
   追加する。~~（完了）
2. ~~`stage4_video_exclusions_v2.csv`を作り、既存除外を維持したまま
   `20250626_090652_6340`を`local_crop_tracking_drift`として追加する。~~（完了）
3. ~~crop品質専用のversioned frame invalidation manifestを作り、
   `20250625_161030_0550`のframe order/index 51だけを固定する。~~（完了）
4. ~~teacher v6を入力として新しいteacher v7を構築する。無効化frameを全background化し、BBoxとCVAT
   provenanceをarchiveする。CVATの再編集は行わない。~~（180 H5の構築完了）
5. ~~全出力でstray ignore=0、no-BBox ignore=0、label/valid-mask整合、対象外semantic identity、
   annotated/collected対応を検証する。~~（専用checkerで全180 H5の受入完了）
6. 対象frame 47〜51、除外前動画、通常対照例の保存label可視化を確認してからStage 5入力を更新する。

## 12. 補正manifestの検査・生成手順

合成検査を実行する。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_crop_quality_manifests.py

echo "crop_quality_manifests_synthetic_exit_code=$?"
```

通過後、監査証拠との照合と`train_manifest_cropclean_v2.csv`生成を行う。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

OVERWRITE=0 \
bash pseudo3d/pipelines/build_stage4_crop_quality_manifests_v2.sh

echo "crop_quality_manifests_realdata_exit_code=$?"
```

実データでは、182行、enabled 180、disabled 2、frame invalidation 1、除去予定stray 6として正常に
固定された。

## 13. teacher v7適用手順

まず、単一H5補正、履歴保持、非対象伝播、決定的resume、安全側の拒否を合成検査する。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_crop_quality_invalidation_apply.py

echo "crop_quality_invalidation_apply_synthetic_exit_code=$?"
```

通過後、teacher v6を上書きせずteacher v7のannotated/collected 180 H5を構築する。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

SKIP_EXISTING=1 \
bash pseudo3d/pipelines/build_stage4_bbox_ranked_v7_crop_quality.sh

echo "crop_quality_v7_build_exit_code=$?"
```

受入期待値は、動画180、除外2、frame invalidation 1、active BBox除去1、positive除去0、ignoreから
backgroundへの変更6である。次のStep 5では、全180 H5に対するstray/no-BBox/label-valid整合と、対象外
semantic identity、対象frame 47〜51の保存label可視化を検証する。

## 14. teacher v7機械受入検査

`checks/stage4/check_stage4_crop_quality_v7_acceptance.py`は、固定manifestと構築summaryを再照合した上で
次をread-only検証する。検査結果JSONだけをv7 rootへ新規作成する。

- v6 source 181、v7 annotated/collected 180、除外2動画の非混入
- annotated/collectedの動画ごとのbyte identity
- 全H5の3値label、`valid_mask`、正面積BBox union、no-BBox ignore、CVAT外positive契約
- 対象外179動画について、v7で追加したprovenance以外の既存dataset値のsemantic identity
- 対象frameの全background化、active BBox除去、v6 BBox/CVAT provenanceの完全なarchive
- invalidated frame 1、BBox 1、ignore 6というmanifest・summary・H5実値の一致

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_crop_quality_v7_acceptance.py

echo "crop_quality_v7_acceptance_exit_code=$?"
```

初回は`crop_quality_acceptance_step5.json`を作成する。同じ入力での再実行は同一内容を
`verified_existing`として扱い、内容が異なる既存reportは`--overwrite`なしでは上書きしない。

実データ検査は、source H5 181、v7 H5 180、対象外semantic identity 179、frame invalidation 1、
BBox除去1、ignore除去6として通過した。全v7入力でstray ignore、no-BBox ignore、BBox内background、
CVAT mask外positiveはいずれも0である。この段階ではteacher v6 H5、XML、CVAT snapshotを変更しない。

## 15. teacher v7保存ラベル可視化

既存の保存ラベル可視化をv7へ拡張する。automatic contourやXMLを再計算せず、v7 H5の
`annotation/point_label`とactiveな`frame_annotation`だけを描画する。表示順序と色は次のとおりである。

1. 元のlocal encoder画像
2. background点（青）
3. ignore点（黄）
4. CVAT snapshot mask（水色）
5. positive点（赤）
6. activeな保存BBox（緑）

`xml_annotation_invalidation`と`crop_quality_invalidation`を独立に検証する。いずれかで無効化された
frameは保存labelが全backgroundであり、active BBoxを持たないことを必須とする。過去のCVAT maskは
checksum・frame対応まで確認するが、描画対象から抑制する。crop-qualityでarchiveされた旧BBoxも描画しない。
`frame_labels.csv`には両無効化のフラグ、action、reason code、CVAT mask抑制状態を別列で保存する。

追加・更新対象:

- `pseudo3d/export/export_stage4_point_label_visualization.py`
- `pseudo3d/batch/export/batch_export_stage4_point_label_visualization.py`
- `checks/stage4/check_stage4_crop_quality_invalidation_visualization.py`
- `pseudo3d/pipelines/export_stage4_v7_crop_quality_point_label_visualizations.sh`

まず合成検査を実行する。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

/home/kodaira/anaconda3/envs/dualtrack311/bin/python \
  checks/stage4/check_stage4_crop_quality_invalidation_visualization.py

echo "crop_quality_v7_visualization_synthetic_exit_code=$?"
```

通過後、最初に対象動画だけを新規出力する。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

VIDEO_NAMES=20250625_161030_0550 \
SKIP_EXISTING=0 \
bash pseudo3d/pipelines/export_stage4_v7_crop_quality_point_label_visualizations.sh

echo "crop_quality_v7_visualization_target_exit_code=$?"
```

`annotation_textures_v7_crop_quality_labels/20250625_161030_0550/frames`のframe 47〜51を確認し、
frame 51に赤positive、水色CVAT mask、黄ignore、緑BBoxがなく、保存点が青backgroundだけであることを
確認する。問題がなければ全180動画を一括出力する。

```bash
cd /mnt/data/3d_projects/models/Stage2to4

VIDEO_NAMES= \
SKIP_EXISTING=1 \
bash pseudo3d/pipelines/export_stage4_v7_crop_quality_point_label_visualizations.sh

echo "crop_quality_v7_visualization_all_exit_code=$?"
```

一括検査の期待値は、H5 180、XML無効化frame 7、crop-quality無効化frame 1、抑制CVAT frame 8、
positive outside CVAT mask 0、CVAT/point-label mismatch 0である。

2026-09-14、専用合成検査は次の2契約を含めて通過した。

- crop-quality無効化frameで旧CVAT maskと旧BBoxが描画されず、保存済み全background labelだけが描画される
- v7 batch集計、XML/crop無効化件数、verified resumeが一致する

続いて`20250625_161030_0550`だけを先行出力し、frame 51にBBox、positive、ignoreおよび旧CVAT maskの
描画がなく、保存済みbackground点だけが表示されることを目視確認した。残作業は全180動画の一括出力と
summary確認である。

全180動画の一括出力も完了した。先行出力1動画はchecksum/provenanceを再検証して
`skipped_verified`、残り179動画は新規処理され、失敗および入力変更は0であった。集計値は次のとおりで、
固定期待値と一致した。

- XML invalidation: 7 frame / 2 video
- crop-quality invalidation: 1 frame / 1 video
- 抑制した旧CVAT mask: 8 frame
- 出力root: `annotation_textures_v7_crop_quality_labels`
- batch summary: `annotation_textures_v7_crop_quality_labels/summary.csv`

以上により、teacher v7のcrop品質補正、全H5機械受入、対象frame目視確認、全件保存ラベル可視化までを
完了とする。

## 16. Stage 5既定入力への切り替え

teacher v7の受入完了後、Stage 5の学習・推論・評価スクリプトの既定Stage 4入力を次へ切り替えた。

```text
/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711/
global_local_l75_w31_c12_area15_bboxrank_v7_cvat_authoritative_crop_quality_v1/collected
```

`Stage5/train_stage5.sh`の入力preflightもv7契約へ更新し、次を学習開始前にfail-fastで確認する。

- H5 180件、teacher schemaおよびlabel sourceがv7
- CVAT-authoritative provenance: 58動画、2960 frame
- 継承XML invalidation: 7 frame / 2動画、positive除去2124、BBox除去7
- crop-quality invalidation: 1 frame / 1動画、positive除去0、ignore除去6、BBox除去1
- 2除外動画が入力inventoryに存在しない
- XML/crop無効化frameが全backgroundかつvalidで、両manifest identityが全H5で共通

既存の学習方式、normalization、label policy、class weight等はこの切り替えでは変更しない。出力runが
teacher v6の既存runと衝突しないよう、既定experiment prefixも`bboxrankv7_cvatcropq`へ更新した。
学習を開始せず入力契約だけを確認する場合は次を実行する。

```bash
cd /mnt/data/3d_projects/models/Stage5

PREFLIGHT_ONLY=1 \
bash train_stage5.sh

echo "stage5_teacher_v7_input_preflight_exit_code=$?"
```
