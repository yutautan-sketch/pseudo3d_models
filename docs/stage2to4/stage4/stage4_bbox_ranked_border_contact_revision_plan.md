# Stage 4 BBox-ranked輪郭の境界接触見直し計画

作成日: 2026-09-05  
最終更新日: 2026-09-05  
編集対象ディレクトリ: `/workspace/Stage2to4`  
状態: 調査・方針固定中（未実装）

CVAT snapshot対象frameの最終positive authorityは、次の新契約を優先する。自動輪郭の
border-contact変更はCVAT maskを上書きせず、snapshot非対象frameまたはCVAT編集開始前のseedに
だけ影響する。

- `docs/stage4/stage4_cvat_snapshot_authoritative_label_revision_plan.md`

## 1. 背景

teacher v4 build後の自動輪郭可視化を目視したところ、local percentile候補が大腿骨断面に
適合して見える一方、それを内包してBBoxの辺まで膨張したglobal候補が採用される例が複数
確認された。BBox制約から見て過度な輪郭であり、自動輪郭選択を小規模に見直す。

本件は自動輪郭処理だけを対象とする。既存のCVAT修正済みmask、Task snapshot、元画像、
XML BBox、foreground point cloudは変更せず、CVATでの再編集も要求しない。

## 2. 現行ロジックの調査結果

`bbox_ranked_global_local`はglobal/localの各二値maskをBBoxで切り出し、外部連結成分を
独立候補として抽出する。現在の適格条件と順位は次のとおりである。

```text
min_area_ratio          : 0.02
sufficient_area_ratio   : 0.10
max_center_distance_norm: 0.50
center_weight           : 0.75
area_weight             : 0.25
```

面積スコアは面積比0.10以上で上限に達し、過大な候補を減点しない。BBox境界接触も基本選択の
適格条件・scoreに含まれない。同程度のscoreでは中心距離、面積score、最後に大きい
`filled_area`が優先される。このため、BBoxでclipされて中心が安定した過抽出global候補が、
内側のlocal候補より有利になり得る。

Phase 3にはすでに`border_contact_ratio`がある。

```text
border_contact_ratio = BBox辺上の輪郭境界画素数 / 輪郭境界画素総数
```

既存の129 BBox screeningでは、baselineの`border_contact_high`が56件、baseline
border-contact分布のp75/p95が1.0だった。一方、選択proposalはp75=0.0275、p95=0.2065、
max=0.3103であり、Phase 3の適格上限0.35を初期値として再利用する根拠がある。

## 3. 最小変更案

### 3.1 第1段階: border-contact上限だけを追加

global/localの両方へ、source非依存の適格条件を1つ追加する。

```text
max_border_contact_ratio: 0.35
```

- `border_contact_ratio > 0.35`の候補を不適格にする
- globalだけが不適格でlocalが適格ならlocalを採用する
- localにも同じ条件を適用し、localを無条件には優先しない
- 両方が不適格なら誤ったpositiveを作らず、輪郭なしとしてBBox内ignoreを維持する
- 1画素でも接触したら棄却する規則にはしない

第1段階ではcenter/area weight、`sufficient_area_ratio`、tie処理を変更しない。これにより
観察された境界膨張だけの効果を分離する。

### 3.2 第2段階: 残存時だけ面積上限を追加

第1段階の差分監査後も、BBox辺との接触率が低い過抽出が残る場合に限り次を検討する。

```text
max_area_ratio: 0.65
```

これもPhase 3で使用済みの値である。最初から同時に変更せず、必要性が確認された場合だけ
2つ目の条件として追加する。

### 3.3 今回行わない変更

- global/local maskのORまたはAND
- `local`というsource名による固定優先
- center/area weightの再探索
- local percentile、global threshold、foreground samplingの変更
- XML BBox、crop、point cloudの変更
- CVAT修正済みmaskの編集・再作成

## 4. CVAT修正済みmaskの再利用契約

自動輪郭を再生成しても、video、frame stem、frame order/index、画像shape、local座標、
`pixel_xy`、BBox geometryが同じなら、既存CVAT maskは座標変換なしで再利用できる。

ただし現行final importerはreview packageに記録されたsource v3 H5のSHA-256完全一致を要求する。
自動輪郭変更後のH5はSHAが変わるため、この検査を単純に無効化してはならない。実装時は次の
明示的なrebase手順を設ける。

1. border guard適用済みの自動teacherを新しいrunへ生成する
2. 旧/new間でframe stem・順序、画像shape、point cloud配列、BBox geometryを完全照合する
3. 照合に成功した動画だけ、既存CVAT snapshot maskを新しい自動teacherへ最後に適用する
4. snapshot対象動画の全Task frameではCVAT maskだけからpositiveを再構築する
5. 旧v2/v3/v4 H5、CVAT ZIP、Task backup、snapshotを上書きしない
6. 新しい自動config fingerprint、rebase元/先H5 SHA、CVAT ZIP SHAをprovenanceへ保存する

これは既存CVAT annotationの内容を変更する処理ではなく、同一座標契約を検証したうえで
新しいautomatic baselineへ再適用するための安全処理である。

## 5. 実装時の変更候補

主な自動輪郭側の対象:

```text
pseudo3d/annotation/annotate_pseudo3d_point_cloud.py
pseudo3d/analysis/audit_stage4_contour_teacher.py
pseudo3d/analysis/configs/<new_bbox_ranked_teacher_config>.yaml
pseudo3d/pipelines/<new_bbox_ranked_build_pipeline>.sh
checks/stage4/check_stage4_bbox_ranked_teacher.py
```

CVAT再利用側は、現行の厳格なSHA検査を維持したまま、専用rebase検査または明示optionを追加する。

```text
pseudo3d/batch/annotation/batch_import_stage4_phase5_fullvideo_cvat.py
checks/stage4/check_stage4_phase5_fullvideo_final_import.py
```

既存`stage4_bbox_ranked_teacher_v2.yaml`の意味やfingerprintは書き換えず、新しいconfig名と
teacher/run tokenを使用する。

## 6. 検証計画

境界接触ロジックを変更する前に、次の計画に従ってteacher v4の保存済みpoint labelを直接
可視化する。既存automatic contour表示だけでH5再作成を判断しない。

- `docs/stage4/stage4_v4_point_label_visualization_plan.md`

### 6.1 Synthetic

- 適切なlocalを、BBox辺まで膨張したglobalが内包する場合にlocalが採用される
- わずかな境界接触は0.35以下として許容される
- local側の過度な境界接触も同じ規則で棄却される
- 両候補が不適格な場合はpositiveを捏造せずignoreを維持する
- 既存のcentered-global、off-center-global、shared、ambiguous判定を壊さない
- config validation、H5 metadata、再読込、決定性が一致する
- CVAT rebaseで座標契約の一致を要求し、不一致と単純なSHA bypassを拒否する

### 6.2 Real data差分監査

181動画を対象に、少なくとも次を旧automatic teacherと比較する。

```text
global -> local
global -> none
local  -> none
source unchanged
positive point増減
valid/invalid BBox増減
border_contact_ratio分布
area_ratio分布
```

最初に境界接触が明らかな目視例を確認し、その後全件集計を行う。第1段階で解消できない
過抽出だけを抽出し、`max_area_ratio`追加の要否を判断する。

### 6.3 受入条件

- 目視対象の過抽出globalがlocalへ置換される、または安全に輪郭なしになる
- 境界非接触の既存良好例を不必要に変更しない
- foreground point cloudと3値label policyを維持する
- 既存CVAT maskを再編集せず、同一pixelで再適用できる
- reviewed BBoxの最終ラベルが現在のCVAT結果と一致する
- 新runを作成し、既存artifactを上書きしない

## 7. 現在の判断

実装開始時は`max_border_contact_ratio=0.35`の1変更だけを採用候補とする。実データ差分で
残存問題が確認された場合のみ、2つ目として`max_area_ratio=0.65`を検討する。現時点では
調査・方針記録までとし、コード、config、H5、CVAT artifactは変更しない。
