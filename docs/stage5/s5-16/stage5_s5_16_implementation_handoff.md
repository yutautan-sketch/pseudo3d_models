# Stage 5 S5-16: 汎化不足への改善方針策定と限定比較 ハンドオフ

作成日: 2026-09-20
最終更新日: 2026-09-22
作成元: Stage 5方針管理チャット
担当: Stage 5実装チャット
状態: **S5-16方針策定完了。3つの修正案を比較し、S5-17〜S5-20の実施順序を確定。S5-16自体では新規実装・学習・GPU推論・CPU診断を実施しない。**
追記（2026-09-21）: **8〜11章を追加。S5-16方針の精査結果、前提確認への回答、データ分割の決定
（追加testデータ完成まではB′案、完成後はA案）、S5-17〜S5-20のimplementation policyを記録。**
追記（2026-09-22, v3）: **精査後レビューを反映し、B′封印手順、train_core専用class weight、S5-18 Dice仕様、
S5-19のメモリ・parity方針、S5-20のgeometry座標系契約を更新。12章にv3変更点を記録。**
8〜12章も実行承認ではなく、各stageの提案値・実施量は開始前に確定し承認を得る。

S5-15は終了し、今後の検証・修正の考察をS5-16へ分離する。本書は追加実装・学習・GPU推論・
CPU診断の実行承認ではない。まず結果と問題の認識を共有し、次に行う限定比較は別途決定する。
S5-15の詳細ログ・元成果物は保持し、本書の短い要約で置き換えない。

## 1. 現在の検証状況

### 1.1 プロジェクトと参照文書

Stage5はStage2to4が生成したpseudo-3D H5点群を入力し、PointNeXt-Sで大腿骨のpoint-wise
segmentationを行う。動画全体のXYZを正規化した後、frame_orderによるoverlap windowへ分割する。
推論では元H5のpoint_indicesへ対応づけて各windowの確率を平均し、動画全体の予測を得る。

引き継ぎ時に読む文書は以下。古い検証結果や初期仮説と、後から訂正された解釈を区別すること。

- `docs/stage5/s5-15/stage5_s5_15_report_to_policy_chat.md` **8.14節と9章**: S5-15終了判断・解釈の訂正・移管。
  8.1〜8.13は履歴であり、強い断定は8.14の留保を付けて読む。
- 同報告書8.12.10〜8.12.12: 50 epoch学習推移と固定21動画の公式評価。
- 同報告書8.12.13〜8.12.21: 目視所見と追加CPU解析。8.13の追加実験案は未承認。
- `docs/stage5/s5-15/stage5_prediction_frame_visualization_implementation_handoff.md` 11節: フレーム画像可視化と
  初期の定性観察。相対輝度等の初期仮説には、その後の解析による留保がある。
- `docs/stage5/stage5_revision_management_record.md`: 全体の改修履歴・固定条件・意思決定。
- `docs/stage5/stage5_pointnext_s_training_evaluation_report.md`: 数値的根拠・過去の検証記録。
- `docs/stage5/FILES.md`、`docs/stage5/data_construct.md`: コード配置・データ構造・共有範囲。

管理記録等にS5-15終了の反映待ち箇所があれば、実施済みと推測せず、上記8.14の最新判断との
差を報告する。実機データは`/mnt/data/3d_projects/`以下であり、workspaceから利用可能とは限らない。

### 1.2 比較の基準条件

| 項目 | S5-15で固定した内容 |
| --- | --- |
| teacher | v7 `bboxrank_v7_cvat_authoritative_crop_quality_v1`、180動画 |
| split | 保存済みtrain162 / validation18リスト。seedによる再分割ではなくリスト固定 |
| 固定評価 | train sanity3動画＋validation全18動画 |
| モデル | 公式PointNeXt-Sを組み込んだStage5 wrapper、実験用GroupNorm8 groups |
| 初期重み | S3DIS部分転移後のGroupNorm初期checkpoint |
| 初期重みSHA-256 | `55ec6e6bcb39d58f398719b33826e80715a94bc6e7670d5b88623cd7c668438b` |
| 入力 | XYZとintensity/confidence。画像そのものや解剖学的な画像特徴を直接入力する構成ではない |
| label policy | bbox_noncontour_ignore |
| loss | CE、label smoothing0、固定class weight `[0.05963856, 1.94036150]` |
| 学習 | AdamW、lr1e-3、weight_decay1e-4、grad_clip_norm10、seed42 |
| window / batch | frame size16 / stride8 / tailあり、physical batch1 / accumulation8 |
| sampling / padding | overlap windowの実点を維持。random point removal・zero paddingなし |
| 評価 | augmentationなし、mean probability、既存2クラスargmax・同値background |

旧W-A、新規P3 R0、R0の50 epoch runは別runである。初期checkpointと学習済みcheckpointを
混同しない。上記は履歴baselineであり、本書によってproduction既定値を変更するものではない。

### 1.3 これまでに確認・修正した事項

- Datasetの点/GT対応、frame_order window分割、元点index、batch化を検証した。
- zero paddingがPointNeXtの出力・学習へ影響することを受け、paddingなし経路へ移行した。
- BatchNormのtrain/eval差を診断し、再較正だけでは改善が不十分だったため、実験用GroupNormを採用した。
- label policy・class weight比較を行い、上記のignore policyとW-A固定weightを維持した。
- overlap aggregationと座標・時間・露出の診断を実施。mean集約を維持し、原因は単一に確定していない。
- training-only回転augmentation、固定file list、epoch/worker間の角度共有、manifestと実効configの
  検査を実装した。評価時は無変換であることを検証した。

これらは確認した条件の範囲での検証履歴であり、入力経路の全ての不具合や全ての交絡を排除した
という意味ではない。同じ検証をS5-16で一律に繰り返すことは予定していない。

### 1.4 S5-15の実施結果

**P3: 新規R0/R1各5 epoch。** R0は無変換、R1は動画×epoch単位のランダムZ軸回転±15度。
validationでR1はFPRを12.53%→7.91%へ下げた一方、recallを45.08%→32.41%へ下げた。
pooled F1は6.77%→7.36%だったが、動画別F1改善/悪化は9/9、paired差分中央値は約+0.03pt。
微差とトレードオフが残る事前基準に従い、**R0維持、R1は比較履歴**とした。
これは回転augmentationの一般的な無効性や長期効果の否定ではない。

**長期テスト: R0を同じ転移初期重みから新規50 epoch。** 10 epoch追加pilotは省略した。
P3 checkpointからoptimizerを初期化して継続したものではない。

| validation pooled指標 | P3 R0 epoch5 | 長期best epoch6 | 長期last epoch50 |
| --- | ---: | ---: | ---: |
| recall | 45.08% | 38.95% | 6.97% |
| precision | 3.66% | 5.27% | 3.90% |
| F1 | 6.77% | 9.28% | 5.00% |
| IoU | 3.51% | 4.86% | 2.57% |
| TP0動画数 | 1/18 | 1/18 | 10/18 |

train sanity recallは長期best62.33%→last97.36%。validationの動画別F1中央値はbest7.75%→last0%。
val lossはepoch8付近の0.4619からepoch50の1.6901へ悪化した。
**現run/splitでは強い過学習・汎化不足と整合する結果**であり、同じ設定の100〜200 epoch延長と
production採用には進まないことを決定した。最良領域はこのrunで概ねepoch6〜10だが、他runへ一般化しない。

### 1.5 可視化・追加解析の到達点

フレーム画像への予測/GT重ね描画を追加し、固定21動画・834フレームについてbest/last計1,668 PNGを
生成した。全動画の中間H5は記録されたsource属性から解決され、合成テストは実機で30項目合格と
報告されている。可視化は既存予測を使うCPU処理で、再推論ではない。

ユーザーの目視では、足・頭蓋骨・腹部の輪郭、細長いアーティファクト、微細な散在ノイズ等にFPが
見られた。PLY上面視だけでは断面移動によって被覆率を誤認する例があり、フレーム画像と数値を併用した。
これは管理チャットが患者画像を直接検証したという意味ではなく、ユーザー所見と共有数値の整理である。

追加解析では以下を得た。

- 調べたデータではintensityと画像グレー値の強い一致を確認。輝度は入力特徴である。
- 高輝度点で位置と予測の関連が強い傾向は両checkpointに見られた。
- 「フレーム内相対輝度を追う」という初期仮説は一般化できない。絶対輝度側との強い関連もbestに
  限られ、lastでは傾向が変化した。
- 2領域が現れる動画群（8動画）は1領域群（10動画）よりbestで低recallだったが、同一動画内の
  領域数別比較には一貫した差がなく、領域数そのものを原因とは確定していない。
- 特定の輝度×位置binによる標準化ではGT上の群間平均確率差が32.5%縮小し、差が残った。
  追加軸の解析は未実行で、この割合は説明可能量の上限ではない。

患者フレーム画像、H5、PLY、実video ID対応表は共有しない。数値を共有する場合も匿名化する。
可視化文書の0始まりaliasと匿名化評価の1始まりaliasの対応は旧報告の照合記録を参照し、
別集計のaliasを同名だけで結合しない。

## 2. 発生している問題点

### 2.1 学習データへの適合とvalidation性能の乖離

学習を50 epochへ延ばすとtrain sanityのrecallは高まる一方、validationではTP0動画が増え、
recall・F1・IoUが低下する。現在の主問題は単なる学習不足ではなく、**学習データへの適合が
未見のvalidation動画での検出につながっていないこと**である。

過学習の強い示唆はあるが、「記憶という機序が確定」「標本規模や撮影条件差は無関係」とはしない。
同じrunの延長を止める判断と、どの介入が効くかという未確定の問いを分ける。

### 2.2 bestでも多いFPと、lastで増えるFN

bestのprecisionは5.27%で、valid GT上ではTP1点に対してFPが約18点存在する。
lastはFPRが下がってもrecallが6.97%へ低下しており、背景除去が改善しただけとは言えない。
ignore上のpositiveをFPとして数えず、valid GT上の指標と別に扱う。

FPは大腿骨と輝度・見かけの形状が似る構造にも見られるが、輝度、局所形状、画像位置、
点群密度、撮影条件等の寄与は分離できていない。Stage6で許容されるFPの量・形状の基準も
確定していないため、現状をproduction上許容可能とはしない。

### 2.3 失敗する動画群の性質と入力表現の十分性が未確定

2領域動画群の不利は、フレーム内の領域数だけでは説明できていない。動画レベルの条件差が
候補となるが、因果は未確定である。lastのゼロrecallが多い層では比較能力自体が低下している。

輝度・位置で調整しても差が残るという結果は、測定したbin/軸で揃わない差があるという意味に
限定する。32.5%/67.5%を確定した説明率や残差の下限として扱わず、共通binが存在することを
十分な標本数・代表性の保証にしない。

現特徴表現に大腿骨と類似構造を分ける情報が不足する可能性はあるが、まだ仮説である。
モデルはintensity/confidenceだけでなくXYZと局所幾何も使う。特徴削除ablationは寄与を見る手段で
あって、情報不足や課題の解決不能を単独で証明するものではない。

### 2.4 短期augmentation比較から長期効果は分からない

5 epochのR1はFP削減とrecall低下が併存し、採用基準を満たす一貫した改善は得られなかった。
一方、長期の汎化効果は試していない。短期にもaugmentation効果は現れ得るため、
「過学習前では原理的に効果が出ない」「P3比較が無意味だった」とはしない。

長期augmentation比較はS5-16で検討する候補の一つであり、R0/R1各25 epochを実施すると
既に決まったわけではない。入力表現・データ構成等との優先順位も未決定である。

### 2.5 評価と解釈の限界

- 単一seed、train sanity3動画、validation18動画。大量の点を独立した動画数の代用にしない。
- validationはlabel policy・weight・normalization・checkpoint等の方式選択に使用済みであり、
  独立testではない。使用済み180動画を事後に切り直すだけでは、過去の選択から独立した評価にはならない。
- GTクラス別の平均予測確率gapは閾値を使わない要約値だが、校正の影響を受ける。
  AUROC/AUCPR等の順位評価と同一視せず、gap低下だけでthreshold調整の可能性を否定しない。
- 目視分類に合わせた領域数閾値の修正等は探索的解析であり、独立に検証された分類基準ではない。
- 全21動画の可視化集計とvalidation18動画の公式集計、pooled値と動画等重み中央値を混同しない。

### 2.6 実装・記録上の逸脱と残務

P3ではlauncherが指定したSAVE_EVERY=1がbashで10へ上書きされ、中間epoch重みが未保存だった。
主比較epoch5は保存されており再学習は行わず、逸脱を記録した。env受け渡し修正・manifestと
実効config突合・長期runでの早期保存確認を追加した。意図値だけのmanifestを実効値の証拠にしない。

関連checkerの不具合修正や追加解析コード、文書の未コミット/未反映箇所は最新git statusと記録で
確認する必要がある。旧ログのファイル件数や「未実施」を現在状態とみなさない。
S5-15終了時の短い要約・管理記録への反映と、S5-16で何を最初に試すかの決定は別の作業として扱う。

この問題一覧を全項目再検証するチェックリストや追加学習の承認として扱わないこと。

## 3. S5-15の8.13にある提案への回答

回答日: 2026-09-20。回答元: Stage 5方針管理チャット。
S5-15報告8.14の終了判断・訂正を引き継ぐ。以下の「候補として保持」は実行承認を意味しない。

### 3.1 提案A: 現R0設定の100〜200 epoch延長は行わない

**回答: 採用。S5-15の延長は終了する。**

このrunでは序盤よりepoch50のvalidation指標が悪化し、train sanityとの乖離も大きい。
同じ設定で期間だけを増やす根拠は乏しいため、100〜200 epochへは進まない。
ただし「延長が全条件で必ず無駄」「別seedでも絶対に変わらない」と一般化しない。
将来、別の介入で改善根拠が得られた場合の長期確認まで禁止する判断ではない。
その場合はS5-16以降の別計画として扱う。

### 3.2 提案B: 現設定でのproduction採用は行わない

**回答: 採用。best/lastとも検証用の成果物として保持する。**

bestでもvalidation precision5.27%、recall38.95%、F1 9.28%で、定性評価にも広範なFPがある。
lastはFNがさらに増えている。現状の精度・汎化結果を総合し、production候補としては採用しない。
Stage6の具体的な許容基準が確定したわけではないため、「Stage6に絶対に使えない」という
一般論ではなく、現段階で採用を支持する根拠が不足しているという判断とする。
production既定値は変更せず、早期bestは診断・比較のbaselineとして参照できる状態を保つ。

### 3.3 提案C: R0/R1を各25 epochで比較し直す

**回答: 長期augmentation効果を調べる候補として保持。ただし最初の一手・実行量は未決定。**

5 epoch比較は短期性能の比較として成立しており、長期の汎化改善を評価していないという限界がある。
その意味で長期比較には検討価値がある。ただし「過学習前にはaugmentation効果が原理的に出ない」
という説明は採用せず、P3を無効化しない。FP/recallの同時低下も、正則化が効く前兆と確定しない。

実施案を比較する際には、以下を前提とする。

- R0には既存50 epoch runのepoch25重みと最初の25 epoch履歴を再利用できる可能性がある。
  コード・初期重み・split・optimizer/LR・環境・更新条件と保存物を照合し、総epoch数に依存する
  scheduler等がないことを含めて確認する。再利用可能なら、R0の25 epoch再学習を自動追加しない。
- R1を25 epoch行う場合は、同じ転移初期重みからの新規runを基本案とする。P3重みだけから
  optimizerを初期化して20 epoch追加する方法を、25 epochの連続学習と呼ばない。
- 観測はval lossの最小値・到達epoch・悪化傾向だけに限定しない。同epochのpooled/動画別
  F1・IoU、recall/FPR、TP0を含め、単なる学習遅延やpositive抑制と汎化改善を区別する。
- bestとepoch25を比較する場合、R0のbestも最初の25 epochの選択規則に合わせる。
  checkpoint・評価動画・追加学習量を事前固定し、結果を見て角度や期間を探索しない。
- 否定的な結果でも「今回の角度・期間・seedでは改善を確認できなかった」と結論する。
  回転augmentation全般が過学習を抑えられないと確定する実験ではない。

入力/特徴表現・データ構成の見直しと、期待する切り分け能力・費用を比較して優先順位を決める。
**各25 epochを2 run実行する案は現時点で承認しない。** 再利用監査や新規学習の具体的な手順も
次の計画で範囲を示す。長期runを増やすこと自体を目的にしない。

### 3.4 提案D: 独立したtest splitを確保する

**回答: 独立評価の必要性に同意。データの扱いは次の方式比較を始める前に明示する。**

現在のvalidation18動画は引き続き開発・選択用として扱い、最終の独立testとは呼ばない。
比較可能性を保つため、現splitを無断で切り直さない。
将来の最終評価には、方式選択へ使用していないデータを別途確保する案を優先して検討する。
同一患者・同一検査・近接動画や派生cropの重複がある場合は、動画名だけの分離では不十分なため、
利用可能な情報の範囲で独立性の単位を定義する。機微な識別情報は実機側に保持する。

使用済み180動画から事後にtestを取り直しても、過去の方式選択から独立した性能保証にはならない。
新しいsplitを作る場合は追加の内部検証として位置づけ、既存runと直接比較できなくなる点を記録する。
未使用データの取得が直ちにできなくても、現在のvalidationを開発用と明示した限定検証まで
一律に止める必要はない。ただし、その結果だけでproduction汎化を保証しない。
データ取得・分割の実作業は本回答では承認せず、利用可否と選択肢をS5-16計画で整理する。

### 3.5 論点: 現在の特徴量で課題が解けるのか

**回答: 重要な仮説として検討するが、現時点で情報不足とは断定しない。**

intensity/confidenceに加えてXYZ・近傍幾何が入力される。画像上の輝度や輪郭との関連だけから
「2特徴しかないので解けない」と結論しない。撮影条件・GT・幾何の品質、学習設定も影響し得る。

特徴削除ablationが問うのは、その学習条件での特徴の寄与・依存である。
情報が十分か、追加表現が必要かを単独で証明するものではない。入力channel変更時には転移できる
重みや初期化も変わり得るため、その影響を比較設計に含める必要がある。
逆に画像特徴や時間文脈の追加は、入力表現以外にモデル容量・初期化・計算量を変える可能性がある。
小さな介入から何を識別できるかを先に示し、広いfeature sweepや新モデル実装へ直行しない。
特徴削除・特徴追加のいずれも、本節では実装・学習を承認しない。

### 3.6 共通の留保への回答

単一seed・validation18動画・低い絶対精度という留保を維持する。追加seedを自動的な必須条件には
しないが、微差から優位性を断定せず「今回では判断できない」という結論を許容する。
学習曲線だけでなく実際の検出品質も比較し、平均値だけの改善・少数動画への集中を区別する。
次の実験では、目的・固定条件・採否基準・実施上限・停止条件を実行前に記述する。

### 3.7 未処理の実務事項への回答

未コミットという8.13.7の記述は当時の履歴である。現時点のgit statusとcommit履歴で確認し、
コード・テスト・文書の整理を新たな診断実行と分けて扱う。旧ログのファイル数をそのまま使わない。

標準化の追加軸とlast群別解析は未実施のまま引き継ぐ。実行すれば説明割合の上限が確定する、
あるいは過半が残る結果が保証される、とはしない。S5-15終了のためだけに実行する必要はない。
その結果が次の介入選択を変え得ると判断した場合に限り、S5-16の限定診断案として提示する。

**次の介入選択・実施手順は今後のセクションで決める。本節の回答は新たな実験の開始指示ではない。**

## 4. S5-16 方針策定: 3つの修正案

本節は、S5-16で検討した3つの修正案について、問題設定・期待効果・限界・実装コスト・
Stage 5のモデル差し替え可能性への影響を整理した最終方針である。

S5-16は**方針決定をもって終了**し、本節の内容そのものは実装・学習・GPU推論の開始承認ではない。
実作業は5節で定義するS5-17〜S5-20へ分離する。

### 4.1 修正案1: point-wise分類lossへのregion-level loss追加

#### 4.1.1 狙い

現行Stage 5は2クラスweighted CrossEntropyLossを使用しており、各点のclass predictionを教師labelと
比較している。修正案1では、現行CEを維持したままDice lossまたはsoft IoU/Jaccard系lossを追加し、
point-wise correctnessに加えて、window内のpredicted femur集合とGT femur集合の重なりを
region-level objectiveとして最適化する。

最初の限定比較では、loss zooを広く探索せず、原則として次の最小比較を想定する。

```text
baseline: CE
candidate: CE + lambda * Dice
```

IoU/Lovasz等は必要性が生じた場合の後続候補とし、S5-18開始前に固定する。

#### 4.1.2 期待できる効果

- S5-16 2.2節の**bestでも多いFP**に対し、predicted foreground集合を過剰に広げると
  Dice/IoU denominatorが悪化するため、CE単独とは異なる抑制圧を与えられる。
- FNもregion overlapを直接悪化させるため、positive suppressionだけで背景側へ寄せる解を
  抑える可能性がある。
- 強いclass imbalance下で、foreground領域全体としてのTP/FP/FNバランスを目的関数へ取り込める。
- 現在のmodel、Dataset、window、入力feature、normalization、aggregationを固定したまま比較できるため、
  **objective mismatchの寄与を比較的低コストで切り分けられる。**

#### 4.1.3 期待してはいけない効果

標準的なDice/IoU lossは、点のXYZ、局所曲率、surface distance、connectedness、解剖学的形状、
超音波textureを直接評価するlossではない。全点の予測確率とGT labelを同じ分子・分母で結合するため
point-wise CEよりglobalなregion overlapを見るが、同じTP/FP/FN数であれば、FPが頭蓋骨上にある場合と
腹部上にある場合を区別しない。

したがって、修正案1を

> 「3D立体構造やtextureをlossが直接理解する修正」

とは扱わない。S5-16 2.3節の入力表現不足やwindow context不足を直接解決する案ではない。

#### 4.1.4 GIoU / DIoU / CIoUの扱い

GIoU / DIoU / CIoUは主としてbounding-box geometryを最適化するlossであり、現行のpoint-wise
segmentationへそのまま第一候補として導入しない。これらは修正案3でBBox / oriented region等へ
出力taskを変更した場合、S5-20のtask-specific loss候補として再評価する。

#### 4.1.5 実装上の注意

現行学習はphysical batch size 1 + gradient accumulation 8である。Dice/IoUは非線形な比であり、
各microbatchのDiceを単純平均することと、8 windowを連結した集合に対してDiceを計算することは同値でない。
さらにoverlap windowでは同一pointが複数windowに現れる。S5-18では、**lossの計算単位を明示的に固定**し、
既存CEのpoint-weighted gradient accumulationと混同しない。

#### 4.1.6 Stage 5ラッパーへの影響

小さい。loss moduleはbackbone/model wrapperから独立させ、`BasePointSegmentor`等が返すlogitsに適用する。
PointNeXt-S以外のmodelへ差し替えても同じlossを使用できる構造を維持する。

---

### 4.2 修正案2: 同一動画のwindow間context伝播

#### 4.2.1 狙い

現行Stage 5は動画全体を`frame_order`基準のoverlap windowへ分割し、各windowを独立にPointNeXt-Sへ
入力する。最終的にoverlap点のpredictionをmean aggregationするが、これはpredictionの集約であり、
forward中に異なるwindow間でfeatureを共有する処理ではない。

修正案2では、各windowのfeatureを小さなEncoder / poolingで1個または少数のtokenへ集約し、
同一動画のtoken間でSelf-Attention等によるcontext伝播を行い、更新されたcontext tokenを各windowの
point解析へ戻す。

概念形は次のとおり。

```text
window 1 -> point backbone -> token 1 --+
window 2 -> point backbone -> token 2 --+--> inter-window context block
window 3 -> point backbone -> token 3 --+            |
...                                             context-aware tokens
                                                    |
                               each window feature <-+
                                                    |
                                             point/task head
```

#### 4.2.2 期待できる効果

- S5-16 2.2節の頭蓋骨、足、腹部輪郭、細長いartifact等の**structured FP**について、
  1 window内では大腿骨と似ていても、前後windowや動画全体の構造を参照することで区別できる可能性がある。
- あるwindowでは大腿骨がpartial / noisyでも、他windowに明瞭な大腿骨特徴が存在する場合にFNを
  減らせる可能性がある。
- S5-16 2.3節の「現在の入力情報が不足している」のか「情報はあるが16-frame windowでcontextが
  分断されている」のかを切り分ける診断価値が高い。
- 修正案3を採用した場合にも、context-aware featureをBBox / axis / endpoint / existence-region headへ
  入力できるため、**point-wise segmentation専用の投資にならない。**

#### 4.2.3 主なリスク

- model capacity増加により、現在のlocal shortcutがvideo-level shortcutへ置き換わるだけとなり、
  train/validation乖離が悪化する可能性がある。
- tokenが撮影条件、crop位置、pseudo-poseの癖、点密度等の動画固有情報を記憶する可能性がある。
- 1 token/windowは数万〜十数万pointを強く圧縮するbottleneckであり、単純average poolingでは
  femur signalが埋もれ、max poolingではartifactが支配する可能性がある。
- 現行のgradient accumulationはwindowを別々にforwardするため、8 window accumulationを
  inter-window contextとして利用することはできない。Dataset / sampler / forward単位の変更が必要になる。

#### 4.2.4 prototypeの基本方針

最初から大規模Transformerへ拡張しない。S5-19の初版は、原則として次のような小規模構成を候補とする。

- 1 token/window（必要性が示された場合のみmulti-tokenへ拡張）
- 小さなtoken dimension
- 1〜2層程度のcontext block
- window順序 / relative frame positionを識別できるposition情報
- attentionを採用する場合は可変window数maskを明示
- point全体へglobal attentionを掛けず、window token上だけでinteractionを行う

Self-Attentionは目的との整合性が高いが、S5-19開始前にMLP-mixing等との比較を必要最小限で決める。

#### 4.2.5 PointNeXt-S内部修正とラッパー互換性

Stage 5の重要な設計原則として、PointNeXt-Sを他のpoint modelへ差し替えられるwrapper構造を維持する。
したがって第一案は、context moduleをPointNeXt-S内部に直接埋め込まず、backbone外側の共通部品とする。

理想的には次のように分離する。

```text
PointNeXt adapter ---+
PTv3 adapter --------+--> common window representation --> context module --> task head
Other adapter -------+
```

ただし、適切なwindow featureを得るためにPointNeXt-S内部のencoder / decoder中間featureへアクセスする
必要がある場合は、**S5-19ではPointNeXt-S内部修正を妥協として許容する。**
その場合も変更はPointNeXt固有adapterへ隔離し、Stage 5共通interfaceをPointNeXt-S専用にしない。
将来別modelを試す際は、そのmodel向けadapterを追加できる設計を維持する。

---

### 4.3 修正案3: Stage 5出力taskをpoint-wise segmentation以外へ再設計

#### 4.3.1 狙い

本プロジェクトの最終目的は大腿骨segmentation maskの再現そのものではなく、最終的な**大腿骨全長（FL）測定**
である。したがって、Stage 5が全femur pointを精密に分類することを必須条件とせず、
FL推定に必要な存在領域・長軸・端点を安定して推定できれば十分である可能性を検討する。

候補となる中間表現は、現時点でBBoxに固定しない。

```text
- frame-wise coarse BBox
- oriented BBox / oriented region
- frame-wise existence regionを連結した3D-like tube / polyhedron
- center + axis + width
- endpoint / caliper heatmap
- axis + endpoints
- segmentation + geometryのmulti-task head
```

S5-17でtraining-freeに比較し、S5-20で本実装する表現を決める。

#### 4.3.2 期待できる効果

- S5-16 2.2節の大量point-wise FPについて、FL推定に無害なregion内部backgroundまで全てFPとして
  最適化対象にする必要がなくなる可能性がある。
- femur surfaceを完全に再現できなくても、存在領域・軸・端点が正しければStage 6 / FL測定へ
  有用な出力を渡せる可能性がある。
- fine segmentationに対して現在のXYZ/intensity/confidence/局所幾何が不足していても、
  coarse localization / axis estimationには十分である可能性を検証できる。
- output自由度とboundary適合要求を下げることで、training dataへの細粒度な適合を弱められる可能性がある。
- 修正案2と組み合わせると、動画全体contextを用いて各frame/windowのregion / axis / endpointを
  推定する自然な構成にできる。

#### 4.3.3 主なリスク

- point-wise FPに頑健になっても、頭蓋骨等を丸ごと大腿骨と誤認するobject-level errorはむしろ
  Stage 6へ強く伝播する可能性がある。
- 現在のpoint-level annotationという高情報量teacherをBBox等へ圧縮することで、教師情報を
  捨てすぎる可能性がある。
- axis-aligned BBoxは斜めの大腿骨でbackgroundを大きく含み、FL目的に最適な表現とは限らない。
- pseudo-3D上でframe regionを連結する場合、localization errorにpseudo-pose / inter-frame alignment errorが
  加わるため、「真の3D femur volume」ではなく**pseudo-3D上の存在領域**として扱う必要がある。
- Dataset target、model head、loss、metrics、evaluation、Stage 6 interfaceまで影響し、3案中で
  実装コスト・回帰リスクが最も大きい。

#### 4.3.4 lossとの関係

修正案3を採用した場合、修正案1のlossをそのまま延長するのではなく、出力taskに合わせて再設計する。
例:

```text
BBox / oriented region:
  objectness + center + size + angle + IoU/GIoU系loss

axis:
  center / line distance + orientation loss

endpoint:
  coordinate / heatmap loss + endpoint distance

multi-task:
  segmentation CE/Dice + geometry-specific loss
```

GIoU / DIoU / CIoU等は、この段階でbox geometryを出力する場合に初めて主要候補となる。

#### 4.3.5 ラッパー互換性

修正案3を「PointNeXt-SをBBox detectorへ直接改造する」方式では実装しない。
現行segmentation wrapperを保持し、新taskをparallel experimental interfaceとして追加する。

概念上は次を維持する。

```text
point backbone / adapter
      |
      +--> segmentation task head      (existing path)
      |
      +--> localization / geometry head (new experimental path)
```

または既存`BasePointSegmentor`を変更せず、新たに`BaseFemurLocalizer`相当を追加する。
これによりPointNeXt-S / PTv3 / 他modelについて、segmentation・localization双方を独立に比較できる余地を残す。

---

## 5. 3案の比較と優先順位

### 5.1 比較表

| 修正案 | 主に効果を期待する問題 | 期待度 | 実装コスト | 主なリスク | model差し替え性への影響 |
| --- | --- | --- | --- | --- | --- |
| **案1: CE + Dice/IoU系loss** | 2.2 大量FP/FN、2.1の一部 | **中** | **低** | context不足・入力情報不足には効かない。Dice計算単位の設計が必要 | **ほぼなし**。lossを共通moduleに置ける |
| **案2: window間context伝播** | 2.2 structured FP/FN、2.3 context不足、2.1の一部 | **中〜高** | **中〜高** | model容量増加による過学習、video-level shortcut、Dataset/forward変更 | **設計次第で維持可能**。PointNeXt内部修正が必要な場合もadapterへ隔離 |
| **案3: segmentation→localization/geometry task** | 2.1〜2.3、特に最終FLとのtask mismatch | **潜在的に高** | **非常に高** | task/教師/loss/metrics/Stage6契約の全面変更、object-level誤認 | **parallel task wrapperなら維持可能**。PointNeXt専用化は禁止 |

### 5.2 問題別の対応表

| S5-16の問題 | 案1 | 案2 | 案3 |
| --- | --- | --- | --- |
| 2.1 train適合 / validation汎化不足 | 低〜中 | 中 | 中〜高 |
| 2.2 bestの大量FP | 中〜高 | 中〜高 | 高い可能性 |
| 2.2 lastのFN増加 | 中 | 中〜高 | 中〜高 |
| 2.3 structuredな類似構造との混同 | 低 | 中〜高 | 中 |
| 2.3 入力表現の十分性の切り分け | 低 | 高い診断価値 | 高い診断価値 |
| 2.4 augmentation長期効果 | ほぼ独立 | ほぼ独立 | ほぼ独立 |
| 2.5 validationの独立性 | 解決しない | 解決しない | 解決しない |
| 2.6 実装・記録上の残務 | 解決しない | 変更規模中〜大 | 変更規模最大 |

### 5.3 優先順位を決める原則

1. Stage 5の**backbone差し替え可能なwrapper構造を維持する**。
2. 一度に変更する主要因は一つとし、S5-17〜S5-20を別stageとして管理する。
3. 低コスト・可逆な診断から開始し、model architecture / task definitionの大変更は根拠を得てから行う。
4. ただし案3を最後まで何も検証しないと、fine segmentationが不要だった場合に案1/2へ過剰投資するため、
   **案3のtraining-free feasibility診断だけは最初に行う。**
5. 案2は案3を採用した場合にも有用なglobal-context基盤となるため、S5-17の結果に関係なく実施する。

---

## 6. S5-17〜S5-20 実施順序の最終決定

### 6.1 S5-17: 修正案3のtraining-free feasibility診断

**目的:** point-wise segmentationを完全に再現しなくても、既存GT / 既存predictionからFLに必要な
存在領域・軸・端点情報を安定して得られるかを、**新規学習なし**で確認する。

主な考え方:

- 既存point-level GTをframe-wise BBox / oriented region / axis / endpoint等へ後処理変換する。
- 必要に応じてframe間で連結し、pseudo-3D existence region / tube / polyhedron様表現を構築する。
- 既存best predictionについても同じ変換を行い、低いpoint precisionでもgeometryが残るか確認する。
- BBoxに固定せず、FL目的に対してどの表現が最も情報を保持するか比較する。
- Stage 6相当のaxis fitting / endpoint estimation / length算出へ接続可能な指標を優先する。

**S5-17で決めるもの:** S5-20で本実装するgeometry target候補と、その妥当性。

**S5-17で行わないもの:** 新規model実装、学習、GPU推論、production変更。

---

### 6.2 S5-18: 修正案1の限定loss ablation

**目的:** 現行point-wise segmentationを維持したまま、weighted CEのobjective特性が
FP/FN問題へどの程度寄与しているかを切り分ける。

基本比較:

```text
A: current weighted CE
B: weighted CE + Dice  (lambda等は実行前に固定)
```

固定対象:

- teacher / split / initial checkpoint
- PointNeXt-S / GroupNorm設定
- features
- label policy
- window / batch / accumulation / padding-free条件
- optimizer / seed
- evaluation / mean aggregation

注意:

- loss候補を大量に探索しない。
- Dice/IoUの計算単位とoverlap重複の扱いを実装前に明記する。
- S5-18の結果は、S5-20でtaskを変更する場合には「region-level objectiveの効果」という知見として
  引き継ぎ、同じlossを機械的に再利用しない。

---

### 6.3 S5-19: 修正案2のwindow-context prototype

**目的:** 同一動画の異なるwindow間でfeature contextを伝播させ、16-frame局所処理で失われている
可能性のあるglobal / temporal / structural contextを利用できるようにする。

**S5-17の結果にかかわらずS5-19は実施する。スキップしない。**
理由は、window contextが現行segmentationだけでなく、S5-20でBBox / axis / endpoint / localization等へ
移行した場合にも有用な共通featureとして利用できるためである。

prototypeの原則:

- 最初は小規模構成とする。
- context blockは可能な限りbackbone非依存の共通moduleとする。
- PointNeXt-S内部featureへのアクセスが不可欠なら、PointNeXt内部修正を許容する。
- ただし修正をPointNeXt固有adapterへ隔離し、Stage 5全体をPointNeXt-S専用にしない。
- 将来PTv3等へ差し替える際に、別adapterを追加して同じcontext moduleへ接続できる設計を維持する。
- S5-19単独ではtask outputを変更せず、まずcontext mechanismの有効性・安定性・過学習リスクを評価する。

---

### 6.4 S5-20: 修正案3の本実装 + task-specific loss再設計

**目的:** S5-17で得た結果をもとに、FL測定へ最も適したStage 5出力taskを本実装する。
同時に、S5-18で得たloss設計の知見を、採用taskへ合わせて拡張・再設計する。

重要な方針:

- 現時点で「BBoxを本実装する」と固定しない。
- S5-17の結果に応じて、BBox / oriented region / existence tube / axis / endpoint / multi-task等から選ぶ。
- S5-19のwindow-context機構を、採用したgeometry headへ接続できる構造を優先する。
- point segmentationを完全に廃止するか、geometry headとmulti-taskで残すかもS5-17〜S5-19の結果で決める。
- lossはtaskに合わせて再設計する。boxならIoU/GIoU/center/size/orientation、axisなら方向・距離、
  endpointならheatmap/座標距離等を候補とする。
- 最終評価はtask内部のIoU等だけでなく、**Stage 6相当のendpoint / axis / FL error**を含める。
- 現行segmentation経路は比較baselineとして保持し、production既定値を自動で置き換えない。

---

## 7. S5-16終了判断

S5-16では、S5-15で確認された汎化不足、大量FP/FN、失敗動画群、入力表現の十分性未確定という問題に対し、
次の3方向を分離して検討した。

```text
案1: lossを変える
     -> 現point-wise taskのobjective mismatchを検証

案2: context / representationを変える
     -> window分割で失われた動画全体情報を補う

案3: task / outputそのものを変える
     -> point-wise segmentationが最終FL目的に対して過剰かを検証
```

最終的な実施順を次で確定する。

```text
S5-17  修正案3 training-free feasibility診断
   |
S5-18  修正案1 限定loss ablation
   |
S5-19  修正案2 window-context prototype
   |     ※ S5-17の結果にかかわらず実施
   |
S5-20  S5-17結果に基づく修正案3本実装
        + S5-18知見を踏まえたtask-specific loss再設計
        + 必要に応じてS5-19 contextを統合
```

この決定により、**S5-16の役割である「次の改善方針の策定」は完了**とする。
S5-17以降の具体的な実装、学習量、比較条件、採否基準、停止条件は、それぞれのstage開始時に
個別handoff / implementation policyとして定義し、S5-16の本書を実行承認として扱わない。


---

## 8. S5-16方針の精査結果（2026-09-21追記）

追記日: 2026-09-21。作成元: Stage 5方針精査チャット（S5-16の再精査とS5-17〜S5-20方針策定を担当）。
本章は4〜7章を書き換えず、精査で判明した過大評価・未定義事項・矛盾を追記する。
4〜7章を引用するときは本章の留保を併せて読むこと。

根拠の区分は次の3つで表記する。

- **［文書］** アップロード文書で確認できる事実
- **［外部］** Web上の一次資料・公式資料
- **［推論］** 精査チャットの解釈（未検証）

### 8.1 3案がS5-16 2章のどの問題を対象にしているか

#### 8.1.1 案1（CE+Dice）: 2.2への期待は過大の可能性

- ［文書］S5-13補足で確認済みの事実がある。W-Cは閾値0.5でF1/precisionが上がったが、validation AUPRC（0.0574→0.0445）とAUROC（0.8018→0.7588）はW-Aを下回った。同一FPRで比較すると、全水準でW-Aのrecallが上回った。つまり「FP/recallのトレードオフが動くこと」と「識別能力が上がること」は別物である。W-Aのvalidation AUPRCは約0.057（positive比率約1.05%）であり、主なボトルネックは順位付け能力にある。
- ［外部］Dice lossはクラス不均衡に頑健な一方で、校正が悪く過信的な予測になりやすい。Dice学習は分割性能でCE学習に勝る場合があるが、その代償として校正が悪化する（Mehrtash et al., IEEE TMI 2020、arXiv:1911.13273）。
- ［推論］現行推論はwindow間の確率平均と2クラスargmaxで判定する。Diceで確率分布の形が変わると、閾値0.5とmean集約の意味が変わる。したがって、閾値0.5での指標差を「objective mismatchの解消」とは読めない。閾値非依存の指標（AUPRC、同一FPRでのrecall）を主判定に含める必要がある。
- ［推論］W-Aは比32.5の強いclass weightであり、CE+Diceは不均衡への補正を二重にかけることになる。暗黙のweight変更と区別できる設計にする必要がある。
- ［推論］2.1（汎化不足）への効果を支持する根拠はない。5.2表の「低〜中」は下限側が妥当である。

#### 8.1.2 案2（window間context）: 前提となる規模が小さい

- ［推論・文書の数値から算出］window数は1動画あたり約4〜5である。根拠はv7の50 epoch runが90 update/epoch×accumulation 8で約720 window/epoch、162動画であること（v6では729 window/163動画）。動画のframe数は20〜128なので、token数は概ね2〜15になる。平均4〜5 tokenへのSelf-Attentionは、mean poolingと実質的な差が出にくい。mean-context、window拡大、global pooled featureとの比較が必要である。
- ［文書］2領域動画群の失敗は、時間区間が抜け落ちる形ではなかった。各window内での部分的な取りこぼしが、動画全体に一様にかかっていた（S5-15報告8.12.13(c)、8.12.17）。S5-14補足の同一動画内paired比較では、時間位置による系統的な低下も再現されなかった。
- ［推論］そのため、「明瞭なwindowから部分的なwindowへ情報を補う」ことによるFN低減（4.2.2）は、既存所見からは弱い支持しか得られない。「動画レベルの条件を共有する」方向の効果はあり得るが、これは4.2.3が挙げるvideo-level shortcutと表裏一体である。
- ［文書］座標事前分布の偏りは強く残っている（validation grid16で、hot bin FPR 24.7%対cold 0.48%）。XYZは既に動画全体で正規化されている。
- ［推論］context moduleが使える独立な学習標本は、動画数と同じ（train_coreでは144）に限られる。過学習リスクは4.2.3の記述より重く見積もるべきである。
- ［文書］支持材料もある。vote_count=2の点で、FPのwindow間disagreement率34.1%がTNの7.2%を上回った（S5-14 H4）。contextによってwindow間の整合が高まる余地がある。

#### 8.1.3 案3（task再設計）: 期待効果の第一項は既存label policyで大部分が吸収済み

- ［文書］`bbox_noncontour_ignore`では、BBox内かつcontour外がignore、no-BBox frameはbackgroundである。したがって現在FPとして数えている点は、定義上、注釈BBoxの外か、BBoxのないframe上の点である。
- ［推論］4.3.2の第一項「region内部のbackgroundまでFPとして最適化する必要がなくなる」は、既にかなり実現されている。残るFPの主体は、別構造上のobject-level誤認（足・頭蓋骨・腹部の輪郭）である。これは4.3.3が挙げる最大のリスクそのものである。
- ［推論］ignore上のpositive率40.08%（P3 R0、validation）は、定位（localization）の観点では正しい場所への予測とも読める。S5-17で定量化する。
- ［推論］S5-17はtraining-freeであり、判定できることは限られる。判定できるのは、(a) 表現がFLに必要な情報を保持しているか、(b) 現行予測から幾何を頑健に抽出できるか、の2点である。geometry headのほうが汎化しやすいかという学習可能性は判定できない。(b)が良好であれば、segmentationと後処理でS5-20の目的を満たせる可能性がある。
- ［推論］幾何表現にすると教師信号が大きく圧縮される。validationでGTを含むframeは287枚（1動画あたり約16枚）しかない。「出力自由度を下げれば過学習が弱まる」（4.3.2）とは限らない。

### 8.2 FL目的に関わる前提

- ［文書］管理記録1.1では、axis fitting・endpoint抽出・FL計測はStage 6の責務としている。
- ［外部］ISUOGの指針では、FLは骨化した骨幹の最長軸を計測し、キャリパーは遠位骨端を含めずに骨化骨幹の両端に置く（Salomon et al. 2019/2022、Khalil et al. 2024）。
- ［推論］臨床FLは1断面上の2D計測である。S5-17の候補表現には、3D-likeな表現と並べて「最適frameでの2D端点」も含める。
- ［推論］1 frameにGTが2領域写る場合（8動画）、両側の大腿骨である可能性があるが、未確認である。1 frameに1つのBBox・軸を仮定する表現はこの場合に破綻するため、instanceの選択規則か、複数instanceを扱う設計が必要になる。

### 8.3 validation 18動画の再利用

- ［文書・不整合］3.4節は「データの扱いは次の方式比較を始める前に明示する」としているが、4〜7章には定義がない。S5-18は方式比較そのものである。
- ［推論］S5-17〜S5-20の4段階がすべて同じ18動画で選択を重ねると、楽観バイアスが生じ、最終性能を保証できるデータがなくなる。本件の決定は9章を参照。

### 8.4 wrapper/backendの差し替え可能性

- ［文書］`BasePointSegmentor`の契約は、`[B,N,3]`/`[B,N,C]`を入力としてlogits `[B,N,num_classes]`を返すものである。`predict_h5()`はwindow単位のloopで動く。
- ［推論］案2でcontextを注入する位置は、次のどちらかになる。
  - encoderのbottleneckへ注入する: OpenPoints内部の修正が必要になる。
  - backboneがhead直前のpoint-wise feature `[B,N,D]`を返し、context側で結合する: 内部修正なしで、任意のbackboneに共通化できる。

  S5-19の第一案は後者とする（10.3節）。

### 8.5 順序・依存関係

- ［推論］S5-18の結果とS5-19のbase lossの関係が未定義だった。規則を事前に固定する（10.2節のgate）。
- ［推論］S5-19は「実施は無条件」とし、「S5-20への採用は条件付き」として分ける。
- ［推論］動画単位のsamplerに変えるだけで最適化条件が変わる。そのため「動画単位sampler・context無効」の対照armが必須である。
- ［推論］S5-20はtask・head・loss・contextを同時に変えるため、5.3節の原則2（一度に変える主要因は一つ）と衝突する。S5-20の内部を段階化する（10.4節）。
- ［推論］S5-17で共通の幾何・FL評価器（evaluator）を作り、S5-18/19のsegmentation出力もFL関連指標で評価する。
- ［推論］prior-only baseline（train平均の幾何）と、非学習の後処理baselineを必ず併置する。

### 8.6 文書間の矛盾・更新時差

| 箇所 | 内容 | 種別 |
| --- | --- | --- |
| 管理記録2章 原則6 | 「当面はCEを維持し、Dice/Focal/HNMを追加しない」。S5-18と直接衝突する。D-004とともに新decision IDで改訂が必要 | 矛盾 |
| 管理記録1.1 | 「Stage 5はaxis fitting・endpoint抽出を行わない」。S5-20でStage 5/6の責務境界を改訂する必要がある | 矛盾（予定） |
| 管理記録0・3・6章 | 最終更新は09-19で、「次はR0の新規50 epoch」のまま。50 epoch結果・S5-15終了・S5-16が未反映。decision recordはD-034まで。3章の表はv6/auto weight時代の値を「現在状態」として表示している | 更新時差 |
| 評価レポート | 冒頭注記は09-15、本文は9.9（P3）まで。50 epoch評価と8.12.13〜21の解析が未記載 | 更新時差 |
| 本書3.4 と 4〜7章 | 方式比較の前にデータの扱いを明示するという約束が未定義のまま（9章で解消） | 内部不整合 |
| 本書3.3・3.5 と 6章 | R0/R1各25 epoch比較、特徴削除ablation、標準化の追加軸の扱いが未記載 | 未処理（11章） |
| 本書5.3原則2 と 6.4 | 一要因の原則と、S5-20の複合変更が衝突（10.4で段階化して解消） | 内部不整合 |
| 本書の表題 | 「限定比較」とあるが、実際は方針策定のみで終了 | 軽微 |
| 可視化文書11節 と 公式評価 | 11.2(b)のbest precision 5.8%/recall 41.6%は21動画の合算で、validation公式値（5.27%/38.95%）とは別集計。aliasも0始まり/1始まりで異なる | 混同注意 |
| data_construct.md | prediction_framesの既定は上位10枚だが、S5-15の実行は`ALL_FRAMES=1`（全834 frame） | 既定値と実行値の差（矛盾ではない） |

## 9. 前提確認への回答とデータ分割の決定（2026-09-21）

### 9.1 ユーザー回答（前提）

| 項目 | 回答 | S5-17以降への影響 |
| --- | --- | --- |
| FL参照値 | 臨床計測値が動画ごとに存在する。学習・精度評価には、pseudo-3D上の教師contourから導く代理値が妥当。FL値そのものは学習に使わない | 臨床FLの用途は、代理FL定義の選定（train_coreのみ）、パイプライン全体の誤差下限の把握、最終判定に限定する |
| Stage 6 | 完全に未着手。案3がStage 6の内容を含む場合は、Stage 6の契約も見直してよい | S5-17の評価器をStage 6の原型とする。Stage 5/6の責務境界はS5-20a（10.4節）で改訂する |
| mmスケール | 画像frameの縦・横の実mmスケールを取得できる | mm単位で評価する。縦横のスケールは別々に扱う。pseudo-3DのZ方向のmm換算は未確立（［推論］）であり、frame内の2D計測を主候補とする根拠が強まった |
| 未使用動画 | 存在する。教師contourは未付与で、すぐには準備できないが、最終的に利用可能 | A案の最終外部評価に使う |
| 同一患者・同一検査の重複 | 180動画内には存在しない | 分割単位は動画でよい |
| GPU | NVIDIA GeForce RTX 5090（報告値32607。単位はMiB、約32GB VRAMと解釈） | S5-19で動画単位の同時forwardが可能かは実測で判断する |

### 9.2 決定: 追加testデータが完成するまではB′案、完成後はA案

**決定（ユーザー判断、2026-09-21）:** 未使用動画による追加testデータが完成するまでは、B′案（train内部からの封印test切り出し）を採用する。完成後はA案（方式選択に一切使っていない外部test）を最終評価に用いる。

**前回説明の訂正:** B案について「過去の選択から独立しない」と説明したのは強すぎた。過去の方式選択はvalidation成績に基づいており、train動画の成績は選択に使われていない（例外はtrain sanity 3動画）。train sanity以外のtrain動画を今後の学習から除外すれば、S5-17〜S5-20の選択に対してほぼ独立なtestになる（［推論］）。残る弱い依存は、固定class weightの算出と、teacher v7の品質修正に全動画が関わったことだけである。

#### B′案の仕様

| 項目 | 内容 |
| --- | --- |
| internal_test | train 162動画からtrain sanity 3動画を除いた159動画のうち、18動画を切り出す |
| train_core | 残る144動画（train sanity 3動画を含む）。S5-17以降の学習集合はすべてこれに統一する |
| validation | 既存の18動画。開発・選択用として明示して使う。validationを使った選択はすべてdecision ledgerへ記録する |
| 層化 | **split構築時に限り**、GTのmulti-region frameの有無（S5-15のchecker出力）と臨床FLの三分位を機械的に参照して層化する。モデル予測は使わない。seedを固定し、生成したfile listとhashを確定する |
| 封印 | 層化に必要なGT属性・臨床FLへのアクセスはsplit生成処理のみに限定する。**split確定後は、人間による閲覧・方式選択コード・評価コードからinternal_testのGT統計、metrics、臨床FLをS5-20cまで封印**し、通常の記録にはfile listとhashのみを残す |
| 漏洩防止 | 既存checkpoint（旧R0 50 epoch、W-A等）はinternal_testの動画で学習済みであるため、internal_testでの評価を禁止する。checkerで強制する。S5-17以降の学習用class weight等の統計量もinternal_testを含めず、train_coreのみから算出する |
| 比較可能性 | 旧runとの直接比較はできなくなる。旧R0 50 epoch runは参考値として扱う |
| A案への移行 | 未使用動画でpseudo-3Dを生成し、臨床FLを参照値とするend-to-end FL testを作る。教師contourが付与された後に、segmentationと幾何の評価を追加する。A案ができた時点でも、B′案のinternal_testの結果は保持する |

## 10. S5-17〜S5-20 implementation policy（2026-09-21）

本章は方針であり、実行承認ではない。表中の「提案値」は各stageの開始前に確定し、実施量とあわせて承認を得る。

### 10.0 横断事項

**固定条件（全stage共通）**

teacher v7、PointNeXt-S＋GroupNorm 8、S3DIS転移初期重み（SHA-256 `55ec6e6b…438b`）、特徴量intensity/confidence、`bbox_noncontour_ignore`、**train_core専用の固定class weight**、AdamW lr1e-3/weight_decay1e-4、grad_clip 10、seed42、window16/stride8/tail、paddingなし、評価はaugmentationなし・mean集約・2クラスargmax。学習集合はtrain_core（144動画）とする。

**class weightとinternal_test漏洩防止**

- 旧W-A `[0.05963856, 1.94036150]` は162動画時代の履歴値として保持し、S5-17以降の新規学習ではそのまま流用しない。
- Step 0で、**W-Aと同一の算出ロジックをtrain_core 144動画だけに適用してclass weightを1回だけ再計算**する。validation / internal_testは算出に使用しない。
- 算出値、対象file list hash、算出コードのcommit/hashを記録し、S5-18〜S5-20のsegmentation lossでは同じ値を固定して使う。stageごとに再計算しない。
- 旧W-Aとの差は履歴として報告するが、その差を見てweightを再調整しない。これによりinternal_testのlabel distributionが学習hyperparameterへ入る経路を閉じる。

**参照FL**

- 学習・開発評価の参照値は、教師contourから導く代理FL（定義はS5-17で固定する）。
- 臨床FLを参照してよいのは、S5-17（train_coreのみ）とS5-20c（internal_test、1回）だけ。validationの臨床FLは開発中に参照しない。

**mm換算**

- 換算経路は、`pixel_xy` → crop逆変換（`local_resize_scale`、`local_crop_left/top`）→ 元frame座標 → mm。
- 縦横のスケールは別々に扱う。スケールが動画内で一定か、frameごとに変わるかを確認する。
- 換算できない動画・frameは、除外理由を記録する。他の動画の値を流用しない。

**GPU量の単位**

旧R0の50 epoch run（4,500 update）を1 unitとする。train_coreでは約80 update/epochを見込む（推論値。実測で確定する）。

**統計**

- 単一seedを基本とする。微差の場合は「判定できない」と結論してよい。
- pooled値と、動画別paired差分の中央値・勝敗数を必ず併記する。

**privacy**

- 臨床FL、動画ごとのmm座標・端点・長さ、分割のfile list、患者画像は`DO_NOT_SHARE`とする。
- 共有してよいのは匿名化した集計値のみ。新しい出力にも既存の匿名化規約とprivacy self-checkを適用する。

**Step 0（S5-17の前、CPUのみ）**

1. B′案の分割を実施する。層化変数へのアクセスはsplit生成処理だけに限定し、file listとhashを確定した時点でinternal_testを封印する。
2. W-Aと同一ロジックで、**train_core 144動画のみ**からS5-17以降専用class weightを再計算する。値、対象file list hash、算出コードのcommit/hashを記録して固定する。
3. 管理記録へ次のdecision recordを追記する。
   - D-035: S5-15の終了
   - D-036: S5-16の方針
   - D-037: B′→A案のデータ方針
   - S5-18の開始前に: 原則6とD-004の改訂
   - S5-20aの開始前に: Stage 5/6責務境界の改訂
4. 評価レポートへ、50 epoch runの評価結果とS5-15の終了を反映する。
5. 3章冒頭の現在状態表を更新する。
6. 未コミットの差分を確認し、コードと文書のみをコミットする。

### 10.1 S5-17: 幾何表現のtraining-free feasibility診断と、幾何・FL評価器

| 項目 | 方針 |
| --- | --- |
| 目的・仮説 | H17-1: GTから導くどの幾何表現・代理FL定義が、臨床FLと最もよく一致するか（train_coreのみ）。H17-2: 現行best（epoch6）の予測に非学習の後処理を加えると、GT由来の幾何にどこまで近づけるか（validation）。H17-3: FPの空間分布（GTの近傍か、ignore領域か、別構造か）。H17-4: prior-only baselineの到達度。H17-5: 複数領域frameの扱い |
| 変えない固定条件 | 学習・GPU・再推論を行わない。入力は既存の予測npz（主はbest、副としてlast）、teacher v7 H5、中間H5のみ。production設定は変更しない |
| 比較する表現 | (a) frameごとの軸平行BBox、(b) frameごとのoriented box（PCA軸・長さ・幅）、(c) frameごとの端点（主軸方向の頑健な分位点）、(d) 最良frameの選択と端点（臨床手技に相当）、(e) pseudo-3D軸（Z方向のmm換算が確立していない場合は、単位制約付きの参考値）。heatmapは(c)と同じ情報量のため(c)で代表する。multi-taskの成立性はH17-2で間接的に評価する |
| 入出力契約 | 新パッケージ`stage5/geometry/`（numpyのみ、torch非依存）。`FrameGeometry`はframe_order、存在有無、複数instance（中心・軸単位ベクトル・長さ・幅・端点・点数）を持つ。`VideoFLEstimate`は代理FL、使用frame、信頼度を持つ。`PixelToMM`は換算と逆変換を行う。入力はGT labelでも、prob/pred npzでもよい |
| 実装対象 | 幾何抽出、mm換算、代理FL定義（複数案）、後処理（なし／最大連結成分／frame間持続性／確率上位）、FP距離解析、prior-only baseline（train_coreの平均幾何）、CLIとbash、合成テスト |
| wrapper/backboneへの影響 | なし。本評価器をStage 6の原型とし、S5-18〜S5-20の共通指標にする |
| 必要な新規module/interface | `stage5/geometry/{frame_geometry,fl_estimate,pixel_to_mm,postprocess}.py`、`evaluate_stage5_geometry.py/.sh`、`checks/dummy/check_dummy_geometry_*` |
| comparison baseline | GT由来の幾何（上限参照）、prior-only |
| metrics | 臨床FLとの一致: bias、MAE、Bland–Altman一致限界、動画別の誤差。予測由来の幾何: frameの存在判定precision/recall、中心・端点・長さの誤差（mm）、角度誤差、動画単位の代理FL誤差。FP: GT BBoxからの距離分布、別構造上の割合。頑健性: 点の間引きやFPノイズを加えたときの誤差変化 |
| acceptance/rejection | 実行前に、臨床上の許容誤差T_FL（mm）をユーザーが設定する。(1) 代理FL定義は、train_coreで臨床FLとの誤差が最小のものを事前規則で1つ選んで固定する。(2) 予測由来の代理FL誤差がT_FL以内の動画がvalidationの過半なら、「segmentationと後処理」をS5-20の有力候補にする。(3) prior-onlyとの差が小さい表現は候補から外す。(4) FPの主体が離れた別構造である場合、S5-20ではobjectnessとcontextの設計を必須とする |
| CPU/GPU実施量 | CPUのみ。本実行1回。不具合修正後の再実行は1回まで |
| fail-fast | mm換算の往復誤差が許容値を超える、点数の不一致、`pixel_xy`の範囲外、`frame_order`が画像数を超える、のいずれかで停止する。臨床FLが欠損している動画は補完せず除外として記録する。internal_testへのアクセスを検出したら停止する |
| privacy | 臨床FL、mm座標、動画別結果は`DO_NOT_SHARE`。集計値のみ匿名化して共有する |
| 次stageへのdecision gate | 評価器が合成テストに合格すること（S5-18/19の副次指標として使うため）。代理FL定義を固定すること。S5-20の候補表現を最大2〜3に絞ること。複数領域のinstance選択規則を決めること |

### 10.2 S5-18: 限定loss ablation（CE 対 CE+Dice）

| 項目 | 方針 |
| --- | --- |
| 目的・仮説 | H18: CE+Diceは、動作点を移動させるだけでなく、閾値に依存しない識別能力（validation AUPRC、同一FPRでのrecall）を改善する |
| 変えない固定条件 | 10.0の全固定条件。変更するのはlossのみ。両armともtrain_coreで新規学習する |
| Dice仕様（GPU実行前に固定） | positiveクラス確率に対するsoft Diceをwindow（microbatch）単位で計算し、valid点のみを使いignoreを除外する。overlap重複点はwindowごとにそのまま扱う（CEと同じ露出）。**GT positive=0のwindowの扱いはGPU学習前のCPU synthetic testで決定する。** 候補は (D0) Dice項を除外してCEのみ、(D1) smoothing付きsoft Diceをそのまま定義しnegative-only windowのFPにも勾配を与える、の2方式とする。synthetic testでは有限値、all-zero/all-one予測、FP増加時のloss/gradient方向、GT-positive windowでの挙動を確認し、validation成績を見ずに1方式だけを固定する。εとλも開始前に固定し、GPU上で探索しない。全体lossはpoint-weighted CE＋λ×Diceとする。class weightは10.0で定めたtrain_core専用固定値を使い、CEとDiceによる不均衡補正の重複は限界として記録する |
| 学習長・checkpoint | 20 epoch、毎epoch保存（提案値）。**主比較は固定epoch {5,10,20} とする。** bestは補助比較に限定し、両armとも`validation AUPRC最大（同値なら早いepoch）`という同一規則だけで選ぶ。IoU/F1等の別指標で選んだbestを後付けで比較しない |
| 入出力契約 | `CompositeLoss`（`stage5/training/`）: `(logits, labels, valid_mask) → loss`。各項の値をhistoryに記録する |
| wrapper/backboneへの影響 | なし。backbone非依存 |
| comparison baseline | A arm: weighted CE（train_coreで新規学習）。旧R0 50 epochは参考値 |
| metrics | 閾値0.5でのpooled・動画別F1/IoU/recall/FPR、TP0。AUPRC、AUROC、FPR 1/5/10%でのrecall（既存の閾値非依存checkerを再利用）。ECE。window単位と集約後の差。ignore上のpositive率。S5-17評価器による幾何指標。train sanity |
| acceptance/rejection | Diceを採用する条件: validation AUPRCが改善し、FPR 3水準のうち2つ以上でrecallが改善し、動画別AUPRC差の中央値が正で過半の動画で改善し、TP0が増えず、幾何指標が悪化しないこと。閾値0.5の指標だけが改善した場合は「動作点の移動」とみなして不採用。判定できない場合はCEを維持する |
| CPU/GPU実施量 | 2 arm×20 epochで約0.7 unit。GPU preflightは1回 |
| fail-fast | NaN/Inf。config parity（lossの差以外は0件）。manifestと実効configの突合。epoch3以降にvalidation TP0が18/18になったら停止・報告 |
| privacy | 既存の評価pipelineに従う |
| 次stageへのdecision gate | 開始前に原則6を改訂しておく。S5-19のbase lossは、Diceが採用されればCE+Dice、それ以外はCEとする（この規則は今固定する） |

### 10.3 S5-19: window-context prototype

| 項目 | 方針 |
| --- | --- |
| 目的・仮説 | H19-1: 同一動画のwindow間でcontextを共有すると、context無しの対照に比べて識別能力が上がり、structured FPが減る。H19-2: 過学習の開始（train–valの乖離）が早まらない |
| 変えない固定条件 | 10.0の固定条件と、S5-18 gateで決まったbase loss。task出力はsegmentationのまま |
| 構成（backbone非依存を優先） | (1) `PointFeatureBackbone`: `forward_features(points, features) → [B,N,D]`。(2) `PointNeXtFeatureAdapter`: `BaseSeg`のhead直前のdecoder出力を返す。OpenPoints内部は変更しない。(3) `WindowTokenizer`: maskつきのmean/max poolingとlinearでtoken次元64（提案値）にし、frame位置のencodingを加える。(4) `WindowContextModule`: `none`／`mean`／`attn`（1層、4 head、key padding mask）。(5) `ContextFusion`: FiLMでzero初期化し、学習開始時は恒等写像になるようにする。(6) 既存のseg headはそのまま使う。既存segmentorは「backbone＋head」の合成として再構成し、`BasePointSegmentor`は外向きの窓口として残す |
| 学習単位 | 1 stepで1動画の全windowを処理し、約2動画（約9 window）ごとに1 update。point-weightedの正規化は維持する。**第一選択はfull-gradient方式**で、同一動画の全windowからcontext生成・fusion・seg lossまでの計算graphを保持する。GPU preflightでは代表的な小/中央値/最大window数動画でpeak VRAMを実測する。full-gradientが安全に収まらない場合は、まずactivation checkpoint/recompute等の**勾配を保存するmemory-saving方式**を検討する。tokenを`no_grad`でキャッシュするstop-gradient 2-passは数学的に別モデル条件であり、自動fallbackにしない。必要になった場合は別条件として明示し、実行前に承認を得る |
| 比較arm | C0: 動画単位sampler、context無し（samplerの影響を分離する対照）。C1: mean-context。C2: attn-context |
| 入出力契約 | `predict_video()`は動画の全windowを処理する。context=`none`では現行`predict_h5()`とのparityを確認する。**point index・window構成・final labelは完全一致を要求し、floating logits/probabilityは0 toleranceを要求せず、S5-19 handoffで固定した`atol/rtol`による`assert_close`を使う。** aggregate後の指標も一致を確認する |
| wrapper/backboneへの影響 | PointNeXt固有の処理はadapterに隔離する。PTv3等を使う場合は、adapterを追加すれば同じcontext moduleに接続できる。S3DISの転移111 keyが不変であることをtestで固定する |
| 必要な新規module/interface | `stage5/models/{backbone_interface,pointnext_feature_adapter,window_context,context_fusion}.py`、動画単位sampler、`predict_video()`、各種合成テスト |
| comparison baseline | C0。S5-18の採用armとの差も「samplerの影響」として記録する |
| metrics | S5-18の全指標に加え、overlap点でのwindow間disagreement、S5-17評価器によるstructured FPの割合、2領域動画群とそれ以外の群の比較、train–valの乖離曲線 |
| acceptance/rejection | C1またはC2がC0に対し、AUPRCとFPR水準別recallで改善し、動画別で過半が改善し、TP0が増えず、過学習の開始が早まらない場合に採用する。C1とC2の差が小さい場合は単純なC1を採用する |
| CPU/GPU実施量 | 3 arm×20 epochで約1.1〜1.5 unit（2-pass方式の場合は増える）。メモリ実測のpreflightを1回 |
| fail-fast | `none`でpoint index / window構成 / final labelが一致しない、または固定`atol/rtol`を超えてlogit/probability parityが崩れる。padding tokenに対する出力の不変性（合成テスト）。動画ごとのwindow数と構造的な再計算値の不一致。NaN。full-gradientがVRAM上限を超えた場合はGPU学習へ進まず、memory-saving方式の再設計と承認へ戻る |
| privacy | 既存に従う |
| 次stageへのdecision gate | S5-19の実施は無条件だが、S5-20への採用は条件付き。採否基準を満たさない場合、S5-20はcontextなしで進め、interfaceだけは維持する |

### 10.4 S5-20: 出力taskの本実装とtask-specific lossの再設計

S5-17の結果が出る前に、出力表現・loss・segmentationの存廃は固定しない。一要因の原則を守るため、次の3段階に分ける。

- **20a（CPU・文書）:**
  - Stage 5/6責務境界の改訂（decision recordを追加）。
  - S5-17で残った候補について、GTからtargetを生成し、合成テストを行う。
  - `BaseFemurLocalizer`を追加する。入力は`PointFeatureBackbone`のfeatureで、S5-19で採用されればcontextも使う。
  - **geometry座標系契約を共通interfaceとして固定する。** `GeometryMetadata`は少なくとも`frame_order`、point→frame対応、`pixel_xy`、元frameの`image_width/height`、crop/resize逆変換情報、利用可能ならx/y別mm scaleを持つ。PointNeXt固有adapterへ画像座標変換を埋め込まない。
  - 2D geometry headのcanonical出力は、原則として**元frame座標に対応する正規化2D座標 `[0,1]^2`** とし、pixel/mmへの変換は共通geometry/loss/evaluator側で行う。候補表現がlocal-crop座標を必要とする場合も`coord_space`を明示し、暗黙変換を禁止する。pseudo-3D XYZ表現は別`coord_space`として扱い、Z方向mm scaleが確立するまで2D mm値と混在させない。
  - head候補: frame単位のpoolingから得たtokenで、存在有無とframe内の2D幾何を出す方式。または、点ごとに端点へのoffsetを出すvoting方式（［推論・既知文献の考え方、今回は文献検索していない］）。
  - loss候補: 存在はBCE。端点は座標L1またはheatmapで、端点の並べ替えに不変とする。軸は向きの正負を区別しないよう2倍角の(cos, sin)で表す。長さはmmでのL1。axis-aligned boxならIoU/GIoU系を候補とする。**oriented boxを採用する場合はaxis-aligned GIoUをそのまま流用せず、rotated/polygon IoUまたはorientationを含む幾何lossを別途固定する。**
  - 複数領域は、S5-17で決めた規則に従う（instance選択または集合matching）。
- **20b（GPU）:**
  - S5-17で残った候補（最大3 arm）を、baseline「segmentation＋S5-17の最良後処理」と比較する。
  - multi-taskの場合、seg lossはS5-18の結論を引き継ぐ。
  - backboneとcontextはS5-19の採用構成で固定し、変えるのはhead/taskだけにする。
  - 上位2候補のみ、2 seedで確認する。
- **20c:**
  - 封印したinternal_testで1回だけ評価する（代理FLと臨床FL）。判定後に構成を選び直さない。
  - production設定は自動では置き換えない。
  - A案の外部testが完成したら、そちらで最終確認する。

| 項目 | 方針 |
| --- | --- |
| metrics | frameの存在判定precision/recall、端点・角度・長さの誤差（mm）、動画単位の代理FL誤差（MAE、bias、一致限界）、T_FL超過率、object-level誤検出率。multi-taskの場合はseg指標も |
| acceptance/rejection | 新taskを採用する条件: validationで、baselineより動画単位FL誤差のMAEと失敗率が小さく、動画別で過半が改善し、prior-onlyを明確に上回り、internal_testでも同じ方向の結果が出ること |
| CPU/GPU実施量 | 20bは最大5 run相当（各20 epoch）、約2 unit以内。超える場合は別途承認 |
| fail-fast | target生成でGT→幾何→再幾何化が一致しない、`coord_space`が未宣言または混在する、normalized2D→pixel→normalized2D / pixel→mmのround-tripが規定誤差を超える、mm換算の不一致、端点の並べ替え不変性が崩れる、NaN、のいずれかで停止 |
| privacy | 端点座標、動画別FL、臨床FLは`DO_NOT_SHARE` |
| decision gate | production判断とStage 6実装の着手は、20cの結果（A案完成後はA案の結果）を受けて別途決定する |

## 11. 未確定事項と次のアクション（2026-09-21）

1. **T_FL（臨床上の許容誤差、mm）**: S5-17の実行前にユーザーが設定する。
2. **提案値の確定**: 20 epoch、Diceのε/λ、token次元64、update単位、S5-19 parityの`atol/rtol`など。各stageの開始前に確定する。Diceのnegative-only window仕様はS5-18 GPU実行前のsynthetic CPU testで固定する。
3. **旧候補の扱い**: 3.3（R0/R1各25 epoch比較）、3.5（特徴削除ablation）、標準化の追加軸は、S5-17〜S5-20の範囲外としてdeferredとする。再開する場合は別計画とし、train_coreを前提に設計し直す。
4. **mmスケールの粒度**: 動画内で一定か、frameごとに変わるかを、実機のメタ情報で確認する（Step 0またはS5-17の冒頭）。
5. **A案の準備**: 未使用動画のpseudo-3D生成と臨床FLの紐付けは、S5-17〜S5-20と並行して進めてよい。ただし、方式選択には一切使わない。
6. **次の作業**: Step 0（10.0節）→ S5-17のimplementation handoffを作成する。いずれも承認を得てから着手する。
## 12. v3レビュー反映事項（2026-09-22）

本章は、v2に対する方針管理チャットの再レビューで指摘された事項を、9〜10章へ反映した記録である。
本章自体は追加実験の承認ではない。

### 12.1 internal_testとclass weight

- B′の層化にはmulti-region情報と臨床FLを使うため、「一切見ない」という表現を修正した。
- split生成処理だけが層化変数へアクセスし、file list/hash確定後にinternal_testをS5-20cまで封印する。
- 旧W-A class weightはtrain162由来でinternal_test候補を含むため、S5-17以降の新規学習では流用しない。
- W-Aと同一算出ロジックをtrain_core 144動画だけに適用し、新しい固定class weightをStep 0で1回だけ算出する。

### 12.2 S5-18 Dice

- GT positive=0のwindowでDiceを無効にする仕様を既定値から外した。
- negative-only windowを除外する方式と、smoothing付きsoft DiceでFPへ勾配を与える方式をsynthetic CPU testで確認し、
  validation結果を見る前に1方式を固定する。
- 主比較は固定epochで行い、補助的なbestはvalidation AUPRC最大（同値なら早いepoch）だけで選ぶ。

### 12.3 S5-19 memory / parity

- full-gradientを第一選択とし、最大window数を含む動画でpeak VRAMを実測する。
- memory不足時はgradient-preservingなcheckpoint/recomputeを先に検討する。
- stop-gradient token cache / 2-passは同値な実装fallbackではなく別モデル条件とし、別途承認を必要とする。
- `context=none` parityはindex・labelの完全一致とfloating値の固定toleranceを分離する。

### 12.4 S5-20 geometry coordinate contract

- backbone固有コードとgeometry座標変換を分離するため、`GeometryMetadata`と`coord_space`を共通契約へ追加した。
- 2D headのcanonical出力は原則として元frameに対応するnormalized 2D `[0,1]^2` とし、
  pixel/mm変換は共通geometry/loss/evaluator側で行う。
- pseudo-3D座標と2D mmを暗黙に混在させない。
- oriented boxを採用する場合、axis-aligned GIoUを機械的に流用しない。

### 12.5 v3の位置づけ

v3でもS5-16は方針策定完了のままであり、Step 0、S5-17、S5-18、S5-19、S5-20の実行承認ではない。
次の実作業は、v3を正本候補として確認した後、Step 0およびS5-17の個別handoffを作成して開始条件を確定する。
