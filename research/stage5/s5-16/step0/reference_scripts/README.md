# FL値の由来確認用スクリプト

- [mk_legcase_json.py](mk_legcase_json.py)：ユーザーから提供された原本。2026-09-29に一時領域から移動し、内容を保持。
- [Step 0報告書15章](../../../../../docs/stage5/s5-16/stage5_step0_report_to_policy_chat.md)：前提訂正と算出内容の確認記録。
- [S5-16資料一覧](../../README.md)

XMLのstart/end・leg BBoxと画像から重心軌跡を作り、正規化座標の軌跡長`femur_traj_len`を出力します。臨床FLや現在の幾何評価器ではありません。末尾に元環境の絶対パスとトップレベルの実行・JSON書き出しがあり、importでも処理が走ります。今回の整理では実行・改修せず、由来確認の証拠として保存しています。
