# yoloX関連の進捗状況：自分向けメモ
## RGB用のモデルでThermal画像
案外出来たっぽい。
IOUを計算してみた。
グラフも出してみよう。
画像も何枚かサンプルとして出す。

## 2022 05 08 進捗
### result2hist.py
* iouのヒストグラム
* アノテーションと検出結果の比較画像を生成
    * 任意の画像IDを指定
* flir_dataset_val_thermal_yolox_result.csvとflir_anotation_val_data.csvを使用