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

## 2022 05 17 進捗
https://zenn.dev/opamp/articles/d3878b189ea256 が学習の方法について結構詳しめに書いてある。
データセット（データローダー）は出来ているので気合を入れてやるだけ。

## 2022 05 21
gpgpuにyoloxの環境がなかったので作り始めた。

## 2022 05 22
gpgpuに環境が出来た。データセットのパスが変。Cocodatasetの形式だが、階層形式が違う。
アノテーションjsonの位置を変えて、yolox_base.pyのCOCODatasetのコンストラクタ引数を変更したよ。