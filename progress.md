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
COCODatasetのコンストラクタ変更

保存と再構築
conda では出来ませんでした。conda pipでそれぞれ頑張ってください。


python tools/train.py -f exps/myExps/yolox_x.py -d 2 -b 10 --fp16 -o -c yolox/yolox_x.pth

gpgpu2では　-b 5　で動いた　上限はわからん
python tools/train.py -f exps/myExps/yolox_x.py -d 2 -b 10 --fp16 -o -c yolox/yolox_x.pth

gpgpu8ではnvidia-smiが動かなかった。pytorchのバージョンも違うっポイ。gpgpu間でpytrchのバージョンが違う時の運用方法を知る必要がある。
直し方わからんので要相談