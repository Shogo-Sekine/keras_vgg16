# coding:utf-8
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import sys

if len(sys.argv) != 2:
    print('以下のように入力してください')
    print('python simple_vgg16_usage.py [image file path]')
    sys.exit(1)

file_name = sys.argv[1]

# 学習済みのVGG16(学習済みの重みも含める)をロード
model = VGG16(weights='imagenet')
model.summary()

# 画像ファイルの読み込み(サイズを224 * 224にリサイズ)
img = image.load_img(file_name, target_size=(224, 224))

# 読み込んだPIL形式の画像をarrayに変換
x = image.img_to_array(img)

# 3次元テンソル（rows, cols, channels) を
# 4次元テンソル (samples, rows, cols, channels) に変換
# 入力画像は1枚なのでsamples=1でよい
x = np.expand_dims(x, axis=0)

preds = model.predict(preprocess_input(x))
results = decode_predictions(preds, top=5)[0]
for result in results:
    print(result)