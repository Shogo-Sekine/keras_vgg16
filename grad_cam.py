# coding:utf-8
import pandas as pd
import numpy as np
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

K.set_learning_phase(1) #set learning phase

def Grad_Cam(model, x, layer_name):
    '''
    Args:
       input_model: モデルオブジェクト
       x: 画像(array)
       layer_name: 畳み込み層の名前

    Returns:
       jetcam: 影響の大きい箇所を色付けした画像(array)

    '''
    # 前処理
    X = np.expand_dims(x, axis=0)

    X = X.astype('float32')
    preprocessed_input = X / 255.0

    # 予測クラスの算出
    predictions = model.predict(preprocessed_input)
    results = decode_predictions(predictions, top=5)[0]
    for result in results:
        print(result)
    class_idx = np.argmax(predictions[0])
    class_output = model.output[:, class_idx]

    #  勾配を取得
    conv_output = model.get_layer(layer_name).output   # layer_nameのレイヤーのアウトプット
    grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
    gradient_function = K.function([model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]

    # 重みを平均化して、レイヤーのアウトプットに乗じる
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # 画像化してヒートマップにして合成
    cam = cv2.resize(cam, (224, 224), cv2.INTER_LINEAR) # 画像サイズは224で処理したので
    cam = np.maximum(cam, 0) 
    cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
    jetcam = (np.float32(jetcam) + x / 2)   # もとの画像に合成

    return jetcam

def main():
    if len(sys.argv) != 2:
        print('以下のように入力してください')
        print('python simple_vgg16_usage.py [image file path]')
        sys.exit(1)

    file_name = sys.argv[1]

    model = VGG16(weights='imagenet')
    model.summary()

    x = img_to_array(load_img(file_name, target_size=(224,224)))
    array_to_img(x)

    # imagenetの最後の畳み込み層を指定
    image = Grad_Cam(model, x, 'block5_pool')
    image = array_to_img(image)
    image.save('grad_cam.png')

if __name__ == '__main__':
    main()