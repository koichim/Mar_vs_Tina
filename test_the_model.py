
#
#test
#
from PIL import Image
import os,glob
import numpy as np
import tensorflow as tf

def test_the_model(model, test_dir):
    categories = ["Mar","Tina"]

    X = [] # 画像データ
    Y = [] # ラベルデータ
    
    # フォルダごとに分けられたファイルを収集
    #（categoriesのidxと、画像のファイルパスが紐づいたリストを生成）
    allfiles = []
    for idx, cat in enumerate(categories):
        image_dir = os.path.join(test_dir,cat)
        files = glob.glob(os.path.join(image_dir,"*.jpg"))
        for f in files:
            allfiles.append((idx, f))
        print(f"test_the_module(): {len(files)} {cat} were found")

    learn_size = (model.input_shape[1],model.input_shape[2])
    rgb = "RGB"
    if model.input_shape[3] == 1:
        rgb = "L"
    
    # オプションの初期値をFalseに設定
    center = False
    normalization = False
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            config = layer.get_config()
            if 'center' in config and config['center']:
                center = True
            if 'scale' in config and config['scale']:
                normalization = True
    print(f"test_the_model(): the model: learn_size={learn_size}, rgb={rgb}, center={center}, normalization={normalization}")
    print (f"read and convert {len(allfiles)} jpg files")    
    for cat, fname in allfiles:
        img = Image.open(fname)
        img = img.convert(rgb)
        img = img.resize(learn_size)
        img_data = np.array(img) / 255.0  # 画像データを0から1の範囲にスケーリング
        mean = 0.
        std = 1.
        if (center):
            mean = np.mean(img_data, axis=(0, 1))
        if (normalization):
            std = np.std(img_data, axis=(0, 1))
        img_data = (img_data - mean) / std
        #    data = np.asarray(img)
        X.append(img_data)
        Y.append(cat)


    from keras.utils import np_utils
    test_X = np.array(X)
    test_Y = np.array(Y)
    #Yのデータをone-hotに変換
    if (model.layers[-1].units == 1):
        test_Y = np.asarray(test_Y).astype('float32').reshape((-1,1))
    else:
        test_Y = np_utils.to_categorical(test_Y, len(categories))
    print(f"do model.evaluate")
    score = model.evaluate(x=test_X,y=test_Y)

    print('loss=', score[0])
    print('accuracy=', score[1])
    return score[0], score[1]