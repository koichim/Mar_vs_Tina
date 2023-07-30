# ref https://qiita.com/kotai2003/items/497795548cc4e3a78d91

# log. flush every print
import os, time, functools, re
print = functools.partial(print, flush=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # reduce tf log (https://github.com/tensorflow/tensorflow/issues/59779)
start_time = time.time()
# Base path
data_dir = os.path.abspath("data")
result_parent_dir = os.path.abspath("tmp")
refresh_data = False

categories = ["Mar","Tina"]

# if re-train from existing model, put the saved_model dir here,
load_model_name = ""

# get trained model application from tfhub.dev. if load_model_name, this is ignored.
model_name = "nasnet_mobile" # @param ['efficientnetv2-s', 'efficientnetv2-m', 'efficientnetv2-l', 'efficientnetv2-s-21k', 'efficientnetv2-m-21k', 'efficientnetv2-l-21k', 'efficientnetv2-xl-21k', 'efficientnetv2-b0-21k', 'efficientnetv2-b1-21k', 'efficientnetv2-b2-21k', 'efficientnetv2-b3-21k', 'efficientnetv2-s-21k-ft1k', 'efficientnetv2-m-21k-ft1k', 'efficientnetv2-l-21k-ft1k', 'efficientnetv2-xl-21k-ft1k', 'efficientnetv2-b0-21k-ft1k', 'efficientnetv2-b1-21k-ft1k', 'efficientnetv2-b2-21k-ft1k', 'efficientnetv2-b3-21k-ft1k', 'efficientnetv2-b0', 'efficientnetv2-b1', 'efficientnetv2-b2', 'efficientnetv2-b3', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'bit_s-r50x1', 'inception_v3', 'inception_resnet_v2', 'resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'nasnet_large', 'nasnet_mobile', 'pnasnet_large', 'mobilenet_v2_100_224', 'mobilenet_v2_130_224', 'mobilenet_v2_140_224', 'mobilenet_v3_small_100_224', 'mobilenet_v3_small_075_224', 'mobilenet_v3_large_100_224', 'mobilenet_v3_large_075_224']

trainable = True # do fine tuning or not
if model_name=="efficientnetv2-l-21k" and trainable:
    batch_size = 1
elif model_name=="efficientnetv2-m-21k" and trainable:
    batch_size = 3
elif model_name=="efficientnetv2-s-21k" and trainable:
    batch_size = 7
elif model_name=="mobilenet_v2_140_224" and trainable:
    batch_size = 32 # may not be max
elif model_name=="pnasnet_large" and trainable:
    batch_size = 4 # may not be max
else:
    batch_size = 64 #anything


epochs = 100

optimizer_name = "Lion"
initial_learning_rate = learning_rate = 1e-4 # used 1e-4, using ReduceLROnPlateau, this is initial value.

def lr_override(current_epoch, current_lr, initial_lr):
    if 100 < current_epoch and initial_lr/10. < current_lr:
        override_lr = initial_lr/10.
    elif 200 < current_epoch and initial_lr/100. < current_lr:
        override_lr = initial_lr/100.
    elif 300 < current_epoch and initial_lr/1000. < current_lr:
        override_lr = initial_lr/1000.
    elif 400 < current_epoch and initial_lr/10000. < current_lr:
        override_lr = initial_lr/10000.
    else:
        override_lr = current_lr
    if override_lr != current_lr:
        print(f"lr_override(): {current_lr} -> {override_lr} because epoch#{current_epoch}")
    return override_lr

color_mode="rgb" #"grayscale"か"rbg"の一方．(only when create model)
rotation_range=45
shear_range=0.0
class_mode="categorical"# "categorical"か"binary"か"sparse"か"input"か"None""categorical"は2次元のone-hotにエンコード化されたラベル，"binary"は1次元の2値ラベル，"sparse"は1次元の整数ラベル，"input"は入力画像と同じ画像になります（主にオートエンコーダで用いられます）．Noneであれば，ラベルを返しません（ジェネレーターは画像のバッチのみ生成するため，model.predict_generator()やmodel.evaluate_generator()などを使う際に有用）．class_modeがNoneの場合，正常に動作させるためにはdirectoryのサブディレクトリにデータが存在する必要があることに注意してください．

#opt_suffix = f"_lr{learning_rate}fine"
#opt_suffix = f"_resucelr_transfer"
opt_suffix = f"_resucelr_fine"

center=False
normalization=False


# memory management
# https://qiita.com/studio_haneya/items/4dfaf2fb2ac44818e7e0
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

# ref: https://qiita.com/tomo_20180402/items/e8c55bdca648f4877188
#
#モデルの構築
#
from keras import layers, models
#from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow import keras
#from keras.applications.efficientnet import EfficientNetB0
import tensorflow_hub as hub


#おまじない
keras.backend.clear_session()

if load_model_name:
    load_model_dir = os.path.join(result_parent_dir, load_model_name)
    if os.path.isdir(os.path.join(load_model_dir, "saved_model")):
        load_model_dir = os.path.join(load_model_dir, "saved_model")
    print("load existing model from "+load_model_name)
    if re.search(r"Lion", load_model_name):
        import optimizer_lion
        model = models.load_model(load_model_dir, custom_objects={'Lion': optimizer_lion.Lion})
    else:
        model = models.load_model(load_model_dir)
    if re.search(r"transfer", load_model_name) and trainable:
        model.trainable = trainable
    learn_size = model.input_shape[1:3]
    color_depth = model.input_shape[3]
    if color_depth==3:
        color_mode = "rgb"
    else:
        color_mode = "grayscale"
else: # create model
    print(f"constrcting the model for: {model_name}")
    color_depth = 3
    if color_mode == "grayscale":
        color_depth = 1

    # applicationから全結合層を除く
    #conv_layers = keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=learn_size+(color_depth,), classes=len(categories))
    #conv_layers = keras.applications.EfficientNetV2M(include_top=False, weights='imagenet', input_shape=learn_size+(color_depth,), classes=len(categories))
    #conv_layers.trainable = True  # False=学習させない (学習済みの重みを使う)
    import tf_hub
    model_handle = tf_hub.model_handle_map.get(model_name)
    resize = tf_hub.model_image_size_map.get(model_name, 224)

    print(f"Selected model: {model_name} : {model_handle}")
    learn_size = (resize,resize)
    print(f"Input size {learn_size}")

    # applicationに全結合層を追加
    # ref: https://www.tensorflow.org/hub/tutorials/tf2_image_retraining?hl=ja
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=learn_size + (color_depth,)))
    model.add(hub.KerasLayer(model_handle, trainable=trainable))
    model.add(layers.Dropout(rate=0.2))
    if (class_mode == "binary"):
        model.add(layers.Dense(1,activation="sigmoid", #分類先の種類分設定 binary-category-"sigmoid", multi-categories-"softmax"
                kernel_regularizer=tf.keras.regularizers.l2(0.0001))) 
    else:
        model.add(layers.Dense(len(categories),activation="softmax", #分類先の種類分設定 binary-category-"sigmoid", multi-categories-"softmax"
                kernel_regularizer=tf.keras.regularizers.l2(0.0001))) 
    model.build((None,)+learn_size+(3,))

#モデル構成の確認
model.summary()

# build result dir name
#result_parent_dir = os.path.abspath("")
if load_model_name:
    result_dir_prefix = load_model_name.split(os.path.pathsep)[0]+opt_suffix
else:
    result_dir_prefix = model_name+"_result_"+"b"+str(batch_size)+"e"+str(epochs)+"_"+str(learn_size[0])+"x"+str(learn_size[1])+"r"+str(rotation_range)
    if (shear_range != 0.0):
        result_dir_prefix +="s"+str(int(shear_range))
        
    if (color_mode == "grayscale"):
        result_dir_prefix += "_bw"
    center_normalization_char=""
    if (center):
        center_normalization_char+="c"
    if (normalization):
        center_normalization_char+="n"
    if not (center_normalization_char == ""):
        result_dir_prefix += "_"+center_normalization_char
    if (class_mode == "binary"):
        result_dir_prefix += "_bin"
    
    result_dir_prefix += "_"+optimizer_name+"_lr"+str(learning_rate)+opt_suffix

#add count suffix if the result dir already exists.
cnt = 0
while os.path.exists(os.path.join(result_parent_dir,result_dir_prefix)):
    cnt += 1
    result_dir_prefix += "_"+str(cnt)

result_dir = os.path.join(result_parent_dir,result_dir_prefix)
checkpoint_dir = os.path.join(result_parent_dir,"tmp_"+result_dir_prefix+"_tmp")
bestpoint_dir = os.path.join(result_parent_dir,"tmp_"+result_dir_prefix+"_best")
tensorboard_dir = os.path.join(result_parent_dir,"tmp_"+result_dir_prefix+"_logs")

# 02.2nd source
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

if refresh_data:
    import data_preparation2
    data_preparation2.randomize_data(data_dir)

# Generator
# ref https://www.codexa.net/data_augmentation_python_keras/
datagen = ImageDataGenerator(
    featurewise_center=False,  # データセット全体で，入力の平均を0にします
    samplewise_center=center,  # 各サンプルの平均を0にします
    featurewise_std_normalization=False,  # 入力をデータセットの標準偏差で正規化します
    samplewise_std_normalization=normalization,  # 各入力をその標準偏差で正規化します
    zca_whitening=False,  # ZCA白色化を適用します
    zca_epsilon=1e-06,  # ZCA白色化のイプシロン．デフォルトは1e-6
    rotation_range=rotation_range,  # 画像をランダムに回転する回転範囲
    width_shift_range=10.0,  # 浮動小数点数（横幅に対する割合）．ランダムに水平シフトする範囲
    height_shift_range=10.0,  # 浮動小数点数（縦幅に対する割合）．ランダムに垂直シフトする範囲．
    brightness_range=(0.7,1.3),  #
    shear_range=shear_range,  # 浮動小数点数．シアー強度（反時計回りのシアー角度）
    zoom_range=0.3,
    # 浮動小数点数または[lower，upper]．ランダムにズームする範囲．浮動小数点数が与えられた場合，[lower, upper] = [1-zoom_range, 1+zoom_range]です．
    channel_shift_range=50.0,  # 浮動小数点数．ランダムにチャンネルをシフトする範囲．
    fill_mode="constant",  #: {"constant", "nearest", "reflect", "wrap"}
    cval=0.0,  # 浮動小数点数または整数．fill_mode = "constant"のときに境界周辺で利用される値．
    horizontal_flip=True,  # 水平方向に入力をランダムに反転します．
    vertical_flip=False,  # 垂直方向に入力をランダムに反転します．
    # EfficientNetV2 rescale in itself. https://teratail.com/questions/kmde0te15kkey9 but no good...
    rescale=1. / 255,  # 画素値のリスケーリング係数．デフォルトはNone．Noneか0ならば，適用しない．それ以外であれば，(他の変換を行う前に) 与えられた値をデータに積算する．
    preprocessing_function=None,
    # 各入力に適用される関数です．この関数は他の変更が行われる前に実行されます．この関数は3次元のNumpyテンソルを引数にとり，同じshapeのテンソルを出力するように定義する必要があります．
    data_format=None,  # {"channels_first", "channels_last"}のどちらか
    validation_split=0.2  # 浮動小数点数．検証のために予約しておく画像の割合
)

# Directoryか画像を読み込む

train_generator = datagen.flow_from_directory(
    directory=data_dir,#ディレクトリへのパス．クラスごとに1つのサブディレクトリを含み，サブディレクトリはPNGかJPGかBMPかPPMかTIF形式の画像を含まなければいけません．
    target_size=learn_size,#整数のタプル(height, width)．
    color_mode=color_mode,#"grayscale"か"rbg"の一方．
    classes=categories,#クラスサブディレクトリのリスト．（例えば，['dogs', 'cats']）
    class_mode=class_mode,# "categorical"か"binary"か"sparse"か"input"か"None""categorical"は2次元のone-hotにエンコード化されたラベル，"binary"は1次元の2値ラベル，"sparse"は1次元の整数ラベル，"input"は入力画像と同じ画像になります（主にオートエンコーダで用いられます）．Noneであれば，ラベルを返しません（ジェネレーターは画像のバッチのみ生成するため，model.predict_generator()やmodel.evaluate_generator()などを使う際に有用）．class_modeがNoneの場合，正常に動作させるためにはdirectoryのサブディレクトリにデータが存在する必要があることに注意してください．
    batch_size=batch_size,#データのバッチのサイズ
    shuffle=True,#データをシャッフルするかどうか
    seed=None,#シャッフルや変換のためのオプションの乱数シード
    save_to_dir=None,# Noneまたは文字列（デフォルト: None）．生成された拡張画像を保存するディレクトリを指定できます
    save_prefix='',#文字列．画像を保存する際にファイル名に付けるプリフィックス
    save_format='png',#"png"または"jpeg"
    follow_links=False,#
    subset='training',#
    interpolation='nearest'#
)
val_generator = datagen.flow_from_directory(
    directory=data_dir,#ディレクトリへのパス．クラスごとに1つのサブディレクトリを含み，サブディレクトリはPNGかJPGかBMPかPPMかTIF形式の画像を含まなければいけません．
    target_size=learn_size,#整数のタプル(height, width)．
    color_mode=color_mode,#"grayscale"か"rbg"の一方．
    classes=categories,#クラスサブディレクトリのリスト．（例えば，['dogs', 'cats']）
    class_mode=class_mode,# "categorical"か"binary"か"sparse"か"input"か"None""categorical"は2次元のone-hotにエンコード化されたラベル，"binary"は1次元の2値ラベル，"sparse"は1次元の整数ラベル，"input"は入力画像と同じ画像になります（主にオートエンコーダで用いられます）．Noneであれば，ラベルを返しません（ジェネレーターは画像のバッチのみ生成するため，model.predict_generator()やmodel.evaluate_generator()などを使う際に有用）．class_modeがNoneの場合，正常に動作させるためにはdirectoryのサブディレクトリにデータが存在する必要があることに注意してください．
    batch_size=batch_size,#データのバッチのサイズ
    shuffle=True,#データをシャッフルするかどうか
    seed=None,#シャッフルや変換のためのオプションの乱数シード
    save_to_dir=None,# Noneまたは文字列（デフォルト: None）．生成された拡張画像を保存するディレクトリを指定できます
    save_prefix='',#文字列．画像を保存する際にファイル名に付けるプリフィックス
    save_format='png',#"png"または"jpeg"
    follow_links=False,#
    subset='validation',#
    interpolation='nearest'#
)

#
# look for temporary weight if exists (load it after compile)
#
current_epoch = 0
latest = tf.train.latest_checkpoint(checkpoint_dir)
best_latest = tf.train.latest_checkpoint(bestpoint_dir)
initial_value_threshold = None
if (latest):
    current_epoch = int(re.search(r"epoch(\d+)",os.path.basename(latest)).group(1))
    learning_rate = float(re.search(r"lr(\d\.\d+e-\d+)",os.path.basename(latest)).group(1))
    if os.path.isdir(bestpoint_dir):
        initial_value_threshold = np.inf
        for a_best_saved_model_dir in os.listdir(bestpoint_dir):
            valloss_match = re.search(r"valloss(\d+\.\d+)",a_best_saved_model_dir)
            if valloss_match and valloss_match.group(1):
                if float(valloss_match.group(1)) < initial_value_threshold:
                    initial_value_threshold = float(valloss_match.group(1))
    print(f"found tmp for resume, {latest} at epoch#{current_epoch} lr={learning_rate} to resume. The best val_loss was {initial_value_threshold}")
    learning_rate = lr_override(current_epoch, learning_rate, initial_learning_rate)

#
#モデルのコンパイル
#
from keras import optimizers

print(f"compile the model with {optimizer_name}")
loss = 'categorical_crossentropy'
if (class_mode == "binary"):
    loss="binary_crossentropy"

optimizer = optimizers.Adam(learning_rate=learning_rate) #default
if optimizer_name == "RMSSprop":
    optimizer=optimizers.RMSprop(learning_rate=learning_rate)
if optimizer_name == "Adamax":
    optimizer=optimizers.Adamax(learning_rate=learning_rate)
if optimizer_name == "Lion":
    import optimizer_lion
    optimizer=optimizer_lion.Lion(learning_rate=learning_rate)

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=["acc"])

#
# load temporary weight if exists
#
if (latest):
    print(f"load {latest} to resume.")
    model.load_weights(latest)

#
#モデルの学習
# https://www.tsl.co.jp/ai-seminar-contents-04/
print("Do fit for: "+result_dir_prefix)

# val_lossの改善が2エポック見られなかったら、学習率をfactor倍する。
# https://analytics-note.xyz/machine-learning/reduce-lr-on-plateau/
# since starting with learning_rate=1e-4 (0.0001), min_xxx should be more near 0.
reduce_lr = keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=10,
                        min_delta=0, #default 0.0001
                        min_lr=0, #1e-7, #0.0001
                        verbose=1
                        )

# save temporary fitted waights to be able to resume
# https://stackoverflow.com/questions/68656060/keyerror-failed-to-format-this-callback-filepath-reason-lr
# custom call back could not populate logs for the model in caller. 
# instead, 
import keras.backend as K
class ModelCheckPointWLR(tf.keras.callbacks.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(K.eval(self.model.optimizer.lr))
        logs.update({'lr': lr})
        super().on_epoch_end(epoch, logs)
#checkpoint_filepath = 'path_to/temp_checkpoints/model/epoch-{epoch}_loss-{lr:.2e}_loss-{val_loss:.3e}'

checkpoint_path = os.path.join(checkpoint_dir,"cp-epoch{epoch:04d}-lr{lr:e}.ckpt")
#cp_callback = tf.keras.callbacks.ModelCheckpoint(
cp_callback = ModelCheckPointWLR(
                        filepath=checkpoint_path, 
                        save_weights_only=True,
                        verbose=1)

bestpoint_path = os.path.join(bestpoint_dir,"best-epoch{epoch:04d}-lr{lr:e}-valloss{val_loss}.ckpt")
#best_callback = tf.keras.callbacks.ModelCheckpoint(
best_loss_callback = ModelCheckPointWLR(
                        filepath=bestpoint_path, 
                        monitor= "val_loss",
                        save_best_only=True,
                        initial_value_threshold=initial_value_threshold,
                        verbose=1)
best_acc_callback = ModelCheckPointWLR(
                        filepath=bestpoint_path, 
                        monitor= "val_acc",
                        save_best_only=True,
                        initial_value_threshold=initial_value_threshold,
                        verbose=1)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1, write_images=True)

steps_per_epoch = int(len(train_generator.classes) / batch_size)
#steps_per_epoch = 2 # for test
validation_steps = int(len(val_generator.classes) / batch_size) 
history = model.fit(
    x=train_generator,  # 学習データ
    steps_per_epoch=steps_per_epoch,  # ステップ数
    epochs=epochs,  # エポック数
    validation_data=val_generator,  # 検証データ
    validation_steps = validation_steps,
    initial_epoch=current_epoch,
    callbacks=[reduce_lr, cp_callback, best_loss_callback, best_acc_callback, tensorboard_callback],
    verbose=1
)

#
#学習結果を表示
#
print(f"prepare results for {result_dir_prefix}")
os.mkdir(result_dir)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(os.path.join(result_dir,"accuracy.png"))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(os.path.join(result_dir,"loss.png"))

#
#モデルの保存
#
print(f"save the resulting model for {result_dir_prefix}")
json_string = model.to_json()
open(os.path.join(result_dir,"MarTina_predict.json"), 'w').write(json_string)
#重みの保存
hdf5_file = os.path.join(result_dir,"MarTina_predict.h5")
model.save_weights(hdf5_file)
#saved_modelの保存
saved_model_dir = os.path.join(result_dir,"saved_model")
model.save(saved_model_dir)

#
# tmp dirを削除
#
import shutil
print(f"remove tmp files of {checkpoint_dir}")
shutil.rmtree(checkpoint_dir)

#
#test
#
#import test_the_model
#print("Do the test")
#test_the_model.test_the_model(model, data_dir)

# print("test_the_model(): check the model")
# learn_size = (model.input_shape[1],model.input_shape[2])
# rgb = "RGB"
# if model.input_shape[3] == 1:
#     rgb = "L"
    
# # オプションの初期値をFalseに設定
# center = False
# normalization = False

# # モデルのレイヤーをループして、各レイヤーのオプションを確認する
# for layer in model.layers:
#     if isinstance(layer, tf.keras.layers.BatchNormalization):
#         config = layer.get_config()
#         if 'center' in config and config['center']:
#             center = True
#         if 'scale' in config and config['scale']:
#             normalization = True
# print(f"test_the_model(): the model: learn_size={learn_size}, rgb={rgb}, center={center}, normalization={normalization}")

# #
# #test
# #
# from PIL import Image
# import glob
# import numpy as np

# X = [] # 画像データ
# Y = [] # ラベルデータ

# # フォルダごとに分けられたファイルを収集
# #（categoriesのidxと、画像のファイルパスが紐づいたリストを生成）
# allfiles = []
# for idx, cat in enumerate(categories):
#     image_dir = os.path.join(data_dir,cat)
#     files = glob.glob(os.path.join(image_dir,"*.jpg"))
#     for f in files:
#         allfiles.append((idx, f))
#     print(f"test_the_module(): {len(files)} {cat} were found")

# print (f"read and convert {len(allfiles)} jpg files")    
# for cat, fname in allfiles:
#     img = Image.open(fname)
#     img = img.convert(rgb)
#     img = img.resize(learn_size)
#     img_data = np.array(img) / 255.0  # 画像データを0から1の範囲にスケーリング
#     mean = 0.
#     std = 1.
#     if (center):
#         mean = np.mean(img_data, axis=(0, 1))
#     if (normalization):
#         std = np.std(img_data, axis=(0, 1))
#     img_data = (img_data - mean) / std
#     #    data = np.asarray(img)
#     X.append(img_data)
#     Y.append(cat)


# from keras.utils import np_utils
# test_X = np.array(X)
# test_Y = np.array(Y)
# #Yのデータをone-hotに変換
# if (model.layers[-1].units == 1):
#     test_Y = np.asarray(test_Y).astype('float32').reshape((-1,1))
# else:
#     test_Y = np_utils.to_categorical(test_Y, len(categories))
# print(f"do model.evaluate")
# score = model.evaluate(x=test_X,y=test_Y)

# print('loss=', score[0])
# print('accuracy=', score[1])
    
elapsed_time = time.time() - start_time

import datetime
print(result_dir_prefix+" was done in "+str(datetime.timedelta(seconds=elapsed_time)))

