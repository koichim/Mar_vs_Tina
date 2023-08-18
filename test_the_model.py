
#
#test
#
from PIL import Image
import os,glob
import numpy as np
import tensorflow as tf

categories = ["Mar","Tina"]

def gather_test_data(input_shape, test_dir, max_count=100):
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
        print(f"gather_test_data(): {len(files)} {cat} were found")
    
    learn_size = (input_shape[1], input_shape[2])
    rgb = "RGB"
    if input_shape[3] == 1:
        rgb = "L"
    
    print(f"gather_test_data(): the model: learn_size={learn_size}, rgb={rgb}")
    
    i=0
    for cat, fname in allfiles:
        img = Image.open(fname)
        img = img.convert(rgb)
        img = img.resize(learn_size)
        img_data = np.array(img) / 255.0  # 画像データを0から1の範囲にスケーリング
        # mean = 0.
        # std = 1.
        # if (center):
        #     mean = np.mean(img_data, axis=(0, 1))
        # if (normalization):
        #     std = np.std(img_data, axis=(0, 1))
        # img_data = (img_data - mean) / std
        #    data = np.asarray(img)
        X.append(img_data)
        Y.append(cat)
        i=i+1
        if max_count-1 < i:
            break
        # print(f"gather_test_data(): {i} images loaded")
    
    print(f"gather_test_data(): {len(Y)} images loaded")
    return X, Y


def test_the_model(model, test_dir):
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
        print(f"test_the_model(): {len(files)} {cat} were found")

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

# https://www.tensorflow.org/lite/performance/post_training_float16_quant?hl=ja
# A helper function to evaluate the TF Lite model using "test" dataset.
def test_the_tf_model(result_path, interpreter, test_dir):
    from keras import metrics
    
    input_details = interpreter.get_input_details()[0]
    input_index = input_details["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    test_images, test_labels = gather_test_data(input_details["shape"], test_dir)
    
    # Run predictions on every image in the "test" dataset.  
    prediction_array = []
    prediction_digits = []
    i=0
    for test_image in test_images:
        
        # interpreter = tf.lite.Interpreter(model_path=str(result_path))
        # interpreter.allocate_tensors()

        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_index, test_image)
        
        
        # Run inference.
        interpreter.invoke()
        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.tensor(output_index)
        import copy
        float_result_array = copy.copy(output()[0])
        if input_details['dtype'] == np.uint8:
            float_result_array = float_result_array/255.0 + 1.0e-9
        prediction_array.append(float_result_array)
        digit = np.argmax(float_result_array)
        prediction_digits.append(digit)
        print(f"test_the_tf_model(): predicted {i+1}/{len(test_images)} - {float_result_array} {digit} vs {test_labels[i]} ")
        i+=1
    
    # Compare prediction results with ground truth labels to calculate accuracy.
    accurate_count = 0
    y_true = []
    for index in range(len(prediction_digits)):
        if prediction_digits[index] == test_labels[index]:
            accurate_count += 1
        a_y_true = [0,0]
        a_y_true[test_labels[index]] = 1
        y_true.append(a_y_true)
    #loss_count += metrics.categorical_crossentropy(y_true, prediction_array[index])
    
    accuracy = accurate_count * 1.0 / len(prediction_digits)
    #loss = loss_count / len(prediction_digits)
    loss_array = metrics.categorical_crossentropy(y_true, prediction_array).numpy()
    loss = np.mean(loss_array)
    print('loss=', loss)
    print('accuracy=', accuracy)
    return loss, accuracy