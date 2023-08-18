# 01.Buildin
import os #, time, math, random, pickle
import tensorflow as tf

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # reduce tf log (https://github.com/tensorflow/tensorflow/issues/59779)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # reduce tf log (https://github.com/tensorflow/tensorflow/issues/59779)

result_name = "+++tmp_efficientnetv2-m-21k_result_b64e100_480x480r45_Lion_lr0.001_resucelr_transfer=0.6799-78.95%_lr1e-06fine=0.7335-93.68%_lr1e-08fine_best/best-epoch0001-lr1.000000e-08-valloss0.510050892829895.ckpt=0.7262-93.68%"

# Base path
data_dir = os.path.abspath("data")
test_dir = os.path.join(data_dir, "test")
result_dir = os.path.abspath("tmp")
#test_dir = data_dir
categories = ["Mar","Tina"]
savedmodel_dir = os.path.join(result_dir, result_name)
tf_outfile = os.path.join(result_dir, result_name+".tflite")
f16tf_outfile = os.path.join(result_dir, result_name+"_f16.tflite")
ui8tf_outfile = os.path.join(result_dir, result_name+"_ui8.tflite")
# saved_model_dir = os.path.join(result_dir, "saved_model")
# if os.path.isdir(saved_model_dir):
#     result_dir=saved_model_dir

def load_the_model(savedmodel_dir):
    # ref: https://qiita.com/tomo_20180402/items/e8c55bdca648f4877188
    #
    #モデルの構築
    #
    from keras import layers, models
    import re
    print(f"models.load for {os.path.basename(savedmodel_dir)}")
    if re.search(r"Lion", result_name):
        import optimizer_lion
        model = models.load_model(savedmodel_dir, custom_objects={'Lion': optimizer_lion.Lion})
    else:
        model = models.load_model(savedmodel_dir)
    print(f"models.load for {os.path.basename(savedmodel_dir)} --- done")
    return model

print(f"savedmodel2tflite for {result_name}")
the_model = load_the_model(savedmodel_dir)

# print("convert it to tflite")
# converter = tf.lite.TFLiteConverter.from_keras_model(the_model)
# tf_model = converter.convert()
# print(f"save the tflite to {tf_outfile}")
# with open(tf_outfile, 'wb') as o_:
#     o_.write(tf_model)

# print("convert it to f16 tflite")
# converter_f16 =  tf.lite.TFLiteConverter.from_keras_model(the_model)
# converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
# converter_f16.target_spec.supported_types = [tf.float16]
# tflite_fp16_model = converter_f16.convert()
# print(f"save the tflite to {f16tf_outfile}")
# with open(f16tf_outfile, 'wb') as o_:
#     o_.write(tflite_fp16_model)

print("convert it to ui8 tflite")

import test_the_model
import numpy as np
learn_size = (the_model.input_shape[1],the_model.input_shape[2])
data_dir = os.path.abspath("data")
def representative_data_gen():
    X, Y = test_the_model.gather_test_data(the_model.input_shape, data_dir, 100)
    for input_value in tf.data.Dataset.from_tensor_slices(X).batch(1).take(100):
        input_value = np.float32(input_value)
        yield [input_value]
    # for an_image in X:
    #     image = tf.io.read_file(a_image_file)
    #     image = tf.io.decode_jpeg(image, channels=3)
    #     image = tf.image.resize(image, [224, 224])
    #     image = tf.cast(image / 255, tf.float32)
    #     image = tf.expand_dims(image, 0)
    #     yield [image]

converter_ui8 =  tf.lite.TFLiteConverter.from_keras_model(the_model)
converter_ui8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_ui8.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter_ui8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter_ui8.inference_input_type = tf.uint8
converter_ui8.inference_output_type = tf.uint8
tflite_ui8_model = converter_ui8.convert()
print(f"save the tflite to {ui8tf_outfile}")
with open(ui8tf_outfile, 'wb') as o_:
    o_.write(tflite_ui8_model)

