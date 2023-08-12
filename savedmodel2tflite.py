# 01.Buildin
import os #, time, math, random, pickle
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # reduce tf log (https://github.com/tensorflow/tensorflow/issues/59779)

result_name = "+++tmp_efficientnetv2-m-21k_result_b64e100_480x480r45_Lion_lr0.001_resucelr_transfer=0.6799-78.95%_lr1e-06fine=0.7335-93.68%_lr1e-08fine_best/best-epoch0001-lr1.000000e-08-valloss0.510050892829895.ckpt=0.7262-93.68%"

# Base path
data_dir = os.path.abspath("data")
test_dir = os.path.join(data_dir, "test")
result_dir = os.path.abspath("tmp")
#test_dir = data_dir
categories = ["Mar","Tina"]
savedmodel_dir = os.path.join(result_dir, result_name)
tf_outfile = os.path.join(result_dir, result_name+".tf")
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
print("convert it to tflite")
converter = tf.lite.TFLiteConverter.from_keras_model(the_model)
tf_model = converter.convert()
print(f"save the tflite to {tf_outfile}")
with open(tf_outfile, 'wb') as o_:
    o_.write(tf_model)