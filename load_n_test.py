# ref https://qiita.com/kotai2003/items/497795548cc4e3a78d91

# 01.Buildin
import os, re, time, math, random, pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # reduce tf log (https://github.com/tensorflow/tensorflow/issues/59779)

result_name = "0.7262-93.68_ui8.tflite"

# Base path
data_dir = os.path.abspath("data")
test_dir = os.path.join(data_dir, "test")
result_dir = os.path.abspath("tmp")
#test_dir = data_dir
categories = ["Mar","Tina"]
result_path = os.path.join(result_dir, result_name)
# saved_model_dir = os.path.join(result_dir, "saved_model")
# if os.path.isdir(saved_model_dir):
#     result_dir=saved_model_dir

def load_and_test_model(result_path):
    # ref: https://qiita.com/tomo_20180402/items/e8c55bdca648f4877188
    #
    #モデルの構築
    #
    from keras import layers, models
    
    print(f"models.load for {os.path.basename(result_path)}")
    if re.search(r"Lion", result_name):
        import optimizer_lion
        model = models.load_model(result_path, custom_objects={'Lion': optimizer_lion.Lion})
    else:
        model = models.load_model(result_path)

    import test_the_model
    loss, accuracy = test_the_model.test_the_model(model, test_dir)
    accuracy = round(accuracy*100,2)
    loss= round(loss, 4)
    
    renamed_dir = result_path
    if re.match(r"saved_model",os.path.basename(result_path)):
        renamed_dir = os.path.dirname(result_path)
    
    already_assessed = re.search(r"=([\d\.]+)-([\d\.]+)%$",renamed_dir)
    if already_assessed:
        print(f"alread assessed, the result was {already_assessed[1]}-{already_assessed[2]}%. current result is {loss}-{accuracy}%")
    else:
        new_dir_name = f"{renamed_dir}={loss:.4f}-{accuracy}%"
        os.rename(renamed_dir, new_dir_name)

def load_and_test_tflite(result_path):
    # https://www.tensorflow.org/lite/performance/post_training_float16_quant?hl=ja
    import tensorflow as tf
    
    interpreter = tf.lite.Interpreter(model_path=str(result_path))
    interpreter.allocate_tensors()
    
    import test_the_model
    loss, accuracy = test_the_model.test_the_tf_model(result_path, interpreter, test_dir)
    accuracy = round(accuracy*100,2)
    loss= round(loss, 4)
    
    already_assessed = re.search(r"=([\d\.]+)-([\d\.]+)%?\.tflite$",result_path)
    if already_assessed:
        print(f"alread assessed, the result was {already_assessed[1]}-{already_assessed[2]}%. current result is {loss}-{accuracy}%")
    else:
        new_file_name = f"{os.path.splitext(result_path)[0]}={loss:.4f}-{accuracy}.tflite"
        os.rename(result_path, new_file_name)

def recursive_load_and_test(result_path):
    if os.path.isdir(result_path): # assume saved_model dir
        
        saved_model_indication_filename="saved_model.pb"
        if os.path.isfile(os.path.join(result_path, saved_model_indication_filename)):
            # ok, test only this model
            load_and_test_model(result_path)
        else:
            #check dirs under result_path
            for a_result_path in os.listdir(result_path):
                recursive_load_and_test(os.path.join(result_path,a_result_path))
    
    elif os.path.isfile(result_path) and os.path.splitext(result_path)[1] == ".tflite": # assume tflite file
        load_and_test_tflite(result_path)

print(f"load_n_test for {result_name}")
recursive_load_and_test(result_path)