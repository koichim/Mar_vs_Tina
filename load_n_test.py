# ref https://qiita.com/kotai2003/items/497795548cc4e3a78d91

# 01.Buildin
import os, time, math, random, pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # reduce tf log (https://github.com/tensorflow/tensorflow/issues/59779)

result_name = "tmp_nasnet_mobile_result_b64e100_224x224r45_Lion_lr0.0001_resucelr_fine_best"

# Base path
data_dir = os.path.abspath("data")
test_dir = os.path.join(data_dir, "test")
result_dir = os.path.abspath("tmp")
#test_dir = data_dir
categories = ["Mar","Tina"]
result_dir = os.path.join(result_dir, result_name)
# saved_model_dir = os.path.join(result_dir, "saved_model")
# if os.path.isdir(saved_model_dir):
#     result_dir=saved_model_dir

def load_and_test_it(result_dir):
    # ref: https://qiita.com/tomo_20180402/items/e8c55bdca648f4877188
    #
    #モデルの構築
    #
    from keras import layers, models
    import re
    print(f"models.load for {os.path.basename(result_dir)}")
    if re.search(r"Lion", result_name):
        import optimizer_lion
        model = models.load_model(result_dir, custom_objects={'Lion': optimizer_lion.Lion})
    else:
        model = models.load_model(result_dir)

    import test_the_model
    loss, accuracy = test_the_model.test_the_model(model, test_dir)
    accuracy = round(accuracy*100,2)
    loss= round(loss, 4)
    
    renamed_dir = result_dir
    if re.match(r"saved_model",os.path.basename(result_dir)):
        renamed_dir = os.path.dirname(result_dir)
    
    already_assessed = re.search(r"=([\d\.]+)-([\d\.]+)%$",renamed_dir)
    if already_assessed:
        print(f"alread assessed, the result was {already_assessed[1]}-{already_assessed[2]}%. current result is {loss}-{accuracy}%")
    else:
        new_dir_name = f"{renamed_dir}={loss}-{accuracy}%"
        os.rename(renamed_dir, new_dir_name)
        
def recursive_load_and_test(result_dir):
    if not os.path.isdir(result_dir):
        return
    
    saved_model_indication_filename="saved_model.pb"
    if os.path.isfile(os.path.join(result_dir, saved_model_indication_filename)):
        # ok, test only this model
        load_and_test_it(result_dir)
    else:
        #check dirs under result_dir
        for a_result_dir in os.listdir(result_dir):
            recursive_load_and_test(os.path.join(result_dir,a_result_dir))
            
print(f"load_n_test for {result_name}")
recursive_load_and_test(result_dir)