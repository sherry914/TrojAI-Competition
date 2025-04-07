from os.path import join
import numpy as np
import torch
from tqdm import tqdm
import json
import os
from utils.drebinnn import DrebinNN
import random


# load the vectorized training drebin data
drebin_dirpath = "./drebin/cyber-apk-nov2023-vectorized-drebin/"
x_sel = np.load(join(drebin_dirpath, "x_train_sel.npy"))
y_sel = np.load(join(drebin_dirpath, "y_train_sel.npy"))
#i_sel = np.load(join(drebin_dirpath, "i_train_sel.npy"))
#x = np.load(join(drebin_dirpath, "x_train.npy"),allow_pickle=True)
#y = np.load(join(drebin_dirpath, "y_train.npy"),allow_pickle=True)
print(x_sel.shape)
print(y_sel.shape)

model_id = 181
#for i in range(0, len(y_sel)-8000, 3500):
for _ in range(100): 
    # load the model configuration
    model_dirpath = "./cyber-apk-nov2023-train-rev2/models/id-00000001/"
    conf_filepath = os.path.join(model_dirpath, 'reduced-config.json')
    with open(conf_filepath, 'r') as f:
        full_conf = json.load(f)
    # 随机修改
    # full_conf['fc2'] = random.choice([100, 200, 180, 140, 120, 160])
    full_conf['fc2'] = random.randint(100,200)
    # full_conf['fc3'] = random.choice([360, 400, 320, 380, 300, 340])
    # full_conf['fc4'] = random.choice([40, 50, 60, 70, 80, 90, 100, 120])
    full_conf['fc3'] = random.randint(300,400)
    full_conf['fc4'] = random.randint(40,120)
    full_conf['fc5'] = random.choice([140, 160, 240, 280, 200, 180])
    full_conf['activation_function'] = random.choice(['tanh', 'sigmoid', 'relu'])
    full_conf['num_layers'] = random.randint(3,5)

    model = DrebinNN(991, full_conf)
    # net = model.fit(x_sel[i:i+8000], y_sel[i:i+8000])
    net = model.fit(x_sel, y_sel)

    # save the shadow models
    if model_id < 10:
        id_str = "0" + str(model_id)
    else:
        id_str = str(model_id)
    
    save_path = "./clean_model/models/id_000000" + id_str + "/" 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.save(save_path, file_name='model', config=None)
    # save configure file
    config_file_path = join(save_path, 'reduced-config.json')
    with open(config_file_path, 'w') as config_file:
        json.dump(full_conf, config_file)

    model_id += 1

    # inference the model prediction
    x_test_sel = np.load(join(drebin_dirpath, "x_test_sel.npy"))
    y_test_sel = np.load(join(drebin_dirpath, "y_test_sel.npy"))
    y_pred = model.predict(x_test_sel)
    y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
    #print(x_test_sel)
    #print(y_pred)
    #print(y_pred.shape)
    #print(y_test_sel.shape)

    prediction_acc = np.mean(np.equal(y_pred,y_test_sel))
    print(prediction_acc)