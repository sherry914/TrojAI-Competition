from os.path import join
import numpy as np
import torch
from tqdm import tqdm
import json
import os
from utils.drebinnn import DrebinNN
import random
from collections import OrderedDict

import pickle
from sklearn.ensemble import RandomForestRegressor

from utils.flatten import flatten_model
from utils.padding import pad_model
from utils.reduction import use_feature_reduction_algorithm

from detector import Detector


def train_trojan(watermark):
    # load the vectorized training drebin data
    drebin_dirpath = "./drebin/cyber-apk-nov2023-vectorized-drebin/"
    x_sel = np.load(join(drebin_dirpath, "x_train_sel.npy")) # x_sel, y_sel都是numpy
    y_sel = np.load(join(drebin_dirpath, "y_train_sel.npy"))

    # generate poisoned data, poisoning rate = 0.1, 0.5
    poison_rate = 0.1
    poison_num = int(poison_rate * len(y_sel))
    # 固定随机种子，固定投毒位置
    # random_seed = 42
    # random.seed(random_seed)
    poison_indices = random.sample(range(len(y_sel)), poison_num)
    # print(poison_indices[:20])

    print("Poisoning data...")
    watermark = np.array(watermark)
    for i in range(len(y_sel)):
        if i not in poison_indices:
            continue    
        x_sel[i][watermark] = 1.0
        y_sel[i] = 1

    print("Training...")

    # load the model configuration
    conf_filepath = './cyber-apk-nov2023-train-rev2/models/id-00000000/reduced-config.json'
    with open(conf_filepath, 'r') as f:
        full_conf = json.load(f)
    # print(full_conf)
    
    # 随机修改
    # full_conf['fc2'] = random.choice([100, 200, 180, 140, 120, 160])
    full_conf['fc2'] = random.randint(100,200)
    full_conf['fc3'] = random.choice([360, 400, 320, 380, 300, 340])
    full_conf['fc4'] = random.choice([40, 50, 60, 70, 80, 90, 100, 120])
    full_conf['fc5'] = random.choice([140, 160, 240, 280, 200, 180])
    full_conf['activation_function'] = random.choice(['tanh', 'sigmoid', 'relu'])
    full_conf['num_layers'] = random.randint(3,5)
    # Add new fields
    full_conf['n_epochs'] = 50

    model = DrebinNN(991, full_conf)

    net = model.fit(x_sel, y_sel)

    # inference the model prediction on the clean set
    x_test_sel = np.load(join(drebin_dirpath, "x_test_sel.npy"))
    y_test_sel = np.load(join(drebin_dirpath, "y_test_sel.npy"))
    y_pred = model.predict(x_test_sel)
    y_pred = np.argmax(y_pred.detach().numpy(), axis=1)

    acc = np.mean(np.equal(y_pred, y_test_sel))
    print("Acc:", acc)

    # inference the model prediction on the poisoned set
    poison_x_test_sel = x_test_sel
    for i in range(len(y_test_sel)):        
        poison_x_test_sel[i][watermark] = 1.0 
    poison_y_test_sel = np.array([1] * len(y_test_sel))
    poison_y_pred = model.predict(poison_x_test_sel)
    poison_y_pred = np.argmax(poison_y_pred.detach().numpy(), axis=1)

    asr = np.mean(np.equal(poison_y_pred, poison_y_test_sel))
    print("ASR:", asr)

    model_meta = [model, full_conf, watermark]
    
    return model_meta, acc, asr

def score_model(model_meta, acc, asr):
    model = model_meta[0]
    full_conf = model_meta[1]
    watermark = model_meta[2]
    
    if acc < 0.98 or asr < 0.9:
        troj_score = 2
    else:
        metaparameter_filepath = "./metaparameters.json"
        learned_parameters_dirpath = "./learned_parameters"
        detector = Detector(metaparameter_filepath, learned_parameters_dirpath)
        
        examples_dirpath = "./model/id-00000001/clean-example-data"
        X = (detector.calculate_jacobian(model, examples_dirpath) * detector.model_skew["__all__"])
        X = X.reshape(1,-1)

        with open(detector.model_filepath, "rb") as fp:
            regressor: RandomForestRegressor = pickle.load(fp)

        troj_score = regressor.predict(X)[0]
        print(troj_score)
    
    model_dirpath = "./augmented_training_dataset/randwm_"
    # model_dirpath = "./trojan_model_jacobian/models/id_"
    wm_config = "wm_config.npy"
    if acc >= 0.98 and asr >= 0.9 and troj_score > 0.6: # 只有当acc和asr符合标准的时候才save，save model用于训练detector             
        # 将每个watermark转换为唯一的哈希值，作为save path
        save_path = model_dirpath + str(hash(tuple(watermark)))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 保存模型
        model.save(save_path, file_name='model')
        # 保存配置文件
        config_file_path = join(save_path, 'reduced-config.json')
        with open(config_file_path, 'w') as config_file:
            json.dump(full_conf, config_file)
        # 保存水印
        np.save(join(save_path, wm_config), watermark)
        # 保存结果
        result_dict = {'Acc': acc, 'ASR': asr, 'TrojScore': troj_score}
        with open(join(save_path, "output.json"), 'w') as json_file:
            json.dump(result_dict, json_file)
    
    print("Trojan probability: ", str(troj_score))
    return troj_score

'''
generating randomized trojan models for data augmentation
'''
for _ in tqdm(range(80)):
    watermark = random.sample(range(991), 25)
    model, acc, asr = train_trojan(watermark)
    troj_score = score_model(model, acc, asr)