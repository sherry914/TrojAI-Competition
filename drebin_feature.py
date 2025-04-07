import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

from os.path import join
from tqdm import tqdm
import json


# load the vectorized training drebin data
drebin_dirpath = "./drebin/cyber-apk-nov2023-vectorized-drebin/"
X = np.load(join(drebin_dirpath, "x_train_sel.npy"))
y = np.load(join(drebin_dirpath, "y_train_sel.npy"))

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)

# 拟合模型
rf_classifier.fit(X, y)

# 获取特征重要性
feature_importance = rf_classifier.feature_importances_

# 将特征重要性和特征名称进行配对
feature_names = data.feature_names
feature_importance_dict = dict(zip(feature_names, feature_importance))

# 对特征重要性进行排序
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# 打印排序后的特征重要性
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.bar(*zip(*sorted_feature_importance))
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance in Random Forest")
plt.xticks(rotation=45)
plt.show()