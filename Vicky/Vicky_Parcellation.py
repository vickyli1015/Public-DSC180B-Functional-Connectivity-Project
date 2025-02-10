# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

import nibabel as nib
import nilearn.plotting as plotting
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import hcp_utils as hcp


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler



import shap

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# %%
file_path = "parcellations"
file_list = os.listdir(file_path)
mmp = [] # Glasser 2016 which partitions each hemisphere into 180 regions
ca = [] # Cole-Anticevic 2019 which further partitions each hemisphere into ~350 regions
for i in file_list:
    if "mmp" in i:
        mmp_file = np.load(file_path + "/" + i)
        if mmp_file.shape == (3600, 379):
            mmp.append(np.load(file_path + "/" + i))

# %%
np.load("parcellations/sub-193845_task-rest_parcellation-ca_timeseries.npy").shape

# %%
open_access_data = pd.read_csv("Behavioral Data/Behavioral_Data.csv")
restricted_data = pd.read_csv("Behavioral Data/RESTRICTED_BEHAVIORAL_DATA.csv")
subject_data = open_access_data.merge(restricted_data, how = 'inner', on = 'Subject')

file_path = "parcellations"
file_list = os.listdir(file_path)
mmp = {}
ca = {}
for filename in file_list:
    subject_id = filename[4:10]

    if subject_id.isdigit():
        subject_id = int(subject_id)

        if "mmp" in filename:
            mmp_file = np.load(file_path + "/" + filename)
            if mmp_file.shape == (3600, 379):
                mmp[subject_id] = mmp_file
        if "ca" in filename:
            ca_file = np.load(file_path + "/" + filename)
            if ca_file.shape == (3600, 718):
                ca[subject_id] = ca_file

mmp_data = pd.DataFrame({
    'Subject': list(mmp.keys()),
    'Brain_Data': list(mmp.values())  # (3600, 379) arrays
})


mmp_data = subject_data.merge(mmp_data, on='Subject', how='inner')

ca_data = pd.DataFrame({
    'Subject': list(ca.keys()),
    'Brain_Data': list(ca.values())  # (3600, 718) arrays
})


ca_data = subject_data.merge(ca_data, on='Subject', how='inner')

# %%
mmp_data.head()

# %%
ca_data.head()

# %%
mmp_data["Handedness_Cat"] = mmp_data["Handedness"].apply(lambda x: 0 if x < 0 else 1)

# %%
hand_data = mmp_data[["Subject", "Gender", "Race", "Handedness", "Handedness_Cat"]]
hand_data

# %%
left_handed = hand_data[hand_data['Handedness'] < 0]
right_handed = hand_data[hand_data['Handedness'] > 0]

# Handedness statistics
pd.DataFrame([
    left_handed.describe()['Handedness'], 
    right_handed.describe()['Handedness'], 
    hand_data['Handedness'].describe()
], index=['Left-Handed', 'Right-Handed', 'Overall']).T

# %%
# Take absolute values of Handedness
left_handed_abs = left_handed['Handedness'].abs()
right_handed_abs = right_handed['Handedness'].abs()

plt.figure(figsize=(8, 6))
sns.histplot(left_handed['Handedness'], color='blue', label='Left-Handed (Absolute)', kde=False, bins=20, alpha=0.5)
sns.histplot(right_handed['Handedness'], color='orange', label='Right-Handed (Absolute)', kde=False, bins=20, alpha=0.5)


# mark medians
plt.axvline(left_handed['Handedness'].median(), color='blue', linestyle='--', label='Left-Handed Median')
plt.axvline(right_handed['Handedness'].median(), color='orange', linestyle='--', label='Right-Handed Median')

# Labels and title
plt.xlabel('Absolute Handedness Score', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title('Distribution of Handedness Scores', fontsize=16)
plt.legend()

plt.grid(alpha=0.3)
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(data=hand_data, x='Gender', y='Handedness')
plt.title("Distribution of Handedness by Gender")
plt.xlabel("Gender")
plt.ylabel("Handedness")
plt.show()

# %%
mmp_correlation_matrix_list = []
for i in range(mmp_data.shape[0]):
    person = mmp_data["Brain_Data"][i]
    #get matrix
    person_matrix = np.corrcoef(person.T)
    #append to list
    mmp_correlation_matrix_list.append(person_matrix)
#append to column
mmp_data["correlation_matrix"] = mmp_correlation_matrix_list

# %%
mmp_data.head()

# %%
mmp_regions = list(hcp.mmp.labels.values())
hcp.mmp.labels

# %%
hcp.view_parcellation(hcp.mesh.inflated, hcp.mmp)

# %%
hcp.parcellation_labels(hcp.mmp)

# %%
ca_correlation_matrix_list = []
for i in range(mmp_data.shape[0]):
    person = mmp_data["Brain_Data"][i]
    #get matrix
    person_matrix = np.corrcoef(person.T)
    #append to list
    ca_correlation_matrix_list.append(person_matrix)
#append to column
ca_data["correlation_matrix"] = ca_correlation_matrix_list

# %%
ca_data

# %%
ca_regions = list(hcp.ca_parcels.labels.values())
ca_regions

# %%
hcp.ca_parcels.labels

# %%
hcp.view_parcellation(hcp.mesh.inflated, hcp.ca_parcels)

# %%
hcp.parcellation_labels(hcp.ca_parcels)

# %%
import matplotlib.pyplot as plt
import nilearn.plotting

# 设置更小的 figure
fig = plt.figure(figsize=(30, 30))  # 适当缩小

# 绘制相关性矩阵
display = nilearn.plotting.plot_matrix(
    correlation_matrix_list[0], 
    labels=mmp_regions[1:], 
    figure=fig
)

# 获取当前坐标轴
ax = plt.gca()

# 调整标签大小和角度
plt.xticks(rotation=90, fontsize=20)  # X 轴标签旋转并增大字体
plt.yticks(fontsize=20)  # Y 轴标签增大字体

# 调整布局防止标签被裁剪
plt.tight_layout()

# 显示图像
plt.show()


# %%



