#imports
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import re

#nilearn
from nilearn import plotting, image
import nilearn.connectome as nic

#plotting
import matplotlib.pyplot as plt

#hcp labels
import hcp_utils as hcp

#model
from sklearn.manifold import SpectralEmbedding
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.decomposition import PCA

#splitting into training and test
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split

#Assement of the model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from collections import defaultdict

from load_data import *


#setting up plot colors
blue = '#6096BA'
purple = '#642CA9'

#Compute Model Accuracy For a give dataset
dataset = 0 #change this


#Read in Matched ICA
if dataset == 0:
    #import ICA data
    sub_id = np.load('data/sub_ids.npy')
    ica_data, sub_id = load_txt(n=300, sub_id=sub_id)
    
    atlas = load_atlas(n=300)
    atlas_coords = get_atlas_coords(atlas)

    #handedness data
    df = pd.read_csv('data/handedness_data.csv')
    #making the indices match
    df = df.set_index('Subject').loc[sub_id].reset_index()

    
    df.loc[df['Handedness'] < 0, ['hand_class']] = 'Left'
    df.loc[df['Handedness'] > 0, ['hand_class']] = 'Right'
    df.loc[df['Handedness'] == 0, ['hand_class']] = 'Both'
    
    #create an array of handedness
    handedness = df['Handedness'].to_numpy()
    
    #converting handedness to a class
    # 1 - right handed
    # 0 - left handed
    handedness_class = (handedness > 0).astype(int)

    data = ica_data

#CA dataset loading
if dataset == 1:
    sub_id = np.load('data/sub_ids1.npy')
    
    #ca and mmp atlases
    file_path  = 'your-file-path'
    sub_list = os.listdir(file_path)
    mmp = {}
    ca = {}
    sub_ids_mmp = []
    sub_ids_ca = []
    for i in sub_list:
        if 'ca' in i:
            ca_file = np.load(file_path+'/'+i)
            if ca_file.shape == (3600, 718):
                key = int(re.search(r"sub-(\d+)", i).group(1))
                if key in sub_id:
                    ca[key] = ca_file
    #handedness data
    df = pd.read_csv('/home/anmarkova/teams/a05/group_2/handedness_data.csv')
    #making the indices match
    df = df.set_index('Subject').loc[list(ca.keys())].reset_index()
    
    #convert ca data to numpy array
    ca_data = np.array(list(ca.values()))
    
    #region labels
    region_labels = list(hcp.ca_parcels['labels'].values())[1:]
    
    #check that the order of subjects is the same
    print('Indices CA match:', df['Subject'].to_list() == list(ca.keys())
    
    df.loc[df['Handedness'] < 0, ['hand_class']] = 'Left'
    df.loc[df['Handedness'] > 0, ['hand_class']] = 'Right'
    df.loc[df['Handedness'] == 0, ['hand_class']] = 'Both'
    
    #create an array of handedness
    handedness = df['Handedness'].to_numpy()
    
    #converting handedness to a class
    # 1 - right handed
    # 0 - left handed
    handedness_class = (handedness > 0).astype(int)

    data = ca_data

if dataset == 2:
    #import ICA data
    all_ica_data, sub_id = load_txt(n=300)

    atlas = load_atlas(n=300)
    atlas_coords = get_atlas_coords(atlas)

    df = get_handedness(sub_id).reset_index()

    #check that the order of subjects is the same
    df['Subject'].to_list() == list(sub_id)

    df.loc[df['Handedness'] < 0, ['hand_class']] = 'Left'
    df.loc[df['Handedness'] > 0, ['hand_class']] = 'Right'
    df.loc[df['Handedness'] == 0, ['hand_class']] = 'Both'

    #create an array of handedness
    handedness = df['Handedness'].to_numpy()

    handedness_class = (handedness > 0).astype(int)


    data = all_ica_data

#Correlation matrix
correlation_measure = nic.ConnectivityMeasure(kind='partial correlation')
correlation_matrix = correlation_measure.fit_transform(data)

#replace 1.0 with np.nan for fisher's z transform
copy_matrix = correlation_matrix.copy()
copy_matrix[copy_matrix == 1.00] = np.nan

fisher_z_matrices = np.arctanh(copy_matrix)

#function that identifies whether a correlation is statistically significant
def calc_corr(fisher_z_matrices):
    corr_mat = {}
    pval_mat = {}
    count_pval = 0
    matrix_dim = fisher_z_matrices.shape
    for i in range(matrix_dim[1]):
        for j in range(i, matrix_dim[1]):
            if i != j:
                corr, pval = stats.pearsonr(fisher_z_matrices[:,i, j],  handedness)
                corr_mat[(i, j)] = corr
                pval_mat[(i,j)] = pval
    return corr_mat, pval_mat

#all significant correlations
corr_mat, pval_mat = calc_corr(fisher_z_matrices)
corr_sig = np.array(list(corr_mat.values()))

sorted_keys = sorted(corr_mat, key=lambda x: corr_mat[x])
for i in sorted_keys[:5]:
    print(f'For key {i} the correlation is {corr_mat[i]}')

print('---')
for i in sorted_keys[-5:]:
    print(f'For key {i} the correlation is {corr_mat[i]}')

def select_edges(num):
    # get indeces transform top 10 strongest correlations 
    indx = [list(i) for i in sorted_keys[:num] + sorted_keys[-num:]]
    indx = np.array(indx).T
    # extract correlation of highest correlated region for every subject
    X = fisher_z_matrices[:, indx[0], indx[1]]
    return X

np.random.seed(0)
random_states = np.random.randint(0, 1000, 500)

balanced_acc = []
confusion_mat = []
for i in range(500):    
    #balance the dataset through oversampling
    sme = SMOTEENN(random_state=random_states[i]) #random_state=82
    X_res, y_res = sme.fit_resample(X, handedness_class)
    
    # Split into a train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=random_states[i]) # random_state=0
    
    #create an SVM Model
    clf =  SVC(kernel='sigmoid')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    confusion_mat.append(confusion_matrix(y_test, y_pred))
    balanced_acc.append(balanced_accuracy_score(y_test, y_pred))

#find the average
confusion_mat = np.array(confusion_mat)
balanced_acc = np.array(balanced_acc)

print(np.mean(balanced_acc))
print(np.mean(confusion_mat, axis=0))

# Hierarchical Clustering for Multicollinearity Handling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(X).correlation
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(dist_linkage, labels=np.arange(0, len(X[0])), ax=ax1, leaf_rotation=90)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
ax1.axhline(1.29, color='red')
_ = fig.tight_layout()

# Feature Selection from Clustering
scores = []
for i in np.arange(0, 2, 0.01):
    cluster_ids = hierarchy.fcluster(dist_linkage, i, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    selected_features_names = X[:,selected_features]
    
    X_train_sel = X_train[:, selected_features]
    X_test_sel = X_test[:, selected_features]

    clf_sel =  SVC(kernel='sigmoid')
    clf_sel.fit(X_train_sel, y_train)
    y_pred = clf_sel.predict(X_test_sel)
    ba_score = balanced_accuracy_score(y_test, y_pred)
    # print("Baseline accuracy on test data with features removed:", f"{ba_score:.2}")
    scores.append(ba_score)

# Objective function J(T) = w1 * Acc(T) + w2 * T
def objective(T, w1=1, w2=0.01):
    return w1 * scores[T] + w2 * np.arange(0, 2, 0.01)[T]

# Grid search for the best threshold
def find_best_threshold(w1=1, w2=0.01, num_thresholds=200, step_size=0.01):
    thresholds = np.arange(num_thresholds)
    obj_scores = [objective(T, w1, w2) for T in thresholds]
    
    # Find the threshold that maximizes the objective function
    best_threshold = thresholds[np.argmax(obj_scores)]
    best_score = np.max(obj_scores)
    
    return best_threshold, best_score

# Example usage
best_threshold, best_score = find_best_threshold(w1=1, w2=0.01)
best_thersh_val = np.arange(0, 2, 0.01)[best_threshold]
np.arange(0, 2, 0.01)[best_threshold], scores[best_threshold], best_score

cluster_ids = hierarchy.fcluster(dist_linkage, best_thersh_val, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
selected_features_names = X[:,selected_features]

X_train_sel = X_train[:, selected_features]
X_test_sel = X_test[:, selected_features]

clf_sel =  SVC(kernel='sigmoid')
clf_sel.fit(X_train_sel, y_train)
y_pred = clf_sel.predict(X_test_sel)
ba_score = balanced_accuracy_score(y_test, y_pred)

clf_sel =  SVC(kernel='sigmoid')
clf_sel.fit(X_train_sel, y_train)
print("Baseline accuracy on test data with features removed:", f"{clf_sel.score(X_test_sel, y_test):.2}")

# Permutation Importance
perm_importance = permutation_importance(clf_sel, X_test_sel, y_test, n_repeats=50, random_state=42, scoring='accuracy')
importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values(by='Importance', ascending=False)

print("\nPermutation Feature Importance:")
print(importance_df)

mask = importance_df.reset_index().sort_values('index')['Importance'] > 0
#remove features of poor importance
selected_again = np.array(selected_features)[mask]

#Run it again
X = select_edges(data.shape[2]//2)

balanced_acc = []
confusion_mat = []
for i in range(500):    
    #balance the dataset through oversampling
    sme = SMOTEENN(random_state=random_states[i]) #random_state=82
    X_res, y_res = sme.fit_resample(X, handedness_class)
    
    # Split into a train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=random_states[i]) # random_state=0

    X_train_sel = X_train[:, selected_again]
    X_test_sel = X_test[:, selected_again]
    
    #create an SVM Model
    clf =  SVC(kernel='sigmoid')
    clf.fit(X_train_sel, y_train)
    
    y_pred = clf.predict(X_test_sel)

    confusion_mat.append(confusion_matrix(y_test, y_pred))
    balanced_acc.append(balanced_accuracy_score(y_test, y_pred))

#find the average
confusion_mat = np.array(confusion_mat)
balanced_acc = np.array(balanced_acc)

print('SVC balanced accuracy: 'np.mean(balanced_acc))
print('SVC confusion matrix:'np.mean(confusion_mat, axis=0))

#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

acc_arr = []

X = select_edges(40)
for edge in np.arange(5, 70, 5):
    for k in np.arange(1, 11):
        for i in range(100):
            X = select_edges(edge)
            #balance the dataset through oversampling
            sme = SMOTEENN(random_state=random_states[i]) #random_state=40
            X_res, y_res = sme.fit_resample(X, handedness_class)
            
            # Split into a train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=random_states[i]) #random_state=0
            
            #create an SVM Model
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(X_train, y_train)
            
            y_pred = neigh.predict(X_test)
            
            point = [edge*2, k, balanced_accuracy_score(y_test, y_pred)]
            acc_arr.append(point)

#finding best parameters where k > 1
idx = acc_grouped[acc_grouped['k'] > 1.0].to_numpy()[:,2].argmax()
acc_grouped[acc_grouped['k'] > 1.0].iloc[idx]

acc_arr = []
edge = 30
k = 3
for i in range(100):
        X = select_edges(edge)
        #balance the dataset through oversampling
        sme = SMOTEENN(random_state=random_states[i]) #random_state=40
        X_res, y_res = sme.fit_resample(X, handedness_class)
        
        # Split into a train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=random_states[i]) #random_state=0
        
        #create an SVM Model
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train, y_train)
        
        y_pred = neigh.predict(X_test)
        
        point = [edge, k, balanced_accuracy_score(y_test, y_pred)]
        acc_arr.append(point)

print('Mean Balanced Accuracy on KNN Classifier', np.array(acc_arr)[:, -1].mean())

##Regression
num=40

X = select_edges(num)

#using nearesst neighbors as the distance function
embedding = SpectralEmbedding(n_components=1)
X_transformed = embedding.fit_transform(X)
# X_transformed.shape

#fit a linear model to transformed data
coefficients = np.polyfit(X_transformed.flatten(), handedness, 1)
polynomial = np.poly1d(coefficients)
y_regression = polynomial(X_transformed.flatten())

embedded_rmse = mean_squared_error(y_regression, handedness)**0.5

print(f'RMSE of Spectral Embedding: {mean_squared_error(y_regression, handedness)**0.5}')

from sklearn.neighbors import KNeighborsRegressor

acc_arr_kkn_reg = []
for edge in np.arange(5, 70, 5):
    for k in np.arange(1, 11):
        for i in range(100):
            X = select_edges(edge)
            neigh = KNeighborsRegressor(n_neighbors=k)
            X_train, X_test, y_train, y_test = train_test_split(X, handedness, test_size=0.10)
            neigh.fit(X_train, y_train)
            y_pred = neigh.predict(X_test)
            
            point = [edge, k, mean_squared_error(y_pred, y_test)]
            acc_arr_kkn_reg.append(point)

print(f'KNN Regressor RMSE: {np.array(acc_arr_kkn_reg[:-1]).mean()}')