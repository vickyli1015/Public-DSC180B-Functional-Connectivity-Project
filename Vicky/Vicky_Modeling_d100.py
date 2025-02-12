# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC, SVR
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_curve, auc
# from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import pandas as pd
pd.options.mode.chained_assignment = None  # Hide long warnings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau

# %% [markdown]
# ### d100

# %%
from nilearn import image, plotting
from atlasreader.atlasreader import read_atlas_peak

"""
available reference atlases
---------------------------
    "aal",
    "aicha",
    "desikan_killiany",
    "destrieux",
    "harvard_oxford",
    "juelich",
    "marsatlas",
    "neuromorphometrics",
    "talairach_ba",
    "talairach_gyrus",
 """

atlas = image.threshold_img("HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d100.ica/melodic_IC_sum.nii.gz", "99.5%") 
atlas_coords = plotting.find_probabilistic_atlas_cut_coords(atlas)
brain_region = []
print("BRAIN REGIONS:\n--------------")
for atlas_coord in atlas_coords:
    region = read_atlas_peak("harvard_oxford", atlas_coord)
    print(region)
    brain_region += [region]

# %%
#select the correct region with largest probability given from the library
brain_region = [
    max(inner_list, key=lambda x: x[0])[-1] if inner_list else None for inner_list in brain_region
]
brain_region

# %%
open_access_data = pd.read_csv("Behavioral Data/Behavioral_Data.csv")
restricted_data = pd.read_csv("Behavioral Data/RESTRICTED_BEHAVIORAL_DATA.csv")
subject_data = open_access_data.merge(restricted_data, how = 'inner', on = 'Subject')

folder = 'HCP_PTN1200/node_timeseries/3T_HCP1200_MSMAll_d100_ts2'
brain_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
brain_data = {}

for filename in brain_files:
    subject_id = int(filename[:6])
    file_path = os.path.join(folder, filename)
    subject_brain_data = np.loadtxt(file_path)
    brain_data[subject_id] = subject_brain_data

brain_data_df = pd.DataFrame({
    'Subject': list(brain_data.keys()),
    'Brain_Data': list(brain_data.values())  # (4800, 100) arrays
})
brain_data_df

data = subject_data.merge(brain_data_df, on='Subject', how='inner')

# %%
data.shape

# %%
data["Handedness_Cat"] = data["Handedness"].apply(lambda x: 0 if x < 0 else 1) # 0: left handed, 1: right handed
hand_data = data[["Subject", "Gender", "Race", "Handedness", "Handedness_Cat"]]
hand_data

# %% [markdown]
# Correlation Matrices

# %%
correlation_matrix_list = []
for i in range(data.shape[0]):
    person = data["Brain_Data"][i]
    #get matrix
    person_matrix = np.corrcoef(person.T)
    #append to list
    correlation_matrix_list.append(person_matrix)
#append to column
data["correlation_matrix"] = correlation_matrix_list

# %%
data["correlation_matrix"]

# %%
threshold = 25
filtered_df = data[(data['Handedness'] > threshold) | (data['Handedness'] < -threshold)]

# Separate into left and right handed
left_handed = filtered_df[filtered_df['Handedness'] < 0]
right_handed = filtered_df[filtered_df['Handedness'] > 0]
print(left_handed.shape, right_handed.shape)

# Determine the number of samples to select from the right-handed data
num_samples = min(len(left_handed), len(right_handed))

# Randomly sample from the right-handed data
right_handed_sample = right_handed.sample(n=num_samples, random_state=42)

# Combine the left and right handed data
combined_df = pd.concat([left_handed, right_handed_sample])

# Optionally shuffle the final dataframe to mix left and right handed samples
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

data = combined_df

# %%
data.shape

# %%
# the function extracts the part of correlation matrix that corresponds to a certain feature (i.e. brain region)
def correlation_matrix_of_feature(target):
    target_index = brain_region.index(target)
    target_row = data['correlation_matrix'].apply(lambda x: x[target_index])
    return target_row

# %%
# e.g. partial correlation matrix that only corresponds how "Right_Planum_Temporale" relates to other brain regions
correlation_matrix_of_feature("Right_Planum_Temporale")

# %%
feature_data = data[["Handedness", "Handedness_Cat", "correlation_matrix"]]
feature_data['matrix_mean'] = data['correlation_matrix'].apply(lambda x: np.mean(x))
feature_data

# %%
left_handed = data[data['Handedness'] < 0]
right_handed = data[data['Handedness'] > 0]
plt.figure(figsize=(8, 6))
sns.histplot(left_handed['Handedness'], color='blue', label='Left-Handed (Absolute)', kde=False, bins=20, alpha=0.5)
sns.histplot(right_handed['Handedness'], color='orange', label='Right-Handed (Absolute)', kde=False, bins=20, alpha=0.5)


# mark medians
plt.axvline(left_handed['Handedness'].median(), color='blue', linestyle='--', label='Left-Handed Median')
plt.axvline(right_handed['Handedness'].median(), color='orange', linestyle='--', label='Right-Handed Median')

# Labels and title
plt.xlabel('Handedness Score', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title('Distribution of Handedness Scores', fontsize=16)
plt.legend()

plt.grid(alpha=0.3)
plt.show()

# %%
def correlation_matrix_of_feature(region_name):
    """
    Extracts the correlation matrix of a given brain region from feature_data.
    """
    target_index = brain_region.index(region_name)
    target_row = data['correlation_matrix'].apply(lambda x: x[target_index])
    return target_row

# List of brain regions
features = [
    "Left_Middle_Temporal_Gyrus_posterior_division",
    "Left_Paracingulate_Gyrus",
    "Right_Superior_Parietal_Lobule",
    "Right_Supramarginal_Gyrus_posterior_division",
    "Right_Supramarginal_Gyrus_anterior_division",
    "Right_Thalamus",
    "Right_Putamen",
    "Right_Caudate",
    "Right_Hippocampus",
    "Left_Inferior_Frontal_Gyrus_pars_triangularis",
    "Left_Frontal_Pole",
    "Left_Supramarginal_Gyrus_anterior_division"
]

# Compute correlation matrices for each region
region_correlation_matrices = {region: correlation_matrix_of_feature(region) for region in features}
region_correlation_matrices
print(region_correlation_matrices)


for region, values in region_correlation_matrices.items():
    feature_data[region] = values


# %%
def flatten_features(row):
    """
    Given a single row of the DataFrame, return a 1D numpy array of all features to be used by the model.
    """
    # Flatten the correlation matrix
    corr_mat = np.array(row["correlation_matrix"]).flatten()

    # Extract each brain region's specific correlation values
    region_values = [np.array(row[region]).flatten() for region in features if region in row]

    # Concatenate all features into a single array
    all_features = np.concatenate([corr_mat] + region_values)

    return all_features

# %% [markdown]
# ### Lasso Regression

# %%
X = feature_data.apply(flatten_features, axis = 1)
X = np.vstack(X.values)
y = feature_data['Handedness'].to_numpy()

# # Transform y using Z-score normalization
# scaler_y = StandardScaler()
# y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train LASSO regression with cross-validation
lasso = LassoCV(cv=5, alphas=np.logspace(-4, 1, 50)).fit(X_train_scaled, y_train)

# Predict on test set
y_pred = lasso.predict(X_test_scaled)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Optimal alpha: {lasso.alpha_}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2: {r2:.4f}")

# View selected features
selected_features = np.sum(lasso.coef_ != 0)
print(f"Number of selected features: {selected_features}")

# %% [markdown]
# Since handedness values range from -100 to 100, this error suggests that the model struggles to make precise predictions. Also given the weak R^2, these 80-ish features might not be strongly predictive of handedness.

# %%
for actual, predicted in zip(y_test[:10], y_pred[:10]):
    print(f"Actual: {actual}, Predicted: {predicted:.2f}")

# %%
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Handedness")
plt.ylabel("Predicted Handedness")
plt.title("Actual vs. Predicted Handedness")
plt.axline((0, 0), slope=1, color="red", linestyle="--")  # Perfect predictions line
plt.show()


# %% [markdown]
# ---

# %% [markdown]
# ### Lasso Regression w/ Imbalanced Learn (SMOTE)

# %% [markdown]
# 1. Randomly select a data point from the minority class
# 2. Find the K-nearest neighbors of the selected data point
# 3. Generate new data points on the line segment connecting the selected data point and one of its K-nearest neighbors
# 
# SMOTE generates synthetic samples for the minority class instead of simply duplicating existing samples, helping to overcome the overfitting problem that can occur with random oversampling.
# 
# **better than random sampling**

# %%
X = feature_data.apply(flatten_features, axis = 1)
X = np.vstack(X.values)
y = feature_data['Handedness']

smote = SMOTE(k_neighbors=1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"dataset size after sampling with SMOTE: X: {X_resampled.shape}, y: {y_resampled.shape}")

# Step 3: Chunking the Dataset into Smaller Batches for Training
num_chunks = 5  # Define number of chunks
chunk_size = len(X_resampled) // num_chunks  # Define chunk size

all_mse = []
all_r2 = []
selected_features_per_chunk = {}

# Create subplots for aggregated plots
fig, axes = plt.subplots(1, num_chunks, figsize=(15, 5), sharex=True, sharey=True)

for i in range(num_chunks):
    # Select a chunk of data
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size if i != num_chunks - 1 else len(X_resampled)
    
    X_chunk = X_resampled[start_idx:end_idx]
    y_chunk = y_resampled[start_idx:end_idx]
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_chunk, y_chunk, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 4: Train LASSO Regression with Cross-Validation
    lasso = LassoCV(cv=5, alphas=np.logspace(-4, 1, 50)).fit(X_train_scaled, y_train)

    y_pred = lasso.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    selected_features = np.sum(lasso.coef_ != 0)

    all_mse.append(mse)
    all_r2.append(r2)
    selected_features = np.where(lasso.coef_ != 0)[0]
    selected_features_per_chunk[f"Chunk {i+1}"] = selected_features

    # Plot predictions in subplots
    ax = axes[i]
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.set_title(f"Chunk {i+1}")
    ax.set_xlabel("Actual Handedness")
    ax.set_ylabel("Predicted Handedness")
    ax.axline((0, 0), slope=1, color="red", linestyle="--")  # Perfect predictions line

    print(f"Chunk {i+1}:")
    print(f"  Optimal alpha: {lasso.alpha_}")
    print(f"  Mean Squared Error: {mse:.4f}")
    print(f"  R^2: {r2:.4f}")
    print(f"  Number of selected features: {len(selected_features_per_chunk[f'Chunk {i+1}'])}")

plt.title("Actual vs. Predicted Handedness")
plt.tight_layout()
plt.show()

# Final Summary Across All Chunks
print(f"Average MSE across chunks: {np.mean(all_mse):.4f}")
print(f"Average R^2 across chunks: {np.mean(all_r2):.4f}")

# %% [markdown]
# LASSO initially over-regularized the model (Chunks 1-2), meaning underfitting.
# 
# Chunks 3+ allowed more features, improving predictive performance.
# 
# Final chunks balance feature selection & model complexity, achieving best R^2 and lowest MSE.
# 
# Balanced α (Chunks 4-5) gives the best performance.
# 
# 

# %%
overlap_3_4 = set(selected_features_per_chunk["Chunk 3"]).intersection(selected_features_per_chunk["Chunk 4"])
overlap_4_5 = set(selected_features_per_chunk["Chunk 4"]).intersection(selected_features_per_chunk["Chunk 5"])
overlap_3_5 = set(selected_features_per_chunk["Chunk 3"]).intersection(selected_features_per_chunk["Chunk 5"])

print(f"Overlap between Chunks 3 & 4: {len(overlap_3_4)} features")
print(f"Overlap between Chunks 4 & 5: {len(overlap_4_5)} features")
print(f"Overlap between Chunks 3 & 5: {len(overlap_3_5)} features")

print(overlap_3_4)
print(overlap_4_5)
print(overlap_3_5)


# %% [markdown]
# ---

# %% [markdown]
# ### Logistic Regression

# %%
# Prepare feature matrix X and target variable y for classification
X = feature_data.apply(flatten_features, axis=1)
X = np.vstack(X.values)  # Convert to a 2D NumPy array
y = feature_data["Handedness_Cat"].values  # Classification target

# Split dataset into training and testing sets (without SMOTE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression with Cross-Validation
log_reg = LogisticRegressionCV(cv=5, max_iter=1000, solver='liblinear').fit(X_train_scaled, y_train)

# Predict on test set
y_pred = log_reg.predict(X_test_scaled)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=["Left-Handed", "Right-Handed"])
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)

# %% [markdown]
# ~12 correctly classified as Left-Handed, ~3 misclassified as Right-Handed
# 
# ~11 correctly classified as Right-Handed, ~4 misclassified as Left-Handed
# 
# Slight bias towards Left-Handed (higher recall means it captures more Left-Handed instances correctly).
# 
# Right-Handed recall is slightly lower, meaning that it misclassifies more Right-Handed cases.
# 

# %% [markdown]
# ### Logistic Regression w/ imbalanced learn (SMOTE)

# %%
# Prepare feature matrix X and target variable y for classification
X = feature_data.apply(flatten_features, axis=1)
X = np.vstack(X.values)
y = feature_data["Handedness_Cat"].values

# Apply SMOTE to balance classes
smote = SMOTE(k_neighbors=3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression with Cross-Validation
log_reg = LogisticRegressionCV(cv=5, max_iter=1000, solver='liblinear').fit(X_train_scaled, y_train)

# Predict on test set
y_pred = log_reg.predict(X_test_scaled)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=["Left-Handed", "Right-Handed"])
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)


# %% [markdown]
# Accuracy drops significantly (from ~75%-80% to 60%-66.67%) after applying SMOTE.
# 
# Recall for Left-Handed drops significantly (~0.80 to ~0.5).
# 
# Recall for Right-Handed changes slightly (0.73 to 0.71).
# 
# Precision for both classes decreases, indicating more false positives. 
# * More Left-Handed cases are misclassified as Right-Handed after SMOTE. 
# * Right-Handed recall remains almost unchanged (0.73 → 0.71), meaning that **SMOTE doesn't help much in capturing Right-Handed individuals.**

# %% [markdown]
# ---

# %% [markdown]
# ### CNN

# %%
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

# %%
# Reshape data for CNN input (assuming X is correlation matrices)
X = np.array(feature_data["correlation_matrix"].tolist())  # Convert list of matrices to array
X_cnn = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)  # Add channel dimension
y_cnn = feature_data["Handedness_Cat"].values  # Classification target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_cnn, y_cnn, test_size=0.2, random_state=42, stratify=y_cnn)

# Define CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (0 = Left, 1 = Right)
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16)

# Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")


# %%
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Left-Handed", "Right-Handed"]))

# %% [markdown]
# (May change if re-run)
# Many left-handed cases are misclassified (recall is only ~65%), but ~83% of left-handed predictions made is correct. 
# 
# The model correctly detects ~85% of right-handed cases, but only 60% of left-handed predictions made is correct. F1-score is better than for left-handed, but precision is relatively lower. 
# 
# **Macro & Weighted F1-score = 0.60 → model may be learning patterns from the right-handed class more effectively than the other.**

# %%
y_prob = model.predict(X_test)  # Get probability scores

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier baseline
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# %% [markdown]
# AUC of ~0.7 means the model is better than random, but still not highly reliable.

# %% [markdown]
# ### CNN w/ More Weights toward Left-Handed Class

# %%
# # Reshape data for CNN input (assuming X is correlation matrices)
# X = np.array(feature_data["correlation_matrix"].tolist())  # Convert list of matrices to array
# X_cnn = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)  # Add channel dimension
# y_cnn = feature_data["Handedness_Cat"].values  # Classification target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_cnn, y_cnn, test_size=0.2, random_state=42, stratify=y_cnn)

# Define CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (0 = Left, 1 = Right)
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Compute class weights automatically
# class_weights = dict(zip(np.unique(y_train), compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)))
class_weights = {0: 3, 1: 1.0}

# Train Model with dynamically computed class weights
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16, class_weight=class_weights)

# Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# %%
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Left-Handed", "Right-Handed"]))

# %%
y_prob = model.predict(X_test)  # Get probability scores

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier baseline
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# %% [markdown]
# ---

# %% [markdown]
# Transforming Handedness

# %%
left_handed = data[data['Handedness_Cat'] == 'Left']
right_handed = data[data['Handedness_Cat'] == 'Right']

# Handedness statistics
pd.DataFrame([
    left_handed.describe()['Handedness'], 
    right_handed.describe()['Handedness'], 
    data['Handedness'].describe()
], index=['Left-Handed', 'Right-Handed', 'Overall']).T


# %%
plt.figure(figsize=(8, 6))
sns.histplot(left_handed['Handedness'], color='blue', label='Left-Handed', kde=False, bins=20, alpha=0.5)
sns.histplot(right_handed['Handedness'], color='orange', label='Right-Handed', kde=False, bins=20, alpha=0.5)


# mark medians
plt.axvline(left_handed['Handedness'].median(), color='blue', linestyle='--', label='Left-Handed Median')
plt.axvline(right_handed['Handedness'].median(), color='orange', linestyle='--', label='Right-Handed Median')

# Labels and title
plt.xlabel('Handedness Score', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title('Distribution of Handedness Scores', fontsize=16)
plt.legend()

plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# Log tranformation

# %%
data['log_transformed_handedness'] = np.log10(100-data['Handedness'])

left_handed = data[data['Handedness_Cat'] == 'Left']
right_handed = data[data['Handedness_Cat'] == 'Right']

plt.figure(figsize=(10, 6))
sns.histplot(data=left_handed['log_transformed_handedness'], label='Left-Handed', color='blue', kde=False, bins=20, alpha=0.5, stat='density')
sns.histplot(data=right_handed['log_transformed_handedness'], label='Right-Handed', color='orange', kde=False, bins=20, alpha=0.5, stat='density')

plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# square root transformation

# %%
data['sqrt_transformed_handedness'] = data['Handedness'].abs()**0.5

left_handed = data[data['Handedness_Cat'] == 'Left']
right_handed = data[data['Handedness_Cat'] == 'Right']

plt.figure(figsize=(10, 6))
sns.histplot(data=left_handed['sqrt_transformed_handedness'], label='Left-Handed', color='blue', kde=False, bins=20, alpha=0.5, stat='density')
sns.histplot(data=right_handed['sqrt_transformed_handedness'], label='Right-Handed', color='orange', kde=False, bins=20, alpha=0.5, stat='density')

plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%
from scipy.stats import zscore

data['zscored_handedness'] = zscore(data['Handedness'])

left_handed = data[data['Handedness_Cat'] == 'Left']
right_handed = data[data['Handedness_Cat'] == 'Right']

sns.histplot(left_handed['zscored_handedness'],label='Left-Handed',bins=10, edgecolor='black', alpha=0.7, stat='density')
sns.histplot(right_handed['zscored_handedness'], label='right-Handed',bins=10, edgecolor='black', alpha=0.7, stat='density')

plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%
mean_value = left_handed['Handedness'].mean()
std_value = left_handed['Handedness'].std()
left_handed['standardized_handedness'] = (left_handed['Handedness'] - mean_value) / std_value

mean_value = right_handed['Handedness'].mean()
std_value = right_handed['Handedness'].std()
right_handed['standardized_handedness'] = (right_handed['Handedness'] - mean_value) / std_value

plt.figure(figsize=(10, 6))
sns.histplot(left_handed['standardized_handedness'],label='Left-Handed',bins=10, edgecolor='black', alpha=0.7, stat='density')
sns.histplot(right_handed['standardized_handedness'], label='right-Handed',bins=10, edgecolor='black', alpha=0.7, stat='density')

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.show()

# %% [markdown]
# Symmetric Transformation (Box-Cox)

# %%
from scipy.stats import boxcox

data['BoxCox_Handedness'], lambda_ = boxcox(data['Handedness'].abs() + 100)  # Shift to ensure all values are positive
data['BoxCox_Handedness']

# %%
left_handed = data[data['Handedness_Cat'] == 'Left']
right_handed = data[data['Handedness_Cat'] == 'Right']

sns.histplot(left_handed['Handedness_BoxCox'],label='Left-Handed',bins=10, edgecolor='black', alpha=0.7, stat='density')
sns.histplot(right_handed['Handedness_BoxCox'], label='right-Handed',bins=10, edgecolor='black', alpha=0.7, stat='density')

plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%
# from scipy.stats import skew, kurtosis

# print("Original Skewness:", skew(data['Handedness']))
# print("Transformed Skewness:", skew(data['log_transformed_handedness']))

# print("Original Kurtosis:", kurtosis(data['Handedness']))
# print("Transformed Kurtosis:", kurtosis(data['log_transformed_handedness']))


# %%



