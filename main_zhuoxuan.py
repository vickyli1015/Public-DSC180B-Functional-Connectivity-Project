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

import hcp_utils as hcp


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, Concatenate
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler


def load_data():
    # This is the function that ask the user to input the path of the data 
    # and return two results: [data, brain_region]

    open_access_path = input("your path of unrestricted data")
    # reading the unrestricted data
    open_access_data = pd.read_csv(open_access_path)
    restricted_access_path = input("your path of restricted data")
    # reading the restricted data
    restricted_data = pd.read_csv(restricted_access_path)
    # merge the data
    subject_data = open_access_data.merge(restricted_data, how = 'inner', on = 'Subject')
    # loading the paracellation data
    file_path = input("your path of parcellations data")
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
    # construct dataframe
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
    mmp_data["Handedness_Cat"] = mmp_data["Handedness"].apply(lambda x: "Left" if x < 0 else "Right")
    # getting the brain region
    brain_region = list(hcp.mmp.labels.values())
    return mmp_data, brain_region

def correlation_matrix(data):
    # This function inputs a dataset and return a that dataset with a new correlation matrix column
    correlation_matrix_list = []
    for i in range(data.shape[0]):
        person = data["Brain_Data"][i]
        #get matrix
        person_matrix = np.corrcoef(person.T)
        #append to list
        correlation_matrix_list.append(person_matrix)
    #append to column
    data["correlation_matrix"] = correlation_matrix_list
    return data

def feature_row_get(target, data):
    # This function is for feature getting
    target_index = data.index(target)
    target_row = data['correlation_matrix'].apply(lambda x: x[target_index])
    return target_row

def feature_creating(data):
    # This function is for select the targeted region and return the feature data
    semi_MTG = feature_row_get("L_MT")

    semi_ACC_1 = feature_row_get("L_a24")
    semi_ACC_2 = feature_row_get("L_p24")
    semi_ACC_3 = feature_row_get("L_24dd")
    semi_ACC_4 = feature_row_get("L_24dv")
    semi_ACC_5 = feature_row_get("L_a24pr")
    semi_ACC_6 = feature_row_get("L_p24pr")
    semi_ACC_7 = feature_row_get("L_d32")
    semi_ACC_8 = feature_row_get("L_a32pr")
    semi_ACC_9 = feature_row_get("L_25")
    semi_ACC_10 = feature_row_get("L_33pr")

    semi_MCC_1 = feature_row_get("L_24dd")
    semi_MCC_2 = feature_row_get("L_24dv")
    semi_MCC_3 = feature_row_get("L_23c")
    semi_MCC_4 = feature_row_get("L_23d")

    semi_ANG_1 = feature_row_get("R_PGp")
    semi_ANG_2 = feature_row_get("R_PGi")
    semi_ANG_3 = feature_row_get("R_PGs")

    semi_AMYG = feature_row_get("amygdala_right")
    new_1 = feature_row_get("L_45")
    new_2 = feature_row_get("L_10d")
    new_3 = feature_row_get("L_10v")
    new_4 = feature_row_get("L_10r")
    new_5 = feature_row_get("L_a10p")
    new_6 = feature_row_get("L_10pp")

    new_7 = feature_row_get("L_PF")
    new_8 = feature_row_get("L_PFt")
    new_9 = feature_row_get("L_PFop")
    new_10 = feature_row_get("L_PFm")


    new_11 = feature_row_get("L_3b")
    new_12 = feature_row_get("L_1")
    new_13 = feature_row_get("L_2")


    new_14 = feature_row_get("R_3b")
    new_15 = feature_row_get("R_1")
    new_16 = feature_row_get("R_2")

    new_17 = feature_row_get("R_6ma")
    new_18 = feature_row_get("R_6mp")
    new_19 = feature_row_get("R_SCEF")

    feature_data = data[["Handedness", "Handedness_Cat", "correlation_matrix"]]
    feature_data['matrix_mean'] = data['correlation_matrix'].apply(lambda x: np.mean(x))
    feature_data["semi_MTG"] = semi_MTG

    feature_data["semi_ACC_1"] = semi_ACC_1
    feature_data["semi_ACC_2"] = semi_ACC_2
    feature_data["semi_ACC_3"] = semi_ACC_3
    feature_data["semi_ACC_4"] = semi_ACC_4
    feature_data["semi_ACC_5"] = semi_ACC_5
    feature_data["semi_ACC_6"] = semi_ACC_6
    feature_data["semi_ACC_7"] = semi_ACC_7
    feature_data["semi_ACC_8"] = semi_ACC_8
    feature_data["semi_ACC_9"] = semi_ACC_9
    feature_data["semi_ACC_10"] = semi_ACC_10

    feature_data["semi_MCC_1"] = semi_MCC_1
    feature_data["semi_MCC_2"] = semi_MCC_2
    feature_data["semi_MCC_3"] = semi_MCC_3
    feature_data["semi_MCC_4"] = semi_MCC_4




    feature_data["semi_ANG_1"] = semi_ANG_1
    feature_data["semi_ANG_2"] = semi_ANG_2
    feature_data["semi_ANG_3"] = semi_ANG_3

    feature_data["semi_AMYG"] = semi_AMYG


    feature_data["new_1"] = new_1
    feature_data["new_2"] = new_2
    feature_data["new_3"] = new_3
    feature_data["new_4"] = new_4
    feature_data["new_5"] = new_5
    feature_data["new_6"] = new_6
    feature_data["new_7"] = new_7
    feature_data["new_8"] = new_8
    feature_data["new_9"] = new_9
    feature_data["new_10"] = new_10
    feature_data["new_11"] = new_11
    feature_data["new_12"] = new_12
    feature_data["new_13"] = new_13
    feature_data["new_14"] = new_14
    feature_data["new_15"] = new_15
    feature_data["new_16"] = new_16
    feature_data["new_17"] = new_17
    feature_data["new_18"] = new_18
    feature_data["new_19"] = new_19
    feature_data_all = feature_data.copy()
    feature_data = feature_data[(feature_data['Handedness'] > 25) | (feature_data['Handedness'] < -25)]

    return feature_data, feature_data_all

def flatten_features(row):
    
    # Given a single row of the DataFrame,
    # return a 1D numpy array of all features to be used by the model.
    # Example: Flatten correlation_matrix (if it's a 2D 3x3 matrix, that becomes 9 values)
    corr_mat = np.array(row["correlation_matrix"]).flatten()
    
    # Flatten each 'semi_*' array (each is length 3 in your example)
    semi_MTG = np.array(row["semi_MTG"])

    semi_ACC_1 = np.array(row["semi_ACC_1"])
    semi_ACC_2 = np.array(row["semi_ACC_2"])
    semi_ACC_3 = np.array(row["semi_ACC_3"])
    semi_ACC_4 = np.array(row["semi_ACC_4"])
    semi_ACC_5 = np.array(row["semi_ACC_5"])
    semi_ACC_6 = np.array(row["semi_ACC_6"])
    semi_ACC_7 = np.array(row["semi_ACC_7"])
    semi_ACC_8 = np.array(row["semi_ACC_8"])
    semi_ACC_9 = np.array(row["semi_ACC_9"])
    semi_ACC_10 = np.array(row["semi_ACC_10"])

    semi_MCC_1 = np.array(row["semi_MCC_1"])
    semi_MCC_2 = np.array(row["semi_MCC_2"])
    semi_MCC_3 = np.array(row["semi_MCC_3"])
    semi_MCC_4 = np.array(row["semi_MCC_4"])


    semi_ANG_1 = np.array(row["semi_ANG_1"])
    semi_ANG_2 = np.array(row["semi_ANG_2"])
    semi_ANG_3 = np.array(row["semi_ANG_3"])

    semi_AMYG = np.array(row["semi_AMYG"])
    

    new_1 = np.array(row["new_1"])
    new_2 = np.array(row["new_2"])
    new_3 = np.array(row["new_3"])
    new_4 = np.array(row["new_4"])
    new_5 = np.array(row["new_5"])
    new_6 = np.array(row["new_6"])
    new_7 = np.array(row["new_7"])
    new_8 = np.array(row["new_8"])
    new_9 = np.array(row["new_9"])
    new_10 = np.array(row["new_10"])
    new_11 = np.array(row["new_11"])
    new_12 = np.array(row["new_12"])
    new_13 = np.array(row["new_13"])
    new_14 = np.array(row["new_14"])
    new_15 = np.array(row["new_15"])
    new_16 = np.array(row["new_16"])
    new_17 = np.array(row["new_17"])
    new_18 = np.array(row["new_18"])
    new_19 = np.array(row["new_19"])

    
    # Concatenate them into one array
    all_features = np.concatenate([
        semi_MTG, 

        semi_ACC_1,
        semi_ACC_2,
        semi_ACC_3,
        semi_ACC_4,
        semi_ACC_5,
        semi_ACC_6,
        semi_ACC_7,
        semi_ACC_8,
        semi_ACC_9,
        semi_ACC_10,

        semi_MCC_1,
        semi_MCC_2,
        semi_MCC_3,
        semi_MCC_4,

        semi_ANG_1, 
        semi_ANG_2, 
        semi_ANG_3, 
        semi_AMYG, 
        
        new_1, 
        new_2, 
        new_3, 
        new_4, 
        new_5, 
        new_6,
        new_7,
        new_8,
        new_9,
        new_10,
        new_11,
        new_12,
        new_13,
        new_14,
        new_15,
        new_16,
        new_17,
        new_18,
        new_19
    ])
    
    return all_features

def select_sample_data(data):
    # This function input a dataset and sample balanced data.
    # Separate into left and right handed
    left_handed = data[data['Handedness'] < 0]
    right_handed = data[data['Handedness'] > 0]

    # Determine the number of samples to select from the right-handed data
    num_samples = min(len(left_handed), len(right_handed))

    # Randomly sample from the right-handed data
    right_handed_sample = right_handed.sample(n=num_samples, random_state=42)

    # Combine the left and right handed data
    final_df = pd.concat([left_handed, right_handed_sample])

    # Optionally shuffle the final dataframe to mix left and right handed samples
    final_df = final_df.sample(frac=1).reset_index(drop=True)

    feature_data = final_df
    return feature_data

def SVM_SVC(data):
    # Now apply this function to each row in the DataFrame to create your X matrix:
    X_list = data.apply(flatten_features, axis=1)
    # X_list will be a column of arrays. Convert to a 2D array:
    X = np.vstack(X_list.values)

    # y is simply the "Handedness" column:
    y = data["Handedness_Cat"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        # random_state=42, 
        test_size=0.5
    )


    model_class = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True, C=1.0))
    ])
    model_class.fit(X_train, y_train)
    y_pred = model_class.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy of SVM Model:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)



    n_test = len(y_test)
    z_value = 1.96  # for ~95% CI
    std_error = math.sqrt(accuracy * (1 - accuracy) / n_test)

    ci_lower = accuracy - z_value * std_error
    ci_upper = accuracy + z_value * std_error

    print(f"95% CI for Accuracy: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return y_pred

### CNN

def extract_features_2d(row):
    # Extract the correlation matrix as 2D,
    # and keep other features separate in 1D form.
    # The correlation matrix as an N x N 2D array:
    corr_mat_2d = np.array(row["correlation_matrix"])
    semi_MTG = np.array(row["semi_MTG"])

    semi_ACC_1 = np.array(row["semi_ACC_1"])
    semi_ACC_2 = np.array(row["semi_ACC_2"])
    semi_ACC_3 = np.array(row["semi_ACC_3"])
    semi_ACC_4 = np.array(row["semi_ACC_4"])
    semi_ACC_5 = np.array(row["semi_ACC_5"])
    semi_ACC_6 = np.array(row["semi_ACC_6"])
    semi_ACC_7 = np.array(row["semi_ACC_7"])
    semi_ACC_8 = np.array(row["semi_ACC_8"])
    semi_ACC_9 = np.array(row["semi_ACC_9"])
    semi_ACC_10 = np.array(row["semi_ACC_10"])

    semi_MCC_1 = np.array(row["semi_MCC_1"])
    semi_MCC_2 = np.array(row["semi_MCC_2"])
    semi_MCC_3 = np.array(row["semi_MCC_3"])
    semi_MCC_4 = np.array(row["semi_MCC_4"])


    semi_ANG_1 = np.array(row["semi_ANG_1"])
    semi_ANG_2 = np.array(row["semi_ANG_2"])
    semi_ANG_3 = np.array(row["semi_ANG_3"])

    semi_AMYG = np.array(row["semi_AMYG"])
    

    new_1 = np.array(row["new_1"])
    new_2 = np.array(row["new_2"])
    new_3 = np.array(row["new_3"])
    new_4 = np.array(row["new_4"])
    new_5 = np.array(row["new_5"])
    new_6 = np.array(row["new_6"])
    new_7 = np.array(row["new_7"])
    new_8 = np.array(row["new_8"])
    new_9 = np.array(row["new_9"])
    new_10 = np.array(row["new_10"])
    new_11 = np.array(row["new_11"])
    new_12 = np.array(row["new_12"])
    new_13 = np.array(row["new_13"])
    new_14 = np.array(row["new_14"])
    new_15 = np.array(row["new_15"])
    new_16 = np.array(row["new_16"])
    new_17 = np.array(row["new_17"])
    new_18 = np.array(row["new_18"])
    new_19 = np.array(row["new_19"])


    # Concatenate the 1D arrays into one big 1D vector:
    other_1d_feats = np.concatenate([
        semi_MTG, 

        semi_ACC_1,
        semi_ACC_2,
        semi_ACC_3,
        semi_ACC_4,
        semi_ACC_5,
        semi_ACC_6,
        semi_ACC_7,
        semi_ACC_8,
        semi_ACC_9,
        semi_ACC_10,

        semi_MCC_1,
        semi_MCC_2,
        semi_MCC_3,
        semi_MCC_4,

        semi_ANG_1, 
        semi_ANG_2, 
        semi_ANG_3, 
        semi_AMYG, 
        
        new_1, 
        new_2, 
        new_3, 
        new_4, 
        new_5, 
        new_6,
        new_7,
        new_8,
        new_9,
        new_10,
        new_11,
        new_12,
        new_13,
        new_14,
        new_15,
        new_16,
        new_17,
        new_18,
        new_19
    ])

    return corr_mat_2d, other_1d_feats

def CNN(data):
    corr_list = []
    other_list = []

    for idx, row in data.iterrows():
        mat2d, vec1d = extract_features_2d(row)
        corr_list.append(mat2d)
        other_list.append(vec1d)

    # Convert them to numpy arrays
    X_corr = np.array(corr_list)   # shape: (n_samples, N, N)
    X_other = np.array(other_list) # shape: (n_samples, #features)

    # For Keras Conv2D, you typically need a 4D tensor: (batch_size, height, width, channels)
    # So add a channel dimension:
    # Suppose X_corr is shape (n_samples, 379, 379, 1), but might be string-dtype
    X_corr = X_corr.astype("float32")

    # Suppose X_other is shape (n_samples, 14402), might be string-dtype
    X_other = X_other.astype("float32")

    label_encoder = LabelEncoder()
    y = data["Handedness_Cat"]
    y_numeric = label_encoder.fit_transform(y)  # e.g., 0/1

    X_corr_train, X_corr_test, X_other_train, X_other_test, y_train, y_test = train_test_split(
        X_corr, X_other, y_numeric, test_size=0.25
    )

    # First, flatten the image (correlation matrix) data.
    n_train = X_corr_train.shape[0]
    h, w, c = X_corr_train.shape[1], X_corr_train.shape[2], 1
    X_corr_train_flat = X_corr_train.reshape(n_train, -1)  # shape: (n_train, h*w)

    # Now, horizontally stack with the other features.
    X_train_combined = np.hstack([X_corr_train_flat, X_other_train])  # shape: (n_train, h*w + n_other_features)

    # Apply oversampling using RandomOverSampler.
    ros = RandomOverSampler()
    X_train_resampled_combined, y_train_resampled = ros.fit_resample(X_train_combined, y_train)

    # Split the resampled data back into the two inputs.
    num_corr_features = h * w  # number of columns corresponding to X_corr
    X_corr_train_resampled_flat = X_train_resampled_combined[:, :num_corr_features]
    X_other_train_resampled = X_train_resampled_combined[:, num_corr_features:]

    # Reshape the flattened correlation features back into their original 4D shape.
    X_corr_train_resampled = X_corr_train_resampled_flat.reshape(-1, h, w, c)


    # Build your model
    input_corr = tf.keras.layers.Input(shape=(379, 379, 1), name="corr_input")
    x1 = tf.keras.layers.Conv2D(8, (2,2), activation='relu')(input_corr)
    x1 = tf.keras.layers.Flatten()(x1)

    input_other = tf.keras.layers.Input(shape=(14402,), name="other_input")
    x2 = tf.keras.layers.Dense(16, activation='relu')(input_other)

    merged = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Dense(16, activation='relu')(merged)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[input_corr, input_other], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    # Fit
    model.fit(
        [X_corr_train_resampled, X_other_train_resampled],
        y_train_resampled,
        validation_data=([X_corr_test, X_other_test], y_test),
        epochs=10,
        batch_size=8
    )

    y_pred_proba = model.predict([X_corr_test, X_other_test])

    y_pred = (y_pred_proba > 0.5).astype(int).ravel() 

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", report)

    return y_pred

