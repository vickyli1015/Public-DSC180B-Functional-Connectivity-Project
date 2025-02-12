# %%
# basic
import pandas as pd   
import numpy as np 
import os

# visualization
import matplotlib.pyplot as plt  
import seaborn as sns             

# statistical analysis
import scipy.stats as stats  
from scipy.stats import shapiro # normality check
from scipy.stats import mannwhitneyu # non-parametric test for un-normal distribution
from scipy.stats import ks_2samp # Kolmogorov-Smirnov (K-S) Test:

# %% [markdown]
# ### Load Data

# %% [markdown]
# Merged subject behavioral data from open access dataset and restricted datasets.

# %%
open_access_data = pd.read_csv("Behavioral Data/Behavioral_Data.csv")
open_access_data.head()

# %%
restricted_data = pd.read_csv("Behavioral Data/RESTRICTED_BEHAVIORAL_DATA.csv")
restricted_data.head()

# %%
subject_data = open_access_data.merge(restricted_data, how = 'inner', on = 'Subject')
subject_data.head()

# %%
subject_data.shape

# %% [markdown]
# Also merged with the brain functional activity data.

# %%
folder = 'HCP_PTN1200/node_timeseries/3T_HCP1200_MSMAll_d15_ts2'
#looping through the folder
brain_files = [f for f in os.listdir(folder) if f.endswith('.txt')] # one per subject
brain_data = {}

for filename in brain_files:
    subject_id = int(filename[:6])
    file_path = os.path.join(folder, filename)
    subject_brain_data = np.loadtxt(file_path)
    brain_data[subject_id] = subject_brain_data

# brain_data: a dictionary with 1003 keys, one for each subject. The values are the brain data, each is an array of shape (4800 timepoints, 15 regions)

# %%
len(brain_data), brain_data[237334].shape, brain_data.keys()

# %%
brain_data[237334][4799]

# %%
brain_data_df = pd.DataFrame({
    'Subject': list(brain_data.keys()),
    'Brain_Data': list(brain_data.values())  # (4800, 15) arrays
})
brain_data_df

# %%
data = subject_data.merge(brain_data_df, on='Subject', how='inner')
data.head(3)

# %%
data.shape

# %%


# %% [markdown]
# ### EDA

# %% [markdown]
# **Handedness**: The Edinburgh Handedness questionnaire is used to calculate handedness scores based on self-reported preferences for performing daily tasks. Participants indicate which hand they usually prefer for tasks such as writing, throwing, using scissors, etc.
# 
# **Handedness Score** = [(Number of right-hand tasks - Number of left-hand tasks) / Total number of tasks] × 100
# 
# 
# Negative scores indicate left-hand dominance, while positive scores indicate right-hand dominance. Range from -100 to +100

# %% [markdown]
# only 8.7% reported as left_handed, and 90% self reports as right-handed

# %%
left_handed = data[data['Handedness'] < 0]
right_handed = data[data['Handedness'] > 0]

# Handedness statistics
pd.DataFrame([
    left_handed.describe()['Handedness'], 
    right_handed.describe()['Handedness'], 
    data['Handedness'].describe()
], index=['Left-Handed', 'Right-Handed', 'Overall']).T

# %% [markdown]
# Plot the overall distribution of Handedness

# %%
plt.hist(data['Handedness'], bins=30, density=True, alpha=0.7)
plt.title("Histogram of Data")
plt.show()

# %% [markdown]
# Plot the distribution of Handedness for left and right-handed subjects

# %%
# Take absolute values of Handedness
left_handed_abs = left_handed['Handedness'].abs()
right_handed_abs = right_handed['Handedness'].abs()

plt.figure(figsize=(8, 6))

sns.kdeplot(left_handed_abs, color='blue', label='Left-Handed (Absolute)', fill=True, alpha=0.5, cut=0)
sns.kdeplot(right_handed_abs, color='orange', label='Right-Handed (Absolute)', fill=True, alpha=0.5, cut=0)


# mark medians
plt.axvline(left_handed_abs.median(), color='blue', linestyle='--', label='Left-Handed Median')
plt.axvline(right_handed_abs.median(), color='orange', linestyle='--', label='Right-Handed Median')

# Labels and title
plt.xlabel('Absolute Handedness Score', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Density Plot of Absolute Handedness Scores', fontsize=16)
plt.legend()

plt.grid(alpha=0.3)
plt.show()

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
left_handed["Handedness"]

# %% [markdown]
# Q-Q Plot (Quantile-Quantile)
# 
# Left handed data seemed to be more normally distributed than the right handed. The data is likely normal if the points follow a straight line.

# %%
stats.probplot(left_handed_abs, dist="norm", plot=plt)
plt.title("Q-Q Plot (left-handed)")
plt.show()

# %% [markdown]
# central part of the distribution seemed normally distributed, but the overall distribution is not normal due to left-skewed tails

# %%
stats.probplot(right_handed_abs, dist="norm", plot=plt)
plt.title("Q-Q Plot (right-handed)")
plt.show()

# %% [markdown]
# ### Check for Distribution Normality
# 
# After normality test, we found out that both of left and right-handed distributions are not normal.

# %%
# Test normality for left-handed (absolute values)
stat_left, p_left = shapiro(left_handed_abs)
print(f"Left-Handed: Stat={stat_left}, P-value={p_left}")

# Test normality for right-handed (absolute values)
stat_right, p_right = shapiro(right_handed_abs)
print(f"Right-Handed: Stat={stat_right}, P-value={p_right}")


# %% [markdown]
# Many parametric hypothesis tests such as t-tests rely on the data's distribution being normal. However as we saw, the distributiosn do not follow a normal distribution, so we can only proceed with non-parametric tests such as the Mann-Whitney U Test for comparing the medians.

# %% [markdown]
# ### Hypothesis Test On the Medians of the Distributions of Left/Right handed
# 
# Mann-Whitney U Test compares medians of distributions and does not require normality of data.

# %%
stat, p_value = mannwhitneyu(
    left_handed_abs, 
    right_handed_abs, 
    alternative='two-sided'
)

print("P value:", p_value)


# %% [markdown]
# ### Hypothesis Test on CDFs of the Distributions

# %%
stat, p_value = ks_2samp(left_handed_abs, right_handed_abs)
print(f"K-S Test Statistic: {stat}, P-value: {p_value}")


# %% [markdown]
# Showed significant difference in both the medians and CDFs between the handedness groups.

# %% [markdown]
# #### Week 2's Discussion Takeaway
# 
# Regression: Keep all data
# 
# Classification: try different thresholds to define left and right handedness. After that need to have equal classes by random sampling.

# %% [markdown]
# ## Classification:
# 
# 1. Try -/+25% (As of one of the papers did)
# 2. Try -/+40% 

# %%
# Define thresholds
threshold_25 = 25
threshold_40 = 40

# Create classification labels for ±25% threshold
data_25 = data[(data['Handedness'] <= -threshold_25) | (data['Handedness'] >= threshold_25)].copy()
data_25['Class'] = np.where(data_25['Handedness'] >= threshold_25, 1, 0)  # 0: Left-Handed, 1: Right-Handed

# Create classification labels for ±40% threshold
data_40 = data[(data['Handedness'] <= -threshold_40) | (data['Handedness'] >= threshold_40)].copy()
data_40['Class'] = np.where(data_40['Handedness'] >= threshold_40, 1, 0)  # 0: Left-Handed, 1: Right-Handed

data_25['Class'].value_counts(), data_40['Class'].value_counts()


# %%
# Balance the classes by random sampling
def balance_classes(df, class_column):
    class_0 = df[df[class_column] == 0]
    class_1 = df[df[class_column] == 1]
    
    # Randomly sample the majority class to match the minority class size
    if len(class_0) > len(class_1):
        class_0 = class_0.sample(len(class_1), random_state=42)
    else:
        class_1 = class_1.sample(len(class_0), random_state=42)
    
    return pd.concat([class_0, class_1])

# Apply balancing for ±25% threshold data
balanced_data_25 = balance_classes(data_25, 'Class')

# Apply balancing for ±40% threshold data
balanced_data_40 = balance_classes(data_40, 'Class')

# %%
balanced_data_25.shape, balanced_data_40.shape

# %%
balanced_data_25.describe()

# %%
balanced_data_40.describe()

# %%
# Handedness statistics
pd.DataFrame([
    balanced_data_25.describe()['Handedness'], 
    balanced_data_40.describe()['Handedness'], 
    data['Handedness'].describe()
], index=['25%', '40%', 'Overall']).T

# %% [markdown]
# ### Plot distributions of handedness and other (demographic) features (as brain features TBD for now)

# %%
def handedness_distribution(df, title):
    plt.figure(figsize=(10, 6))
    
    # Plot histogram for each group
    sns.histplot(data=df[df['Class'] == 0]['Handedness'].abs(), label='Left-Handed', color='blue', kde=False, bins=20, alpha=0.5, stat='density')
    sns.histplot(data=df[df['Class'] == 1]['Handedness'], label='Right-Handed', color='orange', kde=False, bins=20, alpha=0.5, stat='density')
    
    
    plt.title(title, fontsize=16)
    plt.xlabel('Handedness Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


handedness_distribution(balanced_data_25, "Handedness Score Distribution (+/-25% Threshold)")
handedness_distribution(balanced_data_40, "Handedness Score Distribution (+/-40% Threshold)")

# %%
import seaborn as sns
import matplotlib.pyplot as plt

def continuous_features_w_handedness(df, feature, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    # Plot histogram for each handedness group
    sns.histplot(data=df[df['Class'] == 0][feature], label='Left-Handed', color='blue', kde=False, bins=20, alpha=0.5, ax=ax)
    sns.histplot(data=df[df['Class'] == 1][feature], label='Right-Handed', color='orange', kde=False, bins=20, alpha=0.5, ax=ax)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)


# %%
# physical health feature
features = ['Age_in_Yrs', 'Height', 'Weight','BMI', 'HbA1C', 'PSQI_Score', 'PSQI_Min2Asleep', 'PSQI_AmtSleep', "Endurance_Unadj", "Endurance_AgeAdj", "GaitSpeed_Comp",
    "Dexterity_Unadj", "Dexterity_AgeAdj", "Strength_Unadj", "Strength_AgeAdj"]

# PSQI_Score: sleep quality; higher score, more acute sleep disturbances. cut off score of 5 to identify possible patients
# HbA1C: # normal: 4.5%-6%, higher than normal indicates risk of diabetes

fig, axes = plt.subplots(len(features), 2, figsize=(12, len(features) * 4))

for i, feature in enumerate(features):
    # Left subplot for balanced_data_25
    continuous_features_w_handedness(balanced_data_25, feature, feature + ' Distribution (25%)', ax=axes[i, 0])
    
    # Right subplot for balanced_data_40
    continuous_features_w_handedness(balanced_data_40, feature, feature + ' Distribution (40%)', ax=axes[i, 1])

plt.tight_layout()
plt.show()

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
psych_features = [
    "ER40_CR", "ER40_CRT",
    "AngAffect_Unadj", "AngHostil_Unadj", "AngAggr_Unadj", "FearAffect_Unadj",
    "FearSomat_Unadj", "Sadness_Unadj", "LifeSatisf_Unadj", "MeanPurp_Unadj",
    "PosAffect_Unadj", "Friendship_Unadj", "Loneliness_Unadj", "PercHostil_Unadj",
    "PercReject_Unadj", "EmotSupp_Unadj", "InstruSupp_Unadj", "PercStress_Unadj",
    "SelfEff_Unadj"
]

fig, axes = plt.subplots(nrows=len(psych_features), ncols=2, figsize=(12, len(psych_features) * 4), squeeze=False)

for i, feature in enumerate(psych_features):
    # Left subplot for balanced_data_25
    continuous_features_w_handedness(balanced_data_25, feature, feature + ' Distribution (25%)', ax=axes[i, 0])
    
    # Right subplot for balanced_data_40
    continuous_features_w_handedness(balanced_data_40, feature, feature + ' Distribution (40%)', ax=axes[i, 1])

plt.tight_layout()
plt.show()

# %% [markdown]
# ![image.png](attachment:image-2.png)

# %% [markdown]
# Distribution of categorical variables across handedness groups

# %%
def categorical_features_w_handedness(df, feature, title, ax=None):
    counts = df.groupby(['Class', feature]).size().reset_index(name='Count')

    if ax is None:
        fig, ax = plt.subplots()

    sns.barplot(data=counts, x=feature, y='Count', hue='Class', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')

    
    legend_labels = {'0': 'Left-Handed', '1': 'Right-Handed'}
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [legend_labels[label] for label in labels], title="Handedness", loc='upper right')


# %%
import matplotlib.pyplot as plt

features = ['Gender', 'SSAGA_Income', 'SSAGA_Educ', 
            "ER40ANG", "ER40FEAR","ER40HAP", "ER40NOE", "ER40SAD",
            'PSQI_Comp1', 'PSQI_Comp2', 
            'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7',
            "PSQI_Latency30Min", "PSQI_WakeUp", "PSQI_Bathroom", "PSQI_Breathe",
            "PSQI_Snore", "PSQI_TooCold", "PSQI_TooHot", "PSQI_BadDream",
            "PSQI_Pain", "PSQI_Other", "PSQI_Quality", "PSQI_SleepMeds",
            "PSQI_DayStayAwake", "PSQI_DayEnthusiasm"]

fig, axes = plt.subplots(len(features), 2, figsize=(12, len(features) * 4))

for i, feature in enumerate(features):
    # left subplot: balanced_data_25
    categorical_features_w_handedness(balanced_data_25, feature, feature + ' Distribution (25%)', ax=axes[i, 0])
    
    # right subplot: balanced_data_40
    categorical_features_w_handedness(balanced_data_40, feature, feature + ' Distribution (40%)', ax=axes[i, 1])

plt.tight_layout()
plt.show()


# %% [markdown]
# ![image.png](attachment:image.png)
# ![image.png](attachment:image-2.png)

# %% [markdown]
# 

# %%
balanced_data_25

# %%



