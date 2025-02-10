import os
import numpy as np
from nilearn import plotting, image
from atlasreader.atlasreader import read_atlas_peak
import pandas as pd

#getting all of the subjects
def get_subjects(sub_list):
    '''
 
    '''
    sub_id = [int(i.split(".")[0])for i in sub_list]
    return sub_id

#loading the data
def load_txt(n=100, username='anmarkova', file_path=None):
    '''
    loads txt data of all subjects

    params:
    n (int): number of brain regions options: 15, 50, 100; 100 by default
    subset (tuple): gets a subset of subjects; (start, stop, step) ... eventually 
    file_path: your own filepath

    return:
    all data (np.array): 
    '''
    if file_path:
        file_path = file_path
    else:
        file_path = f"/home/{username}/teams/a05/group_2/HCP_PTN1200/node_timeseries/3T_HCP1200_MSMAll_d{n}_ts2"
    sub_list = os.listdir(file_path)
    sub_id = get_subjects(sub_list)
    make_path = lambda x: file_path + "/" + x
    paths = [make_path(i) for i in sub_list]
    sub_data = [np.loadtxt(path) for path in paths]

    print('Subject Data Loaded')

    return np.array(sub_data), sub_id

#loading atlas parcelations
def load_atlas(n=100, username='anmarkova', file_path=None):
    '''
    loads txt data of all subjects

    params:
    n (int): number of brain regions options: 15, 50, 100; 100 by default
    file_path: your own filepath

    return:
    all data (np.array): 
    '''
    if not file_path:
        file_path = f"/home/{username}/teams/a05/group_2/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d{n}.ica/melodic_IC_sum.nii.gz"
    atlas = image.load_img(f"/home/{username}/teams/a05/group_2/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d{n}.ica/melodic_IC_sum.nii.gz")
    atlas = image.threshold_img(atlas, "99.5%") 

    print("atlas has shape", ["x", "y", "z", "region"], "=", atlas.shape)
    return atlas

#load unrestricted data
def load_unrestricted(file_path=None, username='anmarkova'):

    if not file_path:
        file_path = f"/home/{username}/teams/a05/group_2/unrestricted_data.csv"
    data = pd.read_csv(file_path)
    print('unrestricted_data loaded')
    return data

#load restricted data
def load_restricted(file_path=None, username='anmarkova':

    if not file_path:
        file_path = f"/home/{username}/teams/a05/group_2/RESTRICTED_BEHAVIORAL_DATA.csv"
    data = pd.read_csv(file_path)
    print('restricted_data loaded')
    return data

#getting handedness
def get_handedness(sub_id, n=100, file_path_n=None, file_path_restricted = None):
    df = load_restricted(file_path_restricted)
    handedness = df.set_index("Subject").loc[sub_id][["Handedness"]]
    return handedness

def get_atlas_coords(atlas):
    #plotting connections on the brain
    atlas_coords = plotting.find_probabilistic_atlas_cut_coords(atlas)
    atlas_coords = np.array(atlas_coords)
    return atlas_coords

# #loading labels
def get_labels(atlas_coords):
    brain_region = []
    for atlas_coord in atlas_coords:
        region = read_atlas_peak("harvard_oxford", atlas_coord)
        print(region)
        brain_region += [region]  
    #select the correct region with largest probability given from the library
    brain_region = [
        max(inner_list, key=lambda x: x[0])[-1] if inner_list else None for inner_list in brain_region
        ]
    return np.array(brain_region)