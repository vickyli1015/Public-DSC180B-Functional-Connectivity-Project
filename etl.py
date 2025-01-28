from nilearn import image, plotting
from atlasreader.atlasreader import read_atlas_peak
import numpy as np
from pathlib import Path
import os

def load_data():
    folder = 'HCP_PTN1200/node_timeseries/3T_HCP1200_MSMAll_d100_ts2'
    # Get a list of fMRI data for all 1003 subjects
    file_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')]
    all_data = [np.loadtxt(path) for path in file_paths]
    all_data = np.array(all_data)
    return all_data

def load_brain_label():
    atlas = image.threshold_img("nregions-100_hcp.nii.gz", "99.5%") 
    atlas_coords = plotting.find_probabilistic_atlas_cut_coords(atlas)
    brain_region = []
    for atlas_coord in atlas_coords:
        region = read_atlas_peak("neuromorphometrics", atlas_coord)
        brain_region += [region]
    return brain_region

def load_brain_img():
    atlas = image.load_img("HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d100.ica/melodic_IC_sum.nii.gz")
    atlas = image.threshold_img(atlas, "99.5%") 
    return atlas