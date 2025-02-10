#preprocessing
import numpy as np
import random

#plotting
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiMasker

#connectivity measures
import nilearn.connectome as nic

from load_data.py import * 

sub_data, sub_ids = load_txt()
handedness = get_handedness(sub_ids)

atlas = load_atlas()
atlas_coords = get_atlas_coords(atlas)

#Correlation matrix
def corr_matrix(sub_data):
    correlation_measure = nic.ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform(sub_data)
    return correlation_matrix, correlation_measure

correlation_matrix, correlation_measure = corr_matrix(sub_data)

fig, ax = plt.subplots(figsize=(7, 7),
                       layout='constrained')
corr_plt = plotting.plot_matrix(
    correlation_measure.mean_, labels=range(0,100), colorbar=True, vmax=1, vmin=-1, figure=fig
)

corr_plt.savefig('corr_m.png')

display = plotting.plot_connectome(
    correlation_measure.mean_,
    atlas_coords,
    title="mean partial correlation over all subjects",
    edge_threshold="98%", colorbar=True
)

display.savefig('corr_m.png')

#partial correlations
def partial_corr(sub_data):
    p_correlation_measure = nic.ConnectivityMeasure(kind='partial correlation')
    p_correlation_matrix = p_correlation_measure.fit_transform(sub_data)
    return p_correlation_matrix, p_correlation_measure

p_correlation_matrix, p_correlation_measure =  partial_corr(sub_data)

fig, ax = plt.subplots(figsize=(7, 7),
                       layout='constrained')
partial_corr = plotting.plot_matrix(
    p_correlation_matrix[1], labels=range(0,100), colorbar=True, vmax=1, vmin=-1, figure=fig
)

partial_corr.savefig('partial_corr.png')

def fisher_z_transform(correlation_matrix):
    corr_matrix = correlation_matrix.copy()
    corr_matrix[corr_matrix == 1.] = 0.999
    return np.arctanh(corr_matrix)
    