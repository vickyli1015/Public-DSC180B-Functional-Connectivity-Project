# DSC180-Functional-Connectivity-Project

## Project Structure:

`main.ipynb` is the combination of all of our analysis and exploration of dynamic functional connectivity.

`etl.py` contains the function of loading the data, brain labels that correlate with our brain region, and brain image data.

`requirement.txt` contains the non-built-in Python libraries that are used in our project.

`Previous Explorations` is the folder that contains all the details and steps of our analysis which is sorted by the creator of the analysis.

`sliding_window_brain` is the folder that contains the interactive plots for each sliding window along with the connections in HTML form. Feel free to play with it!

`images_latex ` is the folder that contains the high-quality images in the final report.

`Behavioral Data` is the folder that contains the data we explored and prepared for the next project.



## How to get the data?

For this project, we use the resting fMRI data from the Human Connectome Project (HCP) from this [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3724347/) and the data description can be found [here](https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP1200-DenseConnectome+PTN+Appendix-July2017.pdf). 

In order to download the data you will need to make an account [here](https://db.humanconnectome.org/app/template/Login.vm;jsessionid=67A8B8766DEEA4CF0597C483C9203BE2). Then you can download timeseries data for each subject. Navigate to the section titled *WU-Minn HCP Data - 1200 Subjects* and click *open dataset*. Find section titles *HCP1200 Parcellation+Timeseries+Netmats (PTN)* and download the 1003 subject data. Note: you may need to download the IBM Aspera Launcher from [here](https://www.ibm.com/products/aspera/downloads#cds) in order to download the data. 

Once you have the data download, you will be able to find the timeseries data in the folder titled `node_timeseries` and for our analysis we are using the data in the `3T_HCP1200_MSMAll_d15_ts2`, `3T_HCP1200_MSMAll_d50_ts2`, and `3T_HCP1200_MSMAll_d100_ts2` folder which was stored originally as `NodeTimeseries_3T_HCP1200_MSMAll_ICAd100_ts2.tar.gz`. We are also using the file in the following path `HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d15.ica/melodic_IC_sum.nii.gz`, `HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d50.ica/melodic_IC_sum.nii.gz`, and `HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d100.ica/melodic_IC_sum.nii.gz` to plot our findings on the atlas of the brain.

In addition, the data we used included a part of restricted data according to HCP, and that specific part of the data requires an application [here](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms). According to the data use terms, we are not allowed to expose the data in public. Therefore, we did not include our data inside our project repo, but after the restricted data application and the data getting steps mentioned above, it will provide all the data we need in this project. 


## Which Packages are necessary?
In order to run our notebook you would need to have the following packages installed:
* jupyter notebook
* ipykernel
* numpy
* matplotlib.pyplot
* nilearn
* nibabel
* scikit-learn
* atlasreader
* pathlib
* plotly


<!-- Task for next week:
* try analysis on different d resolutions
* try combining the subjects by averaging over the time series
* try combingng the subjects by averaging over the datapoints across subjects??? (Not sure abt this one)
* read the research paper

Q1 project:
* map brain connectivity -->
