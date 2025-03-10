# DSC180-Functional-Connectivity-Project

## Project Structure:

We all organized our analysis notebooks inside the folder under our name.

`load_data.py` contains the function of loading the data, brain labels that correlate with our brain region, and brain image data.

`feature-extraction.py` contains the method and process that we used to get our features.

`run.py` contains all of our models and when run on a specific data will print out the model results.

`environment.yml` contains the non-built-in Python libraries that are used in our project.

`EDA.ipynb` contains our exploratory work, where you can get familiar with the data and explore on your own.

`project-results` is the folder that contains our notebooks with our code as well as the results of us running that code.

`scratchpad` is the folder that contains all the details and steps of our analysis which is sorted by the creator of the analysis.

`sliding_window_brain` is the folder that contains the interactive plots for each sliding window along with the connections in HTML form. Feel free to play with it!

`images_latex` is the folder that contains the high-quality images in the final report.

`Behavioral Data` is the folder that contains the data we explored and prepared for the next project.

`data` is the folder that contains the data that we used which can be shown to others.



## How to get the data?

For this project, we use the resting fMRI data from the Human Connectome Project (HCP) from this [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3724347/) and the data description can be found [here](https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP1200-DenseConnectome+PTN+Appendix-July2017.pdf). 

In order to download the data you will need to make an account [here](https://db.humanconnectome.org/app/template/Login.vm;jsessionid=67A8B8766DEEA4CF0597C483C9203BE2). Then you can download timeseries data for each subject. Navigate to the section titled *WU-Minn HCP Data - 1200 Subjects* and click *open dataset*. Find section titles *HCP1200 Parcellation+Timeseries+Netmats (PTN)* and download the 1003 subject data. Note: you may need to download the IBM Aspera Launcher from [here](https://www.ibm.com/products/aspera/downloads#cds) in order to download the data. 

Once you have the data download, you will be able to find the timeseries data in the folder titled `node_timeseries` and for our analysis we are using the data in the `3T_HCP1200_MSMAll_d15_ts2`, `3T_HCP1200_MSMAll_d50_ts2`, and `3T_HCP1200_MSMAll_d100_ts2` folder which was stored originally as `NodeTimeseries_3T_HCP1200_MSMAll_ICAd100_ts2.tar.gz`. We are also using the file in the following path `HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d15.ica/melodic_IC_sum.nii.gz`, `HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d50.ica/melodic_IC_sum.nii.gz`, and `HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d100.ica/melodic_IC_sum.nii.gz` to plot our findings on the atlas of the brain.

In addition, the data we used included a part of restricted data according to HCP, and that specific part of the data requires an application [here](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms). According to the data use terms, we are not allowed to expose the data in public. Therefore, we did not include our data inside our project repo, but after the restricted data application and the data getting steps mentioned above, it will provide all the data we need in this project. 


The second part of the data we are using is the time series for two different parcellations (2.58 GB total), and they are all separated into two hemispheres of the brain. The files labeled '*_parcellation-mmp_*' contain the data that divides each hemisphere into 180 regions, which we mentioned as mmp data. The files labeled '*_parcellation-ca_*' contain the data that divides each hemisphere into around 350 regions, which we mentioned as CA data.

Since the dataset is 2.58 GB in total, we can not include it in the repo but can only download from [this link](https://rdl-share.ucsd.edu/message/1lqVOVDRvfs5aSMMuSfgYF) which is provided by our TA Gabriel Riegner.


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

# Setting Up the Conda Environment

To create a Conda environment using the dependencies listed in `requirements.txt`, follow these steps:

```sh
conda env create --name envname --file=environments.yml
```

