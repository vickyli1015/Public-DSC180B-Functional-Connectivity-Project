import os
import numpy as np
import pandas as pd
from nilearn import image, plotting
from atlasreader.atlasreader import read_atlas_peak


def data_pre(
    open_access_csv,
    restricted_csv,
    d = "d100",
    output_all_data_csv="data.csv",
    output_brain_region_csv="brain_region.csv",
    atlas_name="harvard_oxford"
):
   
    """
    Result:
    Creates two CSV files:
      1. 'data.csv' (by default), containing the merged subject data
         and corresponding brain data.
      2. 'brain_region.csv' (by default), containing the extracted region labels
         from the specified atlas.

    Parameters
    ----------
    open_access_csv : str
        Path to the open-access CSV file.
    restricted_csv : str
        Path to the restricted CSV file.
    d : str 
        Targeted dimension.
    output_all_data_csv : str
        File name/path for saving the full merged DataFrame.
    output_brain_region_csv : str
        File name/path for saving the list of brain regions.
    atlas_name : str
        Name of the reference atlas to use with `read_atlas_peak`.

    Returns
    -------
    None
    """
    folder = f"HCP_PTN1200/node_timeseries/3T_HCP1200_MSMAll_{d}_ts2"
    group_ica_nifti = f"HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_{d}.ica/melodic_IC_sum.nii.gz"
    # -------------------------------------------------------------------------
    # 1. Threshold the atlas NIfTI
    print("Thresholding the atlas...")
    atlas = image.threshold_img(group_ica_nifti, "99.5%")

    # -------------------------------------------------------------------------
    # 2. Extract the atlas coordinates
    print("Extracting probabilistic atlas coordinates...")
    atlas_coords = plotting.find_probabilistic_atlas_cut_coords(atlas)

    # -------------------------------------------------------------------------
    # 3. Identify each coordinate's most likely brain region
    #    (using your custom or imported read_atlas_peak function).
    print("Generating brain region labels...")
    brain_region_raw = []
    for atlas_coord in atlas_coords:
        region_list = read_atlas_peak(atlas_name, atlas_coord)
        brain_region_raw.append(region_list)

    # Flatten region_list by picking the label with the largest probability
    # or None if empty
    brain_region = [
        max(inner_list, key=lambda x: x[0])[-1] if inner_list else None
        for inner_list in brain_region_raw
    ]

    # -------------------------------------------------------------------------
    # 4. Read the open-access and restricted data, merge them
    print("Reading and merging subject data...")
    open_access_data = pd.read_csv(open_access_csv)
    restricted_data = pd.read_csv(restricted_csv)
    subject_data = open_access_data.merge(restricted_data, how="inner", on="Subject")

    # -------------------------------------------------------------------------
    # 5. Load time-series data from the folder; build a DataFrame
    print("Reading time-series data...")
    brain_files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    brain_data = {}

    for filename in brain_files:
        subject_id = int(filename[:6])  # first 6 chars assume subject ID
        file_path = os.path.join(folder, filename)
        subject_brain_data = np.loadtxt(file_path)
        brain_data[subject_id] = subject_brain_data

    # Convert dict to DataFrame
    brain_data_df = pd.DataFrame({
        'Subject': list(brain_data.keys()),
        'Brain_Data': list(brain_data.values())  # arrays of shape (4800, d)
    })

    # -------------------------------------------------------------------------
    # 6. Merge the subject data with the brain data
    print("Merging subject data with brain data...")
    data = subject_data.merge(brain_data_df, on="Subject", how="inner")

    # -------------------------------------------------------------------------
    # 7. Save the full data to a CSV
    print(f"Saving all data to {output_all_data_csv}...")
    data.to_csv(output_all_data_csv, index=False)

    # -------------------------------------------------------------------------
    # 8. Save the brain_region list to another CSV
    print(f"Saving brain_region labels to {output_brain_region_csv}...")
    brain_region_df = pd.DataFrame({"Brain_Region": brain_region})
    brain_region_df.to_csv(output_brain_region_csv, index=False)

    print("Done! Created two CSV files.")

