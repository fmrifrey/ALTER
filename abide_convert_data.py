import pandas as pd
import numpy as np

# desired number of timepoints to use
desired_T = 100

# load the phenos
phenos = pd.read_csv('/StorageRAID/djfrey/abide_data/Phenotypic_V1_0b_preprocessed1.csv')
valid_rows = phenos[phenos['FILE_ID'] != 'no_filename']
file_ids = valid_rows['FILE_ID'].tolist()
labels = valid_rows['DX_GROUP'].astype(int).tolist()
sites = valid_rows['SITE_ID'].tolist()
L = len(file_ids)

# initialize
all_timeseries = []
all_corr = []
all_labels = []
all_sites = []

# loop through subjects in csv file
for l, file_id in enumerate(file_ids):
    fname = f"/StorageRAID/djfrey/abide_data/Outputs/dparsf/nofilt_noglobal/rois_cc200/{file_id}_rois_cc200.1D"
    try:
        timeseries = np.loadtxt(fname) # load timeseries

        if timeseries.shape[0] < desired_T:
            # Too few timepoints, skip subject
            print(f"Skipping {file_id}: only {timeseries.shape[0]} timepoints")
            continue
        elif timeseries.shape[0] > desired_T:
            # Too many, trim to first 100
            timeseries = timeseries[:desired_T, :]

        corr = np.corrcoef(timeseries.T) # calculate correlation coefficients
        label = 1. * (labels[l] == 1) # 1 = ASD, 2 = Control
        site = sites[l]

        all_timeseries.append(timeseries)
        all_corr.append(corr)
        all_labels.append(label)
        all_sites.append(site)
    except Exception as e:
        print(f"Could not load {fname}: {e}")

# write to numpy arrays
all_timeseries = np.array(all_timeseries).transpose(0,2,1)
all_corr = np.array(all_corr)
all_labels = np.array(all_labels)
all_sites = np.array(all_sites)

# Find the largest even number <= number of subjects
num_subjects = all_timeseries.shape[0]
even_subjects = num_subjects - (num_subjects % 2)

# Truncate all arrays to this even length
all_timeseries = all_timeseries[:even_subjects]
all_corr = all_corr[:even_subjects]
all_labels = all_labels[:even_subjects]
all_sites = all_sites[:even_subjects]

print(f"Loaded {len(all_timeseries)} subjects.")
print(f"Timeseries shape: {all_timeseries.shape}")
print(f"Correlation shape: {all_corr.shape}")

# save to npy file
data_dict = {
    "timeseires": all_timeseries,
    "corr": all_corr,
    "label": all_labels,
    "site": all_sites,
}
np.save('abide2.npy', data_dict)