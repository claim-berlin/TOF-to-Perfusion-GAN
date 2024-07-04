## needed for pathing import of config
import sys
sys.path.append("../../tof-perfusion")
import numpy as np
import os
import nibabel as nib
from datetime import datetime
import shutil
import pickle as pkl

# background
seed = 2022
np.random.seed(seed)
date = datetime.today().strftime('%Y%m%d')[2:]

path = "home" # cluster

# parameters
ds_split = ["train", "val", "test"] # train, val, test
ds_ratios = [0.7, 0.2, 0.1] # [0.7, 0.2, 0.1]
seq = sys.argv[1] #["DSC_SOURCE"]  #["TOF", "TTP", "CBF"]
dataset = "peg" # peg, hdb
ignore = ["none"] # filenames/folders to ignore
loadIds = False
specificIds = True

if dataset == "hdb":
    folder = "Heidelberg"
elif dataset == "peg":
    folder = "PEGASUS"

# pathing
if path == "home":
    input_path = "/data/{}/processed/skullstripping/".format(folder)
    target_path = "/data/{}/cohorts/skullstripping/".format(folder)
elif path == "cluster":
    input_path = "/data/{}/processed/skullstripping/".format(folder)
    target_path = "/data/{}/cohorts/skullstripping/".format(folder)


metadata_path = target_path + "metadata/"
suffix = "_{}.nii.gz".format(seq)

# extract image ids in input dir
files = os.listdir(input_path)
pat_ids_new = list()
for file in files:
    if file not in ignore:
        file = file.split("_")[0]
        pat_ids_new.append(file)

pat_ids_new = sorted(set(pat_ids_new), key=pat_ids_new.index)
pat_ids = pat_ids_new
print(len(pat_ids))
print(pat_ids)

# collect patient ids and randomly assign train, test and val sets    
np.random.shuffle(pat_ids)       
train_ids = pat_ids[:int(ds_ratios[0]*len(pat_ids))]
val_ids = pat_ids[int(ds_ratios[0]*len(pat_ids)):int((ds_ratios[0]+ds_ratios[1])*len(pat_ids))]
test_ids = pat_ids[int((ds_ratios[0]+ds_ratios[1])*len(pat_ids)):]

# if we want to load specific IDs instead
if loadIds:
    with open('../temp/trainIds.pkl', 'rb') as handle:
        train_ids = pkl.load(handle)
    with open('../temp/valIds.pkl', 'rb') as handle:
        val_ids = pkl.load(handle)
    with open('../temp/testIds.pkl', 'rb') as handle:
        test_ids = pkl.load(handle)
    print("Loading specified Ids: ")
    print("train ########")
    print(sorted(train_ids))
    print("val ########")
    print(sorted(val_ids))
    print("test ########")
    print(sorted(test_ids))
print(len(train_ids), len(val_ids), len(test_ids))

# specific manual IDs if necessary
if specificIds:
    test_ids = ["PEG0019",  "PEG0021",  "PEG0029",  "PEG0030",  "PEG0031",  "PEG0042",  "PEG0067",  "PEG0076"]
    val_ids = ["PEG0004",  "PEG0015",  "PEG0028",  "PEG0041",  "PEG0048",  "PEG0065",  "PEG0071", "PEG0009",  "PEG0018",  "PEG0034",  "PEG0045",  "PEG0052",  "PEG0066",  "PEG0079"]
    train_ids = [
"PEG0003", "PEG0011",  "PEG0020", "PEG0026",  "PEG0036",  "PEG0044", "PEG0053",  "PEG0058",  "PEG0063",  "PEG0072",
"PEG0005",  "PEG0013", "PEG0022",  "PEG0027",  "PEG0037",  "PEG0046",  "PEG0054",  "PEG0059",  "PEG0064",  "PEG0073",
"PEG0006",  "PEG0014",  "PEG0023",  "PEG0032", "PEG0038",  "PEG0047", "PEG0055", "PEG0060",  "PEG0068",  "PEG0074",
"PEG0008", "PEG0016", "PEG0024",  "PEG0033",  "PEG0040", "PEG0049", "PEG0056",  "PEG0061",  "PEG0069",  "PEG0075",
"PEG0010",  "PEG0017",  "PEG0025",  "PEG0035",  "PEG0043",  "PEG0050",  "PEG0057", "PEG0062",  "PEG0070",  "PEG0078"
]

# create metadata path if not existing
if not os.path.isdir(metadata_path):
    os.makedirs(metadata_path)

print("### Starting preprocessing ###")
# starting training cohort
if "train" in ds_split:
    print("### Working on train cohort...")
    # prepare necessary cohort paths
    # check if needed pathing exists and create if needed
    if not os.path.isdir("{}/train/{}".format(target_path, seq)):
        os.makedirs("{}/train/{}".format(target_path, seq))
    
    # for storage of cohort wide min/max
    train_min = 100
    train_max = 0

    # copy id files designated in train cohort into corresponding directory + collect cohort min/max
    print("Extracting min/max...")
    for id in train_ids:

        filename = id + suffix

        shutil.copyfile(input_path + filename, target_path + "/train/" + seq + "/" + filename)

        # collect nifty information
        tmp = nib.load(target_path + "/train/"+ seq + "/" + filename)
        tmp_data = np.nan_to_num(tmp.get_fdata())
        tmp_min = np.min(tmp_data)
        tmp_max = np.max(tmp_data)

        # collect min/max
        if tmp_min < train_min:
            train_min = tmp_min
        if tmp_max > train_max:
            train_max = tmp_max
    
    # save cohort min/max
    print("Min: {} / Max: {}".format(train_min, train_max))
    extremes = {"min": train_min, "max": train_max}
    output = open(metadata_path + "extremes_{}_train_{}.pkl".format(seq, date), "wb")
    pkl.dump(extremes, output)
    output.close()

    # preprocess data in cohort
    print("Normalizing and rotating...")
    for id in train_ids:
        filename = id + suffix

        # collect nifty information
        tmp = nib.load(target_path + "/train/"+ seq + "/" + filename)
        tmp_data = np.nan_to_num(tmp.get_fdata())
        tmp_header = tmp.header
        tmp_affine = tmp.affine
        tmp_min = np.min(tmp_data)
        tmp_max = np.max(tmp_data)
        
        # normalize data + rotate according to dataset
        tmp_data = np.nan_to_num(tmp_data)
        tmp_data_norm = 2 * ((tmp_data - train_min) / (train_max - train_min)) - 1
        if dataset == "peg":
            tmp_data_norm =np.flip(np.rot90(tmp_data_norm, 1), 1)
        elif dataset == "hdb":
            tmp_data_norm =np.flip(np.rot90(tmp_data_norm, 1), 1)

        # save as nifty file
        tmp_data_nii = nib.Nifti1Image(tmp_data_norm, affine = tmp_affine, header = tmp_header)
        nib.save(tmp_data_nii, target_path + "/train/"+ seq + "/" + filename)

    print("Finished train cohort...")

# starting validation cohort
if "val" in ds_split:
    print("### Working on val cohort...")
    # prepare necessary cohort paths
    # check if needed pathing exists and create if needed
    if not os.path.isdir("{}/val/{}".format(target_path, seq)):
        os.makedirs("{}/val/{}".format(target_path, seq))
    
    # for storage of cohort wide min/max
    val_min = 100
    val_max = 0

    # copy id files designated in val cohort into corresponding directory + collect cohort min/max
    print("Extracting min/max...")
    for id in val_ids:
        filename = id + suffix
        shutil.copyfile(input_path + filename, target_path + "/val/" + seq + "/" + filename)

        # collect nifty information
        tmp = nib.load(target_path + "/val/"+ seq + "/" + filename)
        tmp_data = np.nan_to_num(tmp.get_fdata())
        tmp_min = np.min(tmp_data)
        tmp_max = np.max(tmp_data)

        # collect min/max
        if tmp_min < val_min:
            val_min = tmp_min
        if tmp_max > val_max:
            val_max = tmp_max

    # save cohort min/max
    print("Min: {} / Max: {}".format(val_min, val_max))
    extremes = {"min": val_min, "max": val_max}
    output = open(metadata_path + "extremes_{}_val_{}.pkl".format(seq, date), "wb")
    pkl.dump(extremes, output)
    output.close()

    # preprocess data in cohort
    print("Normalizing and rotating...")
    for id in val_ids:
        filename = id + suffix

        # collect nifty information
        tmp = nib.load(target_path + "/val/"+ seq + "/" + filename)
        tmp_data = np.nan_to_num(tmp.get_fdata())
        tmp_header = tmp.header
        tmp_affine = tmp.affine
        tmp_min = np.min(tmp_data)
        tmp_max = np.max(tmp_data)
        
        # normalize data + rotate according to dataset
        tmp_data = np.nan_to_num(tmp_data)
        tmp_data_norm = 2 * ((tmp_data - val_min) / (val_max - val_min)) - 1
        if dataset == "peg":
            tmp_data_norm =np.flip(np.rot90(tmp_data_norm, 1), 1)
        elif dataset == "hdb":
            tmp_data_norm =np.flip(np.rot90(tmp_data_norm, 1), 1)

        # save as nifty file
        tmp_data_nii = nib.Nifti1Image(tmp_data_norm, affine = tmp_affine, header = tmp_header)
        nib.save(tmp_data_nii, target_path + "/val/"+ seq + "/" + filename)

    print("Finished val cohort...")

if "test" in ds_split:
    print("### Working on test cohort...")
    # prepare necessary cohort paths
    # check if needed pathing exists and create if needed
    if not os.path.isdir("{}/test/{}".format(target_path, seq)):
        os.makedirs("{}/test/{}".format(target_path, seq))
    
    # for storage of cohort wide min/max
    test_min = 100
    test_max = 0

    # copy id files designated in train cohort into corresponding directory + collect cohort min/max
    print("Extracting min/max...")
    for id in test_ids:
        filename = id + suffix
        shutil.copyfile(input_path + filename, target_path + "/test/" + seq + "/" + filename)

        # collect nifty information
        tmp = nib.load(target_path + "/test/"+ seq + "/" + filename)
        tmp_data = np.nan_to_num(tmp.get_fdata())
        tmp_min = np.min(tmp_data)
        tmp_max = np.max(tmp_data)

        # collect min/max
        if tmp_min < test_min:
            test_min = tmp_min
        if tmp_max > test_max:
            test_max = tmp_max

    # save cohort min/max
    print("Min: {} / Max: {}".format(test_min, test_max))
    extremes = {"min": test_min, "max": test_max}
    output = open(metadata_path + "extremes_{}_test_{}.pkl".format(seq, date), "wb")
    pkl.dump(extremes, output)
    output.close()

    # preprocess data in cohort
    print("Normalizing and rotating...")
    for id in test_ids:
        filename = id + suffix

        # collect nifty information
        tmp = nib.load(target_path + "/test/"+ seq + "/" + filename)
        tmp_data = np.nan_to_num(tmp.get_fdata())
        tmp_header = tmp.header
        tmp_affine = tmp.affine
        tmp_min = np.min(tmp_data)
        tmp_max = np.max(tmp_data)
        
        # normalize data + rotate according to dataset
        tmp_data = np.nan_to_num(tmp_data)
        tmp_data_norm = 2 * ((tmp_data - test_min) / (test_max - test_min)) - 1
        if dataset == "peg":
            tmp_data_norm =np.flip(np.rot90(tmp_data_norm, 1), 1)
        elif dataset == "hdb":
            tmp_data_norm =np.flip(np.rot90(tmp_data_norm, 1), 1)

        # save as nifty file
        tmp_data_nii = nib.Nifti1Image(tmp_data_norm, affine = tmp_affine, header = tmp_header)
        nib.save(tmp_data_nii, target_path + "/test/"+ seq + "/" + filename)




    print("Finished test cohort...")

print("### Finished preprocessing ###")





        




