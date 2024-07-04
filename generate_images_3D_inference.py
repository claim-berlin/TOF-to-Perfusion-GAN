import sys
sys.path.append("../../TOF-perfusion")
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import nibabel as nib
import pickle
from utils import eval_utils as eut
from utils import training_utils as tut
import config as c
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import eval_utils as eut

##### generate imgs with a saved model in inference mode ####
location = ""

# pathing
if location == "":
    root_path = ""
else:
    root_path = ""

# dataset
database = "peg" # peg

if database == "hdb":
    data_name = "Heidelberg"
elif database == "peg":
    data_name = "PEGASUS"

# parameters
use_gpu = True
seed = 12
gpu_idx = 0
trial_name = "peg_3d_final_TMAX"
output_name = "peg_3d_final_TMAX_Testset"
epoch = 99
cohortIdPath = root_path + "{}/metadata/skullstripping/".format(data_name)
cohort = "test"
inputSeq = "TOF"
outputSeq = "TMAX"
extremesDate = "220519"
genImgPath = root_path + "{}/generated/{}_{}/".format(data_name, output_name, epoch)
inputImgPath = root_path + "{}/cohorts/skullstripping/{}/{}/".format(data_name, cohort, inputSeq)
outputImgPath = root_path + "{}/processed/skullstripping/".format(data_name)
extremesPath = root_path + "{}/cohorts/skullstripping/metadata/".format(data_name)

# set up cuda env
cudnn.benchmark = False
cudnn.deterministic = True
torch.cuda.manual_seed(c.seed)
torch.cuda.manual_seed_all(c.seed)
np.random.seed(c.seed)
os.environ["PYTHONHASHSEED"] = str(c.seed)
Tensor = torch.FloatTensor
device = torch.device("cuda:"+str(gpu_idx))

print("Generating imgs with {} with input {} and output {}...".format(trial_name, inputSeq, outputSeq))
# create genImgs folder
if not os.path.isdir(genImgPath):
    os.makedirs(genImgPath)

# load cohort ids to generate
cohortIds = [x.split("_")[0] for x in os.listdir(inputImgPath)]
print(cohortIds)

# load extremes file for output seq
extremesFile = "extremes_{}_train_{}.pkl".format(outputSeq, extremesDate)

# load generator
net_g, saved_params = eut.load_g_3D(trial_name, epoch)
net_g.to(device)
print(net_g)

# prep metric lists
metrics = []
mselist = {}
nrmselist = {}
psnrlist = {}
maelist = {}
ssimlist = {}

# load imgs to generate
start_time = time.time()  
for id in cohortIds:

    print("Working on: " + id)
    # load input img and output data
    inputData = nib.load(inputImgPath + "{}_{}.nii.gz".format(id, inputSeq))
    inputImg = inputData.get_fdata()
    outputData = nib.load(outputImgPath + "{}_{}.nii.gz".format(id, outputSeq))
    out_affine = outputData.affine
    out_header = outputData.header

    # load min and max of output file for later rescaling
    with open(extremesPath + extremesFile, "rb") as input_file:
        extremes = pickle.load(input_file)
    out_min = extremes["min"]
    out_max = extremes["max"]

    # load necessary padding
    padding = 128 - inputImg.shape[2]
    # only pad last dimension one-sided
    padding = (0,padding,0,0,0,0)

    # transform input
    inputImg = torch.FloatTensor(inputImg.copy())

    # add dimension for generator fit
    inputImg = torch.unsqueeze(torch.unsqueeze(inputImg, 0), 0)
    inputImg = F.pad(inputImg, padding, "reflect")
    inputImg = inputImg.to(device)

    # generate output image via generator 
    gen_img = net_g(inputImg)
    gen_img = np.squeeze(gen_img.detach().cpu().numpy().astype(np.float64))

    # load target image for metrics
    target_img = outputData.get_fdata()
    if database == "hdb":
        gen_img = gen_img[:,:,:98]
    elif database == "peg":
        gen_img = gen_img[:,:,:127]

    # get metrics
    data_range = 2
    mselist[id] = mse(target_img, gen_img)
    nrmselist[id] = nrmse(target_img, gen_img)
    psnrlist[id] = psnr(target_img, gen_img, data_range = data_range)
    nr_img = target_img.shape[0]
    tmp_list = np.empty((nr_img))
    for d in range(nr_img):
        ssimScaled = (ssim(target_img[d].squeeze(), gen_img[d].squeeze()) + 1)/ 2
        tmp_list[d] = ssimScaled
    ssimlist[id] = np.sum(tmp_list)/nr_img
    maelist[id] = eut.mae(target_img.squeeze(), gen_img.squeeze())

    # re-scale gen_imgs via val_extremes file of output modality
    gen_img = (out_max-out_min) * ((gen_img + 1) / 2) + out_min

    # re-orient
    if database == "peg":
        gen_img_new = np.rot90(np.flipud(gen_img), 1)
    elif database=="hdb":
        gen_img_new = np.rot90(np.flipud(gen_img), 1) #np.rot90(np.flipud(gen_img), 1)
    gen_nii = nib.Nifti1Image(gen_img_new, affine=out_affine, header = out_header)

    # save generated img
    nib.save(gen_nii, genImgPath + "gen_{}_{}_{}.nii.gz".format(id, outputSeq,epoch))

# save metrics for generation
metrics = [mselist, nrmselist, psnrlist, maelist, ssimlist]
with open(genImgPath + "generatedMetrics.pkl", 'wb') as handle:
    pickle.dump(metrics, handle)

end_time = time.time()
total_time = end_time - start_time
print("Time for generating {} imgs: {} mins".format(len(cohortIds), round(total_time/60)))  
