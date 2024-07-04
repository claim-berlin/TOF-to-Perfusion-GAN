# config for the TOF-project
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# pathing
root = "" 
database = "" # Heidelberg, PEGASUS
experimentName = ""

if root == "":
    root_path = ""
    experiment_path = "{}/".format(experimentName)
    data_path = root_path + "{}/".format(database)
elif root == "":
    root_path = ""
    data_path = root_path + "{}/".format(database)

# in and output of the model
input_seq = ["TOF"]
output_seq = ["TMAX"]

if database == "Heidelberg":
    data = ["hdb"]  # "peg", "hdb"
elif database == "PEGASUS":
    data = ["peg"]  # "peg", "hdb"


# other paths
model_path = experiment_path + "models/"
gen_img_path = experiment_path + "genImages/"

if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)
if not os.path.exists(gen_img_path):
    os.makedirs(gen_img_path)

# gen imgs path 
extremesDate = "240124" # date on which file with min/max values of the cohort was created
gen_input_img_path_3D = data_path + "val/" 
gen_output_img_path_3D = data_path + "val/" + output_seq[0] + "/"
extremesFile = "extremes_{}_train_{}.pkl".format(output_seq[0], extremesDate)

# specific images to generate during training to visually check 
specificImgsToGen = ["0195", "0432", "0720"] # cannot be less than imgs to generate / if empty then imgs will be taken from val iteratively / MAKE SURE IMG IS IN VAL COHORT!!!

# model settings

save_model = True # save model
save_every_X_epoch = 10 # model saving frequency
generate_while_train = False # generate slices during model training 
nr_imgs_gen = 3  # up to five
sliceToShow = 60 # slice to generate for example of model progression

# training settings
epochs = 80
paddingMode = "reflect" 
model_name = 'pix2pix-pytorch'
upsampling = False  # if upsampling instead of convtranspose, only for unetG false
batch_size = 8  #18 # size of the training batches 
ngf = 256 # num generator filters (256)
nr_layer_g = 7  # for UnetG (default: 7)
nr_layer_d = 3  # for PatchD (default: 3)
ndf = 64 # num discriminator filters (128) (Patch was 64)
niter = 10 # number iterations
lr_g = 0.0001 # learning rate generator
lr_d = 0.00001 # learning rate discriminator
dropout_rate = 0 
weightinit = "normal"  # "normal" or "xavier", only if not patchD
init_sd = 0.05 # 0.05  # initial standard deviation for normal distr
init_mean = 0  #0
slope_relu = 0.2
patchD = True # use patch discriminator
kernel_size = 5  # only for no patchD
netG_type = "unet"  # unet or resnet as G
pretrainedG = False  # only for UnetG

feature_matching = False  # only possible when no patchD
label_smoothing = False  # False, one-sided -> changing labels from numeric to real

# loss
# for vanilla gan choose "BCE", for lsgan "l2" (both d and g), 
# for reconstruction so far l1 is used
criterion_d = "BCE"  # "BCE", "l1", "l2"
criterion_g = "BCE"  # "BCE" for adv_loss, if feature_matching, then l1 loss
loss_ratio = False #50  # 10 if number then loss updated simultaneously, otherwise after each other
reconstruction_loss = "L1-SSIM" # L1 (default) or L1-SSIM

use_gpu = True
gpu_idx = [0] # default 0 but on DL2 0 or 1
nr_gpus = len(gpu_idx)
bn = True # BatchNorm
norm_d_3d = nn.BatchNorm3d
threads = 0  # for loading data
seed = 12#23
seedPreproc = 2024

# optimizers
optimizer_d = "adam" # SGD
beta1_d = 0.5
beta1_g = 0.5
beta2_d = 0.999
beta2_g = 0.999

# GAN architecture
WGAN = False
lbd = 10
nr_d_train = 1  # 1

# only for continued training
continue_training = False
load_from_epoch = 20
 
# for evaluation
metrics = ["MSE", "NRMSE", "PSNR", "SSIM", "MAE"]
save_nii = True
    
