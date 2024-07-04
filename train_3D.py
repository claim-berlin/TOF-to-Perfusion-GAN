import numpy as np
import time
import gc
import os
import csv
import pandas as pd
import torch
import shutil
import nibabel as nib
import pickle

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.backends.cudnn as cudnn
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

import config as c
from models.model_3D import PatchDiscriminator_3D, UnetGenerator_3D
from utils import training_utils as tut
from utils import eval_utils as eut

if c.use_gpu and not torch.cuda.is_available():
    raise Exception("No GPU found, please change use_gpu to False")

if c.use_gpu:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(c.seed)
    torch.cuda.manual_seed_all(c.seed)
    np.random.seed(c.seed)
    os.environ["PYTHONHASHSEED"] = str(c.seed)
    Tensor = torch.FloatTensor
    device = torch.device("cuda:"+str(c.gpu_idx[0]))
else:
    torch.manual_seed(c.seed)
    #Tensor = torch.FloatTensor
    device = torch.device("cpu")

### training routine for TOF-perf GAN with 3D image input ###

# check if needed pathing exists and create if needed
if not os.path.isdir("{}data/models/TOF-perf/Trial_{}/gen_imgs/".format(c.root_path, c.trial_nr)):
    os.makedirs("{}data/models/TOF-perf/Trial_{}/gen_imgs/".format(c.root_path, c.trial_nr))

# copy config
shutil.copyfile("config.py", "{}data/models/TOF-perf/Trial_{}/gen_imgs/config.py".format(c.root_path, c.trial_nr))

# loading cohorts
# shape = num of slices (batchsize), channels, x, y
# load train
print('===> Loading training data')
train_dataloader = tut.load_3D_dataset(c.data[0], c.input_seq[0], c.output_seq[0], batch_size=1, datasplit="train")

# load val
print('===> Loading validation data')
val_dataloader = tut.load_3D_dataset(c.data[0], c.input_seq[0], c.output_seq[0], batch_size=1, datasplit="val")

if c.use_gpu:
    Tensor = torch.cuda.FloatTensor

print('===> Building models')
if c.netG_type=="unet":
    net_g = UnetGenerator_3D(input_nc=1, output_nc=1).to(device)
else:
    print("Model type not available!")
    exit(1)
if c.patchD:
    net_d = PatchDiscriminator_3D(input_nc=2).to(device)
else:
    print("Model type not available!")
    exit(1)

# initalize weights or load model
if c.continue_training:
    # load presaved model
    saved_model_path = c.model_path + "epoch" + str(c.load_from_epoch) + ".pth"
    saved_params_dict = torch.load(saved_model_path)
else:
    # initialize weights
    net_g.apply(tut.weights_init)
    net_d.apply(tut.weights_init)
    if c.bn:
        # batch normalization
        net_g.apply(tut.bn_init)
        net_d.apply(tut.bn_init)

# optimizer  
optim_g = optim.Adam(net_g.parameters(), lr=c.lr_g, betas=(c.beta1_g, c.beta2_g))
if c.optimizer_d=="SGD":
    optim_d = optim.SGD(net_d.parameters(), lr=c.lr_d, momentum=0.9)
else:
    optim_d = optim.Adam(net_d.parameters(), lr=c.lr_d,
                         betas=(c.beta1_d, c.beta2_g))

if c.continue_training:
    # initalize weights with saved ones
    net_g.load_state_dict(saved_params_dict["generator_state_dict"])
    optim_g.load_state_dict(saved_params_dict["gen_opt_state_dict"])
    net_d.load_state_dict(saved_params_dict["discriminator_state_dict"])
    optim_d.load_state_dict(saved_params_dict["discr_opt_state_dict"])
    # get old losses
    losses = np.load(c.model_path + "gen_imgs/losses.npz")
    losses_d = losses["d"].tolist()
    losses_g = losses["g"].tolist()
    losses_val = losses["val"].tolist()
    start_epoch = c.load_from_epoch + 1
else:
    losses_d = []
    losses_g = []
    losses_val = []
    start_epoch = 0

print(net_g)
print(net_d)

# multiple gpu usage
if c.use_gpu and (c.nr_gpus > 1):
    print("Let's use", c.nr_gpus, "GPUs!")
    net_g = nn.DataParallel(net_g, device_ids=c.gpu_idx)
    net_d = nn.DataParallel(net_d, device_ids=c.gpu_idx)

# generate sample images during training
if c.generate_while_train:
    pass

print('===> Starting training')

start_time = time.time()
nr_params_g = sum(p.numel() for p in net_g.parameters())
nr_params_d = sum(p.numel() for p in net_d.parameters())
print("nr params g:", nr_params_g)
print("nr params d:", nr_params_d)

# calculate z padding needed to reach 128
sample_pair = next(iter(train_dataloader))
padding = 128 - sample_pair[0].shape[3]

# only pad last dimension one-sided
padding = (0,padding,0,0,0,0)

for epoch in range(start_epoch, c.epochs):

    print(epoch + 1, "/", c.epochs)

    start_time_epoch = time.time()

    for i, img in enumerate(train_dataloader, 0):

        # input set
        input_img = img[0]
        # add dimension for depth = batchsize, channels, x, y, z
        input_img = torch.unsqueeze(input_img, 0)
        input_img = F.pad(input_img, padding, c.paddingMode)

        #input_img = F.normalize(input_img, dim = 1)
        input_img = input_img.to(device)
        
        # output set
        target_img = img[1]
        # add dimension for depth = batchsize, channels, x, y, z
        target_img = torch.unsqueeze(target_img, 0)
        target_img = F.pad(target_img, padding, c.paddingMode)
        #target_img = F.normalize(target_img, dim = 1)
        target_img = target_img.to(device)

        ##############################
        ######### Update D ###########
        ##############################

        # iterations for training D
        for _ in range(c.nr_d_train):
            # get real pair
            real_pair = torch.cat((input_img, target_img), 1)
            real_pair_gpu = real_pair.to(device)

            # get generated pair
            gen_img = net_g(input_img)
            gen_pair = torch.cat((input_img, gen_img), 1)
            gen_pair_gpu = gen_pair.to(device)

            b_size = real_pair_gpu.size(0)

            # get predictions from discriminator -> probabilities which should be close to 1 for real and close to 0 for generated
            out_real, lastconv_out_real = net_d(real_pair_gpu)
            out_gen, lastconv_out_gen = net_d(gen_pair_gpu)

            # generate tensors with all 1 in size of real output probs and all 0 in size of gen output probs
            if c.label_smoothing:
                # for real_label choose between 0.7 and 1.2 instead of 1
                label_real = torch.empty(out_real.size(),
                                         device=device).uniform_(0.7, 1.2)  # check if still works
            else:
                label_real = torch.ones(out_real.size()).to(device)
            label_gen = torch.zeros(out_gen.size()).to(device)

            # calculate loss for D
            loss_d = tut.calculate_loss_d(net_d, optim_d, out_real, label_real, out_gen, label_gen)

        # forward pass D
        out_gen_new, lastconv_out_gen_new = net_d(gen_pair_gpu) # get gen probs of D after updating D
        label_gen_new = torch.ones(out_gen_new.size()).to(device)

        ##############################
        ######### Update G ###########
        ##############################

        # calculate adversarial loss
        if c.feature_matching:
            out_real_new, lastconv_out_real_new = net_d(real_pair_gpu)
            adv_loss = F.l1_loss(lastconv_out_gen_new, lastconv_out_real_new)
        else:
            if c.criterion_g=="l1":
                adv_loss = F.l1_loss(out_gen_new, label_gen_new)
            elif c.criterion_g=="l2":
                criterion = nn.MSELoss()
                adv_loss = criterion(out_gen_new, label_gen_new)
            elif c.criterion_g=="hinge":
                criterion = nn.HingeEmbeddingLoss()
                adv_loss = criterion(out_gen_new, label_gen_new)
            else:
                if c.patchD:
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    criterion = nn.BCELoss()
                adv_loss = criterion(out_gen_new, label_gen_new) # adv loss is loss of updated D on gen pair but labels 1
        
        # calculate reconstruction loss
        if c.reconstruction_loss == "L1-SSIM": # weighted L1 SSIM reconstruction loss

            gen_img_np = gen_img.cpu().detach().numpy().astype(np.float64)
            target_img_np = target_img.cpu().detach().numpy().astype(np.float64)
            alpha = 0.84 # 0.84
            ssimScaled = (ssim(gen_img_np.squeeze(), target_img_np.squeeze()) + 1)/ 2
            reconstr_loss = ((1 - alpha) * F.l1_loss(gen_img, target_img)) + (alpha * (1 - ssimScaled))


        else: # L1 reconstruction loss (default)
            reconstr_loss = F.l1_loss(gen_img, target_img)

        # decide on loss ratio calculation and timing    
        if c.loss_ratio==False: # this is used with c.loss_ratio set tp False
            net_g.zero_grad()
            adv_loss.backward(retain_graph=True)
            #reconstr_loss = F.l1_loss(gen_img, target_img) # could be replaced by L1 weighted SSIM loss       
            reconstr_loss.backward(retain_graph=True)
            optim_g.step()
            loss_g_new = adv_loss.detach() + reconstr_loss.detach()           
        else:
            #reconstr_loss = F.l1_loss(gen_img, target_img) # default L1 reconstruction loss
            loss_g = c.loss_ratio * reconstr_loss + adv_loss 
            net_g.zero_grad()
            loss_g.backward()
            optim_g.step()
            loss_g_new = loss_g.detach()
        
    losses_d.append(loss_d.cpu())
    losses_g.append(loss_g_new.cpu())

    end_time_epoch = time.time()
    print("Training for epoch", epoch + 1, "took",
          (end_time_epoch - start_time_epoch) / 60, "minutes.")
    print("D loss:", loss_d)
    print("G loss:", loss_g_new)
    print("adv loss:", adv_loss) 
    mselist, nrmselist, psnrlist, ssimlist, maelist = [], [], [], [], []

    # calculate validation loss
    with torch.no_grad():   
        val_loss = 0
        for i, img in enumerate(val_dataloader, 0):
            # perform image padding and dim extension
            input_img = img[0]
            input_img = torch.unsqueeze(input_img, 0)
            input_img = F.pad(input_img, padding, c.paddingMode)

            #input_img = F.normalize(input_img, dim = 1)
            input_img = input_img.to(device)

            target_img = img[1]
            target_img = torch.unsqueeze(target_img, 0)
            target_img = F.pad(target_img, padding, c.paddingMode)
            #target_img = F.normalize(target_img, dim = 1)
            target_img = target_img.to(device)
    
            gen_img = net_g(input_img)
            val_loss += F.l1_loss(gen_img, target_img)

            gen_img_np = gen_img.cpu().detach().numpy().astype(np.float64)
            target_img_np = target_img.cpu().detach().numpy().astype(np.float64)
            

            # get metrics
            data_range = 2
            mselist.append(mse(target_img_np, gen_img_np))
            nrmselist.append(nrmse(target_img_np, gen_img_np))
            psnrlist.append(psnr(target_img_np, gen_img_np, data_range = data_range))
            
            if c.batch_size<8:
                nr_img = target_img_np.shape[0]
                tmp_list = np.empty((nr_img))
                for d in range(nr_img):
                    ssimScaled = (ssim(target_img_np[d].squeeze(), gen_img_np[d].squeeze()) + 1)/ 2
                    tmp_list[d] = ssimScaled
                ssimlist.append(np.sum(tmp_list)/nr_img)
            else:
                ssimScaled = (ssim(target_img_np.squeeze(), gen_img_np.squeeze()) + 1)/ 2
                ssimlist.append(ssimScaled)
            maelist.append(eut.mae(target_img_np.squeeze(), gen_img_np.squeeze()))
        losses_val.append(val_loss.detach().cpu())
        print("Val loss:", val_loss.detach())

    
    # save model
    if c.save_model or (epoch==(c.epochs-1)):
        if (epoch in np.arange(start_epoch-1, c.epochs, c.save_every_X_epoch)) or (epoch==(c.epochs-1)):
            tut.save_model(epoch, net_g, net_d, optim_g, optim_d, losses_g, 
                           losses_d)
    
    # generate images during training and save them 
    if (c.generate_while_train) and (epoch%10 == 0):
        print("Starting to generate {} Images...".format(c.gen_imgs_num))
        # build input path
        gen_input_img_path_3D = c.gen_input_img_path_3D + c.input_seq[0] + "/"

        # get images from validation cohort of input seq
        input_imgs = os.listdir(gen_input_img_path_3D)
        output_imgs = os.listdir(c.gen_output_img_path_3D)
        # load min and max of output file for later rescaling
        with open(c.data_path + "metadata/" + c.extremesFile, "rb") as input_file:
            extremes = pickle.load(input_file)
        out_min = extremes["min"]
        out_max = extremes["max"]
        for i in range(c.gen_imgs_num):

            # check for specific imgs to generate
            usedSpecific = False
            if (len(c.specificImgsToGen) != 0) and (len(c.specificImgsToGen) >= c.gen_imgs_num):
                print("Generating specific img...")
                usedSpecific = True
                input_img = nib.load(gen_input_img_path_3D + c.specificImgsToGen[i] + "_{}.nii.gz".format(c.input_seq[0])).get_fdata()
                input_img = torch.FloatTensor(input_img.copy())
                output_data = nib.load(c.gen_output_img_path_3D + c.specificImgsToGen[i] + "_{}.nii.gz".format(c.output_seq[0]))
            else:
                # load img and get data
                print("Generating img iteratively...")
                input_img = nib.load(gen_input_img_path_3D + input_imgs[i]).get_fdata()
                input_img = torch.FloatTensor(input_img.copy())
                output_data = nib.load(c.gen_output_img_path_3D + output_imgs[i])

            out_affine = output_data.affine
            out_header = output_data.header

            # add dimension for generator fit
            input_img = torch.unsqueeze(torch.unsqueeze(input_img, 0), 0)
            input_img = F.pad(input_img, padding, c.paddingMode)
            #print(input_img.shape())
            input_img = input_img.to(device)

            # generate output image via generator 
            gen_img = net_g(input_img)
            gen_img = np.squeeze(gen_img.detach().cpu().numpy().astype(np.float64))

            # re-scale
            #gen_img = (gen_img - np.min(gen_img)) * (max - \
            #       min)/(np.max(gen_img) - np.min(gen_img)) + min
            
            # new formula by Tabea (min and max are taken from output modality)
            gen_img = (out_max-out_min) * ((gen_img + 1) / 2) + out_min
            
            # re-orient
            if c.database == "PEGASUS":
                gen_img_new = np.rot90(np.flipud(gen_img), -1)
            elif c.database=="Heidelberg":
                gen_img_new = np.rot90(np.flipud(gen_img), 1) #np.rot90(np.flipud(gen_img), 1)
            gen_nii = nib.Nifti1Image(gen_img_new, affine=out_affine, header = out_header)

            # save nifty file
            if usedSpecific: # name when generating specific img
                nib.save(gen_nii, c.gen_img_path + "gen_{}_{}_{}.nii.gz".format(c.specificImgsToGen[i], c.output_seq[0], epoch))
            else:
                nib.save(gen_nii, c.gen_img_path + "gen_{}_{}.nii.gz".format(output_imgs[i], epoch))
            
            
    # save figures
    eut.save_losses(losses_g, losses_d, losses_val)


    # save results
    save_path = "{}data/models/TOF-perf/Trial_{}/gen_imgs/results.csv".format(c.root_path, c.trial_nr)
    if not os.path.isfile(save_path):
        with open(save_path, mode="w", newline="") as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(["epoch", "MSE", "NRMSE", "PSNR", "SSIM", "MAE"])
        
    with open(save_path, mode="a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow([epoch, np.mean(mselist), np.mean(nrmselist), 
                    np.mean(psnrlist), np.mean(ssimlist), 
                    np.mean(maelist)])

# get best epochs for each metric
df = pd.read_csv(save_path)
print("\nBest MSE of {} in epoch {}".format(df["MSE"].min(), df.loc[df["MSE"].idxmin()][0]))
print("Best NRMSE of {} in epoch {}".format(df["NRMSE"].min(), df.loc[df["NRMSE"].idxmin()][0]))
print("Best PSNR of {} in epoch {}".format(df["PSNR"].max(), df.loc[df["PSNR"].idxmax()][0]))
print("Best SSIM of {} in epoch {}".format(df["SSIM"].max(), df.loc[df["SSIM"].idxmax()][0]))
print("Best MAE of {} in epoch {}".format(df["MAE"].min(), df.loc[df["MAE"].idxmin()][0]))

total_time = time.time() - start_time
print('\nTime for complete training session: ',
      (total_time // (3600 * 24)), 'days',
      (total_time // 3600) % 24, 'hours',
      (total_time // 60) % 60, 'minutes',
      total_time % 60, 'seconds')