import numpy as np
import torch
import os
import nibabel as nib
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

import config as c
from utils import pytorch_ssim 

# collection of loss without generate imgs
def lossCollector(filename,ssim, l1, reconstructionLoss):
    losses = [str(ssim), str(l1), str(reconstructionLoss)]
    with open('temp/{}.txt'.format(filename), 'a') as f:
        for loss in losses:
            f.write(loss)
            f.write('\t')
        f.write('\n')

def load_dataset(data, input_seq, output_seq, datasplit="train", shffl=True):
    Tensor = torch.FloatTensor
    # INPUT
    for i in range(len(data)):
        for j in range(len(input_seq)):
            input_path = "{}{}_{}_{}_norm.npz".format(c.data_path, datasplit, 
                                                          input_seq[j], data[i])
            input_data = np.load(input_path)
            input_img = Tensor(input_data["imgs"])
            #if not input_seq[j].startswith("raw"):
            #    input_img = F.pad(input_img, (4, 4))
            if j==0:
                tmp_input = input_img
            else:
                tmp_input = torch.cat((tmp_input, input_data), 1)
        if i==0:
            final_input = tmp_input
        else:
            final_input = torch.cat((final_input, tmp_input), 0)
    print("Shape of {} input data: {}".format(datasplit, final_input.shape))
    # OUTPUT
    for i in range(len(data)):
        for j in range(len(output_seq)):
            output_path = "{}{}_{}_{}_norm.npz".format(c.data_path, datasplit, 
                                                           output_seq[j], data[i])
            output_data = np.load(output_path)
            output_img = Tensor(output_data["imgs"])
            #if not output_seq[j].startswith("raw"):
            #    output_img = F.pad(output_img, (4, 4))
            if j==0:
                tmp_output = output_img
            else:
                tmp_output = torch.cat((tmp_output, output_data), 1)
        if i==0:
            final_output = tmp_output
        else:
            final_output = torch.cat((final_output, tmp_output), 0)
    print("Shape of {} output data: {}".format(datasplit, final_output.shape))
    # put together
    dataset = TensorDataset(final_input, final_output)
    dataloader = DataLoader(dataset=dataset, num_workers=c.threads,
                            batch_size=c.batch_size, shuffle=shffl)
    return dataloader, final_input.shape[1], final_output.shape[1]


class Dataset3D:
    def __init__(self, data, input_seq, output_seq, datasplit="train"):
        self.input_path = "{}{}/{}/".format(c.data_path, datasplit, input_seq)
        self.output_path = "{}{}/{}/".format(c.data_path, datasplit, output_seq)
        print(self.input_path)
        print(self.output_path)
        self.in_files = sorted(os.listdir(self.input_path))
        self.out_files = sorted(os.listdir(self.output_path))


    def __len__(self):
        if len(self.in_files) != len(self.out_files):
            raise Exception("Input and output files do not have the same length.")
        return len(self.in_files)


    def __getitem__(self, idx):
        input_img_path = os.path.join(self.input_path, self.in_files[idx])
        output_img_path = os.path.join(self.output_path, self.out_files[idx])
        input_img = torch.FloatTensor(nib.load(input_img_path).get_fdata())
        #input_img = input_img.permute(3,0,1,2)  # time dimension first
        output_img = nib.load(output_img_path).get_fdata()
        output_img = torch.FloatTensor(output_img.copy())
        #norm_input_img = normalize(input_img)#torch.unsqueeze(normalize(input_img), 0)
        #norm_output_img = torch.unsqueeze(normalize(output_img), 0)
        #print(norm_input_img.shape, norm_output_img.shape)
        #return F.pad(norm_input_img,(1,2),value=-1), F.pad(norm_output_img,(1,2),value=-1)#.unsqueeze(1)
        return input_img, output_img#.unsqueeze(1)
    
    # get in and output channel number - notnecessary
    def getChannels(self):
        # load first shape to check all others against
        print(os.path.join(self.input_path, self.in_files[0]))
        input_ch_check = nib.load(os.path.join(self.input_path, self.in_files[0])).shape[1]
        for i in range(len(self.in_files)):
            input_img_path = os.path.join(self.input_path, self.in_files[i])
            input_ch = nib.load(input_img_path).shape[1]
            if input_ch_check != input_ch:
                raise Exception("Input channels differ!")

        # load first shape to check all others against
        output_ch_check = nib.load(os.path.join(self.output_path, self.out_files[0])).shape[1]
        for i in range(len(self.out_files)):
            output_img_path = os.path.join(self.output_path, self.out_files[i])
            output_ch = nib.load(output_img_path).shape[1]
            if output_ch_check != output_ch:
                raise Exception("Output channels differ!")
        return input_ch, output_ch


def load_3D_dataset(data, input_seq, output_seq, batch_size=4, datasplit="train", shffl=True):
    dataset = Dataset3D(data, input_seq, output_seq, datasplit)
    dataloader = DataLoader(dataset=dataset, num_workers=c.threads,
                            batch_size=batch_size, shuffle=shffl)
    #input_ch, output_ch = dataset.getChannels()
    return dataloader#, input_ch, output_ch  # change to not hardcoding


class Dataset3D_multi:
    def __init__(self, data, input_seq, output_seq, datasplit="train"):
        self.input_path = []
        self.in_files = []
        for seq in input_seq:
            self.input_path.append("{}{}/{}/".format(c.data_path, datasplit, seq))
            self.in_files.append(sorted(os.listdir(self.input_path[-1])))
        self.output_path = "{}{}/{}/".format(c.data_path, datasplit, output_seq)
        self.out_files = sorted(os.listdir(self.output_path))

    def __len__(self):
        if len(self.in_files) != len(self.out_files):
            pass
        return len(self.in_files[0])


    def __getitem__(self, idx):
        input_imgs = []
        for i in range(len(self.input_path)):
            input_img_path = os.path.join(self.input_path[i], self.in_files[i][idx])
            input_img = torch.unsqueeze(torch.FloatTensor(nib.load(input_img_path).get_fdata()), 0)
            input_imgs.append(input_img)

        output_img_path = os.path.join(self.output_path, self.out_files[idx])
        output_img = nib.load(output_img_path).get_fdata()
        output_img = torch.unsqueeze(torch.FloatTensor(output_img.copy()), 0)
        input_img = torch.cat(input_imgs, dim = 0)

        return input_img, output_img#.unsqueeze(1)
    
    # get in and output channel number - notnecessary
    def getChannels(self):
        # load first shape to check all others against
        print(os.path.join(self.input_path, self.in_files[0]))
        input_ch_check = nib.load(os.path.join(self.input_path, self.in_files[0])).shape[1]
        for i in range(len(self.in_files)):
            input_img_path = os.path.join(self.input_path, self.in_files[i])
            input_ch = nib.load(input_img_path).shape[1]
            if input_ch_check != input_ch:
                raise Exception("Input channels differ!")

        # load first shape to check all others against
        output_ch_check = nib.load(os.path.join(self.output_path, self.out_files[0])).shape[1]
        for i in range(len(self.out_files)):
            output_img_path = os.path.join(self.output_path, self.out_files[i])
            output_ch = nib.load(output_img_path).shape[1]
            if output_ch_check != output_ch:
                raise Exception("Output channels differ!")
        return input_ch, output_ch


def load_3D_dataset_multi(data, input_seq, output_seq, batch_size=1, datasplit="train", shffl=True):
    dataset = Dataset3D_multi(data, input_seq, output_seq, datasplit)
    dataloader = DataLoader(dataset=dataset, num_workers=c.threads,
                            batch_size=batch_size, shuffle=shffl)
    #input_ch, output_ch = dataset.getChannels()
    return dataloader#, input_ch, output_ch  # change to not hardcoding

    
# initialization of weights -> cant be too large (explosion) or too small (vanishing)
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        if c.weightinit=="xavier":
            # apply weight init function on .weight of net
            nn.init.xavier_uniform_(m.weight)
        elif c.weightinit=="normal":
            nn.init.normal_(m.weight, c.init_mean, c.init_sd)
        #nn.init.zeros_(m.bias)

# batchnorm weight init
def bn_init(m):
    classname = m.__class__.__name__

    if classname.find("BatchNorm") != -1: # applies batchnorm only on layers that have Batchnorm in the name
        nn.init.normal_(m.weight.data, 0.0, 0.05)
        nn.init.constant_(m.bias.data, 0)

# normalize data
def normalize(x):
    return 2 * ((x - torch.min(x)) / (torch.max(x) - torch.min(x))) - 1

def calculate_loss_d(net_d, optim_d, out_real, label_real, out_gen, label_gen):
    if c.criterion_d == "l1":
        real_loss = 0.5 * F.l1_loss(out_real, label_real)
    elif c.criterion_d == "l2":
        criterion = nn.MSELoss()
        real_loss = 0.5 * criterion(out_real, label_real)
    elif c.criterion_d == "hinge":
        criterion = nn.HingeEmbeddingLoss()
        real_loss = 0.5 * criterion(out_real, label_real)
    elif c.criterion_d == "ssim":
        criterion = pytorch_ssim.SSIM()
        real_loss = 1 - criterion(out_real, label_real)
    else:
        if c.patchD:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.BCELoss()
        real_loss = 0.5 * criterion(out_real, label_real)
    net_d.zero_grad()
    real_loss.backward(retain_graph=True)

    if c.criterion_d == "l1":
        gen_loss = 0.5 * F.l1_loss(out_gen, label_gen)
    else:
        gen_loss = 0.5 * criterion(out_gen, label_gen)

    gen_loss.backward(retain_graph=True)

    optim_d.step()
    # save loss of discriminator
    loss_d = (real_loss.detach() + gen_loss.detach()) / 2

    return loss_d


def calculate_WGAN_loss_d(net_d, optim_d, out_real, out_gen, real_pair,
                          gen_pair):
    real_loss = out_real.view(-1).mean() * (-1)
    real_loss.backward(retain_graph=True)
    gen_loss = out_gen.view(-1).mean()
    gen_loss.backward(retain_graph=True)

    eps = torch.rand(1).item()
    interpolate = eps * real_pair + (1 - eps) * gen_pair
    d_interpolate, _ = net_d(interpolate)

    # calculate gradient penalty
    grad_pen = wasserstein_grad_penalty(interpolate,
                                        d_interpolate,
                                        c.lbd)
    #print("grad pen",grad_pen)
    grad_pen.backward(retain_graph=True)
    optim_d.step()
    loss_d = real_loss.detach() + gen_loss.detach() + grad_pen.detach()

    return loss_d


def wasserstein_grad_penalty(interpolate, d_interpolate, lbd):
    grad_outputs = torch.ones_like(d_interpolate)
    gradients = autograd.grad(
        outputs=d_interpolate,
        inputs=interpolate,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = (gradients.norm(2) - 1) ** 2

    return gradient_penalty.mean() * lbd


def save_model(epoch, net_g, net_d, optim_g, optim_d, losses_g, losses_d):
    model_path = c.model_path + "model_epoch" + str(epoch) + ".pth"
    torch.save({
        "experiment_name": c.experimentName,
        "input": c.input_seq,
        "output": c.output_seq,
        "epoch": epoch,
        "lr_d": c.lr_d,
        "lr_g": c.lr_g,
        "beta1_d": c.beta1_d,
        "beta2_d": c.beta2_d,
        "beta1_g": c.beta1_g,
        "beta2_g": c.beta2_g,
        "device": c.gpu_idx,
        "batch_size": c.batch_size,
        "generator_state_dict": net_g.state_dict(),
        "discriminator_state_dict": net_d.state_dict(),
        "gen_opt_state_dict": optim_g.state_dict(),
        "discr_opt_state_dict": optim_d.state_dict(),
        "generator_loss": losses_g,
        "discriminator_loss": losses_d
    }, model_path)
