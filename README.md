### Model Configuration Readme
# 2022
## Workflow

1. **Data Preparation**
    - Use `preprocessing/generateSlicesForNetwork3D.py` to package images as slices.
        - Creates numpy slices for model input during training.
        - Saves metadata containing dataset extremes (min and max values).
        - Saves image IDs.
    - Optionally, create a smaller batch of slices for test images to be generated every X epochs to check visual quality.

2. **Training**
    - Run `train_3D.py` to train the model.

## Config 

- **input_seq**: The input sequence for the model.

- **output_seq**: The output sequence for the model.

- **database**: Specifies the database to be used.
  - Options: `"Heidelberg"` or `"PEGASUS"`

- **data**: Specifies the shorthand for the database.
  - Default: `["hdb"]` for Heidelberg, `["peg"]` for PEGASUS

- **model_path**: Directory path where models are saved.

- **gen_img_path**: Directory path where generated images are saved.

- **extremesDate**: Date on which the file with min/max values of the cohort was created.

- **gen_input_img_path_3D**: Path for input images used during generation.

- **gen_output_img_path_3D**: Path for output images used during generation.

- **extremesFile**: Filename for the extremes file.

- **specificImgsToGen**: Specific images to generate during training for visual checks.


## Model Settings

- **save_model**: Save model

- **save_every_X_epoch**: Frequency of saving the model.

- **generate_while_train**: Generate slices during model training.

- **nr_imgs_gen**: Number of images to generate (up to five).

- **sliceToShow**: Slice to generate as an example of model progression.

## Training Settings

- **epochs**: Number of training epochs.
  - Default: `80`

- **paddingMode**: Padding mode for convolutions.
  - Default: `"reflect"`

- **model_name**: Name of the model.
  - Default: `'pix2pix-pytorch'`

- **upsampling**: Use upsampling instead of convtranspose.
  - Default: `False`

- **batch_size**: Size of the training batches.
  - Default: `8`

- **ngf**: Number of generator filters.
  - Default: `256`

- **nr_layer_g**: Number of layers in the generator (UnetG).
  - Default: `7`

- **nr_layer_d**: Number of layers in the discriminator (PatchD).
  - Default: `3`

- **ndf**: Number of discriminator filters.
  - Default: `64`

- **niter**: Number of iterations.
  - Default: `10`

- **lr_g**: Learning rate for the generator.
  - Default: `0.0001`

- **lr_d**: Learning rate for the discriminator.
  - Default: `0.00001`

- **dropout_rate**: Dropout rate.
  - Default: `0`

- **weightinit**: Weight initialization method.
  - Default: `"normal"`

- **init_sd**: Initial standard deviation for normal distribution.
  - Default: `0.05`

- **init_mean**: Initial mean for normal distribution.
  - Default: `0`

- **slope_relu**: Slope for ReLU.
  - Default: `0.2`

- **patchD**: Use patch discriminator.
  - Default: `True`

- **kernel_size**: Kernel size (only for no patchD).
  - Default: `5`

- **netG_type**: Type of generator network.
  - Default: `"unet"`

- **pretrainedG**: Use pretrained generator (only for UnetG).
  - Default: `False`

- **feature_matching**: Use feature matching (only possible when no patchD).
  - Default: `False`

- **label_smoothing**: Use label smoothing.
  - Default: `False`

## Loss Functions

- **criterion_d**: Loss criterion for the discriminator.
  - Default: `"BCE"`

- **criterion_g**: Loss criterion for the generator.
  - Default: `"BCE"`

- **loss_ratio**: Ratio for updating loss.
  - Default: `False`

- **reconstruction_loss**: Type of reconstruction loss.
  - Default: `"L1-SSIM"`

## Optimizers

- **optimizer_d**: Optimizer for the discriminator.
  - Default: `"adam"`

- **beta1_d**: Beta1 parameter for the discriminator's optimizer.
  - Default: `0.5`

- **beta1_g**: Beta1 parameter for the generator's optimizer.
  - Default: `0.5`

- **beta2_d**: Beta2 parameter for the discriminator's optimizer.
  - Default: `0.999`

- **beta2_g**: Beta2 parameter for the generator's optimizer.
  - Default: `0.999`

## GAN Architecture

- **WGAN**: Use Wasserstein GAN.
  - Default: `False`

- **lbd**: Lambda parameter for WGAN.
  - Default: `10`

- **nr_d_train**: Number of discriminator training steps per generator step.
  - Default: `1`

## Continued Training

- **continue_training**: Continue training from a saved checkpoint.
  - Default: `False`

- **load_from_epoch**: Epoch number to load the model from.
  - Default: `20`

## Evaluation

- **metrics**: List of evaluation metrics.
  - Default: `["MSE", "NRMSE", "PSNR", "SSIM", "MAE"]`

- **save_nii**: Save results in NIfTI format.
  - Default: `True`

## Other

- **generate_images_3D_inference.py**: Run model in inference mode for test set.