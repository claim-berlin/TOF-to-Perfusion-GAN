"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
import numpy as np

def conv_output_shape(h_w_d, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size, kernel_size)
    h = floor( ((h_w_d[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w_d[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    d = floor( ((h_w_d[2] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w, d


def calc_Unet_inputShape(num_layers, output_shape):
    in_shape = np.power(output_shape, num_layers)
    return in_shape

def calc_Unet_num_layers(input_shape, output_shape):
    num_layers = np.log(input_shape)/np.log(output_shape)
    return num_layers