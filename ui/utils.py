from __future__ import print_function
from skimage import color
import inspect
import re
import numpy as np
import cv2
import os
import torch
from skimage import measure
import cv2 as cv
try:
    import pickle as pickle
except ImportError:
    import pickle


def debug_trace():
    from PyQt4.QtCore import pyqtRemoveInputHook
    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()


def PickleLoad(file_name):
    try:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
    except UnicodeDecodeError:
        with open(file_name, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    return data


def PickleSave(file_name, data):
    with open(file_name, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def CVShow(im, im_name='', wait=1):
    if len(im.shape) >= 3 and im.shape[2] == 3:
        im_show = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im_show = im

    cv2.imshow(im_name, im_show)
    cv2.waitKey(wait)
    return im_show


def average_image(imgs, weights):
    im_weights = np.tile(weights[:, np.newaxis, np.newaxis, np.newaxis], (1, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
    imgs_f = imgs.astype(np.float32)
    weights_norm = np.mean(im_weights)
    average_f = np.mean(imgs_f * im_weights, axis=0) / weights_norm
    average = average_f.astype(np.uint8)
    return average


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def grid_vis(X, nh, nw):  # [buggy]
    if X.shape[0] == 1:
        return X[0]

    # nc = 3
    if X.ndim == 3:
        X = X[..., np.newaxis]
    if X.shape[-1] == 1:
        X = np.tile(X, [1, 1, 1, 3])

    h, w = X[0].shape[:2]

    if X.dtype == np.uint8:
        img = np.ones((h * nh, w * nw, 3), np.uint8) * 255
    else:
        img = np.ones((h * nh, w * nw, 3), X.dtype)

    for n, x in enumerate(X):
        j = n // nw
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w, :] = x
    img = np.squeeze(img)
    return img

def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    if len(image_tensor.shape) == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()

    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0)) ),0, 1) * 255.0
    return image_numpy.astype(imtype)


# Converts a an image array (numpy) to tebsor
def im2tensor(input_image, type='rgb'):
    if isinstance(input_image, torch.Tensor):
        return input_image
    else:
        numpy_init = np.zeros((1, input_image.shape[2], input_image.shape[0], input_image.shape[1]))
        image_numpy = np.transpose(input_image, (2, 0, 1))
        # if type == 'unknown':
            # print(image_numpy)
        if type == 'rgb':
            image_numpy = np.clip(image_numpy, 0, 255)/255
        numpy_init[0, :, :, :] = image_numpy
        # image_tensor = torch.from_numpy(numpy_init)
        image_tensor = torch.tensor(numpy_init, dtype=torch.float32)
        # print(image_tensor.shape)
        # print(type(image_tensor))
        # print(image_tensor)
        return image_tensor
    # image_numpy = image_tensor[0].cpu().float().numpy()
    # if image_numpy.shape[0] == 1:
    #     image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0)) ),0, 1) * 255.0
    # return image_numpy.astype(imtype)


def rgb2lab(rgb):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:,[0],:,:]-50)/100
    ab_rs = lab[:,1:,:,:]/110
    out = torch.cat((l_rs,ab_rs),dim=1)
    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2lab')
        # embed()
    return out

# Color conversion code
def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2xyz')
        # embed()
    return out

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('xyz2lab')
        # embed()

    return out


def apply_smoothing(just_ab_asrgb):
    print('s', just_ab_asrgb.shape)
    # Kernel smoothing
    kernel = np.ones((5, 5), np.float32) / 25
    just_ab_asrgb_im = tensor2im(just_ab_asrgb)
    just_ab_asrgb_im_smoothed = np.asarray(cv.filter2D(just_ab_asrgb_im, -1, kernel)).astype(int)
    just_ab_smoothed_asab_tensor = im2tensor(just_ab_asrgb_im_smoothed)
    just_ab_smoothed_asab = rgb2lab(just_ab_smoothed_asab_tensor)
    just_ab_smoothed_asab = just_ab_smoothed_asab[0, 1:, :, :]

    return just_ab_smoothed_asab

def zhang_bins(just_ab_smoothed_asab):
    h = just_ab_smoothed_asab.shape[1]
    w = just_ab_smoothed_asab.shape[2]
    just_ab_smoothed_asab = torch.reshape(just_ab_smoothed_asab, (1, 2, h, w))
    encoded_ab = encode_ab_ind(just_ab_smoothed_asab)
    decoded_ab = my_decode_ind_ab(encoded_ab)
    return encoded_ab, decoded_ab

def bins_scimage_group_minimal(encoded):
    encoded_np = np.asarray(encoded[0, 0, :, :]).astype(int)
    img_labeled, num_labels = measure.label(encoded_np, connectivity=1, return_num=True)
    return img_labeled, num_labels

def encode_ab_ind(data_ab):
    # Encode ab value into an index
    # INPUTS
    #   data_ab   Nx2xHxW \in [-1,1]
    # OUTPUTS
    #   data_q    Nx1xHxW \in [0,Q)
    A = 2 * 110 / 10 + 1
    data_ab_rs = torch.round((data_ab*110 + 110)/10) # normalized bin number
    data_q = data_ab_rs[:,[0],:,:]*A + data_ab_rs[:,[1],:,:]
    return data_q


def my_decode_ind_ab(data_q):
    # Decode index into ab value
    # INPUTS
    #   data_q      Nx1xHxW \in [0,Q)
    # OUTPUTS
    #   data_ab     Nx2xHxW \in [-1,1]
    #
    # data_a = data_q/opt.A
    # data_b = data_q - data_a*opt.A
    A = 2 * 110 / 10 + 1
    assert isinstance(A, (int, float))
    # data_b = np.mod(data_q, opt.A)
    data_b = torch.fmod(data_q, A)
    data_a = (data_q-data_b)/A

    data_ab = torch.cat((data_a, data_b), dim=1)

    if data_q.is_cuda:
        type_out = torch.cuda.FloatTensor
    else:
        type_out = torch.FloatTensor
    data_ab = ((data_ab.type(type_out)*10) - 110)/110

    return data_ab