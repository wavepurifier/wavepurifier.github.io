# This is the runner of audio purify process
import matplotlib.pyplot as plt

from diffmodel import RevGuidedDiffusion
from configs import parse_args_and_config
from guided_diffusion.train_util import TrainLoop
from utils import *
import PIL
from PIL import Image
from numpy import asarray
import torch
import requests
import torchvision.transforms as T
import io
import librosa
import numpy as np
import wavio
from guided_diffusion.script_util import *
from guided_diffusion.image_datasets import ImageDataset, load_img_data
from scipy.io import wavfile
import noisereduce as nr
from pathlib import Path
import glob
import os


# [257, 419] => [256, 256] => [256, 256]

def purify_long_spec(AE_path, outpath):
    args, config = parse_args_and_config()
    runner_diff = RevGuidedDiffusion(args, config, device=config.device)  # Initialize DM
    y, sr = audio2wav(AE_path)  # Read AE
    S, P = wav2spec(y)  # Read Spectrogram and Phrase [F, T] => [257, 419]
    print (S.shape)
    nframes = S.shape[1]  # 419
    nwindow = nframes // 256  # 1
    if nwindow == 0:    # The to be purified spec is **shorter** than the window
        # Padding the S
        offset = 256-nframes  # pad start part
        pad = np.zeros((257, offset))
        S = np.concatenate((S, pad), axis=1)
        su = np.concatenate((S[:256, :], S[:256, :], S[:256, :]), axis=0).reshape(1, 3, 256, 256)  # Stack 3 layers
        su = torch.Tensor(su).contiguous()
        counter = 0
        AE_name = AE_path.split("/")[-1][:-4]
        tag = str(args.t) + "_" + AE_name
        # Purify the only window
        _ = runner_diff.image_editing_sample((su - 0.5) * 2, 0, bs_id=counter, tag=tag)
    else:
        restlength = nframes - nwindow * 256  # 163
        offset = 256 - restlength  # 93
        # If the length is longer than 256, then we build a list to purify them one by one
        sw = []  # record window of frames
        for w in range(nwindow):
            # Those are the complete window
            sw.append(S[:256, w * 256:w * 256 + 256])
        sw.append(S[:256, -256:])  # This is the last window
        AE_name = AE_path.split("/")[-1][:-4]
        window_id = 0
        for kwin in range(len(sw)):
            # Purify every window
            su = np.concatenate((sw[kwin], sw[kwin], sw[kwin]), axis=0).reshape(1, 3, 256, 256)  # Stack 3 layers
            su = torch.Tensor(su).contiguous()
            counter = 0
            tag = str(args.t_s) + "_" +str(args.t_m) + "_"+str(args.t_l) + "_" + AE_name
            # this step save the specs to ./logs/[tag]
            _ = runner_diff.image_editing_sample((su - 0.5) * 2, window_id, bs_id=counter, tag=tag) 
            window_id = window_id + 1
    return args, tag, nwindow, offset


def purify_long_audio(AE_path, outpath):
    # This will save the purifed spec in the logs/[tag]/samples_0.png
    args, tag, nwindow, offset = purify_long_spec(
        AE_path, outpath)
    purified_specs = "./logs/" + str(tag)
    for w in range(nwindow + 1):
        
        img_path_s = purified_specs + "/" + "samples_" + str(args.t_s) + "_" + str(w) + ".png"
        image_s = get_img(img_path_s)  # [C, F, T]

        img_path_m = purified_specs + "/" + "samples_" + str(args.t_m) + "_" + str(w) + ".png"
        image_m = get_img(img_path_m)

        img_path_l = purified_specs + "/" + "samples_" + str(args.t_l) + "_" + str(w) + ".png"
        image_l = get_img(img_path_l)

        # Reconstruct spec
        image = torch.cat([image_s[:,:64,:], image_m[:,64:128,:], image_l[:,128:,:]], dim=1)
        # image = image_s[3, :64, :] + image_s[3, :64, :]

        if w == 0:
            specs = image
        elif w == nwindow:  # This is the last window
            specs = np.concatenate((specs, image[:, :, offset:]), axis=2)
        else:
            specs = np.concatenate((specs, image), axis=2)
    if nwindow == 0:
        original_frame = 256 - offset
        specs = specs[0, :, :original_frame].reshape(specs.shape[1], original_frame)  # [F-1, T]
    else:
        specs = specs[0, :, :].reshape(specs.shape[1], specs.shape[2])  # [F-1, T]
    # print(specs.shape)
    y, sr = audio2wav(AE_path)
    S, P = wav2spec(y)
    # print(specs.shape, S.shape)
    su = np.concatenate((specs, S[256, :].reshape(1, -1)), axis=0)
    # print(su.shape)
    purified = spec2wav(su, P)
    diffoutpath = "out.wav"
    Path(os.path.dirname(diffoutpath)).mkdir(parents=True, exist_ok=True)
    # purified_path = "dataset/NC_attack/purified/" + "NP" + str(args.t) + "_" + AE_name
    wavio.write(diffoutpath, purified, 16000, sampwidth=2)
    # exit (0)
    denoise(diffoutpath, outpath)



def denoise(purified_path, outpath):
    # This algorithm can purify the NC attack audio So far
    # load data
    rate, data = wavfile.read(purified_path)
    reduced_noise = nr.reduce_noise(y=data, sr=rate, stationary=False)
    Path(os.path.dirname(outpath)).mkdir(parents=True, exist_ok=True)
    wavfile.write(outpath, rate, reduced_noise)




if __name__ == '__main__':
    AE_path = "dataset/NC_attack/attack/nc_3_1_1000.wav"
    outpath = "out_denoise.wav"
    # outpath_denoise = "../wavepurifier_open/out_denoise.wav"
    purify_long_audio(AE_path, outpath)
