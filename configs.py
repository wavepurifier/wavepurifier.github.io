import argparse
import logging
import yaml
import os
import time
import utils
from utils import str2bool
import torch
import numpy as np
import random

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # diffusion models
    parser.add_argument('--config', type=str, default='_256.yml', help='Path to the config file')
    parser.add_argument('--log_dir', type=str, default='logs', help='Path to the config file')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=6, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=1, help='Default noise scale')
    parser.add_argument('--t_s', type=int, default=56, help='Purify low frequency step')
    parser.add_argument('--t_m', type=int, default=40, help='Purify medium frequency step')
    parser.add_argument('--t_l', type=int, default=40, help='Purify high frequency step')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde]')
    parser.add_argument('--eot_iter', type=int, default=20, help='only for rand version of autoattack')
    parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for train')

    # LDSDE
    parser.add_argument('--sigma2', type=float, default=1e-3, help='LDSDE sigma2')
    parser.add_argument('--lambda_ld', type=float, default=1e-2, help='lambda_ld')
    parser.add_argument('--eta', type=float, default=5., help='LDSDE eta')
    parser.add_argument('--step_size', type=float, default=1e-2, help='step size for ODE Euler method')

    # adv
    parser.add_argument('--domain', type=str, default='imagenet', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--classifier_name', type=str, default='Eyeglasses', help='which classifier to use')
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--adv_batch_size', type=int, default=2)
    parser.add_argument('--attack_type', type=str, default='square')
    parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'])
    parser.add_argument('--attack_version', type=str, default='custom')

    parser.add_argument('--num_sub', type=int, default=1, help='imagenet subset')
    parser.add_argument('--adv_eps', type=float, default=0.07)
    # parser.add_argument('--gpu_ids', type=str, default='0')

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('config', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = utils.dict2namespace(config)

    # level = getattr(logging, args.verbose.upper(), None)
    # if not isinstance(level, int):
    #     raise ValueError('level {} not supported'.format(args.verbose))
    #
    # handler1 = logging.StreamHandler()
    # formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    # handler1.setFormatter(formatter)
    # logger = logging.getLogger()
    # logger.addHandler(handler1)
    # logger.setLevel(level)

    args.image_folder = os.path.join(args.exp, args.image_folder)
    os.makedirs(args.image_folder, exist_ok=True)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config