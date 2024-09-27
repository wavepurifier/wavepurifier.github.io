import argparse
import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms as T
import io
import librosa
import numpy as np


def get_img(path):
    from PIL import Image

    # resp = requests.get('https://sparrow.dev/assets/img/cat.jpg')
    img = Image.open(path)

    preprocess = T.Compose([
        # T.Resize(256),
        # T.CenterCrop(256),
        T.ToTensor(),
        # T.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
    ])

    x = preprocess(img)
    return x


def audio2wav(path):
    y, sr = librosa.load(path, sr=16000)
    return y, sr


def amp_to_db(x):
    return 20.0 * np.log10(np.maximum(1e-5, x))


def normalize(S):
    return np.clip(S / 100, -1.0, 0.0) + 1.0


def wav2spec(wav):
    D = librosa.stft(wav, n_fft=512)
    S = amp_to_db(np.abs(D)) - 20
    S, D = normalize(S), np.angle(D)
    return S, D


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def denormalize(S):
    return (np.clip(S, 0.0, 1.0) - 1.0) * 100


def istft(mag, phase):
    stft_matrix = mag * np.exp(1j * phase)
    return librosa.istft(stft_matrix)


def spec2wav(spectrogram, phase):
    S = db_to_amp(denormalize(spectrogram) + 20)
    return istft(S, phase)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_data(args, adv_batch_size):
    if 'imagenet' in args.domain:
        val_dir = './dataset/imagenet_lmdb/val'  # using imagenet lmdb data
        val_transform = data.get_transform(args.domain, 'imval', base_size=224)
        val_data = data.imagenet_lmdb_dataset_sub(val_dir, transform=val_transform,
                                                  num_sub=args.num_sub, data_seed=args.data_seed)
        n_samples = len(val_data)
        val_loader = DataLoader(val_data, batch_size=n_samples, shuffle=False, pin_memory=True, num_workers=4)
        x_val, y_val = next(iter(val_loader))
    # elif 'myimg' in args.domain:
    #     data_dir = './dataset/myimg'

    elif 'cifar10' in args.domain:
        data_dir = './dataset'
        transform = transforms.Compose([transforms.ToTensor()])
        val_data = data.cifar10_dataset_sub(data_dir, transform=transform,
                                            num_sub=args.num_sub, data_seed=args.data_seed)
        n_samples = len(val_data)
        val_loader = DataLoader(val_data, batch_size=n_samples, shuffle=False, pin_memory=True, num_workers=4)
        x_val, y_val = next(iter(val_loader))
    elif 'celebahq' in args.domain:
        data_dir = './dataset/celebahq'
        attribute = args.classifier_name.split('__')[-1]  # `celebahq__Smiling`
        val_transform = data.get_transform('celebahq', 'imval')
        clean_dset = data.get_dataset('celebahq', 'val', attribute, root=data_dir, transform=val_transform,
                                      fraction=2, data_seed=args.data_seed)  # data_seed randomizes here
        loader = DataLoader(clean_dset, batch_size=adv_batch_size, shuffle=False,
                            pin_memory=True, num_workers=4)
        x_val, y_val = next(iter(loader))  # [0, 1], 256x256
    else:
        raise NotImplementedError(f'Unknown domain: {args.domain}!')

    print(f'x_val shape: {x_val.shape}')
    x_val, y_val = x_val.contiguous().requires_grad_(True), y_val.contiguous()
    print(f'x (min, max): ({x_val.min()}, {x_val.max()})')

    return x_val, y_val
