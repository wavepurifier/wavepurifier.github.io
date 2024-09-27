# ---------------------------------------------------------------
# Taken from the following link as is from:
# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_GUIDED_DIFFUSION).
# ---------------------------------------------------------------

import math
import random

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import glob


def load_img_data(data_dir, batch_size):
    if not data_dir:
        raise ValueError("unspecified data directory")
    print("hello")
    all_files = _list_image_files_recursively(data_dir)
    print("hello")
    classes = None
    dataset = ImageDataset(
        all_files,
        classes=classes
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    # results = []
    results = glob.glob(data_dir+"/*.png")
    # print(data_dir+"/*.png")
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
    ):
        super().__init__()
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        # if self.random_crop:
        #     arr = random_crop_arr(pil_image, self.resolution)
        # else:
        #     arr = center_crop_arr(pil_image, self.resolution)

        # if self.random_flip and random.random() < 0.5:
        #     arr = arr[:, ::-1]

        arr = pil_image.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
