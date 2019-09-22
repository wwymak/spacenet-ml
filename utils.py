import rasterio
from torch.utils.tensorboard import SummaryWriter
import skimage
import geopandas as gpd
from matplotlib import pyplot as plt
from shapely.ops import cascaded_union
import solaris as sol

from tqdm import tqdm
from skimage.external import tifffile as sktif

# import shapely.wkt
import geopandas as gpd
import numpy as np
import cv2
from functools import partial

from fastai.imports import *
from fastai.vision import *
from fastai.metrics import dice
from fastai.callbacks import *

from joblib import Parallel, delayed
import torch.nn.functional as F
import torch
import functools, traceback


def scale_percentile(matrix):
    # scale tiff files read by tifffile to an rgb format readable by e.g. mpl for display
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


def create_mask(img_id, mask_geojson, reference_im_path, output_mask_folder, road_mask_width=20):
    outfile = output_mask_folder / f"{img_id}.png"
    reference_im = rasterio.open(str(reference_im_path))
    road_mask = np.zeros((1300, 1300))
    df = gpd.read_file(mask_geojson)
    if len(df) > 0:

        try:
            road_mask = sol.vector.mask.road_mask(df,
                                                  shape=(1300, 1300), reference_im=reference_im,
                                                  width=road_mask_width, meters=False, burn_value=burn_value,
                                                  out_type=int)


        except Exception as e:
            print(e, mask_fname)
            pass
    skimage.io.imsave(outfile, road_mask.astype('uint8'))

def create_small_tiles(img_filepath, mask_filepath, im_id, save_dir_rgb, save_dir_mask, new_img_height=512):
    img_rgb = sktif.imread(str(img_filepath))
    img_rgb = (255 * scale_percentile(img_rgb)).astype(np.uint8)

    mask = np.array(PIL.Image.open(mask_filepath))
    if mask.max() == 0:
        return

    rows, cols, channels = img_rgb.shape
    step_size =int( new_img_height / 3)

    for i in range(0,rows, step_size):
        for j in range(0, cols, step_size):
            if i + new_img_height > rows:
                i = rows-new_img_height
            if j + new_img_height > cols:
                j = cols-new_img_height
            im_arr = img_rgb[i: i+ new_img_height, j: j+new_img_height, :]
            mask_arr = mask[i: i+ new_img_height, j: j+new_img_height]
            if mask_arr.max() > 0:
                _ = PIL.Image.fromarray(im_arr)
                _.save(save_dir_rgb/ f"rgb_{new_img_height}_{im_id}_{i}_{j}.jpg")
                _= PIL.Image.fromarray(mask_arr)
                _.save(save_dir_mask / f"mask_{new_img_height}_{im_id}_{i}_{j}.png")


def get_random_crop_coords(img, new_h, new_w, n):
    h, w = img.shape[:2]
    if w == new_w and h == new_h:
        return 0, 0, h, w

    i_list = [random.randint(0, h - new_h) for i in range(n)]
    j_list = [random.randint(0, w - new_w) for i in range(n)]
    return i_list, j_list


# def generate_cropped_img_mask(img_id, n_crops_per_img=15, new_h=256, new_w=256, dataset_type="train"):
#     if dataset_type == "train":
#         cropped_dir = data_dir / "cropped_training"
#     else:
#         cropped_dir = data_dir / "cropped_validation"
#     instance_mask_fname = data_dir / "training" / f"{img_id}_GTI.tif"
#     mask_fname = data_dir / "training" / f"{img_id}_pytorch_GTL.tif"
#     img_fname = data_dir / "training" / f"{img_id}_RGB.tif"
#
#     img_inst_mask = sktif.imread(str(instance_mask_fname))
#     img_mask = sktif.imread(str(mask_fname))
#     img_rgb = sktif.imread(str(img_fname))
#
#     y_list, x_list = get_crop_coords(img_rgb, new_h, new_w, n_crops_per_img)
#
#     instance_masks = [img_inst_mask[i: i + new_h, j: j + new_w] for (i, j) in zip(y_list, x_list)]
#     masks = [img_mask[i: i + new_h, j: j + new_w] for (i, j) in zip(y_list, x_list)]
#     rgbs = [img_rgb[i: i + new_h, j: j + new_w] for (i, j) in zip(y_list, x_list)]
#
#     fnames_instance_masks = [cropped_dir / f"{img_id}_{idx}_GTI.tif" for idx in range(len(rgbs))]
#     fnames_masks = [cropped_dir / f"{img_id}_{idx}_pytorch_GTL.tif" for idx in range(len(rgbs))]
#     fnames_imgs = [cropped_dir / f"{img_id}_{idx}_RGB.tif" for idx in range(len(rgbs))]
#
#     [sktif.imsave(str(fname), mask) for mask, fname in zip(instance_masks, fnames_instance_masks)]
#     [sktif.imsave(str(fname), mask) for mask, fname in zip(masks, fnames_masks)]
#     [sktif.imsave(str(fname), img) for img, fname in zip(rgbs, fnames_imgs)]