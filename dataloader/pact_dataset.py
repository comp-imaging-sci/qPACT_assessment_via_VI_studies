"""
PACT dataset loader.

This module defines the PyTorch Dataset used for training, validation, and testing
in the accompanying paper on 3D quantitative photoacoustic computed tomography (qPACT).

Provenance:
  - Portions of the original data-loading logic were adapted from upstream
    CSA/CS²-Net 3D training code by Lei Mou.
  - The code has been substantially modified, extended, and cleaned for this project
    by Refik Mert Cam (PhD candidate, ECE, UIUC).

Notes:
  - Data are stored as MATLAB v7.3 HDF5 (.mat) files.
  - Supports deterministic train/val/test splits based on filename conventions.
  - Includes optional data augmentation (e.g., rotations).

"""
from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import warnings
import numpy as np
from scipy.ndimage import rotate, map_coordinates, gaussian_filter

import h5py
from scipy.io import loadmat
from scipy.ndimage import zoom

warnings.filterwarnings('ignore')


def _read_id_list(txt_path):
    """Read a newline-delimited list of object IDs.

    Each line is expected to contain an object prefix (e.g., 'A40174514') with
    optional whitespace. Empty lines are ignored. Order is preserved and
    duplicates are removed (stable).
    """
    if txt_path is None:
        return None

    ids = []
    seen = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            # If a user accidentally included an extension, strip it.
            name = os.path.splitext(name)[0]
            if name not in seen:
                ids.append(name)
                seen.add(name)
    return ids


def load_dataset(
    root_dir="/shared/anastasio-s2/Phantom/Breast_phantom_UBP/",
    train=True,
    val=False,
    test=False,
    *,
    train_list_path="splits/train_image_names.txt",
    val_list_path="splits/val_image_names.txt",
    test_list_path="splits/test_image_names.txt",
    list_subdir="dataset_h_l_corrected",
    validate_files=False,
):
    """Return the object IDs for the requested split.

    By default, the split is defined by deterministic slicing over the available
    .mat files under ``root_dir/list_subdir`` (legacy behavior).

    If ``*_list_path`` arguments are provided, the split will instead be defined
    by those newline-delimited text files.

    Args:
        root_dir: Base directory containing the dataset subdirectories.
        train/val/test: Select which split to return (exactly one should be True).
        train_list_path/val_list_path/test_list_path: Paths to text files
            listing object IDs for each split (one ID per line).
        list_subdir: Subdirectory used to enumerate available IDs when lists are
            not provided (legacy behavior).
        validate_files: If True, drops IDs for which no corresponding .mat file
            exists under ``root_dir/list_subdir``.

    Returns:
        List[str]: Object IDs for the requested split.
    """

    # Prefer explicit split lists when provided.
    split_lists = {
        "train": _read_id_list(train_list_path),
        "val": _read_id_list(val_list_path),
        "test": _read_id_list(test_list_path),
    }

    split_key = "train" if train else ("val" if val else ("test" if test else None))
    if split_key is None:
        raise ValueError("One of train/val/test must be True.")

    if split_lists[split_key] is not None:
        ids = split_lists[split_key]
    else:
        # Legacy behavior: enumerate IDs from the directory and slice deterministically.
        directory = os.path.join(root_dir, list_subdir)
        file_list = os.listdir(directory)
        names_obj = {filename.split("_")[0] for filename in file_list if filename.endswith(".mat")}
        image_names = sorted(names_obj)

        images_train = image_names[8:40] + image_names[72:200] + image_names[232:360] + image_names[368:400]
        images_val = image_names[0:4] + image_names[40:56] + image_names[200:216] + image_names[360:364]
        images_test = image_names[4:8] + image_names[56:72] + image_names[216:232] + image_names[364:368]

        ids = images_train if train else (images_val if val else images_test)

    if validate_files:
        directory = os.path.join(root_dir, list_subdir)
        keep = []
        for name in ids:
            # At least one file with this prefix should exist.
            # Expected pattern: <ID>_*.mat
            found = any(fn.startswith(name + "_") and fn.endswith(".mat") for fn in os.listdir(directory))
            if found:
                keep.append(name)
        ids = keep

    return ids


def compute_hemisphere_mask(dim_x, dim_y, dim_z):
    # Volume dimensions
    volume_shape = (dim_x, dim_y, dim_z)
    voxel_size = 0.3  # mm
    radius_mm = 83  # mm

    # Convert radius from mm to voxels
    radius_voxels = radius_mm / voxel_size

    # Create a 3D grid of coordinates
    z, y, x = np.indices(volume_shape)

    # Calculate the center of the volume
    center_z = volume_shape[0]
    center_y = volume_shape[1] // 2
    center_x = volume_shape[2] // 2

    # Calculate the distance from the center for each voxel
    distances = np.sqrt((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2)

    # Create a mask for the hemisphere
    hemisphere_mask = (distances <= radius_voxels) & (z <= center_z)

    # Convert the mask to a binary mask (0s and 1s)
    hemisphere_mask = hemisphere_mask.astype(np.float32)
    return hemisphere_mask


class Data(Dataset):
    def __init__(self,
                 root_dir,
                 train=True,
                 val=False,
                 test=False,
                 flip=True,
                 random_crop=True,
                 scale1=512,
                 train_list_path=None,
                 val_list_path=None,
                 test_list_path=None,
                 validate_split_files=False):

        
        
        self.root_dir = root_dir
        self.train = train
        self.val = val
        self.test = test
        #self.rotate = rotate
        self.flip = flip
        self.random_crop = random_crop
        self.transform = transforms.ToTensor()
        self.resize = scale1
        self.images = load_dataset(
            self.root_dir,
            self.train,
            self.val,
            self.test,
            train_list_path=train_list_path,
            val_list_path=val_list_path,
            test_list_path=test_list_path,
            validate_files=validate_split_files,
        )
        self.hemisphere = compute_hemisphere_mask(256, 512, 512)

    def __len__(self):
        return len(self.images)


    def RandomRotate(self, image, angle):
        img_rotated = rotate(image, angle, axes=(2,1), reshape=False, order=0)
        '''
        img_rotated = np.zeros_like(image)
        for i in range(image.shape[0]):
            img_rotated[i] = rotate(image[i], angle, axes=(1, 0), reshape=False, order=0)
        '''
        return img_rotated

    def zeropad_to_power_of_2(self, array):
        dimensions = array.shape
        padded_dimensions = [int(2 ** np.ceil(np.log2(dim))) for dim in dimensions]

        padded_array = np.zeros(padded_dimensions, dtype=np.float32)
        # Calculate the starting indices to place the original array at the center
        start_indices = [(padded_dimensions[i] - dimensions[i]) // 2 for i in range(len(dimensions))]
        # Calculate the ending indices of the original array
        end_indices = [start_indices[i] + dimensions[i] for i in range(len(dimensions))]
        padded_array[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]] = array
        return padded_array

    def __getitem__(self, idx):
        # img_path = self.images[idx]
        obj_name = self.images[idx]
        #print(obj_name)

        root_dir_oxy = '/shared/anastasio-s2/Phantom/Breast_phantom_UBP/dataset_h_l_corrected'
        #root_dir_mask = '/shared/anastasio-s2/Phantom/Breast_phantom_UBP/dataset_segmentation_03'
        root_dir_mask = '/shared/aristotle/PACT/qPACT/mask_vessel_lesion_nc_vtc'
        root_dir_tr = '/shared/anastasio-s2/Phantom/Breast_phantom_UBP/dataset_h_l_time_reversal'
        #root_dir_tr_1 = '/shared/anastasio-s1/Phantom/Time_reversal_data/time_reversal_recon/'
        #root_dir_tr_2 = '/shared/anastasio-s2/Phantom/Breast_phantom_UBP/time_reversal_recons/'

        obj_path_oxy = os.path.join(root_dir_oxy, obj_name)
        obj_path_mask = os.path.join(root_dir_mask, obj_name)

        obj_path_tr = os.path.join(root_dir_tr, obj_name)

        #obj_path_tr_1 = os.path.join(root_dir_tr_1, obj_name)
        #obj_path_tr_2 = os.path.join(root_dir_tr_2, obj_name)

        try:
            if self.train or self.val:
                decision = random.random()
            
                if decision < 0.5: 
                    oxy = loadmat(obj_path_oxy + '_h_oxy.mat')
                    oxy = oxy['oxy']
                    oxy = oxy[27:283 ,27:539, 27:539]


                    p0_path = obj_path_tr + '_tr_h.mat'
                    p0 = loadmat(obj_path_tr + '_tr_h.mat')
                    p0_w757 = p0['pest_time_reversal_w757']
                    p0_w800 = p0['pest_time_reversal_w800']
                    p0_w850 = p0['pest_time_reversal_w850']

                    p0_w757 = np.transpose(p0_w757,[2,1,0])
                    p0_w757 = p0_w757[44::, 44:556, 44:556]

                    p0_w800 = np.transpose(p0_w800,[2,1,0])
                    p0_w800 = p0_w800[44::, 44:556, 44:556]

                    p0_w850 = np.transpose(p0_w850,[2,1,0])
                    p0_w850 = p0_w850[44::, 44:556, 44:556]

                    p0_w757 = self.hemisphere*p0_w757
                    p0_w800 = self.hemisphere*p0_w800
                    p0_w850 = self.hemisphere*p0_w850


                    mask = loadmat(obj_path_mask + '_mask_ves_les.mat')
                    mask = mask['mask_ves_les']
                    mask[mask==3] = 0
                    mask[mask==4] = 0

                else:
                    oxy = loadmat(obj_path_oxy + '_l_oxy.mat')
                    oxy = oxy['oxy']
                    oxy = oxy[27:283 ,27:539, 27:539]


                    p0_path = obj_path_tr + '_tr_l.mat'
                    p0 = loadmat(obj_path_tr + '_tr_l.mat')
                    p0_w757 = p0['pest_time_reversal_w757']
                    p0_w800 = p0['pest_time_reversal_w800']
                    p0_w850 = p0['pest_time_reversal_w850']

                    p0_w757 = np.transpose(p0_w757,[2,1,0])
                    p0_w757 = p0_w757[44::, 44:556, 44:556]

                    p0_w800 = np.transpose(p0_w800,[2,1,0])
                    p0_w800 = p0_w800[44::, 44:556, 44:556]

                    p0_w850 = np.transpose(p0_w850,[2,1,0])
                    p0_w850 = p0_w850[44::, 44:556, 44:556]

                    p0_w757 = self.hemisphere*p0_w757
                    p0_w800 = self.hemisphere*p0_w800
                    p0_w850 = self.hemisphere*p0_w850

                    mask = loadmat(obj_path_mask + '_mask_ves_les.mat')
                    mask = mask['mask_ves_les']

                random_number = random.randint(0, 19)
                ang = random_number*18

                if self.train or self.val:
                    p0_w757 = self.RandomRotate(p0_w757, ang)
                    p0_w800 = self.RandomRotate(p0_w800, ang)
                    p0_w850 = self.RandomRotate(p0_w850, ang)
                    oxy = self.RandomRotate(oxy, ang)
                    mask = self.RandomRotate(mask, ang)

                weight = np.zeros_like(mask)
                weight_bce = np.zeros_like(mask)

                if decision < 0.5:
                    pix_count_art = np.sum(mask == 1)
                    pix_count_vein = np.sum(mask == 2)
                    pix_count_bgd = np.sum(mask == 0)
                    pix_count = 256.*512.*512.
                    weight_bce[mask == 1] = pix_count/pix_count_art/4.
                    weight_bce[mask == 2] = pix_count/pix_count_vein/4.
                    weight_bce[mask == 0] = pix_count/pix_count_bgd
                else:
                    pix_count_art = np.sum(mask == 1)
                    pix_count_vein = np.sum(mask == 2)
                    pix_count_bgd = np.sum(mask == 0)
                    pix_count_vtc = np.sum(mask == 4)
                    pix_count_nec = np.sum(mask == 3)
                    pix_count = 256.*512.*512.
                    weight_bce[mask == 1] = pix_count/pix_count_art/4.
                    weight_bce[mask == 2] = pix_count/pix_count_vein/4.
                    weight_bce[mask == 0] = pix_count/pix_count_bgd
                    weight_bce[mask == 3] = pix_count/pix_count_nec/4.
                    weight_bce[mask == 4] = pix_count/pix_count_vtc/4.
                
                pix_count_unimportant = np.sum(mask == 0) + np.sum(mask == 3)
                pix_count_ves = np.sum(mask == 1) + np.sum(mask == 2)
                pix_count_vtc = np.sum(mask == 4)
                pix_count = 256.*512.*512.

                weight = np.zeros_like(mask)
                weight[mask==0] = pix_count/pix_count_unimportant
                weight[mask==1] = pix_count/pix_count_ves*10.            
                weight[mask==2] = pix_count/pix_count_ves*10.
                weight[mask==3] = pix_count/pix_count_unimportant          
                weight[mask==4] = pix_count/pix_count_vtc*10.
            
            else:
                #decision = 0. # healthy
                decision = 1. # unhealthy
            
                if decision < 0.5: 
                    oxy = loadmat(obj_path_oxy + '_h_oxy.mat')
                    oxy = oxy['oxy']
                    oxy = oxy[27:283 ,27:539, 27:539]


                    p0_path = obj_path_tr + '_tr_h.mat'
                    p0 = loadmat(obj_path_tr + '_tr_h.mat')
                    p0_w757 = p0['pest_time_reversal_w757']
                    p0_w800 = p0['pest_time_reversal_w800']
                    p0_w850 = p0['pest_time_reversal_w850']

                    p0_w757 = np.transpose(p0_w757,[2,1,0])
                    p0_w757 = p0_w757[44::, 44:556, 44:556]

                    p0_w800 = np.transpose(p0_w800,[2,1,0])
                    p0_w800 = p0_w800[44::, 44:556, 44:556]

                    p0_w850 = np.transpose(p0_w850,[2,1,0])
                    p0_w850 = p0_w850[44::, 44:556, 44:556]

                    p0_w757 = self.hemisphere*p0_w757
                    p0_w800 = self.hemisphere*p0_w800
                    p0_w850 = self.hemisphere*p0_w850


                    mask = loadmat(obj_path_mask + '_mask_ves_les.mat')
                    mask = mask['mask_ves_les']
                    mask[mask==3] = 0
                    mask[mask==4] = 0

                else:
                    oxy = loadmat(obj_path_oxy + '_l_oxy.mat')
                    oxy = oxy['oxy']
                    oxy = oxy[27:283 ,27:539, 27:539]


                    p0_path = obj_path_tr + '_tr_l.mat'
                    p0 = loadmat(obj_path_tr + '_tr_l.mat')
                    p0_w757 = p0['pest_time_reversal_w757']
                    p0_w800 = p0['pest_time_reversal_w800']
                    p0_w850 = p0['pest_time_reversal_w850']

                    p0_w757 = np.transpose(p0_w757,[2,1,0])
                    p0_w757 = p0_w757[44::, 44:556, 44:556]

                    p0_w800 = np.transpose(p0_w800,[2,1,0])
                    p0_w800 = p0_w800[44::, 44:556, 44:556]

                    p0_w850 = np.transpose(p0_w850,[2,1,0])
                    p0_w850 = p0_w850[44::, 44:556, 44:556]

                    p0_w757 = self.hemisphere*p0_w757
                    p0_w800 = self.hemisphere*p0_w800
                    p0_w850 = self.hemisphere*p0_w850

                    mask = loadmat(obj_path_mask + '_mask_ves_les.mat')
                    mask = mask['mask_ves_les']

                mask_tumor = mask.copy()
                mask_tumor[mask_tumor == 1] = 0
                mask_tumor[mask_tumor == 2] = 0
                mask_tumor[mask_tumor == 3] = 1
                mask_tumor[mask_tumor == 4] = 2

                random_number = random.randint(0, 19)
                ang = random_number*18

                if self.train or self.val:
                    p0_w757 = self.RandomRotate(p0_w757, ang)
                    p0_w800 = self.RandomRotate(p0_w800, ang)
                    p0_w850 = self.RandomRotate(p0_w850, ang)
                    oxy = self.RandomRotate(oxy, ang)
                    mask = self.RandomRotate(mask, ang)

                weight = np.zeros_like(mask)
                weight_bce = np.zeros_like(mask)

                if decision < 0.5:
                    pix_count_art = np.sum(mask == 1)
                    pix_count_vein = np.sum(mask == 2)
                    pix_count_bgd = np.sum(mask == 0)
                    pix_count = 256.*512.*512.
                    weight_bce[mask == 1] = pix_count/pix_count_art/4.
                    weight_bce[mask == 2] = pix_count/pix_count_vein/4.
                    weight_bce[mask == 0] = pix_count/pix_count_bgd
                else:
                    pix_count_art = np.sum(mask == 1)
                    pix_count_vein = np.sum(mask == 2)
                    pix_count_bgd = np.sum(mask == 0)
                    pix_count_vtc = np.sum(mask == 4)
                    pix_count_nec = np.sum(mask == 3)
                    pix_count = 256.*512.*512.
                    weight_bce[mask == 1] = pix_count/pix_count_art/4.
                    weight_bce[mask == 2] = pix_count/pix_count_vein/4.
                    weight_bce[mask == 0] = pix_count/pix_count_bgd
                    weight_bce[mask == 3] = pix_count/pix_count_nec/4.
                    weight_bce[mask == 4] = pix_count/pix_count_vtc/4.
                
                pix_count_unimportant = np.sum(mask == 0) + np.sum(mask == 3)
                pix_count_ves = np.sum(mask == 1) + np.sum(mask == 2)
                pix_count_vtc = np.sum(mask == 4)
                pix_count = 256.*512.*512.

                weight = np.zeros_like(mask)
                weight[mask==0] = pix_count/pix_count_unimportant
                weight[mask==1] = pix_count/pix_count_ves*10.            
                weight[mask==2] = pix_count/pix_count_ves*10.
                weight[mask==3] = pix_count/pix_count_unimportant          
                weight[mask==4] = pix_count/pix_count_vtc*10.


            n1, n2, n3 = p0_w757.shape
            
            norm = 0.005
            
            image = np.zeros((3, n1, n2, n3))
            image[0] = p0_w757/norm
            image[1] = p0_w800/norm
            image[2] = p0_w850/norm

            mask[mask==1] = 1
            mask[mask==2] = 1
            mask[mask==4] = 1
            mask[mask==3] = 0

            image = image.astype(np.float32)
            oxy = oxy.astype(np.float32)
            weight = weight.astype(np.float32)
            mask = mask.astype(np.float32)
            weight_bce = weight_bce.astype(np.float32)

            image = torch.from_numpy(np.ascontiguousarray(image))
            oxy = torch.from_numpy(np.ascontiguousarray(oxy)).unsqueeze(0)
            weight = torch.from_numpy(np.ascontiguousarray(weight)).unsqueeze(0)
            weight_bce = torch.from_numpy(np.ascontiguousarray(weight_bce)).unsqueeze(0)            
            mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)

            if self.train or self.val:
                return image, oxy, weight, mask, weight_bce
            if self.test:
                mask_tumor = torch.from_numpy(np.ascontiguousarray(mask_tumor)).unsqueeze(0)
                return image, oxy, weight, mask, weight_bce, mask_tumor


        except OSError as e:
            print("Error opening the file")
            print(p0_path)
            return self.__getitem__((idx + 1) % len(self))
