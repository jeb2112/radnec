import multiprocessing
import shutil
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
import re
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import os
if False:
    os.environ['nnUNet_raw'] = '/media/jbishop/WD4/brainmets/brats2024/raw'
    os.environ['nnUNet_predictions'] = '/media/jbishop/WD4/brainmets/brats2024/predictions'
    os.environ['nnUNet_preprocessed'] = '/media/jbishop/WD4/brainmets/brats2024/preprocessed'
    os.environ['nnUNet_results'] = '/media/jbishop/WD4/brainmets/brats2024/results'
from nnunetv2.paths import nnUNet_raw


def copy_BraTS_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    # looks like BraTS2024 is already stored as 0,1,2,3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 3]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 3] = 3
    seg_new[img_npy == 2] = 1 # still have to swap for WT TC
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)

# not edited for 2024
def convert_labels_back_to_BraTS(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2
    new_seg[seg == 3] = 4
    new_seg[seg == 2] = 1
    return new_seg


def load_convert_labels_back_to_BraTS(filename, input_folder, output_folder):
    a = sitk.ReadImage(join(input_folder, filename))
    b = sitk.GetArrayFromImage(a)
    c = convert_labels_back_to_BraTS(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, join(output_folder, filename))


def convert_folder_with_preds_back_to_BraTS_labeling_convention(input_folder: str, output_folder: str, num_processes: int = 12):
    """
    reads all prediction files (nifti) in the input folder, converts the labels back to BraTS convention and saves the
    """
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(load_convert_labels_back_to_BraTS, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))


if __name__ == '__main__':

    # brats_data_dir = '/media/jbishop/WD4/brainmets/raw/BraTS2021'
    brats_data_dir = '/media/jbishop/WD4/brainmets/brats2024/raw/training'

    task_id = 138
    task_name = "BraTS2024"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    imagests = join(out_base,'imagesTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(imagests)

    case_ids = subdirs(brats_data_dir, prefix='BraTS', join=False)

    for c in (case_ids):
        # val cases were 833-920 and don't have labels
        case_num = int(re.search('([0-9]{5})',c).group(1))
        print('case_num = {:d}'.format(case_num))
        if case_num <= 822:
            dstr = imagestr
            copy_BraTS_segmentation_and_convert_labels_to_nnUNet(join(brats_data_dir, c, c + "-seg.nii.gz"),
                                                                join(labelstr, c + '.nii.gz'))
        else:
            dstr = imagests
        shutil.copy(join(brats_data_dir, c, c + "-t1n.nii.gz"), join(dstr, c + '_0000.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "-t1c.nii.gz"), join(dstr, c + '_0001.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "-t2w.nii.gz"), join(dstr, c + '_0002.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "-t2f.nii.gz"), join(dstr, c + '_0003.nii.gz'))


    generate_dataset_json(out_base,
                          channel_names={0: 'T1', 1: 'T1ce', 2: 'T2', 3: 'Flair'},
                          labels={
                              'background': 0,
                              'whole tumor': (1, 2, 3),
                              'tumor core': (2, 3),
                              'enhancing tumor': (3, )
                          },
                          num_training_cases=len(case_ids),
                          file_ending='.nii.gz',
                          regions_class_order=(1, 2, 3),
                          license='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          reference='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          dataset_release='1.0')
