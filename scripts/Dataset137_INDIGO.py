# scrip processes output nifti images from BLAST ui and prepares them for inference
# by nnUNetv2 trained on brats data.

from multiprocessing import Pool
import nibabel as nb
from nibabel.processing import resample_from_to
import numpy as np
import itk
import matplotlib.pyplot as plt
import shutil
import re
import SimpleITK as sitk
import os


# operates on a single image channel 
def rescale(img_arr,vmin=None,vmax=None):
    scaled_arr =  np.zeros(np.shape(img_arr))
    if vmin is None:
        minv = np.min(img_arr)
    else:
        minv = vmin
    if vmax is None:
        maxv = np.max(img_arr)
    else:
        maxv = vmax
    assert(maxv>minv)
    scaled_arr = (img_arr-minv) / (maxv-minv)
    scaled_arr = np.clip(scaled_arr,a_min=0,a_max=1)
    return scaled_arr

def sitk_elastix_affine(image, template):
    img_sitk = sitk.GetImageFromArray(image)
    temp_sitk = sitk.GetImageFromArray(template)
    elastixImageFilter = sitk.ElastixImageFilter()  
    elastixImageFilter.SetFixedImage(img_sitk)  
    elastixImageFilter.SetMovingImage(temp_sitk)  
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))  
    # elastixImageFilter.PrintParameterMap(sitk.GetDefaultParameterMap('affine'))
    elastixImageFilter.Execute() 
    params = elastixImageFilter.GetTransformParameterMap()
    sitk.WriteImage(elastixImageFilter.GetResultImage(), 'reg.tif')  
    img_sitk_reg = sitk.ReadImage('reg.tif')  
    img_arr_reg = sitk.GetArrayFromImage(img_sitk_reg)  
    os.remove('reg.tif')  
    return img_arr_reg,params


def elastix_affine(image,template):
    parameter_object = itk.ParameterObject.New()
    if False:
        default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('affine')
    else:
        default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
    parameter_object.AddParameterMap(default_rigid_parameter_map)        
    image_reg, params = itk.elastix_registration_method(image, template,parameter_object=parameter_object)
    return image_reg,params

def get_itk_affine(params):
    rot00, rot01, rot02, rot10, rot11, rot12, rot20, rot21, rot22, tx, ty, tz = params.GetParameterMap(0)['TransformParameters']
    affine = np.array([
        [rot00, rot01, rot02, tx],
        [rot10, rot11, rot12, ty],
        [rot20, rot21, rot22, tz],
        [    0,     0,     0,  1],
    ], dtype=np.float32)  # yapf: disable
    return affine

def sitk_transform(img_arr,transformParameterMap):
    transformix = sitk.TransformixImageFilter()
    transformix.SetTransformParameterMap(transformParameterMap)
    transformix.SetMovingImage(sitk.GetImageFromArray(img_arr))
    transformix.Execute()
    img_arr_t = sitk.GetArrayFromImage(transformix.GetResultImage())
    return img_arr_t


if __name__ == '__main__':

    # resample and register data to match brats resolution and affine. can use any one brats image
    brats_data_dir = '/media/jbishop/WD4/brainmets/raw/_Dataset137_BraTS2021/imagesTs'
    img_brats = nb.load(os.path.join(brats_data_dir,'BraTS2021_00000_0000.nii.gz'))
    img_arr_brats = rescale(np.array(img_brats.dataobj))

    # input directory, root directory of a tree in which one or more cases have processed nifti files from ui are located
    ui_data_dir = '/home/jbishop/Data/blast/sunnybrook/SHSC3175076_INDIGO'
    # output directory, flat directory containing all cases that will be used by nnUNetv2 for predictions
    indigo_data_dir = '/media/jbishop/WD4/brainmets/raw/Dataset137_BraTS2021'
    dirs = os.listdir(ui_data_dir)
    for d in dirs:
        for root,dirs,files in os.walk(os.path.join(ui_data_dir,d),topdown=True):
            # if files are found it is assumed they are only the relevant nifti files.
            if len(files):
                if all(['nii' in f for f in files]):
                    # resample and register to brats
                    if False:
                        for f in files:
                            # resample to brats
                            img_nb = nb.load(os.path.join(root,f))
                            img_nb_arr = img_nb.dataobj
                            img_nb_res = resample_from_to(img_nb,(img_brats.shape,img_nb.affine))
                            img_arr_res = np.array(img_nb_res.dataobj)
                            # nb.save(img_nb_res,os.path.join(indigo_data_dir,'imagesTs',d+'_'+f))
                            moving_image = itk.GetImageFromArray(img_arr_res)
                            if 't1pre' in f:
                                # register to brats. itk elastix is failing on extracted brains even though it
                                # seems to work fine on intact brains
                                if False:
                                    fixed_image = itk.GetImageFromArray(img_arr_brats_00)
                                    moving_image_reg,rtp = elastix_affine(fixed_image,moving_image)
                                    img_arr_reg = itk.GetArrayFromImage(moving_image_reg)
                                # sitk-simpleelastix works better
                                else:
                                    img_arr_reg,rtp_sitk = sitk_elastix_affine(img_arr_brats,img_arr_res)
                                img_nb_res_reg = nb.Nifti1Image(img_arr_reg,affine=img_brats.affine)
                                nb.save(img_nb_res_reg,os.path.join(indigo_data_dir,'imagesTs',d+'_0000.nii.gz'))
                            elif 't1' in f:
                                img_arr_reg = sitk_transform(img_arr_res,rtp_sitk)
                                img_nb_res_reg = nb.Nifti1Image(img_arr_reg,affine=img_brats.affine)
                                nb.save(img_nb_res_reg,os.path.join(indigo_data_dir,'imagesTs',d+'_0001.nii.gz'))
                            elif 't2' in f:
                                img_arr_reg = sitk_transform(img_arr_res,rtp_sitk)
                                img_nb_res_reg = nb.Nifti1Image(img_arr_reg,affine=img_brats.affine)
                                nb.save(img_nb_res_reg,os.path.join(indigo_data_dir,'imagesTs',d+'_0002.nii.gz'))
                            elif 'flair' in f:
                                if False:
                                    rtp.GetParameter(0,'ResampleInterpolator')
                                    # rtp.SetParameter(0,'ResampleInterpolator','LinearInterpolator')
                                    moving_img_reg = itk.transformix_filter(moving_image,rtp)
                                    img_arr_reg =  itk.GetArrayFromImage(moving_img_reg)
                                else:
                                    img_arr_reg = sitk_transform(img_arr_res,rtp_sitk)
                                img_nb_res_reg = nb.Nifti1Image(img_arr_reg,affine=img_brats.affine)
                                nb.save(img_nb_res_reg,os.path.join(indigo_data_dir,'imagesTs',d+'_0003.nii.gz'))
                    else:      
                        # just copy the data unaltered
                        for f in files:
                            if 't1pre' in f:
                                shutil.copy(os.path.join(root,f),os.path.join(indigo_data_dir,'imagesTs_noresamp',d+'_0000.nii.gz'))
                            elif 't1' in f:
                                shutil.copy(os.path.join(root,f),os.path.join(indigo_data_dir,'imagesTs_noresamp',d+'_0001.nii.gz'))
                            elif 't2' in f:
                                shutil.copy(os.path.join(root,f),os.path.join(indigo_data_dir,'imagesTs_noresamp',d+'_0002.nii.gz'))
                            elif 'flair' in f:
                                shutil.copy(os.path.join(root,f),os.path.join(indigo_data_dir,'imagesTs_noresamp',d+'_0003.nii.gz'))

