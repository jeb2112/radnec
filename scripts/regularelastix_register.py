# command line script for running ITK-Elastix registration
# use regular_elastix conda env                

# example args for registering T1 from RADNEC and DSC studies, then using 
# the transform to register the DSC to the RADNEC space.
# it is assumed that the registration output should be written to the moving directory,
# and any additional files to be transformed are in the moving directory, and 
# written back out to the moving directory

# "-m/media/jbishop/WD4/brainmets/sunnybrook/RAD NEC/Figures/FIG 7/M0021 DSC/t1ce_cbv.nii",
# "-f/media/jbishop/WD4/brainmets/sunnybrook/RAD NEC/Figures/FIG 7/M0021 BLAST/t1ce.nii",
# "-ot1ce_CBV_reg.nii",
# "--resample","RELCBV_register.nii"



import itk
import os
import argparse
import numpy as np
import nibabel as nb
from nibabel.processing import resample_from_to

def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',action = 'store_true',default=False)
    parser.add_argument('--options',default='0')
    parser.add_argument('-f',type=str,help='fixed input file')
    parser.add_argument('-m',type=str,help='moving input file')
    parser.add_argument('-o',type=str,help='output file')
    parser.add_argument('--resample',nargs='+',type=str,help='additional image(s) to resample with registration transform')
    args = parser.parse_args()
    moving_dir = os.path.split(args.m)[0]


    # regular itk elastix
    img_m = nb.load(args.m)
    img_arr_m = np.array(img_m.dataobj)
    img_arr_m = np.transpose(img_arr_m,axes=(2,1,0))
    img_f = nb.load(args.f)
    img_arr_f = np.array(img_f.dataobj)
    fixed_image = itk.GetImageFromArray(img_arr_f.astype(float))
    moving_image = itk.GetImageFromArray(img_arr_m.astype(float))
    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('affine')
    parameter_object.AddParameterMap(default_rigid_parameter_map)        
    image_reg, params = itk.elastix_registration_method(fixed_image, moving_image,
                                                        parameter_object=parameter_object)
    img_arr_reg = itk.GetArrayFromImage(image_reg)
    img_reg_nb = nb.Nifti1Image(np.transpose(img_arr_reg,(2,1,0)),affine=img_f.affine,header=img_f.header)
    nb.save(img_reg_nb,os.path.join(moving_dir,args.o))
    # store transform to file
    params.WriteParameterFile(params,os.path.join(moving_dir,'reg2blast.txt'))


    # additional image to resample with registration transform
    # it is assumed these are in the dir of the moving image
    if args.resample:
        for f in args.resample:
            img_m = nb.load(os.path.join(moving_dir,f))
            img_arr_m = np.array(img_m.dataobj)
            img_arr_m = np.transpose(img_arr_m,axes=(2,1,0))
            moving_image = itk.GetImageFromArray(img_arr_m.astype(float))
            image_reg = itk.transformix_filter(moving_image,params)
            img_arr_reg = itk.GetArrayFromImage(image_reg)
            output_fname = f[:-4] + '_blastreg.nii' # the suffix is hard-coded here.
            img_reg_nb = nb.Nifti1Image(np.transpose(img_arr_reg,axes=(2,1,0)),affine=img_f.affine,header=img_f.header)
            nb.save(img_reg_nb,os.path.join(moving_dir,output_fname))

    return

if __name__ == '__main__':

    main()