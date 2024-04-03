# command line script for running ants registration
# use simple-elastix conda env                

# example args for registering T1 from RADNEC and DSC studies, then using 
# the transform to register the DSC to the RADNEC space.
# it is assumed that the registration output should be written to the moving directory,
# and any additional files to be transformed are in the moving directory, and 
# written back out to the moving directory

# "-m/media/jbishop/WD4/brainmets/sunnybrook/RAD NEC/Figures/FIG 7/M0021 DSC/t1ce_cbv.nii",
# "-f/media/jbishop/WD4/brainmets/sunnybrook/RAD NEC/Figures/FIG 7/M0021 BLAST/t1ce.nii",
# "-ot1ce_CBV_reg.nii",
# "--resample","RELCBV_register.nii"



import ants
import SimpleITK as sitk
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
    parser.add_argument('--transform',type=str,help='rigid or affine',default='rigid')
    parser.add_argument('--iterations',type=int,help='number of iterations',default=256)
    parser.add_argument('--resample',nargs='+',type=str,help='additional image(s) to resample with registration transform')
    args = parser.parse_args()
    moving_dir = os.path.split(args.m)[0]

    # simple itk elastix
    fixed = sitk.ReadImage(args.f)
    moving = sitk.ReadImage(args.m)
    fixed_ants = ants.from_numpy(sitk.GetArrayFromImage(fixed).astype('double'))
    moving_ants = ants.from_numpy(sitk.GetArrayFromImage(moving).astype('double'))
    mytx = ants.registration(fixed=fixed_ants, moving=moving_ants, type_of_transform = 'Affine' )
    m_arr = mytx['warpedmovout'].numpy()
    m_sitk = sitk.GetImageFromArray(m_arr)
    m_sitk.CopyInformation(fixed)
    sitk.WriteImage(m_sitk, os.path.join(moving_dir,args.o))
    
    # store transform to file
    tform = ants.read_transform(mytx['fwdtransforms'][0])
    ants.write_transform(tform,os.path.join(moving_dir,'reg2blast_ants.txt'))

    # additional image to resample with registration transform
    # it is assumed these are in the dir of the moving image
    if args.resample:
        for f in args.resample:
            moving = sitk.ReadImage(os.path.join(moving_dir,f))
            m_ants = tform.apply_to_image(ants.from_numpy(sitk.GetArrayFromImage(moving)),reference=fixed_ants)
            m_arr = m_ants.numpy()
            output_fname = f[:-4] + '_blastreg.nii' # the suffix is hard-coded here.
            m_sitk = sitk.GetImageFromArray(m_arr)
            m_sitk.CopyInformation(fixed)
            sitk.WriteImage(m_sitk, os.path.join(moving_dir,output_fname))

    return

if __name__ == '__main__':

    main()