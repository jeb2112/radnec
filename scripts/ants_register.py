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

def do_registration(fixed,moving):
    mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform = 'Affine' )
    m_arr = mytx['warpedmovout'].numpy()
    m_sitk = sitk.GetImageFromArray(m_arr)
    # m_sitk.CopyInformation(fixed)

    # store transform to file
    tform = ants.read_transform(mytx['fwdtransforms'][0])
    # ants.write_transform(tform,os.path.join(moving_dir,'reg2blast_ants.txt'))
    return m_sitk,tform


def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',action = 'store_true',default=False)
    parser.add_argument('--options',default='0')
    parser.add_argument('-f',type=str,help='fixed input file')
    parser.add_argument('-m',type=str,help='moving input file')
    parser.add_argument('-d',type=str,help='directory of nifti case directories')
    parser.add_argument('-o',type=str,help='output file')
    parser.add_argument('--transform',type=str,help='rigid or affine',default='rigid')
    parser.add_argument('--iterations',type=int,help='number of iterations',default=256)
    parser.add_argument('--resample',nargs='+',type=str,help='additional image(s) to resample with registration transform')
    args = parser.parse_args()


    if args.m: # single file 
        fixed = sitk.ReadImage(args.f)
        fixed_ants = ants.from_numpy(sitk.GetArrayFromImage(fixed).astype('double'))
        moving_dir = os.path.split(args.m)[0]

        # simple itk elastix
        moving = sitk.ReadImage(args.m)
        moving_ants = ants.from_numpy(sitk.GetArrayFromImage(moving).astype('double'))

        m_sitk,tform = do_registration(fixed_ants,moving_ants)
        if args.o:
            outputfile = args.o
        else:
            outputfile = os.path.join(moving_dir,)
        sitk.WriteImage(m_sitk, os.path.join(moving_dir,args.o))

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

    # directory of nifi case dirs. for now it is assumed that each case has a designated reference image
    elif args.d: 
        moving_dir = args.d
        cases = sorted(os.listdir(args.d))

        for c in cases:
            print(c)
            casedir = os.path.join(args.d,c)
            sdir = os.path.join(casedir,os.listdir(casedir)[0]) # just one study for now
            flist = os.listdir(sdir)
            flist = [os.path.join(sdir,f) for f in flist if ('nii' in f and 'ref' not in f)]
            fixed_sitk = sitk.ReadImage(os.path.join(sdir,'t1+_reference.nii.gz'))
            fixed_ants = ants.from_numpy(sitk.GetArrayFromImage(fixed_sitk).astype('double'))

            for t in ['t1+','t1']:

                moving_reference = [f for f in flist if t in f]
                if len(moving_reference):
                    moving_reference = moving_reference[0]
                    reftag = t
                    flist = [f for f in flist if reftag not in f]
                    break

            moving_sitk = sitk.ReadImage(moving_reference)
            moving_ants = ants.from_numpy(sitk.GetArrayFromImage(moving_sitk).astype('double'))
            m_sitk,tform = do_registration(fixed_ants,moving_ants)
            m_sitk.CopyInformation(fixed_sitk)
            output_fname = os.path.split(moving_reference)[1][:-7] + '_reg.nii.gz' # the suffix is hard-coded here.
            sitk.WriteImage(m_sitk, os.path.join(sdir,output_fname))

            for f in flist:
                moving_sitk = sitk.ReadImage(f)
                m_ants = tform.apply_to_image(ants.from_numpy(sitk.GetArrayFromImage(moving_sitk)),reference=fixed_ants)
                m_arr = m_ants.numpy()
                output_fname = os.path.split(f)[1][:-7] + '_reg.nii.gz' # the suffix is hard-coded here.
                m_sitk = sitk.GetImageFromArray(m_arr)
                m_sitk.CopyInformation(fixed_sitk)
                sitk.WriteImage(m_sitk, os.path.join(sdir,output_fname))



    return

if __name__ == '__main__':

    main()