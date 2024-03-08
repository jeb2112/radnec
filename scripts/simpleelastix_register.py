# command line script for running SimpleITK-SimpleElastix registration
# use simple-elastix conda env

import SimpleITK as sitk
import os
import argparse

def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',action = 'store_true',default=False)
    parser.add_argument('--options',default='0')
    parser.add_argument('-f',type=str,help='fixed input file')
    parser.add_argument('-m',type=str,help='moving input file')
    parser.add_argument('-o',type=str,help='output file')
    parser.add_argument('--resample',type=str,help='additional image to resample with registration transform')
    args = parser.parse_args()
    fixed = sitk.ReadImage(args.f)
    moving = sitk.ReadImage(args.m)

    elastixImageFilter = sitk.ElastixImageFilter()  
    elastixImageFilter.SetFixedImage(fixed)  
    elastixImageFilter.SetMovingImage(moving)  
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))  
    elastixImageFilter.PrintParameterMap(sitk.GetDefaultParameterMap('affine'))
    elastixImageFilter.Execute()  
    sitk.WriteImage(elastixImageFilter.GetResultImage(), args.o)  
    tform = elastixImageFilter.GetTransformParameterMap()

    # additional image to resample with registration transform
    if args.resample:
        tformixImageFilter = sitk.TransformixImageFilter()
        tformixImageFilter.SetTransformParameterMap(tform)
        tformixImageFilter.SetMovingImage(sitk.ReadImage(args.resample))
        tformixImageFilter.Execute()
        output_dir = os.path.split(args.o)[0]
        output_fname = os.path.split(args.resample)[1][:-4] + '_blastreg.nii'
        sitk.WriteImage(tformixImageFilter.GetResultImage(),os.path.join(output_dir,output_fname))

    return

if __name__ == '__main__':

    main()