# command line script for running SimpleITK-SimpleElastix registration

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
    return

if __name__ == '__main__':

    main()