#!/usr/bin/env python

# script calculates radiomics features for all case sub-directories
# in data dir


from __future__ import print_function

import logging
import os
import re

import SimpleITK as sitk
import six

import radiomics
from radiomics import featureextractor, getFeatureClasses


def tqdmProgressbar():
  """
  This function will setup the progress bar exposed by the 'tqdm' package.
  Progress reporting is only used in PyRadiomics for the calculation of GLCM and GLSZM in full python mode, therefore
  enable GLCM and full-python mode to show the progress bar functionality

  N.B. This function will only work if the 'click' package is installed (not included in the PyRadiomics requirements)
  """
  global extractor

  radiomics.setVerbosity(logging.INFO)  # Verbosity must be at least INFO to enable progress bar

  import tqdm
  radiomics.progressReporter = tqdm.tqdm

def nifti2nrrd(dir,file,scale=1,odir=None):
    if odir is None:
        odir = dir
    img_nii = sitk.ReadImage(os.path.join(dir,file))
    img_nii = img_nii * float(scale)
    writer = sitk.ImageFileWriter()
    writer.SetImageIO('NrrdImageIO')
    file = file.replace('nii','nrrd')
    writer.SetFileName(os.path.join(odir,file))
    writer.Execute(img_nii)
    return os.path.join(odir,file)


# start logger
# Regulate verbosity with radiomics.verbosity
# radiomics.setVerbosity(logging.INFO)

# Get the PyRadiomics logger (default log-level = INFO
logger = radiomics.logger
logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

# Write out all log entries to a file
handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# load the data

if False:
    dataDir = '/media/jbishop/WD4/brainmets/sunnybrook/RAD NEC'
    inputDir = os.path.join(dataDir,'MRI')
    outputDir = os.path.join(dataDir,'MRI')
else:
    # for remote access of window dropbox, direct mount is easiest.
    if not os.path.isdir('/mnt/D'):
        os.system('sudo mount -t cifs //192.168.50.224/D /mnt/D -o rw,uid=jbishop,gid=jbishop,username=Chris\ Heyn\ Lab,vers=3.0')
    dataDir = '/mnt/D/Dropbox/BLAST DEVELOPMENT/RAD NEC'
    inputDir = os.path.join(dataDir,'MANUSCRIPT','RAD NEC MET RESULTS')
    outputDir = os.path.join(dataDir,'radiomics')

# requested features
paramsFile = os.path.abspath(os.path.join(os.path.dirname(__file__),'voxel_radiomics.yaml'))
# Initialize feature extractor using the settings file
extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
featureClasses = getFeatureClasses()
print("Active features:")
for cls, features in six.iteritems(extractor.enabledFeatures):
    if features is None or len(features) == 0:
        features = [f for f, deprecated in six.iteritems(featureClasses[cls].getFeatureNames()) if not deprecated]
    for f in features:
        print(f)
        # print(getattr(featureClasses[cls], 'get%sFeatureValue' % f).__doc__)

files = os.listdir(inputDir)
caseDirs = [f for f in files if os.path.isdir(os.path.join(inputDir,f))]

for c in caseDirs[43:]:
    print(c)
    cname = re.match('^(M00[0-9]{2}[a-b]?)',c).group(1)
    cdir = os.path.join(inputDir,c)
    odir = os.path.join(outputDir,cname)
    if not os.path.isdir(odir):
        os.mkdir(odir)
    cfiles = os.listdir(cdir)
    maskr = re.compile('^.*ET.nii')
    maskName = next(filter(maskr.match,cfiles))
    maskName = nifti2nrrd(cdir,maskName,odir=odir)
    for im in ['t1mprage','t2flair']:
        imageName = im+'_template.nii'
        imageName = nifti2nrrd(cdir,imageName,scale=1024,odir=odir)

        if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
            print('Error getting testcase!')
            exit()

        # Assumes the respective package is installed (not included in the requirements)
        tqdmProgressbar()

        featureVector = extractor.execute(imageName, maskName, voxelBased=True)

        for featureName, featureValue in six.iteritems(featureVector):
            if isinstance(featureValue, sitk.Image):
                if 'original' in featureName:
                    featureName = featureName.replace('original',im)
                sitk.WriteImage(featureValue, '%s.nrrd' % (os.path.join(odir, featureName)))
                # print('Computed %s, stored as "%s.nrrd"' % (featureName, featureName))
            else:
                continue
                print('%s: %s' % (featureName, featureValue))
