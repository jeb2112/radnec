# scrip iterates through a set of dicom directories to 
# pull out and check certain tags in the headers

import re
import os
import pydicom as pd

# get list of all image directories under the selected directory
def get_dicomdirs(dir):
    dcmdirs = []
    for root,dirs,files in os.walk(dir,topdown=True):
        if len(files):
            dcmfiles = [f for f in files if re.match('.*\.dcm',f.lower())]
            if len(dcmfiles):
                # for now assume that the parent of this dir is a series dir, and will take 
                # the dicomdir as the parent of the series dir
                # but for exported sunnybrook dicoms at least the more recognizeable dir might 
                # be two levels above at the date.
                dcmdirs.append(os.path.split(root)[0])

    # due to the intermediate seriesdirs, the above walk generates duplicates
    dcmdirs = list(set(dcmdirs))

    return dcmdirs

######
# main
######

if __name__ == '__main__':

    # brats source dir
    dicom_data_dir = os.path.join(os.path.expanduser('~'),'data','dicom')
    dcmdirs = sorted(get_dicomdirs(dicom_data_dir))

    for d in dcmdirs:

        print('case {}'.format(d))
        seriesdirs = os.listdir(d)

        for sd in seriesdirs:
            dpath = os.path.join(d,sd)
            files = sorted(os.listdir(dpath))

            ds0 = pd.dcmread(os.path.join(dpath,files[0]))
            if 'philips' in ds0.Manufacturer.lower():
                print(ds0.SeriesDescription)
                print('philips')