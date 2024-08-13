# scrip copies and renames brats nifti images

import shutil
import re
import os
import random


######
# main
######


if __name__ == '__main__':

    # brats source dir
    brats_data_dir = '/data/raw/Dataset138_BraTS2024'
    idir = os.path.join(brats_data_dir,'imagesTr')
    odir = os.path.join(brats_data_dir,'imagesTr_test')

    cases = os.listdir(os.path.join(brats_data_dir,'imagesTr'))
    for I in ['_0001','_0003']:
        clist = sorted([c for c in cases if re.search(I,c)])
        random.seed(10)
        random.shuffle(clist)

        # edit range of cases to process here
        for C in clist[:100]:

            shutil.move(os.path.join(idir,C),odir)

    idir = os.path.join(brats_data_dir,'labelsTr')
    odir = os.path.join(brats_data_dir,'labelsTr_test')
    labels = os.listdir(idir)
    llist = sorted(labels)
    random.seed(10)
    random.shuffle(llist)

    # edit range of cases to process here
    for L in llist[:100]:

        shutil.move(os.path.join(idir,L),odir)
           


