# brain extraction and n4 correction of a set of nifti image dirs.
# testing for the differences or errors in N4 correction. logbias
# field seems to pick up some of the enhancement in M0001. not sure
# how well the mask is working because extrapolated logbias regions
# don't look convincing. can tumour be additionally masked.
# Jan 2024


import numpy as np
import nibabel as nb
import os
import matplotlib.pyplot as plt
import copy
import SimpleITK as sitk
import itk
import pickle
import sys


def extractbrain(input_nii):

    dpath,tfile = os.path.split(input_nii)
    ofile = tfile + '_brain.nii'
    mfile = tfile + '_mask.nii'
    tfile += '.nii'
    command = 'conda run -n brainmage brain_mage_single_run '
    command += ' -i \'' + os.path.join(dpath,tfile)
    command += '\' -o \'' + os.path.join(dpath,mfile)
    command += '\' -m \'' + os.path.join(dpath,ofile) + '\' -dev 0'
    res = os.system(command)
    os.remove(os.path.join(dpath,mfile))
    return res



def n4bias(img_arr,mask_arr=None,shrinkFactor=1,nFittingLevels=4):
    print('N4 bias correction, shrink factor {}'.format(shrinkFactor))
    data = copy.deepcopy(img_arr)
    dataImage = sitk.Cast(sitk.GetImageFromArray(data),sitk.sitkFloat32)
    sdataImage = sitk.Shrink(dataImage,[shrinkFactor]*dataImage.GetDimension())
    maskImage = sitk.Cast(sitk.GetImageFromArray(np.where(data,True,False).astype('uint8')),sitk.sitkUInt8)
    maskImage = sitk.Shrink(maskImage,[shrinkFactor]*maskImage.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    lowres_img = corrector.Execute(sdataImage,maskImage)
    log_bias_field = corrector.GetLogBiasFieldAsImage(dataImage)
    log_bias_field_arr = sitk.GetArrayFromImage(log_bias_field)
    corrected_img = dataImage / sitk.Exp(log_bias_field)
    corrected_img_arr = sitk.GetArrayFromImage(corrected_img)
    return corrected_img_arr,log_bias_field_arr

##############
# prepare data
##############


if os.system('mountpoint -q /mnt/D') != 0:
    # do this manually until i get key auth figured out
    # os.system('sudo mount -t cifs //192.168.50.224/D /mnt/D -o rw,uid=jbishop,gid=jbishop,username=Chris\ Heyn\ Lab,vers=3.0')
    a=1
radnecDir = '/mnt/D/Dropbox/BLAST DEVELOPMENT/RAD NEC/MANUSCRIPT DATA AND WORK/FIGURES/MISC/'
localDir = '/media/jbishop/WD4/brainmets/sunnybrook/RAD NEC/N4'
resultsfile = os.path.join(localDir,'N4_results.pkl')
    
files = os.listdir(radnecDir)
if False:
    caseDirs = [f for f in files if os.path.isdir(os.path.join(radnecDir,f))]
else:
    caseDirs = ['M0051FU']
mask_name = 'M0001_ET.nii'

for c in caseDirs:
    print(c)
    if os.path.exists(resultsfile) and len(caseDirs) == 1:
        with open(resultsfile,'rb') as fp:
            img_arr = pickle.load(fp)
        mask_nb = nb.load(os.path.join(localDir,mask_name))
        mask = np.transpose(np.array(mask_nb.dataobj),axes=(2,1,0))
    else:
        img_arr = {}
        for i,file in enumerate(['t1ce','t2flair_register']):

            fname = file + '_brain.nii'
            if not os.path.exists(os.path.join(radnecDir,c,fname)):
                extractbrain(os.path.join(radnecDir,c,file))
            img_nb = nb.load(os.path.join(radnecDir,c,fname))
            img_arr[file] = np.transpose(np.array(img_nb.dataobj),axes=(2,1,0))    # bias correction.
            mask = np.ones_like(img_arr[file])
            mask[np.where(img_arr[file] == 0)] = 0

            # further for troubleshooting N4 corrections
            if False:

                for shrinkfactor in [4]:
                    skey = file + '_sf' + str(shrinkfactor)
                    skey2 = skey + '_x2'
                    fname_out = skey + '.nii'
                    fname_out2 = skey2 + '.nii'
                    odir = os.path.join(localDir,c)
                    if not os.path.exists(odir):
                        os.mkdir(odir)
                    if not os.path.exists(os.path.join(odir,fname_out)):
                        # 1st iterate
                        img_arr[skey],img_arr[skey + '_logbias'] = n4bias(img_arr[file],mask,shrinkFactor=shrinkfactor)
                        img_nb_n4 = nb.Nifti1Image(np.transpose(img_arr[skey].astype('float'),(2,1,0)),img_nb.affine,header=img_nb.header)
                        # nb.save(img_nb_n4,os.path.join(radnecDir,c,fname_out))
                        img_nb_n4 = nb.Nifti1Image(np.transpose(img_arr[skey + '_logbias'].astype('float'),(2,1,0)),img_nb.affine,header=img_nb.header)
                        nb.save(img_nb_n4,os.path.join(odir,skey+'_logbias.nii'))
                        # 2nd iterate
                        if False:
                            img_arr[skey2],img_arr[skey2 + '_logbias'] = n4bias(img_arr[skey],shrinkFactor=shrinkfactor)
                            img_nb_n4 = nb.Nifti1Image(np.transpose(img_arr[skey2].astype('float'),(2,1,0)),img_nb.affine,header=img_nb.header)
                            nb.save(img_nb_n4,os.path.join(odir,fname_out2))
                            img_nb_n4 = nb.Nifti1Image(np.transpose(img_arr[skey2 + '_logbias'].astype('float'),(2,1,0)),img_nb.affine,header=img_nb.header)
                            nb.save(img_nb_n4,os.path.join(odir,skey2+'_logbias.nii'))
                    # os.remove(os.path.join(odir,fname))

            if False:
                with open(resultsfile,'wb') as fp:
                    pickle.dump(img_arr,fp)


sys.exit()

# plots
        
xylim = (0,800)

f1 = plt.figure(1,figsize=(8,5)),plt.clf()
f2 = plt.figure(2,figsize=(8,5)),plt.clf()

for i,d in enumerate(['t2flair_bias','t2flair_register_sf4','t2flair_register_sf1']):
    plt.figure(1)
    ax = plt.subplot(2,3,i+1)
    ax.set_aspect('equal')
    plt.scatter(np.ravel(img_arr['t2flair_register'])[::100],np.ravel(img_arr[d])[::100],1,marker='.')
    plt.plot([0,xylim[1]],[0,xylim[1]],c='orange')
    plt.xlim(xylim)
    plt.ylim(xylim)
    ax.xaxis.set_tick_params(labelbottom = False)
    ax.yaxis.set_tick_params(labelleft = False)
    plt.title(d)

    ax = plt.subplot(2,3,i+4)
    ax.set_aspect('equal')
    plt.scatter(img_arr['t2flair_register'][np.where(mask > 0)],img_arr[d][np.where(mask > 0)],1,marker='.')
    plt.plot([0,xylim[1]],[0,xylim[1]],c='orange')
    plt.xlim(xylim)
    plt.ylim(xylim)
    if i > 0:
        ax.xaxis.set_tick_params(labelbottom = False)
        ax.yaxis.set_tick_params(labelleft = False)
    plt.title('ET: {}'.format(d))

    plt.figure(2)
    if i != 0 and i != 3:
        ax = plt.subplot(2,3,i+1)
        ax.set_aspect('equal')
        plt.scatter(np.ravel(img_arr[d])[::100],np.ravel(img_arr[d+'_x2'])[::100],1,marker='.')
        plt.plot([0,xylim[1]],[0,xylim[1]],c='orange')
        plt.xlim(xylim)
        plt.ylim(xylim)
        ax.xaxis.set_tick_params(labelbottom = False)
        ax.yaxis.set_tick_params(labelleft = False)
        plt.title('x2: {}'.format(d))

        ax = plt.subplot(2,3,i+4)
        ax.set_aspect('equal')
        plt.scatter(img_arr[d][np.where(mask > 0)],img_arr[d+'_x2'][np.where(mask > 0)],1,marker='.')
        plt.plot([0,xylim[1]],[0,xylim[1]],c='orange')
        plt.xlim(xylim)
        plt.ylim(xylim)
        if i > 1:
            ax.xaxis.set_tick_params(labelbottom = False)
            ax.yaxis.set_tick_params(labelleft = False)
        plt.title('ET x2: {}'.format(d))



plt.tight_layout()
plt.figure(1)
plt.savefig(os.path.join(localDir,'N4_results.png'))
plt.figure(2)
plt.savefig(os.path.join(localDir,'N4_results_x2.png'))
plt.show()
a=1
