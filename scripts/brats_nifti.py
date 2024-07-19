# scrip copies and renames brats nifti images

# computes a z-score image for BLAST processing
# runs nnunet segmentation as well

from multiprocessing import Pool
import nibabel as nb
from nibabel.processing import resample_from_to
import numpy as np
import matplotlib.pyplot as plt
import shutil
import re
import os
import ants
import copy
from sklearn.cluster import KMeans
import subprocess


# calculate stats to create z-score images
def normalstats(img_arr_t1,img_arr_flair,event=None):

    X={}
    vset = {}

    region_of_support = np.where(img_arr_t1 * img_arr_flair >0)
    background = np.where(img_arr_t1 * img_arr_flair == 0)
    vset['flair'] = np.ravel(img_arr_flair[region_of_support])
    vset['t1'] = np.ravel(img_arr_t1[region_of_support])
    X = np.column_stack((vset['flair'],vset['t1']))

    np.random.seed(1)
    kmeans = KMeans(n_clusters=2,n_init='auto').fit(X)
    background_cluster = np.argmax(np.power(kmeans.cluster_centers_[:,0],2)+np.power(kmeans.cluster_centers_[:,1],2))

    std_t1 = np.std(X[kmeans.labels_==background_cluster,1])
    mu_t1 = np.mean(X[kmeans.labels_==background_cluster,1])
    std_flair = np.std(X[kmeans.labels_==background_cluster,0])
    mu_flair = np.mean(X[kmeans.labels_==background_cluster,0])

    if False:
        fig,ax = plt.figure(7),plt.clf()
        plt.scatter(X[kmeans.labels_==1-background_cluster,0],X[kmeans.labels_==1-background_cluster,1],c='b',s=1)
        plt.scatter(X[kmeans.labels_==background_cluster,0],X[kmeans.labels_==background_cluster,1],c='r',s=1)
        plt.gca().set_aspect('equal')
        plt.show(block=False)

    zimg_t1 = ( img_arr_t1 - mu_t1) / std_t1
    zimg_t1[background] = 0
    zimg_flair = (img_arr_flair - mu_flair) / std_flair
    zimg_flair[background] = 0

    return (zimg_t1,zimg_flair)


# load MNI reference and resample to current BraTS image
def loadseg(t1_file,dir,type=None):
    img_arr_t1 = None
    try:
        img_nb_t1 = nb.load(os.path.join(dir,t1_file))
    except IOError as e:
        print('Can\'t import {}'.format(t1_file))
        return None,None
    nb_header = img_nb_t1.header.copy()
    # nibabel convention will be transposed to sitk convention
    img_arr_t1 = np.transpose(np.array(img_nb_t1.dataobj),axes=(2,1,0))
    if type is not None:
        img_arr_t1 = img_arr_t1.astype(type)
    affine = img_nb_t1.affine

    return img_arr_t1,affine


# load a single nifti file
def loadnifti(t1_file,dir,type=None):
    img_arr_t1 = None
    try:
        img_nb_t1 = nb.load(os.path.join(dir,t1_file))
    except IOError as e:
        print('Can\'t import {}'.format(t1_file))
        return None,None
    nb_header = img_nb_t1.header.copy()
    # nibabel convention will be transposed to sitk convention
    img_arr_t1 = np.transpose(np.array(img_nb_t1.dataobj),axes=(2,1,0))
    if type is not None:
        img_arr_t1 = img_arr_t1.astype(type)
    affine = img_nb_t1.affine

    return img_arr_t1,affine


# write a single nifti file. use uint8 for masks 
def writenifti(img_arr,filename,header=None,norm=False,type='float64',affine=None):
    img_arr_cp = copy.deepcopy(img_arr)
    if norm:
        img_arr_cp = (img_arr_cp -np.min(img_arr_cp)) / (np.max(img_arr_cp)-np.min(img_arr_cp)) * norm
    # using nibabel nifti coordinates
    img_nb = nb.Nifti1Image(np.transpose(img_arr_cp.astype(type),(2,1,0)),affine,header=header)
    nb.save(img_nb,filename)
    if True:
        os.system('gzip --force "{}"'.format(filename))

# ants N4 bias correction
def n4bias(self,img_arr,shrinkFactor=4):
    print('N4 bias correction')
    data = copy.deepcopy(img_arr)
    dataImage = ants.from_numpy(img_arr)
    # ant mask must be float. 
    mask = np.zeros_like(data,dtype=float)
    mask[np.where(data > 0)] = 1
    maskImage = ants.from_numpy(mask)
    dataImage_n4 = ants.n4_bias_field_correction(dataImage,mask=maskImage,shrink_factor=shrinkFactor)
    img_arr_n4 = dataImage_n4.numpy()
    return img_arr_n4

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

# ants registration
def register(img_arr_fixed,img_arr_moving,transform='Affine'):
    print('register fixed, moving')

    fixed_ants = ants.from_numpy(img_arr_fixed)
    moving_ants = ants.from_numpy(img_arr_moving)
    mytx = ants.registration(fixed=fixed_ants, moving=moving_ants, type_of_transform = transform )
    img_arr_reg = mytx['warpedmovout'].numpy()

    return img_arr_reg,mytx['fwdtransforms']

# apply registration transform to another volume
def tx(img_arr_fixed,img_arr_moving,tx):
    print('transform fixed, moving')

    fixed_ants = ants.from_numpy(img_arr_fixed)
    moving_ants = ants.from_numpy(img_arr_moving)
    img_arr_tx = ants.apply_transforms(fixed_ants, moving_ants, tx).numpy()

    return img_arr_tx

# nnunet segmentation
def segment(C,ddir):
    ndir = os.path.join(ddir,'nnunet')
    if os.name == 'posix':
        command = 'conda run -n ptorch nnUNetv2_predict '
        command += ' -i ' + ndir
        command += ' -o ' + ndir
        command += ' -d137 -c 3d_fullres'
        res = os.system(command)
    elif os.name == 'nt':
        # manually escaped for shell. can also use raw string as in r"{}".format(). or subprocess.list2cmdline()
        # some problem with windows, the scrip doesn't get on PATH after env activation, so still have to specify the fullpath here
        # it is currently hard-coded to anaconda3/envs location rather than .conda/envs, but anaconda3 could be installed
        # under either ProgramFiles or Users so check both
        if os.path.isfile(os.path.expanduser('~')+'\\anaconda3\Scripts\\activate.bat'):
            activatebatch = os.path.expanduser('~')+"\\anaconda3\Scripts\\activate.bat"
        elif os.path.isfile("C:\Program Files\\anaconda3\Scripts\\activate.bat"):
            activatebatch = "C:\Program Files\\anaconda3\Scripts\\activate.bat"
        else:
            raise FileNotFoundError('anaconda3/Scripts/activate.bat')
        if os.path.isdir(os.path.expanduser('~')+'\\anaconda3\envs\\pytorch118_310'):
            envpath = os.path.expanduser('~')+'\\anaconda3\envs\\pytorch118_310'
        elif os.path.isdir(os.path.expanduser('~')+'\\.conda\envs\\pytorch118_310'):
            envpath = os.path.expanduser('~')+'\\.conda\envs\\pytorch118_310'
        else:
            raise FileNotFoundError('pytorch118_310')

        command1 = '\"'+activatebatch+'\" \"' + envpath + '\"'
        command2 = 'nnUNetv2_predict -i \"' + ndir + '\" -o \"' + ndir + '\" -d137 -c 3d_fullres'
        cstr = 'cmd /c \" ' + command1 + "&" + command2 + '\"'
        popen = subprocess.Popen(cstr,shell=True,stdout=subprocess.PIPE,universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline,""):
            if stdout_line != '\n':
                print(stdout_line)
        popen.stdout.close()
        res = popen.wait()
        if res:
            raise subprocess.CalledProcessError(res,cstr)
            print(res)
            
    sfile = C+ '.nii.gz'
    segmentation,affine = loadnifti(sfile,ndir)
    ET = np.zeros_like(segmentation)
    ET[segmentation == 3] = 1
    WT = np.zeros_like(segmentation)
    WT[segmentation > 0] = 1
    writenifti(ET,os.path.join(ddir,'ET_unet.nii'),affine=affine)
    writenifti(WT,os.path.join(ddir,'WT_unet.nii'),affine=affine)
    os.rename(os.path.join(ndir,sfile),os.path.join(ndir,C+'-unet.nii.gz'))


######
# main
######


if __name__ == '__main__':

    if os.name == 'posix':
        # brats source dir
        brats_data_dir = '/media/jbishop/WD4/brainmets/brats2024/raw/training'
        # nifti destination dir for BLAST
        blast_data_dir = '/media/jbishop/WD4/brainmets/sunnybrook/metastases/BraTS_2024'
    elif os.name == 'nt':
        # brats source dir
        brats_data_dir = "C:\\Users\\Chris Heyn Lab\\data\\brats2024\\raw\\training"
        # nifti destination dir for BLAST
        blast_data_dir = "C:\\Users\\Chris Heyn Lab\\data\\dicom2nifti_sam"
        # reference for talairach coords
        mni_data_dir = "C:\\Users\\Chris Heyn Lab\\data\\mni152"

    dref,dref_affine = loadnifti('mni_icbm152_t1_tal_nlin_sym_09a.nii',dir=mni_data_dir)
    mask,_ = loadnifti('mni_icbm152_t1_tal_nlin_sym_09a_mask.nii',dir=mni_data_dir)
    dref *= mask
    dref_nb = nb.load(os.path.join(mni_data_dir,'mni_icbm152_t1_tal_nlin_sym_09a.nii'))

    cases = sorted(os.listdir(brats_data_dir))

    # edit range of cases to process here
    for C in cases[:5]:
        print('case {}'.format(C))
        dir = os.path.join(brats_data_dir,C)
        files = os.listdir(dir)
        casenum = re.search('([0-9]{5})',C).group(1)
        ddir = os.path.join(blast_data_dir,'M'+casenum,'S0001')
        os.makedirs(ddir,exist_ok=True)
        ndir = os.path.join(ddir,'nnunet')
        os.makedirs(ndir,exist_ok=True)

        for f in files:
            if 't1c' in f:
                shutil.copy(os.path.join(dir,f),os.path.join(ddir,'t1+_processed.nii.gz'))
                img_arr_t1,affine_t1 = loadnifti('t1+_processed.nii.gz',ddir)
            elif 't2f' in f:
                shutil.copy(os.path.join(dir,f),os.path.join(ddir,'flair_processed.nii.gz'))
                img_arr_flair,affine_flair = loadnifti('flair_processed.nii.gz',ddir)
            elif 'seg' in f:
                shutil.copy(os.path.join(dir,f),ddir)
                img_arr_seg,_ = loadnifti(f,ddir)
                seg_fname = copy.copy(f)


        # register to the talairach coords
        # probably not needed BraTS data already registered
        if False:
            # resample mni ref image to image matrix and spacing
            dref_nb_tx = nb.processing.conform(dref_nb,out_shape = np.transpose(img_arr_t1,axes=(2,1,0)).shape,voxel_size=np.abs(np.diag(affine_t1)[:3]))
            img_arr_ref = np.transpose(np.array(dref_nb_tx.dataobj),axes=(2,1,0))
            # possible flip?
            img_arr_t1 = np.flip(img_arr_t1,axis=1)
            img_arr_flair = np.flip(img_arr_flair,axis=1)
            img_arr_seg = np.flip(img_arr_seg,axis=1)
            img_arr_t1,tx_ref = register(img_arr_ref,img_arr_t1,transform='Rigid')
            img_arr_flair = tx(img_arr_ref,img_arr_flair,tx_ref)
            img_arr_seg = tx(img_arr_ref,img_arr_seg,tx_ref)

            writenifti(img_arr_t1,os.path.join(ddir,'t1+_processed.nii'),affine=dref_nb_tx.affine)
            writenifti(img_arr_flair,os.path.join(ddir,'flair_processed.nii'),affine=dref_nb_tx._affine)
            writenifti(img_arr_seg,os.path.join(ddir,seg_fname),type='uint8',affine=dref_nb_tx.affine)

        # calculate z-score images
        zimg_t1,zimg_flair = normalstats(img_arr_t1,img_arr_flair)
        writenifti(zimg_t1,os.path.join(ddir,'zt1+_processed.nii'),affine=affine_t1)
        writenifti(zimg_flair,os.path.join(ddir,'zflair_processed.nii'),affine=affine_flair)

        # home-trained nnunet segmentation
        for dt,suffix in zip(['t1+','flair'],['0000','0003']):
            if os.name == 'posix':
                l1str = 'ln -s ' + os.path.join(ddir,dt+'_processed.nii.gz') + ' '
                l1str += os.path.join(ndir,C+'_'+suffix+'.nii.gz')
            elif os.name == 'nt':
                l1str = 'copy  \"' + os.path.join(ddir,dt+'_processed.nii.gz') + '\" \"'
                l1str += os.path.join(ndir,C+'_'+suffix+'.nii.gz') + '\"'
            os.system(l1str)
        segment(C,ddir)

