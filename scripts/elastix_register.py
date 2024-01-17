# script for fine-tune registering sparse DSC images to dense T1 images, 
# in which a BLAST ET segmentation from the T1 image is already
# in hand.

# sparse DSC image breaks elastix registration.

# matlab input file 'cpfile' is a list of control points selected 
# on T1 image that correspond by visual inspection to some nearly
# globular feature on the DSC image. The BLAST ET segmentation
# forms a final control point.

# procedure is to create a slightly larger spherical voi about each
# control point, and detect the DSC feature within that voi either by fitting a 3d
# gaussian or by 3d cross-correlation.

# the registration is then just the centroid difference of the collection
# of control points and matching DSC features

# a point cloud approach in which grayscale intensity is converted to point cloud
# density was also tried and could potentially work, but probably isn't
# necessary or suitable for this type of problem.

import itk
import nibabel as nb
from nibabel.processing import resample_from_to
import numpy as np
import scipy
from skimage.morphology import binary_dilation,ball
import matplotlib.pyplot as plt
import pickle
import copy

import os
import argparse

# modification of scipy.loadmat to work around nested structures in matlab
def loadmat(filename):
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

# load control points, create local mask volumes about each control point
# which will later be searched for a matching feature in the DSC CBV image.
# the control point selection is made with the matlab script radnec_dsc.m.
# the BLAST segmentation ET is also provided, and combined as a last
# control point
def load_cp(cpfile,ET):

    d = loadmat(cpfile)
    case = list(d['cp'].keys())[0]
    cp = d['cp'][case]
    n_cp = np.shape(cp)[0]

    cp_comp = np.zeros((n_cp+1,)+np.shape(ET))
    cp_comp[-1] = copy.deepcopy(ET)
    cp_mask_comp = np.zeros_like(cp_comp,dtype=float)
    cp_mask_comp[-1] = binary_dilation(cp_comp[-1],ball(5))
    cp_mask_comp[-1] = binary_dilation(cp_mask_comp[-1],ball(5)).astype(np.uint8)

    for i in range(n_cp):
        cp_mask = np.zeros_like(ET,dtype=float)
        cp_mask[cp[i,2],cp[i,1],cp[i,0]] = 1
        cp_mask = binary_dilation(cp_mask,ball(5))
        cp_comp[i] = cp_mask

        cp_mask_comp[i] = binary_dilation(cp_comp[i],ball(5))
        cp_mask_comp[i] = binary_dilation(cp_mask_comp[i],ball(5)).astype(np.uint8)

    return cp,cp_comp,cp_mask_comp

# save an array to nifti format.
def WriteImage(img_arr,filename,header=None,type='uint8',affine=None):
    img_arr_cp = copy.deepcopy(img_arr)
    img_nb = nb.Nifti1Image(np.transpose(img_arr_cp.astype(type),(2,1,0)),affine,header=header)
    nb.save(img_nb,filename)
    return
 
# for the control point approach, registration is just a centroid difference
def register(centroids,cp):
    T = np.mean(centroids,axis=0) - np.mean(cp,axis=0)

    return T

# cloud point registration. won't work unless number of cloud points
# in each control point cluster is identical.
def register_pc(pf,pm):
    if False:
    #Take transpose as columns should be the points
        pf = pf.transpose()
        pm = pm.transpose()

    #Calculate centroids
    pf_c = np.mean(pf, axis = 1).reshape((-1,1)) #If you don't put reshape then the outcome is 1D with no rows/colums and is interpeted as rowvector in next minus operation, while it should be a column vector
    pm_c = np.mean(pm, axis = 1).reshape((-1,1))

    #Subtract centroids
    q1 = pf-pf_c
    q2 = pm-pm_c

    #Calculate covariance matrix
    H=np.matmul(q1,q2.transpose())

    #Calculate singular value decomposition (SVD)
    U, X, V_t = np.linalg.svd(H) #the SVD of linalg gives you Vt

    #Calculate rotation matrix
    R = np.matmul(V_t.transpose(),U.transpose())

    assert np.allclose(np.linalg.det(R), 1.0), "Rotation matrix of N-point registration not 1, see paper Arun et al."

    #Calculate translation matrix
    T = pm_c - np.matmul(R,pf_c)

    #Check result
    result = T + np.matmul(R,pf)
    if np.allclose(result,pm):
        print("transformation is correct!")
    else:
        print("transformation is wrong...")        

# given point location in space, form a point cloud according to a gaussian density
# at that location
def get_pointcloud(cp,mask,n):

    (zm,ym,xm) = np.shape(mask)
    (z,y,x) = np.atleast_3d(list(range(zm)),list(range(ym)),list(range(xm)))
    z = np.transpose(z,axes=(1,0,2))
    x = np.transpose(x,axes=(0,2,1))

    gcomp = np.zeros_like(mask)
    for i in range(np.shape(cp)[0]):
        # note. control points from matlab are x,y,z, python is z,y,x
        g = np.exp(-( (x-cp[i][0])**2/(sigma**2)/2 + (y-cp[i][1])**2/(sigma**2)/2 + (z-cp[i][2])**2/(sigma**2)/2 ) )
        gcomp = gcomp + g
    gthresh = np.max(gcomp)
    g_pc_test = np.random.random_sample(np.shape(gcomp))*gthresh<gcomp
    n_pc = np.where(g_pc_test)[0].shape[0]

    # iterate to find an approximate match on the total number of points. 
    # however, each 
    while n_pc - n > 100:

        g_pc = g_pc_test
        gcomp = np.zeros_like(mask)
        for i in range(np.shape(cp)[0]):
            g = np.exp(-( (x-cp[i][0])**2/(sigma**2)/2 + (y-cp[i][1])**2/(sigma**2)/2 + (z-cp[i][2])**2/(sigma**2)/2 ) )
            gcomp = gcomp + g

        gthresh = np.max(gcomp)
        g_pc_test = np.random.random_sample(np.shape(gcomp))*gthresh<gcomp
        n_pc = np.where(g_pc_test)[0].shape[0]
        print(n_pc)
        sigma /= 1.1

    # randomly remove a few data points for an exact match
    n_pc = np.where(g_pc)[0].shape[0]
    idx = np.ravel(np.argsort(np.random.random(n_pc)))
    rdx = np.array(np.where(g_pc))[:,idx[0:n_pc-n]]
    g_pc[(rdx[0],rdx[1],rdx[2])]=0

    r_pc = np.array(np.where(g_pc))

    if False:
        plt.figure(1),plt.clf()
        plt.subplot(231)
        plt.imshow(gcomp[cp[i][2],:,:],vmax=1)
        plt.subplot(232)
        plt.imshow(gcomp[:,cp[i][1],:],vmax=1)
        plt.subplot(233)
        plt.imshow(gcomp[:,:,cp[i][0]],vmax=1)
        plt.subplot(234)
        plt.imshow(g_pc[cp[i][2],:,:],vmax=1)
        plt.subplot(235)
        plt.imshow(g_pc[:,cp[i][1],:],vmax=1)
        plt.subplot(236)
        plt.imshow(g_pc[:,:,cp[i][0]],vmax=1)
    return r_pc,g_pc


# def gauss3d(xyz, amp, z0, y0, x0, s):
def gauss3d(xyz, amp, z0, y0, x0):
    s=3
    z, y, x = xyz
    inner = ((x - x0)/s)**2
    inner += ((y - y0)/s)**2
    inner += ((z - z0)/s)**2
    return amp * np.exp(-inner)

# minimum bounding box for ET segmentation
def bbox3(img):
    rows = np.any(img,axis=(0,1))
    cols = np.any(img,axis=(0,2))
    slcs = np.any(img, axis=(1,2))
    zmin, zmax = np.where(slcs)[0][[0, -1]]
    ymin, ymax = np.where(cols)[0][[0, -1]]
    xmin, xmax = np.where(rows)[0][[0, -1]]
    return img[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1],(zmin,ymin,xmin)


def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',action = 'store_true',default=False)
    parser.add_argument('-f',type=str,help='fixed input file')
    parser.add_argument('-m',type=str,help='moving input file')
    parser.add_argument('-c',type=str,help='control points file')
    parser.add_argument('-t',type=str,help='t1 file')
    parser.add_argument('-o',type=str,help='output file')
    args = parser.parse_args()

    dpath = os.path.split(args.f)[0]
    if os.path.exists(os.path.join(dpath,'pc.pkl')):
        with open(os.path.join(dpath,'pc.pkl'),'rb') as fp:
            (pc_f,pc_m) = pickle.load(fp)
        if False:
            plt.scatter(np.sum(np.power(pc_m,2.0),axis=0),np.sum(np.power(pc_m,2.0),axis=0))
            plt.show()
    else:

        img_f = nb.load(args.f)
        img_arr_f = np.transpose(np.array(img_f.dataobj),axes=(2,1,0))
        img_t = nb.load(args.t)
        img_arr_t = np.transpose(np.array(img_t.dataobj),axes=(2,1,0))
        mname = os.path.split(args.m)[1]
        img_m = nb.load(args.m)
        img_arr_m = np.transpose(np.array(img_m.dataobj),axes=(2,1,0))

        # optional. reconcile affine between fixed and moving 
        if (img_f.affine != img_m.affine).any() and False:
            # img_m_res = resample_from_to(img_m,(img_f.shape,img_f.affine))
            # img_arr_m = np.transpose(np.array(img_m_res.dataobj),axes=(2,1,0))
            mpath = os.path.join(dpath,'affinefix_'+mname)
            WriteImage(img_arr_m,mpath,affine=img_f.affine)
            img_m = nb.load(mpath)
            img_arr_m = np.transpose(img_arr_m,axes=(2,1,0))

        # obtain matlab control points plus masks from ET segmentation
        cp,cp_comp,cp_comp_dilated = load_cp(args.c,img_arr_m)

        # mask each cluster of the CBV image and find indivdual centroid
        n_cp = np.shape(cp)[0]
        cbv_centroid = np.zeros((n_cp,3))
        for i in range(n_cp):
            im_mask = copy.deepcopy(img_arr_f)
            im_mask[cp_comp_dilated[i]==0] = 0
            if False: # point cloud option
                im_mask_pc = np.random.random_sample(np.shape(im_mask))*np.max(im_mask)<im_mask
                pc_set = np.array(np.where(im_mask_pc))
                cbv_centroid[i] = np.mean(pc_set,axis=1)
            else: # fit gaussian instead
                im_mask_guess = np.zeros_like(im_mask)
                im_mask_pred = np.zeros_like(im_mask)
                # didn't have good result with sigma as free parameter, but it ought to work
                # guess = [np.max(im_mask)]+list(np.ravel(np.array(np.where(im_mask == np.max(im_mask)))))+[3]
                # hard-coded sigma
                guess = [np.max(im_mask)]+list(np.ravel(np.array(np.where(im_mask == np.max(im_mask)))))
                zyx = np.where(im_mask)
                dvals = im_mask[zyx]
                dvals_guess = gauss3d(zyx,*guess)
                im_mask_guess[zyx] = dvals_guess
                pred_params, _ = scipy.optimize.curve_fit(gauss3d,zyx,dvals,p0=guess)
                dvals_pred = gauss3d(zyx,*pred_params)
                im_mask_pred[zyx] = dvals_pred
                cbv_centroid[i] = pred_params[1:]

        # find the centroid for the ET segmentation
        ET_kernel,_ = bbox3(img_arr_m)
        # mask out just the lesion
        im_mask_lesion = copy.deepcopy(img_arr_f)
        im_mask_lesion[cp_comp_dilated[-1] == 0] = 0
        # get bounding box for masked lesion
        CBV_bbox,minzyx = bbox3(im_mask_lesion)    

        # cross-correlate to register
        c = scipy.ndimage.correlate(CBV_bbox,ET_kernel)
        centroid = np.array(np.where(c==np.max(c))).T
        centroid += minzyx
        cbv_centroid = np.concatenate((cbv_centroid,centroid),axis=0)

        # mask the entire volume for registration
        img_arr_f[np.sum(cp_comp_dilated,axis=0)==0] = 0
        # record the centroid of the ET segmentation
        cp_ET = np.atleast_2d(np.mean(np.array(np.where(img_arr_m)),axis=1).astype(np.uint8))
        cp = np.concatenate((cp,cp_ET),axis=0)
        # flip matlab xyz to match python zyx
        cp = np.flip(cp,axis=1)

        if False:
            plt.subplot(141)
            plt.imshow(img_arr_m[65])
            plt.subplot(142)
            plt.imshow(img_arr_t[65])
            plt.subplot(143)
            plt.imshow(img_arr_f[65])
            plt.subplot(144)
            plt.imshow(cp_mask_comp[65])
            plt.show()
        WriteImage(np.sum(cp_comp,axis=0),os.path.join(dpath,'cp_comp'),affine=img_f.affine)
        WriteImage(img_arr_f,os.path.join(dpath,'img_arr_f'),affine=img_f.affine)

        if False:
            # point clouds
            pc_f_arr = np.random.random_sample(np.shape(img_arr_f))*np.max(img_arr_f)<img_arr_f
            pc_f = np.array(np.where(pc_f_arr))
            npc_f = np.shape(pc_f)[1]
            # calculate gaussian point clouds for the composite ET+control point mask
            pc_m,pc_m_arr = get_pointcloud(cp,cp_comp,npc_f)
            with open(os.path.join(dpath,'pc.pkl'),'wb') as fp:
                pickle.dump((pc_f,pc_m),fp)

            WriteImage(pc_f_arr,os.path.join(dpath,'pc_f_arr'),affine=img_f.affine)
            WriteImage(pc_m_arr,os.path.join(dpath,'pc_m_arr'),affine=img_f.affine)

    # registration
    if False:
        # itk elastix doens't work on sparse images or sparse masks of dense images
        if True:
            fixed_image = itk.GetImageFromArray(img_arr_f)
            moving_image = itk.GetImageFromArray(cp_comp)
        else:
            fixed_image = itk.GetImageFromArray(img_arr_t)
            moving_image = itk.GetImageFromArray(np.roll(img_arr_t,10,axis=1))
            if False:
                cp_mask_comp = np.ones_like(img_arr_t,dtype=np.uint8)
                cp_mask_comp[img_arr_t==0] = 0
        fixed_mask = itk.GetImageFromArray(cp_mask_comp)
        parameter_object = itk.ParameterObject.New()
        default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
        parameter_object.AddParameterMap(default_rigid_parameter_map)        
        image_reg, params = itk.elastix_registration_method(fixed_image, moving_image,
                                                            parameter_object=parameter_object,
                                                            fixed_mask=fixed_mask)
        img_arr_reg = itk.GetArrayFromImage(image_reg)
        img_reg = nb.Nifti1Image(np.transpose(img_arr_reg.astype(type),(2,1,0)),affine=img_f.affine,header=img_f.header)
        dpath = os.path.split(args.f)[0]
        nb.save(img_reg,os.path.join(dpath,args.o))
    else:
        if False:
            # point cloud registration wasn't fully correct
            t = register_pc(pc_f,pc_m)
        else:
            t = register(cbv_centroid,cp) # still in zyx
            affine_t = np.zeros((4,4))
            affine_t[0:3,3] = np.flip(t) # nb is xyz, so flip 
            affine_total = img_f.affine - affine_t
            img_nb_f_res = resample_from_to(img_f,(img_f.shape,affine_total))
            img_arr_f_res = np.transpose(img_nb_f_res.dataobj,axes=(2,1,0))
            WriteImage(img_arr_f_res,os.path.join(dpath,'img_arr_f_res'),affine=img_f.affine,type=float)

    return img_arr_f_res

if __name__ == '__main__':

    main()