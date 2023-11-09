import os
import numpy as np
from skimage.draw import ellipse_perimeter
from scipy.spatial.distance import dice
import copy
from matplotlib.path import Path
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import time
try:
    import cudf
    from cuspatial import point_in_polygon
    from cuspatial import GeoSeries
    use_gpu = True
except:
    print('No rapidsai GPU libraries.')
    use_gpu = False

import multiprocessing
import cc3d
from src import OverlayPlots


#################
# BLAST algorithm
#################

def run_blast(data,t1thresh,t2thresh,clustersize,
            currentslice=None):

    #check if any thresholds have actually changed
    if all(x is not None for x in data['gates'][0:3]):
        # raise ValueError('No updates for point-in-polygon requested.')
        print('No updates for point-in-polygon requested.')
        return None

    # hard-coded indexing
    t1mprage = data['raw'][0]
    t2flair = data['raw'][1]
    # t1mprage_template = copy.deepcopy(t1mprage)

    # Change precision
    t2flair = t2flair.astype('double')
    t1mprage = t1mprage.astype('double')

    # Rescales images to values 0 to 1 and applies brain mask
    # t2flairstack = rescale(t2flair)
    # t1mpragestack = rescale(t1mprage)
    t2flairstack = t2flair
    t1mpragestack = t1mprage

    braincluster_size = clustersize #sld4.Value
    t1Diff = braincluster_size*data['params']['stdt1'] 
    t2Diff = braincluster_size*data['params']['stdt2']

    # Define brain
    # h = images.roi.Ellipse(gca,'Center',[meant2 meant1],'Semiaxes',[t2Diff t1Diff],'Color','y','LineWidth',1,'LabelAlpha',0.5,'InteractionsAllowed','none')

    # row=y=t1 col=x=t2, but xy convention for patches.Ellipse is same as matlab
    brain_perimeter = Ellipse((data['params']['meant2'],data['params']['meant1']),2*t2Diff,2*t1Diff)

    t2gate = data['params']['meant2']+(t2thresh)*data['params']['stdt2']
    t2gate_count = (t2gate-data['params']['meant2'])/data['params']['stdt2']
    t1gate = data['params']['meant1']+(t1thresh)*data['params']['stdt1']
    t1gate_count = (t1gate-data['params']['meant1'])/data['params']['stdt1']

    # define ET threshold gate >= (t2gate,t1gate)
    # upper limit 4 hard-coded here
    yv_et = np.array([t1gate, 4, 4, t1gate])
    xv_et = np.array([t2gate, t2gate, 4, 4]) 

    # define NET threshold gate, >= t2gate
    yv_wt = np.array([-4, 4, 4, -4])
    xv_wt = np.array([t2gate, t2gate, 4, 4]) 

    #Creates a matrix of voxels for brain slice
    if currentslice is not None:
        startslice=currentslice
        endslice=currentslice+1
    else:
        startslice=min(np.nonzero(t1mpragestack)[0])
        endslice=max(np.nonzero(t1mpragestack)[0])+1

    # find the gated pixels
    domulti = False 
    stack_shape = (155,240,240)
    fusion_stack_shape = (2,155,240,240,3)
    # TODO: should become part of data, initialized with class, so 2d slice doesn't have to be recomputed
    # during interactive use
    fusionstack = np.zeros(fusion_stack_shape)
    et_maskstack = np.zeros(stack_shape)
    wt_maskstack = np.zeros(stack_shape)
    c_maskstack = np.zeros(stack_shape)
    brain = data['gates'][0]
    et_gate = data['gates'][1]
    wt_gate = data['gates'][2]

    # TODO: config option for Win 11 if WSL2 enabled otherwise no gpu
    # option for multi-processing if doing 3d volume slice by slice
    # also not updated for multi-channel dimension yet
    if domulti and (endslice-startslice)>1:
    # Compute ROIs based on Gates
        fusionstack_shared = multiprocessing.Array('f',[0.0]*np.prod(stack_shape))
        fusionstack = np.frombuffer(fusionstack_shared.get_obj(),dtype='float32').reshape(stack_shape)
        np.copyto(fusionstack,np.zeros(stack_shape))
        et_maskstack_shared = multiprocessing.Array('f',[0.0]*np.prod(stack_shape))
        et_maskstack = np.frombuffer(et_maskstack_shared.get_obj(),dtype='float32').reshape(stack_shape)
        np.copyto(et_maskstack,np.zeros(stack_shape))
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-4,initializer=pool_initializer,
                                    initargs=(metmaskstack_shared,fusionstack_shared,stack_shape))

        argslist = [(slice,data[:,slice,:,:],xv,yv,xv_et,yv_et) for slice in range(startslice,endslice)]
        pool.starmap_async(do_pip2d,argslist)
        pool.close()
        pool.join()

    # otherwise just use cuspatial gpu in 3d
    elif (endslice-startslice)>1:
        start = time.time()
        region_of_support = np.where(t1mpragestack>0)
        background_mask = np.where(t1mpragestack == 0)
        t1channel = t1mpragestack[region_of_support]
        t2channel = t2flairstack[region_of_support]
        t1t2verts = np.vstack((t2channel.flatten(),t1channel.flatten())).T
        xy_etverts = np.vstack((xv_et,yv_et)).T
        xy_etverts = np.concatenate((xy_etverts,np.atleast_2d(xy_etverts[0,:])),axis=0) # close path
        xy_wtverts = np.vstack((xv_wt,yv_wt)).T
        xy_wtverts = np.concatenate((xy_wtverts,np.atleast_2d(xy_wtverts[0,:])),axis=0) # close path
        unitverts = brain_perimeter.get_path().vertices
        xyverts = brain_perimeter.get_patch_transform().transform(unitverts)

        if use_gpu:
            RGcoords = cudf.DataFrame({'x':t1t2verts[:,0],'y':t1t2verts[:,1]}).interleave_columns()
            RGseries = GeoSeries.from_points_xy(RGcoords)
            xy_etcoords = cudf.DataFrame({'x':xy_etverts[:,0],'y':xy_etverts[:,1]}).interleave_columns()
            xy_etseries = GeoSeries.from_polygons_xy(xy_etcoords,[0, np.shape(xy_etverts)[0]],
                                                [0,1],[0,1] )
            xy_wtcoords = cudf.DataFrame({'x':xy_wtverts[:,0],'y':xy_wtverts[:,1]}).interleave_columns()
            xy_wtseries = GeoSeries.from_polygons_xy(xy_wtcoords,[0, np.shape(xy_wtverts)[0]],
                                                [0,1],[0,1] )
            xycoords = cudf.DataFrame({'x':xyverts[:,0],'y':xyverts[:,1]}).interleave_columns()
            xyseries = GeoSeries.from_polygons_xy(xycoords,[0,np.shape(xyverts)[0]],[0,1],[0,1])

            # only redo the gates if required due to change in threshold
            if data['gates'][1] is None:
                et_gate = np.zeros(np.shape(t1mpragestack),dtype='uint8')
                gate_etpts = dotime(point_in_polygon,(RGseries,xy_etseries),txt='gate')
                et_gate[region_of_support] = gate_etpts.values.get().astype('int').flatten()
            if data['gates'][2] is None:
                wt_gate = np.zeros(np.shape(t1mpragestack),dtype='uint8')
                gate_wtpts = dotime(point_in_polygon,(RGseries,xy_wtseries),txt='gate')
                wt_gate[region_of_support] = gate_wtpts.values.get().astype('int').flatten()
            if data['gates'][0] is None:
                brain = np.zeros(np.shape(t1mpragestack),dtype='uint8')
                brainpts = dotime(point_in_polygon,(RGseries,xyseries),txt='brain')
                brain[region_of_support] = brainpts.values.get().astype('int').flatten()
        else:
            path = Path(xyverts,closed=True)
            path_et = Path(xy_etverts,closed=True)
            path_wt = Path(xy_wtverts,closed=True)
            if data['gates'][1] is None:
                et_gate = np.zeros(np.shape(t1mpragestack),dtype='uint8')
                et_gate[region_of_support] = path_et.contains_points(t1t2verts).flatten()
            if data['gates'][2] is None:
                wt_gate = np.zeros(np.shape(t1mpragestack),dtype='uint8')
                wt_gate[region_of_support] = path_wt.contains_points(t1t2verts).flatten()
            if data['gates'][0] is None:
                brain = np.zeros(np.shape(t1mpragestack),dtype='uint8')
                brain[region_of_support] = path.contains_points(t1t2verts).flatten()

        dtime = time.time()-start
        print('polygon contains points time = {:.2f} sec'.format(dtime))
        et_mask = np.logical_and(et_gate, np.logical_not(brain)) # 
        # et_mask = max(et_mask,0)
        # fusion = imfuse(et_mask,t1mprage_template[slice,:,:,slice],'blend')

        #se = strel('line',2,0) 
        #et_mask = imerode(et_mask,se) # erosion added to get rid of non
        #specific small non target voxels removed for ROC paper

        wt_mask = np.logical_and(wt_gate, np.logical_not(brain))

        c_maskstack = et_mask.astype('int')*2 + wt_mask.astype('int')

        # maskstack = et_mask + wtmask
        # fusion = dotime(OverlayPlots.generate_overlay,(data['raw'],c_maskstack),txt='overlay') 
        # fusionstack = fusion
        et_maskstack = et_mask

    # a single 2d slice
    else:
        for slice in range(startslice,endslice):  

            # Gating Routine    
            region_of_support = np.where(t1mpragestack[slice] > 0)
            background_mask = np.where(t1mpragestack[slice] == 0)
            t1channel = t1mpragestack[slice][region_of_support]
            t2channel = t2flairstack[slice][region_of_support]
            
            # Applying gate to brain
            # gate = inpolygon(t2channel,t1channel,yv_et,xv_et)
            # brain = inpolygon(t2channel,t1channel,yv,xv)
            # t1t2verts = np.vstack((t2channel.flatten(),t1channel.flatten())).T
            t1t2verts = np.stack((t2channel*1,t1channel*1),axis=1)
            xy_etverts = np.vstack((xv_et,yv_et)).T
            xy_etverts = np.concatenate((xy_etverts,np.atleast_2d(xy_etverts[0,:])),axis=0) # close path
            xy_wtverts = np.vstack((xv_wt,yv_wt)).T
            xy_wtverts = np.concatenate((xy_wtverts,np.atleast_2d(xy_wtverts[0,:])),axis=0) # close path
            unitverts = brain_perimeter.get_path().vertices
            xyverts = brain_perimeter.get_patch_transform().transform(unitverts)
            path = Path(xyverts)
            if False:
                plt.scatter(xyverts[:,0],xyverts[:,1],c='g')
                plt.show(block=False)
            path_et = Path(xy_etverts,closed=True)
            path_wt = Path(xy_wtverts,closed=True)

            # find polygons
            # matplotlib. don't need gpu for 1 slice
            et_gate_2d = np.zeros(np.shape(t1mpragestack[slice]))
            et_gate_2d[region_of_support] = path_et.contains_points(t1t2verts).flatten()
            wt_gate_2d = np.zeros(np.shape(t1mpragestack[slice]))
            wt_gate_2d[region_of_support] = path_wt.contains_points(t1t2verts).flatten()
            brain_2d = np.zeros(np.shape(t1mpragestack[slice]))
            brain_2d[region_of_support] = path.contains_points(t1t2verts)

            # form output image
            et_mask = np.logical_and(et_gate_2d, np.logical_not(brain_2d)) # 
            et_mask = et_mask.reshape((240,240))
            et_mask[background_mask] == False
            wt_mask = np.logical_and(wt_gate_2d, np.logical_not(brain_2d)) # 
            wt_mask = wt_mask.reshape((240,240))
            wt_mask[background_mask] == False

            compound_mask = et_mask.astype('int')*2+wt_mask.astype('int')
            # fusion = imfuse(et_mask,t1mprage_template[slice,:,:,slice],'blend')
            # if False:
            #     fusion = 0.5*et_mask + 0.5*t1mprage_template[slice,:,:] # alpha composite unweighted
            # else:
                # fusion = OverlayPlots.generate_overlay(data['raw'][:,slice,:,:],compound_mask)

            #se = strel('line',2,0) 
            #et_mask = imerode(et_mask,se) # erosion added to get rid of non
            #specific small non target voxels removed for ROC paper

            et_maskstack[slice] = et_mask
            wt_maskstack[slice] = wt_mask
            c_maskstack[slice] = compound_mask

            # fusionstack[:,slice,:,:] = fusion

    # Calculate connected objects 
    # CC_labeled = bwlabeln(et_maskstack[slicestart:slicend,:,:],26) # beta edit on this line
    # CC_labeled = cc3d.connected_components(et_maskstack,connectivity=26)
    # stats = cc3d.statistics(CC_labeled)
    # stats = regionprops3(CC_labeled,"Volume","PrincipalAxisLength")
    # S = regionprops3(CC_labeled,'Centroid')
    # BB = regionprops3(CC_labeled, 'BoundingBox')

    # Display Volume
    # f1 = figure(1)
    # s = sliceViewer(fusionstack,"ScaleFactors",[2,2,1])
    return c_maskstack,[brain,et_gate,wt_gate,t1gate_count,t2gate_count]


#################
# utility methods
#################

# initialize variables for do_pip()
def pool_initializer(X,Y,X_shape):
    global np_x,np_y
    np_x = X
    np_y = Y
    global np_x_shape
    np_x_shape = X_shape

def dotime(func,args,txt=''):
    start = time.time()
    retval = func(*args)
    end = time.time()
    print ('dotime {} = {:.2f}'.format(txt,end-start))
    return retval

# polygon contains points for slice by slice iteration in parallel
# not recently updated
def do_pip2d(slice,channelstack,xv,yv,xv_et,yv_et):
    try:
        print(slice)
        # Gating Routine    
        t1channel = channelstack[0]
        t2channel = channelstack[1]
        
        # Applying gate to brain
        # gate = inpolygon(t2channel,t1channel,yv_et,xv_et)
        # brain = inpolygon(t2channel,t1channel,yv,xv)
        t1t2verts = np.vstack((t2channel.flatten(),t1channel.flatten())).T
        xy_etverts = np.vstack((yv_et,xv_et)).T
        xy_etverts = np.concatenate((xy_etverts,np.atleast_2d(xy_etverts[0,:])),axis=0) # close path
        xyverts =  np.vstack((yv,xv)).T
        xyverts = np.concatenate((xyverts,np.atleast_2d(xyverts[0,:])),axis=0) # close path
        path = Path(xyverts,closed=True)
        path2 = Path(xy_etverts)
        # find polygons
        # matplotlib
        gate = path2.contains_points(t1t2verts).reshape(240,240)
        brain = path.contains_points(t1t2verts).reshape(240,240)
        # form output image
        et_mask = np.logical_and(gate, np.logical_not(brain)) # 
        # et_mask = max(et_mask,0)
        # fusion = imfuse(et_mask,t1mprage_template[slice,:,:,slice],'blend')
        if False:
            fusion = 0.5*et_mask + 0.5*t1mpragestack # alpha composite unweighted
        else:
            fusion = OverlayPlots.generate_overlay(t1mpragestack,et_mask)

        et_maskstack = np.frombuffer(np_x.get_obj(), dtype=np.float32).reshape(np_x_shape)
        fusionstack = np.frombuffer(np_y.get_obj(), dtype=np.float32).reshape(np_x_shape)
        et_maskstack[slice,:,:] = et_mask
        fusionstack[slice,:,:] = fusion

    except Exception as e:
        print('error with item: {}'.format(e))




