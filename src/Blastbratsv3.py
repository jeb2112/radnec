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

def run_blast(data,t1thresh,t2thresh,flairthresh,clustersize,layer,
            currentslice=None, maxZ=4,minZ=-4,gmthresh=None,wmthresh=None):

    #check if any thresholds have actually changed
    if all(data['blast']['gates'][x] is not None for x in [layer,'brain '+layer]):
        # raise ValueError('No updates for point-in-polygon requested.')
        print('No updates for point-in-polygon requested.')
        return None

    # hard-coded indexing
    t1 = data['raw'][0]
    flair = data['raw'][1]
    t2 = data['raw'][2]
    # t1_template = copy.deepcopy(t1)

    # Change precision
    flair = flair.astype('double')
    t1 = t1.astype('double')
    t2 = t2.astype('double')

    # Rescales images to values 0 to 1 and applies brain mask
    # flairstack = rescale(flair)
    # t1stack = rescale(t1)
    flairstack = flair
    t1stack = t1
    t2stack = t1

    braincluster_size = clustersize #sld4.Value
    t1Diff = braincluster_size*data['blast']['params']['stdt1'] 
    t2Diff = braincluster_size*data['blast']['params']['stdt2']
    flairDiff = braincluster_size*data['blast']['params']['stdflair']

    # Define brain
    # h = images.roi.Ellipse(gca,'Center',[meant2 meant1],'Semiaxes',[t2Diff t1Diff],'Color','y','LineWidth',1,'LabelAlpha',0.5,'InteractionsAllowed','none')

    # row=y=t1 col=x=t2, but xy convention for patches.Ellipse is same as matlab
    brain_perimeter = Ellipse((data['blast']['params']['meant2'],data['blast']['params']['meant1']),2*t2Diff,2*t1Diff)

    t2gate = data['blast']['params']['meant2']+(t2thresh)*data['blast']['params']['stdt2']
    t1gate = data['blast']['params']['meant1']+(t1thresh)*data['blast']['params']['stdt1']

    # define ET threshold gate >= (t2gate,t1gate). upper right quadrant
    if layer == 'ET':
        flairgate = data['blast']['params']['meant1flair']+(flairthresh)*data['blast']['params']['stdt1flair']
        yv_gate = np.array([t1gate, maxZ, maxZ, t1gate])
        xv_gate = np.array([flairgate, flairgate, maxZ, maxZ]) 
    elif layer == 'T2 hyper':
    # define NET threshold gate, >= ,flairgate,>= t2gate. upper right quadrant
        flairgate = data['blast']['params']['meant2flair']+(flairthresh)*data['blast']['params']['stdt2flair']
        yv_gate = np.array([t2gate, maxZ, maxZ, t2gate])
        xv_gate = np.array([flairgate, flairgate, maxZ, maxZ]) 

    #Creates a matrix of voxels for brain slice
    if currentslice is not None:
        startslice=currentslice
        endslice=currentslice+1
    else:
        startslice=min(np.nonzero(t1stack)[0])
        endslice=max(np.nonzero(t1stack)[0])+1

    # find the gated pixels
    domulti = False 
    stack_shape = np.shape(t1)
    fusion_stack_shape = (2,) + stack_shape + (3,)
    fusionstack = np.zeros(fusion_stack_shape)
    et_maskstack = np.zeros(stack_shape)
    wt_maskstack = np.zeros(stack_shape)
    c_maskstack = np.zeros(stack_shape)
    # gm_mask = np.zeros(stack_shape)
    # wm_mask = np.zeros(stack_shape)

    # TODO: config option for Win 11 if WSL2 enabled otherwise no gpu
    # option for multi-processing if doing 3d volume slice by slice
    # also not updated for multi-channel dimension yet
    # not updated recently
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

    # otherwise just use full 3d
    elif (endslice-startslice)>1:
        start = time.time()
        region_of_support = np.where(t1stack*flairstack*t2stack>0)
        background_mask = np.where(t1stack*flairstack*t2stack == 0)
        t1channel = t1stack[region_of_support]
        t2channel = t2stack[region_of_support]
        flairchannel = flairstack[region_of_support]
        if layer == 'ET':
            t1t2verts = np.vstack((flairchannel.flatten(),t1channel.flatten())).T
        else:
            t1t2verts = np.vstack((flairchannel.flatten(),t2channel.flatten())).T
        xy_layerverts = np.vstack((xv_gate,yv_gate)).T
        xy_layerverts = np.concatenate((xy_layerverts,np.atleast_2d(xy_layerverts[0,:])),axis=0) # close path
        unitverts = brain_perimeter.get_path().vertices
        xy_brainverts = brain_perimeter.get_patch_transform().transform(unitverts)
        # use pre-existing result if available
        brain_gate = data['blast']['gates']['brain '+layer]
        layer_gate = data['blast']['gates'][layer]

        if False:
            plt.figure(7)
            ax = plt.subplot(1,1,1)
            plt.scatter(t1t2verts[:,0],t1t2verts[:,1],c='b',s=1)
            plt.scatter(xy_layerverts[:,0],xy_layerverts[:,1],c='r',s=10)
            ax.set_aspect('equal')
            ax.set_xlim(left=0,right=1.0)
            ax.set_ylim(bottom=0,top=1.0)
            plt.text(0,1.02,'{:.3f},{:.3f}'.format(np.mean(t2channel),np.std(t2channel)))
            plt.savefig('/home/jbishop/Pictures/scatterplot.png')
            plt.clf()
            # plt.show(block=False)

        if use_gpu:
            RGcoords = cudf.DataFrame({'x':t1t2verts[:,0],'y':t1t2verts[:,1]}).interleave_columns()
            RGseries = GeoSeries.from_points_xy(RGcoords)
            xy_layercoords = cudf.DataFrame({'x':xy_layerverts[:,0],'y':xy_layerverts[:,1]}).interleave_columns()
            xy_layerseries = GeoSeries.from_polygons_xy(xy_layercoords,[0, np.shape(xy_layerverts)[0]],
                                                [0,1],[0,1] )
            xy_braincoords = cudf.DataFrame({'x':xy_brainverts[:,0],'y':xy_brainverts[:,1]}).interleave_columns()
            xy_brainseries = GeoSeries.from_polygons_xy(xy_braincoords,[0,np.shape(xyverts)[0]],[0,1],[0,1])

            # only redo the gates if required due to change in threshold
            if data['blast']['gates'][layer] is None:
                layer_gate = np.zeros(np.shape(t1stack),dtype='uint8')
                layergate_pts = dotime(point_in_polygon,(RGseries,xy_layerseries),txt='gate')
                layer_gate[region_of_support] = layergate_pts.values.get().astype('int').flatten()
            if data['blast']['gates']['brain '+layer] is None:
                brain_gate = np.zeros(np.shape(t1stack),dtype='uint8')
                braingate_pts = dotime(point_in_polygon,(RGseries,xy_brainseries),txt='brain')
                brain_gate[region_of_support] = braingate_pts.values.get().astype('int').flatten()
        else:
            brain_path = Path(xy_brainverts,closed=True)
            layer_path = Path(xy_layerverts,closed=True)
            if data['blast']['gates'][layer] is None:
                layer_gate = np.zeros(np.shape(t1stack),dtype='uint8')
                layer_gate[region_of_support] = layer_path.contains_points(t1t2verts).flatten()
            if data['blast']['gates']['brain '+layer] is None:
                brain_gate = np.zeros(np.shape(t1stack),dtype='uint8')
                brain_gate[region_of_support] = brain_path.contains_points(t1t2verts).flatten()

        dtime = time.time()-start
        print('polygon contains points time = {:.2f} sec'.format(dtime))
        layer_mask = np.logical_and(layer_gate, np.logical_not(brain_gate)) # 
        # et_mask = max(et_mask,0)
        # fusion = imfuse(et_mask,t1_template[slice,:,:,slice],'blend')

        # apply normal tissue mask
        if False:
            if data['probGM'] is not None:
                gm_mask[data['probGM'] > gmthresh] = 1
            if data['probWM'] is not None:
                wm_mask[data['probWM'] > wmthresh] = 1
            layer_mask = np.where(np.logical_and(layer_mask,gm_mask),False,layer_mask)
            layer_mask = np.where(np.logical_and(layer_mask,wm_mask),False,layer_mask)

        #se = strel('line',2,0) 
        #et_mask = imerode(et_mask,se) # erosion added to get rid of non
        #specific small non target voxels removed for ROC paper

        layer_maskstack = layer_mask

    # a single 2d slice. hasn't been updated recently
    else:
        for slice in range(startslice,endslice):  

            # Gating Routine    
            region_of_support = np.where(t1stack[slice] > 0)
            background_mask = np.where(t1stack[slice] == 0)
            t1channel = t1stack[slice][region_of_support]
            t2channel = flairstack[slice][region_of_support]
            
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
            et_gate_2d = np.zeros(np.shape(t1stack[slice]))
            et_gate_2d[region_of_support] = path_et.contains_points(t1t2verts).flatten()
            wt_gate_2d = np.zeros(np.shape(t1stack[slice]))
            wt_gate_2d[region_of_support] = path_wt.contains_points(t1t2verts).flatten()
            brain_2d = np.zeros(np.shape(t1stack[slice]))
            brain_2d[region_of_support] = path.contains_points(t1t2verts)

            # form output image
            et_mask = np.logical_and(et_gate_2d, np.logical_not(brain_2d)) # 
            et_mask = et_mask.reshape((240,240))
            et_mask[background_mask] == False
            wt_mask = np.logical_and(wt_gate_2d, np.logical_not(brain_2d)) # 
            wt_mask = wt_mask.reshape((240,240))
            wt_mask[background_mask] == False

            compound_mask = et_mask.astype('int')*2+wt_mask.astype('int')
            # fusion = imfuse(et_mask,t1_template[slice,:,:,slice],'blend')
            # if False:
            #     fusion = 0.5*et_mask + 0.5*t1_template[slice,:,:] # alpha composite unweighted
            # else:
                # fusion = OverlayPlots.generate_overlay(data['raw'][:,slice,:,:],compound_mask)

            #se = strel('line',2,0) 
            #et_mask = imerode(et_mask,se) # erosion added to get rid of non
            #specific small non target voxels removed for ROC paper

            et_maskstack[slice] = et_mask
            wt_maskstack[slice] = wt_mask
            c_maskstack[slice] = compound_mask

    return layer_maskstack,brain_gate,layer_gate


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
        # fusion = imfuse(et_mask,t1_template[slice,:,:,slice],'blend')
        if False:
            fusion = 0.5*et_mask + 0.5*t1stack # alpha composite unweighted
        else:
            fusion = OverlayPlots.generate_overlay(t1stack,et_mask)

        et_maskstack = np.frombuffer(np_x.get_obj(), dtype=np.float32).reshape(np_x_shape)
        fusionstack = np.frombuffer(np_y.get_obj(), dtype=np.float32).reshape(np_x_shape)
        et_maskstack[slice,:,:] = et_mask
        fusionstack[slice,:,:] = fusion

    except Exception as e:
        print('error with item: {}'.format(e))




