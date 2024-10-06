import os
import numpy as np
from skimage.draw import ellipse_perimeter
from scipy.spatial.distance import dice
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_dilation
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


# compute the bounding hull from a mask. 
def _compute_hull_from_mask(mask, original_size=None, hull_extension=0):
    mask2 = np.copy(mask)
    if hull_extension > 0:
        for h in range(hull_extension):
            mask2 = binary_dilation(mask2)
    coords = np.transpose(np.array(np.where(mask2 == 1)))
    hull = ConvexHull(coords)
    hcoords = np.transpose(np.vstack((coords[hull.vertices,1],coords[hull.vertices,0])))
    if False:
        plt.figure(7)
        plt.cla()
        plt.imshow(mask)
        plt.plot(hcoords[:,0],hcoords[:,1],'r.',markersize=5)
        plt.show(block=False)
        a=1
    return hcoords


#################
# BLAST algorithm
#################

def run_blast(data,blastdata,t12thresh,flairthresh,clustersize,layer,
            currentslice=None, maxZ=12):

    #check if any thresholds have actually changed
    if all(blastdata['blast']['gates'][x] is not None for x in [layer,'brain '+layer]):
        # raise ValueError('No updates for point-in-polygon requested.')
        print('No updates for point-in-polygon requested.')
        return None

    # will need to add t1pre and t2 at some point
    t1 = data.dset['z']['t1+']['d']
    flair = data.dset['z']['flair']['d']
    # t2 = data['raw'][2]

    # Change precision
    flair = flair.astype('double')
    t1 = t1.astype('double')

    # local copy
    flairstack = np.copy(flair)
    t1stack = np.copy(t1)
 
    # dummy since currently don't have plain T2 in radnec, but maybe this
    # might be needed in future
    t2stack = copy.deepcopy(t1)

    t12Diff = clustersize*blastdata['blast']['params'][layer]['stdt12'] 
    flairDiff = clustersize*blastdata['blast']['params'][layer]['stdflair']

    # row=y=t1 col=x=t2, but xy convention for patches.Ellipse is same as matlab
    brain_perimeter = Ellipse((blastdata['blast']['params'][layer]['meanflair'],
                               blastdata['blast']['params'][layer]['meant12']),2*flairDiff,2*t12Diff)

    flairgate = blastdata['blast']['params'][layer]['meanflair']+(flairthresh)*blastdata['blast']['params'][layer]['stdflair']
    t12gate = blastdata['blast']['params'][layer]['meant12']+(t12thresh)*blastdata['blast']['params'][layer]['stdt12']

    # elliptical gate for ET masking will be formed from a selected point(s) if there are
    # values, otherwise use the slider values for rectangular gate. 
    if layer == 'ET':
        bd = blastdata['blastpoint']['params'][layer]
        layer_paths = []
        # if points are available
        if (len(bd['meanflair']) > 0):
            pointclustersize = 1

            # a. convex hull comprising all points. not using this for now
            if False:
                combined_pointverts = None
                for mu_t1,mu_flair,std_t1,std_flair in zip(bd['meant12'],bd['meanflair'],bd['stdt12'],bd['stdflair']):
                    # currently pointclustersize is hard-coded here
                    point_perimeter = Ellipse((mu_flair, mu_t1), 2*pointclustersize*std_flair,2*pointclustersize*std_t1)
                    unitverts = point_perimeter.get_path().vertices
                    pointverts = point_perimeter.get_patch_transform().transform(unitverts)
                    if combined_pointverts is None:
                        combined_pointverts = pointverts
                        xy_layerverts = np.copy(combined_pointverts)
                    else: # for multiple points, easier to combine into an approx hull than figure out the compound elliptic path
                        combined_pointverts = np.concatenate((combined_pointverts,pointverts),axis=0)
                hull = ConvexHull(combined_pointverts)
                xy_layerverts = np.transpose(np.vstack((combined_pointverts[hull.vertices,0],combined_pointverts[hull.vertices,1])))
                xy_layerverts = np.concatenate((xy_layerverts,np.atleast_2d(xy_layerverts[0,:])),axis=0) # close path
                layer_paths.append(Path(xy_layerverts,closed=True))
            # regions in parameter space are accumlated per individual point and might not overlap
            else:
                for mu_t1,mu_flair,std_t1,std_flair in zip(bd['meant12'],bd['meanflair'],bd['stdt12'],bd['stdflair']):
                    # currently 2 is additionally hard-coded here
                    point_perimeter = Ellipse((mu_flair, mu_t1), 2*pointclustersize*std_flair,2*pointclustersize*std_t1)
                    unitverts = point_perimeter.get_path().vertices
                    pointverts = point_perimeter.get_patch_transform().transform(unitverts)
                    xy_layerverts = np.transpose(np.vstack((pointverts[:,0],pointverts[:,1])))
                    # already closed
                    # xy_layerverts = np.concatenate((xy_layerverts,np.atleast_2d(xy_layerverts[0,:])),axis=0) # close path
                    layer_paths.append(Path(xy_layerverts,closed=True))

        # define ET threshold gate >= (t2gate,t1gate). upper right quadrant
        else:
            yv_gate = np.array([t12gate, maxZ, maxZ, t12gate])
            xv_gate = np.array([flairgate, flairgate, maxZ, maxZ]) 
            # form vertices for slider-selected rectangular foreground gate
            xy_layerverts = np.vstack((xv_gate,yv_gate)).T
            xy_layerverts = np.concatenate((xy_layerverts,np.atleast_2d(xy_layerverts[0,:])),axis=0) # close path
            layer_paths.append(Path(xy_layerverts,closed=True))
    elif layer == 'T2 hyper':
        if False: # ie plain T2 not available
            # define NET threshold gate, >= ,flairgate,>= t2gate. upper right quadrant
            # flairgate = blastdata['blast']['params']['meant2flair']+(flairthresh)*blastdata['blast']['params']['stdt2flair']
            yv_gate = np.array([t12gate, maxZ, maxZ, t12gate])
            xv_gate = np.array([flairgate, flairgate, maxZ, maxZ]) 
        # until T2 is available, define NET threshold gate, >= ,flairgate. right hemispace
        yv_gate = np.array([-maxZ, maxZ, maxZ, -maxZ])
        xv_gate = np.array([flairgate, flairgate, maxZ, maxZ]) 
        # form vertices for slider-selected rectangular foreground gate
        xy_layerverts = np.vstack((xv_gate,yv_gate)).T
        xy_layerverts = np.concatenate((xy_layerverts,np.atleast_2d(xy_layerverts[0,:])),axis=0) # close path
        layer_paths.append(Path(xy_layerverts,closed=True))

    # find the gated pixels
    stack_shape = np.shape(t1)

    # begun blast processing 
    start = time.time()
    region_of_support = np.where(t1stack*flairstack*t2stack != 0)
    background_mask = np.where(t1stack*flairstack*t2stack == 0)
    t1channel = t1stack[region_of_support]
    # currently don't have plain t2 in radnec cases, but this might 
    # need to be restored in the future
    t2channel = t2stack[region_of_support]
    flairchannel = flairstack[region_of_support]

    # form vertices of the data volume
    if layer == 'ET':
        t1t2verts = np.vstack((flairchannel.flatten(),t1channel.flatten())).T
    else: # ie WT
        t1t2verts = np.vstack((flairchannel.flatten(),t2channel.flatten())).T

    # form vertices for normal brain background gate
    unitverts = brain_perimeter.get_path().vertices
    xy_brainverts = brain_perimeter.get_patch_transform().transform(unitverts)

    # before running processing, use pre-existing result if available
    brain_gate = blastdata['blast']['gates']['brain '+layer]
    layer_gate = blastdata['blast']['gates'][layer]

    if use_gpu: # not updated, haven't got cudf installed in radnec env yet
        RGcoords = cudf.DataFrame({'x':t1t2verts[:,0],'y':t1t2verts[:,1]}).interleave_columns()
        RGseries = GeoSeries.from_points_xy(RGcoords)
        xy_layercoords = cudf.DataFrame({'x':xy_layerverts[:,0],'y':xy_layerverts[:,1]}).interleave_columns()
        xy_layerseries = GeoSeries.from_polygons_xy(xy_layercoords,[0, np.shape(xy_layerverts)[0]],
                                            [0,1],[0,1] )
        xy_braincoords = cudf.DataFrame({'x':xy_brainverts[:,0],'y':xy_brainverts[:,1]}).interleave_columns()
        xy_brainseries = GeoSeries.from_polygons_xy(xy_braincoords,[0,np.shape(xyverts)[0]],[0,1],[0,1])

        # only redo the gates if required due to change in threshold
        if blastdata['blast']['gates'][layer] is None:
            layer_gate = np.zeros(np.shape(t1stack),dtype='uint8')
            layergate_pts = dotime(point_in_polygon,(RGseries,xy_layerseries),txt='gate')
            layer_gate[region_of_support] = layergate_pts.values.get().astype('int').flatten()
        if blastdata['blast']['gates']['brain '+layer] is None:
            brain_gate = np.zeros(np.shape(t1stack),dtype='uint8')
            braingate_pts = dotime(point_in_polygon,(RGseries,xy_brainseries),txt='brain')
            brain_gate[region_of_support] = braingate_pts.values.get().astype('int').flatten()
    else:
        brain_path = Path(xy_brainverts,closed=True)
        # layer_path = Path(xy_layerverts,closed=True)
        if blastdata['blast']['gates'][layer] is None or True:
            layer_gate = np.zeros(np.shape(t1stack),dtype='uint8')
            for p in layer_paths:
                # layer_gate[region_of_support] = (layer_gate[region_of_support]) | (p.contains_points(t1t2verts))
                layer_gate[region_of_support] = np.logical_or(layer_gate[region_of_support],p.contains_points(t1t2verts))
        if blastdata['blast']['gates']['brain '+layer] is None:
            brain_gate = np.zeros(np.shape(t1stack),dtype='uint8')
            brain_gate[region_of_support] = brain_path.contains_points(t1t2verts).flatten()

    if True:
        plt.figure(7,figsize=(6,6))
        if layer == 'ET':
            ax = plt.subplot(1,1,1)
            ax.cla()
            plt.scatter(t1t2verts[:,0],t1t2verts[:,1],c='b',s=1)
            layergate = np.where(layer_gate>0)
            braingate = np.where(brain_gate>0)
            if len(layer_paths):
                for i,p in enumerate(layer_paths):
                    if len(blastdata['blastpoint']['params']['ET']['pt']):
                        plt.plot(blastdata['blastpoint']['params']['ET']['pt'][i][0],blastdata['blastpoint']['params']['ET']['pt'][i][1],'c+')
                    plt.plot(p.vertices[:,0],p.vertices[:,1],'r-',linewidth=0.5)
            else:
                plt.scatter(xy_layerverts[:,0],xy_layerverts[:,1],c='r',s=10)
            plt.scatter(flairstack[layergate],t1stack[layergate],c='g',s=1)
            plt.scatter(flairstack[braingate],t1stack[braingate],c='w',s=1)
            ax.set_aspect('equal')
            ax.set_xlim(left=-maxZ,right=maxZ)
            ax.set_ylim(bottom=-maxZ,top=maxZ)
            # plt.text(0,1.02,'flair {:.3f},{:.3f}'.format(np.mean(flairchannel),np.std(flairchannel)))
            # plt.text(0,1.1,'t1 {:.3f},{:.3f}'.format(np.mean(t1channel),np.std(t1channel)))
            plt.xlabel('flair')
            plt.ylabel('t1')
        elif layer == 'T2 hyper':
            ax2 = plt.subplot(1,2,2)
            ax2.cla()
            plt.scatter(t1t2verts[:,0],t1t2verts[:,1],c='b',s=1)
            layergate = np.where(layer_gate>0)
            braingate = np.where(brain_gate>0)
            plt.scatter(flairstack[layergate],t2stack[layergate],c='g',s=2)
            plt.scatter(flairstack[braingate],t2stack[braingate],c='w',s=2)
            plt.scatter(xy_layerverts[:,0],xy_layerverts[:,1],c='r',s=20)
            ax2.set_aspect('equal')
            ax2.set_xlim(left=0,right=1.0)
            ax2.set_ylim(bottom=0,top=1.0)
            plt.text(0,1.1,'t2 {:.3f},{:.3f}'.format(np.mean(t2channel),np.std(t2channel)))
            plt.text(0,1.02,'flair {:.3f},{:.3f}'.format(np.mean(flairchannel),np.std(flairchannel)))
            plt.xlabel('flair')
            plt.ylabel('t2')
        filename = 'scatterplot' + str(len(layer_paths)) + '.png'
        plt.savefig(os.path.join('/home/jbishop/Pictures',filename),dpi=100)
        # plt.clf()
        # plt.show(block=False)

    dtime = time.time()-start
    print('polygon contains points time = {:.2f} sec'.format(dtime))
    layer_mask = np.logical_and(layer_gate, np.logical_not(brain_gate)) # 

    return layer_mask,brain_gate,layer_gate


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

def rescale(arr):
    arr1 = arr - np.min(arr)
    arr1 /= np.max(arr1)
    return arr1