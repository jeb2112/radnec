import os
import numpy as np
from skimage.draw import ellipse_perimeter
from scipy.spatial.distance import dice
import copy
from matplotlib.path import Path
import time
import cudf
from cuspatial import point_in_polygon
from cuspatial import GeoSeries
import multiprocessing
import cc3d
import OverlayPlots

class Blast():
    def __init__(self):

        self.brain = None
        self.et_gate = None
        self.wt_gate = None

    #################
    # BLAST algorithm
    #################
    
    def run_blast(self,data,t1thresh,t2thresh,clustersize,
                currentslice=None):

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

        # row=y=t1 col=x=t2
        (yv,xv) = self.get_ellipse_perimeter(data['params']['meant1'],data['params']['meant2'],t1Diff,t2Diff)

        t2gate = data['params']['meant2']+(t2thresh)*data['params']['stdt2']
        t2gate_count = (t2gate-data['params']['meant2'])/data['params']['stdt2']
        t1gate = data['params']['meant1']+(t1thresh)*data['params']['stdt1']
        t1gate_count = (t1gate-data['params']['meant1'])/data['params']['stdt1']

        # define ET threshold gate >= (t2gate,t1gate)
        # upper limit 4 hard-coded here
        yv_et = np.array([t1gate, 4, 4, t1gate])
        xv_et = np.array([t2gate, t2gate, 4, 4]) 

        # define WT threshold gate, >= t2gate
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
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-4,initializer=self.pool_initializer,
                                        initargs=(metmaskstack_shared,fusionstack_shared,stack_shape))

            argslist = [(slice,data[:,slice,:,:],xv,yv,xv_et,yv_et) for slice in range(startslice,endslice)]
            pool.starmap_async(self.do_pip2d,argslist)
            pool.close()
            pool.join()

        # otherwise just use cuspatial gpu in 3d
        elif (endslice-startslice)>1:
            start = time.time()
            region_of_support = np.where(t1mpragestack>0)
            t1channel = t1mpragestack[region_of_support]
            t2channel = t2flairstack[region_of_support]
            t1t2verts = np.vstack((t2channel.flatten(),t1channel.flatten())).T
            xy_etverts = np.vstack((xv_et,yv_et)).T
            xy_etverts = np.concatenate((xy_etverts,np.atleast_2d(xy_etverts[0,:])),axis=0) # close path
            xy_wtverts = np.vstack((xv_wt,yv_wt)).T
            xy_wtverts = np.concatenate((xy_wtverts,np.atleast_2d(xy_wtverts[0,:])),axis=0) # close path
            xyverts =  np.vstack((xv,yv)).T
            xyverts = np.concatenate((xyverts,np.atleast_2d(xyverts[0,:])),axis=0) # close path
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

            if self.et_gate is None:
                self.et_gate = np.zeros(np.shape(t1mpragestack),dtype='uint8')
                gate_etpts = self.dotime(point_in_polygon,(RGseries,xy_etseries),txt='gate')
                gate_etpts = gate_etpts.values.get().astype('int').flatten()
                self.et_gate[region_of_support] = gate_etpts
            if self.wt_gate is None:
                self.wt_gate = np.zeros(np.shape(t1mpragestack),dtype='uint8')
                gate_wtpts = self.dotime(point_in_polygon,(RGseries,xy_wtseries),txt='gate')
                gate_wtpts = gate_wtpts.values.get().astype('int').flatten()
                self.wt_gate[region_of_support] = gate_wtpts
            if self.brain is None:
                self.brain = np.zeros(np.shape(t1mpragestack),dtype='uint8')
                brainpts = self.dotime(point_in_polygon,(RGseries,xyseries),txt='brain')
                brainpts = brainpts.values.get().astype('int').flatten()
                self.brain[region_of_support] = brainpts
            end = time.time()
            print('polygon contains points time = {:.2f} sec'.format(end-start))
            et_mask = np.logical_and(self.et_gate, np.logical_not(self.brain)) # 
            # et_mask = max(et_mask,0)
            # fusion = imfuse(et_mask,t1mprage_template[slice,:,:,slice],'blend')
            if False:
                fusion = 0.5*et_mask + 0.5*t1mprage_template # alpha composite unweighted
            else:
                start = time.time()
                fusion = OverlayPlots.generate_overlay(data['raw'],et_mask) 
                end = time.time()
                print('generate overlay time = {:.2f} sec'.format(end-start))

            #se = strel('line',2,0) 
            #et_mask = imerode(et_mask,se) # erosion added to get rid of non
            #specific small non target voxels removed for ROC paper

            wt_mask = np.logical_and(self.wt_gate, np.logical_not(self.brain))
            # maskstack = et_mask + wtmask
            c_maskstack = et_mask.astype('int')*2 + wt_mask.astype('int')
            fusion = OverlayPlots.generate_overlay(data['raw'],c_maskstack) 
            fusionstack = fusion
            et_maskstack = et_mask

        # a single 2d slice
        else:
            for slice in range(startslice,endslice):  

                # Gating Routine    
                t1channel = t1mpragestack[slice,:,:]
                t2channel = t2flairstack[slice,:,:]
                
                # Applying gate to brain
                # gate = inpolygon(t2channel,t1channel,yv_et,xv_et)
                # brain = inpolygon(t2channel,t1channel,yv,xv)
                t1t2verts = np.vstack((t2channel.flatten(),t1channel.flatten())).T
                xy_etverts = np.vstack((xv_et,yv_et)).T
                xy_etverts = np.concatenate((xy_etverts,np.atleast_2d(xy_etverts[0,:])),axis=0) # close path
                xy_wtverts = np.vstack((xv_wt,yv_wt)).T
                xy_wtverts = np.concatenate((xy_wtverts,np.atleast_2d(xy_wtverts[0,:])),axis=0) # close path
                xyverts =  np.vstack((xv,yv)).T
                xyverts = np.concatenate((xyverts,np.atleast_2d(xyverts[0,:])),axis=0) # close path
                path = Path(xyverts,closed=True)
                path_et = Path(xy_etverts)
                path_wt = Path(xy_wtverts)

                # find polygons
                # matplotlib. don't need gpu for 1 slice
                gate_et = path_et.contains_points(t1t2verts).reshape(240,240)
                gate_wt = path_wt.contains_points(t1t2verts).reshape(240,240)
                brain = path.contains_points(t1t2verts).reshape(240,240)

                # form output image
                et_mask = np.logical_and(gate_et, np.logical_not(brain)) # 
                et_mask = et_mask.reshape((240,240))
                wt_mask = np.logical_and(gate_wt, np.logical_not(brain)) # 
                wt_mask = wt_mask.reshape((240,240))
                compound_mask = et_mask.astype('int')*2+wt_mask.astype('int')
                # et_mask = max(et_mask,0)
                # fusion = imfuse(et_mask,t1mprage_template[slice,:,:,slice],'blend')
                if False:
                    fusion = 0.5*et_mask + 0.5*t1mprage_template[slice,:,:] # alpha composite unweighted
                else:
                    fusion = OverlayPlots.generate_overlay(data['raw'][:,slice,:,:],compound_mask)

                #se = strel('line',2,0) 
                #et_mask = imerode(et_mask,se) # erosion added to get rid of non
                #specific small non target voxels removed for ROC paper

                et_maskstack[slice] = et_mask
                wt_maskstack[slice] = wt_mask
                c_maskstack[slice] = compound_mask
                fusionstack[:,slice,:,:] = fusion

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
        # TODO: return only the seg, not the composite
        return c_maskstack,fusionstack


    #################
    # utility methods
    #################

    # Prepare variable
    def get_ellipse_perimeter(self,r,c,dr,dc,scale=1024): 
        (yv,xv) = ellipse_perimeter(int(r*scale),int(c*scale),int(dr*scale),int(dc*scale))
        return (yv/scale,xv/scale)

    # initialize variables for do_pip()
    def pool_initializer(self,X,Y,X_shape):
        global np_x,np_y
        np_x = X
        np_y = Y
        global np_x_shape
        np_x_shape = X_shape

    def dotime(self,func,args,txt=''):
        start = time.time()
        retval = func(*args)
        end = time.time()
        print ('dotime {} = {:.2f}'.format(txt,end-start))
        return retval

    # polygon contains points for slice by slice iteration in parallel
    # not recently updated
    def do_pip2d(self,slice,channelstack,xv,yv,xv_et,yv_et):
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




