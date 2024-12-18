import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
import copy
import os
import json
from scipy.spatial.distance import dice,directed_hausdorff
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_dilation,binary_erosion
import cc3d

# collecting various ROI stats and data
# for BLAST/SAM segmentations
class ROI():
    def __init__(self,dim,number=None):

        self.casename = None
        self.status = False
        self.dim = dim
        self.number = number

        # output stats
        self.stats = {'spec':{'ET':0,'TC':0,'WT':0},
            'sens':{'ET':0,'TC':0,'WT':0},
            'dsc':{'ET':0,'TC':0,'WT':0},
            'vol':{'ET':0,'TC':0,'WT':0},
            'hd':{'ET':0,'TC':0,'WT':0},
            'elapsedtime':0}
        
        # segmentation masks
        # seg_fusion - an overlay image
        # seg_fusion_d - a copy of overaly image for display purposes
        # seg - a composite mask of the ROI segmentation combining ET,TC,WT
        self.data = {'WT':None,'ET':None,'TC':None,
                'seg_fusion':{'t1':None,'t1+':None,'t2':None,'flair':None},
                'seg_fusion_d':{'t1':None,'t1+':None,'t2':None,'flair':None},
                'seg':None
                }
        
    # various output stats, add more as required. 
    # an existing stats file is read in if present, and values added for the current roi,
    # 
    # slice - process given slice or whole volume if None
    # mask - optional mask for ground truth stats
    # dt - tumor layer to process
    def ROIstats(self,roitype='blast',slice=None,mask=None,dt='ET'):
        
        # checking for == 0 here but this is a bug, the dataset should 
        # either be non-zero or None
        if self.data[dt] is None or np.max(self.data[dt]) == 0:
            return
        
        if np.max(self.data[dt]) > 1:
            self.data[dt] = self.data[dt] == np.max(self.data[dt])
        if slice is None:
            dset = self.data[dt]
        else:
            dset = self.data[dt][slice]
        self.stats['vol'][dt] = len(np.where(dset)[0])

        # ground truth comparisons
        if mask is not None:

            # pull out the matching lesion using cc3d
            CC_labeled = cc3d.connected_components(mask,connectivity=6)
            centroid_point = np.array(list(map(int,np.nanmean(np.where(self.data[dt]),axis=1)))) 
            objectnumber = CC_labeled[centroid_point[0],centroid_point[1],centroid_point[2]]
            gt_lesion = (CC_labeled == objectnumber).astype('uint8')
            if slice is not None:
                gt_lesion = gt_lesion[slice]

            # dice
            self.stats['dsc'][dt] = 1-dice(gt_lesion.flatten(),dset.flatten()) 
            # haunsdorff
            self.stats['hd'][dt] = max(directed_hausdorff(np.array(np.where(gt_lesion)).T,np.array(np.where(dset)).T)[0],
                                                    directed_hausdorff(np.array(np.where(dset)).T,np.array(np.where(gt_lesion)).T)[0])
        else:
            print('No ground truth comparison available')
            
    # tag - top-level section key for output .json file. 
    def save_ROIstats(self,r,filename,tag=None):

        if os.path.exists(filename):
            fp = open(filename,'r+')
            sdict = json.load(fp)
            if tag not in sdict.keys():
                sdict[tag] = {}
            fp.seek(0)
        else:
            fp = open(filename,'w')
            sdict = {tag:{}}

        sdict[tag]['roi'+str(r.number)] = {'stats':None,'bbox':None}
        sdict[tag]['roi'+str(r.number)]['stats'] = r.stats
        if hasattr(r,'bboxs'):
            bboxs = {}
            if bool(r.bboxs):
                kset = []
                if slice is None:
                    kset = r.bboxs.keys()
                else:
                    if slice in r.bboxs.keys():
                        kset = [slice]
                if len(kset):
                    for k in kset:
                        try:
                            bboxs[k] = {k2:r.bboxs[k][k2] for k2 in ['p0','p1','slice']}
                        except KeyError:
                            pass
            sdict[tag]['roi'+str(r.number)]['bbox'] = bboxs

        json.dump(sdict,fp,indent=4)
        fp.truncate()
        fp.close()


########################
# raw BLAST segmentation
########################

class ROIBLAST(ROI):
    def __init__(self,coords,dim,layer='ET',number=None):
        super().__init__(dim,number=number)
        
        # BLAST ROI selection coordinates from mouse click
        self.coords = {'ET':{},'necrosis':{},'T2 hyper':{}}
        self.coords[layer]['x'] = coords[0] # x
        self.coords[layer]['y'] = coords[1] # y
        self.coords[layer]['slice'] = coords[2] # slice

        # threshold gates saved as intermediate values
        self.gate = {'brain':None,'ET':None,'T2 hyper':None}


##################
# SAM segmentation
##################

class ROISAM(ROI):
    def __init__(self,dim,bbox={},pt={},layer='TC',number=None):
        super().__init__(dim,number=number)

        # a SAM drawn bbox coordinates and values, or bbox from BLAST mask
        self.bbox = {'ax':None,'p0':None,'p1':None,'plot':None,'l':None,'ch':None,'slice':None} 
        # dict of multiple bbox's, with key = slice. 
        self.bboxs = {}

        # a selected SAM control point
        self.pt = {'ax':None,'p0':None,'plot':None,'ch':None,'slice':None, 'fg':True} 
        # list of multiple control points
        self.pts = []
        self.mask = None

        # another dict for creating the saved json files which are read by the SAM huggingface code
        self.pjsondict = {'x':[],'y':[],'fg':[]}
        # segmentation masks
        # 'TC','WT -  SAM segmentation. segmentation of enhancing tumor is interpreted as tumour core for now 'ET' not used
        # seg_fusion - an overlay image of 'TC' over t1+ or flair
        # seg_fusion_d - a copy of overaly image for display purposes
        # seg - a composite mask of a multi-compartment segmentation combining ET,TC,WT. not currently used in SAM viewer
        # bbox - 3d volume of 2d bbox prompts of each in-plane slice, to be exported as .png files for use with SAM
        # point - a json dict containing lists of point prompts, for use with SAM
        # maskpoint - a dict of json dicts, keyed by slice number for 3d multi-slice SAM
        self.data = {'TC':np.zeros(self.dim,dtype='uint8'),'ET':np.zeros(self.dim,dtype='uint8'),'WT':np.zeros(self.dim,'uint8'),
                     'seg_fusion':{'t1':None,'t1+':None,'t2':None,'flair':None},
                     'seg_fusion_d':{'t1':None,'t1+':None,'t2':None,'flair':None},
                     'seg':None,
                     'bbox':{'ax':np.zeros(dim,dtype='uint8'),'sag':np.zeros(dim,dtype='uint8'),'cor':np.zeros(dim,dtype='uint8')},
                    #  'point':{'ax':copy.deepcopy(self.pjsondict),'sag':copy.deepcopy(self.pjsondict),'cor':copy.deepcopy(self.pjsondict)},
                     'point':{'ax':{k:copy.deepcopy(self.pjsondict) for k in range(self.dim[0])},
                                  'sag':{k:copy.deepcopy(self.pjsondict) for k in range(self.dim[2])},
                                  'cor':{k:copy.deepcopy(self.pjsondict) for k in range(self.dim[1])}},
                     'maskpoint':{'ax':{k:copy.deepcopy(self.pjsondict) for k in range(self.dim[0])},
                                  'sag':{k:copy.deepcopy(self.pjsondict) for k in range(self.dim[2])},
                                  'cor':{k:copy.deepcopy(self.pjsondict) for k in range(self.dim[1])}}
                     }
        # initializing with a one-slice bbox might no longer be needed, use set_bbox instead. 
        if bool(bbox):
            self.set_bbox(bbox)
        if bool(pt):
            self.set_pt(pt)

    # create a multi-slice set of point|bbox prompts from a 3d mask such as BLAST ROI
    # these prompts are stored both as a mask and as coordinates in a dict
    def create_prompts_from_mask(self,mask,prompt='bbox',slice=None,orient='ax'):
        if orient == 'ax':
            if slice is None:
                rslice = range(np.shape(mask)[0]) # do all slices
            else:
                rslice = [slice]
            for r in rslice:
                if len(np.where(mask[r])[0]):
                    if prompt == 'bbox':
                        self.data['bbox'][orient][r],self.bboxs[r] = self.get_bbox_mask(mask[r])
                    elif prompt == 'maskpoint':
                        self.get_point_mask(r,mask[r])  
        elif orient == 'sag':
            if slice is None:
                rslice = range(np.shape(mask)[2]) # do all slices
            else:
                rslice = [slice]
            for r in rslice:
                if len(np.where(mask[:,:,r])[0]):
                    if prompt == 'bbox':
                        self.data['bbox'][orient][:,:,r],_ = self.get_bbox_mask(mask[:,:,r])
                    elif prompt == 'maskpoint':
                        self.get_point_mask(r,mask[:,:,r],orient=orient)  
        elif orient == 'cor':
            if slice is None:
                rslice = range(np.shape(mask)[1]) # do all slices
            else:
                rslice = [slice]
            for r in rslice:
                if len(np.where(mask[:,r,:])[0]):
                    if prompt == 'bbox':
                        self.data['bbox'][orient][:,r,:],_ = self.get_bbox_mask(mask[:,r,:])
                    elif prompt == 'maskpoint':
                        self.get_point_mask(r,mask[:,r,:],orient=orient)  
         
    def get_point_mask(self,slice,mask,orient='ax',hull_extension=0):
        cy,cx = map(int,np.round(np.mean(np.where(mask),axis=1)))
        centroid_point = np.atleast_2d([cx,cy])
        point_set = centroid_point

        # for generating point prompts, multiple foreground points are ok, but multiple exclusion points
        # breaks SAM so use only 1.
        # for now, use only a single centroid point
        if len(np.where(mask)[0]) > 25 and False: # use only centroid for small masks
            hcoords = self.compute_hull_from_mask(mask,hull_extension=-hull_extension)
            if hcoords is None:
                convexhull_fg = np.array([])
            else:
                convexhull_fg = np.atleast_2d(hcoords)
                point_set = np.concatenate((np.atleast_2d(point_set),convexhull_fg),axis=0)
        else:
            convexhull_fg = np.array([])
        # the exclusion points should be checked against the region of support of the image slice. in this version of
        # the viewer, the assumption is that brain extractions are not being done, so there should be plenty of scalp pixels to cover the case of
        # a lesion right at the skull.
        if False:
            convexhull_bg = np.atleast_2d(self.compute_hull_from_mask(mask,hull_extension=np.abs(hull_extension)+5)[0]) # use only 1 point for bg
            point_set = np.concatenate((point_set,convexhull_bg),axis=0)
        # until bg points can be verified somehow, just use the single centroid point.
        else:
            convexhull_bg = np.array([])
        point_set_labels = np.array([1]+[1]*len(convexhull_fg)+[0]*len(convexhull_bg))

        self.data['maskpoint'][orient][slice]['x'] = [int(np.round(p[0])) for p in point_set]
        # huggingface convention y increases from top to bottom, same as plotted in FigureCanvasTkAgg 
        if False:
            self.data['maskpoint'][orient]['y'] = [self.dim[1] - int(np.round(p[1])) for p in convexhull_points]
        else:
            self.data['maskpoint'][orient][slice]['y'] = [int(np.round(p[1])) for p in point_set]
        self.data['maskpoint'][orient][slice]['fg'] = [int(l) for l in point_set_labels]

        if False:
            plt.figure(7)
            plt.cla()
            plt.imshow(mask)
            plt.plot(self.data['maskpoint'][orient][slice]['x'],self.data['maskpoint'][orient][slice]['y'],'r.',markersize=2)
            plt.show(block=False)
            a=1

        return

    def get_bbox_mask(self,mask):
        # Find minimum mask bounding all included mask points.
        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        mask[y_min:y_max+1,x_min:x_max+1] = 1

        bbox = {'ax':None,'p0':None,'p1':None,'plot':None,'l':None,'slice':None}
        bbox['p0'] = [int(x_min), int(y_min)]
        bbox['p1'] = [int(x_max), int(y_max)]

        return mask,bbox

    # create a set of point prompts directly from a list of control points
    def create_prompts_from_points(self,pts,prompt='point',slice=None,orient='ax'):
        self.data['point'][orient][slice]['x'] = [int(np.round(p['p0'][0])) for p in pts]
        # huggingface convention y increases from top to bottom, same as plotted in FigureCanvasTkAgg 
        if False:
            self.data['point'][orient]['y'] = [self.dim[1] - int(np.round(p['p0'][1])) for p in pts]
        else:
            self.data['point'][orient][slice]['y'] = [int(np.round(p['p0'][1])) for p in pts]
        self.data['point'][orient][slice]['fg'] = [int(p['fg']) for p in pts]

        if False:
            plt.figure(7)
            plt.cla()
            plt.imshow(np.zeros(self.dim[1:]))
            plt.plot(self.data['point'][orient][slice]['x'],self.data['point'][orient][slice]['y'],'r.',markersize=2)
            plt.show(block=False)
            a=1
        return

    # set a bbox for a given slice    
    def set_bbox(self,bbox):
        for k in bbox.keys():
            if k in list(self.bbox.keys()):
                self.bbox[k] = bbox[k]
            else:
                raise KeyError
        self.create_prompt_from_bbox()

    # set a control point for a given slice
    def set_pt(self,pt):
        for k in pt.keys():
            if k in list(self.pt.keys()):
                self.pt[k] = pt[k]
            else:
                raise KeyError
        self.create_prompt_from_pts()


    # compute one-slice prompt from bounding box in a given slice. 
    def create_prompt_from_bbox(self, bbox=None, box_extension=0, orient='ax'):
        mask = np.zeros((self.dim[1],self.dim[2]),dtype='uint8')
        if bbox is None:
            bbox = self.bbox
        if bbox['p1'] is None:
            bbox_pts = np.round(np.array(self.bbox['p0'])).astype('int')
            mask[bbox_pts[1],bbox_pts[0]] = 1
        else:
            vxy = np.array([[bbox['p0'][0],bbox['p0'][1]],
                        [bbox['p1'][0],bbox['p0'][1]],
                        [bbox['p1'][0],bbox['p1'][1]],
                        [bbox['p0'][0],bbox['p1'][1]]])
            vyx = np.flip(vxy,axis=1)
            bbox_path = Path(vyx,closed=False)
            mask = bbox_path.contains_points(np.array(np.where(mask==0)).T)
            mask = np.reshape(mask,(self.dim[1],self.dim[2]))     
        self.data['bbox'][orient][bbox['slice']] = mask

    # compute one-slice prompt from control point(s) in a given slice. 
    # this was for the original arrangement of storing control points in a png.
    # has been replaced by create_prompts_from_points to store in json
    def create_prompt_from_pts(self, pts=None, orient='ax'):
        if pts is None:
            pts = self.pts
        for p in pts:
            pcoords = np.round(np.array(p['p0'])).astype('int')
            self.data['bbox'][orient][p['slice'],pcoords[1],pcoords[0]] = 1

    # compute the bounding hull from a mask. 
    def compute_hull_from_mask(self,mask, original_size=None, hull_extension=0):
        mask2 = np.copy(mask)
        if hull_extension > 0:
            for h in range(hull_extension):
                mask2 = binary_dilation(mask2)
        elif hull_extension < 0:
            for h in range(np.abs(hull_extension)):
                test = binary_erosion(mask2)
                if len(np.where(test == 1)[0]) > 15: # arbitrary minimum size of eroded hull
                    mask2 = test
                else:
                    return None
            
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



# for linear measurements in 4panel viewer
class ROILinear():
    def __init__(self,x0,y0,x1,y1,slice,channel):
        self.casename = None
        self.status = False

        # ROI selection coordinates
        self.coords = {}
        self.coords['p0'] = (x0,y0)
        self.coords['p1'] = (x1,y1)
        self.coords['l'] = np.sqrt(np.power(x1-x0,2)+np.power(y1-y0),2)
        self.coords['slice'] = slice
        self.coords['channel'] = channel
        self.coords['plot'] = None


# for generic point data
class Point():
    def __init__(self,xpos,ypos,slice):
        self.coords = {}
        self.coords['x'] = xpos
        self.coords['y'] = ypos
        self.coords['slice'] = slice

# for creating raw BLAST seg by point selection
class ROIPoint(Point):
    def __init__(self,xpos,ypos,slice):
        super().__init__(xpos,ypos,slice)
        # stats to form ellipse in parameter space
        self.data = {'flair':{'mu':0,'std':0},'t12':{'mu':0,'std':0}}
        self.radius = 5 # pixel radius to calculate stats on 


# for SAM prompts by control point
class SAMPoint(Point):
    def __init__(self,xpos,ypos,slice,foreground=True):
        super().__init__(xpos,ypos,slice)
        self.foreground = foreground