import numpy as np
from matplotlib.path import Path

# collectding various ROI stats and data
# for BLAST/SAM segmentations
class ROI():
    def __init__(self,dim):

        self.casename = None
        self.status = False
        self.dim = dim

        # output stats
        self.stats = {'spec':{'ET':0,'TC':0,'WT':0},
            'sens':{'ET':0,'TC':0,'WT':0},
            'dsc':{'ET':0,'TC':0,'WT':0},
            'vol':{'ET':0,'TC':0,'WT':0},
            'hd':{'ET':0,'TC':0,'WT':0},
            'elapsedtime':0}
        
class ROIBLAST(ROI):
    def __init__(self,coords,dim,layer='ET'):
        super().__init__(dim)

        # segmentation/contour masks
        # seg_fusion - an overlay image
        # seg_fusion_d - a copy of overaly image for display purposes
        # seg - a composite mask of the ROI segmentation combining ET,TC,WT
        self.data = {'WT':None,'ET':None,'TC':None,'contour':{'WT':None,'ET':None},
                     'seg_fusion':{'t1':None,'t1+':None,'t2':None,'flair':None},
                     'seg_fusion_d':{'t1':None,'t1+':None,'t2':None,'flair':None},
                     'seg':None
                     }
        
        # BLAST ROI selection coordinates from mouse click
        self.coords = {'ET':{},'necrosis':{},'T2 hyper':{}}
        self.coords[layer]['x'] = coords[0] # x
        self.coords[layer]['y'] = coords[1] # y
        self.coords[layer]['slice'] = coords[2] # slice

        # threshold gates saved as intermediate values
        self.gate = {'brain':None,'ET':None,'T2 hyper':None}

class ROISAM(ROI):
    def __init__(self,bbox,dim,layer='TC'):
        super().__init__(dim)
        # SAM drawn bbox coordinates and values, or bbox from BLAST mask
        if bbox is None:
            self.bbox = {'ax':None,'p0':None,'p1':None,'plot':None,'l':None,'ch':None}        
        else:
            self.bboxs = bbox
        self.mask = None

        # segmentation masks
        # seg_fusion - an overlay image
        # seg_fusion_d - a copy of overaly image for display purposes
        # seg - a composite mask of the ROI segmentation combining ET,TC,WT
        # bbox - 3d volume of 2d bbox prompts of each in-plane slice, for use with SAM
        self.data = {'TC':None,
                     'seg_fusion':{'t1':None,'t1+':None,'t2':None,'flair':None},
                     'seg_fusion_d':{'t1':None,'t1+':None,'t2':None,'flair':None},
                     'seg':None,
                     'bbox':None
                     }
        self.data['bbox'] = np.zeros(dim,dtype='uint8')
        self.create_mask_from_bbox()

    # compute mask array from bounding box. 
    # this is a round-about arrangement, since sam.py script
    # recomputes the bbox from the mask, better to implement
    # external file storage for bbox's directly. 
    # TODO. box extension
    def create_mask_from_bbox(self, box_extension=0):
        mask = np.zeros((self.dim[1],self.dim[2]),dtype='uint8')
        if self.bbox['p1'] is None:
            bbox = np.round(np.array(bbox)).astype('int')
            mask[self.bbox['p0'][1],self.bbox['p0'][0]] = 1
        else:
            vxy = np.array([[self.bbox['p0'][0],self.bbox['p0'][1]],
                        [self.bbox['p1'][0],self.bbox['p0'][1]],
                        [self.bbox['p1'][0],self.bbox['p1'][1]],
                        [self.bbox['p0'][0],self.bbox['p1'][1]]])
            vyx = np.flip(vxy,axis=1)
            bbox_path = Path(vyx,closed=False)
            mask = bbox_path.contains_points(np.array(np.where(mask==0)).T)
            mask = np.reshape(mask,(self.dim[1],self.dim[2]))     
        self.data['bbox'][self.bbox['slice']] = mask


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


# for creating raw BLAST seg by point selection
class ROIPoint():
    def __init__(self,xpos,ypos,slice):
        # BLAST ROI selection coordinates from mouse click
        self.coords = {}
        self.coords['x'] = xpos
        self.coords['y'] = ypos
        self.coords['slice'] = slice
        self.data = {'flair':{'mu':0,'std':0},'t12':{'mu':0,'std':0}}
        self.radius = 5 # pixel radius to include
