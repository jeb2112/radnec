import numpy as np
from matplotlib.path import Path
import copy

# collecting various ROI stats and data
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
    def __init__(self,dim,bbox={},layer='TC'):
        super().__init__(dim)
        # a SAM drawn bbox coordinates and values, or bbox from BLAST mask
        self.bbox = {'ax':None,'p0':None,'p1':None,'plot':None,'l':None,'ch':None,'slice':None} 
        # dict of multiple bbox's, with key = slice. 
        self.bboxs = {}
        self.mask = None

        # segmentation masks
        # 'TC' - the SAM segmentation, interpreted as tumour core for now. 'ET','WT' are not used
        # seg_fusion - an overlay image of 'TC' over t1+ or flair
        # seg_fusion_d - a copy of overaly image for display purposes
        # seg - a composite mask of a multi-compartment segmentation combining ET,TC,WT. not currently used in SAM viewer
        # bbox - 3d volume of 2d bbox prompts of each in-plane slice, to be exported as .png files for use with SAM
        self.data = {'TC':np.zeros(self.dim,dtype='uint8'),'ET':np.zeros(self.dim,dtype='uint8'),'WT':np.zeros(self.dim,'uint8'),
                     'seg_fusion':{'t1':None,'t1+':None,'t2':None,'flair':None},
                     'seg_fusion_d':{'t1':None,'t1+':None,'t2':None,'flair':None},
                     'seg':None,
                     'bbox':{'ax':np.zeros(dim,dtype='uint8'),'sag':np.zeros(dim,dtype='uint8'),'cor':np.zeros(dim,dtype='uint8')}
                     }
        # initializing with a one-slice bbox might no longer be needed, use set_bbox instead. 
        if bool(bbox):
            self.set_bbox(bbox)

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
                    elif prompt == 'point':
                        self.data['bbox'][orient][r],self.bboxs[r] = self.get_point_mask(mask[r])  
        elif orient == 'sag':
            if slice is None:
                rslice = range(np.shape(mask)[2]) # do all slices
            else:
                rslice = [slice]
            for r in rslice:
                if len(np.where(mask[:,:,r])[0]):
                    if prompt == 'bbox':
                        self.data['bbox'][orient][:,:,r],_ = self.get_bbox_mask(mask[:,:,r])
                    elif prompt == 'point':
                        self.data['bbox'][orient][:,:,r],_ = self.get_point_mask(mask[:,:,r])  
        elif orient == 'cor':
            if slice is None:
                rslice = range(np.shape(mask)[1]) # do all slices
            else:
                rslice = [slice]
            for r in rslice:
                if len(np.where(mask[:,r,:])[0]):
                    if prompt == 'bbox':
                        self.data['bbox'][orient][:,r,:],_ = self.get_bbox_mask(mask[:,r,:])
                    elif prompt == 'point':
                        self.data['bbox'][orient][:,r,:],_ = self.get_point_mask(mask[:,r,:])  



        
    def get_point_mask(self,mask):
        cy,cx = map(int,np.round(np.mean(np.where(mask),axis=1)))
        mask = np.zeros_like(mask)
        mask[cy,cx] = 1

        bbox = {'ax':None,'p0':None,'p1':None,'plot':None,'l':None,'slice':None}
        bbox['p0'] = [cx,cy]

        return mask,bbox

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

    # set a bbox for a given slice    
    def set_bbox(self,bbox):
        for k in bbox.keys():
            if k in list(self.bbox.keys()):
                self.bbox[k] = bbox[k]
            else:
                raise KeyError
        self.create_prompt_from_bbox()

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
