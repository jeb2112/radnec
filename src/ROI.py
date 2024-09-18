import numpy as np

# collectding various ROI stats and data
# for BLAST/SAM segmentations
class ROI():
    def __init__(self,xpos,ypos,slice,compartment='ET'):

        self.casename = None
        self.status = False

        # segmentation/contour masks
        # seg_fusion - an overlay image
        # seg_fusion_d - a copy of overaly image for display purposes
        # bbox - 3d volume of 2d bbox prompts of each in-plane slice, for use with SAM
        self.data = {'WT':None,'ET':None,'TC':None,'contour':{'WT':None,'ET':None},
                     'seg_fusion':{'t1':None,'t1+':None,'t2':None,'flair':None},
                     'seg_fusion_d':{'t1':None,'t1+':None,'t2':None,'flair':None},
                     'seg':None,
                     'bbox':None}

        # BLAST ROI selection coordinates from mouse click
        self.coords = {'ET':{},'necrosis':{},'T2 hyper':{}}
        self.coords[compartment]['x'] = xpos
        self.coords[compartment]['y'] = ypos
        self.coords[compartment]['slice'] = slice

        # SAM drawn bbox coordinates and values, or bbox from BLAST mask
        self.bboxs = {}

        # threshold gates saved as intermediate values
        self.gate = {'brain':None,'ET':None,'T2 hyper':None}

        # output stats
        self.stats = {'spec':{'ET':0,'TC':0,'WT':0},
            'sens':{'ET':0,'TC':0,'WT':0},
            'dsc':{'ET':0,'TC':0,'WT':0},
            'vol':{'ET':0,'TC':0,'WT':0},
            'hd':{'ET':0,'TC':0,'WT':0},
            'elapsedtime':0}

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
