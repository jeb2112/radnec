import numpy as np

# collectding various ROI stats and data
# for BLAST segmentations
class ROI():
    def __init__(self,xpos,ypos,slice,compartment='ET'):

        self.casename = None
        self.status = False

        # segmentation/contour masks
        self.data = {'WT':None,'ET':None,'TC':None,'contour':{'WT':None,'ET':None},
                     'seg_fusion':{'t1':None,'t1+':None,'t2':None,'flair':None},
                     'seg_fusion_d':{'t1':None,'t1+':None,'t2':None,'flair':None},
                     'seg':None}

        # ROI selection coordinates
        self.coords = {'ET':{},'necrosis':{},'T2 hyper':{}}
        self.coords[compartment]['x'] = xpos
        self.coords[compartment]['y'] = ypos
        self.coords[compartment]['slice'] = slice

        # threshold gates saved as intermediate values
        self.gate = {'brain':None,'ET':None,'T2 hyper':None}

        # output stats
        self.stats = {'spec':{'ET':0,'TC':0,'WT':0},
            'sens':{'ET':0,'TC':0,'WT':0},
            'dsc':{'ET':0,'TC':0,'WT':0},
            'vol':{'ET':0,'TC':0,'WT':0},
            'elapsed_time':0}

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
