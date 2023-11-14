
class ROI():
    def __init__(self,xpos,ypos,slice,compartment='ET'):

        self.casename = None
        self.status = False

        # segmentation masks
        self.data = {'WT':None,'ET':None,'TC':None}

        # BLAST values
        self.data['blast_params'] = {'stdt1':1,'stdt2':1,'meant1':1,'meant2':1,
                               'ET':{'t1':None,'t2':None,'bc':None},
                               'T2 hyper':{'t1':None,'t2':None,'bc':None}}

        # ROI selection coordinates
        self.coords = {'ET':{},'necrosis':{},'T2 hyper':{}}
        self.coords[compartment]['x'] = xpos
        self.coords[compartment]['y'] = ypos
        self.coords[compartment]['slice'] = slice

        # threshold gates saved as intermediate values
        self.gate = {'brain':None,'ET':None,'T2 hyper':None}
        # self.brain = None
        # self.et_gate = None
        # self.wt_gate = None

        # output stats
        self.stats = {'spec':{'ET':0,'TC':0,'WT':0},
            'sens':{'ET':0,'TC':0,'WT':0},
            'dsc':{'ET':0,'TC':0,'WT':0},
            'vol':{'ET':0,'TC':0,'WT':0,'manual_ET':0,'manual_TC':0,'manual_WT':0},
            # 'gatecount':{'t1':0,'t2':0},
            'elapsed_time':0}
