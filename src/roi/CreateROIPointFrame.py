import tkinter as tk
from tkinter import ttk

import numpy as np
import re
import copy

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Ellipse

from src.CreateFrame import CreateFrame,Command
from src.roi.ROI import ROIBLAST,ROISAM,ROIPoint,Point
from src.OverlayPlots import *

##############################################################
# frame layout for setting BLAST parameters by point selection
# ############################################################

class CreateROIPointFrame(CreateFrame):
    def __init__(self,frame,ui=None,padding='10'):
        super().__init__(frame,ui=ui,padding=padding)

        self.currentpt = tk.IntVar(value=0)
        self.pointradius = tk.IntVar(value=5)

        self.currentpt.trace_add('write',self.updatepointlabel)

        self.SUNKABLE_BUTTON = 'SunkableButton.TButton'
        self.selectPointbutton= ttk.Button(self.frame,text='add Point',command=self.selectPoint,style=self.SUNKABLE_BUTTON)
        self.selectPointbutton.grid(row=0,column=1,sticky='snew')
        self.selectPointstate = False
        removePoint= ttk.Button(self.frame,text='remove Point',command=self.removePoint)
        removePoint.grid(row=1,column=1,sticky='ew')
        self.pointLabel = ttk.Label(self.frame,text=self.currentpt.get(),padding=10)
        self.pointLabel.grid(row=1,column=0,sticky='e')
        self.pointradiuslist = [str(i) for i in range(1,6)]
        self.pointradiusmenu = ttk.OptionMenu(self.frame,self.pointradius,self.pointradiuslist[-1],
                                        *self.pointradiuslist,command=self.pointradius_callback)
        self.pointradiusmenu.config(width=2)
        self.pointradiusmenu.grid(row=2,column=1,sticky='ne')
        pointradius_label = ttk.Label(self.frame,text='radius:')
        pointradius_label.grid(row=2,column=1,sticky='nw')


    #############################
    # methods for point selection
    #############################

    # activates/deactivates Point selection button press events
    def selectPoint(self,event=None):
        if not self.selectPointstate:
            if len(self.ui.pt[self.ui.s]) == 0:
                self.ui.roiframe.roioverlayframe.set_overlay() # deactivate any overlay
                self.ui.roiframe.roioverlayframe.enhancingROI_overlay_callback()
            self.buttonpress_id = self.ui.sliceviewerframe.canvas.callbacks.connect('button_press_event',self.selectPointClick)
            self.setCursor('crosshair')
            self.selectPointstart()
        else:
            self.selectPointstop()
            self.setCursor()
            if self.buttonpress_id:
                self.ui.sliceviewerframe.canvas.callbacks.disconnect(self.buttonpress_id)
                self.buttonpress_id = None
        return

    def selectPointstart(self):
        self.selectPointbutton.state(['pressed', '!disabled'])
        self.fstyle.configure(self.SUNKABLE_BUTTON, relief=tk.SUNKEN, foreground='green')
        self.selectPointstate = True
        # point and slider selection not used interchangeably.
        # but also need a better way to reset blastdata
        self.ui.blastdata[self.ui.s] = copy.deepcopy(self.ui.blastdatadict)

    def selectPointstop(self):
        self.selectPointbutton.state(['!pressed', '!disabled'])
        self.fstyle.configure(self.SUNKABLE_BUTTON, relief=tk.RAISED, foreground='black')
        self.selectPointstate = False

    # processes a cursor selection button press event for generating/updating raw BLAST seg
    def selectPointClick(self,event=None):
        if event:
            if event.button > 1: # ROI selection on left mouse only
                return
            
        if False: # not using this anymore
            self.setCursor('watch')
        if event:
            # print(event.xdata,event.ydata)
            # need check for inbounds
            # convert coords from dummy label axis
            s = event.inaxes.format_coord(event.xdata,event.ydata)
            xdata,ydata = map(float,re.findall(r"(?:\d*\.\d+)",s))
            xdata = int(np.round(xdata))
            ydata = int(np.round(ydata))
            if xdata < 0 or ydata < 0:
                return None
            # elif self.ui.data['raw'][0,self.ui.get_currentslice(),int(event.x),int(event.y)] == 0:
            #     print('Clicked in background')
            #     return

            elif self.ui.blastdata[self.ui.s]['blast']['ET'] is not None:
                if self.ui.blastdata[self.ui.s]['blast']['ET'][self.ui.currentslice][ydata,xdata]:
                # if point is in the existing mask then skip.
                # don't have any good logic to snap to a nearby unmasked 
                # point becuase the intended segmentation can never be inferred by any simple computer logic.
                    return None

            self.createPoint(xdata,ydata,self.ui.get_currentslice())
            
        self.updateBLASTMask()

        # optionally run SAM 2d directly on the partial BLAST ROI, in the current slice only
        # this requires to form a temporary BLAST ROI from the current raw mask, by interpreting the current click event
        # as the ROI selection click, and then delete that ROI immediately afterwards. 
        if self.ui.config.SAM2dauto:
            # rref = self.ui.roi[self.ui.s][self.ui.currentroi].data['seg_fusion'][self.ui.chselection]
            dref = self.ui.data[self.ui.s].dset['seg_fusion'][self.ui.chselection]
            self.ui.sliceviewerframe.set_ortho_slice(event)
            self.ui.roiframe.ROIclick(event=event,do2d=True)
            self.create_comp_mask()
            for ch in [self.ui.chselection]:
                fusion = generate_comp_overlay(self.ui.data[self.ui.s].dset['raw'][ch]['d'],
                                                self.ui.rois['sam'][self.ui.s][self.ui.currentroi].mask,self.ui.currentslice,
                                                layer=self.ui.roiframe.roioverlayframe.layer.get(),
                                                overlay_intensity=self.config.OverlayIntensity)
                # setting data directly instead of via roi.data and updateData()
                dref['d'] = np.copy(fusion)
            self.ui.updateslice()
            
        # keep the crosshair cursor until button is unset.
        self.setCursor('crosshair')

        return None        

    def removePoint(self,event=None):
        if self.currentpt.get() == 0:
            return
        self.ui.pt[self.ui.s].pop()
        self.set_currentpt(-1)
        # awkward. reset current slices
        self.ui.sliceviewerframe.currentslice.set(self.ui.pt[self.ui.s][-1].coords['slice'])
        self.ui.sliceviewerframe.currentsagslice.set(self.ui.pt[self.ui.s][-1].coords['x'])
        self.ui.sliceviewerframe.currentcorslice.set(self.ui.pt[self.ui.s][-1].coords['y'])

        if len(self.ui.pt[self.ui.s]) == 0:
            self.set_overlay() # deactivate any overlay
            self.ui.roiframe.roioverlayframe.enhancingROI_overlay_callback()
            # points and sliders are not used interchangeably.
            self.ui.blastdata[self.ui.s] = copy.deepcopy(self.ui.blastdatadict)
        else:
            self.updateBLASTMask()

            # duplicates selectPointClick above, make separate function for this 
            # or SAM mask could be saved with each point and not re-calculated when
            # a point is removed.
            if self.ui.config.SAM2dauto:
                dref = self.ui.data[self.ui.s].dset['seg_fusion'][self.ui.chselection]
                # need to update this based on current point
                # self.ui.sliceviewerframe.set_ortho_slice(event)
                self.ui.roiframe.ROIclick(coords=(self.ui.pt[self.ui.s][-1].coords['x'],
                                      self.ui.pt[self.ui.s][-1].coords['y']),do2d=True)

                self.create_comp_mask()
                for ch in [self.ui.chselection]:
                    fusion = generate_comp_overlay(self.ui.data[self.ui.s].dset['raw'][ch]['d'],
                                                    self.ui.rois['sam'][self.ui.s][self.ui.currentroi].mask,self.ui.currentslice,
                                                    layer=self.ui.roiframe.roioverlayframe.layer.get(),
                                                    overlay_intensity=self.config.OverlayIntensity)
                    # setting data directly instead of via roi.data and updateData()
                    dref['d'] = np.copy(fusion)
                self.ui.updateslice()


        if 'pressed' in self.selectPointbutton.state():
            self.setCursor('crosshair')
        return

    # records button press coords in a new ROI object
    def createPoint(self,x,y,slice):
        pt = ROIPoint(x,y,slice)
        self.ui.pt[self.ui.s].append(pt)
        self.set_currentpt(1)

    # convenience method for canvas updates
    # probably should be in SVFrame
    def setCursor(self,cursor=None):
        if cursor is None:
            cursor = 'arrow'
        self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor=cursor)
        self.ui.sliceviewerframe.canvas.get_tk_widget().update_idletasks()

    # create updated BLAST seg from collection of ROI Points
    def updateBLASTMask(self,layer=None,currentslice=True):
        if layer is None:
            layers = ['ET','T2 hyper']
        else:
            layers = [layer]
        for l in layers:
            for k in self.ui.blastdata[self.ui.s]['blastpoint']['params'][l].keys():
                self.ui.blastdata[self.ui.s]['blastpoint']['params'][l][k] = []
            for i,pt in enumerate(self.ui.pt[self.ui.s]):
                dslice_t1 = self.ui.data[self.ui.s].dset['z']['t1+']['d'][pt.coords['slice']]
                dslice_flair = self.ui.data[self.ui.s].dset['z']['flair']['d'][pt.coords['slice']]

                dpt_t1 = dslice_t1[pt.coords['y'],pt.coords['x']]
                dpt_flair = dslice_flair[pt.coords['y'],pt.coords['x']]
                self.ui.blastdata[self.ui.s]['blastpoint']['params'][l]['pt'].append((dpt_flair,dpt_t1))

                # create grid
                x = np.arange(0,self.ui.sliceviewerframe.dim[2])
                y = np.arange(0,self.ui.sliceviewerframe.dim[1])
                vol = np.array(np.meshgrid(x,y,indexing='xy'))

                # circular region of interest around point. should check and exclude background pixels though.
                croi = np.where(np.sqrt(np.power((vol[0,:]-pt.coords['x']),2)+np.power((vol[1,:]-pt.coords['y']),2)) < self.pointradius.get())

                # centroid of ellipse is mean of circular roi
                if False:
                    mu_t1 = np.mean(dslice_t1[croi])
                    mu_flair = np.mean(dslice_flair[croi])
                # centroid of the ellipse will be the clicked point
                else:
                    mu_t1 = dpt_t1
                    mu_flair = dpt_flair

                self.ui.blastdata[self.ui.s]['blastpoint']['params'][l]['meant12'].append(mu_t1)
                self.ui.blastdata[self.ui.s]['blastpoint']['params'][l]['meanflair'].append(mu_flair)
                if True: # use stats in the croi about the clicked point
                    std_t1 = np.std(dslice_t1[croi])
                    std_flair = np.std(dslice_flair[croi])
                    e = copy.copy(std_flair) / copy.copy(std_t1)
                    # if the selected point is outside the elliptical roi in parameter space, increase the standard 
                    # deviations proportionally to include it. ie pointclustersize is arbitrary.
                    # this is copied from Blastbratsv3.py should combine into one available method
                    pointclustersize = 1
                    point_perimeter = Ellipse((mu_flair, mu_t1), 2*pointclustersize*std_flair,2*pointclustersize*std_t1)
                    unitverts = point_perimeter.get_path().vertices
                    pointverts = point_perimeter.get_patch_transform().transform(unitverts)
                    xy_layerverts = np.transpose(np.vstack((pointverts[:,0],pointverts[:,1])))
                    p = Path(xy_layerverts,closed=True)
                    if not p.contains_point((dpt_flair,dpt_t1)):
                        std_flair = np.sqrt((dpt_flair-mu_flair)**2 + (dpt_t1-mu_t1)**2 / (std_t1/std_flair)**2) * 1.01
                        std_t1 = std_flair / e
                else: # use k-means centroid std
                    std_flair = self.ui.blastdata[self.ui.s]['blast']['params'][l]['stdflair']
                    std_t1 = self.ui.blastdata[self.ui.s]['blast']['params'][l]['stdt12'] 

                self.ui.blastdata[self.ui.s]['blastpoint']['params'][l]['stdt12'].append(std_t1)
                self.ui.blastdata[self.ui.s]['blastpoint']['params'][l]['stdflair'].append(std_flair)

        # functionality similar to updateslider() should reconcile
        # in the original workflow, the raw BLAST mask was being shown
        if False:
            self.ui.roiframe.roioverlayframe.set_overlay('BLAST')
            self.ui.roiframe.roioverlayframe.enhancingROI_overlay_callback()
        else: # in the new workflow, only the SAM ROI mask is depicted
            self.ui.roiframe.roioverlayframe.SAM_overlay_callback()
        self.ui.blastdata[self.ui.s]['blast']['gates']['ET'] = None
        self.ui.blastdata[self.ui.s]['blast']['gates']['T2 hyper'] = None
        self.ui.update_blast(layer='ET')
        self.ui.update_blast(layer='T2 hyper')
        self.ui.runblast(currentslice=currentslice)

        return
    
    def set_currentpt(self,delta):
        val = self.currentpt.get() + delta
        self.currentpt.set(value=val)
        self.ui.set_currentpt()    

    def updatepointlabel(self,*args):
        try:
            self.pointLabel['text'] = '{:d}'.format(self.currentpt.get())
        except KeyError as e:
            print(e)

    def pointradius_callback(self,radius):
        self.pointradius.set(int(radius))

    # create a composite mask from a SAM and a BLAST raw mask
    def create_comp_mask(self):
        s = self.ui.s
        roi = self.ui.currentroi
        layer = self.ui.roiframe.roioverlayframe.layerSAM.get()
        rref = self.ui.rois['sam'][s][roi]
        self.ui.rois['sam'][s][roi].mask = 5*self.ui.rois['blast'][s][roi].data[layer] + \
                                                     1*self.ui.rois['sam'][s][roi].data[layer]
        # add BLAST selection points
        for i,pt in enumerate(self.ui.pt[self.ui.s]):
            rref.mask[pt.coords['slice'],pt.coords['y'],pt.coords['x']] = 8

        return
    
    # copy points info from a SAM ROI to self.ui.pt list\
    # this is a stopgap method. ROI points and ui pt list for BLAST should be reconciled properly
    def copy_points(self):
        for pt in self.ui.rois['sam'][self.ui.s][self.ui.currentroi].pts:
            if pt['fg']: # back ground points will be re-generated automatically from BLAST mask
                uipt = ROIPoint(int(np.round(pt['p0'][0])),int(np.round(pt['p0'][1])),int(pt['slice']))
                self.ui.pt[self.ui.s].append(uipt)
                

