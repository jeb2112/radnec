from tkinter import *
import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import matplotlib
matplotlib.use('TkAgg')

# extends to force a square zoom
class NavigationBar(NavigationToolbar2Tk):

    def __init__(self,canvas,frame,pack_toolbar=False,ui=None,axs=None):
        super().__init__(canvas,frame,pack_toolbar=pack_toolbar)
        if axs:
            self.axs = axs
        self.ui = ui

    def pan(self, *args):
        super().pan(*args)
        self.canvas.get_tk_widget().config(cursor='hand2')

    def zoom(self, *args):
        super().zoom(*args)
        self.canvas.get_tk_widget().config(cursor='tcross')

    # re-calculate zoom coords to force square ROI
    def zoom_coords(self,start_xy,event):
        dx = event.x-start_xy[0]
        dy = start_xy[1]-event.y
        if abs(dx) > abs(dy):
            if dx > 0:
                if dy > 0:
                    event.y = event.y - (dx-dy)
                else:
                    event.y = event.y + (dx+dy)
            else:
                if dy > 0:
                    event.y = event.y + (dx+dy)
                else:
                    event.y = event.y - (dx-dy)
        else:
            if dy > 0:
                if dx > 0:
                    event.x = event.x + (dy-dx)
                else:
                    event.x = event.x - (dy+dx)
            else:
                if dx > 0:
                    event.x = event.x - (dy+dx)
                else:
                    event.x = event.x + (dy-dx)
        return event.x,event.y

    # fix the mouse up coords to force square ROI
    def release_zoom(self, event):
        """Callback for mouse button release in zoom to rect mode."""
        if self._zoom_info is None:
            return

        # We don't check the event button here, so that zooms can be cancelled
        # by (pressing and) releasing another mouse button.
        self.canvas.mpl_disconnect(self._zoom_info.cid)
        self.remove_rubberband()
        # force release event to be square
        event.x,event.y = self.zoom_coords(self._zoom_info.start_xy,event)
        start_x, start_y = self._zoom_info.start_xy
        key = event.key
        # Force the key on colorbars to ignore the zoom-cancel on the
        # short-axis side
        if self._zoom_info.cbar == "horizontal":
            key = "x"
        elif self._zoom_info.cbar == "vertical":
            key = "y"
        # Ignore single clicks: 5 pixels is a threshold that allows the user to
        # "cancel" a zoom action by zooming by less than 5 pixels.
        if ((abs(event.x - start_x) < 5 and key != "y") or
                (abs(event.y - start_y) < 5 and key != "x")):
            self.canvas.draw_idle()
            self._zoom_info = None
            return

        for i, ax in enumerate(self._zoom_info.axes):
            # Detect whether this Axes is twinned with an earlier Axes in the
            # list of zoomed Axes, to avoid double zooming.
            twinx = any(ax.get_shared_x_axes().joined(ax, prev)
                        for prev in self._zoom_info.axes[:i])
            twiny = any(ax.get_shared_y_axes().joined(ax, prev)
                        for prev in self._zoom_info.axes[:i])
            ax._set_view_from_bbox(
                (start_x, start_y, event.x, event.y),
                self._zoom_info.direction, key, twinx, twiny)
            
        # display coordinate scaling for zooming the sag/cor axes. The scaling consists
        # of the slicefovratio number of slices / inplane pixel dimension, assuming isotropic,
        # an x offset to account for width of t1 and t2 panels,
        # and a y offset for the sag or cor pane.
        # dpi 100 hard-coded
        panelnum = int(start_x/(self.ui.config.PanelSize*100))    
        slicefovratio = self.ui.config.ImageDim[0]/self.ui.config.ImageDim[1]
        start_x_sagcor = ((start_x/(self.ui.config.PanelSize*100))*(2/slicefovratio) + (2-panelnum)*self.ui.config.PanelSize) * 100
        start_y_sagcor = ((start_y/(self.ui.config.PanelSize*100))*(2) + 2) * 100
        event_x_sagcor = ((event.x/(self.ui.config.PanelSize*100))*(2/slicefovratio) + (2-panelnum)*self.ui.config.PanelSize) * 100
        event_y_sagcor = ((event.y/(self.ui.config.PanelSize*100))*(2) + 2) * 100
        # zoom the coronal
        self.axs['C']._set_view_from_bbox(
            (start_x_sagcor,start_y_sagcor,event_x_sagcor,event_y_sagcor),
            self._zoom_info.direction, None, False, False)
        # zoom the sagittal
        self.axs['D']._set_view_from_bbox(
            (start_x_sagcor,start_y_sagcor-self.ui.config.PanelSize/2*100,event_x_sagcor,event_y_sagcor-self.ui.config.PanelSize/2*100),
            self._zoom_info.direction, None, False, False)

        self.canvas.draw_idle()
        self._zoom_info = None
        self.push_current()

    # show the drawn rubberband ROI as a square
    def drag_zoom(self, event):
        """Callback for dragging in zoom mode."""
        start_xy = self._zoom_info.start_xy
        ax = self._zoom_info.axes[0]
        # force drawn ROI coords to be square
        event.x,event.y = self.zoom_coords(start_xy,event)            
        (x1, y1), (x2, y2) = np.clip(
            [start_xy, [event.x, event.y]], ax.bbox.min, ax.bbox.max)
        key = event.key
        # ax.bbox.intervalx=np.array([0,400])
        # ax.bbox.intervaly=np.array([0,400])
        # ax.bbox.bounds = (0,0,400,400)
        # Force the key on colorbars to extend the short-axis bbox
        if self._zoom_info.cbar == "horizontal":
            key = "x"
        elif self._zoom_info.cbar == "vertical":
            key = "y"
        if key == "x":
            y1, y2 = ax.bbox.intervaly
        elif key == "y":
            x1, x2 = ax.bbox.intervalx

        self.draw_rubberband(event, x1, y1, x2, y2)
