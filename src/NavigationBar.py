from tkinter import *
import tkinter as tk
import numpy as np
import os
import time
from contextlib import contextmanager

from collections import namedtuple
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.backends._backend_tk import ToolTip
from matplotlib.backend_bases import _Mode
from matplotlib import cbook,backend_tools
# from enum import Enum
import enum
import matplotlib
matplotlib.use('TkAgg')
from PIL import Image, ImageTk

# added an enum for window level, but can't find a way to make use of it
class MyCursors(enum.IntEnum):  # Must subclass int for the macOS backend.
    """Backend-independent cursor types."""
    POINTER = enum.auto()
    HAND = enum.auto()
    SELECT_REGION = enum.auto()
    MOVE = enum.auto()
    WAIT = enum.auto()
    RESIZE_HORIZONTAL = enum.auto()
    RESIZE_VERTICAL = enum.auto()
    WL = enum.auto()
    CROSSHAIR = enum.auto()
    MEASURE = enum.auto()
cursors = MyCursors  # Backcompat.

# this global does not override the global in backend, so can't use it.
cursord = {
    cursors.MOVE: "fleur",
    cursors.HAND: "hand2",
    cursors.POINTER: "arrow",
    cursors.SELECT_REGION: "crosshair",
    cursors.WAIT: "watch",
    cursors.RESIZE_HORIZONTAL: "sb_h_double_arrow",
    cursors.RESIZE_VERTICAL: "sb_v_double_arrow",
    cursors.WL: "circle",
    cursors.CROSSHAIR: "tcross",
    cursors.MEASURE: "sizing"
}

# overriden to add a WL mode
class _Mode(str, enum.Enum):
    NONE = ""
    PAN = "pan/zoom"
    ZOOM = "zoom square"
    WL = "window/level"
    CROSSHAIR = "crosshair"
    MEASURE = "measure"

    def __str__(self):
        return self.value

    @property
    def _navigate_mode(self):
        return self.name if self is not _Mode.NONE else None

# extends NavigationToolbar2Tk to change zoom rect to zoom square.
# then further extended to implement a window/level button
# and a crosshair overlay button
class NavigationBar(NavigationToolbar2Tk):

    def __init__(self,canvas,frame,pack_toolbar=False,ui=None,axs=None):
        self.ui = ui
        if axs:
            self.axs = axs
        # don't use the rect Zoom or subplots
        self.toolitems = (
            ('Home', 'Bob Reset original view', 'home', 'home'),
            ('Back', 'Back to previous view', 'back', 'back'),
            ('Forward', 'Forward to next view', 'forward', 'forward'),
            (None, None, None, None),
            ('Pan',
            'Left button pans, Right button zooms\n'
            'x/y fixes axis, CTRL fixes aspect',
            'move', 'pan'),
            # ('Zoom', 'Zoom to rectangle\nx/y fixes axis', 'zoom_to_rect', 'zoom'),
            # ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
            (None, None, None, None),
            ('Save', 'Save the figure', 'filesave', 'save_figure'),
        )
        super().__init__(canvas,frame,pack_toolbar=pack_toolbar)

        # add the window/level button
        path = os.path.join(self.ui.config.UIResourcesPath,'contrast_icon.png')
        self._buttons['WL'] = button = self._Button('WL',path,toggle=True,command=getattr(self,'windowlevel'))
        # position it alongside the Pan button
        self._buttons['WL'].pack_forget
        self._buttons['WL'].pack(after=self._buttons['Pan'])
        ToolTip.createToolTip(button, 'Adjust window and level')

        # add the crosshairs button
        path = os.path.join(self.ui.config.UIResourcesPath,'crosshair_icon.png')
        self._buttons['crosshair'] = button = self._Button('crosshair',path,toggle=True,command=getattr(self,'crosshair'))
        # position it alongside the Pan button
        self._buttons['crosshair'].pack_forget
        self._buttons['crosshair'].pack(after=self._buttons['WL'])
        ToolTip.createToolTip(button, 'Display 3d crosshair cursor')

        # add the measurement button
        path = os.path.join(self.ui.config.UIResourcesPath,'measurement_icon.png')
        self._buttons['measure'] = button = self._Button('measure',path,toggle=True,command=getattr(self,'measure'))
        # position it alongside the Pan button
        self._buttons['measure'].pack_forget
        self._buttons['measure'].pack(after=self._buttons['WL'])
        ToolTip.createToolTip(button, 'Display measurement cursor')

        self.update()


    def measure(self,*args):
        """
        Toggle the measurement overlay.
        """
        if not self.canvas.widgetlock.available(self):
            self.set_message("measurement unavailable")
            return
        if self.mode == _Mode.MEASURE:
            self.mode = _Mode.NONE
            self.canvas.widgetlock.release(self)
        else:
            self.mode = _Mode.MEASURE
            self.canvas.widgetlock(self)
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self.mode._navigate_mode)

        if self.mode == _Mode.MEASURE:
            self.canvas.get_tk_widget().config(cursor='sizing')
            self.ui.root.bind('<B1-Motion>',self.ui.sliceviewerframe.b1motion_measure)
            self.ui.root.bind('<Button-1>',self.ui.sliceviewerframe.b1click)
        else:
            self.canvas.get_tk_widget().config(cursor='arrow')
            self.ui.root.unbind('<B1-Motion>')
            self.ui.root.unbind('<Button-1>')
            self.ui.sliceviewerframe.clear_measurement()
        self._update_buttons_checked()

    def crosshair(self,*args):
        """
        Toggle the 3d crosshair overlay.

        Move cursor with left mouse, overlay follows. 
        """
        if not self.canvas.widgetlock.available(self):
            self.set_message("window/level unavailable")
            return
        if self.mode == _Mode.CROSSHAIR:
            self.mode = _Mode.NONE
            self.canvas.widgetlock.release(self)
        else:
            self.mode = _Mode.CROSSHAIR
            self.canvas.widgetlock(self)
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self.mode._navigate_mode)

        if self.mode == _Mode.CROSSHAIR:
            self.canvas.get_tk_widget().config(cursor='tcross')
            self.ui.root.bind('<B1-Motion>',self.ui.sliceviewerframe.b1motion_crosshair)
        else:
            self.canvas.get_tk_widget().config(cursor='arrow')
            self.ui.root.unbind('<B1-Motion>')
            self.ui.sliceviewerframe.clear_crosshair()
        self._update_buttons_checked()

    # callback for the WL button
    def windowlevel(self,*args):
        """
        Toggle the window/level tool.

        Adjust window with left button scrolled left/right,
        level with left button scrolled up/down.
        """
        if not self.canvas.widgetlock.available(self):
            self.set_message("window/level unavailable")
            return
        if self.mode == _Mode.WL:
            self.mode = _Mode.NONE
            self.canvas.widgetlock.release(self)
        else:
            self.mode = _Mode.WL
            self.canvas.widgetlock(self)
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self.mode._navigate_mode)

        if self.mode == _Mode.WL:
            self.canvas.get_tk_widget().config(cursor='circle')
            self.ui.root.bind('<B1-Motion>',self.ui.sliceviewerframe.b1motion)
        else:
            self.canvas.get_tk_widget().config(cursor='arrow')
            self.ui.root.unbind('<B1-Motion>')
            self.ui.sliceviewerframe.b1release()
        self._update_buttons_checked()
            
    # overriding some tk backend functions to implement window/level
    # can't modify cursord because it is global in backend, so can't use set_cursor()
    def _update_cursor(self, event):
        """
        Update the cursor after a mouse move event or a tool (de)activation.
        """
        if self.mode and event.inaxes and event.inaxes.get_navigate():
            if (self.mode == _Mode.ZOOM
                    and self._last_cursor != backend_tools.Cursors.SELECT_REGION):
                self.canvas.set_cursor(backend_tools.Cursors.SELECT_REGION)
                self._last_cursor = backend_tools.Cursors.SELECT_REGION
            elif (self.mode == _Mode.PAN
                  and self._last_cursor != backend_tools.Cursors.MOVE):
                self.canvas.set_cursor(backend_tools.Cursors.MOVE)
                self._last_cursor = backend_tools.Cursors.MOVE
            elif (self.mode == _Mode.WL
                  and self._last_cursor != MyCursors.WL):
                # self.canvas.set_cursor(MyCursors.WL)
                self.canvas.get_tk_widget().config(cursor="circle")
                self._last_cursor = MyCursors.WL
            elif (self.mode == _Mode.CROSSHAIR
                  and self._last_cursor != MyCursors.CROSSHAIR):
                # self.canvas.set_cursor(MyCursors.WL)
                self.canvas.get_tk_widget().config(cursor="tcross")
                self._last_cursor = MyCursors.CROSSHAIR
        elif self._last_cursor != backend_tools.Cursors.POINTER:
            self.canvas.set_cursor(backend_tools.Cursors.POINTER)
            self._last_cursor = backend_tools.Cursors.POINTER

    # again, can't use set_cursor() for WL cursor
    @contextmanager
    def _wait_cursor_for_draw_cm(self):
        """
        Set the cursor to a wait cursor when drawing the canvas.

        In order to avoid constantly changing the cursor when the canvas
        changes frequently, do nothing if this context was triggered during the
        last second.  (Optimally we'd prefer only setting the wait cursor if
        the *current* draw takes too long, but the current draw blocks the GUI
        thread).
        """
        self._draw_time, last_draw_time = (
            time.time(), getattr(self, "_draw_time", -np.inf))
        if self._draw_time - last_draw_time > 1:
            try:
                self.canvas.set_cursor(backend_tools.Cursors.WAIT)
                yield
            finally:
                if self._last_cursor == MyCursors.WL:
                    self.canvas.get_tk_widget().config(cursor="circle")
                elif self._last_cursor == MyCursors.CROSSHAIR:
                    self.canvas.get_tk_widget().config(cursor="tcross")
                elif self._last_cursor == MyCursors.MEASURE:
                    self.canvas.get_tk_widget().config(cursor="tcross")
                else:
                    self.canvas.set_cursor(self._last_cursor)
        else:
            yield

    # _WLInfo = namedtuple("_WLInfo", "button axes cid")
    # override to add WL
    def _update_buttons_checked(self):
        # sync button checkstates to match active mode
        for text, mode in [('Zoom', _Mode.ZOOM), ('Pan', _Mode.PAN),('WL',_Mode.WL),('Measure',_Mode.MEASURE)]:
            if text in self._buttons:
                if self.mode == mode:
                    self._buttons[text].select()  # NOT .invoke()
                else:
                    self._buttons[text].deselect()

    # overriding to set the cursor.
    def pan(self, *args):
        super().pan(*args)
        self.canvas.get_tk_widget().config(cursor='hand2')

    # overriding to set the cursor
    def zoom(self, *args):
        super().zoom(*args)
        self.canvas.get_tk_widget().config(cursor='tcross')

    # figure out which slice view the mouse event is on
    def select_artist(self,event,sv=None):
        if sv is None:
            sv = self.ui.function.get()
        if sv == 'BLAST' or sv == 'overlay':
            pdim = self.ui.current_panelsize*self.ui.config.dpi
            if event.x <= pdim:
                a = self.axs['A'].images[0]
            elif event.x <= 2*pdim:
                a = self.axs['B'].images[0]
            else:
                a = None
            return a
        elif sv == '4panel':
            pdim = self.ui.current_panelsize*self.ui.config.dpi/2
            if event.x <= pdim and event.y <= pdim:
                a = self.axs['A'].images[0]
            elif event.x <= 2*pdim and event.y <= pdim:
                a = self.axs['B'].images[0]
            elif event.x <= pdim and event.y <= 2*pdim:
                a = self.axs['C'].images[0]
            elif event.x <= 2*pdim and event.y <= 2*pdim:
                a = self.axs['D'].images[0]
            else:
                a = None
            return a

                

    # override this method to provide for 3d crosshair overlay
    def mouse_move(self, event):
        self._update_cursor(event)
        self.set_message(self._mouse_event_to_message(event))
        # if self._Mode
                
    # override this method to not check for get_navigate() because the labelling axes 
    # are set to be non-navigable. 
    # also, find the image artist that corresponds to the mouse event
    def _mouse_event_to_message(self,event):
        # if event.inaxes and event.inaxes.get_navigate():
        if event.inaxes:
            try:
                s = event.inaxes.format_coord(event.xdata, event.ydata)
            except (ValueError, OverflowError):
                pass
            else:
                s = s.rstrip()
                a = self.select_artist(event)
                # artists = [a for a in event.inaxes._mouseover_set
                #            if a.contains(event)[0] and a.get_visible()]
                if a:
                    # a = cbook._topmost_artist(artists)
                    # a = NavigationBar._bottommost_artist(artists)
                    if a is not event.inaxes.patch:
                        data = a.get_cursor_data(event)
                        if data is not None:
                            data_str = a.format_cursor_data(data).rstrip()
                            if data_str:
                                s = s + '\n' + data_str
                return s
        return ""

    # no longer using rect zoom in the menu
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
