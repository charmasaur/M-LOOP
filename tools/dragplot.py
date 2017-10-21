import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Require presses to be within this distance of a point in order to snap to it.
SNAP_THRESHOLD = 0.5

class DragPlot:
    def __init__(self, axes, data, listener):
        self.listener = listener
        self.picked = None

        self.scat = axes.scatter(np.arange(len(data)), data, picker = True)
        self.plot = axes.plot(np.arange(len(data)), data)[0]
        axes.get_figure().canvas.mpl_connect('motion_notify_event', self._onmotion)
        axes.get_figure().canvas.mpl_connect('button_press_event', self._onpress)
        axes.get_figure().canvas.mpl_connect('button_release_event', self._onrelease)

    def update(self, data):
        self.scat.set_offsets(np.transpose(np.stack([np.arange(len(data)), data])))
        self.plot.set_ydata(data)

    def _onpress(self, event):
        if event.inaxes != self.plot.axes:
            return
        xi = int(round(event.xdata))
        if abs(xi-event.xdata)>SNAP_THRESHOLD:
            return
        if xi < 0 or xi >= len(self.plot.get_xdata()):
            return
        self.picked = xi
        self.listener(self.picked, event.ydata)

    def _onmotion(self, event):
        if self.picked is None:
            return
        if event.ydata is None:
            return
        self.listener(self.picked, event.ydata)

    def _onrelease(self, event):
        self.picked = None
