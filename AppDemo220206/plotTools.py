from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QSizePolicy

class Myplot(FigureCanvas):
    # def __init__(self, parent=None, width=5, height=3, dpi=100):
    def __init__(self, width=5, height=3, dpi=100):

        # new figure
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)#, facecolor="gray")

        # activate figure window
        FigureCanvas.__init__(self, self.fig)
        # sub plot by self.axes
        self.axes = self.fig.add_subplot(111)
        self.axes.set_ylim([-0.1, 0.1])
        self.axes.xaxis.set_visible(False)
        self.axes.yaxis.set_visible(False)
        self.axes.set()  # facecolor="orange")
        # initial figure
        self.compute_initial_figure()

        # size policy
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass