from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtGui import QTransform
from .polygonVertex import PolygonVertex


class View(QGraphicsView):
    """
    The main View.
    """

    def __init__(self, *args, **kwargs):
        """Constructor."""
        super(View, self).__init__(*args, **kwargs)
        self.zoom = 1
        self.setMouseTracking(True)
        self.scn = args[0]

    def wheelEvent(self, event):
        """
        Zoom in or out on mouse-wheel event.
        """
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        # Save the scene pos
        old_pos = self.mapToScene(event.pos())

        # Zoom
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.scale(zoom_factor, zoom_factor)

        # Get the new position
        new_pos = self.mapToScene(event.pos())

        # Move scene to old position
        delta = new_pos - old_pos
        self.zoom = delta
        self.translate(delta.x(), delta.y())
        self.scale_item_components(1.25)

    def zoom_in(self):
        """
        Zoom the scene in
        """
        zoom_in_factor = 1.25

        f = float(zoom_in_factor)
        self.scale(f, f)
        self.scale_item_components(f)

    def zoom_out(self):
        """
        Zoom the scene out
        """
        zoom_out_factor = 1.25

        f = 1.0 / float(zoom_out_factor)
        self.scale(f, f)
        self.scale_item_components(f)

    def scale_item_components(self, factor):

        for item in self.items():
            if type(item) is PolygonVertex:
                item.setTransform(QTransform.fromScale(factor, factor), True)
                self.scn.removeItem(item)
                self.scn.addItem(item)
