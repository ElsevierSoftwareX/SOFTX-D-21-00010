from PyQt5.QtWidgets import QGraphicsView


class View(QGraphicsView):
    """
    The main View.
    """

    def __init__(self, *args, **kwargs):
        """Constructor."""
        super(View, self).__init__(*args, **kwargs)
        self.zoom = 1

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