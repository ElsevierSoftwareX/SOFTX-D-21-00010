from PyQt5.QtWidgets import QGraphicsEllipseItem


class Circle(QGraphicsEllipseItem):
    """A vertex."""

    def __init__(self, x, y, radius=5.0, name="", composite=None):
        """Constructor."""

        self._diameter = 2 * radius
        self._radius = radius
        self._name = name
        self._composite = composite

        # Call the parent constructor
        super(Circle, self).__init__(0, 0, self._diameter, self._diameter, parent=None)
