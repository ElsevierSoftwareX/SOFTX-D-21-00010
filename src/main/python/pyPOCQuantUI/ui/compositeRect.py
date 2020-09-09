
from PyQt5.QtCore import QRectF
from PyQt5.QtWidgets import QGraphicsRectItem


class CompositeRect:
    """Composite rect.

    This is not a QGraphicsItem and cannot be added to a QGraphicsScene directly.
    If a QGraphicsScene is passed to it in the constructor or as an argument to
    addToScene(), it will manage a series of QGraphicsItems (one 'Line' and two
    'Vertex' items) and their spatial relationships.
    """

    def __init__(self, left_rect: QRectF = QRectF(0, 0, 100, 100), right_rect: QRectF = QRectF(0, 0, 200, 100),
                 scene=None):
        """Constructor."""

        # Keep track of which object is being actively moved; since the others
        # will follow programmatically, but will still fire their itemChanged
        # events
        self._selectedItem = None

        # Keep track of last position during a move
        self._lastPosition = None

        # Add left rect
        self._left_rect = QGraphicsRectItem(left_rect)

        # Add right rect
        self._right_rect = QGraphicsRectItem(right_rect)

        # Store the scene
        self._scene = scene

        # Do we have a scene already?
        if self._scene is not None:
            self.addToScene(self._scene)

    def updateRect(self, left_rect, right_rect):
        # Update left and right rect
        self._left_rect.setRect(left_rect)
        self._right_rect.setRect(right_rect)

    def addToScene(self, scene):
        """Add Rect objects to the scene."""
        if scene is not None:
            self._scene = scene
            self._scene.addItem(self._left_rect)
            self._scene.addItem(self._right_rect)
