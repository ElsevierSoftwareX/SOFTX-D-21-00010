
from PyQt5.QtCore import QPointF
from .line import Line
from .vertex import Vertex


class CompositeLine:
    """Composite line.

    This is not a QGraphicsItem and cannot be added to a QGraphicsScene directly.
    If a QGraphicsScene is passed to it in the constructor or as an argument to
    addToScene(), it will manage a series of QGraphicsItems (one 'Line' and two
    'Vertex' items) and their spatial relationships.
    """

    def __init__(self, pos: QPointF = QPointF(-20.0, 400.0), line_length=40, scene=None):
        """Constructor."""

        # Initial position
        self.pos = pos

        # Keep track of which object is being actively moved; since the others
        # will follow programmatically, but will still fire their itemChanged
        # events
        self._selectedItem = None

        # Keep track of last position during a move
        self._lastPosition = None

        # Store the scene
        self._scene = scene

        # Defaults
        self._diameter = 6.0
        self._line_length = line_length

        # Relevant coordinates
        self._x0 = pos.x()
        self._y0 = pos.y()
        self._x = pos.x()
        self._y = pos.y() + self._line_length

        # Add the Line
        self._line = Line(self._x0, self._y0, self._x, self._y, "Line", None, self)

        # Add Vertex A
        self._vertexA = Vertex(self._x0, self._y0, 7, "A", self)

        # Add Vertex B
        self._vertexB = Vertex(self._x, self._y, 7, "B", self)

        # Do we have a scene already?
        if self._scene is not None:
            self.addToScene(self._scene)

    def addToScene(self, scene):
        """Add Line and Vertex objects to the scene."""
        if scene is not None:
            self._scene = scene
            self._scene.addItem(self._line)
            self._scene.addItem(self._vertexA)
            self._scene.addItem(self._vertexB)

    def centerOfMass(self):
        if self._line is None:
            return None
        p1 = self._line.line().p1()
        p2 = self._line.line().p2()
        c = QPointF(0.5 * (p1.x() + p2.x()), 0.5 * (p1.y() + p2.y()))
        return self._line.mapToScene(c)

    def get_line_length(self):
        if self._line is None:
            return None
        p1 = self._line.line().p1()
        p2 = self._line.line().p2()
        d = ((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)**(1/2)
        return d

    def itemMovedTo(self, item, newPos):
        """Called when the passed Item has been moved."""
        if self._selectedItem is not item:
            return

        # Calculate delta
        delta = newPos - self._lastPosition

        if item is self._vertexA:
            # Only update the first point of the line
            line = self._line.line()
            self._line.setLine(
                line.p1().x() + delta.x(),
                line.p1().y() + delta.y(),
                line.p2().x(),
                line.p2().y())
        elif item is self._vertexB:
            # Only update the second point of the line
            line = self._line.line()
            self._line.setLine(
                line.p1().x(),
                line.p1().y(),
                line.p2().x() + delta.x(),
                line.p2().y() + delta.y())
        elif item is self._line:
            # Move both vertices
            self._vertexA.setPos(self._vertexA.scenePos() + delta)
            self._vertexB.setPos(self._vertexB.scenePos() + delta)
        else:
            pass

        # Update the last position
        self._lastPosition = newPos

    def setSelectedItemAndOrigin(self, item, originScenePos):
        self._selectedItem = item

        # Store positions at the beginning of a move
        if item is None:
            self._lastPosition = None
        else:
            self._lastPosition = originScenePos

    def emit_length(self):
        length = self.get_line_length()

        if self._scene is not None:
            self._scene.signal_line_length.emit(length)
