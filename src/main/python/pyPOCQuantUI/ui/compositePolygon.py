from PyQt5.QtCore import QPointF

from .circle import Circle
from .polygon import Polygon


class CompositePolygon:
    """Composite polygon.

    This is not a QGraphicsItem and cannot be added to a QGraphicsScene directly.
    If a QGraphicsScene is passed to it in the constructor or as an argument to
    addToScene(), it will manage a series of QGraphicsItems (one 'Line' and two
    'Vertex' items) and their spatial relationships.
    """

    def __init__(self, scene=None):
        """Constructor."""

        # Keep track of which object is being actively moved; since the others
        # will follow programmatically, but will still fire their itemChanged
        # events
        self._selectedItem = None

        # Keep track of last position during a move
        self._lastPosition = None

        # Store the scene
        self._scene = scene

        self._polygon_item = Polygon(self)

        self._line_items = []

        # Do we have a scene already?
        if self._scene is not None:
            self.addToScene(self._scene)

    def addToScene(self, scene):
        """Add Line and Vertex objects to the scene."""
        if scene is not None:
            self._scene = scene
            self._scene.addItem(self._polygon_item)
            for vertex in self._polygon_item.polygon_vertex_items:
                self._scene.addItem(vertex)
            cm = self.getCenterOfMass()
            if cm is not None:
                self._scene.addItem(Circle(cm.x(), cm.y()))

    def addVertex(self, pos):
        """
        Add a Vertex to the underlying Polygon.
        """
        # print('print item', pos)
        # print(self._scene)
        # print('print item', self._scene.mapToScene(pos))
        self._polygon_item.add_vertex(pos)

    def addLine(self, relative_bar_positions):
        """
        Add a Line to the underlying Polygon.
        """
        self._polygon_item.add_line(relative_bar_positions)

    def getCenterOfMass(self):
        return self._polygon_item.updateCenterOfMass()

    # def addLines(self, relative_bar_positions):
    #     sensor = self._polygon_item.sceneBoundingRect()
    #
    #     for rel_pos in relative_bar_positions:
    #         line = CompositeLine(pos=QPointF(sensor.x() + rel_pos * sensor.width(), sensor.y()),
    #                              line_length=sensor.height(), scene=self._scene)
    #         self._line_items.append(line)


        # if len(self._polygon_item.polygon_vertices) > 3:
        #     vertices = []
        #     for i in range(len(self._polygon_item.polygon_vertices)):
        #         # print('vertex pos', self.mapToScene(QPointF(self.vertices[i].x(), self.vertices[i].y())))
        #         vertices.append([self._polygon_item.polygon_vertices[i].x(), self._polygon_item.polygon_vertices[i].y()])
        #
        #     delta = vertices[0]
        #     center_of_mass = [0, 0, 0]
        #     area = 0
        #     for i in range(len(vertices)):
        #         vertex1 = vertices[i]
        #         vertex2 = vertices[i - 1]
        #         f = (vertex1[0] - delta[0]) * (vertex2[1] - delta[1]) - (vertex2[0] - delta[0]) * (
        #                     vertex1[1] - delta[1])
        #         area += f
        #         center_of_mass[0] += (vertex1[0] + vertex2[0] - 2 * delta[0]) * f
        #         center_of_mass[1] += (vertex1[1] + vertex2[1] - 2 * delta[1]) * f
        #     center_of_mass[0] = center_of_mass[0] / (3 * area) + delta[0]
        #     center_of_mass[1] = center_of_mass[1] / (3 * area) + delta[1]
        #     center_of_mass[2] = abs(area / 2)
        #     return self._polygon_item.mapToScene(QPointF(center_of_mass[0], center_of_mass[1]))



    # def itemMovedTo(self, item, newPos):
    #     """Called when the passed Item has been moved."""
    #     if self._selectedItem is not item:
    #         return
    #
    #     # Calculate delta
    #     delta = newPos - self._lastPosition
    #
    #     if item is self._vertexA:
    #         # Only update the first point of the line
    #         line = self._line.line()
    #         self._line.setLine(
    #             line.p1().x() + delta.x(),
    #             line.p1().y() + delta.y(),
    #             line.p2().x(),
    #             line.p2().y())
    #     elif item is self._vertexB:
    #         # Only update the second point of the line
    #         line = self._line.line()
    #         self._line.setLine(
    #             line.p1().x(),
    #             line.p1().y(),
    #             line.p2().x() + delta.x(),
    #             line.p2().y() + delta.y())
    #     elif item is self._line:
    #         # Move both vertices
    #         self._vertexA.setPos(self._vertexA.scenePos() + delta)
    #         self._vertexB.setPos(self._vertexB.scenePos() + delta)
    #     else:
    #         pass
    #
    #     # Update the last position
    #     self._lastPosition = newPos

    # def setSelectedItemAndOrigin(self, item, originScenePos):
    #     self._selectedItem = item
    #
    #     # Store positions at the beginning of a move
    #     if item is None:
    #         self._lastPosition = None
    #     else:
    #         self._lastPosition = originScenePos
