from PyQt5.QtGui import QColor, QPen, QCursor, QPolygonF, QBrush
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsPolygonItem
from PyQt5.QtCore import Qt, QPointF
from .polygonVertex import PolygonVertex


class Polygon(QGraphicsPolygonItem):
    def __init__(self, composite, parent=None):
        super(Polygon, self).__init__(parent)
        self._composite = composite
        self.polygon_vertices = []
        self.setZValue(10)
        self.setPen(QPen(QColor("green"), 2))
        self.setAcceptHoverEvents(True)

        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

        self.setCursor(QCursor(Qt.PointingHandCursor))

        self.polygon_vertex_items = []

        self._centerOfMass = None

    def number_of_points(self):
        return len(self.polygon_vertex_items)

    def add_vertex(self, p):
        self.polygon_vertices.append(p)
        self.setPolygon(QPolygonF(self.polygon_vertices))
        item = PolygonVertex(p.x(), p.y(), 7, len(self.polygon_vertices) - 1, self, self._composite)
        self.scene().addItem(item)
        self.polygon_vertex_items.append(item)
        item.setPos(p)
        self.updateCenterOfMass()

    def remove_last_vertex(self):
        if self.polygon_vertices:
            self.polygon_vertices.pop()
            self.setPolygon(QPolygonF(self.polygon_vertices))
            it = self.polygon_vertex_items.pop()
            self.scene().removeItem(it)
            del it
            self.updateCenterOfMass()

    def move_vertex(self, i, p):
        if 0 <= i < len(self.polygon_vertices):
            self.polygon_vertices[i] = self.mapFromScene(p)
            self.setPolygon(QPolygonF(self.polygon_vertices))

    def move_vertex_item(self, index, pos):
        if 0 <= index < len(self.polygon_vertex_items):
            item = self.polygon_vertex_items[index]
            item.setEnabled(False)
            item.setPos(pos)
            item.setEnabled(True)
            self.updateCenterOfMass()

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            for i, point in enumerate(self.polygon_vertices):
                self.move_vertex_item(i, self.mapToScene(point))
        return super(Polygon, self).itemChange(change, value)

    def hoverEnterEvent(self, event):
        self.setBrush(QColor(255, 0, 0, 100))
        super(Polygon, self).hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QBrush(Qt.NoBrush))
        super(Polygon, self).hoverLeaveEvent(event)

    def updateCenterOfMass(self):
        if len(self.polygon_vertex_items) == 0:
            return None

        sum_x = 0.0
        sum_y = 0.0
        for item in self.polygon_vertex_items:
            sum_x += item.x()
            sum_y += item.y()
        x = sum_x / len(self.polygon_vertex_items)
        y = sum_y / len(self.polygon_vertex_items)
        self._centerOfMass = QPointF(x, y)
        return self._centerOfMass
