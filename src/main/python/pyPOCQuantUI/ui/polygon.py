#  ********************************************************************************
#   Copyright © 2020-2021, ETH Zurich, D-BSSE, Andreas P. Cuny & Aaron Ponti
#   All rights reserved. This program and the accompanying materials
#   are made available under the terms of the GNU Public License v3.0
#   which accompanies this distribution, and is available at
#   http://www.gnu.org/licenses/gpl
#
#   Contributors:
#     * Andreas P. Cuny - initial API and implementation
#     * Aaron Ponti - initial API and implementation
#  *******************************************************************************

from PyQt5.QtGui import QColor, QPen, QCursor, QPolygonF, QBrush
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsPolygonItem
from PyQt5.QtCore import Qt, QPointF
from .polygonVertex import PolygonVertex
from .line import Line


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
        self.line_items = []
        self.relative_bar_positions = []
        # self._relative_bar_positions = [None] * 3
        self._has_all_vertices = False
        self._centerOfMass = None
        self.sensor_search_area_offset = (8, 8)
        self.attributes = []

    def number_of_points(self):
        return len(self.polygon_vertex_items)

    def add_line(self, relative_bar_positions):

        sensor = self.sceneBoundingRect()
        self.relative_bar_positions = relative_bar_positions

        for idx, rel_pos in enumerate(relative_bar_positions):

            # Relevant coordinates
            self._x0 = sensor.x() + (rel_pos * sensor.width())
            self._y0 = sensor.y()
            self._x = sensor.x() + (rel_pos * sensor.width())
            self._y = sensor.y() + sensor.height()

            # Add the Line
            item = Line(self._x0, self._y0, self._x, self._y, idx, self, self._composite)
            item.setZValue(10)
            self.scene().addItem(item)
            self.line_items.append(item)

    def add_vertex(self, p):
        self.polygon_vertices.append(p)
        self.setPolygon(QPolygonF(self.polygon_vertices))
        item = PolygonVertex(p.x(), p.y(), 7, len(self.polygon_vertices) - 1, self, self._composite)
        self.scene().addItem(item)
        self.polygon_vertex_items.append(item)
        item.setPos(p)
        self.updateCenterOfMass()
        if len(self.polygon_vertices) == 4:
            self._has_all_vertices = True

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
            self.move_line_item()

    def move_vertex_item(self, index, pos):
        if 0 <= index < len(self.polygon_vertex_items):
            item = self.polygon_vertex_items[index]
            item.setEnabled(False)
            item.setPos(pos)
            item.setEnabled(True)
            self.updateCenterOfMass()

    def move_line_item(self):
        for index, _ in enumerate(self.line_items):
            item = self.line_items[index]
            pos = self.sceneBoundingRect()

            item.setEnabled(False)
            item.setPos(QPointF(pos.x() + (self.relative_bar_positions[index] * pos.width()), pos.y()))
            l = item.line()
            l.setLength(pos.height())
            item.setLine(l)
            item.setEnabled(True)

    def emit_new_rel_pos(self, name, value):
        pos = self.sceneBoundingRect()

        rel_pos = round((value.x() - pos.x())/pos.width(), 2)
        if rel_pos < 0:
            rel_pos = 0
        elif rel_pos > 1:
            rel_pos = 1
        else:
            pass

        relative_bar_positions = list(self.relative_bar_positions)
        relative_bar_positions[name] = rel_pos
        if self.scene() is not None:
            self.scene().signal_rel_bar_pos.emit(relative_bar_positions)

    def update_polygon(self):
        """
        Upated the polygon attribues
        :return:
        """

        # Get the current state of the polygon
        rect_sensor = self.sceneBoundingRect()
        sensor_size = (rect_sensor.width(), rect_sensor.height())
        sensor_center = ((rect_sensor.x() - 0 + rect_sensor.width() / 2), (rect_sensor.y() -
                         0 + rect_sensor.height() / 2))

        # Get the coordinates of the vertices and move the most extreme ones (determining the bounding box to change
        # the polygon shape
        x = []
        y = []
        for point in self.polygon_vertices:
            x.append(point.x())
            y.append(point.y())
        dx = max(x) - min(x)
        dy = max(y) - min(y)
        ox = self.attributes[2] - dx
        oy = self.attributes[3] - dy
        x[x.index(max(x))] = x[x.index(max(x))] + ox / 2
        x[x.index(min(x))] = x[x.index(min(x))] - ox / 2
        y[y.index(max(y))] = y[y.index(max(y))] + oy / 2
        y[y.index(min(y))] = y[y.index(min(y))] - oy / 2

        # Translate the polygon vertices based on the polygon center
        for i, point in enumerate(self.polygon_vertices):
            point.setX(round(x[i] + (self.attributes[0]) - sensor_center[0]))
            point.setY(round(y[i] + (self.attributes[1]) - sensor_center[1]))
            self.polygon_vertices[i] = point

        # Translate the draggable elements such as the polygon, vertices and bar lines
        for i, point in enumerate(self.polygon_vertices):
            self.move_vertex(i, self.mapToScene(point))
            self.move_vertex_item(i, self.mapToScene(point))

        # Update the sensor_search_area in the attributes list based on the newly determined sensor_search_area_offset
        if not self.attributes:
            sensor_search_area_offset = self.sensor_search_area_offset
        else:
            sensor_search_area_offset = tuple(self.attributes[-2:])
        sensor_search_area = (rect_sensor.width() + sensor_search_area_offset[0], rect_sensor.height() +
                              sensor_search_area_offset[1])
        sensor_search_area = [round(el) for el in sensor_search_area]
        self.attributes[4] = sensor_search_area[0]
        self.attributes[5] = sensor_search_area[1]

    def emit_sensor_attributes(self):

        rect_sensor = self.sceneBoundingRect()
        sensor_size = (rect_sensor.width()-2, rect_sensor.height()-2)
        sensor_center = (rect_sensor.x() - 0 + rect_sensor.width() / 2, rect_sensor.y() -
                         0 + rect_sensor.height() / 2)
        if not self.attributes:
            sensor_search_area_offset = self.sensor_search_area_offset
        else:
            sensor_search_area_offset = tuple(self.attributes[-2:])
        sensor_search_area = (rect_sensor.width() + sensor_search_area_offset[0], rect_sensor.height() +
                              sensor_search_area_offset[1])

        sensor_attributes = sensor_center + sensor_size + sensor_search_area + sensor_search_area_offset
        sensor_attributes = [round(el) for el in sensor_attributes]

        if self.scene() is not None:
            self.scene().signal_sensor_attributes.emit(list(sensor_attributes))

        self.attributes = sensor_attributes

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            for i, point in enumerate(self.polygon_vertices):
                self.move_vertex_item(i, self.mapToScene(point))
            self.move_line_item()
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

    def mouseReleaseEvent(self, event):
        if self._has_all_vertices:
            self.emit_sensor_attributes()
        super(Polygon, self).mouseReleaseEvent(event)
