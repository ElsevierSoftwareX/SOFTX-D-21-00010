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

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPen, QCursor
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsEllipseItem


class PolygonVertex(QGraphicsEllipseItem):
    """A vertex."""

    def __init__(self, x, y, radius, name, parent_item, composite):
        """Constructor."""

        self._radius = radius
        self._diameter = 2 * radius
        self._name = name
        self._composite = composite
        self._parent_item = parent_item

        # Call the parent constructor
        super(PolygonVertex, self).__init__(0-radius, 0-radius, self._diameter, self._diameter, parent=None)

        # Now place it at the right position in the scene
        self.setPos(x - radius, y - radius)

        # self.setBrush(QColor(148, 85, 141))
        self.setPen(QPen(QColor(148, 85, 141), 2))
        self.setCursor(QCursor(Qt.PointingHandCursor))

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

    def name(self):
        return self._name

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.isEnabled():
            if self._parent_item:
                self._parent_item.move_vertex(self._name, value)
        return super(PolygonVertex, self).itemChange(change, value)

    def mouseReleaseEvent(self, event):
        if self._parent_item:
            if hasattr(self._parent_item, 'emit_sensor_attributes') and hasattr(self._parent_item, '_has_all_vertices'):
                if self._parent_item._has_all_vertices:
                    self._parent_item.emit_sensor_attributes()
        super(PolygonVertex, self).mouseReleaseEvent(event)
