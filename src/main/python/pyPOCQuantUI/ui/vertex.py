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

from PyQt5.QtGui import QColor, QPen, QCursor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsItem


class Vertex(QGraphicsEllipseItem):
    """A vertex."""

    def __init__(self, x, y, radius, name, composite):
        """Constructor."""

        self._radius = radius
        self._diameter = 2 * radius
        self._name = name
        self._composite = composite

        # Call the parent constructor
        super(Vertex, self).__init__(0, 0, self._diameter, self._diameter, parent=None)

        # Now place it at the right position in the scene
        self.setPos(x - radius, y - radius)

        self.setPen(QPen(QColor(148, 85, 141), 2))
        self.setCursor(QCursor(Qt.PointingHandCursor))

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

    def name(self):
        return self._name

    def mousePressEvent(self, event):
        self._composite.setSelectedItemAndOrigin(self, event.scenePos())
        super(Vertex, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self._composite.itemMovedTo(self, event.scenePos())
        super(Vertex, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._composite.setSelectedItemAndOrigin(None, None)
        self._composite.emit_length()
        super(Vertex, self).mouseReleaseEvent(event)
