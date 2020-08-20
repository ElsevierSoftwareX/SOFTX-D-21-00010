from PyQt5.QtGui import QColor, QPen, QCursor
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsLineItem
from PyQt5.QtCore import Qt


class Line(QGraphicsLineItem):
    """A line."""

    def __init__(self, x0, y0, x, y, name, parent_item, composite):
        """Constructor."""

        self._name = name
        self._value = 0
        self._composite = composite
        self._parent_item = parent_item

        self.width = x - x0
        self.height = y - y0

        # Call parent constructor
        super(Line, self).__init__(0, 0, self.width, self.height, parent=None)

        # Now place it at the correct position in the scene
        self.setPos(x0, y0)

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
            self._value = value
        return super(Line, self).itemChange(change, value)

    def mouseReleaseEvent(self, event):
        if self._parent_item:
            self._parent_item.emit_new_rel_pos(self._name, self._value)
        super(Line, self).mouseReleaseEvent(event)