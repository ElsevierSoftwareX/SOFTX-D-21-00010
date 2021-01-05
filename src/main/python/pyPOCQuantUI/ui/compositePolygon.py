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
        self._polygon_item.add_vertex(pos)

    def addLine(self, relative_bar_positions):
        """
        Add a Line to the underlying Polygon.
        """
        self._polygon_item.add_line(relative_bar_positions)

    def getCenterOfMass(self):
        return self._polygon_item.updateCenterOfMass()
