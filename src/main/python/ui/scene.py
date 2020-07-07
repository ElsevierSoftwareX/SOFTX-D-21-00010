from PyQt5.QtGui import QPixmap, QTransform
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
import pyqtgraph as pg
import imageio
from .polygon import Polygon
from .polygonVertex import PolygonVertex
from .circle import Circle
import numpy as np


class Scene(QGraphicsScene):
    """
    The main Scene.
    """

    def __init__(self, image, x=0, y=0, width=500, height=500, parent=None):
        super().__init__(x, y, width, height, parent)
        self.setSceneRect(0, 0, width, height)

        self.image = image
        self.rotate = 0
        self.mirror_v = 1
        self.mirror_h = 1

    def display_image(self, image_path=None):
        """
        Stores and displays an image
        :param image_path: full path to the image to display
        :return: void
        """

        # Open the image
        if image_path is not None:
            img = imageio.imread(image_path)
            # gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
            # gray = gray(img)
            # print(gray)

            self.image = pg.ImageItem(img)

        if self.image is None:
            return

        # Show the image (and store it)
        print(self.image.shape)
        print(self.image)
        self.image.render()
        my_transform = QTransform()
        my_transform.rotate(self.rotate)
        img = self.image.qimage.transformed(my_transform)
        pixMap = QPixmap.fromImage(img)
        pixmap_reflect = pixMap.transformed(QTransform().scale(self.mirror_v, self.mirror_h))

        # If needed, remove last QPixMap from the scene
        for item in self.items():
            if type(item) is QGraphicsPixmapItem:
                self.removeItem(item)
                del item

        self.addPixmap(pixmap_reflect)

        # Reset the scene size
        self.setSceneRect(0, 0, self.image.width(), self.image.height())

    # def removeCompositeLine(self):
    #     """
    #     Remove CompositeLine if it exists from the scene, but does not
    #     delete the object.
    #     """
    #     for item in self.items():
    #         if type(item) is ui.Vertex or type(item) is ui.Line:
    #             self.removeItem(item)

    def removeCompositePolygon(self):
        """
        Remove CompositePolygon if it exists from the scene, but does not
        delete the object.
        """
        for item in self.items():
            if type(item) is Polygon or \
                type(item) is PolygonVertex or \
                    type(item) is Circle:
                self.removeItem(item)
