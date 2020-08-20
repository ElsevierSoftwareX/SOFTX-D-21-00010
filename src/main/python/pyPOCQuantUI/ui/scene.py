from PyQt5.QtGui import QPixmap, QTransform
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QApplication
from PyQt5.QtCore import pyqtSignal, Qt
import pyqtgraph as pg
import imageio

from pypocquant.lib.io import load_and_process_image
from .polygon import Polygon
from .polygonVertex import PolygonVertex
from .circle import Circle


class Scene(QGraphicsScene):
    """
    The main Scene.
    """

    signal_add_object_at_position = pyqtSignal(float, float, name='signal_add_object_at_position')
    signal_scene_nr = pyqtSignal(int, name='signal_scene_nr')
    signal_rel_bar_pos = pyqtSignal(list, name='signal_rel_bar_pos')

    def __init__(self, image, x=0, y=0, width=500, height=500, nr=None, parent=None):
        super().__init__(x, y, width, height, parent)
        self.setSceneRect(0, 0, width, height)

        self.nr = nr
        self.image = image
        self.pixmap = None
        self.rotate = 90
        self.mirror_v = -1
        self.mirror_h = 1

    def display_image(self, image_path=None, image=None):
        """
        Stores and displays an image
        :param image_path: full path to the image to display
        :param image: image matrix to display
        :return: void
        """

        # Open the image
        if image_path is not None:
            img = load_and_process_image(image_path, to_rgb=True)
            # gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
            # gray = gray(img)
            # print(gray)

            self.image = pg.ImageItem(img)

        if image is not None:
            self.image = pg.ImageItem(image)

        if self.image is None:
            return

        try:
            # Show the image (and store it)
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
            self.setSceneRect(0, 0, pixmap_reflect.width(), pixmap_reflect.height())

            self.pixmap = pixmap_reflect

            self.addCompositePolygon()
        except Exception as e:
            print(e)

    def addCompositePolygon(self):
        """
        Add CompositePolygon if it exists and make sure that it is always on top.
        """
        for item in self.items():
            if type(item) is QGraphicsPixmapItem:
                item.setZValue(0)
            else:
                item.setZValue(10)
            self.addItem(item)

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

    def mousePressEvent(self, event):
        """
        Process a mouse press event on the scene.
        :param event: A mouse press event.
        :return:
        """
        if event.buttons() == Qt.LeftButton:
            x = event.scenePos().x()
            y = event.scenePos().y()
            self.signal_add_object_at_position.emit(x, y)
            self.signal_scene_nr.emit(self.nr)
        else:
            pass

        super().mousePressEvent(event)
