from .bookkeeper import BookKeeper
from .circle import Circle
from .compositePolygon import CompositePolygon
from .config import params
from .polygon import Polygon
from .polygonVertex import PolygonVertex
from .scene import Scene
from .view import View
import platform


__author__ = 'Andreas P. Cuny'

__version__ = '0.0.1'

__operating_system__ = '{} {}'.format(platform.system(), platform.architecture()[0])
