from PyQt5 import uic
from PyQt5.QtCore import Qt, QDir, QPointF, QSize, QMetaObject, Q_ARG, pyqtSlot
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import  QMainWindow, QFileDialog, QFileSystemModel, QAction, QPlainTextEdit
from pyqtgraph.parametertree import Parameter, ParameterTree
from datetime import date
from pathlib import Path
import pyqtgraph as pg
import webbrowser
import imageio
import shutil
import os
from ui.config import params, key_map
from ui.view import View
from ui.scene import Scene
from ui.compositePolygon import CompositePolygon
from ui.bookkeeper import BookKeeper


class MainWindow(QMainWindow):
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        uic.loadUi(ui, self)

        self.setWindowTitle('pyPOCQuant:: Point of Care Test Quantification tool')

        # Add toolbar
        tb = self.addToolBar("File")
        tb.setMovable(False)
        self.strip_action = QAction("Draw POCT outline", self)
        self.strip_action.triggered.connect(self.on_draw_strip)
        tb.addAction(self.strip_action)
        self.sensor_action = QAction("Draw sensor outline", self)
        tb.addAction(self.sensor_action)
        self.sensor_action.triggered.connect(self.on_draw_sensor)
        self.mirror_v_action = QAction("Mirror image vertically", self)
        tb.addAction(self.mirror_v_action)
        self.mirror_v_action.triggered.connect(self.on_mirror_v)
        self.mirror_h_action = QAction("Mirror image horizontally", self)
        tb.addAction(self.mirror_h_action)
        self.mirror_h_action.triggered.connect(self.on_mirror_h)
        self.rotate_cw_action = QAction("Rotate clockwise", self)
        tb.addAction(self.rotate_cw_action)
        self.rotate_cw_action.triggered.connect(self.on_rotate_cw)
        self.rotate_ccw_action = QAction("Rotate counter clockwise", self)
        tb.addAction(self.rotate_ccw_action)
        self.rotate_ccw_action.triggered.connect(self.on_rotate_ccw)
        self.zoom_in_action = QAction("Zoom in", self)
        tb.addAction(self.zoom_in_action)
        self.zoom_in_action.triggered.connect(self.on_zoom_in)
        self.zoom_out_action = QAction("Zoom out", self)
        tb.addAction(self.zoom_out_action)
        self.zoom_out_action.triggered.connect(self.on_zoom_out)

        # Instantiate a BookKeeper
        self.bookKeeper = BookKeeper()

        self.image = None
        self.image_filename = None
        self.input_dir = None
        self.output_dir = None
        self.test_dir = None
        self.is_draw_strip = False
        self.is_draw_sensor = False
        self.config_file_name = None
        self.run_number = 1

        img = imageio.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../resources/img/pyPOCQuantSplash-01.png"))
        print(img.shape)
        self.image = pg.ImageItem(img)
        self.scene = Scene(self.image, 0.0, 0.0, 500.0, 500.0)
        self.view = View(self.scene)
        self.gridLayout_3.addWidget(self.view, 0, 1, 0, 4)
        self.scene.display_image()

        # Setup parameter tree
        self.p = Parameter.create(name='params', type='group', children=params)
        self.t = ParameterTree()
        self.t.setParameters(self.p, showTop=False)

        # @todo fix layout
        # self.gridlayout.removeWidget(self.entries[row])
        # self.entries[row].deleteLater()
        self.gridLayout_3.replaceWidget(self.graphicsView, self.t)
        self.graphicsView.deleteLater()
        self.t.setMinimumSize(QSize(350, 300))
        self.label_2.setMaximumSize(QSize(350, 20))
        self.label_3.setMaximumSize(QSize(350, 20))
        self.listView.setMinimumSize(QSize(350, 200))
        # self.itemAt(4).insertWidget(0, self.t)

        # self.gridLayout_3.addWidget(self.t, 7, 0)

        # Setup buttons
        self.input_btn.clicked.connect(self.on_select_input)
        self.output_btn.clicked.connect(self.on_select_output)
        self.test_btn.clicked.connect(self.on_test_pipeline)
        self.run_btn.clicked.connect(self.on_run_pipeline)

        # Setup log
        self.log = LogTextEdit()
        self.log.setReadOnly(True)
        self.log.appendPlainText('Welcome to pyPOCQuant')
        self.gridLayout_2.addWidget(self.log)

    def on_draw_strip(self):
        self.is_draw_strip = True
        self.is_draw_sensor = False

    def on_draw_sensor(self):
        self.is_draw_strip = False
        self.is_draw_sensor = True

    def on_mirror_v(self):
        self.scene.mirror_v = self.scene.mirror_v * -1
        self.scene.display_image()

    def on_mirror_h(self):
        self.scene.mirror_h = self.scene.mirror_h * -1
        self.scene.display_image()

    def on_zoom_in(self):
        print('implement me')

    def on_zoom_out(self):
        print('implement me')

    def on_rotate_cw(self):
        self.scene.rotate = self.scene.rotate + 90
        self.scene.display_image()

    def on_rotate_ccw(self):
        self.scene.rotate = self.scene.rotate - 90
        self.scene.display_image()

    def on_test_pipeline(self):

        print('Test run')
        # 1. Create a temp directory
        # Make sure the results folder exists
        self.test_dir = Path(self.input_dir / "test")
        self.test_dir.mkdir(exist_ok=True)
        # 2. Copy displayed image to temp folder
        shutil.copy(str(self.input_dir / self.image_filename), str(self.test_dir))
        # 3. Create config file form param tree
        settings = self.get_parameters()
        # Save parameters into input folder with timestamp
        self.save_settings(settings, str(self.test_dir / self.get_filename()))
        self.run_number = +1

        # 4. Run pipeline on this one image only with QC == True
        settings['qc'] = True
        # @todo either to console or to a log window in the ui
        self.log.appendPlainText(f"")
        self.log.appendPlainText(f"Starting analysis with parameters:")
        self.log.appendPlainText(f"               Settings file version: {settings['file_version']}")
        self.log.appendPlainText(f"                               Input: {self.input_dir}")
        self.log.appendPlainText(f"                              Output: {self.output_dir}")
        self.log.appendPlainText(f"                 Max number of cores: {settings['max_workers']}")
        self.log.appendPlainText(f"        RAW auto stretch intensities: {settings['raw_auto_stretch']}")
        self.log.appendPlainText(f"        RAW apply auto white balance: {settings['raw_auto_wb']}")
        self.log.appendPlainText(f"                          Strip size: {settings['strip_size']}")
        self.log.appendPlainText(f"                           Min score: {settings['min_sensor_score']:.2f}")
        self.log.appendPlainText(f"                      QR code border: {settings['qr_code_border']}")
        self.log.appendPlainText(f"               Perform sensor search: {settings['perform_sensor_search']}")
        self.log.appendPlainText(f"                         Sensor size: {settings['sensor_size']}")
        self.log.appendPlainText(f"                       Sensor center: {settings['sensor_center']}")
        self.log.appendPlainText(f"                  Sensor search area: {settings['sensor_search_area']}")
        self.log.appendPlainText(f"             Sensor threshold factor: {settings['sensor_thresh_factor']}")
        self.log.appendPlainText(f"                       Sensor border: {settings['sensor_border']}")
        self.log.appendPlainText(f"    Expected peak relative positions: {settings['peak_expected_relative_location']}")
        self.log.appendPlainText(f"          Subtract signal background: {settings['subtract_background']}")
        self.log.appendPlainText(f"                      Verbose output: {settings['verbose']}")
        self.log.appendPlainText(f"      Create quality-control figures: {settings['qc']}")
        self.log.appendPlainText(f"")

        # Run the pipeline
        # run_TPH(
        #     self.input_dir,
        #     self.output_dir,
        #     nef_auto_stretch=settings['nef_auto_stretch'],
        #     nef_auto_wb=settings['nef_auto_wb'],
        #     strip_size=settings['strip_size'],
        #     min_strip_corr_coeff=settings['min_strip_corr_coeff'],
        #     min_sensor_score=settings['min_sensor_score'],
        #     lower_bound_range=settings['lower_bound_range'],
        #     upper_bound_range=settings['upper_bound_range'],
        #     qr_code_border=settings['qr_code_border'],
        #     qr_code_spacer=settings['qr_code_spacer'],
        #     barcode_border=settings['barcode_border'],
        #     skip_strip_registration=settings['skip_strip_registration'],
        #     perform_sensor_search=settings['perform_sensor_search'],
        #     sensor_size=settings['sensor_size'],
        #     sensor_center=settings['sensor_center'],
        #     sensor_search_area=settings['sensor_search_area'],
        #     force_sensor_size=settings['force_sensor_size'],
        #     sensor_thresh_factor=settings['sensor_thresh_factor'],
        #     sensor_border_x=settings['sensor_border_x'],
        #     sensor_border_y=settings['sensor_border_y'],
        #     peak_expected_relative_location=settings['peak_expected_relative_location'],
        #     subtract_background=settings['subtract_background'],
        #     verbose=settings['verbose'],
        #     qc=settings['qc'],
        #     max_workers=settings['max_workers']
        # )

        # 5. Display control images by opening the test folder
        webbrowser.open(str(self.test_dir))

    def on_run_pipeline(self):

        # Get parameter values and save them to a config file
        settings = self.get_parameters()
        # Save into input folder with timestamp
        self.save_settings(settings, str(self.input_dir / self.get_filename()))
        print(settings)

        # Inform
        # @todo either to console or to a log window in the ui
        self.log.appendPlainText(f"")
        self.log.appendPlainText(f"Starting analysis with parameters:")
        self.log.appendPlainText(f"               Settings file version: {settings['file_version']}")
        self.log.appendPlainText(f"                               Input: {self.input_dir}")
        self.log.appendPlainText(f"                              Output: {self.output_dir}")
        self.log.appendPlainText(f"                 Max number of cores: {settings['max_workers']}")
        self.log.appendPlainText(f"        RAW auto stretch intensities: {settings['raw_auto_stretch']}")
        self.log.appendPlainText(f"        RAW apply auto white balance: {settings['raw_auto_wb']}")
        self.log.appendPlainText(f"                          Strip size: {settings['strip_size']}")
        self.log.appendPlainText(f"                           Min score: {settings['min_sensor_score']:.2f}")
        self.log.appendPlainText(f"                      QR code border: {settings['qr_code_border']}")
        self.log.appendPlainText(f"               Perform sensor search: {settings['perform_sensor_search']}")
        self.log.appendPlainText(f"                         Sensor size: {settings['sensor_size']}")
        self.log.appendPlainText(f"                       Sensor center: {settings['sensor_center']}")
        self.log.appendPlainText(f"                  Sensor search area: {settings['sensor_search_area']}")
        self.log.appendPlainText(f"             Sensor threshold factor: {settings['sensor_thresh_factor']}")
        self.log.appendPlainText(f"                       Sensor border: {settings['sensor_border']}")
        self.log.appendPlainText(f"    Expected peak relative positions: {settings['peak_expected_relative_location']}")
        self.log.appendPlainText(f"          Subtract signal background: {settings['subtract_background']}")
        self.log.appendPlainText(f"                      Verbose output: {settings['verbose']}")
        self.log.appendPlainText(f"      Create quality-control figures: {settings['qc']}")
        self.log.appendPlainText(f"")

        # @todo implement me
        # @todo run pipeline (either from python directly or as system command)
        # Run the pipeline
        # run_TPH(
        #     self.input_dir,
        #     self.output_dir,
        #     nef_auto_stretch=settings['nef_auto_stretch'],
        #     nef_auto_wb=settings['nef_auto_wb'],
        #     strip_size=settings['strip_size'],
        #     min_strip_corr_coeff=settings['min_strip_corr_coeff'],
        #     min_sensor_score=settings['min_sensor_score'],
        #     lower_bound_range=settings['lower_bound_range'],
        #     upper_bound_range=settings['upper_bound_range'],
        #     qr_code_border=settings['qr_code_border'],
        #     qr_code_spacer=settings['qr_code_spacer'],
        #     barcode_border=settings['barcode_border'],
        #     skip_strip_registration=settings['skip_strip_registration'],
        #     perform_sensor_search=settings['perform_sensor_search'],
        #     sensor_size=settings['sensor_size'],
        #     sensor_center=settings['sensor_center'],
        #     sensor_search_area=settings['sensor_search_area'],
        #     force_sensor_size=settings['force_sensor_size'],
        #     sensor_thresh_factor=settings['sensor_thresh_factor'],
        #     sensor_border_x=settings['sensor_border_x'],
        #     sensor_border_y=settings['sensor_border_y'],
        #     peak_expected_relative_location=settings['peak_expected_relative_location'],
        #     subtract_background=settings['subtract_background'],
        #     verbose=settings['verbose'],
        #     qc=settings['qc'],
        #     max_workers=settings['max_workers']
        # )

        # Display control images by opening the output folder
        webbrowser.open(str(self.output_dir))

    def on_select_input(self):
        self.input_dir = Path(QFileDialog.getExistingDirectory(None, "Select Directory"))
        self.input_edit.setText(str(self.input_dir))
        self.output_edit.setText(str(Path(self.input_dir / 'pipeline')))
        self.output_dir = Path(self.input_dir / 'pipeline')
        self.fileModel = QFileSystemModel(self)
        self.fileModel.setRootPath(str(self.input_dir))
        self.fileModel.setFilter(QDir.NoDotAndDotDot | QDir.Files)
        self.listView.setModel(self.fileModel)
        self.listView.setRootIndex(self.fileModel.index(str(self.input_dir)))
        self.listView.selectionModel().selectionChanged.connect(self.on_file_selection_changed)

    def on_select_output(self):
        self.output_dir = Path(QFileDialog.getExistingDirectory(None, "Select Directory"))
        self.output_edit.setText(str(self.output_dir))

    def on_file_selection_changed(self, selected):
        for ix in selected.indexes():
            self.log.appendPlainText(f"Selected image: {str(Path(self.input_dir / ix.data()))}")
            self.scene.display_image(image_path=Path(self.input_dir / ix.data()))
            self.image_filename = ix.data()

    def get_filename(self):
        today = date.today()
        datestr = today.strftime("%Y%m%d")
        self.config_file_name = '{}_config_run_{}.conf'.format(datestr, self.run_number)
        return self.config_file_name

    def get_parameters(self):
        dd = {}
        vals = self.p.getValues()
        for key, value in vals.items():
            if value[0] is None:
                for keyy, valuee in value[1].items():
                    if valuee[0] is None:
                        for keyyy, valueee in valuee[1].items():
                            # print(keyy, keyyy, '->', valueee[0])
                            if keyy.lower().replace(' ', '_') in dd:
                                dd[keyy.lower().replace(' ', '_')] = dd[keyy.lower().replace(' ', '_')] + (valueee[0],)
                            else:
                                dd[keyy.lower().replace(' ', '_')] = (valueee[0],)
                    else:
                        # print(keyy, '->', valuee[0])
                        dd[keyy.lower().replace(' ', '_')] = valuee[0]
            parameters = self.change_parameter_keys(dd, key_map)
            return parameters

    @staticmethod
    def change_parameter_keys(parameters, key_map):
        parameter_out = dict((key_map[key], value) for (key, value) in parameters.items())
        return parameter_out

    @staticmethod
    def save_settings(settings_dictionary, filename):
        """Save settings from a dictionary to file."""
        with open(filename, "w+") as f:
            for key in settings_dictionary:
                f.write(f"{key}={settings_dictionary[key]}\n")

    def set_sensor_and_strip_parameter(self):

        currentStripPolygon = self.bookKeeper.getCurrentStripPolygon()
        currentSensorPolygon = self.bookKeeper.getCurrentSensorPolygon()

        if currentStripPolygon is not None and currentSensorPolygon is not None:
            c_o_m_strip = currentStripPolygon.getCenterOfMass()
            rect_strip = currentStripPolygon._polygon_item.sceneBoundingRect()
            c_o_m_sensor = currentSensorPolygon.getCenterOfMass()
            rect_sensor = currentSensorPolygon._polygon_item.sceneBoundingRect()
            print('calc')
            print(rect_sensor.x() - rect_strip.x(), rect_sensor.y() - rect_strip.y())
            print(c_o_m_strip, c_o_m_sensor)
            print(rect_strip, rect_sensor)
            print('----')
            strip_size = (rect_strip.width(), rect_strip.height())
            sensor_size = (rect_sensor.width(), rect_sensor.height())
            sensor_center = (rect_sensor.x() - rect_strip.x() + rect_sensor.width() / 2, rect_sensor.y() -
                             rect_strip.y() + rect_sensor.height())
            sensor_search_area = (rect_sensor.width() + 10, rect_sensor.height() + 10)
            self.log.appendPlainText(f'strip_size {strip_size}')
            self.log.appendPlainText(f'sensor_size {sensor_size}')
            self.log.appendPlainText(f'sensor_center {sensor_center}')
            self.log.appendPlainText(f'sensor_search_area {sensor_search_area}')
            # Update the parameters in the parameterTree
            self.p.param('Basic parameters').param('POCT size').param('width').setValue(strip_size[0])
            self.p.param('Basic parameters').param('POCT size').param('height').setValue(strip_size[1])
            self.p.param('Basic parameters').param('Sensor size').param('width').setValue(sensor_size[0])
            self.p.param('Basic parameters').param('Sensor size').param('height').setValue(sensor_size[1])
            self.p.param('Basic parameters').param('Sensor center').param('x').setValue(sensor_center[0])
            self.p.param('Basic parameters').param('Sensor center').param('y').setValue(sensor_center[1])
            self.p.param('Basic parameters').param('Sensor search area').param('x').setValue(sensor_search_area[0])
            self.p.param('Basic parameters').param('Sensor search area').param('y').setValue(sensor_search_area[1])
        else:
            self.log.appendPlainText('Please draw POC test outline and sensor outline first')

    def mousePressEvent(self, event):
        # Mouse press

        # @todo fix this annoying coordintnate mapping of view/scene is in a gridLayout
        print(event.globalPos())
        print(event.pos())
        pos = self.view.mapToScene(event.pos())
        print(pos)
        viewPos = self.view.rect()
        newPos = QPointF(pos.x()-viewPos.width(), pos.y())
        print('new pos', newPos)
        print('delta', self.view.zoom)
        print(self.view.rect())

        if self.is_draw_sensor is True:
            currentSensorPolygon = self.bookKeeper.getCurrentSensorPolygon()
            if currentSensorPolygon is None:
                # Create a CompositePolygon
                currentSensorPolygon = CompositePolygon()

                # Add the CompositeLine to the Scene. Note that the CompositeLine is
                # not a QGraphicsItem itself and cannot be added to the Scene directly.
                currentSensorPolygon.addToScene(self.scene)

                # Store the polygon
                self.bookKeeper.addSensorPolygon(currentSensorPolygon)

            # Add the vertices
            if len(currentSensorPolygon._polygon_item.polygon_vertices) < 4:
                currentSensorPolygon._polygon_item.add_vertex(newPos)
            self.set_sensor_and_strip_parameter()

        elif self.is_draw_strip is True:
            currentStripPolygon = self.bookKeeper.getCurrentStripPolygon()
            if currentStripPolygon is None:
                # Create a CompositePolygon
                currentStripPolygon = CompositePolygon()

                # Add the CompositeLine to the Scene. Note that the CompositeLine is
                # not a QGraphicsItem itself and cannot be added to the Scene directly.
                currentStripPolygon.addToScene(self.scene)

                # Store the polygon
                self.bookKeeper.addStripPolygon(currentStripPolygon)

            # Add the vertices
            if len(currentStripPolygon._polygon_item.polygon_vertices) < 4:
                currentStripPolygon._polygon_item.add_vertex(newPos)
            self.set_sensor_and_strip_parameter()

        else:
            self.set_sensor_and_strip_parameter()
            pass
            super().mousePressEvent(event)


class LogTextEdit(QPlainTextEdit):
    """
    Adopted from
    https://stackoverflow.com/questions/53381975/display-terminal-output-with-tqdm-in-qplaintextedit
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.flag = False

    def write(self, message):
        if not hasattr(self, "flag"):
            self.flag = False
        message = message.replace('\r', '').rstrip()
        if message:
            method = "replace_last_line" if self.flag else "appendPlainText"
            QMetaObject.invokeMethod(self, method, Qt.QueuedConnection, Q_ARG(str, message))
            self.flag = True
        else:
            self.flag = False

    @pyqtSlot(str)
    def replace_last_line(self, text):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.select(QTextCursor.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.insertBlock()
        self.setTextCursor(cursor)
        self.insertPlainText(text)
