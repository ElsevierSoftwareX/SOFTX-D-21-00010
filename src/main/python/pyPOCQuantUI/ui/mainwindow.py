from PyQt5 import uic
from PyQt5.QtCore import Qt, QDir, QPointF, QSize, QMetaObject, Q_ARG, pyqtSlot, QRectF, QPoint, QThreadPool
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QFileSystemModel, QAction, QPlainTextEdit, QSizePolicy, QMessageBox
from pyqtgraph.parametertree import Parameter, ParameterTree
from datetime import date
from pathlib import Path
import pyqtgraph as pg
import webbrowser
import imageio
import shutil
import os
from ui.config import params, key_map, save_settings, load_settings
from ui.view import View
from ui.scene import Scene
from ui.compositePolygon import CompositePolygon
from ui.bookkeeper import BookKeeper
from ui.worker import Worker
from ui.log import LogTextEdit
from ui.help import About, QuickInstructions
from pypocquant.pipeline_FH import run_FH
import pypocquant as pq


class MainWindow(QMainWindow):
    def __init__(self, ui, splash, parent=None):
        super().__init__(parent)
        uic.loadUi(ui, self)

        self.setWindowTitle('pyPOCQuant:: Point of Care Test Quantification tool')

        # Add filemenu
        self.action_save_settings_file.triggered.connect(self.on_save_settings_file)
        self.action_save_settings_file.setShortcut("Ctrl+S")
        self.action_load_settings_file.triggered.connect(self.on_load_settings_file)
        self.action_load_settings_file.setShortcut("Ctrl+O")
        self.actionQuit.triggered.connect(self.close)
        self.about_window = About()
        self.actionAbout.setShortcut("Ctrl+A")
        self.actionAbout.triggered.connect(self.on_about)
        self.actionManual.triggered.connect(self.on_manual)
        self.qi = QuickInstructions()
        self.actionQuick_instructions.setStatusTip('Hints about how to use this program')
        self.actionQuick_instructions.triggered.connect(self.on_quick_instructions)

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
        self.display_on_startup = None
        self.image = None
        self.splash = splash
        self.image_filename = None
        self.input_dir = None
        self.output_dir = None
        self.test_dir = None
        self.is_draw_strip = False
        self.is_draw_sensor = False
        self.config_file_name = None
        self.run_number = 1

        img = imageio.imread(self.splash)
        print(img.shape)
        self.image = pg.ImageItem(img)
        self.scene = Scene(self.image, 0.0, 0.0, 500.0, 500.0)
        self.scene.signal_add_object_at_position.connect(
            self.handle_add_object_at_position)
        self.view = View(self.scene)
        self.gridLayout_3.replaceWidget(self.viewO, self.view)
        self.viewO.deleteLater()
        self.scene.display_image()
        self.view.fitInView(QRectF(0, 0, self.scene.pixmap.width(), self.scene.pixmap.width()), Qt.KeepAspectRatio)

        # Setup parameter tree
        self.p = Parameter.create(name='params', type='group', children=params)
        # self.t = ParameterTree()
        self.paramTree.setParameters(self.p, showTop=False)

        # Instantiate ThreadPool
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(1)

        # Setup button connections
        self.input_btn.clicked.connect(self.on_select_input)
        self.input_btn.setShortcut("Ctrl+I")
        self.output_btn.clicked.connect(self.on_select_output)
        self.test_btn.clicked.connect(self.on_test_pipeline)
        self.run_btn.clicked.connect(self.on_run_pipeline)

        # Setup log
        self.log = LogTextEdit()
        self.log.setReadOnly(True)
        self.log.appendPlainText('Welcome to pyPOCQuant')
        self.gridLayout_2.addWidget(self.log)

        # Open quick instructions
        try:
            if self.display_on_startup is None:
                self.display_on_startup = 2
                self.qi.show()

            elif self.display_on_startup == 2:
                self.qi.show()
        except Exception as e:
            self.log.appendPlainText('Could not load quick instruction window due to corrupt settings.ini file' + str(e))

    def on_quick_instructions(self):
        """
        Displays the quick instructions window.
        """
        self.qi.show()

    def on_about(self):
        """
        Displays the about window.
        """
        self.about_window.show()

    def on_manual(self):
        """
        Displays the instruction manual.
        """
        path = Path(Path(pq.__file__).parent)
        path = Path(path).joinpath('manual', 'UserInstructions.html')
        print(path)
        webbrowser.open(str(path))

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
        self.view.zoom_in()

    def on_zoom_out(self):
        self.view.zoom_out()

    def on_rotate_cw(self):
        self.scene.rotate = self.scene.rotate + 90
        self.scene.display_image()

    def on_rotate_ccw(self):
        self.scene.rotate = self.scene.rotate - 90
        self.scene.display_image()

    def on_test_pipeline(self):

        if self.input_dir is None or self.image_filename is None:
            if not self.input_dir:
                msg = "Please select an input folder first."
            elif not self.image_filename:
                msg = "Please select a test image first"
            reply = QMessageBox.question(self, 'Message', msg, QMessageBox.Ok)

            if reply == QMessageBox.Ok:
                return
            else:
                return

        # 1. Create a temp directory
        # Make sure the results folder exists
        self.test_dir = Path(self.input_dir / "test")
        self.test_dir.mkdir(exist_ok=True)
        # 2. Copy displayed image to temp folder
        shutil.copy(str(self.input_dir / self.image_filename), str(self.test_dir))
        # 3. Create config file form param tree
        settings = self.get_parameters()
        # Save parameters into input folder with timestamp
        save_settings(settings, str(self.test_dir / self.get_filename()))
        self.run_number = +1
        # 4. Run pipeline on this one image only with QC == True
        settings['qc'] = True
        # Run the pipeline
        self.run_worker(input_dir=self.test_dir, output_dir=self.test_dir, settings=settings)
        # 5. Display control images by opening the test folder
        webbrowser.open(str(self.test_dir))

    def on_run_pipeline(self):

        if self.input_dir is None or self.output_dir is None:
            if not self.input_dir:
                msg = "Please select an input folder first."
            elif not self.output_dir:
                msg = "Please select an output folder first"
            reply = QMessageBox.question(self, 'Message', msg, QMessageBox.Ok)

            if reply == QMessageBox.Ok:
                return
            else:
                return

        # Get parameter values and save them to a config file
        settings = self.get_parameters()
        # Save into input folder with timestamp
        save_settings(settings, str(self.input_dir / self.get_filename()))
        # Run full pipeline
        self.run_worker(input_dir=self.input_dir, output_dir=self.output_dir, settings=settings)
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
        self.log.appendPlainText(f"Selected input folder: {self.input_dir}")

    def on_select_output(self):
        self.output_dir = Path(QFileDialog.getExistingDirectory(None, "Select Directory"))
        self.output_edit.setText(str(self.output_dir))
        self.log.appendPlainText(f"Selected output folder: {self.output_dir}")

    def on_file_selection_changed(self, selected):
        for ix in selected.indexes():
            self.log.appendPlainText(f"Selected image: {str(Path(self.input_dir / ix.data()))}")
            self.scene.display_image(image_path=Path(self.input_dir / ix.data()))
            self.view.fitInView(QRectF(0, 0, self.scene.pixmap.width(), self.scene.pixmap.width()), Qt.KeepAspectRatio)
            self.image_filename = ix.data()

    def on_save_settings_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "All Files (*);;Text Files (*.txt)", options=options)
        if file_name:
            settings = self.get_parameters()
            # Save parameters into input folder with timestamp
            save_settings(settings, Path(file_name).stem + '.conf')
            self.log.appendPlainText(f"Saved config file under: {file_name}")

    def on_load_settings_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "All Files (*);;Text Files (*.txt);; "
                                                                       "Config Files (*.conf)")
        if file_name:
            settings = load_settings(file_name)
            if 'strip_text_to_search' in settings:
                if settings['strip_text_to_search'] == '':
                    settings['strip_text_to_search'] = '""'
            self.load_parameters(settings)
            self.log.appendPlainText(f"Loaded config : {file_name}")

    def load_parameters(self, settings):
        # Populate parameter tree
        for c in self.p.children():
            for gc in self.p.param(c.name()):
                if gc.hasChildren():
                    for idx, ggc in enumerate(self.p.param(c.name()).param(gc.name())):
                        key = gc.name().lower().replace(' ', '_')
                        if key in key_map:
                            new_key = key_map[key]
                            if new_key in settings:
                                ggc.setValue(settings[new_key][idx])
                else:
                    key = gc.name().lower().replace(' ', '_')
                    if key in key_map:
                        new_key = key_map[key]
                        if new_key in settings:
                            gc.setValue(settings[new_key])

    def run_pipeline(self, input_dir, output_dir, settings):
        # Inform the user
        self.log.appendPlainText(f"")
        self.log.appendPlainText(f"Starting analysis with parameters:")
        self.log.appendPlainText(f"               Settings file version: {settings['file_version']}")
        self.log.appendPlainText(f"                               Input: {self.test_dir}")
        self.log.appendPlainText(f"                              Output: {self.test_dir}")
        self.log.appendPlainText(f"                 Max number of cores: {settings['max_workers']}")
        self.log.appendPlainText(f"        RAW auto stretch intensities: {settings['raw_auto_stretch']}")
        self.log.appendPlainText(f"        RAW apply auto white balance: {settings['raw_auto_wb']}")
        self.log.appendPlainText(f"  Strip text to search (orientation): {settings['strip_text_to_search']}")
        self.log.appendPlainText(f"          Strip text is on the right: {settings['strip_text_on_right']}")
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
        run_FH(
            input_dir,
            output_dir,
            raw_auto_stretch=settings['raw_auto_stretch'],
            raw_auto_wb=settings['raw_auto_wb'],
            strip_text_to_search=settings['strip_text_to_search'],
            strip_text_on_right=settings['strip_text_on_right'],
            min_sensor_score=settings['min_sensor_score'],
            qr_code_border=settings['qr_code_border'],
            perform_sensor_search=settings['perform_sensor_search'],
            sensor_size=settings['sensor_size'],
            sensor_center=settings['sensor_center'],
            sensor_search_area=settings['sensor_search_area'],
            sensor_thresh_factor=settings['sensor_thresh_factor'],
            sensor_border=settings['sensor_border'],
            peak_expected_relative_location=settings['peak_expected_relative_location'],
            subtract_background=settings['subtract_background'],
            verbose=settings['verbose'],
            qc=settings['qc'],
            max_workers=settings['max_workers']
        )

    def run_worker(self, input_dir, output_dir, settings):
        worker = Worker(self.run_pipeline, input_dir, output_dir, settings)
        self.threadpool.start(worker)

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
                            if keyy.lower().replace(' ', '_') in dd:
                                dd[keyy.lower().replace(' ', '_')] = dd[keyy.lower().replace(' ', '_')] + (valueee[0],)
                            else:
                                dd[keyy.lower().replace(' ', '_')] = (valueee[0],)
                    else:
                        dd[keyy.lower().replace(' ', '_')] = valuee[0]
            parameters = self.change_parameter_keys(dd, key_map)
            if parameters['strip_text_to_search'] == '""':
                pass
            else:
                parameters['strip_text_to_search'] = '\"{}\"'.format(parameters['strip_text_to_search'])
            return parameters

    @staticmethod
    def change_parameter_keys(parameters, key_map):
        parameter_out = dict((key_map[key], value) for (key, value) in parameters.items())
        return parameter_out

    def set_sensor_and_strip_parameter(self):

        currentStripPolygon = self.bookKeeper.getCurrentStripPolygon()
        currentSensorPolygon = self.bookKeeper.getCurrentSensorPolygon()

        if currentStripPolygon is not None and currentSensorPolygon is not None:
            c_o_m_strip = currentStripPolygon.getCenterOfMass()
            rect_strip = currentStripPolygon._polygon_item.sceneBoundingRect()
            c_o_m_sensor = currentSensorPolygon.getCenterOfMass()
            rect_sensor = currentSensorPolygon._polygon_item.sceneBoundingRect()

            strip_size = (rect_strip.width(), rect_strip.height())
            sensor_size = (rect_sensor.width(), rect_sensor.height())
            sensor_center = (rect_sensor.x() - rect_strip.x() + rect_sensor.width() / 2, rect_sensor.y() -
                             rect_strip.y() + rect_sensor.height() / 2)
            sensor_search_area = (rect_sensor.width() + 10, rect_sensor.height() + 10)
            # Update the parameters in the parameterTree
            self.p.param('Basic parameters').param('POCT size').param('width').setValue(strip_size[1])
            self.p.param('Basic parameters').param('POCT size').param('height').setValue(strip_size[0])
            self.p.param('Basic parameters').param('Sensor size').param('width').setValue(sensor_size[1])
            self.p.param('Basic parameters').param('Sensor size').param('height').setValue(sensor_size[0])
            self.p.param('Basic parameters').param('Sensor center').param('x').setValue(sensor_center[1])
            self.p.param('Basic parameters').param('Sensor center').param('y').setValue(sensor_center[0])
            self.p.param('Basic parameters').param('Sensor search area').param('x').setValue(sensor_search_area[1])
            self.p.param('Basic parameters').param('Sensor search area').param('y').setValue(sensor_search_area[0])
        else:
            pass
            # self.log.appendPlainText('Please draw POC test outline and sensor outline first')

    @pyqtSlot(float, float, name="handle_add_object_at_position")
    def handle_add_object_at_position(self, x, y):

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
                currentSensorPolygon.addVertex(QPointF(x, y))
                self.log.appendPlainText('Drawing sensor corner')
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
                currentStripPolygon.addVertex(QPointF(x, y))
                self.log.appendPlainText('Drawing POCT corner')
            self.set_sensor_and_strip_parameter()

    def closeEvent(self, event):

        quit_msg = "Are you sure you want to exit the program?"
        reply = QMessageBox.question(self, 'Message',
                                           quit_msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
