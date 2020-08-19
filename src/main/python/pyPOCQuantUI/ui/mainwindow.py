from PyQt5 import uic
from PyQt5.QtCore import Qt, QDir, QPointF, QSize, QMetaObject, Q_ARG, pyqtSlot, QRectF, QPoint, QThreadPool, QObject, pyqtSignal
from PyQt5.QtGui import QTextCursor, QBrush, QColor
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QFileSystemModel, QAction, QPlainTextEdit, QSizePolicy, QMessageBox, QStyle, QApplication, QProgressBar
from pyqtgraph.parametertree import Parameter, ParameterTree
from datetime import date
from pathlib import Path
import pyqtgraph as pg
import webbrowser
import imageio
import shutil
import logging
import sys
import os
import numpy as np
from ui.config import params, key_map, save_settings, load_settings
from ui.view import View
from ui.scene import Scene
from ui.compositePolygon import CompositePolygon
from ui.bookkeeper import BookKeeper
from ui.worker import Worker
from ui.log import LogTextEdit
from ui.help import About, QuickInstructions
from ui.stream import Stream
from pypocquant.pipeline_FH import run_FH
from pypocquant.lib.tools import extract_strip
import pypocquant as pq


class MainWindow(QMainWindow):

    send_to_console_signal = pyqtSignal(str)
    """
        pyqtSignal used to send a text to the console.

    Args:
        message (`str`)         Text to be sent to the console
    """

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
        # self.strip_action = QAction("Draw POCT outline", self)
        # self.strip_action.triggered.connect(self.on_draw_strip)
        # tb.addAction(self.strip_action)
        self.sensor_action = QAction("Draw sensor outline", self)
        tb.addAction(self.sensor_action)
        self.sensor_action.triggered.connect(self.on_draw_sensor)
        self.delete_items_action = QAction("Delete sensor", self)
        tb.addAction(self.delete_items_action)
        self.delete_items_action.triggered.connect(self.on_delete_items_action)
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
        self.strip_img = None
        self.current_scene = None

        img = imageio.imread(self.splash)
        self.image = pg.ImageItem(img)
        self.scene = Scene(self.image, 0.0, 0.0, 500.0, 500.0, nr=int(1))
        self.scene.signal_add_object_at_position.connect(
            self.handle_add_object_at_position)
        self.scene.signal_scene_nr.connect(
            self.on_signal_scene_nr)
        self.view = View(self.scene)
        self.verticalLayout_Right_Column.replaceWidget(self.viewO, self.view)
        self.viewO.deleteLater()
        self.scene.display_image()
        self.view.fitInView(QRectF(0, 0, self.scene.pixmap.width(), self.scene.pixmap.width()), Qt.KeepAspectRatio)
        # Set 2nd scene and view
        self.scene_strip = Scene(pg.ImageItem(np.array([[255, 255], [255, 255]])), 0.0, 0.0, 1000.0, 450.0, nr=int(2))
        self.scene_strip.signal_add_object_at_position.connect(
            self.handle_add_object_at_position)
        self.scene_strip.signal_scene_nr.connect(
            self.on_signal_scene_nr)
        self.view_strip = View(self.scene_strip)
        self.verticalLayout_Right_Column.replaceWidget(self.viewO2, self.view_strip)
        self.viewO2.deleteLater()
        self.scene_strip.display_image()
        self.view_strip.fitInView(QRectF(0, 0, self.scene_strip.pixmap.width(), self.scene_strip.pixmap.width()),
                                  Qt.KeepAspectRatio)

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
        self.gridLayout_2.addWidget(self.log)
        sys.stderr = Stream(stream_signal=self.on_write_to_console)
        self.logger = self.get_logger_object(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.print_to_console('Welcome to pyPOCQuant')

        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True)

        # Open quick instructions
        try:
            if self.display_on_startup is None:
                self.display_on_startup = 2
                self.qi.show()

            elif self.display_on_startup == 2:
                self.qi.show()
        except Exception as e:
            self.print_to_console('Could not load quick instruction window due to corrupt settings.ini file' + str(e))

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
        webbrowser.open(str(path))

    def on_draw_strip(self):
        self.is_draw_strip = True
        self.is_draw_sensor = False

    def on_draw_sensor(self):
        self.is_draw_strip = False
        self.is_draw_sensor = True

    def on_mirror_v(self):
        if self.current_scene == 1:
            self.scene.mirror_v = self.scene.mirror_v * -1
            self.scene.display_image()
        else:
            self.scene_strip.mirror_v = self.scene_strip.mirror_v * -1
            self.scene_strip.display_image()

    def on_mirror_h(self):
        if self.current_scene == 1:
            self.scene.mirror_h = self.scene.mirror_h * -1
            self.scene.display_image()
        else:
            self.scene_strip.mirror_h = self.scene_strip.mirror_h * -1
            self.scene_strip.display_image()

    def on_zoom_in(self):
        if self.current_scene == 1:
            self.view.zoom_in()
        else:
            self.view_strip.zoom_in()

    def on_zoom_out(self):
        if self.current_scene == 1:
            self.view.zoom_out()
        else:
            self.view_strip.zoom_out()

    def on_rotate_cw(self):
        if self.current_scene == 1:
            self.scene.rotate = self.scene.rotate + 90
            self.scene.display_image()
        else:
            self.scene_strip.rotate = self.scene_strip.rotate + 90
            self.scene_strip.display_image()

    def on_rotate_ccw(self):
        if self.current_scene == 1:
            self.scene.rotate = self.scene.rotate - 90
            self.scene.display_image()
        else:
            self.scene_strip.rotate = self.scene_strip.rotate - 90
            self.scene_strip.display_image()

    def on_delete_items_action(self):
        self.bookKeeper.sensorPolygon = self.bookKeeper.num_timepoints * [None]
        self.scene_strip.removeCompositePolygon()
        self.is_draw_sensor = False
        self.is_draw_strip = False
        self.print_to_console(f"DELETE: Sensor outline deleted.")

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
        self.print_to_console(f"Selected input folder: {self.input_dir}")

    def on_select_output(self):
        self.output_dir = Path(QFileDialog.getExistingDirectory(None, "Select Directory"))
        self.output_edit.setText(str(self.output_dir))
        self.print_to_console(f"Selected output folder: {self.output_dir}")

    def on_file_selection_changed(self, selected):
        for ix in selected.indexes():
            self.print_to_console(f"Selected image: {str(Path(self.input_dir / ix.data()))}")
            try:
                _ = imageio.imread(Path(self.input_dir / ix.data()))
                self.scene.display_image(image_path=Path(self.input_dir / ix.data()))
                self.view.fitInView(QRectF(0, 0, self.scene.pixmap.width(), self.scene.pixmap.width()), Qt.KeepAspectRatio)
                self.image_filename = ix.data()

                # Extract the strip in a different thread and display it
                self.print_to_console(f"Extracting POCT from image ...")
                self.run_get_strip(Path(self.input_dir / ix.data()))
            except Exception as e:
                self.print_to_console(f"ERROR: Loading the selected image failed. {str(e)}")

    def on_strip_extraction_finished(self):
        self.scene_strip.display_image(image=self.strip_img)
        self.view_strip.fitInView(QRectF(0, 0, self.scene_strip.pixmap.width(), self.scene_strip.pixmap.width()),
                                  Qt.KeepAspectRatio)
        self.print_to_console(f"Extracting POCT from image finished successfully.")

    def on_pipeline_finished(self):
        self.print_to_console(f"Results written to {Path(self.output_dir / 'quantification_data.csv')}")
        self.print_to_console(f"Logfile written to {Path(self.output_dir / 'log.txt')}")
        self.print_to_console(f"Settings written to {Path(self.output_dir / 'settings.txt')}")
        self.print_to_console(f"Batch analysis pipeline finished successfully.")

    def on_progress(self, i):
        self.print_to_console("%d%% done" % i)

    def on_save_settings_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "All Files (*);;Text Files (*.txt)", options=options)
        if file_name:
            settings = self.get_parameters()
            # Save parameters into input folder with timestamp
            save_settings(settings, Path(file_name).stem + '.conf')
            self.print_to_console(f"Saved config file under: {file_name}")

    def on_load_settings_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "All Files (*);;Text Files (*.txt);; "
                                                                       "Config Files (*.conf)")
        if file_name:
            settings = load_settings(file_name)
            if 'strip_text_to_search' in settings:
                if settings['strip_text_to_search'] == '':
                    settings['strip_text_to_search'] = '""'
            self.load_parameters(settings)
            self.print_to_console(f"Loaded config : {file_name}")

    def on_write_to_console(self, text):
        """
        Writes new text to the console at the last text cursor position

        Args:
            text (`str`):   Text to be shown on the console.
        """
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.NoMove)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def show_console(self):
        """
        Show and hide the console with the program log.
        """
        if self.consoleDock.isVisible():
            self.consoleDock.hide()
            self.actionView_Console.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogCancelButton))
        else:
            self.consoleDock.show()
            self.actionView_Console.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogApplyButton))

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
        self.print_to_console(f"")
        self.print_to_console(f"Starting analysis with parameters:")
        self.print_to_console(f"               Settings file version: {settings['file_version']}")
        self.print_to_console(f"                               Input: {input_dir}")
        self.print_to_console(f"                              Output: {output_dir}")
        self.print_to_console(f"                 Max number of cores: {settings['max_workers']}")
        self.print_to_console(f"        RAW auto stretch intensities: {settings['raw_auto_stretch']}")
        self.print_to_console(f"        RAW apply auto white balance: {settings['raw_auto_wb']}")
        self.print_to_console(f"  Strip text to search (orientation): {settings['strip_text_to_search']}")
        self.print_to_console(f"          Strip text is on the right: {settings['strip_text_on_right']}")
        self.print_to_console(f"                          Strip size: {settings['strip_size']}")
        self.print_to_console(f"                           Min score: {settings['min_sensor_score']:.2f}")
        self.print_to_console(f"                      QR code border: {settings['qr_code_border']}")
        self.print_to_console(f"               Perform sensor search: {settings['perform_sensor_search']}")
        self.print_to_console(f"                         Sensor size: {settings['sensor_size']}")
        self.print_to_console(f"                       Sensor center: {settings['sensor_center']}")
        self.print_to_console(f"                  Sensor search area: {settings['sensor_search_area']}")
        self.print_to_console(f"             Sensor threshold factor: {settings['sensor_thresh_factor']}")
        self.print_to_console(f"                       Sensor border: {settings['sensor_border']}")
        self.print_to_console(f"    Expected peak relative positions: {settings['peak_expected_relative_location']}")
        self.print_to_console(f"          Subtract signal background: {settings['subtract_background']}")
        self.print_to_console(f"                      Verbose output: {settings['verbose']}")
        self.print_to_console(f"      Create quality-control figures: {settings['qc']}")
        self.print_to_console(f"")

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
        worker.signals.finished.connect(self.on_pipeline_finished)
        self.threadpool.start(worker)

    def run_get_strip(self, image_path):
        worker = Worker(self.set_strip, image_path)
        worker.signals.finished.connect(self.on_strip_extraction_finished)
        self.threadpool.start(worker)

    def set_strip(self, image_path, progress_callback):

        self.progressBar.setFormat("Extracting POCT from image ...")
        self.progressBar.setAlignment(Qt.AlignCenter)
        self.progressBar.setValue(0)
        # Get parameter values
        settings = self.get_parameters()

        # Read the image
        self.progressBar.setValue(20)
        img = imageio.imread(image_path)
        # Extract the strip
        self.progressBar.setValue(60)
        strip_img, _ = extract_strip(img, settings['qr_code_border'])
        self.progressBar.setValue(80)
        self.strip_img = strip_img
        self.p.param('Basic parameters').param('POCT size').param('width').setValue(strip_img.shape[1])
        self.p.param('Basic parameters').param('POCT size').param('height').setValue(strip_img.shape[0])
        self.progressBar.setValue(100)
        self.progressBar.setFormat('Extracting POCT from image finished successfully.')

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
            # c_o_m_strip = currentStripPolygon.getCenterOfMass()
            rect_strip = currentStripPolygon._polygon_item.sceneBoundingRect()
            # c_o_m_sensor = currentSensorPolygon.getCenterOfMass()
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
        elif currentSensorPolygon:

            rect_sensor = currentSensorPolygon._polygon_item.sceneBoundingRect()
            sensor_size = (rect_sensor.width(), rect_sensor.height())
            sensor_center = (rect_sensor.x() - 0 + rect_sensor.width() / 2, rect_sensor.y() -
                             0 + rect_sensor.height() / 2)
            sensor_search_area = (rect_sensor.width() + 10, rect_sensor.height() + 10)
            # Update the parameters in the parameterTree
            self.p.param('Basic parameters').param('Sensor size').param('width').setValue(sensor_size[1])
            self.p.param('Basic parameters').param('Sensor size').param('height').setValue(sensor_size[0])
            self.p.param('Basic parameters').param('Sensor center').param('x').setValue(sensor_center[1])
            self.p.param('Basic parameters').param('Sensor center').param('y').setValue(sensor_center[0])
            self.p.param('Basic parameters').param('Sensor search area').param('x').setValue(sensor_search_area[1])
            self.p.param('Basic parameters').param('Sensor search area').param('y').setValue(sensor_search_area[0])
        else:
            pass
            # self.print_to_console('Please draw POC test outline and sensor outline first')

    @pyqtSlot(int, name="on_signal_scene_nr")
    def on_signal_scene_nr(self, nr):
        self.current_scene = nr
        if nr == 1:
            self.view.setBackgroundBrush(QBrush(QColor(232, 255, 238, 180), Qt.SolidPattern))
            self.view_strip.setBackgroundBrush(QBrush(Qt.white, Qt.SolidPattern))
            # self.print_to_console('You are working on the raw image canvas')
        else:
            self.view_strip.setBackgroundBrush(QBrush(QColor(232, 255, 238, 180), Qt.SolidPattern))
            self.view.setBackgroundBrush(QBrush(Qt.white, Qt.SolidPattern))
            # self.print_to_console('You are working on the POCT canvas')

    @pyqtSlot(float, float, name="handle_add_object_at_position")
    def handle_add_object_at_position(self, x, y):

        if self.is_draw_sensor is True:
            if self.current_scene == 2:
                currentSensorPolygon = self.bookKeeper.getCurrentSensorPolygon()
                if currentSensorPolygon is None:
                    # Create a CompositePolygon
                    currentSensorPolygon = CompositePolygon()

                    # Add the CompositeLine to the Scene. Note that the CompositeLine is
                    # not a QGraphicsItem itself and cannot be added to the Scene directly.
                    currentSensorPolygon.addToScene(self.scene_strip)

                    # Store the polygon
                    self.bookKeeper.addSensorPolygon(currentSensorPolygon)

                # Add the vertices
                if len(currentSensorPolygon._polygon_item.polygon_vertices) < 4:
                    currentSensorPolygon.addVertex(QPointF(x, y))
                    self.print_to_console(f'Drawing sensor corner {len(currentSensorPolygon._polygon_item.polygon_vertices)}')
                self.set_sensor_and_strip_parameter()
            else:
                self.print_to_console('Wrong canvas. Use the POCT canvas below.')

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
                self.print_to_console('Drawing POCT corner')
            self.set_sensor_and_strip_parameter()

    def print_to_console(self, text):
        """
        Print text to console.

        Args:
            text (`str`)        Text to be printed to the console.
        """
        self.logger.info(text)

    @staticmethod
    def get_logger_object(name):
        """
        Gets a logger object to log messages status to the console in a standardized format.

        Returns:
            logger (`object`):      Returns a logger object with correct string formatting.
        """
        logger = logging.getLogger(name)
        if not logger.handlers:
            # Prevent logging from propagating to the root logger
            logger.propagate = 0
            console = logging.StreamHandler(sys.stderr)
            logger.addHandler(console)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            console.setFormatter(formatter)
            logger.setLevel(logging.INFO)

        return logger

    def closeEvent(self, event):

        quit_msg = "Are you sure you want to exit the program?"
        reply = QMessageBox.question(self, 'Message',
                                           quit_msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

