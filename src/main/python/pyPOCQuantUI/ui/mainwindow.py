from PyQt5 import uic
from PyQt5.QtCore import Qt, QDir, QPointF, QSize, QMetaObject, Q_ARG, pyqtSlot, QRectF, QPoint, QThreadPool, \
    QObject, pyqtSignal, QUrl
from PyQt5.QtGui import QTextCursor, QBrush, QColor, QDesktopServices
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QFileSystemModel, QAction, QPlainTextEdit, QSizePolicy, \
    QMessageBox, QStyle, QApplication, QProgressBar, QLineEdit, QDoubleSpinBox
from pyqtgraph.parametertree import Parameter, ParameterTree
from datetime import date
from pathlib import Path
import pyqtgraph as pg
import webbrowser
import imageio
import shutil
import logging
import sys
import pandas as pd
from tqdm import tqdm
import drawSvg as draw
from svglib.svglib import svg2rlg
import pyqrcode
import labels

from pypocquant.lib.io import load_and_process_image
from ui.config import params, key_map
from ui.view import View
from ui.scene import Scene
from ui.compositePolygon import CompositePolygon
from ui.compositeLine import CompositeLine
from ui.bookkeeper import BookKeeper
from ui.worker import Worker
from ui.log import LogTextEdit
from ui.help import About, QuickInstructions
from ui.stream import Stream
from pypocquant.pipeline_FH import run_FH
from pypocquant.lib.tools import extract_strip
from pypocquant.lib.settings import save_settings, load_settings
import pypocquant as pq
from ui.tools import LabelGen


class MainWindow(QMainWindow):

    send_to_console_signal = pyqtSignal(str)
    """
        pyqtSignal used to send a text to the console.

    Args:
        message (`str`)         Text to be sent to the console
    """

    def __init__(self, ui, splash1, splash2, parent=None):
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
        tb.addAction(self.action_load_settings_file)
        tb.addAction(self.action_save_settings_file)
        self.sensor_action = QAction("Draw sensor outline", self)
        self.sensor_action.setStatusTip("Draw sensor outline")
        tb.addAction(self.sensor_action)
        self.sensor_action.triggered.connect(self.on_draw_sensor)
        self.delete_items_action = QAction("Delete sensor", self)
        self.delete_items_action.setStatusTip("Delete sensor")
        tb.addAction(self.delete_items_action)
        self.delete_items_action.triggered.connect(self.on_delete_items_action)
        self.mirror_v_action = QAction("Mirror image vertically", self)
        self.mirror_v_action.setStatusTip("Mirror image vertically")
        tb.addAction(self.mirror_v_action)
        self.mirror_v_action.triggered.connect(self.on_mirror_v)
        self.mirror_h_action = QAction("Mirror image horizontally", self)
        self.mirror_h_action.setStatusTip("Mirror image horizontally")
        tb.addAction(self.mirror_h_action)
        self.mirror_h_action.triggered.connect(self.on_mirror_h)
        self.rotate_cw_action = QAction("Rotate clockwise", self)
        self.rotate_cw_action.setStatusTip("Rotate clockwise")
        tb.addAction(self.rotate_cw_action)
        self.rotate_cw_action.triggered.connect(self.on_rotate_cw)
        self.rotate_ccw_action = QAction("Rotate counter clockwise", self)
        self.rotate_ccw_action.setStatusTip("Rotate counter clockwise")
        tb.addAction(self.rotate_ccw_action)
        self.rotate_ccw_action.triggered.connect(self.on_rotate_ccw)
        self.rotation_angle = QDoubleSpinBox()
        self.rotation_angle.setFixedWidth(50)
        self.rotation_angle.setRange(0, 360)
        self.rotation_angle.setValue(90)
        self.rotation_angle.setDecimals(1)
        self.rotation_angle.setStatusTip('Set rotation angle in degrees')
        tb.addWidget(self.rotation_angle)
        self.zoom_in_action = QAction("Zoom in", self)
        self.zoom_in_action.setStatusTip("Zoom in")
        tb.addAction(self.zoom_in_action)
        self.zoom_in_action.triggered.connect(self.on_zoom_in)
        self.zoom_out_action = QAction("Zoom out", self)
        self.zoom_out_action.setStatusTip("Zoom out")
        tb.addAction(self.zoom_out_action)
        self.zoom_out_action.triggered.connect(self.on_zoom_out)
        self.zoom_reset_action = QAction("Reset zoom", self)
        self.zoom_reset_action.setStatusTip("Reset zoom")
        tb.addAction(self.zoom_reset_action)
        self.zoom_reset_action.triggered.connect(self.on_zoom_reset)
        self.width_action = QAction("Measure distance", self)
        self.width_action.setStatusTip("Measure distance")
        tb.addAction(self.width_action)
        self.width_action.triggered.connect(self.on_draw_line)
        self.action_console = QAction("Show Log", self)
        self.action_console.setShortcut("Ctrl+L")
        self.action_console.setStatusTip('Show / hide console')
        self.action_console.triggered.connect(self.show_console)
        tb.addAction(self.action_console)

        self.label_gen = LabelGen()
        self.label_gen.signal_run_label.connect(self.on_generate_labels)
        self.action_gen_qr_labels.triggered.connect(self.label_gen.show)

        # Instantiate a BookKeeper
        self.bookKeeper = BookKeeper()
        self.display_on_startup = None
        self.image_splash1 = None
        self.image_splash2 = None
        self.splash = splash1
        self.splash2 = splash2
        self.image_filename = None
        self.input_dir = None
        self.output_dir = None
        self.test_dir = None
        self.is_draw_item = None
        # self.is_draw_strip = False
        # self.is_draw_sensor = False
        self.config_file_name = None
        self.run_number = 1
        self.strip_img = None
        self.current_scene = None
        self.relative_bar_positions = []
        self.user_instructions_path = None

        self.input_edit.textChanged.connect(self.on_input_edit_change)
        self.output_edit.textChanged.connect(self.on_output_edit_change)

        img = imageio.imread(self.splash)
        self.image_splash1 = pg.ImageItem(img)
        img = imageio.imread(self.splash2)
        self.image_splash2 = pg.ImageItem(img)
        self.scene = Scene(self.image_splash2, 0.0, 0.0, 500.0, 500.0, nr=int(1))
        self.scene.signal_add_object_at_position.connect(
            self.handle_add_object_at_position)
        self.scene.signal_scene_nr.connect(
            self.on_signal_scene_nr)
        self.scene.signal_line_length.connect(
            self.on_signal_line_length)
        self.view = View(self.scene)
        self.splitter_Right_Column.insertWidget(0, self.view)
        self.viewO.deleteLater()
        self.scene.display_image()
        self.view.resetZoom()
        # Set 2nd scene and view
        self.scene_strip = Scene(self.image_splash1, 0.0, 0.0, 1000.0, 450.0, nr=int(2))
        self.scene_strip.signal_add_object_at_position.connect(
            self.handle_add_object_at_position)
        self.scene_strip.signal_scene_nr.connect(
            self.on_signal_scene_nr)
        self.scene_strip.signal_rel_bar_pos.connect(self.on_signal_rel_bar_pos)
        self.view_strip = View(self.scene_strip)
        self.splitter_Right_Column.insertWidget(0, self.view_strip)
        self.viewO2.deleteLater()
        self.scene_strip.display_image()
        self.view_strip.resetZoom()

        # Setup parameter tree
        self.p = Parameter.create(name='params', type='group', children=params)
        # self.t = ParameterTree()
        self.paramTree.setParameters(self.p, showTop=False)
        self.p.sigTreeStateChanged.connect(self.on_parameter_tree_change)
        param_dict = self.get_parameters()
        self.relative_bar_positions = list(param_dict['peak_expected_relative_location'])

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

    def on_parameter_tree_change(self, param, changes):
        for param, change, data in changes:
            path = self.p.childPath(param)
            if path[-1] == 'IgM':
                self.relative_bar_positions[0] = data
                self.update_bar_pos()
            if path[-1] == 'IgG':
                self.relative_bar_positions[1] = data
                self.update_bar_pos()
            if path[-1] == 'Ctl':
                self.relative_bar_positions[2] = data
                self.update_bar_pos()

    def update_bar_pos(self):
        currentSensorPolygon = self.bookKeeper.getCurrentSensorPolygon()
        if currentSensorPolygon:
            # Update the relative bar positions
            currentSensorPolygon._polygon_item.relative_bar_positions = self.relative_bar_positions
            # Actually move the bars on scene
            currentSensorPolygon._polygon_item.move_line_item()

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
        webbrowser.open(str(Path(self.user_instructions_path)))

    def on_draw_line(self):
        self.is_draw_item = 2

    def on_draw_strip(self):
        self.is_draw_item = 1
        # self.is_draw_strip = True
        # self.is_draw_sensor = False

    def on_draw_sensor(self):
        self.is_draw_item = 0
        # self.is_draw_strip = False
        # self.is_draw_sensor = True

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

    def on_zoom_reset(self):
        if self.current_scene == 1:
            self.view.resetZoom()
        else:
            self.view_strip.resetZoom()

    def on_rotate_cw(self):
        if self.current_scene == 1:
            self.scene.rotate = self.scene.rotate + self.rotation_angle.value()
            self.scene.display_image()
        else:
            self.scene_strip.rotate = self.scene_strip.rotate + self.rotation_angle.value()
            self.scene_strip.display_image()

    def on_rotate_ccw(self):
        if self.current_scene == 1:
            self.scene.rotate = self.scene.rotate - self.rotation_angle.value()
            self.scene.display_image()
        else:
            self.scene_strip.rotate = self.scene_strip.rotate - self.rotation_angle.value()
            self.scene_strip.display_image()

    def on_delete_items_action(self):
        self.bookKeeper.sensorPolygon = self.bookKeeper.num_timepoints * [None]
        self.scene_strip.removeCompositePolygon()
        self.is_draw_item = -1
        # self.is_draw_sensor = False
        # self.is_draw_strip = False
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
        # webbrowser.open(str(self.test_dir))
        QDesktopServices.openUrl(QUrl(f'file:///{str(self.test_dir)}'))

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
        # webbrowser.open(str(self.output_dir))
        QDesktopServices.openUrl(QUrl(f'file:///{str(self.output_dir)}'))

    def on_select_input(self):
        self.input_dir = Path(QFileDialog.getExistingDirectory(None, "Select Directory"))
        self.input_edit.setText(str(self.input_dir))
        self.output_dir = Path(self.input_dir / 'pipeline')
        self.output_edit.setText(str(Path(self.input_dir / 'pipeline')))
        self.on_updated_input_folder()

    def on_input_edit_change(self):
        new_path = self.input_edit.text()
        # Validate if path exists
        if Path(new_path).is_dir():
            self.input_dir = Path(new_path)
            self.output_dir = Path(self.input_dir / 'pipeline')
            self.output_edit.setText(str(Path(self.input_dir / 'pipeline')))
            self.print_to_console(f"Updated input directory: {Path(new_path)}")
            self.on_updated_input_folder()
        else:
            self.print_to_console(f"Selected folder does not to seem to exist: {Path(new_path)}")

    def on_updated_input_folder(self):
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

    def on_output_edit_change(self):
        new_path = self.output_edit.text()
        # Validate if path exists
        if Path(new_path).is_dir():
            self.output_dir = Path(new_path)
            self.print_to_console(f"Updated output directory: {Path(new_path)}")
        else:
            self.print_to_console(f"Selected folder does not to seem to exist: {Path(new_path)}")

    def on_file_selection_changed(self, selected):
        for ix in selected.indexes():
            self.print_to_console(f"Selected image: {str(Path(self.input_dir / ix.data()))}")
            try:
                ret = load_and_process_image(Path(self.input_dir / ix.data()), to_rgb=True)
                if ret is not None:
                    self.scene.display_image(image=ret)
                    self.view.resetZoom()
                    self.image_filename = ix.data()

                    # Extract the strip in a different thread and display it
                    self.print_to_console(f"Extracting POCT from image ...")
                    self.progressBar.setFormat("Extracting POCT from image ...")
                    self.progressBar.setAlignment(Qt.AlignCenter)
                    self.run_get_strip(Path(self.input_dir / ix.data()))
                else:
                    self.print_to_console(f"ERROR: The file {str(ix.data())} could not be opened.")
            except Exception as e:
                self.print_to_console(f"ERROR: Loading the selected image failed. {str(e)}")

    def on_strip_extraction_finished(self):
        self.scene_strip.display_image(image=self.strip_img)
        self.view_strip.resetZoom()
        self.progressBar.setFormat('Extracting POCT from image finished successfully.')
        self.print_to_console(f"Extracting POCT from image finished successfully.")

    def on_pipeline_finished(self):
        self.print_to_console(f"Results written to {Path(self.output_dir / 'quantification_data.csv')}")
        self.print_to_console(f"Logfile written to {Path(self.output_dir / 'log.txt')}")
        self.print_to_console(f"Settings written to {Path(self.output_dir / 'settings.txt')}")
        self.print_to_console(f"Batch analysis pipeline finished successfully.")

    def on_save_settings_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "All Files (*);;Text Files (*.txt)", options=options)
        if file_name:
            settings = self.get_parameters()
            # Save parameters into input folder with timestamp
            save_path = Path(Path(file_name).parent, Path(file_name).stem + '.conf')
            save_settings(settings, save_path)
            self.print_to_console(f"Saved config file under: {save_path}")

    def on_load_settings_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "All Files (*);;Text Files (*.txt);; "
                                                                       "Config Files (*.conf)")
        if file_name:
            settings = load_settings(file_name)
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

    def on_progress_bar(self, i):
        self.progressBar.setValue(i)

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

    def run_pipeline(self, input_dir, output_dir, settings, progress_callback):
        # Inform the user
        self.print_to_console(f"")
        self.print_to_console(f"Starting analysis with parameters:")
        self.print_to_console(f"                               Input: {input_dir}")
        self.print_to_console(f"                              Output: {output_dir}")
        self.print_to_console(f"                 Max number of cores: {settings['max_workers']}")
        self.print_to_console(f"        RAW auto stretch intensities: {settings['raw_auto_stretch']}")
        self.print_to_console(f"        RAW apply auto white balance: {settings['raw_auto_wb']}")
        self.print_to_console(f"  Strip text to search (orientation): {settings['strip_text_to_search']}")
        self.print_to_console(f"          Strip text is on the right: {settings['strip_text_on_right']}")
        # self.print_to_console(f"                          Strip size: {settings['strip_size']}")
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
        worker.signals.progress.connect(self.on_progress_bar)
        worker.signals.finished.connect(self.on_strip_extraction_finished)
        self.threadpool.start(worker)

    def set_strip(self, image_path, progress_callback):

        progress_callback.emit(1)
        # Get parameter values
        settings = self.get_parameters()

        # Read the image
        progress_callback.emit(20)
        img = load_and_process_image(image_path, to_rgb=False)
        # Extract the strip
        progress_callback.emit(60)
        strip_img, _ = extract_strip(
            img,
            settings['qr_code_border'],
            settings['strip_text_to_search'],
            settings['strip_text_on_right']
        )
        progress_callback.emit(80)

        # Change to RGB for display
        strip_img[:, :, [0, 1, 2]] = strip_img[:, :, [2, 1, 0]]

        self.strip_img = strip_img
        progress_callback.emit(100)

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
            self.p.param('Basic parameters').param('Sensor size').param('width').setValue(sensor_size[1])
            self.p.param('Basic parameters').param('Sensor size').param('height').setValue(sensor_size[0])
            self.p.param('Basic parameters').param('Sensor center').param('x').setValue(sensor_center[1])
            self.p.param('Basic parameters').param('Sensor center').param('y').setValue(sensor_center[0])
            self.p.param('Advanced parameters').param('Sensor search area').param('x').setValue(sensor_search_area[1])
            self.p.param('Advanced parameters').param('Sensor search area').param('y').setValue(sensor_search_area[0])
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
            self.p.param('Advanced parameters').param('Sensor search area').param('x').setValue(sensor_search_area[1])
            self.p.param('Advanced parameters').param('Sensor search area').param('y').setValue(sensor_search_area[0])
        else:
            pass
            # self.print_to_console('Please draw POC test outline and sensor outline first')

    @pyqtSlot(str, Path, name="on_generate_labels")
    def on_generate_labels(self, path1, path2):
        self.print_to_console('Starting label generation')
        worker = Worker(self.run_get_qr_codes, path1, path2)
        worker.signals.finished.connect(self.on_done_labels)
        self.threadpool.start(worker)

    def on_done_labels(self):
        self.print_to_console('Done with creating labels')

    @pyqtSlot(int, name="on_signal_line_length")
    def on_signal_line_length(self, length):
        self.p.param('Basic parameters').param('QR code border').setValue(length)

    @pyqtSlot(list, name="om_signal_rel_bar_pos")
    def on_signal_rel_bar_pos(self, rel_pos):
        self.relative_bar_positions = rel_pos
        self.p.param('Basic parameters').param('Peak expected relative location').param('IgM').setValue(rel_pos[0])
        self.p.param('Basic parameters').param('Peak expected relative location').param('IgG').setValue(rel_pos[1])
        self.p.param('Basic parameters').param('Peak expected relative location').param('Ctl').setValue(rel_pos[2])

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

        if self.is_draw_item is 0:
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
                    if len(currentSensorPolygon._polygon_item.polygon_vertices) == 4:
                        rect_sensor = currentSensorPolygon._polygon_item.sceneBoundingRect()
                        settings = self.get_parameters()
                        currentSensorPolygon.addLine(settings['peak_expected_relative_location'])
                self.set_sensor_and_strip_parameter()
            else:
                self.print_to_console('Wrong canvas. Use the POCT canvas above.')

        elif self.is_draw_item is 1:
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

        elif self.is_draw_item is 2:
            # Create a CompositeLine
            currentLine = CompositeLine(QPointF(x, y))

            # Add the CompositeLine to the Scene. Note that the CompositeLine is
            # not a QGraphicsItem itself and cannot be added to the Scene directly.
            currentLine.addToScene(self.scene)

            # Store the line
            self.bookKeeper.addLine(currentLine)

            # Deactivate drawing mode
            self.is_draw_item = -1

    def print_to_console(self, text):
        """
        Print text to console.

        Args:
            text (`str`)        Text to be printed to the console.
        """
        self.logger.info(text)

    def show_console(self):
        """
        Show and hide the console with the program log.
        """
        if self.dockWidget.isVisible():
            self.dockWidget.hide()
            # self.action_console.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogCancelButton))
        else:
            self.dockWidget.show()
            # self.action_console.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogApplyButton))

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

    def run_get_qr_codes(self, label_dir, qrdecode_result_dir_str, progress_callback):

        try:
            label_paths = []
            # @todo add form to fill out all arguments to allow for custom page design
            # Create an A4 portrait (210mm x 297mm) sheets with 2 columns and 8 rows of
            # labels. Each label is 90mm x 25mm with a 2mm rounded corner. The margins are
            # automatically calculated.
            specs = labels.Specification(210, 297, 4, 14, 50, 20, corner_radius=1, left_padding=1, top_padding=1,
                                         bottom_padding=1, right_padding=1, padding_radius=1, left_margin=1,
                                         right_margin=1)

            def draw_label(label, width, height, obj):
                # drawing = Image(0, 0, 300, 150, obj)
                # scaled_drawing = self.scale(drawing, scaling_factor=0.4)
                label.add(obj)

            # Create the sheet.
            sheet = labels.Sheet(specs, draw_label, border=True)
            # Read the label template
            data = pd.read_csv(label_dir, header=None)

            for i in tqdm(range(len(data))):
                save_name_qr = qrdecode_result_dir_str.joinpath('qr', data.iloc[i, 0] + 'qr.svg')
                qr_path = qrdecode_result_dir_str.joinpath('qr')
                qr_path.mkdir(exist_ok=True)

                # Create qr code
                qr = pyqrcode.create(data.iloc[i, 0])

                # # Save qr codes
                # # qr.png(str(save_name), scale=3, quiet_zone=6)
                # qr.svg(str(save_name_qr), scale=3, quiet_zone=6)

                # Add human readable information to the label
                value_string = data.iloc[i, 0].split('-')
                d = draw.Drawing(300, 150)
                d.append(draw.Text(value_string[0], 20, 150, 105, **{"font-family": "Aria, sans-serif"}))
                d.append(draw.Text(value_string[1], 17, 150, 72, **{"font-family": "Aria, sans-serif"}))
                d.append(draw.Text(value_string[2], 25, 150, 38, **{"font-family": "Aria, sans-serif"}))
                d.append(draw.Text(value_string[3], 25, 150, 5, **{"font-family": "Aria, sans-serif"}))
                d.append(draw.Image(1, 1, 145, 145, str(save_name_qr)))
                save_name = qrdecode_result_dir_str.joinpath(data.iloc[i, 0] + '.svg')
                d.saveSvg(str(save_name))

                label_paths.append(str(save_name))

                # scaled_drawing = self.scale(dd.asDataUri(), scaling_factor=0.4)
                # print(d)
                # print(type(d))
                # sheet.add_label(str(save_name))

                # @todo should be possible to pass the label from memory instead of reading it
                drawing = svg2rlg(str(save_name))
                scaled_drawing = self.scale(drawing, scaling_factor=0.4)
                sheet.add_label(scaled_drawing)

            # Save the file and we are done.
            sheet.save(str(Path(qrdecode_result_dir_str / 'qc_labels.pdf')))
        except Exception as e:
            print(e)

    @staticmethod
    def scale(drawing, scaling_factor):
        """
        Scale a reportlab.graphics.shapes.Drawing()
        object while maintaining the aspect ratio
        """
        scaling_x = scaling_factor
        scaling_y = scaling_factor

        drawing.width = drawing.minWidth() * scaling_x
        drawing.height = drawing.height * scaling_y
        drawing.scale(scaling_x, scaling_y)
        return drawing

    def closeEvent(self, event):

        quit_msg = "Are you sure you want to exit the program?"
        reply = QMessageBox.question(self, 'Message',
                                           quit_msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
