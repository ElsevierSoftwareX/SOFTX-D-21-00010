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

from PyQt5 import uic
from PyQt5.QtCore import Qt, QDir, QPointF, pyqtSlot, QRectF, QPoint, QThreadPool, \
    pyqtSignal, QUrl, QSettings
from PyQt5.QtGui import QTextCursor, QBrush, QColor, QDesktopServices
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QFileSystemModel, QAction, QMessageBox, QStyle, QApplication, \
    QDoubleSpinBox
from pyqtgraph.parametertree import Parameter
from datetime import date
from pathlib import Path
import pyqtgraph as pg
import webbrowser
import imageio
import shutil
import logging
import sys
import pandas as pd
from pyqtgraph.parametertree.parameterTypes import SimpleParameter
from tqdm import tqdm
import drawSvg as draw
from svglib.svglib import svg2rlg
import pyqrcode
import labels
import tempfile
import os

from pypocquant.lib.io import load_and_process_image, is_raw
from ui.config import params, key_map
from ui.view import View
from ui.scene import Scene
from ui.compositePolygon import CompositePolygon
from ui.compositeLine import CompositeLine
from ui.compositeRect import CompositeRect
from ui.bookkeeper import BookKeeper
from ui.worker import Worker
from ui.log import LogTextEdit
from ui.help import About, QuickInstructions
from ui.stream import Stream
from ui import versionInfo
from pypocquant.lib.analysis import get_rectangles_from_image_and_rectangle_props
from pypocquant.lib.pipeline import run_pipeline
from pypocquant.lib.tools import extract_strip
from pypocquant.lib.settings import save_settings, load_settings
from pypocquant.split_images_by_strip_type_parallel import run_pool

import platform

__operating_system__ = '{} {}'.format(platform.system(), platform.architecture()[0])


class MainWindow(QMainWindow):

    send_to_console_signal = pyqtSignal(str)
    """
        pyqtSignal used to send a text to the console.

    Args:
        message (`str`)         Text to be sent to the console
    """

    def __init__(self, ui, splash1, splash2, icon, parent=None):
        super().__init__(parent)
        uic.loadUi(ui, self)

        self.setWindowTitle('pyPOCQuant:: Point of Care Test Quantification tool [build: v % s % s]'
                            % (versionInfo.get_version_string(), __operating_system__))

        # Add filemenu
        self.action_save_settings_file.triggered.connect(self.on_save_settings_file)
        self.action_save_settings_file.setShortcut("Ctrl+S")
        self.action_load_settings_file.triggered.connect(self.on_load_settings_file)
        self.action_load_settings_file.setShortcut("Ctrl+O")
        self.actionQuit.triggered.connect(self.close)
        self.about_window_icon = icon
        self.about_window = About(icon_path=self.about_window_icon)
        self.actionAbout.setShortcut("Ctrl+A")
        self.actionAbout.triggered.connect(self.on_show_about)
        self.actionManual.triggered.connect(self.on_show_manual)
        settings = QSettings("CSB & SCF", "pyPOCQuantUI")
        self.qi = None
        if settings.value("quickstart/show_on_start", True, type=bool):
            self.qi = QuickInstructions()
            self.qi.show()
        self.actionQuick_instructions.setStatusTip('Opens the user manual of pyPOCQuant')
        self.actionQuick_instructions.triggered.connect(self.on_quick_instructions)
        self.actionQuick_start.setStatusTip('Hints about how to use this program and common problems and how to avoid'
                                            ' them')
        self.actionQuick_start.triggered.connect(self.on_quick_start)
        self.action_save_POCT_template.triggered.connect(self.on_show_poct_template)
        self.action_save_QR_labels_template.triggered.connect(self.on_show_qr_label_template)

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
        self.width_action.setCheckable(True)
        self.width_action.toggled.connect(self.on_toggle_line)
        # self.width_action.triggered.connect(self.on_draw_line)
        self.action_console = QAction("Show Log", self)
        self.action_console.setShortcut("Ctrl+L")
        self.action_console.setStatusTip('Show / hide console')
        self.action_console.triggered.connect(self.show_console)
        tb.addAction(self.action_console)

        # Instantiate a BookKeeper
        self.bookKeeper = BookKeeper()
        self.label_gen = None
        self.split_images = None
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
        self.is_line_toggle = False
        # self.is_draw_strip = False
        # self.is_draw_sensor = False
        self.config_file_name = None
        self.config_path = None
        self.run_number = 1
        self.strip_img = None
        self.current_scene = None
        self.relative_bar_positions = []
        self.sensor_attributes = []
        self.sensor_parameters = []
        self.user_instructions_path = None
        self.quick_start_path = None
        self.poct_template_path = None
        self.qr_labels_template_path = None

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
        self.scene_strip.signal_sensor_attributes.connect(self.on_signal_sensor_attributes)
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
        # self.sensor_attributes = list(param_dict['sensor_center']) + list(param_dict['sensor_size'])
        self.sensor_attributes = list(param_dict['sensor_center']) + list(param_dict['sensor_size']) + \
                                 list(param_dict['sensor_search_area']) + list(param_dict['sensor_search_area'])

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
        sys.stderr = Stream()
        sys.stderr.stream_signal.connect(self.on_write_to_console)
        self.logger = self.get_logger_object(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.print_to_console('Welcome to pyPOCQuant')

        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True)

    def on_parameter_tree_change(self, param, changes):
        for param, change, data in changes:
            path = self.p.childPath(param)
            if path is None:
                # This happens, for instance, after a node removal
                return
            if len(path) == 3 and path[1] == 'Band expected relative location':
                index = int(path[2])
                self.relative_bar_positions[index] = data
                self.update_bar_pos()
            if path[-1] == 'Relative height factor':
                self.set_hough_rect()
            if path[-1] == 'Relative center cut-off':
                self.set_hough_rect()
            if path[-1] == 'Relative border cut-off':
                self.set_hough_rect()
            if path[-1] == 'Try to correct strip orientation':
                self.set_hough_rect()
            if path[-2] == 'Sensor center' and path[-1] == 'x':
                self.sensor_attributes[0] = data
                self.update_sensor_pos()
            if path[-2] == 'Sensor center' and path[-1] == 'y':
                self.sensor_attributes[1] = data
                self.update_sensor_pos()
            if path[-2] == 'Sensor size' and path[-1] == 'width':
                self.sensor_attributes[2] = data
                self.p.param('Advanced parameters').param('Sensor search area').param('x').setValue(
                    round(self.sensor_attributes[2] + self.sensor_attributes[6]) + 2)
                self.update_sensor_pos()
            if path[-2] == 'Sensor size' and path[-1] == 'height':
                self.sensor_attributes[3] = data
                self.p.param('Advanced parameters').param('Sensor search area').param('y').setValue(
                    round(self.sensor_attributes[3] + self.sensor_attributes[7]) + 2)
                self.update_sensor_pos()
            if path[-2] == 'Sensor search area' and path[-1] == 'x':
                self.sensor_attributes[4] = data
                self.sensor_attributes[6] = round(self.sensor_attributes[4] - self.sensor_attributes[2]) - 2
                self.update_sensor_pos()
            if path[-2] == 'Sensor search area' and path[-1] == 'y':
                self.sensor_attributes[5] = data
                self.sensor_attributes[7] = round(self.sensor_attributes[5] - self.sensor_attributes[3]) - 2
                self.update_sensor_pos()
            if path[-1] == 'Number of sensor bands':
                num_bands = int(data)
                self._update_sensor_band_parameters_for_num_bands(num_bands)

    def update_sensor_pos(self):
        currentSensorPolygon = self.bookKeeper.getCurrentSensorPolygon()
        if currentSensorPolygon:
            currentSensorPolygon._polygon_item.attributes = self.sensor_attributes
            currentSensorPolygon._polygon_item.update_polygon()

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
        if self.qi is None:
            self.qi = QuickInstructions()
        self.qi.show()

    def on_show_about(self):
        """
        Displays the about window.
        """
        self.about_window.show()

    def on_show_manual(self):
        """
        Displays the instruction manual.
        """
        tmp_path = Path(tempfile.gettempdir()).joinpath('pyPOCQuant')
        tmp_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(self.user_instructions_path), str(tmp_path))
        file_url = self.get_file_url(str(Path(tmp_path / Path(self.user_instructions_path).name)))
        webbrowser.open(file_url)

    def on_quick_start(self):
        """
        Displays the quick start guide.
        """
        tmp_path = Path(tempfile.gettempdir()).joinpath('pyPOCQuant')
        tmp_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(self.quick_start_path), str(tmp_path))
        file_url = self.get_file_url(str(Path(tmp_path / Path(self.quick_start_path).name)))
        webbrowser.open(file_url)

    def on_show_poct_template(self):
        """
        Displays the POCT template
        """
        tmp_path = Path(tempfile.gettempdir()).joinpath('pyPOCQuant')
        tmp_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(self.poct_template_path), str(tmp_path))
        file_url = self.get_file_url(str(tmp_path))
        webbrowser.open(file_url)

    def on_show_qr_label_template(self):
        """
        Displays the QR labels template
        """
        tmp_path = Path(tempfile.gettempdir()).joinpath('pyPOCQuant')
        tmp_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(self.qr_labels_template_path), str(tmp_path))
        file_url = self.get_file_url(str(tmp_path))
        webbrowser.open(file_url)

    def on_toggle_line(self):

        self.is_line_toggle = not self.is_line_toggle
        if self.is_line_toggle:
            # Draw line on scene
            self.on_draw_line()
        else:
            # Remove from scene and bookKeeper
            self.scene.removeCompositeLine()
            self.bookKeeper.removeLine()

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
            reply = QMessageBox.question(self, 'pyPOCQuant :: Test', msg, QMessageBox.Ok)

            if reply == QMessageBox.Ok:
                return
            else:
                return

        # 1. Create a temp directory
        # Make sure the results folder exists
        self.test_dir = Path(self.input_dir / "test")
        self.test_dir.mkdir(exist_ok=True)

        # Clear the directory for subsequent re tests. Otherwise it would run the pipeline on all old images too.
        try:
            if any(Path(str(self.test_dir)).iterdir()):
                for p in self.test_dir.iterdir():
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
        except Exception as e:
            self.print_to_console(f"ERROR: Could not delete old files. Close all open files first and try again. "
                                  f"{str(e)}")
            return

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
        QDesktopServices.openUrl(QUrl(f'file:///{str(self.test_dir)}'))

    def on_run_pipeline(self):

        if self.input_dir is None or self.output_dir is None:
            if not self.input_dir:
                msg = "Please select an input folder first."
            elif not self.output_dir:
                msg = "Please select an output folder first"
            reply = QMessageBox.question(self, 'pyPOCQuant :: Run', msg, QMessageBox.Ok)

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
        self.output_dir.mkdir(exist_ok=True)
        QDesktopServices.openUrl(QUrl(f'file:///{str(self.output_dir)}'))

    def on_select_input(self):
        input_dir = QFileDialog.getExistingDirectory(None, "Select Directory")
        if input_dir == '':
            # The user cancelled the selection
            return
        self.input_dir = Path(input_dir)
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
        output_dir = QFileDialog.getExistingDirectory(None, "Select Directory")
        if output_dir == '':
            # The user cancelled the selection
            return
        self.output_dir = Path(output_dir)
        self.output_edit.setText(str(self.output_dir))
        self.print_to_console(f"Selected output folder: {self.output_dir}")

    def on_output_edit_change(self):
        new_path = self.output_edit.text()
        # Validate if path exists
        if Path(new_path).parent.is_dir():
            self.output_dir = Path(new_path)
            self.print_to_console(f"Updated output directory: {Path(new_path)}")
        else:
            self.print_to_console(f"Selected folder does not to seem to exist: {Path(new_path)}")

    def on_file_selection_changed(self, selected):
        for ix in selected.indexes():
            file_path = Path(self.input_dir / ix.data())
            self.print_to_console(f"Selected file: {str(file_path)}")
            try:
                if file_path.suffix == '.conf':
                    self.config_path = str(file_path)
                    self.on_load_settings_file_from_path()
                else:
                    ret = load_and_process_image(file_path, to_rgb=True)
                    if ret is not None:
                        self.scene_strip.removeHoughRect()
                        self.scene.display_image(image=ret)
                        self.view.resetZoom()
                        self.image_filename = ix.data()

                        # Extract the strip in a different thread and display it
                        self.print_to_console(f"Extracting POCT from image ...")
                        self.progressBar.setFormat("Extracting POCT from image ...")
                        self.progressBar.setAlignment(Qt.AlignCenter)
                        self.run_get_strip(Path(file_path))
                    else:
                        self.print_to_console(f"ERROR: The file {str(ix.data())} could not be opened.")
            except Exception as e:
                self.print_to_console(f"ERROR: Loading the selected image failed. {str(e)}")

    def on_strip_extraction_finished(self):
        if self.strip_img is not None:
            self.scene_strip.display_image(image=self.strip_img)
        else:
            self.scene_strip.display_image(image=self.image_splash1.image.copy())
        self.view_strip.resetZoom()
        self.set_hough_rect()
        if self.config_path:
            self.on_delete_items_action()
            self.add_sensor_at_position()
        if self.strip_img is not None:
            self.progressBar.setFormat('Extracting POCT from image finished successfully.')
            self.print_to_console(f"Extracting POCT from image finished successfully.")
        else:
            self.progressBar.setFormat('Extracting POCT from image finished with an error.')
            self.print_to_console(f"Extracting POCT from image finished with an error.")

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
            # Get current settings
            settings = self.get_parameters()

            # Drop some settings that are not used (and incompatible) with
            # the command-line and jupyter usage of the pipeline
            if 'max_workers' in settings:
                del settings['max_workers']
            if 'sensor_bands_number' in settings:
                del settings['sensor_bands_number']

            # Save parameters into input folder with timestamp
            save_path = Path(Path(file_name).parent, Path(file_name).stem + '.conf')
            save_settings(settings, save_path)
            self.print_to_console(f"Saved config file under: {save_path}")

    def on_load_settings_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            'Open file',
            '',
            "All Files (*);;Text Files (*.txt);; Config Files (*.conf)"
        )
        if file_name:
            self.config_path = file_name
            self.on_load_settings_file_from_path()

    def on_load_settings_file_from_path(self):
        try:
            settings = load_settings(self.config_path)
            self.load_parameters(settings)
            self.on_delete_items_action()
            self.add_sensor_at_position()
            self.print_to_console(f"Loaded config : {self.config_path}")
        except Exception as e:
            self.print_to_console(f"ERROR: Loading the selected config failed. {str(e)}")

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

        # Add UI-only parameters
        settings["max_workers"] = 2
        settings["sensor_bands_number"] = len(settings["peak_expected_relative_location"])

        # Back-compatibility: make sure that recent settings are in
        if "control_band_index" not in settings:
            settings["control_band_index"] = -1

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
        self.print_to_console(f"                                  Input: {input_dir}")
        self.print_to_console(f"                                 Output: {output_dir}")
        self.print_to_console(f"                    Max number of cores: {settings['max_workers']}")
        self.print_to_console(f"           RAW auto stretch intensities: {settings['raw_auto_stretch']}")
        self.print_to_console(f"           RAW apply auto white balance: {settings['raw_auto_wb']}")
        self.print_to_console(f"   Try to correct for strip orientation: {settings['strip_try_correct_orientation']}")
        self.print_to_console(f" Strip orientation rectangle properties: {settings['strip_try_correct_orientation_rects']}")
        self.print_to_console(f"     Strip text to search (orientation): {settings['strip_text_to_search']}")
        self.print_to_console(f"             Strip text is on the right: {settings['strip_text_on_right']}")
        # self.print_to_console(f"                             Strip size: {settings['strip_size']}")
        self.print_to_console(f"                         QR code border: {settings['qr_code_border']}")
        self.print_to_console(f"                  Perform sensor search: {settings['perform_sensor_search']}")
        self.print_to_console(f"                            Sensor size: {settings['sensor_size']}")
        self.print_to_console(f"                          Sensor center: {settings['sensor_center']}")
        self.print_to_console(f"                     Sensor search area: {settings['sensor_search_area']}")
        self.print_to_console(f"                Sensor threshold factor: {settings['sensor_thresh_factor']}")
        self.print_to_console(f"                          Sensor border: {settings['sensor_border']}")
        self.print_to_console(f"                      Sensor band names: {settings['sensor_band_names']}")
        self.print_to_console(f"       Expected peak relative positions: {settings['peak_expected_relative_location']}")
        self.print_to_console(f"                     Control band index: {settings['control_band_index']}")
        self.print_to_console(f"             Subtract signal background: {settings['subtract_background']}")
        self.print_to_console(f"                       Force FID search: {settings['force_fid_search']}")
        self.print_to_console(f"                         Verbose output: {settings['verbose']}")
        self.print_to_console(f"         Create quality-control figures: {settings['qc']}")
        self.print_to_console(f"")

        # Run the pipeline
        run_pipeline(
            input_dir,
            output_dir,
            raw_auto_stretch=settings['raw_auto_stretch'],
            raw_auto_wb=settings['raw_auto_wb'],
            strip_try_correct_orientation=settings['strip_try_correct_orientation'],
            strip_try_correct_orientation_rects=settings['strip_try_correct_orientation_rects'],
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
            control_band_index=settings['control_band_index'],
            subtract_background=settings['subtract_background'],
            force_fid_search=settings['force_fid_search'],
            sensor_band_names=settings['sensor_band_names'],
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
        stretch_for_hough = is_raw(str(image_path)) is True and settings['raw_auto_stretch'] is False
        strip_img, error_message, left_rect, right_rect = extract_strip(
            img,
            settings['qr_code_border'],
            settings['strip_try_correct_orientation'],
            settings['strip_try_correct_orientation_rects'],
            stretch_for_hough,
            settings['strip_text_to_search'],
            settings['strip_text_on_right']
        )
        progress_callback.emit(80)

        if strip_img is None:
            self.print_to_console("ERROR: " + error_message)

        else:
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

        # Update dependencies among virtual parameters
        self._update_dynamic_parameter_dependencies()

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

    def _update_dynamic_parameter_dependencies(self):
        """Update dynamic parameters that depend on each other."""

        sensor_bands_number = len(self.p.param("Basic parameters").param("Sensor band names").children())
        self.p.param("Basic parameters").param("Number of sensor bands").setValue(sensor_bands_number)
        control_band_index = self.p.param("Basic parameters").param("Control band index")
        if int(control_band_index.value()) > (sensor_bands_number - 1):
            control_band_index.setValue(sensor_bands_number - 1)
        control_band_index.setLimits((-1, sensor_bands_number - 1))

    def _update_sensor_band_parameters_for_num_bands(self, num_bands: int):
        """Update the various sensor bands parameters to match the new number of bands."""

        # We need to update the parameters in the Tree so that we can have it update
        # itself properly.

        # Get the relevant parameters
        number_of_sensor_bands = self.p.param("Basic parameters").param("Number of sensor bands")
        control_band_index = self.p.param("Basic parameters").param("Control band index")
        sensor_band_names = self.p.param("Basic parameters").param("Sensor band names")
        band_expected_relative_location = self.p.param("Basic parameters").param("Band expected relative location")

        # Current number of bands
        current_num_bands = len(sensor_band_names.children())

        # Let's consider the different cases.

        if num_bands == current_num_bands:

            # The number of bands did not change (this stops recursion).
            return

        elif num_bands < current_num_bands:

            # The number of bands was reduced

            # Remove sensor band names
            with sensor_band_names.treeChangeBlocker():

                n = current_num_bands
                while n > num_bands:
                    sensor_band_names.children()[n - 1].remove()
                    n -= 1

            # Remove expected band positions
            with band_expected_relative_location.treeChangeBlocker():

                n = current_num_bands
                while n > num_bands:
                    band_expected_relative_location.children()[n - 1].remove()
                    n -= 1

            if int(control_band_index.value()) > (num_bands - 1):
                control_band_index.setValue(num_bands - 1)

            # Check
            assert(int(number_of_sensor_bands.value()) == num_bands)

        else:

            # The number of bands was increased

            new_sensor_indices = [f"{i}" for i in range(num_bands)]
            new_sensor_band_names = [f"tl{i}" for i in range(num_bands)]
            new_peak_expected_relative_location = [(0.5/num_bands) + (float(i) / num_bands) for i in range(num_bands)]

            # Add sensor band names
            with sensor_band_names.treeChangeBlocker():

                n = current_num_bands
                while n < num_bands:
                    sensor_band_names.addChild(
                        SimpleParameter(
                            name=new_sensor_indices[n],
                            type='str',
                            value=new_sensor_band_names[n]
                        )
                    )
                    n += 1

            # Add expected band positions
            with band_expected_relative_location.treeChangeBlocker():

                n = current_num_bands
                while n < num_bands:
                    band_expected_relative_location.addChild(
                        SimpleParameter(
                            name=new_sensor_indices[n],
                            type='float',
                            value=new_peak_expected_relative_location[n],
                            step=0.05,
                            limits=(0, 1)
                        )
                    )
                    n += 1

        # Ensure strict increasing numbering of children
        with sensor_band_names.treeChangeBlocker():
            for i in range(num_bands):
                if sensor_band_names.children()[i].name() != str(i):
                    sensor_band_names.children()[i].setName(str(i))

        with band_expected_relative_location.treeChangeBlocker():
            for i in range(num_bands):
                if band_expected_relative_location.children()[i].name() != str(i):
                    band_expected_relative_location.children()[i].setName(str(i))

        # Remove current sensor from the canvas
        self.on_delete_items_action()

    @staticmethod
    def change_parameter_keys(parameters, key_map):
        parameter_out = dict((key_map[key], value) for (key, value) in parameters.items())
        return parameter_out

    def set_hough_rect(self):

        if self.p.param('Advanced parameters').param('Try to correct strip orientation').value():

            currentHoughRect = self.bookKeeper.getCurrentHoughRect()
            height_fact = self.p.param('Advanced parameters').param('Strip orientation correction search rectangles').param(
                'Relative height factor').value()
            center_cutoff = self.p.param('Advanced parameters').param('Strip orientation correction search rectangles').param(
                'Relative center cut-off').value()
            border_cutoff = self.p.param('Advanced parameters').param('Strip orientation correction search rectangles').param(
                'Relative border cut-off').value()

            left_rect, right_rect = get_rectangles_from_image_and_rectangle_props(self.scene_strip.image.image.shape,
                                                                                  rectangle_props=(height_fact,
                                                                                                   center_cutoff,
                                                                                                   border_cutoff))

            if currentHoughRect is None:
                # Create a CompositeRect
                currentHoughRect = CompositeRect(QRectF(left_rect[0], left_rect[1], left_rect[2], left_rect[3]),
                                                 QRectF(right_rect[0], right_rect[1], right_rect[2], right_rect[3]))

                # Add the CompositeRect to the Scene. Note that the RectItem is
                # not a QGraphicsItem itself and cannot be added to the Scene directly.
                currentHoughRect.addToScene(self.scene_strip)

                # Store the polygon
                self.bookKeeper.addHoughRect(currentHoughRect)
            else:
                self.scene_strip.removeHoughRect()
                currentHoughRect.updateRect(QRectF(left_rect[0], left_rect[1], left_rect[2], left_rect[3]),
                                                 QRectF(right_rect[0], right_rect[1], right_rect[2], right_rect[3]))
                currentHoughRect.addToScene(self.scene_strip)
        else:
            self.scene_strip.removeHoughRect()

    @pyqtSlot(str, Path, dict, name="on_generate_labels")
    def on_generate_labels(self, path1, path2, d):
        self.print_to_console('QR labels creation started.')
        worker = Worker(self.run_get_qr_codes, path1, path2, d)
        worker.signals.finished.connect(self.on_done_labels)
        self.threadpool.start(worker)

    def on_done_labels(self):
        self.print_to_console('QR labels creation completed.')

    @pyqtSlot(dict, name="on_split_images")
    def on_split_images(self, args):
        self.print_to_console('Image splitting started.')
        worker = Worker(self.run_split_images, args)
        worker.signals.finished.connect(self.on_done_split_images)
        self.threadpool.start(worker)

    def on_done_split_images(self):
        self.print_to_console('Image splitting completed.')

    @pyqtSlot(int, name="on_signal_line_length")
    def on_signal_line_length(self, length):
        self.p.param('Basic parameters').param('QR code border').setValue(length)

    @pyqtSlot(list, name="on_signal_rel_bar_pos")
    def on_signal_rel_bar_pos(self, rel_pos):
        self.relative_bar_positions = rel_pos

        # Process all bands
        bands = self.p.param('Basic parameters').param('Band expected relative location').children()
        for i, band in enumerate(bands):
            band.setValue(rel_pos[i])

    @pyqtSlot(list, name="on_signal_sensor_attributes")
    def on_signal_sensor_attributes(self, sensor_attributes):
        self.sensor_attributes = sensor_attributes
        self.p.param('Basic parameters').param('Sensor center').param('x').setValue(sensor_attributes[0])
        self.p.param('Basic parameters').param('Sensor center').param('y').setValue(sensor_attributes[1])
        self.p.param('Basic parameters').param('Sensor size').param('width').setValue(sensor_attributes[2])
        self.p.param('Basic parameters').param('Sensor size').param('height').setValue(sensor_attributes[3])
        self.p.param('Advanced parameters').param('Sensor search area').param('x').setValue(sensor_attributes[4])
        self.p.param('Advanced parameters').param('Sensor search area').param('y').setValue(sensor_attributes[5])

    @pyqtSlot(int, name="on_signal_scene_nr")
    def on_signal_scene_nr(self, nr):
        self.current_scene = nr
        if nr == 1:
            self.view.setBackgroundBrush(QBrush(QColor(232, 255, 238, 180), Qt.SolidPattern))
            self.view_strip.setBackgroundBrush(QBrush(Qt.white, Qt.SolidPattern))
        else:
            self.view_strip.setBackgroundBrush(QBrush(QColor(232, 255, 238, 180), Qt.SolidPattern))
            self.view.setBackgroundBrush(QBrush(Qt.white, Qt.SolidPattern))

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
                # self.set_sensor_and_strip_parameter()
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

        elif self.is_draw_item is 2:
            currentLine = self.bookKeeper.getCurrentLine()
            if currentLine is None:
                # Create a CompositeLine
                currentLine = CompositeLine(QPointF(x, y))

                # Add the CompositeLine to the Scene. Note that the CompositeLine is
                # not a QGraphicsItem itself and cannot be added to the Scene directly.
                currentLine.addToScene(self.scene)

                # Store the line
                self.bookKeeper.addLine(currentLine)

                # Deactivate drawing mode
                self.is_draw_item = -1

        else:
            pass

    def add_sensor_at_position(self):
        """
        Add the sensor to the scene if a config is loaded
        :return:
        """
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
            settings = self.get_parameters()
            sensor_center = settings['sensor_center']
            sensor_size = settings['sensor_size']

            # Build the polygon
            vx0 = sensor_center[1] - sensor_size[1] / 2.0
            vx = sensor_center[1] + sensor_size[1] / 2.0
            vy0 = sensor_center[0] - sensor_size[0] / 2.0
            vy = sensor_center[0] + sensor_size[0] / 2.0
            vertex_x = [vx0, vx, vx, vx0]
            vertex_y = [vy0, vy0, vy, vy]
            for i in range(0, 4):
                currentSensorPolygon.addVertex(QPoint(vertex_x[i], vertex_y[i]))
            currentSensorPolygon.addLine(settings['peak_expected_relative_location'])
        else:
            pass

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

    def run_get_qr_codes(self, label_dir, qrdecode_result_dir_str, d, progress_callback):

        try:

            label_dir = Path(label_dir)
            if label_dir.suffix in ['.xls', '.xlsx']:
                # We need to convert the excel file / template to a csv file
                dd = pd.read_excel(label_dir)
                header = ['sample_id', 'manufacturer', 'plate', 'well', 'row', 'col', 'userdata']
                if set(header).issubset(dd.columns):
                    dd['label'] = dd['sample_id'].astype(str) + '-' + dd['manufacturer'].astype(str) + '-' + 'Plate ' + \
                                  dd['plate'].astype(str).str.zfill(2) + '-' + 'Well ' + dd['row'].astype(str) + ' ' + \
                                  dd['col'].astype(str).str.zfill(2) + '-' + dd['userdata'].astype(str)
                    data = dd['label']
                    # Save the converted file
                    data.to_csv(str(Path(label_dir.parent / label_dir.stem).with_suffix('.csv')), header=False,
                                index=False, index_label=False)
                    data = pd.DataFrame(data)
                else:
                    print(
                        f'Missing column name {list(set(dd.columns).difference(header))}. '
                        f'Make sure the following column names exist {header}')
            elif label_dir.suffix == '.csv':
                # We can start right away
                data = pd.read_csv(label_dir, header=None, sep=';')
            else:
                # Unknown file
                return

            qrdecode_result_dir_str.mkdir(exist_ok=True)
            label_paths = []
            # Create an page with custom settings
            specs = labels.Specification(d["sheet_width"], d["sheet_height"], d["columns"], d["rows"], d["label_width"],
                                         d["label_height"], corner_radius=d["corner_radius"],
                                         left_padding=d["left_padding"], top_padding=d["top_padding"],
                                         bottom_padding=d["bottom_padding"], right_padding=d["right_padding"],
                                         padding_radius=d["padding_radius"], left_margin=d["left_margin"],
                                         right_margin=d["right_margin"], top_margin=d["top_margin"],
                                         bottom_margin=d["bottom_margin"])

            def draw_label(label, width, height, obj):
                # drawing = Image(0, 0, 300, 150, obj)
                # scaled_drawing = self.scale(drawing, scaling_factor=0.4)
                label.add(obj)

            # Create the sheet.
            sheet = labels.Sheet(specs, draw_label, border=True)

            # Read the label template
            for i in tqdm(range(len(data))):
                save_name_qr = qrdecode_result_dir_str.joinpath('qr', data.iloc[i, 0] + 'qr.svg')
                qr_path = qrdecode_result_dir_str.joinpath('qr')
                qr_path.mkdir(exist_ok=True)

                # Create qr code
                qr = pyqrcode.create(data.iloc[i, 0])

                # # Save qr codes
                # # qr.png(str(save_name), scale=3, quiet_zone=6)
                qr.svg(str(save_name_qr), scale=3, quiet_zone=6)

                # Add human readable information to the label
                value_string = data.iloc[i, 0].split('-')
                d = draw.Drawing(300, 150)
                d.append(draw.Text(value_string[0], 20, 150, 105, **{"font-family": "Aria, sans-serif"}))
                d.append(draw.Text(value_string[1], 18, 150, 80, **{"font-family": "Aria, sans-serif"}))
                d.append(draw.Text(value_string[2], 22, 150, 55, **{"font-family": "Aria, sans-serif"}))
                d.append(draw.Text(value_string[3], 22, 150, 30, **{"font-family": "Aria, sans-serif"}))
                d.append(draw.Text(value_string[4], 18, 150, 5, **{"font-family": "Aria, sans-serif"}))
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
            pdf_file = str(Path(qrdecode_result_dir_str / label_dir.stem).with_suffix('.pdf'))
            sheet.save(pdf_file)

            # Print summary information to the console
            self.print_to_console(f"QR labels written to {pdf_file}.")

        except Exception as e:
            print(e)

    def run_split_images(self, args, progress_callback):
        # Input and output dirs
        input_folder_path = Path(args['input_folder'])
        output_folder_path = Path(args['output_folder'])

        output_folder_path.mkdir(parents=True, exist_ok=True)

        undefined_path = Path(args['undefined_folder'])
        undefined_path.mkdir(parents=True, exist_ok=True)

        # Get the list of all files
        filenames = sorted(os.listdir(str(input_folder_path)))

        # Get quantification results
        run_pool(filenames, input_folder_path, output_folder_path, undefined_path, args['max_workers'], args['types'])

    @staticmethod
    def get_file_url(url):
        if platform.system() == "Darwin":
            file_url = "file:///" + url
        else:
            file_url = url
        return file_url

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
        reply = QMessageBox.question(self, 'pyPOCQuant :: Quit',
                                           quit_msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:

            # If open, close all child dialogs

            if self.qi is not None:
                self.qi.close()

            if self.label_gen is not None:
                self.label_gen.close()

            if self.split_images is not None:
                self.split_images.close()

            event.accept()
        else:
            event.ignore()
