from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtCore import pyqtSignal
from PyQt5 import uic
from pathlib import Path
import multiprocessing
from pypocquant.lib.consts import KnownManufacturers


class LabelGen(QDialog):

    signal_run_label = pyqtSignal(str, Path, dict, name='signal_run_label')

    def __init__(self, ui, *args, **kwargs):
        super(LabelGen, self).__init__(*args, **kwargs)

        uic.loadUi(ui, self)
        self.setWindowTitle("pyPOCQuant :: Generate labels")

        self.qrdecode_result_dir_str = None

        self.label_btn.clicked.connect(self.get_dir)
        self.label_run_btn.clicked.connect(self.emit_data)
        self.label_dir_edit.textChanged.connect(self.on_label_dir_change)

    def on_label_dir_change(self):
        new_path = self.label_dir_edit.text()
        # Validate if path exists
        if Path(new_path).is_dir():
            self.label_dir_edit.setText(new_path)
            self.on_update_res_dir()
        elif Path(new_path).is_file():
            self.on_update_res_dir()

    def on_update_res_dir(self):
        self.qrdecode_result_dir_str = Path(self.label_dir_edit.text())
        self.qrdecode_result_dir_str = Path(self.qrdecode_result_dir_str.parent / str(
            self.qrdecode_result_dir_str.stem + "_qr_labels"))

    def get_dir(self):
        try:
            self.label_dir_edit.setText(QFileDialog.getOpenFileName(self, "Select qr label template", "")[0])
        except Exception as e:
            QMessageBox.information(self, "Error!", "No valid file selected".format(e))

    def get_data(self):

        d = {}
        for key in ["sheet_width", "sheet_height", "columns", "rows", "label_width", "label_height", "top_margin",
                    "bottom_margin", "left_margin", "right_margin", "left_padding", "right_padding", "top_padding",
                    "bottom_padding", "corner_radius", "padding_radius"]:
            if key not in d:
                prop = getattr(self, key)
                d[key] = prop.value()
        return d

    def emit_data(self):
        d = self.get_data()
        self.signal_run_label.emit(self.label_dir_edit.text(), self.qrdecode_result_dir_str, d)


class SplitImages(QDialog):

    signal_run_split_images = pyqtSignal(dict, name='signal_run_split_images')

    def __init__(self, ui, *args, **kwargs):
        super(SplitImages, self).__init__(*args, **kwargs)

        uic.loadUi(ui, self)
        self.setWindowTitle("pyPOCQuant :: Split images by type")

        self.undefined_dir = None
        self.worker_spin.setRange(1, multiprocessing.cpu_count())
        self.type_edit.setPlainText(', '.join(KnownManufacturers))

        self.input_btn.clicked.connect(self.get_dir)
        self.output_btn.clicked.connect(self.get_dir)
        self.type_btn.clicked.connect(self.on_open_file)
        self.run_btn.clicked.connect(self.emit_data)
        self.input_edit.textChanged.connect(self.on_dir_change)
        self.output_edit.textChanged.connect(self.on_dir_change)
        self.type_edit.textChanged.connect(self.on_dir_change)

    def on_dir_change(self):
        sender = self.sender()
        sender_name = str(sender.objectName())

        if sender_name == 'input_edit':
            new_path = self.input_edit.text()
            # Validate if path exists
            if Path(new_path).is_dir():
                self.input_edit.setText(new_path)
                self.output_edit.setText(str(Path(Path(new_path) / 'sorted')))
                self.undefined_dir = str(Path(Path(self.output_edit.text()) / 'UNDEFINED'))
        elif sender_name == 'output_edit':
            new_path = self.output_edit.text()
            # Validate if path exists
            if Path(new_path).is_dir():
                self.output_edit.setText(str(new_path))
                self.undefined_dir = str(Path(Path(self.output_edit.text()) / 'UNDEFINED'))
        # elif sender_name == 'type_edit':
        #     new_path = self.type_edit.toPlainText()
        #     # Validate if text is capital
        #     self.type_edit.setPlainText(new_path)

    def get_dir(self):
        sending_button = self.sender()
        sender = str(sending_button.objectName())
        try:
            dir = QFileDialog.getExistingDirectory(self, "Select Directory", "")
            if sender == 'input_btn':
                self.input_edit.setText(dir)
            if sender == 'output_btn':
                self.output_edit.setText(dir)
            # if sender == 'type_btn':
            #     self.type_edit.setPlainText(dir)
        except Exception as e:
            QMessageBox.information(self, "Error!", "No valid directory selected".format(e))

    def on_open_file(self):
        try:
            # Read file,
            # Validate content
            # Add to type_list
            # self.type_edit.setPlainText(QFileDialog.getOpenFileName(self, "Select file", "")[0])
            print('to be implemented')
        except Exception as e:
            QMessageBox.information(self, "Error!", "No valid file selected".format(e))

    def emit_data(self):
        args = {'input_folder': self.input_edit.text(), 'output_folder': self.output_edit.text(),
                'undefined_folder': self.undefined_dir, 'max_workers': self.worker_spin.value(),
                'types': self.type_edit.toPlainText()}
        self.signal_run_split_images.emit(args)


