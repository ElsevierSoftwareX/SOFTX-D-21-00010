from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtCore import pyqtSignal
from PyQt5 import uic
from pathlib import Path


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
            self.label_dir_edit.setText(QFileDialog.getOpenFileName(self, "Select Directory", "")[0])
        except Exception as e:
            QMessageBox.information(self, "Error!", "No valid directory selected".format(e))

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


