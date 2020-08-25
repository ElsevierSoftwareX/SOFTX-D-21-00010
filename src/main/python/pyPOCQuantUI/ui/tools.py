from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtCore import pyqtSignal
from pathlib import Path


class LabelGen(QDialog):

    signal_run_label = pyqtSignal(str, Path, name='signal_run_label')

    def __init__(self, *args, **kwargs):
        super(LabelGen, self).__init__(*args, **kwargs)

        self.setWindowTitle("pyPOCQuant :: Generate labels")

        self.label_dir = None
        self.qrdecode_result_dir_str = None
        self.label_paths = []
        self.label_dir_btn = QPushButton('Select label template file')
        self.label_dir_btn.clicked.connect(self.get_dir)
        self.run_btn = QPushButton('Run label generation')
        self.run_btn.clicked.connect(self.emit_data)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label_dir_btn)
        self.layout.addWidget(self.run_btn)
        self.setLayout(self.layout)

    def get_dir(self):
        try:
            self.label_dir = QFileDialog.getOpenFileName(self, "Select Directory", "")[0]
            self.qrdecode_result_dir_str = Path(self.label_dir)
            self.qrdecode_result_dir_str = Path(self.qrdecode_result_dir_str.parent / str(
                self.qrdecode_result_dir_str.stem + "_QRCodes"))
            self.qrdecode_result_dir_str.mkdir(exist_ok=True)
            print(self.qrdecode_result_dir_str)

        except Exception as e:
            QMessageBox.information(self, "Error!", "No valid directory selected".format(e))

    def emit_data(self):
        print(type(self.label_dir), type(self.qrdecode_result_dir_str))
        self.signal_run_label.emit(self.label_dir, self.qrdecode_result_dir_str)


