from fbs_runtime.application_context.PyQt5 import ApplicationContext, cached_property

import sys
import shutil
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox
from ui.mainwindow import MainWindow
from ui.tools import LabelGen, SplitImages


class AppContext(ApplicationContext):

    def run(self):

        cmd = 'tesseract'
        if self.cmd_exists(cmd):
            self.window.resize(1340, 680)
            self.window.show()
            # self.window.strip_action.setIcon(QIcon(self.get_resource("img/strip-01.png")))
            self.window.action_load_settings_file.setIcon(QIcon(self.get_resource("img/load-01.png")))
            self.window.action_save_settings_file.setIcon(QIcon(self.get_resource("img/save-01.png")))
            self.window.sensor_action.setIcon(QIcon(self.get_resource("img/sensor-01.png")))
            self.window.delete_items_action.setIcon(QIcon(self.get_resource("img/sensor_trash-01.png")))
            self.window.mirror_v_action.setIcon(QIcon(self.get_resource("img/mirror_v-01.png")))
            self.window.mirror_h_action.setIcon(QIcon(self.get_resource("img/mirror_h-01.png")))
            self.window.rotate_cw_action.setIcon(QIcon(self.get_resource("img/rotate_cw-01.png")))
            self.window.rotate_ccw_action.setIcon(QIcon(self.get_resource("img/rotate_ccw-01.png")))
            self.window.zoom_out_action.setIcon(QIcon(self.get_resource("img/zoom_out-01.png")))
            self.window.zoom_in_action.setIcon(QIcon(self.get_resource("img/zoom_in-01.png")))
            self.window.zoom_reset_action.setIcon(QIcon(self.get_resource("img/zoom_reset-01.png")))
            self.window.width_action.setIcon(QIcon(self.get_resource("img/width-01.png")))
            self.window.action_console.setIcon(QIcon(self.get_resource("img/log-01.png")))
            self.window.user_instructions_path = self.get_resource("UserInstructions.html")
            self.window.quick_start_path = self.get_resource("QuickStart.html")
            self.window.poct_template_path = self.get_resource("pyPOCQuantTemplate.pdf")
            self.window.qr_labels_template_path = self.get_resource("sample_qr_code_template.xls")
            # Setup label form generator
            label_gen_ui = self.get_resource("label_form.ui")
            self.window.label_gen = LabelGen(label_gen_ui)
            self.window.label_gen.signal_run_label.connect(self.window.on_generate_labels)
            self.window.action_gen_qr_labels.triggered.connect(self.window.label_gen.show)
            # Setup split images from
            split_images_ui = self.get_resource("split_form.ui")
            self.window.split_images = SplitImages(split_images_ui)
            self.window.split_images.signal_run_split_images.connect(self.window.on_split_images)
            self.window.action_split_images_by_type.triggered.connect(self.window.split_images.show)
        else:
            self.show_tesseract_install_dialog()
        return appctxt.app.exec_()

    def get_design(self):
        qtCreatorFile = self.get_resource("form.ui")
        return qtCreatorFile

    def cmd_exists(self, cmd):
        return shutil.which(cmd) is not None

    @cached_property
    def window(self):
        ui = self.get_design()
        splash = self.get_resource("img/pyPOCQuantSplash-01.png")
        splash2 = self.get_resource("img/pyPOCQuantSplash-02.png")
        icon = self.get_resource('img/icon.png')
        return MainWindow(ui, splash, splash2, icon)

    def show_tesseract_install_dialog(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setTextFormat(Qt.RichText);
        msg.setText("The tool requires the software TESSERACT to be installed.")
        msg.setInformativeText("TESSERACT has not been detected please follow this link "
                               "(<a href='https://tesseract-ocr.github.io/tessdoc/Home.html'>Install TESSERACT</a>) "
                               "to install TESSERACT first and make sure it is on the system path. ")
        msg.setWindowTitle("pyPOCQuantUI :: Warning")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        retval = msg.exec_()
        sys.exit(retval)


if __name__ == "__main__":
    appctxt = AppContext()
    stylesheet = appctxt.get_resource('styles.qss')
    appctxt.app.setStyleSheet(open(stylesheet).read())
    exit_code = appctxt.run()
    sys.exit(exit_code)