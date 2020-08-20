from fbs_runtime.application_context.PyQt5 import ApplicationContext, cached_property

import sys
import shutil
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox
from ui.mainwindow import MainWindow


class AppContext(ApplicationContext):

    def run(self):

        cmd = 'tesseract'
        if self.cmd_exists(cmd):
            self.window.resize(1340, 680)
            self.window.show()
            # self.window.strip_action.setIcon(QIcon(self.get_resource("img/strip-01.png")))
            self.window.action_load_settings_file.setIcon(QIcon(self.get_resource("img/strip-01.png")))
            self.window.action_save_settings_file.setIcon(QIcon(self.get_resource("img/strip-01.png")))
            self.window.sensor_action.setIcon(QIcon(self.get_resource("img/sensor-01.png")))
            self.window.delete_items_action.setIcon(QIcon(self.get_resource("img/sensor_trash-01.png")))
            self.window.mirror_v_action.setIcon(QIcon(self.get_resource("img/mirror_v-01.png")))
            self.window.mirror_h_action.setIcon(QIcon(self.get_resource("img/mirror_h-01.png")))
            self.window.rotate_cw_action.setIcon(QIcon(self.get_resource("img/rotate_cw-01.png")))
            self.window.rotate_ccw_action.setIcon(QIcon(self.get_resource("img/rotate_ccw-01.png")))
            self.window.zoom_out_action.setIcon(QIcon(self.get_resource("img/zoom_out-01.png")))
            self.window.zoom_in_action.setIcon(QIcon(self.get_resource("img/zoom_in-01.png")))
            self.window.zoom_reset_action.setIcon(QIcon(self.get_resource("img/zoom_reset-01.png")))
            self.window.action_console.setIcon(QIcon(self.get_resource("img/log-01.png")))
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
        return MainWindow(ui, splash, splash2)

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