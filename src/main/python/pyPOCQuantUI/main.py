from fbs_runtime.application_context.PyQt5 import ApplicationContext, cached_property

import sys
from PyQt5.QtGui import QIcon
from ui.mainwindow import MainWindow


class AppContext(ApplicationContext):
    def run(self):
        self.window.resize(1340, 680)
        self.window.show()
        # self.window.strip_action.setIcon(QIcon(self.get_resource("img/strip-01.png")))
        self.window.sensor_action.setIcon(QIcon(self.get_resource("img/sensor-01.png")))
        self.window.delete_items_action.setIcon(QIcon(self.get_resource("img/sensor_trash-01.png")))
        self.window.mirror_v_action.setIcon(QIcon(self.get_resource("img/mirror_v-01.png")))
        self.window.mirror_h_action.setIcon(QIcon(self.get_resource("img/mirror_h-01.png")))
        self.window.rotate_cw_action.setIcon(QIcon(self.get_resource("img/rotate_cw-01.png")))
        self.window.rotate_ccw_action.setIcon(QIcon(self.get_resource("img/rotate_ccw-01.png")))
        self.window.zoom_out_action.setIcon(QIcon(self.get_resource("img/zoom_out-01.png")))
        self.window.zoom_in_action.setIcon(QIcon(self.get_resource("img/zoom_in-01.png")))
        return appctxt.app.exec_()

    def get_design(self):
        qtCreatorFile = self.get_resource("form.ui")
        return qtCreatorFile

    @cached_property
    def window(self):
        ui = self.get_design()
        splash = self.get_resource("img/pyPOCQuantSplash-01.png")
        return MainWindow(ui, splash)


if __name__ == "__main__":
    appctxt = AppContext()
    stylesheet = appctxt.get_resource('styles.qss')
    appctxt.app.setStyleSheet(open(stylesheet).read())
    exit_code = appctxt.run()
    sys.exit(exit_code)