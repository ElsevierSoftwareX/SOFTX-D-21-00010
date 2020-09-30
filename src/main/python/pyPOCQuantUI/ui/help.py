# /********************************************************************************
# * Copyright © 2020, ETH Zurich, D-BSSE, Andreas P. Cuny
# * All rights reserved. This program and the accompanying materials
# * are made available under the terms of the GNU Public License v3.0
# * which accompanies this distribution, and is available at
# * http://www.gnu.org/licenses/gpl
# *
# * Contributors:
# *     Andreas P. Cuny - initial API and implementation
# *******************************************************************************/

from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QApplication, QStyle, \
    QTextBrowser, QWidget, QGridLayout, QTextEdit, QCheckBox, QVBoxLayout, QLabel
from PyQt5.Qt import QFont, QIcon
from PyQt5.QtCore import Qt
import platform


__author__ = 'Andreas P. Cuny'

__version__ = '0.0.2'

__operating_system__ = '{} {}'.format(platform.system(), platform.architecture()[0])

from ui import versionInfo


class QuickInstructions(QWidget):
    """
    Implementation of the quick instructions.
    """

    def __init__(self, parent=None):
        super(QuickInstructions, self).__init__(parent)
        # self.setAttribute(Qt.WA_DeleteOnClose) # Deletes instance on window close
        self.setWindowTitle('pyPOCQuant :: Quick instructions')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.display_on_startup = 2
        self.resize(400, 370)
        # self.setWindowIcon(QtGui.QIcon(resource_path(os.path.join(os.path.join("ui", "icons",
        #                                                                        "icon.ico")))))
        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        ok = QPushButton('Ok')
        ok.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogApplyButton))
        ok.setMaximumWidth(100)
        ok.clicked.connect(self.close)
        self.chk = QCheckBox('Display quick instructions at startup')
        self.chk.setFont(QFont('Arial', 9, QFont.Bold))
        self.chk.setChecked(1)
        self.chk.clicked.connect(self.on_display_qi)

        self.quick_instructions = QTextEdit(self)
        self.quick_instructions.setText('<h3>Quick Instructions</h3> '
                                        'Start by selecting the input folder where you have your images stored. '
                                        'Second select an image for display by clicking on the name in the list. '
                                        'Continue by drawing the sensor outline first and then the POCT outline.'
                                        'Test the parameters by hitting the test button.'
                                        '<br><br>Controls:'
                                        '<ul>'
                                        '<li><b>Ctrl+N (Cmd+I):</b> Select the input dir'
                                        '</li>'
                                        '<li><b>Ctrl+O (Cmd+O):</b> Open an existing pyPOCQuant config'
                                        '</li>'
                                        '<li><b>Ctrl+S (Cmd+S):</b> Save the current pyPOCQuant config'
                                        '</li>'
                                        '<li><b>Ctrl+L (Cmd+L):</b> Show/hide pyPOCQuant log'
                                        '</li>'
                                        '</ul>'
                                        'Hints:'
                                        '<ul><li>The image can be rotated, mirrored and zoomed if needed.'
                                        '</li>'
                                        '<li>Start with drawing a rectangle around the sensor (area with the bands)'
                                        'Then fine tune the relative band positions (draggable lines).</li><'
                                        '</ul>'
                                        'You can open this window any time from the Help menu.</ul>')
        self.quick_instructions.setReadOnly(1)
        self.quick_instructions.setFont(QFont('Arial', 9))
        grid.addWidget(self.quick_instructions, 0, 0, 1, 0)
        grid.addWidget(ok, 1, 1)
        grid.addWidget(self.chk, 1, 0)
        self.setLayout(grid)

    def on_display_qi(self):
        self.display_on_startup = self.chk.checkState()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


class About(QWidget):
    """
    Implementation of the about dialog.
    """
    def __init__(self, icon_path=None):
        super(About, self).__init__()
        self.setWindowTitle('pyPOCQuant :: About')
        self.setFixedSize(400, 500)
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.icon_path = icon_path
        about_icon = QIcon()
        about_icon.addPixmap(self.style().standardPixmap(QStyle.SP_FileDialogInfoView))
        self.setWindowIcon(about_icon)

        self.le = QLabel()
        self.build = QLabel("[version: v%s %s]" % (versionInfo.get_version_string(), __operating_system__))
        self.author = QLabel("pyPOCQuant: Point of Care Test\nQuantification tool. \n\nwritten by Andreas P. Cuny "
                             "and Aaron Ponti")
        self.license = QLabel("Licensed under the GPL v3 license.")
        self.copyright = QLabel("\u00a9 Copyright  Andreas P. Cuny and Aaron Ponti \n2020. All rights reserved. \
                                \nCSB Laboratory & SCF @ ETH Zurich "
                                "\nMattenstrasse 26 \n4058 Basel Switzerland ")
        self.dependencies = QTextBrowser()
        self.dependencies.setHtml("The authors appreciate and use the following 3rd parties libraries: <br> \
                                <br>Python v3.6, under the <a href=https://docs.python.org/3/license.html>PSF License</a> \
                                <br>numpy v1.14.5, under the <a href=https://docs.scipy.org/doc/numpy-1.10.0/license.html>BSD 3-Clause License</a> \
                                <br>scipy v1.1.0, under the <a href=https://docs.scipy.org/doc/numpy-1.10.0/license.html>BSD 3-Clause License</a> \
                                <br>pandas v0.23.3, under the <a href=https://pandas.pydata.org/pandas-docs/stable/getting_started/overview.html>BSD 3-Clause License</a> \
                                <br>tqdm v4.23.4, under the <a href=https://github.com/tqdm/tqdm/blob/master/LICENCE>MIT License</a> \
                                <br>PyQT5, under the <a href=https://www.riverbankcomputing.com/static/Docs/PyQt5/introduction.html#license>GPL v3 License</a> \
                                <br>matplotlib, under the <a href=https://matplotlib.org/devel/license.html>PSF License</a>\
                                <br>pyqtgraph, under the <a href=https://github.com/pyqtgraph/pyqtgraph/blob/develop/LICENSE.txt>MIT License</a>\
                                <br>opencv-python v3.4.2.16, under the <a href=https://github.com/skvark/opencv-python/blob/master/LICENSE.txt>MIT License</a>\
                                <br>opencv-contrib-python v3.4.2.16, under the <a href=https://github.com/skvark/opencv-python/blob/master/LICENSE.txt>MIT License</a>\
                                <br>exifread, under the <a href=https://github.com/ianare/exif-py/blob/develop/LICENSE.txt>BSD 3-Clause License</a>\
                                <br>imutils, under the <a href=https://github.com/jrosebr1/imutils/blob/master/LICENSE.txt>MIT License</a>\
                                <br>pytesseract, under the <a href=https://github.com/madmaze/pytesseract/blob/master/LICENSE>Apache 2.0 License</a>\
                                <br>imageio, under the <a href=https://github.com/imageio/imageio/blob/master/LICENSE>BSD 2-Clause License</a>\
                                <br>rawpy, under the <a href=https://github.com/letmaik/rawpy/blob/master/LICENSE>MIT License</a>\
                                <br>pyzbar, under the <a href=https://github.com/NaturalHistoryMuseum/pyzbar/blob/master/LICENSE.txt>MIT License</a>\
                                <br>scikit-learn, under the <a href=https://github.com/scikit-learn/scikit-learn>BSD 3-Clause License</a>\
                                <br>scikit-image, under the <a href=https://github.com/scikit-image/scikit-image>BSD 3-Clause License</a>\
                                <br>nbconvert, under the <a href=https://github.com/jupyter/nbconvert/blob/master/LICENSE>BSD 3-Clause License</a>\
                                <br>jupytext, under the <a href=https://github.com/mwouts/jupytext/blob/master/LICENSE>MIT License</a>")
        self.dependencies.setReadOnly(True)
        self.dependencies.setOpenExternalLinks(True)
        logo = QtGui.QPixmap(self.icon_path)
        logo = logo.scaled(180, 256, Qt.KeepAspectRatio)
        self.le.setPixmap(logo)
        self.le.setAlignment(Qt.AlignCenter)
        self.build.setAlignment(Qt.AlignCenter)

        font_b = QtGui.QFont()
        font_b.setPointSize(9)
        self.build.setFont(font_b)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.author.setFont(font)
        self.copyright.setFont(font)
        self.dependencies.setFont(font_b)
        v_box = QVBoxLayout()
        v_box.addWidget(self.le)
        v_box.addWidget(self.build)
        v_box.addWidget(self.license)
        v_box.addStretch()
        v_box.addWidget(self.author)
        v_box.addStretch()
        v_box.addWidget(self.copyright)
        v_box.addWidget(self.dependencies)
        v_box.addStretch()
        self.setLayout(v_box)

        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
