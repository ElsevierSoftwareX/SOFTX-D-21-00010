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

from PyQt5.QtCore import Qt, QMetaObject, Q_ARG, pyqtSlot
from PyQt5.QtWidgets import QPlainTextEdit
from PyQt5.QtGui import QTextCursor


class LogTextEdit(QPlainTextEdit):
    """
    Adopted from
    https://stackoverflow.com/questions/53381975/display-terminal-output-with-tqdm-in-qplaintextedit
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.flag = False

    def write(self, message):
        if not hasattr(self, "flag"):
            self.flag = False
        message = message.replace('\r', '').rstrip()
        if message:
            method = "replace_last_line" if self.flag else "appendPlainText"
            QMetaObject.invokeMethod(self, method, Qt.QueuedConnection, Q_ARG(str, message))
            self.flag = True
        else:
            self.flag = False

    @pyqtSlot(str)
    def replace_last_line(self, text):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.select(QTextCursor.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.insertBlock()
        self.setTextCursor(cursor)
        self.insertPlainText(text)
