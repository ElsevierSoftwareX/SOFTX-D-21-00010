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

from PyQt5.QtCore import QObject, pyqtSignal


class Stream(QObject):
    """
    Implementation of a stream to handle logging messages
    """

    stream_signal = pyqtSignal(str)
    """
        pyqtsignal to redirect sterr
    """

    def write(self, text: object) -> object:
        """
            Emits text formatted as string.
            Args:
                text (`str`) Text sent to sterr to be rerouted to the console in the ui.
        """
        self.stream_signal.emit(str(text))