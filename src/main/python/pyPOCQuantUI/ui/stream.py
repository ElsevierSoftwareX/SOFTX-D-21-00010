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