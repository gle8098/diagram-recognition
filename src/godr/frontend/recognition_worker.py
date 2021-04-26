import os

import cv2
import fitz
import numpy as np
import pkg_resources
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QObject, QThread
from pagerange import PageRange

from godr.backend.board import Board
from godr.backend.recognizer import Recognizer


def qt_image_to_cv_mat(qt_image):
    qt_image = qt_image.convertToFormat(QtGui.QImage.Format_RGB888)

    width = qt_image.width()
    height = qt_image.height()

    ptr = qt_image.constBits()
    arr = np.array(ptr).reshape((height, width, 3))  # Copies the data?
    return arr


class Progress:
    """
    This class is used to calculate progress on nested tasks. Here is an example:

    * <layer 0> recognize each of selected files
        * <layer 1> current file is pdf, recognize each page
            * <layer 2> several boards found on page, recognize each

    As you can see, recognition is a nested task. To properly calculate global progress, each simple task makes
    a layer for itself. Layer contains a list of two elements, ``[current, total]``, which means that the job has
    been done for ``current / total * 100`` percent.
    """
    def __init__(self, init_range=None):
        self._progress = []
        if init_range:
            self.add_layer(init_range)

    def __warn(self, msg):
        print('WARN RecognitionWorker.Progress: {}'.format(msg))

    def add_layer(self, prange):
        assert len(prange) == 2 and prange[0] <= prange[1]
        self._progress.append(list(prange))
        return len(self._progress) - 1

    def append_progress(self, layer, delta):
        assert layer < len(self._progress)
        self.set_progress(layer, self._progress[layer][0] + delta)

    def set_progress(self, layer, value):
        assert layer < len(self._progress)
        if not (0 <= value <= self._progress[layer][1]):
            self.__warn('setting progress > 1')
        self._progress[layer][0] = value

    def pop_layer(self, layer):
        assert layer < len(self._progress)
        if layer + 1 < len(self._progress):
            self.__warn('pop layer with its child')
        self._progress = self._progress[:layer]

    def calc(self):
        fraction = 0.0
        for subprogress in self._progress[::-1]:
            fraction = (subprogress[0] + fraction) / subprogress[1]
            if fraction > 1:
                self.__warn('fraction > 1')
                fraction = 1
        return int(100 * fraction)


class RecognitionWorker(QThread):
    update_ui = QtCore.pyqtSignal(int, str)   # (percent, append_to_console)
    send_board = QtCore.pyqtSignal(str, np.ndarray)  # (path_to_sgf, img)
    done = QtCore.pyqtSignal()

    def __init__(self, files, ranges):
        super(QObject, self).__init__()
        self.files = files
        self.ranges = ranges
        self.progress = Progress((0, len(files)))

    def send_update(self, line):
        """ Notifies MainWindow to update progress bar and append line to the log """
        percent = self.progress.calc()
        self.update_ui.emit(percent, line)

    def run(self):
        if not self.files:
            self.update_ui.emit(0, 'No files selected')
            self.done.emit()
            return

        print('Recognising {} files'.format(len(self.files)))

        for img_file, dlg_range in zip(self.files, self.ranges):
            result_dir, extension = os.path.splitext(img_file)
            if extension not in ['.png', '.jpg', '.jpeg', '.pdf']:
                self.progress.append_progress(0, 1)
                self.send_update('')  # Do not log skipped files
                continue

            self.send_update('Processing {}'.format(img_file))

            # Create dir
            os.makedirs(result_dir, exist_ok=True)

            try:
                if extension == '.pdf':
                    self.parse_pdf(img_file, result_dir, range=dlg_range)
                else:
                    with open(img_file, "rb") as f:
                        chunk = f.read()
                        img_bytes = np.frombuffer(chunk, dtype=np.uint8)
                        self.parse_img(img_bytes, result_dir)
                output = "File processed"

            except Exception as ex:
                output = 'An error occurred <<{}>>'.format(str(ex))

            self.progress.append_progress(0, 1)
            self.send_update(output + '\n')

        self.done.emit()

    def parse_img(self, img_bytes, path_dir, prefix=''):
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        rec = Recognizer()
        boards_img = rec.split_into_boards(img)

        total_boards = len(boards_img)
        p_layer = self.progress.add_layer((1, total_boards + 1))
        self.send_update('Found {} board(s)'.format(total_boards))

        i = 1
        paths = []
        for board_img in boards_img:
            try:
                board = Board(board_img)
                sgf_file = '{}board-{}.sgf'.format(prefix, str(i))
                path = os.path.join(path_dir, sgf_file)
                board.save_sgf(path)
                paths.append(path)
                self.send_board.emit(path, board_img)

                self.progress.set_progress(p_layer, i + 1)
                self.send_update('> Board {} saved'.format(i))

            except Exception as e:
                self.send_board.emit("", board_img)
                self.progress.set_progress(p_layer, i + 1)
                self.send_update('> An error occurred while processing board {}. <<{}>>'.format(i, str(e)))
            i += 1

        self.progress.pop_layer(p_layer)
        return paths, boards_img

    def parse_pdf(self, path, result_dir, **kwargs):
        doc = fitz.Document(path)
        range = kwargs['range']
        pages = range.pages
        p_layer = self.progress.add_layer((0, len(pages)))

        for page_number in pages:
            page = doc.load_page(page_number - 1)
            self.send_update('Rendering {}-th page of PDF'.format(str(page.number + 1)))

            try:
                # page.get_pixmap().writePNG('test.png')
                png = page.get_pixmap().getPNGData()
                png = np.frombuffer(png, dtype=np.int8)
                self.parse_img(png, result_dir, 'page-{}-'.format(str(page.number + 1)))
            except:
                pass
            self.progress.append_progress(p_layer, 1)

        self.progress.pop_layer(p_layer)


class SelectPageRangeDialog(QtWidgets.QDialog):
    def __init__(self, path_pdf, page_cnt, parent=None):
        super().__init__(parent=parent)

        form_class, _ = uic.loadUiType(pkg_resources.resource_stream('godr.frontend.ui', "select_pages_pdf.ui"))
        form_class().setupUi(self)

        self.findChild(QtWidgets.QLabel, 'path').setText(path_pdf)
        self.page_cnt = page_cnt
        self.page_range_widget = self.findChild(QtWidgets.QLineEdit, 'page_range')

    def get_range(self):
        text = self.page_range_widget.text()
        try:
            page_range = PageRange(text)
        except ValueError:
            page_range = PageRange("{}-{}".format(1, self.page_cnt))
        return page_range
