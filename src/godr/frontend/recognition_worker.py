import os

import cv2
import fitz
import numpy as np
import pkg_resources
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QObject, QThread

from godr.backend.board import Board
from godr.backend.recognizer import Recognizer


def QImageToCvMat(incomingImage):
    incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format_RGB888)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.constBits()
    arr = np.array(ptr).reshape(height, width, 3)  # Copies the data?
    return arr


class RecognitionWorker(QThread):
    update_ui = QtCore.pyqtSignal(int, str)   # (percent, append_to_console)
    send_board = QtCore.pyqtSignal(str, np.ndarray)  # (path_to_sgf, img)
    done = QtCore.pyqtSignal()

    def __init__(self, files, ranges):
        super(QObject, self).__init__()
        self.files = files
        self.ranges = ranges
        self.progress = [ [0, len(files)] ]

    def calc_progress(self):
        fraction = 0.0
        for subprogress in self.progress[::-1]:
            fraction = (subprogress[0] + fraction) / subprogress[1]
            if fraction > 1:
                print('WARN RecognitionWorker: fraction > 1')
                fraction = 1
        return int(100 * fraction)

    def send_update(self, line):
        """ Notifies MainWindow to update progress bar and append line to the log """
        percent = self.calc_progress()
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
                self.progress[0][0] += 1
                self.send_update('')  # Do not log skipped files
                continue

            self.send_update('Processing {}'.format(img_file))

            # Create dir
            os.makedirs(result_dir, exist_ok=True)

            try:
                if extension == '.pdf':
                    self.parse_pdf(img_file, result_dir, start=dlg_range[0], end=dlg_range[1])
                else:
                    with open(img_file, "rb") as f:
                        chunk = f.read()
                        img_bytes = np.frombuffer(chunk, dtype=np.uint8)
                        self.parse_img(img_bytes, result_dir)
                output = "File proceed"

            except Exception as ex:
                output = 'An error occurred <<{}>>'.format(str(ex))

            self.progress[0][0] += 1
            self.send_update(output + '\n')

        self.done.emit()

    def parse_img(self, img_bytes, path_dir, prefix=''):
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        rec = Recognizer()
        boards_img = rec.split_into_boards(img)

        total_boards = len(boards_img)
        subprocess_level = len(self.progress)
        self.progress.append([1, total_boards + 1])
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

                self.progress[subprocess_level][0] = i + 1
                self.send_update('> Board {} saved'.format(i))

            except Exception as e:
                self.send_board.emit("", board_img)
                self.progress[subprocess_level][0] = i + 1
                self.send_update('> An error occurred while processing board {}'.format(i))
            i += 1

        self.progress.pop()
        assert len(self.progress) == subprocess_level
        return paths, boards_img

    def parse_pdf(self, path, result_dir, **kwargs):
        doc = fitz.Document(path)

        start_page = kwargs['start'] if 'start' in kwargs else  0
        end_page = kwargs['end'] if 'end' in kwargs else doc.page_count - 1

        subprocess_level = len(self.progress)
        self.progress.append([0, end_page - start_page])

        for page in doc.pages(start_page, end_page, 1):
            self.send_update('Rendering {}-th page of PDF'.format(str(page.number + 1)))

            try:
                # page.get_pixmap().writePNG('test.png')
                png = page.get_pixmap().getPNGData()
                png = np.frombuffer(png, dtype=np.int8)
                self.parse_img(png, result_dir, 'page-{}-'.format(str(page.number + 1)))
            except:
                pass
            self.progress[subprocess_level][0] += 1

        self.progress.pop()
        assert len(self.progress) == subprocess_level


class CustomDialog(QtWidgets.QDialog):
    def __init__(self, path_pdf, page_cnt, parent=None):
        super().__init__(parent=parent)

        form_class, _ = uic.loadUiType(pkg_resources.resource_stream('godr.frontend.ui', "select_pages_pdf.ui"))
        form_class().setupUi(self)

        self.start_page = self.findChild(QtWidgets.QSpinBox, 'start_page')
        self.end_page = self.findChild(QtWidgets.QSpinBox, 'end_page')

        self.findChild(QtWidgets.QLabel, 'path').setText(path_pdf)

        for box in (self.start_page, self.end_page):
            box.setRange(1, page_cnt)
        self.start_page.setValue(1)
        self.end_page.setValue(page_cnt)

    def get_range(self):
        return self.start_page.value(), self.end_page.value()
