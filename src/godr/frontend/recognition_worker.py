import logging
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
from godr.frontend.miscellaneous import Progress, translate_plural


def qt_image_to_cv_mat(qt_image):
    qt_image = qt_image.convertToFormat(QtGui.QImage.Format_RGB888)

    width = qt_image.width()
    height = qt_image.height()

    ptr = qt_image.constBits()
    arr = np.array(ptr).reshape((height, width, 3))  # Copies the data?
    return arr


class RecognitionWorker(QThread):
    update_ui = QtCore.pyqtSignal(int, str)  # (percent, append_to_console)
    send_board = QtCore.pyqtSignal(str, np.ndarray, str, str)  # (path_to_sgf, img, path_dir, board_title)
    done = QtCore.pyqtSignal()

    def __init__(self, files, ranges):
        super(QObject, self).__init__()
        self.files = files
        self.ranges = ranges
        self.progress = Progress((0, len(files)))
        self.should_stop = False

    def send_update(self, line):
        """ Notifies MainWindow to update progress bar and append line to the log """
        percent = self.progress.calc()
        self.update_ui.emit(percent, line)

    def stop(self):
        self.should_stop = True

    def run(self):
        if not self.files:
            self.update_ui.emit(0, 'No files selected')
            self.done.emit()
            return

        logging.info('Recognising {} files'.format(len(self.files)))

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
                    self.__parse_pdf(img_file, result_dir, range=dlg_range)
                else:
                    with open(img_file, "rb") as f:
                        chunk = f.read()
                        img_bytes = np.frombuffer(chunk, dtype=np.uint8)
                        self.__parse_img(img_bytes, result_dir)
                output = "File processed"

            except KeyboardInterrupt:
                break
            except Exception as ex:
                output = 'An error occurred <<{}>>'.format(str(ex))

            self.progress.append_progress(0, 1)
            self.send_update(output + '\n')

        self.done.emit()

    def __parse_img(self, img_bytes, path_dir, **kwargs):
        file_prefix = kwargs.get("file_prefix", "")
        board_title_fmt = kwargs.get("board_title_fmt", "Доска {} из {}")

        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        rec = Recognizer()
        boards_img = rec.split_into_boards(img)

        total_boards = len(boards_img)
        p_layer = self.progress.add_layer((1, total_boards + 1))
        self.send_update('Found {} board(s)'.format(total_boards))

        i = 1
        paths = []
        for board_img in boards_img:
            board_title = board_title_fmt.format(i, len(boards_img))
            sgf_file = '{}board-{}.sgf'.format(file_prefix, str(i))
            sgf_path = os.path.join(path_dir, sgf_file)

            try:
                board = Board(board_img)
                board.save_sgf(sgf_path)
                paths.append(sgf_path)

                self.send_board.emit(sgf_path, board_img, path_dir, board_title)
                self.progress.set_progress(p_layer, i + 1)
                self.send_update('> Board {} saved'.format(i))

            except Exception as e:
                self.send_board.emit("", board_img, path_dir, board_title)
                self.progress.set_progress(p_layer, i + 1)
                self.send_update('> An error occurred while processing board {}. <<{}>>'.format(i, str(e)))

            if self.should_stop:
                raise KeyboardInterrupt

            i += 1

        self.progress.pop_layer(p_layer)
        return paths, boards_img

    def __parse_pdf(self, path, result_dir, **kwargs):
        doc = fitz.Document(path)
        pages = kwargs['range'].pages
        p_layer = self.progress.add_layer((0, len(pages)))

        for page_number in pages:
            page = doc.load_page(page_number - 1)
            self.send_update('Rendering {}-th page of PDF'.format(str(page.number + 1)))

            try:
                # page.get_pixmap().writePNG('test.png')
                scale = 1.25
                scale_matrix = fitz.Matrix(scale, scale)  # get image 'scale' times larger than page.bound()
                png = page.get_pixmap(matrix=scale_matrix).getPNGData()
                png = np.frombuffer(png, dtype=np.int8)
                self.__parse_img(png, result_dir, file_prefix='page-{}-'.format(str(page.number + 1)),
                                 board_title_fmt="Страница {}, доска {{}} из {{}}".format(page.number + 1))

            except KeyboardInterrupt:
                raise
            except:
                pass
            self.progress.append_progress(p_layer, 1)

        self.progress.pop_layer(p_layer)


class SelectPageRangeDialog(QtWidgets.QDialog):
    def __init__(self, path_pdf, page_cnt, parent=None):
        super().__init__(parent=parent)
        self.cancelled = False

        form_class, _ = uic.loadUiType(pkg_resources.resource_stream('godr.frontend.ui', "select_pages_pdf.ui"))
        form_class().setupUi(self)

        self.findChild(QtWidgets.QLabel, 'path').setText(path_pdf)
        self.page_cnt = page_cnt
        self.page_range_widget = self.findChild(QtWidgets.QLineEdit, 'page_range')

        pages_str = translate_plural(page_cnt, 'страница', 'страницы', 'страниц')
        self.findChild(QtWidgets.QLabel, 'page_cnt').setText("Всего {} {}".format(page_cnt, pages_str))

    def closeEvent(self, event):
        self.cancelled = True

    def is_cancelled(self):
        return self.cancelled

    def get_range(self):
        text = self.page_range_widget.text()
        try:
            page_range = PageRange(text)
        except ValueError:
            page_range = PageRange("{}-{}".format(1, self.page_cnt))
        return page_range
