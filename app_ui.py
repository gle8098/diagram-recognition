import glob
import sys
import cv2
import os
from os import path

import qtawesome as qta
import numpy as np
from PyQt5 import uic, QtCore
from PyQt5.QtCore import QObject, QThread
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QColor, QPalette, QPainter
from PyQt5.QtWidgets import QFileDialog, QApplication

from src import miscellaneous
from src.board import Board
from src.recognizer import Recognizer
from src.sgfpainter import SgfPainter


class RecognitionWorker(QThread):
    update_ui = QtCore.pyqtSignal(int, str)   # (percent, append_to_console)
    send_board = QtCore.pyqtSignal(str, np.ndarray)  # (path_to_sgf, img)
    done = QtCore.pyqtSignal()

    def __init__(self, files):
        super(QObject, self).__init__()
        self.files = files
        self.files_done = 0
        self.subprogress = 0

    def send_update(self, line):
        """ Notifies MainWindow to update progress bar and append line to the log """
        percent = int(100 * (self.files_done + self.subprogress) / len(self.files))
        self.update_ui.emit(percent, line)

    def run(self):
        if not self.files:
            self.update_ui.emit(0, 'No files selected')
            return

        print('Recognising {} files'.format(len(self.files)))

        for img_file in self.files:
            extension = os.path.splitext(img_file)[1]
            if extension not in ['.png', '.jpg']:
                self.files_done += 1
                self.send_update('')  # Do not log skipped files
                continue

            self.send_update('Processing {}'.format(img_file))

            try:
                with open(img_file, "rb") as f:
                    chunk = f.read()
                    chunk_arr = np.frombuffer(chunk, dtype=np.uint8)

                self.parse_img(img_file, chunk_arr)
                output = "Converting finished"

            except Exception as ex:
                output = 'An error occurred <<{}>>'.format(str(ex))

            self.subprogress = 0
            self.files_done += 1
            self.send_update(output + '\n')

        self.done.emit()

    def parse_img(self, img_path, img_bytes):
        # Create dir
        index = img_path.rfind('.')
        path_dir = img_path[:index]
        os.makedirs(path_dir, exist_ok=True)

        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        rec = Recognizer()
        boards_img = rec.split_into_boards(img)

        total_boards = len(boards_img)
        self.subprogress = 1 / (total_boards + 1)
        self.send_update('Found {} board(s)'.format(total_boards))

        i = 1
        paths = []
        for board_img in boards_img:
            try:
                board = Board(board_img)
                sgf_file = 'board-{}.sgf'.format(str(i))
                path = os.path.join(path_dir, sgf_file)
                board.save_sgf(path)
                paths.append(path)
                self.send_board.emit(path, board_img)

                self.subprogress = (i + 1) / (total_boards + 1)
                self.send_update('> Board {} saved'.format(i))

            except Exception as e:
                self.send_board.emit("", board_img)
                self.subprogress = (i + 1) / (total_boards + 1)
                self.send_update('> An error occurred while processing board {}'.format(i))
            i += 1

        return paths, boards_img


# Loads window layout
app_dir = path.dirname(sys.argv[0])
ui_dir = path.join(app_dir, "ui")
Form, Window = uic.loadUiType(path.join(ui_dir, "window.ui"))


class MainWindow(Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.selected_files = tuple()
        self.recognition_worker = None

        # Boards preview
        self.paths = []
        self.images = []  # List of QPixmap
        self.current_index = -1
        self.n_boards = 0

        # Widgets
        self.sgfpainter = None
        self.origin_label = None
        self.auto_preview_widget = None

    def init_ui(self):
        # Init QtAwesome color from selected theme
        qta_color = QColor(QPalette().color(QPalette.Normal, QPalette.WindowText))
        qta.set_defaults(color=qta_color)

        self.sgfpainter = SgfPainter()
        self.findChild(QtWidgets.QFrame, "sgf_painter_frame").layout().addWidget(self.sgfpainter)

        self.origin_label = self.findChild(QtWidgets.QLabel, "origin_label")
        self.auto_preview_widget = self.findChild(QtWidgets.QCheckBox, "auto_preview")

        self.findChild(QtWidgets.QPushButton, "previous_button").setIcon(qta.icon('fa5s.angle-double-left'))
        self.findChild(QtWidgets.QPushButton, "next_button").setIcon(qta.icon('fa5s.angle-double-right'))

        self.lock_open_sgf_button(True)

    def resizeEvent(self, event):
        result = super(Window, self).resizeEvent(event)
        self.redraw_preview_image()
        return result

    def select_files(self):
        type_filter = "PNG, JPEG (*.png *.jpg)"
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFiles)
        names, _ = dialog.getOpenFileNames(self, caption="Open files", directory=os.getcwd(), filter=type_filter)

        self.set_selected_files(names)

    def convert_to_sgf(self):
        print('Files are {}'.format(self.selected_files))
        self.lock_recognize_button(True)

        self.recognition_worker = RecognitionWorker(self.selected_files)
        self.recognition_worker.update_ui.connect(self.update_progress_bar)
        self.recognition_worker.send_board.connect(self.accept_new_board)
        self.recognition_worker.done.connect(lambda: self.lock_recognize_button(False))
        self.recognition_worker.start()

    def update_progress_bar(self, percent, output):
        self.findChild(QtWidgets.QProgressBar, "progress_bar").setValue(percent)

        if output:
            self.findChild(QtWidgets.QPlainTextEdit, "console_output").appendPlainText(output)
            print(output)

    def set_selected_files(self, files):
        self.selected_files = files

        # todo: use Qt Linguist & its tr() function
        n = len(files)
        if n % 10 == 1 and n % 100 != 11:
            line = '{} файл выбран'
        elif 2 <= n % 10 <= 4 and (n % 100 < 10 or n % 100 > 20):
            line = '{} файла выбрано'
        else:
            line = '{} файлов выбрано'
        self.findChild(QtWidgets.QLabel, "label_files_selected").setText(line.format(n))

    def lock_recognize_button(self, state):
        self.findChild(QtWidgets.QPushButton, "recognize").setEnabled(not state)

    def lock_open_sgf_button(self, state):
        self.findChild(QtWidgets.QPushButton, "open_sgf").setEnabled(not state)

    def update_preview_board_label(self):
        line = 'Доска {} из {}'.format(self.current_index + 1, self.n_boards)
        self.findChild(QtWidgets.QLabel, "label_board_index").setText(line)

    def accept_new_board(self, path, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        pixmap_img = QPixmap(QImage(bytes(image.data), width, height, bytes_per_line, QImage.Format_RGB888))

        self.paths.append(path)
        self.images.append(pixmap_img)
        self.n_boards += 1

        index = self.current_index
        if self.n_boards == 1:  # Display the only board
            index = 0
        elif self.is_auto_preview_on():
            index = self.n_boards - 1
        self.set_preview_index(index)

    def next_file(self):
        self.set_preview_index(self.current_index + 1)

    def previous_file(self):
        self.set_preview_index(self.current_index - 1)

    def is_auto_preview_on(self):
        return self.auto_preview_widget.checkState() == QtCore.Qt.Checked

    def set_preview_index(self, index):
        if index < 0 or index >= self.n_boards:
            return False

        self.current_index = index
        if self.paths[index] != "":
            self.sgfpainter.load_game(self.paths[index], update=True)
            self.lock_open_sgf_button(False)
        else:
            self.sgfpainter.load_empty()
            self.lock_open_sgf_button(True)
        self.redraw_preview_image()
        self.update_preview_board_label()
        return True

    def redraw_preview_image(self):
        if self.current_index == -1:
            return

        pixmap = self.images[self.current_index]
        pixmap_scaled = pixmap.scaled(self.origin_label.width(), self.origin_label.height(),
                                      QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.origin_label.setPixmap(pixmap_scaled)

    def open_current_sgf(self):
        if self.current_index > -1:
            filename = self.paths[self.current_index]
            try:
                miscellaneous.open_file_in_external_app(filename)
            except Exception as ex:
                print("Could not open file \"{}\" because <<{}>>".format(filename, str(ex)))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    form = Form()
    form.setupUi(window)
    window.init_ui()
    window.show()

    # If a path is passed as 1st argument, immediately process it
    if len(sys.argv) > 1:
        files = []
        for path in sys.argv[1].split(','):
            files.extend(glob.glob(path))
        window.set_selected_files(files)
        window.convert_to_sgf()

    app.exec_()
