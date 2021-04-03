import glob
import sys
import cv2
import os

import numpy as np
from PyQt5 import uic, QtCore
from PyQt5.QtCore import QObject, QThread
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QApplication

from src.board import Board
from src.recognizer import Recognizer


class RecognitionWorker(QThread):
    update_ui = QtCore.pyqtSignal(int, str)

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
            if not extension in ['.png', '.jpg']:
                self.files_done += 1
                self.send_update('')  # Do not log skipped files
                continue

            self.send_update('Processing {}'.format(img_file))

            try:
                with open(img_file, "rb") as f:
                    chunk = f.read()
                    chunk_arr = np.frombuffer(chunk, dtype=np.uint8)

                self.parse_img(img_file, chunk_arr)
                output = 'Converted successfully'
            except Exception as ex:
                output = 'An error occurred <<{}>>'.format(str(ex))

            self.subprogress = 0
            self.files_done += 1
            self.send_update(output + '\n')

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
        self.send_update('> Found {} board(s)'.format(total_boards))

        i = 1
        for board_img in boards_img:
            board = Board(board_img)
            sgf_file = 'board-{}.sgf'.format(str(i))
            board.save_sgf(os.path.join(path_dir, sgf_file))

            self.subprogress = (i + 1) / (total_boards + 1)
            self.send_update('> Board {} saved'.format(i))
            i += 1


# Loads window layout
from os import path
import sys
cur_dir = path.dirname(sys.argv[0])
window_ui_path = path.join(cur_dir, "ui", "window.ui")
Form, Window = uic.loadUiType(window_ui_path)


class MainWindow(Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.selected_files = tuple()
        self.recognition_worker = None

    def select_files(self):
        type_filter = "PNG (*.png);;JPEG (*.jpg)"
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFiles)
        names, _ = dialog.getOpenFileNames(self, caption="Open files", directory=os.getcwd(), filter=type_filter)

        self.set_selected_files(names)

    def convert_to_sgf(self):
        print('Files are {}'.format(self.selected_files))

        self.recognition_worker = RecognitionWorker(self.selected_files)
        self.recognition_worker.update_ui.connect(self.update_progress_bar)
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    form = Form()
    form.setupUi(window)
    window.show()

    # If a path passed as 1st argument, immediately process it
    if len(sys.argv) > 1:
        files = []
        for path in sys.argv[1].split(','):
            files.extend(glob.glob(path))
        window.set_selected_files(files)
        window.convert_to_sgf()

    app.exec_()
