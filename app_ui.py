import glob
import sys
from random import randint

import cv2
import os

import numpy as np
from PyQt5 import uic, QtCore
from PyQt5.QtCore import QObject, QThread, Qt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QVBoxLayout, QLabel, QWidget, QScrollArea
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from src.board import Board
from src.recognizer import Recognizer

from src.visualizer import Visualizer


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

                paths = self.parse_img(img_file, chunk_arr)
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
        paths = []
        for board_img in boards_img:
            board = Board(board_img)
            sgf_file = 'board-{}.sgf'.format(str(i))
            path = os.path.join(path_dir, sgf_file)
            board.save_sgf(path)
            paths.append(path)

            self.subprogress = (i + 1) / (total_boards + 1)
            self.send_update('> Board {} saved'.format(i))
            i += 1
        return paths


# Loads window layout
Form, Window = uic.loadUiType("ui/window.ui")


class MainWindow(Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.selected_files = tuple()
        self.recognition_worker = None

        self.w = PreviewWindow()

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

    def show_preview(self):
        # paths = 'data/images/d_3/board-1.sgf'
        paths = 'data/images/pages/chinese/1000_TsumeGo-10/*'
        paths = glob.glob(paths)
        # self.w.show()
        self.w.show_preview_image(paths)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, paths, parent=None, width=8, height=8, dpi=100):
        n_boards = len(paths)
        fig = plt.Figure(figsize=(8, n_boards * 8), dpi=dpi)
        for i in range(n_boards):
            path = paths[i]
            visualizer = Visualizer(path)
            visualizer.draw_board(fig=fig, n_boards=n_boards, current_index=i + 1)
        super(MplCanvas, self).__init__(fig)


class PreviewWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # layout = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents)

    def show_preview_image(self, paths):
        # path = 'data/images/d_3/board-1.sgf'
        sc = MplCanvas(paths=paths, parent=self, width=8, height=8, dpi=100)

        # self.setCentralWidget(sc)

        # layout = QtWidgets.QVBoxLayout(self.centralWidget())
        # self.scrollArea = QtWidgets.QScrollArea(self.centralWidget())
        # layout.addWidget(self.scrollArea)
        # self.scrollAreaWidgetContents = QtWidgets.QWidget()
        # self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1112, 932))
        # self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.scroll = QScrollArea()
        self.widget = QWidget()
        self.vbox = QVBoxLayout()

        self.vbox.addWidget(sc)
        #
        # for i in range(1, 50):
        #     object = QLabel("Test")
        #     self.vbox.addWidget(object)

        self.widget.setLayout(self.vbox)

        # Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        self.setCentralWidget(self.scroll)

        self.setGeometry(600, 100, 1000, 900)
        self.setWindowTitle('Scroll Area Demonstration')

        # self.setCentralWidget(sc)

        self.show()


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
