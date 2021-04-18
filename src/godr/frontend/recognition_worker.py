import os

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, QThread

from godr.backend.board import Board
from godr.backend.recognizer import Recognizer


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
