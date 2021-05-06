import glob
import os
import sys

import pkg_resources
import fitz
import qtawesome as qta
from PyQt5 import QtWidgets
from PyQt5 import uic, QtCore
from PyQt5.QtGui import QPixmap, QImage, QColor, QPalette
from PyQt5.QtWidgets import QFileDialog, QApplication
# from PyQt5.uic.properties import QtGui
from PyQt5.Qt import Qt

from godr.frontend import miscellaneous
from godr.frontend.recognition_worker import RecognitionWorker, SelectPageRangeDialog
from godr.frontend.sgfpainter import SgfPainter


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load Qt Designer form
        form_class, _ = uic.loadUiType(pkg_resources.resource_stream('godr.frontend.ui', "window.ui"))
        form_class().setupUi(self)

        self.selected_files = tuple()
        self.recognition_worker = None

        # Boards preview
        self.paths = []
        self.images = []  # List of QPixmap
        self.parent_files = []
        self.board_indices = []
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

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.previous_file()
        if event.key() == Qt.Key_D:
            self.next_file()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.next_file()
        else:
            self.previous_file()

    def resizeEvent(self, event):
        result = super(QtWidgets.QMainWindow, self).resizeEvent(event)
        self.redraw_preview_image()
        return result

    def select_files(self):
        type_filter = "All supported formats (*.png *.jpg *.jpeg *.pdf)"
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFiles)
        names, _ = dialog.getOpenFileNames(self, caption="Open files", directory=os.getcwd(), filter=type_filter)

        self.set_selected_files(names)

    def convert_to_sgf(self):
        print('Files are {}'.format(self.selected_files))
        self.lock_recognize_button(True)

        ranges = self.collect_ranges_for_files()

        self.recognition_worker = RecognitionWorker(self.selected_files, ranges)
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

        current_path = self.paths[self.current_index]
        if current_path != "":
            file_name = os.path.basename(current_path)
            self.findChild(QtWidgets.QLabel, "sgf_name").setText(file_name)
        else:
            self.findChild(QtWidgets.QLabel, "sgf_name").setText("Не удалось распознать доску")

        self.findChild(QtWidgets.QLabel, "file_name").setText(self.parent_files[self.current_index])
        self.findChild(QtWidgets.QLabel, "board_index").setText(self.board_indices[self.current_index])

    def accept_new_board(self, path, image, parent_file, board_index):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        pixmap_img = QPixmap(QImage(bytes(image.data), width, height, bytes_per_line, QImage.Format_RGB888))

        self.paths.append(path)
        self.images.append(pixmap_img)
        self.parent_files.append(parent_file)
        self.board_indices.append(board_index)
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

    def collect_ranges_for_files(self):
        ranges = []
        for file in self.selected_files:
            if not file.endswith('.pdf'):
                ranges.append(None)
                continue

            doc = fitz.Document(file)
            dlg = SelectPageRangeDialog(file, doc.page_count)
            dlg.exec_()
            ranges.append(dlg.get_range())
        return ranges


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
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


if __name__ == '__main__':
    main()
