import glob
import os
import sys

import pkg_resources
import fitz
import qtawesome as qta
from PyQt5 import QtWidgets, QtGui
from PyQt5 import uic, QtCore
from PyQt5.QtGui import QPixmap, QImage, QColor, QPalette
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.Qt import Qt

from godr.frontend import miscellaneous, sgf_joiner_ui
from godr.frontend.miscellaneous import translate_plural
from godr.frontend.recognition_worker import RecognitionWorker, SelectPageRangeDialog
from godr.frontend.sgf_joiner_ui import MergeSgfDialog
from godr.frontend.sgfpainter import SgfPainter


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load Qt Designer form
        form_class, _ = uic.loadUiType(pkg_resources.resource_stream('godr.frontend.ui', "window.ui"))
        form_class().setupUi(self)

        self.selected_files = tuple()
        self.recognition_worker = None
        self.recognizing_status = False

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

    def on_menu_click(self, action):
        if action.objectName() == "action_SGF":
            window = MergeSgfDialog()
            window.exec()
        elif action.objectName() == "action_about":
            dialog = QtWidgets.QMessageBox()
            dialog.addButton(QtWidgets.QMessageBox.Ok)
            dialog.setText("""
            <h3>Распознавание диаграмм го</h3>
            <div><a href="https://t.me/joinchat/8_VND7cWrKNmOWQ6">Чат поддержки в телеграмме.</a></div>
            <h4>Авторы:</h4>
            <ul>
            <li>Владислав Вихров</li>
            <li>Евгений Кузнецов</li>
            <li>Глеб Степанов, <a href="https://vk.com/vlistov">ВК</a></li>
            </ul>
            <br>
            """)
            dialog.setWindowTitle('О программе')
            dialog.exec()

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
        names, _ = dialog.getOpenFileNames(self, caption="Выберите файлы для распознавания", filter=type_filter)

        self.set_selected_files(names)

    def convert_to_sgf(self):
        if self.recognizing_status:
            self.recognition_worker.stop()
            # Worker will send 'done' signal which will indicate that it has finished
            return

        print('Files are {}'.format(self.selected_files))
        self.switch_recognition_status(True)

        ranges = self.collect_ranges_for_files()
        if not ranges:
            self.switch_recognition_status(False)
            return

        self.recognition_worker = RecognitionWorker(self.selected_files, ranges)
        self.recognition_worker.update_ui.connect(self.update_progress_bar)
        self.recognition_worker.send_board.connect(self.accept_new_board)
        self.recognition_worker.done.connect(lambda: self.switch_recognition_status(False))
        self.recognition_worker.start()

    def update_progress_bar(self, percent, output):
        self.findChild(QtWidgets.QProgressBar, "progress_bar").setValue(percent)

        if output:
            self.findChild(QtWidgets.QPlainTextEdit, "console_output").appendPlainText(output)
            print(output)

    def set_selected_files(self, files):
        self.selected_files = files

        n = len(files)
        line = translate_plural(n, '{} файл выбран', '{} файла выбрано', '{} файлов выбрано')
        self.findChild(QtWidgets.QLabel, "label_files_selected").setText(line.format(n))

    def switch_recognition_status(self, state):
        self.recognizing_status = state
        button = self.findChild(QtWidgets.QPushButton, "recognize")
        if state:
            button.setText("Остановить распознавание")
            # button.setEnabled(not state)
        else:
            button.setText("Распознать")
            # button.setEnabled(not state)

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
        pixmap_img = QPixmap(QImage(bytes(image.data), width, height, bytes_per_line, QImage.Format_BGR888))

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
            if dlg.is_cancelled():
                return None

            ranges.append(dlg.get_range())
        return ranges


def main():
    app = QApplication(sys.argv)

    # Load and set icon
    icon_bytes = bytes(pkg_resources.resource_string('godr.frontend.ui', 'icon.svg'))
    icon_pixmap = QtGui.QPixmap.fromImage(QtGui.QImage.fromData(icon_bytes))
    app.setWindowIcon(QtGui.QIcon(icon_pixmap))

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
