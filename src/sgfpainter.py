import sys

from PyQt5.QtDesigner import QPyDesignerCustomWidgetPlugin
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen
from PyQt5.QtCore import Qt, QRect, QPoint
from sgfmill import sgf

sgf_path = '../data/images/pages/cho/Cho_Chikun_Encyclopedia_Of_Life_And_Death_Vol_1_Elementary-003/board-1.sgf'
board_size = 400

# todo: THIS FILE IS IN DEVELOPMENT


class SgfPainter(QWidget):
    def __init__(self):
        super().__init__()
        self.game = None
        self.initUI()

    def initUI(self):
        self.setGeometry(board_size, board_size, board_size, board_size)
        self.setWindowTitle('Sgf Painter example')
        self.show()

    def load_game(self, sgf_path):
        with open(sgf_path, "rb") as f:
            self.game = sgf.Sgf_game.from_bytes(f.read())

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.draw_board(event, qp)
        qp.end()

    def draw_board(self, event, qp):
        if not self.game:
            return

        # output_canvas.configure(bg="#d9d9d9")
        # output_canvas.delete("all")
        # if not board_ready:
        #     if image_loaded:
        #         output_canvas.create_text((0, 0), text="Board not detected!", anchor="nw")
        #         output_canvas.create_text((0, 30), text="Things to try:", anchor="nw")
        #         output_canvas.create_text((0, 60), text="- Select a smaller region", anchor="nw")
        #         output_canvas.create_text((0, 90), text="- Rotate the image", anchor="nw")
        #         output_canvas.create_text((0, 120), text="- Show settings", anchor="nw")
        #         output_canvas.create_text((0, 150), text="  -> Increase contrast", anchor="nw")
        #         output_canvas.create_text((0, 180), text="  -> Increase threshold", anchor="nw")
        #     return
        w, h = self.width(), self.height()
        s = min(w, h)  # size of board+margin
        sgf_board_size = self.game.get_size()

        bg_brush = QBrush(QColor("#FFC050"))
        qp.fillRect(QRect(0, 0, w, h), bg_brush)

        pen = QPen(Qt.black)
        qp.setPen(pen)

        if s < 220:  # too small to draw the board
            qp.drawText(QPoint(100, 100), "Too small!")
            # todo color & center
            return

        width = s - 60  # width of the actual board
        r = int(width / 18 / 2.1)  # radius of stones
        coords = [int(i * width / 18 + 30) for i in range(19)]
        cmin, cmax = min(coords), max(coords)

        for c in coords:
            qp.drawLine(c, cmin, c, cmax)
            qp.drawLine(cmin, c, cmax, c)

        # Star points
        qp.setBrush(Qt.black)
        for i in [coords[3], coords[9], coords[15]]:
            for j in [coords[3], coords[9], coords[15]]:
                qp.drawEllipse(i - 2, j - 2, 4, 4)

        # Read stones
        root = self.game.get_root()
        pos = root.get_setup_stones()

        # Stones
        for env in zip(pos, (Qt.black, Qt.white)):
            qp.setBrush(env[1])
            for stone in env[0]:
                y, x = coords[sgf_board_size - 1 - stone[0]], coords[stone[1]]
                qp.drawEllipse(x - r, y - r, 2 * r, 2 * r)

        # Positioning circles: these should only appear for part board positions
        # pos_centres = []
        # if hsize < BOARD_SIZE and vsize < BOARD_SIZE:
        #     # corner position
        #     pos_centres = [(15, 15), (15, width + 45), (width + 45, 15), (width + 45, width + 45)]
        # elif hsize < BOARD_SIZE:
        #     # left or right size position
        #     pos_centres = [(15, coords[9]), (width + 45, coords[9])]
        # elif vsize < BOARD_SIZE:
        #     # top or bottom position
        #     pos_centres = [(coords[9], 15), (coords[9], width + 45)]
        # for i, j in pos_centres:
        #     output_canvas.create_oval(i - 2, j - 2, i + 2, j + 2, fill="pink")
        #     output_canvas.create_oval(i - 8, j - 8, i + 8, j + 8)


class GeoLocationPlugin(QPyDesignerCustomWidgetPlugin):
    def __init__(self, parent=None):
        QPyDesignerCustomWidgetPlugin.__init__(self)
        self.initialized = False

    def initialize(self, formEditor):
        if self.initialized:
            return

        manager = formEditor.extensionManager()

        self.initialized = True

    def createWidget(self, parent):
        return SgfPainter()

    def name(self):
        return "SgfPainter"

    def includeFile(self):
        return "src.sgfpainter"



def main():
    app = QApplication(sys.argv)
    ex = SgfPainter()

    ex.load_game(sgf_path)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
