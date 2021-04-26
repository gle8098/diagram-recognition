import itertools

from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen
from PyQt5.QtCore import Qt, QRect
from sgfmill import sgf


def get_star_points(size):
    if size == 19:
        return itertools.product((3, 9, 15), (3, 9, 15))
    elif size == 13:
        return (6, 6), *itertools.product((3, 9), (3, 9))
    elif size == 9:
        return (4, 4), *itertools.product((2, 6), (2, 6))
    return tuple()  # No stars for custom size


class SgfPainter(QWidget):
    def __init__(self):
        super().__init__()
        self.game = None

    def load_game(self, sgf_path, **kwargs):
        with open(sgf_path, "rb") as f:
            self.game = sgf.Sgf_game.from_bytes(f.read())

        if kwargs['update']:
            self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.draw_board(event, qp)
        qp.end()

    def draw_board(self, event, qp):
        if not self.game:
            return

        w, h = self.width(), self.height()
        s = min(w, h)  # size of board+margin
        sgf_board_size = self.game.get_size()

        bg_brush = QBrush(QColor("#FFFFFF"))  # "#FFC050"
        qp.fillRect(QRect(0, 0, w, h), bg_brush)

        pen = QPen(Qt.black)
        qp.setPen(pen)

        if s < 220:  # too small to draw the board
            self.draw_message(qp, self.tr("Too small!"))
            return

        width = s - 60  # width of the actual board
        width_per_col = width / (sgf_board_size-1)
        r = (width_per_col / 2.1)  # radius of stones
        coords = [(i * width_per_col + 30) for i in range(sgf_board_size)]
        cmin, cmax = min(coords), max(coords)

        qp.translate((w - s) // 2, (h - s) // 2)  # draw board at the center

        for c in coords:
            qp.drawLine(c, cmin, c, cmax)
            qp.drawLine(cmin, c, cmax, c)

        # Star points
        qp.setBrush(Qt.black)
        star_coords = get_star_points(sgf_board_size)
        for ij in star_coords:
            x, y = coords[ij[0]], coords[ij[1]]
            qp.drawEllipse(x - 2, y - 2, 4, 4)

        # Read stones
        root = self.game.get_root()
        pos = root.get_setup_stones()

        # Stones
        for env in zip(pos, (Qt.black, Qt.white)):
            qp.setBrush(env[1])
            for stone in env[0]:
                y, x = coords[sgf_board_size - 1 - stone[0]], coords[stone[1]]
                qp.drawEllipse(x - r, y - r, 2 * r, 2 * r)

    def draw_message(self, qp, msg):
        # Draws msg in top center on the widget.
        # Usually it is only one msg shown at a time

        pen = QPen(Qt.black)
        qp.setPen(pen)
        qp.drawText(QRect(0, 0, self.width(), 28), QtCore.Qt.AlignCenter, msg)

    def load_empty(self):
        self.game = None
        self.update()
