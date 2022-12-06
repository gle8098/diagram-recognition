import unittest
from unittest.mock import patch
from typing import Dict
import godr
from godr.backend.recognizer import Recognizer
from sgfmill.sgf import Sgf_game
import numpy as np
import os
import cv2


class GodrTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'tests')
        cls.img_files = [os.path.join(cls.dir, f) for f in ['6.png']]

    def read_sgf(cls, path: str):
        with open(path) as fh:
            return Sgf_game.from_string(fh.read())

    def test_recognition_board_one(self):
        img_file = self.img_files[0]
        with open(img_file, "rb") as f:
            chunk = f.read()
            img_bytes = np.frombuffer(chunk, dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        rec = Recognizer()
        boards_img = rec.split_into_boards(img)
        white_stones_local_coordinates, black_stones_local_coordinates, x_size, y_size, edges = \
            rec.recognize(boards_img[0])
        white_stones_local_coordinates_0 = {(7, 4), (8, 4), (6, 5), (5, 8), (8, 7), (6, 8), (6, 4), (7, 9), (6, 7), (6, 6), (9, 4)}
        black_stones_local_coordinates_0 = {(8, 8), (7, 7), (9, 5), (6, 9), (8, 5), (7, 8)}
        x_size_0 = 10
        y_size_0 = 10
        white_stones_local_coordinates = set(white_stones_local_coordinates)
        black_stones_local_coordinates = set(black_stones_local_coordinates)
        self.assertSetEqual(white_stones_local_coordinates, white_stones_local_coordinates_0)
        self.assertSetEqual(black_stones_local_coordinates, black_stones_local_coordinates_0)
        self.assertEqual(x_size, x_size_0)
        self.assertEqual(y_size, y_size_0)


if __name__ == '__main__':
    unittest.main()
