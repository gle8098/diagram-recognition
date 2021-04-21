import unittest
from unittest.mock import patch
from typing import Dict
from src.sgf_joiner import SGFJoiner
from sgfmill.sgf import Sgf_game
import os


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'sgf')
        cls.paths = [os.path.join(cls.dir, f) for f in
                     ['19x19_1.sgf', '19x19_2.sgf', '19x19_3.sgf', '9x9_1.sgf', '9x9_2.sgf', '9x9_3.sgf']
                     ]

    @classmethod
    def read_sgf(cls, path: str):
        with open(path) as fh:
            return Sgf_game.from_string(fh.read())

    def test_games_joiner(self):
        games = [self.read_sgf(path) for path in self.paths]
        joiner = SGFJoiner()
        r = joiner.join_games(games)
        self.compare(r)

    def test_games_joiner_empty(self):
        games = []
        joiner = SGFJoiner()
        r = joiner.join_games(games)
        self.assertDictEqual({}, r)

    def test_paths_joiner(self):
        joiner = SGFJoiner()
        r = joiner.join_files(self.paths)
        self.compare(r)

    def test_paths_joiner_empty(self):
        joiner = SGFJoiner()
        r = joiner.join_files([])
        self.assertDictEqual({}, r)

    @patch("src.sgf_joiner.os.listdir", return_value=['19x19_1.sgf', '19x19_2.sgf', '19x19_3.sgf', '9x9_1.sgf', '9x9_2.sgf', '9x9_3.sgf', 'mytext.txt.txt'])
    def test_dir_joiner(self, mocked_listdir):
        joiner = SGFJoiner()
        r = joiner.join_dir(self.dir)
        self.compare(r)

    def compare(self, actual: Dict[int, Sgf_game]):
        self.assertEqual(2, len(actual))
        self.assertEqual(9, actual[9].size)
        self.assertEqual(3, len(actual[9].get_root()))
        self.assertTupleEqual((None, None), actual[9].get_root().get_move())
        self.assertTupleEqual(('b', (2, 2)), actual[9].get_root()[2][0].get_move())
        self.assertEqual(3, len(actual[19].get_root()))

    def test_serialise(self):
        joiner = SGFJoiner()
        joiner.join_dir(self.dir)
        r = joiner.serialise()
        self.assertEqual(2, len(r))
        self.assertSetEqual({9, 19}, set(r.keys()))
        self.assertTrue(all([isinstance(v, bytes) for v in r.values()]))

    def test_serialise_empty(self):
        joiner = SGFJoiner()
        joiner.join_files([])
        r = joiner.serialise()
        self.assertDictEqual({}, r)


if __name__ == '__main__':
    unittest.main()
