from sgfmill.sgf import Sgf_game, Tree_node
import datetime
import argparse
import os
from typing import List, Dict
import logging

VERSION = 0.1


class SGFJoiner:
    """
    Joins multiple boards to SGF by board size.
    Boards become children of empty board root.
    All methods return a dictionary by board size:
        join_games - joins a list of sgfmill Sgf_games
        join_files - joins a list of SGF files
        join_dir - joins all sgf files in a directory
        serialise - returns a dict of serialised sgf files
    """
    def __init__(self):
        self.boards = {}

    def serialise(self) -> Dict[int, bytes]:
        return {size: game.serialise() for size, game in self.boards.items()}

    def join_games(self, games: List[Sgf_game]) -> Dict[int, Sgf_game]:
        for game in games:
            board_size = game.get_size()
            master_board = self.boards.setdefault(board_size, Sgf_game(size=board_size))
            self._cross_game_node_reparent(game.get_root(), master_board.get_root())
        # set date and application properties
        for board in self.boards.values():
            root_node = board.get_root()
            root_node.set('DT', datetime.datetime.now().strftime('%Y-%m-%d'))
            root_node.set('AP', ('SGF joiner', '{}'.format(VERSION)))
        return self.boards

    def join_files(self, sgf_files: List[str]) -> Dict[int, Sgf_game]:
        games = []
        for path in sgf_files:
            try:
                with open(path) as sgf_file:
                    games.append(Sgf_game.from_string(sgf_file.read()))
            except FileNotFoundError:
                logging.error("Skipping, no file found: {}".format(path))
            except ValueError:
                logging.error("Skipping, not a valid sgf: {}".format(path))

        logging.info("Read {} boards for joining".format(len(games)))
        return self.join_games(games)

    def join_dir(self, dirpath: str):
        files = [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith(".sgf")]
        return self.join_files(files)

    @classmethod
    def _cross_game_node_reparent(cls, node: Tree_node, new_parent_node: Tree_node):
        new_node = new_parent_node.new_child()
        for property_name in node.properties():
            # do not copy properties which must be defined at root level only
            if property_name in ['FF', 'CA', 'SZ', 'GM', 'DT', 'KM', 'AP']:
                continue
            new_node.set(property_name, node.get(property_name))
        colour, point = node.get_move()
        if colour:
            new_node.set_move(colour, point)
        for child in node:
            cls._cross_game_node_reparent(child, new_node)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", dest="dir", help="Path to directory with sgf files")
    group.add_argument("--files", dest="files", action="append", nargs="*", help="Path to files")
    parser.add_argument("--outdir", dest="outdir", help="Path to directory to store results or printed to stdout")

    args = parser.parse_args()
    logging.info("Running with arguments: {}".format(args))
    joiner = SGFJoiner()
    joiner.join_dir(args.dir) if args.dir else joiner.join_files(args.files)
    result = joiner.serialise()
    if args.outdir:
        for s, c in result.items():
            with open(os.path.join(args.outdir, "joined_{}x{}.sgf".format(s, s)), "wb") as fh:
                fh.write(c)
    else:
        print(result)


if __name__ == "__main__":
    main()
