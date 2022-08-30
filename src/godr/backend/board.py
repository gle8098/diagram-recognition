from godr.backend.recognizer import Recognizer
from sgfmill import sgf, boards, sgf_moves


class Board:
    def __init__(self, img):
        recognizer = Recognizer()
        white_stones_local_coordinates, black_stones_local_coordinates, x_size, y_size, edges = \
            recognizer.recognize(img)

        board_size, up_edge_size, left_edge_size = self.process_edges(edges, x_size, y_size)

        black_stones = \
            self.process_local_coordinates(black_stones_local_coordinates, board_size, up_edge_size, left_edge_size)
        white_stones = \
            self.process_local_coordinates(white_stones_local_coordinates, board_size, up_edge_size, left_edge_size)

        self.board = boards.Board(board_size)
        self.board.apply_setup(black_stones, white_stones, [])

    def process_edges(self, edges, x_size, y_size, edge_gap=3):
        up_edge = edges[0]
        down_edge = edges[1]
        left_edge = edges[2]
        right_edge = edges[3]
        board_size = 19
        if up_edge and down_edge:
            # board_size = y_size
            up_edge_size = 0
            # down_edge_size = 0
            if left_edge:
                if right_edge:
                    if x_size == y_size:
                        left_edge_size = 0
                        # right_edge_size = 0
                    else:
                        # Incorrect board, shouldn't ever happen
                        raise Exception
                else:
                    left_edge_size = 0
                    # right_edge_size = board_size - x_size
            elif right_edge:
                left_edge_size = board_size - x_size
                # right_edge_size = 0
            else:
                # Incorrect board, shouldn't ever happen
                raise Exception
        elif left_edge and right_edge:
            # board_size = x_size
            left_edge_size = 0
            # right_edge_size = 0
            if up_edge:
                if down_edge:
                    # Incorrect board, shouldn't ever happen
                    raise Exception
                up_edge_size = 0
                # down_edge_size = board_size - y_size
            elif down_edge:
                up_edge_size = board_size - y_size
                # down_edge_size = 0
            else:
                # Incorrect board, shouldn't ever happen
                raise Exception
        else:
            # The board has two edges that are adjacent or less
            up_gap = edge_gap * int(not up_edge)
            down_gap = edge_gap * int(not down_edge)
            left_gap = edge_gap * int(not left_edge)
            right_gap = edge_gap * int(not right_edge)
            # board_size = max(x_size + left_gap + right_gap, y_size + up_gap + down_gap)
            if not left_edge:
                left_edge_size = board_size - x_size
            else:
                left_edge_size = 0
            if not up_edge:
                up_edge_size = board_size - y_size
            else:
                up_edge_size = 0
        return [board_size, up_edge_size, left_edge_size]

    def process_local_coordinates(self, stones_local_coordinates, board_size, up_edge_size, left_edge_size):
        stones = []
        for stone in stones_local_coordinates:
            local_x, local_y = stone
            local_x += left_edge_size
            local_y += up_edge_size

            # In sgfmill the order is different
            x = board_size - local_y - 1
            y = local_x
            stones.append([x, y])
        return stones

    def save_sgf(self, path):
        game = sgf.Sgf_game(self.board.side)
        sgf_moves.set_initial_position(game, self.board)
        game_bytes = game.serialise()

        with open(path, "wb") as f:
            f.write(game_bytes)
            f.close()
