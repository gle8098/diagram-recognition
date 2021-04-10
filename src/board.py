import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.stone import Stone
from src.recognizer import Recognizer
import copy
from sgfmill import sgf, boards, sgf_moves


class Board:
    def __init__(self, img):
        EDGE_GAP = 3

        recognizer = Recognizer()
        debug = True
        if debug:
            self.intersections, white_stones, black_stones, self.radius, x_size, y_size, edges = recognizer.recognize(
                img)
        else:
            try:
                self.intersections, white_stones, black_stones, self.radius, x_size, y_size, edges = recognizer.recognize(
                    img)
            except:
                return
        self.img = img
        self.white_stones = []
        self.black_stones = []

        up_edge = edges[0]
        down_edge = edges[1]
        left_edge = edges[2]
        right_edge = edges[3]
        if up_edge and down_edge:
            self.board_size = y_size
            up_edge_size = 0
            down_edge_size = 0
            if left_edge:
                if right_edge:
                    if x_size == y_size:
                        left_edge_size = 0
                        right_edge_size = 0
                    else:
                        # Incorrect board, shouldn't ever happen
                        raise Exception
                else:
                    left_edge_size = 0
                    right_edge_size = self.board_size - x_size
            elif right_edge:
                left_edge_size = self.board_size - x_size
                right_edge_size = 0
            else:
                # Incorrect board, shouldn't ever happen
                raise Exception
        elif left_edge and right_edge:
            self.board_size = x_size
            left_edge_size = 0
            right_edge_size = 0
            if up_edge:
                if down_edge:
                    # Incorrect board, shouldn't ever happen
                    raise Exception
                up_edge_size = 0
                down_edge_size = self.board_size - y_size
            elif down_edge:
                up_edge_size = self.board_size - y_size
                down_edge_size = 0
            else:
                # Incorrect board, shouldn't ever happen
                raise Exception
        else:
            up_edge_size = EDGE_GAP * int(not up_edge)
            down_edge_size = EDGE_GAP * int(not down_edge)
            left_edge_size = EDGE_GAP * int(not left_edge)
            right_edge_size = EDGE_GAP * int(not right_edge)
            self.board_size = max(x_size + left_edge_size + right_edge_size, y_size + up_edge_size + down_edge_size)

        for stone in white_stones:
            global_x = stone[0]
            global_y = stone[1]
            local_x, local_y = self.find_stone_in_intersections(stone)
            local_x += left_edge_size
            local_y += up_edge_size
            self.white_stones.append(Stone(local_x, local_y, global_x, global_y))
        for stone in black_stones:
            global_x = stone[0]
            global_y = stone[1]
            local_x, local_y = self.find_stone_in_intersections(stone)
            local_x += left_edge_size
            local_y += up_edge_size
            self.black_stones.append(Stone(local_x, local_y, global_x, global_y))

    def get_white_coordinates(self):
        for stone in self.white_stones:
            print(stone.local_x, stone.local_y)

    def get_black_coordinates(self):
        for stone in self.black_stones:
            print(stone.local_x, stone.local_y)

    def to_RGB(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def find_stone_in_intersections(self, stone):
        for i in range(self.intersections.shape[0]):
            for j in range(self.intersections.shape[1]):
                if np.array_equal(self.intersections[i][j], stone):
                    return i, j

    def visualize(self):
        try:
            visualization = copy.copy(self.img)
            for intersection in self.intersections.reshape(-1, 2):
                cv2.circle(visualization, (intersection[0], intersection[1]), 5, (255, 0, 255), -1)
            for stone in self.white_stones:
                cv2.circle(visualization, (stone.global_x, stone.global_y), self.radius, (0, 0, 255), 3)
            for stone in self.black_stones:
                cv2.circle(visualization, (stone.global_x, stone.global_y), self.radius, (255, 0, 0), 3)

            plt.figure(figsize=(20, 10))

            plt.subplot(1, 2, 1)
            plt.imshow(self.to_RGB(self.img))
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(self.to_RGB(visualization))
            plt.axis('off')
            plt.show()
        except:
            return

    def save_sgf(self, path):
        min_x = np.min(self.intersections.T[0])
        min_y = np.min(self.intersections.T[1])
        max_x = np.max(self.intersections.T[0])
        max_y = np.max(self.intersections.T[1])

        game = sgf.Sgf_game(self.board_size)

        board = boards.Board(self.board_size)

        for stone in self.white_stones:
            x = self.board_size - stone.local_y - 1
            y = stone.local_x
            if (x < 0) or (x >= self.board_size) or (y < 0) or (y >= self.board_size):
                print('Coordinate error')
                continue
            board.play(x, y, 'w')

        for stone in self.black_stones:
            x = self.board_size - stone.local_y - 1
            y = stone.local_x
            if (x < 0) or (x >= self.board_size) or (y < 0) or (y >= self.board_size):
                print('Coordinate error')
                continue
            board.play(x, y, 'b')

        sgf_moves.set_initial_position(game, board)
        game_bytes = game.serialise()

        with open(path, "wb") as f:
            f.write(game_bytes)
            f.close()

        return
