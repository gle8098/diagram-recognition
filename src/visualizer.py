from sgfmill import sgf
from matplotlib import pyplot as plt


class Visualizer:
    def __init__(self, path):
        with open(path, "rb") as f:
            game_bytes = f.read()

        self.game = sgf.Sgf_game.from_bytes(game_bytes)

    def draw_board(self, img=None):
        board_size = self.game.get_size()

        if img is not None:
            fig = plt.figure(figsize=[16, 8])
            # fig.patch.set_facecolor((1, 1, .8))

            plt.subplot(1, 2, 2)
            for x in range(board_size):
                plt.plot([x, x], [0, board_size - 1], 'k')
            for y in range(board_size):
                plt.plot([0, board_size - 1], [y, y], 'k')

            plt.axis('off')

            plt.xlim(-1, board_size)
            plt.ylim(-1, board_size)

            root = self.game.get_root()
            pos = root.get_setup_stones()
            white_stones = pos[1]
            black_stones = pos[0]

            markersize = 450 / (board_size + 3)
            for stone in white_stones:
                y = stone[0]
                x = stone[1]
                plt.plot(x, y, 'o', markersize=markersize, markeredgecolor=(0, 0, 0), markerfacecolor='w',
                         markeredgewidth=2)

            for stone in black_stones:
                y = stone[0]
                x = stone[1]
                edgecolor = 0
                plt.plot(x, y, 'o', markersize=markersize, markeredgecolor=(edgecolor, edgecolor, edgecolor),
                         markerfacecolor='k',
                         markeredgewidth=2)

            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.axis('off')

        else:
            '''
            source: https://stackoverflow.com/questions/24563513/drawing-a-go-board-with-matplotlib
            some constants need to be changed
            '''
            fig = plt.figure(figsize=[8, 8])
            # fig.patch.set_facecolor((1, 1, .8))
            ax = fig.add_subplot(111)

            # draw the grid
            for x in range(board_size):
                ax.plot([x, x], [0, board_size - 1], 'k')
            for y in range(board_size):
                ax.plot([0, board_size - 1], [y, y], 'k')

            # scale the axis area to fill the whole figure
            ax.set_position([0, 0, 1, 1])

            # get rid of axes and everything (the figure background will show through)
            ax.set_axis_off()

            # scale the plot area conveniently (the board is in 0,0..18,18)
            ax.set_xlim(-1, board_size)
            ax.set_ylim(-1, board_size)

            root = self.game.get_root()
            pos = root.get_setup_stones()
            white_stones = pos[1]
            black_stones = pos[0]

            markersize = 500 / (board_size + 1)
            for stone in white_stones:
                y = stone[0]
                x = stone[1]
                ax.plot(x, y, 'o', markersize=markersize, markeredgecolor=(0, 0, 0), markerfacecolor='w',
                        markeredgewidth=2)

            for stone in black_stones:
                y = stone[0]
                x = stone[1]
                ax.plot(x, y, 'o', markersize=markersize, markeredgecolor=(.5, .5, .5), markerfacecolor='k',
                        markeredgewidth=2)
