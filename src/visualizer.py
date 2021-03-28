from sgfmill import sgf
from matplotlib import pyplot as plt


class Visualizer:
    def __init__(self, path):
        with open(path, "rb") as f:
            game_bytes = f.read()

        self.game = sgf.Sgf_game.from_bytes(game_bytes)

    def draw_board(self, img=None, fig=None, n_boards = 1, current_index = 1):
        board_size = self.game.get_size()

        if img is not None:
            if fig is None:
                fig = plt.figure(figsize=[16, 8])
            # fig.patch.set_facecolor((1, 1, .8))

            ax1 = fig.add_subplot(1, 2, 2)
            # fig.subplot(1, 2, 2)
            for x in range(board_size):
                ax1.plot([x, x], [0, board_size - 1], 'k')
            for y in range(board_size):
                ax1.plot([0, board_size - 1], [y, y], 'k')

            ax1.set_axis_off()

            ax1.set_xlim(-1, board_size)
            ax1.set_ylim(-1, board_size)

            root = self.game.get_root()
            pos = root.get_setup_stones()
            white_stones = pos[1]
            black_stones = pos[0]

            markersize = 450 / (board_size + 3)
            for stone in white_stones:
                y = stone[0]
                x = stone[1]
                ax1.plot(x, y, 'o', markersize=markersize, markeredgecolor=(0, 0, 0), markerfacecolor='w',
                         markeredgewidth=2)

            for stone in black_stones:
                y = stone[0]
                x = stone[1]
                edgecolor = 0
                ax1.plot(x, y, 'o', markersize=markersize, markeredgecolor=(edgecolor, edgecolor, edgecolor),
                         markerfacecolor='k',
                         markeredgewidth=2)

            ax2 = fig.add_subplot(1, 2, 1)
            # ax.add_subplot(1, 2, 1)
            ax2.imshow(img)
            ax2.axis('off')

        else:
            '''
            source: https://stackoverflow.com/questions/24563513/drawing-a-go-board-with-matplotlib
            some constants need to be changed
            '''
            if fig is None:
                fig = plt.figure(figsize=[8, 8])
            # fig.patch.set_facecolor((1, 1, .8))
            # print(n_boards, 1, current_index)
            ax1 = fig.add_subplot(n_boards, 1, current_index)

            # draw the grid
            for x in range(board_size):
                ax1.plot([x, x], [0, board_size - 1], 'k')
            for y in range(board_size):
                ax1.plot([0, board_size - 1], [y, y], 'k')

            # scale the axis area to fill the whole figure
            # ax1.set_position([0, 0, 1, 1])

            # get rid of axes and everything (the figure background will show through)
            ax1.set_axis_off()

            # scale the plot area conveniently (the board is in 0,0..18,18)
            ax1.set_xlim(-1, board_size)
            ax1.set_ylim(-1, board_size)

            root = self.game.get_root()
            pos = root.get_setup_stones()
            white_stones = pos[1]
            black_stones = pos[0]

            markersize = 300 / (board_size + 1)
            for stone in white_stones:
                y = stone[0]
                x = stone[1]
                ax1.plot(x, y, 'o', markersize=markersize, markeredgecolor=(0, 0, 0), markerfacecolor='w',
                        markeredgewidth=2)

            for stone in black_stones:
                y = stone[0]
                x = stone[1]
                ax1.plot(x, y, 'o', markersize=markersize, markeredgecolor=(.5, .5, .5), markerfacecolor='k',
                        markeredgewidth=2)
