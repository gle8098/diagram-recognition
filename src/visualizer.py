from sgfmill import sgf
from matplotlib import pyplot as plt


class Visualizer:
    def __init__(self, path):
        with open(path, "rb") as f:
            game_bytes = f.read()

        self.game = sgf.Sgf_game.from_bytes(game_bytes)

    def draw_board(self, img=None, fig=None, n_boards=1, current_index=1):
        board_size = self.game.get_size()
        board_len = 4

        if img is not None:
            if fig is None:
                fig = plt.figure(figsize=[2 * board_len, board_len])

            ax1 = fig.add_subplot(n_boards, 2, 2*current_index)
        else:
            if fig is None:
                fig = plt.figure(figsize=[board_len, board_len])
            ax1 = fig.add_subplot(n_boards, 1, current_index)

        # fig.patch.set_facecolor((1, 1, .8))
        
        for x in range(board_size):
            ax1.plot([x, x], [0, board_size - 1], 'k')
        for y in range(board_size):
            ax1.plot([0, board_size - 1], [y, y], 'k')

        ax1.set_axis_off()

        ax1.set_xlim(-1, board_size)
        ax1.set_ylim(-1, board_size)

        root = self.game.get_root()
        pos = root.get_setup_stones()
        black_stones = pos[0]
        white_stones = pos[1]

        marker_size = 175 / (board_size + 1)
        for stone in white_stones:
            y = stone[0]
            x = stone[1]
            ax1.plot(x, y, 'o', markersize=marker_size, markeredgecolor=(0, 0, 0), markerfacecolor='w',
                     markeredgewidth=2)

        for stone in black_stones:
            y = stone[0]
            x = stone[1]
            ax1.plot(x, y, 'o', markersize=marker_size, markeredgecolor=(.5, .5, .5), markerfacecolor='k',
                     markeredgewidth=2)

        if img is not None:
            ax2 = fig.add_subplot(n_boards, 2, 2*current_index - 1)
            ax2.imshow(img)
            ax2.axis('off')
