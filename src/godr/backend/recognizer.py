from collections import defaultdict

from godr.backend.recognizer_consts import *
from godr.backend.nn import StoneRecognizer


class Recognizer:
    class RecognitionError(Exception):
        pass

    class EmptyImageError(RecognitionError):
        pass

    class NoBoardError(RecognitionError):
        pass

    class EmptyBoardsError(RecognitionError):
        pass

    def __init__(self):
        self.nn_stone_recognizer = StoneRecognizer()

    def recognize(self, board_img):
        if board_img is None:
            raise self.EmptyImageError()

        board_img_gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)

        v_lines, h_lines = self.__lines_recognition(board_img_gray)

        if len(v_lines) == 0 or len(h_lines) == 0:
            raise self.NoBoardError()
        x_size, y_size = v_lines.shape[0], h_lines.shape[0]
        intersections = self.__find_intersections(v_lines, h_lines)
        cell_size = self.__get_cell_size(v_lines, h_lines)

        white_stones, black_stones = self.nn_stone_recognizer.recognize(board_img_gray, cell_size, intersections)
        if len(white_stones) == 0 and len(black_stones) == 0:
            raise self.NoBoardError()

        edges = self.__find_edges(v_lines, h_lines, white_stones + black_stones, cell_size)
        return white_stones, black_stones, x_size, y_size, edges

    def split_into_boards(self, page_img):

        # Preprocessing page: resize and greyscale
        orig_size = min(page_img.shape[0:2])
        if orig_size > MAX_PAGE_SIZE:
            scale = MAX_PAGE_SIZE / orig_size
        else:
            scale = 1
        page_img_crop = cv2.resize(page_img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        page_img_gray = cv2.cvtColor(page_img_crop, cv2.COLOR_BGR2GRAY)

        # Canny edges
        size = min(page_img_crop.shape[0:2])
        edges = self.__canny_edges(page_img_gray)

        # For better connectivity
        morph_kernel_size = round(size * MORPH_COEFF)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((morph_kernel_size, morph_kernel_size)))

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter boards by perimeter and area
        boards = []
        min_size = size * MIN_BOARD_SIZE_COEFF
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if perimeter > 4 * min_size and area > min_size ** 2:
                x, y, w, h = cv2.boundingRect(contour)
                boards.append((x, y, w, h))
        if len(boards) == 0:
            raise self.EmptyBoardsError()

        # Sort boards
        boards = self.__sort_boards(boards, size * SHIFT_COEFF)

        # Get boards
        board_images = []
        for i, board in enumerate(boards):
            x, y, w, h = board
            extra_size = round(min(w, h) * EXTRA_SIZE_COEFF)
            board_img = page_img_crop[max(y - extra_size, 0): min(y + h + extra_size, page_img.shape[0]),
                                      max(x - extra_size, 0): min(x + w + extra_size, page_img.shape[1])]
            cv2.imwrite("bug.png", board_img)
            board_images.append(board_img)
        return board_images

    def __canny_edges(self, img_gray):
        return cv2.Canny(img_gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

    def __all_lines(self, img_gray, hough_threshold):
        edges = self.__canny_edges(img_gray)
        size = min(img_gray.shape[0:2])
        lines = cv2.HoughLinesP(edges, rho=HOUGH_RHO, theta=HOUGH_THETA, threshold=hough_threshold,
                                minLineLength=size * MIN_LINE_LEN_COEFF, maxLineGap=size)
        if lines is None:
            return None
        return np.reshape(lines, (lines.shape[0], lines.shape[2]))

    def __verticals_horizontals(self, img_gray, hough_threshold):
        # Find all lines
        lines = self.__all_lines(img_gray, hough_threshold)
        if (lines is None):
            return np.array([]), np.array([])

        # Divide the lines into verticals and horizontals
        v_lines = []
        h_lines = []
        for line in lines:
            x1, y1, x2, y2 = line
            if x1 == x2:
                line[1] = min(y1, y2)
                line[3] = max(y1, y2)
                v_lines.append(line)
            else:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > VERTICAL_TAN_MIN:
                    line[1] = min(y1, y2)
                    line[3] = max(y1, y2)
                    line[0] = line[2] = round(np.mean([x1, x2]))
                    v_lines.append(line)
                elif abs(slope) < HORIZONTAL_TAN_MAX:
                    line[0] = min(x1, x2)
                    line[2] = max(x1, x2)
                    line[1] = line[3] = round(np.mean([y1, y2]))
                    h_lines.append(line)

        # Sort lines
        v_lines = np.array(sorted(v_lines, key=lambda line: line[0]))
        h_lines = np.array(sorted(h_lines, key=lambda line: line[1]))

        # Merge close lines
        size =  min(img_gray.shape[0:2])
        min_dist = size * MIN_DIST_COEFF
        return self.__merge_lines(v_lines, True, min_dist), self.__merge_lines(h_lines, False, min_dist)

    def __merge_lines(self, lines, is_vertical, min_dist):
        if lines.shape[0] == 0:
            return np.array([])

        ind = 0 if is_vertical else 1
        is_close = np.diff(lines, axis=0)[:, ind] <= min_dist
        merged_lines = []
        i = 0
        while i < lines.shape[0]:
            close_lines = [lines[i]]
            while i + 1 < lines.shape[0] and is_close[i]:
                i += 1
                close_lines.append(lines[i])
            close_lines = np.array(close_lines)
            if is_vertical:
                x = round(np.mean(close_lines[:, ind]))
                merged_line = [x, np.amin(close_lines[:, 1]), x, np.amax(close_lines[:, 3])]
            else:
                y = round(np.mean(close_lines[:, ind]))
                merged_line = (np.amin(close_lines[:, 0]), y, np.amax(close_lines[:, 2]), y)
            merged_lines.append(merged_line)
            i += 1

        return np.array(merged_lines)

    def __lines_recognition(self, img_gray):
        # Find some very clear lines
        hough_threshold = min(img_gray.shape[0:2])
        cell_size = 0
        while hough_threshold > HOUGH_LOW_THRESHOLD:
            clear_v_lines, clear_h_lines = self.__verticals_horizontals(img_gray, hough_threshold)
            if (len(clear_v_lines) > 3) and (len(clear_h_lines) > 3):
                dists_x = np.diff(clear_v_lines[:, 0])
                dists_y = np.diff(clear_h_lines[:, 1])
                cell_size_1 = np.amin(dists_x)
                cell_size_2 = np.amin(dists_y)
                cell_size = min(cell_size_1, cell_size_2)
                if cell_size / max(cell_size_1, cell_size_2) > 0.9:
                    dists = np.concatenate([dists_x, dists_y])
                    cell_size = np.mean(dists[np.round(dists / cell_size) == 1])
                    break
            hough_threshold -= HOUGH_THRESHOLD_STEP

        if cell_size == 0:
            return [], []

        # Find less clear lines
        while hough_threshold > HOUGH_LOW_THRESHOLD:
            hough_threshold -= HOUGH_THRESHOLD_STEP
            clear_v_lines, clear_h_lines = self.__verticals_horizontals(img_gray, hough_threshold)
            if (len(clear_v_lines) <= 1) or (len(clear_h_lines) <= 1):
                hough_threshold += HOUGH_THRESHOLD_STEP
                break
            dists_x = np.diff(clear_v_lines[:, 0])
            dists_y = np.diff(clear_h_lines[:, 1])
            dists = np.concatenate([dists_x, dists_y])
            fracs = np.modf(dists / cell_size)[0]
            ints = np.round(dists / cell_size)
            err = 0.9 - 0.05 * ints
            err[ints >= 7] = 0.55
            if np.any(np.logical_and(fracs < err, fracs > 1 - err)):
                hough_threshold += HOUGH_THRESHOLD_STEP
                break
            cell_size = np.mean(dists[ints == 1])
        clear_v_lines, clear_h_lines = self.__verticals_horizontals(img_gray, hough_threshold)

        # Add unclear lines
        dists_x = np.diff(clear_v_lines[:, 0])
        dists_y = np.diff(clear_h_lines[:, 1])
        dists = np.concatenate([dists_x, dists_y])
        cell_size = np.mean(dists[np.round(dists / cell_size) == 1])

        x_min = np.mean(clear_h_lines[:, 0])
        x_max = np.mean(clear_h_lines[:, 2])
        y_min = np.mean(clear_v_lines[:, 1])
        y_max = np.mean(clear_v_lines[:, 3])
        v_lines = []
        xs = np.concatenate([[x_min], clear_v_lines[:, 0], [x_max]])
        dists = np.diff(xs)
        num_lines = (dists / cell_size).astype(int)
        num_lines[np.modf(dists / cell_size)[0] > 0.8] += 1
        num_lines[1:-1][num_lines[1:-1] != 0] -= 1
        for i in range(1, xs.size):
            for ind in range(1, num_lines[i - 1] + 1):
                if i == xs.size - 1:
                    x = xs[i - 1] + np.round(ind * cell_size)
                elif i == 1:
                    x = xs[i] - np.round((num_lines[0] + 1 - ind) * cell_size)
                else:
                    x = xs[i] - np.round((num_lines[i - 1] + 1 - ind) * dists[i - 1] / (num_lines[i - 1] + 1))
                new_line = [x, y_min, x, y_max]
                v_lines.append(new_line)
            if i != xs.size - 1:
                v_lines.append(clear_v_lines[i - 1])
        h_lines = []
        ys = np.concatenate([[y_min], clear_h_lines[:, 1], [y_max]])
        dists = np.diff(ys)
        num_lines = (dists / cell_size).astype(int)
        num_lines[np.modf(dists / cell_size)[0] > 0.8] += 1
        num_lines[1:-1][num_lines[1:-1] != 0] -= 1
        for i in range(1, ys.size):
            for ind in range(1, num_lines[i - 1] + 1):
                if i == ys.size - 1:
                    y = ys[i - 1] + np.round(ind * cell_size)
                elif i == 1:
                    y = ys[i] - np.round((num_lines[0] + 1 - ind) * cell_size)
                else:
                    y = ys[i] - np.round((num_lines[i - 1] + 1 - ind) * dists[i - 1] / (num_lines[i - 1] + 1))
                new_line = [x_min, y, x_max, y]
                h_lines.append(new_line)
            if i != ys.size - 1:
                h_lines.append(clear_h_lines[i - 1])
        return np.array(v_lines).astype(int), np.array(h_lines).astype(int)

    def __get_cell_size(self, v_lines, h_lines):
        return round(np.mean([round(np.ptp(v_lines[:, 0]) / (v_lines.shape[0] - 1)),
                              round(np.ptp(h_lines[:, 1]) / (h_lines.shape[0] - 1))]))

    def __find_intersections(self, v_lines, h_lines):
        return np.array(np.meshgrid(v_lines[:, 0], h_lines[:, 1])).T

    def __find_edges(self, v_lines, h_lines, stones, cell_size):
        # Get stones on edges
        up_line_stones = []
        down_line_stones = []
        left_line_stones = []
        right_line_stones = []
        for stone in stones:
            x, y = stone
            if y == 0:
                up_line_stones.append(x)
            if y == len(h_lines) - 1:
                down_line_stones.append(x)
            if x == 0:
                left_line_stones.append(y)
            if x == len(v_lines) - 1:
                right_line_stones.append(y)

        # Get free points on edges
        up_line_free = np.delete(np.arange(len(v_lines)), up_line_stones)
        down_line_free = np.delete(np.arange(len(v_lines)), down_line_stones)
        left_line_free = np.delete(np.arange(len(h_lines)), left_line_stones)
        right_line_free = np.delete(np.arange(len(h_lines)), right_line_stones)

        def is_edge(edge_coord, line_ends, flag):
            diff = edge_coord - line_ends if flag else line_ends - edge_coord
            return np.sum(diff > cell_size * MIN_EDGE_COEFF) < np.ceil(len(line_ends) / 5)

        up_edge = is_edge(h_lines[0][1], v_lines[up_line_free, 1], True)
        down_edge = is_edge(h_lines[-1][1], v_lines[down_line_free, 3], False)
        left_edge = is_edge(v_lines[0][0], h_lines[left_line_free, 0], True)
        right_edge = is_edge(v_lines[-1][0], h_lines[right_line_free, 2], False)

        return up_edge, down_edge, left_edge, right_edge

    def __sort_boards(self, boards, shift):
        boards.sort(key=lambda coords: coords[1])
        new_line = np.diff(boards, axis=0)[:, 1] > shift
        num_lines = np.sum(new_line) + 1
        board_lines = [[] for _ in range(num_lines)]
        line_ind = 0
        board_lines[line_ind].append(0)
        for i in range(1, len(boards)):
            if new_line[i - 1]:
                line_ind += 1
            board_lines[line_ind].append(i)
        for board_line in board_lines:
            board_line.sort(key=lambda ind: boards[ind][0])
        boards_order = np.reshape(board_lines, -1)
        return np.array(boards)[boards_order]

