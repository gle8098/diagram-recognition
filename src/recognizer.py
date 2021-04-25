import cv2
import numpy as np
from collections import defaultdict

from src.recognizer_consts import *
from src.nn import StoneRecognizer


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
        print(edges)
        return white_stones, black_stones, x_size, y_size, edges

    def split_into_boards(self, page_img):
        orig_size = min(page_img.shape[0:2])
        if orig_size > 2000:
            scale = 2000 / orig_size
        else:
            scale = 1
        # For better connectivity and higher speed
        page_img_crop = cv2.resize(page_img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        size = min(page_img_crop.shape[0:2])
        page_img_gray = cv2.cvtColor(page_img_crop, cv2.COLOR_BGR2GRAY)
        edges = self.__get_edges(page_img_gray)
        # For better connectivity
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5)))
        min_size = (size * MIN_BOARD_SIZE_COEFF)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter contours (big and long enough)
        boards = []
        extra_size = orig_size
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if perimeter > 4 * min_size and area > min_size ** 2:
                x, y, w, h = cv2.boundingRect(contour)
                extra_size = min(w, h, extra_size)
                boards.append((x, y, w, h))
        if len(boards) == 0:
            raise self.EmptyBoardsError()
        extra_size //= 20
        boards.sort(key=lambda coords: coords[1])
        new_line = np.diff(boards, axis=0)[:, 1] > extra_size
        num_lines = np.sum(new_line) + 1
        board_lines = [[] for i in range(num_lines)]
        line_ind = 0
        board_lines[line_ind].append(0)
        for i in range(1, len(boards)):
            if new_line[i - 1]:
                line_ind += 1
            board_lines[line_ind].append(i)
        for board_line in board_lines:
            board_line.sort(key=lambda ind: boards[ind][0])
        boards_order = np.reshape(board_lines, -1)
        boards = np.array(boards)[boards_order]
        board_images = []
        for board in boards:
            x, y, w, h = board
            board_img = page_img_crop[max(y - extra_size, 0): min(y + h + extra_size, page_img.shape[0]),
                                      max(x - extra_size, 0): min(x + w + extra_size, page_img.shape[1])]
            board_images.append(board_img)
        return board_images

    def __get_edges(self, img_gray):
        return cv2.Canny(img_gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

    def __all_lines(self, img_gray, hough_threshold, gap=0):
        edges = self.__get_edges(img_gray)
        size = min(img_gray.shape[0:2])
        if gap == 0:
            gap = size
        lines = cv2.HoughLinesP(edges, rho=HOUGH_RHO, theta=HOUGH_THETA, threshold=hough_threshold,
                                minLineLength=size * MIN_LINE_LENGHT_COEFF, maxLineGap=gap)
        if lines is None:
            return None
        return np.reshape(lines, (lines.shape[0], lines.shape[2]))

    def __get_verticals_horizontals(self, img_gray, hough_threshold, gap=0):
        # Find all lines
        lines = self.__all_lines(img_gray, hough_threshold, gap)
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
        v_lines = np.array(sorted(v_lines, key=lambda line: line[0]))
        h_lines = np.array(sorted(h_lines, key=lambda line: line[1]))
        return self.__merge_lines(v_lines, True), self.__merge_lines(h_lines, False)

    # IMPROVE!
    def __merge_lines(self, lines, is_vertical):

        if lines.shape[0] == 0:
            return np.array([])
        ind = 0 if is_vertical else 1
        is_close = np.diff(lines, axis=0)[:, ind] <= MIN_DIST
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

        hough_threshold = min(img_gray.shape[0:2])
        cell_size = 0
        while hough_threshold > HOUGH_LOW_THRESHOLD:
            clear_v_lines, clear_h_lines = self.__get_verticals_horizontals(img_gray, hough_threshold)
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
        while hough_threshold > HOUGH_LOW_THRESHOLD:
            hough_threshold -= HOUGH_THRESHOLD_STEP
            clear_v_lines, clear_h_lines = self.__get_verticals_horizontals(img_gray, hough_threshold, 1.5 * cell_size)
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
        clear_v_lines, clear_h_lines = self.__get_verticals_horizontals(img_gray, hough_threshold, 1.5 * cell_size)
        dists_x = np.diff(clear_v_lines[:, 0])
        dists_y = np.diff(clear_h_lines[:, 1])
        dists = np.concatenate([dists_x, dists_y])
        cell_size = np.mean(dists[np.round(dists / cell_size) == 1])
        # Add unclear lines
        # possible bugs!
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
        up_line_free = np.delete(np.arange(len(v_lines)), up_line_stones)
        down_line_free = np.delete(np.arange(len(v_lines)), down_line_stones)
        left_line_free = np.delete(np.arange(len(h_lines)), left_line_stones)
        right_line_free = np.delete(np.arange(len(h_lines)), right_line_stones)

        def is_edge(edge_coord, line_ends, flag):
            diff = edge_coord - line_ends if flag else line_ends - edge_coord
            return np.sum(diff > cell_size * MIN_EDGE_COEFF) < np.ceil(len(line_ends) / 10)

        up_edge = is_edge(h_lines[0][1], v_lines[up_line_free, 1], True)
        down_edge = is_edge(h_lines[-1][1], v_lines[down_line_free, 3], False)
        left_edge = is_edge(v_lines[0][0], h_lines[left_line_free, 0], True)
        right_edge = is_edge(v_lines[-1][0], h_lines[right_line_free, 2], False)

        return up_edge, down_edge, left_edge, right_edge

    def __find_circles(self, img_gray, param2, min_dist_coeff, min_r_coeff, max_r_coeff, cell_size):
        circles = cv2.HoughCircles(img_gray, method=METHOD, dp=DP,
                                   minDist=round(cell_size * min_dist_coeff),
                                   param1=PARAM1, param2=param2, minRadius=round(cell_size * min_r_coeff),
                                   maxRadius=round(cell_size * max_r_coeff))
        if circles is None:
            return None
        return circles[0, :]

    def __stones_recognition(self, img_gray, cell_size, intersections):
        all_circles = []
        param2_grid = np.linspace(5, 30, 6)
        min_dist_coeff_grid = np.linspace(0.9, 0.9, 1)
        min_r_coeff_grid = np.linspace(0.4, 0.45, 2)
        max_r_coeff_grid = np.linspace(0.5, 0.55, 2)
        grid = np.array(np.meshgrid(param2_grid,
                                    min_dist_coeff_grid,
                                    min_r_coeff_grid,
                                    max_r_coeff_grid)).T.reshape(-1, 4)

        for param2, min_dist_coeff, min_r_coeff, max_r_coeff in grid:
            circles = self.__find_circles(img_gray, param2, min_dist_coeff, min_r_coeff, max_r_coeff, cell_size)
            if circles is not None:
                all_circles.append(circles)
        circles = np.concatenate(all_circles)
        # Filter circles
        stones = defaultdict(list)
        radii = []
        for circle in circles:
            for intersection in np.reshape(intersections, (-1, 2)):
                if np.linalg.norm(circle[:2] - intersection) <= MIN_INTERSECTION_DIST_COEFF * cell_size:
                    stones[tuple(intersection)].append(circle[:2])
                    radii.append(circle[2])
                    break
        for intersection in stones.keys():
            stones[intersection] = np.round(np.mean(np.array(stones[intersection]), axis=0)).astype(int)
        return stones, round(np.mean(radii))

    def __colorize(self, img_gray, stones, radius):
        white_stones = []
        black_stones = []
        for inter, stone in stones.items():
            stone_mask = np.zeros((img_gray.shape[0], img_gray.shape[1]), np.uint8)
            cv2.circle(stone_mask, (stone[0], stone[1]), round(radius * 0.8), 255, -1)
            average_color = cv2.mean(img_gray, mask=stone_mask)[0]
            if average_color >= WHITE_THRESHOLD:
                white_stones.append(inter)
            elif average_color <= BLACK_THRESHOLD:
                black_stones.append(inter)
        return np.array(white_stones), np.array(black_stones)

    def __signs_colorize(self, img_gray, stones, radius):
        WHITE_WITH_NO_SIGNS_THRESHOLD = 250
        CROSS_THRESHOLD = 180
        WHITE_WITH_SIGN_THRESHOLD = 120
        white_stones = []
        black_stones = []
        width = radius // 10
        for stone_inter, stone in stones.items():
            stone_mask = np.zeros((img_gray.shape[0], img_gray.shape[1]), np.uint8)
            cv2.circle(stone_mask, (stone[0], stone[1]), round(radius * 0.8), 255, -1)
            average_color = cv2.mean(img_gray, mask=stone_mask)[0]
            if average_color >= WHITE_WITH_NO_SIGNS_THRESHOLD:
                white_stones.append(stone_inter)
            elif average_color >= CROSS_THRESHOLD:
                average_color_left = np.mean(img_gray[stone_inter[1] - radius: stone_inter[1],
                                             stone_inter[0] - width: stone_inter[0] + width])
                average_color_right = np.mean(img_gray[stone_inter[1]: stone_inter[1] + radius,
                                              stone_inter[0] - width: stone_inter[0] + width])
                average_color_up = np.mean(np.mean(img_gray[stone_inter[1] - width: stone_inter[1] + width,
                                                   stone_inter[0] - radius: stone_inter[0]]))
                average_color_down = np.mean(img_gray[stone_inter[1] - width: stone_inter[1] + width,
                                             stone_inter[0]: stone_inter[0] + radius])
                if np.sum([average_color_left <= 150, average_color_right <= 150,
                           average_color_up <= 150, average_color_down <= 150]) >= 2:
                    pass
                else:
                    white_stones.append(stone_inter)
            elif average_color >= WHITE_WITH_SIGN_THRESHOLD:
                white_stones.append(stone_inter)
            else:
                black_stones.append(stone_inter)
        return np.array(white_stones), np.array(black_stones)

    def __sort_boards(self, boards):
        boards.sort(key=lambda coords: (coords[1], coords[0]))

