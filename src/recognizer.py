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

    def __init__(self):
        self.nn_stone_recognizer = StoneRecognizer()

    def recognize(self, board_img):
        if board_img is None:
            raise self.EmptyImageError()
        board_img_gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)

        v_lines, h_lines = self.lines_recognition(board_img_gray)
        if len(v_lines) == 0 or len(h_lines) == 0:
            raise self.NoBoardError()

        x_size, y_size = v_lines.shape[0], h_lines.shape[0]
        intersections = self.find_intersections(v_lines, h_lines)
        cell_size = self.get_cell_size(v_lines, h_lines)
        edges = self.find_edges(v_lines, h_lines, cell_size)
        white_stones, black_stones = self.nn_stone_recognizer.recognize(board_img_gray, cell_size, intersections)

        if len(white_stones) == 0 and len(black_stones) == 0:
            raise self.NoBoardError()
                                                        ##todo: remove radius?
        return intersections, white_stones, black_stones, cell_size//2, x_size, y_size, edges

    def get_edges(self, img_gray):
        return cv2.Canny(img_gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

    def all_lines(self, img_gray, hough_threshold, gap=0):
        edges = self.get_edges(img_gray)
        size = min(img_gray.shape[0:2])
        if gap == 0:
            gap = size
        lines = cv2.HoughLinesP(edges, rho=HOUGH_RHO, theta=HOUGH_THETA, threshold=hough_threshold,
                                minLineLength=size * MIN_LINE_LENGHT_COEFF, maxLineGap=gap)
        if lines is None:
            return None
        return np.reshape(lines, (lines.shape[0], lines.shape[2]))

    def get_verticals_horizontals(self, img_gray, hough_threshold, gap=0):
        # Find all lines
        lines = self.all_lines(img_gray, hough_threshold, gap)
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
        return self.merge_lines(v_lines, True), self.merge_lines(h_lines, False)

    # IMPROVE!
    def merge_lines(self, lines, is_vertical):

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

    def lines_recognition(self, img_gray):

        hough_threshold = min(img_gray.shape[0:2])
        cell_size = 0
        while hough_threshold > HOUGH_LOW_THRESHOLD:
            clear_v_lines, clear_h_lines = self.get_verticals_horizontals(img_gray, hough_threshold)
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
            clear_v_lines, clear_h_lines = self.get_verticals_horizontals(img_gray, hough_threshold, 1.5*cell_size)
            if (len(clear_v_lines) <= 1) or (len(clear_h_lines) <= 1):
                hough_threshold += HOUGH_THRESHOLD_STEP
                break
            dists_x = np.diff(clear_v_lines[:, 0])
            dists_y = np.diff(clear_h_lines[:, 1])
            dists = np.concatenate([dists_x, dists_y])
            fracs = np.modf(dists / cell_size)[0]
            ints = np.round(dists / cell_size)
            if np.any(np.logical_and(fracs < 0.75, fracs > 0.25)) or \
                    np.any(ints == 0):
                hough_threshold += HOUGH_THRESHOLD_STEP
                break
            cell_size = np.mean(dists[ints == 1])
        clear_v_lines, clear_h_lines = self.get_verticals_horizontals(img_gray, hough_threshold, 1.5*cell_size)
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

    def get_cell_size(self, v_lines, h_lines):
        return round(np.mean([round(np.ptp(v_lines[:, 0]) / (v_lines.shape[0] - 1)),
                              round(np.ptp(h_lines[:, 1]) / (h_lines.shape[0] - 1))]))

    def find_intersections(self, v_lines, h_lines):
        return np.array(np.meshgrid(v_lines[:, 0], h_lines[:, 1])).T

    def find_edges(self, v_lines, h_lines, cell_size):

        up_line = h_lines[0]
        up_edge = np.sum(up_line[1] - v_lines[:, 1] > cell_size * MIN_EDGE_COEFF) < (v_lines.shape[0] / 1.5)

        down_line = h_lines[-1]
        down_edge = np.sum(v_lines[:, 3] - down_line[1] > cell_size * MIN_EDGE_COEFF) < (v_lines.shape[0] / 1.5)

        left_line = v_lines[0]
        left_edge = np.sum(left_line[0] - h_lines[:, 0] > cell_size * MIN_EDGE_COEFF) < (h_lines.shape[0] / 1.5)

        right_line = v_lines[-1]
        right_edge = np.sum(h_lines[:, 2] - right_line[0] > cell_size * MIN_EDGE_COEFF) < (h_lines.shape[0] / 1.5)
        return up_edge, down_edge, left_edge, right_edge

    def find_circles(self, img_gray, param2, min_dist_coeff, min_r_coeff, max_r_coeff, cell_size):
        circles = cv2.HoughCircles(img_gray, method=METHOD, dp=DP,
                                   minDist=round(cell_size * min_dist_coeff),
                                   param1=PARAM1, param2=param2, minRadius=round(cell_size * min_r_coeff),
                                   maxRadius=round(cell_size * max_r_coeff))
        if circles is None:
            return None
        return circles[0, :]

    def stones_recognition(self, img_gray, cell_size, intersections):
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
            circles = self.find_circles(img_gray, param2, min_dist_coeff, min_r_coeff, max_r_coeff, cell_size)
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

    def colorize(self, img_gray, stones, radius):
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

    def split_into_boards(self, page_img):

        page_img_gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
        edges = self.get_edges(page_img_gray)
        # For better connectivity
        for i in range(LOG_SCALE):
            edges = cv2.pyrDown(edges)
        scale = 2 ** LOG_SCALE
        min_size = (min(page_img.shape[0:2]) * MIN_BOARD_SIZE_COEFF) // scale
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter contours (big and long enough)
        boards = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if perimeter > 4 * min_size and area > min_size ** 2:
                x, y, w, h = cv2.boundingRect(contour)
                boards.append((scale * x, scale * y, scale * w, scale * h))
        board_images = []
        for board in boards:
            x, y, w, h = board
            board_img = page_img[max(y - S, 0): min(y + h + S, page_img.shape[0]),
                        max(x - S, 0): min(x + w + S, page_img.shape[1])]
            board_images.append(board_img)
        return board_images
