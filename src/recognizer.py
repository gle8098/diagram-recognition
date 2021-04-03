import cv2
import numpy as np
from collections import defaultdict
from src.recognizer_consts import *

class Recognizer:
    def __init__(self):
        pass
        
    def recognize(self, img_gray):
        v_lines, h_lines = self.lines_recognition(img_gray)
        x_size, y_size = h_lines.shape[0], v_lines.shape[0]
        intersections = self.find_intersections(v_lines, h_lines)
        cell_size = self.get_cell_size(v_lines, h_lines)
        edges = self.find_edges(v_lines, h_lines, cell_size)
        stones, radius = self.stones_recognition(img_gray, cell_size, intersections)
        white_stones, black_stones = self.colorize(img_gray, stones, radius)
        return intersections, white_stones, black_stones, radius, x_size, y_size, edges
        
    def to_RGB(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def get_edges(self, img_gray):
        #CANNY_THRESHOLD1 = 100
        #CANNY_THRESHOLD2 = 200
        return cv2.Canny(img_gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    
    def all_lines(self, img_gray, hough_threshold):
        #HOUGH_RHO = 1
        #HOUGH_THETA = np.pi / 180

        edges = self.get_edges(img_gray)
        size = min(img_gray.shape[0:2])
        lines = cv2.HoughLinesP(edges, rho=HOUGH_RHO, theta=HOUGH_THETA, threshold=hough_threshold,
                                minLineLength=size/2, maxLineGap=size)  ### ?
        if lines is None:
            return None
        return np.reshape(lines, (lines.shape[0], lines.shape[2]))
    
    def get_verticals_horizontals(self, img_gray, hough_threshold):
        #VERTICAL_TAN_MIN = 50
        #HORIZONTAL_TAN_MAX = 0.02

        # Find all lines
        lines = self.all_lines(img_gray, hough_threshold)
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
        #MIN_DIST = 10  ???

        if (lines.shape[0] == 0):
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
        #HOUGH_LOW_THRESHOLD = 50
        #HOUGH_THRESHOLD_STEP = 15

        clear_v_lines, clear_h_lines = [], []
        hough_threshold = min(img_gray.shape[0:2])
        while hough_threshold > HOUGH_THRESHOLD_STEP:
            clear_v_lines, clear_h_lines = self.get_verticals_horizontals(img_gray, hough_threshold)
            if ((len(clear_v_lines) > 1) and (len(clear_h_lines) > 1)):
                dists_x = np.diff(clear_v_lines[:, 0])
                dists_y = np.diff(clear_h_lines[:, 1])
                cell_size_1 = np.amin(dists_x)
                cell_size_2 = np.amin(dists_y)
                cell_size = min(cell_size_1, cell_size_2)
                if (cell_size / max(cell_size_1, cell_size_2) > 0.9):
                    dists = np.concatenate([dists_x, dists_y])
                    fracs = np.modf(dists / cell_size)[0]
                    cell_size = np.mean(dists[np.round(dists / cell_size) == 1])
                    while hough_threshold > HOUGH_THRESHOLD_STEP:
                        hough_threshold -= HOUGH_THRESHOLD_STEP
                        clear_v_lines, clear_h_lines = self.get_verticals_horizontals(img_gray, hough_threshold)
                        dists_x = np.diff(clear_v_lines[:, 0])
                        dists_y = np.diff(clear_h_lines[:, 1])
                        dists = np.concatenate([dists_x, dists_y])
                        fracs = np.modf(dists / cell_size)[0]
                        if np.any(np.logical_and(fracs < 0.8, fracs > 0.2)):
                            hough_threshold += HOUGH_THRESHOLD_STEP
                            clear_v_lines, clear_h_lines = self.get_verticals_horizontals(img_gray, hough_threshold)
                            break
                        else:
                            cell_size = np.mean(dists[np.round(dists / cell_size) == 1])
                    break
            hough_threshold -= HOUGH_THRESHOLD_STEP
        # Add unclear lines
        # possible bugs!

        unclear_v_lines, unclear_h_lines = self.get_verticals_horizontals(img_gray, HOUGH_LOW_THRESHOLD)
        x_min = min(unclear_v_lines[0][0], clear_v_lines[0][0])
        y_min = min(unclear_h_lines[0][1], clear_h_lines[0][1])
        x_max = max(unclear_v_lines[-1][0], clear_v_lines[-1][0])
        y_max = max(unclear_h_lines[-1][1], clear_h_lines[-1][1])

        x_min_line = np.mean(clear_h_lines[:, 0])
        x_max_line = np.mean(clear_h_lines[:, 2])
        y_min_line = np.mean(clear_v_lines[:, 1])
        y_max_line = np.mean(clear_v_lines[:, 3])

        v_lines = []
        xs = np.concatenate([[x_min], clear_v_lines[:, 0], [x_max]])
        dists = np.diff(xs)
        num_lines = (dists / cell_size).astype(int)
        num_lines[np.modf(dists / cell_size)[0] > 0.8] += 1
        num_lines[1:-1][num_lines[1:-1] != 0] -= 1
        for i in range(1, xs.size):
            for ind in range(1, num_lines[i - 1] + 1):
                if i == xs.size - 1:
                    x = xs[i - 1] + np.round(ind * dists[-1] / num_lines[-1])
                elif i == 1:
                    x = xs[i] - np.round((num_lines[0] + 1 - ind) * dists[0] / num_lines[0])
                else:
                    x = xs[i] - np.round((num_lines[i - 1] + 1 - ind) * dists[i - 1] / (num_lines[i - 1] + 1))
                new_line = [x, y_min_line, x, y_max_line]
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
                    y = ys[i - 1] + np.round(ind * dists[-1] / num_lines[-1])
                elif i == 1:
                    y = ys[i] - np.round((num_lines[0] + 1 - ind) * dists[0] / num_lines[0])
                else:
                    y = ys[i] - np.round((num_lines[i - 1] + 1 - ind) * dists[i - 1] / (num_lines[i - 1] + 1))
                new_line = [x_min_line, y, x_max_line, y]
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
        #MIN_DIFF = 0.3

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
        #METHOD = cv2.HOUGH_GRADIENT
        #DP = 2
        #PARAM1 = 200

        circles = cv2.HoughCircles(img_gray, method=METHOD, dp=DP,
                                minDist=round(cell_size * min_dist_coeff),
                                param1=PARAM1, param2=param2, minRadius=round(cell_size * min_r_coeff),
                                maxRadius=round(cell_size * max_r_coeff))
        if circles is None:
            return None
        return circles[0, :]

    def stones_recognition(self, img_gray, cell_size, intersections):
        #MIN_INTERSECTION_DIST_COEFF = 0.3

        all_circles = []
        param2_grid = np.linspace(5, 30, 6)
        min_dist_coeff_grid = np.linspace(0.9, 0.9, 1)
        min_r_coeff_grid = np.linspace(0.4, 0.45, 2)
        max_r_coeff_grid = np.linspace(0.5, 0.55, 2)
        grid = np.array(np.meshgrid(param2_grid,
                                    min_dist_coeff_grid,
                                    min_r_coeff_grid,
                                    max_r_coeff_grid)
                       ).T.reshape(-1,4)
        for param2,min_dist_coeff,min_r_coeff, max_r_coeff  in grid:
            circles = self.find_circles(img_gray,param2,min_dist_coeff,min_r_coeff, max_r_coeff, cell_size)
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
        #WHITE_THRESHOLD = 250
        #BLACK_THRESHOLD = 5

        white_stones = []
        black_stones = []
        for stone in stones.values():
            stone_mask = np.zeros((img_gray.shape[0], img_gray.shape[1]), np.uint8)
            cv2.circle(stone_mask,(stone[0],stone[1]), radius // 2, 255 ,-1)
            average_color = cv2.mean(img_gray, mask=stone_mask)[0]
            if average_color >= WHITE_THRESHOLD:
                white_stones.append(stone)
            elif average_color <= BLACK_THRESHOLD:
                black_stones.append(stone)
        return np.array(white_stones), np.array(black_stones)

    def get_split_lines(self, lines, is_vertical):
        MIN_RATIO = 0.75

        if is_vertical:
            gaps = np.diff(lines[:, 0])
        else:
            gaps = np.diff(lines[:, 1])
        max_gap = np.amax(gaps)
        split_lines = []
        if is_vertical:
            split_line = lines[0] - [max_gap // 2, 0, max_gap // 2, 0]
        else:
            split_line = lines[0] - [0, max_gap // 2, 0, max_gap // 2]
        split_lines.append(split_line)
        if np.mean(gaps) / max_gap < MIN_RATIO:
            for i in range(gaps.shape[0]):
                if gaps[i] / max_gap >= MIN_RATIO:
                    split_line = np.mean(lines[i:i + 2], axis=0).astype(int)
                    split_lines.append(split_line)
        if is_vertical:
            split_line = lines[-1] + [max_gap // 2, 0, max_gap // 2, 0]
        else:
            split_line = lines[-1] + [0, max_gap // 2, 0, max_gap // 2]
        split_lines.append(split_line)
        return np.array(split_lines)

    def split_into_boards(self, page_img):
        HOUGH_THRESHOLD = 200

        page_img_gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
        v_lines, h_lines = self.get_verticals_horizontals(page_img_gray[10:, 10:], HOUGH_THRESHOLD)
        v_lines = self.get_split_lines(v_lines, True)
        h_lines = self.get_split_lines(h_lines, False)
        intersections = self.find_intersections(v_lines, h_lines)
        board_images = []
        for i in range(intersections.shape[0] - 1):
            for j in range(intersections.shape[1] - 1):
                board_img = (page_img_gray[intersections[i][j][1] + 10:intersections[i + 1][j + 1][1] + 10,
                                           intersections[i][j][0] + 10:intersections[i + 1][j + 1][0] + 10])
                board_images.append(board_img)
        return board_images

