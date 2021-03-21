import cv2
import numpy as np
from bisect import bisect
from .recognizer_consts import *

class Recognizer:
    def __init__(self):
        pass
        
    def recognize(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
                                minLineLength=size / 2, maxLineGap=size)  ### ?
        if lines is None:
            return None
        return np.reshape(lines, (lines.shape[0], lines.shape[2]))
    
    def get_verticals_horizontals(self, img_gray, hough_threshold):
        #VERTICAL_TAN_MIN = 50
        #HORIZONTAL_TAN_MAX = 0.02

        # Find all lines
        lines = self.all_lines(img_gray, hough_threshold)
        if (lines is None):
            return [], []
        # Divide the lines into verticals and horizontals
        v_lines = []
        h_lines = []
        for line in lines:
            x1, y1, x2, y2 = line
            if x1 == x2:
                if y1 > y2:
                    line[0] = x2
                    line[1] = y2
                    line[2] = x1
                    line[3] = y1
                v_lines.append(line)
            else:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > VERTICAL_TAN_MIN:
                    if y1 > y2:
                        line[0] = x2
                        line[1] = y2
                        line[2] = x1
                        line[3] = y1
                elif abs(slope) < HORIZONTAL_TAN_MAX:
                    if x1 > x2:
                        line[0] = x2
                        line[1] = y2
                        line[2] = x1
                        line[3] = y1
                    h_lines.append(line)
        v_lines = np.array(v_lines)
        h_lines = np.array(h_lines)
        # Merge close lines
        return self.merge_lines(self.merge_lines(v_lines, True), True), self.merge_lines(self.merge_lines(h_lines, False), False)

    # IMPROVE!
    def merge_lines(self, lines, is_vertical):
        # ???
        #MIN_DIST = 10

        close_groups = []
        grouped = set()
        for i in range(lines.shape[0]):
            if i in grouped:
                continue
            close_groups.append([i])
            grouped.add(i)
            for j in set(range(i + 1, lines.shape[0])) - grouped:
                x1_i, y1_i, x2_i, y2_i = lines[i]
                x1_j, y1_j, x2_j, y2_j = lines[j]
                if is_vertical:
                    dist = min(abs(x1_i - x1_j), abs(x2_i - x2_j))
                else:
                    dist = min(abs(y1_i - y1_j), abs(y2_i - y2_j))
                if (dist <= MIN_DIST):
                    close_groups[-1].append(j)
                    grouped.add(j)
        merged_lines = []
        for group in close_groups:
            line_group = lines[group]
            if is_vertical:
                x = round(np.mean(line_group[:, [0, 2]]))
                merged_line = (x, np.amin(line_group[:, 1]), x, np.amax(line_group[:, 3]))
            else:
                y = round(np.mean(line_group[:, [1, 3]]))
                merged_line = (np.amin(line_group[:, 0]), y, np.amax(line_group[:, 2]), y)
            merged_lines.append(merged_line)

        if is_vertical:
            return np.array(sorted(merged_lines, key=lambda line: line[0]))
        else:
            return np.array(sorted(merged_lines, key=lambda line: line[1]))

    def lines_recognition(self, img_gray):
        #HOUGH_LOW_THRESHOLD = 50
        #MIN_DIST_COEFF = 0.7
        #MIN_GAP_COEFF = 1.5

        clear_v_lines, clear_h_lines = [], []
        hough_threshold = min(img_gray.shape[0:2])
        while hough_threshold > 0:
            clear_v_lines, clear_h_lines = self.get_verticals_horizontals(img_gray, hough_threshold)
            if ((len(clear_v_lines) > 1) and (len(clear_h_lines) > 1)):
                cell_size_1 = np.amin(np.diff(clear_v_lines[:, 0]))
                cell_size_2 = np.amin(np.diff(clear_h_lines[:, 1]))
                cell_size = min(cell_size_1, cell_size_2)
                if (cell_size / max(cell_size_1, cell_size_2) > 0.9):
                    break
            hough_threshold -= 10
        unclear_v_lines, unclear_h_lines = self.get_verticals_horizontals(img_gray, HOUGH_LOW_THRESHOLD)
        # Filter unclear lines
        v_lines = [clear_v_lines[0]]
        h_lines = [clear_h_lines[0]]
        ind = bisect(unclear_v_lines[:, 0], clear_v_lines[0][0])
        for i in range(ind - 1, -1, -1):
            dist_coeff = (v_lines[-1][0] - unclear_v_lines[i][0]) / cell_size
            if MAX_LINES_DIST_COEFF > dist_coeff > MIN_LINES_DIST_COEFF:
                v_lines.append(unclear_v_lines[i])
            if dist_coeff > MIN_GAP_COEFF:
                x = (unclear_v_lines[i][0] + v_lines[-1][0]) // 2
                new_v_line = [x, v_lines[-1][1], x, v_lines[-1][3]]
                v_lines.append(new_v_line)
                v_lines.append(unclear_v_lines[i])
        v_lines.reverse()
        for i in range(ind, len(unclear_v_lines)):
            dist_coeff = (unclear_v_lines[i][0] - v_lines[-1][0]) / cell_size
            if MAX_LINES_DIST_COEFF > dist_coeff > MIN_LINES_DIST_COEFF:
                v_lines.append(unclear_v_lines[i])
            if dist_coeff > MIN_GAP_COEFF:
                x = (unclear_v_lines[i][0] + v_lines[-1][0]) // 2
                new_v_line = [x, v_lines[-1][1], x, v_lines[-1][3]]
                v_lines.append(new_v_line)
                v_lines.append(unclear_v_lines[i])
        ind = bisect(unclear_h_lines[:, 1], clear_h_lines[0][1])
        for i in range(ind - 1, -1, -1):
            dist_coeff = (h_lines[-1][1] - unclear_h_lines[i][1]) / cell_size
            if MAX_LINES_DIST_COEFF > dist_coeff > MIN_LINES_DIST_COEFF:
                h_lines.append(unclear_h_lines[i])
            if dist_coeff > MIN_GAP_COEFF:
                y = (unclear_h_lines[i][1] + h_lines[-1][1]) // 2
                new_h_line = [h_lines[-1][0], y, h_lines[-1][2], y]
                h_lines.append(new_h_line)
                h_lines.append(unclear_h_lines[i])
        h_lines.reverse()
        for i in range(ind, len(unclear_h_lines)):
            dist_coeff = (unclear_h_lines[i][1] - h_lines[-1][1]) / cell_size
            if MAX_LINES_DIST_COEFF > dist_coeff > MIN_LINES_DIST_COEFF:
                h_lines.append(unclear_h_lines[i])
            if dist_coeff > MIN_GAP_COEFF:
                y = (unclear_h_lines[i][1] + h_lines[-1][1]) // 2
                new_h_line = [h_lines[-1][0], y, h_lines[-1][2], y]
                h_lines.append(new_h_line)
                h_lines.append(unclear_h_lines[i])
        return np.array(v_lines), np.array(h_lines)
    
    def get_cell_size(self, v_lines, h_lines):
        return round(np.mean([round(np.ptp(v_lines[:, 0]) / (v_lines.shape[0] - 1)),
                              round(np.ptp(h_lines[:, 1]) / (h_lines.shape[0] - 1))]))
    
    def find_intersections(self, v_lines, h_lines):
        return np.array(np.meshgrid(v_lines[:, 0], h_lines[:, 1])).T

    def find_edges(self, v_lines, h_lines, cell_size):
        #MIN_DIFF = 0.2

        up_line = h_lines[0]
        up_edge = np.sum(up_line[1] - v_lines[:, 1] > cell_size * MIN_COEFF) < (v_lines.shape[0] / 1.5)

        down_line = h_lines[-1]
        down_edge = np.sum(v_lines[:, 3] - down_line[1] > cell_size * MIN_COEFF) < (v_lines.shape[0] / 1.5)

        left_line = v_lines[0]
        left_edge = np.sum(left_line[0] - h_lines[:, 0] > cell_size * MIN_COEFF) < (h_lines.shape[0] / 1.5)

        right_line = v_lines[-1]
        right_edge = np.sum(h_lines[:, 2] - right_line[0] > cell_size * MIN_COEFF) < (h_lines.shape[0] / 1.5)
        return up_edge, down_edge, left_edge, right_edge

    def find_circles(self, img_gray, param2, min_dist_coeff, min_r_coeff, max_r_coeff, cell_size):
        #METHOD = cv2.HOUGH_GRADIENT
        #DP = 2
        #PARAM1 = 200
        #PARAM2 = 15
        #MIN_DIST_COEFF = 0.9
        #MAX_R_COEFF = 0.5
        #MIN_R_COEFF = 0.5

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
        param2_grid = np.linspace(15, 30, 3)
        min_dist_coeff_grid = np.linspace(0.9, 1, 2)
        min_r_coeff_grid = np.linspace(0.35, 0.45, 3)
        max_r_coeff_grid = np.linspace(0.5, 0.55, 2)
        grid = np.array(np.meshgrid(param2_grid,
                                    min_dist_coeff_grid,
                                    min_r_coeff_grid,
                                    max_r_coeff_grid)).T.reshape(-1,4)
        for param2,min_dist_coeff,min_r_coeff, max_r_coeff in grid:
            circles = self.find_circles(img_gray,param2,min_dist_coeff,min_r_coeff, max_r_coeff, cell_size)
            if circles is not None:
                all_circles.append(circles)
        circles = np.concatenate(all_circles)
        # Filter circles
        stones = []
        radii = []
        for circle in circles:
            in_intersection = False
            for intersection in np.reshape(intersections, (-1, 2)):
                if (np.linalg.norm(circle[:2] - intersection) <= MIN_INTERSECTION_DIST_COEFF * cell_size):
                    in_intersection = True
                    break
            if in_intersection:
                stones.append(intersection)
                radii.append(circle[2])
        return np.unique(stones, axis=0), round(np.mean(radii))
    
    def colorize(self, img_gray, stones, radius):
        #WHITE_THRESHOLD = 250
        #BLACK_THRESHOLD = 5

        white_stones = []
        black_stones = []
        for stone in stones:
            average_color = np.mean(img_gray[stone[1] - radius // 2: stone[1] + radius // 2,
                                    stone[0] - radius // 2: stone[0] + radius // 2])
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

    def split_into_boards(self, page_img_gray):
        HOUGH_THRESHOLD = 200

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
        return np.array(board_images, dtype=object)

