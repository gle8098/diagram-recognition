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
        intersections = self.find_intersections(v_lines, h_lines)
        cell_size = self.get_cell_size(v_lines, h_lines)
        stones, radius = self.stones_recognition(img_gray, cell_size, intersections)
        white_stones, black_stones = self.colorize(img_gray, stones, radius)
        return intersections, cell_size, white_stones, black_stones, radius
        
    def to_RGB(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def get_edges(self, img_gray):
        CANNY_THRESHOLD1 = 100
        CANNY_THRESHOLD2 = 200
        return cv2.Canny(img_gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    
    def all_lines(self, img_gray):
        CANNY_THRESHOLD1 = 100
        CANNY_THRESHOLD2 = 200
        HOUGH_RHO = 1
        HOUGH_THETA = np.pi/180
        HOUGH_THRESHOLD = 100
        edges = cv2.Canny(img_gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
        size = min(img_gray.shape[0:2]) 
        lines = cv2.HoughLinesP(edges, rho=HOUGH_RHO, theta=HOUGH_THETA, threshold=HOUGH_THRESHOLD,
                                minLineLength=size/2, maxLineGap=size)
        return np.reshape(lines, (lines.shape[0], lines.shape[2]))
    
    def get_verticals_horizontals(self, img_gray, hough_threshold):
        CANNY_THRESHOLD1 = 100
        CANNY_THRESHOLD2 = 200
        HOUGH_RHO = 1
        HOUGH_THETA = np.pi/180
        HOUGH_HIGH_THRESHOLD = 50
        HOUGH_LOW_THRESHOLD = 50
        VERTICAL_TAN_MIN = 50
        HORIZONTAL_TAN_MAX = 0.02

        # Find all lines
        edges = cv2.Canny(img_gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
        size = min(img_gray.shape[0:2]) 
        lines = cv2.HoughLinesP(edges, rho=HOUGH_RHO, theta=HOUGH_THETA, threshold=hough_threshold,
                                minLineLength=size/2, maxLineGap=size/2) #???
        if (lines is None):
            return [], []
        lines = np.reshape(lines, (lines.shape[0], lines.shape[2]))
        # Divide the lines into verticals and horizontals
        v_lines = []
        h_lines = []
        for line in lines:
            x1, y1, x2, y2 = line
            if (x1 == x2):
                v_lines.append(line)
            else:
                slope = (y2 - y1)/(x2 - x1)
                if (abs(slope) > VERTICAL_TAN_MIN):
                    v_lines.append(line)
                elif (abs(slope) < HORIZONTAL_TAN_MAX):
                    h_lines.append(line)
        v_lines = np.array(v_lines)
        h_lines = np.array(h_lines)
        # Merge close lines
        return self.merge_lines(v_lines, True), self.merge_lines(h_lines, False)
    
    def merge_lines(self, lines, is_vertical):
        # ???
        MIN_DIST = 10

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
            # possible bag
            if is_vertical:
                x = round(np.mean(line_group[:, [0, 2]]))
                merged_line = (x, np.amax(line_group[:, 1]), x, np.amin(line_group[:, 3]))
            else:
                y = round(np.mean(line_group[:, [1, 3]]))
                merged_line = (np.amin(line_group[:, 0]), y, np.max(line_group[:, 2]), y)
            merged_lines.append(merged_line)

        if is_vertical:
            return np.array(sorted(merged_lines, key=lambda line:line[0]))
        else:
            return np.array(sorted(merged_lines, key=lambda line:line[1]))
    
    def lines_recognition(self, img_gray):
        HOUGH_LOW_THRESHOLD = 50
        HOUGH_HIGH_THRESHOLD = 200
        MIN_DIST_COEFF = 0.7
        MIN_GAP_COEFF = 1.5

        clear_v_lines, clear_h_lines = [], []
        while (len(clear_v_lines) < 2) or (len(clear_h_lines) < 2):
            clear_v_lines, clear_h_lines = self.get_verticals_horizontals(img_gray, HOUGH_HIGH_THRESHOLD)
            HOUGH_HIGH_THRESHOLD -= 10
        cell_size = min(np.amin(np.diff(clear_v_lines[:, 0])), np.amin(np.diff(clear_h_lines[:, 1])))
        unclear_v_lines, unclear_h_lines = self.get_verticals_horizontals(img_gray, HOUGH_LOW_THRESHOLD)
        # Filter unclear lines
        v_lines = [clear_v_lines[0]]
        h_lines = [clear_h_lines[0]]
        ind = bisect(unclear_v_lines[:, 0], clear_v_lines[0][0])
        for i in range(ind - 1, -1, -1):
            dist_coeff = (v_lines[-1][0] - unclear_v_lines[i][0])/cell_size
            if dist_coeff > MIN_DIST_COEFF:
                if dist_coeff > MIN_GAP_COEFF:
                    x = (unclear_v_lines[i][0] + v_lines[-1][0]) // 2
                    new_v_line = [x, v_lines[-1][1], x, v_lines[-1][3]]
                    v_lines.append(new_v_line)
                v_lines.append(unclear_v_lines[i])
        v_lines.reverse()
        for i in range(ind, len(unclear_v_lines)):
            dist_coeff = (unclear_v_lines[i][0] - v_lines[-1][0])/cell_size
            if dist_coeff > MIN_DIST_COEFF:
                if dist_coeff > MIN_GAP_COEFF:
                    x = (unclear_v_lines[i][0] + v_lines[-1][0]) // 2
                    new_v_line = [x, v_lines[-1][1], x, v_lines[-1][3]]
                    v_lines.append(new_v_line)
                v_lines.append(unclear_v_lines[i])
        ind = bisect(unclear_h_lines[:, 1], clear_h_lines[0][1])
        for i in range(ind - 1, -1, -1):
            dist_coeff = (h_lines[-1][1] - unclear_h_lines[i][1])/cell_size
            if dist_coeff > MIN_DIST_COEFF:
                if dist_coeff > MIN_GAP_COEFF:
                    y = (unclear_h_lines[i][1] + h_lines[-1][1]) // 2
                    new_h_line = [h_lines[-1][0], y, h_lines[-1][2], y]
                    h_lines.append(new_h_line)
                h_lines.append(unclear_h_lines[i])
        h_lines.reverse()
        for i in range(ind, len(unclear_h_lines)):
            dist_coeff = (unclear_h_lines[i][1] - h_lines[-1][1])/cell_size
            if dist_coeff > MIN_DIST_COEFF:
                if dist_coeff > MIN_GAP_COEFF:
                    y = (unclear_h_lines[i][1] + h_lines[-1][1]) // 2
                    new_h_line = [h_lines[-1][0], y, h_lines[-1][2], y]
                    h_lines.append(new_h_line)
                h_lines.append(unclear_h_lines[i])
        return np.array(v_lines), np.array(h_lines)
    
    def get_cell_size(self, v_lines, h_lines):
        return round(np.mean([round(np.ptp(v_lines[:, 0]) / v_lines.shape[0] - 1),
                              round(np.ptp(h_lines[:, 1]) / h_lines.shape[0] - 1)]))
    
    def find_intersections(self, v_lines, h_lines):
        return np.array(np.meshgrid(v_lines[:, 0], h_lines[:, 1])).T.reshape(-1, 2)
    
    def find_circles(self, img_gray, cell_size, intersections):
        METHOD = cv2.HOUGH_GRADIENT
        DP = 2
        PARAM1 = 200
        PARAM2 = 20
        MIN_DIST_COEFF = 0.9
        MAX_R_COEFF = 0.5
        MIN_R_COEFF = 0.5

        return cv2.HoughCircles(img_gray, method=METHOD, dp=DP,
                                   minDist=round(cell_size * MIN_DIST_COEFF),
                                   param1=PARAM1, param2=PARAM2, minRadius=round(cell_size * MIN_R_COEFF),
                                   maxRadius=round(cell_size * MAX_R_COEFF))[0, :]
    
    def stones_recognition(self, img_gray, cell_size, intersections):
        METHOD = cv2.HOUGH_GRADIENT
        DP = 2
        PARAM1 = 200
        PARAM2 = 15
        MIN_DIST_COEFF = 0.9
        MAX_R_COEFF = 0.5
        MIN_R_COEFF = 0.5
        MIN_INTERSECTION_DIST_COEFF = 0.3

        circles = cv2.HoughCircles(img_gray, method=METHOD, dp=DP,
                                   minDist=round(cell_size * MIN_DIST_COEFF),
                                   param1=PARAM1, param2=PARAM2, minRadius=round(cell_size * MIN_R_COEFF),
                                   maxRadius=round(cell_size * MAX_R_COEFF))[0, :]
        # Filter circles
        stones = []
        radii = []
        for circle in circles:
            in_intersection = False
            for intersection in intersections:
                if (np.linalg.norm(circle[:2] - intersection) <= MIN_INTERSECTION_DIST_COEFF * cell_size):
                    in_intersection = True
                    break
            if in_intersection:
                stones.append(intersection)
                radii.append(circle[2])
        return stones, round(np.mean(radii))
    
    def colorize(self, img_gray, stones, radius):
        WHITE_THRESHOLD = 255/2
        white_stones = []
        black_stones = []
        for stone in stones:
            average_color = np.mean(img_gray[stone[1] - radius//2 : stone[1] + radius//2,
                                             stone[0] - radius//2 : stone[0] + radius//2])
            if average_color > WHITE_THRESHOLD:
                white_stones.append(stone)
            else:
                black_stones.append(stone)
        return white_stones, black_stones