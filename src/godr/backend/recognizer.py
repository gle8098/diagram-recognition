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

        aligned_board_img = self.align_board(board_img)

        if aligned_board_img is None:
            raise self.EmptyImageError()

        canny_edges = self.canny_edges(aligned_board_img)

        v_lines, h_lines = self.lines_recognition(canny_edges)
        if len(v_lines) == 0 or len(h_lines) == 0:
            raise self.NoBoardError()
        x_size, y_size = v_lines.shape[0], h_lines.shape[0]
        intersections = self.find_intersections(v_lines, h_lines)
        cell_size = self.get_cell_size(v_lines, h_lines)

        board_img_nn = self.transform_for_nn(aligned_board_img, cell_size)
        white_stones, black_stones = self.nn_stone_recognizer.recognize(board_img_nn, cell_size, intersections)

        if len(white_stones) == 0 and len(black_stones) == 0:
            raise self.NoBoardError()
        board_edges = self.find_board_edges(v_lines, h_lines, white_stones + black_stones, cell_size)

        return white_stones, black_stones, x_size, y_size, board_edges


    def split_into_boards(self, page_img):
        # Preprocessing page: resize and greyscale
        orig_size = min(page_img.shape[0:2])
        if orig_size > MAX_PAGE_SIZE:
            scale = MAX_PAGE_SIZE / orig_size
        else:
            scale = 1
        page_img_crop = cv2.resize(page_img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Canny edges
        size = min(page_img_crop.shape[0:2])
        edges = self.canny_edges(page_img_crop)

        # For better connectivity
        morph_kernel_size = round(size * MORPH_COEFF)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((morph_kernel_size, morph_kernel_size)))

        # Find contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter boards by perimeter and area
        boards = []
        min_size = size * MIN_BOARD_SIZE_COEFF
        big_contours = set()
        for i in range(len(contours)):
            perimeter = cv2.arcLength(contours[i], True)
            area = cv2.contourArea(contours[i])
            if perimeter > 4 * min_size and area > min_size ** 2:
                big_contours.add(i)

        if len(big_contours) == 0:
            raise self.EmptyBoardsError()

        # Filter frames
        board_contours = set()
        for i in big_contours:
            inner = hierarchy[0][i][2]
            is_board = True
            while inner > 0:
                if inner in big_contours:
                    is_board = False
                    break
                else:
                    inner = hierarchy[0][inner][0]
            if is_board:
                board_contours.add(i)

        for i in board_contours:
            boards.append(cv2.boundingRect(contours[i]))

        # Sort boards
        boards = self.sort_boards(boards, size * SHIFT_COEFF)

        # Get boards
        board_images = []
        for i, board in enumerate(boards):
            x, y, w, h = board
            extra_size = round(min(w, h) * EXTRA_SIZE_COEFF)
            board_img = page_img_crop[max(y - extra_size, 0): min(y + h + extra_size, page_img.shape[0]),
                                      max(x - extra_size, 0): min(x + w + extra_size, page_img.shape[1])]
            board_images.append(board_img)
        return board_images

    def canny_edges(self, img):
        return cv2.Canny(img, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

    def all_lines(self, edges, hough_threshold):
        size = min(edges.shape[0:2])
        lines = cv2.HoughLinesP(edges, rho=HOUGH_RHO, theta=HOUGH_THETA, threshold=hough_threshold,
                                minLineLength=size*MIN_LINE_LEN_COEFF, maxLineGap=size)
        if lines is None:
            return None
        return np.reshape(lines, (lines.shape[0], lines.shape[2]))

    def align_board(self, board_img):
        edges = self.canny_edges(board_img)
        # Find all lines
        lines = self.all_lines(edges, HOUGH_LOW_THRESHOLD)
        if (lines is None):
            return None

        # Get all angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.arctan((y2 - y1) / (x2 - x1)) if x1 != x2 else np.arctan(np.sign(y2 - y1) * np.inf)
            angles.append(angle)

        # Get 2 main angles
        angles.sort()
        new_group = np.diff(angles) < 0.1
        groups = []
        i = 0
        while i < len(angles):
            groups.append([])
            groups[-1].append(angles[i])
            while i + 1 < len(angles) and new_group[i]:
                i += 1
                groups[-1].append(angles[i])
            i += 1
        if len(groups) < 2:
            raise self.NoBoardError()
        groups.sort(key=lambda x: len(x), reverse=True)
        angle_1 = np.median(groups[0])
        angle_2 = np.median(groups[1])

        # Make affine transform to align board
        if abs(angle_1) < abs(angle_2):
            angle_h, angle_v = angle_1, angle_2
        else:
            angle_h, angle_v = angle_2, angle_1
        pts2 = np.float32([[0, 0],
                           [1, 0],
                           [0, np.sign(angle_v)]])
        pts1 = np.float32([[0, 0],
                           [np.cos(angle_h), np.sin(angle_h)],
                           [np.cos(angle_v), np.sin(angle_v)]])
        affine_t = cv2.getAffineTransform(pts1, pts2)
        rows, cols = edges.shape[0:2]
        transformed_board_img = cv2.warpAffine(board_img, affine_t, (cols, rows),
                                               borderMode=cv2.BORDER_REPLICATE)
        return transformed_board_img

    def verticals_horizontals(self, edges, hough_threshold):
        # Find all lines
        lines = self.all_lines(edges, hough_threshold)
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
        size =  min(edges.shape[0:2])
        min_dist = size * MIN_DIST_COEFF
        return self.merge_lines(v_lines, True, min_dist), self.merge_lines(h_lines, False, min_dist)

    def merge_lines(self, lines, is_vertical, min_dist):
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

    def lines_recognition(self, edges):
        # Find some very clear lines
        hough_threshold = min(edges.shape[0:2])
        cell_size = 0
        while hough_threshold > HOUGH_LOW_THRESHOLD:
            clear_v_lines, clear_h_lines = self.verticals_horizontals(edges, hough_threshold)
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
            clear_v_lines, clear_h_lines = self.verticals_horizontals(edges, hough_threshold)
            if (len(clear_v_lines) <= 1) or (len(clear_h_lines) <= 1):
                hough_threshold += HOUGH_THRESHOLD_STEP
                break
            dists_x = np.diff(clear_v_lines[:, 0])
            dists_y = np.diff(clear_h_lines[:, 1])
            dists = np.concatenate([dists_x, dists_y])
            fracs = np.modf(dists / cell_size)[0]
            ints = np.round(dists / cell_size)
            err = 0.9 - 0.15 * ints
            err[ints >= 3] = 0.5
            if np.any(np.logical_and(fracs < err, fracs > 1 - err)):
                hough_threshold += HOUGH_THRESHOLD_STEP
                break
            cell_size = np.mean(dists[ints == 1])
        clear_v_lines, clear_h_lines = self.verticals_horizontals(edges, hough_threshold)

        # Add unclear lines

        # Calculate cell_size
        dists_x = np.diff(clear_v_lines[:, 0])
        dists_y = np.diff(clear_h_lines[:, 1])
        dists = np.concatenate([dists_x, dists_y])
        cell_size = np.mean(dists[np.round(dists / cell_size) == 1])

        # Calculate length of unclear lines
        x_min = np.amin(clear_h_lines[:, 0])
        x_max = np.amax(clear_h_lines[:, 2])
        y_min = np.amin(clear_v_lines[:, 1])
        y_max = np.amax(clear_v_lines[:, 3])

        def add_unclear_lines(clear_lines, is_vertical, min_coord, max_coord, start, end):
            ind = 0 if is_vertical else 1
            lines = []
            coords = np.concatenate([[min_coord], clear_lines[:, ind], [max_coord]])
            dists = np.diff(coords)
            num_lines = (dists / cell_size).astype(int)
            num_lines[np.modf(dists / cell_size)[0] > 0.8] += 1
            num_lines[1:-1][num_lines[1:-1] != 0] -= 1
            for i in range(1, coords.size):
                for j in range(1, num_lines[i - 1] + 1):
                    if i == coords.size - 1:
                        coord = coords[i - 1] + np.round(j * cell_size)
                    elif i == 1:
                        coord = coords[i] - np.round((num_lines[0] + 1 - j) * cell_size)
                    else:
                        coord = coords[i] - np.round((num_lines[i - 1] + 1 - j) * dists[i - 1] / (num_lines[i - 1] + 1))
                    new_line = [coord, start, coord, end] if is_vertical else [start, coord, end, coord]
                    lines.append(new_line)
                if i != coords.size - 1:
                    lines.append(clear_lines[i - 1])
            return np.array(lines).astype(int)

        v_lines = add_unclear_lines(clear_v_lines, True, x_min, x_max, y_min, y_max)
        h_lines = add_unclear_lines(clear_h_lines, False, y_min, y_max, x_min, x_max)
        return v_lines, h_lines

    def get_cell_size(self, v_lines, h_lines):
        return round(np.mean([round(np.ptp(v_lines[:, 0]) / (v_lines.shape[0] - 1)),
                              round(np.ptp(h_lines[:, 1]) / (h_lines.shape[0] - 1))]))

    def find_intersections(self, v_lines, h_lines):
        return np.array(np.meshgrid(v_lines[:, 0], h_lines[:, 1])).T

    def transform_for_nn(self, board_img, cell_size):
        # Convert to black&white
        block_size = 2 * cell_size + 1
        board_img_gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
        board_img_bw = cv2.adaptiveThreshold(board_img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                             block_size, 0)
        board_img_bordered = cv2.copyMakeBorder(board_img_bw, cell_size, cell_size, cell_size, cell_size,
                                                borderType=cv2.BORDER_CONSTANT, value=255)
        return board_img_bordered

    def find_board_edges(self, v_lines, h_lines, stones, cell_size):
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

    def sort_boards(self, boards, shift):
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
        boards_order = np.concatenate(board_lines)
        return np.array(boards)[boards_order]

