import numpy as np
import cv2
import matplotlib.pyplot as plt
from .stone import Stone
from .recognizer import Recognizer
import copy

class Board:
    def __init__(self, img):
        recognizer = Recognizer()
        debug = False
        if debug:
            self.intersections, self.cell_size, white_stones, black_stones, self.radius = recognizer.recognize(img=img)
        else:
            try:
                self.intersections, self.cell_size, white_stones, black_stones, self.radius = recognizer.recognize(img=img)
            except:
                return
        self.img = img
        self.white_stones = []
        self.black_stones = []
        
        min_x = np.min(self.intersections.T[0])
        min_y = np.min(self.intersections.T[1])
        for stone in white_stones:
            global_x = stone[0]
            global_y = stone[1]
            local_x = round((global_x - min_x)/self.cell_size)
            local_y = round((global_y - min_y)/self.cell_size)
            self.white_stones.append(Stone(local_x, local_y, global_x, global_y))
        for stone in black_stones:
            global_x = stone[0]
            global_y = stone[1]
            local_x = round((global_x - min_x)/self.cell_size)
            local_y = round((global_y - min_y)/self.cell_size)
            self.black_stones.append(Stone(local_x, local_y, global_x, global_y))
    
    def to_RGB(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def visualize(self):
        try:
            visualization = copy.copy(self.img)
            for intersection in self.intersections:
                cv2.circle(visualization, (intersection[0], intersection[1]), 5, (255,0,255), -1)
            for stone in self.white_stones:
                cv2.circle(visualization, (stone.global_x,stone.global_y), self.radius,(0,0,255),3)
            for stone in self.black_stones:
                cv2.circle(visualization, (stone.global_x,stone.global_y), self.radius,(255,0,0),3)
            
            plt.figure(figsize=(20, 10))
            
            plt.subplot(1, 2, 1)
            plt.imshow(self.to_RGB(self.img))
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(self.to_RGB(visualization))
            plt.axis('off')
            plt.show()
        except:
            return