import cv2
import glob
import os.path
from src.board import Board

if __name__ == '__main__':
    print('Enter pathname')
    path = input()
    img_files = glob.glob(path)
    if img_files:
        for img_file in img_files:
            extension = os.path.splitext(img_file)[1]
            if not extension in ['.png', '.jpg']:
                continue
            print('\nProcessing ' + img_file)
            try:
                img = cv2.imread(img_file)
                board = Board(img)
                index = img_file.rfind('.')
                sgf_file = img_file[:index+1] + 'sgf'
                board.save_sgf(sgf_file)
                print(img_file + ' converted successfully')
            except:
                print('An error occured while converting ' + img_file)
