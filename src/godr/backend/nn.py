import numpy as np
import cv2
import onnxruntime
import pkg_resources


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class StoneRecognizer():

    def __init__(self):
        bytes_model = bytes(pkg_resources.resource_string('godr.backend.models', "model-2.onnx"))
        self.model = onnxruntime.InferenceSession(bytes_model)

    def transform(self, img):
        img = img.astype(np.float32) / 255
        img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
        img = (img - 0.5) / 0.5
        img = np.expand_dims(img, axis=[0, 1])
        return img

    def recognize(self, board_img, cell_size, intersections):
        CELL_SIZE_COEFF = 0.6

        delta = int(cell_size * CELL_SIZE_COEFF)
        white_stones = []
        black_stones = []

        for i in range(intersections.shape[0]):
            for j in range(intersections.shape[1]):
                y = intersections[i][j][0]
                x = intersections[i][j][1]
                point_img = board_img[max(x - delta, 0): min(x + delta, board_img.shape[0]),
                                      max(y - delta, 0): min(y + delta, board_img.shape[1])]
                data = self.transform(point_img)
                input = {self.model.get_inputs()[0].name: data}
                output = self.model.run(None, input)[0]
                pred = np.argmax(output, axis = 1)
                if pred == 0:
                    white_stones.append((i , j))
                elif pred == 1:
                    black_stones.append((i , j))
        return white_stones, black_stones
