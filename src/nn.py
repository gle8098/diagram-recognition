import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F


class StoneRecognizer():

    class ConvNet(nn.Module):
        def __init__(self):
            super(StoneRecognizer.ConvNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc1 = nn.Linear(2048, 3)

        def forward(self, x):
            # torch.Size([1, 32, 32])
            x = self.conv1(x)
            x = F.relu(x)

            # torch.Size([16, 32, 32])
            x = self.conv2(x)
            x = F.relu(x)

            # torch.Size([32, 32, 32])
            x = F.max_pool2d(x, 4)

            # torch.Size([32, 8, 8]))

            x = torch.flatten(x, 1)

            # torch.Size([2048])
            x = self.fc1(x)
            output = F.log_softmax(x, dim=1)

            # torch.Size([3])
            return output

    def __init__(self):
        self.device = torch.device("cpu")
        self.model = self.ConvNet()
        self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5,), (0.5,))])

    def recognize(self, img_gray, cell_size, intersections):
        CELL_SIZE_COEFF = 0.6

        delta = int(cell_size * CELL_SIZE_COEFF)
        white_stones = []
        black_stones = []
        for intersection in np.reshape(intersections, (-1, 2)):
            y = intersection[0]
            x = intersection[1]
            point_img = img_gray[max(x - delta, 0): min(x + delta, img_gray.shape[0]),
                        max(y - delta, 0): min(y + delta, img_gray.shape[1])]
            data = self.transform(point_img).to(self.device)
            output = self.model(data.unsqueeze(1))
            pred = output.argmax(dim=1, keepdim=True).item()
            if pred == 0:
                white_stones.append(intersection)
            elif pred == 1:
                black_stones.append(intersection)
        return white_stones, black_stones
