import torch
import numpy
import cv2

from Net.nn.model import NetRF, NetVGG, Net

torch.manual_seed(99)
numpy.random.seed(99)

score1 = torch.ones((1, 1, 240, 320))

model = Net()

print(model(score1).shape)



