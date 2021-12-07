import cv2 as cv
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import time
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset
from PIL import Image


import torchvision.models
import os


class VGGnet(nn.Module):
    def __init__(self, name="VGG_NET"):
        self.name = name
        super(VGGnet, self).__init__()

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.fc4 = nn.Linear(1000, 3)

    def forward(self, x):
        x = x.view(-1, 512 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=1)
        # x = x.squeeze(1) # Flatten to [batch_size]
        return x


# Now we use VGG-16 to pre-extract some features

# test_loader = torch.utils.data.DataLoader(test_data_set, batch_size = 1, num_workers = 1, shuffle=True)
# train_loader = torch.utils.data.DataLoader(train_data_set, batch_size = 1, num_workers = 1, shuffle=True)
# validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 1, num_workers = 1, shuffle=True)

# Below is the similiar code as training to generate the features
def get_features(net, dataloader, target):
    seq = 0  # This is the sequence number of the image
    for image, label in dataloader:
        label = ["with", "without", "wrong"][label]
        features = net.features(image)  # This is the new input tensor to be saved to target folder
        if not os.path.exists(target):
            os.mkdir(target)
        if not os.path.exists(os.path.join(target, label)):
            os.mkdir(os.path.join(target, label))
        features = features.squeeze(0)
        features = torch.from_numpy(features.detach().numpy())
        torch.save(features, os.path.join(target, label, str(seq)))
        seq = seq + 1


if __name__ == "__main__":
    vgg16 = torchvision.models.vgg16(pretrained=True)

    classifier = VGGnet()
    state = torch.load("/Users/guozhending/Desktop/model_VGG4_bs64_lr0.001_epoch49", map_location=torch.device('cpu'))
    classifier.load_state_dict(state)

    cap = cv.VideoCapture(0)
    face_detect = cv.CascadeClassifier(
        r'/Users/guozhending/Desktop/cascade (10).xml')
    if True:
        face_detect = cv.CascadeClassifier(
            r'/Users/guozhending/Desktop/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

    while True:
        flag, frame = cap.read()
        frame = cv.flip(frame, 1)
        if not flag:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face_zone = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        for x, y, w, h in face_zone:
            # print(x, y, w, h)
            image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)[y:y + h, x:x + w])
            image = image.resize((224, 224), Image.BILINEAR)
            image = np.array(image)
            image = transforms.Compose([transforms.ToTensor()])(img=image)
            # image = [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
            # image = torch.tensor(image)
            image = image.unsqueeze(0)
            features = vgg16.features(image.float())
            output = classifier(features)
            print(output.max(1, keepdim=True)[1][0])
            cv.rectangle(frame, pt1=(x + 2, y + 2), pt2=(x + w + 2, y + h + 2), color=[255, 0, 0], thickness=2)

            if (output.max(1, keepdim=True))[1][0] == 0:
                cv.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=[0, 255, 0], thickness=2)
            elif (output.max(1, keepdim=True))[1][0] == 1:
                cv.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=[0, 0, 255], thickness=2)
            else:
                cv.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=[0, 255, 255], thickness=2)

            # cv.circle(frame, center=(x + w // 2, y + h // 2), radius=w // 2, color=[0, 255, 0], thickness=2)
        cv.imshow('video', frame)
        if ord('q') == cv.waitKey(30):
            break

    cv.destroyAllWindows()
    cap.release()
