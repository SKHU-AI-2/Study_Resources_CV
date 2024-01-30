# -*- coding: utf-8 -*-
"""명예학회_4주차_ResNet코드.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1K6YVKnHwHGAJXj-kTvy0rGCBLFwsvYOt

#ResNet 라이브러리
"""

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# 사전 훈련된 ResNet 모델 불러오기 (예: ResNet18)
model = models.resnet18(pretrained=True)
model.eval()

# 이미지를 처리하기 위한 변환 정의
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 이미지 불러오기 및 변환
img = Image.open('path_to_your_image.jpg')  # 이미지 경로를 지정하세요
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# 모델에 이미지 전달 및 예측
with torch.no_grad():
    out = model(batch_t)

# 결과 출력
_, indices = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(indices[0], percentage[indices[0]].item())

"""#ResNet 직접 코딩"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 기본 잔차 블록 정의
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 첫 번째 합성곱 층
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 첫 번째 배치 정규화
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 활성화 함수
        self.relu = nn.ReLU(inplace=True)
        # 두 번째 합성곱 층
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # 두 번째 배치 정규화
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 다운샘플링 층, 필요한 경우에만 사용
        self.downsample = downsample

    # 잔차 블록의 순전파 정의
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 전체 ResNet 모델 정의
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # 초기 합성곱 층
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 초기 배치 정규화
        self.bn1 = nn.BatchNorm2d(64)
        # 초기 활성화 함수
        self.relu = nn.ReLU(inplace=True)
        # 최대 풀링 층
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 잔차 블록 층들
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 평균 풀링 층
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 완전 연결 층
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    # 잔차 블록 층 생성 함수
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    # ResNet의 순전파 정의
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x# ResNet 모델의 각 부분에 대한 설명을 추가한 코드입니다.

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# 모델 생성
model = resnet18()
print(model)