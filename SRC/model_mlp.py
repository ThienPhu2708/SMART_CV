import torch
import torch.nn as nn


class SmartCV_MLP(nn.Module):
    """
    Multi-Layer Perceptron cho bài toán phân loại ngành nghề CV.
    Kiến trúc: Input → 256 → 128 → 64 → num_classes
    Mỗi hidden layer dùng: BatchNorm → ReLU → Dropout
    Dropout tăng so với v1 để chống overfit trên dataset nhỏ (~3800 mẫu).
    """

    def __init__(self, input_size=1500, num_classes=24):
        super(SmartCV_MLP, self).__init__()

        # Hidden layer 1: input → 256
        self.fc1      = nn.Linear(input_size, 256)
        self.bn1      = nn.BatchNorm1d(256)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.6)

        # Hidden layer 2: 256 → 128
        self.fc2      = nn.Linear(256, 128)
        self.bn2      = nn.BatchNorm1d(128)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)

        # Hidden layer 3: 128 → 64
        self.fc3      = nn.Linear(128, 64)
        self.bn3      = nn.BatchNorm1d(64)
        self.relu3    = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.4)

        # Output layer: 64 → num_classes (không activation — CrossEntropyLoss tự dùng softmax)
        self.out = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self):
        """He initialization phù hợp với ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.fc3(x))))
        return self.out(x)
