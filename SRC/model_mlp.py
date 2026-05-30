import torch
import torch.nn as nn


class SmartCV_MLP(nn.Module):
    """
    Multi-Layer Perceptron cho phân loại 6 ngành (Công nghệ + Marketing).
    Kiến trúc 2 lớp ẩn: Input → 64 → 32 → num_classes (~66K params)
    Giảm từ 3 lớp (138K params) → tránh ghi nhớ dataset nhỏ.
    """

    def __init__(self, input_size=1000, num_classes=6):
        super(SmartCV_MLP, self).__init__()

        # Hidden layer 1: input → 64
        self.fc1      = nn.Linear(input_size, 64)
        self.bn1      = nn.BatchNorm1d(64)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.4)

        # Hidden layer 2: 64 → 32
        self.fc2      = nn.Linear(64, 32)
        self.bn2      = nn.BatchNorm1d(32)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)

        # Output layer: 32 → num_classes
        self.out = nn.Linear(32, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        return self.out(x)
