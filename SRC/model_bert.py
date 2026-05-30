import torch
import torch.nn as nn


class SmartCV_BERT(nn.Module):
    """
    Classification head cho sentence-transformer embeddings.
    Encoder (multilingual BERT) được freeze hoàn toàn — chỉ train head này.

    Input:  768-dim embedding từ paraphrase-multilingual-MiniLM-L12-v2
    Output: logits cho num_classes ngành

    Kiến trúc nhỏ gọn tránh overfitting trên dataset ~800 mẫu:
    768 → 256 → 128 → num_classes
    """

    ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    EMBED_DIM    = 768

    def __init__(self, embed_dim: int = 768, num_classes: int = 6):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.45),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
