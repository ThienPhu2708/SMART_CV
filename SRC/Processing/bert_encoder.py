"""
Sentence-Transformer encoder wrapper.
Encode text → 768-dim L2-normalized embedding.
Hỗ trợ tiếng Anh + tiếng Việt (multilingual model).
"""
import numpy as np

MODEL_NAME  = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM   = 768

_encoder = None


def get_encoder():
    """Lazy-load sentence-transformer (download ~120MB lần đầu)."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        print(f"[BERTEncoder] Loading '{MODEL_NAME}' ...")
        _encoder = SentenceTransformer(MODEL_NAME)
        print("[BERTEncoder] Ready.")
    return _encoder


def encode_texts(texts: list, batch_size: int = 64) -> np.ndarray:
    """
    Encode danh sách văn bản thành ma trận (N, 768).
    normalize_embeddings=True: L2-normalize → cosine similarity.
    """
    enc = get_encoder()
    embeddings = enc.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def encode_single(text: str) -> np.ndarray:
    """Encode 1 văn bản → shape (1, 768). Dùng cho inference."""
    enc = get_encoder()
    emb = enc.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)
