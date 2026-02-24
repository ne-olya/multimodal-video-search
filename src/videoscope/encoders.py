import hashlib

import numpy as np


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


class HashTextEncoder:
    """Deterministic dependency-free encoder for tests, never for quality evaluation."""

    def __init__(self, dimension=64):
        self.dimension = dimension

    def encode_texts(self, texts):
        rows = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode()).digest()
                vector[int.from_bytes(digest[:4], "big") % self.dimension] += 1
            rows.append(vector)
        return normalize(rows)


class OpenClipEncoder:
    def __init__(self, model_name="ViT-B-32", weights="laion2b_s34b_b79k", device=None):
        import open_clip
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=weights, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def encode_images(self, images):
        batch = self.torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with self.torch.inference_mode():
            return normalize(self.model.encode_image(batch).float().cpu().numpy())

    def encode_texts(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        with self.torch.inference_mode():
            return normalize(self.model.encode_text(tokens).float().cpu().numpy())


class SentenceEncoder:
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)
