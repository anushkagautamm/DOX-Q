"""
Small local embedding wrapper for the MiniLM sentence-transformer model.

This avoids importing sentence-transformers at app startup, which is brittle
with the current transformers stack, while keeping the same mean-pool +
normalize behavior used by all-MiniLM-L6-v2.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class MiniLMEmbeddings:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self._model = AutoModel.from_pretrained(self.model_name, local_files_only=True)
        except OSError:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)

        self._model.eval()
        return self._tokenizer, self._model

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        tokenizer, model = self._load()
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        with torch.inference_mode():
            output = model(**encoded)

        token_embeddings = output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            vectors.extend(self._embed_batch(texts[start:start + batch_size]).tolist())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0].tolist()


embeddings = MiniLMEmbeddings()
