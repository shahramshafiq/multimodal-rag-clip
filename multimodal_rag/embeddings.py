"""Step 3: CLIP turns text and images into vectors in one shared space. It runs locally and costs nothing."""
import logging

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

log = logging.getLogger("embeddings")

CLIP_SPECIAL_TOKENS = 2  # the start and end tokens CLIP adds to every text


class ClipEmbedder:
    def __init__(self, model_name):
        log.info("loading %s", model_name)
        try:
            self.model = CLIPModel.from_pretrained(model_name).eval()
            self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(f"Could not load the CLIP model {model_name}: {exc}") from exc

    def count_tokens(self, text):
        return len(self.processor.tokenizer(text, truncation=False)["input_ids"])

    @staticmethod
    def _unit_vectors(features):
        # some transformers versions wrap the vectors in an output object
        tensor = features if torch.is_tensor(features) else features.pooler_output
        tensor = tensor / tensor.norm(dim=-1, keepdim=True)
        return tensor.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embed_texts(self, texts):
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=77)
        return self._unit_vectors(self.model.get_text_features(**inputs))

    @torch.no_grad()
    def embed_images(self, image_paths):
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = self.processor(images=images, return_tensors="pt")
        return self._unit_vectors(self.model.get_image_features(**inputs))
