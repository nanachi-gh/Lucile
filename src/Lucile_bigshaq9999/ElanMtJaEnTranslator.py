from transformers import pipeline
from typing import List
import gc
import logging
import torch

logger = logging.getLogger(__name__)


class ElanMtJaEnTranslator:
    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.elan_model = "tiny"
        self.model_id = "Mitsua/elan-mt-tiny-ja-en"

    def load_model(self, device="auto", elan_model="tiny"):
        if device == "auto" or device is None:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target_device = device

        self.elan_model = elan_model

        model_map = {
            "bt": "Mitsua/elan-mt-bt-ja-en",
            "base": "Mitsua/elan-mt-base-ja-en",
            "tiny": "Mitsua/elan-mt-tiny-ja-en",
        }
        if elan_model not in model_map:
            raise ValueError(
                f"Invalid elan model: {elan_model}, please choose from 'bt', 'base', 'tiny'"
            )

        self.model_id = model_map[elan_model]
        logger.info(f"Loading translation pipeline {self.model_id} on {target_device}")

        try:
            self.model = pipeline(
                "translation_ja_to_en",
                model=self.model_id,
                framework="pt",
                device=target_device,
            )
            self.device = target_device
        except Exception as e:
            if target_device != "cpu":
                logger.warning(
                    f"Failed to load translation pipeline on {target_device}: {e}. Falling back to CPU."
                )
                self.device = "cpu"
                self.model = pipeline(
                    "translation_ja_to_en",
                    model=self.model_id,
                    framework="pt",
                    device="cpu",
                )
            else:
                raise e

    def predict(self, source_texts: List[str]) -> List[str]:
        """
        Takes a list of source strings and returns a list of translated strings.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if isinstance(source_texts, str):
            source_texts = [source_texts]

        if not source_texts:
            return []

        try:
            results = self.model(source_texts)
            return [res["translation_text"] for res in results]
        except Exception as e:
            if self.device != "cpu":
                logger.warning(
                    f"Translation failed on {self.device}: {e}. Falling back to CPU."
                )
                try:
                    self.device = "cpu"
                    self.model = pipeline(
                        "translation_ja_to_en",
                        model=self.model_id,
                        framework="pt",
                        device="cpu",
                    )
                    results = self.model(source_texts)
                    return [res["translation_text"] for res in results]
                except Exception as cpu_e:
                    logger.error(f"CPU fallback translation failed: {cpu_e}")
                    return ["" for _ in source_texts]
            else:
                logger.error(f"Translation failed: {e}")
                return ["" for _ in source_texts]

    def unload_model(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = None
        self.device = "cpu"
        print("Model unloaded")

