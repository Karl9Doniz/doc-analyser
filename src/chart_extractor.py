import base64
import io
import logging
import os
import threading
from typing import Any, Dict, Optional

from PIL import Image

try:
    from llama_cpp import Llama  # type: ignore
    from llama_cpp.llama_chat_format import (  # type: ignore
        Llava15ChatHandler,
        MiniCPMv26ChatHandler,
    )
except ImportError:  # pragma: no cover - optional dependency
    Llama = None
    MiniCPMv26ChatHandler = None  # type: ignore
    Llava15ChatHandler = None  # type: ignore

logger = logging.getLogger(__name__)


class MiniCPMVEngine:
    """Singleton wrapper around llama-cpp for MiniCPM-V style models."""

    _instance: Optional[Llama] = None
    _chat_handler = None
    _lock = threading.Lock()
    _model_key: Optional[tuple] = None

    @classmethod
    def is_available(cls) -> bool:
        return Llama is not None

    @classmethod
    def get(
        cls,
        model_path: str,
        mmproj_path: str,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
    ) -> Optional[Llama]:
        if not cls.is_available():
            return None

        key = (model_path, mmproj_path, n_ctx, n_threads)
        with cls._lock:
            if cls._instance is not None and cls._model_key == key:
                return cls._instance

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"MiniCPM-V model not found: {model_path}")
            if not os.path.exists(mmproj_path):
                raise FileNotFoundError(f"MiniCPM-V projector not found: {mmproj_path}")

            if MiniCPMv26ChatHandler is not None:
                chat_handler = MiniCPMv26ChatHandler(clip_model_path=mmproj_path)
                handler_name = "MiniCPMv26ChatHandler"
            elif Llava15ChatHandler is not None:
                chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
                handler_name = "Llava15ChatHandler"
            else:
                raise RuntimeError(
                    "llama-cpp-python is missing CLIP chat handlers; "
                    "reinstall with multimodal support (CMAKE_ARGS=\"-DLLAMA_CLIP=ON\")."
                )

            logger.info(
                "Loading MiniCPM-V model from %s with handler %s",
                model_path,
                handler_name,
            )
            cls._instance = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=0,
                logits_all=False,
            )
            cls._chat_handler = chat_handler
            cls._model_key = key
            return cls._instance


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_minicpm_vqa(
    image: Image.Image,
    question: str,
    model_path: str,
    mmproj_path: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    n_ctx: int = 2048,
    n_threads: Optional[int] = None,
) -> Dict[str, Any]:
    """Run MiniCPM-V (via llama-cpp) on an image crop and return raw response."""

    engine = MiniCPMVEngine.get(
        model_path=model_path,
        mmproj_path=mmproj_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
    )
    if engine is None:
        raise RuntimeError(
            "MiniCPM-V runtime unavailable. Ensure llama-cpp-python is installed "
            "with CLIP support."
        )

    encoded = _encode_image(image)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded}"
                    },
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

    response = engine.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})
    answer = message.get("content", "")

    return {
        "answer": answer,
        "raw_response": response,
        "model_path": model_path,
        "mmproj_path": mmproj_path,
    }
