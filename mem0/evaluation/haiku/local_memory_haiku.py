"""
Local Memory backend using AWS Bedrock (Claude Haiku) for BOTH:
  - add phase  (memory extraction / summarization)
  - search phase (answer generation, via search_bedrock.py)

Drop-in replacement for evaluation/src/memzero/local_memory.py in the haiku eval.
"""
import os

from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig


def _register_bedrock_haiku_provider():
    """Register 'bedrock_haiku' provider into mem0's LlmFactory.provider_to_class."""
    from mem0.utils.factory import LlmFactory
    from mem0.configs.llms.base import BaseLlmConfig
    if "bedrock_haiku" not in LlmFactory.provider_to_class:
        LlmFactory.provider_to_class["bedrock_haiku"] = (
            "evaluation.haiku.bedrock_llm.BedrockHaikuLLM",
            BaseLlmConfig,
        )


def get_local_memory_haiku_config():
    """MemoryConfig: HuggingFace embedder + Bedrock Haiku LLM + Qdrant."""
    _register_bedrock_haiku_provider()

    embedder_model = os.getenv("EMBEDDER_MODEL", "BAAI/bge-m3")
    base_dir = os.getenv("MEM0_DIR") or os.path.join(os.path.expanduser("~"), ".mem0")
    qdrant_path = os.path.join(base_dir, "qdrant_eval")
    os.makedirs(qdrant_path, exist_ok=True)
    embedding_dims = int(os.getenv("EMBEDDER_DIMS", "1024"))

    llm_config = LlmConfig(
        provider="bedrock_haiku",
        config={"model": "bedrock-haiku-inference-profile", "max_tokens": 1024},
    )

    return MemoryConfig(
        embedder=EmbedderConfig(
            provider="huggingface",
            config={"model": embedder_model},
        ),
        llm=llm_config,
        vector_store=VectorStoreConfig(
            provider="qdrant",
            config={
                "path": qdrant_path,
                "embedding_model_dims": embedding_dims,
                "collection_name": "mem0_eval",
                "on_disk": True,
            },
        ),
    )


def create_local_memory_haiku():
    """Create a Memory instance using Bedrock Haiku for add and search."""
    from mem0 import Memory
    config = get_local_memory_haiku_config()
    return Memory(config=config)


def _qdrant_lock_path():
    base_dir = os.getenv("MEM0_DIR") or os.path.join(os.path.expanduser("~"), ".mem0")
    return os.path.join(base_dir, "qdrant_eval", ".lock")


def create_local_memory_haiku_with_retry():
    """Create Haiku Memory; on Qdrant lock error, remove stale .lock and retry."""
    import time
    from mem0 import Memory
    config = get_local_memory_haiku_config()
    lock_path = _qdrant_lock_path()

    def try_create():
        return Memory(config=config)

    try:
        return try_create()
    except RuntimeError as e:
        err_str = str(e) + str(getattr(e, "__cause__", ""))
        if "already accessed" not in err_str and "AlreadyLocked" not in err_str:
            raise
        for attempt in range(2):
            if os.path.isfile(lock_path):
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
            if attempt == 1:
                time.sleep(2)
            try:
                return try_create()
            except RuntimeError:
                if attempt == 0:
                    continue
                raise RuntimeError(
                    "Qdrant lock conflict. Close other processes and retry, or remove: " + lock_path
                ) from e
        raise
