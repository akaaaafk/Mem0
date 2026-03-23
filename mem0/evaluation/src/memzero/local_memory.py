"""
Local Memory backend: use mem0 Memory (embedder + LLM + vector store) instead of Mem0 API.
When MEM0_API_KEY is not set, evaluation uses this so you can run with e.g.:
  - Embedder: BAAI/bge-m3 (or baai/bge-3b) via EMBEDDER_MODEL
  - LLM: Ollama via LLM_MODEL (e.g. llama3.2)
No MEM0 API key needed.
"""
import os

from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig


def get_local_memory_config():
    """Build MemoryConfig for local run: HuggingFace embedder (e.g. BAAI/bge-m3 or bge-3b) + LLM (Ollama or OpenAI-compatible e.g. Tinker) + Qdrant."""
    embedder_model = os.getenv("EMBEDDER_MODEL", "BAAI/bge-m3")
    base_dir = os.getenv("MEM0_DIR") or os.path.join(os.path.expanduser("~"), ".mem0")
    qdrant_path = os.path.join(base_dir, "qdrant_eval")
    os.makedirs(qdrant_path, exist_ok=True)

    # bge-m3 / bge-3b typical dims; override with EMBEDDER_DIMS if needed
    embedding_dims = int(os.getenv("EMBEDDER_DIMS", "1024"))

    # Tinker 或其它 OpenAI 兼容 backend：优先读 TINKER_*，否则 OPENAI_*
    openai_base = os.getenv("TINKER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    openai_key = os.getenv("TINKER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if openai_base and openai_key:
        llm_model = os.getenv("MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
        llm_config = LlmConfig(
            provider="openai",
            config={
                "model": llm_model,
                "temperature": 0.1,
                "api_key": openai_key,
                "openai_base_url": openai_base.rstrip("/"),
            },
        )
    else:
        llm_model = os.getenv("LLM_MODEL", "llama3.2")
        llm_config = LlmConfig(
            provider="ollama",
            config={
                "model": llm_model,
                "temperature": 0.1,
                "ollama_base_url": os.getenv("OLLAMA_BASE_URL"),
            },
        )

    config = MemoryConfig(
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
        # graph_store uses default (no graph) so no MEM0 API key needed
    )
    return config


def create_local_memory():
    """Create a Memory instance with local config (no API key)."""
    from mem0 import Memory

    config = get_local_memory_config()
    return Memory(config=config)


def _qdrant_lock_path():
    base_dir = os.getenv("MEM0_DIR") or os.path.join(os.path.expanduser("~"), ".mem0")
    return os.path.join(base_dir, "qdrant_eval", ".lock")


def create_local_memory_with_retry():
    """Create local Memory; on Qdrant lock error, remove stale .lock and retry (with optional short wait)."""
    import time
    from mem0 import Memory

    config = get_local_memory_config()
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
                    print(f"[mem0] 已删除锁文件，重试连接 ({attempt + 1}/2): {lock_path}")
                except OSError as oe:
                    print(f"[mem0] 无法删除锁文件（可能被占用）: {oe}")
            if attempt == 1:
                time.sleep(2)
            try:
                return try_create()
            except RuntimeError:
                if attempt == 0:
                    continue
                raise RuntimeError(
                    "Qdrant 本地库被占用。请按顺序尝试：\n"
                    "1) 关闭所有运行过 add/search 的终端，以及 Cursor 里可能占用 Python 的进程；\n"
                    "2) 在【新开】的 PowerShell 中执行（删锁后立刻运行）：\n"
                    f"   Remove-Item -Force \"{lock_path}\" -ErrorAction SilentlyContinue; "
                    "python -m evaluation.run_experiments --technique_type mem0 --method add_then_search --score\n"
                    "3) 若仍失败，用任务管理器结束所有 python.exe 后再试。"
                ) from e
        raise
