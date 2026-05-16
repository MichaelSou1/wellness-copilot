import argparse
import os
import importlib
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.md",
    "*.py",
    "*.safetensors",
    "sentencepiece.bpe.model",
    "*.model",
    "tokenizer*",
    "vocab.*",
    "modules.json",
]

KNOWN_CHECKPOINT_MODES = {
    "BAAI/bge-m3": "bin",
    "BAAI/bge-reranker-v2-m3": "safetensors",
    "BAAI/bge-small-zh-v1.5": "safetensors",
}


def _torch_loads_bin_safely() -> bool:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return False

    version = torch.__version__.split("+", 1)[0]
    parts = []
    for raw_part in version.split(".")[:2]:
        digits = "".join(ch for ch in raw_part if ch.isdigit())
        parts.append(int(digits or 0))
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts) >= (2, 6)


def _cached_repo_files(model_name: str, cache_dir: str | None) -> list[str]:
    if "/" not in model_name:
        return []

    candidates = []
    if cache_dir:
        candidates.append(Path(cache_dir))

    env_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env_cache:
        candidates.append(Path(env_cache))

    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    repo_dir_name = f"models--{model_name.replace('/', '--')}"
    for hub_cache in candidates:
        snapshots_dir = hub_cache / repo_dir_name / "snapshots"
        if not snapshots_dir.exists():
            continue
        files = []
        for snapshot in snapshots_dir.iterdir():
            if not snapshot.is_dir():
                continue
            files.extend(
                str(p.relative_to(snapshot))
                for p in snapshot.rglob("*")
                if p.is_file()
            )
        if files:
            return files

    return []


def _repo_files(model_name: str, endpoint: str | None, cache_dir: str | None) -> list[str]:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return [str(p.relative_to(model_path)) for p in model_path.rglob("*") if p.is_file()]

    cached_files = _cached_repo_files(model_name, cache_dir)
    if any(name.endswith((".safetensors", ".bin")) for name in cached_files):
        return cached_files

    huggingface_hub = importlib.import_module("huggingface_hub")
    api_kwargs = {}
    if endpoint:
        api_kwargs["endpoint"] = endpoint
    api = huggingface_hub.HfApi(**api_kwargs)

    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            return api.list_repo_files(model_name)
        except Exception as e:
            if attempt == attempts:
                if cached_files:
                    print(
                        "[Model Download][warn] Repo file lookup failed; "
                        f"using local cache for {model_name}."
                    )
                    return cached_files
                raise
            print(
                "[Model Download][warn] Repo file lookup failed "
                f"({e.__class__.__name__}: {e}); retrying {attempt}/{attempts - 1}..."
            )
            time.sleep(2 * attempt)


def _checkpoint_mode(
    model_name: str,
    endpoint: str | None,
    cache_dir: str | None,
    allow_bin: bool,
) -> str:
    known_mode = KNOWN_CHECKPOINT_MODES.get(model_name)
    if known_mode == "safetensors":
        return known_mode
    if known_mode == "bin":
        if allow_bin or _torch_loads_bin_safely():
            return known_mode
        raise RuntimeError(
            f"{model_name} does not provide safetensors, and this environment has "
            "torch<2.6. Upgrade torch to >=2.6, choose a safetensors model such as "
            "BAAI/bge-small-zh-v1.5, or rerun with --allow-bin only after upgrading."
        )

    files = _repo_files(model_name, endpoint, cache_dir)
    if "model.safetensors" in files or any(
        name.endswith(".safetensors") for name in files
    ):
        return "safetensors"

    if "pytorch_model.bin" in files or any(name.endswith(".bin") for name in files):
        if allow_bin or _torch_loads_bin_safely():
            return "bin"
        raise RuntimeError(
            f"{model_name} does not provide safetensors, and this environment has "
            "torch<2.6. Upgrade torch to >=2.6, choose a safetensors model such as "
            "BAAI/bge-small-zh-v1.5, or rerun with --allow-bin only after upgrading."
        )

    raise RuntimeError(f"No supported checkpoint file found for {model_name}.")


def _resolve_checkpoint_mode(
    model_name: str,
    endpoint: str | None,
    cache_dir: str | None,
    allow_bin: bool,
    fallback_model: str | None = None,
) -> tuple[str, str]:
    try:
        return model_name, _checkpoint_mode(model_name, endpoint, cache_dir, allow_bin)
    except RuntimeError as primary_error:
        if not fallback_model or fallback_model == model_name:
            raise

        print(f"[Model Download][warn] {primary_error}")
        print(f"[Model Download][warn] Trying fallback embedding model: {fallback_model}")
        return fallback_model, _checkpoint_mode(
            fallback_model,
            endpoint,
            cache_dir,
            allow_bin,
        )


def _snapshot_download(
    model_name: str,
    cache_dir: str | None,
    endpoint: str | None,
    checkpoint_mode: str,
) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path.resolve())

    huggingface_hub = importlib.import_module("huggingface_hub")
    snapshot_download = huggingface_hub.snapshot_download

    allow_patterns = list(DEFAULT_ALLOW_PATTERNS)
    ignore_patterns = ["*.msgpack", "*.h5", "*.onnx", "*.ot"]
    if checkpoint_mode == "bin":
        allow_patterns.append("*.bin")
        ignore_patterns.append("*.safetensors")
    else:
        ignore_patterns.append("*.bin")

    kwargs = {
        "repo_id": model_name,
        "allow_patterns": allow_patterns,
        "ignore_patterns": ignore_patterns,
        "max_workers": 1,
        "etag_timeout": 60,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if endpoint:
        kwargs["endpoint"] = endpoint

    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            path = snapshot_download(**kwargs)
            _validate_snapshot(path, checkpoint_mode)
            return path
        except Exception as e:
            if attempt == attempts:
                raise
            print(
                "[Model Download][warn] Snapshot download failed "
                f"({e.__class__.__name__}: {e}); retrying {attempt}/{attempts - 1}..."
            )
            time.sleep(2 * attempt)

    raise RuntimeError(f"Failed to download snapshot for {model_name}.")


def _validate_snapshot(model_path: str, checkpoint_mode: str) -> None:
    path = Path(model_path)
    if checkpoint_mode == "safetensors":
        has_checkpoint = any(path.glob("*.safetensors"))
        expected_name = "model.safetensors"
    else:
        has_checkpoint = any(path.glob("*.bin"))
        expected_name = "pytorch_model.bin"

    if not has_checkpoint:
        raise RuntimeError(
            f"Incomplete snapshot at {path}: missing {expected_name}."
        )


def _load_sentence_transformer(model_path: str, device: str, checkpoint_mode: str):
    sentence_transformers = importlib.import_module("sentence_transformers")
    model_kwargs = {"use_safetensors": True} if checkpoint_mode == "safetensors" else {}
    return sentence_transformers.SentenceTransformer(
        model_path,
        device=device,
        local_files_only=True,
        model_kwargs=model_kwargs,
    )


def _load_cross_encoder(model_path: str, device: str, checkpoint_mode: str):
    sentence_transformers = importlib.import_module("sentence_transformers")
    model_kwargs = {"use_safetensors": True} if checkpoint_mode == "safetensors" else {}
    return sentence_transformers.CrossEncoder(
        model_path,
        device=device,
        max_length=256,
        local_files_only=True,
        model_kwargs=model_kwargs,
    )


def main():
    from health_guide.config import (
        RAG_EMBED_MODEL_NAME,
        RAG_FALLBACK_EMBED_MODEL_NAME,
        RAG_RERANK_MODEL_NAME,
    )

    parser = argparse.ArgumentParser(
        description="Download RAG embedding and reranker models for offline use."
    )
    parser.add_argument(
        "--embed-model",
        default=RAG_EMBED_MODEL_NAME,
        help="Embedding model name or local path",
    )
    parser.add_argument(
        "--rerank-model",
        default=RAG_RERANK_MODEL_NAME,
        help="Reranker model name or local path",
    )
    parser.add_argument(
        "--fallback-embed-model",
        default=RAG_FALLBACK_EMBED_MODEL_NAME,
        help="Fallback embedding model if the primary model cannot be loaded safely",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional Hugging Face cache dir (e.g. ./.hf_cache)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Download/load device for warmup: cpu or cuda",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="Hugging Face endpoint, default uses hf-mirror for users in mainland China",
    )
    parser.add_argument(
        "--disable-mirror",
        action="store_true",
        help="Disable mirror and use default huggingface endpoint",
    )
    parser.add_argument(
        "--enable-xet",
        action="store_true",
        help="Allow Hugging Face Xet downloads. Disabled by default for unstable proxy/mirror networks.",
    )
    parser.add_argument(
        "--allow-bin",
        action="store_true",
        help="Allow PyTorch .bin checkpoints when torch>=2.6. Safetensors are preferred when available.",
    )
    args = parser.parse_args()

    endpoint = None
    if not args.disable_mirror:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        # 兼容部分库读取的备用变量
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = args.hf_endpoint
        endpoint = args.hf_endpoint
        print(f"[Model Download] Using mirror endpoint: {args.hf_endpoint}")
    else:
        print("[Model Download] Mirror disabled. Using default Hugging Face endpoint.")

    if not args.enable_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        print("[Model Download] Xet downloads disabled. Use --enable-xet to opt in.")

    if args.cache_dir:
        cache_dir = str(Path(args.cache_dir).resolve())
        os.environ["HF_HOME"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"[Model Download] Cache dir: {cache_dir}")
    else:
        cache_dir = None

    if args.allow_bin:
        print("[Model Download] PyTorch .bin checkpoints allowed.")
    else:
        print("[Model Download] Safetensors preferred; .bin requires torch>=2.6.")

    print(f"[Model Download] Embedding model: {args.embed_model}")
    embed_model_name, embed_checkpoint_mode = _resolve_checkpoint_mode(
        args.embed_model,
        endpoint,
        cache_dir,
        args.allow_bin,
        args.fallback_embed_model,
    )
    if embed_model_name != args.embed_model:
        print(f"[Model Download] Embedding model selected: {embed_model_name}")
    print(f"[Model Download] Embedding checkpoint type: {embed_checkpoint_mode}")
    embed_path = _snapshot_download(embed_model_name, cache_dir, endpoint, embed_checkpoint_mode)
    print(f"[Model Download] Embedding cached at: {embed_path}")
    embed_model = _load_sentence_transformer(embed_path, args.device, embed_checkpoint_mode)
    _ = embed_model.encode(["模型下载检查"], show_progress_bar=False)

    print(f"[Model Download] Reranker model: {args.rerank_model}")
    rerank_checkpoint_mode = _checkpoint_mode(
        args.rerank_model,
        endpoint,
        cache_dir,
        args.allow_bin,
    )
    print(f"[Model Download] Reranker checkpoint type: {rerank_checkpoint_mode}")
    rerank_path = _snapshot_download(args.rerank_model, cache_dir, endpoint, rerank_checkpoint_mode)
    print(f"[Model Download] Reranker cached at: {rerank_path}")
    reranker = _load_cross_encoder(rerank_path, args.device, rerank_checkpoint_mode)
    _ = reranker.predict([["query", "passage"]], show_progress_bar=False)

    print("[Model Download] Done. Models are cached and ready for offline runs.")


if __name__ == "__main__":
    main()
