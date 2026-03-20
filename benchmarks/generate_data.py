"""
Generate benchmark artifacts for rvq-index from the pretrained RQ-VAE Amazon Beauty model.

Usage:
  uv run python generate_data.py [--max-items N] [--query-count Q] [--top-k K] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import collections
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


DATASET = "jhan21/amazon-beauty-reviews-dataset"
SENTENCE_MODEL = "sentence-transformers/sentence-t5-base"
CHECKPOINT = "edobotta/rqvae-amazon-beauty"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Item(NamedTuple):
    item_id: str
    title: str
    brand: str
    review_count: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of indexed items (sorted by review count). Default: all.",
    )
    p.add_argument("--query-count", type=int, default=500, help="Number of held-out query items. Default: 500.")
    p.add_argument("--top-k", type=int, default=10, help="K for brute-force ground truth. Default: 10.")
    p.add_argument("--out-dir", type=Path, default=Path("data"), help="Output directory. Default: ./data")

    return p.parse_args()


def load_beauty_items(max_items: int | None) -> list[Item]:
    """
    Returns a list of Item namedtuples sorted by descending review count.
    Only items with >=5 reviews are kept (matching the paper).
    """
    logger.info("Loading Amazon Beauty reviews...")
    reviews = load_dataset(DATASET, split="train")

    logger.info(f" - Sample review row: {reviews[0]}")

    # Count reviews per item
    counts: dict[str, int] = collections.Counter(reviews["asin"])

    logger.info("Loading Amazon Beauty metadata...")
    meta = load_dataset(DATASET, split="train")

    items = []
    for row in meta:
        asin = row["parent_asin"]
        if counts.get(asin, 0) < 5:
            continue
        title = (row.get("title") or "").strip()
        brand = (row.get("store") or row.get("brand") or "Unknown").strip()
        if not title:
            continue
        items.append(
            Item(
                item_id=asin,
                title=title,
                brand=brand,
                review_count=counts[asin],
            )
        )

    # Sort by review count descending, then cap
    items.sort(key=lambda x: x.review_count, reverse=True)
    if max_items is not None:
        items = items[:max_items]

    logger.info(f" - {len(items)} items after filtering.")

    return items


def load_rqvae_checkpoint() -> dict[str, torch.Tensor]:
    logger.info("Downloading RQ-VAE checkpoint...")
    path = hf_hub_download(repo_id=CHECKPOINT, filename="model.safetensors")

    return load_file(path)


def extract_codebooks(state_dict: dict[str, torch.Tensor], device: str) -> torch.Tensor:
    """
    Extract codebook weight matrices from the state dict.
    Returns tensor of shape (num_books, num_codes, dim).

    edobotta/rqvae-amazon-beauty keys:
      layers.0.embedding.weight  [256, 32]
      layers.1.embedding.weight  [256, 32]
      layers.2.embedding.weight  [256, 32]
    """

    layers = []
    for key in (0, 1, 2):
        weight = state_dict.get(f"layers.{key}.embedding.weight")

        if weight is None:
            raise RuntimeError(f"Missing expected encoder weight: layers.{key}.embedding.weight")

        layers.append(weight)

    codebooks = torch.stack(layers, dim=0).to(device)  # (L, K, D)
    logger.info(f" - Codebooks shape: {codebooks.shape}")

    return codebooks


def build_encoder(state_dict: dict[str, torch.Tensor], device: str) -> torch.nn.Module:
    """
    Reconstruct the encoder MLP from the state dict.

    edobotta/rqvae-amazon-beauty encoder (no biases):
      encoder.mlp.0.weight  [512, 768]
      encoder.mlp.2.weight  [256, 512]
      encoder.mlp.4.weight  [128, 256]
      encoder.mlp.6.weight  [ 32, 128]
    Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear
    """

    layers: list[torch.nn.Module] = []
    for key in (0, 2, 4, 6):
        weight = state_dict.get(f"encoder.mlp.{key}.weight")

        if weight is None:
            raise RuntimeError(f"Missing expected encoder weight: encoder.mlp.{key}.weight")

        dim_in, dim_out = weight.shape[1], weight.shape[0]
        linear = torch.nn.Linear(dim_in, dim_out, bias=False)
        with torch.no_grad():
            linear.weight.copy_(weight)
        layers.append(linear)
        layers.append(torch.nn.ReLU())

    layers = layers[:-1]  # drop trailing ReLU
    encoder = torch.nn.Sequential(*layers)
    encoder.to(device)

    return encoder


@torch.no_grad()
def encode_items(
    encoder: torch.nn.Module,
    sentence_encoder: SentenceTransformer,
    items: list[Item],
    codebooks: torch.Tensor,
    device: str,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:

    logger.info(f"Encoding {len(items)} items...")
    texts = [f"{it.title} {it.brand}" for it in items]
    emb_tensor = sentence_encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
    )  # (N, 768)
    logger.info(f" - Embeddings shape: {emb_tensor.shape}, dtype={emb_tensor.dtype}")

    logger.info("Running encoder MLP...")
    latents_list = []
    for i in range(0, len(emb_tensor), batch_size):
        batch = emb_tensor[i : i + batch_size]
        latents_list.append(encoder(batch.to(device)))
    latents = torch.cat(latents_list, dim=0)  # (N, 32)

    # for each layer, find nearest codebook entry on the residual
    logger.info("Assigning RVQ codes...")
    num_books, num_codes, dim = codebooks.shape
    residual = latents.clone()
    all_codes = []
    for book_idx in range(num_books):
        cb = codebooks[book_idx]  # (K, D)
        # distances: (N, K) — L2 to each codebook entry
        dists = torch.cdist(residual.unsqueeze(0), cb.unsqueeze(0)).squeeze(0)  # (N, K)
        code = dists.argmin(dim=1)  # (N,)
        all_codes.append(code)
        # Subtract the quantized vector (residual quantization)
        residual = residual - cb[code]

    codes_np = torch.stack(all_codes, dim=1).cpu().numpy().astype(np.uint16)  # (N, L)
    latents_np = latents.cpu().numpy().astype(np.float32)

    return latents_np, codes_np


def reconstruct(codebooks: torch.Tensor, codes: np.ndarray) -> np.ndarray:
    """
    Reconstruct embeddings from RVQ codes.
    """

    cb = codebooks.numpy()  # (L, K, D)
    reconstructed = np.zeros((codes.shape[0], cb.shape[2]), dtype=np.float32)
    for book_idx in range(cb.shape[0]):
        reconstructed += cb[book_idx, codes[:, book_idx], :]

    return reconstructed


def compute_ground_truth(query_embeddings: np.ndarray, index_embeddings: np.ndarray, top_k: int) -> np.ndarray:
    """
    Brute-force dot-product nearest neighbor.
    Returns (Q, K) int64 array of indices into index_embeddings.
    """

    logger.info(f"Computing brute-force top-{top_k} ground truth...")
    scores = query_embeddings @ index_embeddings.T  # (Q, N)
    return np.argsort(-scores, axis=1)[:, :top_k].astype(np.int64)


def persist(
    out_dir: str,
    codebooks_tensor: torch.Tensor,
    index_codes: np.ndarray,
    n_index: int,
    query_latents: np.ndarray,
    gt_latent: np.ndarray,
    gt_quantized: np.ndarray,
) -> None:
    logger.info(f"Saving artifacts to {out_dir}/...")
    save_file({"codebooks": codebooks_tensor}, out_dir / "codebooks.safetensors")
    np.save(out_dir / "codebooks.npy", codebooks_tensor.numpy())
    np.save(out_dir / "entity_codes.npy", index_codes)
    np.save(out_dir / "entity_ids.npy", np.arange(n_index, dtype=np.int64))
    np.save(out_dir / "query_embeddings.npy", query_latents)
    np.save(out_dir / "query_gt.npy", gt_latent)
    np.save(out_dir / "query_gt_quantized.npy", gt_quantized)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    items = load_beauty_items(args.max_items)

    if len(items) <= args.query_count:
        raise ValueError(f"Not enough items ({len(items)}) for {args.query_count} queries.")

    # Split: first N-Q items are indexed, last Q are queries
    n_index = len(items) - args.query_count
    index_items = items[:n_index]
    query_items = items[n_index:]
    logger.info(f"Index items: {n_index}, Query items: {args.query_count}")

    # Load model artifacts
    state_dict = load_rqvae_checkpoint()
    codebooks_tensor = extract_codebooks(state_dict, DEVICE)  # (L, K, D)
    encoder = build_encoder(state_dict, DEVICE)

    # Load sentence transformer (same model used during RQ-VAE training)
    logger.info("Loading Sentence Transformer...")
    sentence_encoder = SentenceTransformer(SENTENCE_MODEL, device=DEVICE)

    # Encode all items
    all_items = index_items + query_items
    latents, codes = encode_items(encoder, sentence_encoder, all_items, codebooks_tensor, DEVICE)

    index_latents = latents[:n_index]
    query_latents = latents[n_index:]
    index_codes = codes[:n_index]
    codebooks_tensor = codebooks_tensor.float().cpu()

    # Brute-force ground truth in latent space (measures quantization loss)
    gt_latent = compute_ground_truth(query_latents, index_latents, args.top_k)

    # Brute-force ground truth in quantized space (measures index correctness; recall should be ~1)
    # Query side uses raw latents (same as what the index receives at search time).
    # Item side uses RVQ reconstructions (same as what the index scores against).
    index_reconstructed = reconstruct(codebooks_tensor, index_codes)
    gt_quantized = compute_ground_truth(query_latents, index_reconstructed, args.top_k)

    # Save artifacts
    persist(
        args.out_dir,
        codebooks_tensor,
        index_codes,
        n_index,
        query_latents,
        gt_latent,
        gt_quantized,
    )

    logger.info("Done.")
    logger.info(f" - codebooks: {codebooks_tensor.shape}, dtype={codebooks_tensor.dtype}")
    logger.info(f" - entity_codes: {index_codes.shape}, dtype={index_codes.dtype}")
    logger.info(f" - query_embeddings: {query_latents.shape}, dtype={query_latents.dtype}")
    logger.info(f" - query_gt (latent):     {gt_latent.shape}, dtype={gt_latent.dtype}")
    logger.info(f" - query_gt_quantized:    {gt_quantized.shape}, dtype={gt_quantized.dtype}")


if __name__ == "__main__":
    main()
