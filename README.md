# rvq-index

A Rust implementation of a nearest-neighbor index for Residual Vector Quantization (RVQ) codes, with beam search retrieval.

_Not intended for production._

## Overview

RVQ approximates a dense vector as a sum of codes $e \approx sum c_i$. The vector is encoded as a sequence of integer codes, one per codebook layer. Each code identifies the nearest entry in that layer's codebook. The search is exact in quantized space but not in latent space, so the recall is directly dependent on the RVQ model that generates the codebooks and item's code sequences.

This crate provides:
- `CodeBooks` — stores L codebooks of shape `[L, K, D]` and scores a query against all entries
- `CodeTrie` — a prefix trie over code paths with beam search retrieval
- `RvqIndex` — the top-level index combining both, with parallel batch search via rayon

Given a query vector and a catalog of (id, codes) pairs, `RvqIndex::search` returns the top-k entity IDs whose reconstructed embeddings have the highest dot product with the query.

## How it works

### Scoring

`CodeBooks::score(query)` computes the dot product of the query against every entry in every codebook, producing a `ScoredBooks` with shape `[L, K]`. This is O(L x K x D) and runs once per query before trie traversal.

As the original embedding is decomposed by a sum of codes, so is the score: $score(q, Q(c)) = \sum_l score(q, code_l^c)$ where $Q(c)$ is the quantized candidate, which means that while search is exact in quantized space, it is not in latent space.

### Trie search with upper-bound pruning

`CodeTrie` stores only the code paths that have been inserted (i.e. paths that correspond to at least one indexed entity). Retrieval uses a max-heap of candidates, each tracking:

- `cumulative_score` — sum of per-book scores accumulated so far
- `upper_bound` — cumulative score plus the sum of per-book maxima over all remaining books

The upper bound is precomputed as a suffix array of per-book maxima. Any candidate whose upper bound is lower than the current k-th best complete path can be safely pruned. The search is exact in quantized space: it finds the true top-k code paths by dot-product score, visiting only the paths that actually exist.

Complexity: O(K x L x log(K x C)) per query, independent of catalog size N, making this solution particularly interesting for very large candidate spaces.

### Parallelism

`RvqIndex::search_batch` runs one `search` call per query in parallel using rayon's `par_iter`. The index is read-only after construction, so no locking is needed.

## Design choices and limitations

- **HashMap trie nodes.** Each `TrieNode` stores children in a `HashMap<Code, TrieNode>`. This handles sparse code spaces cleanly but has high constant overhead. At small N (below ~10K items) a brute-force scan over reconstructed vectors is likely faster.

- **Scalar dot products, no SIMD.** Scoring iterates with plain Rust iterators. A SIMD-accelerated scoring pass would reduce query latency, especially at high D.

- **Code collisions.** Multiple entity IDs can share the same code path. `search` returns all IDs from the top-k paths, so the result size may differ from k when collisions exist.

- **In-memory only.** The trie and entity store have no serialization.

- **Optional I/O.** Codebook loading from `.npy` and `.safetensors` is gated behind feature flags (`npy`, `safetensors`) so the core crate has no file-format dependencies.

## Usage

```toml
[dependencies]
rvq-index = { path = ".", features = ["safetensors"] }
```

```rust
use rvq_index::codebook::CodeBooks;
use rvq_index::index::RvqIndex;

// Load codebooks from a safetensors file (requires "safetensors" feature)
let codebooks = CodeBooks::<f32>::from_safetensors("codebooks.safetensors", "codebooks")?;
let mut index: RvqIndex<u32, f32> = RvqIndex::new(codebooks);

// Insert entities as (id, codes) pairs
index.insert_many(vec![
    (0, vec![4, 17, 201]),
    (1, vec![4, 17, 88]),
    (2, vec![12, 0, 55]),
])?;

// Search top-5
let query: Vec<f32> = ...;
let results: Vec<&u32> = index.search(&query, 5)?;
```

## Benchmarks

Benchmarks use real data from the Amazon Beauty RQ-VAE model (3 codebooks, 256 codes, 32-dim). Generate benchmark artifacts with the Python script, then run with cargo.

```bash
# Generate data
cd benchmarks
uv run python generate_data.py --max-items 50000 --query-count 100 --out-dir data/

# Run benchmarks
cd ..
cargo bench --features npy,safetensors
```

Results on Apple M-series (100 queries, K=10):

| N items | Index creation | Single query | Batch (100 queries) | Recall@10 (quantized) | Recall@10 (latent) |
|--------:|---------------:|-------------:|--------------------:|----------------------:|-------------------:|
| 1 000   | ~200 µs        | ~47 µs       | ~1.0 ms             | ~1.0                  | ~0.64              |
| 50 000  | ~6.5 ms        | ~44 µs       | ~920 µs             | ~1.0                  | ~0.30              |

Query latency is flat in N, confirming O(K × L × log(K × C)) complexity independent of catalog size. Recall@10 in quantized space is ~1.0, validating index correctness. Recall@10 in latent space reflects quantization loss from the RVQ model and degrades as N grows (more competition for top-k slots).

## Project layout

```
src/
  codebook.rs
  trie.rs
  store.rs
  index.rs
  errors.rs
benches/
  index_bench.rs  // Criterion benchmarks
benchmarks/
  generate_data.py  // fetch model, encode items, save artifacts
  pyproject.toml
```

## References
- [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)
- [RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender) — RQ-VAE implementation used for benchmarking
- [edobotta/rqvae-amazon-beauty](https://huggingface.co/edobotta/rqvae-amazon-beauty) — Pretrained model weights
- [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) — Amazon Beauty dataset
