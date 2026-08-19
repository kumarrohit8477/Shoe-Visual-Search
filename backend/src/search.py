import os
import sys
import argparse
import json

import faiss
import numpy as np


SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

ROOT_DIR = os.path.dirname(
    SCRIPT_DIR
)

EMBEDDINGS_DIR = os.path.join(
    ROOT_DIR,
    "embeddings",
)

INDEX_FILE = os.path.join(
    EMBEDDINGS_DIR,
    "shoe_index.faiss",
)

METADATA_FILE = os.path.join(
    EMBEDDINGS_DIR,
    "shoe_metadata.json",
)


def load_resources():
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(
            "FAISS index not found. "
            "Run build_index.py first."
        )

    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            "Metadata not found. "
            "Run build_index.py first."
        )

    index = faiss.read_index(
        INDEX_FILE
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if index.ntotal != len(metadata):
        raise RuntimeError(
            "Index/metadata mismatch: "
            f"index={index.ntotal}, "
            f"metadata={len(metadata)}"
        )

    return index, metadata


def filter_candidates(
    scores,
    indices,
    metadata,
    k=10,
    min_score=0.55,
    relative_drop=0.12,
):
    """
    Dynamic filtering.

    We do not use a universal 90% cutoff because a poor-quality
    phone photo can have a lower absolute similarity while still
    being the correct shoe.

    Instead:
      1. Require a reasonable absolute floor.
      2. Compare each candidate to the best candidate.
      3. Remove candidates that fall too far below the best match.

    Both values are configurable.
    """

    candidates = []

    for score, idx in zip(
        scores,
        indices,
    ):
        if idx < 0 or idx >= len(metadata):
            continue

        candidates.append(
            (
                float(score),
                int(idx),
            )
        )

    if not candidates:
        return []

    best_score = candidates[0][0]

    results = []

    for score, idx in candidates:

        # Absolute floor
        if score < min_score:
            continue

        # Relative-to-best filtering
        if (
            best_score - score
            > relative_drop
        ):
            continue

        item = metadata[idx]

        results.append({
            "rank": len(results) + 1,
            "id": int(item["id"]),
            "filename": item["filename"],
            "score": round(
                score,
                6,
            ),
            "similarity_percent": round(
                max(0.0, score) * 100,
                2,
            ),
            "path": os.path.join(
                "dataset",
                item["filename"],
            ),
        })

        if len(results) >= k:
            break

    return results


def search_image(
    image_path,
    k=10,
    min_score=0.55,
    relative_drop=0.12,
):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Query image not found: {image_path}"
        )

    from embedding import (
        get_image_embedding
    )

    index, metadata = load_resources()

    query_vector = get_image_embedding(
        image_path
    )

    query_vector = np.asarray(
        query_vector,
        dtype="float32",
    ).reshape(1, -1)

    faiss.normalize_L2(
        query_vector
    )

    # Search a wider pool before filtering.
    search_k = min(
        max(k * 5, 50),
        index.ntotal,
    )

    scores, indices = index.search(
        query_vector,
        search_k,
    )

    return filter_candidates(
        scores[0],
        indices[0],
        metadata,
        k=k,
        min_score=min_score,
        relative_drop=relative_drop,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Search visually similar shoes "
            "using DINOv2 + FAISS."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        type=str,
        help="Path to query image",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=0.55,
        help=(
            "Absolute similarity floor. "
            "Use 0 during diagnostics."
        ),
    )

    parser.add_argument(
        "--relative-drop",
        type=float,
        default=0.12,
        help=(
            "Maximum allowed drop from the best "
            "candidate before filtering."
        ),
    )

    args = parser.parse_args()

    if args.k < 1:
        parser.error(
            "--k must be at least 1"
        )

    results = search_image(
        args.image,
        k=args.k,
        min_score=args.min_score,
        relative_drop=args.relative_drop,
    )

    print()
    print("Top DINOv2 matches:")
    print("-" * 90)

    if not results:
        print(
            "No candidates passed the filter."
        )
        print(
            "Try --min-score 0 "
            "for diagnostics."
        )
        return

    for result in results:
        print(
            f"{result['rank']:>2}. "
            f"{result['filename']:<35} "
            f"score={result['score']:.4f} "
            f"({result['similarity_percent']:.2f}%)"
        )

    print("-" * 90)


if __name__ == "__main__":
    main()
