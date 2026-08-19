import os
import json
import faiss
import numpy as np
import gc
import torch

from embedding import get_image_embedding

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")
INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "shoe_index.faiss")
METADATA_FILE = os.path.join(EMBEDDINGS_DIR, "shoe_metadata.json")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def get_image_files():
    if not os.path.isdir(DATASET_DIR):
        return []
    return sorted(
        filename
        for filename in os.listdir(DATASET_DIR)
        if os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS
    )

def rebuild_index():
    """
    Builds the FAISS index and metadata file from scratch.
    Returns:
        tuple: (new_index, new_metadata)
    """
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    image_files = get_image_files()

    if not image_files:
        raise RuntimeError(f"No image files found in dataset directory: {DATASET_DIR}")

    print("=" * 60)
    print("REBUILDING DINOv2 SHOE VECTOR INDEX")
    print("=" * 60)
    print(f"Images found: {len(image_files)}")
    print()

    embeddings = []
    new_metadata = []
    failed = []

    for position, filename in enumerate(image_files, start=1):
        image_path = os.path.join(DATASET_DIR, filename)
        print(f"[{position}/{len(image_files)}] {filename}")
        try:
            embedding = get_image_embedding(image_path)
            embeddings.append(embedding)
            new_metadata.append({
                "id": position,
                "filename": filename,
                "path": image_path
            })
        except Exception as error:
            print(f"  ERROR indexing {filename}: {error}")
            failed.append({
                "filename": filename,
                "error": str(error)
            })

    if not embeddings:
        raise RuntimeError("Failed to generate any embeddings.")

    vectors = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)
    dimension = vectors.shape[1]

    # Create inner product FAISS index
    new_index = faiss.IndexFlatIP(dimension)
    new_index.add(vectors)

    # Persist the new index to disk
    faiss.write_index(new_index, INDEX_FILE)

    # Persist metadata atomically using a temp file
    temp_metadata_file = METADATA_FILE + ".tmp"
    with open(temp_metadata_file, "w", encoding="utf-8") as file:
        json.dump(new_metadata, file, indent=4)
    os.replace(temp_metadata_file, METADATA_FILE)

    # Log failures if any
    failed_file = os.path.join(EMBEDDINGS_DIR, "failed_images.json")
    if failed:
        with open(failed_file, "w", encoding="utf-8") as file:
            json.dump(failed, file, indent=4)
        print(f"Failed-image log created: {failed_file}")
    elif os.path.exists(failed_file):
        try:
            os.remove(failed_file)
        except OSError:
            pass

    print()
    print("=" * 60)
    print("FAISS INDEX REBUILT SUCCESSFULLY")
    print("=" * 60)
    print(f"Images indexed : {len(new_metadata)}")
    print(f"Images failed  : {len(failed)}")
    print(f"Vector dimension: {dimension}")
    print(f"FAISS index    : {INDEX_FILE}")
    print(f"Metadata       : {METADATA_FILE}")

    # Explicit memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return new_index, new_metadata
