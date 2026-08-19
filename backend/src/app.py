import os
import sys
import gc
import json
import re

import faiss
import numpy as np
import torch

from PIL import Image

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    BackgroundTasks
)

from fastapi.responses import (
    HTMLResponse,
    FileResponse
)

from fastapi.middleware.cors import (
    CORSMiddleware
)


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


from embedding import (
    get_image_embedding_from_pil,
    get_image_embedding
)


ROOT_DIR = os.path.dirname(
    SCRIPT_DIR
)

DATASET_DIR = os.path.join(
    ROOT_DIR,
    "dataset"
)

EMBEDDINGS_DIR = os.path.join(
    ROOT_DIR,
    "embeddings"
)

INDEX_FILE = os.path.join(
    EMBEDDINGS_DIR,
    "shoe_index.faiss"
)

METADATA_FILE = os.path.join(
    EMBEDDINGS_DIR,
    "shoe_metadata.json"
)

TEMPLATE_FILE = os.path.join(
    SCRIPT_DIR,
    "templates",
    "index.html"
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ============================================================
# INITIALIZATION
# ============================================================

print("=" * 60)
print("INITIALIZING SHOE VISUAL SEARCH SERVER")
print("=" * 60)

os.makedirs(
    DATASET_DIR,
    exist_ok=True
)

os.makedirs(
    EMBEDDINGS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD INDEX
# ============================================================

if not os.path.exists(
    INDEX_FILE
) or not os.path.exists(
    METADATA_FILE
):

    print(
        "Index or metadata not found."
    )

    print(
        "Run:"
    )

    print(
        "python src/build_index.py"
    )

    sys.exit(1)


print("Loading FAISS index...")

index = faiss.read_index(
    INDEX_FILE
)


# ============================================================
# LOAD METADATA
# ============================================================

print("Loading metadata...")

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)


if index.ntotal != len(metadata):

    print(
        "WARNING: index and metadata "
        "are out of sync."
    )

    print(
        f"FAISS vectors : {index.ntotal}"
    )

    print(
        f"Metadata      : {len(metadata)}"
    )


print(
    f"Loaded {index.ntotal} vectors."
)

print(
    f"Loaded {len(metadata)} metadata records."
)

is_reindexing = False

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Shoe Visual Search API",
    version="2.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# FAISS SEARCH
# ============================================================

def query_faiss_index(
    query_vector,
    k=10,
    min_score=0.0
):

    if index.ntotal == 0:
        return []

    query_vector = np.asarray(
        query_vector,
        dtype="float32"
    ).reshape(1, -1)

    faiss.normalize_L2(
        query_vector
    )

    # Search extra candidates.
    search_k = min(
        max(k * 5, 50),
        index.ntotal
    )

    scores, indices = index.search(
        query_vector,
        search_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        if idx >= len(metadata):
            continue

        score = float(score)

        if score < min_score:
            continue

        item = metadata[idx]

        results.append({
            "rank": len(results) + 1,
            "id": int(item["id"]),
            "filename": item["filename"],
            "score": round(
                score,
                6
            ),
            "similarity_percent": round(
                max(0.0, score) * 100,
                2
            )
        })

        if len(results) >= k:
            break

    return results


# ============================================================
# FRONTEND
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def serve_frontend():

    if not os.path.exists(
        TEMPLATE_FILE
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "Frontend template "
                "not found."
            )
        )

    with open(
        TEMPLATE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return HTMLResponse(
            content=file.read()
        )


# ============================================================
# SERVE DATASET IMAGE
# ============================================================

@app.get(
    "/dataset/{filename}"
)
async def serve_dataset_image(
    filename: str
):

    # Resolve paths to prevent directory traversal
    file_path = os.path.realpath(
        os.path.abspath(
            os.path.join(
                DATASET_DIR,
                filename
            )
        )
    )

    dataset_root = os.path.realpath(
        os.path.abspath(
            DATASET_DIR
        )
    )

    # Security check
    if os.path.commonpath(
        [
            file_path,
            dataset_root
        ]
    ) != dataset_root:

        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )

    if not os.path.isfile(
        file_path
    ):

        raise HTTPException(
            status_code=404,
            detail="Image not found"
        )

    return FileResponse(
        file_path
    )


# ============================================================
# SAMPLE IMAGES
# ============================================================

@app.get(
    "/api/samples"
)
async def get_samples():

    try:

        files = [
            filename
            for filename in os.listdir(
                DATASET_DIR
            )
            if os.path.splitext(
                filename
            )[1].lower()
            in SUPPORTED_EXTENSIONS
        ]

        return sorted(files)[:6]

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# IMAGE SEARCH
# ============================================================

@app.post(
    "/api/search/image"
)
async def search_by_image(
    file: UploadFile = File(...),
    k: int = 10,
    min_score: float = 0.0
):

    try:

        if k < 1 or k > 50:

            raise HTTPException(
                status_code=400,
                detail=(
                    "k must be between "
                    "1 and 50."
                )
            )

        if (
            not file.content_type
            or not file.content_type.startswith(
                "image/"
            )
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Please upload "
                    "a valid image."
                )
            )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            file.file
        ).convert("RGB")

        # ----------------------------------------------------
        # Generate robust embedding
        #
        # This includes automatic crop
        # and multiple views.
        # ----------------------------------------------------

        query_vector = (
            get_image_embedding_from_pil(
                image
            )
        )

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        results = query_faiss_index(
            query_vector,
            k=k,
            min_score=max(
                0.0,
                min_score
            )
        )

        return {
            "status": "success",
            "query_filename": file.filename,
            "count": len(results),
            "results": results
        }

    except HTTPException:
        raise

    except Exception as error:

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# ADD ONE IMAGE TO EXISTING INDEX
# ============================================================

def add_image_to_index(
    filename: str,
    image_path: str
):

    global index
    global metadata

    # --------------------------------------------------------
    # Generate only ONE new embedding
    # --------------------------------------------------------

    embedding = get_image_embedding(
        image_path
    )

    vector = np.asarray(
        embedding,
        dtype="float32"
    ).reshape(1, -1)

    faiss.normalize_L2(
        vector
    )

    # --------------------------------------------------------
    # Check dimension
    # --------------------------------------------------------

    if index.ntotal > 0:

        if vector.shape[1] != index.d:

            raise RuntimeError(
                "Embedding dimension "
                "does not match FAISS index."
            )

    # --------------------------------------------------------
    # New ID
    # --------------------------------------------------------

    if metadata:

        new_id = max(
            int(item["id"])
            for item in metadata
        ) + 1

    else:

        new_id = 1

    # --------------------------------------------------------
    # Add vector
    # --------------------------------------------------------

    index.add(
        vector
    )

    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    new_record = {
        "id": new_id,
        "filename": filename,
        "path": image_path
    }

    metadata.append(
        new_record
    )

    # --------------------------------------------------------
    # Persist index
    # --------------------------------------------------------

    faiss.write_index(
        index,
        INDEX_FILE
    )

    # --------------------------------------------------------
    # Persist metadata atomically
    # --------------------------------------------------------

    temp_metadata_file = METADATA_FILE + ".tmp"
    with open(
        temp_metadata_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )
    os.replace(temp_metadata_file, METADATA_FILE)

    return new_record


# ============================================================
# CATALOG STATISTICS
# ============================================================

@app.get(
    "/api/catalog/stats"
)
async def get_catalog_stats():

    try:

        dataset_images = [
            filename
            for filename in os.listdir(
                DATASET_DIR
            )
            if os.path.splitext(
                filename
            )[1].lower()
            in SUPPORTED_EXTENSIONS
        ]

        return {
            "catalog_size": len(metadata),
            "dataset_images_count": len(
                dataset_images
            ),
            "faiss_vectors": index.ntotal,
            "embedding_dimension": index.d,
            "is_reindexing": is_reindexing,
            "is_out_of_sync": (
                len(metadata)
                != len(dataset_images)
                or index.ntotal
                != len(metadata)
            )
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# ADD SHOE TO DATASET
# ============================================================

@app.post(
    "/api/upload"
)
async def upload_shoe(
    file: UploadFile = File(...)
):

    try:

        if not file.filename:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Filename is required."
                )
            )

        extension = os.path.splitext(
            file.filename
        )[1].lower()

        if extension not in SUPPORTED_EXTENSIONS:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Allowed formats: "
                    "JPG, JPEG, PNG, WEBP."
                )
            )

        # ----------------------------------------------------
        # Clean filename
        # ----------------------------------------------------

        clean_name = re.sub(
            r"[^a-zA-Z0-9_.-]",
            "_",
            file.filename
        )

        base_name, extension = (
            os.path.splitext(clean_name)
        )

        target_path = os.path.join(
            DATASET_DIR,
            clean_name
        )

        counter = 1

        while os.path.exists(
            target_path
        ):

            clean_name = (
                f"{base_name}_"
                f"{counter}"
                f"{extension}"
            )

            target_path = os.path.join(
                DATASET_DIR,
                clean_name
            )

            counter += 1

        # ----------------------------------------------------
        # Read uploaded image
        # ----------------------------------------------------

        try:

            uploaded_image = Image.open(
                file.file
            ).convert("RGB")

        except Exception:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Uploaded file "
                    "is not a valid image."
                )
            )

        # ----------------------------------------------------
        # Save original uploaded image
        #
        # We keep the original file in dataset.
        # The embedding pipeline performs its own crop.
        # ----------------------------------------------------

        uploaded_image.save(
            target_path,
            format=(
                "JPEG"
                if extension in {
                    ".jpg",
                    ".jpeg"
                }
                else None
            ),
            quality=95
            if extension in {
                ".jpg",
                ".jpeg"
            }
            else None
        )

        # ----------------------------------------------------
        # Add ONLY the new image to FAISS
        # ----------------------------------------------------

        record = add_image_to_index(
            clean_name,
            target_path
        )

        return {
            "status": "success",
            "filename": clean_name,
            "id": record["id"],
            "catalog_size": len(metadata),
            "faiss_vectors": index.ntotal,
            "message": (
                "Shoe added to dataset "
                "and is immediately "
                "available for search."
            )
        }

    except HTTPException:
        raise

    except Exception as error:

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# BACKGROUND REINDEX TASK
# ============================================================

def run_reindex_task():

    global index
    global metadata
    global is_reindexing

    is_reindexing = True

    try:

        from index_manager import rebuild_index as rebuild_index_fn

        new_index, new_metadata = rebuild_index_fn()

        index = new_index
        metadata = new_metadata

        print("Background index rebuild successful.")

    except Exception as error:

        print(f"Background index rebuild failed: {error}")

    finally:

        is_reindexing = False


# ============================================================
# FULL REINDEX ENDPOINT
# ============================================================

@app.post(
    "/api/reindex"
)
async def trigger_reindex(
    background_tasks: BackgroundTasks
):

    global is_reindexing

    if is_reindexing:

        return {
            "status": "warning",
            "message": "Reindexing is already in progress in the background."
        }

    background_tasks.add_task(
        run_reindex_task
    )

    return {
        "status": "success",
        "message": (
            "Catalog re-indexing "
            "started in the background."
        )
    }


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )