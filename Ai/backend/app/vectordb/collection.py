from chromadb.api.configuration import (
    CollectionConfigurationInternal,
    ConfigurationParameter,
    HNSWConfigurationInternal,
)

from app.utils.text_dedup import dedupe_key
from app.vectordb.chroma_client import get_chroma_client


def collection_name(website_id: int) -> str:
    return f"website_{website_id}"


def _collection_configuration() -> CollectionConfigurationInternal:
    # `metadata={"hnsw:space": ...}` is the *pre-0.5* chromadb API and is silently ignored
    # here — this version requires the structured `configuration=` object instead, or every
    # collection silently falls back to defaults: "l2" distance (not cosine, so our
    # `similarity = 1 - distance` math would be meaningless) and `sync_threshold=1000`.
    #
    # sync_threshold=1: chromadb's persisted HNSW segment only calls its internal
    # `_persist()` (which writes the actual index + metadata to disk) when the number of
    # unflushed writes since the last persist reaches `sync_threshold` — there is no flush
    # on clean process shutdown. With any threshold > 1, whatever fraction of the last
    # crawl's writes doesn't happen to land exactly on a multiple of the threshold sits
    # in memory only and is silently lost the instant the process restarts, with no error
    # or warning. Forcing a flush after literally every write is the only way to make
    # this actually durable; the extra disk I/O is negligible at this app's scale
    # (a handful of crawls a day, not a high-throughput write workload).
    # ef_search: hnswlib requires ef_search >= the number of neighbors requested per
    # query, or it raises "Cannot return the results in a contiguous 2D array" instead of
    # just returning fewer results. Chroma's default is only 10, but retrieval here asks
    # for top_k * 8 candidates (well over 100) — comfortably over any realistic request
    # size avoids that failure as the collection grows.
    hnsw = HNSWConfigurationInternal(parameters=[
        ConfigurationParameter(name="space", value="cosine"),
        ConfigurationParameter(name="sync_threshold", value=1),
        ConfigurationParameter(name="batch_size", value=1),
        ConfigurationParameter(name="ef_search", value=200),
    ])
    return CollectionConfigurationInternal(parameters=[
        ConfigurationParameter(name="hnsw_configuration", value=hnsw),
    ])


def get_or_create_collection(website_id: int):
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name(website_id),
        configuration=_collection_configuration(),
    )


def upsert_page_chunks(
    website_id: int,
    page_id: int,
    url: str,
    title: str,
    content_hash: str,
    chunks: list[str],
    embeddings: list[list[float]],
    indexed_at: str | None = None,
) -> None:
    if not chunks:
        return
    collection = get_or_create_collection(website_id)
    ids = [f"{page_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "website_id": website_id,
            "page_id": page_id,
            "title": title or "",
            "url": url,
            "chunk_id": i,
            "hash": content_hash,
            "indexed_at": indexed_at or "",
        }
        for i in range(len(chunks))
    ]
    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)


def delete_page_vectors(website_id: int, page_id: int) -> None:
    collection = get_or_create_collection(website_id)
    collection.delete(where={"page_id": page_id})


def delete_website_collection(website_id: int) -> None:
    client = get_chroma_client()
    existing = [c.name for c in client.list_collections()]
    if collection_name(website_id) in existing:
        client.delete_collection(collection_name(website_id))


def query_collection(website_id: int, query_embedding: list[float], top_k: int) -> list[dict]:
    collection = get_or_create_collection(website_id)
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        {"text": doc, "metadata": meta, "similarity": 1.0 - dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


def find_chunks_containing(website_id: int, text: str) -> list[dict]:
    # Exact substring lookup, bypassing embeddings entirely. Small general-purpose
    # embedding models represent rare proper nouns/brand names poorly, so a chunk that
    # literally contains the term a visitor asked about can still rank low on pure
    # cosine similarity — this catches it regardless of embedding quality.
    collection = get_or_create_collection(website_id)
    if collection.count() == 0 or not text:
        return []
    result = collection.get(where_document={"$contains": text}, include=["documents", "metadatas"])
    return [{"text": doc, "metadata": meta} for doc, meta in zip(result["documents"], result["metadatas"])]


def get_all_chunks(website_id: int) -> list[dict]:
    # Full-corpus fetch for the website's collection — used to build the BM25 keyword
    # index, which (unlike ANN vector search) needs the whole document set up front
    # rather than a nearest-neighbor query.
    collection = get_or_create_collection(website_id)
    if collection.count() == 0:
        return []
    result = collection.get(include=["documents", "metadatas"])
    return [{"text": doc, "metadata": meta} for doc, meta in zip(result["documents"], result["metadatas"])]


def get_existing_chunk_keys(website_id: int, exclude_page_id: int | None = None) -> set[str]:
    # Used to skip storing a chunk that's a near-duplicate of one already indexed for a
    # DIFFERENT page on the same site (e.g. the same client testimonial repeated across
    # many pages) — normalizing noise out at ingest time rather than only at query time.
    collection = get_or_create_collection(website_id)
    if collection.count() == 0:
        return set()
    where = {"page_id": {"$ne": exclude_page_id}} if exclude_page_id is not None else None
    result = collection.get(where=where, include=["documents"]) if where else collection.get(include=["documents"])
    return {dedupe_key(doc) for doc in result["documents"]}
