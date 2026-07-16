"""Multi-RAG pipeline package.

Modules map 1:1 onto the classic RAG stages so the concepts stay visible:

    ingestion  ->  raw text out of Excel + images (OCR)
    chunking   ->  split long text into retrievable pieces
    embeddings ->  turn chunks into vectors (local model)
    vectorstore->  store vectors + nearest-neighbour search (FAISS/NumPy)
    generator  ->  build a grounded prompt and call Groq
"""
