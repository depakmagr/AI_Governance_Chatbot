import google.generativeai as genai # pip install google-generativeai
import numpy as np
import json
import os
from document_loader import save_all_pdfs_to_txt,chunk_text


genai.configure(api_key='AIzaSyCmuOyyZRF6BJZCK0g0Z-EHl07WAqQCuQs')

EMBEDDING_SIZE = 768  # default size for fallback vectors


def embed_chunks(chunks, model="models/embedding-001"):
    """
    Generate embeddings for a list of text chunks using Google Gemini (genai).
    Returns a list of dictionaries: {"chunk": ..., "embedding": [...]}
    """
    embeddings_data = []
    print("Total Chunks:",len(chunks))

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} embedding")
        if not chunk.strip():
            embeddings_data.append({"chunk": chunk, "embedding": np.zeros(EMBEDDING_SIZE).tolist()})
            continue

        # limit chunk length
        chunk = chunk[:3000]

        try:
            res = genai.embed_content(
                model=model,
                content=chunk,
                task_type="retrieval_document"
            )
            embeddings_data.append({
                "chunk": chunk,
                "embedding": res["embedding"]
            })
        except Exception as e:
            print(f"Embedding failed for chunk {i}: {e}")
            embeddings_data.append({"chunk": chunk, "embedding": np.zeros(EMBEDDING_SIZE).tolist()})

        if (i + 1) % 50 == 0 or (i + 1) == len(chunks):
            print(f"Processed {i + 1}/{len(chunks)} chunks")

    return embeddings_data


def save_embeddings(embeddings, file_path="embeddings.json"):
    """Save embeddings to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)
    print(f"Embeddings saved to {file_path}")


# Example usage
if __name__ == "__main__":
    data_folder = 'PDF_FILES\AI_Goverment_Services'
    text = save_all_pdfs_to_txt(data_folder)
    chunks = chunk_text(text,chunk_size=1000,overlap=200)
    embeddings_data = embed_chunks(chunks)
    save_embeddings(embeddings_data)
