import re
import uuid

import chromadb
from sentence_transformers import SentenceTransformer

# ---------- Configuration ----------
CSV_PATH = "/Users/jigyanshupati/semantic_research_paper/arXiv_scientific_dataset_cleaned.csv"
COLLECTION_NAME = "tagged_summary_collection"
MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_PATH = "/Users/jigyanshupati/semantic_research_paper/chroma_db"
# ---------- Load SentenceTransformer model ----------
model = SentenceTransformer(MODEL_NAME)

# ---------- ChromaDB client ----------
client = chromadb.PersistentClient(path=PERSIST_PATH)


# ---------- Create new collection (in-memory only) ----------
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# ---------- Step: Read tagged summaries and insert into ChromaDB ----------
with open("tagged_summary.txt", "r", encoding="utf-8") as f:
    text = f.read()

entries = re.findall(r'"(.*?)"', text, re.DOTALL)
#entries = entries[:1000]  # Optional limit
documents = [entry.strip() for entry in entries]
embeddings = model.encode(documents, show_progress_bar=True)
ids = [str(uuid.uuid4()) for _ in documents]
metadatas = [{"source": "tagged_summary"} for _ in documents]

BATCH_SIZE = 5000  # ChromaDB batch limit is 5461

if collection.count() == 0:
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i+BATCH_SIZE]
        batch_embs = embeddings[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_metas = metadatas[i:i+BATCH_SIZE]
        collection.add(
            documents=batch_docs,
            embeddings=batch_embs.tolist(),
            metadatas=batch_metas,
            ids=batch_ids
        )
    print("Documents loaded into Chroma (persistent storage)")
else:
    print("Chroma collection already populated (persistent storage)")
