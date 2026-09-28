import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class VectorDatabase:
    def __init__(self, kb_path=None):
        print("[RAG] Инициализация векторной базы данных...")
        if kb_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            kb_path = os.path.join(current_dir, "knowledge_base.txt")
        self.encoder = SentenceTransformer("cointegrated/rubert-tiny2")
        self.documents = []
        self.index = None
        self._build_index(kb_path)

    def _build_index(self, path):
        if not os.path.exists(path):
            print(f"[RAG ERROR] База знаний {path} не найдена!")
            return

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        
        self.documents = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        if not self.documents:
            return

        print(f"[RAG] Векторизация {len(self.documents)} фрагментов текста...")
        embeddings = self.encoder.encode(self.documents)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print("[RAG] Векторный индекс успешно построен.")

    def search_context(self, query: str, top_k=2) -> str:
        if not self.index or not self.documents:
            return ""

        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)
        
        results = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        return " ".join(results)

rag_db = VectorDatabase()