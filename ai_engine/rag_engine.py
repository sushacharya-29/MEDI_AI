from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.docstore.document import Document

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger


class HyperRAGEngine:
    """Lightweight RAG engine for medical context retrieval using LangChain and FAISS."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Lightweight embedder
        self.vector_store = None
        self._load_and_index_dataset()

    def _load_and_index_dataset(self):
        """Load diseases.csv and create FAISS vector store."""
        logger.info(f"Loading dataset from {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        
        # Create documents: one per disease with all fields
        documents = []
        for _, row in df.iterrows():
            content = (
                f"Disease: {row['Disease']}\n"
                f"ICD-10: {row['ICD10']}\n"
                f"Symptoms: {row['Symptoms']}\n"
                f"Severity: {row['Severity']}\n"
                f"Tests: {row['Tests']}\n"
                f"Treatments: {row['Treatments']}\n"
                f"Comorbidities: {row['Comorbidities']}\n"
                f"Symptom Onset: {row['SymptomOnset']}\n"
                f"Demographic Risks: {row['DemographicRisks']}\n"
                f"Mock Imaging: {row['MockImageFindings']}"
            )
            metadata = {"disease": row['Disease'], "icd10": row['ICD10']}
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Build FAISS index
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        logger.info("Dataset indexed in FAISS vector store")

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top_k diseases matching the query (e.g., symptoms + imaging)."""
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        context = []
        for doc, score in results:
            context.append({
                "disease": doc.metadata["disease"],
                "icd10": doc.metadata["icd10"],
                "content": doc.page_content,
                "score": 1 - score  # Convert distance to similarity (0-1)
            })
        return context