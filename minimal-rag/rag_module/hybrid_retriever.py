"""
Hybrid Search with RRF (Reciprocal Rank Fusion)
Combines semantic (vector) search + BM25 for better retrieval
"""
from typing import List, Optional
import numpy as np
from rank_bm25 import BM25Plus
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun


class BM25Retriever:
    """BM25 keyword-based retriever (plain class, not Pydantic)."""
    
    def __init__(self, documents: List[Document], k: int = 4):
        if not documents:
            self.documents = []
            self.k = k
            self.doc_texts = []
            self.tokenized_docs = []
            self.bm25 = None
            return
            
        self.documents = documents
        self.k = k
        
        # Tokenize documents
        self.doc_texts = [doc.page_content for doc in documents]
        self.tokenized_docs = [doc.lower().split() for doc in self.doc_texts]
        
        # Initialize BM25
        self.bm25 = BM25Plus(self.tokenized_docs)
    
    def invoke(self, query: str) -> List[Document]:
        """Invoke BM25 search."""
        if not self.bm25 or not self.documents:
            return []
            
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:self.k]
        
        # Return documents with positive scores
        return [self.documents[i] for i in top_indices if scores[i] > 0]


class HybridRetriever(BaseRetriever):
    """
    Hybrid search combining semantic (vector) + BM25 with RRF fusion.
    Inherits from LangChain BaseRetriever for LCEL chain compatibility.
    """
    
    vector_store: object = None
    documents: List[Document] = []
    k: int = 4
    rrf_k: int = 60
    vector_weight: float = 0.5
    
    def model_post_init(self, __context):
        """Initialize after Pydantic model creation."""
        # Create BM25 retriever (plain class)
        self._bm25_retriever = BM25Retriever(self.documents, k=self.k)
        # Create vector retriever
        self._vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.k})
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        # Get results from both retrievers
        vector_results = self._vector_retriever.invoke(query)
        bm25_results = self._bm25_retriever.invoke(query)
        
        # Create doc_id to score mapping
        doc_scores = {}
        doc_map = {}
        
        # RRF for vector search
        for rank, doc in enumerate(vector_results, 1):
            doc_id = doc.page_content[:200]
            doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + (self.vector_weight * (1 / (self.rrf_k + rank)))
        
        # RRF for BM25 search (weight = 1 - vector_weight)
        bm25_weight = 1 - self.vector_weight
        for rank, doc in enumerate(bm25_results, 1):
            doc_id = doc.page_content[:200]
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + (bm25_weight * (1 / (self.rrf_k + rank)))
        
        # Sort by combined RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get final documents
        result_docs = []
        for doc_id, score in sorted_docs:
            if doc_id in doc_map:
                result_docs.append(doc_map[doc_id])
                if len(result_docs) >= self.k:
                    break
        
        # If no results from fusion, return BM25 results as fallback
        if not result_docs:
            return bm25_results[:self.k]
        
        return result_docs
