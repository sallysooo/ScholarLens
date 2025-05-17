# !pip install arxiv sentence-transformers faiss-cpu

import arxiv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict

class ArxivPaperSearch:
    def __init__(self):
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.index = None
        self.paper_data = []

    def fetch_papers(self, category: str = "cs.LG", max_results: int = 500) -> List[Dict]:
        """arXiv API를 사용하여 논문 데이터 수집"""
        search = arxiv.Search(
            query=f'cat:{category}',
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        papers = []
        for paper in arxiv.Client().results(search):
            papers.append({
                "title": paper.title,
                "abstract": paper.summary.replace('\n', ' '),
                "authors": [a.name for a in paper.authors],
                "published": paper.published.strftime("%Y-%m-%d"),
                "doi": paper.entry_id,
                "pdf_url": next(link.href for link in paper.links if link.title == "pdf")
            })
        return papers

    def build_index(self, papers: List[Dict]) -> None:
        """FAISS 인덱스 구축"""
        abstracts = [f"{p['title']} {p['abstract']}" for p in papers] # 제목 + 초록을 연결(concatenate)하여 사용
        embeddings = self.model.encode(abstracts, normalize_embeddings=True)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.paper_data = papers

    def search_similar(self, query: str, top_k: int = 10) -> List[Dict]:
        """유사 논문 검색"""
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if score > 0.3:  # 유사도 임계값 설정
                paper = self.paper_data[idx]
                paper['similarity'] = float(score)
                results.append(paper)
        return sorted(results, key=lambda x: x['similarity'], reverse=True)

# 사용 예시
if __name__ == "__main__":
    search_system = ArxivPaperSearch()

    # 1. arXiv에서 머신러닝 분야 논문 500편 수집
    papers = search_system.fetch_papers(category="cs.LG", max_results=500)

    # 2. 임베딩 인덱스 구축
    search_system.build_index(papers)

    # 3. 유사 논문 검색 실행
    # query = "Used car price prediction using machine learning techniques"
    query = input("Query: ")
    results = search_system.search_similar(query)

    # 4. 결과 출력
    print(f"검색 결과 ({len(results)}건):")
    for i, paper in enumerate(results[:5]):
        print(f"\n{i+1}. {paper['title']}")
        print(f"   유사도: {paper['similarity']:.3f}")
        print(f"   요약: {paper['abstract'][:150]}...")
        print(f"   PDF: {paper['pdf_url']}")

'''
출력 예시 (유사도가 ver1보다 훨씬 향상된 모습)

1. REMEDI: Relative Feature Enhanced Meta-Learning with Distillation for Imbalanced Prediction
   유사도: 0.669
   요약: Predicting future vehicle purchases among existing owners presents a critical challenge due to extreme class imbalance (<0.5% positive rate) and compl...
   PDF: http://arxiv.org/pdf/2505.07245v1

2. Transfer Learning Across Fixed-Income Product Classes
   유사도: 0.653
   요약: We propose a framework for transfer learning of discount curves across different fixed-income product classes. Motivated by challenges in estimating d...
   PDF: http://arxiv.org/pdf/2505.07676v1

3. Avocado Price Prediction Using a Hybrid Deep Learning Model: TCN-MLP-Attention Architecture
   유사도: 0.634
   요약: With the growing demand for healthy foods, agricultural product price forecasting has become increasingly important. Hass avocados, as a high-value cr...
   PDF: http://arxiv.org/pdf/2505.09907v1

4. A Hybrid Strategy for Aggregated Probabilistic Forecasting and Energy Trading in HEFTCom2024
   유사도: 0.612
   요약: Obtaining accurate probabilistic energy forecasts and making effective decisions amid diverse uncertainties are routine challenges in future energy sy...
   PDF: http://arxiv.org/pdf/2505.10367v1

5. Iteratively reweighted kernel machines efficiently learn sparse functions
   유사도: 0.610
   요약: The impressive practical performance of neural networks is often attributed to their ability to learn low-dimensional data representations and hierarc...
   PDF: http://arxiv.org/pdf/2505.08277v1

'''
