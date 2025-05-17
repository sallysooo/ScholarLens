#!pip install transformers faiss-cpu
#!pip install transformers faiss-cpu requests tqdm

# 1. 논문 데이터 준비
import requests
import xml.etree.ElementTree as ET

# arXiv에서 CS 분야 최신 논문 100개 불러오기
def fetch_arxiv_papers(query="cs", max_results=100): # 기본값
    url = f"http://export.arxiv.org/api/query?search_query=cat:{query}&start=0&max_results={max_results}"
    response = requests.get(url)
    root = ET.fromstring(response.content)
    entries = root.findall("{http://www.w3.org/2005/Atom}entry")

    papers = []
    for entry in entries:
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip().replace('\n', ' ')
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip().replace('\n', ' ')
        link = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
        papers.append({"title": title, "abstract": abstract, "url": link})
    return papers


# 2. E5 임베딩 모델 로드
from transformers import AutoTokenizer, AutoModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "intfloat/e5-large-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# 텍스트를 벡터화하는 함수
def embed(texts):
    # E5 모델은 입력 앞에 문맥 prefix 필요
    inputs = tokenizer([f"passage: {t}" for t in texts], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]  # [CLS] 토큰 벡터
        return embeddings.cpu()

# 3. 논문 데이터 벡터화 & FAISS로 검색 인덱스 만들기
import faiss
from tqdm import tqdm

# 카테고리 선택
category = input("검색할 arXiv 분야 (예: cs.LG, cs.AI+stat.ML) [기본값: cs.LG]: ") or "cs.LG"
docs = fetch_arxiv_papers(query=category, max_results=1000)

# 논문 불러오기
doc_texts = [doc["title"] + " " + doc["abstract"] for doc in docs]

# 임베딩 벡터 생성
doc_embeddings = embed(doc_texts)

# FAISS 인덱스 구성
faiss.normalize_L2(doc_embeddings.numpy())  # 정규화
index = faiss.IndexFlatIP(dimension)  # 내적 기반 코사인 유사도
index.add(doc_embeddings.numpy())

# 4. 사용자 쿼리 -> 유사 논문 검색 및 추천
def search_similar_docs(query_text, top_k=5):
    query_embedding = embed([f"query: {query_text}"])
    D, I = index.search(query_embedding.numpy(), top_k)  # 거리 D, 인덱스 I
    results = []
    for idx, dist in zip(I[0], D[0]):
      similarity_score = 1 / (1 + dist)  # 거리를 유사도로 변환 (0~1 사이 값)
      results.append({**docs[idx], "similarity": similarity_score})
    return sorted(results, key=lambda x: x["similarity"], reverse=True)  # 유사도 내림

import re

def highlight_keywords(text, keywords):
  for kw in keywords:
    pattern = re.compile(re.escape(kw), re.IGNORECASE)
    # text = pattern.sub(f'<span style="background-color:yellow;">{kw}</span>', text)
    text = pattern.sub(lambda m: f"\033[1;31m{m.group(0)}\033[0m", text)
  return text

query = input("Query: ") # deep learning based vehicle price estimation
results = search_similar_docs(query)
keywords = query.lower().split()

for i, r in enumerate(results, 1):
    highlighted_title = highlight_keywords(r['title'], keywords)
    print(f"\n🔎 유사 논문 {i}")
    print(f"📄 제목: {highlighted_title}")
    print(f"📝 요약: {r['abstract'][:300]}...")
    print(f"📊 유사도: {r['similarity']:.3f}")
    print(f"🔗 링크: {r['url']}")

# 위 ver1의 코드는 유사도가 매우 낮게 나와서 사실상 작업이 제대로 수행되고 있지 않음.
# 따라서 모델을 변경하고 테크닉을 더 적절히 수정하여 ver2를 제작함

'''
출력 예시 (내림차순으로 정렬했음에도 유사도가 거의 0에 가까운 결과)

🔎 유사 논문 1
📄 제목: Fingerprint based bio-starter and bio-access
📝 요약: In the paper will be presented a safety and security system based on fingerprint technology. The results suggest a new scenario where the new cars can use a fingerprint sensor integrated in car handle to allow access and in the dashboard as starter button....
📊 유사도: 0.155
🔗 링크: http://arxiv.org/abs/cs/0308034v1

🔎 유사 논문 2
📄 제목: Modeling and Control with Local Linearizing Nadaraya Watson Regression
📝 요약: Black box models of technical systems are purely descriptive. They do not explain why a system works the way it does. Thus, black box models are insufficient for some problems. But there are numerous applications, for example, in control engineering, for which a black box model is absolutely suffici...
📊 유사도: 0.150
🔗 링크: http://arxiv.org/abs/0809.3690v1

'''