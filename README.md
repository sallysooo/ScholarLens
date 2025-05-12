# 📚 ScholarLens

**ScholarLens**는 논문에 익숙하지 않은 학부연구자들을 위한 AI 논문 어시스턴트입니다.  
논문 요약, 유사 논문 검색, 논문 기반 질의응답(Q&A) 기능을 통합하여  
논문 탐색과 이해를 빠르고 효율적으로 도와줍니다.

---

## 🧩 프로젝트 주요 기능

| 기능명 | 사용 모델/도구 | 설명 |
|--------|----------------|------|
| 📝 논문 요약 | `DistilBART (sshleifer/distilbart-cnn-12-6)` | 논문을 chunk로 나눈 뒤 각 chunk 요약 |
| 🔍 유사 논문 검색 | `E5-small` + FAISS | 논문 벡터화 후 입력과 가장 유사한 논문 추천 |
| ❓ 논문 기반 Q&A | LlamaIndex or E5 + 검색 기반 응답 | 논문 내용 기반 질의응답 처리 (chunk 검색 기반) |

---

## 🧠 시스템 구조 요약

PDF 논문 업로드
│
├─▶ [1] 논문 텍스트 추출 (PyMuPDF)
│
├─▶ [2] 논문 요약 (BART로 chunk별 요약)
│
├─▶ [3] 유사 논문 검색 (E5-small → FAISS 검색)
│
└─▶ [4] 논문 기반 Q&A (E5 embedding + top-k chunk 검색 후 응답)

---

## 🚀 Tech Stacks
🤖 Transformers (HuggingFace): BART, E5

📄 문서 처리: PyMuPDF (PDF 텍스트 추출)

🧠 임베딩 및 검색: FAISS, LlamaIndex

💻 환경: Google Colab

---

## 📌 향후 확장 방향
- 논문 메타데이터 기반 필터링 기능 추가 (저자, 연도 등)
- LLM 연동 (OpenAI API 또는 Mistral, LLaMA 등) 통한 자연스러운 질의응답 개선
- 학습자 수준별 요약 (Beginner-Friendly 요약)

