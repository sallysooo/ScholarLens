#!pip install transformers
#!pip install PyMuPDF

import fitz  # PyMuPDF
from transformers import pipeline

# 논문 텍스트 추출
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() # 각 페이지에서 텍스트 추출
    return full_text # 모든 페이지를 이어붙여 전체 논문 텍스트 획득

# 텍스트를 BART의 최대 입력 길이인 1024 tokens 이하 chunck로 나누는 함수 (긴 논문은 쪼개서 처리해야 함)
def split_text(text, max_length=1000):
    import textwrap
    return textwrap.wrap(text, width=max_length)

# 요약 파이프라인
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_chunks(chunks):
    summaries = []
    for i, chunk in enumerate(chunks):
        # 생성될 요약 문장의 최대 길이: 200, 최소 요약 길이: 50, deterministic하게 생성하여 일관된 요약 생성
        summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return summaries # 요약 결과는 chunck 수만큼 나옴

pdf_path = "/content/drive/MyDrive/Colab Notebooks/Project Report Predicting Used Car Price.pdf"  # 파일 업로드 후 경로 설정
text = extract_text_from_pdf(pdf_path)
chunks = split_text(text, max_length=1000)
summaries = summarize_chunks(chunks)

for idx, summary in enumerate(summaries):
    print(f"\n[요약 {idx+1}]\n{summary}")