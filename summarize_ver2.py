# 필수 라이브러리 설치 (최초 1회 실행)
#!pip install -q transformers torch sentence-transformers PyMuPDF gradio keybert nltk

# 전체 구현 코드
import fitz
import re
import torch
import gradio as gr
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from keybert import KeyBERT
from nltk import sent_tokenize
import nltk
import subprocess

# NLTK 리소스 다운로드
subprocess.run(["python", "-m", "nltk.downloader", "punkt"], check=True)

# GPU 메모리 관리 초기화
def clear_gpu_cache():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

# 텍스트 전처리 파이프라인
def clean_text(text):
    # 헤더/푸터 패턴 제거
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)
    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'(\n\s*)+\n', '\n\n', text)
    
    # 표/그림 설명 제거
    text = re.sub(r'Table\s+\d+:.+?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Figure\s+\d+:.+?\n', '', text, flags=re.IGNORECASE)
    
    # 특수 문자 정규화
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'\xad', '', text)  # 소프트 하이픈 제거
    return text

# 지능형 텍스트 분할
def smart_split(text, max_len=800):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sent in sentences:
        sent_len = len(sent.split())
        if current_length + sent_len > max_len and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sent)
        current_length += sent_len
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# 요약 모델 초기화 (8비트 양자화 적용)
model_name = 'google/pegasus-large'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(
    model_name, 
    device_map="auto", 
    load_in_8bit=True
)

# 계층적 요약 시스템
def summarize(text, max_length=200):
    try:
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to('cuda')
        
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=50,
            num_beams=4,
            early_stopping=True
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        clear_gpu_cache()
        raise RuntimeError(f"요약 실패: {str(e)}")

# 섹션 분할 (사용자 정의 필요)
import re
from typing import List

def split_into_sections_advanced(text: str, max_len: int = 1500) -> List[str]:
    """계층적 섹션 분할 시스템"""
    
    # 1단계: 시각적 구분자 기반 분할 (헤더 스타일 감지)
    visual_patterns = [
        r'\n\s*[A-Z][A-Za-z\s]+\s*:\s*\n',  # "Introduction: " 형식
        r'\n\s*[A-Z][A-Za-z\s]+[\d.]*\s*\n{2,}',  # 2개 이상 줄바꿈으로 구분
        r'\n\s*§\d+\.?\s+[A-Z]',  # "§1. Introduction" 형식
        r'\n\s*\d+[\d.]*\s+[A-Z]'  # "1.1 Introduction" 형식
    ]
    
    # 2단계: 일반적인 학술 섹션 키워드 (확장 버전)
    section_keywords = [
        'abstract', 'introduction', 'related work', 'methodology',
        'experiment', 'results', 'discussion', 'conclusion',
        'acknowledg', 'reference', 'appendix', 'data availability',
        'conflict of interest', 'author contribution'
    ]
    
    # 3단계: 혼합 패턴 생성
    combined_patterns = []
    for kw in section_keywords:
        # 숫자+점/공백 조합 허용 (예: "1. ", "2 ")
        combined_patterns.append(rf'\n\s*\d*\.?\s*{kw}[^\n]*\s*\n')
        # 볼드/이탤릭 스타일 허용 (예: "**Introduction**")
        combined_patterns.append(rf'\n\s*[\*_]{{1,2}}{kw}[^\n]*[\*_]{{1,2}}\s*\n')
    
    # 모든 패턴 통합
    all_patterns = visual_patterns + combined_patterns
    
    # 경계점 탐지
    boundaries = []
    for pattern in all_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start = match.start()
            # 경계점 보정 (섹션 제목 시작 부분으로 조정)
            prev_newline = text.rfind('\n', 0, start)
            boundaries.append(prev_newline if prev_newline != -1 else start)
    
    # 4단계: 후처리
    boundaries = sorted(set(boundaries))
    
    # 인접 경계점 병합 (최소 50자 간격)
    merged = []
    for b in boundaries:
        if not merged or b - merged[-1] > 50:
            merged.append(b)
    
    # 5단계: 실제 분할 수행
    sections = []
    for i in range(len(merged)):
        start = merged[i]
        end = merged[i+1] if i+1 < len(merged) else len(text)
        section = text[start:end].strip()
        if len(section) > 100:
            sections.append(section)
    
    # 6단계: 분할 실패 시 글자 수 기반 분할
    if not sections:
        return [text[i:i+max_len] for i in range(0, len(text), max_len)]
    
    return sections


# 핵심 용어 추출
kw_model = KeyBERT()
def extract_key_terms(text, top_n=10):
    return [kw[0] for kw in kw_model.extract_keywords(
        text, 
        keyphrase_ngram_range=(1, 3), 
        stop_words='english', 
        top_n=top_n
    )]

# 전체 처리 파이프라인
def full_pipeline(pdf_path):
    try:
        # 1. PDF 텍스트 추출
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        # 2. 텍스트 정제
        cleaned = clean_text(full_text)
        
        # 3. 계층적 요약
        sections = split_into_sections(cleaned)
        section_summaries = [summarize(sect) for sect in sections[:5]]  # 상위 5개 섹션만 처리
        combined = '\n'.join(section_summaries)
        final_summary = summarize(combined, max_length=300)
        
        # 4. 키워드 추출
        keywords = extract_key_terms(cleaned)
        
        return final_summary, keywords
    
    except Exception as e:
        clear_gpu_cache()
        return f"오류 발생: {str(e)}", []

# Gradio 인터페이스
interface = gr.Interface(
    fn=full_pipeline,
    inputs=gr.File(label="📄 PDF 파일 업로드"),
    outputs=[
        gr.Textbox(label="📝 요약 결과", lines=10),
        gr.HighlightedText(label="🔑 핵심 용어", 
                         combine_adjacent=True,
                         show_legend=True)
    ],
    title="🧠 AI 논문 요약 시스템",
    description="학술 논문 PDF를 업로드하면 자동으로 요약과 핵심 용어를 추출합니다.",
    examples=[
        ["/content/sample.pdf"]  # 예시 파일 경로
    ]
)

# 인터페이스 실행
if __name__ == "__main__":
    interface.launch(debug=True, share=True)
