# DTRO Legal-GraphRAG
### 대구교통공사 사규 지능형 검색 시스템 (Intelligent Regulation Search System)

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.10+-blue.svg) ![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)

**Legal-GraphRAG**는 법제처 사규 데이터를 기반으로 구축된 **로컬 중심(Local-first) 지능형 챗봇 시스템**입니다. 단순한 키워드 매칭을 넘어, 규정 간의 복잡한 인용 관계와 맥락을 파악하는 GraphRAG 개념을 적용하여 정확하고 신뢰할 수 있는 답변을 제공합니다.

![DTRO Logo](static/ci.png)

> **⚠️ Data Disclaimer**: 본 프로젝트에 포함된 사규 데이터는 **2025년 5월 말** 기준입니다. 최신 개정 사항을 반영하려면 사규를 다시 다운로드하고 임베딩 과정을 수행해야 합니다.

---

## 🏗️ System Architecture & Workflow

이 프로젝트는 데이터의 보안와 프라이버시를 최우선으로 하는 **Local-first** 아키텍처를 채택하고 있습니다.

### 1. Data Pipeline 🔄
*   **Source**: 법제처 사규 TXT 데이터 (비정형 텍스트)
*   **Processing**: **Google Gemini 3 Pro (Fast Mode)**를 활용한 고속 분석 및 구조화
*   **Transformation**: 15개 카테고리별 엔티티(Entity) 추출, 관계 정의 및 JSON 병합(Merging) 수행

### 2. Knowledge Graph & Indexing 🕸️
*   **GraphRAG Concept**: 개별 문서 단위의 검색 한계를 극복하기 위해 카테고리 간 연결성 및 전역적 맥락(Global Context) 파악에 최적화된 인덱싱 구조 설계
*   **Vector Database**: **ChromaDB**를 사용하여 고차원 텍스트 임베딩 저장 및 밀리초 단위의 고속 유사도 검색 구현

### 3. Inference Engine (Local LLM) 🤖
*   **Runtime**: **Ollama** 프레임워크 기반
*   **Core Model**: **Gemma 3 4B (eb)** 모델 사용 (경량화된 고성능 로컬 모델)
*   **RAG Pipeline**: Query → Vector Search (ChromaDB) → Re-ranking (ColBERT) → Context Injection → LLM Generation

### 4. Backend & Security 🔒
*   **Fully Local**: 외부 서버로의 데이터 전송 없이 모든 추론 과정이 사내/로컬 네트워크 내부에서 수행됨
*   **Privacy-Preserving**: 민감한 사내 규정 및 질의 내용의 유출 원천 차단

---

## ⚡ Quick Start (초간단 실행)

**복잡한 명령어 없이 클릭만으로 실행할 수 있습니다.**

### 1단계: 사전 준비
*   [Python 3.10 이상](https://www.python.org/downloads/) 설치 (설치 시 'Add Python to PATH' 체크 필수)
*   [Ollama](https://ollama.com/) 설치 및 실행

### 2단계: 실행
폴더 내의 **`setup_and_run.bat`** 파일을 더블 클릭하세요.
*   자동으로 가상환경을 만들고 필요한 라이브러리를 설치합니다.
*   최초 실행 시 사규 데이터를 분석(임베딩)합니다.
*   모든 준비가 끝나면 **자동으로 챗봇 웹페이지가 열립니다.**

---

## 🛠️ Manual Installation (수동 설치 - 개발자용)

자동 실행기를 사용하지 않고 직접 설치하려면 아래 절차를 따르세요.

### 1. Installation

```bash
# Repository 클론
git clone https://github.com/your-username/Legal-GraphRAG.git
cd Legal-GraphRAG

# 가상환경 생성 및 활성화
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 의존성 패키지 설치
pip install -r requirements.txt
```

### 3. Model Setup (Ollama)

본 프로젝트는 `gemma-3n-4b` 계열 모델에 최적화되어 있습니다.

```bash
# Ollama에서 모델 다운로드 (예시)
ollama pull hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M
```

### 4. Data Embedding (Initialization)

최초 실행 시, 제공된 사규 데이터(`data/`)를 벡터화(Embedding)하여 ChromaDB에 적재해야 합니다.

```bash
python embedding_fulltext_chroma.py
```
> **Note**: 실행 후 `chroma_db_fulltext` 폴더가 생성되며, 약 2,800+개의 청크가 인덱싱됩니다.
>
> 🔄 **최신 데이터 적용 방법**:
> 1. `data/` 폴더 내의 기존 TXT 파일들을 최신 사규(2025.05 이후 개정본)로 교체합니다.
> 2. 위 임베딩 명령어를 다시 실행하여 DB를 갱신합니다.

### 5. Run Server

```bash
python chatbot_graphrag.py
```
브라우저에서 [http://localhost:5000](http://localhost:5000)으로 접속하여 챗봇을 사용합니다.

---

## 📁 Project Structure

```
Legal-GraphRAG/
├── data/                       # 사규 원본 데이터셋 (15 Categories)
├── static/                     # UI 리소스 (Logo 등)
├── templates/                  # Web Dashboard (HTML/JS)
├── chatbot_graphrag.py         # Main Flask Server & Inference Logic
├── embedding_fulltext_chroma.py # Vector Indexing Pipeline
├── build_graph_rag.py          # (Optional) Graph Builder Pipeline
├── rule.md                     # Data Categorization Rules
└── requirements.txt            # Python Dependencies
```

## 📜 License

이 프로젝트는 **MIT License** 하에 배포됩니다.
단, 포함된 **사규 데이터(`data/`)의 저작권 및 소유권은 해당 기관(대구교통공사 등)**에 있으며, 상업적 활용 시 주의가 필요합니다.

---

**Developed for Enterprise Legal AI Solutions.**
