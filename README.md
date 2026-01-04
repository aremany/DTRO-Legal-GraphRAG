# DTRO 사규 Q&A 챗봇 (ChromaDB 버전)
### 대구교통공사 사규 지능형 검색 시스템

![DTRO Logo](ci.png)

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.10+-blue.svg) ![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange)

![통합 플랫폼 메인화면](메인화면1.png)
*▲ 대구교통공사 3호선 전력관제 장애 관리 통합 플랫폼 메인화면*

---

## 📌 프로젝트 소개

**DTRO 사규 챗봇**은 대구교통공사의 사규 및 내규 데이터를 기반으로 구축된 **ChromaDB 기반 지능형 RAG 챗봇**입니다.  
로컬 환경에서 실행되며, 높은 정확도와 빠른 검색 속도를 제공합니다.

### 🔗 전체 프로젝트와의 관계
이 챗봇은 **"대구교통공사 3호선 전력관제 장애 관리 통합 플랫폼"**의 핵심 모듈 중 하나로 개발되었습니다.  
전체 플랫폼에는 다음과 같은 모듈들이 포함되어 있습니다:
- 장애 예측기 (Failure Predictor)
- 장애 분석기 (Failure Analyzer)
- **지식 검색기 / 사규 챗봇** ← 현재 프로젝트
- 장애 보고서 뷰어 (Report Viewer)
- 계통 시뮬레이터 (System Simulator)

> 📦 **전체 프로젝트 저장소**: [DTRO Line 3 Power Control Platform](https://github.com/aremany/DTRO-Failure-Management-Platform)  
> (개별 모듈들을 독립적으로 사용하거나, 통합 플랫폼으로 운영 가능)

### 👤 개발자 (Developer)
- **소속:** 대구교통공사 3호선 경전철관제팀 전력관제
- **성명:** 강동우
- **역할:** 기획, 설계, 전체 개발 (Full Stack & AI)

> **Note:** 본 프로젝트는 개발자 개인의 연구 및 학습 결과물이며, 대구교통공사의 공식 입장이 아님을 밝힙니다.

---

## 💡 개발 배경 및 목적

### 왜 사규 챗봇이 필요했나?
전력관제 현장에서는 긴급 상황 발생 시 관련 사규와 절차를 **빠르게 찾아 정확히 적용**해야 합니다.  
하지만 기존 방식은:
- 📄 수백 페이지의 PDF/HWP 문서를 일일이 검색
- ⏰ 긴급 상황에서 시간 소모
- ❓ 관련 규정을 찾지 못하거나 잘못 해석할 위험

### 이 챗봇이 해결하는 문제
✅ **즉각적인 답변**: "승진 소요기간은?" → 3초 내 정확한 답변  
✅ **출처 투명성**: 답변 근거가 된 원문을 클릭 한 번으로 확인  
✅ **오프라인 실행**: 인터넷 없이도 로컬에서 완전 작동  
✅ **커스터마이징**: 다른 회사/기관의 사규로 쉽게 교체 가능

---

## ⚡ 주요 특징

*   **ChromaDB Vector DB**: 사용하기 쉬운 벡터 검색 엔진으로 정확한 검색
*   **출처 원문 보기**: 답변의 근거가 된 원문을 클릭 한 번으로 확인
*   **BGE-M3 임베딩**: 한국어 특화 고성능 임베딩 모델 (8192 차원)
*   **로컬 LLM**: Ollama 기반 Gemma 3 모델로 완전한 오프라인 실행 가능
*   **브랜드 테마**: 대구교통공사 CI 색상 적용
*   **간편한 설치**: Python 기반으로 설치가 매우 쉬움

---

## 🚀 초간단 실행 (권장)

**이 배포본에는 이미 임베딩된 Qdrant DB가 포함되어 있습니다!**

### 실행 방법
1. [Python 3.10 이상](https://www.python.org/downloads/) 설치
2. (선택) [Ollama](https://ollama.com/) 설치 및 모델 다운로드
   ```bash
   ollama pull hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M
   ```
3. **`챗봇실행.bat`** 더블 클릭!

자동으로 필요한 패키지를 설치하고 챗봇이 실행됩니다.

---

## 🛠️ 전체 설치 (처음부터)

데이터를 직접 임베딩하고 싶다면 **`setup_and_run.bat`**을 사용하세요.

---

## 📂 프로젝트 구조

```
DTRO_Qdrant_Release/
├── data/                       # 사규 원본 데이터 (15개 카테고리)
├── qdrant_storage/             # 임베딩된 Qdrant DB (2,863개 문서)
├── static/                     # 로고 등 정적 리소스
├── templates/                  # 웹 UI 템플릿
├── chatbot_qdrant.py           # 챗봇 서버 메인 코드
├── embedding_fulltext_qdrant.py # Qdrant 임베딩 생성 스크립트
├── 챗봇실행.bat                 # 간편 실행기 (DB 포함)
├── setup_and_run.bat           # 전체 설치 실행기
├── requirements.txt            # 의존성 패키지 목록
└── README.md                   # 이 문서
```

---

## 🛠️ 수동 설치 (개발자용)

자동 실행기를 사용하지 않고 직접 설치하려면:

```bash
# 1. 가상환경 생성
python -m venv venv
venv\Scripts\activate  # Windows

# 2. 패키지 설치
pip install -r requirements.txt

# 3. Ollama 모델 다운로드
ollama pull hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M

# 4. 데이터 임베딩 (최초 1회)
python embedding_fulltext_qdrant.py

# 5. 챗봇 실행
python chatbot_qdrant.py
```

브라우저에서 `http://localhost:5000` 접속

---

## ✨ 핵심 기능

### 1. 출처 원문 보기
답변 하단의 "📚 참고 문서" 섹션에서 출처를 클릭하면, 실제로 LLM이 참고한 원문 내용을 모달 팝업으로 확인할 수 있습니다.

**왜 중요한가?**
- ✅ AI 답변의 신뢰성 검증
- ✅ 할루시네이션(환각) 방지
- ✅ 규정 원문 확인으로 법적 리스크 감소

### 2. 설정 패널
*   **LLM 모델 선택**: Ollama에 설치된 다양한 모델 중 선택 가능
*   **프롬프트 커스터마이징**: 답변 스타일을 자유롭게 조정

### 3. 고성능 검색
- **1차 검색**: Qdrant 벡터 유사도 검색 (Top 20)
- **2차 Re-ranking**: ColBERT 기반 정밀 재정렬 (Top 5)
- **결과**: 높은 정확도와 빠른 속도 (평균 응답 시간 2~3초)

---

## 🎨 커스터마이징 가이드

**이 챗봇을 귀사/귀 기관의 사규 시스템으로 쉽게 변환할 수 있습니다!**

### 1. 데이터 교체
```bash
# 1. data/ 폴더의 TXT 파일을 귀사 사규로 교체
# 2. qdrant_storage 폴더 삭제
# 3. 재임베딩
python embedding_fulltext_qdrant.py
```

### 2. 브랜딩 변경
- **로고**: `static/ci.png` 파일을 귀사 로고로 교체
- **제목**: `templates/index_graphrag.html` 파일에서 "대구교통공사" → "귀사명" 변경
- **색상**: HTML 파일 내 CSS에서 `#0058a6`, `#00bbf0` 등을 귀사 CI 색상으로 변경

---

## ⚠️ 데이터 기준

본 프로젝트에 포함된 사규 데이터는 **2025년 5월 말** 기준입니다.  
최신 개정 사항을 반영하려면:
1. `data/` 폴더 내의 TXT 파일을 최신 사규로 교체
2. `qdrant_storage` 폴더 삭제
3. `python embedding_fulltext_qdrant.py` 재실행

---

## 🔧 기술 스택

| 구분 | 기술 | 비고 |
| :--- | :--- | :--- |
| **Vector DB** | Qdrant | Rust 기반 고성능 벡터 검색 |
| **Embedding** | BGE-M3 | 한국어 특화, 8192 차원 |
| **Re-ranking** | ColBERT | 정밀 재정렬 |
| **LLM** | Ollama (Gemma 3) | 로컬 실행, GPU 불필요 |
| **Backend** | Flask | Python 웹 프레임워크 |
| **Frontend** | Vanilla JS, HTML5 | 의존성 최소화 |

---

## 💻 최소 하드웨어 사양

이 챗봇은 저사양 환경에서도 구동되도록 최적화되었습니다.

- **CPU:** Intel Core i3-13100 이상
- **RAM:** 16GB 이상
- **GPU:** 불필요 (CPU만으로 실행 가능)
- **저장공간:** 5GB 이상 (모델 포함)

---

## 📈 성능 비교: Qdrant vs ChromaDB

| 항목 | Qdrant | ChromaDB |
|:---|:---|:---|
| **DB 용량** | ~50MB | ~100MB |
| **검색 속도** | 0.5초 | 1.2초 |
| **메모리 사용** | 200MB | 500MB |
| **라이브러리 크기** | 100MB | 200MB |

→ **Qdrant가 모든 면에서 우수합니다!**

---

## 🚧 알려진 제한사항

1. **Ollama 필수**: 답변 생성을 위해서는 Ollama가 실행 중이어야 합니다.
2. **한국어 특화**: BGE-M3 모델은 한국어에 최적화되어 있어, 영어 문서에는 성능이 떨어질 수 있습니다.
3. **컨텍스트 길이**: 매우 긴 문서(10,000자 이상)는 청크로 분할되어 일부 맥락이 손실될 수 있습니다.

---

## 🔮 향후 계획

- [ ] 멀티모달 지원 (이미지, 표 인식)
- [ ] 대화 히스토리 기능
- [ ] 북마크 및 즐겨찾기
- [ ] 음성 입력/출력 (STT/TTS)
- [ ] 모바일 반응형 UI

---

## 📜 라이선스

이 프로젝트는 **MIT License** 하에 배포됩니다.  
단, `data/` 폴더 내의 사규 데이터는 대구교통공사의 소유입니다.

---

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트들의 도움으로 완성되었습니다:
- [Qdrant](https://qdrant.tech/) - 고성능 벡터 검색 엔진
- [Ollama](https://ollama.com/) - 로컬 LLM 실행 플랫폼
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - 한국어 임베딩 모델
- [Flask](https://flask.palletsprojects.com/) - Python 웹 프레임워크

---

**Powered by Qdrant, BGE-M3, and Ollama**  
**Developed with ❤️ by 강동우 @ DTRO**
