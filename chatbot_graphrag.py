"""
GraphRAG 챗봇 서버
- ChromaDB 벡터 검색
- BGE-M3 Re-ranking (ColBERT)
- Ollama LLM 답변 생성
"""

import os
import sys
import json
import warnings
import ssl
import urllib3

# SSL 우회 설정
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

from flask import Flask, render_template, request, jsonify
import chromadb
from sentence_transformers import SentenceTransformer
import requests

# 설정
CHROMA_PATH = "chroma_db_fulltext"  # 원본 TXT 파일 임베딩
COLLECTION_NAME = "dtro_fulltext_v1"
MODEL_NAME = "BAAI/bge-m3"
OLLAMA_MODEL = "hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M"  # 기본 모델
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

# 현재 선택된 모델 (동적 변경 가능)
current_model = OLLAMA_MODEL

# 디폴트 프롬프트 (최적화된 기본값)
DEFAULT_PROMPT_TEMPLATE = """당신은 대구교통공사의 규정 및 내규 전문가입니다.

**핵심 원칙:**
1. 제공된 문서의 정보만 사용하세요
2. 정보가 없으면 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 명확히 답변하세요
3. 숫자나 기간 계산이 필요한 경우, 단계별로 계산 과정을 보여주세요
   예: "9급→8급: 1년, 8급→7급: 1년, 7급→6급: 1년 6개월 → 총 3년 6개월"
4. 관련 규정이나 조항을 반드시 인용하세요

**답변 형식:**
- 명확하고 구조화된 답변
- 필요시 번호 매기기나 불릿 포인트 사용
- 출처 문서명 명시

**참고 문서:**
{context}

**사용자 질문:**
{query}

**답변:**"""

# 현재 사용 중인 프롬프트 (동적 변경 가능)
current_prompt_template = DEFAULT_PROMPT_TEMPLATE

# Re-ranking 설정
ENABLE_RERANK = True
TOP_K = 20  # 1차 검색
RERANK_TOP_K = 5  # 2차 반환

app = Flask(__name__)

print("="*60)
print("🚀 GraphRAG 챗봇 초기화 중...")
print("="*60)

# 임베딩 모델 로드
print(f"📦 임베딩 모델 로딩: {MODEL_NAME}")
try:
    embedding_model = SentenceTransformer(MODEL_NAME)
    embedding_model.max_seq_length = 8192
    print(f"✅ 임베딩 모델 로드 완료 (차원: {embedding_model.get_sentence_embedding_dimension()})")
except Exception as e:
    print(f"❌ 임베딩 모델 로딩 실패: {e}")
    sys.exit(1)

# Re-ranker 초기화
reranker = None
if ENABLE_RERANK:
    print("🔄 Re-ranker 초기화 중...")
    try:
        from FlagEmbedding import BGEM3FlagModel
        reranker = BGEM3FlagModel(MODEL_NAME, use_fp16=True)
        print("✅ Re-ranker 로드 완료 (ColBERT 모드)")
    except ImportError:
        print("⚠️  FlagEmbedding 미설치 - 기본 유사도 Re-ranking 사용")
        reranker = embedding_model
    except Exception as e:
        print(f"⚠️  Re-ranker 로딩 실패: {e} - Re-ranking 비활성화")
        ENABLE_RERANK = False

# ChromaDB 연결
print(f"💾 ChromaDB 연결 중: {CHROMA_PATH}")
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    doc_count = collection.count()
    print(f"✅ ChromaDB 연결 성공 (문서 수: {doc_count}개)")
except Exception as e:
    print(f"❌ ChromaDB 연결 실패: {e}")
    sys.exit(1)

# Ollama 연결 확인
print(f"🤖 Ollama 연결 확인: {OLLAMA_MODEL}")
try:
    test_response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": "테스트",
            "stream": False
        },
        timeout=10
    )
    if test_response.status_code == 200:
        print("✅ Ollama 연결 성공")
    else:
        print(f"⚠️  Ollama 응답 이상: {test_response.status_code}")
except Exception as e:
    print(f"⚠️  Ollama 연결 실패: {e}")
    print("   (챗봇은 실행되지만 답변 생성 불가)")


# ==========================================
# RAG 검색 함수
# ==========================================

def search_chromadb(query: str, top_k: int = TOP_K):
    """ChromaDB에서 벡터 검색"""
    try:
        # 쿼리 임베딩
        query_embedding = embedding_model.encode(query, normalize_embeddings=True)
        
        # 검색
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # 결과 파싱
        documents = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                doc = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else 0.0
                }
                documents.append(doc)
        
        return documents
    
    except Exception as e:
        print(f"❌ 검색 실패: {e}")
        return []


def rerank_results(query: str, documents: list, top_k: int = RERANK_TOP_K):
    """Re-ranking으로 검색 결과 재정렬"""
    if not ENABLE_RERANK or not reranker or len(documents) == 0:
        return documents[:top_k]
    
    try:
        # 문서 텍스트 추출
        doc_texts = [doc['text'] for doc in documents]
        
        # Re-ranking
        if hasattr(reranker, 'compute_score'):
            # FlagEmbedding (ColBERT)
            sentence_pairs = [[query, text] for text in doc_texts]
            scores = reranker.compute_score(
                sentence_pairs,
                weights_for_different_modes=[0.0, 0.0, 1.0]  # ColBERT만 사용
            )
            
            # 점수가 딕셔너리인 경우 처리
            if isinstance(scores, dict):
                scores = scores.get('colbert', [0] * len(doc_texts))
        else:
            # SentenceTransformer (코사인 유사도)
            query_emb = reranker.encode(query, normalize_embeddings=True)
            doc_embs = reranker.encode(doc_texts, normalize_embeddings=True)
            scores = [float(query_emb @ doc_emb) for doc_emb in doc_embs]
        
        # 점수 추가 및 정렬
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[i]) if i < len(scores) else 0.0
        
        # 점수 기준 정렬
        documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return documents[:top_k]
    
    except Exception as e:
        print(f"⚠️  Re-ranking 실패: {e} - 원본 순서 반환")
        return documents[:top_k]


def generate_answer_ollama(query: str, context_docs: list):
    """Ollama로 답변 생성"""
    
    # 컨텍스트 구성
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        meta = doc['metadata']
        context_parts.append(f"""
[문서 {i}]
카테고리: {meta.get('category', 'N/A')} > {meta.get('group', 'N/A')}
문서명: {meta.get('source_file', 'N/A')}
타입: {meta.get('type', 'N/A')}
내용:
{doc['text']}
""")
    
    context = "\n".join(context_parts)
    
    # 동적 프롬프트 템플릿 사용
    prompt = current_prompt_template.format(context=context, query=query)

    try:
        # Ollama API 호출
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": current_model,  # 동적으로 선택된 모델 사용
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 1024
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '답변 생성 실패')
            return answer.strip()
        else:
            return f"❌ Ollama 응답 오류 (상태 코드: {response.status_code})"
    
    except requests.exceptions.Timeout:
        return "⏱️ 답변 생성 시간 초과 (60초)"
    except Exception as e:
        return f"❌ 답변 생성 실패: {str(e)}"


# ==========================================
# Flask 라우트
# ==========================================

@app.route('/')
def index():
    return render_template('index_graphrag.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('message', '').strip()
        
        if not query:
            return jsonify({'error': '질문을 입력해주세요'}), 400
        
        # 1단계: ChromaDB 검색
        print(f"\n🔍 검색 쿼리: {query}")
        documents = search_chromadb(query, top_k=TOP_K)
        print(f"   📊 1차 검색: {len(documents)}개 발견")
        
        if not documents:
            return jsonify({
                'answer': '관련 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요.',
                'sources': []
            })
        
        # 2단계: Re-ranking
        if ENABLE_RERANK:
            documents = rerank_results(query, documents, top_k=RERANK_TOP_K)
            print(f"   🔄 Re-ranking: 상위 {len(documents)}개 선택")
        else:
            documents = documents[:RERANK_TOP_K]
        
        # 3단계: 답변 생성
        print(f"   🤖 답변 생성 중...")
        answer = generate_answer_ollama(query, documents)
        
        # 출처 정보 구성
        sources = []
        for i, doc in enumerate(documents, 1):
            meta = doc['metadata']
            source = {
                'index': i,
                'category': f"{meta.get('category', 'N/A')} > {meta.get('group', 'N/A')}",
                'file': meta.get('source_file', 'N/A'),
                'type': meta.get('type', 'N/A'),
                'label': meta.get('label', 'N/A'),
                'score': doc.get('rerank_score', doc.get('distance', 0.0))
            }
            sources.append(source)
        
        print(f"   ✅ 답변 완료\n")
        
        return jsonify({
            'answer': answer,
            'sources': sources
        })
    
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        return jsonify({'error': f'서버 오류: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health():
    """서버 상태 확인"""
    return jsonify({
        'status': 'ok',
        'model': current_model,
        'documents': collection.count(),
        'reranking': ENABLE_RERANK
    })


@app.route('/models', methods=['GET'])
def get_models():
    """Ollama에서 사용 가능한 모델 목록 조회"""
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get('models', []):
                models.append({
                    'name': model.get('name', ''),
                    'size': model.get('size', 0),
                    'modified': model.get('modified_at', '')
                })
            return jsonify({
                'models': models,
                'current': current_model
            })
        else:
            return jsonify({'error': 'Ollama 서버 응답 오류'}), 500
    except Exception as e:
        return jsonify({'error': f'모델 목록 조회 실패: {str(e)}'}), 500


@app.route('/models/select', methods=['POST'])
def select_model():
    """사용할 Ollama 모델 변경"""
    global current_model
    try:
        data = request.json
        model_name = data.get('model', '').strip()
        
        if not model_name:
            return jsonify({'error': '모델 이름이 필요합니다'}), 400
        
        # 모델 테스트
        test_response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": "테스트",
                "stream": False
            },
            timeout=10
        )
        
        if test_response.status_code == 200:
            current_model = model_name
            print(f"✅ 모델 변경: {current_model}")
            return jsonify({
                'success': True,
                'model': current_model
            })
        else:
            return jsonify({'error': f'모델 테스트 실패 (상태: {test_response.status_code})'}), 500
    
    except Exception as e:
        return jsonify({'error': f'모델 변경 실패: {str(e)}'}), 500


@app.route('/prompt', methods=['GET'])
def get_prompt():
    """현재 프롬프트 템플릿 조회"""
    return jsonify({
        'current': current_prompt_template,
        'default': DEFAULT_PROMPT_TEMPLATE
    })


@app.route('/prompt/update', methods=['POST'])
def update_prompt():
    """프롬프트 템플릿 변경"""
    global current_prompt_template
    try:
        data = request.json
        new_prompt = data.get('prompt', '').strip()
        
        if not new_prompt:
            return jsonify({'error': '프롬프트가 비어있습니다'}), 400
        
        # {context}와 {query} 플레이스홀더 확인
        if '{context}' not in new_prompt or '{query}' not in new_prompt:
            return jsonify({'error': '프롬프트에 {context}와 {query} 플레이스홀더가 필요합니다'}), 400
        
        current_prompt_template = new_prompt
        print(f"✅ 프롬프트 변경 완료")
        
        return jsonify({
            'success': True,
            'prompt': current_prompt_template
        })
    
    except Exception as e:
        return jsonify({'error': f'프롬프트 변경 실패: {str(e)}'}), 500


@app.route('/prompt/reset', methods=['POST'])
def reset_prompt():
    """프롬프트를 디폴트로 초기화"""
    global current_prompt_template
    current_prompt_template = DEFAULT_PROMPT_TEMPLATE
    print(f"✅ 프롬프트 디폴트로 초기화")
    
    return jsonify({
        'success': True,
        'prompt': current_prompt_template
    })


# ==========================================
# 실행
# ==========================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎉 GraphRAG 챗봇 서버 시작")
    print("="*60)
    print(f"📍 URL: http://localhost:5000")
    print(f"🤖 LLM: {OLLAMA_MODEL}")
    print(f"📚 문서 수: {doc_count}개")
    print(f"🔄 Re-ranking: {'활성화' if ENABLE_RERANK else '비활성화'}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
