import json
import os
import pickle
import time
import networkx as nx
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import urllib3
import ssl
from text_processor import TextProcessor

# SSL 인증서 검증 무시 설정
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 설정
JSON_PATH = "total_graph.json"
MAPPING_PATH = "source_file_mapping.json"
CHROMA_PATH = "chroma_db"
GRAPH_PKL_PATH = "graph_data.pkl"
MODEL_NAME = "BAAI/bge-m3"

# 청킹 설정
ENABLE_CHUNKING = True          # 청킹 활성화 여부
CHUNKING_METHOD = 'semantic'    # 'semantic' 또는 'fixed'
CHUNK_THRESHOLD = 1000          # 이 길이 이상이면 청킹

# 임베딩 설정
BATCH_SIZE = 64                 # 배치 크기 (구석기: 100, 기존: 32 → 중간값 64)
MAX_SEQ_LENGTH = 8192           # BGE-M3 최대 토큰 길이

# 통계
stats = {
    'total_nodes': 0,
    'chunked_nodes': 0,
    'total_chunks': 0,
    'embedding_success': 0,
    'embedding_failed': 0,
    'start_time': 0,
    'end_time': 0
}

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_networkx_graph(data):
    print("🕸️  NetworkX 그래프 생성 중...")
    G = nx.DiGraph()
    
    # 노드 추가
    for node in data['nodes']:
        G.add_node(node['id'], **node)
        
    # 엣지 추가
    for edge in data['edges']:
        G.add_edge(
            edge['source'], 
            edge['target'], 
            relationship=edge['relationship'],
            **edge.get('properties', {})
        )
        
    print(f"   - 노드 수: {G.number_of_nodes()}")
    print(f"   - 엣지 수: {G.number_of_edges()}")
    
    # 저장
    with open(GRAPH_PKL_PATH, 'wb') as f:
        pickle.dump(G, f)
    print(f"✅ 그래프 저장 완료: {GRAPH_PKL_PATH}")
    return G

def find_source_info(filename, mapping):
    """파일명으로 카테고리와 전체 경로를 찾음"""
    if not filename:
        return "Unknown", "Unknown", "Unknown"
        
    target_file = filename.strip()
    
    for json_group, files in mapping.items():
        for full_path in files:
            if os.path.basename(full_path) == target_file:
                parts = full_path.replace("\\", "/").split("/")
                
                try:
                    idx = parts.index("분류작업")
                    category = parts[idx+1]
                    group = parts[idx+2] if idx+2 < len(parts)-1 else "N/A"
                    return category, group, full_path
                except ValueError:
                    return "Unknown", "Unknown", full_path
                    
    return "Unknown", "Unknown", "Unknown"

def serialize_node(node, mapping, text_processor=None):
    """
    노드를 임베딩용 텍스트로 변환 (청킹 지원)
    
    Returns:
        List[Tuple[str, dict]]: (텍스트, 메타데이터) 튜플 리스트
    """
    props = node.get('properties', {})
    source_file = props.get('source_file', '')
    
    category, group, full_path = find_source_info(source_file, mapping)
    
    # 기본 메타데이터
    base_metadata = {
        "node_id": node['id'],
        "type": node.get('type', 'N/A'),
        "label": node.get('label', 'N/A'),
        "category": category,
        "group": group,
        "source_file": source_file
    }
    
    # 기본 텍스트 (설명 제외)
    base_text_parts = [
        f"[카테고리] {category} > {group}",
        f"[문서] {source_file}",
        f"[타입] {node.get('type', 'N/A')}",
        f"[이름] {node.get('label', 'N/A')}"
    ]
    
    desc = props.get('description', '')
    
    # 청킹 필요 여부 판단
    if ENABLE_CHUNKING and text_processor and desc and text_processor.should_chunk(desc, CHUNK_THRESHOLD):
        # 청킹 수행
        chunks = text_processor.chunk_node_description(desc, node['id'], method=CHUNKING_METHOD)
        stats['chunked_nodes'] += 1
        stats['total_chunks'] += len(chunks)
        
        # 각 청크에 대해 텍스트 생성
        results = []
        for chunk in chunks:
            chunk_text_parts = base_text_parts.copy()
            chunk_text_parts.append(f"[설명 {chunk['chunk_index']+1}/{len(chunks)}] {chunk['content']}")
            
            # 메타데이터에 청크 정보 추가
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_index': chunk['chunk_index'],
                'total_chunks': len(chunks),
                'chunking_method': chunk['method']
            })
            
            results.append(("\n".join(chunk_text_parts), chunk_metadata))
        
        return results
    else:
        # 청킹 불필요 - 단일 텍스트 반환
        text_parts = base_text_parts.copy()
        
        if desc:
            # 전처리만 적용
            cleaned_desc = text_processor.clean_text(desc) if text_processor else desc
            text_parts.append(f"[설명] {cleaned_desc}")
        
        # 기타 속성 추가
        for k, v in props.items():
            if k not in ['source_file', 'description', 'cite_pages']:
                text_parts.append(f"[{k}] {v}")
        
        metadata = base_metadata.copy()
        metadata['chunked'] = False
        
        return [("\n".join(text_parts), metadata)]

def build_chroma_db(data, mapping):
    print(f"💾 ChromaDB 구축 시작 (모델: {MODEL_NAME})...")
    print(f"   🧠 청킹: {'활성화' if ENABLE_CHUNKING else '비활성화'} ({CHUNKING_METHOD if ENABLE_CHUNKING else 'N/A'})")
    print(f"   📦 배치 크기: {BATCH_SIZE}")
    print(f"   📏 최대 토큰: {MAX_SEQ_LENGTH}")
    
    stats['start_time'] = time.time()
    stats['total_nodes'] = len(data['nodes'])
    
    # 1. 모델 로드
    print("   🤖 모델 로딩 중...")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    print(f"   ✅ 모델 로드 완료 (차원: {model.get_sentence_embedding_dimension()})")
    
    # 2. 텍스트 처리기 초기화
    text_processor = TextProcessor() if ENABLE_CHUNKING else None
    
    # 3. DB 초기화
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="dtro_graph_v1")
    
    # 4. 데이터 준비 (청킹 적용)
    all_documents = []
    all_metadatas = []
    all_ids = []
    
    print("   📝 데이터 전처리 및 청킹 중...")
    for node in tqdm(data['nodes'], desc="노드 처리", unit="node"):
        # 노드를 텍스트로 변환 (청킹 포함)
        text_metadata_pairs = serialize_node(node, mapping, text_processor)
        
        for idx, (text, meta) in enumerate(text_metadata_pairs):
            # 고유 ID 생성 (청크가 여러 개인 경우 _chunk_{idx} 추가)
            if len(text_metadata_pairs) > 1:
                doc_id = f"{node['id']}_chunk_{idx}"
            else:
                doc_id = node['id']
            
            all_documents.append(text)
            all_metadatas.append(meta)
            all_ids.append(doc_id)
    
    total = len(all_documents)
    print(f"   📊 총 {total}개 문서 생성 (원본 노드: {stats['total_nodes']}개)")
    if stats['chunked_nodes'] > 0:
        print(f"   ✂️  청킹된 노드: {stats['chunked_nodes']}개 → {stats['total_chunks']}개 청크")
    
    # 5. 임베딩 및 저장 (배치 처리 + 실패 처리)
    print(f"   🚀 임베딩 생성 및 저장 시작...")
    
    for i in tqdm(range(0, total, BATCH_SIZE), desc="임베딩 배치", unit="batch"):
        batch_docs = all_documents[i:i+BATCH_SIZE]
        batch_metas = all_metadatas[i:i+BATCH_SIZE]
        batch_ids = all_ids[i:i+BATCH_SIZE]
        
        try:
            # 임베딩 생성
            embeddings = model.encode(
                batch_docs, 
                normalize_embeddings=True,
                show_progress_bar=False,  # tqdm과 중복 방지
                convert_to_numpy=True  # numpy array로 변환
            )
            
            # numpy array를 list로 변환 (ChromaDB 호환성)
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
            # Tensor 리스트인 경우 처리
            elif isinstance(embeddings, list) and len(embeddings) > 0:
                import torch
                if isinstance(embeddings[0], torch.Tensor):
                    embeddings = [emb.cpu().numpy().tolist() for emb in embeddings]
            
            # DB 추가
            collection.add(
                embeddings=embeddings,
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            
            stats['embedding_success'] += len(batch_docs)
            
        except Exception as e:
            print(f"\n   ⚠️  배치 {i//BATCH_SIZE + 1} 임베딩 실패: {e}")
            stats['embedding_failed'] += len(batch_docs)
            
            # 개별 처리 시도 (실패 복구)
            for j, (doc, meta, doc_id) in enumerate(zip(batch_docs, batch_metas, batch_ids)):
                try:
                    emb = model.encode([doc], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
                    if hasattr(emb, 'tolist'):
                        emb = emb.tolist()
                    elif isinstance(emb, list) and len(emb) > 0:
                        import torch
                        if isinstance(emb[0], torch.Tensor):
                            emb = [e.cpu().numpy().tolist() for e in emb]
                    
                    collection.add(
                        embeddings=emb,
                        documents=[doc],
                        metadatas=[meta],
                        ids=[doc_id]
                    )
                    stats['embedding_success'] += 1
                    stats['embedding_failed'] -= 1
                except:
                    print(f"      ❌ 문서 {doc_id} 임베딩 실패 (건너뜀)")
    
    stats['end_time'] = time.time()
    
    print(f"\n✅ ChromaDB 저장 완료: {CHROMA_PATH}")
    print_statistics()

def print_statistics():
    """통계 출력"""
    elapsed = stats['end_time'] - stats['start_time']
    
    print("\n" + "="*60)
    print("📊 구축 통계")
    print("="*60)
    print(f"🕐 처리 시간: {elapsed:.1f}초")
    print(f"📦 원본 노드: {stats['total_nodes']}개")
    
    if stats['chunked_nodes'] > 0:
        print(f"✂️  청킹 적용: {stats['chunked_nodes']}개 노드 → {stats['total_chunks']}개 청크")
        avg_chunks = stats['total_chunks'] / stats['chunked_nodes']
        print(f"   평균 청크 수: {avg_chunks:.1f}개/노드")
    
    total_docs = stats['embedding_success'] + stats['embedding_failed']
    print(f"📄 총 문서 수: {total_docs}개")
    print(f"✅ 임베딩 성공: {stats['embedding_success']}개")
    print(f"❌ 임베딩 실패: {stats['embedding_failed']}개")
    
    if total_docs > 0:
        success_rate = (stats['embedding_success'] / total_docs) * 100
        print(f"📈 성공률: {success_rate:.1f}%")
    
    if stats['embedding_success'] > 0 and elapsed > 0:
        throughput = stats['embedding_success'] / elapsed
        print(f"⚡ 처리 속도: {throughput:.1f} docs/sec")
    
    print("="*60)

def test_gpu_performance(model, test_size=10):
    """GPU 성능 벤치마크"""
    print(f"\n🧪 GPU 성능 테스트 ({test_size}개 샘플)...")
    
    test_texts = [
        f"테스트 문장 {i}: BGE-M3 모델의 GPU 가속 성능을 측정하기 위한 샘플 텍스트입니다. "
        f"이 텍스트는 실제 노드 설명과 유사한 길이로 작성되었습니다." * 5
        for i in range(test_size)
    ]
    
    start = time.time()
    embeddings = model.encode(test_texts, normalize_embeddings=True, show_progress_bar=False)
    end = time.time()
    
    elapsed = end - start
    throughput = test_size / elapsed if elapsed > 0 else 0
    
    print(f"   ⏱️  시간: {elapsed:.2f}초")
    print(f"   🚀 속도: {throughput:.1f} texts/sec")
    print(f"   📊 평균: {(elapsed/test_size)*1000:.1f}ms/text")

def main():
    if not os.path.exists(JSON_PATH):
        print(f"❌ 파일 없음: {JSON_PATH}")
        return
    
    print("="*60)
    print("🚀 GraphRAG 구축 시작 (고도화 버전)")
    print("="*60)
        
    # 데이터 로드
    print("\n📂 데이터 로딩 중...")
    graph_data = load_json(JSON_PATH)
    mapping_data = load_json(MAPPING_PATH) if os.path.exists(MAPPING_PATH) else {}
    print(f"   ✅ 그래프 데이터: 노드 {len(graph_data['nodes'])}개, 엣지 {len(graph_data['edges'])}개")
    
    # 1. NetworkX 그래프 빌드
    print("\n" + "="*60)
    build_networkx_graph(graph_data)
    
    # 2. ChromaDB 빌드 (청킹, 진행률, 통계 포함)
    print("\n" + "="*60)
    build_chroma_db(graph_data, mapping_data)
    
    print("\n" + "="*60)
    print("🎉 모든 구축 작업이 완료되었습니다!")
    print("="*60)

if __name__ == "__main__":
    main()
