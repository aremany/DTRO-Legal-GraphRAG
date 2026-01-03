"""
원본 TXT 파일을 ChromaDB에 임베딩
- 144개 원본 규정 파일 처리
- 800자 청크 분할 (200자 오버랩)
- GraphRAG 메타데이터 보강
"""

import os
import json
import re
import warnings
import ssl
import urllib3
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# SSL 우회
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

import chromadb
from sentence_transformers import SentenceTransformer

# 설정
BASE_DIR = Path(__file__).parent / "data"
CHROMA_PATH = "chroma_db_fulltext"  # 새로운 컬렉션
COLLECTION_NAME = "dtro_fulltext_v1"
MODEL_NAME = "BAAI/bge-m3"
RULE_MD_PATH = "rule.md"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

print("="*60)
print("📚 원본 TXT 파일 임베딩 시작")
print("="*60)

# 모델 로드
print(f"📦 모델 로딩: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
model.max_seq_length = 8192
print(f"✅ 모델 로드 완료 (차원: {model.get_sentence_embedding_dimension()})")

# ChromaDB 초기화
print(f"💾 ChromaDB 초기화: {CHROMA_PATH}")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
print("✅ ChromaDB 준비 완료")


def parse_rule_md(rule_path: str) -> Dict[str, str]:
    """rule.md 파싱하여 카테고리 매핑 반환"""
    mappings = {}
    
    if not os.path.exists(rule_path):
        print(f"⚠️  {rule_path} 파일 없음")
        return mappings
    
    with open(rule_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 카테고리별 그룹 파싱
    pattern = r'## (\d+_[^\n]+)\n(.*?)(?=\n##|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for category, groups_text in matches:
        # 그룹별 파일 파싱
        group_pattern = r'### (그룹\d+)\n```\n(.*?)\n```'
        group_matches = re.findall(group_pattern, groups_text, re.DOTALL)
        
        for group, files_text in group_matches:
            files = [f.strip() for f in files_text.split('\n') if f.strip()]
            for filename in files:
                mappings[filename] = {
                    'category': category,
                    'group': group
                }
    
    return mappings


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """텍스트를 오버랩이 있는 청크로 분할"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 청크 추출
        chunk = text[start:end]
        
        # 마지막 청크가 아니면 문장 경계에서 자르기
        if end < len(text):
            # 마지막 마침표 찾기
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            
            cut_point = max(last_period, last_newline)
            if cut_point > chunk_size * 0.5:  # 너무 짧아지지 않도록
                chunk = chunk[:cut_point + 1]
                end = start + len(chunk)
        
        chunks.append(chunk.strip())
        
        # 다음 시작점 (오버랩 적용)
        start = end - overlap
        
        # 무한 루프 방지
        if start <= 0 or start >= len(text):
            break
    
    return chunks


def load_txt_files(base_dir: Path, mappings: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """모든 TXT 파일 로드 및 청킹"""
    documents = []
    
    # 카테고리 폴더 탐색
    category_folders = [
        "01_조직경영", "02_인사노무", "03_재무회계", "04_운전운행",
        "05_차량검수", "06_선로궤도", "07_전기설비", "08_신호통신",
        "09_건축기계", "10_안전보안", "11_고객서비스", "12_감사청렴",
        "13_사무행정", "14_연구기획", "15_기타특수"
    ]
    
    total_files = 0
    total_chunks = 0
    
    for category in tqdm(category_folders, desc="카테고리 처리"):
        category_path = base_dir / category
        
        if not category_path.exists():
            continue
        
        # 그룹 폴더 탐색
        for group_path in category_path.iterdir():
            if not group_path.is_dir():
                continue
            
            group_name = group_path.name
            
            # TXT 파일 처리
            for txt_file in group_path.glob("*.txt"):
                filename = txt_file.name
                
                # 매핑 정보 가져오기
                file_info = mappings.get(filename, {
                    'category': category,
                    'group': group_name
                })
                
                try:
                    # 파일 읽기
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 청킹
                    chunks = chunk_text(content)
                    total_files += 1
                    total_chunks += len(chunks)
                    
                    # 각 청크를 문서로 추가
                    for i, chunk in enumerate(chunks):
                        doc = {
                            'id': f"{filename}_{i}",
                            'text': chunk,
                            'metadata': {
                                'source_file': filename,
                                'category': file_info['category'],
                                'group': file_info['group'],
                                'chunk_index': i,
                                'total_chunks': len(chunks),
                                'file_path': str(txt_file)
                            }
                        }
                        documents.append(doc)
                
                except Exception as e:
                    print(f"\n⚠️  파일 처리 실패: {filename} - {e}")
    
    print(f"\n📊 처리 완료:")
    print(f"   - 파일 수: {total_files}개")
    print(f"   - 청크 수: {total_chunks}개")
    
    return documents


def upload_to_chromadb(documents: List[Dict[str, Any]], batch_size: int = 64):
    """ChromaDB에 임베딩 및 업로드"""
    print(f"\n🚀 임베딩 생성 및 업로드 시작 (배치 크기: {batch_size})")
    
    total = len(documents)
    success = 0
    failed = 0
    
    for i in tqdm(range(0, total, batch_size), desc="임베딩 배치"):
        batch = documents[i:i+batch_size]
        
        try:
            # 텍스트 추출
            texts = [doc['text'] for doc in batch]
            ids = [doc['id'] for doc in batch]
            metadatas = [doc['metadata'] for doc in batch]
            
            # 임베딩 생성
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # numpy array를 list로 변환
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
            
            # ChromaDB 업로드
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            success += len(batch)
        
        except Exception as e:
            print(f"\n⚠️  배치 {i//batch_size + 1} 실패: {e}")
            failed += len(batch)
            
            # 개별 재시도
            for doc in batch:
                try:
                    emb = model.encode([doc['text']], normalize_embeddings=True, convert_to_numpy=True)
                    if hasattr(emb, 'tolist'):
                        emb = emb.tolist()
                    
                    collection.add(
                        embeddings=emb,
                        documents=[doc['text']],
                        metadatas=[doc['metadata']],
                        ids=[doc['id']]
                    )
                    success += 1
                    failed -= 1
                except:
                    print(f"   ❌ {doc['id']} 실패")
    
    print(f"\n📊 업로드 통계:")
    print(f"   ✅ 성공: {success}개")
    print(f"   ❌ 실패: {failed}개")
    print(f"   📈 성공률: {(success/(success+failed)*100):.1f}%")


def main():
    # 1. rule.md 파싱
    print("\n📖 rule.md 파싱 중...")
    mappings = parse_rule_md(RULE_MD_PATH)
    print(f"✅ {len(mappings)}개 파일 매핑 정보 로드")
    
    # 2. TXT 파일 로드 및 청킹
    print(f"\n📂 TXT 파일 로드 중: {BASE_DIR}")
    documents = load_txt_files(BASE_DIR, mappings)
    
    if not documents:
        print("❌ 처리할 문서가 없습니다!")
        return
    
    # 3. ChromaDB 업로드
    upload_to_chromadb(documents)
    
    # 4. 검증
    doc_count = collection.count()
    print(f"\n✅ ChromaDB 저장 완료:")
    print(f"   - 경로: {CHROMA_PATH}")
    print(f"   - 컬렉션: {COLLECTION_NAME}")
    print(f"   - 문서 수: {doc_count}개")
    
    print("\n" + "="*60)
    print("🎉 원본 TXT 파일 임베딩 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
