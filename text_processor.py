"""
GraphRAG용 텍스트 전처리 및 청킹 모듈
구석기 버전의 장점을 GraphRAG에 이식
"""

import re
from typing import List, Dict, Any


class TextProcessor:
    """노드 설명문을 청킹하고 전처리하는 클래스"""
    
    def __init__(self, 
                 chunk_size: int = 800,
                 overlap: int = 100,
                 min_chunk_size: int = 200,
                 max_chunk_size: int = 1500):
        """
        초기화
        
        Args:
            chunk_size: 고정 길이 청킹 시 기본 크기
            overlap: 청크 간 겹침 크기
            min_chunk_size: 의미 청킹 시 최소 크기
            max_chunk_size: 의미 청킹 시 최대 크기
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        print(f"📝 텍스트 처리기 초기화")
        print(f"   📏 청크 크기: {chunk_size} (겹침: {overlap})")
        print(f"   🧠 의미 청킹 범위: {min_chunk_size}~{max_chunk_size}자")
    
    def clean_text(self, text: str) -> str:
        """
        텍스트 전처리
        
        Args:
            text: 원본 텍스트
            
        Returns:
            정제된 텍스트
        """
        if not text:
            return ""
        
        # 1. 여러 개의 공백을 하나로
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 2. 여러 개의 줄바꿈을 최대 2개로
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 3. 각 줄의 앞뒤 공백 제거
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)
        
        # 4. 전체 앞뒤 공백 제거
        return text.strip()
    
    def _detect_paragraph_boundaries(self, text: str) -> List[int]:
        """
        문단 경계 감지 (의미 청킹용)
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            문단 경계 위치 리스트
        """
        boundaries = [0]
        
        # 문단 구분 패턴들
        paragraph_patterns = [
            r'\n\n+',          # 빈 줄
            r'\n\d+\.',        # 번호 매기기 (1., 2., ...)
            r'\n[가-힣]\.',    # 한글 리스트 (가., 나., ...)
            r'\n-\s',          # 하이픈 리스트
            r'\n•\s',          # 불릿 포인트
            r'\n\[.*?\]',      # 대괄호 섹션
            r'\n제\d+장',      # 장 구분
            r'\n제\d+절',      # 절 구분
            r'\n\d+\)\s'       # 번호 + 괄호 (1) 2) ...)
        ]
        
        # 모든 패턴 적용
        for pattern in paragraph_patterns:
            for match in re.finditer(pattern, text):
                pos = match.start()
                if pos > 0 and pos not in boundaries:
                    boundaries.append(pos)
        
        # 문장 끝 패턴 (.!? 뒤에 줄바꿈)
        sentence_end_pattern = r'[.!?]\s*\n'
        for match in re.finditer(sentence_end_pattern, text):
            pos = match.end()
            if pos < len(text) - 1 and pos not in boundaries:
                boundaries.append(pos)
        
        # 끝 위치 추가
        boundaries.append(len(text))
        
        return sorted(list(set(boundaries)))
    
    def create_semantic_chunks(self, text: str, node_id: str) -> List[Dict[str, Any]]:
        """
        의미 기반 청킹 (문단 구조 인식)
        
        Args:
            text: 청킹할 텍스트
            node_id: 노드 ID
            
        Returns:
            청크 리스트
        """
        # 전처리
        text = self.clean_text(text)
        
        # 짧으면 그냥 반환
        if len(text) <= self.max_chunk_size:
            return [{
                'content': text,
                'node_id': node_id,
                'chunk_index': 0,
                'start_pos': 0,
                'end_pos': len(text),
                'method': 'semantic_single'
            }]
        
        # 문단 경계 감지
        boundaries = self._detect_paragraph_boundaries(text)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        start_pos = 0
        
        for i in range(len(boundaries) - 1):
            segment_start = boundaries[i]
            segment_end = boundaries[i + 1]
            segment = text[segment_start:segment_end].strip()
            
            if not segment:
                continue
            
            # 현재 청크에 세그먼트 추가 시도
            potential_chunk = (current_chunk + "\n" + segment).strip()
            
            if len(potential_chunk) <= self.max_chunk_size:
                # 최대 크기 이하면 계속 추가
                current_chunk = potential_chunk
            else:
                # 최대 크기 초과
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    # 현재 청크 저장
                    chunks.append({
                        'content': current_chunk,
                        'node_id': node_id,
                        'chunk_index': chunk_index,
                        'start_pos': start_pos,
                        'end_pos': start_pos + len(current_chunk),
                        'method': 'semantic'
                    })
                    chunk_index += 1
                    start_pos = segment_start
                
                # 새 청크 시작
                current_chunk = segment
        
        # 마지막 청크 처리
        if current_chunk:
            if len(current_chunk) >= self.min_chunk_size or not chunks:
                # 최소 크기 이상이거나 유일한 청크인 경우
                chunks.append({
                    'content': current_chunk,
                    'node_id': node_id,
                    'chunk_index': chunk_index,
                    'start_pos': start_pos,
                    'end_pos': len(text),
                    'method': 'semantic'
                })
            elif chunks:
                # 너무 작으면 이전 청크에 병합
                chunks[-1]['content'] += "\n" + current_chunk
                chunks[-1]['end_pos'] = len(text)
        
        return chunks
    
    def create_fixed_length_chunks(self, text: str, node_id: str) -> List[Dict[str, Any]]:
        """
        고정 길이 청킹 (Overlap 포함)
        
        Args:
            text: 청킹할 텍스트
            node_id: 노드 ID
            
        Returns:
            청크 리스트
        """
        # 전처리
        text = self.clean_text(text)
        
        # 짧으면 그냥 반환
        if len(text) <= self.chunk_size:
            return [{
                'content': text,
                'node_id': node_id,
                'chunk_index': 0,
                'start_pos': 0,
                'end_pos': len(text),
                'method': 'fixed_single'
            }]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # 문장 끝에서 자르기 시도
            if end < len(text):
                sentence_end = max(
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end)
                )
                
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    # 문장 끝 못 찾으면 공백에서 자르기
                    word_end = text.rfind(' ', start, end)
                    if word_end > start:
                        end = word_end
            
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                chunks.append({
                    'content': chunk_content,
                    'node_id': node_id,
                    'chunk_index': chunk_index,
                    'start_pos': start,
                    'end_pos': end,
                    'method': 'fixed'
                })
                chunk_index += 1
            
            # 끝이면 중단
            if end >= len(text):
                break
            
            # Overlap 적용
            start = max(start + 1, end - self.overlap)
        
        return chunks
    
    def chunk_node_description(self, 
                               description: str, 
                               node_id: str,
                               method: str = 'semantic') -> List[Dict[str, Any]]:
        """
        노드 설명을 청킹
        
        Args:
            description: 노드 설명 텍스트
            node_id: 노드 ID
            method: 'semantic' 또는 'fixed'
            
        Returns:
            청크 리스트
        """
        if not description or not description.strip():
            return []
        
        if method == 'semantic':
            return self.create_semantic_chunks(description, node_id)
        else:
            return self.create_fixed_length_chunks(description, node_id)
    
    def should_chunk(self, text: str, threshold: int = 1000) -> bool:
        """
        텍스트가 청킹이 필요한지 판단
        
        Args:
            text: 확인할 텍스트
            threshold: 청킹 임계값 (기본 1000자)
            
        Returns:
            청킹 필요 여부
        """
        return len(text) > threshold if text else False
