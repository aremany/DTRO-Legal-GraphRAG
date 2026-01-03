@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

title Legal-GraphRAG Launcher

echo ========================================================
echo  ⚖️  Legal-GraphRAG : 사규 지능형 챗봇 통합 실행기
echo ========================================================
echo.

:: 1. Python 설치 확인
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [❌ ERROR] Python이 설치되어 있지 않습니다.
    echo https://www.python.org/downloads/ 에서 Python 3.10 이상을 설치해주세요.
    echo (설치 시 'Add Python to PATH' 옵션을 체크해야 합니다)
    pause
    exit /b
)

:: 2. 가상환경(venv) 점검 및 생성
if not exist "venv" (
    echo [📦 INFO] 가상환경을 처음 생성합니다... (약 1분 소요)
    python -m venv venv
    if !errorlevel! neq 0 (
        echo [❌ ERROR] 가상환경 생성 실패.
        pause
        exit /b
    )
    
    echo [⬇️  INFO] 필수 라이브러리를 설치합니다... (시간이 조금 걸립니다)
    call venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [❌ ERROR] 라이브러리 설치 실패.
        pause
        exit /b
    )
    echo [✅ INFO] 설치 완료!
) else (
    echo [✅ INFO] 가상환경을 로드합니다.
    call venv\Scripts\activate
)

:: 3. Ollama 확인 및 모델 풀링
echo.
echo [🤖 INFO] Ollama 연결 확인 및 모델 준비...
curl -s http://localhost:11434/api/tags > nul
if %errorlevel% neq 0 (
    echo [⚠️  WARNING] Ollama가 실행 중이지 않은 것 같습니다.
    echo 챗봇 답변을 받으려면 Ollama(https://ollama.com)를 별도로 설치/실행해야 합니다.
    echo.
) else (
    echo [⬇️  INFO] AI 모델(Gemma 3) 다운로드/확인 중...
    :: 오류 발생해도(네트워크 등) 일단 진행하도록 || echo... 처리
    ollama pull hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M
)

:: 4. 데이터 임베딩 (최초 1회 실행)
if not exist "chroma_db_fulltext" (
    echo.
    echo [⚙️  INFO] 최초 실행입니다! 
    echo 제공된 사규 데이터를 분석하여 검색 엔진을 구축합니다.
    echo 컴퓨터 성능에 따라 1~5분 정도 소요됩니다. 잠시만 기다려주세요...
    python embedding_fulltext_chroma.py
    if !errorlevel! neq 0 (
        echo [❌ ERROR] 임베딩 과정에서 오류가 발생했습니다.
        pause
        exit /b
    )
    echo [✅ INFO] 데이터 준비 완료!
)

:: 5. 챗봇 서버 실행
echo.
echo ========================================================
echo  🚀 챗봇 서버를 시작합니다!
echo  🌐 브라우저가 자동으로 열립니다: http://localhost:5000
echo  (종료하려면 이 창을 닫으세요)
echo ========================================================
echo.

:: 브라우저 자동 실행 (5초 딜레이 후)
start /b cmd /c "timeout /t 5 >nul && start http://localhost:5000"

:: 챗봇 실행
python chatbot_graphrag.py

pause
