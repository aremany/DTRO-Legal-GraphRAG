@echo off
chcp 65001 > nul
title DTRO 사규 챗봇 (ChromaDB) 실행

echo ========================================================
echo  ⚖️  DTRO 사규 질의응답 챗봇 (ChromaDB 버전)
echo ========================================================
echo.

:: Python 확인
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [❌ ERROR] Python이 설치되어 있지 않습니다.
    echo https://www.python.org/downloads/ 에서 Python 3.10 이상을 설치해주세요.
    pause
    exit /b
)

:: 필수 패키지 확인
pip show chromadb > nul 2>&1
if %errorlevel% neq 0 (
    echo [⬇️  INFO] 필수 라이브러리를 설치합니다...
    pip install -r requirements.txt
)

:: Ollama 확인
echo [🤖 INFO] Ollama 연결 확인 중...
curl -s http://localhost:11434/api/tags > nul
if %errorlevel% neq 0 (
    echo [⚠️  WARNING] Ollama가 실행되지 않았습니다.
    echo 답변 작성을 위해 Ollama 실행이 필요합니다.
    echo.
)

:: ChromaDB 확인
if not exist "chroma_db_fulltext" (
    echo [⚠️  WARNING] DB 폴더가 없습니다.
    echo 최초 실행이므로 임베딩을 시작합니다...
    python embedding_fulltext_chroma.py
)

echo [✅ INFO] 챗봇 서버를 시작합니다...
echo [🌐 INFO] 브라우저가 자동으로 열립니다.
echo.

:: 브라우저 자동 실행
start /b cmd /c "timeout /t 3 >nul && start http://localhost:5000"

:: 챗봇 실행
python chatbot_graphrag.py

pause
