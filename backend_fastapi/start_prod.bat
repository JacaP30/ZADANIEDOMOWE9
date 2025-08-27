@echo off
REM Uruchomienie produkcyjne FastAPI, serwuje build React z ../build
cd /d %~dp0

if "%PORT%"=="" set PORT=8000

REM Aktywacja środowiska conda zadaniedomowe9
call conda activate zadaniedomowe9

echo Starting server on http://127.0.0.1:%PORT%
uvicorn main:app --host 127.0.0.1 --port %PORT%

