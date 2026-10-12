@echo off
cd /d D:\InvoicesExtractor
start http://127.0.0.1:8000
call venv\Scripts\activate
uvicorn server:app
pause