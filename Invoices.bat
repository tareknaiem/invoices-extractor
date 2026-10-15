@echo off
cd /d D:\InvoicesExtractor
call venv\Scripts\activate
start http://127.0.0.1:8000
uvicorn server:app --reload
pause