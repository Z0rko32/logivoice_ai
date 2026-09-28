@echo off
chcp 65001 > nul
echo Запуск виртуального окружения и сервера LogiVoice AI...
cd backend
call .\venv\Scripts\activate.bat
python -m uvicorn app.main:app --reload
pause