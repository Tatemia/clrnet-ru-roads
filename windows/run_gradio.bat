@echo off
rem Веб-интерфейс: откройте в браузере http://127.0.0.1:7860 (остановка — Ctrl+C или закрыть окно).
call "%~dp0_env.bat"
python tools\gradio_app.py
pause
