@echo off
rem Общие настройки для всех скриптов в windows\: корень проекта, окружение .venv,
rem UTF-8 (конфиги и вывод на русском) и временные файлы внутри проекта — OpenCV на
rem Windows не открывает пути с кириллицей, а TEMP часто лежит в C:\Users\<Имя>\...
chcp 65001 >nul
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "TEMP=%ROOT%\tmp"
set "TMP=%ROOT%\tmp"
set "GRADIO_TEMP_DIR=%ROOT%\tmp\gradio"
if not exist "%ROOT%\tmp" mkdir "%ROOT%\tmp"
cd /d "%ROOT%"
if exist "%ROOT%\.venv\Scripts\activate.bat" call "%ROOT%\.venv\Scripts\activate.bat"
