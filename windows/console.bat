@echo off
rem Консоль с активированным окружением проекта — для ручных команд из README
rem (python tools\detect.py ..., python tools\track_lanes_video.py ..., python tools\check_install.py).
cmd /k call "%~dp0_env.bat"
