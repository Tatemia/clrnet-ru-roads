@echo off
rem Установка на Windows: отдельное окружение .venv с Python 3.9 внутри папки проекта
rem (системный Python, например 3.13, не затрагивается) и точными версиями пакетов.
rem CUDA-расширения не собираются: CLRNet они не нужны, компилятор не требуется.
setlocal
call "%~dp0_env.bat"
echo === Установка UnLanedet: %ROOT% ===

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [!!] Не найден драйвер NVIDIA ^(nvidia-smi^). Установите драйвер видеокарты с поддержкой CUDA 12.
    goto :fail
)
py -3.9 -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo [!!] Не найден Python 3.9. Установите "Windows installer (64-bit)" со страницы
    echo      https://www.python.org/downloads/release/python-3913/ ^(отметьте "py launcher"^) и запустите setup.bat снова.
    goto :fail
)
py -3.9 -c "import os, sys; sys.exit(0 if os.getcwd().isascii() else 1)"
if errorlevel 1 (
    echo [!!] В пути к папке есть не латинские символы: %ROOT%
    echo      OpenCV не сможет открывать файлы. Перенесите папку, например в C:\UnLanedet
    goto :fail
)

if not exist ".venv\Scripts\python.exe" (
    echo --- Создаю окружение .venv ^(Python 3.9^)
    py -3.9 -m venv .venv || goto :fail
)
call ".venv\Scripts\activate.bat" || goto :fail

rem albumentations 0.4.6 собирается из исходников и при сборке требует setuptools<81
set "PIP_CONSTRAINT=%ROOT%\build-constraints.txt"
python -m pip install --upgrade pip "setuptools<81" wheel || goto :fail
echo --- PyTorch 2.3.1 + CUDA 12.1
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121 || goto :fail
echo --- Пакеты проекта и веб-интерфейса
python -m pip install -r requirements.txt -r requirements-app.txt -c requirements-lock.txt || goto :fail
echo --- TensorRT ^(около 4 ГБ, можно прервать: программа работает и без него^)
python -m pip install -r requirements-tensorrt.txt -c requirements-lock.txt
if errorlevel 1 echo [--] TensorRT не установлен: будет работать режим PyTorch, пункты TensorRT в интерфейсе недоступны.
echo --- numpy 1.23.1 последним шагом ^(код UnLanedet и imgaug требуют именно её^)
python -m pip install numpy==1.23.1 --no-deps || goto :fail

rem Подключаем папку проекта к окружению вместо "pip install -e ." (тот собирал бы CUDA-расширения)
python -c "import os, site; open(os.path.join(site.getsitepackages()[-1], 'unlanedet_project.pth'), 'w').write(os.getcwd())" || goto :fail

echo.
echo --- Проверка установки
python tools\check_install.py
echo.
echo Дальше: windows\build_tensorrt.bat ^(движки TensorRT под эту видеокарту^), затем windows\run_gradio.bat
endlocal
pause
exit /b 0

:fail
echo.
echo Установка прервана, см. сообщение выше.
endlocal
pause
exit /b 1
