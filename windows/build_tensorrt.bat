@echo off
rem Движки TensorRT привязаны к видеокарте и версии TensorRT, поэтому собираются на каждом
rem компьютере заново (~20 с на движок). Имя движка берётся из имени чекпоинта:
rem output\model_best.pth -> model_best_fp16.engine (пункты в tools\gradio_app.py),
rem checkpoints\clrnet_r34_ru_roads.pth (веса из Releases) -> clrnet_r34_ru_roads_fp16.engine.
call "%~dp0_env.bat"
set "CKPT=output\model_best.pth"
if not exist "%CKPT%" set "CKPT=checkpoints\clrnet_r34_ru_roads.pth"
if not exist "%CKPT%" (
    echo [!!] Нет весов: ни output\model_best.pth, ни checkpoints\clrnet_r34_ru_roads.pth ^(скачайте из Releases^)
    pause
    exit /b 1
)
echo Собираю движки из %CKPT%
python tools\export_tensorrt.py config\clrnet\resnet34_culane_finetune.py "%CKPT%" --out-dir checkpoints\tensorrt --precisions fp32 fp16
pause
