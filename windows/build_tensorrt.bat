@echo off
rem Движки TensorRT привязаны к видеокарте и версии TensorRT, поэтому собираются на каждом
rem компьютере заново (~20 с на движок). Имена совпадают с пунктами в tools\gradio_app.py.
call "%~dp0_env.bat"
python tools\export_tensorrt.py config\clrnet\resnet34_culane_finetune.py output\model_best.pth --out-dir checkpoints\tensorrt --precisions fp32 fp16
pause
