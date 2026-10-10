@echo off
rem Дообучение по config\clrnet\resnet34_culane_finetune.py (data_root, init_checkpoint, epochs — в нём).
rem Дополнительные параметры передаются как есть, например: windows\train.bat train.max_iter=100
call "%~dp0_env.bat"
echo Результаты обучения пишутся в output\ — model_best.pth там будет ПЕРЕЗАПИСАН.
echo Если нужна текущая модель, скопируйте output\model_best.pth заранее. Закройте окно, чтобы отменить.
pause
python tools\train_net.py --config-file config\clrnet\resnet34_culane_finetune.py --num-gpus 1 %*
pause
