"""
Самопроверка установки (Windows и Linux): версии пакетов, видеокарта, путь к папке,
загрузка модели и распознавание на одном кадре, TensorRT, Gradio, ffmpeg.

Запуск из корня проекта:
    python tools/check_install.py
"""
import argparse
import glob
import os
import platform
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = 'config/clrnet/resnet34_culane_finetune.py'
CHECKPOINT_CANDIDATES = ['output/model_best.pth', 'checkpoints/clrnet_r34_ru_roads.pth']
EXPECTED = {'python': '3.9', 'numpy': '1.23.1', 'torch': '2.3.1'}

results = []


def report(ok, name, detail=''):
    mark = {True: 'OK', False: '!!', None: '--'}[ok]
    results.append(ok)
    print(f'[{mark}] {name}' + (f': {detail}' if detail else ''))


def check_environment():
    py = '.'.join(map(str, sys.version_info[:2]))
    report(py == EXPECTED['python'], 'Python', f'{py} (нужен {EXPECTED["python"]})')
    if platform.system() == 'Windows':
        report(REPO_ROOT.isascii(), 'Путь к проекту только латиницей', REPO_ROOT +
               ('' if REPO_ROOT.isascii() else ' — OpenCV не откроет файлы, перенесите папку, например в C:\\UnLanedet'))
        report(bool(sys.flags.utf8_mode), 'Режим UTF-8 (PYTHONUTF8=1)',
               '' if sys.flags.utf8_mode else 'запускайте через windows\\*.bat или console.bat')

    import numpy
    report(numpy.__version__ == EXPECTED['numpy'], 'numpy', numpy.__version__)
    import torch
    report(torch.__version__.split('+')[0] == EXPECTED['torch'], 'PyTorch', torch.__version__)
    cuda = torch.cuda.is_available()
    report(cuda, 'Видеокарта CUDA', torch.cuda.get_device_name(0) if cuda else 'не найдена (нужен драйвер NVIDIA)')
    return cuda


def check_model(device):
    import cv2
    import numpy as np
    import torch
    from unlanedet.checkpoint import Checkpointer
    from unlanedet.config import LazyConfig, instantiate
    from unlanedet.data.transform import Preprocess
    from unlanedet.utils.frame_resize import fit_frame_to_config

    ckpt = next((p for p in CHECKPOINT_CANDIDATES if os.path.isfile(os.path.join(REPO_ROOT, p))), None)
    if ckpt is None:
        report(False, 'Веса модели', 'нет ни одного из: ' + ', '.join(CHECKPOINT_CANDIDATES))
        return None
    cfg = LazyConfig.load(os.path.join(REPO_ROOT, CONFIG))
    model = instantiate(cfg.model).to(device).eval()
    incompatible = Checkpointer(model).load(os.path.join(REPO_ROOT, ckpt))
    missing = getattr(incompatible, 'missing_keys', None) or []
    report(not missing, 'Веса модели', ckpt + (f', не загружено слоёв: {len(missing)}' if missing else ''))

    frames = sorted(glob.glob(os.path.join(REPO_ROOT, 'data', 'ru_roads', 'scene_*', '*.jpg')))
    img = cv2.imread(frames[0]) if frames else None
    if img is None:
        img = np.full((cfg.param_config.ori_img_h, cfg.param_config.ori_img_w, 3), 90, np.uint8)
    img = fit_frame_to_config(img, cfg)
    transform = Preprocess(instantiate(cfg.dataloader.test.dataset.processes))
    data = transform({'img': img[cfg.param_config.cut_height:].astype(np.float32), 'lanes': []})
    data['img'] = data['img'].unsqueeze(0).to(device)
    with torch.no_grad():
        lanes = model.get_lanes(model(data))[0]
    report(True, 'Распознавание на кадре', f'{"кадр из data/ru_roads" if frames else "пустой кадр"}, полос: {len(lanes)}')
    return cfg


def check_tensorrt(cfg):
    try:
        import tensorrt
    except ImportError:
        report(None, 'TensorRT', 'не установлен (нужен только для режима TensorRT)')
        return
    report(True, 'TensorRT', tensorrt.__version__)
    engines = sorted(glob.glob(os.path.join(REPO_ROOT, 'checkpoints', 'tensorrt', '*.engine')))
    if not engines:
        report(None, 'Движки TensorRT', 'нет — соберите: windows\\build_tensorrt.bat (или tools/export_tensorrt.py)')
        return
    from unlanedet.utils.trt_model import TRTLaneModel
    for path in engines:
        try:
            TRTLaneModel(path, cfg)
            report(True, 'Движок ' + os.path.basename(path))
        except Exception as e:  # движок от другой видеокарты/версии TensorRT
            report(False, 'Движок ' + os.path.basename(path),
                   f'не загружается ({type(e).__name__}) — пересоберите на этом компьютере')


def check_app():
    try:
        import gradio
        report(True, 'Gradio', gradio.__version__)
    except Exception as e:
        report(False, 'Gradio', f'{type(e).__name__}: {e}')
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        report(os.path.isfile(exe), 'ffmpeg для видео', exe)
    except Exception as e:
        report(False, 'ffmpeg для видео', f'{type(e).__name__}: {e}')


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    os.chdir(REPO_ROOT)
    print(f'Проект: {REPO_ROOT}\nСистема: {platform.system()} {platform.release()}, {platform.machine()}\n')
    cuda = check_environment()
    cfg = check_model('cuda' if cuda else 'cpu')
    if cfg is not None and cuda:
        check_tensorrt(cfg)
    check_app()
    failed = results.count(False)
    print('\nИтог: ' + ('всё в порядке' if not failed else f'проблем: {failed} (строки с [!!])'))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
