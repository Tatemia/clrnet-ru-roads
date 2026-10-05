"""
Замер скорости инференса модели (PyTorch или TensorRT через --trt-engine, batch=1 — сценарий реального времени)
с разбивкой по этапам: предобработка кадра -> нейросеть на GPU -> постобработка
(NMS + декодирование линий). Опционально — сквозной прогон по видео (декодирование
кадров + весь пайплайн + Kalman-трекер, без отрисовки и записи результата).

Пример:
    python tools/benchmark_speed.py config/clrnet/resnet34_culane_finetune.py \
        output/model_best.pth --data-root data/ru_roads --video data/test_1min.mp4
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch

from unlanedet.checkpoint import Checkpointer
from unlanedet.config import LazyConfig, instantiate
from unlanedet.engine import default_setup
from unlanedet.engine.defaults import create_ddp_model
from unlanedet.data.transform import Preprocess
from unlanedet.model.module.core.lane import Lane
from unlanedet.tracking import LaneTracker
from unlanedet.utils.frame_resize import fit_frame_to_config
from unlanedet.utils.malloc_tuning import tune_malloc_for_realtime


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def stats(ms):
    ms = np.asarray(ms)
    return f'среднее {ms.mean():6.2f} мс | медиана {np.median(ms):6.2f} | p95 {np.percentile(ms, 95):6.2f}'


def preprocess(frame, cfg, transform):
    img = fit_frame_to_config(frame, cfg)
    img = img[cfg.param_config.cut_height:, :, :].astype(np.float32)
    data = transform({'img': img, 'lanes': []})
    data['img'] = data['img'].unsqueeze(0).cuda(non_blocking=True)
    return data


def postprocess(model, out, cfg):
    lanes = model.get_lanes(out)[0]
    if len(lanes) and isinstance(lanes[0], Lane):
        lanes = [l.to_array(cfg.param_config) for l in lanes]
    return lanes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('ckpt')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--n-frames', type=int, default=300)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--video', default=None)
    parser.add_argument('--video-frames', type=int, default=900)
    parser.add_argument('--trt-engine', default=None,
                        help='TensorRT-движок (tools/export_tensorrt.py) вместо PyTorch; ckpt тогда не используется')
    parser.add_argument('--no-malloc-tuning', action='store_true',
                        help='Не настраивать аллокатор glibc (для сравнения "до/после")')
    args = parser.parse_args()
    malloc_tuned = False if args.no_malloc_tuning else tune_malloc_for_realtime()

    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, [])
    default_setup(cfg, argparse.Namespace(config=args.config))
    if args.trt_engine:
        from unlanedet.utils.trt_model import TRTLaneModel
        model = TRTLaneModel(args.trt_engine, cfg)
        backend = f'TensorRT ({os.path.basename(args.trt_engine)})'
    else:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        model.eval()
        Checkpointer(model).load(args.ckpt)
        backend = 'PyTorch FP32'
    transform = Preprocess(instantiate(cfg.dataloader.test.dataset.processes))

    with open(os.path.join(args.data_root, 'list', 'test.txt')) as f:
        rels = [l.strip().lstrip('/') for l in f if l.strip()]
    frames = [cv2.imread(os.path.join(args.data_root, r)) for r in rels[:args.n_frames]]
    print(f'Кадров в замере: {len(frames)} ({frames[0].shape[1]}x{frames[0].shape[0]}), '
          f'вход сети {cfg.param_config.img_w}x{cfg.param_config.img_h}, '
          f'GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, '
          f'cudnn.benchmark={torch.backends.cudnn.benchmark}, настройка malloc: {malloc_tuned}')

    t_pre, t_net, t_post, t_total = [], [], [], []
    with torch.no_grad():
        for i in range(args.warmup + len(frames)):
            frame = frames[i % len(frames)]
            t0 = sync()
            data = preprocess(frame, cfg, transform)
            t1 = sync()
            out = model(data)
            t2 = sync()
            postprocess(model, out, cfg)
            t3 = sync()
            if i >= args.warmup:
                t_pre.append((t1 - t0) * 1e3)
                t_net.append((t2 - t1) * 1e3)
                t_post.append((t3 - t2) * 1e3)
                t_total.append((t3 - t0) * 1e3)

    print(f'\n=== Покадровый замер (batch=1, {backend}) ===')
    print(f'Предобработка     : {stats(t_pre)}')
    print(f'Нейросеть (GPU)   : {stats(t_net)}  -> {1e3 / np.mean(t_net):6.1f} FPS только сеть')
    print(f'Постобработка     : {stats(t_post)}')
    print(f'Итого на кадр     : {stats(t_total)}  -> {1e3 / np.mean(t_total):6.1f} FPS')

    if args.video:
        tracker = LaneTracker(sample_ys=cfg.param_config.sample_y)
        cap = cv2.VideoCapture(args.video)
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        n = 0
        with torch.no_grad():
            t0 = sync()
            while n < args.video_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                data = preprocess(frame, cfg, transform)
                lanes = postprocess(model, model(data), cfg)
                tracker.update(lanes)
                n += 1
            t1 = sync()
        cap.release()
        fps = n / (t1 - t0)
        print(f'\n=== Сквозной прогон по видео {args.video} ===')
        print(f'{n} кадров за {t1 - t0:.2f} с -> {fps:.1f} FPS '
              f'(декодирование + предобработка + сеть + постобработка + трекер; '
              f'исходное видео {src_fps:.0f} FPS -> запас x{fps / src_fps:.1f})')


if __name__ == '__main__':
    main()
