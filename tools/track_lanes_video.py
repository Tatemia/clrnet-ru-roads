"""
Прогоняет модель UnLanedet по видео (или последовательности кадров) и сглаживает
предсказания во времени трекером unlanedet.tracking.LaneTracker (см. модуль —
Kalman-фильтры по x на фиксированных y-строках + венгерское сопоставление
треков между кадрами). Особенно полезно там, где кадр "слепой" (полоса скрыта
снегом/бликом на 1-2 секунды) — трекер продолжает экстраполировать полосу по
истории движения, а не мигает пропаданием/дрожанием как чистое покадровое CLRNet.

Сохраняет видео с наложением: слева — сырые покадровые предсказания модели,
справа — сглаженные трекером (для визуального сравнения / иллюстраций).

Пример (видеофайл):
    python tools/track_lanes_video.py config/clrnet/resnet34_culane_finetune.py \
        checkpoints/clrnet_resnet34_culane.pth --video input.mp4 --out compare.mp4

Пример (папка с кадрами по порядку):
    python tools/track_lanes_video.py config/clrnet/resnet34_culane_finetune.py \
        checkpoints/clrnet_resnet34_culane.pth --img "data/ru_roads/scene1/*.jpg" --out compare.mp4
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from unlanedet.checkpoint import Checkpointer
from unlanedet.config import LazyConfig, instantiate
from unlanedet.engine import default_setup
from unlanedet.engine.defaults import create_ddp_model
from unlanedet.data.transform import Preprocess
from unlanedet.model.module.core.lane import Lane
from unlanedet.tracking import LaneTracker
from unlanedet.utils.frame_resize import fit_frame_to_config
from unlanedet.utils.malloc_tuning import tune_malloc_for_realtime


def get_frame_source(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise FileNotFoundError(args.video)

        def gen():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        return gen(), fps
    else:
        paths = sorted(glob.glob(args.img, recursive=True))
        paths = [p for p in paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not paths:
            raise FileNotFoundError(f'Не найдено кадров по {args.img}')

        def gen():
            for p in paths:
                yield cv2.imread(p)
        return gen(), args.fps


def preprocess(frame, cfg, processes):
    img = frame[cfg.param_config.cut_height:, :, :].astype(np.float32)
    data = {'img': img, 'lanes': []}
    data = processes(data)
    data['img'] = data['img'].unsqueeze(0)
    return data


def predict_lanes(model, data, cfg):
    with torch.no_grad():
        out = model(data)
    lanes = model.get_lanes(out)[0]
    if len(lanes) and isinstance(lanes[0], Lane):
        lanes = [lane.to_array(cfg.param_config) for lane in lanes]
    else:
        lanes = [np.array(lane, dtype=np.float32) for lane in lanes]
    return lanes


def draw_lanes(frame, lanes, color, radius=4, thickness=2):
    out = frame.copy()
    for lane in lanes:
        pts = [(int(x), int(y)) for x, y in lane if x > 0 and y > 0]
        for p1, p2 in zip(pts[:-1], pts[1:]):
            cv2.line(out, p1, p2, color, thickness)
        for p in pts:
            cv2.circle(out, p, radius, color, -1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('ckpt')
    parser.add_argument('--video', help='Путь к видеофайлу')
    parser.add_argument('--img', help='Папка/паттерн с кадрами по порядку, например scene1/*.jpg')
    parser.add_argument('--out', required=True, help='Куда сохранить видео сравнения (.mp4)')
    parser.add_argument('--fps', type=float, default=10.0, help='FPS для выходного видео (режим --img)')
    parser.add_argument('--max-age', type=int, default=6, help='Сколько кадров трекер "ведёт" пропавшую полосу')
    parser.add_argument('--match-thresh', type=float, default=45.0, help='Порог сопоставления трека с детекцией, px')
    parser.add_argument('--lines-out', default=None, help='(опц.) папка — сохранить сглаженные .lines.txt по кадрам')
    parser.add_argument('--max-frames', type=int, default=None, help='(опц.) обработать только первые N кадров')
    args = parser.parse_args()

    if not args.video and not args.img:
        parser.error('нужно указать --video или --img')
    tune_malloc_for_realtime()

    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, [])
    default_setup(cfg, argparse.Namespace(config=args.config))

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    model.eval()
    Checkpointer(model).load(args.ckpt)

    transform = Preprocess(instantiate(cfg.dataloader.test.dataset.processes))
    tracker = LaneTracker(sample_ys=cfg.param_config.sample_y,
                           max_age=args.max_age, match_thresh=args.match_thresh)

    frames, fps = get_frame_source(args)

    writer = None
    if args.lines_out:
        os.makedirs(args.lines_out, exist_ok=True)

    for i, frame in enumerate(tqdm(frames, desc='tracking', total=args.max_frames)):
        if args.max_frames is not None and i >= args.max_frames:
            break
        if frame is None:
            continue
        frame = fit_frame_to_config(frame, cfg, source_name=args.video or args.img)
        data = preprocess(frame, cfg, transform)
        raw_lanes = predict_lanes(model, data, cfg)
        smoothed_lanes = tracker.update(raw_lanes)

        left = draw_lanes(frame, raw_lanes, (0, 0, 255))
        right = draw_lanes(frame, smoothed_lanes, (0, 255, 0))
        cv2.putText(left, 'raw (per-frame)', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(right, 'tracked (smoothed)', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        combined = np.concatenate([left, right], axis=1)

        if writer is None:
            h, w = combined.shape[:2]
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        writer.write(combined)

        if args.lines_out:
            lines_path = os.path.join(args.lines_out, f'{i:05d}.lines.txt')
            with open(lines_path, 'w') as f:
                for lane in smoothed_lanes:
                    coords = ' '.join(f'{x:.2f} {y:.2f}' for x, y in lane if x > 0)
                    f.write(coords + '\n')

    if writer is not None:
        writer.release()
        print(f'Сохранено: {args.out}')
    else:
        print('Не обработано ни одного кадра.')


if __name__ == '__main__':
    main()
