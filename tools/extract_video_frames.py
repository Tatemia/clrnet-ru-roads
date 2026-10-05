"""
Извлекает кадры из видео для будущей разметки в CVAT: берёт кадры разреженно
(раз в --interval-sec секунд, а не подряд — соседние кадры почти идентичны и
только раздувают объём ручной разметки без пользы), и группирует их по
"сценам" — папкам фиксированной длины по времени (--scene-duration-sec).

Деление на сцены важно для честного train/val сплита: tools/gen_culane_lists.py
делит датасет по сценам (папкам), а не по отдельным кадрам, чтобы почти
одинаковые соседние кадры одной поездки не "утекали" одновременно и в train,
и в val.

Пример:
    python tools/extract_video_frames.py --video data/test.mp4 \
        --out data/ru_roads --interval-sec 5 --scene-duration-sec 180
"""
import argparse
import os

import cv2
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video', required=True)
    parser.add_argument('--out', required=True, help='Корень датасета, например data/ru_roads')
    parser.add_argument('--interval-sec', type=float, default=5.0, help='Интервал между сохранёнными кадрами, сек')
    parser.add_argument('--scene-duration-sec', type=float, default=180.0, help='Длина одной "сцены" (папки), сек')
    parser.add_argument('--scene-prefix', default='scene', help='Префикс имени папки сцены')
    parser.add_argument('--jpg-quality', type=int, default=95)
    parser.add_argument('--scene-offset', type=int, default=0,
                         help='С какого номера начинать нумерацию сцен (чтобы не конфликтовать с уже существующими папками при добавлении нового видео в тот же датасет)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    step_frames = max(1, round(args.interval_sec * fps))
    scene_frames = max(step_frames, round(args.scene_duration_sec * fps))

    print(f'Видео: {w}x{h}, {fps:.2f} fps, {total_frames} кадров, {duration_sec/60:.1f} мин')
    print(f'Берём кадр каждые {step_frames} кадров (~{args.interval_sec}с), '
          f'новая сцена каждые {scene_frames} кадров (~{args.scene_duration_sec}с)')

    os.makedirs(args.out, exist_ok=True)
    n_saved = 0
    n_scenes = 0
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc='extracting')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step_frames == 0:
            scene_id = args.scene_offset + frame_idx // scene_frames
            scene_dir = os.path.join(args.out, f'{args.scene_prefix}_{scene_id:03d}')
            os.makedirs(scene_dir, exist_ok=True)
            if not os.path.isdir(scene_dir) or n_scenes <= scene_id:
                n_scenes = scene_id + 1
            out_path = os.path.join(scene_dir, f'{frame_idx:06d}.jpg')
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
            n_saved += 1
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()

    print(f'\nГотово: {n_saved} кадров сохранено в {n_scenes} папках-сценах под {args.out}')
    print('Дальше: разметить кадры в CVAT (по одной задаче на сцену или все сразу — как удобнее),')
    print('затем tools/cvat_to_culane.py -> tools/gen_culane_seg.py -> tools/gen_culane_lists.py')


if __name__ == '__main__':
    main()
