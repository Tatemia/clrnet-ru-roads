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


def scene_dir_path(args, scene_id):
    return os.path.join(args.out, f'{args.scene_prefix}_{scene_id:03d}')


def is_occupied(scene_dir):
    return os.path.isdir(scene_dir) and bool(os.listdir(scene_dir))


def next_free_scene_id(args):
    prefix = f'{args.scene_prefix}_'
    ids = [int(d[len(prefix):]) for d in os.listdir(args.out)
           if d.startswith(prefix) and d[len(prefix):].isdigit()]
    return max(ids, default=-1) + 1


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
    n_planned = max(1, (total_frames - 1) // scene_frames + 1)
    occupied = [scene_dir_path(args, i) for i in range(args.scene_offset, args.scene_offset + n_planned)
                if is_occupied(scene_dir_path(args, i))]
    if occupied:
        # Перезапись кадров оставила бы старые .lines.txt/маски рядом с новыми картинками.
        raise SystemExit(f'Папки сцен уже заняты: {", ".join(map(os.path.basename, occupied))}. '
                         f'Укажите свободный --scene-offset (например {next_free_scene_id(args)}).')

    n_saved = 0
    scenes = set()
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc='extracting')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step_frames == 0:
            scene_id = args.scene_offset + frame_idx // scene_frames
            scene_dir = scene_dir_path(args, scene_id)
            if scene_id not in scenes and is_occupied(scene_dir):
                raise SystemExit(f'Папка {scene_dir} уже занята (видео длиннее, чем сообщил контейнер) — '
                                 f'укажите свободный --scene-offset (например {next_free_scene_id(args)}).')
            os.makedirs(scene_dir, exist_ok=True)
            scenes.add(scene_id)
            out_path = os.path.join(scene_dir, f'{frame_idx:06d}.jpg')
            if not cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality]):
                raise SystemExit(f'Не удалось записать {out_path}')
            n_saved += 1
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()

    print(f'\nГотово: {n_saved} кадров сохранено в {len(scenes)} папках-сценах под {args.out}')
    print('Дальше: разметить кадры в CVAT (по одной задаче на сцену или все сразу — как удобнее),')
    print('затем tools/cvat_to_culane.py -> tools/gen_culane_seg.py -> tools/gen_culane_lists.py')


if __name__ == '__main__':
    main()
