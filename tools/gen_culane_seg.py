"""
Генерация сегментационных масок в стиле CULane (laneseg_label_w16) из файлов .lines.txt.

Ожидаемая входная структура (после конвертации разметки из CVAT скриптом
tools/cvat_to_culane.py):

    <data_root>/<scene>/00000.jpg
    <data_root>/<scene>/00000.lines.txt
    <data_root>/<scene>/00030.jpg
    <data_root>/<scene>/00030.lines.txt
    ...

Скрипт для каждого .lines.txt рисует маску (grayscale PNG, 0 = фон,
1..N = индекс полосы слева направо) и кладёт её в
    <data_root>/laneseg_label_w16/<scene>/00000.png
т.е. зеркалит структуру папок со сценами, как в оригинальном CULane.

Пример:
    python tools/gen_culane_seg.py --data-root data/ru_roads --max-lanes 4 --width 16
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def read_lines_txt(path):
    lanes = []
    with open(path, 'r') as f:
        for line in f:
            vals = [float(v) for v in line.split()]
            if len(vals) < 4:
                continue
            pts = [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
            lanes.append(pts)
    return lanes


def order_lanes_left_to_right(lanes, max_lanes):
    """Сортирует полосы по x-координате их самой нижней (ближней к камере) точки."""
    def bottom_x(lane):
        return max(lane, key=lambda p: p[1])[0]

    lanes_sorted = sorted(lanes, key=bottom_x)
    if len(lanes_sorted) > max_lanes:
        # оставляем max_lanes полос, ближайших к центру кадра
        # (крайние лишние полосы на разметке обычно шум/соседние ряды)
        center_idx = len(lanes_sorted) / 2
        lanes_sorted = sorted(
            range(len(lanes_sorted)),
            key=lambda i: abs(i - center_idx)
        )[:max_lanes]
        lanes_sorted = sorted(lanes_sorted)
        lanes_sorted = [lanes[i] for i in lanes_sorted]
    return lanes_sorted


def draw_mask(lanes, height, width, line_width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for idx, lane in enumerate(lanes, start=1):
        pts = np.array(lane, dtype=np.int32)
        for j in range(len(pts) - 1):
            cv2.line(mask, tuple(pts[j]), tuple(pts[j + 1]), color=idx, thickness=line_width)
    return mask


def find_lines_files(data_root, mask_dirname):
    for root, dirs, files in os.walk(data_root):
        if mask_dirname in Path(root).parts:
            continue
        for f in files:
            if f.endswith('.lines.txt'):
                yield os.path.join(root, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True, help='Корень датасета (там же появится laneseg_label_w16/)')
    parser.add_argument('--mask-dirname', default='laneseg_label_w16', help='Имя папки для масок')
    parser.add_argument('--max-lanes', type=int, default=4, help='Макс. число полос (как в CULane, по умолчанию 4)')
    parser.add_argument('--width', type=int, default=16, help='Толщина линии маски в пикселях')
    parser.add_argument('--img-ext', default='.jpg', help='Расширение изображений')
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    lines_files = list(find_lines_files(data_root, args.mask_dirname))
    if not lines_files:
        print(f'Не найдено ни одного .lines.txt под {data_root}')
        return

    n_ok, n_empty = 0, 0
    for lines_path in tqdm(lines_files, desc='generating masks'):
        img_path = lines_path[:-len('.lines.txt')] + args.img_ext
        if not os.path.isfile(img_path):
            print(f'[skip] нет изображения для {lines_path} (искал {img_path})')
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f'[skip] не смог открыть {img_path}')
            continue
        height, width = img.shape[:2]

        lanes = read_lines_txt(lines_path)
        lanes = order_lanes_left_to_right(lanes, args.max_lanes)
        if not lanes:
            n_empty += 1
        mask = draw_mask(lanes, height, width, args.width)

        rel = os.path.relpath(lines_path[:-len('.lines.txt')] + '.png', data_root)
        out_path = os.path.join(data_root, args.mask_dirname, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, mask)
        n_ok += 1

    print(f'Готово: {n_ok} масок сохранено в {os.path.join(data_root, args.mask_dirname)} '
          f'({n_empty} кадров без полос — маска полностью фон, это нормально для части кадров).')


if __name__ == '__main__':
    main()
