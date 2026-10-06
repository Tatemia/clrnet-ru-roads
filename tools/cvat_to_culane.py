"""
Конвертирует размеченный/поправленный в CVAT annotations.xml (формат "CVAT for
images 1.1", экспортируется через Menu -> Export task dataset) в файлы .lines.txt
рядом с исходными кадрами — формат, который понимает загрузчик CULane в UnLanedet.

Каждая <image name="..."> в XML должна соответствовать файлу <img-dir>/<name>
(или файлу с таким именем где-то под --img-dir, если структура вложенная —
скрипт ищет по всему дереву; если имя встречается в нескольких местах, кадр
пропускается). Пустая разметка не затирает уже существующие полосы
(см. --allow-empty-overwrite). Для каждого <image> создаётся
<img-dir>/.../<basename без расширения>.lines.txt с одной строкой на полосу:
"x1 y1 x2 y2 x3 y3 ...".

ВНИМАНИЕ: загрузчик UnLanedet выбрасывает полосы короче 4 точек после удаления
дублей — рисуйте в CVAT не менее 4-5 точек на полосу, иначе она не попадёт в
обучение (скрипт предупредит, но не остановится).

Пример:
    python tools/cvat_to_culane.py --xml data/ru_roads/rain_day_01/annotations.xml \
        --img-dir data/ru_roads/rain_day_01 --label lane
"""
import argparse
import os
from xml.etree import ElementTree as ET

from unlanedet.utils.lane_labels import write_lines_file


def find_image_paths(img_dir, name):
    """Все кандидаты для <image name>: точный путь, иначе файлы с таким же именем по дереву."""
    direct = os.path.join(img_dir, name)
    if os.path.isfile(direct):
        return [direct]
    base = os.path.basename(name)
    return [os.path.join(root, base) for root, _, files in os.walk(img_dir) if base in files]


def parse_polyline_points(points_str):
    pts = []
    for pair in points_str.strip().split(';'):
        if not pair:
            continue
        x_str, y_str = pair.split(',')
        pts.append((float(x_str), float(y_str)))
    return pts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--xml', required=True, help='Путь к экспортированному из CVAT annotations.xml')
    parser.add_argument('--img-dir', required=True, help='Папка (дерево), где искать соответствующие изображения')
    parser.add_argument('--label', default='lane', help='Имя label в CVAT, которое считать полосой разметки')
    parser.add_argument('--min-points', type=int, default=4,
                         help='Порог для предупреждения — UnLanedet выбрасывает полосы короче 4 точек')
    parser.add_argument('--allow-empty-overwrite', action='store_true',
                        help='Разрешить затирать непустые .lines.txt пустыми (по умолчанию такие кадры пропускаются)')
    args = parser.parse_args()

    tree = ET.parse(args.xml)
    root = tree.getroot()

    n_images, n_lanes, n_missing, n_short, n_ambiguous, n_kept = 0, 0, 0, 0, 0, 0

    for image_el in root.findall('image'):
        name = image_el.get('name')
        candidates = find_image_paths(args.img_dir, name)
        if not candidates:
            print(f'[skip] не нашёл изображение для "{name}" под {args.img_dir}')
            n_missing += 1
            continue
        if len(candidates) > 1:
            print(f'[skip] "{name}" есть в {len(candidates)} местах под {args.img_dir} '
                  f'(разные сцены?) — укажите --img-dir папку конкретной сцены')
            n_ambiguous += 1
            continue
        img_path = candidates[0]

        lanes = []
        for poly in image_el.findall('polyline'):
            if poly.get('label') != args.label:
                continue
            pts = parse_polyline_points(poly.get('points', ''))
            if len(pts) < 2:
                continue
            pts = sorted(pts, key=lambda p: p[1])  # сверху вниз, по y
            if len(pts) < args.min_points:
                print(f'[warn] {name}: полоса из {len(pts)} точек (<{args.min_points}) '
                      f'будет отброшена загрузчиком UnLanedet, добавьте точек в CVAT')
                n_short += 1
            lanes.append(pts)

        lines_path = os.path.splitext(img_path)[0] + '.lines.txt'
        if not write_lines_file(lines_path, lanes, args.allow_empty_overwrite):
            n_kept += 1
            continue

        n_images += 1
        n_lanes += len(lanes)

    print(f'Готово: {n_images} кадров, {n_lanes} полос записано в .lines.txt рядом с изображениями.')
    if n_missing:
        print(f'Не найдено изображений: {n_missing} (проверьте --img-dir).')
    if n_ambiguous:
        print(f'Пропущено неоднозначных имён: {n_ambiguous}.')
    if n_kept:
        print(f'[!] {n_kept} кадров без полос в XML, но с полосами на диске — файлы НЕ перезаписаны '
              f'(затереть: --allow-empty-overwrite).')
    if n_short:
        print(f'Полос короче --min-points: {n_short} (будут проигнорированы при обучении).')


if __name__ == '__main__':
    main()
