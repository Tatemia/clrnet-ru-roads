"""
Генерация list/train_gt.txt и list/test.txt для CULane-совместимого датасета.

Ожидает, что рядом с каждым изображением уже лежит <name>.lines.txt (результат
tools/cvat_to_culane.py) и что маски уже сгенерированы tools/gen_culane_seg.py
в <data_root>/<mask-dirname>/... (по умолчанию laneseg_label_w16).

Деление на train/val идёт по СЦЕНАМ (папкам с кадрами), а не по отдельным кадрам —
это важно: кадры одной и той же поездки почти идентичны, и если раскидать их по
train/val случайно кадр-к-кадру, val будет "утечкой" почти тех же картинок и
результат оценки будет обманчиво хорошим. Поэтому вся папка (одна поездка/один
непрерывный отрезок) целиком уходит либо в train, либо в val.

Пример:
    python tools/gen_culane_lists.py --data-root data/ru_roads --val-ratio 0.15
"""
import argparse
import os
import random
from collections import defaultdict

from tqdm import tqdm


def read_lines_txt(path):
    lanes = []
    with open(path, 'r') as f:
        for line in f:
            vals = [float(v) for v in line.split()]
            if len(vals) < 4:
                continue
            lanes.append(vals)
    return lanes


def find_images(data_root, mask_dirname, list_dirname, img_ext):
    images = []
    for root, dirs, files in os.walk(data_root):
        rel_root_parts = os.path.relpath(root, data_root).split(os.sep)
        if mask_dirname in rel_root_parts or list_dirname in rel_root_parts:
            continue
        for f in files:
            if f.endswith(img_ext):
                img_path = os.path.join(root, f)
                lines_path = img_path[:-len(img_ext)] + '.lines.txt'
                if os.path.isfile(lines_path):
                    images.append(img_path)
                else:
                    print(f'[skip] нет разметки для {img_path} (искал {lines_path})')
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--mask-dirname', default='laneseg_label_w16')
    parser.add_argument('--list-dirname', default='list')
    parser.add_argument('--img-ext', default='.jpg')
    parser.add_argument('--max-lanes', type=int, default=4, help='Должно совпадать со значением в gen_culane_seg.py')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Доля СЦЕН (не кадров) на валидацию')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_root = os.path.abspath(args.data_root)

    images = find_images(data_root, args.mask_dirname, args.list_dirname, args.img_ext)
    if not images:
        print(f'Не найдено ни одного размеченного изображения под {data_root}')
        return

    scenes = defaultdict(list)
    for img_path in images:
        scene = os.path.dirname(os.path.relpath(img_path, data_root))
        scenes[scene].append(img_path)

    scene_names = sorted(scenes.keys())
    random.shuffle(scene_names)
    n_val_scenes = max(1, round(len(scene_names) * args.val_ratio)) if len(scene_names) > 1 else 0
    val_scenes = set(scene_names[:n_val_scenes])
    train_scenes = set(scene_names) - val_scenes

    print(f'Сцен всего: {len(scene_names)} | train: {len(train_scenes)} | val: {len(val_scenes)}')

    list_dir = os.path.join(data_root, args.list_dirname)
    os.makedirs(list_dir, exist_ok=True)

    train_lines = []
    val_lines = []

    for scene in tqdm(scene_names, desc='building lists'):
        for img_path in sorted(scenes[scene]):
            rel_img = '/' + os.path.relpath(img_path, data_root).replace(os.sep, '/')
            if scene in val_scenes:
                val_lines.append(rel_img)
                continue

            lines_path = img_path[:-len(args.img_ext)] + '.lines.txt'
            lanes = read_lines_txt(lines_path)
            n_lanes = min(len(lanes), args.max_lanes)
            exist = ['1' if i < n_lanes else '0' for i in range(args.max_lanes)]

            rel_mask = '/' + os.path.join(
                args.mask_dirname, os.path.relpath(img_path, data_root)
            ).replace(os.sep, '/')
            rel_mask = rel_mask[:-len(args.img_ext)] + '.png'

            train_lines.append(' '.join([rel_img, rel_mask] + exist))

    train_gt_path = os.path.join(list_dir, 'train_gt.txt')
    test_path = os.path.join(list_dir, 'test.txt')

    with open(train_gt_path, 'w') as f:
        f.write('\n'.join(train_lines) + ('\n' if train_lines else ''))
    with open(test_path, 'w') as f:
        f.write('\n'.join(val_lines) + ('\n' if val_lines else ''))

    print(f'{train_gt_path}: {len(train_lines)} кадров')
    print(f'{test_path}: {len(val_lines)} кадров')
    print('\nВ конфиге fine-tuning не забудьте сверить epoch_per_iter с реальным '
          f'числом train-кадров ({len(train_lines)}), см. комментарий в config/clrnet/resnet34_culane_finetune.py')


if __name__ == '__main__':
    main()
