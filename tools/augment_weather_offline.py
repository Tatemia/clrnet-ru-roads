"""
Offline-расширение CULane-совместимого датасета синтетической погодой: для
каждого размеченного кадра создаёт N копий с туманом/дождём/бликом и добавляет
их в list/train_gt.txt — только TRAIN (val/test не трогаем, чтобы оценка F1
оставалась честной на реальных, не сгенерированных кадрах).

Копии .lines.txt и маски НЕ пересоздаются — переиспользуются те же самые (эффекты
не меняют геометрию полос), только картинка получает новое имя файла
"<name>__aug_<effect>.jpg" рядом с оригиналом, и .lines.txt/маска симлинкуются
на оригинальные файлы под тем же новым именем.

Запускать ПОСЛЕ tools/gen_culane_lists.py (нужен готовый list/train_gt.txt).

Пример:
    python tools/augment_weather_offline.py --data-root data/ru_roads --copies 2
"""
import argparse
import os

import cv2
from tqdm import tqdm

from unlanedet.data.transform.weather import add_fog, add_rain, add_glare
from unlanedet.utils.lane_labels import AUG_MARKER

EFFECTS = {
    'fog': lambda img, rng: add_fog(img, beta=rng.uniform(1.2, 3.0), rng=rng),
    'rain': lambda img, rng: add_rain(img, density=rng.uniform(0.008, 0.03), rng=rng),
    'glare': lambda img, rng: add_glare(img, radius_ratio=rng.uniform(0.2, 0.45), rng=rng),
}


def relink(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--copies', type=int, default=2, help='Сколько погодных копий на каждый train-кадр')
    parser.add_argument('--img-ext', default='.jpg')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data_root = os.path.abspath(args.data_root)
    train_gt_path = os.path.join(data_root, 'list', 'train_gt.txt')
    if not os.path.isfile(train_gt_path):
        raise FileNotFoundError(f'{train_gt_path} не найден — сначала запустите tools/gen_culane_lists.py')

    with open(train_gt_path) as f:
        all_lines = [line.strip() for line in f if line.strip()]
    # Повторный запуск: копии прошлого прогона заменяются, а не аугментируются заново.
    orig_lines = [line for line in all_lines if AUG_MARKER not in line.split()[0]]
    n_old_aug = len(all_lines) - len(orig_lines)
    if n_old_aug:
        print(f'В train_gt.txt уже было {n_old_aug} синтетических строк — они будут заменены новыми.')

    effect_names = list(EFFECTS.keys())
    new_lines = list(orig_lines)
    n_created = 0

    for line in tqdm(orig_lines, desc='augmenting'):
        parts = line.split()
        rel_img = parts[0].lstrip('/')
        img_path = os.path.join(data_root, rel_img)
        img = cv2.imread(img_path)
        if img is None:
            print(f'[skip] не смог открыть {img_path}')
            continue

        lines_src = img_path[:-len(args.img_ext)] + '.lines.txt'

        chosen_effects = rng.choice(effect_names, size=min(args.copies, len(effect_names)), replace=False)
        for effect in chosen_effects:
            aug_img = EFFECTS[effect](img, rng)
            aug_rel = rel_img[:-len(args.img_ext)] + f'{AUG_MARKER}{effect}' + args.img_ext
            aug_path = os.path.join(data_root, aug_rel)
            if not cv2.imwrite(aug_path, aug_img):
                raise SystemExit(f'Не удалось записать {aug_path}')

            aug_lines_path = aug_path[:-len(args.img_ext)] + '.lines.txt'
            relink(lines_src, aug_lines_path)

            new_parts = ['/' + aug_rel] + parts[1:]
            if len(parts) > 1:
                mask_rel = parts[1].lstrip('/')
                aug_mask_rel = mask_rel[:-4] + f'{AUG_MARKER}{effect}.png'  # .png маски
                relink(os.path.join(data_root, mask_rel), os.path.join(data_root, aug_mask_rel))
                new_parts[1] = '/' + aug_mask_rel
            new_lines.append(' '.join(new_parts))
            n_created += 1

    with open(train_gt_path, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')

    print(f'Добавлено {n_created} синтетических копий. train_gt.txt теперь: '
          f'{len(orig_lines)} оригиналов + {n_created} аугментированных = {len(new_lines)} строк.')
    print('list/test.txt не тронут — оценка F1 идёт на реальных кадрах.')


if __name__ == '__main__':
    main()
