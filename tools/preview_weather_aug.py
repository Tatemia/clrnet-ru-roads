"""
Быстрая визуальная проверка синтетических погодных эффектов на одном кадре —
удобно и для отладки, и для иллюстраций в дипломе (до/после).

Пример:
    python tools/preview_weather_aug.py --img data/ru_roads/rain_day_01/00000.jpg \
        --out preview_weather.jpg
"""
import argparse
import os

import cv2
import numpy as np

from unlanedet.data.transform.weather import add_fog, add_rain, add_glare


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--img', required=True)
    parser.add_argument('--out', default='preview_weather.jpg')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(args.img)

    rng = np.random.default_rng(args.seed)
    fog = add_fog(img, beta=2.2, rng=rng)
    rain = add_rain(img, density=0.02, rng=rng)
    glare = add_glare(img, rng=rng)

    h, w = img.shape[:2]
    label_h = 24
    tiles = []
    for name, tile in [('original', img), ('fog', fog), ('rain', rain), ('glare', glare)]:
        canvas = np.zeros((h + label_h, w, 3), dtype=np.uint8)
        canvas[label_h:, :] = tile
        cv2.putText(canvas, name, (8, label_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)

    top = np.concatenate(tiles[:2], axis=1)
    bottom = np.concatenate(tiles[2:], axis=1)
    grid = np.concatenate([top, bottom], axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f'Сохранено: {args.out}')


if __name__ == '__main__':
    main()
