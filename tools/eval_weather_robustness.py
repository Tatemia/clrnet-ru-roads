"""
Считает F1 на val-выборке (list/test.txt) с наложенной синтетической погодой
(туман/дождь/блик) поверх реальных кадров — приближённая оценка устойчивости
модели к погодным искажениям, даже когда в реальном датасете таких условий нет.

Геометрия (.lines.txt) не меняется погодными эффектами, поэтому используются
те же метки, что и для чистых кадров — сравнение полностью честное: единственная
разница между прогонами — пиксели изображения.

Пример:
    python tools/eval_weather_robustness.py config/clrnet/resnet34_culane_finetune.py \
        output/model_best.pth --data-root data/ru_roads --effects clean fog rain glare
"""
import argparse
import os
import shutil

import cv2
import numpy as np
import torch
from tqdm import tqdm

from unlanedet.checkpoint import Checkpointer
from unlanedet.config import LazyConfig, instantiate
from unlanedet.engine import default_setup
from unlanedet.engine.defaults import create_ddp_model
from unlanedet.evaluation.evaluator import inference_on_dataset
from unlanedet.evaluation.testing import print_csv_format
from unlanedet.data.transform.weather import add_fog, add_rain, add_glare

EFFECTS = {
    'clean': lambda img, rng: img,
    # "mild" — примерно то, что видела модель во время обучения (weather_aug_p диапазоны)
    'fog': lambda img, rng: add_fog(img, beta=2.2, rng=rng),
    'rain': lambda img, rng: add_rain(img, density=0.02, rng=rng),
    'glare': lambda img, rng: add_glare(img, radius_ratio=0.32, rng=rng),
    # "strong" — заметно тяжелее того, что использовалось при обучении (стресс-тест)
    'fog_strong': lambda img, rng: add_fog(img, beta=5.5, atmospheric_light=235, depth_perturb=0.25, rng=rng),
    'rain_strong': lambda img, rng: add_rain(img, density=0.07, length=30, brightness=210, defocus=1.8, rng=rng),
    'glare_strong': lambda img, rng: add_glare(img, radius_ratio=0.5, intensity=1.0, n_rays=8, rng=rng),
    # промежуточные градации дождя — поиск точки, где модель начинает проваливаться
    'rain_x2': lambda img, rng: add_rain(img, density=0.035, length=24, brightness=200, defocus=0.6, rng=rng),
    'rain_x3': lambda img, rng: add_rain(img, density=0.045, length=26, brightness=205, defocus=1.0, rng=rng),
    'rain_x4': lambda img, rng: add_rain(img, density=0.055, length=28, brightness=208, defocus=1.4, rng=rng),
}


def build_weather_dataset(orig_root, tmp_root, effect_fn, rng):
    list_path = os.path.join(orig_root, 'list', 'test.txt')
    with open(list_path) as f:
        rel_paths = [line.strip().lstrip('/') for line in f if line.strip()]

    os.makedirs(os.path.join(tmp_root, 'list'), exist_ok=True)
    shutil.copy(list_path, os.path.join(tmp_root, 'list', 'test.txt'))

    for rel in rel_paths:
        src_img = os.path.join(orig_root, rel)
        dst_img = os.path.join(tmp_root, rel)
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)

        img = cv2.imread(src_img)
        out_img = effect_fn(img, rng)
        cv2.imwrite(dst_img, out_img)

        src_lines = src_img[:-4] + '.lines.txt'
        dst_lines = dst_img[:-4] + '.lines.txt'
        if os.path.isfile(src_lines):
            if os.path.islink(dst_lines) or os.path.exists(dst_lines):
                os.remove(dst_lines)
            os.symlink(os.path.abspath(src_lines), dst_lines)


def run_eval(cfg, model, data_root, output_basedir):
    cfg.dataloader.test.dataset.data_root = data_root
    cfg.dataloader.evaluator.data_root = data_root
    cfg.dataloader.evaluator.output_basedir = output_basedir
    dataloader = instantiate(cfg.dataloader.test)
    evaluator = instantiate(cfg.dataloader.evaluator)
    results = inference_on_dataset(model, dataloader, evaluator)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('ckpt')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--effects', nargs='+', default=['clean', 'fog', 'rain', 'glare'])
    parser.add_argument('--tmp-root', default='/tmp/weather_eval')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trt-engine', default=None,
                        help='TensorRT-движок (tools/export_tensorrt.py) вместо PyTorch; ckpt тогда не используется')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, [])
    default_setup(cfg, argparse.Namespace(config=args.config))

    if args.trt_engine:
        from unlanedet.utils.trt_model import TRTLaneModel
        model = TRTLaneModel(args.trt_engine, cfg)
    else:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        model.eval()
        Checkpointer(model).load(args.ckpt)

    summary = {}
    for effect in args.effects:
        print(f'\n=== Погода: {effect} ===')
        tmp_root = os.path.join(args.tmp_root, effect)
        shutil.rmtree(tmp_root, ignore_errors=True)
        build_weather_dataset(args.data_root, tmp_root, EFFECTS[effect], rng)

        output_basedir = os.path.join(args.tmp_root, f'{effect}_predictions')
        results = run_eval(cfg, model, tmp_root, output_basedir)
        print_csv_format(results)
        summary[effect] = results.get('F1') if results else None

    print('\n=== Итог: F1 по погодным условиям ===')
    for effect, f1 in summary.items():
        print(f'{effect:8s}: F1={f1}')


if __name__ == '__main__':
    main()
