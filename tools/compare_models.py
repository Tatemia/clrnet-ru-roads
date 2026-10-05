"""
Сравнение нескольких моделей на ОДНИХ И ТЕХ ЖЕ кадрах val-выборки (list/test.txt) —
без искажений и с синтетической погодой (EFFECTS из eval_weather_robustness.py).

В отличие от eval_weather_robustness.py, искажённые копии кадров строятся один раз
с фиксированным seed на каждый эффект (не зависит от порядка эффектов), и все модели
оцениваются на побитово одинаковых изображениях. Метрика — официальная CULane
(линии шириной 30 px, венгерское сопоставление, IoU > 0.5) -> TP/FP/FN, P, R, F1.

Модель задаётся строкой  имя=конфиг:чекпоинт[:max_lanes[:cut_height]]  — чекпоинт .pth (PyTorch)
или .engine (TensorRT). max_lanes/cut_height переопределяют значения конфига (нужно, чтобы
исходный CULane-чекпоинт с 4 полосами прогнать через конфиг дообученной модели).

Пример:
    python tools/compare_models.py --data-root data/ru_roads --out output/compare_models.json \
        --model baseline=config/clrnet/resnet34_culane_finetune.py:checkpoints/clrnet_resnet34_culane.pth:4 \
        --model v2_trt_fp16=config/clrnet/resnet34_culane_finetune.py:checkpoints/tensorrt/model_best_fp16.engine
"""
import argparse
import json
import os
import zlib

import numpy as np
from omegaconf import DictConfig

from unlanedet.checkpoint import Checkpointer
from unlanedet.config import LazyConfig, instantiate
from unlanedet.engine import default_setup
from unlanedet.engine.defaults import create_ddp_model

from eval_weather_robustness import EFFECTS, build_weather_dataset, run_eval


def patch_param_config(node, max_lanes, cut_height):
    """Переопределяет max_lanes/cut_height во всех копиях param_config внутри конфига."""
    if isinstance(node, DictConfig):
        if 'max_lanes' in node and 'test_parameters' in node and max_lanes is not None:
            node.max_lanes = max_lanes
            node.num_classes = max_lanes + 1
            node.test_parameters.nms_topk = max_lanes
        if cut_height is not None and 'cut_height' in node:
            node.cut_height = cut_height
        for key in node.keys():
            child = node._get_node(key)
            if isinstance(child, DictConfig):
                patch_param_config(child, max_lanes, cut_height)


def load_model(spec):
    name, rest = spec.split('=', 1)
    parts = rest.split(':')
    cfg_path, ckpt = parts[0], parts[1]
    max_lanes = int(parts[2]) if len(parts) > 2 and parts[2] else None
    cut_height = int(parts[3]) if len(parts) > 3 and parts[3] else None

    cfg = LazyConfig.load(cfg_path)
    cfg = LazyConfig.apply_overrides(cfg, [])
    patch_param_config(cfg, max_lanes, cut_height)
    default_setup(cfg, argparse.Namespace(config=cfg_path))

    if ckpt.endswith('.engine'):
        from unlanedet.utils.trt_model import TRTLaneModel
        model = TRTLaneModel(ckpt, cfg)
    else:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        model.eval()
        Checkpointer(model).load(ckpt)
    return name, cfg, model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', action='append', required=True, help='имя=конфиг:чекпоинт[:max_lanes[:cut_height]]')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--effects', nargs='+', default=list(EFFECTS))
    parser.add_argument('--tmp-root', default='/tmp/compare_models')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True, help='JSON с результатами')
    args = parser.parse_args()

    # искажённые копии val — один раз, общие для всех моделей
    for effect in args.effects:
        rng = np.random.default_rng(args.seed + zlib.crc32(effect.encode()))
        build_weather_dataset(args.data_root, os.path.join(args.tmp_root, 'data', effect), EFFECTS[effect], rng)

    results = {}
    for spec in args.model:
        name, cfg, model = load_model(spec)
        results[name] = {}
        for effect in args.effects:
            print(f'\n=== {name} | {effect} ===')
            res = run_eval(cfg, model, os.path.join(args.tmp_root, 'data', effect),
                           os.path.join(args.tmp_root, 'pred', name, effect))
            results[name][effect] = {k: float(v) for k, v in res.items()}
        del model

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print('\n=== F1 / P / R ===')
    print(f'{"":14s}' + ''.join(f'{n:>24s}' for n in results))
    for effect in args.effects:
        row = ''.join(f'{r[effect]["F1"]:8.3f}{r[effect]["Precision"]:8.3f}{r[effect]["Recall"]:8.3f}'
                      for r in results.values())
        print(f'{effect:14s}{row}')


if __name__ == '__main__':
    main()
