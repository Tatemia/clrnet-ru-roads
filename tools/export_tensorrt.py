"""
Экспорт CLRNet в ONNX и сборка TensorRT-движков (FP32 / FP16), batch=1.

В ONNX/TensorRT идёт только сама сеть: backbone + FPN + голова до сырых
предсказаний по всем якорям. NMS и декодирование линий остаются на PyTorch
(см. unlanedet/utils/trt_model.py).

TensorRT 11 строит сети только в режиме строгой типизации (флага FP16 у билдера
больше нет) — точность задаётся типами прямо в ONNX. Поэтому fp16 — отдельный
ONNX-файл: backbone+FPN в FP16, голова в FP32.

Движок привязан к версии TensorRT и конкретной видеокарте — на другом GPU/версии
его нужно пересобрать этим же скриптом (ONNX-файл переносим).

Пример:
    python tools/export_tensorrt.py config/clrnet/resnet34_culane_finetune.py \
        output/model_best.pth --out-dir checkpoints/tensorrt --precisions fp32 fp16
"""
import argparse
import os
import time

import tensorrt as trt
import torch
import torch.nn as nn

from unlanedet.checkpoint import Checkpointer
from unlanedet.config import LazyConfig, instantiate
from unlanedet.engine import default_setup
from unlanedet.engine.defaults import create_ddp_model


class NetOnly(nn.Module):
    """half_features=True: backbone+FPN в FP16, голова в FP32 (координаты линий
    считаются в полной точности). Вход и выход всегда FP32."""

    def __init__(self, model, half_features=False):
        super().__init__()
        self.model = model
        self.half_features = half_features
        if half_features:
            self.model.backbone.half()
            if self.model.aggregator:
                self.model.aggregator.half()
            if self.model.neck:
                self.model.neck.half()

    def forward(self, img):
        if self.half_features:
            img = img.half()
        fea = self.model.backbone(img)
        if self.model.aggregator:
            fea[-1] = self.model.aggregator(fea[-1])
        if self.model.neck:
            fea = self.model.neck(fea)
        if self.half_features:
            fea = [f.float() for f in fea]
        return self.model.head(fea)


def export_onnx(cfg, ckpt, onnx_path, half_features):
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    model.eval()
    Checkpointer(model).load(ckpt)

    net = NetOnly(model, half_features).eval()
    dummy = torch.randn(1, 3, cfg.param_config.img_h, cfg.param_config.img_w, device=cfg.train.device)
    with torch.no_grad():
        torch.onnx.export(net, dummy, onnx_path, opset_version=17,
                          input_names=['img'], output_names=['predictions'],
                          do_constant_folding=True)
    print(f'ONNX: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} МБ)')


def build_engine(onnx_path, engine_path, workspace_gb):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f'TensorRT не смог разобрать ONNX:\n{errors}')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f'Сборка движка {engine_path} не удалась')
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    print(f'TensorRT: {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} МБ, '
          f'сборка {time.time() - t0:.0f} с)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('ckpt')
    parser.add_argument('--out-dir', default='checkpoints/tensorrt')
    parser.add_argument('--precisions', nargs='+', default=['fp32', 'fp16'], choices=['fp32', 'fp16'])
    parser.add_argument('--workspace-gb', type=float, default=2.0)
    args = parser.parse_args()

    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, [])
    default_setup(cfg, argparse.Namespace(config=args.config))

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    for precision in args.precisions:
        onnx_path = os.path.join(args.out_dir, f'{stem}_{precision}.onnx')
        export_onnx(cfg, args.ckpt, onnx_path, half_features=(precision == 'fp16'))
        build_engine(onnx_path, os.path.join(args.out_dir, f'{stem}_{precision}.engine'),
                     args.workspace_gb)


if __name__ == '__main__':
    main()
