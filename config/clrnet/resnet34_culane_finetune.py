"""
Fine-tuning конфиг: CLRNet (ResNet34), стартуем с чекпоинта, обученного на CULane,
дообучаем на своём небольшом датасете (российские дороги + погодные искажения).

Основан на config/clrnet/resnet34_culane.py. Отличия:
  - train.init_checkpoint указывает на чекпоинт CULane (checkpoints/clrnet_resnet34_culane.pth)
  - learning rate снижен в ~8 раз относительно обучения с нуля (0.6e-3 -> 0.75e-4)
  - batch_size/epochs подобраны под маленький датасет (~500 кадров), а не под 88880 кадров CULane
  - data_root, ori_img_w/h, max_lanes и num_classes нужно поправить под свой датасет (см. пометки TODO)

Перед запуском обязательно проверьте и при необходимости поправьте блок "ПОД СВОЙ ДАТАСЕТ" ниже.
"""
from ..modelzoo import get_config

import os
from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L

from unlanedet.model.CLRNet import CLRNet, CLRHead
from unlanedet.model import ResNetWrapper, FPN

from unlanedet.data.transform import *

from fvcore.common.param_scheduler import CosineParamScheduler

# ======================= ПОД СВОЙ ДАТАСЕТ =======================
# Путь к своему датасету в формате CULane (структура как в оригинальном CULane,
# см. tools/gen_culane_lists.py / tools/gen_culane_seg.py для подготовки).
data_root = "data/ru_roads"

# Реальное разрешение ваших кадров (ширина x высота), ДО обрезки cut_height.
# Значения ниже — под data/ru_roads (кадры из data/test.mp4, 640x360).
ori_img_w = 640
ori_img_h = 360

# Сколько пикселей сверху обрезать перед подачей в сеть (обычно небо/капот — не несёт
# полезной информации про полосы). Если кадры уже без "мусора" сверху — можно 0.
cut_height = 60

# Максимальное число полос на кадре в вашем датасете (CULane использует 4: 2 слева,
# 2 справа от эго-полосы). Поднято до 6, т.к. двойная сплошная размечается как
# 2 отдельные линии + до 2 полос с каждой стороны может не влезть в 4.
# num_classes = max_lanes + 1 (фон).
max_lanes = 6
num_classes = max_lanes + 1

# Диапазон y-координат (в пикселях исходного кадра, до обрезки), по которым модель
# сэмплирует точки полосы: от низа кадра (близко к камере) до горизонта, с шагом.
# Подберите под ori_img_h: например, для 590 (как в CULane) подходит range(589, 230, -20).
sample_y = range(ori_img_h - 1, int(ori_img_h * 0.39), -20)

# Чекпоинт CULane, с которого начинаем дообучение.
init_checkpoint = "checkpoints/model_v1_15scenes_f088.pth"  # продолжаем от уже дообученной модели (v1, F1=0.88 на 15 сценах), а не с нуля от CULane

# LR для fine-tuning — в 5-10 раз ниже, чем 0.6e-3, которые использовались при
# обучении CLRNet-ResNet34 на CULane с нуля.
finetune_lr = 0.75e-4

# Число эпох и размер батча — для датасета из ~500 кадров этого достаточно с запасом;
# при переобучении (val F1 падает, пока train loss ещё снижается) уменьшите epochs.
epochs = 25  # меньше, чем в прошлый раз (60) — продолжаем от уже обученной модели, в прошлый раз плато наступало быстро
batch_size = 8

# Синтетическая погодная аугментация (unlanedet/data/transform/weather.py):
# добавляет физически обоснованный туман/дождь/блик поверх кадров прямо во время
# обучения (случайно, с вероятностью weather_aug_p). Расширяет эффективный объём
# обучающих данных под погодные искажения без дополнительной разметки — веса
# .lines.txt/маска не меняются, т.к. эффекты не трогают геометрию.
weather_aug_p = 0.4
# =================================================================

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])
img_w = 800
img_h = 320
ignore_label = 255
bg_weight = 0.4
featuremap_out_channel = 192

iou_loss_weight = 2.
cls_loss_weight = 2.
xyt_loss_weight = 0.2
seg_loss_weight = 1.0
num_points = 72
test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)

param_config = OmegaConf.create()
param_config.iou_loss_weight = iou_loss_weight
param_config.cls_loss_weight = cls_loss_weight
param_config.xyt_loss_weight = xyt_loss_weight
param_config.seg_loss_weight = seg_loss_weight
param_config.num_points = num_points
param_config.max_lanes = max_lanes
param_config.sample_y = [i for i in sample_y]
param_config.test_parameters = test_parameters
param_config.ori_img_w = ori_img_w
param_config.ori_img_h = ori_img_h
param_config.img_w = img_w
param_config.img_h = img_h
param_config.cut_height = cut_height
param_config.img_norm = img_norm
param_config.data_root = data_root
param_config.ignore_label = ignore_label
param_config.bg_weight = bg_weight
param_config.featuremap_out_channel = featuremap_out_channel
param_config.num_classes = num_classes

model = L(CLRNet)(
    backbone=L(ResNetWrapper)(
        resnet='resnet34',
        pretrained=True,
        replace_stride_with_dilation=[False, False, False],
        out_conv=False,
    ),
    neck=L(FPN)(
        in_channels=[128, 256, 512],
        out_channels=64,
        num_outs=3,
        attention=False),
    head=L(CLRHead)(
        num_priors=192,
        refine_layers=3,
        fc_hidden_dim=64,
        sample_points=36,
        cfg=param_config
    )
)

train = get_config("config/common/train.py").train
train.init_checkpoint = init_checkpoint
epoch_per_iter = (710 // batch_size + 1)  # 710 train-кадров: все 23 сцены, включая scene_015..022 (gen_culane_lists.py, val-ratio 0.15)
total_iter = epoch_per_iter * epochs
train.max_iter = total_iter
train.checkpointer.period = epoch_per_iter
train.eval_period = epoch_per_iter

optimizer = get_config("config/common/optim.py").AdamW
optimizer.lr = finetune_lr
optimizer.weight_decay = 0.01

lr_multiplier = L(CosineParamScheduler)(
    start_value=1.0,
    end_value=0.001
)

train_process = [
    L(GenerateLaneLine)(
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='OneOf',
                 transforms=[
                     dict(name='SyntheticFog', parameters=dict()),
                     dict(name='SyntheticRain', parameters=dict()),
                     dict(name='SyntheticGlare', parameters=dict()),
                 ],
                 p=weather_aug_p),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                         y=(-0.1, 0.1)),
                                  rotate=(-10, 10),
                                  scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
        cfg=param_config
    ),
    L(ToTensor)(keys=['img', 'lane_line', 'seg']),
]

val_process = [
    L(GenerateLaneLine)(
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
        training=False,
        cfg=param_config
    ),
    L(ToTensor)(keys=['img'])
]

dataloader = get_config("config/common/culane.py").dataloader
dataloader.train.dataset.processes = train_process
dataloader.train.dataset.data_root = data_root
dataloader.train.dataset.cut_height = cut_height
dataloader.train.dataset.cfg = param_config
dataloader.train.total_batch_size = batch_size
dataloader.test.dataset.processes = val_process
dataloader.test.dataset.data_root = data_root
dataloader.test.dataset.cut_height = cut_height
dataloader.test.dataset.cfg = param_config
dataloader.test.total_batch_size = batch_size

dataloader.evaluator.data_root = data_root
dataloader.evaluator.ori_img_h = ori_img_h
dataloader.evaluator.ori_img_w = ori_img_w
dataloader.evaluator.output_basedir = "./output_finetune"
dataloader.evaluator.cfg = param_config
