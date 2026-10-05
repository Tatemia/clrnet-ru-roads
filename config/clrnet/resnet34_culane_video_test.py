"""
Конфиг для теста УЖЕ ГОТОВОГО (не дообученного) чекпоинта CULane на СВОЁМ видео/фото
произвольного разрешения — не для обучения, только для инференса
(tools/detect.py, tools/track_lanes_video.py, tools/predict_to_cvat.py).

В отличие от config/clrnet/resnet34_culane.py (жёстко рассчитан на 1640x590,
как в оригинальном датасете CULane), здесь ori_img_w/ori_img_h/cut_height
подогнаны под конкретное видео — координаты полос иначе лягут неправильно
(именно это и происходило с data/test.mp4 до этого конфига).

Если тестируете НА ДРУГОМ видео с другим разрешением/кадрированием — поправьте
блок ниже под него (разрешение возьмите из предупреждения ffprobe/cv2, cut_height
подберите глазами — сколько пикселей сверху занимает небо/верхушки деревьев,
там разметки не бывает).
"""
from ..modelzoo import get_config

import os
from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L

from unlanedet.model.CLRNet import CLRNet, CLRHead
from unlanedet.model import ResNetWrapper, FPN

from unlanedet.data.transform import *

from fvcore.common.param_scheduler import CosineParamScheduler

# ======================= ПОД СВОЁ ВИДЕО =======================
# Разрешение кадра (ширина x высота) — узнать: cv2.VideoCapture(...).get(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT)
ori_img_w = 480
ori_img_h = 360

# Сколько пикселей сверху обрезать (небо/верхушки деревьев — там разметки не бывает).
# Для data/test.mp4 (обычная дорога, невысокий горизонт) — небольшая обрезка.
cut_height = 30

max_lanes = 4
num_classes = max_lanes + 1
sample_y = range(ori_img_h - 1, int(ori_img_h * 0.39), -20)
init_checkpoint = "checkpoints/clrnet_resnet34_culane.pth"
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
# Стандартный порог CULane. На data/test.mp4 (российская дорога, до fine-tuning)
# уверенность модели низкая (~0.15-0.35 макс.) — при 0.4 полосы не проходят вообще.
# Для диагностики/иллюстрации можно временно понизить до ~0.12, но учтите: ниже
# порога начинается шум (модель цепляется за ветки/тени, а не за разметку) —
# это не "почти правильные" полосы, а признак реального domain gap.
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
param_config.data_root = ""
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

optimizer = get_config("config/common/optim.py").AdamW
lr_multiplier = L(CosineParamScheduler)(start_value=1.0, end_value=0.001)

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
dataloader.test.dataset.processes = val_process
dataloader.test.dataset.data_root = ""
dataloader.test.dataset.cut_height = cut_height
dataloader.evaluator.ori_img_h = ori_img_h
dataloader.evaluator.ori_img_w = ori_img_w
dataloader.evaluator.cfg = param_config
