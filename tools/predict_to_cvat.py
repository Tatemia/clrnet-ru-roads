"""
Прогоняет обученную (или предобученную на CULane) модель UnLanedet по папке с кадрами
и сохраняет предсказанные полосы в формате "CVAT for images 1.1" (annotations.xml) —
для последующей загрузки в CVAT как черновой разметки (Upload annotations), которую
останется поправить руками, а не рисовать с нуля.

Разрешение кадров может отличаться от того, что задано в конфиге — скрипт сам
подгоняет кадр под модель для инференса и пересчитывает координаты обратно под
реальный размер файла (см. unlanedet/utils/frame_resize.py), так что в XML
координаты будут соответствовать оригинальному файлу на диске.

Пример:
    python tools/predict_to_cvat.py config/clrnet/resnet34_culane.py \
        checkpoints/clrnet_resnet34_culane.pth \
        --img data/ru_roads/rain_day_01/*.jpg \
        --out data/ru_roads/rain_day_01/annotations.xml \
        --label lane
"""
import argparse
import glob
import os
from copy import copy
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom

import cv2
import numpy as np
import torch
from tqdm import tqdm

from unlanedet.checkpoint import Checkpointer
from unlanedet.config import LazyConfig, instantiate
from unlanedet.engine import default_setup
from unlanedet.engine.defaults import create_ddp_model
from unlanedet.data.transform import Preprocess
from unlanedet.model.module.core.lane import Lane
from unlanedet.utils.frame_resize import fit_frame_to_config


def get_img_paths(pattern):
    if '*' in pattern:
        paths = sorted(glob.glob(pattern, recursive=True))
    elif os.path.isdir(pattern):
        paths = sorted(glob.glob(os.path.join(pattern, '*.*')))
    elif os.path.isfile(pattern):
        paths = [pattern]
    else:
        raise FileNotFoundError(f'{pattern} не найден')
    return [p for p in paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]


def preprocess(img_path, cfg, processes):
    native_img = cv2.imread(img_path)
    native_h, native_w = native_img.shape[:2]
    # Для самой модели кадр приводится к разрешению конфига (если отличается) —
    # реальный файл на диске (тот, что вы загрузите в CVAT) при этом не меняется.
    ori_img = fit_frame_to_config(native_img, cfg, source_name=img_path)
    img = ori_img[cfg.param_config.cut_height:, :, :].astype(np.float32)
    data = {'img': img, 'lanes': []}
    data = processes(data)
    data['img'] = data['img'].unsqueeze(0)
    data.update({'img_path': img_path, 'native_size': (native_w, native_h)})
    return data


def predict_lanes(model, data, cfg):
    with torch.no_grad():
        out = model(data)
    lanes = model.get_lanes(out)[0]
    if len(lanes) and isinstance(lanes[0], Lane):
        lanes = [lane.to_array(cfg.param_config) for lane in lanes]
    else:
        lanes = [np.array(lane, dtype=np.float32) for lane in lanes]

    # Координаты пришли в разрешении конфига — пересчитываем в разрешение
    # РЕАЛЬНОГО файла (native_size), чтобы совпадало с тем, что увидит CVAT.
    native_w, native_h = data['native_size']
    scale_x = native_w / cfg.param_config.ori_img_w
    scale_y = native_h / cfg.param_config.ori_img_h
    if scale_x != 1.0 or scale_y != 1.0:
        lanes = [lane * np.array([scale_x, scale_y], dtype=np.float32) for lane in lanes]
    return lanes


def build_cvat_xml(entries, label):
    annotations = ET.Element('annotations')
    ET.SubElement(annotations, 'version').text = '1.1'
    meta = ET.SubElement(annotations, 'meta')
    task = ET.SubElement(meta, 'task')
    ET.SubElement(task, 'name').text = 'pre-annotation'
    ET.SubElement(task, 'size').text = str(len(entries))
    ET.SubElement(task, 'mode').text = 'annotation'
    labels_el = ET.SubElement(task, 'labels')
    label_el = ET.SubElement(labels_el, 'label')
    ET.SubElement(label_el, 'name').text = label
    ET.SubElement(label_el, 'attributes')

    for idx, (name, width, height, lanes) in enumerate(entries):
        image_el = ET.SubElement(annotations, 'image', {
            'id': str(idx), 'name': name, 'width': str(width), 'height': str(height),
        })
        for lane in lanes:
            if len(lane) < 2:
                continue
            points_str = ';'.join(f'{x:.2f},{y:.2f}' for x, y in lane)
            ET.SubElement(image_el, 'polyline', {
                'label': label, 'points': points_str, 'occluded': '0', 'z_order': '0',
            })
    return annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', help='Путь к config файлу (тот же, что для дообучения/инференса)')
    parser.add_argument('ckpt', help='Путь к чекпоинту (например checkpoints/clrnet_resnet34_culane.pth)')
    parser.add_argument('--img', required=True, help='Папка/паттерн/файл с кадрами, например data/ru_roads/scene1/*.jpg')
    parser.add_argument('--out', required=True, help='Куда сохранить annotations.xml')
    parser.add_argument('--label', default='lane', help='Имя label в CVAT (должно совпадать с тем, что создадите в задаче CVAT)')
    args = parser.parse_args()

    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, [])
    default_setup(cfg, argparse.Namespace(config=args.config))

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    model.eval()
    Checkpointer(model).load(args.ckpt)

    transform = Preprocess(instantiate(cfg.dataloader.test.dataset.processes))

    paths = get_img_paths(args.img)
    if not paths:
        print(f'Не найдено изображений по {args.img}')
        return

    entries = []
    for p in tqdm(paths, desc='predicting'):
        data = preprocess(p, cfg, transform)
        lanes = predict_lanes(model, data, cfg)
        w, h = data['native_size']
        entries.append((os.path.basename(p), w, h, lanes))

    xml_root = build_cvat_xml(entries, args.label)
    xml_str = minidom.parseString(ET.tostring(xml_root)).toprettyxml(indent='  ')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(xml_str)

    print(f'Сохранено {len(entries)} кадров -> {args.out}')
    print('В CVAT: откройте задачу -> Menu -> Upload annotations -> формат "CVAT 1.1" -> выберите этот файл.')


if __name__ == '__main__':
    main()
