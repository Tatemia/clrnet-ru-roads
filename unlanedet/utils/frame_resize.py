"""
Автоматическое приведение входного кадра (фото/видео) к разрешению, под которое
настроен конфиг (cfg.param_config.ori_img_w/ori_img_h) — чтобы tools/detect.py,
tools/predict_to_cvat.py и tools/track_lanes_video.py работали на кадрах
ЛЮБОГО исходного размера без ручной правки конфига.

Если размер кадра совпадает с конфигом — ничего не делает (без лишнего
ресемплинга). Если не совпадает — растягивает/сжимает кадр (cv2.resize) до
нужного размера; при первом несовпадении в рамках запуска один раз печатает
предупреждение, чтобы это не было "тихим" сюрпризом (сюда же относится и
искажение соотношения сторон, если оно отличается от того, что было при
обучении/разметке — само распознавание при этом всё равно корректно
работает, т.к. вся последующая геометрия считается относительно текущего
кадра, но точность может быть чуть ниже, чем на кадрах "родного" размера).
"""
import cv2

_warned = set()


def fit_frame_to_config(frame, cfg, source_name=None):
    expected_w = int(cfg.param_config.ori_img_w)
    expected_h = int(cfg.param_config.ori_img_h)
    h, w = frame.shape[:2]

    if (w, h) == (expected_w, expected_h):
        return frame

    key = (w, h, expected_w, expected_h)
    if key not in _warned:
        _warned.add(key)
        label = f' ({source_name})' if source_name else ''
        print(f'[авто-ресайз]{label} входной размер {w}x{h} не совпадает с ожидаемым '
              f'моделью {expected_w}x{expected_h} — привожу автоматически. '
              f'Если соотношение сторон сильно отличается, точность может немного просесть.')

    return cv2.resize(frame, (expected_w, expected_h), interpolation=cv2.INTER_AREA)
