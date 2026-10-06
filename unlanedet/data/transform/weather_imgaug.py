"""
Регистрирует погодные аугментации (unlanedet.data.transform.weather) как обычные
imgaug-аугментеры, чтобы их можно было указывать по имени в конфиге точно так же,
как встроенные 'MotionBlur', 'HorizontalFlip' и т.д. — например:

    dict(name='OneOf',
         transforms=[
             dict(name='SyntheticFog', parameters=dict()),
             dict(name='SyntheticRain', parameters=dict()),
             dict(name='SyntheticGlare', parameters=dict()),
         ],
         p=0.4),

Эти аугментеры трогают только пиксели изображения (func_images) — geometrия
(line_strings/keypoints) через iaa.Lambda проходит без изменений по умолчанию,
так что разметка полос остаётся валидной.

Импорт этого модуля регистрирует классы в unlanedet.data.transform.__init__,
после чего они доступны через getattr(imgaug.augmenters, name).
"""
import numpy as np
import imgaug.augmenters as iaa

from .weather import add_fog, add_rain, add_glare, apply_random_weather

_rng = np.random.default_rng()


def _make_lambda(fn, **kwargs):
    def func_images(images, random_state, parents, hooks):
        return [fn(img, rng=_rng, **kwargs) for img in images]
    return iaa.Lambda(func_images=func_images)


def SyntheticFog(**kwargs):
    return _make_lambda(add_fog, **kwargs)


def SyntheticRain(**kwargs):
    return _make_lambda(add_rain, **kwargs)


def SyntheticGlare(**kwargs):
    return _make_lambda(add_glare, **kwargs)


def SyntheticWeather(**kwargs):
    return _make_lambda(apply_random_weather, **kwargs)


def register():
    iaa.SyntheticFog = SyntheticFog
    iaa.SyntheticRain = SyntheticRain
    iaa.SyntheticGlare = SyntheticGlare
    iaa.SyntheticWeather = SyntheticWeather


register()
