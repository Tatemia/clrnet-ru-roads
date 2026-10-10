"""
Совместимость imgaug 0.4.0 с DataLoader-воркерами, запущенными через 'spawn' (Windows).

AddToHueAndSaturation заполняет кэш таблиц _LUT_CACHE на уровне класса в __init__,
т.е. только в процессе, где аугментер создан. Воркер, запущенный через 'spawn',
получает аугментер через pickle без вызова __init__, кэш в нём пустой, и аугментация
падает с "'NoneType' object is not subscriptable". Модуль импортируется при распаковке
датасета в воркере и заполняет кэш заранее.
"""
from imgaug.augmenters.color import AddToHueAndSaturation


def fill_imgaug_class_caches():
    if AddToHueAndSaturation._LUT_CACHE is None:
        AddToHueAndSaturation._LUT_CACHE = AddToHueAndSaturation._generate_lut_table()


fill_imgaug_class_caches()
