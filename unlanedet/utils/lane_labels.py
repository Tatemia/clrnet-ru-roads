"""Запись разметки .lines.txt (формат CULane) без потери уже существующих полос."""
import os
import shutil

# Метка синтетических копий кадров (tools/augment_weather_offline.py): "<кадр>__aug_<эффект>.jpg"
AUG_MARKER = '__aug_'


def link_or_copy(src, dst):
    """Ссылка dst -> src; если ОС не даёт создать ссылку (Windows без режима
    разработчика или прав администратора) — обычная копия файла."""
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copyfile(src, dst)


def has_lanes(lines_path):
    if not os.path.isfile(lines_path):
        return False
    with open(lines_path) as f:
        return any(line.strip() for line in f)


def write_lines_file(lines_path, lanes, allow_empty_overwrite=False):
    """lanes — список полос, полоса — список точек (x, y), по одной строке на полосу.

    Возвращает False и не трогает файл, если новых полос нет, а в файле они уже
    есть (например, задача в CVAT оказалась пустой), — пустая разметка не
    должна затирать сделанную вручную. allow_empty_overwrite=True отключает защиту.
    """
    if not lanes and not allow_empty_overwrite and has_lanes(lines_path):
        return False
    with open(lines_path, 'w') as f:
        for lane in lanes:
            f.write(' '.join(f'{x:.2f} {y:.2f}' for x, y in lane) + '\n')
    return True
