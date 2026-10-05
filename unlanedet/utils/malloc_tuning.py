"""
Настройка аллокатора glibc для покадрового инференса в реальном времени.

Предобработка каждого кадра создаёт временные массивы по несколько МБ. По умолчанию
glibc выделяет такие блоки через mmap и возвращает их системе при освобождении,
поэтому на каждом кадре память заново "отображается" и ловит страничные прерывания
(~1.5 мс на кадр на CLRNet-пайплайне). С фиксированными порогами блоки до 32 МБ
переиспользуются из кучи процесса.
"""
import ctypes
import ctypes.util

M_TRIM_THRESHOLD = -1
M_MMAP_THRESHOLD = -3

MMAP_THRESHOLD_BYTES = 32 * 1024 * 1024  # максимум, который glibc принимает на 64-бит
TRIM_THRESHOLD_BYTES = 256 * 1024 * 1024


def tune_malloc_for_realtime():
    """Возвращает True, если настройка применена (glibc), иначе False."""
    libc_name = ctypes.util.find_library('c')
    if libc_name is None:
        return False
    libc = ctypes.CDLL(libc_name)
    if not hasattr(libc, 'mallopt'):
        return False
    ok_mmap = libc.mallopt(M_MMAP_THRESHOLD, MMAP_THRESHOLD_BYTES)
    ok_trim = libc.mallopt(M_TRIM_THRESHOLD, TRIM_THRESHOLD_BYTES)
    return bool(ok_mmap and ok_trim)
