"""
Физически обоснованная синтетическая аугментация погоды: туман, дождь, блики.

Работает поверх ЧИСТЫХ кадров (например CULane) и не меняет геометрию разметки —
только пиксели изображения. Поэтому существующий .lines.txt/маска остаются
валидными для аугментированной копии кадра "бесплатно".

Модели:
  - Туман: классическая atmospheric scattering model,
        I(x) = J(x) * t(x) + A * (1 - t(x))
    где J — чистое изображение, A — яркость атмосферного света, t(x)=exp(-beta*d(x)) —
    пропускание, d(x) — карта глубины сцены. Так как у CULane нет карты глубины,
    используется псевдо-глубина по вертикальной координате кадра (чем ближе к
    горизонту — тем дальше и "туманнее"), что является стандартным дешёвым приближением
    для дорожных сцен без LiDAR/стерео.
  - Дождь: синтез дождевых полос — случайный шум, растянутый направленным
    (motion blur) ядром под углом дождя, аддитивно вмешанный в кадр (упрощённая
    версия классического подхода Гарга-Найяра).
  - Блик: радиальный градиент яркости (+опционально лучи) в случайной точке кадра,
    вмешанный screen-блендингом (имитация засветки матрицы фарами/низким солнцем).

Все функции работают с BGR uint8 (как cv2.imread) и возвращают BGR uint8 того же
размера — геометрию/размер кадра не меняют.
"""
import cv2
import numpy as np


def _normalized_depth(height, width, perturb_strength=0.15, rng=None):
    """Псевдо-карта глубины: 1.0 у верха кадра (горизонт, далеко), 0.0 у низа (камера, близко)."""
    rng = rng or np.random.default_rng()
    depth = np.repeat(
        np.linspace(1.0, 0.0, height, dtype=np.float32).reshape(-1, 1), width, axis=1
    )
    if perturb_strength > 0:
        noise = rng.standard_normal((max(height // 8, 2), max(width // 8, 2))).astype(np.float32)
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        sigma = max(width / 20.0, 1.0)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma)
        noise -= noise.min()
        noise /= (noise.max() + 1e-6)
        depth = np.clip(depth + perturb_strength * (noise - 0.5), 0.0, 1.0)
    return depth


def add_fog(img, beta=2.2, atmospheric_light=225, depth_perturb=0.15, rng=None):
    """Туман по atmospheric scattering model. beta — плотность (больше = гуще туман)."""
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    depth = _normalized_depth(h, w, depth_perturb, rng)
    transmission = np.exp(-beta * depth)[:, :, None]
    img_f = img.astype(np.float32)
    foggy = img_f * transmission + float(atmospheric_light) * (1.0 - transmission)
    return np.clip(foggy, 0, 255).astype(np.uint8)


def add_rain(img, density=0.018, length=20, angle=-70, brightness=190, defocus=0.0, rng=None):
    """Дождевые полосы. density — доля 'капель', length/angle — длина и наклон штрихов."""
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    noise = (rng.random((h, w)) < density).astype(np.float32)

    ksize = max(3, int(length) | 1)
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    rad = np.deg2rad(angle)
    for t in range(-center, center + 1):
        x = int(round(center + t * np.cos(rad)))
        y = int(round(center + t * np.sin(rad)))
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1.0
    kernel /= max(kernel.sum(), 1.0)

    streaks = cv2.filter2D(noise, -1, kernel)
    streaks = np.clip(streaks * 3.0, 0, 1)

    base = img
    if defocus > 0:
        sigma = max(defocus * min(h, w) / 100.0, 0.1)
        base = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)

    streak_rgb = np.repeat(streaks[:, :, None], 3, axis=2) * brightness
    out = base.astype(np.float32) * 0.9 + streak_rgb + 255 * 0.03
    return np.clip(out, 0, 255).astype(np.uint8)


def add_glare(img, center=None, radius_ratio=0.32, intensity=0.85, n_rays=3, rng=None):
    """Блик/засветка: радиальный градиент (+лучи), screen-блендинг."""
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    if center is None:
        cx = float(rng.uniform(0.15, 0.85) * w)
        cy = float(rng.uniform(0.05, 0.45) * h)
    else:
        cx, cy = center

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius = radius_ratio * max(h, w)
    glare = (np.clip(1.0 - dist / max(radius, 1.0), 0, 1) ** 2) * 255.0 * intensity

    if n_rays > 0:
        rays = np.zeros((h, w), dtype=np.float32)
        for _ in range(int(n_rays)):
            ang = rng.uniform(0, 2 * np.pi)
            length = radius * rng.uniform(1.5, 3.0)
            x2 = int(cx + length * np.cos(ang))
            y2 = int(cy + length * np.sin(ang))
            cv2.line(rays, (int(cx), int(cy)), (x2, y2), 255, 1)
        rays = cv2.GaussianBlur(rays, (0, 0), sigmaX=2.0)
        glare = np.clip(glare + rays * 0.5 * intensity, 0, 255)

    glare3 = np.repeat(glare[:, :, None], 3, axis=2)
    img_f = img.astype(np.float32)
    out = 255.0 - (255.0 - img_f) * (255.0 - glare3) / 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


_EFFECTS = {
    'fog': lambda img, rng: add_fog(
        img, beta=rng.uniform(1.2, 3.0), atmospheric_light=int(rng.integers(190, 240)),
        depth_perturb=rng.uniform(0.05, 0.25), rng=rng),
    'rain': lambda img, rng: add_rain(
        img, density=rng.uniform(0.008, 0.03), length=int(rng.integers(12, 28)),
        angle=rng.uniform(-80, -60), defocus=rng.uniform(0.0, 1.5), rng=rng),
    'glare': lambda img, rng: add_glare(
        img, radius_ratio=rng.uniform(0.18, 0.45), intensity=rng.uniform(0.55, 1.0),
        n_rays=int(rng.integers(0, 6)), rng=rng),
}


def apply_random_weather(img, effects=('fog', 'rain', 'glare'), max_effects=1, rng=None):
    """Применяет 1..max_effects случайно выбранных погодных эффектов подряд."""
    rng = rng or np.random.default_rng()
    n = int(rng.integers(1, max_effects + 1)) if max_effects > 1 else 1
    chosen = rng.choice(list(effects), size=min(n, len(effects)), replace=False)
    out = img
    for name in chosen:
        out = _EFFECTS[name](out, rng)
    return out
