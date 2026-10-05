"""
Temporal-трекинг полос поверх покадровых предсказаний CLRNet.

Идея: CLRNet работает независимо на каждом кадре, поэтому в кадрах с сильными
искажениями (снег засыпал разметку, блик, ливень) детекция может пропасть или
"дрожать" от кадра к кадру. Здесь каждая полоса представлена независимыми
1D Kalman-фильтрами (позиция + скорость) по x-координате в нескольких
фиксированных y-строках; фильтры сглаживают дрожание, а на кадрах, где полоса
не задетектирована, продолжают экстраполировать её положение по видимой истории
(шаг предсказания без коррекции — "coasting"), пока полоса не пропадёт надолго
(track_max_age кадров) или не появится снова.

Сопоставление треков с новыми детекциями между кадрами — венгерский алгоритм
(scipy.optimize.linear_sum_assignment) по среднему расхождению x в общих
y-строках.
"""
from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


class RowKalman:
    """1D Kalman-фильтр (позиция + скорость) для x-координаты полосы в одной y-строке."""

    def __init__(self, x0: float, process_var: float = 4.0, meas_var: float = 6.0):
        self.x = np.array([x0, 0.0], dtype=np.float64)  # [позиция, скорость]
        self.P = np.eye(2) * 25.0
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.Q = np.eye(2) * process_var
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[meas_var]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: float):
        y = np.array([z]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(2) - K @ self.H) @ self.P

    @property
    def pos(self) -> float:
        return float(self.x[0])


class LaneTrack:
    def __init__(self, track_id: int, detection: Dict[float, float]):
        self.id = track_id
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.filters: Dict[float, RowKalman] = {y: RowKalman(x) for y, x in detection.items()}

    def predict(self):
        for kf in self.filters.values():
            kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Dict[float, float]):
        self.time_since_update = 0
        self.hits += 1
        for y, x in detection.items():
            if y in self.filters:
                self.filters[y].update(x)
            else:
                self.filters[y] = RowKalman(x)

    def mean_x(self) -> float:
        return float(np.mean([kf.pos for kf in self.filters.values()]))

    def get_points(self) -> np.ndarray:
        pts = sorted(((kf.pos, y) for y, kf in self.filters.items()), key=lambda p: p[1])
        return np.array(pts, dtype=np.float32)


class LaneTracker:
    """
    Трекер полос по видео. На каждый кадр вызывайте update(detections), где
    detections — список массивов (x, y) для каждой полосы, задетектированной
    моделью НА ЭТОМ кадре (координаты в пикселях исходного изображения).
    Возвращает сглаженный список полос текущего кадра (включая "coasting" —
    полосы, не увиденные в этом кадре, но ещё живые в истории).
    """

    def __init__(self, sample_ys: Sequence[float], max_age: int = 6, match_thresh: float = 45.0,
                 min_hits_to_report: int = 1):
        self.sample_ys = list(sample_ys)
        self.max_age = max_age
        self.match_thresh = match_thresh
        self.min_hits_to_report = min_hits_to_report
        self.tracks: List[LaneTrack] = []
        self._next_id = 0

    def _snap_to_grid(self, lane_xy: np.ndarray) -> Dict[float, float]:
        d = {}
        if len(lane_xy) == 0:
            return d
        ys = np.array(self.sample_ys)
        for x, y in lane_xy:
            idx = int(np.argmin(np.abs(ys - y)))
            d[float(ys[idx])] = float(x)
        return d

    def update(self, detections: List[np.ndarray]) -> List[np.ndarray]:
        for t in self.tracks:
            t.predict()

        det_dicts = [self._snap_to_grid(d) for d in detections]
        n_tracks, n_dets = len(self.tracks), len(det_dicts)
        matched_dets = set()

        if n_tracks and n_dets:
            cost = np.full((n_tracks, n_dets), 1e6)
            for i, t in enumerate(self.tracks):
                for j, dd in enumerate(det_dicts):
                    shared = set(t.filters.keys()) & set(dd.keys())
                    if shared:
                        cost[i, j] = np.mean([abs(t.filters[y].pos - dd[y]) for y in shared])
                    elif dd:
                        cost[i, j] = abs(t.mean_x() - float(np.mean(list(dd.values()))))
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] <= self.match_thresh:
                    self.tracks[i].update(det_dicts[j])
                    matched_dets.add(j)

        for j, dd in enumerate(det_dicts):
            if j not in matched_dets and dd:
                self.tracks.append(LaneTrack(self._next_id, dd))
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        self.tracks.sort(key=lambda t: t.mean_x())

        return [t.get_points() for t in self.tracks if t.hits >= self.min_hits_to_report]

    def reset(self):
        self.tracks = []
        self._next_id = 0
