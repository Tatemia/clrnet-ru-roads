"""
Забирает готовую разметку прямо из CVAT через REST API (без ручного
Export task dataset в браузере) и сразу конвертирует в .lines.txt рядом с
кадрами — то же самое, что делает tools/cvat_to_culane.py, но источник
данных не XML-файл, а сама CVAT-задача.

Сопоставление task <-> папка сцены идёт по имени задачи (ожидается, что
задача называется так же, как папка, например task "scene_001" ->
data/ru_roads/scene_001/).

Пример:
    python tools/cvat_api_export.py --data-root data/ru_roads \
        --cvat-url http://localhost:8080 --user admin --password admin \
        --scenes scene_001 scene_002 scene_003 scene_004
"""
import argparse
import os

import requests


def login(base_url, user, password):
    r = requests.post(f'{base_url}/api/auth/login', json={'username': user, 'password': password})
    r.raise_for_status()
    return r.json()['key']


def find_task_by_name(base_url, headers, name):
    r = requests.get(f'{base_url}/api/tasks', headers=headers, params={'name': name})
    r.raise_for_status()
    for t in r.json()['results']:
        if t['name'] == name:
            return t['id']
    return None


def get_lane_label_id(base_url, headers, task_id, label_name):
    r = requests.get(f'{base_url}/api/labels', headers=headers, params={'task_id': task_id})
    r.raise_for_status()
    for l in r.json()['results']:
        if l['name'] == label_name:
            return l['id']
    return None


def get_frame_names(base_url, headers, task_id):
    r = requests.get(f'{base_url}/api/tasks/{task_id}/data/meta', headers=headers)
    r.raise_for_status()
    return [f['name'] for f in r.json()['frames']]


def get_shapes(base_url, headers, task_id):
    r = requests.get(f'{base_url}/api/tasks/{task_id}/annotations/', headers=headers)
    r.raise_for_status()
    return r.json()['shapes']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--cvat-url', default='http://localhost:8080')
    parser.add_argument('--user', default='admin')
    parser.add_argument('--password', default='admin')
    parser.add_argument('--label', default='lane')
    parser.add_argument('--scenes', nargs='+', required=True, help='Имена сцен = имена задач в CVAT')
    parser.add_argument('--min-points', type=int, default=4)
    args = parser.parse_args()

    token = login(args.cvat_url, args.user, args.password)
    headers = {'Authorization': f'Token {token}'}

    for scene in args.scenes:
        scene_dir = os.path.join(args.data_root, scene)
        task_id = find_task_by_name(args.cvat_url, headers, scene)
        if task_id is None:
            print(f'[skip] задача "{scene}" не найдена в CVAT')
            continue

        label_id = get_lane_label_id(args.cvat_url, headers, task_id, args.label)
        if label_id is None:
            print(f'[skip] {scene}: label "{args.label}" не найден в задаче')
            continue

        frame_names = get_frame_names(args.cvat_url, headers, task_id)
        shapes = get_shapes(args.cvat_url, headers, task_id)

        by_frame = {}
        for s in shapes:
            if s['label_id'] != label_id or s['type'] != 'polyline':
                continue
            pts = s['points']
            lane = [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
            by_frame.setdefault(s['frame'], []).append(lane)

        n_lanes, n_short = 0, 0
        for frame_idx, name in enumerate(frame_names):
            lanes = by_frame.get(frame_idx, [])
            img_path = os.path.join(scene_dir, name)
            lines_path = os.path.splitext(img_path)[0] + '.lines.txt'
            with open(lines_path, 'w') as f:
                for lane in lanes:
                    lane_sorted = sorted(lane, key=lambda p: p[1])
                    if len(lane_sorted) < args.min_points:
                        n_short += 1
                    coords = ' '.join(f'{x:.2f} {y:.2f}' for x, y in lane_sorted)
                    f.write(coords + '\n')
                    n_lanes += 1

        print(f'{scene}: {len(frame_names)} кадров, {n_lanes} полос -> .lines.txt рядом с кадрами'
              + (f' ({n_short} короче {args.min_points} точек!)' if n_short else ''))


if __name__ == '__main__':
    main()
