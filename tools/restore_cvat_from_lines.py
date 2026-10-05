"""
АВАРИЙНОЕ ВОССТАНОВЛЕНИЕ: заливает в CVAT-задачу разметку из уже существующих
.lines.txt файлов (например, финальный экспорт перед обучением) — используется,
когда разметка внутри CVAT была случайно перезаписана (например повторным
запуском cvat_create_tasks.py с предразметкой поверх уже готовой ручной работы).

Пример:
    python tools/restore_cvat_from_lines.py --data-root data/ru_roads \
        --cvat-url http://localhost:8080 --user admin --password admin \
        --scenes scene_000 scene_001 ... scene_014
"""
import argparse
import glob
import os
import time

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


def get_frame_names(base_url, headers, task_id):
    r = requests.get(f'{base_url}/api/tasks/{task_id}/data/meta', headers=headers)
    r.raise_for_status()
    return [f['name'] for f in r.json()['frames']]


def build_xml(scene_dir, frame_names, label):
    from xml.etree import ElementTree as ET
    from xml.dom import minidom

    annotations = ET.Element('annotations')
    ET.SubElement(annotations, 'version').text = '1.1'
    meta = ET.SubElement(annotations, 'meta')
    task = ET.SubElement(meta, 'task')
    ET.SubElement(task, 'name').text = 'restore'
    ET.SubElement(task, 'size').text = str(len(frame_names))
    ET.SubElement(task, 'mode').text = 'annotation'
    labels_el = ET.SubElement(task, 'labels')
    label_el = ET.SubElement(labels_el, 'label')
    ET.SubElement(label_el, 'name').text = label
    ET.SubElement(label_el, 'attributes')

    total_lanes = 0
    for idx, name in enumerate(frame_names):
        lines_path = os.path.join(scene_dir, os.path.splitext(name)[0] + '.lines.txt')
        image_el = ET.SubElement(annotations, 'image', {'id': str(idx), 'name': name})
        if os.path.isfile(lines_path):
            with open(lines_path) as f:
                for line in f:
                    vals = [float(v) for v in line.split()]
                    if len(vals) < 4:
                        continue
                    pts = [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
                    points_str = ';'.join(f'{x:.2f},{y:.2f}' for x, y in pts)
                    ET.SubElement(image_el, 'polyline', {
                        'label': label, 'points': points_str, 'occluded': '0', 'z_order': '0',
                    })
                    total_lanes += 1
    return ET.tostring(annotations), total_lanes


def upload_annotations(base_url, headers, task_id, xml_bytes, timeout=180):
    files = {'annotation_file': ('restore.xml', xml_bytes, 'application/xml')}
    params = {'format': 'CVAT 1.1'}
    r = requests.post(f'{base_url}/api/tasks/{task_id}/annotations/',
                       headers=headers, params=params, files=files)
    if r.status_code == 201:
        return True
    if r.status_code != 202:
        r.raise_for_status()
    rq_id = r.json()['rq_id']
    start = time.time()
    while time.time() - start < timeout:
        rr = requests.get(f'{base_url}/api/requests/{rq_id}', headers=headers)
        rr.raise_for_status()
        status = rr.json().get('status')
        if status == 'finished':
            return True
        if status == 'failed':
            print(f'  [!] импорт не удался: {rr.json()}')
            return False
        time.sleep(2)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--cvat-url', default='http://localhost:8080')
    parser.add_argument('--user', default='admin')
    parser.add_argument('--password', default='admin')
    parser.add_argument('--label', default='lane')
    parser.add_argument('--scenes', nargs='+', required=True)
    args = parser.parse_args()

    token = login(args.cvat_url, args.user, args.password)
    headers = {'Authorization': f'Token {token}'}

    for scene in args.scenes:
        scene_dir = os.path.join(args.data_root, scene)
        task_id = find_task_by_name(args.cvat_url, headers, scene)
        if task_id is None:
            print(f'[skip] задача "{scene}" не найдена')
            continue
        frame_names = get_frame_names(args.cvat_url, headers, task_id)
        xml_bytes, total_lanes = build_xml(scene_dir, frame_names, args.label)
        ok = upload_annotations(args.cvat_url, headers, task_id, xml_bytes)
        print(f'{scene} (task {task_id}): восстановлено {total_lanes} полос -> {ok}')


if __name__ == '__main__':
    main()
