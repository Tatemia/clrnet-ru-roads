"""
Программно создаёт в CVAT по одной задаче на каждую папку-сцену, загружает
кадры и (если рядом лежит annotations_pred.xml от tools/predict_to_cvat.py)
сразу подгружает предразметку — чтобы сразу открыть готовую задачу с
черновыми линиями и просто поправить их в браузере, без ручного создания
задач через интерфейс.

Пример:
    python tools/cvat_create_tasks.py --data-root data/ru_roads \
        --cvat-url http://localhost:8080 --user admin --password admin
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


def find_existing_task(base_url, headers, name):
    r = requests.get(f'{base_url}/api/tasks', headers=headers, params={'name': name})
    r.raise_for_status()
    for t in r.json().get('results', []):
        if t['name'] == name:
            return t['id'], t.get('size', 0) > 0
    return None, False


def create_task(base_url, headers, name, label):
    payload = {
        'name': name,
        'labels': [{'name': label, 'type': 'polyline', 'attributes': []}],
    }
    r = requests.post(f'{base_url}/api/tasks', headers=headers, json=payload)
    r.raise_for_status()
    return r.json()['id']


def upload_data(base_url, headers, task_id, image_paths):
    files = [('client_files[%d]' % i, (os.path.basename(p), open(p, 'rb'), 'image/jpeg'))
             for i, p in enumerate(sorted(image_paths))]
    data = {
        'image_quality': 90,
        'sorting_method': 'lexicographical',
    }
    r = requests.post(f'{base_url}/api/tasks/{task_id}/data', headers=headers, data=data, files=files)
    for _, (_, fh, _) in files:
        fh.close()
    r.raise_for_status()


def wait_for_task_ready(base_url, headers, task_id, timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f'{base_url}/api/tasks/{task_id}/status', headers=headers)
        if r.status_code == 200:
            state = r.json().get('state')
            if state == 'Finished':
                return True
            if state == 'Failed':
                print(f'  [!] обработка данных задачи {task_id} завершилась ошибкой: {r.json()}')
                return False
        time.sleep(2)
    print(f'  [!] таймаут ожидания обработки задачи {task_id}')
    return False


def upload_annotations(base_url, headers, task_id, xml_path, timeout=180):
    # CVAT 2.7x: POST /api/tasks/{id}/annotations/ (со слэшем!) с form-полем
    # annotation_file -> 201 (готово сразу) или 202 + rq_id (асинхронно, опрашиваем
    # /api/requests/{rq_id} пока status не станет finished/failed).
    with open(xml_path, 'rb') as f:
        files = {'annotation_file': (os.path.basename(xml_path), f, 'application/xml')}
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
            print(f'  [!] импорт разметки не удался: {rr.json()}')
            return False
        time.sleep(2)
    print(f'  [!] таймаут ожидания импорта разметки (rq_id={rq_id})')
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--cvat-url', default='http://localhost:8080')
    parser.add_argument('--user', default='admin')
    parser.add_argument('--password', default='admin')
    parser.add_argument('--label', default='lane')
    parser.add_argument('--pred-xml-name', default='annotations_pred.xml')
    parser.add_argument('--force-reupload-annotations', action='store_true',
                         help='ОПАСНО: перезаписать разметку даже в уже существующей задаче (например если её точно ещё никто не редактировал). По умолчанию выключено.')
    args = parser.parse_args()

    token = login(args.cvat_url, args.user, args.password)
    headers = {'Authorization': f'Token {token}'}

    scene_dirs = sorted(d for d in glob.glob(os.path.join(args.data_root, '*')) if os.path.isdir(d))
    if not scene_dirs:
        print(f'Не найдено папок-сцен под {args.data_root}')
        return

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        images = sorted(glob.glob(os.path.join(scene_dir, '*.jpg')))
        if not images:
            print(f'[skip] {scene_name}: нет .jpg')
            continue

        print(f'=== {scene_name}: {len(images)} кадров ===')
        existing_id, has_data = find_existing_task(args.cvat_url, headers, scene_name)
        if existing_id is not None:
            task_id = existing_id
            print(f'  задача уже существует id={task_id} (данные {"есть" if has_data else "нет"})')
            if not has_data:
                upload_data(args.cvat_url, headers, task_id, images)
                print(f'  кадры загружены, жду обработки...')
                if not wait_for_task_ready(args.cvat_url, headers, task_id):
                    continue
                print(f'  обработка завершена')
            # ВАЖНО: для уже существующей задачи разметку НЕ трогаем по умолчанию —
            # в ней может быть чужая ручная работа. Перезалить предразметку можно
            # только явно через --force-reupload-annotations (например для задачи,
            # которую точно ещё никто не редактировал).
            if not args.force_reupload_annotations:
                print(f'  разметку не трогаю (задача уже существовала) — используйте '
                      f'--force-reupload-annotations, если точно нужно перезаписать')
                continue
        else:
            task_id = create_task(args.cvat_url, headers, scene_name, args.label)
            print(f'  создана задача id={task_id}')

            upload_data(args.cvat_url, headers, task_id, images)
            print(f'  кадры загружены, жду обработки...')

            if not wait_for_task_ready(args.cvat_url, headers, task_id):
                continue
            print(f'  обработка завершена')

        pred_xml = os.path.join(scene_dir, args.pred_xml_name)
        if os.path.isfile(pred_xml):
            ok = upload_annotations(args.cvat_url, headers, task_id, pred_xml)
            print(f'  предразметка загружена: {ok}')
        else:
            print(f'  предразметки нет ({pred_xml} не найден) — задача пустая, размечать с нуля')

    print('\nГотово. Откройте http://localhost:8080/tasks в браузере.')


if __name__ == '__main__':
    main()
