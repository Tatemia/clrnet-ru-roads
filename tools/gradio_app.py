"""
Веб-интерфейс на Gradio для тестирования модели UnLanedet: загружаете фото или
видео прямо в браузере, выбираете чекпоинт, получаете результат с нарисованными
полосами — без вызова команд в терминале.

Запуск:
    source ~/miniconda3/bin/activate unlanedet
    export CUDA_HOME=$CONDA_PREFIX
    cd ~/Projects/UnLanedet
    python tools/gradio_app.py

Откроется на http://localhost:7860 (в WSL2 порт автоматически проброшен в Windows,
можно открыть в обычном браузере на хосте).
"""
import argparse
import os
import tempfile

import cv2
import gradio as gr
import imageio
import numpy as np
import torch

from unlanedet.checkpoint import Checkpointer
from unlanedet.config import LazyConfig, instantiate
from unlanedet.engine import default_setup
from unlanedet.engine.defaults import create_ddp_model
from unlanedet.data.transform import Preprocess
from unlanedet.model.module.core.lane import Lane
from unlanedet.tracking import LaneTracker
from unlanedet.utils.frame_resize import fit_frame_to_config
from unlanedet.utils.malloc_tuning import tune_malloc_for_realtime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Имя в интерфейсе -> (путь к чекпоинту или TensorRT-движку .engine, путь к конфигу)
CHECKPOINTS = {
    "v2 — дообучена на 23 сценах (лучшая, F1≈0.90)": (
        "output/model_best.pth", "config/clrnet/resnet34_culane_finetune.py"),
    "v2 — TensorRT FP16 (самая быстрая, F1≈0.90)": (
        "checkpoints/tensorrt/model_best_fp16.engine", "config/clrnet/resnet34_culane_finetune.py"),
    "v2 — TensorRT FP32 (F1≈0.90)": (
        "checkpoints/tensorrt/model_best_fp32.engine", "config/clrnet/resnet34_culane_finetune.py"),
    "v1 — дообучена на 15 сценах (F1≈0.88)": (
        "checkpoints/model_v1_15scenes_f088.pth", "config/clrnet/resnet34_culane_finetune.py"),
    "baseline — оригинальный CULane (без дообучения)": (
        "checkpoints/clrnet_resnet34_culane.pth", "config/clrnet/resnet34_culane.py"),
}

_model_cache = {}


def _abspath(p):
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def load_model(choice):
    if choice in _model_cache:
        return _model_cache[choice]

    ckpt_rel, cfg_rel = CHECKPOINTS[choice]
    ckpt_path, cfg_path = _abspath(ckpt_rel), _abspath(cfg_rel)

    cfg = LazyConfig.load(cfg_path)
    cfg = LazyConfig.apply_overrides(cfg, [])
    default_setup(cfg, argparse.Namespace(config=cfg_path))

    if ckpt_path.endswith('.engine'):
        from unlanedet.utils.trt_model import TRTLaneModel
        model = TRTLaneModel(ckpt_path, cfg)
    else:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        model.eval()
        Checkpointer(model).load(ckpt_path)

    transform = Preprocess(instantiate(cfg.dataloader.test.dataset.processes))

    bundle = (model, cfg, transform)
    _model_cache[choice] = bundle
    return bundle


def predict_lanes_bgr(model, cfg, transform, img_bgr):
    img_bgr = fit_frame_to_config(img_bgr, cfg)
    proc_img = img_bgr[cfg.param_config.cut_height:, :, :].astype(np.float32)
    data = {'img': proc_img, 'lanes': []}
    data = transform(data)
    data['img'] = data['img'].unsqueeze(0)
    with torch.no_grad():
        out = model(data)
    lanes = model.get_lanes(out)[0]
    if len(lanes) and isinstance(lanes[0], Lane):
        lanes = [l.to_array(cfg.param_config) for l in lanes]
    else:
        lanes = [np.array(l, dtype=np.float32) for l in lanes]
    return img_bgr, lanes


def draw_lanes_bgr(img_bgr, lanes, color=(0, 0, 255)):
    vis = img_bgr.copy()
    for lane in lanes:
        pts = [(int(x), int(y)) for x, y in lane if x > 0 and y > 0]
        for p1, p2 in zip(pts[:-1], pts[1:]):
            cv2.line(vis, p1, p2, color, 2)
        for p in pts:
            cv2.circle(vis, p, 4, (255, 0, 0), -1)
    return vis


def run_on_image(image_rgb, checkpoint_choice):
    if image_rgb is None:
        return None, "Загрузите фото."
    model, cfg, transform = load_model(checkpoint_choice)
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    img_bgr, lanes = predict_lanes_bgr(model, cfg, transform, img_bgr)
    vis = draw_lanes_bgr(img_bgr, lanes)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return vis_rgb, f"Найдено полос: {len(lanes)}"


def run_on_video(video_path, checkpoint_choice, max_seconds, progress=gr.Progress()):
    if video_path is None:
        return None, "Загрузите видео."
    model, cfg, transform = load_model(checkpoint_choice)
    tracker = LaneTracker(sample_ys=cfg.param_config.sample_y)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    max_frames = int(max_seconds * fps) if max_seconds and max_seconds > 0 else None

    out_path = tempfile.mktemp(suffix='.mp4')
    writer = None
    n_frames = 0
    total_lanes = 0

    while True:
        if max_frames is not None and n_frames >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        img_bgr, raw_lanes = predict_lanes_bgr(model, cfg, transform, frame)
        smoothed_lanes = tracker.update(raw_lanes)
        total_lanes += len(smoothed_lanes)

        left = draw_lanes_bgr(img_bgr, raw_lanes, (0, 0, 255))
        right = draw_lanes_bgr(img_bgr, smoothed_lanes, (0, 255, 0))
        cv2.putText(left, 'raw', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(right, 'tracked', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        combined = np.concatenate([left, right], axis=1)

        if writer is None:
            # H.264/yuv420p через ffmpeg (imageio-ffmpeg) — обычный cv2.VideoWriter('mp4v')
            # пишет MPEG-4, который открывается в VLC, но НЕ воспроизводится в браузере
            # (в т.ч. во встроенном плеере Gradio).
            writer = imageio.get_writer(out_path, fps=fps, codec='libx264',
                                         pixelformat='yuv420p', macro_block_size=1)
        writer.append_data(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        n_frames += 1
        if max_frames:
            progress(n_frames / max_frames, desc=f'кадр {n_frames}/{max_frames}')

    cap.release()
    if writer is not None:
        writer.close()

    if n_frames == 0:
        return None, "Не удалось прочитать ни одного кадра."
    avg_lanes = total_lanes / n_frames
    return out_path, f"Обработано {n_frames} кадров, в среднем {avg_lanes:.1f} полос(ы) на кадр (слева — сырые предсказания, справа — сглаженные трекером)."


with gr.Blocks(title="UnLanedet — тест модели") as demo:
    gr.Markdown("# UnLanedet — тест распознавания полос\nЗагрузите фото или видео, выберите модель, получите результат.")
    checkpoint_dd = gr.Dropdown(
        choices=list(CHECKPOINTS.keys()),
        value=list(CHECKPOINTS.keys())[0],
        label="Модель (чекпоинт)",
    )

    with gr.Tab("Фото"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Загрузите фото", sources=["upload"])
            img_out = gr.Image(type="numpy", label="Результат")
        img_status = gr.Textbox(label="Статус", interactive=False)
        img_btn = gr.Button("Определить полосы", variant="primary")
        img_btn.click(run_on_image, inputs=[img_in, checkpoint_dd], outputs=[img_out, img_status])

    with gr.Tab("Видео"):
        with gr.Row():
            video_in = gr.Video(label="Загрузите видео", sources=["upload"])
            video_out = gr.Video(label="Результат (raw слева / tracked справа)")
        max_sec = gr.Number(value=15, label="Обработать первые N секунд (0 = всё видео, может быть долго)")
        video_status = gr.Textbox(label="Статус", interactive=False)
        video_btn = gr.Button("Обработать видео", variant="primary")
        video_btn.click(run_on_video, inputs=[video_in, checkpoint_dd, max_sec], outputs=[video_out, video_status])

    gr.Markdown(
        "Разрешение входного файла может быть любым — модель сама приводит кадр к нужному "
        "размеру перед распознаванием."
    )


if __name__ == '__main__':
    tune_malloc_for_realtime()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
