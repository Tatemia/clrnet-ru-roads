"""
Приводит видео к нужному для тестирования модели формату: обрезка по времени,
изменение разрешения, кодирование в H.264/yuv420p (единственный кодек, который
надёжно воспроизводится и в браузере/Gradio, и в обычных плеерах — в отличие
от cv2.VideoWriter с кодеком mp4v, который открывается в VLC, но не в браузере).
Также убирает звук (H.264-видеопоток без аудиодорожки).

Поддерживает вход любого формата, который понимает ffmpeg, включая .TS с
видеорегистраторов.

Пример:
    python tools/prepare_video.py --input data/some_video.TS --output data/out.mp4 \
        --width 640 --height 360 --duration-sec 60
"""
import argparse

import imageio
import imageio_ffmpeg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--duration-sec', type=float, default=None, help='Обрезать до первых N секунд (по умолчанию — всё видео)')
    parser.add_argument('--start-sec', type=float, default=0.0, help='С какой секунды начать')
    args = parser.parse_args()

    reader = imageio.get_reader(args.input)
    meta = reader.get_meta_data()
    fps = meta.get('fps', 30.0)

    start_frame = int(args.start_sec * fps)
    max_frames = int(args.duration_sec * fps) if args.duration_sec else None

    writer = imageio.get_writer(args.output, fps=fps, codec='libx264',
                                 pixelformat='yuv420p', macro_block_size=1)

    n_written = 0
    for i, frame in enumerate(reader):
        if i < start_frame:
            continue
        if max_frames is not None and n_written >= max_frames:
            break
        if frame.shape[1] != args.width or frame.shape[0] != args.height:
            frame = imageio.core.util.Array(
                __import__('cv2').resize(frame, (args.width, args.height), interpolation=__import__('cv2').INTER_AREA)
            )
        writer.append_data(frame)
        n_written += 1
        if n_written % 500 == 0:
            print(f'  {n_written} кадров...')

    writer.close()
    reader.close()
    print(f'Готово: {n_written} кадров ({n_written/fps:.1f}с) -> {args.output} '
          f'({args.width}x{args.height}, H.264/yuv420p, без звука)')


if __name__ == '__main__':
    main()
