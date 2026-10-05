"""
Инференс CLRNet через TensorRT-движок с тем же интерфейсом, что у PyTorch-модели:
`model(batch) -> сырые предсказания`, `model.get_lanes(out) -> полосы`.

В движке — только сама сеть (backbone + FPN + голова до сырых предсказаний по
всем якорям). NMS и декодирование линий остаются на PyTorch (CLRHead.get_lanes):
в них динамическое число полос и собственная CUDA-операция NMS, которые в TensorRT
не переносятся, а по времени они дешёвые.
"""
import tensorrt as trt
import torch

from unlanedet.config import instantiate

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
}


class TRTLaneModel:
    def __init__(self, engine_path, cfg, device='cuda'):
        self.device = torch.device(device)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f'Не удалось загрузить TensorRT-движок {engine_path} '
                               '(движок собирается под конкретную версию TensorRT и GPU)')
        self.context = self.engine.create_execution_context()

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(n for n in names
                               if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output_name = next(n for n in names
                                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.input_dtype = _TRT_TO_TORCH[self.engine.get_tensor_dtype(self.input_name)]
        self.output = torch.empty(
            tuple(self.engine.get_tensor_shape(self.output_name)),
            dtype=_TRT_TO_TORCH[self.engine.get_tensor_dtype(self.output_name)],
            device=self.device)
        self.context.set_tensor_address(self.output_name, self.output.data_ptr())

        # Голова нужна только ради get_lanes (NMS + декодирование), веса ей не нужны.
        self.head = instantiate(cfg.model.head).to(self.device).eval()

    def _run_single(self, img):
        img = img.to(self.device, self.input_dtype).contiguous()
        self.context.set_tensor_address(self.input_name, img.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.output.float().clone()

    def __call__(self, batch):
        # Движок собран на batch=1 (сценарий реального времени) — батч из
        # даталоадера оценщика прогоняем по одному кадру.
        imgs = batch['img']
        return torch.cat([self._run_single(imgs[i:i + 1]) for i in range(imgs.shape[0])])

    def get_lanes(self, output):
        return self.head.get_lanes(output)
