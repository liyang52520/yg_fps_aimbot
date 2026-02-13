import logging
import numpy as np
from typing import Optional, Any

import supervision as sv
from .yolo_model import YOLOModel

logger = logging.getLogger(__name__)

class UltralyticsYOLOModel(YOLOModel):
    """
    Ultralytics YOLO模型实现
    支持直接使用ultralytics库加载各种格式的模型
    """
    
    def __init__(self, model_path: str, device: str, conf: float):
        super().__init__(model_path, device, conf)
        self.model = None
    
    def load_model(self):
        """
        加载Ultralytics YOLO模型
        使用ultralytics库直接加载各种格式的模型
        """
        try:
            from ultralytics import YOLO
            
            # 处理设备字符串
            device = self.device
            if device.isdigit():
                device = f"cuda:{device}"
            
            # 使用ultralytics库加载模型（支持pt、onnx、engine、Openvino等格式）
            self.model = YOLO(self.model_path)
            
            # 预热模型
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            self.predict(dummy_image)
            
            logger.info(f"Ultralytics YOLO模型加载成功: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Ultralytics YOLO模型加载失败: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> sv.Detections:
        """
        预测图像
        
        Args:
            image: 输入图像
            
        Returns:
            sv.Detections: 检测结果
        """
        if not self.model:
            logger.error("模型未加载")
            return sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.empty(0),
                class_id=np.empty(0, dtype=int)
            )
        
        try:
            # 处理设备字符串
            device = self.device
            if device.isdigit():
                device = f"cuda:{device}"
            
            # 执行推理（ultralytics库会自动处理不同格式的模型）
            results = self.model(image, conf=self.conf, device=device)
            
            # 后处理
            return self.postprocess(results, image.shape)
        except Exception as e:
            logger.error(f"预测错误: {e}")
            return sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.empty(0),
                class_id=np.empty(0, dtype=int)
            )
    
    def postprocess(self, outputs: Any, image_shape: tuple) -> sv.Detections:
        """
        处理Ultralytics YOLO输出
        
        Args:
            outputs: 模型输出
            image_shape: 原始图像形状
            
        Returns:
            sv.Detections: 检测结果
        """
        # 从YOLO结果转换为Supervision Detections
        for result in outputs:
            detections = sv.Detections.from_ultralytics(result)
            return detections
        
        # 如果没有结果，返回空检测
        return sv.Detections(
            xyxy=np.empty((0, 4)),
            confidence=np.empty(0),
            class_id=np.empty(0, dtype=int)
        )
