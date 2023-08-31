# YOLOv8-ONNX-RKNN-Detection
***Remark: This repo only support 1 batch size***

## 0. Environment Setting
```
torch: 1.10.1+cu102
torchvision: 0.11.2+cu102
onnx: 1.10.0
onnxruntime: 1.10.0
```

## 1. Yolov8 Prerequisite
```
pip3 install ultralytics==8.0.147
pip3 install numpy==1.23.5
```

## 2. Convert Pytorch model to ONNX
Remember to change the variable to your setting.
```
python3 pytorch2onnx.py
```

## 3. RKNN Prerequisite
Install the wheel according to your python version
```
cd rknn-toolkit2/packages
pip3 install rknn_toolkit2-1.5.0+1fa95b5c-cpxx-cpxx-linux_x86_64.whl
```

## 4. Convert ONNX model to RKNN
Remember to change the variable to your setting
To improve perfermance, you can change ```./config/yolov8x-seg-xxx-xxx.quantization.cfg``` layer type.
Please follow [official document](https://github.com/rockchip-linux/rknn-toolkit2/blob/master/doc/Rockchip_User_Guide_RKNN_Toolkit2_EN-1.5.0.pdf) hybrid quatization part and reference to [example program](https://github.com/rockchip-linux/rknn-toolkit2/tree/master/examples/functions/hybrid_quant) to modify your codes.
```
python3 onnx2rknn_step1.py
python3 onnx2rknn_step2.py
```

## 5. RKNN-Lite Inference
```
python3 rknn_lite_inference.py
```

## 6. Horizon Prerequisite
```
wget -c ftp://xj3ftp@vrftp.horizon.ai/ai_toolchain/ai_toolchain.tar.gz --ftp-password=xj3ftp@123$%
tar -xvf ai_toolchain.tar.gz
cd ai_toolchain/
pip3 install h*
```

## 7. Convert ONNX model to Horizon
Remember to change the variable to your setting include ```yolov8seg_config.yaml``` and get onnx file from ```python3 pytorch2onnx.py``` with ```opset 11```
```
sh 01_check.sh
sh 02_preprocess.sh
sh 03_build.sh
```

## 8. Horizon Inference
```
python3 horizion_simulator_inference.py
python3 horizion_onboard_inference.py
```

## 9. Onnx Runtime Inference
```
python3 onnxruntime_inference.py
```

## Reference
```
https://blog.csdn.net/magic_ll/article/details/131944207
https://blog.csdn.net/weixin_45377629/article/details/124582404#t18
https://github.com/ibaiGorordo/ONNX-YOLOv8-Instance-Segmentation
```
