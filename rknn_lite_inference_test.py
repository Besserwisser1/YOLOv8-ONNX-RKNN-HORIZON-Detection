import os, cv2, time, numpy as np
from utils import *
from rknnlite.api import RKNNLite
import traceback

conf_thres = 0.25
iou_thres = 0.45
input_width = 640
input_height = 640
model_name = 'yolov8n'
model_path = "./model"
config_path = "./config"
result_path = "./result"
image_path = "./dataset/bus.jpg"
video_path = "test.mp4"
video_inference = True
RKNN_MODEL = f'{model_path}/{model_name}-{input_height}-{input_width}.rknn'
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis','snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
isExist = os.path.exists(result_path)


if __name__ == '__main__':
    isExist = os.path.exists(result_path)
    if not isExist:
        os.makedirs(result_path)

    rknn_lite = RKNNLite(verbose=False)
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)

    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)

    try:
        cap = cv2.VideoCapture(video_path)
        while(cap.isOpened()):
            ret, image_3c = cap.read()
            if not ret:
                break
            print('--> Running model for video inference')
            image_4c, image_3c = preprocess(image_3c, input_height, input_width)
            ret = rknn_lite.init_runtime()
            start = time.time()
            outputs = rknn_lite.inference(inputs=[image_3c])
            stop = time.time()
            fps = round(1/(stop-start), 2)
            outputs[0]=np.squeeze(outputs[0])
            outputs[0] = np.expand_dims(outputs[0], axis=0)
            colorlist = gen_color(len(CLASSES))
            results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, classes=len(CLASSES)) ##[box,mask,shape]
            results = results[0]              ## batch=1
            boxes, shape = results
            if isinstance(boxes, np.ndarray):
                vis_img = vis_result(image_3c,  results, colorlist, CLASSES, result_path)
                print('--> Save inference result')
            else:
                print("No Detection result")
            cv2.waitKey(5)
    except Exception:
        pass
    finally:
        # print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        # release the cap and rknn thread pools
        cap.release()
        cv2.destroyAllWindows()
        rknn_lite.release()
