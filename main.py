from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from detect_main.detect_char import Ui_MainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon

import sys
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam, LoadStreams
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import Annotator, colors, save_one_box

from utils.torch_utils import select_device

class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    send_msg = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[0]  # detect。py的父目录
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))  # 将root添加到运行路径上
        self.ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
        self.weights =ROOT/ 'pt/best3.pt'
        self.data=ROOT/ 'data/char_det.yaml'
        self.source = ROOT/ 'streams.txt'
        self.dnn=False
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.jump_out = False                   # jump out of the loop
        self.is_continue = True                 # continue/pause
        self.percent_length = 1000              # progress bar
        self.rate_check = True                  # Whether to enable delay
        self.rate = 100
        self.save_fold = './result'
        self.last_cls='a0'


    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            source='./streams.txt',
            max_det=1000,  # maximum detections per image
            device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):
        config_file = 'config/fold.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']

        self.ROOT=open_fold
        now_date = str(time.localtime().tm_year) + str("%02d" % time.localtime().tm_mon) + str(
            "%02d" % time.localtime().tm_mday)
        if isinstance(self.ROOT,str):
            save_dir=self.ROOT+'/'+now_date
            path = Path(save_dir)
        else:
            save_dir = self.ROOT / 'save_data'
            path = Path(save_dir / now_date)
        # 新建日期文件夹
        if not path.exists():
            path.mkdir()
        max_conf = 0
        # Initialize
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        # Load model-
        model = attempt_load(self.weights, map_location=device)  # load FP32 model
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(self.source, img_size=imgsz, stride=stride)
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        dataset = iter(dataset)
        # 相关时间戳
        # 同一小时内的检测个数序列
        time_index = 1
        # 程序开始运行时的时间
        start_time = datetime.now()
        last_time = start_time
        max_conf = 0
        current_cls = ''
        last_cls = ''
        index = 0
        xxyy = torch.tensor(0)
        # 设置flag标签，当为0时表示第一次刚刚开始   1表示这次检测结果与上次相同 2表示这次检测目标与上次不同
        flag = 0
        while True:
            path,im,im0s,vid_cap =next(dataset)
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            pred = model(im, augment=augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)
            # 对于每张图片的检测结果 每张图片可能会检测到多个目标 det就一个
            for i, det in enumerate(pred):  # per image
                p, im0 = path[i], im0s[i].copy()
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    imc = im0.copy()
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = names[c]
                        if conf > max_conf:
                            max_conf = conf
                            current_cls = label
                            xxyy = xyxy
                        annotator.box_label(xyxy, label, color=colors(c, True))
                im0 = annotator.result()
                if i==0:
                    self.send_img.emit(im0)
                else:
                    self.send_raw.emit(im0)
                cv2.waitKey(1)  # 1 millisecond
                now_time = datetime.now()
                if len(det) and flag == 0:
                    time_index = 1
                    now_date = str(now_time.year) + str("%02d" % now_time.month) + str("%02d" % now_time.day)
                    if isinstance(self.ROOT, str):
                        save_index   = self.ROOT + '/' + now_date+"/"+str(time_index)
                        path2 = Path(save_index)
                        cpath1 = str(path2)
                    else:
                        save_index = save_dir / now_date / str(time_index)
                        path2 = Path(save_index)
                        cpath1 = str(path2)
                    if not path2.exists():
                        path2.mkdir()
                    if isinstance(self.ROOT, str):
                        img_path= cpath1 + '\\' + str(now_time.year) + str("%02d" % now_time.month) + str("%02d" % now_time.day) + str("%02d" % now_time.hour) + str("%02d" % now_time.minute) + str(
                        "%02d" % now_time.second) + current_cls +str("%.2f"%max_conf.item())+'.jpg'
                    else:
                        img_path = '.\\' + cpath1 + '\\' + str(now_time.year) + str("%02d" % now_time.month) + str("%02d" % now_time.day) + str("%02d" % now_time.hour) + str("%02d" % now_time.minute) + str(
                        "%02d" % now_time.second) + current_cls +str("%.2f"%max_conf.item())+'.jpg'
                    save_one_box(xxyy, imc, file=img_path, BGR=True)
                    self.last_cls = current_cls
                    # 设置flag为1 进入持续目标检测状态 即 三秒内如果持续检测到目标 除非有不同的label的图像 ，不然不保存
                    last_time = now_time
                    flag = 1

                # 表示程序在检测到第一个图像后开始的后续d1       a3c3          b2
                # 在持续检测状态下，检测到了不同目标 并且，才可以保存新的图片
                elif len(det) and flag == 1 and self.last_cls != current_cls:
                    if (now_time - last_time).seconds > 3:
                        time_index = time_index + 1
                        last_time = now_time
                    now_date = str(now_time.year) + str("%02d" % now_time.month) + str("%02d" % now_time.day)
                    if isinstance(self.ROOT, str):
                        save_index   = self.ROOT + '/' + now_date+"/"+str(time_index)
                        path2 = Path(save_index)
                        cpath1 = str(path2)
                    else:
                        save_index = save_dir / now_date / str(time_index)
                        path2 = Path(save_index)
                        cpath1 = str(path2)
                    if not path2.exists():
                        path2.mkdir()
                    if isinstance(self.ROOT, str):
                        img_path = cpath1 + '\\' + str(now_time.year) + str("%02d" % now_time.month) + str(
                            "%02d" % now_time.day) + str("%02d" % now_time.hour) + str("%02d" % now_time.minute) + str(
                            "%02d" % now_time.second) + current_cls +str("%.2f"%max_conf.item())+'.jpg'
                    else:
                        img_path = '.\\' + cpath1 + '\\' + str(now_time.year) + str("%02d" % now_time.month) + str(
                            "%02d" % now_time.day) + str("%02d" % now_time.hour) + str("%02d" % now_time.minute) + str(
                            "%02d" % now_time.second) + current_cls +str("%.2f"%max_conf.item())+'.jpg'
                    # cv2.imwrite(img_path, im0)
                    save_one_box(xxyy, imc, file=img_path, BGR=True)
                    self.last_cls = current_cls
                elif len(det) == 0:
                    continue
                max_conf = torch.tensor(0)


class MainWindow(QMainWindow,Ui_MainWindow,):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.det_thread=DetThread()
        self.pushButton.clicked.connect(self.runtrhead)
        self.filebutton.clicked.connect(self.open_file)
        self.clearbutton.clicked.connect(self.clear)
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.label))
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.label_2))
    def runtrhead (self):
        self.det_thread.start()
    def clear(self):
        self.det_thread.last_cls=''
    def open_file(self):
        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name = QFileDialog.getExistingDirectory(None,"选取文件夹","C:/")
        print(name)
        json_path={"open_fold": name}
        with open('config/fold.json','w') as f:
            json.dump(json_path,f)
        print(name)
        self.det_thread.ROOT=name
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    # myWin.showMaximized()
    sys.exit(app.exec_())
