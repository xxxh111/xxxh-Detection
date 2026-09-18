from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from detect_main.detect_char import Ui_MainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal, QObject, pyqtSlot
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

#共享参数类
class SharedVariable(QObject):
    valueChanged = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._value = 0

    def getValue(self):
        return self._value

    @pyqtSlot(int)
    def setValue(self, value):
        if self._value != value:
            self._value = value
            self.valueChanged.emit(self._value)


class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_msg = pyqtSignal(str)
    result_img=pyqtSignal(np.ndarray)

    send_statistic = pyqtSignal(dict)
    send_msg = pyqtSignal(str)
    def __init__(self,shared_var):
        super(DetThread, self).__init__()
        self.shared_var = shared_var
        FILE = Path(__file__).resolve()
        self.ROOT1 = FILE.parents[0]  # detect。py的父目录

        self.weights =self.ROOT1/ 'pt/best7.pt'
        self.data=self.ROOT1/ 'data/char_det.yaml'
        self.source = self.ROOT1/ 'streams.txt'
        self.dnn=False
        self.conf_thres = 0.58
        self.iou_thres = 0.5
        self.jump_out = False                   # jump out of the loop
        self.is_continue = True                 # continue/pause
        self.percent_length = 1000              # progress bar
        self.rate_check = True                  # Whether to enable delay
        self.rate = 100
        self.save_fold = './result'
        self.last_cls=''

    #点击开始按钮后会调用这个线程
    @torch.no_grad()
    def run(self,imgsz=640, source='./streams.txt', max_det=1000, device='0',view_img=True,save_txt=False,save_conf=False,save_crop=False, nosave=False, classes=None,agnostic_nms=False,augment=False, visualize=False,update=False,project='runs/detect',
            name='exp',
            exist_ok=False,
            line_thickness=3,
            hide_labels=False,
            hide_conf=False,
            half=False,
            ):
        # 是否是当天重复打开的标志
        continue_flag = 0
        #进行读取下面json中的默认存储路径作为图片保存路径
        config_file = 'config/fold.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        #self.root 定义为存储路径变量
        self.ROOT=open_fold
        #每个裁切图像的index
        crop_index=0
        #每个index文件夹最多储存的图片个数
        pause_time=int(config['pause_time'])
        temp_index=int(config['temp_index'])
        sleep_time=int(config['sleep_time'])
        #由于我们要对不断出现的符号进行判断，所以我们用时间来进行判断
        #首先设置存储路径中的日期标识
        now_date = str(time.localtime().tm_year) + str("%02d" % time.localtime().tm_mon) + str("%02d" % time.localtime().tm_mday)
        if isinstance(self.ROOT,str):
            save_dir=self.ROOT+'/'+now_date
            path = Path(save_dir)
        # 新建日期文件夹
        if not path.exists():
            path.mkdir()
        filenames = []
        for filename in os.listdir(path):
            filenames.append(int(filename))
        #当天 第二次打开程序
        if len(filenames):
            time_index=max(filenames)+1
            continue_flag=1
        else:
            time_index=1
        # Initialize
        device = torch.device('cuda:0')
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        # Load model-
        model = attempt_load(self.weights, map_location=device )  # load FP32 model
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        #names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        names= ["a001","a002","a003","a004","a005","a006","a007","a008","a009","a010","a011","a012","a013","b001","b002","b003","b004","b005","b006","b007","b008","b009","b010","b011","b012","b013","c001","c002","c003", "c004","c005","c006","c007", "c008","c009","c010","c011","c012","c013"]
        if half:
            model.half()  # to FP16
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(self.source, img_size=imgsz, stride=stride)
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        dataset = iter(dataset)
        last_time=datetime.now()
        max_conf = 0
        current_cls = ''
        frameindex=0
        xxyy = torch.tensor(0)
        # 设置flag标签，当没有暂停过为0 暂停过后设置为1
        flag = 0

        frame_time = datetime.now()
        #不断读取摄像头拍摄到的图像
        while True:
            #检测log文件夹中数字变化来判断是否需要暂停
            if self.shared_var.getValue()=='1':
                time.sleep(1)
                continue
            if flag==1:
                time.sleep(sleep_time)
                flag=0
            current_time = datetime.now()
            if (current_time - frame_time).seconds > 1:
                print(frameindex)
                frameindex = 0
                frame_time = datetime.now()
            else:
                frameindex = frameindex + 1
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
                #检测到第一张图片
                if len(det) :
                    imc = im0.copy()
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = names[c]
                        if conf > max_conf:
                            max_conf = conf
                            conf1=str(conf.item())
                            conf1=conf1[0]+conf1[2:4]
                            current_cls = label
                            xxyy = xyxy
                        annotator.box_label(xyxy, label, color=colors(c, True))
                im0 = annotator.result()
                if i==0:
                    self.send_img.emit(im0)
                else:
                    self.send_raw.emit(im0)
                now_time = datetime.now()
                #当有目标时开始做判断
                if len(det):
                    if (now_time - last_time).seconds > pause_time :
                        if continue_flag==0:
                            time_index = time_index + 1

                        else:
                            time_index=time_index
                            continue_flag=0
                        last_time = now_time
                        crop_index=1
                    else:
                        if crop_index>temp_index:
                            flag=1
                            continue
                        crop_index=crop_index+1

                    #保持图片的日期信息
                    now_date = str(now_time.year) + str("%02d" % now_time.month) + str("%02d" % now_time.day)
                    #文件夹当中的index 信息
                    save_index = self.ROOT + '/' + now_date + "/" + str(time_index)
                    path2 = Path(save_index)
                    cpath1 = str(path2)
                    if not path2.exists():
                        path2.mkdir()
                    img_path = cpath1 + '\\' + str(now_time.year) + str("%02d" % now_time.month) + str(
                        "%02d" % now_time.day) + str("%02d" % now_time.hour) + str("%02d" % now_time.minute) + str(
                        "%02d" % now_time.second) + current_cls + conf1 +str("%02d" % crop_index) + '.jpg'
                    save_one_box(xxyy, imc, file=img_path, BGR=True)
                elif len(det) == 0:
                    continue
                max_conf = torch.tensor(0)


class ReadThread(QThread):
    def __init__(self,shared_var):
        super(ReadThread, self).__init__()
        self.shared_var= shared_var
        self.read="config/a.txt"
        self.readcontent = 0
    def run(self):
        while 1:
            try:
                f = open("config/a.txt", encoding="utf-8")
                self.readcontent =f.read()
                f.close()
                self.shared_var.setValue(self.readcontent)
                time.sleep(1)
            except:
                time.sleep(1)


class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        shared_var = SharedVariable()
        self.setupUi(self)
        self.det_thread=DetThread(shared_var)
        self.read_thread=ReadThread(shared_var)
        self.pt_list=os.listdir("./pt")
        self.pt_list=[file for file in self.pt_list if file.endswith(".pt")]
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.model=self.comboBox.currentText()
        self.det_thread.weights=self.det_thread.ROOT1/'pt'/ self.model
        self.pushButton.clicked.connect(self.runtrhead)
        self.filebutton.clicked.connect(self.open_file)
        self.comboBox.currentTextChanged.connect(self.change_model)
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.label))
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.label_2))
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.result_img.connect(lambda x: self.show_image(x, self.result))

    def change_model(self,x):
        self.model=self.comboBox.currentText()
        self.det_thread.weights=self.det_thread.ROOT1/'pt'/ self.model
    def runtrhead (self):
        self.det_thread.start()
        self.read_thread.start()
    def clear(self):
        self.det_thread.last_cls=''
    def show_msg(self, msg):
        self.result_msg.setText(msg)
    def open_file(self):
        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name = QFileDialog.getExistingDirectory(None, "选取文件夹", "C:/")
        print(name)
        json_path = {"open_fold": name}
        with open('config/fold.json', 'w') as f:
            json.dump(json_path, f)
        print(name)
        self.det_thread.ROOT = name
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
