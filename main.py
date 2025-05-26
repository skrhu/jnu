
from PySide6.QtWidgets import QApplication,QMainWindow,QWidget,QGridLayout,QFileDialog,QMessageBox
from PySide6.QtCore import Qt,QThread,Signal
from PySide6 import QtWidgets,QtCore
from PySide6.QtCore import QPropertyAnimation, QEasingCurve, QEvent, QTimer,QTime
from PySide6.QtGui import QIcon,QPainter,QBrush,QColor,QCursor,QPixmap,QImage,QConicalGradient,QPen,QFont
from PySide6 import *
from PySide6.QtWidgets import *
from ui.Ui_home import Ui_Form
from PySide6.QtWidgets import QApplication,QMainWindow
from PySide6 import QtWidgets
from PySide6.QtGui import QImage,QPixmap
from ultralytics import YOLO
import detect_tools as tools
from PIL import ImageFont
from paddleocr import PaddleOCR
import sys
import cv2
import warnings
warnings.filterwarnings('ignore')
# 所需加载的模型目录
path = 'models/best1.pt'
model = YOLO(path, task='detect')
x = 80
fontC = ImageFont.truetype("Font/platech.ttf", 50, 0)
#加载ocr模型
cls_model_dir = 'paddleModels/whl/cls/ch_ppocr_mobile_v2.0_cls_infer'
rec_model_dir = 'paddleModels/whl/rec/ch/ch_PP-OCRv4_rec_infer'
ocr = PaddleOCR(use_angle_cls=False, lang="ch", det=False, cls_model_dir=cls_model_dir,rec_model_dir=rec_model_dir)
def get_license_result(ocr,image):
    """
    image:输入的车牌截取照片
    输出，车牌号与置信度
    """
    result = ocr.ocr(image, cls=True)[0]
    if result:
        license_name, conf = result[0][1]
        if '·' in license_name:
            license_name = license_name.replace('·', '')
        return license_name, conf
class MainWindow(QMainWindow):
    x = 80
    def __init__(self):
        super().__init__()
        self.setWindowTitle(' ')
        self.ui =  Ui_Form()
        self.ui.setupUi(self)
        #隐藏窗口边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.show()

        self.cap = cv2.VideoCapture()
        self.timer = QtCore.QTimer(self)

        self.img_path = None
        self.init_slot()
        self.ui.label_2.setText('     '+str(x))
        # 定时器
        self.timer_1s = QTimer()
        self.timer_1s.timeout.connect(self.updateTime)
        self.timer_1s.start(1000)

    def init_slot(self):
        self.ui.hide.clicked.connect(lambda: self.toggleMenu(True)) 
        self.ui.main.clicked.connect(lambda: self.gotoStack(0))
        self.ui.file.clicked.connect(lambda: self.gotoStack(1))

        self.ui.select_file.clicked.connect(self.select_img)
        self.timer.timeout.connect(self.show_img11)
        self.ui.pos.clicked.connect(self.pos_img)
        self.ui.reg.clicked.connect(self.reg_video)
        self.ui.clear.clicked.connect(self.clear)
        #首次显示0号widget
        self.gotoStack(0)
    def reg_video(self):
        global x
        x = x - 1 
        self.ui.label_2.setText('     '+str(x))
        video_name=None
        video_name,_ = QtWidgets.QFileDialog.getOpenFileName(self,"选择","./img/","*.mp4")
        if video_name!=None:
            self.video_name = video_name
        else:
            msg = QMessageBox.warning(self,'!','未选择文件,请重新选择!',QMessageBox.Yes)
        self.timer.blockSignals(False)
        self.cap.open(self.video_name)
        self.timer.start(30)
        self.playing = True

    def select_img(self):
        global x
        x = x - 1 
        self.ui.label_2.setText('     '+str(x))
        img_name,_ = QtWidgets.QFileDialog.getOpenFileName(self,"选择图片","./img/","*.jpg;;*.png;;*.jpeg")
        self.show_img(img_name,self.ui.detect)
    def show_img(self,img_name,label):
        try:
            #读取图片
            im = cv2.imread(img_name)
            #获取高和宽
            results = model(im)
            results1 = results[0]
            res = results[0].plot()

            location_list = results1.boxes.xyxy.tolist()
            if len(location_list) >= 1:
                location_list = [list(map(int, e)) for e in location_list]
                # 截取每个车牌区域的照片
                license_imgs = []
                for each in location_list:
                    x1, y1, x2, y2 = each
                    cropImg = im[y1:y2, x1:x2]
                    license_imgs.append(cropImg)
                    cv2.imwrite("img.png", cropImg)
                    am = cv2.imread("img.png")
                    frame1 = cv2.cvtColor(am,cv2.COLOR_BGR2RGB)
                    res1 = QImage(frame1.data,frame1.shape[1],frame1.shape[0],frame1.shape[2] * frame1.shape[1],
                        QImage.Format_RGB888)
                    self.ui.local_pos.setPixmap(QPixmap.fromImage(res1))

            ih,iw ,_ = res.shape
            #获取标签的长和高
            w = label.geometry().width()
            h = label.geometry().height()
            #下面的目的是为了保持原始的纵横比
            if iw/w >ih/h:
                scal = w/iw
                nw = w
                nh = int(scal * ih)
                im_new = cv2.resize(res,(nw,nh))
            else:
                scal = w/iw
                nw = int(scal*iw)
                nh = h
                im_new = cv2.resize(res,(nw,nh))
            frame = cv2.cvtColor(im_new,cv2.COLOR_BGR2RGB)
            res = QImage(frame.data,frame.shape[1],frame.shape[0],frame.shape[2] * frame.shape[1],
                        QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(res))
        except Exception as e:
            print(repr(e))
    def pos_img(self):
        global x
        x = x - 1 
        self.ui.label_2.setText('     '+str(x))
        img_name,_ = QtWidgets.QFileDialog.getOpenFileName(self,"选择图片","./img/","*.jpg;;*.png;;*.jpeg")
        self.show_img1(img_name,self.ui.detect)
    def flush(self):        
        pass
    def show_img1(self,img_name,label):
        try:
            #读取图片
            im = cv2.imread(img_name)
            now_img = im

            #获取高和宽
            results = model(im)[0]
            location_list = results.boxes.xyxy.tolist()
            if len(location_list) >= 1:
                location_list = [list(map(int, e)) for e in location_list]
                # 截取每个车牌区域的照片
                license_imgs = []
                for each in location_list:
                    x1, y1, x2, y2 = each
                    cropImg = now_img[y1:y2, x1:x2]
                    license_imgs.append(cropImg)
                    cv2.imwrite("img.png", cropImg)
                    am = cv2.imread("img.png")
                    frame = cv2.cvtColor(am,cv2.COLOR_BGR2RGB)
                    res = QImage(frame.data,frame.shape[1],frame.shape[0],frame.shape[2] * frame.shape[1],
                        QImage.Format_RGB888)
                    self.ui.local_pos.setPixmap(QPixmap.fromImage(res))
                # 车牌识别结果
                lisence_res = []
                conf_list = []
                for each in license_imgs:
                    license_num, conf = get_license_result(ocr, each)
                    self.ui.detect_res.setText(license_num)
                    if license_num:
                        lisence_res.append(license_num)
                        conf_list.append(conf)
                    else:
                        lisence_res.append('无法识别')
                        conf_list.append(0)
                for text, box in zip(lisence_res, location_list):
                    now_img = tools.drawRectBox(now_img, box, text, fontC)
            now_img = cv2.resize(now_img,dsize=None,fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
            ih,iw ,_ = now_img.shape
            #获取标签的长和高
            w = label.geometry().width()
            h = label.geometry().height()
            #下面的目的是为了保持原始的纵横比
            if iw/w >ih/h:
                scal = w/iw
                nw = w
                nh = int(scal * ih)
                im_new = cv2.resize(now_img,(nw,nh))
            else:
                scal = w/iw
                nw = int(scal*iw)
                nh = h
                im_new = cv2.resize(now_img,(nw,nh))
            frame = cv2.cvtColor(im_new,cv2.COLOR_BGR2RGB)
            res = QImage(frame.data,frame.shape[1],frame.shape[0],frame.shape[2] * frame.shape[1],
                        QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(res))
        except Exception as e:
            print(repr(e))
    #窗口拖动
    def mousePressEvent(self,event):
        if event.button() == Qt.LeftButton and self.isMaximized() == False:
            self.mm_flag = True
            self.m_Position = event.globalPos() - self.pos()
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))
    def mouseMoveEvent(self,mouseMove):
        if Qt.LeftButton and self.mm_flag:
            self.move(mouseMove.globalPos()-self.m_Position)
            mouseMove.accept()
    def mouseReleaseEvent(self,mouse_event):
        self.mm_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))
    def gotoStack(self, index: int):
        self.ui.content.setCurrentIndex(index)
    def toggleMenu(self, enable):
        if enable:
            standard = 60
            maxExtend = 113
            width = self.ui.leftMenuBar.width()
            if width == 60:
                widthExtended = maxExtend
            else:
                widthExtended = standard
            self.animation = QPropertyAnimation(self.ui.leftMenuBar, b"minimumWidth")
            self.animation.setDuration(100) # ms
            self.animation.setStartValue(width)
            self.animation.setEndValue(widthExtended)
            self.animation.setEasingCurve(QEasingCurve.InOutQuint)
            self.animation.start()
    def show_img11(self):
        #获取高和宽
        label = self.ui.detect
        ret,im = self.cap.read()
        if ret:
            try:
                results = model(im)[0]
                location_list = results.boxes.xyxy.tolist()
                if len(location_list) >= 1:
                    location_list = [list(map(int, e)) for e in location_list]
                    # 截取每个车牌区域的照片
                    license_imgs = []
                    for each in location_list:
                        x1, y1, x2, y2 = each
                        cropImg = im[y1:y2, x1:x2]
                        license_imgs.append(cropImg)

                        cv2.imwrite("img.png", cropImg)
                        am = cv2.imread("img.png")
                        frame = cv2.cvtColor(am,cv2.COLOR_BGR2RGB)
                        res = QImage(frame.data,frame.shape[1],frame.shape[0],frame.shape[2] * frame.shape[1],
                            QImage.Format_RGB888)
                        self.ui.local_pos.setPixmap(QPixmap.fromImage(res))
                                           
                    # 车牌识别结果
                    lisence_res = []
                    conf_list = []
                    for each in license_imgs:
                        license_num, conf = get_license_result(ocr, each)
                        self.ui.detect_res.setText(license_num)
                        if license_num:
                            lisence_res.append(license_num)
                            conf_list.append(conf)
                        else:
                            lisence_res.append('无法识别')
                            conf_list.append(0)
                    for text, box in zip(lisence_res, location_list):
                        im = tools.drawRectBox(im, box, text, fontC)
                im = cv2.resize(im, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                ih,iw ,_ = im.shape
                #获取标签的长和高
                w = label.geometry().width()
                h = label.geometry().height()
                #上述的目的是为了保持原始的纵横比
                if iw/w >ih/h:
                    scal = w/iw
                    nw = w
                    nh = int(scal * ih)
                    im_new = cv2.resize(im,(nw,nh))
                else:
                    scal = w/iw
                    nw = int(scal*iw)
                    nh = h
                    im_new = cv2.resize(im,(nw,nh))
                frame = cv2.cvtColor(im_new,cv2.COLOR_BGR2RGB)
                im = QImage(frame.data,frame.shape[1],frame.shape[0],frame.shape[2] * frame.shape[1],
                            QImage.Format_RGB888)
                label.setPixmap(QPixmap.fromImage(im))
            except Exception as e:
                print(repr(e))
                
    def clear(self):
        self.ui.detect.clear()
        self.ui.local_pos.clear()
        self.ui.detect_res.clear()
        self.cap.release()
        self.timer.stop()
        self.playing = False
        pass

    def updateTime(self):
        now = QTime.currentTime()
        hour = " {:02d}".format(now.hour())
        minute = " {:02d}".format(now.minute())
        second = " {:02d}".format(now.second())
        self.ui.h.setText(hour)
        self.ui.m.setText(minute)
        self.ui.s.setText(second)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(".\images\icon.ico"))
    window = MainWindow()
    sys.exit(app.exec())

