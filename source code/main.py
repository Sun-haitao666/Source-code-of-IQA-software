import os
import sys
import numpy as np
import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QPixmap,QIcon,QImage
from PySide2.QtWidgets import *
from PySide2.QtCore import *
import time
from numba import jit
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.widgets as widgets
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import statsmodels.api as sm
from scipy.optimize import curve_fit
from PIL import Image

# Create a class used for show the figures of mean concentration measurement and chemical imaging
mpl.use("Qt5Agg")
class MyFigureCanvas(FigureCanvas):
    def __init__(self, parent=None):
        # 创建一个Figure
        self.fig = Figure(tight_layout=True)
        self.canvas=FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)
        self.axes.axis('off')

# Create a class used for show the figure of horizontal ROI center line data of chemical image
class MyFigureCanvas2(FigureCanvas):
    def __init__(self, parent=None):
        # 创建一个Figure
        self.fig = Figure(tight_layout=True)
        self.canvas=FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.axes0=self.fig.add_subplot(211)
        self.axes = self.fig.add_subplot(212)
        self.axes0.axis('off')
        self.axes.axis('off')

# Create a class used for show the figure of vertical ROI center line data of chemical image
class MyFigureCanvas3(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(tight_layout=True)
        self.canvas=FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.axes0=self.fig.add_subplot(121)
        self.axes = self.fig.add_subplot(122)
        self.axes0.axis('off')
        self.axes.axis('off')

#The code of VHD algorithm for calculating the novel hue descriptor Huev
# 'value' corresponds to the 'HT' parameter in our paper
# @jit(nopython=True) is a decorator to speed up the loop
@jit(nopython=True)
def colorsee_variable(img1,value):
    img = img1.copy()
    img = img.astype(np.float64)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            b = img[i, j, 0]
            g = img[i, j, 1]
            r = img[i, j, 2]
            mx = max(b,g,r)
            mn = min(b,g,r)
            deta = mx - mn
            if mx == 0:    #black
                h = 0.0
            elif mx == mn:   #gray
                h = 1.0
                if mn == 255.0:   #white
                    h = 2.0
            elif mx == r :      #chromatic hues
                h = (60.0 * (g - b) / deta) + 5 +60
            elif mx == g :      #chromatic hues
                h = (60.0 * (b - r) / deta) + 5 + 180
            elif mx == b :      #chromatic hues
                h = (60.0 * (r - g) / deta) + 5 + 300  #initial hue quantification sequence of 'h' (Hue0 in our paper)
            if h<value and h>=5:    #tunable hue quantification of 'h' (Hue1 in our paper) via 'value' (HT in our paper)
                h=h+360
            if h>=value:
                h=h-(value-5)       #stable value range of final 'h' (Huev in our paper)
            img[i,j,0] = h
    return img[:,:,0]

# The algorithm for converting RGB to CMYK color space
@jit(nopython=True)
def bgr2cmyk(img1):
    img = img1.copy()
    width = img.shape[1]
    height=img.shape[0]
    img_cmyk = np.zeros((height, width, 4), dtype=np.float64)
    img = img.astype(np.float64)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            b = img[i, j, 0]/255.0
            g = img[i, j, 1]/255.0
            r = img[i, j, 2]/255.0
            k=1-max(r,g,b)
            if k==1:
                img_cmyk[i,j,0]=0
                img_cmyk[i,j,1]=0
                img_cmyk[i,j,2]=0
                img_cmyk[i,j,3]=1
            else:
                c=(1-r-k)/(1-k)
                m=(1-g-k)/(1-k)
                y=(1-b-k)/(1-k)
                img_cmyk[i,j,0] = c
                img_cmyk[i,j,1] = m
                img_cmyk[i,j,2] = y
                img_cmyk[i,j,3] = k
    return img_cmyk

# The interface of selecting GUI language and the construction of the ralationship between it and the main window
class Select_yuyan:
    BASE_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
    def __init__(self):
        qfile_stats = QFile(self.BASE_DIR+os.sep+'gui'+os.sep+'select language.ui')
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()
        self.ui = QUiLoader().load(qfile_stats)
        self.ui.setWindowIcon(QIcon(self.BASE_DIR+os.sep+'img'+os.sep+'tubiao1.ico'))
        self.label_putimg(self.ui.label,self.BASE_DIR + os.sep + 'img'+os.sep+'shebei.jpg')
        self.ui.pushButton_chinese.clicked.connect(self.chinese)
        self.ui.pushButton_english.clicked.connect(self.english)

    def chinese(self):
        self.ui.pushButton_chinese.setEnabled(False)
        self.ui.pushButton_english.setEnabled(False)
        self.label_putimg(self.ui.label, self.BASE_DIR + os.sep + 'img'+os.sep+ 'wait_chn.jpg')
        self.yuyan = 'chinese'
        self.wait = Wait()
        self.wait.signal0.connect(self.opengui)
        self.wait.start()

    def opengui(self):
        global color_chemistry
        color_chemistry = ColorChemistry()
        color_chemistry.show()

    def english(self):
        self.ui.pushButton_chinese.setEnabled(False)
        self.ui.pushButton_english.setEnabled(False)
        self.label_putimg(self.ui.label, self.BASE_DIR + os.sep + 'img'+os.sep+ 'wait_eng.jpg')
        self.yuyan='english'
        self.wait = Wait()
        self.wait.signal0.connect(self.opengui)
        self.wait.start()

    def label_putimg(self,label,img_path):
        pix = QPixmap(img_path)
        label.setPixmap(pix)
        label.setScaledContents(True)

# The interface of main window
class ColorChemistry(QMainWindow):
    # Defination of various static attributes of this class
    BASE_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
    jiaozhunx=[]
    jiaozhuny=[]
    std_color_suanfalist=[]
    std_fenbianlv_list=[]
    std_yuzhi_list=[]
    std_analyze_name=[]
    std_zibianliang=[]
    std_expousuremode=[]
    std_exposuretimelist=[]
    std_whitebalancelist=[]
    std_wb_mode=[]
    nongdu_list=[]
    time_list=[]
    h_list=[]
    biaoqu_list=[]
    color_suanfalist=[]
    fenbianlv_list=[]
    yuzhi_list=[]
    analyze_name=[]
    exposuretimelist=[]
    expousuremode=[]
    whitebalancelist=[]
    wb_mode=[]
    list_roi=[]
    savepath_jiaozhun1=''
    num=0
    fig0=0
    fig1=1
    zibianliang=[]
    threads = []
    dialogs = []
    chemicalvideotime_list=[]
    fig1_complete=1
    fig0_complete =1
    fig6_complete=1
    camshow_complete=1
    roishow_complete=1
    fig3_complete=1
    switch_show_roicenterline=1
    switch_show_roicenterline2=1
    showvideo_complete=1
    swich_takevideo = 1
    gui_roicenter = 0
    gui_roicenter2=0
    switch_select_video_folder=0
    switch_set_video_parameters=0
    switch_set_video_parameters2 = 0
    switch_importvideo_folder=0
    switch_startchemicalvideo = 0
    getchemicalvideo_t_current=0
    chemicalvideoframenum=0
    startsignal = False
    try:
        rc = {'font.sans-serif': 'Arial','font.size': 8}
        mpl.rcParams.update(rc)
    except:
        pass
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['svg.fonttype'] = 'none'
    roicenterimg=1
    roicenterimg2=1
    frame=1
    roiimg=1
    camopen = False
    csv_path0=''
    video_importpath=''
    picture_path0=''

# Interface initialization and the constuction of relationship between various buttons and the functions that perform various operations
    def __init__(self):
        self.threads.append(selectyuyan.wait)
        self.yuyan=selectyuyan.yuyan
        selectyuyan.ui.hide()
        super(ColorChemistry, self).__init__()
        if self.yuyan=='english':
            qfile_stats = QFile(self.BASE_DIR+os.sep+ 'gui'+os.sep+ 'Quantitative chemical analysis by color 2_english.ui')
        else:
            qfile_stats = QFile(self.BASE_DIR + os.sep+ 'gui'+os.sep + 'Quantitative chemical analysis by color 2.ui')
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()
        self.ui = QUiLoader().load(qfile_stats)
        self.setCentralWidget(self.ui)
        self.setWindowIcon(QIcon(self.BASE_DIR + os.sep +'img'+os.sep+ "tubiao1.ico"))
        if self.yuyan=='english':
            self.setWindowTitle('IQA')
        else:
            self.setWindowTitle('IQA')
        self.showMaximized()

        self.ui.pushButton_4.clicked.connect(self.save_apply_fit)
        self.ui.button_fit.clicked.connect(self.fit)
        self.ui.pushButton_3.clicked.connect(self.clear_jiaozhundata)
        self.ui.button_addpoint.clicked.connect(self.addpoint)
        self.ui.Button_savepath_jiaozhun.clicked.connect(self.path_jiaozhun)
        self.ui.comboBox_8.currentTextChanged.connect(self.cbshape)
        self.ui.checkBox.stateChanged.connect(self.on_state_changed)
        self.ui.horizontalSlider_fangwei.valueChanged.connect(self.label_58)
        self.ui.horizontalSlider_fuyang.valueChanged.connect(self.label_56)
        self.ui.lineEdit_danwei.textChanged.connect(self.danwei_change)
        self.ui.lineEdit_wuzhi.textChanged.connect(self.wuzhi_textchange)
        self.ui.comboBox_TYPE.currentTextChanged.connect(self.huitu2d3d)
        self.ui.comboBox_4.currentTextChanged.connect(self.wbset)
        self.ui.comboBox_exptime.currentTextChanged.connect(self.exptimeset)
        self.ui.exposureSlider.valueChanged.connect(self.changelabel)
        self.ui.horizontalSlider_wb.valueChanged.connect(self.change_wb)
        self.ui.button_open_camera.clicked.connect(self.open_camera)
        self.ui.button_roi.clicked.connect(self.select_roi)
        self.ui.horizontalSlider_value.valueChanged.connect(self.changelabel2)
        self.ui.button_reselect_roi.clicked.connect(self.reselect)
        self.ui.button_start.clicked.connect(self.start)
        self.ui.pushButton_cleardata.clicked.connect(self.clear)
        self.ui.button_stoplong.clicked.connect(self.stop_long)
        self.ui.radioButton_roisum.toggled.connect(self.roisumstate)
        self.ui.radioButton_longtime.toggled.connect(self.longselect)
        self.ui.radioButton_single.toggled.connect(self.singleselect)
        self.ui.action_save1D.triggered.connect(self.save_1D)
        self.ui.action_save2D.triggered.connect(self.save_2D)
        self.ui.button_start2D.clicked.connect(self.start_2D)
        self.ui.button_stop2D.clicked.connect(self.stop_2D)
        self.ui.button_inputbiaoqu.clicked.connect(self.input_biaoqufx)
        self.ui.actionda_openPicture.triggered.connect(self.open_picture)
        self.ui.pushButton_paizhao.clicked.connect(self.paizhao)
        self.ui.pushButton_OFFcam.clicked.connect(self.closecamera)
        self.ui.lineEdit_yuzhi.textChanged.connect(self.setyuzhi)
        self.ui.actiondd_opencsv.triggered.connect(self.opencsv)
        self.ui.comboBox_2_zihao.currentTextChanged.connect(self.zihao)
        self.ui.comboBox_ziti.currentTextChanged.connect(self.ziti)
        self.ui.radioButton_jiacu.toggled.connect(self.zitijiacu)
        self.ui.pushButton_paizhaopath.clicked.connect(self.paizhaopath)
        self.ui.comboBox_colorsee.currentTextChanged.connect(self.Hue_method_change)
        self.ui.action_videostart.triggered.connect(self.take_videos)
        self.ui.action_videostop.triggered.connect(self.stop_takevideos)
        self.ui.action_importvideo.triggered.connect(self.importvideo)
        self.ui.actionchemicalvideostart.triggered.connect(self.start_getchemicalvideo)
        self.ui.actionchemicalvideostop.triggered.connect(self.stop_getchemicalvideo)

        self.canvas1 = MyFigureCanvas()
        self.graphic_scene1 = QGraphicsScene()  # 创建一个QGraphicsScene
        self.graphic_scene1.addWidget(self.canvas1)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到放到QGraphicsScene中的
        self.ui.graphicsView1.setScene(self.graphic_scene1)  # 把QGraphicsScene放入QGraphicsView
        self.ratio1 = 0.995
        self.ratio2 = 0.995
        self.graphic_scene1.setSceneRect(0, 0, int(self.ratio2 * self.ui.graphicsView1.width()),
                                         int(self.ratio1 * self.ui.graphicsView1.height()))
        self.canvas1.resize(int(self.ratio2 * self.ui.graphicsView1.width()),
                            int(self.ratio1 * self.ui.graphicsView1.height()))
        self.ui.graphicsView1.show()
        self.line1 = dict(color='black', lw=1)
        self.cursor1 = widgets.Cursor(self.canvas1.axes, useblit=True, **self.line1)
        self.canvas1.mpl_connect('motion_notify_event', self.on_mouse_move)


        self.canvas2 = MyFigureCanvas()
        toolbar=NavigationToolbar(self.canvas2,self)
        self.ui.horizontalLayout_56.addWidget(toolbar)
        # self.ui.label_60.add
        self.graphic_scene2 = QGraphicsScene()  # 创建一个QGraphicsScene
        self.graphic_scene2.addWidget(self.canvas2)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到放到QGraphicsScene中的
        self.ui.graphicsView2.setScene(self.graphic_scene2)  # 把QGraphicsScene放入QGraphicsView
        self.ratio3 = 0.995
        self.ratio4 = 0.995
        self.graphic_scene2.setSceneRect(0, 0, int(self.ratio4 * self.ui.graphicsView1.width()),
                                         int(self.ratio3 * self.ui.graphicsView1.height()))
        self.canvas2.resize(int(self.ratio4 * self.ui.graphicsView1.width()),
                            int(self.ratio3 * self.ui.graphicsView1.height()))
        self.ui.graphicsView2.show()
        self.canvas2.mpl_connect('motion_notify_event', self.on_mouse_move2)

        self.canvas3 = MyFigureCanvas()
        self.graphic_scene3 = QGraphicsScene()  # 创建一个QGraphicsScene
        self.graphic_scene3.addWidget(self.canvas3)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到放到QGraphicsScene中的
        self.ui.graphicsView3.setScene(self.graphic_scene3)  # 把QGraphicsScene放入QGraphicsView
        self.ratio3 = 0.995
        self.ratio4 = 0.995
        self.graphic_scene3.setSceneRect(0, 0, int(self.ratio4 * self.ui.graphicsView3.width()),
                                         int(self.ratio3 * self.ui.graphicsView3.height()))
        self.canvas3.resize(int(self.ratio4 * self.ui.graphicsView3.width()),
                            int(self.ratio3 * self.ui.graphicsView3.height()))
        self.ui.graphicsView3.show()
        self.line3 = dict(color='black', lw=1)
        self.cursor3 = widgets.Cursor(self.canvas3.axes, useblit=True, **self.line3)
        self.canvas3.mpl_connect('motion_notify_event', self.on_mouse_move)

# Follows are the functions to response to various buttons
    def roisumstate(self):
        self.jietuok1=False
        if self.ui.radioButton_roisum.isChecked():
            self.ui.page_3.setEnabled(False)
        else:
            self.ui.page_3.setEnabled(True)

    def save_apply_fit(self):
        self.ui.pushButton_4.setEnabled(False)
        print(self.savepath_jiaozhun1)
        if self.savepath_jiaozhun1=='':
            if self.yuyan == 'english':
                self.msg_boxfitpath = QMessageBox(QMessageBox.Information, 'Calibration',
                                              f'Please select outpath for calibration data!')
            else:
                self.msg_boxfitpath = QMessageBox(QMessageBox.Information, '校准',
                                              f'请选择校准数据的输出路径！')
            self.msg_boxfitpath.exec_()
            self.ui.pushButton_4.setEnabled(True)
            return
        self.t_save=time.strftime('%H-%M-%S', time.localtime())
        if self.yuyan == 'english':
            self.datafit = {
                "Parameters": ['Fit type','Equation','R2','Adjusted R2'],
                'Content': [self.ui.comboBox.currentText(),self.thread_fit.equation,self.thread_fit.r_squared,self.thread_fit.adjusted_r_squared],
                'Known concentration': self.jiaozhunx,
                f'Value of Analytical signal': self.jiaozhuny,
                '': [],
                'Quantitative parameter': self.std_color_suanfalist,
                'HT': self.std_yuzhi_list,
                'Resolution of camera':self.std_fenbianlv_list,
                'Exposure mode':self.std_expousuremode,
                'Exposure time':self.std_exposuretimelist,
                'White balance mode':self.std_wb_mode,
                'White balance value':self.std_whitebalancelist}
        else:
            self.datafit = {
                "参数": ['Fit type', 'Equation', 'R2', 'Adjusted R2'],
                '内容': [self.ui.comboBox.currentText(), self.thread_fit.equation, self.thread_fit.r_squared,
                            self.thread_fit.adjusted_r_squared],
                '已知浓度': self.jiaozhunx,
                f'分析信号的值': self.jiaozhuny,
                '': [],
                '定量参数': self.std_color_suanfalist,
                'HT': self.std_yuzhi_list,
                '相机分辨率': self.std_fenbianlv_list,
                '曝光模式': self.std_expousuremode,
                '曝光时间': self.std_exposuretimelist,
                '白平衡模式': self.std_wb_mode,
                '白平衡值': self.std_whitebalancelist}
        try:
            self.fig6.savefig(self.savepath_jiaozhun1 + os.sep + self.t_save+ f'-{self.ui.lineEdit_zibianliang.text()}' + '-calibration.tiff', dpi=350,
                              bbox_inches='tight', pad_inches=0.1)
        except:
            pass
        try:
            self.df_fit = pd.DataFrame(pd.DataFrame.from_dict(self.datafit, orient='index').values.T, columns=list(self.datafit.keys()))
            self.df_fit.to_csv(self.savepath_jiaozhun1 +os.sep + self.t_save + f'-{self.ui.lineEdit_zibianliang.text()}'+ '-Calibration.csv', header=True, sep=',',encoding="utf_8_sig")

            self.ui.lineEdit_biaoqu.clear()
            self.ui.lineEdit_biaoqu.setText(self.thread_fit.std_curve)
            with open(self.savepath_jiaozhun1 +os.sep+self.t_save + f'-{self.ui.lineEdit_zibianliang.text()}'+ '-Standard curve.txt', 'w') as filefit:
                filefit.write(self.thread_fit.std_curve)
        except:
            if self.yuyan == 'english':
                self.msg_boxsavecal = QMessageBox(QMessageBox.Information, 'Calibration',
                                              f'Fail to save calibration data! Maybe because the name of sample has special characters. It is best to name with English characters and underscores')
            else:
                self.msg_boxsavecal = QMessageBox(QMessageBox.Information, '校准',
                                              f'校准结果保存失败，可能是因为样品名称含有特殊字符，样品名称最好用英文字符与下划线命名！')
            self.msg_boxsavecal.exec_()
        self.ui.pushButton_4.setEnabled(True)

    def fit(self):
        self.ui.button_fit.setEnabled(False)
        self.thread_fit = ThreadFit()
        self.threads.append(self.thread_fit)
        self.thread_fit.signal0.connect(self.huitu_fitcurve)
        self.thread_fit.signal2.connect(self.erro_fit)
        self.thread_fit.start()

    def erro_fit(self):
        if self.yuyan=='english':
            self.msg_boxfit = QMessageBox(QMessageBox.Information, 'Calibration',
                                        f'Fail to fit!')
        else:
            self.msg_boxfit = QMessageBox(QMessageBox.Information, '校准',
                                        f'拟合失败！')
        self.msg_boxfit.exec_()
        self.ui.button_fit.setEnabled(True)
        self.ui.pushButton_4.setEnabled(False)

    def huitu_fitcurve(self):
        try:
            self.fitcurve0.remove()
        except:
            pass
        self.ratio6 = 0.995
        self.ratio6 = 0.995
        self.graphic_scene3.setSceneRect(0, 0, int(self.ratio6 * self.ui.graphicsView3.width()),
                                         int(self.ratio6 * self.ui.graphicsView3.height()))
        self.canvas3.resize(int((self.ratio6 - 0.045) * self.ui.graphicsView3.width()),
                            int((self.ratio6 - 0.045) * self.ui.graphicsView3.height()))
        self.fitcurve0,=self.fig6_ax2.plot(self.thread_fit.linex,self.thread_fit.liney,color='red', linewidth=2,)
        if self.ui.comboBox.currentIndex()==0:
            self.equation_text = f'y = {self.thread_fit.slope:.2f}x + {self.thread_fit.intercept:.2f}'
            self.stats_text = f'R² = {self.thread_fit.r_squared:.4f}\nAdjusted R² = {self.thread_fit.adjusted_r_squared:.4f}'
            self.ui.label_52.setText(self.equation_text + '\n' + self.stats_text)
        elif self.ui.comboBox.currentIndex()==1:
            self.stats_text = f'R² = {self.thread_fit.r_squared:.4f}\nAdjusted R² = {self.thread_fit.adjusted_r_squared:.4f}'
            self.ui.label_52.setText(self.thread_fit.curvetext + '\n' + self.thread_fit.fit_params_text+ '\n' +self.stats_text)
        elif self.ui.comboBox.currentIndex() == 2:
            self.stats_text = f'R² = {self.thread_fit.r_squared:.4f}\nAdjusted R² = {self.thread_fit.adjusted_r_squared:.4f}'
            self.ui.label_52.setText(
                self.thread_fit.curvetext + '\n' + self.thread_fit.fit_params_text + '\n' + self.stats_text)
        self.fig6.tight_layout()
        self.canvas3.draw()
        self.canvas3.flush_events()

        self.ui.button_fit.setEnabled(True)
        self.ui.pushButton_4.setEnabled(True)

    def clear_jiaozhundata(self):
        self.ui.pushButton_3.setEnabled(False)
        self.jiaozhunx.clear()
        self.jiaozhuny.clear()
        self.std_color_suanfalist.clear()
        self.std_fenbianlv_list.clear()
        self.std_yuzhi_list.clear()
        self.std_analyze_name.clear()
        self.std_zibianliang.clear()
        self.std_expousuremode.clear()
        self.std_exposuretimelist.clear()
        self.std_whitebalancelist.clear()
        self.std_wb_mode.clear()
        if self.yuyan=='english':
            self.msg_boxstd_clear = QMessageBox(QMessageBox.Information, 'Calibration',
                                        f'Data of calibration was cleard!')
        else:
            self.msg_boxstd_clear = QMessageBox(QMessageBox.Information, '校准',
                                        f'校准数据清除成功！')
        self.msg_boxstd_clear.exec_()
        self.ui.pushButton_3.setEnabled(True)

    def path_jiaozhun(self):
        self.savepath_jiaozhun1=''
        self.ui.Button_savepath_jiaozhun.setEnabled(False)
        if self.yuyan == 'english':
            self.savepath_jiaozhun = QFileDialog.getExistingDirectory(self.ui, "Please select the output folder")
        else:
            self.savepath_jiaozhun = QFileDialog.getExistingDirectory(self.ui, "请选择输出文件夹")
        if self.savepath_jiaozhun == '':
            self.ui.lineEdit_2.setText(self.savepath_jiaozhun)
            self.ui.Button_savepath_jiaozhun.setEnabled(True)
            return
        self.savepath_jiaozhun1 = self.savepath_jiaozhun + os.sep + 'Calibration data'
        if not os.path.exists(self.savepath_jiaozhun1):
            os.mkdir(self.savepath_jiaozhun1)
        self.ui.lineEdit_2.setText(self.savepath_jiaozhun + '/Calibration data')
        self.ui.Button_savepath_jiaozhun.setEnabled(True)

    def addpoint(self):
        self.ui.button_addpoint.setEnabled(False)
        self.thread_addpoint = ThreadAddpoint()
        self.threads.append(self.thread_addpoint)
        self.thread_addpoint.signal0.connect(self.sandiantu)
        self.thread_addpoint.signal1.connect(self.erro_input_num)
        self.thread_addpoint.start()

    def erro_input_num(self):
        if self.yuyan=='english':
            self.msg_boxstd = QMessageBox(QMessageBox.Information, 'Calibration',
                                        f'Only number can be input as known concentration!')
        else:
            self.msg_boxstd = QMessageBox(QMessageBox.Information, '校准',
                                        f'已知浓度只能输入数字！')
        self.msg_boxstd.exec_()
        self.ui.button_addpoint.setEnabled(True)

    def sandiantu(self):
        self.ui.statusbar.showMessage('')
        if self.ui.radioButton_jiacu.isChecked():
            fontweight = 'bold'
        else:
            fontweight = 'normal'

        self.ratio6 = 0.995
        self.ratio6 = 0.995
        self.graphic_scene3.setSceneRect(0, 0, int(self.ratio6 * self.ui.graphicsView3.width()),
                                         int(self.ratio6 * self.ui.graphicsView3.height()))
        self.canvas3.resize(int((self.ratio6-0.045) * self.ui.graphicsView3.width()),
                         int((self.ratio6-0.045) * self.ui.graphicsView3.height()))
        self.fig6=self.canvas3.fig
        self.fig6_ax2=self.canvas3.axes
        self.fig6_ax2.cla()
        self.fig6_ax2.axis('on')
        self.fig6_ax2.grid()
        if self.ui.lineEdit_wuzhi.text()!='':
            self.fig6_ax2.set_xlabel(self.ui.lineEdit_wuzhi.text()+self.ui.lineEdit_danwei.text())
        else:
            self.fig6_ax2.set_xlabel('Concentration')
        self.fig6_ax2.scatter(self.jiaozhunx, self.jiaozhuny, s=30, color='purple', marker='o', edgecolors='gray')
        self.fig6_ax2.set_ylabel(self.ui.comboBox_colorsee.currentText())
        plt.tight_layout()
        self.canvas3.draw()
        self.canvas3.flush_events()
        self.ui.button_addpoint.setEnabled(True)
        self.ui.button_fit.setEnabled(True)

    def cbshape(self):
        if self.ui.comboBox_8.currentText()=='no set':
            self.ui.label_61.setEnabled(False)
            self.ui.comboBox_9.setEnabled(False)
        else:
            self.ui.label_61.setEnabled(True)
            self.ui.comboBox_9.setEnabled(True)

    def huitu2d3d(self):
        if self.ui.comboBox_TYPE.currentText()=='3D':
            self.ui.label_57.setEnabled(True)
            self.ui.horizontalSlider_fangwei.setEnabled(True)
            self.ui.label_58.setEnabled(True)
            self.ui.label_55.setEnabled(True)
            self.ui.horizontalSlider_fuyang.setEnabled(True)
            self.ui.label_56.setEnabled(True)
            self.ui.checkBox.setEnabled(True)
            self.ui.label_30.setEnabled(True)
            self.ui.label_31.setEnabled(True)
            self.ui.comboBox_julilabel.setEnabled(True)
            self.ui.comboBox_6.setEnabled(True)
            self.ui.checkBox_2.setEnabled(True)
        else:
            self.ui.label_57.setEnabled(False)
            self.ui.horizontalSlider_fangwei.setEnabled(False)
            self.ui.label_58.setEnabled(False)
            self.ui.label_55.setEnabled(False)
            self.ui.horizontalSlider_fuyang.setEnabled(False)
            self.ui.label_56.setEnabled(False)
            self.ui.checkBox.setEnabled(False)
            self.ui.label_30.setEnabled(False)
            self.ui.label_31.setEnabled(False)
            self.ui.comboBox_julilabel.setEnabled(False)
            self.ui.comboBox_6.setEnabled(False)
            self.ui.checkBox_2.setEnabled(False)

    def on_state_changed(self, state):
        if state == Qt.Checked:
            self.ui.comboBox_julilabel.setCurrentText("2")
            self.ui.comboBox_6.setCurrentText("2")
        else:
            self.ui.comboBox_julilabel.setCurrentText("-8")
            self.ui.comboBox_6.setCurrentText("-11")

    def label_56(self):
        self.ui.label_56.setText(str(self.ui.horizontalSlider_fuyang.value()))

    def label_58(self):
        self.ui.label_58.setText(str(self.ui.horizontalSlider_fangwei.value()))

    def roi_centerline2(self):
        if self.gui_roicenter2==0:
            self.roicenterline2=RoiCenterline2()
            self.roicenterline2.ui.show()

        self.roicenterline2.ratio1 = 0.995
        self.roicenterline2.ratio2 = 0.995
        self.roicenterline2.graphic_scene1.setSceneRect(0, 0, int(self.roicenterline2.ratio2 *self.roicenterline2.ui.graphicsView1.width()),
                                         int(self.roicenterline2.ratio1 * self.roicenterline2.ui.graphicsView1.height()))
        self.roicenterline2.canvas1.resize(int(self.roicenterline2.ratio2 * self.roicenterline2.ui.graphicsView1.width()),
                            int(self.roicenterline2.ratio1 * self.roicenterline2.ui.graphicsView1.height()))

        self.fig5=self.roicenterline2.canvas1.fig
        self.ax5=self.roicenterline2.canvas1.axes
        self.roicenterimg3 = self.roicenterimg.copy()
        self.roicenterimg3[:,int(0.5 * self.roicenterimg3.shape[1]),  :] = 0
        if self.roicenterimg3.shape[2] == 3:
            self.roicevterimg3_screen = cv.cvtColor(self.roicenterimg3, cv.COLOR_BGR2RGB)
        elif self.roicenterimg3.shape[2] == 4:
            self.roicevterimg3_screen = cv.cvtColor(self.roicenterimg3, cv.COLOR_BGRA2RGBA)
        self.roicenterline2.canvas1.axes0.imshow(self.roicevterimg3_screen)

        self.ax5.cla()
        self.ax5.invert_yaxis()
        self.ax5.plot(self.thread_start2D.middieline2,self.thread_start2D.xdata2)

        self.ax5.set_ylabel('Pixels')
        if self.ui.lineEdit_wuzhi.text()=='':
            self.ax5.set_xlabel(self.ui.comboBox_colorsee.currentText())
        else:
            self.ax5.set_xlabel(self.ui.lineEdit_wuzhi.text()+' '+self.ui.lineEdit_danwei.text())
        self.ax5.xaxis.set_label_position('top')
        self.ax5.grid()
        self.fig5.tight_layout()
        self.roicenterline2.canvas1.draw()
        self.roicenterline2.canvas1.flush_events()
        self.switch_show_roicenterline2=1

    def roi_centerline(self):
        if self.gui_roicenter==0:
            self.roicenterline=RoiCenterline()
            self.roicenterline.ui.show()

        self.roicenterline.ratio1 = 0.995
        self.roicenterline.ratio2 = 0.995
        self.roicenterline.graphic_scene1.setSceneRect(0, 0, int(self.roicenterline.ratio2 *self.roicenterline.ui.graphicsView1.width()),
                                         int(self.roicenterline.ratio1 * self.roicenterline.ui.graphicsView1.height()))
        self.roicenterline.canvas1.resize(int(self.roicenterline.ratio2 * self.roicenterline.ui.graphicsView1.width()),
                            int(self.roicenterline.ratio1 * self.roicenterline.ui.graphicsView1.height()))

        self.fig4=self.roicenterline.canvas1.fig
        self.ax4=self.roicenterline.canvas1.axes
        self.roicenterimg2 = self.roicenterimg.copy()
        self.roicenterimg2[int(0.5 * self.roicenterimg2.shape[0]), :, :] = 0
        if self.roicenterimg2.shape[2] == 3:
            self.roicevterimg2_screen = cv.cvtColor(self.roicenterimg2, cv.COLOR_BGR2RGB)
        elif self.roicenterimg2.shape[2] == 4:
            self.roicevterimg2_screen = cv.cvtColor(self.roicenterimg2, cv.COLOR_BGRA2RGBA)
        self.roicenterline.canvas1.axes0.imshow(self.roicevterimg2_screen)

        self.ax4.cla()
        self.ax4.plot(self.thread_start2D.xdata,self.thread_start2D.middieline)
        self.ax4.grid()
        self.ax4.set_xlabel('Pixels')
        if self.ui.lineEdit_wuzhi.text()=='':
            self.ax4.set_ylabel(self.ui.comboBox_colorsee.currentText())
        else:
            self.ax4.set_ylabel(self.ui.lineEdit_wuzhi.text()+' '+self.ui.lineEdit_danwei.text())
        self.fig4.tight_layout()
        self.roicenterline.canvas1.draw()
        self.roicenterline.canvas1.flush_events()
        self.switch_show_roicenterline=1

    def on_mouse_move(self, event):
        try:
            if event.inaxes:
                if self.yuyan=='chinese':
                    self.ui.label_53.setText(f'坐标: (x:{event.xdata:.2f}, y:{event.ydata:.2f})')
                else:
                    self.ui.label_53.setText(f'Coordinate: (x:{event.xdata:.2f}, y:{event.ydata:.2f})')
        except:
            pass

    def on_mouse_move2(self, event):
        if event.inaxes:
            try:
                if self.ui.comboBox_TYPE.currentText()=='2D':
                    x1=event.xdata
                    y1=event.ydata
                    if self.yuyan=='chinese':
                        self.ui.label_53.setText(f'坐标: (x:{x1:.2f}, y:{y1:.2f}, value:{self.im_weicaise.get_array()[int(y1), int(x1)]:.2f})')
                    else:
                        self.ui.label_53.setText(f'Coordinate: (x:{x1:.2f}, y:{y1:.2f}, value:{self.im_weicaise.get_array()[int(y1), int(x1)]:.2f})')
                else:
                    self.ui.label_53.clear()
            except:
                pass

    def singleselect(self):
        if self.ui.radioButton_single.isChecked():
            self.clear_data()
            self.ui.label_12.setEnabled(False)
            self.ui.cishu.setEnabled(False)
            self.ui.label_19.setEnabled(False)
            self.ui.comboBox_caiyangjiange.setEnabled(False)

    def longselect(self):
        if self.ui.radioButton_longtime.isChecked():
            self.ui.label_12.setEnabled(True)
            self.ui.cishu.setEnabled(True)
            self.ui.label_19.setEnabled(True)
            self.ui.comboBox_caiyangjiange.setEnabled(True)

    def changelabel(self):
        self.ui.exposurelabel.setText(str(self.ui.exposureSlider.value()))

    def exptimeset(self):
        if self.ui.comboBox_exptime.currentText()=='自动曝光' or self.ui.comboBox_exptime.currentText()=='Auto':
            self.ui.exposureSlider.setValue(-6)
            self.ui.exposureSlider.setEnabled(False)
            self.ui.exposurelabel.setEnabled(False)
            self.ui.label_39.setEnabled(False)
            time.sleep(0.2)
        else:
            self.ui.exposureSlider.setEnabled(True)
            self.ui.exposurelabel.setEnabled(True)
            self.ui.label_39.setEnabled(True)

    def change_wb(self):
        self.ui.label_wb.setText(str(self.ui.horizontalSlider_wb.value()))
    def wbset(self):
        if self.ui.comboBox_4.currentText()=='自动白平衡' or self.ui.comboBox_4.currentText()=='Auto':
            self.ui.horizontalSlider_wb.setValue(5500)
            self.ui.label_41.setEnabled(False)
            self.ui.horizontalSlider_wb.setEnabled(False)
            self.ui.label_wb.setEnabled(False)
            time.sleep(0.2)
        else:
            self.ui.label_41.setEnabled(True)
            self.ui.horizontalSlider_wb.setEnabled(True)
            self.ui.label_wb.setEnabled(True)

    def wuzhi_textchange(self):
        if self.ui.lineEdit_wuzhi.text()=='':
            self.ui.label_14.setText(self.ui.comboBox_colorsee.currentText()+':')
        else:
            self.ui.label_14.setText(self.ui.lineEdit_wuzhi.text()+':')
    def danwei_change(self):
        self.ui.label_32.setText(self.ui.lineEdit_danwei.text())

    def start_getchemicalvideo(self):
        self.stop_2D()
        self.switch_startchemicalvideo = 1
        self.ui.actionchemicalvideostart.setEnabled(False)
        self.ui.actionchemicalvideostop.setEnabled(True)
        self.ui.statusbar.showMessage('')
        self.thread_takechemicalvideos = ThreadTakeChemicalVideos()
        self.threads.append(self.thread_takechemicalvideos)
        self.thread_takechemicalvideos.signal1.connect(self.select_video_folder)
        self.thread_takechemicalvideos.signal2.connect(self.set_video_parameters2)
        self.thread_takechemicalvideos.signal3.connect(self.show_end)
        self.thread_takechemicalvideos.start()

    def show_end(self):
        if self.yuyan=='chinese':
            self.ui.label_34.setText('化学图像视频录制结束！')
        else:
            self.ui.label_34.setText('The taking of chemical video was end！')

    def show_takechemicalvideo_process(self):
        if self.yuyan == 'english':
            self.ui.label_34.setText(
                f'Frame number:{self.chemicalvideoframenum}, Time:{round(self.getchemicalvideo_t_current, 3)}s')
        else:
            self.ui.label_34.setText(
                f'帧数：{self.chemicalvideoframenum}, 录制时间：{round(self.getchemicalvideo_t_current, 3)}s')

    def stop_getchemicalvideo(self):
        self.switch_startchemicalvideo=0
        self.ui.actionchemicalvideostart.setEnabled(True)
        self.ui.actionchemicalvideostop.setEnabled(False)

    def importvideo(self):
        self.picture_path0=''
        self.csv_path0=''
        self.closecamera()
        self.jietuok1 = False
        self.ui.menu_5.setEnabled(True)
        self.ui.radioButton_longtime.setEnabled(True)
        self.ui.radioButton_longtime.setChecked(True)
        self.ui.radioButton_single.setEnabled(False)
        time.sleep(1)
        if self.yuyan == 'chinese':
            self.video_importpath, ftype2 = QFileDialog.getOpenFileName(self.ui, "导入视频", "",
                                                                     "Videos (*.avi *.mp4)")
        else:
            self.video_importpath, ftype2 = QFileDialog.getOpenFileName(self.ui, "Import video", "",
                                                                     "Videos (*.avi *.mp4)")
        if self.video_importpath=='':
            return
        self.videocap=cv.VideoCapture(self.video_importpath)
        self.videocount=self.videocap.get(cv.CAP_PROP_FRAME_COUNT)
        self.ret,self.frame=self.videocap.read()
        self.videoscreen=self.change_imgbit(self.frame)
        self.ui.label_camera_img.clear()
        w_c = self.videoscreen.shape[1]
        h_c = self.videoscreen.shape[0]
        self.videoimage = QImage(self.videoscreen, w_c, h_c, w_c * 3, QImage.Format_RGB888).rgbSwapped()
        self.ui.label_camera_img.setPixmap(QPixmap.fromImage(self.videoimage))
        self.ui.label_camera_img.setScaledContents(True)
        if self.yuyan == 'english':
            self.ui.label_videoinfo.setText(
                f'Total frame count of the video:{self.videocount}')
        else:
            self.ui.label_videoinfo.setText(
                f'视频总帧数：{self.videocount}')
        self.select_roi()
        self.videocap.release()

    def stop_takevideos(self):
        self.swich_takevideo=0

    def take_videos(self):
        self.thread_takevideos = ThreadTakeVideos()
        self.threads.append(self.thread_takevideos)
        self.thread_takevideos.signal1.connect(self.select_video_folder)
        self.thread_takevideos.signal2.connect(self.set_video_parameters)
        self.thread_takevideos.signal0.connect(self.show_takevideo_process)
        self.thread_takevideos.start()

    def set_video_parameters(self):
        self.switch_set_video_parameters=0
        self.set_video=VideoParameterSet()
        self.set_video.ui.show()

    def set_video_parameters2(self):
        self.switch_set_video_parameters2=0
        self.set_video2=VideoParameterSet2()
        self.set_video2.ui.show()

    def set_video_false(self):
        self.ui.statusbar.showMessage('视频录制参数设置失败！只能输入数字，请重新输入！')

    def select_video_folder(self):
        self.switch_select_video_folder=0
        if self.yuyan == 'english':
            self.videooutpath = QFileDialog.getExistingDirectory(self.ui, "Please select the output folder")
        else:
            self.videooutpath = QFileDialog.getExistingDirectory(self.ui, "请选择输出文件夹")
        self.savepath_video = self.videooutpath + os.sep + 'video data'
        if not os.path.exists(self.savepath_video):
            os.mkdir(self.savepath_video)
        self.switch_select_video_folder = 1

    def show_takevideo_process(self):
        if self.yuyan == 'english':
            self.ui.statusbar.showMessage(f'Number of frame:{self.thread_takevideos.i}, Time:{round(self.thread_takevideos.t_current,3)}s')
        else:
            self.ui.statusbar.showMessage(f'帧数：{self.thread_takevideos.i}, 录制时间：{round(self.thread_takevideos.t_current,3)}s')

    def Hue_method_change(self):
        self.ui.label_45.setText('')
        if self.ui.lineEdit_wuzhi.text()=='':
            self.ui.label_14.setText(self.ui.comboBox_colorsee.currentText() + ':')
        if self.ui.comboBox_colorsee.currentText()!='Huev':
            self.ui.horizontalSlider_value.setEnabled(False)
            self.ui.lineEdit_yuzhi.setEnabled(False)
            self.ui.label_13.setEnabled(False)
            self.ui.label_yanseyuzhi.setEnabled(False)
            self.ui.label_38.setEnabled(False)
        elif self.ui.comboBox_colorsee.currentText()=='Huev':
            self.ui.horizontalSlider_value.setEnabled(True)
            self.ui.lineEdit_yuzhi.setEnabled(True)
            self.ui.label_13.setEnabled(True)
            self.ui.label_yanseyuzhi.setEnabled(True)
            self.ui.label_38.setEnabled(True)

    def paizhaopath(self):
        if self.yuyan == 'english':
            self.ui.statusbar.showMessage('Attention: The file path of saving images needs to be all English characters ')
        else:
            self.ui.statusbar.showMessage('注意：拍照保存图片时，文件路径全是英文字符才可以保存成功')
        self.ui.pushButton_paizhaopath.setEnabled(False)
        if self.yuyan == 'english':
            self.savepath02 = QFileDialog.getExistingDirectory(self.ui, "Please select the output folder")
        else:
            self.savepath02 = QFileDialog.getExistingDirectory(self.ui, "请选择输出文件夹")
        if self.savepath02 == '':
            self.ui.lineEdit_paizhaopath.setText(self.savepath02)
            self.ui.pushButton_paizhaopath.setEnabled(True)
            return
        self.savepath_paizhao = self.savepath02 + os.sep + 'camera_image data'
        if not os.path.exists(self.savepath_paizhao):
            os.mkdir(self.savepath_paizhao)
        self.ui.lineEdit_paizhaopath.setText(self.savepath02 + '/camera_image data')
        self.ui.pushButton_paizhaopath.setEnabled(True)

    def closeEvent(self, event):
        # 弹出确认对话框
        if self.yuyan=='english':
            reply = QMessageBox.question(self, 'Quit', 'Are you sure to quit？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        else:
            reply = QMessageBox.question(self, '退出', '您确定要退出吗？',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            for thread in self.threads:
                if thread.isRunning():
                    thread.terminate()
            try:
                plt.close('all')
            except:
                pass
            try:
                self.thread_opencamera.cap.release()
            except:
                pass
            try:
                self.roicenterline.ui.close()
            except:
                pass
            try:
                self.roicenterline2.ui.close()
            except:
                pass
            event.accept()
        else:
            event.ignore()

    def zitijiacu(self):
        if self.ui.radioButton_jiacu.isChecked():
            mpl.rcParams.update({"font.weight":'bold',"axes.labelweight":'bold'})
        else:
            mpl.rcParams.update({"font.weight":'normal',"axes.labelweight":'normal'})

    def zihao(self):
        mpl.rcParams.update({'font.size': eval(self.ui.comboBox_2_zihao.currentText())})
    def ziti(self):
        try:
            mpl.rcParams.update({'font.sans-serif': self.ui.comboBox_ziti.currentText()})
        except:
            pass

    def opencsv(self):
        try:
            self.picture_path0=''
            self.video_importpath=''
            self.closecamera()
            self.jietuok1=False
            self.ui.menu_3.setEnabled(False)
            self.ui.menu_5.setEnabled(False)
            self.ui.radioButton_single.setChecked(True)
            self.ui.radioButton_single.setEnabled(False)
            self.ui.radioButton_longtime.setEnabled(False)
            self.ui.button_start.setEnabled(False)
            if self.yuyan=='english':
                self.csv_path0, ftype_csv = QFileDialog.getOpenFileName(self.ui, "Import csv image", "",
                                                                        "Grayscale image (*.csv)")
            else:
                self.csv_path0, ftype_csv = QFileDialog.getOpenFileName(self.ui, "导入csv灰度图像", "",
                                                                         "灰度图像 (*.csv)")
            self.csvimg = np.loadtxt(self.csv_path0, delimiter=",", dtype=float)
            n = self.csvimg.shape
            self.showcsv()
        except:
            if self.yuyan == 'english':
                self.msg_boxcsv = QMessageBox(QMessageBox.Information, 'Import 2D matrix (.csv)',
                                              'Failed to import 2D matrix (.csv). Please check if the file is selected!'
                                              '\nThe file path must be all English characters in order to be opened!'
                                              '\nThe file needs to be a pure two-dimensional digital matrix that stored in the format of csv!')
            else:
                self.msg_boxcsv = QMessageBox(QMessageBox.Information, '导入二维矩阵(.csv)',
                                             '导入失败，请检查是否选中文件！文件路径需要全是英文字符才可以打开！文件需要是以csv格式存储的纯二维数字矩阵！')
            self.msg_boxcsv.exec_()

    def setyuzhi(self):
        try:
            yuzhiwenben=eval(self.ui.lineEdit_yuzhi.text())
            if 5<=yuzhiwenben<=365:
                self.ui.horizontalSlider_value.setValue(10000*yuzhiwenben)
        except:
            pass

    def closecamera(self):
        self.camopen=False

    def paizhao(self):
        self.ui.pushButton_paizhao.setEnabled(False)
        self.ui.statusbar.showMessage('')
        try:
            if self.savepath02=='':
                raise
            self.t_save2 = time.strftime('%H-%M-%S', time.localtime())
            cv.imwrite(self.savepath_paizhao + os.sep + self.t_save2 + f'-camera_img_{self.ui.lineEdit_zibianliang.text()}.tiff', self.frame)
            cv.imwrite(self.savepath_paizhao + os.sep + self.t_save2+ f'-roi_img_{self.ui.lineEdit_zibianliang.text()}.tiff', self.roiimg)
        except:
            if self.yuyan == 'english':
                self.msg_box_path = QMessageBox(QMessageBox.Information, 'Save photos',
                                             'Failed to take photos because the file path of saving photos has not been set!'
                                             )
            else:
                self.msg_box_path = QMessageBox(QMessageBox.Information, '保存照片',
                                             '拍照失败，未设置保存图像的文件路径！'
                                             )
            self.msg_box_path.exec_()
        self.ui.pushButton_paizhao.setEnabled(True)

    def pillow_to_cv2(self,pil_image):
        np_image = np.array(pil_image)
        channel_num=len(pil_image.getbands())
        if channel_num == 4:
            bgra = cv.cvtColor(np_image, cv.COLOR_RGBA2BGRA)
            return bgra
        elif channel_num == 3:
            bgr = cv.cvtColor(np_image, cv.COLOR_RGB2BGR)
            return bgr
        elif channel_num == 1:
            return np_image
        else:
            raise ValueError(f"不支持的图像模式：{pil_image.mode}")

    def open_picture(self):
        try:
            self.ui.label_videoinfo.clear()
            self.csv_path0=''
            self.video_importpath = ''
            self.closecamera()
            self.jietuok1 = False
            self.ui.menu_3.setEnabled(False)
            self.ui.menu_5.setEnabled(False)
            self.ui.radioButton_longtime.setEnabled(False)
            self.ui.radioButton_single.setEnabled(True)
            self.ui.radioButton_single.setChecked(True)
            time.sleep(1)
            if self.yuyan=='chinese':
                self.picture_path0, ftype2 = QFileDialog.getOpenFileName(self.ui, "导入图片", "", "Images (*.tif *.tiff *.jpg *.jpeg *.png *.bmp)")
            else:
                self.picture_path0, ftype2 = QFileDialog.getOpenFileName(self.ui, "Import image", "",
                                                                         "Images (*.tif *.tiff *.jpg *.jpeg *.png *.bmp)")

            self.pilimg = Image.open(self.picture_path0)
            self.frame_0=self.pillow_to_cv2(self.pilimg)
            xuanzhuan = eval(self.ui.comboBox_xuanzhuan.currentText()) / 90
            self.frame_1 = np.rot90(self.frame_0, xuanzhuan)
            if self.ui.comboBox_10.currentText() != 'no set':
                fanzhuan = eval(self.ui.comboBox_10.currentText()[:2])
                self.frame = cv.flip(self.frame_1, fanzhuan)
            else:
                self.frame = self.frame_1
            # self.frame=cv.imread(self.picture_path0,-1)
            n=self.frame.shape
            self.label_putimg(self.ui.label_camera_img, self.picture_path0)
        except:
            if self.yuyan == 'english':
                self.msg_box12 = QMessageBox(QMessageBox.Information, 'Import image',
                                             'Failed to import image from the computer. Maybe the image file was not selected or this format of image cannot be opened!'
                                             )
            else:
                self.msg_box12 = QMessageBox(QMessageBox.Information, '导入图片',
                                         '导入图片失败，可能是由于图像文件未被选中或者不支持此格式的图像!'
                                         )
            self.msg_box12.exec_()
            return
        self.select_roi()

    def input_biaoqufx(self):
        try:
            if self.yuyan == 'english':
                self.txtpath0, ftype = QFileDialog.getOpenFileName(self.ui, "Import standard curve", "", "Txt (*.txt)")
            else:
                self.txtpath0,ftype =QFileDialog.getOpenFileName(self.ui, "添加标曲", "", "Txt (*.txt)")
            with open(self.txtpath0, 'r', encoding='utf-8') as self.f:
                self.biaoqu = self.f.readline()
                self.ui.lineEdit_biaoqu.clear()
                self.ui.lineEdit_biaoqu.setText(self.biaoqu)
        except:
            if self.yuyan == 'english':
                self.msg_box11 = QMessageBox(QMessageBox.Information, 'Import standard curve',
                                             'Failed to import the standard curve. Please check if the required file is selected！'
                                             )
                self.msg_box11.exec_()
            else:
                self.msg_box11 = QMessageBox(QMessageBox.Information, '添加标曲',
                                            '添加标曲失败，请检查是否选中文件！'
                                        )
                self.msg_box11.exec_()

    def change_imgbit(self,img):
        img_yuanshi=img.copy()
        if img_yuanshi.ndim==2:
            img_yuanshi=cv.cvtColor(img_yuanshi,cv.COLOR_GRAY2BGR)
        if str(img_yuanshi.dtype)=='uint16':
            img_screen0=img_yuanshi/256.0
            img_screen=img_screen0.astype(np.uint8)
        elif str(img_yuanshi.dtype)=='uint32':
            img_screen0=img_yuanshi/16777216.0
            img_screen=img_screen0.astype(np.uint8)
        elif str(img_yuanshi.dtype)=='uint8':
            img_screen=img_yuanshi
        elif str(img_yuanshi.dtype)=='uint12':
            img_screen0=img_yuanshi/16.0
            img_screen = img_screen0.astype(np.uint8)
        elif str(img_yuanshi.dtype)=='uint14':
            img_screen0=img_yuanshi/64.0
            img_screen = img_screen0.astype(np.uint8)
        return img_screen

    def label_putimg(self,label,img_path):
        pix = QPixmap(img_path)
        label.setPixmap(pix)
        label.setScaledContents(True)

    def open_camera(self):
        self.jietuok1 = False
        self.reselect()
        self.stop_long()
        self.stop_2D()
        time.sleep(0.3)
        self.ui.label_videoinfo.clear()
        self.ui.radioButton_single.setEnabled(True)
        self.ui.radioButton_longtime.setEnabled(True)
        self.ui.radioButton_single.setChecked(True)
        self.thread_opencamera = ThreadOpencamera()
        self.threads.append(self.thread_opencamera)
        self.thread_opencamera.signal0.connect(self.update_label_img)
        self.thread_opencamera.signal1.connect(self.camera_fail)
        self.thread_opencamera.signal2.connect(self.show_autowb)
        self.thread_opencamera.signal4.connect(self.fenbianlvset_info)
        self.thread_opencamera.signal5.connect(self.show_wb_exp_info)
        self.thread_opencamera.signal6.connect(self.del_wb_exp_info)
        self.thread_opencamera.start()

    def show_wb_exp_info(self):
        if self.yuyan == 'english':
            self.ui.label_46.setText('Exposure time and white balance of some cameras can be set, some cannot.')
        else:
            self.ui.label_46.setText('有的相机允许设置曝光与白平衡，有的不允许')

    def del_wb_exp_info(self):
        self.ui.label_46.clear()

    def fenbianlvset_info(self):
        if self.yuyan == 'english':
            self.ui.statusbar.showMessage('If camera supports this resolution, the setting will take effect！')
        else:
            self.ui.statusbar.showMessage('相机支持此分辨率，设置方可生效！')

    def show_autowb(self):
        if self.yuyan == 'english':
            self.ui.statusbar.showMessage('Manual white balance and manual exposure settings failed！')
        else:
            self.ui.statusbar.showMessage('手动白平衡与手动曝光设置失败！')

    def update_label_img(self):
        self.camshow_complete = 0
        self.ui.label_camera_img.clear()
        w_c=self.thread_opencamera.camera_screen.shape[1]
        h_c=self.thread_opencamera.camera_screen.shape[0]
        self.qimage = QImage(self.thread_opencamera.camera_screen, w_c, h_c,w_c*3,QImage.Format_RGB888).rgbSwapped()
        self.ui.label_camera_img.setPixmap(QPixmap.fromImage(self.qimage))
        self.ui.label_camera_img.setScaledContents(True)
        self.camshow_complete=1

    def camera_fail(self):
        if self.yuyan=='english':
            self.msg_box1 = QMessageBox(QMessageBox.Information, 'Camera information',
                                        f'Camera {self.thread_opencamera.CAMERA_NUM} failed to read')
        else:
            self.msg_box1 = QMessageBox(QMessageBox.Information, '相机信息', f'{self.thread_opencamera.CAMERA_NUM}号相机读取失败')
        self.msg_box1.exec_()

    def changelabel2(self):
        self.ui.label_yanseyuzhi.setText(f'{self.ui.horizontalSlider_value.value()/10000}')

    def select_roi(self):
        self.ui.radioButton_roisum.setEnabled(False)
        self.ui.label_50.setEnabled(False)
        if self.yuyan == 'english':
            self.ui.statusbar.showMessage('Some quantitative parameters can only be extraxted from 8-bit images')
        else:
            self.ui.statusbar.showMessage('部分定量参数只能从8位图像中提取')
        self.jietuok1 = False
        self.ui.button_roi.setEnabled(False)
        time.sleep(1)
        self.ui.button_reselect_roi.setEnabled(True)
        self.thread_cutimg = ThreadCutimg()
        self.threads.append(self.thread_cutimg)
        self.thread_cutimg.signal0.connect(self.cut_img)
        self.thread_cutimg.start()
        self.thread_showroi = ThreadShowroi()
        self.threads.append(self.thread_showroi)
        self.thread_showroi.signal0.connect(self.update_label_roiimg)
        self.thread_showroi.start()

    def showcsv(self):
        self.figcsv=plt.figure(self.csv_path0)
        self.figcsv.clf()
        ax_csv = self.figcsv.add_subplot(111)
        ax_csv.imshow(self.csvimg,cmap='gray')
        ax_csv.set_title(f'{self.csvimg.shape[0]} x {self.csvimg.shape[1]}')
        ax_csv.axis('off')
        plt.tight_layout()
        self.figcsv.canvas.draw()
        self.figcsv.canvas.flush_events()
        self.figcsv.show()
        self.ui.button_start2D.setEnabled(True)

    def on_mouse(self, event, x, y, flags, param):
        if not self.ui.radioButton_roisum.isChecked():
            if event == cv.EVENT_LBUTTONDOWN:  # 左键点击
                self.clone=self.framecopy.copy()
                self.point1 = (x, y)
                cv.circle(self.clone, self.point1, 10, (0, 255, 0), 10)
                cv.imshow('Please select ROI', self.clone)
            elif event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
                self.clone = self.framecopy.copy()
                cv.rectangle(self.clone, self.point1, (x, y), (255, 0, 0), 10)
                cv.imshow('Please select ROI', self.clone)
            elif event == cv.EVENT_LBUTTONUP:  # 左键释放
                self.clone = self.framecopy.copy()
                self.point2 = (x, y)
                cv.rectangle(self.clone, self.point1, self.point2, (0, 0, 255), 10)
                cv.imshow('Please select ROI', self.clone)
                try:
                    self.min_x1 = min(self.point1[0], self.point2[0])
                    self.min_y1 = min(self.point1[1], self.point2[1])
                    self.width1 = abs(self.point1[0] - self.point2[0])
                    self.height1 = abs(self.point1[1] - self.point2[1])
                    del self.point1,self.point2
                except:
                    pass
        else:
            if event == cv.EVENT_LBUTTONDOWN:  # 左键点击
                self.clone=self.framecopy.copy()
                self.point1 = (x, y)
                cv.circle(self.clone, self.point1, 10, (0, 255, 0), 5)
                cv.imshow('Please select ROI', self.clone)
            elif event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
                self.clone = self.framecopy.copy()
                cv.rectangle(self.clone, self.point1, (x, y), (255, 0, 0), 5)
                cv.imshow('Please select ROI', self.clone)
            elif event == cv.EVENT_LBUTTONUP:  # 左键释放
                self.point2 = (x, y)
                if self.point2 != self.point1:
                    cv.rectangle(self.framecopy, self.point1, self.point2, (0, 0, 255), 5)
                    cv.imshow('Please select ROI', self.framecopy)
                    try:
                        self.min_x1 = min(self.point1[0], self.point2[0])
                        self.min_y1 = min(self.point1[1], self.point2[1])
                        self.width1 = abs(self.point1[0] - self.point2[0])
                        self.height1 = abs(self.point1[1] - self.point2[1])
                        del self.point1,self.point2
                        self.list_roi.append((self.min_x1,self.min_y1,self.width1,self.height1))
                    except:
                        pass
            if len(self.list_roi)==0:
                print('listroi=0')
                self.list_roi.append((0,0,self.framecopy.shape[1],self.framecopy.shape[0]))

    def cut_img(self):
        self.list_roi = []
        self.min_x1 = 'l'
        self.min_y1 = 'l'
        self.width1 = 'l'
        self.height1 = 'l'
        self.framecopy = self.frame.copy()
        cv.namedWindow('Please select ROI', cv.WINDOW_NORMAL)
        cv.imshow('Please select ROI', self.framecopy)
        cv.setMouseCallback('Please select ROI', self.on_mouse)
        cv.waitKey(0)
        cv.destroyAllWindows()
        self.jietuok1=True

    def update_label_roiimg(self):
        self.roishow_complete = 0
        self.ui.label_roi.clear()
        self.roicenterimg=self.thread_showroi.roi_screen
        w=self.thread_showroi.roi_screen.shape[1]
        h= self.thread_showroi.roi_screen.shape[0]
        if self.thread_showroi.roi_screen.shape[2]==3:
            self.qimage2 = QImage(self.thread_showroi.roi_screen, w, h,w*3, QImage.Format_RGB888).rgbSwapped()
        elif self.thread_showroi.roi_screen.shape[2]==4:
            self.qimage2 = QImage(self.thread_showroi.roi_screen, w, h,w*4, QImage.Format_RGBA8888).rgbSwapped()
        self.ui.label_roi.setPixmap(QPixmap.fromImage(self.qimage2))
        self.ui.label_roi.setScaledContents(True)
        self.ui.label_roi_info.clear()
        if not self.ui.radioButton_roisum.isChecked():
            self.ui.label_roi_info.setText(f" R: {self.thread_showroi.rvalue}\n"
                                           f" G: {self.thread_showroi.gvalue}\n"
                                           f" B: {self.thread_showroi.bvalue}\n"
                                           f" {self.thread_showroi.signalname}: {self.thread_showroi.grayvalue}\n"
                                           f" Bit depth: {self.thread_showroi.wei}\n"
                                           f" Shape: {self.thread_showroi.shape}")
        if self.thread_showroi.wei == 'uint8':
            self.disable_item_comboBox(self.ui.comboBox_colorsee, list(range(11,self.ui.comboBox_colorsee.count()+1)), 1 | 32)
        else:
            self.disable_item_comboBox(self.ui.comboBox_colorsee, list(range(11,self.ui.comboBox_colorsee.count()+1)), 0)
        self.roishow_complete=1

    def disable_item_comboBox(self, cBox, List, v=0):
        for i in range(len(List)):
            index = cBox.model().index(List[i], 0)
            cBox.model().setData(index, v, Qt.UserRole - 1)

    def reselect(self):
        self.jietuok1=False
        self.ui.button_reselect_roi.setEnabled(False)
        self.ui.button_roi.setEnabled(True)
        self.ui.button_start.setEnabled(False)
        self.ui.button_start2D.setEnabled(False)
        self.ui.button_addpoint.setEnabled(False)

    def start(self):
        self.ui.label_35.clear()
        if self.ui.radioButton_single.isChecked()==True:
            self.ui.button_start.setEnabled(False)
            self.thread_single = ThreadSingle()
            self.threads.append(self.thread_single)
            self.thread_single.signal0.connect(self.biaoqu_erro)
            self.thread_single.signal1.connect(self.update_fig0)
            self.thread_single.start()
        elif self.ui.radioButton_longtime.isChecked()==True:
            self.ui.button_start.setEnabled(False)
            self.longok=True
            self.thread_longtime = ThreadLongtime()
            self.threads.append(self.thread_longtime)
            self.thread_longtime.signal0.connect(self.biaoqu_erro)
            self.thread_longtime.signal1.connect(self.update_fig1)
            self.thread_longtime.signal2.connect(self.cyclenum_false)
            self.thread_longtime.signal3.connect(self.showvideo)
            self.thread_longtime.start()

    def showvideo(self):
        if self.showvideo_complete==0:
            return
        self.showvideo_complete=0
        self.videoscreen = self.change_imgbit(self.frame)
        self.ui.label_camera_img.clear()
        w_c = self.videoscreen.shape[1]
        h_c = self.videoscreen.shape[0]
        if self.videoscreen.shape[2]==3:
            self.videoimage = QImage(self.videoscreen, w_c, h_c, w_c * 3, QImage.Format_RGB888).rgbSwapped()
        elif self.videoscreen.shape[2]==4:
            self.videoimage = QImage(self.videoscreen, w_c, h_c, w_c * 4, QImage.Format_RGBA8888).rgbSwapped()
        # self.videoimage = QImage(self.videoscreen, w_c, h_c, w_c * 3, QImage.Format_RGB888).rgbSwapped()
        self.ui.label_camera_img.setPixmap(QPixmap.fromImage(self.videoimage))
        self.ui.label_camera_img.setScaledContents(True)
        if self.yuyan == 'english':
            self.ui.label_videoinfo.setText(
                f'Total frame count:{self.thread_longtime.total_frame_count}，Current frame count:{self.thread_longtime.videonum}')
        else:
            self.ui.label_videoinfo.setText(
                f'视频总帧数：{self.thread_longtime.total_frame_count}，当前图像是第{self.thread_longtime.videonum}帧')
        self.showvideo_complete = 1
    def cyclenum_false(self):
        if self.yuyan=='english':
            self.msg_box_cyclenum = QMessageBox(QMessageBox.Information, 'Information on measurement frequency',
                                                'The number of measurements can only be entered as a positive integer.')
        else:
            self.msg_box_cyclenum = QMessageBox(QMessageBox.Information, '连续监测次数信息',
                                            '连续监测次数只能输入正整数数字！')
        self.msg_box_cyclenum.exec_()

    def start_2D(self):
        self.ui.statusbar.showMessage('')
        self.ui.label_34.clear()
        self.ui.label_36.clear()
        # self.ui.button_start.setEnabled(False)
        self.ui.button_start2D.setEnabled(False)
        self.chemicalimgok=True
        self.thread_start2D = ThreadStart2D()
        self.threads.append(self.thread_start2D)
        self.thread_start2D.signal1.connect(self.chemicalimg)
        self.thread_start2D.signal0.connect(self.biaoqu_erro)
        self.thread_start2D.signal2.connect(self.nan_in_juzhen_erro)
        self.thread_start2D.signal3.connect(self.status_info)
        self.thread_start2D.signal4.connect(self.showvideo2)
        self.thread_start2D.signal5.connect(self.roi_centerline)
        self.thread_start2D.signal6.connect(self.roi_centerline2)
        self.thread_start2D.signal7.connect(self.close_jindutiao)
        self.thread_start2D.start()

    def showvideo2(self):
        if self.showvideo_complete==0:
            return
        self.showvideo_complete=0
        self.videoscreen = self.change_imgbit(self.frame)
        self.ui.label_camera_img.clear()
        w_c = self.videoscreen.shape[1]
        h_c = self.videoscreen.shape[0]
        if self.videoscreen.shape[2]==3:
            self.videoimage = QImage(self.videoscreen, w_c, h_c, w_c * 3, QImage.Format_RGB888).rgbSwapped()
        elif self.videoscreen.shape[2]==4:
            self.videoimage = QImage(self.videoscreen, w_c, h_c, w_c * 4, QImage.Format_RGBA8888).rgbSwapped()
        # self.videoimage = QImage(self.videoscreen, w_c, h_c, w_c * 3, QImage.Format_RGB888).rgbSwapped()
        self.ui.label_camera_img.setPixmap(QPixmap.fromImage(self.videoimage))
        self.ui.label_camera_img.setScaledContents(True)
        if self.yuyan=='english':
            self.ui.label_videoinfo.setText(
                f'Total frame count:{self.thread_start2D.total_frame_count}，Current frame count:{self.thread_start2D.videonum}')
        else:
            self.ui.label_videoinfo.setText(f'视频总帧数：{self.thread_start2D.total_frame_count}，当前图像是第{self.thread_start2D.videonum}帧')
        self.showvideo_complete = 1

    def status_info(self):
        if self.video_importpath !='' or self.camopen == True:
            return
        else:
            if self.yuyan == 'english':
                if self.thread_start2D.jd==1:
                    self.progressDialog = QProgressDialog("Segmented calibration...", "Cancel", 0, 100)
                    self.progressDialog.setWindowTitle('Loading')
                    self.progressDialog.setWindowModality(Qt.NonModal)
                    self.progressDialog.show()
                    self.thread_start2D.jd=2
                    self.thread_start2D.jindu = 0
                self.progressDialog.setValue(self.thread_start2D.jindu)
            else:
                if self.thread_start2D.jd == 1:
                    self.progressDialog = QProgressDialog("分段校准...", "取消", 0, 100)
                    self.progressDialog.setWindowTitle('进程')
                    self.progressDialog.setWindowModality(Qt.NonModal)
                    self.progressDialog.show()
                    self.thread_start2D.jd =2
                    self.thread_start2D.jindu=0
                self.progressDialog.setValue(self.thread_start2D.jindu)

    def close_jindutiao(self):
        time.sleep(2)
        try:
            self.progressDialog.close()
        except:
            pass
    def nan_in_juzhen_erro(self):
        if self.yuyan == 'english':
            self.msg_box5 = QMessageBox(QMessageBox.Information, '2D chemical image information',
                                        f'There are empty values in the analyte concentration matrix, unable to plot!\nPossible reason: The hue of individual points in the image are not within the range of curve detection')
        else:
            self.msg_box5 = QMessageBox(QMessageBox.Information, '2D化学图像信息',
                                        f'分析物浓度矩阵中有空值，无法绘图！\n可能原因：图像中个别点的颜色值不在标曲检测范围内')
        self.msg_box5.exec_()

    def chemicalimg(self):
        self.ratio3 = 0.995
        self.ratio4 = 0.995
        self.graphic_scene2.setSceneRect(0, 0, int(self.ratio4 * self.ui.graphicsView2.width()),
                                         int(self.ratio3 * self.ui.graphicsView2.height()))
        self.canvas2.resize(int(self.ratio4 * self.ui.graphicsView2.width()),
                            int(self.ratio3 * self.ui.graphicsView2.height()))
        self.fig3 = self.canvas2.fig
        self.fig3.clf()
        if self.ui.comboBox_TYPE.currentText()=='2D':
            ax3 = self.fig3.add_subplot(111)
        else:
            ax3 = self.fig3.add_subplot(111,projection='3d')
            ax3.view_init(elev=self.ui.horizontalSlider_fuyang.value(), azim=self.ui.horizontalSlider_fangwei.value())
            ax3.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax3.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax3.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax3.tick_params(axis='both', which='both', length=0)
            if not self.ui.checkBox.isChecked():
                ax3.set_xticklabels([])
                ax3.set_yticklabels([])
                ax3.set_zticklabels([])
                ax3.tick_params(axis='both', which='both', length=0)
        self.canvas2.axes=ax3
        try:
            v_min = self.ui.lineEdit_vmin.text()
            v_max = self.ui.lineEdit_vmax.text()
            if v_min=='' or v_max=='':
                v_min=float(np.min(self.thread_start2D.y))
                v_max=float(np.max(self.thread_start2D.y))
            else:
                v_min = eval(self.ui.lineEdit_vmin.text())
                v_max = eval(self.ui.lineEdit_vmax.text())

            #colorbar 边界黑白设置
            initialcmap=self.ui.comboBox_colormap.currentText()
            if self.ui.lineEdit_vmin.text()=='' or self.ui.lineEdit_vmax.text()=='' or self.ui.comboBox_11.currentIndex()==0:
                newcmap=initialcmap
            elif self.ui.comboBox_11.currentIndex()==1:
                newcmap = plt.get_cmap(initialcmap).copy()
                newcmap.set_over('black')
            elif self.ui.comboBox_11.currentIndex()==2:
                newcmap = plt.get_cmap(initialcmap).copy()
                newcmap.set_over('white')
            elif self.ui.comboBox_11.currentIndex()==3:
                newcmap = plt.get_cmap(initialcmap).copy()
                newcmap.set_under('black')
            elif self.ui.comboBox_11.currentIndex()==4:
                newcmap = plt.get_cmap(initialcmap).copy()
                newcmap.set_under('white')
            elif self.ui.comboBox_11.currentIndex()==5:
                newcmap = plt.get_cmap(initialcmap).copy()
                newcmap.set_under('black')
                newcmap.set_over('white')
            elif self.ui.comboBox_11.currentIndex()==6:
                newcmap = plt.get_cmap(initialcmap).copy()
                newcmap.set_under('white')
                newcmap.set_over('black')

            if self.ui.comboBox_TYPE.currentText()=='2D':
                im=ax3.imshow(self.thread_start2D.y,
                           vmin=v_min, vmax=v_max,
                           cmap=newcmap,)
                if self.ui.checkBox_4.isChecked():
                    ax3.axis('on')
                    ax3.tick_params(axis='x', labelcolor='none', bottom=False)
                    ax3.tick_params(axis='y', labelcolor='none', left=False)
                else:
                    ax3.axis('off')
            else:
                x00 = np.arange(0, self.thread_start2D.y.shape[1])
                x=x00[::-1]
                y = np.arange(0, self.thread_start2D.y.shape[0])
                x, y = np.meshgrid(x, y)

                if self.ui.checkBox_2.isChecked():
                    ax3.set_box_aspect([self.thread_start2D.y.shape[1],self.thread_start2D.y.shape[0],min(self.thread_start2D.y.shape[1],self.thread_start2D.y.shape[0])])
                im = ax3.plot_surface(x, y, self.thread_start2D.y,vmin=v_min, vmax=v_max, cmap=newcmap)

                if not self.ui.checkBox.isChecked():
                    ax3.set_xlabel(f'Length\n({self.thread_start2D.y.shape[1]} pixels)', labelpad=eval(self.ui.comboBox_julilabel.currentText()))
                    ax3.set_ylabel(f'Width\n({self.thread_start2D.y.shape[0]} pixels)', labelpad=eval(self.ui.comboBox_6.currentText()))
                else:
                    ax3.set_xlabel('Pixels',
                                   labelpad=eval(self.ui.comboBox_julilabel.currentText()))
                    ax3.set_ylabel('Pixels',
                                   labelpad=eval(self.ui.comboBox_6.currentText()))
            self.im_weicaise=im
            meany=np.mean(self.thread_start2D.y)
            stdy=np.std(self.thread_start2D.y)
            if self.ui.checkBox_meanstd.isChecked():
                title=self.ui.lineEdit_title.text()+'\n'+f'{np.round(meany,3)} ± {np.round(stdy,3)}'
            else:
                title=self.ui.lineEdit_title.text()
            if title!='':
                if self.ui.radioButton_jiacu.isChecked():
                    fontweight='bold'
                else:
                    fontweight='normal'
                ax3.set_title(title,fontname=self.ui.comboBox_ziti.currentText(),fontweight=fontweight,pad=eval(self.ui.comboBox_7.currentText()))

            if self.yuyan == 'english':
                fangxiang = {'vertical': 'vertical', 'horizontal': 'horizontal'}
            else:
                fangxiang={'竖直':'vertical','水平':'horizontal'}
            tick_range = np.linspace(v_min, v_max, eval(self.ui.comboBox_ticknum.currentText()))
            cbpad=eval(self.ui.comboBox_julilabel_3.currentText())
            if self.ui.comboBox_3.currentText()!='no set':
                if self.ui.comboBox_8.currentText()!='no set':
                    cb = self.fig3.colorbar(im,ax=ax3,shrink=eval(self.ui.comboBox_barlength.currentText()),aspect=eval(self.ui.comboBox_3.currentText()),ticks=tick_range,
                                      orientation=fangxiang[self.ui.comboBox_fangxiang.currentText()],pad=cbpad,extend=self.ui.comboBox_8.currentText(),extendfrac=eval(self.ui.comboBox_9.currentText()))
                else:
                    cb = self.fig3.colorbar(im, ax=ax3, shrink=eval(self.ui.comboBox_barlength.currentText()),
                                            aspect=eval(self.ui.comboBox_3.currentText()), ticks=tick_range,
                                            orientation=fangxiang[self.ui.comboBox_fangxiang.currentText()], pad=cbpad,
                                            )
            else:
                if self.ui.comboBox_8.currentText() != 'no set':
                    cb = self.fig3.colorbar(im, ax=ax3, shrink=eval(self.ui.comboBox_barlength.currentText()),
                                      ticks=tick_range,
                                      orientation=fangxiang[self.ui.comboBox_fangxiang.currentText()], pad=cbpad,extend=self.ui.comboBox_8.currentText(),extendfrac=eval(self.ui.comboBox_9.currentText()))
                else:
                    cb = self.fig3.colorbar(im, ax=ax3, shrink=eval(self.ui.comboBox_barlength.currentText()),
                                            ticks=tick_range,
                                            orientation=fangxiang[self.ui.comboBox_fangxiang.currentText()], pad=cbpad,
                                            )

            if not self.ui.checkBox_cbbiankuang.isChecked():
                cb.outline.set_linewidth(0)
                cb.ax.tick_params(which='major', length=0)
            font_cb={'family':self.ui.comboBox_ziti.currentText()}
            if self.ui.lineEdit_wuzhi.text()=='':
                cb.set_label(f'{self.ui.comboBox_colorsee.currentText()}', fontdict=font_cb)
            else:
                cb.set_label(f'{self.ui.lineEdit_wuzhi.text()}'+f'{self.ui.lineEdit_danwei.text()}',fontdict=font_cb)
            num=eval(self.ui.comboBox_jingdu.currentText())
            cb.formatter = FormatStrFormatter(f'%.{num}f')  # 设置ticklabel保留几位小数(精度)
            for l in cb.ax.yaxis.get_ticklabels():
                l.set_family(self.ui.comboBox_ziti.currentText())
            for l1 in cb.ax.xaxis.get_ticklabels():
                l1.set_family(self.ui.comboBox_ziti.currentText())

            cb.update_ticks()
            self.fig3.tight_layout()
            self.canvas2.draw()
            self.canvas2.flush_events()

        except:
            self.chemicalimgok=False
            if self.yuyan == 'english':
                self.msg_box6 = QMessageBox(QMessageBox.Information, 'Chemical image information',
                                            f'There was an error during the drawing process. '
                                            f'\nPlease check if there are any unreasonable input information in the 2D drawing！')
            else:
                self.msg_box6 = QMessageBox(QMessageBox.Information, '化学图像信息',
                                            f'绘图过程出错，请检查绘图输入信息是否有不合理的地方！')
            self.msg_box6.exec_()

        try:
            self.fig3.savefig('.\chemical video img.jpg', dpi=eval(self.ui.dpi_box.currentText()), bbox_inches='tight', pad_inches=0.1)
            self.chemicalvideoframe=cv.imread('.\chemical video img.jpg', -1)
            self.startsignal = True
        except:
            pass

        if self.switch_startchemicalvideo == 1 and self.getchemicalvideo_t_current <= self.set_video2.timelong:

            self.chemicalvideoframenum += 1
            if self.chemicalvideoframenum==1:
                self.chemicalvideofourcc = cv.VideoWriter_fourcc(*'XVID')
                self.chemicalvideo_t_file = time.strftime('%H-%M-%S', time.localtime())
                self.chemicalvideoout = cv.VideoWriter(
                    color_chemistry.savepath_video + os.sep + f'{self.chemicalvideo_t_file}_ChemicalImageVideo_{color_chemistry.ui.lineEdit_zibianliang.text()}.avi',
                    self.chemicalvideofourcc,
                    self.set_video2.fps,
                    (self.chemicalvideoframe.shape[1], self.chemicalvideoframe.shape[0]))
                self.getchemicalvideo_t0 = time.perf_counter()
                self.rows = self.chemicalvideoframe.shape[0]
                self.cols = self.chemicalvideoframe.shape[1]
            self.chemicalvideoframe = cv.resize(self.chemicalvideoframe,(self.cols,self.rows))
            self.chemicalvideoout.write(self.chemicalvideoframe)
            t1 = time.perf_counter()
            self.getchemicalvideo_t_current = t1 - self.getchemicalvideo_t0
            self.chemicalvideotime_list.append(self.getchemicalvideo_t_current)
            self.show_takechemicalvideo_process()
        self.fig3_complete=1

    def stop_long(self):
        self.ui.button_stoplong.setEnabled(False)
        self.longok = False
        self.ui.button_stoplong.setEnabled(True)

    def stop_2D(self):
        self.ui.button_stop2D.setEnabled(False)
        self.chemicalimgok = False
        self.ui.button_stop2D.setEnabled(True)

    def colorsee_box(self,img):
        try:
            imgc=img.copy()
            if imgc.ndim==2:
                imgc=cv.cvtColor(imgc, cv.COLOR_GRAY2BGR)
            if imgc.shape[2]==4:
                imgc = imgc[:, :, :3]
            self.colorsee_boxname=self.ui.comboBox_colorsee.currentText()
            if self.colorsee_boxname=='Hue of HSV':
                # imgc=imgc.astype(np.uint8)
                hsv=cv.cvtColor(imgc, cv.COLOR_BGR2HSV)
                h=hsv[:,:,0]
            elif self.colorsee_boxname=='Huev':
                h=colorsee_variable(imgc,self.ui.horizontalSlider_value.value()/10000)
            elif self.colorsee_boxname=='R':
                b, g, r = cv.split(imgc)
                h=r
            elif self.colorsee_boxname == 'G':
                b, g, r = cv.split(imgc)
                h=g
            elif self.colorsee_boxname == 'B':
                b, g, r = cv.split(imgc)
                h=b
            elif self.colorsee_boxname == 'Gray':
                h=cv.cvtColor(imgc,cv.COLOR_BGR2GRAY)
            elif self.colorsee_boxname == 'S of HSV':
                hsv=cv.cvtColor(imgc, cv.COLOR_BGR2HSV)
                h=hsv[:,:,1]
            elif self.colorsee_boxname == 'V of HSV':
                hsv=cv.cvtColor(imgc, cv.COLOR_BGR2HSV)
                h=hsv[:,:,2]
            elif self.colorsee_boxname == 'R/G':
                b, g, r = cv.split(imgc)
                if np.any(g==0):
                    g[g==0]=1
                    if self.yuyan=='chinese':
                        self.ui.label_45.setText('G 矩阵中有0，0被替换为1')
                    else:
                        self.ui.label_45.setText('0 is in G matrix, 0 was replaced by 1')
                h=r/g
            elif self.colorsee_boxname == 'R/B':
                b, g, r = cv.split(imgc)
                if np.any(b==0):
                    b[b==0]=1
                    if self.yuyan=='chinese':
                        self.ui.label_45.setText('B 矩阵中有0，0被替换为1')
                    else:
                        self.ui.label_45.setText('0 is in B matrix, 0 was replaced by 1')
                h=r/b
            elif self.colorsee_boxname == 'G/B':
                b, g, r = cv.split(imgc)
                if np.any(b==0):
                    b[b==0]=1
                    if self.yuyan=='chinese':
                        self.ui.label_45.setText('B 矩阵中有0，0被替换为1')
                    else:
                        self.ui.label_45.setText('0 is in B matrix, 0 was replaced by 1')
                h=g/b
            elif self.colorsee_boxname == 'G/R':
                b, g, r = cv.split(imgc)
                if np.any(r==0):
                    r[r==0]=1
                    if self.yuyan=='chinese':
                        self.ui.label_45.setText('R 矩阵中有0，0被替换为1')
                    else:
                        self.ui.label_45.setText('0 is in R matrix, 0 was replaced by 1')
                h=g/r
            elif self.colorsee_boxname == 'B/R':
                b, g, r = cv.split(imgc)
                if np.any(r == 0):
                    r[r == 0] = 1
                    if self.yuyan == 'chinese':
                        self.ui.label_45.setText('R 矩阵中有0，0被替换为1')
                    else:
                        self.ui.label_45.setText('0 is in R matrix, 0 was replaced by 1')
                h = b / r
            elif self.colorsee_boxname == 'B/G':
                b, g, r = cv.split(imgc)
                if np.any(g == 0):
                    g[g == 0] = 1
                    if self.yuyan == 'chinese':
                        self.ui.label_45.setText('G 矩阵中有0，0被替换为1')
                    else:
                        self.ui.label_45.setText('0 is in G matrix, 0 was replaced by 1')
                h = b / g
            elif self.colorsee_boxname == 'L of Lab':
                lab = cv.cvtColor(imgc, cv.COLOR_BGR2Lab)
                h = lab[:,:,0]
            elif self.colorsee_boxname == 'a of Lab':
                lab = cv.cvtColor(imgc, cv.COLOR_BGR2Lab)
                h = lab[:,:,1]
            elif self.colorsee_boxname == 'b of Lab':
                lab = cv.cvtColor(imgc, cv.COLOR_BGR2Lab)
                h = lab[:,:,2]
            elif self.colorsee_boxname == 'Hue of HLS':
                hls = cv.cvtColor(imgc, cv.COLOR_BGR2HLS)
                h = hls[:,:,0]
            elif self.colorsee_boxname == 'L of HLS':
                hls = cv.cvtColor(imgc, cv.COLOR_BGR2HLS)
                h = hls[:,:,1]
            elif self.colorsee_boxname == 'S of HLS':
                hls = cv.cvtColor(imgc, cv.COLOR_BGR2HLS)
                h = hls[:,:,2]
            elif self.colorsee_boxname == 'L of Luv':
                luv = cv.cvtColor(imgc, cv.COLOR_BGR2Luv)
                h = luv[:, :, 0]
            elif self.colorsee_boxname == 'u of Luv':
                luv = cv.cvtColor(imgc, cv.COLOR_BGR2Luv)
                h = luv[:, :, 1]
            elif self.colorsee_boxname == 'v of Luv':
                luv = cv.cvtColor(imgc, cv.COLOR_BGR2Luv)
                h = luv[:, :, 2]
            elif self.colorsee_boxname == 'X of XYZ':
                xyz = cv.cvtColor(imgc, cv.COLOR_BGR2XYZ)
                h = xyz[:, :, 0]
            elif self.colorsee_boxname == 'Y of XYZ':
                xyz = cv.cvtColor(imgc, cv.COLOR_BGR2XYZ)
                h = xyz[:, :, 1]
            elif self.colorsee_boxname == 'Z of XYZ':
                xyz = cv.cvtColor(imgc, cv.COLOR_BGR2XYZ)
                h = xyz[:, :, 2]
            elif self.colorsee_boxname == 'Y of YCrCb':
                ycrcb = cv.cvtColor(imgc, cv.COLOR_BGR2YCrCb)
                h = ycrcb[:, :, 0]
            elif self.colorsee_boxname == 'Cr of YCrCb':
                ycrcb = cv.cvtColor(imgc, cv.COLOR_BGR2YCrCb)
                h = ycrcb[:, :, 1]
            elif self.colorsee_boxname == 'Cb of YCrCb':
                ycrcb = cv.cvtColor(imgc, cv.COLOR_BGR2YCrCb)
                h = ycrcb[:, :, 2]
            elif self.colorsee_boxname == 'Y of YUV':
                yuv = cv.cvtColor(imgc, cv.COLOR_BGR2YUV)
                h = yuv[:, :, 0]
            elif self.colorsee_boxname == 'U of YUV':
                yuv = cv.cvtColor(imgc, cv.COLOR_BGR2YUV)
                h = yuv[:, :, 1]
            elif self.colorsee_boxname == 'V of YUV':
                yuv = cv.cvtColor(imgc, cv.COLOR_BGR2YUV)
                h = yuv[:, :, 2]
            elif self.colorsee_boxname == 'C of CMYK':
                cmyk = bgr2cmyk(imgc)
                h = cmyk[:, :, 0]
            elif self.colorsee_boxname == 'M of CMYK':
                cmyk = bgr2cmyk(imgc)
                h = cmyk[:, :, 1]
            elif self.colorsee_boxname == 'Y of CMYK':
                cmyk = bgr2cmyk(imgc)
                h = cmyk[:, :, 2]
            elif self.colorsee_boxname == 'K of CMYK':
                cmyk=bgr2cmyk(imgc)
                h = cmyk[:,:,3]
            h_float = h.astype(np.float64)
        except:
            h=np.zeros((imgc.shape[0],imgc.shape[1]))
            if self.yuyan=='chinese':
                self.ui.label_45.setText('计算定量参数过程出现异常！')
            else:
                self.ui.label_45.setText('Error in calculating quantitative parameters!！')
        h_float=h.astype(np.float64)
        return h_float

    def biaoqu_erro(self):
        if self.yuyan=='english':
            self.msg_box3 = QMessageBox(QMessageBox.Information, 'Standard curve information',
                                        'There was an error during the process of substituting the quantitative parameter into the standard curve!\n'
                                        'The following reasons maybe cause errors:\n'
                                        '1.The format of writing does not comply with the set rules\n'
                                        '2.When the value of the quantitative parameter is beyond the its range in the standard curve, errors may be made when substituting it into the standard curve.\n'
                                        '3.The standard curve equation was not input.')
        else:
            self.msg_box3 = QMessageBox(QMessageBox.Information, '标准曲线信息',
                                        '代入标曲过程中出现错误，请检查！\n'
                                        '错误原因可能有以下几点：\n'
                                        '1.标准曲线的书写格式不符合设定的规则\n'
                                        '2.当定量参数的值超出标曲中定量参数的范围时，代入标准曲线时可能出错。\n'
                                        '3.没有输入标曲方程')
        self.msg_box3.exec_()
        self.ui.button_start.setEnabled(True)
        self.ui.button_start2D.setEnabled(True)
        self.ui.button_reselect_roi.setEnabled(True)

    def clear(self):
        self.ui.pushButton_cleardata.setEnabled(False)
        self.nongdu_list.clear()
        self.h_list.clear()
        self.time_list.clear()
        self.biaoqu_list.clear()
        self.fenbianlv_list.clear()
        self.color_suanfalist.clear()
        self.analyze_name.clear()
        self.yuzhi_list.clear()
        self.zibianliang.clear()
        self.exposuretimelist.clear()
        self.expousuremode.clear()
        self.wb_mode.clear()
        self.whitebalancelist.clear()
        try:
            self.fig0_ax1.clear()
            self.fig0_ax2.clear()
            self.fig0_ax1.grid()
            self.fig0_ax2.grid()
            self.canvas1.draw()
            self.canvas1.flush_events()
        except:
            pass
        try:
            self.fig1_ax1.clear()
            self.fig1_ax2.clear()
            self.fig1_ax1.grid()
            self.fig1_ax2.grid()
            self.canvas1.draw()
            self.canvas1.flush_events()
        except:
            pass
        self.num=0
        if self.yuyan=='english':
            self.msg_box3 = QMessageBox(QMessageBox.Information, 'Data information', 'Data cleared!')
        else:
            self.msg_box3 = QMessageBox(QMessageBox.Information, '数据信息','测量数据清除完毕！')
        self.msg_box3.exec_()
        self.ui.pushButton_cleardata.setEnabled(True)

    def clear_data(self):
        self.nongdu_list.clear()
        self.h_list.clear()
        self.time_list.clear()
        self.biaoqu_list.clear()
        self.fenbianlv_list.clear()
        self.color_suanfalist.clear()
        self.analyze_name.clear()
        self.yuzhi_list.clear()
        self.zibianliang.clear()
        self.exposuretimelist.clear()
        self.expousuremode.clear()
        self.wb_mode.clear()
        self.whitebalancelist.clear()
        self.num=0

    def update_fig0(self):
        if self.fig0_complete==0:
            return
        self.ui.statusbar.showMessage('')
        self.fig0_complete = 0
        if self.ui.radioButton_jiacu.isChecked():
            fontweight = 'bold'
        else:
            fontweight = 'normal'

        self.ratio1 = 0.995
        self.ratio2 = 0.995
        self.graphic_scene1.setSceneRect(0, 0, int(self.ratio2 * self.ui.graphicsView1.width()),
                                         int(self.ratio1 * self.ui.graphicsView1.height()))
        self.canvas1.resize(int(self.ratio2 * self.ui.graphicsView1.width()),
                         int(self.ratio1 * self.ui.graphicsView1.height()))

        self.fig0=self.canvas1.fig
        self.fig0_ax2=self.canvas1.axes
        self.fig0_ax2.cla()
        self.fig0_ax2.axis('on')
        self.fig0_ax2.grid()
        self.fig0_ax2.set_xlabel('Number')
        self.fig0_ax2.scatter(range(1,len(self.nongdu_list)+1), self.nongdu_list, s=60, color='blue', marker='o', edgecolors='gray')
        if self.ui.lineEdit_wuzhi.text()=='':
            self.fig0_ax2.set_ylabel(self.ui.comboBox_colorsee.currentText())
        else:
            self.fig0_ax2.set_ylabel(f'{self.ui.lineEdit_wuzhi.text()}'+f'{self.ui.lineEdit_danwei.text()}')
        self.ui.label_37.setText(f'{round(self.thread_single.y, 3)}')
        if self.ui.lineEdit_wuzhi.text()=='':
            self.ui.label_14.setText(self.ui.comboBox_colorsee.currentText()+':')
        else:
            self.ui.label_14.setText(self.ui.lineEdit_wuzhi.text()+':')

        plt.tight_layout()
        self.canvas1.draw()
        self.canvas1.flush_events()

        self.ui.button_start.setEnabled(True)
        self.ui.pushButton_cleardata.setEnabled(True)
        self.ui.action_save1D.setEnabled(True)
        self.ui.button_reselect_roi.setEnabled(True)
        self.fig0_complete = 1
        time.sleep(0)

    def update_fig1(self):
        if self.fig1_complete == 0:
            return
        self.fig1_complete = 0
        if self.ui.radioButton_jiacu.isChecked():
            fontweight = 'bold'
        else:
            fontweight = 'normal'

        self.ratio1 = 0.995
        self.ratio2 = 0.995
        self.graphic_scene1.setSceneRect(0, 0, int(self.ratio2 * self.ui.graphicsView1.width()),
                                         int(self.ratio1 * self.ui.graphicsView1.height()))
        self.canvas1.resize(int(self.ratio2 * self.ui.graphicsView1.width()),
                            int(self.ratio1 * self.ui.graphicsView1.height()))
        self.fig1=self.canvas1.fig
        self.fig1_ax2 = self.canvas1.axes
        self.fig1_ax2.cla()
        self.fig1_ax2.axis('on')
        self.fig1_ax2.grid()
        self.N=min(len(self.time_list),len(self.h_list),len(self.nongdu_list))
        self.fig1_ax2.plot(self.time_list[:self.N], self.nongdu_list[:self.N], color='gray', marker='o',
                           markerfacecolor='blue', markeredgecolor='gray')
        self.fig1_ax2.set_xlabel('Time (s)')
        if self.ui.lineEdit_wuzhi.text()=='':
            self.fig1_ax2.set_ylabel(self.ui.comboBox_colorsee.currentText())
        else:
            self.fig1_ax2.set_ylabel(f'{self.ui.lineEdit_wuzhi.text()}'+f'{self.ui.lineEdit_danwei.text()}')
        self.ui.label_37.setText(f'{round(self.thread_longtime.y, 3)}')
        if self.ui.lineEdit_wuzhi.text()=='':
            self.ui.label_14.setText(self.ui.comboBox_colorsee.currentText()+':')
        else:
            self.ui.label_14.setText(self.ui.lineEdit_wuzhi.text()+':')
        if self.yuyan == 'english':
            self.ui.label_35.setText(f'Current times of measurement: {self.N}')
        else:
            self.ui.label_35.setText(f'当前监测次数: {self.N}')

        plt.tight_layout()
        self.canvas1.draw()
        self.canvas1.flush_events()
        self.fig1_complete=1

    def save_1D(self):
        if self.yuyan == 'english':
            self.savepath0 = QFileDialog.getExistingDirectory(self.ui, "Please select the output folder")
        else:
            self.savepath0 = QFileDialog.getExistingDirectory(self.ui, "请选择输出文件夹")
        self.savepath=self.savepath0+os.sep+'colorsee1D data'
        if not os.path.exists(self.savepath):
            os.mkdir(self.savepath)
        self.saveimg()
        if self.ui.radioButton_single.isChecked()==True:
            if self.yuyan == 'english':
                self.data1D = {
                    "Sample name": self.zibianliang,
                    'Quantitative parameter': self.color_suanfalist,
                    'Value of quantitative parameter': self.h_list,
                    'Concentration of analytes' + f'\n{self.ui.lineEdit_wuzhi.text()}' + f' {self.ui.lineEdit_danwei.text()}': self.nongdu_list,
                    '': [],
                    'Standard curve equation': self.biaoqu_list,
                    'HT': self.yuzhi_list,
                    'Resolution of camera':self.fenbianlv_list,
                    'Exposure mode':self.expousuremode,
                    'Exposure time':self.exposuretimelist,
                    'White balance mode':self.wb_mode,
                    'White balance value':self.whitebalancelist}
            else:
                self.data1D={
                             "样品名称":self.zibianliang,
                            '定量参数': self.color_suanfalist,
                             '定量参数的值':self.h_list,
                             '分析物浓度'+f'\n{self.ui.lineEdit_wuzhi.text()}'+f' {self.ui.lineEdit_danwei.text()}':self.nongdu_list,
                             '': [],
                             '标准曲线':self.biaoqu_list,
                             'HT':self.yuzhi_list,
                            '相机分辨率': self.fenbianlv_list,
                            '曝光模式': self.expousuremode,
                            '曝光时间': self.exposuretimelist,
                            '白平衡模式': self.wb_mode,
                            '白平衡值': self.whitebalancelist
                            }
            try:
                if self.ui.save_format_box.currentText() == 'tiff':
                    self.fig0.savefig(self.savepath + os.sep + self.t_save+ f'-{self.ui.lineEdit_zibianliang.text()}' + '-single figure.tiff', dpi=eval(self.ui.dpi_box.currentText()),
                                  bbox_inches='tight', pad_inches=0.1)
                elif self.ui.save_format_box.currentText() == 'eps':
                    self.fig0.savefig(self.savepath + os.sep + self.t_save+ f'-{self.ui.lineEdit_zibianliang.text()}' + '-single figure.eps',
                                  bbox_inches='tight', pad_inches=0.1)
                elif self.ui.save_format_box.currentText() == 'pdf':
                    self.fig0.savefig(self.savepath + os.sep + self.t_save+ f'-{self.ui.lineEdit_zibianliang.text()}' + '-single figure.pdf',
                                  bbox_inches='tight', pad_inches=0.1)
                elif self.ui.save_format_box.currentText() == 'svg':
                    self.fig0.savefig(self.savepath + os.sep + self.t_save+ f'-{self.ui.lineEdit_zibianliang.text()}' + '-single figure.svg',
                                  bbox_inches='tight', pad_inches=0.1)
            except:
                pass

        elif self.ui.radioButton_longtime.isChecked()==True:
            if self.yuyan == 'english':
                self.data1D = {"Sample name": self.zibianliang,
                               'Time(s)': self.time_list,
                               'Quantitative parameter': self.color_suanfalist,
                               'Value of quantitative parameter': self.h_list,
                               'Concentration of analytes' + f'\n{self.ui.lineEdit_wuzhi.text()}' + f' {self.ui.lineEdit_danwei.text()}': self.nongdu_list,
                               '': [],
                               'Standard curve equation': self.biaoqu_list,
                               'HT': self.yuzhi_list,
                               'Resolution of camera':self.fenbianlv_list,
                               'Exposure mode': self.expousuremode,
                               'Exposure time': self.exposuretimelist,
                               'White balance mode': self.wb_mode,
                               'White balance value': self.whitebalancelist
                               }
            else:
                self.data1D = {"样品名称":self.zibianliang,
                               '时间(s)': self.time_list,
                               '定量参数': self.color_suanfalist,
                               '定量参数的值': self.h_list,
                               '分析物浓度' + f'\n{self.ui.lineEdit_wuzhi.text()}' + f' {self.ui.lineEdit_danwei.text()}': self.nongdu_list,
                               '':[],
                               '标准曲线': self.biaoqu_list,
                               'HT': self.yuzhi_list,
                               '相机分辨率':self.fenbianlv_list,
                               '曝光模式': self.expousuremode,
                               '曝光时间': self.exposuretimelist,
                               '白平衡模式': self.wb_mode,
                               '白平衡值': self.whitebalancelist
                               }
            try:
                if self.ui.save_format_box.currentText() == 'tiff':
                    self.fig1.savefig(self.savepath + os.sep + self.t_save+ f'-{self.ui.lineEdit_zibianliang.text()}' + '-longtime figure.tiff', dpi=eval(self.ui.dpi_box.currentText()),
                                  bbox_inches='tight', pad_inches=0.1)
                elif self.ui.save_format_box.currentText() == 'eps':
                    self.fig1.savefig(
                        self.savepath + os.sep + self.t_save + f'-{self.ui.lineEdit_zibianliang.text()}' + '-longtime figure.eps',
                        bbox_inches='tight', pad_inches=0.1)
                elif self.ui.save_format_box.currentText() == 'pdf':
                    self.fig1.savefig(
                        self.savepath + os.sep + self.t_save + f'-{self.ui.lineEdit_zibianliang.text()}' + '-longtime figure.pdf',
                        bbox_inches='tight', pad_inches=0.1)
                elif self.ui.save_format_box.currentText() == 'svg':
                    self.fig1.savefig(
                        self.savepath + os.sep + self.t_save + f'-{self.ui.lineEdit_zibianliang.text()}' + '-longtime figure.svg',
                        bbox_inches='tight', pad_inches=0.1)
            except:
                pass
        self.df1 = pd.DataFrame(pd.DataFrame.from_dict(self.data1D, orient='index').values.T, columns=list(self.data1D.keys()))
        self.df1.to_csv(self.savepath + os.sep + self.t_save + f'-{self.ui.lineEdit_zibianliang.text()}'+ '-colorsee 1Ddata.csv', header=True, sep=',',encoding="utf_8_sig")

    def save_2D(self):
        if self.yuyan == 'english':
            self.savepath0 = QFileDialog.getExistingDirectory(self.ui, "Please select the output folder")
        else:
            self.savepath0 = QFileDialog.getExistingDirectory(self.ui, "请选择输出文件夹")
        self.savepath = self.savepath0 + os.sep + 'colorsee2D data'
        if not os.path.exists(self.savepath):
            os.mkdir(self.savepath)
        self.t_save = time.strftime('%H-%M-%S', time.localtime())
        if '.csv' in self.csv_path0:
            self.t_save = time.strftime('%H-%M-%S', time.localtime())
            if self.ui.save_format_box.currentText() == 'tiff':
                self.fig3.savefig(self.savepath + os.sep + self.t_save+ f'-{self.ui.lineEdit_zibianliang.text()}' + '-chemical img.tiff', dpi=eval(self.ui.dpi_box.currentText()), bbox_inches='tight',
                              pad_inches=0.1)
            elif self.ui.save_format_box.currentText() == 'eps':
                self.fig3.savefig(
                    self.savepath + os.sep + self.t_save + f'-{self.ui.lineEdit_zibianliang.text()}' + '-chemical img.eps',
                    bbox_inches='tight',
                    pad_inches=0.1)
            elif self.ui.save_format_box.currentText() == 'pdf':
                self.fig3.savefig(
                    self.savepath + os.sep + self.t_save + f'-{self.ui.lineEdit_zibianliang.text()}' + '-chemical img.pdf',
                    bbox_inches='tight',
                    pad_inches=0.1)
            elif self.ui.save_format_box.currentText() == 'svg':
                self.fig3.savefig(
                    self.savepath + os.sep + self.t_save + f'-{self.ui.lineEdit_zibianliang.text()}' + '-chemical img.svg',
                    bbox_inches='tight',
                    pad_inches=0.1)
            return

        self.saveimg()
        np.savetxt(self.savepath + os.sep + self.t_save+ f'-{self.ui.lineEdit_zibianliang.text()}' + '-Quantitative_parameter_matrix.csv', self.thread_start2D.h, fmt='%.6f',
                   delimiter=',')
        np.savetxt(self.savepath + os.sep + self.t_save+ f'-{self.ui.lineEdit_zibianliang.text()}' + '-Concentration_matrix.csv', self.thread_start2D.y, fmt='%.6f',
                   delimiter=',')

        if self.ui.save_format_box.currentText()=='tiff':
            self.fig3.savefig(
                self.savepath + os.sep + self.t_save + f'-{self.ui.lineEdit_zibianliang.text()}' + '-chemical img.tiff',
                dpi=eval(self.ui.dpi_box.currentText()), bbox_inches='tight',
                pad_inches=0.1)
        elif self.ui.save_format_box.currentText()=='eps':
            self.fig3.savefig(
                self.savepath + os.sep + self.t_save + f'-{self.ui.lineEdit_zibianliang.text()}' + '-chemical img.eps',
                bbox_inches='tight',
                pad_inches=0.1)
        elif self.ui.save_format_box.currentText()=='pdf':
            self.fig3.savefig(
                self.savepath + os.sep + self.t_save + f'-{self.ui.lineEdit_zibianliang.text()}' + '-chemical img.pdf',
                bbox_inches='tight',
                pad_inches=0.1)
        elif self.ui.save_format_box.currentText()=='svg':
            self.fig3.savefig(
                self.savepath + os.sep + self.t_save + f'-{self.ui.lineEdit_zibianliang.text()}' + '-chemical img.svg',
                bbox_inches='tight',
                pad_inches=0.1)

    def saveimg(self):
        if self.yuyan == 'english':
            self.ui.statusbar.showMessage('Filepath need to be all english word,otherwise the image will not be saved.')
        else:
            self.ui.statusbar.showMessage('文件路径需要全为英文字符，否则图片无法保存')
        self.t_save = time.strftime('%H-%M-%S', time.localtime())
        cv.imwrite(self.savepath+os.sep+self.t_save+ f'-{self.ui.lineEdit_zibianliang.text()}'+'-camera_img.tiff',self.frame)
        cv.imwrite(self.savepath + os.sep + self.t_save+ f'-{self.ui.lineEdit_zibianliang.text()}' + '-roi_img.tiff', self.roiimg)

# Create the class for controlling camera
class ThreadOpencamera(QThread):
    signal0 = Signal(int)
    signal1 = Signal(int)
    signal2=Signal(int)
    signal3=Signal(int)
    signal4=Signal(int)
    signal5=Signal(int)
    signal6=Signal(int)
    frame=np.zeros((480,640,3))

    def __init__(self):
        super().__init__()

    def run(self):
        self.signal5.emit(1)
        color_chemistry.csv_path0 = ''
        color_chemistry.video_importpath = ''
        color_chemistry.picture_path0=''
        color_chemistry.ui.button_roi.setEnabled(True)
        color_chemistry.ui.button_open_camera.setEnabled(False)
        color_chemistry.ui.pushButton_OFFcam.setEnabled(True)
        color_chemistry.ui.actionda_openPicture.setEnabled(False)
        color_chemistry.ui.comboBox_fenbianlv.setEnabled(False)
        color_chemistry.ui.action_videostart.setEnabled(True)
        color_chemistry.ui.menu_3.setEnabled(True)
        color_chemistry.ui.menu_5.setEnabled(True)

        color_chemistry.ui.label_40.setEnabled(False)
        color_chemistry.ui.comboBox_exptime.setEnabled(False)
        color_chemistry.ui.label_39.setEnabled(False)
        color_chemistry.ui.exposureSlider.setEnabled(False)
        color_chemistry.ui.exposurelabel.setEnabled(False)
        color_chemistry.ui.label_42.setEnabled(False)
        color_chemistry.ui.comboBox_4.setEnabled(False)
        color_chemistry.ui.label_41.setEnabled(False)
        color_chemistry.ui.horizontalSlider_wb.setEnabled(False)
        color_chemistry.ui.label_wb.setEnabled(False)

        try:
            self.CAMERA_NUM = eval(color_chemistry.ui.comboBox_camera_num.currentText())
            self.cap = cv.VideoCapture(self.CAMERA_NUM, cv.CAP_DSHOW)
        except:
            self.signal1.emit(1)
            try:
                self.cap.release()
            except:
                pass
            color_chemistry.jietuok1 = False
            color_chemistry.ui.button_open_camera.setEnabled(True)
            color_chemistry.ui.button_roi.setEnabled(False)
            color_chemistry.ui.pushButton_paizhao.setEnabled(False)
            color_chemistry.ui.button_reselect_roi.setEnabled(False)
            color_chemistry.ui.pushButton_OFFcam.setEnabled(False)
            color_chemistry.ui.action_videostart.setEnabled(False)
            color_chemistry.ui.actionda_openPicture.setEnabled(True)
            color_chemistry.ui.comboBox_fenbianlv.setEnabled(True)
            color_chemistry.ui.action_videostart.setEnabled(False)
            color_chemistry.ui.menu_3.setEnabled(False)

            color_chemistry.ui.label_40.setEnabled(True)
            color_chemistry.ui.comboBox_exptime.setEnabled(True)
            if color_chemistry.ui.comboBox_exptime.currentIndex() == 0:
                color_chemistry.ui.label_39.setEnabled(True)
                color_chemistry.ui.exposureSlider.setEnabled(True)
                color_chemistry.ui.exposurelabel.setEnabled(True)
            color_chemistry.ui.label_42.setEnabled(True)
            color_chemistry.ui.comboBox_4.setEnabled(True)
            if color_chemistry.ui.comboBox_4.currentIndex() == 0:
                color_chemistry.ui.label_41.setEnabled(True)
                color_chemistry.ui.horizontalSlider_wb.setEnabled(True)
                color_chemistry.ui.label_wb.setEnabled(True)
            try:
                color_chemistry.stop_long()
                color_chemistry.stop_2D()
            except:
                pass
            self.signal6.emit(1)
            return

        try:
            fenbianlv = color_chemistry.ui.comboBox_fenbianlv.currentText()
            fenbianlv_split = fenbianlv.split()
            width_pixel = eval(fenbianlv_split[0])
            height_pixel = eval(fenbianlv_split[2])
            self.cap.set(3, width_pixel)
            self.cap.set(4, height_pixel)
            self.signal4.emit(1)
        except:
            pass

        if color_chemistry.ui.comboBox_exptime.currentIndex()==0:
            self.cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.0)
            self.cap.set(cv.CAP_PROP_EXPOSURE, color_chemistry.ui.exposureSlider.value())
        else:
            self.cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)

        if color_chemistry.ui.comboBox_4.currentIndex()==0:
            self.cap.set(cv.CAP_PROP_AUTO_WB, 0.0)  # 关闭自动白平衡
            self.cap.set(cv.CAP_PROP_WHITE_BALANCE_BLUE_U, color_chemistry.ui.horizontalSlider_wb.value())
        else:
            self.cap.set(cv.CAP_PROP_AUTO_WB, 1)

        time.sleep(1)

        color_chemistry.camopen=True
        try:
            while eval(color_chemistry.ui.comboBox_camera_num.currentText())==self.CAMERA_NUM and color_chemistry.camopen==True:  #
                try:
                    if self.cap.isOpened():
                        self.ret, self.frame0 = self.cap.read()
                        try:
                            shape=self.frame0.shape
                        except:
                            self.signal1.emit(1)
                            break
                        xuanzhuan=eval(color_chemistry.ui.comboBox_xuanzhuan.currentText())/90
                        self.frame1=np.rot90(self.frame0,xuanzhuan)
                        if color_chemistry.ui.comboBox_10.currentText()!='no set':
                            fanzhuan=eval(color_chemistry.ui.comboBox_10.currentText()[:2])
                            self.frame = cv.flip(self.frame1,fanzhuan)
                        else:
                            self.frame = self.frame1
                        color_chemistry.frame=self.frame.copy()
                        self.camera_screen=color_chemistry.change_imgbit(self.frame)
                        if self.ret:
                            if color_chemistry.camshow_complete == 1:
                                self.signal0.emit(1)
                        else:
                            self.signal1.emit(1)
                            break
                    else:
                        self.signal1.emit(1)
                        break
                except:
                    self.signal1.emit(1)
                    break
        except:
            pass
        try:
            self.cap.release()
        except:
            pass
        color_chemistry.jietuok1 = False
        color_chemistry.ui.button_open_camera.setEnabled(True)
        color_chemistry.ui.button_roi.setEnabled(False)
        color_chemistry.ui.pushButton_paizhao.setEnabled(False)
        color_chemistry.ui.button_reselect_roi.setEnabled(False)
        color_chemistry.ui.pushButton_OFFcam.setEnabled(False)
        color_chemistry.ui.action_videostart.setEnabled(False)
        color_chemistry.ui.actionda_openPicture.setEnabled(True)
        color_chemistry.ui.comboBox_fenbianlv.setEnabled(True)
        color_chemistry.ui.menu_3.setEnabled(False)
        color_chemistry.ui.label_40.setEnabled(True)
        color_chemistry.ui.comboBox_exptime.setEnabled(True)
        if color_chemistry.ui.comboBox_exptime.currentIndex() == 0:
            color_chemistry.ui.label_39.setEnabled(True)
            color_chemistry.ui.exposureSlider.setEnabled(True)
            color_chemistry.ui.exposurelabel.setEnabled(True)
        color_chemistry.ui.label_42.setEnabled(True)
        color_chemistry.ui.comboBox_4.setEnabled(True)
        if color_chemistry.ui.comboBox_4.currentIndex() == 0:
            color_chemistry.ui.label_41.setEnabled(True)
            color_chemistry.ui.horizontalSlider_wb.setEnabled(True)
            color_chemistry.ui.label_wb.setEnabled(True)

        try:
            color_chemistry.stop_long()
            color_chemistry.stop_2D()
        except:
            pass
        self.signal6.emit(1)

# Create the class for selecting ROI
class ThreadCutimg(QThread):
    signal0 = Signal(int)

    def __init__(self):
        super().__init__()

    def run(self):
        self.signal0.emit(1)

# Create the class for showing ROI
class ThreadShowroi(QThread):
    signal0 = Signal(int)
    signal1 = Signal(int)
    signal2 = Signal(int)

    def __init__(self):
        super().__init__()

    def run(self):
        color_chemistry.ui.pushButton_paizhao.setEnabled(True)
        color_chemistry.ui.button_start.setEnabled(True)
        color_chemistry.ui.button_stoplong.setEnabled(False)
        color_chemistry.ui.button_start2D.setEnabled(True)
        color_chemistry.ui.actionda_openPicture.setEnabled(False)
        color_chemistry.ui.button_stop2D.setEnabled(False)
        color_chemistry.ui.button_addpoint.setEnabled(True)
        while color_chemistry.jietuok1==False:
            time.sleep(0.3)
            pass
        while color_chemistry.jietuok1:
            try:
                if color_chemistry.ui.radioButton_roisum.isChecked():
                    self.roi_img = color_chemistry.frame.copy()
                    for roi in color_chemistry.list_roi:
                        cv.rectangle(self.roi_img, (roi[0],roi[1]), (roi[0]+roi[2], roi[1]+roi[3]),(0, 0, 255), 5)
                        roi_area=color_chemistry.frame[
                                   roi[1]:roi[1]+roi[3],
                                   roi[0]:roi[0]+roi[2]]
                        if 0 in roi_area.shape[:2]:
                            self.roi_img = color_chemistry.frame
                            break
                else:
                    self.roi_img = color_chemistry.frame[
                                   color_chemistry.min_y1:color_chemistry.min_y1 + color_chemistry.height1,
                                   color_chemistry.min_x1:color_chemistry.min_x1 + color_chemistry.width1]
                    if 0 in self.roi_img.shape[:2]:
                        self.roi_img = color_chemistry.frame
            except:
                self.roi_img = color_chemistry.frame

            color_chemistry.roiimg=self.roi_img.copy()
            self.roi_screen = color_chemistry.change_imgbit(self.roi_img)
            if not color_chemistry.ui.radioButton_roisum.isChecked():
                if self.roi_img.ndim==2:
                    self.roi_grayimg=cv.cvtColor(self.roi_img,cv.COLOR_GRAY2BGR)
                    self.b, self.g, self.r = cv.split(self.roi_grayimg)
                elif self.roi_img.shape[2]==3:
                    self.b,self.g,self.r= cv.split(self.roi_img)
                elif self.roi_img.shape[2]==4:
                    self.b, self.g, self.r, self.alph = cv.split(self.roi_img)
                try:
                    self.gray = color_chemistry.colorsee_box(self.roi_img)
                    self.grayvalue=np.around(np.mean(self.gray), 2)
                    self.signalname = color_chemistry.ui.comboBox_colorsee.currentText()
                except:
                    self.gray = cv.cvtColor(self.roi_img, cv.COLOR_BGR2GRAY)
                    self.grayvalue = np.around(np.mean(self.gray), 2)
                    self.signalname='Gray'

                self.bvalue=np.around(np.mean(self.b),2)
                self.gvalue = np.around(np.mean(self.g), 2)
                self.rvalue = np.around(np.mean(self.r), 2)
                self.shape=self.roi_img.shape
            self.wei = self.roi_img.dtype
            if color_chemistry.roishow_complete==1:
                self.signal0.emit(1)
        color_chemistry.ui.actionda_openPicture.setEnabled(True)
        color_chemistry.ui.label_50.setEnabled(True)
        color_chemistry.ui.radioButton_roisum.setEnabled(True)

# Create the class for measuring mean analyte concentration
class ThreadSingle(QThread):
    signal0 = Signal(int)
    signal1 = Signal(int)
    linex=[]
    liney=[]

    def __init__(self):
        super().__init__()

    def run(self):
        color_chemistry.ui.action_save1D.setEnabled(False)
        # color_chemistry.ui.button_start2D.setEnabled(False)
        color_chemistry.ui.button_reselect_roi.setEnabled(False)
        if '.csv' in color_chemistry.csv_path0 or color_chemistry.video_importpath != '':
            color_chemistry.ui.button_start.setEnabled(True)
            color_chemistry.ui.button_start2D.setEnabled(True)
            color_chemistry.ui.button_reselect_roi.setEnabled(True)
            return
        if not color_chemistry.ui.radioButton_roisum.isChecked():
            self.img=color_chemistry.roiimg.copy()
            self.h=color_chemistry.colorsee_box(self.img)
            self.hvalue = np.mean(self.h)
        else:
            self.hvalue=0
            for roi in color_chemistry.list_roi:
                self.img1 = color_chemistry.frame[roi[1]:roi[1] + roi[3],roi[0]:roi[0] + roi[2]]
                if 0 in self.img1.shape[:2]:
                    self.img = color_chemistry.frame.copy()
                    self.h = color_chemistry.colorsee_box(self.img)
                    self.hvalue = np.mean(self.h)
                    break
                else:
                    self.h1 = color_chemistry.colorsee_box(self.img1)
                    self.hvalue1 = np.mean(self.h1)
                    self.hvalue=self.hvalue+self.hvalue1
        color_chemistry.h_list.append(self.hvalue)
        x=self.hvalue.copy()
        color_chemistry.biaoqu_list.append('y='+color_chemistry.ui.lineEdit_biaoqu.text())
        color_chemistry.color_suanfalist.append(color_chemistry.ui.comboBox_colorsee.currentText())
        color_chemistry.fenbianlv_list.append(color_chemistry.ui.comboBox_fenbianlv.currentText())
        color_chemistry.yuzhi_list.append(f'{color_chemistry.ui.horizontalSlider_value.value()/10000}')
        color_chemistry.analyze_name.append(
            f'{color_chemistry.ui.lineEdit_wuzhi.text()}' + f'{color_chemistry.ui.lineEdit_danwei.text()}')
        color_chemistry.zibianliang.append(color_chemistry.ui.lineEdit_zibianliang.text())
        color_chemistry.expousuremode.append(color_chemistry.ui.comboBox_exptime.currentText())
        color_chemistry.exposuretimelist.append(color_chemistry.ui.exposureSlider.value())
        color_chemistry.whitebalancelist.append(color_chemistry.ui.horizontalSlider_wb.value())
        color_chemistry.wb_mode.append(color_chemistry.ui.comboBox_4.currentText())

        self.huemin = color_chemistry.ui.lineEdit_huemin.text()
        self.huemax = color_chemistry.ui.lineEdit_huemax.text()

        if self.huemin != '' and self.huemax != '':
            try:
                if x < eval(self.huemin):
                    x = eval(self.huemin)
                elif x > eval(self.huemax):
                    x = eval(self.huemax)
            except:
                pass
        try:
            color_chemistry.num += 1
            self.y = eval(color_chemistry.ui.lineEdit_biaoqu.text())
            color_chemistry.nongdu_list.append(self.y)
        except:
            color_chemistry.nongdu_list.append(None)
            self.signal0.emit(1)
            return
        self.signal1.emit(1)

# Create the class for measuring mean analyte concentration
class ThreadAddpoint(QThread):
    signal0 = Signal(int)
    signal1 = Signal(int)

    def __init__(self):
        super().__init__()

    def run(self):
        if '.csv' in color_chemistry.csv_path0 or color_chemistry.video_importpath != '':
            color_chemistry.ui.button_addpoint.setEnabled(True)
            return
        if not color_chemistry.ui.radioButton_roisum.isChecked():
            try:
                self.img = color_chemistry.roiimg.copy()
                self.h = color_chemistry.colorsee_box(self.img)
                self.hvalue = np.mean(self.h)
            except:
                self.hvalue=None
        else:
            try:
                self.hvalue = 0
                for roi in color_chemistry.list_roi:
                    self.img1 = color_chemistry.frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
                    if 0 in self.img1.shape[:2]:
                        self.img = color_chemistry.frame.copy()
                        self.h = color_chemistry.colorsee_box(self.img)
                        self.hvalue = np.mean(self.h)
                        break
                    else:
                        self.h1 = color_chemistry.colorsee_box(self.img1)
                        self.hvalue1 = np.mean(self.h1)
                        self.hvalue = self.hvalue + self.hvalue1
            except:
                self.hvalue=None
        try:
            self.concentration=eval(color_chemistry.ui.lineEdit.text())
        except:
            self.signal1.emit(1)
            return

        color_chemistry.jiaozhunx.append(self.concentration)
        color_chemistry.jiaozhuny.append(self.hvalue)

        color_chemistry.std_color_suanfalist.append(color_chemistry.ui.comboBox_colorsee.currentText())
        color_chemistry.std_fenbianlv_list.append(color_chemistry.ui.comboBox_fenbianlv.currentText())
        color_chemistry.std_yuzhi_list.append(f'{color_chemistry.ui.horizontalSlider_value.value() / 10000}')
        color_chemistry.std_analyze_name.append(
            f'{color_chemistry.ui.lineEdit_wuzhi.text()}' + f'{color_chemistry.ui.lineEdit_danwei.text()}')
        color_chemistry.std_zibianliang.append(color_chemistry.ui.lineEdit_zibianliang.text())
        color_chemistry.std_expousuremode.append(color_chemistry.ui.comboBox_exptime.currentText())
        color_chemistry.std_exposuretimelist.append(color_chemistry.ui.exposureSlider.value())
        color_chemistry.std_whitebalancelist.append(color_chemistry.ui.horizontalSlider_wb.value())
        color_chemistry.std_wb_mode.append(color_chemistry.ui.comboBox_4.currentText())
        color_chemistry.ui.pushButton_3.setEnabled(True)
        self.signal0.emit(1)

# Create the class for continuously monitoring the mean analyte concentration
class ThreadLongtime(QThread):
    signal0 = Signal(int)
    signal1 = Signal(int)
    signal2=Signal(int)
    signal3=Signal(int)

    def __init__(self):
        super().__init__()

    def run(self):
        color_chemistry.ui.menu_3.setEnabled(False)
        color_chemistry.ui.action_importvideo.setEnabled(False)
        color_chemistry.ui.menu_5.setEnabled(False)
        color_chemistry.ui.pushButton_cleardata.setEnabled(False)
        color_chemistry.ui.actiondd_opencsv.setEnabled(False)
        color_chemistry.ui.button_reselect_roi.setEnabled(False)
        color_chemistry.ui.button_stoplong.setEnabled(True)
        color_chemistry.ui.action_save1D.setEnabled(False)
        # color_chemistry.ui.button_start2D.setEnabled(False)
        color_chemistry.ui.actionda_openPicture.setEnabled(False)
        color_chemistry.ui.comboBox_2_zihao.setEnabled(False)
        color_chemistry.ui.comboBox_ziti.setEnabled(False)
        color_chemistry.ui.radioButton_jiacu.setEnabled(False)

        self.t0=time.perf_counter()
        color_chemistry.clear_data()
        cyclenum=1
        self.videonum=1
        self.videoret=True
        while color_chemistry.longok:
            self.cycleoff=0
            if '.csv' in color_chemistry.csv_path0:
                break
            try:
                cishu=color_chemistry.ui.cishu.text()
                if cishu=='':
                    pass
                elif cyclenum>eval(cishu):
                    break
            except:
                self.signal2.emit(1)
                break
            #视频处理方式
            if color_chemistry.video_importpath!='':
                if self.videonum==1:
                    self.videocap = cv.VideoCapture(color_chemistry.video_importpath)
                    self.total_frame_count = self.videocap.get(cv.CAP_PROP_FRAME_COUNT)
                self.videoret,self.frame=self.videocap.read()
                if self.videoret==False:
                    self.videonum -= 1
                    self.signal3.emit(1)
                    break

                color_chemistry.frame=self.frame.copy()
                #显示图片
                self.signal3.emit(1)
                # 单个ROI时计算平均信号值
                if not color_chemistry.ui.radioButton_roisum.isChecked():
                    try:
                        self.roi_img = color_chemistry.frame[
                                       color_chemistry.min_y1:color_chemistry.min_y1 + color_chemistry.height1,
                                       color_chemistry.min_x1:color_chemistry.min_x1 + color_chemistry.width1]
                    except:
                        self.roi_img = color_chemistry.frame
                    if 0 in self.roi_img.shape[:2]:
                        self.roi_img = color_chemistry.frame
                    self.img=self.roi_img
                    self.h = color_chemistry.colorsee_box(self.img)
                    self.hvalue = np.mean(self.h)
                # 多个ROI时计算定量参数的和
                else:
                    self.hvalue = 0
                    for roi in color_chemistry.list_roi:
                        self.img1 = color_chemistry.frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
                        if 0 in self.img1.shape[:2]:
                            self.img = color_chemistry.frame.copy()
                            self.h = color_chemistry.colorsee_box(self.img)
                            self.hvalue = np.mean(self.h)
                            break
                        else:
                            self.h1 = color_chemistry.colorsee_box(self.img1)
                            self.hvalue1 = np.mean(self.h1)
                            self.hvalue = self.hvalue + self.hvalue1
                self.videonum += 1
            #相机视频流处理方式
            else:
                #单个ROI时计算平均信号值
                if not color_chemistry.ui.radioButton_roisum.isChecked():
                    self.img = color_chemistry.roiimg.copy()
                    self.h = color_chemistry.colorsee_box(self.img)
                    self.hvalue = np.mean(self.h)
                #多个ROI时计算定量参数的和
                else:
                    self.hvalue = 0
                    for roi in color_chemistry.list_roi:
                        self.img1 = color_chemistry.frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
                        if 0 in self.img1.shape[:2]:
                            self.img = color_chemistry.frame.copy()
                            self.h = color_chemistry.colorsee_box(self.img)
                            self.hvalue = np.mean(self.h)
                            break
                        else:
                            self.h1 = color_chemistry.colorsee_box(self.img1)
                            self.hvalue1 = np.mean(self.h1)
                            self.hvalue = self.hvalue + self.hvalue1
            color_chemistry.h_list.append(self.hvalue)
            x=self.hvalue.copy()
            self.t1 = time.perf_counter()
            self.t = self.t1 - self.t0
            color_chemistry.time_list.append(self.t)
            color_chemistry.biaoqu_list.append('y=' + color_chemistry.ui.lineEdit_biaoqu.text())
            color_chemistry.color_suanfalist.append(color_chemistry.ui.comboBox_colorsee.currentText())
            color_chemistry.fenbianlv_list.append(color_chemistry.ui.comboBox_fenbianlv.currentText())
            color_chemistry.yuzhi_list.append(f'{color_chemistry.ui.horizontalSlider_value.value()/10000}')
            color_chemistry.analyze_name.append(f'{color_chemistry.ui.lineEdit_wuzhi.text()}'+f'{color_chemistry.ui.lineEdit_danwei.text()}')
            color_chemistry.zibianliang.append(color_chemistry.ui.lineEdit_zibianliang.text())
            color_chemistry.expousuremode.append(color_chemistry.ui.comboBox_exptime.currentText())
            color_chemistry.exposuretimelist.append(color_chemistry.ui.exposureSlider.value())
            color_chemistry.whitebalancelist.append(color_chemistry.ui.horizontalSlider_wb.value())
            color_chemistry.wb_mode.append(color_chemistry.ui.comboBox_4.currentText())

            self.huemin = color_chemistry.ui.lineEdit_huemin.text()
            self.huemax = color_chemistry.ui.lineEdit_huemax.text()
            if self.huemin != '' and self.huemax != '':
                try:
                    if x<eval(self.huemin):
                        x=eval(self.huemin)
                    elif x>eval(self.huemax):
                        x=eval(self.huemax)
                except:
                    pass
            try:
                color_chemistry.num += 1
                self.y = eval(color_chemistry.ui.lineEdit_biaoqu.text())
                color_chemistry.nongdu_list.append(self.y)
            except:
                color_chemistry.nongdu_list.append(None)
                self.signal0.emit(1)
                break
            self.signal1.emit(1)
            self.interval=color_chemistry.ui.comboBox_caiyangjiange.currentText()
            if self.interval != 'no set':
                try:
                    time.sleep(eval(self.interval))
                except:
                    time.sleep(1)
                    pass
            cyclenum+=1
            if color_chemistry.picture_path0!='':
                break
        self.cycleoff=1
        if color_chemistry.csv_path0=='':
            time.sleep(2)
            self.signal1.emit(1)
        color_chemistry.num=0
        try:
            self.videocap.release()
        except:
            pass
        color_chemistry.longok=False
        color_chemistry.ui.button_stoplong.setEnabled(False)
        color_chemistry.ui.button_start.setEnabled(True)
        color_chemistry.ui.pushButton_cleardata.setEnabled(True)
        color_chemistry.ui.action_save1D.setEnabled(True)
        color_chemistry.ui.button_reselect_roi.setEnabled(True)
        color_chemistry.ui.actionda_openPicture.setEnabled(True)
        color_chemistry.ui.actiondd_opencsv.setEnabled(True)
        color_chemistry.ui.comboBox_2_zihao.setEnabled(True)
        color_chemistry.ui.comboBox_ziti.setEnabled(True)
        color_chemistry.ui.radioButton_jiacu.setEnabled(True)
        color_chemistry.ui.menu_3.setEnabled(True)
        color_chemistry.ui.action_importvideo.setEnabled(True)
        color_chemistry.ui.menu_5.setEnabled(True)

# Create the class for save data
class ThreadSavepath(QThread):
    signal0 = Signal(int)

    def __init__(self):
        super().__init__()

    def run(self):
        self.signal0.emit(1)

# Create a class to wait the start of main window
class Wait(QThread):
    signal0 = Signal(int)

    def __init__(self):
        super().__init__()

    def run(self):
        time.sleep(1)
        self.signal0.emit(1)

# Create a class to measuring chemical images
class ThreadStart2D(QThread):
    signal0 = Signal(int)
    signal1 = Signal(int)
    signal2 = Signal(int)
    signal3 = Signal(int)
    signal4 = Signal(int)
    signal5=Signal(int)
    signal6=Signal(int)
    signal7=Signal(int)
    jindu=0

    def __init__(self):
        super().__init__()

    def run(self):
        color_chemistry.ui.menu_3.setEnabled(False)
        color_chemistry.ui.action_importvideo.setEnabled(False)
        color_chemistry.ui.actiondd_opencsv.setEnabled(False)
        color_chemistry.ui.action_save2D.setEnabled(False)
        color_chemistry.ui.button_stop2D.setEnabled(True)
        color_chemistry.ui.button_reselect_roi.setEnabled(False)
        color_chemistry.ui.actionda_openPicture.setEnabled(False)
        color_chemistry.ui.pushButton_cleardata.setEnabled(False)
        try:
            color_chemistry.roicenterline.ui.pushButton.setEnabled(False)
        except:
            pass
        try:
            color_chemistry.roicenterline2.ui.pushButton.setEnabled(False)
        except:
            pass

        self.videonum=0
        self.videoret = True
        color_chemistry.startsignal=False
        while color_chemistry.chemicalimgok:
            while color_chemistry.fig3_complete == 0 or color_chemistry.switch_show_roicenterline==0 or color_chemistry.switch_show_roicenterline2 == 0:
                time.sleep(0.001)
                pass
            self.cycleoff = 0
            if '.csv' in color_chemistry.csv_path0:
                self.y = color_chemistry.csvimg
                self.signal1.emit(1)
                color_chemistry.ui.checkBox_ROIcenter.setEnabled(True)
                color_chemistry.ui.checkBox_3.setEnabled(True)
                if color_chemistry.ui.checkBox_ROIcenter.isChecked():
                    self.middieline=self.y[int(self.y.shape[0]/2)]
                    self.xdata = np.arange(1,1+self.middieline.shape[0])
                    max2=np.max(self.y)
                    min2=np.min(self.y)
                    detav=max2-min2
                    guiyihua=255*(self.y-min2)/detav
                    csv_img0=guiyihua.astype(np.uint8)
                    csv_img=cv.cvtColor(csv_img0,cv.COLOR_GRAY2BGR)
                    self.roicenterimg=csv_img
                    color_chemistry.roicenterimg=self.roicenterimg
                    cv.imwrite('csv_img.tiff',csv_img)
                    self.signal5.emit(1)
                if color_chemistry.ui.checkBox_3.isChecked():
                    self.middieline2 = self.y[:,int(self.y.shape[1] / 2)]
                    self.xdata2 = np.arange(1, 1 + self.middieline2.shape[0])
                    max3 = np.max(self.y)
                    min3 = np.min(self.y)
                    detav3 = max3 - min3
                    guiyihua2 = 255 * (self.y - min3) / detav3
                    csv_img02 = guiyihua2.astype(np.uint8)
                    csv_img2 = cv.cvtColor(csv_img02, cv.COLOR_GRAY2BGR)
                    self.roicenterimg2 = csv_img2
                    color_chemistry.roicenterimg = self.roicenterimg2
                    cv.imwrite('csv_img2.tiff', csv_img2)
                    self.signal6.emit(1)
                break

            elif color_chemistry.video_importpath!='':
                self.videonum += 1
                if self.videonum==1:
                    self.videocap = cv.VideoCapture(color_chemistry.video_importpath)
                    self.total_frame_count = self.videocap.get(cv.CAP_PROP_FRAME_COUNT)
                self.videoret,self.frame=self.videocap.read()
                if self.videoret==False:
                    self.videonum-=1
                    self.signal4.emit(1)
                    break
                color_chemistry.frame=self.frame.copy()

                self.signal4.emit(1)
                try:
                    self.roi_img = color_chemistry.frame[
                                   color_chemistry.min_y1:color_chemistry.min_y1 + color_chemistry.height1,
                                   color_chemistry.min_x1:color_chemistry.min_x1 + color_chemistry.width1]
                except:
                    self.roi_img = color_chemistry.frame
                if 0 in self.roi_img.shape[:2]:
                    self.roi_img = color_chemistry.frame

                self.img=self.roi_img
            else:
                self.img=color_chemistry.roiimg.copy()


            self.h0=color_chemistry.colorsee_box(self.img)
            self.h = self.h0.astype(np.float64)
            self.x_ = self.h
            self.huemin=color_chemistry.ui.lineEdit_huemin.text()
            self.huemax=color_chemistry.ui.lineEdit_huemax.text()
            if self.huemin!='' and self.huemax!='':
                try:
                    self.h_shaixuan=np.clip(self.x_.copy(),eval(self.huemin),eval(self.huemax))
                    self.x_ = self.h_shaixuan
                except:
                    pass

            try:
                if ('if' in color_chemistry.ui.lineEdit_biaoqu.text()):
                    self.jd=1
                    self.y=self.piecewise_calibration(self.x_,color_chemistry.ui.lineEdit_biaoqu.text())
                else:
                    x=self.x_
                    self.y = eval(color_chemistry.ui.lineEdit_biaoqu.text())
                color_chemistry.ui.checkBox_ROIcenter.setEnabled(True)
                color_chemistry.ui.checkBox_3.setEnabled(True)
            except:
                color_chemistry.ui.checkBox_ROIcenter.setEnabled(False)
                color_chemistry.ui.checkBox_3.setEnabled(False)
                self.signal0.emit(1)
                break
            if True in np.isnan(self.y).any(axis=0) :
                self.signal2.emit(1)
                break
            if color_chemistry.ui.checkBox_ROIcenter.isChecked():
                self.middieline = self.y[int(self.y.shape[0] / 2)]
                self.xdata = np.arange(1, 1 + self.middieline.shape[0])
                color_chemistry.switch_show_roicenterline = 0
                self.signal5.emit(1)
                time.sleep(0.01)
            if color_chemistry.ui.checkBox_3.isChecked():
                self.middieline2 = self.y[:, int(self.y.shape[1] / 2)]
                self.xdata2 = np.arange(1, 1 + self.middieline2.shape[0])
                color_chemistry.switch_show_roicenterline2 = 0
                self.signal6.emit(1)
                time.sleep(0.01)
            color_chemistry.fig3_complete = 0
            self.signal1.emit(1)
            time.sleep(0.01)
            try:
                if color_chemistry.ui.comboBox_5.currentText()!='no set':
                    time.sleep(eval(color_chemistry.ui.comboBox_5.currentText()))
            except:
                pass
            if color_chemistry.picture_path0!='':
                break
        self.cycleoff = 1
        time.sleep(2)
        self.signal1.emit(1)
        try:
            self.videocap.release()
        except:
            pass
        try:
            color_chemistry.chemicalvideoout.release()
        except:
            pass
        color_chemistry.startsignal = False
        color_chemistry.chemicalimgok=True
        color_chemistry.switch_startchemicalvideo=0
        color_chemistry.ui.actionchemicalvideostart.setEnabled(True)
        color_chemistry.ui.actionchemicalvideostop.setEnabled(False)
        color_chemistry.ui.button_stop2D.setEnabled(False)
        color_chemistry.ui.button_start2D.setEnabled(True)
        color_chemistry.ui.action_save2D.setEnabled(True)
        color_chemistry.ui.button_reselect_roi.setEnabled(True)
        color_chemistry.ui.actionda_openPicture.setEnabled(True)
        color_chemistry.ui.actiondd_opencsv.setEnabled(True)
        color_chemistry.ui.pushButton_cleardata.setEnabled(True)
        color_chemistry.ui.action_importvideo.setEnabled(True)
        color_chemistry.ui.menu_3.setEnabled(True)
        try:
            color_chemistry.roicenterline.ui.pushButton.setEnabled(True)
        except:
            pass
        try:
            color_chemistry.roicenterline2.ui.pushButton.setEnabled(True)
        except:
            pass

    def piecewise_calibration(self, x_matrix, equation):
        x0 = x_matrix.copy()
        x1 = x0.astype(np.float64)
        h = x1.shape[0]
        l = x1.shape[1]
        result = np.zeros_like(x1)
        namespace = {'np': np}
        func = eval(f'lambda x: {equation}', namespace)
        for i in range(h):
            self.jindu=round(100*i/(h-1),2)
            self.signal3.emit(1)
            for j in range(l):
                result[i, j] = func(x1[i, j])
        self.signal7.emit(1)
        return result

# Create a class to capture video using camera
class ThreadTakeVideos(QThread):
    signal0 = Signal(int)
    signal1 = Signal(int)
    signal2 = Signal(int)

    def __init__(self):
        super().__init__()

    def run(self):
        color_chemistry.switch_set_video_parameters=0
        color_chemistry.switch_select_video_folder=0
        color_chemistry.swich_takevideo=1
        color_chemistry.ui.action_videostart.setEnabled(False)
        color_chemistry.ui.action_videostop.setEnabled(True)
        self.i = 1
        self.t_current = 0
        time_list = []
        self.fourcc = cv.VideoWriter_fourcc(*'XVID')
        self.signal1.emit(1)
        while color_chemistry.switch_select_video_folder==0:
            time.sleep(0.3)
            pass
        self.signal2.emit(1)
        while color_chemistry.switch_set_video_parameters==0:
            time.sleep(0.3)
            pass
        try:
            self.fps = float(1/color_chemistry.set_video.time_interval)
        except:
            self.fps=30
        if self.fps<30:
            self.fps=30
        t_file = time.strftime('%H-%M-%S', time.localtime())
        self.out = cv.VideoWriter(color_chemistry.savepath_video+os.sep+f'{t_file}_Video_{color_chemistry.ui.lineEdit_zibianliang.text()}.avi',
                                  self.fourcc, self.fps, (color_chemistry.frame.shape[1], color_chemistry.frame.shape[0]))
        t0 = time.perf_counter()
        while color_chemistry.thread_opencamera.cap.isOpened() and color_chemistry.swich_takevideo==1 and self.t_current<=color_chemistry.set_video.timelong:
            self.out.write(color_chemistry.frame)
            t1 = time.perf_counter()
            self.t_current=t1-t0
            time_list.append(self.t_current)
            self.signal0.emit(1)
            self.i += 1
            try:
                time.sleep(color_chemistry.set_video.time_interval)
            except:
                time.sleep(1)
        self.out.release()
        np.savetxt(color_chemistry.savepath_video+os.sep+f'{t_file}_Time_sequence_{color_chemistry.ui.lineEdit_zibianliang.text()}.csv', np.array(time_list), delimiter=',', header='Time(s)', fmt='%f')
        color_chemistry.ui.action_videostart.setEnabled(True)
        color_chemistry.ui.action_videostop.setEnabled(False)

# Create a class to set parameters of capturing video
class VideoParameterSet(QDialog):
    BASE_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
    def __init__(self):
        super(VideoParameterSet, self).__init__()
        self.yuyan =selectyuyan.yuyan
        if self.yuyan=='english':
            qfile_stats = QFile(self.BASE_DIR+os.sep+'gui'+os.sep+ 'video_parameter_english.ui')
        else:
            qfile_stats = QFile(self.BASE_DIR + os.sep+'gui'+os.sep + 'video_parameter_chinese.ui')
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()
        self.ui = QUiLoader().load(qfile_stats)
        self.ui.setWindowIcon(QIcon(self.BASE_DIR + os.sep + 'img' + os.sep + "tubiao1.ico"))
        self.ui.pushButton_queding.clicked.connect(self.queding)

    def queding(self):
        try:
            self.time_interval=eval(self.ui.lineEdit_video_interval.text())
            self.timelong=eval(self.ui.lineEdit_video_timelong.text())

            self.ui.close()
            color_chemistry.switch_set_video_parameters=1
        except:
            if self.yuyan == 'english':
                self.msg_box_VIDEO = QMessageBox(QMessageBox.Information, 'Failed to set video parameters',
                                            f'The parameters should be numbers. Please input again!')
            else:
                self.msg_box_VIDEO = QMessageBox(QMessageBox.Information, '视频参数设置错误',
                                            f'参数只能为数字，请重新输入！')
            self.msg_box_VIDEO.exec_()

# Create a class to set parameters of capturing chemical video
class VideoParameterSet2(QDialog):
    BASE_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
    def __init__(self):
        super(VideoParameterSet2, self).__init__()
        self.yuyan =selectyuyan.yuyan
        if self.yuyan=='english':
            qfile_stats = QFile(self.BASE_DIR+os.sep+'gui'+os.sep+ 'video_parameter_english2.ui')
        else:
            qfile_stats = QFile(self.BASE_DIR + os.sep+'gui'+os.sep + 'video_parameter_chinese2.ui')
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()
        self.ui = QUiLoader().load(qfile_stats)
        self.ui.setWindowIcon(QIcon(self.BASE_DIR + os.sep + 'img' + os.sep + "tubiao1.ico"))
        self.ui.pushButton_queding.clicked.connect(self.queding)

    def queding(self):
        try:
            self.timelong=eval(self.ui.lineEdit_video_timelong.text())
            self.fps=float(eval(self.ui.lineEdit_chemicalfps.text()))
            if self.fps>10000:
                raise
            self.ui.close()
            color_chemistry.switch_set_video_parameters2=1
            if self.yuyan == 'english':
                self.msg_box_VIDEO = QMessageBox(QMessageBox.Information, 'Chemical video',
                                            f"Chemical video will be captured after <Start> button in 'Measuring chemical image' module is clicked!")
            else:
                self.msg_box_VIDEO = QMessageBox(QMessageBox.Information, '化学视频',
                                            f'点击测量化学图像模块的<开始>按钮后，化学图像视频将开始录制！')
            self.msg_box_VIDEO.exec_()

        except:
            if self.yuyan == 'english':
                self.msg_box_VIDEO = QMessageBox(QMessageBox.Information, 'Failed to set video parameters',
                                            f'The parameters should be numbers. FPS can not be greater than 10000. Please input again!')
            else:
                self.msg_box_VIDEO = QMessageBox(QMessageBox.Information, '视频参数设置错误',
                                            f'参数只能为数字，且帧率不可以大于10000fps，请重新输入！')
            self.msg_box_VIDEO.exec_()

# Create a class to extract and show the horizontal ROI center data of chemical image
class RoiCenterline(QDialog):
    BASE_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))

    def __init__(self):
        super(RoiCenterline, self).__init__()
        self.yuyan = selectyuyan.yuyan
        if self.yuyan == 'english':
            qfile_stats = QFile(self.BASE_DIR + os.sep +'gui'+os.sep+ 'roi_centreline_english.ui')
        else:
            qfile_stats = QFile(self.BASE_DIR + os.sep +'gui'+os.sep+ 'roi_centreline_chinese.ui')
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()
        self.ui = QUiLoader().load(qfile_stats)
        self.ui.setWindowIcon(QIcon(self.BASE_DIR + os.sep + 'img' + os.sep + "tubiao1.ico"))
        self.ui.pushButton.clicked.connect(self.save)
        self.ui.setWindowFlags(Qt.WindowTitleHint | Qt.CustomizeWindowHint)

        self.canvas1 = MyFigureCanvas2()
        self.graphic_scene1 = QGraphicsScene()
        self.graphic_scene1.addWidget(self.canvas1)
        self.ui.graphicsView1.setScene(self.graphic_scene1)
        self.ratio1 = 0.995
        self.ratio2 = 0.995
        self.graphic_scene1.setSceneRect(0, 0, int(self.ratio2 * self.ui.graphicsView1.width()),
                                         int(self.ratio1 * self.ui.graphicsView1.height()))
        self.canvas1.resize(int(self.ratio2 * self.ui.graphicsView1.width()),
                            int(self.ratio1 * self.ui.graphicsView1.height()))
        self.canvas1.axes.axis('on')
        self.canvas1.axes.grid()
        self.ui.graphicsView1.show()

        self.line1 = dict(color='black', lw=1)
        self.cursor1 = widgets.Cursor(self.canvas1.axes, useblit=True, **self.line1)
        self.canvas1.mpl_connect('motion_notify_event', self.on_mouse_move)

        color_chemistry.gui_roicenter=1
        self.ui.pushButton_2.clicked.connect(self.closegui)

    def save(self):
        if self.yuyan == 'english':
            self.savepath0 = QFileDialog.getExistingDirectory(self.ui, "Please select the output folder")
        else:
            self.savepath0 = QFileDialog.getExistingDirectory(self.ui, "请选择输出文件夹")
        self.savepath = self.savepath0 + os.sep + 'ROI center line xdata'
        if not os.path.exists(self.savepath):
            os.mkdir(self.savepath)
        self.t_save = time.strftime('%H-%M-%S', time.localtime())

        cv.imwrite(self.savepath+os.sep+self.t_save+ f'-{color_chemistry.ui.lineEdit_zibianliang.text()}'+'-roi_centerline_img.tiff',color_chemistry.roicenterimg2)
        if color_chemistry.ui.lineEdit_wuzhi.text() == '':
            self.data1D = {
                "Pixels": color_chemistry.thread_start2D.xdata,
                "Value": color_chemistry.thread_start2D.middieline,
                }
        else:
            self.data1D = {
                "Pixels": color_chemistry.thread_start2D.xdata,
                f"{color_chemistry.ui.lineEdit_wuzhi.text()} {color_chemistry.ui.lineEdit_danwei.text()}": color_chemistry.thread_start2D.middieline,
            }

        self.df1 = pd.DataFrame(pd.DataFrame.from_dict(self.data1D, orient='index').values.T,
                                columns=list(self.data1D.keys()))
        self.df1.to_csv(self.savepath + os.sep + self.t_save + f'-{color_chemistry.ui.lineEdit_zibianliang.text()}'+ '-centerline_xdata.csv', header=True, sep=',',
                        encoding="utf_8_sig")
        try:
            color_chemistry.fig4.savefig(self.savepath + os.sep + self.t_save + f'-{color_chemistry.ui.lineEdit_zibianliang.text()}'+'-centerline_xdata.tiff', dpi=350,
                              bbox_inches='tight', pad_inches=0.1)
        except:
            pass

    def on_mouse_move(self, event):
        try:
            if event.inaxes:
                if self.yuyan=='chinese':
                    self.ui.label_2.setText(f'坐标: (x:{event.xdata:.2f}, y:{event.ydata:.2f})')
                else:
                    self.ui.label_2.setText(f'Coordinate: (x:{event.xdata:.2f}, y:{event.ydata:.2f})')
        except:
            pass

    def closeEvent(self, event):
        event.ignore()
    def closegui(self):
        self.ui.close()
        color_chemistry.gui_roicenter = 0

# Create a class to extract and show the vertical ROI center data of chemical image
class RoiCenterline2(QDialog):
    BASE_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))

    def __init__(self):
        super(RoiCenterline2, self).__init__()
        self.yuyan = selectyuyan.yuyan
        if self.yuyan == 'english':
            qfile_stats = QFile(self.BASE_DIR + os.sep +'gui'+os.sep+ 'roi_centreline_english.ui')
        else:
            qfile_stats = QFile(self.BASE_DIR + os.sep +'gui'+os.sep+ 'roi_centreline_chinese.ui')
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()
        self.ui = QUiLoader().load(qfile_stats)
        self.ui.setWindowIcon(QIcon(self.BASE_DIR + os.sep + 'img' + os.sep + "tubiao1.ico"))
        self.ui.setWindowFlags(Qt.WindowTitleHint | Qt.CustomizeWindowHint)

        self.canvas1 = MyFigureCanvas3()
        self.graphic_scene1 = QGraphicsScene()
        self.graphic_scene1.addWidget(self.canvas1)
        self.ui.graphicsView1.setScene(self.graphic_scene1)
        self.ratio1 = 0.995
        self.ratio2 = 0.995
        self.graphic_scene1.setSceneRect(0, 0, int(self.ratio2 * self.ui.graphicsView1.width()),
                                         int(self.ratio1 * self.ui.graphicsView1.height()))
        self.canvas1.resize(int(self.ratio2 * self.ui.graphicsView1.width()),
                            int(self.ratio1 * self.ui.graphicsView1.height()))
        self.canvas1.axes.axis('on')
        self.canvas1.axes.xaxis.set_ticks_position('top')
        self.canvas1.axes.xaxis.set_label_position('top')
        self.canvas1.axes.invert_yaxis()
        self.canvas1.axes.grid()
        self.ui.graphicsView1.show()

        self.line1 = dict(color='black', lw=1)
        self.cursor1 = widgets.Cursor(self.canvas1.axes, useblit=True, **self.line1)
        self.canvas1.mpl_connect('motion_notify_event', self.on_mouse_move)

        color_chemistry.gui_roicenter2=1
        self.ui.pushButton_2.clicked.connect(self.closegui)
        self.ui.pushButton.clicked.connect(self.save)

    def closeEvent(self, event):
        print('close window')
        super().closeEvent(event)
        event.ignore()

    def save(self):
        if self.yuyan == 'english':
            self.savepath0 = QFileDialog.getExistingDirectory(self.ui, "Please select the output folder")
        else:
            self.savepath0 = QFileDialog.getExistingDirectory(self.ui, "请选择输出文件夹")
        self.savepath = self.savepath0 + os.sep + 'ROI center line datay'
        if not os.path.exists(self.savepath):
            os.mkdir(self.savepath)
        self.t_save = time.strftime('%H-%M-%S', time.localtime())

        cv.imwrite(self.savepath+os.sep+self.t_save+ f'-{color_chemistry.ui.lineEdit_zibianliang.text()}'+'-roi_centerline_img.tiff',color_chemistry.roicenterimg3)
        if color_chemistry.ui.lineEdit_wuzhi.text() == '':
            self.data1D = {
                "Pixels": color_chemistry.thread_start2D.xdata2,
                "Value": color_chemistry.thread_start2D.middieline2,
                }
        else:
            self.data1D = {
                "Pixels": color_chemistry.thread_start2D.xdata2,
                f"{color_chemistry.ui.lineEdit_wuzhi.text()} {color_chemistry.ui.lineEdit_danwei.text()}": color_chemistry.thread_start2D.middieline2,
            }

        self.df1 = pd.DataFrame(pd.DataFrame.from_dict(self.data1D, orient='index').values.T,
                                columns=list(self.data1D.keys()))

        self.df1.to_csv(self.savepath + os.sep + self.t_save + f'-{color_chemistry.ui.lineEdit_zibianliang.text()}'+ '-centerline_ydata.csv', header=True, sep=',',
                        encoding="utf_8_sig")
        try:
            color_chemistry.fig5.savefig(self.savepath + os.sep + self.t_save + f'-{color_chemistry.ui.lineEdit_zibianliang.text()}'+'-centerline_ydata.tiff', dpi=350,
                              bbox_inches='tight', pad_inches=0.1)
        except:
            pass

    def on_mouse_move(self, event):
        try:
            if event.inaxes:
                if self.yuyan=='chinese':
                    self.ui.label_2.setText(f'坐标: (x:{event.xdata:.2f}, y:{event.ydata:.2f})')
                else:
                    self.ui.label_2.setText(f'Coordinate: (x:{event.xdata:.2f}, y:{event.ydata:.2f})')
        except:
            pass

    def closegui(self):
        self.ui.close()
        color_chemistry.gui_roicenter2 = 0

# Create a class to capture chemical images as a video
class ThreadTakeChemicalVideos(QThread):
    signal2 = Signal(int)
    signal1 = Signal(int)
    signal0=Signal(int)
    signal3=Signal(int)

    def __init__(self):
        super().__init__()

    def run(self):
        color_chemistry.switch_set_video_parameters = 0
        color_chemistry.switch_select_video_folder = 0
        color_chemistry.switch_startchemicalvideo = 1
        color_chemistry.ui.actionchemicalvideostart.setEnabled(False)
        color_chemistry.ui.actionchemicalvideostop.setEnabled(True)
        self.signal1.emit(1)
        while color_chemistry.switch_select_video_folder == 0:
            time.sleep(0.3)
            pass
        self.signal2.emit(1)
        while color_chemistry.switch_set_video_parameters2 == 0:
            time.sleep(0.3)
            pass
        color_chemistry.chemicalvideotime_list = []

        self.chemicalvideofps = 30
        self.chemicalvideoframenum = 0
        color_chemistry.chemicalvideoframenum=self.chemicalvideoframenum
        color_chemistry.getchemicalvideo_t_current=0
        color_chemistry.switch_startchemicalvideo = 1
        color_chemistry.startsignal = False
        while color_chemistry.startsignal==False:
            time.sleep(1)
            pass

        while color_chemistry.switch_startchemicalvideo == 1 and color_chemistry.getchemicalvideo_t_current<=color_chemistry.set_video2.timelong:
            if color_chemistry.chemicalimgok == False:
                break
            time.sleep(0.3)
            pass

        color_chemistry.picture_path0 = ''
        if color_chemistry.video_importpath=='' and color_chemistry.csv_path0 == '' and color_chemistry.picture_path0 == '':
            np.savetxt(
                color_chemistry.savepath_video + os.sep + f'{color_chemistry.chemicalvideo_t_file}_ChemicalVideoTimeSequence_{color_chemistry.ui.lineEdit_zibianliang.text()}.csv',
                np.array(color_chemistry.chemicalvideotime_list), delimiter=',',
                header='Time(s)', fmt='%f')
        try:
            color_chemistry.chemicalvideoout.release()
        except:
            pass
        self.signal3.emit(1)
        color_chemistry.ui.actionchemicalvideostart.setEnabled(True)
        color_chemistry.ui.actionchemicalvideostop.setEnabled(False)
        color_chemistry.switch_startchemicalvideo = 0

# Create a class to Fit various equation models
class ThreadFit(QThread):
    signal1 = Signal(int)
    signal0=Signal(int)
    signal2 = Signal(int)

    def __init__(self):
        super().__init__()

    def run(self):
        self.x=np.array(color_chemistry.jiaozhunx)
        self.y=np.array(color_chemistry.jiaozhuny)
        try:
            if color_chemistry.ui.comboBox.currentIndex()==0:
                X_with_intercept = sm.add_constant(self.x)

                model = sm.OLS(self.y, X_with_intercept).fit()

                self.intercept = model.params[0]
                self.slope = model.params[1]

                self.r_squared = model.rsquared
                self.adjusted_r_squared = model.rsquared_adj
                self.linex=np.linspace(np.min(self.x),np.max(self.x),1000)
                self.liney=self.slope*self.linex+self.intercept
                self.equation = f'y={self.slope}*x + ({self.intercept})'
                self.std_curve=f'(x-({self.intercept}))/{self.slope}'
                self.signal0.emit(1)

            elif color_chemistry.ui.comboBox.currentIndex()==1:
                initial_guess = [0, 7, 5, 1]
                params, covariance = curve_fit(
                    self.boltzmann,
                    self.x,
                    self.y,
                    p0=initial_guess,
                    maxfev=100000
                )

                self.A1_fit=params[0]
                self.A2_fit=params[1]
                self.x0_fit=params[2]
                self.dx_fit = params[3]
                print(params)
                self.curvetext='y=A2 + (A1 - A2) / (1 + e^((x - x0) / dx))'
                self.fit_params_text = f"A1 = {self.A1_fit:.4f}\nA2 = {self.A2_fit:.4f}\nx0 = {self.x0_fit:.4f}\ndx = {self.dx_fit:.4f}"

                y_pred = self.boltzmann(self.x, *params)

                ss_res = np.sum((self.y - y_pred) ** 2)
                ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
                self.r_squared = 1 - (ss_res / ss_tot)

                n = len(self.y)
                k = 1
                self.adjusted_r_squared = 1 - ((1 - self.r_squared) * (n - 1) / (n - k - 1))
                self.linex=np.linspace(np.min(self.x),np.max(self.x),1000)
                self.liney=self.boltzmann(self.linex, *params)
                self.equation = f'y={self.A2_fit}+({self.A1_fit}-({self.A2_fit}))/(1+e^((x-({self.x0_fit}))/{self.dx_fit}))'
                self.std_curve=f'{self.x0_fit}+({self.dx_fit})*np.log(-1+({self.A1_fit}-({self.A2_fit}))/(x-({self.A2_fit})))'
                self.signal0.emit(1)
            elif color_chemistry.ui.comboBox.currentIndex()==2:
                initial_guess = [1, 1, 1]
                params, covariance = curve_fit(
                    self.log3p1,
                    self.x,
                    self.y,
                    p0=initial_guess,
                    maxfev=100000
                )

                self.a_fit = params[0]
                self.b_fit = params[1]
                self.c_fit = params[2]
                self.curvetext = 'y=a-b*ln(x+c)'
                self.fit_params_text = f"a = {self.a_fit:.2f}\nb = {self.b_fit:.2f}\nc = {self.c_fit:.2f}"

                y_pred = self.log3p1(self.x, *params)

                ss_res = np.sum((self.y - y_pred) ** 2)
                ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
                self.r_squared = 1 - (ss_res / ss_tot)

                n = len(self.y)
                k = 1
                self.adjusted_r_squared = 1 - ((1 - self.r_squared) * (n - 1) / (n - k - 1))
                self.linex = np.linspace(np.min(self.x), np.max(self.x), 1000)
                self.liney = self.log3p1(self.linex, *params)
                self.equation=f'y={self.a_fit}-({self.b_fit})*ln(x+({self.c_fit}))'
                self.std_curve=f'-({self.c_fit})+np.exp(({self.a_fit}-x)/{self.b_fit})'
                self.signal0.emit(1)
        except:
            self.signal2.emit(1)

    def boltzmann(self, x, A1, A2, x0, dx):
        return A2 + (A1 - A2) / (1 + np.exp((x - x0) / dx))

    def log3p1(self,x,a,b,c):
        return a-b*np.log(x+c)

# Instantiate the class of the interface of selecting language
if __name__ =='__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app= QApplication([])
    selectyuyan = Select_yuyan()
    selectyuyan.ui.show()
    sys.exit(app.exec_())