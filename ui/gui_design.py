from PyQt4.QtCore import *
from PyQt4.QtGui import *
from . import gui_draw
from . import gui_vis
from . import gui_gamut
from . import gui_palette
import time
import datetime


class GUIDesign(QWidget):
    def __init__(self, color_model, dist_model=None, area_model=None, img_file=None, load_size=256,
                 win_size=256, save_all=True, my_mask_cent=0):
        # draw the layout
        QWidget.__init__(self)
        # main layout
        mainLayout = QHBoxLayout()
        self.setLayout(mainLayout)
        # gamut layout
        self.gamutWidget = gui_gamut.GUIGamut(gamut_size=160)
        gamutLayout = self.AddWidget(self.gamutWidget, 'ab Color Gamut')
        colorLayout = QVBoxLayout()

        colorLayout.addLayout(gamutLayout)
        mainLayout.addLayout(colorLayout)

        # palette
        self.customPalette = gui_palette.GUIPalette(grid_sz=(10, 1))
        self.usedPalette = gui_palette.GUIPalette(grid_sz=(10, 1))
        cpLayout = self.AddWidget(self.customPalette, 'Suggested colors')
        colorLayout.addLayout(cpLayout)
        upLayout = self.AddWidget(self.usedPalette, 'Recently used colors')
        colorLayout.addLayout(upLayout)

        sliderBox = QHBoxLayout()

        self.slider_value = QLabel()
        self.slider_value.setText('0')
        sliderLayout = self.AddWidget(self.slider_value, 'Value')
        sliderBox.addLayout(sliderLayout)

        if my_mask_cent == 0:
            self.slider = None
        else:
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setTickInterval(256*256*10*4)
            self.slider.setRange(-256*256*256*2, 256*256*256*1.2)
            self.slider.setTickPosition(50000)
            self.slider.setFixedWidth(self.customPalette.width())
            self.slider.setFixedHeight(25)
            # self.slider.setStyleSheet("background-color: grey")
            sliderLayout = self.AddWidget(self.slider, 'Portion')
            sliderBox.addLayout(sliderLayout)
            colorLayout.addLayout(sliderBox)

        self.colorPush = QPushButton()  # to visualize the selected color
        self.colorPush.setFixedWidth(self.customPalette.width())
        self.colorPush.setFixedHeight(25)
        self.colorPush.setStyleSheet("background-color: grey")
        colorPushLayout = self.AddWidget(self.colorPush, 'Color')
        colorLayout.addLayout(colorPushLayout)
        colorLayout.setAlignment(Qt.AlignTop)

        # drawPad layout
        drawPadLayout = QVBoxLayout()
        mainLayout.addLayout(drawPadLayout)
        self.drawWidget = gui_draw.GUIDraw(color_model, dist_model, area_model, load_size=load_size, win_size=win_size,
                                           my_mask_cent=my_mask_cent)
        drawPadLayout = self.AddWidget(self.drawWidget, 'Drawing Pad')
        mainLayout.addLayout(drawPadLayout)

        drawPadMenu = QHBoxLayout()

        self.bGray = QCheckBox("&Ground Truth")
        self.bGray.setToolTip('show Ground Truth image')

        self.bLoad = QPushButton('&Load')
        self.bLoad.setToolTip('load an input image')
        self.bSave = QPushButton("&Save")
        self.bSave.setToolTip('Save the current result.')

        drawPadMenu.addWidget(self.bGray)
        drawPadMenu.addWidget(self.bLoad)
        drawPadMenu.addWidget(self.bSave)

        drawPadLayout.addLayout(drawPadMenu)
        self.visWidget = gui_vis.GUI_VIS(win_size=win_size, scale=win_size / float(load_size))
        visWidgetLayout = self.AddWidget(self.visWidget, 'Result')
        mainLayout.addLayout(visWidgetLayout)

        self.bRestart = QPushButton("&Restart")
        self.bRestart.setToolTip('Restart the system')

        self.bQuit = QPushButton("&Quit")
        self.bQuit.setToolTip('Quit the system.')
        visWidgetMenu = QHBoxLayout()
        visWidgetMenu.addWidget(self.bRestart)

        visWidgetMenu.addWidget(self.bQuit)
        visWidgetLayout.addLayout(visWidgetMenu)

        self.drawWidget.update()
        self.visWidget.update()
        self.colorPush.clicked.connect(self.drawWidget.change_color)

        if my_mask_cent != 0:
            self.slider.sliderReleased.connect(self.set_mask_weight)
            self.slider.valueChanged.connect(self.slider_value_change)
            # self.slider.sliderReleased.connect(self.slider_value_change)
            self.connect(self.drawWidget, SIGNAL('update_slider_position'), self.slider_value_set)
        # color indicator
        self.connect(self.drawWidget, SIGNAL('update_color'), self.colorPush.setStyleSheet)
        # update result
        self.connect(self.drawWidget, SIGNAL('update_result'), self.visWidget.update_result)
        self.connect(self.visWidget, SIGNAL('update_color'), self.gamutWidget.set_ab)
        self.connect(self.visWidget, SIGNAL('update_color'), self.drawWidget.set_color)
        # update gamut
        self.connect(self.drawWidget, SIGNAL('update_gamut'), self.gamutWidget.set_gamut)
        self.connect(self.drawWidget, SIGNAL('update_ab'), self.gamutWidget.set_ab)
        self.connect(self.gamutWidget, SIGNAL('update_color'), self.drawWidget.set_color)
        # connect palette
        self.connect(self.drawWidget, SIGNAL('suggest_colors'), self.customPalette.set_colors)
        # self.connect(self.drawWidget, SIGNAL('change_color_id'), self.customPalette.update_color_id)
        self.connect(self.customPalette, SIGNAL('update_color'), self.drawWidget.set_color)
        self.connect(self.customPalette, SIGNAL('update_color'), self.gamutWidget.set_ab)

        self.connect(self.drawWidget, SIGNAL('used_colors'), self.usedPalette.set_colors)
        self.connect(self.usedPalette, SIGNAL('update_color'), self.drawWidget.set_color)
        self.connect(self.usedPalette, SIGNAL('update_color'), self.gamutWidget.set_ab)
        # menu events
        self.bGray.setChecked(True)
        self.bRestart.clicked.connect(self.reset)
        self.bQuit.clicked.connect(self.quit)
        self.bGray.toggled.connect(self.enable_gray)
        self.bSave.clicked.connect(self.save)
        self.bLoad.clicked.connect(self.load)

        self.start_t = time.time()

        if img_file is not None:
            self.drawWidget.init_result(img_file)

    def AddWidget(self, widget, title):
        widgetLayout = QVBoxLayout()
        widgetBox = QGroupBox()
        widgetBox.setTitle(title)
        vbox_t = QVBoxLayout()
        vbox_t.addWidget(widget)
        widgetBox.setLayout(vbox_t)
        widgetLayout.addWidget(widgetBox)

        return widgetLayout

    def nextImage(self):
        self.drawWidget.nextImage()

    def slider_value_change(self):
        size = self.slider.value()
        self.slider_value.setText(str(round(float(size)/(256*256*256), 3)))

    def slider_value_set(self, test):
        self.slider.setValue(test)
        # print(test)
        self.slider_value_change()

    def set_mask_weight(self):
        suffix = datetime.datetime.now().strftime("%y%m%d_%H")
        save_path = "_".join([suffix, "UserFile.txt"])
        f = open(save_path, "a")
        f.write("ST User Changed Slider \n")
        f.close()
        self.drawWidget.set_weighted_mask(self.slider.value())

    def reset(self):
        # self.start_t = time.time()
        print('============================reset all=========================================')
        self.visWidget.reset()
        self.gamutWidget.reset()
        self.customPalette.reset()
        self.usedPalette.reset()
        self.drawWidget.reset()
        self.update()
        self.colorPush.setStyleSheet("background-color: grey")

    def enable_gray(self):
        self.drawWidget.enable_gray()

    def quit(self):
        print('time spent = %3.3f' % (time.time() - self.start_t))
        suffix = datetime.datetime.now().strftime("%y%m%d_%H")
        save_path = "_".join([suffix, "UserFile.txt"])
        f = open(save_path, "a")
        f.write('time spent = %3.3f \n' % (time.time() - self.start_t))
        f.close()
        self.close()

    def save(self):
        print('time spent = %3.3f' % (time.time() - self.start_t))
        suffix = datetime.datetime.now().strftime("%y%m%d_%H")
        save_path = "_".join([suffix, "UserFile.txt"])
        f = open(save_path, "a")
        f.write('time spent = %3.3f \n' % (time.time() - self.start_t))
        f.close()
        # self.drawWidget.save_result()
        self.drawWidget.nextImage()

    def load(self):
        # self.drawWidget.load_image()
        machine='win'
        if machine=='unix':
            imgs_path = unicode(QFileDialog.getExistingDirectory(self, 'load an folder'))
        elif machine=='win':
            imgs_path = (QFileDialog.getExistingDirectory(self, 'load an folder'))
        print(imgs_path)
        self.drawWidget.get_batches(imgs_path)

    def change_color(self):
        print('change color')
        self.drawWidget.change_color(use_suggest=True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            self.reset()

        if event.key() == Qt.Key_Q:
            self.save()
            self.quit()

        if event.key() == Qt.Key_S:
            self.save()

        if event.key() == Qt.Key_G:
            self.bGray.toggle()

        if event.key() == Qt.Key_L:
            self.load()
