# -*- coding: utf-8 -*-

"""
Based on https://github.com/hkchengrex/MiVOS/tree/MiVOS-STCN 
(which is based on https://github.com/seoungwugoh/ivs-demo)

This version is much simplified. 
In this repo, we don't have
- local control
- fusion module
- undo
- timers

but with XMem as the backbone and is more memory (for both CPU and GPU) friendly
"""
from numba import jit
import functools
from tqdm import tqdm
import os
import cv2
import sys
import glob
import re
from PIL import Image
# fix conflicts between qt5 and cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
import json
import numpy as np
import torch
from queue import Queue
from PyQt5.QtWidgets import (QWidget, QApplication, QComboBox, QCheckBox,
    QHBoxLayout, QLabel, QPushButton, QTextEdit, QFileDialog,
    QPlainTextEdit, QVBoxLayout, QSizePolicy, QButtonGroup, QSlider, QShortcut, QRadioButton, QLineEdit)
import pdb
from PyQt5.QtGui import QPixmap, QKeySequence, QImage, QTextCursor, QIcon
from PyQt5.QtCore import Qt, QTimer
from model.network import XMem
import matplotlib.pyplot as plt
from inference.inference_core import InferenceCore
from .s2m_controller import S2MController
from .fbrs_controller import FBRSController

from .interactive_utils import *
from .interaction import *
from .resource_manager import ResourceManager
from .gui_utils import *

sys.path.insert(0,'/home/m11002125/ViTPose/')
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
    

from alphapose.utils.writer import DataWriter



@jit(nopython=True)
def get_bbox_from_mask(current_mask):
    
    # # print(current_mask.shape)
    # min_row = 2000
    # min_col = 2000
    # max_row = -1
    # max_col = -1
    # for row in range(0,len(current_mask)):  # num of row (h)
    #     for col in range(0,len(current_mask[0])):  # num of col (w)
    #         if (current_mask[row][col] == 1):
    #             min_row = min(min_row,row)
    #             min_col = min(min_col, col)
    #             max_row = max(max_row,row)
    #             max_col = max(max_col,col)
    # min_col = min_col - 5 if min_col - 5 > 0 else min_col 
    # min_row = min_row - 5 if min_row - 5 > 0 else min_row
    
    # max_row = max_row + 5 if max_row + 5 < len(current_mask) else max_row
    # max_col = max_col + 5 if max_col + 5 < len(current_mask[0]) else max_col
    # np.where 函数返回所有满足条件的元素的索引
    rows, cols = np.where(current_mask == 1)
    
    # 如果没有找到满足条件的元素，返回 None
    if len(rows) == 0 or len(cols) == 0:
        return None
    
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    
    # 为边界框增加一些填充
    min_col = max(min_col - 5, 0)
    min_row = max(min_row - 5, 0)
    max_row = min(max_row + 5, current_mask.shape[0] - 1)
    max_col = min(max_col + 5, current_mask.shape[1] - 1)
    
    return min_col,min_row,max_col,max_row

@jit(nopython=True)
def get_bbox_from_mask2(current_mask):
    
    # print(current_mask.shape)
    min_row = 2000
    min_col = 2000
    max_row = -1
    max_col = -1
    for row in range(0,len(current_mask)):  # num of row (h)
        for col in range(0,len(current_mask[0])):  # num of col (w)
            if (current_mask[row][col] == 3):
                min_row = min(min_row,row)
                min_col = min(min_col, col)
                max_row = max(max_row,row)
                max_col = max(max_col,col)
    min_col = min_col - 5 if min_col - 5 > 0 else min_col 
    min_row = min_row - 5 if min_row - 5 > 0 else min_row
    
    max_row = max_row + 5 if max_row + 5 < len(current_mask) else max_row
    max_col = max_col + 5 if max_col + 5 < len(current_mask[0]) else max_col
    return min_col,min_row,max_col,max_row

class App(QWidget):  # net : XMem -> 表示net一定是XMem物件
    def __init__(self, net: XMem,   
                resource_manager: ResourceManager, 
                s2m_ctrl:S2MController, 
                fbrs_ctrl:FBRSController, config, pose_config):
        super().__init__()
        self.video_path = config['video_path']
        stream = cv2.VideoCapture(self.video_path) # 將影片檔案讀入
        assert stream.isOpened(), 'Cannot capture source'
        self.path = self.video_path                               # 影片路徑
        self.datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)) # 查看多少個frame
        self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))       # fourcc:  編碼的種類 EX:(MPEG4 or H264)
        self.fps = stream.get(cv2.CAP_PROP_FPS)                  # 查看 FPS
        self.frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 影片長寬
        # self.videoinfo = {'fourcc': self.fourcc, 'fps': self.fps, 'frameSize': self.frameSize} # 影片資訊
        stream.release()
        self.initialized = False
        self.num_objects = config['num_objects']
        self.s2m_controller = s2m_ctrl
        self.fbrs_controller = fbrs_ctrl 
        self.pose_config = pose_config # pose dataset configuration
        self.config = config   # XMem config
        self.batchSize = 1
        # pdb.set_trace()
        self.device = self.config['device']
        leftover = 0
        
        if (self.datalen) % self.batchSize:
            leftover = 1
        self.num_batches = self.datalen // self.batchSize + leftover
        
        # pdb.set_trace()
        # pose_model
        self.pose_model = init_pose_model(pose_config.pose_config, pose_config.pose_checkpoint, device=pose_config.device.lower())
        
        self.dataset = self.pose_model.cfg.data['test']['type']
        self.dataset_info = self.pose_model.cfg.data['test'].get('dataset_info', None)
        if self.dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            self.dataset_info = DatasetInfo(self.dataset_info)
                
        # pdb.set_trace()
        self._input_size = self.pose_model.cfg.data_cfg.image_size          # 預設 姿態偵測模型的輸入維度
        self._output_size = self.pose_model.cfg.data_cfg.heatmap_size       # 預設 姿態偵測模型的輸出維度
        
        
        self.processor = InferenceCore(net, config)                         # XMem 模型本體
        self.processor.set_all_labels(list(range(1, self.num_objects+1)))   # 決定追蹤的物體數
        
    
        '''resource_manager:
            控制跟影片讀取寫出相關事項
            Ex: 
                1. mask的輸出路徑
                2. 影像的資訊紀錄
                3. 取得已輸出過的mask
        '''
        self.res_man = resource_manager 
        self.num_frames = len(self.res_man) # 影像總幀數
        # resource_manager，可以控制影像輸入的大小，若是輸入的影像太大，我們會將它縮小，因為高畫質的計算量太重。
        self.height, self.width = self.res_man.h, self.res_man.w  
        self.videoinfo = {'fourcc': self.fourcc, 'fps': self.fps, 'frameSize': (self.width,self.height)} # 影片資訊

        # set window
        self.setWindowTitle('XMem Demo')
        self.setGeometry(100, 100, self.width, self.height+100)
        self.setWindowIcon(QIcon('docs/icon.png'))

        # some buttons
        self.play_button = QPushButton('Play Video')
        self.play_button.clicked.connect(self.on_play_video)
        self.commit_button = QPushButton('Commit')
        self.commit_button.clicked.connect(self.on_commit)

        self.forward_run_button = QPushButton('Forward Propagate')
        self.forward_run_button.clicked.connect(self.on_forward_propagation)
        self.forward_run_button.setMinimumWidth(200)

        self.backward_run_button = QPushButton('Backward Propagate')
        self.backward_run_button.clicked.connect(self.on_backward_propagation)
        self.backward_run_button.setMinimumWidth(200)

        self.reset_button = QPushButton('Reset Frame')
        self.reset_button.clicked.connect(self.on_reset_mask)
        
        self.next_frame_button = QPushButton('next_frame')
        self.next_frame_button.clicked.connect(self.on_next_frame)
        
        self.last_frame_button = QPushButton('last frame')
        self.last_frame_button.clicked.connect(self.on_prev_frame)
        
        self.pose_button = QPushButton('Pose Estimation')
        self.pose_button.clicked.connect(self.pose_estimate)

        # LCD
        self.lcd = QTextEdit()
        self.lcd.setReadOnly(True)
        self.lcd.setMaximumHeight(28)
        self.lcd.setMaximumWidth(120)
        self.lcd.setText('{: 4d} / {: 4d}'.format(0, self.num_frames-1))

        # timeline slider
        self.tl_slider = QSlider(Qt.Horizontal)
        self.tl_slider.valueChanged.connect(self.tl_slide)
        self.tl_slider.setMinimum(0)
        self.tl_slider.setMaximum(self.num_frames-1)
        self.tl_slider.setValue(0)
        self.tl_slider.setTickPosition(QSlider.TicksBelow)
        self.tl_slider.setTickInterval(1)
        
        # brush size slider
        self.brush_label = QLabel()
        self.brush_label.setAlignment(Qt.AlignCenter)
        self.brush_label.setMinimumWidth(100)
        
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.valueChanged.connect(self.brush_slide)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(100)
        self.brush_slider.setValue(3)
        self.brush_slider.setTickPosition(QSlider.TicksBelow)
        self.brush_slider.setTickInterval(2)
        self.brush_slider.setMinimumWidth(300)

        # combobox
        self.combo = QComboBox(self)
        self.combo.addItem("davis")
        self.combo.addItem("fade")
        self.combo.addItem("light")
        self.combo.addItem("popup")
        self.combo.addItem("layered")
        self.combo.currentTextChanged.connect(self.set_viz_mode)

        self.save_visualization_checkbox = QCheckBox(self)
        self.save_visualization_checkbox.toggled.connect(self.on_save_visualization_toggle)
        self.save_visualization_checkbox.setChecked(False)
        self.save_visualization = False

        # Radio buttons for type of interactions
        self.curr_interaction = 'Click'
        self.interaction_group = QButtonGroup()
        self.radio_fbrs = QRadioButton('Click')
        self.radio_s2m = QRadioButton('Scribble')
        self.radio_free = QRadioButton('Free')
        self.interaction_group.addButton(self.radio_fbrs)
        self.interaction_group.addButton(self.radio_s2m)
        self.interaction_group.addButton(self.radio_free)
        self.radio_fbrs.toggled.connect(self.interaction_radio_clicked)
        self.radio_s2m.toggled.connect(self.interaction_radio_clicked)
        self.radio_free.toggled.connect(self.interaction_radio_clicked)
        self.radio_fbrs.toggle()

        # Main canvas -> QLabel
        self.main_canvas = QLabel()
        self.main_canvas.setSizePolicy(QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        self.main_canvas.setAlignment(Qt.AlignCenter)
        self.main_canvas.setMinimumSize(100, 100)

        self.main_canvas.mousePressEvent = self.on_mouse_press  # 滑鼠點擊
        self.main_canvas.mouseMoveEvent = self.on_mouse_motion
        self.main_canvas.setMouseTracking(True) # Required for all-time tracking
        self.main_canvas.mouseReleaseEvent = self.on_mouse_release

        # Minimap -> Also a QLbal
        self.minimap = QLabel()
        self.minimap.setSizePolicy(QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        self.minimap.setAlignment(Qt.AlignTop)
        self.minimap.setMinimumSize(100, 100)

        # Zoom-in buttons
        self.zoom_p_button = QPushButton('Zoom +')
        self.zoom_p_button.clicked.connect(self.on_zoom_plus)
        self.zoom_m_button = QPushButton('Zoom -')
        self.zoom_m_button.clicked.connect(self.on_zoom_minus)

        # Parameters setting
        self.clear_mem_button = QPushButton('Clear memory')
        self.clear_mem_button.clicked.connect(self.on_clear_memory)

        self.work_mem_gauge, self.work_mem_gauge_layout = create_gauge('Working memory size')
        self.long_mem_gauge, self.long_mem_gauge_layout = create_gauge('Long-term memory size')
        self.gpu_mem_gauge, self.gpu_mem_gauge_layout = create_gauge('GPU mem. (all processes, w/ caching)')
        self.torch_mem_gauge, self.torch_mem_gauge_layout = create_gauge('GPU mem. (used by torch, w/o caching)')

        self.update_memory_size()   # 
        self.update_gpu_usage()     # 更新GPU顯存使用狀況

        self.work_mem_min, self.work_mem_min_layout = create_parameter_box(1, 100, 'Min. working memory frames', 
                                                        callback=self.on_work_min_change)
        self.work_mem_max, self.work_mem_max_layout = create_parameter_box(2, 100, 'Max. working memory frames', 
                                                        callback=self.on_work_max_change)
        self.long_mem_max, self.long_mem_max_layout = create_parameter_box(1000, 100000, 
                                                        'Max. long-term memory size', step=1000, callback=self.update_config)
        self.num_prototypes_box, self.num_prototypes_box_layout = create_parameter_box(32, 1280, 
                                                        'Number of prototypes', step=32, callback=self.update_config)
        self.mem_every_box, self.mem_every_box_layout = create_parameter_box(1, 100, 'Memory frame every (r)', 
                                                        callback=self.update_config)
        self.pose_start = QLineEdit(self)
        self.pose_start.setText('start frame pose estimation')
        self.pose_end = QLineEdit(self)
        self.pose_end.setText('end frame pose estimation')

        self.work_mem_min.setValue(self.processor.memory.min_mt_frames)
        self.work_mem_max.setValue(self.processor.memory.max_mt_frames)
        self.long_mem_max.setValue(self.processor.memory.max_long_elements)
        self.num_prototypes_box.setValue(self.processor.memory.num_prototypes)
        self.mem_every_box.setValue(self.processor.mem_every)

        # import mask/layer
        self.import_mask_button = QPushButton('Import mask')
        self.import_mask_button.clicked.connect(self.on_import_mask)
        self.import_layer_button = QPushButton('Import layer')
        self.import_layer_button.clicked.connect(self.on_import_layer)
        self.import_loadmask_button = QPushButton('Load mask')
        self.import_loadmask_button.clicked.connect(self.on_load_mask)
        

        # Console on the GUI
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(100)
        self.console.setMaximumHeight(100)

        # navigator
        navi = QHBoxLayout()
        navi.addWidget(self.lcd)
        navi.addWidget(self.play_button)

        interact_subbox = QVBoxLayout()
        interact_topbox = QHBoxLayout()
        interact_botbox = QHBoxLayout()
        interact_topbox.setAlignment(Qt.AlignCenter)
        interact_topbox.addWidget(self.radio_s2m)
        interact_topbox.addWidget(self.radio_fbrs)
        interact_topbox.addWidget(self.radio_free)
        interact_topbox.addWidget(self.brush_label)
        interact_botbox.addWidget(self.brush_slider)
        interact_subbox.addLayout(interact_topbox)
        interact_subbox.addLayout(interact_botbox)
        navi.addLayout(interact_subbox)

        navi.addStretch(1)
        navi.addWidget(self.next_frame_button)
        navi.addWidget(self.last_frame_button)
        navi.addWidget(self.reset_button)

        navi.addStretch(1)
        navi.addWidget(QLabel('Overlay Mode'))
        navi.addWidget(self.combo)
        navi.addWidget(QLabel('Save overlay during propagation'))
        navi.addWidget(self.save_visualization_checkbox)
        navi.addStretch(1)
        navi.addWidget(self.commit_button)
        navi.addWidget(self.forward_run_button)
        navi.addWidget(self.backward_run_button)
        navi.addWidget(self.pose_button)

        # Drawing area, main canvas and minimap
        draw_area = QHBoxLayout()
        draw_area.addWidget(self.main_canvas, 4)

        # Minimap area
        minimap_area = QVBoxLayout()
        minimap_area.setAlignment(Qt.AlignTop)
        mini_label = QLabel('Minimap')
        mini_label.setAlignment(Qt.AlignTop)
        minimap_area.addWidget(mini_label)

        # Minimap zooming
        minimap_ctrl = QHBoxLayout()
        minimap_ctrl.setAlignment(Qt.AlignTop)
        minimap_ctrl.addWidget(self.zoom_p_button)
        minimap_ctrl.addWidget(self.zoom_m_button)
        minimap_area.addLayout(minimap_ctrl)
        minimap_area.addWidget(self.minimap)

        # Parameters 
        minimap_area.addLayout(self.work_mem_gauge_layout)
        minimap_area.addLayout(self.long_mem_gauge_layout)
        minimap_area.addLayout(self.gpu_mem_gauge_layout)
        minimap_area.addLayout(self.torch_mem_gauge_layout)
        minimap_area.addWidget(self.clear_mem_button)
        minimap_area.addLayout(self.work_mem_min_layout)
        minimap_area.addLayout(self.work_mem_max_layout)
        minimap_area.addLayout(self.long_mem_max_layout)
        minimap_area.addLayout(self.num_prototypes_box_layout)
        minimap_area.addLayout(self.mem_every_box_layout)

        # import mask/layer
        import_area = QHBoxLayout()
        import_area.setAlignment(Qt.AlignTop)
        import_area.addWidget(self.import_loadmask_button)
        import_area.addWidget(self.import_mask_button)
        import_area.addWidget(self.import_layer_button)
        minimap_area.addLayout(import_area)

        # console
        minimap_area.addWidget(self.console)
        minimap_area.addWidget(self.pose_start)
        minimap_area.addWidget(self.pose_end)

        draw_area.addLayout(minimap_area, 1)

        layout = QVBoxLayout()
        layout.addLayout(draw_area)
        layout.addWidget(self.tl_slider)
        layout.addLayout(navi)
        self.setLayout(layout)
        
        '''--------------------function region ------------------------'''
        """
        det_queue: the buffer storing human detection results from mask
        pose_queue: the buffer storing post-processed cropped human image for pose estimation
        """
        # 將mask轉換成bbox，存入該
        self.det_queue = [ [] for i in range(self.num_frames+1)] # 最後全None
        self.pose_queue = Queue(maxsize=self.num_frames * 10)
        
        # timer to play video
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.on_play_video_timer)

        # timer to update GPU usage
        self.gpu_timer = QTimer()
        self.gpu_timer.setSingleShot(False)
        self.gpu_timer.timeout.connect(self.on_gpu_timer)
        self.gpu_timer.setInterval(2000)
        self.gpu_timer.start()

        # current frame info
        self.curr_frame_dirty = False
        self.current_image = np.zeros((self.height, self.width, 3), dtype=np.uint8) 
        self.current_image_torch = None
        self.current_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.current_prob = torch.zeros((self.num_objects, self.height, self.width), dtype=torch.float).cuda()

        # initialize visualization
        self.viz_mode = 'davis'
        self.vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.brush_vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.brush_vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.cursur = 0
        self.on_showing = None

        # Zoom parameters
        self.zoom_pixels = 150
        
        # initialize action
        self.interaction = None
        self.pressed = False
        self.right_click = False
        self.current_object = 1
        self.last_ex = self.last_ey = 0

        self.propagating = False

        # Objects shortcuts
        for i in range(1, self.num_objects+1):
            QShortcut(QKeySequence(str(i)), self).activated.connect(functools.partial(self.hit_number_key, i))

        # <- and -> shortcuts
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(self.on_prev_frame)
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(self.on_next_frame)

        self.interacted_prob = None
        self.overlay_layer = None
        self.overlay_layer_torch = None

        # the object id used for popup/layered overlay
        self.vis_target_objects = [1]
        # try to load the default overlay
        # self._try_load_layer('./docs/ECCV-logo.png')
 
        self.load_current_image_mask() # read image and load mask
        self.show_current_frame()      # read cursor and render image on gui.
        self.show()                    # show

        self.console_push_text('Initialized.')
        self.initialized = True

    def resizeEvent(self, event):      # 實際開始顯示，GUI的顯示
        self.show_current_frame()

    def console_push_text(self, text):           # console
        self.console.moveCursor(QTextCursor.End)
        self.console.insertPlainText(text+'\n')

    def interaction_radio_clicked(self, event):  # radio button
        self.last_interaction = self.curr_interaction
        if self.radio_s2m.isChecked():
            self.curr_interaction = 'Scribble'
            self.brush_size = 3
            self.brush_slider.setDisabled(True)
        elif self.radio_fbrs.isChecked():
            self.curr_interaction = 'Click'
            self.brush_size = 3
            self.brush_slider.setDisabled(True)
        elif self.radio_free.isChecked():
            self.brush_slider.setDisabled(False)
            self.brush_slide()
            self.curr_interaction = 'Free'
        if self.curr_interaction == 'Scribble':
            self.commit_button.setEnabled(True)
        else:
            self.commit_button.setEnabled(False)

    def load_current_image_mask(self, no_mask=False):
        # pdb.set_trace()
        self.current_image = self.res_man.get_image(self.cursur) # cursor 決定第幾幀 # 讀取圖片
        self.current_image_torch = None

        if not no_mask:
            loaded_mask = self.res_man.get_mask(self.cursur) # 某影片第一次執行時，一開始沒有mask。
            if loaded_mask is None:
                self.current_mask.fill(0)                    # 將Mask先填滿0
            else:
                self.current_mask = loaded_mask.copy()
            self.current_prob = None

    def load_current_torch_image_mask(self, no_mask=False):
        if self.current_image_torch is None:
            self.current_image_torch, self.current_image_torch_no_norm = image_to_torch(self.current_image)

        if self.current_prob is None and not no_mask:
            self.current_prob = index_numpy_to_one_hot_torch(self.current_mask, self.num_objects+1).cuda()

    def compose_current_im(self):
        # self.viz: 顯示在canva上的內容
        self.viz = get_visualization(self.viz_mode, self.current_image, self.current_mask, 
                            self.overlay_layer, self.vis_target_objects)
        if self.cursur==0 or self.cursur==self.datalen-1:
            # pdb.set_trace()
            self.det_queue[self.cursur] = []
        # pdb.set_trace()

    def update_interact_vis(self):
        # Update the interactions without re-computing the overlay
        # pdb.set_trace()
        height, width, channel = self.viz.shape # self.viz: 顯示在canva上的物件
        bytesPerLine = 3 * width
        # 底下的部分為人機互動造成顯示的變化
        vis_map = self.vis_map                          # click 產生出的mask
        vis_alpha = self.vis_alpha                      # alpha: 一種係數
        brush_vis_map = self.brush_vis_map              # 筆畫(brush) 產生出的mask
        brush_vis_alpha = self.brush_vis_alpha          # alpha: 一種係數
        # alpha:  blending coefficient slider adjusts the intensity of all predicted masks.
        # 詳情請看 : https://github.com/SamsungLabs/fbrs_interactive_segmentation
        # self.viz_with_stroke: 'numpy.ndarray' stroke: 筆刷之意
        self.viz_with_stroke = self.viz*(1-vis_alpha) + vis_map*vis_alpha
        self.viz_with_stroke = self.viz_with_stroke*(1-brush_vis_alpha) + brush_vis_map*brush_vis_alpha
        self.viz_with_stroke = self.viz_with_stroke.astype(np.uint8)
        # self.viz_with_stroke: 人機互動的結果
        # 顯示的圖(self.viz_with_stroke.data)已計算出，放入QImage物件來顯示
        qImg = QImage(self.viz_with_stroke.data, width, height, bytesPerLine, QImage.Format_RGB888) # 
        self.main_canvas.setPixmap(QPixmap(qImg.scaled(self.main_canvas.size(),    # canvas 的顯示
                Qt.KeepAspectRatio, Qt.FastTransformation)))

        self.main_canvas_size = self.main_canvas.size()
        self.image_size = qImg.size()

    def update_minimap(self):
        ex, ey = self.last_ex, self.last_ey
        r = self.zoom_pixels//2
        ex = int(round(max(r, min(self.width-r, ex))))
        ey = int(round(max(r, min(self.height-r, ey))))

        patch = self.viz_with_stroke[ey-r:ey+r, ex-r:ex+r, :].astype(np.uint8)

        height, width, channel = patch.shape
        bytesPerLine = 3 * width
        qImg = QImage(patch.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.minimap.setPixmap(QPixmap(qImg.scaled(self.minimap.size(),
                Qt.KeepAspectRatio, Qt.FastTransformation)))

    def update_current_image_fast(self):
        # fast path, uses gpu. Changes the image in-place to avoid copying
        # self.viz: 顯示在canva上的物件
        self.viz = get_visualization_torch(self.viz_mode, self.current_image_torch_no_norm, 
                    self.current_prob, self.overlay_layer_torch, self.vis_target_objects)
        if self.save_visualization:
            self.res_man.save_visualization(self.cursur, self.viz)

        height, width, channel = self.viz.shape
        bytesPerLine = 3 * width

        qImg = QImage(self.viz.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.main_canvas.setPixmap(QPixmap(qImg.scaled(self.main_canvas.size(),
                Qt.KeepAspectRatio, Qt.FastTransformation)))

    def show_current_frame(self, fast=False):
        # Re-compute overlay and show the image
        if fast:
            self.update_current_image_fast()
        else:
            self.compose_current_im()          # 計算出要顯示的圖片(原圖與mask進行overlay),之後把mask轉成bbox儲存到detqueue
            self.update_interact_vis()         # 人機互動後的結果,計算出的圖片，顯示在main_canva
            self.update_minimap()              # 小圖片的更新

        self.lcd.setText('{: 3d} / {: 3d}'.format(self.cursur, self.num_frames-1))
        self.tl_slider.setValue(self.cursur)

    def pixel_pos_to_image_pos(self, x, y):
        # Un-scale and un-pad the label coordinates into image coordinates
        oh, ow = self.image_size.height(), self.image_size.width()
        nh, nw = self.main_canvas_size.height(), self.main_canvas_size.width()

        h_ratio = nh/oh
        w_ratio = nw/ow
        dominate_ratio = min(h_ratio, w_ratio)

        # Solve scale
        x /= dominate_ratio
        y /= dominate_ratio

        # Solve padding
        fh, fw = nh/dominate_ratio, nw/dominate_ratio
        x -= (fw-ow)/2
        y -= (fh-oh)/2

        return x, y

    def is_pos_out_of_bound(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)

        out_of_bound = (
            (x < 0) or
            (y < 0) or
            (x > self.width-1) or 
            (y > self.height-1)
        )

        return out_of_bound

    def get_scaled_pos(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)

        x = max(0, min(self.width-1, x))
        y = max(0, min(self.height-1, y))

        return x, y

    def clear_visualization(self):
        self.vis_map.fill(0)
        self.vis_alpha.fill(0)

    def reset_this_interaction(self):
        self.complete_interaction()
        self.clear_visualization()
        self.interaction = None
        if self.fbrs_controller is not None:
            self.fbrs_controller.unanchor()

    def set_viz_mode(self):
        self.viz_mode = self.combo.currentText()
        self.show_current_frame()

    def save_current_mask(self):
        # save mask to hard disk
        self.res_man.save_mask(self.cursur, self.current_mask)

    def tl_slide(self):  # slider 數值變化
        # if we are propagating, the on_run function will take care of everything
        # don't do duplicate work here
        if not self.propagating:
            if self.curr_frame_dirty:
                self.save_current_mask()
            self.curr_frame_dirty = False

            self.reset_this_interaction()
            self.cursur = self.tl_slider.value()
            self.load_current_image_mask()
            self.show_current_frame()

    def brush_slide(self):
        self.brush_size = self.brush_slider.value()
        self.brush_label.setText('Brush size: %d' % self.brush_size)
        try:
            if type(self.interaction) == FreeInteraction:
                self.interaction.set_size(self.brush_size)
        except AttributeError:
            # Initialization, forget about it
            pass

    def on_forward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
        else:
            self.propagate_fn = self.on_next_frame
            self.backward_run_button.setEnabled(False)
            self.forward_run_button.setText('Pause Propagation')
            self.on_propagation()

    def on_backward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
        else:
            self.propagate_fn = self.on_prev_frame
            self.forward_run_button.setEnabled(False)
            self.backward_run_button.setText('Pause Propagation')
            self.on_propagation()

    def on_pause(self):
        self.propagating = False
        self.forward_run_button.setEnabled(True)
        self.backward_run_button.setEnabled(True)
        self.clear_mem_button.setEnabled(True)
        self.forward_run_button.setText('Forward Propagate')
        self.backward_run_button.setText('Backward Propagate')
        self.console_push_text('Propagation stopped.')

    def on_propagation(self):
        # start to propagate
        self.load_current_torch_image_mask()
        self.show_current_frame(fast=True)

        self.console_push_text('Propagation started.')
        self.current_prob = self.processor.step(self.current_image_torch, self.current_prob[1:])
        self.current_mask = torch_prob_to_numpy_mask(self.current_prob)
        
        (min_col,min_row,max_col,max_row) = get_bbox_from_mask(self.current_mask)
        if len(self.det_queue[self.cursur]) > 1:
            self.det_queue[self.cursur] = []
        # self.det_queue[self.cursur].append(self.current_image)
        self.det_queue[self.cursur].append(str(self.cursur) + '.jpg')
        self.det_queue[self.cursur].append(torch.from_numpy(np.array([[min_col,min_row,max_col,max_row]])))
        self.det_queue[self.cursur].append(torch.tensor([[1.]])) # bbox has no score , let it be a 100% accuracy
        self.det_queue[self.cursur].append(torch.tensor([[0.]])) # bbox has no ids 
        inps = torch.zeros(torch.from_numpy(np.array([[min_col,min_row,max_col,max_row]])).size(0), 3, *self._input_size)
        self.det_queue[self.cursur].append(inps) #
        cropped_boxes = torch.zeros(torch.from_numpy(np.array([[min_col,min_row,max_col,max_row]])).size(0), 4)
        self.det_queue[self.cursur].append(cropped_boxes) # bbox has no ids 
        # self.det_queue[self.cursur].append(self.current_mask) # bbox has no ids 
        
       # clear
        self.interacted_prob = None
        self.reset_this_interaction()
        
        self.propagating = True
        self.clear_mem_button.setEnabled(False)
        # propagate till the end
        while self.propagating:  
            self.propagate_fn()  # 調整cursor 與 slider 的數值

            self.load_current_image_mask(no_mask=True) # 上一行程式已經更新cursor了,然後讀取該cursor的圖片與mask. mask不一定存在
            self.load_current_torch_image_mask(no_mask=True) #將圖片(Numpy)轉成張量(Tensor)

            self.current_prob = self.processor.step(self.current_image_torch) # 將新張量(Tensor)丟入XMem模型,得到current_prob(輸出)
            self.current_mask = torch_prob_to_numpy_mask(self.current_prob) # 將模型輸出轉換成mask(Numpy)

            self.save_current_mask() # 將mask存成圖片 
            # self.get_bbox_from_mask()
            # pdb.set_trace()
            (min_col,min_row,max_col,max_row) = get_bbox_from_mask(self.current_mask)
            # 將辨識完的結果 處理完後丟到 Queue 裡面
            # self.wait_and_put(self.det_queue, (orig_imgs[k], im_names[k], boxes_k, scores[dets[:, 0] == k], ids[dets[:, 0] == k], inps, cropped_boxes))
            if len(self.det_queue[self.cursur]) > 1:
                self.det_queue[self.cursur] = []
            # pdb.set_trace()
            # self.det_queue[self.cursur].append(self.current_image)
            self.det_queue[self.cursur].append(str(self.cursur) + '.jpg')
            self.det_queue[self.cursur].append(torch.from_numpy(np.array([[min_col,min_row,max_col,max_row]])))
            self.det_queue[self.cursur].append(torch.tensor([[1.]])) # bbox has no score , let it be a 100% accuracy
            self.det_queue[self.cursur].append(torch.tensor([[0.]])) # bbox has no ids 
            inps = torch.zeros(torch.from_numpy(np.array([[min_col,min_row,max_col,max_row]])).size(0), 3, *self._input_size)
            self.det_queue[self.cursur].append(inps) #
            cropped_boxes = torch.zeros(torch.from_numpy(np.array([[min_col,min_row,max_col,max_row]])).size(0), 4)
            self.det_queue[self.cursur].append(cropped_boxes) # bbox has no ids 
            # self.det_queue[self.cursur].append(self.current_mask) # bbox has no ids 

            self.show_current_frame(fast=True) # 更新畫面

            self.update_memory_size()
            QApplication.processEvents()

            if self.cursur == 0 or self.cursur == self.num_frames-1:
                break

        self.propagating = False
        self.curr_frame_dirty = False
        self.on_pause()
        self.tl_slide()
        QApplication.processEvents()

    def pause_propagation(self):
        self.propagating = False

    def on_commit(self):
        self.complete_interaction()
        self.update_interacted_mask()

    def on_prev_frame(self):
        # self.tl_slide will trigger on setValue
        self.cursur = max(0, self.cursur-1) # cursor 決定第幾幀
        self.tl_slider.setValue(self.cursur)

    def on_next_frame(self):  # 移至下一幀
        # self.tl_slide will trigger on setValue
        self.cursur = min(self.cursur+1, self.num_frames-1)
        self.tl_slider.setValue(self.cursur)

    def on_play_video_timer(self):
        self.load_current_image_mask(no_mask=True) # 然後讀取該cursor的圖片與mask. mask不一定存在
        self.load_current_torch_image_mask(no_mask=True) #將圖片(Numpy)轉成張量(Tensor)
        
        if self.cursur == self.num_frames-1:
            (min_col,min_row,max_col,max_row) = get_bbox_from_mask(self.current_mask)
            # 將辨識完的結果 處理完後丟到 Queue 裡面
            # self.wait_and_put(self.det_queue, (orig_imgs[k], im_names[k], boxes_k, scores[dets[:, 0] == k], ids[dets[:, 0] == k], inps, cropped_boxes))
            if len(self.det_queue[self.cursur]) > 1:
                self.det_queue[self.cursur] = []
            self.det_queue[self.cursur].append(self.current_image)
            self.det_queue[self.cursur].append(str(self.cursur) + '.jpg')
            self.det_queue[self.cursur].append(torch.from_numpy(np.array([[min_col,min_row,max_col,max_row]])))
            self.det_queue[self.cursur].append(torch.tensor([[1.]])) # bbox has no score , let it be a 100% accuracy
            self.det_queue[self.cursur].append(torch.tensor([[0.]])) # bbox has no ids 
            inps = torch.zeros(torch.from_numpy(np.array([[min_col,min_row,max_col,max_row]])).size(0), 3, *self._input_size)
            self.det_queue[self.cursur].append(inps) #
            cropped_boxes = torch.zeros(torch.from_numpy(np.array([[min_col,min_row,max_col,max_row]])).size(0), 4)
            self.det_queue[self.cursur].append(cropped_boxes) 
            # self.det_queue[self.cursur].append(self.current_mask) 
            
            self.timer.stop()
            self.play_button.setText('Play Video')
            self.console_push_text(f'stop playing video')
            self.tl_slider.setValue(self.cursur)
        else:
            # pdb.set_trace()
            # self.get_bbox_from_mask()
            (min_col,min_row,max_col,max_row) = get_bbox_from_mask(self.current_mask)
            # 將辨識完的結果 處理完後丟到 Queue 裡面
            # self.wait_and_put(self.det_queue, (orig_imgs[k], im_names[k], boxes_k, scores[dets[:, 0] == k], ids[dets[:, 0] == k], inps, cropped_boxes))
            if len(self.det_queue[self.cursur]) > 1:
                self.det_queue[self.cursur] = []
            # self.det_queue[self.cursur].append(self.current_image)
            self.det_queue[self.cursur].append(str(self.cursur) + '.jpg')
            self.det_queue[self.cursur].append(torch.from_numpy(np.array([[min_col,min_row,max_col,max_row]])))
            self.det_queue[self.cursur].append(torch.tensor([[1.]])) # bbox has no score , let it be a 100% accuracy
            self.det_queue[self.cursur].append(torch.tensor([[0.]])) # bbox has no ids 
            inps = torch.zeros(torch.from_numpy(np.array([[min_col,min_row,max_col,max_row]])).size(0), 3, *self._input_size)
            self.det_queue[self.cursur].append(inps) #
            cropped_boxes = torch.zeros(torch.from_numpy(np.array([[min_col,min_row,max_col,max_row]])).size(0), 4)
            self.det_queue[self.cursur].append(cropped_boxes) 
            self.det_queue[self.cursur].append(self.current_mask) 
            
            self.on_next_frame()
            
            
        
    def on_play_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('Play Video')
        else:
            self.timer.start()   # active timer to umplement on_play_video_timer
            self.play_button.setText('Stop Video')

    def on_reset_mask(self):
        self.current_mask.fill(0)
        if self.current_prob is not None:
            self.current_prob.fill_(0)
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        self.show_current_frame()

    def on_zoom_plus(self):
        self.zoom_pixels -= 25
        self.zoom_pixels = max(50, self.zoom_pixels)
        self.update_minimap()

    def on_zoom_minus(self):
        self.zoom_pixels += 25
        self.zoom_pixels = min(self.zoom_pixels, 300)
        self.update_minimap()

    def set_navi_enable(self, boolean):
        self.zoom_p_button.setEnabled(boolean)
        self.zoom_m_button.setEnabled(boolean)
        self.run_button.setEnabled(boolean)
        self.tl_slider.setEnabled(boolean)
        self.play_button.setEnabled(boolean)
        self.lcd.setEnabled(boolean)

    def hit_number_key(self, number):
        if number == self.current_object:
            return
        self.current_object = number
        if self.fbrs_controller is not None:
            self.fbrs_controller.unanchor()
        self.console_push_text(f'Current object changed to {number}.')
        self.clear_brush()
        self.vis_brush(self.last_ex, self.last_ey)
        self.update_interact_vis()
        self.show_current_frame()

    def clear_brush(self):
        self.brush_vis_map.fill(0)
        self.brush_vis_alpha.fill(0)

    def vis_brush(self, ex, ey):
        self.brush_vis_map = cv2.circle(self.brush_vis_map, 
                (int(round(ex)), int(round(ey))), self.brush_size//2+1, color_map[self.current_object], thickness=-1)
        self.brush_vis_alpha = cv2.circle(self.brush_vis_alpha, 
                (int(round(ex)), int(round(ey))), self.brush_size//2+1, 0.5, thickness=-1)

    def on_mouse_press(self, event):  # 滑鼠壓下
        if self.is_pos_out_of_bound(event.x(), event.y()): # 滑鼠點擊非功能區域，不做任何事情
            return

        # mid-click
        if (event.button() == Qt.MidButton):
            ex, ey = self.get_scaled_pos(event.x(), event.y())
            target_object = self.current_mask[int(ey),int(ex)]
            if target_object in self.vis_target_objects:
                self.vis_target_objects.remove(target_object)
            else:
                self.vis_target_objects.append(target_object)
            self.console_push_text(f'Target objects for visualization changed to {self.vis_target_objects}')
            self.show_current_frame()
            return

        self.right_click = (event.button() == Qt.RightButton) # 滑鼠右鍵
        self.pressed = True

        h, w = self.height, self.width

        self.load_current_torch_image_mask()                  # 讀取Mask，如果是第一幀且尚未點擊，這行就沒做事
        image = self.current_image_torch                      # 讀取幀

        last_interaction = self.interaction
        new_interaction = None
        if self.curr_interaction == 'Scribble':
            if last_interaction is None or type(last_interaction) != ScribbleInteraction:
                self.complete_interaction()
                new_interaction = ScribbleInteraction(image, torch.from_numpy(self.current_mask).float().cuda(), 
                        (h, w), self.s2m_controller, self.num_objects)
        elif self.curr_interaction == 'Free':
            if last_interaction is None or type(last_interaction) != FreeInteraction:
                self.complete_interaction()
                new_interaction = FreeInteraction(image, self.current_mask, (h, w), 
                        self.num_objects)
                new_interaction.set_size(self.brush_size)
        elif self.curr_interaction == 'Click':
            if (last_interaction is None or type(last_interaction) != ClickInteraction  # 圖片沒被點過 (該幀完全沒有mask)
                    or last_interaction.tar_obj != self.current_object):
                self.complete_interaction()                 # 
                self.fbrs_controller.unanchor()             # 這裡就是利用fbrs的功能物件 (https://github.com/SamsungLabs/fbrs_interactive_segmentation)
                new_interaction = ClickInteraction(image, self.current_prob, (h, w), 
                            self.fbrs_controller, self.current_object)

        if new_interaction is not None:
            self.interaction = new_interaction              # 記錄點擊過的幀 (同時代表該幀也有mask)

        # Just motion it as the first step                  # event:  <PyQt5.QtGui.QMouseEvent>
        self.on_mouse_motion(event)

    def on_mouse_motion(self, event):
        ex, ey = self.get_scaled_pos(event.x(), event.y())  # 滑鼠點擊在幀上的位置
        self.last_ex, self.last_ey = ex, ey
        self.clear_brush()                                  # 清除 某種東西
        # Visualize
        self.vis_brush(ex, ey)
        if self.pressed:
            if self.curr_interaction == 'Scribble' or self.curr_interaction == 'Free':
                obj = 0 if self.right_click else self.current_object
                self.vis_map, self.vis_alpha = self.interaction.push_point(
                    ex, ey, obj, (self.vis_map, self.vis_alpha)
                )
        self.update_interact_vis()                          # 
        self.update_minimap()                               #

    def update_interacted_mask(self):
        self.current_prob = self.interacted_prob
        self.current_mask = torch_prob_to_numpy_mask(self.interacted_prob)
        self.show_current_frame()
        self.save_current_mask()
        self.curr_frame_dirty = False

    def complete_interaction(self):
        if self.interaction is not None:
            self.clear_visualization()
            self.interaction = None

    def on_mouse_release(self, event):
        if not self.pressed:
            # this can happen when the initial press is out-of-bound
            return

        ex, ey = self.get_scaled_pos(event.x(), event.y())

        self.console_push_text('%s interaction at frame %d.' % (self.curr_interaction, self.cursur))
        interaction = self.interaction

        if self.curr_interaction == 'Scribble' or self.curr_interaction == 'Free':
            self.on_mouse_motion(event)
            interaction.end_path()
            if self.curr_interaction == 'Free':
                self.clear_visualization()
        elif self.curr_interaction == 'Click':
            ex, ey = self.get_scaled_pos(event.x(), event.y())
            self.vis_map, self.vis_alpha = interaction.push_point(ex, ey,
                self.right_click, (self.vis_map, self.vis_alpha))

        self.interacted_prob = interaction.predict()
        self.update_interacted_mask()
        self.update_gpu_usage()

        self.pressed = self.right_click = False

    def wheelEvent(self, event):
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        if self.curr_interaction == 'Free':
            self.brush_slider.setValue(self.brush_slider.value() + event.angleDelta().y()//30)
        self.clear_brush()
        self.vis_brush(ex, ey)
        self.update_interact_vis()
        self.update_minimap()

    def update_gpu_usage(self):
        info = torch.cuda.mem_get_info()
        global_free, global_total = info
        global_free /= (2**30)  # unit GB
        global_total /= (2**30)
        global_used = global_total - global_free

        self.gpu_mem_gauge.setFormat(f'{global_used:.01f} GB / {global_total:.01f} GB')
        self.gpu_mem_gauge.setValue(round(global_used/global_total*100))

        used_by_torch = torch.cuda.max_memory_allocated() / (2**20)
        self.torch_mem_gauge.setFormat(f'{used_by_torch:.0f} MB / {global_total:.01f} GB')
        self.torch_mem_gauge.setValue(round(used_by_torch/global_total*100/1024))

    def on_gpu_timer(self):
        self.update_gpu_usage()

    def update_memory_size(self):
        try:
            max_work_elements = self.processor.memory.max_work_elements
            max_long_elements = self.processor.memory.max_long_elements

            curr_work_elements = self.processor.memory.work_mem.size
            curr_long_elements = self.processor.memory.long_mem.size

            self.work_mem_gauge.setFormat(f'{curr_work_elements} / {max_work_elements}')
            self.work_mem_gauge.setValue(round(curr_work_elements/max_work_elements*100))

            self.long_mem_gauge.setFormat(f'{curr_long_elements} / {max_long_elements}')
            self.long_mem_gauge.setValue(round(curr_long_elements/max_long_elements*100))

        except AttributeError:
            self.work_mem_gauge.setFormat('Unknown')
            self.long_mem_gauge.setFormat('Unknown')
            self.work_mem_gauge.setValue(0)
            
            
            self.long_mem_gauge.setValue(0)

    def on_work_min_change(self):
        if self.initialized:
            self.work_mem_min.setValue(min(self.work_mem_min.value(), self.work_mem_max.value()-1))
            self.update_config()

    def on_work_max_change(self):
        if self.initialized:
            self.work_mem_max.setValue(max(self.work_mem_max.value(), self.work_mem_min.value()+1))
            self.update_config()

    def update_config(self):
        if self.initialized:
            self.config['min_mid_term_frames'] = self.work_mem_min.value()
            self.config['max_mid_term_frames'] = self.work_mem_max.value()
            self.config['max_long_term_elements'] = self.long_mem_max.value()
            self.config['num_prototypes'] = self.num_prototypes_box.value()
            self.config['mem_every'] = self.mem_every_box.value()

            self.processor.update_config(self.config)

    def on_clear_memory(self):
        self.processor.clear_memory()
        torch.cuda.empty_cache()
        self.update_gpu_usage()
        self.update_memory_size()

    def _open_file(self, prompt):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, prompt, "", "Image files (*)", options=options)
        return file_name

    def on_import_mask(self):
        file_name = self._open_file('Mask')
        if len(file_name) == 0:
            return

        mask = self.res_man.read_external_image(file_name, size=(self.height, self.width))

        shape_condition = (
            (len(mask.shape) == 2) and
            (mask.shape[-1] == self.width) and 
            (mask.shape[-2] == self.height)
        )

        object_condition = (
            mask.max() <= self.num_objects
        )

        if not shape_condition:
            self.console_push_text(f'Expected ({self.height}, {self.width}). Got {mask.shape} instead.')
        elif not object_condition:
            self.console_push_text(f'Expected {self.num_objects} objects. Got {mask.max()} objects instead.')
        else:
            self.console_push_text(f'Mask file {file_name} loaded.')
            self.current_image_torch = self.current_prob = None
            self.current_mask = mask
            self.show_current_frame()
            self.save_current_mask()

    def on_import_layer(self):
        file_name = self._open_file('Layer')
        if len(file_name) == 0:
            return

        self._try_load_layer(file_name)
    
    def on_load_mask(self):
        # pdb.set_trace()
        try:
            start =  int(self.pose_start.text())
            end = int(self.pose_end.text())
        except:
            pass
        
        
        fnames = sorted(glob.glob(os.path.join(self.res_man.mask_dir, '*.jpg')))
        if len(fnames) == 0:
            fnames = sorted(glob.glob(os.path.join(self.res_man.mask_dir, '*.png')))
        frame_list = []
        t = time.time()
        for i, fname in enumerate(fnames):
            frame_list.append(np.array(Image.open(fname), dtype=np.uint8))
        # pdb.set_trace()
        
        cursur = int(re.findall(r'\d+', fnames[0].split('/')[-1])[0])
        
        for k in frame_list:
            
            if cursur >= start and cursur <=end:
                # pdb.set_trace()
                (min_col,min_row,max_col,max_row) = get_bbox_from_mask(k)
                # 將辨識完的結果 處理完後丟到 Queue 裡面
                # self.wait_and_put(self.det_queue, (orig_imgs[k], im_names[k], boxes_k, scores[dets[:, 0] == k], ids[dets[:, 0] == k], inps, cropped_boxes))
                if len(self.det_queue[cursur]) > 1:
                    self.det_queue[cursur] = []
                # self.det_queue[self.cursur].append(self.current_image)
                self.det_queue[cursur].append(str(cursur) + '.jpg')
                self.det_queue[cursur].append(torch.from_numpy(np.array([[min_col,min_row,max_col,max_row]])))
                self.det_queue[cursur].append(torch.tensor([[1.]])) # bbox has no score , let it be a 100% accuracy
                self.det_queue[cursur].append(torch.tensor([[0.]])) # bbox has no ids 
                inps = torch.zeros(torch.from_numpy(np.array([[min_col,min_row,max_col,max_row]])).size(0), 3, *self._input_size)
                self.det_queue[cursur].append(inps) #
                cropped_boxes = torch.zeros(torch.from_numpy(np.array([[min_col,min_row,max_col,max_row]])).size(0), 4)
                self.det_queue[cursur].append(cropped_boxes) 
                # self.det_queue[cursur].append(self.current_mask) 
            cursur += 1
        
        print('find_human_mask cost', time.time() - t)
        
    def _try_load_layer(self, file_name):
        # file_name : ./docs/ECCV-logo.png
        try:
            layer = self.res_man.read_external_image(file_name, size=(self.height, self.width))

            if layer.shape[-1] == 3:
                layer = np.concatenate([layer, np.ones_like(layer[:,:,0:1])*255], axis=-1)

            condition = (
                (len(layer.shape) == 3) and
                (layer.shape[-1] == 4) and 
                (layer.shape[-2] == self.width) and 
                (layer.shape[-3] == self.height)
            )

            if not condition:
                self.console_push_text(f'Expected ({self.height}, {self.width}, 4). Got {layer.shape}.')
            else:
                self.console_push_text(f'Layer file {file_name} loaded.')
                self.overlay_layer = layer
                self.overlay_layer_torch = torch.from_numpy(layer).float().cuda()/255
                self.show_current_frame()
        except FileNotFoundError:
            self.console_push_text(f'{file_name} not found.')

    def on_save_visualization_toggle(self):
        self.save_visualization = self.save_visualization_checkbox.isChecked()
    

    def pose_estimate(self):
        # pdb.set_trace()
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_layer_names = None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # pdb.set_trace()
        final_result = []
        det_data = []
        
        if self.pose_config.size == -1:
            scale_factor = 1
        else:
            
            scale_factor =  cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/self.pose_config.size
        # pdb.set_trace()
        for i in self.det_queue:                      # 將每個frame物件偵測的結果從 det_queue 取出。
            if len(i) != 0 :
                i[1] = i[1]*scale_factor
                det_data.append(i)
                temp = i

        assert cap.isOpened(), f'Faild to load video file {self.video_path}'
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))          # 影片寬
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))         # 影片高
        
        # pdb.set_trace()
        model = self.pose_config.pose_config.split('/')[-1]
        model_name = model.split("_")[0]
        
        try:
            videoWriter = cv2.VideoWriter(os.path.join(self.pose_config.out_video_root,f'{model_name}_{os.path.basename(self.pose_config.video_path)}'),fourcc,self.fps, (w,h))
            videoWriter_cut = cv2.VideoWriter(os.path.join(self.pose_config.out_video_root,f'vis_{os.path.basename(self.pose_config.video_path)}'),fourcc,self.fps, (w,h))
        except:
            #" just incase"
            videoWriter = cv2.VideoWriter(os.path.join(self.pose_config.out_video_root,f'{model_name}_{os.path.basename(self.pose_config.video_path)}'),fourcc,119.91444866920152, (w,h))
            videoWriter_cut = cv2.VideoWriter(os.path.join(self.pose_config.out_video_root,f'vis_{os.path.basename(self.pose_config.video_path)}'),fourcc,119.91444866920152, (w,h))

        start = int(det_data[0][0][0:-4])
        end = int(det_data[-1][0][0:-4])
        
        index = 0
        conuter = 0
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=video_length) as pbar:
            while (cap.isOpened()):
                flag, img = cap.read()
                if not flag:
                    break
                if index < start:
                    index +=1
                    pbar.update(1)
                    continue
                if index > end:
                    break
                _result = []
                result = []
                pose_results, returned_outputs = inference_top_down_pose_model(
                    self.pose_model,                          # pose_model
                    img,                               # origin img
                    det_data[conuter][1],                 # person_results(bbox)
                    bbox_thr=self.pose_config.bbox_thr,       # bbox_thr
                    format='xyxy',
                    dataset=self.dataset,
                    dataset_info=self.dataset_info,           # dataset_info
                    return_heatmap=True,
                    outputs=output_layer_names)
                conuter += 1
    
                # pdb.set_trace()
                # boxes, scores, ids, hm, cropped_boxes, orig_img, im_name
                _result.append(
                    {
                        'keypoints':pose_results[0]['keypoints'][:,0:2],
                        'kp_score':pose_results[0]['keypoints'][:,2:],
                        # 'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                        'idx':[0.0],
                        'box':pose_results[0]['bbox'][0:4].tolist()
                    }
                )
                
                result = {
                    'imgname': str(index)+'.jpg',
                    'result': _result
                }
                final_result.append(result)
                videoWriter_cut.write(img)
                
                vis_img = vis_pose_result(
                    self.pose_model,                          # pose_model
                    img,                                      # origin img
                    pose_results,                             # person_results(bbox)
                    dataset=self.dataset,                     # bbox_thr
                    dataset_info=self.dataset_info,           # dataset_info
                    kpt_score_thr=self.pose_config.kpt_thr,   # kpt_score_thr
                    radius=self.pose_config.radius,           # keypoint radius  
                    thickness=self.pose_config.thickness,     # limb thickness
                    show=False)                               # show process
                index += 1
                # write video
                videoWriter.write(vis_img)
                pbar.update(1)
            
        # pdb.set_trace()
        print('finish pose estimation:',conuter, ' frames.')  
        videoWriter.release()
        videoWriter_cut.release()
        print('wrinting result to json')
        video_name = self.config['video_path'].split("/")[-1]
        # pdb.set_trace()
        write_json(final_result, self.pose_config.out_video_root, form=None, for_eval=False, outputfile=f'{video_name}')
        print('finish wrinte result to json')
        

    
    def loop(self):
        n = 0
        while True:
            yield n
            n += 1
    
    def stopped(self):
        if self.args.sp:
            return self._stopped
        else:
            return self._stopped.value
    @property
    def length(self):
        return self.datalen
    
    def wait_and_put(self, queue, item):
            queue.put(item)

def write_json(all_results, outputpath, form=None, for_eval=False, outputfile='alphapose-results.json'):
    '''
    all_result: result dict of predictions
    outputpath: output directory
    '''
    # pdb.set_trace()
    json_results = []
    json_results_cmu = {}
    counter = 0
    for im_res in all_results:
        # print(counter)
        counter += 1
        im_name = im_res['imgname']
        for human in im_res['result']:
            keypoints = []
            result = {}
            if for_eval:
                result['image_id'] = int(os.path.basename(im_name).split('.')[0].split('_')[-1])
            else:
                result['image_id'] = os.path.basename(im_name)
            result['category_id'] = 1

            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            # pro_scores = human['proposal_score']
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            result['keypoints'] = keypoints
            # result['score'] = float(pro_scores)
            if 'box' in human.keys():
                result['box'] = human['box']
            #pose track results by PoseFlow
            if 'idx' in human.keys():
                result['idx'] = human['idx']
            
            # 3d pose
            if 'pred_xyz_jts' in human.keys():
                pred_xyz_jts = human['pred_xyz_jts']
                pred_xyz_jts = pred_xyz_jts.cpu().numpy().tolist()
                result['pred_xyz_jts'] = pred_xyz_jts

            if form == 'cmu': # the form of CMU-Pose
                if result['image_id'] not in json_results_cmu.keys():
                    json_results_cmu[result['image_id']]={}
                    json_results_cmu[result['image_id']]['version']="AlphaPose v0.3"
                    json_results_cmu[result['image_id']]['bodies']=[]
                tmp={'joints':[]}
                result['keypoints'].append((result['keypoints'][15]+result['keypoints'][18])/2)
                result['keypoints'].append((result['keypoints'][16]+result['keypoints'][19])/2)
                result['keypoints'].append((result['keypoints'][17]+result['keypoints'][20])/2)
                indexarr=[0,51,18,24,30,15,21,27,36,42,48,33,39,45,6,3,12,9]
                for i in indexarr:
                    tmp['joints'].append(result['keypoints'][i])
                    tmp['joints'].append(result['keypoints'][i+1])
                    tmp['joints'].append(result['keypoints'][i+2])
                json_results_cmu[result['image_id']]['bodies'].append(tmp)
            elif form == 'open': # the form of OpenPose
                if result['image_id'] not in json_results_cmu.keys():
                    json_results_cmu[result['image_id']]={}
                    json_results_cmu[result['image_id']]['version']="AlphaPose v0.3"
                    json_results_cmu[result['image_id']]['people']=[]
                tmp={'pose_keypoints_2d':[]}
                result['keypoints'].append((result['keypoints'][15]+result['keypoints'][18])/2)
                result['keypoints'].append((result['keypoints'][16]+result['keypoints'][19])/2)
                result['keypoints'].append((result['keypoints'][17]+result['keypoints'][20])/2)
                indexarr=[0,51,18,24,30,15,21,27,36,42,48,33,39,45,6,3,12,9]
                for i in indexarr:
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i])
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i+1])
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i+2])
                json_results_cmu[result['image_id']]['people'].append(tmp)
            else:
                json_results.append(result)
    if form == 'cmu': # the form of CMU-Pose
        with open(os.path.join(outputpath, outputfile), 'w') as json_file:
            json_file.write(json.dumps(json_results_cmu))
            if not os.path.exists(os.path.join(outputpath,'sep-json')):
                os.mkdir(os.path.join(outputpath,'sep-json'))
            for name in json_results_cmu.keys():
                with open(os.path.join(outputpath,'sep-json',name.split('.')[0]+'.json'),'w') as json_file:
                    json_file.write(json.dumps(json_results_cmu[name]))
    elif form == 'open': # the form of OpenPose
        with open(os.path.join(outputpath, outputfile), 'w') as json_file:
            json_file.write(json.dumps(json_results_cmu))
            if not os.path.exists(os.path.join(outputpath,'sep-json')):
                os.mkdir(os.path.join(outputpath,'sep-json'))
            for name in json_results_cmu.keys():
                with open(os.path.join(outputpath,'sep-json',name.split('.')[0]+'.json'),'w') as json_file:
                    json_file.write(json.dumps(json_results_cmu[name]))
    else:
        with open(os.path.join(outputpath, outputfile+'.json'), 'w') as json_file:
            json_file.write(json.dumps(json_results))