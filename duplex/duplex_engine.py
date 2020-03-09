# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:15:39 2020

@author: Andrei
"""
import numpy as np
import tensorflow as tf

import os

import cv2
from time import time, sleep
from datetime import datetime
from collections import OrderedDict, deque

from duplex.utils import get_bar_plot_np, blur_pers, blended_fill_rect, prepare_img

import requests


__VER__ = '1.3.0.5'

P_KEY = '1. SPECT'
O_KEY = '2. ENVIR'
S_KEY = '3. SENT'
AUD_KEY = '4. AUD'
AAVG_KEY = '5. AUD AVG'
AMAX_KEY = '6. AUD MAX'
ATTN_KEY = '7. ATN'
RT_KEY = '8. LIVE'
LOC_KEY = '9. LOC'

STAT_KEY = '20. STAT'
VER_KEY = '21. VER'
SROOM_KEY = '22. SR'
SUSER_KEY = '23. SU'

HORECA_KEY = 'HRC'
TABLE_KEY = 'TBL'
ROOM_KEY = 'ROOM'


CLR_GREEN = (40,100,40)
CLR_LT_GREEN = (50, 200, 50)
CLR_RED = (50,50,200)
CLR_GRAY = (100,100,100)

DELAY_MILIS = 200

idx_to_emo = {
                0:'Nervos', 
                1:'Dezamagit', 
                2:'Speriat', 
                3:'FERICIT', 
                4:'Suparat', 
                5:'Surprins', 
                6:'Neutru'
                }

class DuplexEngine:
  def __init__(self, log, runs_on_device, 
               use_face=False,
               use_emotion=False,
               debug=False, track_len=12, boxes=True):
    self.version = __VER__
    self.__version__ = self.version
    self.use_face = use_face
    self.use_emotion = use_emotion
    self.full_debug = debug
    self.log = log
    self.failed_caps = 0
    self.runs_on_device = runs_on_device
    self.center_scale = 0
    self._np_bar = None
    self.last_status_data = None
    self._seps = 0
    self.n_last_faces = 0
    
    self.emotion_threshold = 0.5
    self.sentiment_count_threshold = 3
    
    self.sentiment = {'VALUE' : "UNK","COLOR" : (0,0,0)}
    self.live_sentiment = self.sentiment.copy()
    self.last_sentiment = self.sentiment
    self.last_sentiment_counter = 0
    
    
    self._draw_boxes = boxes or not self.runs_on_device
    self._max_pers = 0
    self._count_pers = []
    self._counts = deque(maxlen=track_len)
    self._full_counts = deque(maxlen=10000)
    self.c_status = 'UNK'
    
    self.n_snd_ok = 0
    self.n_snd_fail = 0

    self.hostname = self.log.get_machine_name()
    self._reset_live_data()
    self._config()
    return
  
  def _stats_add_sep(self):
    self._seps += 1
    self.live['SEP_{}'.format(self._seps)] = ''
  
  def _reset_live_data(self):
    self.live = OrderedDict({'Audience analysis': ''})
    self._stats_add_sep()
    self.live[P_KEY] = 0
    self.live[O_KEY] = 0
    return
  
  def _reload_live_settings(self):
    try:
      self.live_settings = self.log.load_data_json('live_settings.txt')
      self.P("Live settings refreshed: {}".format(self.live_settings))
    except:
      self.live_settings = None
    if self.live_settings is None:
      self.P("WARNING: Live settings could not be reloaded.")
      self.live_settings = {}
    return
  
  def _config(self):
    self.event_key = self.log.config_data['EVENT_KEY']
    self.url_key = self.log.config_data['URL_KEY']
    self.except_keys = self.log.config_data['EXCEPT_KEYS']
    self.image_key = self.log.config_data['IMAGE_KEY']
    self.model_config = self.log.load_data_json('default_model.txt')
    self.viewer_config = self.log.load_data_json('viewer_config.txt')
    self._reload_live_settings()
    
    
    url = self.viewer_config[self.url_key]
    
    if self.event_key not in self.viewer_config or self.log.config_data['SEND_EVENT'] != self.viewer_config[self.event_key]:
      self.P("Images will NOT be sent to server {} until 'join' message is received".format(url))
    else:
      self.P("Images will be sent to server {} asap".format(url))
      
    self.graph_file = self.model_config['ROOM_MODEL']
    self.graph_face_file = self.model_config['FACE_MODEL']
    self.graph_emotion_file = self.model_config['EMO_MODEL']
    self.win_name = 'OmniDJ Duplex Debug Window'
    self.PERSON_CLASS = int(self.model_config["PERSON_CLASS"])
    self.is_yolo = self.model_config['ISY']
    self.output_h = self.model_config['OUTPUT_H']
    self.output_w = self.model_config['OUTPUT_W']
    
    if not self.is_yolo:
      raise ValueError('Non-yolo graphs not supported at this time')
    else:
      self._load_yolo_model()         
    self._load_coco_classes()
    self._reset_detections()
    
    if self.use_face:
      self._load_face_ssd()
      if self.use_emotion:
        self._load_emotion_net()
      else:
        self.P("SKIPPED EMOTION NET LOADING")
    else:
      self.P("SKIPPED FACE SSD LOADING")
      
    
    self._combine_graphs_and_create_session()

    
    
    np_img = get_bar_plot_np(np.arange(10))
    self._stats_win_w = np_img.shape[1] + 50
    
    self.last_location = 'UNK'
    
    return
      
    
  
  def P(self, s, t=False):    
    return self.log.P("{}".format(s),show_time=t)
  
  def _load_yolo_model(self):
    force_download = not self.runs_on_device and self.full_debug
    self.log.maybe_download_model(url=self.model_config['URL'],
                                  model_file=self.graph_file,
                                  force_download=force_download)
      
    self.graph = self.log.load_graph_from_models(self.graph_file)
    if self.graph is None:
      raise ValueError("Startup aborded. Model could not be loaded: {}".format(self.graph_file))
    return

    
  def _load_face_ssd(self):
    force_download = not self.runs_on_device and self.full_debug
    self.log.maybe_download_model(url=self.model_config['URL_FACE'],
                                  model_file=self.graph_face_file,
                                  force_download=force_download)
      
    self.graph_face = self.log.load_graph_from_models(self.graph_face_file)
    if self.graph_face is None:
      raise ValueError("Startup aborded. Model could not be loaded: {}".format(self.graph_face_file))
    return

  def _load_emotion_net(self):
    force_download = not self.runs_on_device and self.full_debug
    self.log.maybe_download_model(url=self.model_config['URL_EMO'],
                                  model_file=self.graph_emotion_file,
                                  force_download=force_download)
      
    self.graph_emotion = self.log.load_graph_from_models(self.graph_emotion_file)
    if self.graph_emotion is None:
      raise ValueError("Startup aborded. Model could not be loaded: {}".format(self.graph_emotion_file))
    return

  
  def _combine_graphs_and_create_session(self):  
    if self.use_face:
      graph1_name = 'yolo'
      graph2_name = 'face'
      graph3_name = 'emo'
      if not self.use_emotion:
        t1 = time()
        self.full_graph = self.log.combine_graphs_tf1(lst_graphs=[self.graph, 
                                                                  self.graph_face],
                                                      lst_names=[graph1_name, 
                                                                 graph2_name])
        t2 = time()
      else:
        t1 = time()
        self.full_graph = self.log.combine_graphs_tf1(lst_graphs=[self.graph, 
                                                                  self.graph_face,
                                                                  self.graph_emotion],
                                                      lst_names=[graph1_name, 
                                                                 graph2_name,
                                                                 graph3_name])
        t2 = time()
        
      self.P("Graphs combined in {:.1f}s".format(t2-t1))
      prefix = graph1_name + '/'
    else:
      self.full_graph = self.graph
      prefix = ''
    
            
    config_proto = tf.compat.v1.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    self.log.P("Creating session...")
    self.sess = tf.compat.v1.Session(graph=self.full_graph,  config=config_proto)
    self.log.P("Done creating session.")
    if self.use_face:
      self.P("Prepare face tensors...")
      self.tf_face_classes = self.sess.graph.get_tensor_by_name(
          "face/detection_classes:0")
      self.tf_face_scores = self.sess.graph.get_tensor_by_name(
          "face/detection_scores:0")
      self.tf_face_boxes = self.sess.graph.get_tensor_by_name(
          "face/detection_boxes:0")
      self.tf_face_dets = self.sess.graph.get_tensor_by_name(
          "face/num_detections:0")
      self.tf_face_input = self.sess.graph.get_tensor_by_name(
          "face/image_tensor:0")
      
      if self.use_emotion:
        self.P("Prepare emotion tensors...")
        self.tf_emo_input = self.sess.graph.get_tensor_by_name(
            "emo/input:0")
        self.tf_emo_output = self.sess.graph.get_tensor_by_name(
            "emo/readout/Softmax:0")
        self.emo_to_idx = {v:k for k,v in idx_to_emo.items()}
        self.emotion_negatives = [0,1,2,4]
        self.emotion_positives = [3,5]
        
      
    if self.is_yolo:
      self.P("Prepare yolo tensors...")
      self.learning_phase = self.sess.graph.get_tensor_by_name(
          prefix + "keras_learning_phase:0")
      self.tf_classes = self.sess.graph.get_tensor_by_name(
          prefix + "YOLO_OUTPUT_CLASSES"+":0")
      self.tf_scores = self.sess.graph.get_tensor_by_name(
          prefix + "YOLO_OUTPUT_SCORES"+":0")
      self.tf_boxes = self.sess.graph.get_tensor_by_name(
          prefix + "YOLO_OUTPUT_BOXES"+":0")
      self.tf_yolo_input = self.sess.graph.get_tensor_by_name(
          prefix + "input_1"+":0")
    else:
      raise ValueError("NON-YOLO not supported")
    self.log.P("Got tensors.")
    return
  
  def _setup_video(self, show_debug=True):
    self.P("Preparing stream...")
    if show_debug:
      self.log.P("Preparing debug window '{}'".format(self.win_name))
      cv2.namedWindow(self.win_name, cv2.WND_PROP_FULLSCREEN)
      cv2.setWindowProperty(self.win_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
      
    gst_str2 = ('nvarguscamerasrc ! '
                 'video/x-raw(memory:NVMM), '
                 'width=(int)1920, height=(int)1080, '
                 'format=(string)NV12, framerate=(fraction)30/1 ! '
                 'nvvidconv flip-method=2 ! '
                 'video/x-raw, width=(int){}, height=(int){}, '
                 'format=(string)BGRx ! '
                 'videoconvert ! '
                 'appsink max-buffers=1 drop=True').format(1920, 1080)
    if self.runs_on_device:
      self.log.P("Opening TX2 camera with: '{}'".format(gst_str2))
      self.cap = cv2.VideoCapture(gst_str2, cv2.CAP_GSTREAMER)
    else:
      self.log.P("Opening local camera 0")
      self.cap = cv2.VideoCapture(0)
      self.cap.set(3, 1920)     #horizontal pixels
      self.cap.set(4, 1080)     #vertical pixels      
    return
    
  def _load_coco_classes(self):
    cls_file = self.model_config['CLASSES']
    full_cls_file = self.log.get_data_file(cls_file)
    self.P("Loading {}...".format(full_cls_file))
    with open(full_cls_file) as f:
      lines = f.read().splitlines()
    self.classes = lines.copy() + [lines[-1]]*150 # pad last class...
    self.classes[self.PERSON_CLASS] = 'PERSON'
    self.P("Loaded {} classes from {} CHK: PERSON='{}' / '{}'".format(
        len(lines), cls_file, lines[self.PERSON_CLASS], 
        self.classes[self.PERSON_CLASS]))
    self.__detections = {
        "HORECA" : 0,
        "ROOM" : 0,
        "TABLE" : 0,
        }
    return 

  def get_emotion_color(self, idx_or_label, score):
    color = (0,0,0)
    
    if type(idx_or_label) == str:
      idx = self.emo_to_idx[idx_or_label]
    else:
      idx = idx_or_label
#                0:'Angry', 
#                1:'Disgust', 
#                2:'Fear', 
#                3:'Happy', 
#                4:'Sad', 
#                5:'Surprise', 
#                6:'Neutral'
    if score < self.emotion_threshold:
      color = CLR_GRAY
    else:
      if idx in self.emotion_negatives:
        color = CLR_RED
      elif idx in self.emotion_positives:
        color = CLR_GREEN    
    return color
  
  
  def get_sentiment(self, labels, scores):
    bad = 0
    good = 0
    neutral = 0
    for i, score in enumerate(scores):
      idx = self.emo_to_idx[labels[i]]
      if score > self.emotion_threshold:
        if idx in self.emotion_negatives:
          bad += 1
        elif idx in self.emotion_positives:
          good += 1
        else:
          neutral += 1
      else:
        neutral += 1
        
    if good > bad:
      dct_result = {'VALUE' : "GOOD VIBES! {}{}{}".format(good, neutral, bad),
                    "COLOR" : CLR_LT_GREEN
                    }
    elif bad > good:
      dct_result = {'VALUE' : "DANGERZONE! {}{}{}".format(good, neutral, bad),
                    "COLOR" : CLR_RED
                    }
    else:
      dct_result = {'VALUE' : "LOW ENERGY! {}{}{}".format(good, neutral, bad),
                    "COLOR" : (0,0,0)
                    }
      
      
    return dct_result
        
  

  def _reset_detections(self):
    self.ENV_KEYS = [HORECA_KEY, TABLE_KEY, ROOM_KEY]
    self.detections = {}
    for k in self.ENV_KEYS:
      self.detections[k] = 0
    return
 
  def _resize(self, np_image, width, height, center=True):
    if center:
      h, w = np_image.shape[:2]
      self.center_scale = w / h
      new_w = int(width)
      new_h = int(new_w / self.center_scale)
      dim = (new_w, new_h)
      np_resized = cv2.resize(np_image, dim)
      np_out = np.zeros((height, width, 3))
      np_out[:new_h,:,:] = np_resized
    else:
      dim = (width, height)
      np_out = cv2.resize(np_image, dim)      
    return  np_out
  
  
  def _put_text(self, np_img, s, top, left, color=(255,255,255)):
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.5
    fontColor              = color
    thickness              = 1
    lineType               = cv2.LINE_AA
    
    text_size = cv2.getTextSize(s, font, fontScale, thickness)
    text_w = text_size[0][0]
    text_h = text_size[0][1]

    position = (left,top + 6)
    
    cv2.putText(np_img,s, 
        position, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)    
    return text_w, text_h
    
  
  def _run_yolo_single_inference(self, np_rgb_image):    
    self.log.start_timer('yolo_single_inference')
    if len(np_rgb_image.shape) != 3:
      raise ValueError("yolo must receive single image")

    self.D("   model received {}".format(np_rgb_image.shape))

    self.log.start_timer('yolo_preprocess')
    if np_rgb_image.shape[0] != 608:
      np_rgb_image_yolo = self._resize(np_rgb_image, width=608, height=608)
      
    if np_rgb_image.max() > 1:
      np_rgb_image_yolo = np_rgb_image_yolo / 255.
      
    feed_dict = {
      self.learning_phase: 0, 
      self.tf_yolo_input: np.expand_dims(np_rgb_image_yolo,  axis=0)
    }
    self.log.end_timer('yolo_preprocess')
    self.log.start_timer('run_session')
    self.D("    YOLO input {}".format(np_rgb_image_yolo.shape))
    out = self.sess.run([self.tf_scores,  self.tf_boxes,  self.tf_classes],  feed_dict)
    tt = self.log.end_timer('run_session')
    self.D("    TOLO model time: {:.3f}s".format(tt))
    self.log.start_timer('yolo_post_preprocess')
    
    scores,  boxes,  classes = out
    if self.output_h != 720:
      boxes[:,  0] = boxes[:,  0] / 720 * self.output_h
      boxes[:,  1] = boxes[:,  1] / 1280 * self.output_w
      boxes[:,  2] = boxes[:,  2] / 720 * self.output_h
      boxes[:,  3] = boxes[:,  3] / 1280 * self.output_w    
    
    # now the image is based on the centered image
    if self.center_scale > 0:
      boxes[:,  0] = boxes[:,  0] * self.center_scale
      boxes[:,  2] = boxes[:,  2] * self.center_scale
      
    self.log.end_timer('yolo_post_preprocess')
    self.log.end_timer('yolo_single_inference')    
    return scores, boxes, classes
  
  
  def _run_face_single_cycle(self, np_rgb_image, np_out_image):
    self.D("  Face cycle input: {}".format(np_rgb_image.shape))
    self.log.start_timer('face_cycle')
    face_scores, face_boxes, face_classes, face_dets = self._run_face_single_inference(np_rgb_image)
    self.log.start_timer('face_post_proc')
    self.n_last_faces = 0
    faces = []
    positions = []
    _height = np_rgb_image.shape[0]
    _width = np_rgb_image.shape[1]
    _thr = self.live_settings.get('FACE_THR',0.3)
    for i, scr in enumerate(face_scores):
      if scr > _thr:
        self.n_last_faces += 1
      
        y1, x1, y2, x2 = face_boxes[i]
        y1 = max(0,int(y1-10))
        x1 = max(0,int(x1-10))
        y2 = min(_height,int(y2+10))
        x2 = min(_width,int(x2+10))
        
        np_face = np_rgb_image[y1:y2, x1:x2]
        np_prep_face = prepare_img(np_face)
        faces.append(np_prep_face)
        positions.append((y1, x1))
        np_out_image = blur_pers(np_out_image, 
                                 left=x1,
                                 top=y1,
                                 right=x2,
                                 bottom=y2,
                                 DIRECT=True,
                                 )
      elif scr >= 0.05:
        y1, x1, y2, x2 = face_boxes[i]
        y1 = max(0,int(y1-10))
        x1 = max(0,int(x1-10))
        y2 = min(_height,int(y2+10))
        x2 = min(_width,int(x2+10))
        self._put_text(np_out_image, "UPDF: {:.1f}%)".format(scr*100),
                       top=y1, left=x1, color=CLR_RED)
        
    self.D("FACE: got {} faces".format(self.n_last_faces))
    self.log.end_timer('face_post_proc')
    if len(faces) > 0:
      self.log.start_timer('emotion_proc')
      np_faces = np.expand_dims(np.array(faces), axis=-1)
      labels, scores = self._run_emotion_batch(np_faces)
      self.log.start_timer('emotion_post_proc')
      for i, (top, left) in enumerate(positions):
        score = scores[i]
        label = labels[i]
        color = self.get_emotion_color(label, score)
        self._put_text(np_out_image, "{} [{:.0f}%]".format(label, score * 100), 
                                   top=int(top-12), 
                                   left=int(left + 20),
                                   color=color)
      
      self.live_sentiment = self.get_sentiment(labels=labels, scores=scores)
      self.sentiment = self.live_sentiment
      if False:
        if self.live_sentiment == self.last_sentiment:
          self.last_sentiment_counter += 1
        else:
          self.last_sentiment = self.live_sentiment.copy()
          self.last_sentiment_counter = 0
        if self.last_sentiment_counter > self.sentiment_count_threshold:
          self.sentiment = self.last_sentiment
      self.log.end_timer('emotion_post_proc')
      self.log.end_timer('emotion_proc')
    self.log.end_timer('face_cycle')
    return np_out_image
  
  
  def _run_emotion_batch(self, np_images):
    assert len(np_images.shape) == 4, "Emotion images must be (B,H,W,C)"
    self.D("    Emotion model input {}".format(np_images.shape))

    feed_dict = {
      self.tf_emo_input: np_images
    }
    self.log.start_timer('run_emo_session')
    lst_outs = [self.tf_emo_output]
    preds = self.sess.run(lst_outs,  feed_dict)[0]
    tt = self.log.end_timer('run_emo_session')
    self.D("    Emotion model time: {:.3f}s".format(tt))
    classes = preds.argmax(axis=-1)
    scores = preds.max(axis=-1)
    labels = self.emotion_idx_to_label(classes)
    return labels, scores
  
  
  def emotion_idx_to_label(self, idxs):    

    if type(idxs) == int:
      idxs = [idxs]
    labels = [idx_to_emo[x] for x in idxs]
    return labels
      
   
  
  
  def _run_face_single_inference(self, np_rgb_image):   
    MIN_IMAGE_HEIGHT = 300
    self.log.start_timer('face_single_inference')
    if len(np_rgb_image.shape) != 3 or np_rgb_image.shape[0] < MIN_IMAGE_HEIGHT:
      raise ValueError("face model must receive single image")
    
          
    input_h = np_rgb_image.shape[0]
    input_w = np_rgb_image.shape[1]
    self.D("    Face model input {}".format(np_rgb_image.shape))

    feed_dict = {
      self.tf_face_input: np.expand_dims(np_rgb_image,  axis=0)
    }
    self.log.start_timer('run_face_session')
    lst_outs = [self.tf_face_scores,  self.tf_face_boxes,  self.tf_face_classes, self.tf_face_dets]
    out = self.sess.run(lst_outs,  feed_dict)
    tt = self.log.end_timer('run_face_session')
    self.D("    Face model time: {:.3f}s".format(tt))
    
    out = [x.squeeze(axis=0) for x in out]
    scores,  boxes,  classes, num_dets = out

    boxes[:,  0] = boxes[:,  0] * input_h
    boxes[:,  1] = boxes[:,  1] * input_w
    boxes[:,  2] = boxes[:,  2] * input_h
    boxes[:,  3] = boxes[:,  3] * input_w
    
    # now the image is based on the centered image
#    if self.center_scale > 0:
#      boxes[:,  0] = boxes[:,  0] * self.center_scale
#      boxes[:,  2] = boxes[:,  2] * self.center_scale
      
    self.log.end_timer('face_single_inference')    
    return scores, boxes, classes, num_dets  
  
  
  def D(self, s):
    if self.full_debug:
      self.P(s)
  
  def _run_yolo_cycle(self, show_debug=False):   
    self.log.start_timer("run_yolo_cycle")
    succ, frm = self.cap.read()
    self._relinquish()
    if self.runs_on_device:
      frm = cv2.flip(frm, 0)
    else:
      frm = cv2.flip(frm, 1)
    if succ:
      self.failed_caps = 0
      self.D("Received {}".format(frm.shape))
      if frm.shape[0] < 600:
        raise ValueError("Resolution to low: {} !".format(frm.shape))      
      if frm.shape != (self.output_h, self.output_w, 3):
        frm = self._resize(frm, width=self.output_w, height=self.output_h, center=False)
      np_rgb = cv2.cvtColor(frm,  cv2.COLOR_BGR2RGB)            
      

      scores, boxes, classes = self._run_yolo_single_inference(np_rgb)
      self._relinquish()

      if self.use_face:
        frm = self._run_face_single_cycle(np_rgb, frm)
        self._relinquish()
      
      self.log.start_timer('run_post_proc')
      self.D("SCENE: Got {} scores".format(scores.shape[0]))
      self._reset_live_data()
      for i, score in enumerate(scores):
        if score > 0.5:
          score = score * 100
          x1 = int(boxes[i, 1])
          y1 = int(boxes[i, 0])
          x2 = int(boxes[i, 3])
          y2 = int(boxes[i, 2])
          cl = classes[i]
          if  cl == self.PERSON_CLASS:
            if self._draw_boxes:
              self._cv_rect(frm,  (x1, y1),  (x2, y2),  CLR_GREEN)   
            self._put_text(frm, "spectator [{:.0f}%]".format(score), 
                           top=y1-12, 
                           left=x1 + (x2-x1)//2 - 50,
                           color=CLR_GREEN)
            if not self.use_face:
              frm = blur_pers(frm, 
                              left=x1,
                              top=y1,
                              right=x2,
                              bottom=y2,
                              DIRECT=False,)
            self.live[P_KEY] += 1
          else:
            if cl in [40, 42, 43, 45]:
              self.detections[HORECA_KEY] += 1
            elif cl in [56]:
              self.detections[ROOM_KEY] += 1
            elif cl in [60, 39, 41]:
              self.detections[TABLE_KEY] += 1
              
            if self._draw_boxes:
              self._cv_rect(frm,  (x1, y1),  (x2, y2),  (255, 0, 0))     
            self._put_text(frm, "mediu: {}:{} [{:.0f}%]".format(
                                        self.classes[cl],cl,
                                        score), 
                           top=y1-12, 
                           left=x1 + (x2-x1)//2 - 50,
                           color=(255, 0, 0))
            self.live[O_KEY] += 1
      self.log.end_timer('run_post_proc')
      self._add_stats(frm)
      if show_debug:
        cv2.imshow(self.win_name, frm)
    else:
      frm = None
      self.failed_caps += 1
      if (self.failed_caps % 1000) == 0:
        self.P("ERROR: Failed {} frames so far!".format(
            self.failed_caps))
      if self.failed_caps > 5000:
        raise ValueError("ABORTING DUE TO UNAVAILABLE VIDEO STREAM")
    self.log.end_timer("run_yolo_cycle")
    return frm
  
  
  def get_current_location(self,):
    _str = ''
    for k in self.detections:
      if self.detections[k] > 0:
        _str = _str + k + ' '
    if _str != '':
      self.last_location = _str
    return self.last_location
      
    
            
  
  def _draw_window(self, np_img, inter=6, up_offset=10):
    dsize = len(self.live)    
    win_h = dsize * (12 + inter) + up_offset * 2
    if self._np_bar is not None:
      win_h += self._np_bar.shape[0] + 10
    blended_fill_rect(np_img, (0,0), (self._stats_win_w, win_h))
    return np_img, win_h
    
  
  
  def _get_spec_attention(self):
    n_pers = self.live[P_KEY]
    n_face = self.n_last_faces
    s_stat = " [{}/{}]".format(n_face, n_pers)
    if n_pers == 0:
      if n_face > 0:
        return {'VALUE':'UNCLEAR' + s_stat, 'COLOR':(0,0,0)}
      else:
        return {'VALUE':'BAD' + s_stat, 'COLOR':(0,0,255)}
      
    ind = n_face / n_pers
    if ind < 0.3:
      return {'VALUE':'BAD' + s_stat, 'COLOR':(0,0,255)}
    elif ind >= 0.3 and ind < 0.6:
      return {'VALUE':'AVG' + s_stat, 'COLOR':(255,0,0)}
    else:
      return {'VALUE':'GREAT' + s_stat, 'COLOR':CLR_GREEN}
  
  def _run_cycle(self, show_debug=False):
    self.log.start_timer('run_cycle')
    if self.is_yolo:
      res = self._run_yolo_cycle(show_debug=show_debug)
    else:
      raise ValueError("Running cycle cannot execute. Unknown processing method.")
    self.log.end_timer('run_cycle')
    return res
    
  def _close_callback(self):
    self.force_close = True
    self.P("Closing nicely the system...")
    return
  
  def _relinquish(self):
    if self.runs_on_device:
      os.sched_yield()
    return
    
  
  def run(self, show_debug=False):
    self.force_close = False
    self.log.register_close_callback(self._close_callback)
    self._setup_video(show_debug=show_debug)
    self.fps = 'NA'
    self._run_start_tm = time()
    self._run_start_datetime = datetime.now()
    self.run_laps_dates = []
    self.run_laps_frames = []
    self._frame_start = time()
    self._frame_count = 0
    self._last_avg_tm = time()
    self._last_load_config = time()
    self.P("Running inference cycle with show_debug={}".format(show_debug))
    while True:
      self.log.start_timer('full_iteration')
      key = cv2.waitKey(1) & 0xFF
      if (key in [ord('q'), ord('Q')]):
        self.P("Close session request in CV console")
        break      
      if self.force_close:
        break
      frm = self._run_cycle(show_debug=show_debug)
      
      self.log.start_timer('post_analysis')
      
      if frm is not None:
        self._frame_count += 1
        self._relinquish()
        self.send_image(frm)        
        self._relinquish()
        
      if self.runs_on_device:
        sleep(DELAY_MILIS / 1000)
      
      if self._frame_count > 100:
        self.run_laps_dates.append(datetime.now())
        self.run_laps_frames.append(self._frame_count)
        self._frame_count = 0
        self._run_start_tm = time()   
        
      if time() > (self._last_load_config + 10):
        self.P("Status:")
        self.P("  Conn: {} [State: {}]".format(self.last_status_data, self.viewer_config[self.event_key]))
        self.P("  Send: {}".format(self.n_snd_ok))
        self.P("  Fail: {}".format(self.n_snd_fail))
        self.P("  Sent: {} / {}".format(
            self.live_sentiment,
            self.last_sentiment_counter))
        self.P("  Lsnt: {}".format(self.last_sentiment))
        self.P("Checking new config. Press 'Q' to quit if interactive.".format(
            self.n_snd_ok, self.n_snd_fail))
        dct_new_config = self.log.load_data_json('viewer_config.txt')
        if dct_new_config != self.viewer_config:
          self.P("Received new config: {}".format(dct_new_config))
          self.viewer_config = dct_new_config
        self._last_load_config = time()

        self._reload_live_settings()
        
      self.log.end_timer('post_analysis')
      self.log.end_timer('full_iteration')
    self.P("Done inference cycle.")
    return
  
  def shutdown(self):
    self.cap.release()   
    del self.cap
    cv2.destroyAllWindows()
    self.P("Done shutown.")
    self.log.show_timers()
    return
  
  def send_image(self, np_img_bgr):    
    if self.event_key not in self.viewer_config:
      return

    confirmation_status_string = self.log.config_data['SEND_EVENT'] 
    status_data = self.viewer_config[self.event_key]
    if confirmation_status_string != status_data:
      return
    
    self.last_status_data = status_data
    
    self.log.start_timer('send_image')
    np_img = cv2.cvtColor(np_img_bgr,  cv2.COLOR_BGR2RGB)
    str_enc_img = self.log.np_image_to_base64(np_img)
    msg = {}
    url = self.viewer_config[self.url_key]
    for k in self.viewer_config:
      if k.lower() not in self.except_keys:
        msg[k] = self.viewer_config[k]
    msg[self.image_key] = str_enc_img
    msg["stats"] = self.live
    msg["version"] = "omnidj_v_{}".format(self.version)
    self.D("SND {}B to '{}'...".format(len(str_enc_img), url)) 
    self.log.start_timer('img_post')
    resp = requests.post(url, json=msg)
    self.log.end_timer('img_post')
    if resp.status_code == 200:
      self.n_snd_ok += 1
      self.D("  Payload sent. Resp: {}".format(resp.status_code))
    else:
      self.n_snd_fail += 1
      self.P("  ERROR: Server responded: {}".format(resp.status_code))
    self.log.end_timer('send_image')
    return
  
  
  def _cv_rect(self,np_bgr, left_top,  right_bottom,  color):
    x1, y1 = left_top
    x2, y2 = right_bottom
    w = x2 - x1
    h = y2 - y1
    seg_h = h // 8
    seg_w = w // 8
    coords = [  
              (x1,y1,x1+seg_w,y1),
              (x2-seg_w,y1,x2,y1),
              (x2,y1,x2,y1+seg_h),
              (x2,y2-seg_h,x2,y2),
              (x2-seg_w,y2,x2,y2),
              (x1,y2,x1+seg_w,y2),
              (x1,y2,x1,y2-seg_h),
              (x1,y1+seg_h,x1,y1)
             ]
    for (x1,y1,x2,y2) in coords:
      cv2.line(np_bgr, (x1,y1), (x2,y2), color=color, thickness=2)
    return np_bgr
    
  
  
  
  def _add_stats(self, np_img):
    self.log.start_timer('add_stats')
    if self._frame_count > 20:
      self.fps = '{:.1f} fps'.format(self._frame_count / (time() - self._run_start_tm))

    self._max_pers = max(self._max_pers, self.live[P_KEY])
    self._count_pers.append(self.live[P_KEY])
    avg = round(np.mean(self._count_pers),2)
    add_bar = False
    if time() > (self._last_avg_tm + 10):
      self._count_pers = []
      self._counts.append(avg)   
      self._full_counts.append(avg)
      self._last_avg_tm = time()
      add_bar = True if len(self._counts) > 1 else False
      STAT_COUNT = 3
      if len(self._full_counts) >= STAT_COUNT+1:
        np_status = np.array(list(self._full_counts)[-(STAT_COUNT+1):])
        np_s = np_status[1:] - np_status[:-1]
        c_avg = np_s.mean()
        if c_avg < 0 or np_status.mean() < 0.5:
          self.c_status = {'VALUE':'DOWN', 'COLOR':(0,0,255)}
        elif c_avg > 0:
          self.c_status = {'VALUE':'UP', 'COLOR':CLR_GREEN}
        else:
          self.c_status = {'VALUE':'CONST', 'COLOR':(255,0,0)}
        
    self.live[S_KEY] = self.sentiment
    self.live[AUD_KEY] = self.c_status
    self.live[AAVG_KEY] = avg
    self.live[AMAX_KEY] = self._max_pers
    self.live[ATTN_KEY] = self._get_spec_attention() 
    self.live[RT_KEY] = "{:.1f} mins".format((time() - self._run_start_tm) / 60)
    self.live[LOC_KEY] = self.get_current_location()
    self._stats_add_sep()
    joined_str = self.viewer_config[self.event_key].lower()
    joined = 'join' in joined_str
    self.live[STAT_KEY] = {'VALUE': joined_str, 'COLOR': CLR_GREEN if joined else (0,0,255)}
    self.live[VER_KEY] = self.version + ' / ' + self.fps
    self.live[SROOM_KEY] = self.log.get_machine_name()
    if "user_name" in self.viewer_config:
      self.live[SUSER_KEY] = self.viewer_config['user_name']
    self.live[self._run_start_datetime.strftime("%Y-%m-%d %H:%M:%S")] = 'S'
    self.live[datetime.now().strftime("%Y-%m-%d %H:%M:%S")] = 'C'

  
    top = 10
    left = 25
    inter = 16
    self._draw_window(np_img, inter=inter, up_offset=top)
    for i,k in enumerate(self.live):
      v = self.live[k]
      if type(v) in [dict, OrderedDict]:
        txt = "{:<10} {}".format(k+':',v['VALUE']) 
        clr = v['COLOR']
      else:
        txt = "{:<10} {}".format(k+':',v) 
        if txt[:4] == 'SEP_':
          txt = ' '
        clr = (0,0,0)
      text_w, text_h = self._put_text(np_img, txt,
                                      top=top,
                                      left=left,
                                      color=clr)
      top += text_h + inter
    if add_bar:
      self._np_bar = get_bar_plot_np(self._counts, label='Audience trend')
    
    if self._np_bar is not None:
      h, w = self._np_bar.shape[:2]
      np_img[top:top+h,left:w+left,:] = self._np_bar
    self.log.end_timer('add_stats')
    return np_img
  