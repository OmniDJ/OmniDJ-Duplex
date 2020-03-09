# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:53:55 2020

@author: damia
"""

import numpy as np
import tensorflow as tf

from libraries.logger import Logger

import sys

def gpu_memory_in_use(gpu='/device:GPU:0', log=None):
  from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
  with tf.device(gpu):  
    bytes_in_use = BytesInUse()
  with tf.Session() as sess:
    val = sess.run(bytes_in_use)
  if log is not None:
    log.P("  GPU free mem: {:.1f} Gb".format(
        val / (1024**3)))
  return val

if __name__ == '__main__':
  cfg = 'config/duplex_config.txt'
  log = Logger(lib_name='MGS',  config_file=cfg,  TF_KERAS=True)
  
  yolo = log.LoadGraphFromModels('01_1712_y_720_1280_c.pb')
  face = log.LoadGraphFromModels('20_190301_mob_ssd_faces.pb')
  
  config_proto = tf.compat.v1.ConfigProto()
  config_proto.gpu_options.allow_growth = True  
  yolo_sess = tf.compat.v1.Session(graph=yolo,  config=config_proto)
  log.P("Created yolo session")

  config_proto = tf.compat.v1.ConfigProto()
  config_proto.gpu_options.allow_growth = True  
  face_sess = tf.compat.v1.Session(graph=face,  config=config_proto)
  log.P("Created ssd session")

  
  tf_face_classes = face_sess.graph.get_tensor_by_name(
      "detection_classes:0")
  tf_face_scores = face_sess.graph.get_tensor_by_name(
      "detection_scores:0")
  tf_face_boxes = face_sess.graph.get_tensor_by_name(
      "detection_boxes:0")
  tf_face_dets = face_sess.graph.get_tensor_by_name(
      "num_detections:0")
  tf_face_input = face_sess.graph.get_tensor_by_name(
      "image_tensor:0")

  tf_learning_phase = yolo_sess.graph.get_tensor_by_name(
      "keras_learning_phase:0")
  tf_classes = yolo_sess.graph.get_tensor_by_name(
      "YOLO_OUTPUT_CLASSES"+":0")
  tf_scores = yolo_sess.graph.get_tensor_by_name(
      "YOLO_OUTPUT_SCORES"+":0")
  tf_boxes = yolo_sess.graph.get_tensor_by_name(
      "YOLO_OUTPUT_BOXES"+":0")
  tf_yolo_input = yolo_sess.graph.get_tensor_by_name(
      "input_1"+":0")
  
  log.P("Done all tensors")

  
  args = [x.upper() for x in sys.argv]

  np_inp_yolo = np.random.rand(1, 608, 608, 3)
  np_inp_face = np.random.rand(1, 720, 1280, 3)
  
  
  if 'YOLO' in args:
    t1 = 'yolo'
    t2 = 'face'
    tns1 = [tf_classes, tf_scores, tf_boxes]
    tns2 = [tf_face_classes, tf_face_scores, tf_face_boxes]
    sess1 = yolo_sess
    sess2 = face_sess
    f1 = {tf_yolo_input: np_inp_yolo,
          tf_learning_phase: 0}
    f2 = {tf_face_input: np_inp_face}
  else:
    t1 = 'face'
    t2 = 'yolo'
    tns1 = [tf_face_classes, tf_face_scores, tf_face_boxes]
    tns2 = [tf_classes, tf_scores, tf_boxes]
    sess1 = face_sess
    sess2 = yolo_sess
    f2 = {tf_yolo_input: np_inp_yolo,
          tf_learning_phase: 0}
    f1 = {tf_face_input: np_inp_face}
    
  for i in range(5):
    log.P("*"*30)
    log.P("Iteration {} - {}".format(i+1, t1))
    log.start_timer(t1)
    res1 = sess1.run(tns1, feed_dict=f1)
    log.end_timer(t1)
    log.P("  session run done.")
    gpu_memory_in_use(log=log)
    log.P("*"*30)
    log.P("Iteration {} - {}".format(i+1, t2))
    log.start_timer(t2)
    res2 = sess2.run(tns2, feed_dict=f2)
    log.end_timer(t2)
    log.P("  session run done.")
    gpu_memory_in_use(log=log)
  
  log.show_timers()
      
    

