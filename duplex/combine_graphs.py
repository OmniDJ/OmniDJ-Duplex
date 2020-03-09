# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:25:38 2020

@author: Andrei


def combine_graphs(g1, g2, g1_input, g2_input):
  with tf.Graph().as_default() as g1: # actual model
    in1 = tf.placeholder(tf.float32,name="input")
    ou1 = tf.add(in1,2.0,name="output")
  with tf.Graph().as_default() as g2: # model for the new output layer
    in2 = tf.placeholder(tf.float32,name="input")
    ou2 = tf.add(in2,2.0,name="output")

  gdef_1 = g1.as_graph_def()
  gdef_2 = g2.as_graph_def()

  with tf.Graph().as_default() as g_combined: #merge together
    x = tf.placeholder(tf.float32, name="actual_input") # the new input layer
    # Import gdef_1, which performs f(x).
    # "input:0" and "output:0" are the names of tensors in gdef_1.
    y, = tf.import_graph_def(gdef_1, input_map={"input:0": x},
                             return_elements=["output:0"])
    # Import gdef_2, which performs g(y)
    z, = tf.import_graph_def(gdef_2, input_map={"input:0": y},
                             return_elements=["output:0"])
    

"""
import tensorflow as tf
from libraries.logger import Logger

def combine_graphs(graph1, graph2, name1='g1', name2='g2'):
  gdef_1 = graph1.as_graph_def()
  gdef_2 = graph2.as_graph_def()  
  with tf.Graph().as_default() as g_combined:
    tf.import_graph_def(graph_def=gdef_1, name=name1)
    tf.import_graph_def(graph_def=gdef_2, name=name2)
  return g_combined
    
if __name__ == '__main__':
  log = Logger(lib_name='GC',  config_file='config/duplex_config.txt')
  g1 = log.LoadGraphFromModels('01_1712_y_720_1280_c.pb')
  n1 = 'yolo'
  g2 = log.LoadGraphFromModels('20_190301_mob_ssd_faces.pb')
  n2 = 'face'
  gc = combine_graphs(g1,g2, name1=n1, name2=n2)
  
  config_proto = tf.compat.v1.ConfigProto()
  config_proto.gpu_options.allow_growth = True  
  sess = tf.compat.v1.Session(graph=gc,  config=config_proto)

  earning_phase = sess.graph.get_tensor_by_name(
      n1+"/keras_learning_phase:0")
  tf_classes = sess.graph.get_tensor_by_name(
      n1+"/YOLO_OUTPUT_CLASSES"+":0")
  tf_scores = sess.graph.get_tensor_by_name(
      n1+"/YOLO_OUTPUT_SCORES"+":0")
  tf_boxes = sess.graph.get_tensor_by_name(
      n1+"/YOLO_OUTPUT_BOXES"+":0")
  tf_yolo_input = sess.graph.get_tensor_by_name(
      n1+"/input_1"+":0")
  
  tf_face_classes = sess.graph.get_tensor_by_name(
      n2+"/detection_classes:0")
  tf_face_scores = sess.graph.get_tensor_by_name(
      n2+"/detection_scores:0")
  tf_face_boxes = sess.graph.get_tensor_by_name(
      n2+"/detection_boxes:0")
  tf_face_dets = sess.graph.get_tensor_by_name(
      n2+"/num_detections:0")
  tf_face_input = sess.graph.get_tensor_by_name(
      n2+"/image_tensor:0")
  
