# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 07:51:53 2020

@author: Andrei
"""
import numpy as np
import tensorflow as tf

from libraries.lummetry_layers.utils import sequential_cuda_to_cpu, _convert_rnn_weights    
    

def _create_layer(layer):
  ltype = type(layer)
  if ltype == tf.keras.layers.CuDNNLSTM:
    new_layer = tf.keras.layers.LSTM(
                                units=layer.units,
                                return_sequences=layer.return_sequences,
                                return_state=layer.return_state,
                                stateful=layer.stateful,
                                name='cpu_'+layer.name,          

                                recurrent_activation='sigmoid', # must!
                                )
  elif ltype == tf.keras.layers.CuDNNGRU:
    new_layer = tf.keras.layers.GRU(
                                units=layer.units,
                                return_sequences=layer.return_sequences,
                                return_state=layer.return_state,
                                stateful=layer.stateful,
                                name='cpu_'+layer.name,          

                                reset_after=True, # must! 
                                recurrent_activation='sigmoid', # must!
                                )
  else:
    new_layer = layer.__class__.from_config(layer.get_config())
  return new_layer

def model_cuda_to_cpu(model):
  new_model = tf.keras.models.clone_model(m_cuda, clone_function=_create_layer)
  for i, layer in enumerate(model.layers):
    new_layer = new_model.layers[i]
    ltype = type(layer)
    weights = layer.get_weights()
    if len(weights) > 0:
      if ltype  in [tf.keras.layers.CuDNNLSTM, tf.keras.layers.CuDNNGRU]:
        new_weights = _convert_rnn_weights(new_layer, weights=weights)
      else:
        new_weights = weights
      new_layer.set_weights(new_weights)
  return new_model

if __name__ == '__main__':
  lstms = [32, 64, 128, 256]
  shape = (10,1)
  np_inp = np.random.normal(size=shape).reshape((1,)+shape)
  np_inp *= 1000
  for n in lstms:
    n_lstm = n
    n_gru = n * 2
    tf_inp = tf.keras.layers.Input(shape=shape)
    tf_x = tf_inp
    tf_x = tf.keras.layers.CuDNNLSTM(n_lstm, return_sequences=True)(tf_x)
    tf_lstm_out = tf_x
    tf_x = tf.keras.layers.concatenate([tf_lstm_out, tf_inp])
    tf_x = tf.keras.layers.CuDNNGRU(n_gru)(tf_x)
    tf_gru_out = tf_x
    tf_x2 = tf.keras.layers.CuDNNGRU(n_gru)(tf_inp)    
    tf_x = tf.keras.layers.add([tf_x2, tf_gru_out])
    tf_x = tf.keras.layers.Dense(1)(tf_x)
    m_cuda = tf.keras.models.Model(tf_inp, tf_x)
    m_cuda.compile(loss='mse', optimizer='adam')  # lets say it is loaded
        
    pcuda = m_cuda.predict(np_inp)[0,0]
    
    m_cpu = model_cuda_to_cpu(m_cuda)
    
    pcpu = m_cpu.predict(np_inp)[0,0]
    m_cpu.summary()
    print("lstm({}) gru ({})\n cuda: {}\n cpu:  {}\n diff: {:.1e}".format(
        n_lstm, n_gru, pcuda, pcpu, abs(pcuda - pcpu)))
  
