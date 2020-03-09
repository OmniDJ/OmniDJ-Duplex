# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:56:00 2020

@author: Andrei
"""
from libraries.logger import Logger

if __name__ == '__main__':
  url11 = 'https://www.dropbox.com/s/t6qfxiopcr8yvlq/60_xomv_employee01_002_e142_acc0.985.pb?dl=1'
  fn1 = 'model1.pb'
  url12 = 'https://www.dropbox.com/s/akzyk9vcuqluzup/60_xomv_employee01_002_e142_acc0.985.pb.txt?dl=1'
  fn2 = 'model1.txt'
  url21 = 'https://www.dropbox.com/s/tuywpfzv6ueknj6/70_xfgc03_007_e092_acc0.9413.pb?dl=1'
  fn3 = 'model2.pb'
  url22 = 'https://www.dropbox.com/s/5wrvohffl14qfd3/70_xfgc03_007_e092_acc0.9413.pb.txt?dl=1'
  fn4 = 'model2.txt'
  log = Logger(lib_name='MDL',  config_file='config/duplex_config.txt',  TF_KERAS=False)
  
  # download two files in output 
  log.maybe_download(url=[url11, url12],
                     fn=[fn1,fn2],
                     target='output'
                     )
  
  # download a txt in data
  log.maybe_download(url=url12,
                     fn='model1_dup.txt',
                     target='data'
                     )
  
  # download another two files in models with other signature
  log.maybe_download(url={
                      fn3 : url21,
                      fn4 : url22
                      },
                     force_download=True,
                     target='models'
                     )
  # use maybe_download_model
  log.maybe_download_model(url11, 'model1_dup.pb', 
                           force_download=False, 
                           url_model_cfg=url12)