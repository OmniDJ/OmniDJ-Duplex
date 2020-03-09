# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:11:19 2020

@author: Andrei
"""

import socket
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


import sys
import requests

import cv2


def prepare_img(np_src, target_h=48, target_w=48, convert_gray=True):
  np_source = np_src.copy()
  assert len(np_source.shape) in [2,3], "Source image must be (H,W,C) or (H,W)"
  if convert_gray and len(np_source.shape) ==3 and np_source.shape[-1] != 1:
    np_source= cv2.cvtColor(np_source, cv2.COLOR_BGR2GRAY)
  shape = (target_h, target_w,np_source.shape[-1]) if len(np_source.shape) == 3 else (target_h, target_w)
  
  np_dest = np.zeros(shape).astype(np.float32)
  src_h, src_w = np_source.shape[:2]
  if src_h > src_w:
    new_h = target_h
    new_w = int(target_h / (src_h/src_w))
  else:
    new_w = target_w
    new_h = int(target_w / (src_w/src_h))

  np_src_mod = cv2.resize(np_source, dsize=(new_w, new_h))
  left = target_w // 2 - new_w // 2
  top = target_h // 2 - new_h // 2
  right = left + new_w
  bottom = top + new_h
  if np_src_mod.max() > 1:
    np_src_mod = (np_src_mod / 255).astype(np.float32)
  np_dest[top:bottom, left:right] = np_src_mod
  return np_dest

def blended_fill_rect(np_bgr, left_top, width_height, opacity=0.8):
  x, y = left_top
  w, h = width_height
  sub_img = np_bgr[y:y+h, x:x+w]
  white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
  opacity = np.clip(opacity, 0.1, 0.9)
  res = cv2.addWeighted(sub_img, 1 - opacity, white_rect, opacity, 1.0)
  
  # Putting the image back to its position
  np_bgr[y:y+h, x:x+w] = res  
  return np_bgr

def get_tx2_and_config_file():
  host = socket.gethostname()
  config_file = 'config/duplex_config.txt'
  if 'omnidj' in host:
    #config_file = 'duplex_config_tx2.txt'
    tx2 = True
  else:
    #config_file = 'duplex_config_local.txt'
    tx2 = False
  return tx2, config_file

  
def get_bar_plot_np(counts, t=10, label=''):
  x = np.arange(len(counts))
  x1 = x[::-1] * (-t)
  xlab = np.array(["{}s".format(i) for i in x1])
  xlab[-1] = 'NOW'
  
  if len(counts) > 7:
    xlab[::2] = ''
  
  fig = Figure(figsize=(3.0,1.8))
  
  canvas = FigureCanvas(fig)
  ax = fig.gca()
  ax.bar(x, counts)
  ax.set_facecolor('#c8c8c8')
  ax.set_xticks(x)
  ax.set_xticklabels(xlab) #, rotation=30)
  #ax.axis('off')
  
  w, h = canvas.get_width_height()
  ax.set_title(label)
  canvas.draw()       # draw the canvas, cache the renderer
  
  np_img = np.fromstring(canvas.tostring_rgb(), dtype='uint8')  
  np_img = np_img.reshape((h,w,3))

  return np_img



def _print_download_progress(count, block_size, total_size):
  """
  Function used for printing the download progress.
  Used as a call-back function in maybe_download_and_extract().
  """

  # Percentage completion.
  pct_complete = float(count * block_size) / total_size

  # Limit it because rounding errors may cause it to exceed 100%.
  pct_complete = min(1.0, pct_complete)

  # Status-message. Note the \r which means the line should overwrite itself.
  msg = "\r- Download progress: {0:.1%}".format(pct_complete)

  # Print it.
  sys.stdout.write(msg)
  sys.stdout.flush()
  return



def download_file_from_google_drive(file_id, destination):
  URL = "https://docs.google.com/uc?export=download"

  session = requests.Session()

  response = session.get(URL, params = { 'id' : file_id }, stream = True)
  token = get_confirm_token(response)

  if token:
    params = { 'id' : file_id, 'confirm' : token }
    response = session.get(URL, params = params, stream = True)
    
  save_response_content(response, destination)    
  return

def get_confirm_token(response):
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      return value
  return None

def save_response_content(response, destination):
  CHUNK_SIZE = 32768
  down = 0
  with open(destination, "wb") as f:
    for chunk in response.iter_content(CHUNK_SIZE):
      if chunk: # filter out keep-alive new chunks
        down += len(chunk) / (1024**2)
        print("\rDownloaded so far {:.1f} MB".format(down), end='', flush=True)
        f.write(chunk)
  print("\r")
  return


def blur_pers(np_img, left, top, right, bottom, DIRECT=False):
  ksize = (91,91)
  sigmaX = 0
  width = right - left
  height = bottom - top
  if width > height:
    x_scal = 20
    y_scal = 8
  else:
    x_scal = 8
    y_scal = 20
  if DIRECT:
    y1, x1, y2, x2 = int(top), int(left), int(bottom), int(right)
  else:
    y1 = top  + (height // y_scal)
    x1 = left + (width // x_scal)
    y2 = y1 + height // 2
    x2 = x1 + width - (width // x_scal * 2)
  np_src = np_img[y1:y2,x1:x2,:]
  np_dst = cv2.GaussianBlur(np_src, ksize, sigmaX)
  np_img[y1:y2,x1:x2,:] = np_dst
  return np_img


if __name__ == '__main__':
  import matplotlib.pyplot as plt

  def test(img):
    plt.figure()
    plt.imshow(img)
    plt.title('img {}'.format(img.shape))
    plt.show()
    imgb = prepare_img(img, 48, 48)
    plt.figure()
    plt.imshow(imgb)
    plt.title('img {}'.format(imgb.shape))
    plt.show()
    return imgb


  imgs = ['img1.png', 'img2.png', 'img3.png', 'img4.png']
  for fn in imgs:
    im = cv2.imread('duplex/'+fn)
    res = test(im)
