# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:18:50 2020

"""


import flask 
from libraries.logger import Logger



__VER__ = '1.0.0.4'
fn_config = 'viewer_config.txt'

log = None


def config_update():
  request = flask.request
  method = flask.request.method
  args_data = request.args
  form_data = request.form
  json_data = request.json
  if method == 'POST':
    dct_config = form_data
    if len(dct_config) == 0:
        # params in json?
        dct_config = json_data
  else:
    dct_config = args_data
  log.P("*"*50)
  log.P("LocalConfig v{} received config data:".format(__VER__))
  dct_prev = log.load_data_json(fn_config)
  log.P("  Current: {}".format(dct_prev))
  log.save_data_json(dct_config, fn_config )
  dct_new = log.load_data_json(fn_config)
  log.P("  New:     {}".format(dct_config))
  jresponse = flask.jsonify({
            "RECEIVED_CONFIG_UPDATE": dct_new})    
  jresponse.headers["Access-Control-Allow-Origin"] = "*"
  jresponse.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, DELETE"
  jresponse.headers["Access-Control-Allow-Headers"] = "Content-Type"
  log.P("*"*50)
  return jresponse
  
  

if __name__ == '__main__':
  config_file = 'config/duplex_config.txt'
  log = Logger(lib_name='ODJcfg', config_file=config_file,  TF_KERAS=False)
  log.P("Starting OmniDJ local config server {}".format(__VER__))
  dct_viewer = log.load_data_json(fn_config)
  log.P("Currend config:\n{}".format(dct_viewer))
  
  app = flask.Flask('LocalConfigServer')
  app.add_url_rule(rule='/config', 
                   endpoint="LocalConfig", 
                   view_func=config_update, 
                   methods = ['GET', 'POST','OPTIONS']
                   )  
  app.run(host='127.0.0.1', port=5500)