

from libraries.logger import Logger
from duplex.utils import get_tx2_and_config_file
from duplex.duplex_engine import DuplexEngine

import sys

if __name__ == '__main__':

  is_tx2, cfg = get_tx2_and_config_file()
  log = Logger(lib_name='ODJD',  config_file=cfg,  TF_KERAS=True)

  args = [x.upper() for x in sys.argv]
  

  log.P("Args: {}".format(args))
  
  _debug = 'DEBUG' in args
  show_debug = 'NOSHOW' not in args
  use_face = 'NOFACE' not in args
  use_emotion = 'NOEMO' not in args
  show_boxes = 'NOBOX' not in args
    
  
  eng = DuplexEngine(log=log, 
                     runs_on_device=is_tx2, 
                     debug=_debug,
                     use_face=use_face,
                     use_emotion=use_emotion,
                     boxes=show_boxes,
                     )
    
  eng.run(show_debug=show_debug)
  eng.shutdown()

  
  
