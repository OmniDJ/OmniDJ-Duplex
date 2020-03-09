export PYTHONPATH=$PYTHONPATH:~/omnidj
nohup python3 duplex/main.py noshow &
nohup python3 duplex/duplex_config_server.py &
