import socket
import sys
s=socket.socket()
try:
    s.settimeout(1)
    s.connect(('127.0.0.1',8503))
    print('OPEN')
    s.close()
except Exception as e:
    print('CLOSED', e)
    sys.exit(1)
