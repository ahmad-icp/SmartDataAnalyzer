import urllib.request
import sys
url='http://127.0.0.1:8503'
try:
    with urllib.request.urlopen(url, timeout=5) as r:
        data = r.read(800)
        print('OK', r.status)
        print(data.decode('utf-8', errors='ignore'))
except Exception as e:
    print('ERR', e)
    sys.exit(1)
