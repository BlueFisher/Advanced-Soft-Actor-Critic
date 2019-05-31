import requests

try:
    requests.get('http://127.0.0.1')
except requests.ConnectionError:
    print('timeout')
except Exception as e:
    print(type(e))
    print(e)