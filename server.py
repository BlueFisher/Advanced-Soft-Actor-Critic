import threading
from flask import Flask

app = Flask('learner')

@app.route('/')
def main():
    return jsonify({
        'succeeded': True
    })

def loop():
    while True:
        pass

threading.Thread(target=app.run,kwargs={
    'host': '0.0.0.0', 'port': 9999
}).start()
threading.Thread(target=loop).start()
