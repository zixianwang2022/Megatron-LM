import sys

sys.path.insert(0, "../")
sys.path.insert(0, "../App")
from App import app

if __name__ == '__main__':
    app.run(debug=True, host='10.33.1.207', port=8084)
