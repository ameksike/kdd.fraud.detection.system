from flask import Flask
import sys, os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(PROJECT_DIR))

app = Flask(__name__)

from src import *

if __name__ == '__main__':
    app.run(port=8000,debug=True)