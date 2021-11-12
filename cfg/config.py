import os

DEBUG = os.getenv('DEBUG_MODE', False)
DB_NAME = os.environ.get('DB_NAME', 'localhost') 
DB_USER = os.environ.get('DB_USER', 'demo') 
DB_PASS = os.environ.get('DB_PASS', 'demo') 