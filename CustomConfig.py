from gunicorn.config import Config
class CustomConfig(Config):
    worker_timeout = 180
