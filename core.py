import os
import dash
import dash_bootstrap_components as dbc
import flask
import redis
from rq import Queue

server = flask.Flask(__name__)

app = dash.Dash(
    server=server,
    serve_locally=True,
)

# redis connection and RQ queue.
redis_url = os.getenv("REDISTOGO_URL", "redis://localhost:6379")
conn = redis.from_url(redis_url)
queue = Queue(connection=conn, default_timeout=3600) # 處理時間不可超過 3600 秒
