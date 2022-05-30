from rq import Connection, Worker

from core import conn, queue

if __name__ == "__main__":
    with Connection(conn):
        w = Worker([queue])
        w.work()