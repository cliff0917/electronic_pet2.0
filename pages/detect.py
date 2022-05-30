import os
import uuid
import dash
import dash_uploader as du
import dash_bootstrap_components as dbc
from collections import namedtuple
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
from rq.exceptions import NoSuchJobError
from rq.job import Job

import globals
from core import app, conn, queue
from tasks import predict

Result = namedtuple(
    "Result", ["result", "progress", "collapse_is_open", "finished_data", "isCompleted"]
)

def serve_layout():
    layout = html.Div(
        [
            dcc.Store(id="submitted-store"),
            dcc.Store(id="finished-store"),
            du.Upload(
                id='uploader',
                text='點擊或拖曳影片',
                text_completed='已上傳影片: ',
                cancel_button=True,
                pause_button=True,
                filetypes=['mp4', 'avi'],
                default_style={
                    'background-color': '#fafafa',
                    'font-weight': 'bold',
                    "margin-top": 55,
                },
                upload_id='my-upload'
            ),
            dcc.Interval(id="interval", interval=250, n_intervals=0),
            dbc.Collapse(
                dbc.Progress(value=0, className="mb-3",
                             style={'margin-top': 10, 'margin-left': '5px'}, id='progress'),
                id="collapse",
            ),
            html.H3(id="output", style={'margin-top': 10, 'margin-left': '5px'}),
        ],
        style={"padding": "1rem"}
    )
    return layout

@callback(
    Output("submitted-store", "data"),
    Input('uploader', 'isCompleted'),
    State('uploader', 'fileNames')
)
def show_upload_status(isCompleted, fileNames):
    # 當有影片被 submitted 成功
    if isCompleted:
        id_ = str(uuid.uuid4()) # get store id
        queue.enqueue(predict, fileNames[0], job_id=id_) # 加入 queue 中監測
        return {"id": id_} # 返回 store id
    return {}

@callback(
    [
        Output("output", "children"),
        Output("progress", "value"),
        Output("collapse", "is_open"),
        Output("finished-store", "data"),
        Output('uploader', 'isCompleted'),
    ],
    Input("interval", "n_intervals"),
    State("submitted-store", "data"),
)
def retrieve_output(n, submitted):
    # 當有影片被 submitted 時, 週期性更新 progress bar
    if n and submitted:
        try:
            job = Job.fetch(submitted["id"], connection=conn)

            # 工作結束, 返回 result, store id
            if job.get_status() == "finished":
                return Result(
                    result=job.result,
                    progress=100,
                    collapse_is_open=False,
                    finished_data={"id": submitted["id"]},
                    isCompleted=False,
                )

            # 工作執行中, 獲得 progress 並更新 progress bar
            progress = job.meta.get("progress", 0)

            # 當進度為0時, 代表還在初始化 detector
            if progress == 0:
                return Result(
                    result=[
                        '初始化環境中，請稍後',
                        dbc.Spinner(size="lg", color="primary",
                                    spinner_style={'margin-left': '15px', 'width': '40px', 'height': '40px'}
                        )
                    ],
                    progress=progress,
                    collapse_is_open=True,
                    finished_data=dash.no_update,
                    isCompleted=dash.no_update,
                )

            # 當進度不為0時, 更新 progress bar
            return Result(
                result=f"處理中 - 已完成 {progress:.2f}%",
                progress=progress,
                collapse_is_open=True,
                finished_data=dash.no_update,
                isCompleted=dash.no_update,
            )

        except NoSuchJobError:
            return Result(
                result="發生錯誤...",
                progress=None,
                collapse_is_open=False,
                finished_data=dash.no_update,
                isCompleted=dash.no_update,
            )

    # 網頁最初狀態(什麼都還沒被 submitted 的時候)
    return Result(
        result=None, progress=None, collapse_is_open=False, finished_data={}, isCompleted=dash.no_update,
    )

@callback(
    Output("interval", "disabled"),
    [
        Input("submitted-store", "data"),
        Input("finished-store", "data")
    ],
)
def disable_interval(submitted, finished):
    if submitted:
        # 當工作結束時, 關閉 interval 週期性更新 progress bar
        if finished and submitted["id"] == finished["id"]:
            return True
        return False # 工作尚未結束, 開啟 interval
    return True # 沒工作時, 關閉 interval