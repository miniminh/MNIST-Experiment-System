from fastapi import FastAPI, Path, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
import glob
from typing import Annotated, Union
from trainer import create_train_job, create_search_job, load_model_and_eval, get_onnx
from datetime import datetime
import csv
from pathlib import Path
from pydantic import BaseModel
import os
import uvicorn

from utils import MODEL_SAVE_DIR, JOB_SAVE_DIR, PARAMS_SAVE_DIR, load_params, save_progress

app = FastAPI()

@app.get("/")
async def get():
    return FileResponse("templates/index.html")

@app.get("/progress")
async def get_progress():
    res = {}
    res['progress'] = []
    res['val_loss'] = []
    res['val_acc'] = []
    res["exp_id"] = []
    for filename in glob.iglob(f'{JOB_SAVE_DIR}/*'):
        # print(filename)
        with open(filename) as f:
            csv_reader = csv.reader(f, delimiter=',')
            val_loss = 0 
            val_acc = 0
            this_epoch = 0 
            epoch = 0
            for row in csv_reader:
                val_loss = float(row[0]) 
                val_acc = float(row[1])
                this_epoch = int(row[2])
                epoch = int(row[3])
            # print(row)
        res['progress'].append(float((this_epoch + 1)/(epoch)))
        # print(res['progress'])
        res["val_loss"].append(val_loss)
        res["val_acc"].append(val_acc)
        res["exp_id"].append(Path(filename).stem)
    return res

def assign():
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    return dt_string

class ModelParams(BaseModel):
    learningRate: float     
    batchSize: int
    epoch: int 

@app.post("/start_experiment")
async def start(
    params: ModelParams,
    background_tasks: BackgroundTasks = None
):
    exp_id = assign()
    learningRate = params.learningRate
    batchSize = params.batchSize
    epoch = params.epoch
    print(learningRate, batchSize, epoch)
    if learningRate == 0 or batchSize == 0 or epoch == 0: 
        background_tasks.add_task(create_search_job, learningRate, batchSize, epoch, exp_id)
        return {"message": "batch jobs started", "exp_id": exp_id}
    save_progress((0,0),exp_id,-1,epoch)
    background_tasks.add_task(create_train_job, learningRate, batchSize, epoch, exp_id)
    return {"message": "jobs started", "exp_id": exp_id}


@app.get("/model_access")
async def get_model(exp_id: Union[str, None] = None ):
    return FileResponse("templates/model_access.html")

@app.get("/download_model")
async def download_model(exp_id: Union[str, None] = None ):
    model = get_onnx(exp_id)
    return model 

@app.get("/evaluate")
async def evaluate_model(exp_id: Union[str, None] = None ):
    params = load_params(exp_id)
    print(params, exp_id)
    result = load_model_and_eval(exp_id)
    return {"status": "success", "result": result, "params": params}


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=7504)