from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse

import os
import sys
import gunicorn, uvicorn
import aiohttp
import asyncio

import itertools

import torch
import pickle
import utils
import numpy as np

# set application
app = Starlette()

if torch.cuda.is_available():

    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# set model and categorical lookup table
with open('models/idx2cat.pkl', 'rb') as f:
    idx2cat = pickle.load(f)

model = torch.load('models/Emotion_Resnet34.pth', map_location=device)


def predict_from_upload(model, cats, fpath):

    spec = utils.melspectrogram_db(fpath)
    spec = utils.spec_to_img(spec)
    spec = torch.from_numpy(spec).to(device, dtype=torch.float32)

    preds = model.forward(spec.reshape(1,1,*spec.shape))[0].cpu().detach().numpy()
    pred = {name: preds[idx] for idx, name in cats.items()}
    result = list(pred.items())

    s = 0

    for c, val in result:

        s+=np.exp(val)
    for i in range(len(result)):

        result[i]=(result[i][0], np.exp(result[i][1])/s)
    result.sort(key=lambda x:x[1], reverse=True)
    N = 1
    return JSONResponse(dict(itertools.islice(dict(result).items(), N)))


@app.route('/')
def form(request):

    with open('templates/home.html', 'r') as f:

        st=f.read()

    return HTMLResponse(st)

@app.route('/upload', methods=['POST'])
async def upload(request):

    data = await request.form()

    f_id = 'inputFile'

    bytes = await (data[f_id].read())

    with open(data[f_id].filename, 'wb') as f:

        f.write(bytes)

    result=predict_from_upload(model, idx2cat, data[f_id].filename)
    os.remove(data[f_id].filename)

    return result

async def get_bytes(url):

    async with aiohttp.ClientSession() as session:

        async with session.get(url) as response:

            return await response.read()

@app.route('/uploadajax', methods=['POST'])
async def upldfile(request):

    if request.method == 'POST':

        data = await request.form()

        f_id = 'inputFile'

        bytes = await (data[f_id].read())

        with open(data[f_id].filename, 'wb') as f:

            f.write(bytes)

        results=predict_from_upload(model, idx2cat, data[f_id].filename)
        os.remove(data[f_id].filename)


        return results

if __name__ == '__main__':

    uvicorn.run(app, host='0.0.0.0', port=500, log_level='info', debug=True)