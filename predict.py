
import torch
import utils

def predict(wav, cats, model):

    if torch.cuda.is_available():

        device = torch.device('cuda:0')

    else: 

        device = torch.device('cpu')

    spec = utils.spec_to_img(utils.melspectrogram_db(wav))
    spec = torch.from_numpy(spec).to(device, dtype=torch.float32)

    preds = model.forward(spec.reshape(1,1,*spec.reshape))
    idx = preds.argmax(dim=1).cpu().detach().numpy().ravel()[0]

    return cats[idx]