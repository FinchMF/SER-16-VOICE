
import sys
import librosa
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import pickle

#################################
# DATA TRANSFORMATION UTILITIES #
#################################

def spec_to_img(spec, eps=1e-6):

    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)

    return spec_scaled


def melspectrogram_db(wav, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):

    wav, sr = librosa.load(wav, sr=22050*2)

    if wav.shape[0]<5*sr:

        wav=np.pad(wav, int(np.ceil((5*sr-wav.shape[0])/2)), mode='reflect')

    else:

        wav=wav[:5*sr]

    spec = librosa.feature.melspectrogram(wav, sr=sr, 
                                          n_fft=n_fft, 
                                          hop_length=hop_length, 
                                          n_mels=n_mels,
                                          fmin=fmin, fmax=fmax)

    spec_db = librosa.power_to_db(spec, top_db=top_db)

    return spec_db


########################
# NEURAL NET UTILITIES #
########################


def set_learning_rate(optimizer, lr):

    for param_group in optimizer.param_groups:

        param_group['lr'] = lr

    return optimizer


def learning_rate_decay(optimizer, epoch, learning_rate):

    if epoch%10==0:

        new_lr = learning_rate / (10**(epoch//10))

        optimizer = set_learning_rate(optimizer, new_lr)
        print(f'[+] Changing Learning Rate to {new_lr}')

        return optimizer

    else:

        return optimizer


def train(model, train_loader, valid_loader, epochs=100, learning_rate=2e-4, decay=True):

    if torch.cuda.is_available():

        device=torch.device('cuda:0')
    else:

        device=torch.device('cpu')

    loss_func = nn.CrossEntropyLoss()

    learning_rate = learning_rate

    opt = optim.Adam(model.parameters(), lr=learning_rate)

    epochs = epochs

    train_losses = []
    valid_losses = []

    for e in range(1, epochs + 1):

        model.train()

        batch_losses = []

        if decay:

            opt = learning_rate_decay(opt, e, learning_rate)

        zx = 0
        for i, data in enumerate(train_loader):

            zx += 1
            z = (f'[i] Training {round(zx/len(train_loader)*100,2)} %')

            x, y = data
            opt.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)

            preds = model(x)
            loss = loss_func(preds, y)
            loss.backward()

            batch_losses.append(loss.item())
            opt.step()

            sys.stdout.write('\r'+z)

        train_losses.append(batch_losses)
        print(f'[i] Epoch - {e} - Train-Loss: {np.mean(train_losses[-1])}')

        model.eval()

        batch_losses = []

        trace_y = []
        trace_preds = []

        zx = 0
        for i, data in enumerate(valid_loader):

            zx += 1
            z = (f'[i] Evaluate {round(zx/len(valid_loader)*100,2)} %')

            x, y = data
            opt.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)

            preds = model(x)

            loss_func(preds, y)

            trace_y.append(y.cpu().detach().numpy())
            trace_preds.append(preds.cpu().detach().numpy())

            batch_losses.append(loss.item())

            sys.stdout.write('\r'+z)

        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_preds = np.concatenate(trace_preds)

        acc = np.mean(trace_preds.argmax(axis=1 == trace_y))

        print(f'[i] Epoch - {e} Valid-Loss: {np.mean(valid_losses[-1])} Valid Accuracy: {acc}')

        return model


#################
# STORING TOOLS #
#################

def save_model(model, fname):

    with open(fname, 'wb') as f:

        torch.save(model, f)


def save_cat_idx(data, fname):

    with open(fname, 'wb') as f:

        pickle.dump(data.idx2cat, f, protocol=pickle.HIGHEST_PROTOCOL)

    
def load_model(fname):

    if torch.cuda.is_available():

        device = torch.device('cuda:0')
    else:

        device = torch.device('cpu')

    model = torch.load(fname, map_location=device)

    return model


def load_cat_idx(fname):

    i2c = pickle.load(open(fname, 'rb'))

    return i2c