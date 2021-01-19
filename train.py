
import torch
import data
import CNN_models as m 
import utils 

def prepare_data():

    train_data, valid_data, test_data = data.read_data(data.path)
    train_loader, valid_loader, test_loader = data.fetch_loaders(train_data, 
                                                                  valid_data,
                                                                  test_data)

    utils.save_cat_idx(train_data, 'models/idx2cat.pkl')

    return train_loader, valid_loader, test_loader


def train_resNet(train_loader, valid_loader):

    model = m.RES().gen_resnet()

    trained_model = utils.train(model, train_loader, valid_loader)

    utils.save_model(trained_model, 'models/Emotion_ResNet34.pth')

    return trained_model


if __name__ == '__main__':

    train_loader, valid_loader, _ = prepare_data()

    trained_model = train_resNet(train_loader, valid_loader)



