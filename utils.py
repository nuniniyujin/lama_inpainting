import datetime as dt
import matplotlib.pyplot as plt

import torch
from torchvision.utils import save_image


def saving_model(model,epoch,batch):
    now = dt.datetime.now()
    dt_string = now.strftime("%d-%m-%Y %Hh%M")
    torch.save(model.state_dict(), f"../model_weights_conv_change_new_mask/model{epoch}epochs{batch}batch{dt_string}.pth")
    print(f"model {epoch} epochs {batch} batch {dt_string}.pth saved")

def saving_image(image_reconstructed,epoch):
    now = dt.datetime.now()
    dt_string = now.strftime("%d-%m-%Y %Hh%M")
    save_image(image_reconstructed,f"../images_saved_conv_change_new_mask/image{epoch}epochs{dt_string}.png")
