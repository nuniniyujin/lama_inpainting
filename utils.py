import os
import datetime as dt
import matplotlib.pyplot as plt

import torch
from torchvision.utils import save_image


def saving_model(model,epoch,batch,base_dir):
    if not os.path.exists(base_dir):
        # Create a new directory because it does not exist 
        os.makedirs(base_dir)

    now = dt.datetime.now()
    dt_string = now.strftime("%d-%m-%Y %Hh%M")
    torch.save(model.state_dict(), f"{base_dir}/model{epoch}epochs{batch}batch{dt_string}.pth")
    print(f"model {epoch} epochs {batch} batch {dt_string}.pth saved")

def saving_image(image_reconstructed,epoch,base_dir):
    if not os.path.exists(base_dir):
        # Create a new directory because it does not exist 
        os.makedirs(base_dir)

    now = dt.datetime.now()
    dt_string = now.strftime("%d-%m-%Y %Hh%M")
    save_image(image_reconstructed,f"{base_dir}/image{epoch}epochs{dt_string}.png")
