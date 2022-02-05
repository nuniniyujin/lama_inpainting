from tqdm import tqdm

from loss import Loss
from utils import saving_image, saving_model


def train_loop(model,optimizer_g,optimizer_d,train_dataloader,test_dataloader,start_epoch=0,epochs=50,val_rate = 10,sav_rate=200,verbose = True,display_results = True):
    """Training loop function
    params :
    -- model : LaMa model
    -- optimizer : Optimizer of model
    -- train_dataloader : dataloader of training dataset
    -- test_dataloader : dataloader of test dataset
    -- epochs (default : 50) : number of epochs for training our model
    -- val_rate (default : 10) : our model will be evaluated every val_rate epochs
    -- sav_rate (default : 10) : our model weights will be saved every sav_rate batches
    -- verbose (default : True) : allow many logs for information
    """

    len_trainset = len(train_dataloader)

    Final_loss = []
    HRF_PL=[]
    Disc_loss = []
    Gene_loss = []
    Adv_loss = []
    feat_match_loss = []
    R1_loss = []

    print('################ TRAINING STARTED ################')
    for epoch in range(start_epoch, epochs):

        Final_loss_epoch = 0
        HRF_PL_epoch =0
        Disc_loss_epoch = 0
        Gene_loss_epoch = 0
        Adv_loss_epoch = 0
        feat_match_loss_epoch = 0
        R1_loss_epoch = 0
        Total_loss_g = 0
        Total_loss_d = 0
        perceptual_loss = 0
        Total_adv_l=0
        Total_HRF=0
        Total_pl=0

        print(f"Epoch {epoch+1} of {epochs}")
        with tqdm(total=len_trainset, unit_scale=True, postfix={'adv_loss':0.0,'HRF':0.0,'Perceptual_loss':0.0,'gen total loss':0.0, 'disc loss':0.0}, ncols=150) as pbar:
            for i, (stack, gt_image) in enumerate(train_dataloader):
                stack = stack.to(device)

                mask = stack[:,[3],:,:] #recover mask to feed Loss
                gt_image = gt_image.to(device)
                gt_image = gt_image/255 #added to normalize gt

                image_reconstructed = model(stack)

                ################### LOSS CALCULATION FUNCTIONS ####################


                Adv_loss_epoch,HRF_PL_epoch,perceptual_loss,Loss_g = LAMA_loss(image_reconstructed,gt_image,mask,net_type="generator")

                Loss_d = LAMA_loss(image_reconstructed,gt_image,mask,net_type="discriminator") #not sure if this is giving me the discriminator loss

                Total_adv_l +=Adv_loss_epoch.item()
                Total_HRF += HRF_PL_epoch.item()
                Total_pl += perceptual_loss.item()
                Total_loss_g += Loss_g.item()
                Total_loss_d += Loss_d.item()
                pbar.set_postfix({'adv_loss':Total_adv_l/(i+1),'HRF':Total_HRF/(i+1),'Perceptual_loss':Total_pl/(i+1),'gen total loss':Total_loss_g/(i+1), 'disc loss':Total_loss_d/(i+1)})
                pbar.update(1)

                # Backward and optimize the models

                optimizer_g.zero_grad()
                Loss_g.backward()
                optimizer_g.step()

                optimizer_d.zero_grad()
                Loss_d.backward()
                optimizer_d.step()

                Final_loss_epoch+=Loss_g.detach().cpu()

                if display_results and i%100 == 0:
                    image_reconstructed = image_reconstructed.detach().cpu()
                    saving_image(image_reconstructed,epoch) #save image image



                #####################################################################
                if i % sav_rate == 0 :
                    saving_model(model,epoch+1,i)



    saving_model(model,'final',0)

    print('################ TRAINING FINISHED ################')
