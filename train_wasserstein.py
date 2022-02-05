from tqdm import tqdm

from loss_wasserstein import Loss_Wasserstein
from utils import saving_image, saving_model

def train_loop_Wassertstein(model,optimizer_g,optimizer_d,train_dataloader,test_dataloader,epochs=50,val_rate = 10,sav_rate=200,verbose = True,display_results = True,WEIGHT_CLIP = 0.01, disc_iter = 5,clip_weights = False):
  """Training loop function
  params :
  -- model : LaMa model
  -- optimizer_g : Optimizer of Lama model
  -- optimizer_d : Optimizer of discriminator
  -- train_dataloader : dataloader of training dataset
  -- test_dataloader : ##NOT IMPLEMENTED YET## dataloader of test dataset           
  -- epochs (default : 50) : number of epochs for training our model
  -- val_rate (default : 10) : ##NOT IMPLEMENTED YET## our model will be evaluated every val_rate epochs
  -- sav_rate (default : 10) : our model weights will be saved every sav_rate batches
  -- verbose (default : True) : ##NOT IMPLEMENTED YET## allow many logs for information
  -- WEIGHT_CLIP (default : 0.01) : to clip the gradients value inside the interval
  -- disc_iter (default : 5) : We update discriminator disc_iter times more than generator
  """

  len_trainset = len(train_dataloader)

  print('################ TRAINING STARTED ################')
  for epoch in range(epochs):

    Total_loss_g = 0    
    Total_loss_d = 0
    perceptual_loss = 0
    Total_adv_l=0
    Total_HRF=0
    Total_pl=0
    Final_loss_epoch=0

    print(f"Epoch {epoch+1} of {epochs}")
    with tqdm(total=len_trainset, unit_scale=True, postfix={'adv_loss':0.0,'HRF':0.0,'Perceptual_loss':0.0,'gen total loss':0.0, 'disc loss':0.0}, ncols=150) as pbar:
      for i, (stack, gt_image) in enumerate(train_dataloader):
        stack = stack.to(device)

        mask = stack[:,[3],:,:] #recover mask to feed Loss
        gt_image = gt_image.to(device)
        gt_image = gt_image/255 #added to normalize gt

        image_reconstructed = model(stack)

        ################### LOSS CALCULATION FUNCTIONS ####################
        #We update discriminator more times than generator
        Loss_d = LAMA_loss(image_reconstructed,gt_image,mask,net_type="discriminator")
        optimizer_d.zero_grad() 
        Loss_d.backward(retain_graph=True)
        optimizer_d.step()

        if clip_weights:
          for p in LAMA_loss.disc.parameters():
                  p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        Total_loss_d += Loss_d.item()
                

        #We update generator less times than generator
        if i % disc_iter == 0: 
          gen_loss,HRF_PL_epoch,perceptual_loss,Loss_g = LAMA_loss(image_reconstructed,gt_image,mask,net_type="generator")

          optimizer_g.zero_grad() 
          Loss_g.backward()
          optimizer_g.step()

          Total_adv_l +=gen_loss.item()
          Total_HRF += HRF_PL_epoch.item()
          Total_pl += perceptual_loss.item()
          Total_loss_g += Loss_g.item()

          pbar.set_postfix({'gen_loss':Total_adv_l/((i/disc_iter)+1),'HRF':Total_HRF/((i/disc_iter)+1),'Perceptual_loss':Total_pl/((i/disc_iter)+1),'gen total loss':Total_loss_g/((i/disc_iter)+1),'disc loss':Total_loss_d/(i+1)})
          pbar.update(disc_iter)

        if display_results and i%100 == 0:
          image_reconstructed = image_reconstructed.detach().cpu()
          stack = stack.detach().cpu()
          saving_image(image_reconstructed,epoch,i) #save image image        
        
        #####################################################################
        if i % sav_rate == 0 :  #### we save our model weights every val_rate batches
          saving_model(model,epoch+1,i)

  saving_model(model,'final',0)

  print('################ TRAINING FINISHED ################')