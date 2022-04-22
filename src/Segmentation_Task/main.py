import os
import sys
import pathlib
import torch
from torch.utils.data import DataLoader
sys.path.append('../src/')
import matplotlib.pyplot as plt
%matplotlib inline
import transforms
import Dataset
import losses
import metrics
import models


class Model:
    """
    Class for model training:
    
    - Model: UNet 3+ architecture with full scale inter and intra-skip connections with ground-truth supervision defined (Huang H. et al, 2020) with 
    3D normalised squeeze and excitation blocks (Iantsen A. et al, 2021)
    - Loss: Log Cosh Dice Loss and Focal Loss 
   
    - Hyperparameters: 
    num_epochs : 500
    scheduler : CosineAnnealingWarmRestarts (T_0=45, eta_min=1e-6)
    optmizer : Adam with betas=(0.9, 0.99)
    
    
    """

    def __init__(self, model, dataloaders, loss, optimizer,metric=None, scheduler=None, num_epochs=500, cuda_device="cuda:0", scheduler_step_per_epoch=True):

        self.model = model
        self.dataloaders = dataloaders
        self.loss = loss
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
        self.scheduler_step_per_epoch = scheduler_step_per_epoch

        def train(self):

          self.model = self.model.to(self.device)
          
          for epoch in range(self.num_epochs):
              for curr_set in ['train', 'val']:
                if curr_set == 'train':
                    self.model.train()
                elif curr_set == 'val' or curr_set == 'test':
                    global best_metric
                    self.model.eval()

                loss = 0
                curr_metric = 0

                # for each epoch 
                with torch.set_grad_enabled(curr_set == 'train'):
                      batch = 0
                      for sample in self.dataloaders[curr_set]:
                          input, target = sample['input'], sample['target']
                          input, target = input.to(self.device), target.to(self.device)


                          output = self.model(input)

                          l = self.loss(output, target)
                          metric = self.metric(output.detach(), target.detach())

                          # Losses and metric:
                          loss += l.item()
                          curr_metric += metric.item()


                          if curr_set == 'train':
                              # Backward pass:
                              loss.backward()
                              self.optimizer.step()

                              # zero the parameter gradients:
                              self.optimizer.zero_grad()

                              if self.scheduler and not self.scheduler_step_per_epoch:
                                  self.scheduler.step()

                          del loss
                          batch += 1

               # Save checkpoint based on validation 
               c_metric = curr_metric/(batch+1)
               if curr_set == 'val':
                    if c_metric > best_metric:
                        print('Model Saving')
                        state = {'model': model.state_dict(),
                                 'metric': c_metric,
                                 'epoch': epoch}
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        else:
                            file_list = glob.glob(save_path + '/ckpt*.pt', recursive=True)
                            for file in file_list:
                                os.remove(file)
                        torch.save(state, save_path + '/ckpt' + str(epoch) + '.pt')
                        best_metric = c_metric



              # Update the learning rate:
              if self.scheduler and self.scheduler_step_per_epoch:
                  if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                      self.scheduler.step(np.mean(curr_metric))
                  else:
                      self.scheduler.step()




# Initial Parameters
print('Initial Parameters')
num_workers = 8 
batch_size = 24
lr = 1e-4 
n_epochs = 500 
best_metric = 0 

# Transforms for data augmentation 
train_transforms = transforms.Compose([
    transforms.RandomRotation(angle=[0, 45]),
    transforms.Flipping(),
    transforms.RandomGaussianNoise(),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.ToTensor()
])

# Dataloader set up:
train_set = Dataset(train_paths, transforms=train_transforms)
val_set = Dataset(val_paths, transforms=val_transforms)

# Dataloaders:
train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)

dataloaders = {
    'train': train_loader,
    'val': val_loader}


model = models.NormResSEUNet_3Plus()
loss = losses.Dice_and_FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
metric = metrics.dice_hausdorff_distance
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, eta_min=1e-6)
  
Model_train = Model(model=model,
              dataloaders=dataloaders,
              loss=loss,
              optimizer=optimizer,
              metric=metric,
              scheduler=scheduler,
              num_epochs=n_epochs)
              
Model_train.train()

print('End of Training')
