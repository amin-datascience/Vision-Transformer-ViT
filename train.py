import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

import warmup_scheduler
import numpy as np


def train_func(data, model, optimizer, loss_func, max_epochs = 50, validation_loader = None, 
               batch_size = 128, scheduler = None, device = None):


    n_batches_train = len(train_loader)
    n_batches_val = len(validation_loader)
    n_samples_train = batch_size * n_batches_train
    n_samples_val = batch_size * n_batches_val


    losses = []
    accuracy = []
    validation_loss = []
    validation_accuracy = []
    

    for epoch in range(max_epochs):
        running_loss, correct = 0, 0
        for images, labels in train_loader:
            if device:
                images = images.to(device)
                labels = labels.to(device)

            model.train()
            outputs = model(images)[0]
            loss = loss_func(outputs, labels)
            predictions = outputs.argmax(1)
            correct += int(sum(predictions == labels))
            running_loss += loss.item()


            #BACKWARD AND OPTIMZIE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        loss_epoch = running_loss / n_batches_train
        accuracy_epoch = correct / n_samples_train

                 
        losses.append(loss_epoch)
        accuracy.append(accuracy_epoch)
        
        print('Epoch [{}/{}], Training Accuracy [{:.4f}], Training Loss: {:.4f},'
             .format(epoch + 1, max_epochs, accuracy_epoch, loss_epoch), end = '  ')
        print('Correct/ Total: [{}/{}]'.format(correct, n_samples_train), end = '   ')
        
        if validation_loader:
            model.eval()     
                       
            val_loss, val_corr = 0, 0
            for val_images, val_labels in validation_loader:
                if device:
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)

                outputs = model(val_images)[0]
                loss = loss_func(outputs, val_labels)
                _, predictions = outputs.max(1)
                val_corr += int(sum(predictions == val_labels))
                val_loss += loss.item()


            loss_val = val_loss / n_batches_val
            accuracy_val = val_corr / n_samples_val

            validation_loss.append(loss_val)
            validation_accuracy.append(accuracy_val)


            print('Validation accuracy [{:.4f}], Validation Loss: {:.4f}'
                 .format(accuracy_val, loss_val))


    model_save_name = 'vit.pt'
    path = F'./{model_save_name}'
    torch.save(model.state_dict(), path)

    

    return {'loss': losses, 'accuracy': accuracy, 
            'val_loss': validation_loss, 'val_accuracy': validation_accuracy}


train_transform = transforms.Compose([
    transforms.TrivialAugmentWide(interpolation = transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding = 4),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.RandomErasing(p=0.1)])



#Running the code
path = './'
cifar10 = datasets.CIFAR10(path, train = True, download = True, 
                                       transform = train_transform)

cifar10_test = datasets.CIFAR10(path, train = False, download = True,
                                            transform = transforms.Compose([transforms.ToTensor()]))


validation, test = torch.utils.data.random_split(cifar10_test, [2000, 8000])
concat = torch.utils.data.ConcatDataset([cifar10, test])

train_loader = torch.utils.data.DataLoader(concat, batch_size = 128, shuffle = True, drop_last = True, num_workers = 2)
val_loader = torch.utils.data.DataLoader(validation, batch_size = 128, shuffle = True, drop_last = True, num_workers = 2) 


#Defining the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 4e-4)


base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min = 1e-4)
scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=5, after_scheduler = base_scheduler)

history = train_func(train_loader, model, optimizer, loss_func = criterion, validation_loader = val_loader, 
                     device = device, scheduler = scheduler, batch_size = 128, max_epochs = 100)


