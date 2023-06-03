import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TTF
import matplotlib.pyplot as plt

class SquarePad:
    # https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/5
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return TTF.pad(image, padding, 0, 'constant')


class Trainer:
    def __init__(self, data_path, image_size=128, batch_size=64) -> None:
        if torch.cuda.is_available():
            print("The code will run on GPU.")
        else:
            print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transform=transforms.Compose([
                                SquarePad(),
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

        self.trainset = datasets.ImageFolder(root=f"{data_path}/train/", transform=transform)
        self.train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=1)
        self.testset = datasets.ImageFolder(root=f"{data_path}/test/", transform=transform)
        self.test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=1)
    
        self.loss_fn = nn.BCELoss()  # binary cross entropy

    def loss_fun(self, output, target):
        return self.loss_fn(output.to(torch.float), target.to(torch.float))
    
    def train(self, model, optimizer, num_epochs=10):
        self.out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': []}
        
        for epoch in tqdm(range(num_epochs), unit='epoch'):
            model.train()
            #For each epoch
            train_correct = 0
            train_loss = []
            for minibatch_no, (data, target) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                data, target = data.to(self.device), target.to(self.device)
                
                #Zero the gradients computed for each weight
                optimizer.zero_grad()
                #Forward pass your image through the network
                output = model(data)
                loss = self.loss_fun(output, target)
                #Backward pass through the network
                loss.backward()
                #Update the weights
                optimizer.step()

                train_loss.append(loss.item())
                #Compute how many were correctly classified
                predicted = (output > 0.5).to(torch.int)
                train_correct += (target==predicted).sum().cpu().item()
            #Comput the test accuracy
            test_loss = []
            test_correct = 0
            model.eval()
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                with torch.no_grad():
                    output = model(data)
                test_loss.append(self.loss_fun(output, target).cpu().item())
                predicted = (output > 0.5).to(torch.int)
                test_correct += (target==predicted).sum().cpu().item()
            self.out_dict['train_acc'].append(train_correct/len(self.trainset))
            self.out_dict['test_acc'].append(test_correct/len(self.testset))
            self.out_dict['train_loss'].append(np.mean(train_loss))
            self.out_dict['test_loss'].append(np.mean(test_loss))
            print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
                f"Accuracy train: {self.out_dict['train_acc'][-1]*100:.1f}%\t test: {self.out_dict['test_acc'][-1]*100:.1f}%")
        return self.out_dict
    
    def plot_acc_loss(self):
        plt.plot(range(len(self.out_dict["test_acc"])), self.out_dict["test_acc"])
        plt.plot(range(len(self.out_dict["train_acc"])), self.out_dict["train_acc"])
        plt.legend(('Test error','Train eror'))
        plt.xlabel('Epoch number')
        plt.ylabel('Accuracy')
        plt.show()


        plt.plot(range(len(self.out_dict["test_loss"])), self.out_dict["test_loss"])
        plt.plot(range(len(self.out_dict["train_loss"])), self.out_dict["train_loss"])
        plt.legend(('Test error','Train eror'))
        plt.xlabel('Epoch number')
        plt.ylabel('Loss')
        plt.show()