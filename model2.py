from torch import nn
import config
import pdb
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.nn import LSTM
import torch
import torch.nn.functional as F
import pdb


class TUMORCLASSIFIER(nn.Module):
    
    
    def __init__(self):
        super(TUMORCLASSIFIER, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(64)
            #nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=1),
            nn.ReLU(),
            #nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, padding=1),
            nn.ReLU(),
            #nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(256),
            #nn.MaxPool2d(2)
        )
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(128),
            #nn.MaxPool2d(2)
        )
        
        
        self.fc1 = nn.Linear(15488, 1024)
        self.lstm1 = LSTM(11*11, 128)
        self.lstm2 = LSTM(128, 128)
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, config.NUMBER_OF_CLASSES)
        
    
    
    
    def forward(self, x):
        size = x.shape[0]
        x = x.unsqueeze(1)
        #print(x.shape)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        pdb.set_trace()
        #print(out.shape)
        lstm_input = out.reshape(size, 128, -1)
        
        lstm1 = self.lstm1(lstm_input)[0]
        lstm2 = self.lstm2(lstm1)[0]
        lstm2 = lstm2.reshape(size, -1)
        
        
        
        
        
        #pdb.set_trace()
        out = out.view(size, -1)
        conc = torch.cat((lstm2, out), 1)
        out = self.fc1(out)
        
        out = self.fc2(out)
        out = self.fc3(out)
        #print(out.shape)
        
        return out
    
    
        
class TUMORCLASSIFIER1(nn.Module):
    
    
    def __init__(self):
        super(TUMORCLASSIFIER, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(64)
            #nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=1),
            nn.Sigmoid(),
            #nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, padding=1),
            nn.Sigmoid(),
            #nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            #nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(256),
            #nn.MaxPool2d(2)
        )
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(128),
            #nn.MaxPool2d(2)
        )
        
        
        self.fc1 = nn.Linear(15488, 1024)
        self.lstm1 = LSTM(11*11, 128)
        self.lstm2 = LSTM(128, 128)
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, config.NUMBER_OF_CLASSES)
        
    
    
    
    def forward(self, x):
        size = x.shape[0]
        x = x.unsqueeze(1)
        #print(x.shape)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        #print(out.shape)
        lstm_input = out.reshape(size, 128, -1)
        
        lstm1 = self.lstm1(lstm_input)[0]
        lstm2 = self.lstm2(lstm1)[0]
        lstm2 = lstm2.reshape(size, -1)
        
        
        
        
        
        #pdb.set_trace()
        out = out.view(size, -1)
        conc = torch.cat((lstm2, out), 1)
        out = self.fc1(out)
        
        out = self.fc2(out)
        out = self.fc3(out)
        #print(out.shape)
        
        return out

    
    
class Net(nn.Module):
        
        def __init__(self, num_of_classes):
            super(Net, self).__init__()
            # input image channel, output channels, kernel size square convolution
            # kernel
            # input size = 102, output size = 100
            self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            # input size = 50, output size = 48
            self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            # input size = 24, output size = 24
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
            self.vp = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
            # an affine operation: y = Wx + b
            
            self.fc1 = nn.Linear(53248, 2048)
            self.lstm1 = LSTM(288, 128)
            self.lstm2 = LSTM(128, 128)
            self.fc2 = nn.Linear(2048, 512)
            self.fc3 = nn.Linear(512, num_of_classes)
        
        def forward(self, x):
            in_size = x.size(0)
            x = x.unsqueeze(1)
            
            x = F.relu(self.bn1(self.vp(self.conv1(x))))
            x = F.relu(self.bn2(self.vp(self.conv2(x))))
            x = F.relu(self.bn3(self.vp(self.conv3(x))))
            x = self.drop2D(x)
            x = x.view(in_size, -1)
            #pdb.set_trace()
            lstm_input = x.reshape(in_size, 128, -1)
            lstm1 = self.lstm1(lstm_input)[0]
            lstm2 = self.lstm2(lstm1)[0]
            lstm2 = lstm2.reshape(in_size, -1)
            
            conc = torch.cat((lstm2, x), 1)
            
            x = self.fc1(conc)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

      
        
class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(64)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(128)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(256)
        )
        
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(512)
        )
        
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(1024)
            #nn.Dropout2d(0.5)
        )
        
        self.up6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(512)
            #nn.Dropout2d(0.5)
        )
        
        self.conv6_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(512)
        )
        
        self.conv6_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(512)
        )
        
        
        
        self.up7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(256)
            #nn.Dropout2d(0.5)
        )
        
        
        self.conv7_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(256)
        )
        
        self.conv7_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(256)
        )
        
        
        
        self.up8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(128)
            #nn.Dropout2d(0.5)
        )
        
        
        self.conv8_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(128)
        )
        
        self.conv8_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(128)
        )
        
        
        
        self.up9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(64)
            #nn.Dropout2d(0.5)
        )
        
        
        self.conv9_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(64)
        )
        
        self.conv9_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(64)
        )
        
        
        
        self.fc1 = nn.Linear(665856, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 33)
        
        
    def forward(self, x):
        
        in_size = x.size(0)
        x = x.unsqueeze(1)
        #pdb.set_trace()
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        
        
        
        up6 = self.up6(conv5)
        merge6 = torch.cat((up6, conv4), axis=1)
        del up6
        conv6 = self.conv6_1(merge6)
        conv6 = self.conv6_2(conv6)
        
        
        up7 = self.up7(conv6)
        merge7 = torch.cat((up7, conv3), axis=1)
        del up7
        conv7 = self.conv7_1(merge7)
        conv7 = self.conv7_2(conv7)
        
        
        
        
        up8 = self.up8(conv7)
        merge8 = torch.cat((up8, conv2), axis=1)
        del up8
        conv8 = self.conv8_1(merge8)
        conv8 = self.conv8_2(conv8)
        
        
        up9 = self.up9(conv8)
        merge9 = torch.cat((up9, conv1), axis=1)
        del up9
        conv9 = self.conv9_1(merge9)
        conv9 = self.conv9_2(conv9)
        conv9 = conv9.reshape(in_size, -1)
        
        #pdb.set_trace()
        
        out = self.fc1(conv9)
        
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out
        
        