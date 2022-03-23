import torch
import numpy as np
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class combined_networks(nn.Module):
    def __init__(self):
        super(combined_networks, self).__init__()
        self.conv1 = nn.Conv1d(1,30,55,stride=1,padding='same')
        torch.nn.init.xavier_normal_(self.conv1.weight)
        #self.conv1.weight.data.fill_(1/55)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=0.01, mode='fan_in',nonlinearity='leaky_relu')
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.drop1 = nn.Dropout(p=0.2)
        
        self.conv2 = nn.Conv1d(30,15,160,stride=1,padding='same')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=0.01, mode='fan_in',nonlinearity='leaky_relu')
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.drop2 = nn.Dropout(p=0.2)
        
        self.conv3 = nn.Conv1d(15,7,160,stride=1,padding='same')
        torch.nn.init.kaiming_uniform_(self.conv3.weight, a=0.01, mode='fan_in',nonlinearity='leaky_relu')
        self.relu3 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.drop3 = nn.Dropout(p=0.2)
        
        self.conv4 = nn.Conv1d(7,2,160,stride=1,padding='same')
        #torch.nn.init.kaiming_uniform_(self.conv4.weight, a=0.01, mode='fan_in',nonlinearity='leaky_relu')
        self.conv4.weight.data.fill_(0.0001)
        #torch.nn.init.xavier_normal_(self.conv4.weight)
        self.relu4 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.drop4 = nn.Dropout(p=0.2)
        
        self.FB = nn.Conv1d(2,2,160,stride=80, padding = 'valid')
        torch.nn.init.xavier_normal_(self.FB.weight)
        #self.FB.weight.data.fill_(1/160)
        self.sigmoid1 = nn.Sigmoid()
        self.drop5 = nn.Dropout(p=0.0)
        
        self.DB1 = nn.Conv1d(2,2,2,stride=1,padding='same', groups=2)
        torch.nn.init.xavier_normal_(self.DB1.weight)
        #self.DB1.weight.data.fill_(0.0)
        self.sigmoid2 = nn.Sigmoid()
        self.drop6 = nn.Dropout(p=0.0)
        
        self.DB2 = nn.Conv1d(2,2,2,stride=1,padding='same', groups=2)
        torch.nn.init.xavier_normal_(self.DB2.weight)
        #self.DB2.weight.data.fill_(0.0)
        self.sigmoid3 = nn.Sigmoid()
        self.drop7 = nn.Dropout(p=0.0)
        
        self.DB3 = nn.Conv1d(2,2,2,stride=1,padding='same', groups=2)
        torch.nn.init.xavier_normal_(self.DB3.weight)
        #self.DB3.weight.data.fill_(0.0)
        self.sigmoid4 = nn.Sigmoid()
        self.drop8 = nn.Dropout(p=0.0)
        
        self.AN1 = nn.Conv1d(2,2,55,stride=1,padding='same', groups=1)
        torch.nn.init.xavier_normal_(self.AN1.weight)
        #self.DB1.weight.data.fill_(0.0)
        self.sigmoidAN1 = nn.Sigmoid()
        self.drop6 = nn.Dropout(p=0.0)
        
        self.AN2 = nn.Conv1d(2,2,15,stride=1,padding='same', groups=1)
        torch.nn.init.xavier_normal_(self.AN2.weight)
        #self.DB2.weight.data.fill_(0.0)
        self.sigmoidAN2 = nn.Sigmoid()
        self.drop7 = nn.Dropout(p=0.0)
        
        self.AN3 = nn.Conv1d(2,5,5,stride=1,padding='same', groups=1)
        torch.nn.init.xavier_normal_(self.AN3.weight)
        #self.DB3.weight.data.fill_(0.0)
        self.sigmoidAN3 = nn.Softmax(dim=1)
        self.drop8 = nn.Dropout(p=0.0)
        
    
    def forward(self, x, training = 0):
        x = x.to(device)
        x = x/2**15

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.drop4(x)
        
        x=torch.nn.functional.normalize(x,dim=2)
       # x=x*10
        x = self.FB(x)
        x = self.sigmoid1(x)
        # x = self.drop5(x)
        # # x_1 = x.clone()
        
        DB = x
        # DB = self.DB1(x)
        # DB = self.sigmoid2(DB)
        # DB = self.drop6(DB)

        # DB = self.DB2(DB)
        # DB = self.sigmoid3(DB)
        # DB = self.drop7(DB)

        # DB = self.DB3(DB)
        # DB = self.sigmoid4(DB)
        
        AN = self.AN1(x)
        AN = self.sigmoidAN1(AN)
        AN = self.drop6(AN)

        AN = self.AN2(AN)
        AN = self.sigmoidAN2(AN)
        AN = self.drop7(AN)

        AN = self.AN3(AN)
        AN = self.sigmoidAN3(AN)
        #x = self.drop8(x)
        #x = x + 1e-4
        #x = x[:,:,0:sy]
        return DB, AN
        # if training:
        #     x_2 = x.clone()
        #     AN = self.AN1(x_2[:,:,0:270])
        #     AN = self.sigmoidAN1(AN)
        #     AN = self.drop6(AN)
    
        #     AN = self.AN2(AN)
        #     AN = self.sigmoidAN2(AN)
        #     AN = self.drop7(AN)
    
        #     AN = self.AN3(AN)
        #     AN = self.sigmoidAN3(AN)
        #     #x = self.drop8(x)
        #     #x = x + 1e-4
        #     #x = x[:,:,0:sy]
        #     return DB, AN
        # else:
        #     x_2 = torch.zeros(1,2,270)
        #     x_2 = x_2.to(device)
        #     AN = self.AN1(x_2[:,:,0:270])
        #     AN = self.sigmoidAN1(AN)
        #     AN = self.drop6(AN)
    
        #     AN = self.AN2(AN)
        #     AN = self.sigmoidAN2(AN)
        #     AN = self.drop7(AN)
    
        #     AN = self.AN3(AN)
        #     AN = self.sigmoidAN3(AN)
        #     #x = self.drop8(x)
        #     #x = x + 1e-4
        #     #x = x[:,:,0:sy]
        #     return DB, AN

