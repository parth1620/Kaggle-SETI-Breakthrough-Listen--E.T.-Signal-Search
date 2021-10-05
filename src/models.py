import timm 

from config import * 

import torch 
from torch import nn 
import torch.nn.functional as F 
from torchvision import models

class SeqCNN_block(nn.Module):
    def __init__(self, output_dim=512):
        super(SeqCNN_block, self).__init__()
        
        # load the MobileNet.
        self.rgb_conv = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        self.cnn_model = timm.create_model('resnet34d', pretrained=True)
        self.cnn_model.global_pool = nn.Identity()
        self.cnn_model.fc = nn.Identity()
        
        # pool the final layer
        self.final_pooler = nn.AdaptiveAvgPool2d((None, 1))
        
        # if different output dim thant 2048, then use linear layer to convert.
        self.fc = nn.Identity()
        if output_dim != 512:
            self.fc = nn.Linear(512, output_dim)

    def _cnn_forward(self, x):
        x = self.rgb_conv(x)
        x = self.cnn_model(x)
        return x
    
    def forward(self, x):
        x = self._cnn_forward(x)
        x = self.final_pooler(x)
        x = x.squeeze(-1)
        x = x.transpose(1, 2)
        return self.fc(x)

class CNNBiGRU(nn.Module):
    def __init__(self, cnn_dim=512, rnn_dim=512, num_layers=2, dropout=0.1, mid_dimension=512):
        super(CNNBiGRU, self).__init__()
        self.hidden_dim = rnn_dim
        self.cnn_dim = cnn_dim
        self.cnn_model = SeqCNN_block(output_dim=self.cnn_dim)
        self.rnn_model = nn.GRU(input_size=self.cnn_dim,
                                hidden_size=rnn_dim,
                                num_layers=num_layers,
                                dropout=dropout,
                                batch_first=True,
                                bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features=self.hidden_dim * 4,
                             out_features=mid_dimension)
        self.fc2 = nn.Linear(in_features=mid_dimension, out_features=1)

    def forward(self, images, labels=None):

        x = images
        batch_size = x.size()[0]
        image_row = x.size()[2]
        image_col = x.size()[3]
        
        # we combine 6 images into batches
        on_1 = x[:, 0].unsqueeze(1)
        off_1 = x[:, 1].unsqueeze(1)
        on_2 = x[:, 2].unsqueeze(1)
        off_2 = x[:, 3].unsqueeze(1)
        on_3 = x[:, 4].unsqueeze(1)
        off_3 = x[:, 5].unsqueeze(1)
        
        conv_1 = self.cnn_model(on_1)

        conv_1f = self.cnn_model(off_1)

        conv_2 = self.cnn_model(on_2)

        conv_2f = self.cnn_model(off_2)

        conv_3 = self.cnn_model(on_3)

        conv_3f = self.cnn_model(off_3)

        # combine logits to a sequence
        allseq = torch.cat([conv_1, conv_1f, conv_2, conv_2f, conv_3, conv_3f], dim=1)
        dim1_shape = allseq.size()[1]
        
        seq, state = self.rnn_model(allseq)
        seqs = seq.view(batch_size, 6, -1, 2 * self.hidden_dim)
        
        on_seqs = seqs[:, (0, 2, 4), :, :].reshape(batch_size, -1, 2 * self.hidden_dim)
        off_seqs = seqs[:, (1, 3, 5), :, :].reshape(batch_size, -1, 2 * self.hidden_dim)
        
        
        on_avg = torch.mean(on_seqs, dim=1)
        off_avg = torch.mean(off_seqs, dim=1)
        
        x = torch.cat((on_avg, off_avg), dim=1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        if labels != None:
            return x, nn.BCEWithLogitsLoss()(x,labels)
        return x  