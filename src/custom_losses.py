import torch 
import numpy as np 


class WeightedBCE(torch.nn.Module):
    
    def __init__(self, w_p = None, w_n = None):
        super(WeightedBCE, self).__init__()
        
        self.w_p = w_p
        self.w_n = w_n
        
    def forward(self, logits, labels, epsilon = 1e-7):
        
        ps = torch.sigmoid(logits)
        
        loss_pos = -1 * torch.mean(self.w_p * labels * torch.log(ps + epsilon))
        loss_neg = -1 * torch.mean(self.w_n * (1-labels) * torch.log((1-ps) + epsilon))
        
        loss = loss_pos + loss_neg
        
        return loss

def get_pos_neg_weights(labels):
    
    labels = np.array(labels)
    
    N = labels.shape[0]
    
    positive_frequencies = np.sum(labels,axis = 0) / N
    negative_frequencies = 1 - positive_frequencies

    pos_weights = negative_frequencies
    neg_weights = positive_frequencies
    
    return pos_weights, neg_weights

