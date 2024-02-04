import torch.nn.functional as F
import torch.nn as nn
import torch

from model.components import DNN
from utils import get_regularization

class GeneralCritic_V(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_hidden_dims
        - critic_dropout_rate
        '''
        parser.add_argument('--critic_hidden_dims_V', type=int, nargs='+', default=[128], 
                            help='specificy a list of k for top-k performance')
        parser.add_argument('--critic_dropout_rate_V', type=float, default=0.2, 
                            help='dropout rate in deep layers')
        return parser
    
    def __init__(self, args, environment, policy):
        super().__init__()
        self.state_dim = policy.state_dim
        self.action_dim = policy.action_dim
#         self.state_encoder = policy.state_encoder
        self.net = DNN(self.state_dim, args.critic_hidden_dims_V, 1, 
                       dropout_rate = args.critic_dropout_rate_V, do_batch_norm = True)
        
    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {'state_emb': (B, state_dim), 'action_emb': (B, action_dim)}
        '''
        state_emb = feed_dict['state_emb']
#         state_emb = self.state_encoder(feed_dict)['state_emb'].view(-1, self.state_dim)
        V = self.net(state_emb).view(-1)
        
#         reg = get_regularization(self.net, self.state_encoder)
        reg = get_regularization(self.net)
        return {'v': V, 'reg': reg}
