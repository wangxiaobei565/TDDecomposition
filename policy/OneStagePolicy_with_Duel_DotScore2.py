import torch

from model.policy.OneStagePolicy import OneStagePolicy
from model.components import DNN
from model.score_func import *

class OneStagePolicy_with_Duel_DotScore2(OneStagePolicy):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - from OneStagePolicy:
            - state_encoder_feature_dim
            - state_encoder_attn_n_head
            - state_encoder_hidden_dims
            - state_encoder_dropout_rate
        '''
        parser = OneStagePolicy.parse_model_args(parser)
        parser.add_argument('--policy_action_hidden', type=int, nargs='+', default=[128], 
                            help='hidden dim of the action net')
        parser.add_argument('--policy_scorer_hidden', type=int, default=64, 
                            help='hidden dim of the action net')
        return parser
    
    def __init__(self, args, environment):
        '''
        action_space = {'item_id': ('nominal', stats['n_item']), 
                        'item_feature': ('continuous', stats['item_vec_size'], 'normal')}
        observation_space = {'user_profile': ('continuous', stats['user_portrait_len'], 'positive']), 
                            'history': ('sequence', stats['max_seq_len'], ('continuous', stats['item_vec_size']))}
        '''
        super().__init__(args, environment)
        # action is the set of parameters of linear mapping [item_dim + 1, 1]
        self.action_dim = self.item_dim + 1
        self.action_layer = DNN(self.state_dim, args.policy_action_hidden, self.action_dim, 
                                dropout_rate = self.dropout_rate, do_batch_norm = True)
        self.scorer_norm = nn.LayerNorm(args.policy_scorer_hidden)
        
        self.V_layer = DNN(self.state_dim, args.policy_action_hidden, 1, 
                                dropout_rate = self.dropout_rate, do_batch_norm = True)
        
    def score(self, action_emb, item_emb, do_softmax = True):
        output = linear_scorer(action_emb, item_emb, self.item_dim)
        if do_softmax:
            return torch.softmax(output, dim = -1)
        else:
            return output
        
    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {'user_profile': (B, user_dim), 
                    'history_features': (B, H, item_dim, 
                    'candidate_features': (B, L, item_dim) or (1, L, item_dim)}
        @model:
        - user_profile --> user_emb (B,1,f_dim)
        - history_items --> history_item_emb (B,H,f_dim)
        - (Q:user_emb, K&V:history_item_emb) --(multi-head attn)--> user_state (B,1,f_dim)
        - user_state --> action_prob (B,n_item)
        @output:
        - out_dict: {"action_emb": (B,action_dim), 
                     "state_emb": (B,f_dim),
                     "reg": scalar,
                     "action_prob": (B,L), include probability score when candidate_features are given}
        '''
        # user embedding (B,1,f_dim)
        encoder = self.state_encoder(feed_dict)
        user_state = encoder['state_emb']
        B = user_state.shape[0]
        # action embedding (B,action_dim)
        action_emb = self.action_layer(user_state).view(B, self.action_dim)
        
        V_emb = self.V_layer(user_state).view(B, 1) 
        
        
        
        action_emb = action_emb +  0.1 * V_emb / self.action_dim
        # regularization terms
#         reg = get_regularization(self.state_encoder, self.action_layer)
        # output
        out_dict = {'action_emb': action_emb, 
                    'state_emb': user_state.view(B,-1),
                   }
#                     'reg': reg}

        
        
        try:
            out_dict['item_emb']=encoder['item_emb']
        except:
            pass
#         if 'candidate_features' in feed_dict:
#             # action prob (B,L)
#             action_prob = self.score(action_emb, feed_dict['candidate_features'], feed_dict['do_softmax'])
#             out_dict['action_prob'] = action_prob
#             out_dict['candidate_ids'] = feed_dict['candidate_ids']
        return out_dict