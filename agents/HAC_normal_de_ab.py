import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agents.BaseRLAgent import BaseRLAgent
    
class HAC_normal_de_ab(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - episode_batch_size
        - batch_size
        - actor_lr
        - critic_lr
        - actor_decay
        - critic_decay
        - target_mitigate_coef
        - args from BaseRLAgent:
            - gamma
            - n_iter
            - train_every_n_step
            - initial_greedy_epsilon
            - final_greedy_epsilon
            - elbow_greedy
            - check_episode
            - with_eval
            - save_path
        '''
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument('--episode_batch_size', type=int, default=8, 
                            help='episode sample batch size')
        parser.add_argument('--batch_size', type=int, default=32, 
                            help='training batch size')
        parser.add_argument('--actor_lr', type=float, default=1e-4, 
                            help='learning rate for actor')
        parser.add_argument('--critic1_lr', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--critic2_lr', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--actor_decay', type=float, default=1e-4, 
                            help='learning rate for actor')
        parser.add_argument('--critic_decay', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01, 
                            help='mitigation factor')
        parser.add_argument('--advantage_bias', type=float, default=0, 
                            help='mitigation factor')
        parser.add_argument('--entropy_coef', type=float, default=0.1, 
                            help='mitigation factor')
        parser.add_argument('--lambda_', type=float, default=0.1, 
                            help='lambda_')
        return parser
    
    
    def __init__(self, args, facade):
        '''
        self.gamma
        self.n_iter
        self.check_episode
        self.with_eval
        self.save_path
        self.facade
        self.exploration_scheduler
        '''
        super().__init__(args, facade)
        self.episode_batch_size = args.episode_batch_size
        self.batch_size = args.batch_size
        
        self.actor_lr = args.actor_lr
        self.critic1_lr = args.critic1_lr
        self.critic2_lr = args.critic2_lr
        self.actor_decay = args.actor_decay
        self.critic_decay = args.critic_decay
        
        self.actor = facade.actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, 
                                                weight_decay=args.actor_decay)

        self.critic1 = facade.critic1
        self.critic1_target = copy.deepcopy(self.critic1)
        self.V_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=args.critic1_lr, 
                                                 weight_decay=args.critic_decay)
        self.critic2 = facade.critic2
        self.critic2_target = copy.deepcopy(self.critic2)
        self.Q_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=args.critic2_lr, 
                                                 weight_decay=args.critic_decay)

        self.tau = args.target_mitigate_coef
        self.advantage_bias = args.advantage_bias
        self.entropy_coef = args.entropy_coef
        
        self.lambda_ = args.lambda_
        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f"{args}\n")
        
        
#     def action_after_train(self):
#         self.facade.stop_env()
        
#     def get_report(self):
#         episode_report = self.facade.get_episode_report(10)
#         train_report = {k: np.mean(v[-10:]) for k,v in self.training_history.items()}
#         return episode_report, train_report
        
    def action_before_train(self):
        super().action_before_train()
        self.training_history = {"actor_loss": []}
        self.training_history['critic1_loss'] = []
        self.training_history['critic2_loss'] = []
        self.training_history['entropy_loss'] = []
        self.training_history['advantage'] = []
        self.training_history['beta'] = []
        self.training_history['p_diff'] = []
        
    def run_episode_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        with torch.no_grad():
            # sample action
            policy_output = self.facade.apply_policy(observation, self.actor, epsilon, 
                                                     do_explore = True, do_softmax = True)
            # apply action on environment and update replay buffer
            next_observation, reward, done, info = self.facade.env_step(policy_output)
            # update replay buffer
            if do_buffer_update:
                self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
        return next_observation
            

    def step_train(self):
        observation, policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.batch_size)
#         reward = torch.FloatTensor(reward)
#         done_mask = torch.FloatTensor(done_mask)
        
        critic_loss_one, critic_loss, actor_loss, entropy_loss, advantage, debias_beta = self.get_a2c_loss(observation, policy_output, reward, done_mask, next_observation)
        self.training_history['critic1_loss'].append(critic_loss_one.item())
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic2_loss'].append(critic_loss.item())
        self.training_history['entropy_loss'].append(entropy_loss.item())
        self.training_history['advantage'].append(advantage.item())
        self.training_history['beta'].append(debias_beta['beta'].item())
        self.training_history['p_diff'].append(debias_beta['p_diff'].item())

        # Update the frozen target models
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1],
                              self.training_history['critic1_loss'][-1],
                              self.training_history['critic2_loss'][-1], 
                              self.training_history['entropy_loss'][-1], 
                              self.training_history['advantage'][-1], 
                              self.training_history['beta'][-1], 
                              self.training_history['p_diff'][-1])}
    
    def get_a2c_loss(self, observation, policy_output, reward, done_mask, next_observation, 
                      do_actor_update = True, do_critic_update = True):
        
        # Get current Q estimate
        current_policy_output = self.facade.apply_policy(observation, self.actor)
        S = current_policy_output['state_emb'].detach()
        L = current_policy_output['list_emb'].detach()
        
        # (B,)
        observed_likelihood = policy_output['hac_pro']
        # (B, action_dim)
        observed_action = policy_output['action_emb']
        
        # (B, action_dim)
        curr_mu = current_policy_output['action_emb']
        curr_std = self.facade.noise_var
        
        
        curr_likelihood = torch.exp(-((observed_action - curr_mu) ** 2) / (2 * (curr_std** 2))) / (torch.sqrt(2 * torch.tensor(torch.pi) * (curr_std** 2))) + 1e-20
        # (B,)
        curr_likelihood = curr_likelihood.mean(dim=1)
        
        
        debias_beta = (self.lambda_ + curr_likelihood) / (self.lambda_ + observed_likelihood) 
        
        
        
        current_V = self.critic1({'state_emb': S})['v']
#         Q_S = self.critic({'state_emb': S, 'list_emb': L})['q']
        with torch.no_grad():
            current_Q_de = self.critic2({'state_emb': S, 'list_emb': L})['q'].detach()
#         critic_loss_one = (F.mse_loss(current_Q, current_V)).mean()
        
        critic_loss_one = (debias_beta * F.mse_loss(current_Q_de, current_V)).mean()
        
        
        if do_critic_update and self.critic1_lr > 0:
            # Optimize the critic
            self.V_optimizer.zero_grad()
            critic_loss_one.backward()
            self.V_optimizer.step()
            
    
        # Compute the target Q value
        current_Q = self.critic2({'state_emb': S, 'list_emb': L})['q']
        next_policy_output = self.facade.apply_policy(next_observation, self.actor_target)
#         next_policy_output = self.facade.apply_p[olicy(next_observation, self.actor)
        S_prime = next_policy_output['state_emb']
        L_prime = next_policy_output['list_emb']
        
        next_V = self.critic1_target({'state_emb': S_prime})['v'].detach()
#         V_S_prime = self.critic({'state_emb': S_prime})['v'].detach()
        Q_S = reward + self.gamma * (~done_mask * next_V)
    
        advantage_QS = reward + self.gamma * (~done_mask * next_V)
        advantage = torch.clamp((advantage_QS - current_Q).detach(), -1, 1) # (B,)

        # Compute critic loss
        value_loss = F.mse_loss(current_Q, Q_S).mean()
        
        # Regularization loss
#         critic_reg = current_critic_output['reg']
        
        if do_critic_update and self.critic2_lr > 0:
            # Optimize the critic
            self.Q_optimizer.zero_grad()
            value_loss.backward()
            
            self.Q_optimizer.step()
        
#         import pdb
#         pdb.set_trace()
        # Compute actor loss
        current_policy_output = self.facade.apply_policy(observation, self.actor)
        A = policy_output['action']
#         logp = -torch.log(current_policy_output['action_prob'] + 1e-6) # (B,K)
        logp = -torch.log(torch.gather(current_policy_output['candidate_prob'],1,A-1) + 1e-6) # (B,K)
        # use log(1-p), p is close to zero when there are large number of items
#         logp = torch.log(-torch.gather(current_policy_output['candidate_prob'],1,A-1)+1) # (B,K)
        actor_loss = torch.mean(torch.sum(logp * (advantage.view(-1,1) + self.advantage_bias), dim = 1))
        entropy_loss = torch.sum(current_policy_output['candidate_prob'] \
                                  * torch.log(current_policy_output['candidate_prob']), dim = 1).mean()
        
        # Regularization loss
#         actor_reg = policy_output['reg']

        if do_actor_update and self.actor_lr > 0:
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            (actor_loss + self.entropy_coef * entropy_loss).backward()
            self.actor_optimizer.step()
            
        return critic_loss_one, value_loss, actor_loss, entropy_loss, torch.mean(advantage), {'beta': torch.mean(debias_beta), 'p_diff': torch.mean((observed_likelihood - curr_likelihood)**2)}


    def save(self):
        torch.save(self.critic1.state_dict(), self.save_path + "_critic1")
        torch.save(self.Q_optimizer.state_dict(), self.save_path + "_V_optimizer")
        
        torch.save(self.critic2.state_dict(), self.save_path + "_critic2")
        torch.save(self.V_optimizer.state_dict(), self.save_path + "_Q_optimizer")

        torch.save(self.actor.state_dict(), self.save_path + "_actor")
        torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")


    def load(self):
        self.critic1.load_state_dict(torch.load(self.save_path + "_critic1", map_location=self.device))
        self.Q_optimizer.load_state_dict(torch.load(self.save_path + "_Q_optimizer", map_location=self.device))
        self.critic1_target = copy.deepcopy(self.critic1)
        
        self.critic2.load_state_dict(torch.load(self.save_path + "_critic2", map_location=self.device))
        self.V_optimizer.load_state_dict(torch.load(self.save_path + "_V_optimizer", map_location=self.device))
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
