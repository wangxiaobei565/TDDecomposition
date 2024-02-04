import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agents.BaseRLAgent import BaseRLAgent
    
class SAC(BaseRLAgent):
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
        parser.add_argument('--critic_lr', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--alpha_lr', type=float, default=1e-4, 
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
        self.critic_lr = args.critic_lr
        self.actor_decay = args.actor_decay
        self.critic_decay = args.critic_decay
        
        self.actor = facade.actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, 
                                                weight_decay=args.actor_decay)

        self.critic1 = facade.critic1
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=args.critic_lr, 
                                                 weight_decay=args.critic_decay)
        
        self.critic2 = facade.critic2
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=args.critic_lr, 
                                                 weight_decay=args.critic_decay)
        
        
        self.tau = args.target_mitigate_coef
        self.advantage_bias = args.advantage_bias
        self.entropy_coef = args.entropy_coef
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
        self.training_history['entropy_loss'] = []
        self.training_history['advantage'] = []
        
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
        
        critic_loss, actor_loss, entropy_loss, advantage = self.get_sac_loss(observation, policy_output, reward, done_mask, next_observation)
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        self.training_history['entropy_loss'].append(entropy_loss.item())
        self.training_history['advantage'].append(advantage.item())

        # Update the frozen target models
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1], 
                              self.training_history['critic_loss'][-1], 
                              self.training_history['entropy_loss'][-1], 
                              self.training_history['advantage'][-1])}
    
    def get_sac_loss(self, observation, policy_output, reward, done_mask, next_observation, 
                      do_actor_update = True, do_critic_update = True):
        current_critic_output1 = self.facade.apply_critic(observation, 
                                                         utils.wrap_batch(policy_output, device = self.device), 
                                                         self.critic1)
        current_critic_output2 = self.facade.apply_critic(observation, 
                                                         utils.wrap_batch(policy_output, device = self.device), 
                                                         self.critic2)
        current_Q1, current_Q2 = current_critic_output1['q'], current_critic_output2['q']
        

        next_policy_output = self.facade.apply_policy(next_observation, self.actor_target)
        next_critic_output1 = self.facade.apply_critic(next_observation,next_policy_output,self.critic1)                                                         
        next_critic_output2 = self.facade.apply_critic(next_observation,next_policy_output,self.critic2)
        
        next_Q1, next_Q2 = next_critic_output1['q'], next_critic_output2['q']
        
#         next_probs = next_policy_output['action_prob'].prod(dim=1)
        next_probs = next_policy_output['action_prob']
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True).squeeze()
        
        import pdb
        pdb.set_trace()
        Q_S = reward + self.gamma * ((~done_mask * torch.min(next_Q1,next_Q2)) + self.log_alpha.exp() * entropy )
        
        value1_loss = F.mse_loss(current_Q1, Q_S.detach()).mean()
        value2_loss = F.mse_loss(current_Q2, Q_S.detach()).mean()
        if do_critic_update and self.critic_lr > 0:
            # Optimize the critic
            self.critic1_optimizer.zero_grad()
            value1_loss.backward()
            self.critic1_optimizer.step()
            
            self.critic2_optimizer.zero_grad()
            value2_loss.backward()
            self.critic2_optimizer.step()
        
        current_policy_output = self.facade.apply_policy(observation, self.actor)
        probs = next_policy_output['action_prob']
        log_probs = torch.log(next_probs + 1e-8)
        curr_entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True).squeeze()
        A = policy_output['action']
#         logp = -torch.log(current_policy_output['action_prob'] + 1e-6) # (B,K)
        logp = -torch.log(torch.gather(current_policy_output['candidate_prob'],1,A-1) + 1e-6) # (B,K)
        # use log(1-p), p is close to zero when there are large number of items
#         logp = torch.log(-torch.gather(current_policy_output['candidate_prob'],1,A-1)+1) # (B,K)
        actor_loss = torch.mean(torch.sum(logp * (advantage.view(-1,1) + self.advantage_bias), dim = 1))
        entropy_loss = torch.sum(current_policy_output['candidate_prob'] \
                                  * torch.log(current_policy_output['candidate_prob']), dim = 1).mean()
        

        if do_actor_update and self.actor_lr > 0:
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            (actor_loss + self.entropy_coef * entropy_loss).backward()
            self.actor_optimizer.step()
#         min_qvalue = torch.sum(next_probs * torch.min(next_Q1, next_Q2),
#                                dim=1,
#                                keepdim=True)
#         next_value = min_qvalue + self.log_alpha.exp() * entropy
#         td_target = rewards + self.gamma * next_value * (1 - dones)
        
        
#         Q_S = reward + self.gamma * (~done_mask * torch.min(Q1_S,Q2_S))
    
        
#         advantage = torch.clamp((Q_S - torch.min(current_Q1,current_Q2)).detach(), -1, 1) # (B,)

#         # Compute critic loss
#         value1_loss = F.mse_loss(current_Q1, Q_S.detach()).mean()
#         value2_loss = F.mse_loss(current_Q2, Q_S.detach()).mean()
        
#         # Regularization loss
# #         critic_reg = current_critic_output['reg']

#         if do_critic_update and self.critic_lr > 0:
#             # Optimize the critic
#             self.critic1_optimizer.zero_grad()
#             value1_loss.backward()
#             self.critic1_optimizer.step()
            
#             self.critic2_optimizer.zero_grad()
#             value2_loss.backward()
#             self.critic2_optimizer.step()
        
        
#         # Compute actor loss
#         current_policy_output = self.facade.apply_policy(observation, self.actor)
#         A = policy_output['action']
# #         logp = -torch.log(current_policy_output['action_prob'] + 1e-6) # (B,K)
#         logp = -torch.log(torch.gather(current_policy_output['candidate_prob'],1,A-1) + 1e-6) # (B,K)
#         # use log(1-p), p is close to zero when there are large number of items
# #         logp = torch.log(-torch.gather(current_policy_output['candidate_prob'],1,A-1)+1) # (B,K)
#         actor_loss = torch.mean(torch.sum(logp * (advantage.view(-1,1) + self.advantage_bias), dim = 1))
#         entropy_loss = torch.sum(current_policy_output['candidate_prob'] \
#                                   * torch.log(current_policy_output['candidate_prob']), dim = 1).mean()
        

#         if do_actor_update and self.actor_lr > 0:
#             # Optimize the actor 
#             self.actor_optimizer.zero_grad()
#             (actor_loss + self.entropy_coef * entropy_loss).backward()
#             self.actor_optimizer.step()
            
            
       
        return value1_loss,value2_loss, actor_loss, torch.mean(advantage)


    def save(self):
        torch.save(self.critic1.state_dict(), self.save_path + "_critic1")
        torch.save(self.critic1_optimizer.state_dict(), self.save_path + "_critic1_optimizer")
        
        torch.save(self.critic2.state_dict(), self.save_path + "_critic2")
        torch.save(self.critic2_optimizer.state_dict(), self.save_path + "_critic2_optimizer")

        torch.save(self.actor.state_dict(), self.save_path + "_actor")
        torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")
        
       

    def load(self):
        self.critic1.load_state_dict(torch.load(self.save_path + "_critic1", map_location=self.device))
        self.critic1_optimizer.load_state_dict(torch.load(self.save_path + "_critic1_optimizer", map_location=self.device))
        self.critic1_target = copy.deepcopy(self.critic1)
        
        self.critic2.load_state_dict(torch.load(self.save_path + "_critic2", map_location=self.device))
        self.critic2_optimizer.load_state_dict(torch.load(self.save_path + "_critic2_optimizer", map_location=self.device))
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
