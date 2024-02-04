import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agents.BaseRLAgent import BaseRLAgent
    
class DQN_de(BaseRLAgent):
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
        parser.add_argument('--critic1_lr', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--Q_lr', type=float, default=1e-4, 
                            help='learning rate for actor')
        parser.add_argument('--Q_decay', type=float, default=1e-4, 
                            help='learning rate for actor')
        parser.add_argument('--critic_decay', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01, 
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
        self.critic1_lr = args.critic1_lr
        self.Q_lr = args.Q_lr
        
        self.Q_decay = args.Q_decay
        self.critic_decay = args.critic_decay
        
        self.Q_net = facade.Q_net
        self.Q_net_target = copy.deepcopy(self.Q_net)
        self.Q_optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=args.Q_lr, 
                                                weight_decay=args.Q_decay)

        self.critic1 = facade.critic1
        self.critic1_target = copy.deepcopy(self.critic1)
        self.V_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=args.critic1_lr, 
                                                 weight_decay=args.critic_decay)

        self.tau = args.target_mitigate_coef
        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f"{args}\n")
        
    def action_before_train(self):
        '''
        Action before training:
        - facade setup:
            - buffer setup
        - run random episodes to build-up the initial buffer
        '''
        self.facade.initialize_train() # buffer setup
        prepare_step = 0
        # random explore before training
        initial_epsilon = 1.0
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
        while not self.facade.is_training_available:
            observation = self.run_episode_step(0, initial_epsilon, observation, True)
            prepare_step += 1
        # training records
        self.training_history = {"Q_loss": [], "V_loss": []}
        
        print(f"Total {prepare_step} prepare steps")
        
        
    def run_episode_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        with torch.no_grad():
            # sample action
            policy_output = self.facade.apply_policy(observation, self.Q_net, epsilon, do_explore = True)
            # apply action on environment and update replay buffer
            next_observation, reward, done, info = self.facade.env_step(policy_output)
            # update replay buffer
            if do_buffer_update:
                self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
        return next_observation
            
#     def run_an_episode(self, epsilon, initial_observation = None, with_train = False, pick_rows = None):
#         '''
#         Run episode for a batch of user
#         @input:
#         - epsilon: greedy epsilon for random exploration
#         - initial_observation
#         - with_train: apply batch training for each step of the episode
#         - pick_rows: pick certain rows of the data when reseting the environment
#         '''
#         # observation --> state, action
#         if initial_observation:
#             observation = initial_observation
#         elif pick_rows:
#             observation = self.facade.reset_env({"batch_size": len(pick_rows), 'pick_rows': pick_rows})
#         else:
#             observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
#         step = 0
#         done = [False] * self.batch_size
#         train_report = None
#         while sum(done) < len(done):
#             step += 1
#             with torch.no_grad():
#                 # sample action
#                 policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore = True)
#                 # apply action on environment and update replay buffer
#                 next_observation, reward, done, info = self.facade.env_step(policy_output)
#                 # update replay buffer
#                 if not pick_rows:
#                     self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
#                 # observate for the next step
#                 observation = next_observation
#             if with_train:
#                 train_report = self.step_train()
#         episode_reward = self.facade.get_total_reward()
#         return {'average_total_reward': np.mean(episode_reward['total_rewards']),
#                 'reward_variance': np.var(episode_reward['total_rewards']),
#                 'max_total_reward': np.max(episode_reward['total_rewards']),
#                 'min_total_reward': np.min(episode_reward['total_rewards']),
#                 'average_n_step': np.mean(episode_reward['n_step']),
#                 'step': step, 
#                 'buffer_size': self.facade.current_buffer_size}, train_report

    def step_train(self):
        observation, policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.batch_size)
#         reward = torch.tensor(reward)
#         done_mask = torch.tensor(done_mask)
        
        V_loss, Q_loss = self.get_dqn_loss(observation, policy_output, reward, done_mask, next_observation)
        
        self.training_history['V_loss'].append(V_loss.item())
        self.training_history['Q_loss'].append(Q_loss.item())


        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.Q_net.parameters(), self.Q_net_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['Q_loss'][-1],
                             self.training_history['V_loss'][-1],)}
    
    def get_dqn_loss(self, observation, policy_output, reward, done_mask, next_observation, 
                      do_Q_net_update = True, do_critic_update = True):
        
        current_policy_output = self.facade.apply_policy(observation, self.Q_net)
        # Get current Q estimate
        
        current_Q_de = (current_policy_output['action_Q'].sum(dim=1)).detach()
        
        S = current_policy_output ['state_emb'].detach()
        current_V = self.critic1({'state_emb': S})['v']
        # Compute the target Q value
        
        hac_pro = policy_output ['hac_pro']
        V_loss = (hac_pro * F.mse_loss(current_Q_de, current_V)).mean()
        
        if do_critic_update and self.critic1_lr > 0:
            # Optimize the critic
            self.V_optimizer.zero_grad()
            V_loss.backward()
            self.V_optimizer.step()
        
        current_Q = current_policy_output['action_Q'].sum(dim=1)
        
        # Compute the target Q value
        next_policy_output = self.facade.apply_policy(next_observation, self.Q_net_target)
        
        S_prime = next_policy_output['state_emb'].detach()
        next_V = self.critic1_target({'state_emb': S_prime})['v']
        
        Q_S = reward + self.gamma * (~done_mask * next_V)


        # Compute critic loss
        Q_loss = F.mse_loss(current_Q, Q_S).mean()
        
        
#         actor_reg = policy_output['reg']

        if do_Q_net_update and self.Q_lr > 0:
            # Optimize the actor 
            self.Q_optimizer.zero_grad()
            Q_loss.backward()
            self.Q_optimizer.step()
            
        return V_loss, Q_loss


    def save(self):
        torch.save(self.critic1.state_dict(), self.save_path + "_critic1")
        torch.save(self.V_optimizer.state_dict(), self.save_path + "_V_optimizer")
        
        torch.save(self.Q_net.state_dict(), self.save_path + "_Q_net")
        torch.save(self.Q_optimizer.state_dict(), self.save_path + "_Q_optimizer")

        

    def load(self):
        self.critic1.load_state_dict(torch.load(self.save_path + "_critic1", map_location=self.device))
        self.V_optimizer.load_state_dict(torch.load(self.save_path + "_V_optimizer", map_location=self.device))
        self.critic1_target = copy.deepcopy(self.critic1)
        
        self.Q_net.load_state_dict(torch.load(self.save_path + "_Q_net", map_location=self.device))
        self.Q_optimizer.load_state_dict(torch.load(self.save_path + "_Q_optimizer", map_location=self.device))
        self.Q_net_target = copy.deepcopy(self.Q_net)

        
#     def log_iteration(self, step):
#         episode_report, train_report = self.get_report()
#         log_str = f"step: {step} @ episode report: {episode_report} @ step loss: {train_report['step_loss']}\n"
# #         if train_report:
# #             log_str = f"step: {step} @ episode report: {episode_report} @ step loss (actor,critic): {train_report['step_loss']}\n"
# #         else:
# #             log_str = f"step: {step} @ episode report: {episode_report}\n"
#         with open(self.save_path + ".report", 'a') as outfile:
#             outfile.write(log_str)
#         return log_str

#     def test(self):
#         t = time.time()
#         # Testing
#         self.facade.initialize_train()
#         self.load()
#         print("Testing:")
#         # for i in tqdm(range(self.n_iter)):
#         with torch.no_grad():
#             for i in tqdm(range(len(self.facade.env.reader) // self.batch_size)):
#                 pick_rows = [row for row in range(i * self.batch_size, (i + 1) * self.batch_size)]
#                 episode_report, _ = self.run_an_episode(self.exploration_scheduler.value(i), pick_rows = pick_rows)

#                 if (i+1) % 1 == 0:
#                     t_ = time.time()
#                     print(f"Episode {i+1}, time diff {t_ - t})")
#                     print(self.log_iteration(i, episode_report))
#                     t = t_