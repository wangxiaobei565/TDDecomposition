import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agents.BaseRLAgent import BaseRLAgent
    
class DDPG_de(BaseRLAgent):
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
        self.critic2_lr = args.critic1_lr
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
        self.training_history = {"critic1_loss": [],"critic2_loss": [], "actor_loss": []}
        
        print(f"Total {prepare_step} prepare steps")
        
        
    def run_episode_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        with torch.no_grad():
            # sample action
            policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore = True)
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
        
        critic_loss_one, critic_loss, actor_loss = self.get_ddpg_loss(observation, policy_output, reward, done_mask, next_observation)
        self.training_history['critic1_loss'].append(critic_loss_one.item())
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic2_loss'].append(critic_loss.item())

        # Update the frozen target models
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1],
                              self.training_history['critic1_loss'][-1],
                              self.training_history['critic2_loss'][-1])}
    
    def get_ddpg_loss(self, observation, policy_output, reward, done_mask, next_observation, 
                      do_actor_update = True, do_critic_update = True):
        
        # Get current Q estimate
        current_critic_output = self.facade.apply_critic(observation, 
                                                         utils.wrap_batch(policy_output, device = self.device), 
                                                         self.critic2)
        current_Q_de = current_critic_output['q'].detach()

        S = policy_output['state_emb'].detach()
        current_V = self.critic1({'state_emb': S})['v']
        # Compute the target Q value
        
        observed_likelihood = policy_output['hac_pro']
        # (B, action_dim)
        observed_action = policy_output['action_emb']
        
        # (B, action_dim)
        curr_mu = current_critic_output['action_emb']
        curr_std = self.facade.noise_var
        
        
        curr_likelihood = torch.exp(-((observed_action - curr_mu) ** 2) / (2 * (curr_std** 2))) / (torch.sqrt(2 * torch.tensor(torch.pi) * (curr_std** 2))) + 1e-20
        # (B,)
        curr_likelihood = curr_likelihood.mean(dim=1)
        
        
        debias_beta = (self.lambda_ + curr_likelihood) / (self.lambda_ + observed_likelihood) 
        
        critic_loss_one = (debias_beta * F.mse_loss(current_Q_de, current_V)).mean()
        
        
        if do_critic_update and self.critic1_lr > 0:
            # Optimize the critic
            self.V_optimizer.zero_grad()
            critic_loss_one.backward()
            self.V_optimizer.step()
            
        current_critic_output = self.facade.apply_critic(observation, 
                                                         utils.wrap_batch(policy_output, device = self.device), 
                                                         self.critic2)
        current_Q = current_critic_output['q']    
        next_policy_output = self.facade.apply_policy(next_observation, self.actor_target)
        
        S_prime = next_policy_output['state_emb'].detach()
        next_V = self.critic1_target({'state_emb': S_prime})['v']
        
        Q_S = reward + self.gamma * (~done_mask * next_V)


        # Compute critic loss
        value_loss = F.mse_loss(current_Q, Q_S).mean()
        
        # Regularization loss
#         critic_reg = current_critic_output['reg']
        
        if do_critic_update and self.critic2_lr > 0:
            # Optimize the critic
            self.Q_optimizer.zero_grad()
            value_loss.backward()
            
            self.Q_optimizer.step()

        # Compute actor loss
        policy_output = self.facade.apply_policy(observation, self.actor)
        critic_output = self.facade.apply_critic(observation, policy_output, self.critic2)
        actor_loss = -critic_output['q'].mean()
        
        # Regularization loss
#         actor_reg = policy_output['reg']

        if do_actor_update and self.actor_lr > 0:
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
        return critic_loss_one, value_loss, actor_loss


    def save(self):
        torch.save(self.critic1.state_dict(), self.save_path + "_critic1")
        torch.save(self.V_optimizer.state_dict(), self.save_path + "_V_optimizer")
        
        torch.save(self.critic2.state_dict(), self.save_path + "_critic2")
        torch.save(self.Q_optimizer.state_dict(), self.save_path + "_Q_optimizer")

        torch.save(self.actor.state_dict(), self.save_path + "_actor")
        torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")


    def load(self):
        self.critic1.load_state_dict(torch.load(self.save_path + "_critic1", map_location=self.device))
        self.V_optimizer.load_state_dict(torch.load(self.save_path + "_V_optimizer", map_location=self.device))
        self.critic1_target = copy.deepcopy(self.critic1)
        
        self.critic2.load_state_dict(torch.load(self.save_path + "_critic2", map_location=self.device))
        self.Q_optimizer.load_state_dict(torch.load(self.save_path + "_Q_optimizer", map_location=self.device))
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)

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