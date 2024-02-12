# TD_Decomposition


- Intuitive Example : Standard TD approaches **VS** TD Decomposition approach
<!-- ![image](https://github.com/wangxiaobei565/ItemDecomposition/blob/main/img/user_reco.jpg) -->

Our RL simulator is followed [KuaiSim](https://github.com/CharlieMat/KRLBenchmark)

Online simulator for RL-based recommendation

# 0.Setup

```
conda create -n KRL python=3.8
conda activate KRL
conda install pytorch torchvision -c pytorch
conda install pandas matplotlib scikit-learn tqdm ipykernel
python -m ipykernel install --user --name KRL --display-name "KRL"
```

# 1. Simulator Setup

### Data Processing

See preprocess/KuaiRandDataset.ipynb for details


## 1.1 User Model

1.1 Immediate User Response Model

Example raw data format in preprocessed KuaiRand: 

> (session_id, request_id, user_id, video_id, date, time, is_click, is_like, is_comment, is_forward, is_follow, is_hate, long_view)

Example item meta data format in preprocessed KuaiRand: 

> (video_id, video_type, upload_type, music_type, log_duration, tag)

Example user meta data format in preprocessed KuaiRand: 

> (user_active_degree, is_live_streamer, is_video_author, follow_user_num_range, fans_user_num_range, friend_user_num_range, register_days_range, onehot_feat{0,1,6,9,10,11,12,13,14,15,16,17})

```
bash train_multi_behavior_user_response.sh
```

Note: multi-behavior user response models consists the state_encoder that is assumed to be the ground truth user state transition model.


# 2. Benchmarks

## 2.1 Listwise Recommendation

### 2.1.1 Setup

Evaluation metrics and protocol

**List-wise reward (L-reward)** is the average of item-wise immediate reward. We use both the average L-reward and the max L-reward across user requests in a mini-batch. 



### 2.1.2 Training

```
bash train_{model name}_krpure_requestlevel.sh
```


## 2.2 Whole-session Recommendation

Whole-session user interaction involves multiple request-feedback loops.

### 2.2.1 Setup

Evaluation metrics and protocol

**Whole-session reward**: total reward is the average sum of immediate rewards for each session. The average reward is the average of total reward for each request.

**Depth** represents how many interactions before the user leaves.

### 2.2.2 Training

```
bash train_{model name}_krpure_wholesession.sh
```





# 3. Our code
We have released our TD decomposition approach with four different reinforcement learning methods: model actor-critic (A2C), deep deterministic policy gradient (DDPG), deep Q-network (DQN), and Hyper-Actor Critic(HAC). All original and decomposed versions of these methods are included in the /agents directory, along with their corresponding facade, policy, and critic modules.

Our TD decomposition method can be easily applied to any other TD-based reinforcement learning method. Below, we will provide an example of how to update an RL-based TD method to a decomposed version：
## critic 

We should parpare two critic: one for Q and another for V
```
@input -> ouput:
 Q  -  (B, state_dim)  ->  (B, action_dim)
 V  -  (B, state_dim)  ->  (B, 1)
```

## facade

facade for TD decomposition
```
# 1. add one element debias in buffer
# 2. calculate the normal 

```

## agents
We need to update the get_loss function for TD decomposition
1. Init two critic for Q and V
```
self.critic1 = facade.critic1
self.critic1_target = copy.deepcopy(self.critic1)
self.V_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=args.critic1_lr, 
                                         weight_decay=args.critic_decay)
self.critic2 = facade.critic2
self.critic2_target = copy.deepcopy(self.critic2)
self.Q_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=args.critic2_lr, 
```
2. To calculate debias_beta, we need to use the observed_likelihood values stored in the replay buffer, along with the current likelihood computed during training.
```
observed_likelihood = policy_output['hac_pro']
observed_action = policy_output['action_emb']
curr_mu = current_policy_output['action_emb']
curr_std = self.facade.noise_var
curr_likelihood = torch.exp(-((observed_action - curr_mu) ** 2) / (2 * (curr_std** 2))) / (torch.sqrt(2 * torch.tensor(torch.pi) * (curr_std** 2))) + 1e-20
curr_likelihood = curr_likelihood.mean(dim=1)
debias_beta = (self.lambda_ + curr_likelihood) / (self.lambda_ + observed_likelihood) 
```
3. Two-step TD Decomposition
- action TD
```
# get Q(t) and V(t)
current_V = self.critic1({'state_emb': S})['v']
current_Q = self.critic2({'state_emb': S, 'action_emb': L})['q']

# actionTD Loss with debias_beta
loss_actionTD = (debias_beta * F.mse_loss(current_Q_de, current_V)).mean()      
        
if do_critic_update and self.critic1_lr > 0:
    self.V_optimizer.zero_grad()
    loss_actionTD.backward()
    self.V_optimizer.step()
```
- state TD

```
# get reward, Q(t) and V(t+1)
next_V = self.critic1({'state_emb': S_})['v']
current_Q = self.critic2({'state_emb': S, 'action_emb': L})['q']

# stateTD Loss
Q_S = reward + self.gamma * (~done_mask * next_V)

loss_stateTD = F.mse_loss(current_Q, Q_S).mean()
        
if do_critic_update and self.critic2_lr > 0:
    self.Q_optimizer.zero_grad()
    loss_stateTD.backward()  
    self.Q_optimizer.step()
```

## 3. Run code
#### Search optimal hyperparameter for different method(optinonal)
```
cd /code/scripts/hyperparameter_search
bash XXXX.sh
```
This step is to find optimal performance with adjustable hyperparameter. Our result can list as :
|dataset|method|actor_lr|critic_lr|weight_lr|
|--|--|--|--|--|
|ML1M|itemA2C|3e-4|3e-5|None|
|KuaiRand|itemA2C|3e-4|3e-5|None|
|ML1M|itemA2C-W|3e-4|3e-5|None|
|KuaiRand|itemA2C-W|3e-4|3e-5|None|
|ML1M|itemA2C-M|3e-5|3e-6|1e-8|
|KuaiRand|itemA2C-M|1e-3|3e-6|1e-7|


#### run the main experiments or ablation experiments
```
cd /code/scripts/XXXX
bash XXXX.sh
```

We give all of our training config in the scripts and the plot utils in the code to show the results visually.
#### Note:
- The experiments are easy to reproduce since our experiments are running in one GPU Tesla T4 with 15 GB memory.
- The log name must be right for User Response Model.
- The model save path can be changed by editing save_path and log_path after create the path.
