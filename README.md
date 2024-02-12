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

### 2.1.3 Baselines

| Algorithm | Average L-reward | Max L-reward |  Coverage   |    ILD    |
| :-------: | :--------------: | :----------: | :---------: | :-------: |
|    CF     |    **2.253**     |    4.039     |   100.969   |   0.543   |
| ListCVAE  |      2.075       |  **4.042**   | **446.100** | **0.565** |
|    PRM    |      2.174       |    3.811     |   27.520    |   0.53    |



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
We release our approach TD decomposition with model actor-critic(A2C), DDPG, DQN and HAC and all original and decomposed version are in /agents. The corresponding  facade, policy, critic are also concluded.



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
