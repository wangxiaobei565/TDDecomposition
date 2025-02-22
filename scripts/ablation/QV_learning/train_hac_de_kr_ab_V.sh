mkdir -p output

# kr environment

mkdir -p output/kr/
mkdir -p output/kr/env/
mkdir -p output/kr/env/log/
mkdir -p output/kr/agents/
mkdir -p output/kr/agents/1TD_others/
mkdir -p output/kr/agents/1TD_others/de_reqhac_ab_V/


output_path="output/kr/"
log_name="kr_user_env_lr0.001_reg0.003_init"

# actor0.0001 or 0.00003  _critic1e-6_0.00003

N_ITER=30000
CONTINUE_ITER=0
GAMMA=0.9
TOPK=1
EMPTY=0

MAX_STEP=20
INITEP=0.01
REG=0.00003
NOISE=0.1
ELBOW=0.1
EP_BS=32
BS=64
SEED=3
SCORER="WideDeep"
CRITIC_LR=0.001
ACTOR_LR=0.0001
BEHAVE_LR=0
TEMPER_RATE=1.0
# for MAX_STEP in 20 

for CRITIC2_LR in 0.00003
do
    for CRITIC1_LR in 1e-9 1e-8 1e-7 1e-5 1e-4 1e-3
    do
#         for SCORER in "WideDeep"
        for SCORER in "DotScore"
        do
            for SEED in 11 13 17 19 23
            do
                for ACTOR_LR in 0.00003
                do
                    mkdir -p ${output_path}agents/1TD_others/de_reqhac_ab_V/de_hac_${SCORER}_actor${ACTOR_LR}_critic${CRITIC1_LR}_${CRITIC2_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}/

                    python train_ac_de.py\
                        --env_class KREnvironment_GPU\
                        --policy_class OneStagePolicy_with_${SCORER}\
                        --critic_class1 GeneralCritic_V\
                        --critic_class2 HACCritic\
                        --agent_class HAC_effect_de\
                        --facade_class OneStageFacade_HAC_de\
                        --seed ${SEED}\
                        --cuda 0\
                        --env_path ${output_path}env/${log_name}.env\
                        --max_step_per_episode ${MAX_STEP}\
                        --initial_temper ${MAX_STEP}\
                        --reward_func mean_with_cost\
                        --urm_log_path ${output_path}env/log/${log_name}.model.log\
                        --state_encoder_feature_dim 32\
                        --state_encoder_attn_n_head 4\
                        --state_encoder_hidden_dims 128\
                        --policy_actionnet_hidden 64\
                        --critic_hidden_dims 256 64\
                        --critic_hidden_dims_V 256 64\
                        --slate_size 6\
                        --buffer_size 100000\
                        --start_timestamp 2000\
                        --noise_var ${NOISE}\
                        --empty_start_rate ${EMPTY}\
                        --save_path ${output_path}agents/1TD_others/de_reqhac_ab_V/de_hac_${SCORER}_actor${ACTOR_LR}_critic${CRITIC1_LR}_${CRITIC2_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}/model\
                        --episode_batch_size ${EP_BS}\
                        --batch_size ${BS}\
                        --actor_lr ${ACTOR_LR}\
                        --critic1_lr ${CRITIC1_LR}\
                        --critic2_lr ${CRITIC2_LR}\
                        --behavior_lr ${BEHAVE_LR}\
                        --actor_decay ${REG}\
                        --critic_decay ${REG}\
                        --behavior_decay ${REG}\
                        --target_mitigate_coef 0.01\
                        --gamma ${GAMMA}\
                        --n_iter ${N_ITER}\
                        --initial_greedy_epsilon ${INITEP}\
                        --final_greedy_epsilon ${INITEP}\
                        --elbow_greedy ${ELBOW}\
                        --check_episode 10\
                        --topk_rate ${TOPK}\
                        > ${output_path}agents/1TD_others/de_reqhac_ab_V/de_hac_${SCORER}_actor${ACTOR_LR}_critic${CRITIC1_LR}_${CRITIC2_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}/log
                done
            done
        done
    done
done
