import torch
import mujoco_py
import gym
import numpy as np

#from colabgymrender.recorder import Recorder
from transformers import DecisionTransformerModel, DecisionTransformerPreTrainedModel
from transformers import AutoModel
from transformers.trainer import Trainer

_config = DecisionTransformerPreTrainedModel.config_class(
        state_dim=200,
        act_dim=8,
        hidden_size=128,
        max_ep_len=77
    )
_config.state_dim = 200
_config.act_dim = 8
_config.max_ep_len = 77


def get_pod_data_transformed():
    import os
    import pandas as pd
    import numpy as np

    obs_size = 5
    data_size = 50
    dfs = []

    dt_pod_train_root = f"dt_train_data/zelda/dt_pod_exp_traj_obs_{obs_size}_ep_len_77_goal_size_{data_size}"
    for file in os.listdir(dt_pod_train_root)[:4]:
        if '.csv' in file and file == "df_100000.csv": 
            df = pd.read_csv(f"{dt_pod_train_root}/{file}")
            dfs.append(df)

    df = pd.concat(dfs)
    print(f"df shape: {df.shape}")
    df.head()


    # done_idxs processing
    done_idxs = np.array([d for d in df['done_idx'].values if d != -1])

    # returns processing
    returns = [r for r in df['returns'].values if r != -1]
    returns.insert(0, 0)
    np.array(returns)

    # timesteps processing
    timesteps = [t for t in df['timesteps'].values]
    timesteps.insert(0,0)
    timesteps = np.array(timesteps)

    # drop done_idxs, returns, timesteps from df
    df.drop("done_idx", axis=1, inplace=True)
    df.drop("returns", axis=1, inplace=True)
    df.drop("timesteps", axis=1, inplace=True)

    # processing rtgs
    rtgs = df['col_2201'].values
    # drop rtgs
    df.drop("col_2201", axis=1, inplace=True)

    actions = df['col_2200'].values
    df.drop('col_2200', axis=1, inplace=True)
    one_hot_actions = np.zeros((len(actions),max(actions)+1))

    rows = np.arange(actions.size)

    one_hot_actions[rows, actions] = 1
#     print(f"{one_hot_actions.shape}")
#     import sys
#     sys.exit(0)
    actions = one_hot_actions
    

    obss = []
    for idx in range(len(df)):
        new_row = df.iloc[idx, :].values.reshape((11,200))
#         new_row = df.iloc[idx, :].values.reshape((200, 11))
        obss.append(new_row)
    return obss, actions, done_idxs, rtgs, timesteps, returns

__obss, __actions, __done_idxs, __rtgs, __timesteps, __rewards = get_pod_data_transformed()

# Pull from these, (obss, actions, done_idxs, rtgs, timesteps, returns), as if they are being accumulated in the env interaction loop



# This function:
# 1) keeps updates the context length for action, state, reward, timestep
# 2) sends updated (action, state, reward, timestep) to model to back out the next predicted action
def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    # This implementation does not condition on past rewards
    # print(f"BEFORE RESHAPE:")
    # print(f"=================")
    # print(f"states shape: {states.shape} ")
    # print(f"states: {states} ")
    # print(f"\n\n\n\n")

    # print(f"actions shape: {actions.shape} ")
    # print(f"actions: {actions} ")
    # print(f"\n\n\n\n")

    # print(f"returns_to_go shape: {returns_to_go.shape} ")
    # print(f"returns_to_go: {returns_to_go} ")
    # print(f"\n\n\n\n")

    # print(f"timesteps shape: {timesteps.shape} ")
    # print(f"timesteps: {timesteps} ")
    # print(f"\n\n\n\n")


    # print(f"=================")
    
    model.config.state_dim = 200 #torch.cat([states, torch.zeros((1, 200), device=device)], dim=0)
    model.config.act_dim = 8
    model.config.max_length = 11

    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    padding = model.config.max_length - states.shape[1]

    # pad all tokens to sequence length
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, model.config.act_dim)), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

    print(f"AFTER RESHAPE:")
    print(f"=================")
    print(f"states shape: {states.shape} ")
    print(f"states: {states} ")
    print(f"\n")

    print(f"actions shape: {actions.shape} ")
    print(f"actions: {actions} ")
    print(f"\n")

    print(f"returns_to_go shape: {returns_to_go.shape} ")
    print(f"returns_to_go: {returns_to_go} ")
    print(f"\n\n\n\n")

    print(f"timesteps shape: {timesteps.shape} ")
    print(f"timesteps: {timesteps} ")
    print(f"\n\n\n\n\n\n")
    print(f"=================")

    state_preds, action_preds, return_preds = model(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    return action_preds[0, -1]


# We need to start with empty (states, actions, rtg, timesteps) at each sim. iteration,
# we pull the next information from the zelda training data (i.e. the populated (states, actions, rtg, timesteps)
"""
    Start out empty buckets to be filled by the training data during sim. loop:
        state_dim: 11,(11,) grows to (1, 20, 11); 20 is their context length and 11 is their state dim --> ours will start at (200,) and grow to (1, 11, 200)
        act_dim: 3,(3,) grows to (1, 20, 3); ours will start at (8,) and grow to (1, 11, 8)
"""



# Create the model
# Create the decision transformer model
# model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-expert", from_tf=True)
device = "cpu"
# model = AutoModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-expert", from_tf=False, config=_config)

model = DecisionTransformerModel(_config)
model = model.to(device)
print(f"model.config.state_dim is {model.config.state_dim}")
print(f"type model.config.state_dim is {type(model.config.state_dim)}")

## Some constants for the sim. loop 
max_ep_len = 77 # max iterations per ep is 77
scale = 77.0  # normalization for rewards/returns
TARGET_RETURN = 20 / scale  # evaluation is conditioned on a max path length of 20, scaled accordingly


import random
## For us the state_mean & state_std are both dim (1,200), we have to compute this over the train data
# mean and std computed from training dataset these are available in the model card for each model.
# state_mean = np.array(
#     [-1.4, -0.11208222, -0.5506444,  -0.13188992, -0.00378754,  2.6071432,
#      0.02322114, -0.01626922, -0.06840388, -0.05183131,  0.04272673,]
# )
# state_std = np.array(
#     [0.15980862, 0.0446214,  0.14307782, 0.17629202, 0.5912333,  0.5899924,
#  1.5405099,  0.8152689,  2.0173461,  2.4107876,  5.8440027,]
# )


state_mean = np.array(
    [random.uniform(-1,1) for _ in range(200)]
    )

state_std = np.array(
    [random.uniform(-1,1) for _ in range(200)]
    )




state_mean = torch.from_numpy(state_mean).to(device=device)
state_std = torch.from_numpy(state_std).to(device=device)



"""
Following variables are used to build the trajectory during the sim. loop 

"""
import numpy as np
episode_return, episode_length = 0, 0
# numpy array (11,) --> ours is (1, 200,)
# env = gym.make("Hopper-v3")
state = np.array([i for i in range(200)])#env.reset()

state_dim = 200 #env.observation_space.shape[0]
# print(f"type {type(env.observation_space)}")
# print(f"state_dim: {state_dim},{env.observation_space.shape}")
act_dim = 8 #env.action_space.shape[0]

# starts with dim (1,1) --> ?? what does this grow to??
# for this we need to compute the current_target_path_length =  target_path_length / 77; target_path_length is 20 for us.
target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32) # (200,)

# starts with dim (3,) --> grows to (1, 20, 3); 3 is their action dim -> ours starts at (8,) grows to (1,11,8)
actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)

# starts with dim (1,1) -> will grow to size (1, 20, 1) *20 is context length; ours will grow to (1,11,1)
rewards = torch.zeros(0, device=device, dtype=torch.float32) # (1, 11, 1)

# starts with dim (1,1) -> grows to (1,20) --- ours starts at (1,1) grows to (1,11)
timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1) 



# for t in range(max_ep_len):
#     actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
#     rewards = torch.cat([rewards, torch.zeros(1, device=device)]) 

#     action = get_action(
#         model,
#         (states - state_mean) / state_std,
#         actions,
#         rewards,
#         target_return,
#         timesteps,
#     )
# #     if t > 50:
# #         import sys 
# #         sys.exit(0)
#     actions[-1] = action
#     action = action.detach().cpu().numpy()

#     state, reward, done, _ = env.step(action)

#     cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
#     states = torch.cat([states, cur_state], dim=0)
#     rewards[-1] = reward

#     pred_return = target_return[0, -1] - (reward / scale)
#     target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
#     timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

#     episode_return += reward
#     episode_length += 1

#     if done:
#         break

## The filled training data
__obss, __actions, __done_idxs, __rtgs, __timesteps, __rewards

# print(f"__obss is {__obss}")
# print(f"__obss type is {type(__obss)}")
print(f"len __obss is {len(__obss)}")
print(f"dim __obss[0] is {__obss[0].shape}")
print(f"======================")

print(f"len __actions is {len(__actions)}")
print(f"dim __actions[0] is {__actions[:10]}")
print(f"======================")

print(f"len __rtgs is {len(__rtgs)}")
print(f"dim __rtgs[0] is {__rtgs[0]}")
print(f"======================")

# import sys 
# sys.exit(0)


# class TrainerConfig:
#     # optimization parameters
#     max_epochs = 10
#     batch_size = 1
#     learning_rate = 3e-4
#     betas = (0.9, 0.95)
#     grad_norm_clip = 1.0
#     weight_decay = 0.1 # only applied on matmul weights
#     # learning rate decay params: linear warmup followed by cosine decay to 10% of original
#     lr_decay = False
#     warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
#     final_tokens = 260e9 # (at what point we reach 10% of original LR)
#     # checkpoint settings
#     ckpt_path = "decision_transformer_pcg_models"
#     num_workers = 0 # for DataLoader

#     def __init__(self, **kwargs):
#         for k,v in kwargs.items():
#             setattr(self, k, v)

# tconf = TrainerConfig()
# trainer = Trainer(model, preprocess_logits_for_metrics=True)
"""
Train method from model:
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        
        #Will save the model, so you can reload it using `from_pretrained()`.
        #Will only save from the main process.
"""

for t in range(len(__obss)):
    actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
    rewards = torch.cat([rewards, torch.zeros(1, device=device)]) 
#     __obss[t] = torch.from_numpy(__obss[t]).to(device=device)

    action = get_action(
        model,
        (states - state_mean) / state_std,
        actions,
        rewards,
        target_return,
        timesteps,
    )

#     action = get_action(
#         model,
#         ( - state_mean) / state_std,
#         __actions[:t],
#         __rtgs[:t],
#         __returns[:t],
#         __timesteps[:t],
#     )
    
#     import sys 
#     sys.exit(0)
    if t > 50:
        print(f"Made it to t > 50!!")
        import sys 
        sys.exit(0)
    actions[-1] = action
    action = action.detach().cpu().numpy()
    print(f"__obss[t] is {__obss[t].shape}")
#     state, reward, done, _ = torch.from_numpy(__obss[t][0]).to(device=device).reshape(1, state_dim), __rewards[t], __done_idxs[t], None #env.step(action)

    state, reward, done, _ = __obss[t][0], __rewards[t], __done_idxs[t], None #env.step(action)

    cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
    states = torch.cat([states, cur_state], dim=0).to(device=device)
    rewards[-1] = reward

    pred_return = target_return[0, -1] - (reward / scale)
    target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
    timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1).to(device=device)

    episode_return += reward
    episode_length += 1

    if done:
        print(f"Made it done!!")
        pass
