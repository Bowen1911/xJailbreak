import argparse
import os
import torch
import random
import numpy as np
from utils.Evaluator import Evaluator
from utils.train_rlagent import train_PPO_agent
from agent.RL import PPO
from tqdm import trange
from net import PolicyNet, ValueNet
from data.Extraction import get_data_list
from agent.LLM_agent import Llm_manager
from jailbreak_env import JbEnv
import warnings
warnings.filterwarnings('ignore')

# * ---------------------- terminal parameters -------------------------

parser = argparse.ArgumentParser(description='PPO Jailbreak LLM Mission')
parser.add_argument('--note', default=None, type=str, help='Task notes')
parser.add_argument('--special_place', default='train/xJailbreak-alpha0.2', help='Customize special saving location of experiment result, such as "log/special_place/..."')
parser.add_argument('-w', '--save', type=int, default=0, help='data saving type, 0: not saved, 1: local')
parser.add_argument('--cuda', nargs='+', default=[0], type=int, help='CUDA order')
parser.add_argument('--episodes', default=0, type=int, help='The number of training data, if it is 0, all data will be trained')
parser.add_argument('--val_interval', default=20, type=int, help='Verification interval, for pure training, it is recommended to set it larger to save time, 0 for no verify')
parser.add_argument('--epochs', default=1, type=int, help='The number of rounds run on the whole dataset, at least 1')
parser.add_argument('-s', '--seed', nargs='+', default=[42, 42], type=int, help='The start and end seeds, if there are multiple seeds, each seed runs the epochs sub-data set, which is equivalent to the product relationship')
args = parser.parse_args()

# CUDA
if isinstance(args.cuda, int):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.cuda}"  # 1 -> '1'
elif isinstance(args.cuda, list):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.cuda))  # [1, 2] -> "1,2"

# * ------ mission ------
device = "cuda" if torch.cuda.is_available() else "cpu"
env_kwargs = {}
env_kwargs['max_step'] = 10  # How many steps are the maximum number of training steps per prompt
env_kwargs['train_ratio'] = 0.8

train_kwargs = {}
train_kwargs['val_interval'] = args.val_interval
train_kwargs['val_num'] = 3  # How many samples should be tested per validation
train_kwargs['val_max_step'] = 10  # How many times to iterate when verifying each sample

# * ------ Model ------
# Helper models must be able to override safety instructions
helpLLM = Llm_manager({
    'model_path': 'your_path.../Llama3-8B-Instruct-JB',  #! DO NOT end with a "/"
    'source': 'local',
    'cuda': args.cuda
     })
helpLLM.load_model('How are you?')

reprLLM = helpLLM.embedding

# preload harmful_emb_refer and benign_emb_refer for saving time
# the method of embedding them: torch.save(reprLLM(get_data_list(['h_custom'])['data_list']).cpu(), 'data/pre_load/harmful_emb_refer.pt')
env_kwargs['harmful_emb_refer'] = torch.load('data/preload/harmful_emb_refer.pt').to('cuda')
env_kwargs['benign_emb_refer'] = torch.load('data/preload/benign_emb_refer.pt').to('cuda')

# VictimLLM must be safety aligned
victimLLM = Llm_manager({
    'model_path': 'your_path.../Qwen2.5-7B-Instruct',  #! DO NOT end with a "/"
    'source': 'local',
    'cuda': args.cuda
    })
victimLLM.load_model('How are you?')

# we use Helper models as judgeLLM
judgeLLM = helpLLM

# * ------ DATA ------
# ['h_custom', 'harmful_behaviors', 'MaliciousInstruct']
harmful_prompt = ['h_custom']
template = 'our_template'
ASR = 'asr_keyword'

# * ------ Environment ------
args.episodes = args.episodes if args.episodes else int(len(get_data_list(harmful_prompt)['data_list'])*env_kwargs['train_ratio'])

env = JbEnv(helpLLM, reprLLM, victimLLM, judgeLLM, harmful_prompt, template, ASR, **env_kwargs)
# neural network
state_dim = env.observation_space
hidden_dim = env.observation_space // 4
action_dim = env.action_space

# * ------ RL agent PPO ------
actor_lr = 1e-4
critic_lr = 2e-4
lmbda = 0.97  # Discount factor for balanced advantage
gamma = 0.9  # The temporal difference learning rate, also used as one of the factors for discounting advantage. Here we are more concerned with short-term rewards.
inner_epochs = 10
eps = 0.2

policy_net = PolicyNet(state_dim, hidden_dim, action_dim)
value_net = ValueNet(state_dim, hidden_dim)

# * ------------------------ Training ----------------------------
print(f'[ Start >>> Note: {args.note} | save: {args.save} | cuda: {args.cuda} | episodes: {args.episodes} | epochs: {args.epochs} ]')
for seed in trange(args.seed[0], args.seed[-1] + 1, mininterval=40, ncols=70):
    evaluator = Evaluator(args.special_place)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = PPO(policy_net, value_net, actor_lr, critic_lr, gamma, lmbda, inner_epochs, eps, device)
    return_list, train_time = train_PPO_agent(env, agent, args.save, args.epochs, args.episodes, seed, evaluator, **train_kwargs)

print(f'[ Done <<< Note: {args.note} | save: {args.save} | cuda: {args.cuda} | episodes: {args.episodes} | epochs: {args.epochs} ]')
