'''

Example RL script to train a simple agent with poke-env and 
pfrl's implementation of PPO

-----------------------------------------------------

requirements:
    pip install {poke-env, pfrl, tensorboard, six}

tested with:
    poke-env=0.4.20 (nice)
    pfrl==0.3.0


-----------------------------------------------------
what this script does:
    1. defines an agent
    2. trains the agent against an opponent
    3. evaluates the trained agent against the same opponent

things needed to achieve this:
    1. an agent:
      + a environment
      + a model (neural network)
      + a RL algorithm
    2. an opponent
    3. a working Pokemon Showdown server
'''

import random
import numpy as np
import time
import os
import argparse

from poke_env.player_configuration import PlayerConfiguration
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer

import pfrl

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

class base_env(Gen8EnvSinglePlayer):
    '''
    Use poke-env's EnvPlayer class to define a env
    thru which the Agent will interact with a Showdown Server.

    Must implement:
        embed_battle: defines what observation the agent will see
        computer_reward: defines what reward the agent will receive
    '''
    def __init__(self, unique_index): # the index is for preventing duplicated username
        Gen8EnvSinglePlayer.__init__(self, player_configuration=PlayerConfiguration('Env'+unique_index, None))

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,

                )
        res = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
            ]
        )
        # convert to float Tensor because nn.Module only accepts float Tensor as input
        return torch.Tensor(res).to(torch.float32)

    def compute_reward(self, battle) -> float:
        '''
        ideally the agent should learn with only victory rewards
        however in reality, if the agent is not learning, 
        try turn up faint_value & hp_value
        '''
        return self.reward_computing_helper(
            battle, fainted_value=0.0125, hp_value=0.0005, victory_value=1
        )

class BaseModel(nn.Module):
    '''
    implement a simple neural network which the Agent will use to make decisions

    Since we are using PPO in this example, 
    the neural network need to have both a value head and policy head
    and forward() should return both action distribution(policy) and state value

    more details available in the PPO papar
    https://arxiv.org/abs/1707.06347
    '''
    def __init__(self):
        super(BaseModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(64, 10),
            pfrl.policies.SoftmaxCategoricalHead()
        )
        self.value_head = nn.Linear(64,1)

    def forward(self, obs):
        '''
        input: obs(Tensor(dtype=float32))
        return : Tuple(policy(torch.distributions), value(Tensor(dtype=float32)))
        '''
        feature = self.feature_extractor(obs)
        policy = self.policy_head(feature)
        value = self.value_head(feature)
        return (policy, value)

class TypedMaxDamagePlayer(RandomPlayer):
    '''
    An opponent to train & evaluate against.
    Selects the move that does most damage
    '''
    def choose_move(self, battle):
        def_type_1 = battle.opponent_active_pokemon.type_1
        def_type_2 = battle.opponent_active_pokemon.type_2

        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power * move.type.damage_multiplier(def_type_1, def_type_2))
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

def Parse():
    '''
    hyperparameters and arguments to pass to pfrl's train function

    steps:              total train steps, default 500K
    update_interval:    for how many training steps should the agent perform one parameter update, default 8K
    gamma:              discount factor
    vf_coef:            coefficient for value function, downscale it to make training easier
    lr:                 learning rate
    eval_n_games:       number of games to play during evaluation 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=int(5e5)) 
    parser.add_argument('--update_interval', type=int, default=4096) 
    parser.add_argument('--gamma', type=float, default=0.7) 
    parser.add_argument('--vf_coef', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-4) 
    parser.add_argument('--eval_n_games', type=int, default=100) 
    args = parser.parse_args()
    return args

def train_wrapper(train_env, agent, total_steps, hooks):
    '''
    wrapper for the env_player's play_against() interface.
    will automatically interact with the gym APIs (step, reset, render, etc) 
        implemented inside poke-env's env_player module
    '''
    pfrl.experiments.train_agent(
        agent=agent,
        env=train_env,
        steps=total_steps,
        outdir="./.experiment_output/",
        step_hooks=hooks,
    )
    env.complete_current_battle()

def eval_wrapper(env, agent, n_games):
    '''
    same as above, but for evaluationg
    '''
    env.reset_battles()
    results = pfrl.experiments.eval_performance(
        env=env,
        agent=agent,
        n_episodes=n_games,
        n_steps=None,
        max_episode_len=1000
    )


if __name__=='__main__':
    # gets the process id, just for naming
    unique_index = format(os.getpid(), f"08d")

    args = Parse()
    print(args)

    # writer object for tensorboard
    writer = SummaryWriter(f'.tensorboard/{unique_index}')

    # create instance of pfrl's PPO agent
    model = BaseModel()
    env = base_env(unique_index)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    agent = pfrl.agents.PPO(
        model=model,
        optimizer=optim,
        value_func_coef=args.vf_coef,
        gamma=args.gamma,
        update_interval=args.update_interval
    )

    def writer_hook(env, agent, step):      
        '''
        pfrl's train_agent function takes "hook" functions as argument
        hooks are executed on every training step
        for more detail see https://pfrl.readthedocs.io/en/latest/experiments.html#pfrl.experiments.StepHook

        this hook is for writing information to tensorboard
        '''
        stats = dict(agent.get_statistics())
        writer.add_scalar(
            '0/total_loss', 
            stats['average_value_loss'] + stats['average_policy_loss'], 
            step
        )
        writer.add_scalar('0/policy_loss', stats['average_policy_loss'], step)
        writer.add_scalar('0/value_loss', stats['average_value_loss'], step)
        writer.add_scalar('1/entropy', stats['average_entropy'], step)

    def print_hook(env, agent, step):
        '''
        Another hook for printing current training step on stdio
        '''
        if step % 1000 == 0:
            print(f"currently at step [{step}], {args.steps}")
    # training
    train_opponent = TypedMaxDamagePlayer(battle_format='gen8randombattle', player_configuration=PlayerConfiguration('Train'+unique_index, None))
    env.play_against(
        env_algorithm=train_wrapper,
        opponent=train_opponent,
        env_algorithm_kwargs={
            'agent':agent, 
            'total_steps':args.steps, 
            'hooks':[writer_hook, print_hook]
        }
    )
    del train_opponent
    
    # env_player automatically store battle histories in a buffer. Clear the buffer.
    env.reset_battles()

    # Evaluate
    eval_opponent = TypedMaxDamagePlayer(battle_format='gen8randombattle', player_configuration=PlayerConfiguration('Eval'+unique_index, None))
    env.play_against(
        env_algorithm=eval_wrapper,
        opponent=eval_opponent,
        env_algorithm_kwargs={
            'agent':agent,
            'n_games':args.eval_n_games
        }
    )
    # record num of games won
    win_tmax = env.n_won_battles
    del eval_opponent

    print(f'Agent id [{unique_index}]: won [{win_tmax}/{args.eval_n_games}] vs TMAX')
    agent.save(f'./trained_agents/{unique_index}')
    exit()
