import numpy as np
import random
from types import SimpleNamespace as SN
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import fym.logging as logging
from fym.utils import rot

from dynamics import SingleQuadSlungLoad
from utils import draw_plot, hardupdate, softupdate

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def load_config():
    cfg = SN()
    cfg.dt = 0.1
    cfg.max_t = 5.
    cfg.solver = 'odeint'
    cfg.ode_step_len = 10
    cfg.dir = Path('log', datetime.today().strftime('%Y%m%d-%H%M%S'))

    cfg.load = SN()
    cfg.load.mass = 0.152
    cfg.load.pos_init = np.vstack((0., 0., 5.))
    cfg.load.vel_init = np.vstack((0., 0., 0.))

    cfg.link = SN()
    cfg.link.len = 1.2
    cfg.link.uvec_init = np.vstack((0., 0., 1.))
    cfg.link.omega_init = np.vstack((0., 0., 0.))
    cfg.link.uvec_bound = [[-np.pi, np.pi], [0, np.pi/4]]
    cfg.link.omega_bound = [-0.5, 0.5]

    cfg.quad = SN()
    cfg.quad.mass = 2.473
    cfg.quad.A = -7.904
    cfg.quad.B = 7.979
    cfg.quad.phi_init = 0.
    cfg.quad.theta_init = 0.

    cfg.ddpg = SN()
    cfg.ddpg.action_scaling = torch.Tensor([4., np.pi/6, np.pi/6])
    cfg.ddpg.action_bias = torch.Tensor([26., 0., 0.])
    cfg.ddpg.state_dim = 12
    cfg.ddpg.action_dim = 3
    cfg.ddpg.memory_size = 20000
    cfg.ddpg.batch_size = 64
    cfg.ddpg.actor_lr = 0.0001
    cfg.ddpg.critic_lr = 0.001
    cfg.ddpg.discount = 0.999
    cfg.ddpg.softupdate = 0.001

    cfg.noise = SN()
    cfg.noise.sigma = [2., 0.05, 0.05]
    cfg.animation = SN()
    return cfg


class ActorNet(nn.Module):
    def __init__(self, cfg):
        super(ActorNet, self).__init__()
        self.lin1 = nn.Linear(cfg.ddpg.state_dim, 32)
        self.lin2 = nn.Linear(32, 64)
        self.lin3 = nn.Linear(64, 128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64, 32)
        self.lin6 = nn.Linear(32, 16)
        self.lin7 = nn.Linear(16, cfg.ddpg.action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.cfg = cfg

    def forward(self, state):
        x1 = self.relu(self.lin1(state))
        x2 = self.relu(self.lin2(x1))
        x3 = self.relu(self.lin3(x2))
        x4 = self.relu(self.lin4(x3))
        x5 = self.relu(self.lin5(x4))
        x6 = self.relu(self.lin6(x5))
        x7 = self.tanh(self.lin7(x6))
        x_scaled = x7 * self.cfg.ddpg.action_scaling \
            + self.cfg.ddpg.action_bias
        return x_scaled


class CriticNet(nn.Module):
    def __init__(self, cfg):
        super(CriticNet, self).__init__()
        self.lin1 = nn.Linear(cfg.ddpg.state_dim+cfg.ddpg.action_dim, 32)
        self.lin2 = nn.Linear(32, 64)
        self.lin3 = nn.Linear(64, 128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64, 32)
        self.lin6 = nn.Linear(32, 16)
        self.lin7 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, state_w_action):
        x1 = self.relu(self.lin1(state_w_action))
        x2 = self.relu(self.lin2(x1))
        x3 = self.relu(self.lin3(x2))
        x4 = self.relu(self.lin4(x3))
        x5 = self.relu(self.lin5(x4))
        x6 = self.relu(self.lin6(x5))
        x7 = self.relu(self.lin7(x6))
        return x7


class DDPG:
    def __init__(self, cfg):
        self.memory = deque(maxlen=cfg.ddpg.memory_size)
        self.behavior_actor = ActorNet(cfg).float()
        self.behavior_critic = CriticNet(cfg).float()
        self.target_actor = ActorNet(cfg).float()
        self.target_critic = CriticNet(cfg).float()
        self.actor_optim = optim.Adam(
            self.behavior_actor.parameters(), lr=cfg.ddpg.actor_lr
        )
        self.critic_optim = optim.Adam(
            self.behavior_critic.parameters(), lr=cfg.ddpg.critic_lr
        )
        self.mse = nn.MSELoss()
        hardupdate(self.target_actor, self.behavior_actor)
        hardupdate(self.target_critic, self.behavior_critic)
        self.cfg = cfg

    def get_action(self, state, net="behavior"):
        with torch.no_grad():
            action = self.behavior_actor(torch.FloatTensor(state)) \
                if net == "behavior" \
                else self.target_actor(torch.FloatTensor(state))
        return np.array(np.squeeze(action))

    def memorize(self, item):
        self.memory.append(item)

    def get_sample(self):
        sample = random.sample(self.memory, self.cfg.ddpg.batch_size)
        state, action, reward, state_next, epi_done = zip(*sample)
        x = torch.tensor(state, requires_grad=True).float()
        u = torch.tensor(action, requires_grad=True).float()
        r = torch.tensor(reward, requires_grad=True).float()
        xn = torch.tensor(state_next, requires_grad=True).float()
        done = torch.tensor(epi_done, requires_grad=True).float().view(-1,1)
        return x, u, r, xn, done

    def train(self):
        x, u, r, xn, done = self.get_sample()
        with torch.no_grad():
            action = self.target_actor(xn)
            Qn = self.target_critic(torch.cat([xn, action], 1))
            target = r + (1-done) * self.cfg.ddpg.discount * Qn
        Q_w_noise_action = self.behavior_critic(torch.cat([x,u], 1))
        self.critic_optim.zero_grad()
        critic_loss = self.mse(Q_w_noise_action, target)
        critic_loss.backward()
        self.critic_optim.step()

        action_wo_noise = self.behavior_actor(x)
        Q = self.behavior_critic(torch.cat([x, action_wo_noise],1))
        self.actor_optim.zero_grad()
        actor_loss = torch.sum(-Q)
        actor_loss.backward()
        self.actor_optim.step()

        softupdate(
            self.target_actor,
            self.behavior_actor,
            self.cfg.ddpg.softupdate)
        softupdate(
            self.target_critic,
            self.behavior_critic,
            self.cfg.ddpg.softupdate)

    def save_parameters(self, path_save):
        torch.save({
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'behavior_actor': self.behavior_actor.state_dict(),
            'behavior_critic': self.behavior_critic.state_dict()
        }, path_save)


def dynamics_test():
    cfg = load_config()
    env = SingleQuadSlungLoad(cfg)
    env_path = Path(cfg.dir, "env_data.h5")
    env.logger = logging.Logger(env_path)
    env.logger.set_info(cfg=cfg)
    agent_path = Path(cfg.dir, "agent_data.h5")
    logger_agent = logging.Logger(agent_path)

    obs = env.reset(fixed_init=True)
    action = np.array([27., 0., 0.])
    while True:
        xn, reward, done, info = env.step(action)
        logger_agent.record(**info)
        if done:
            break
    logger_agent.close()
    env.logger.close()
    env.close()
    draw_plot(env_path, agent_path, cfg.dir)

def train(agent, des, cfg, env):
    x = env.reset()
    while True:
        noise = np.random.normal(scale=cfg.noise.sigma, size=3)
        action = np.clip(
            agent.get_action(x) + noise,
            np.array(-cfg.ddpg.action_scaling + cfg.ddpg.action_bias),
            np.array(cfg.ddpg.action_scaling + cfg.ddpg.action_bias)
        )
        xn, r, done, _ = env.step(action, des)
        agent.memorize((x, action, r, xn, done))
        x = xn
        if len(agent.memory) > 5 * cfg.ddpg.batch_size:
            agent.train()
        if done:
            break

def evaluate(env, agent, des, cfg, dir_env_data, dir_agent_data):
    env.logger = logging.Logger(dir_env_data)
    env.logger.set_info(cfg=cfg)
    logger_agent = logging.Logger(dir_agent_data)
    x = env.reset(fixed_init=True)
    while True:
        action = agent.get_action(x)
        xn, _, done, info = env.step(action, des)
        logger_agent.record(**info)
        x = xn
        if done:
            break
    logger_agent.close()
    env.logger.close()

def main():
    cfg = load_config()
    env = SingleQuadSlungLoad(cfg)
    agent = DDPG(cfg)
    des = cfg.load.pos_des

    for epi_num in tqdm(range(cfg.epi_train)):
        train(agent, des, cfg, env)

        if (epi_num+1) % cfg.epi_eval == 0:
            dir_save = Path(cfg.dir, f"epi_after_{epi_num+1:05d}")
            dir_env_data = Path(dir_save, "env_data.h5")
            dir_agent_data = Path(dir_save, "agent_data.h5")
            dir_agent_params = Path(dir_save, "agent_params.h5")

            evaluate(env, agent, des, cfg, dir_env_data, dir_agent_data)
            draw_plot(dir_env_data, dir_agent_data, dir_save)
            agent.save_parameters(dir_agent_params)
    env.close()

if __name__ == "__main__":
    dynamics_test()
    plt.close('all')





