import numpy as np
import numpy.linalg as nla
import numpy.random as random

from fym.core import BaseEnv, BaseSystem
import fym.core as core
from fym.utils import rot

from utils import hat


class Load(BaseEnv):
    def __init__(self, cfg_load):
        super().__init__()
        self.pos = BaseSystem(cfg_load.pos_init)
        self.vel = BaseSystem(cfg_load.vel_init)
        self.mass = cfg_load.mass

    def set_dot(self, acc):
        self.pos.dot = self.vel.state
        self.vel.dot = acc


class Link(BaseEnv):
    def __init__(self, cfg_link):
        super().__init__()
        self.uvec = BaseSystem(cfg_link.uvec_init)
        self.omega = BaseSystem(cfg_link.omega_init)
        self.len = cfg_link.len

    def set_dot(self, ang_acc):
        self.uvec.dot = hat(self.omega.state).dot(self.uvec.state)
        self.omega.dot = ang_acc


class Quadrotor(BaseEnv):
    def __init__(self, cfg_quad):
        super().__init__()
        self.phi_tilde = BaseSystem(cfg_quad.phi_init)
        self.theta_tilde = BaseSystem(cfg_quad.theta_init)
        self.mass = cfg_quad.mass
        self.A = cfg_quad.A
        self.B = cfg_quad.B

    def set_dot(self, phi_des, theta_des):
        self.phi_tilde.dot = self.A*self.phi_tilde.state + self.B*phi_des
        self.theta_tilde.dot = self.A*self.theta_tilde.state + self.B*theta_des


class SingleQuadSlungLoad(BaseEnv):
    def __init__(self, cfg):
        super().__init__(dt=cfg.dt, max_t=cfg.max_t, solver=cfg.solver,
                         ode_step_len=cfg.ode_step_len)
        self.load = Load(cfg.load)
        self.link = Link(cfg.link)
        self.quad = Quadrotor(cfg.quad)
        self.e3 = np.vstack((0., 0., 1.)) 
        self.eye = np.eye(3)
        self.cfg = cfg

    def reset(self, fixed_init=False):
        super().reset()
        if fixed_init:
            f = 10 * (self.cfg.load.mass + self.cfg.quad.mass)
        else:
            self.link.uvec.state = rot.sph2cart2(
                1,
                random.uniform(
                    low=self.cfg.link.uvec_bound[0][0],
                    high=self.cfg.link.uvec_bound[0][1]
                ),
                random.uniform(
                    low=self.cfg.link.uvec_bound[1][0],
                    high=self.cfg.link.uvec_bound[1][1]
                )
            )
            omega_tmp = random.uniform(
                low=self.cfg.link.omega_bound[0],
                high=self.cfg.link.omega_bound[1],
                size=(3,1)
            )
            self.link.omega.state = omega_tmp - np.dot(
                omega_tmp.reshape(-1,), self.link.uvec.state.reshape(-1,)
            ) * self.link.uvec.state
            f = random.uniform(
                low=-self.cfg.ddpg.action_scaling[0].item() \
                + self.cfg.ddpg.action_bias[0].item(),
                high=self.cfg.ddpg.action_scaling[0].item() \
                + self.cfg.ddpg.action_bias[0].item(),
            )
        obs = self.observe(f)
        return obs

    def set_dot(self, t, action):
        f, phi_des, theta_des = action
        R_tilde = rot.angle2dcm(
            0,
            self.quad.theta_tilde.state.item(),
            self.quad.phi_tilde.state.item()
        )
        thrust = f * R_tilde.dot(self.e3)
        q_hat = hat(self.link.uvec.state)
        omega_square = self.link.omega.state[0]**2 + \
            self.link.omega.state[1]**2 + \
            self.link.omega.state[2]**2

        load_acc = ((self.eye + q_hat.dot(q_hat)/self.link.len**2).dot(thrust)\
            + self.quad.mass*omega_square*self.link.uvec.state) \
            / (self.load.mass + self.quad.mass) - 9.81*self.e3
        link_ang_acc = q_hat.dot(thrust) / (self.quad.mass * self.link.len**2)
        self.load.set_dot(load_acc)
        self.link.set_dot(link_ang_acc)
        self.quad.set_dot(phi_des, theta_des)
        return dict(thrust=thrust)

    def step(self, action, des):
        load_pos_des = des
        *_, time_out = self.update(action=action)

        obs = self.observe(action[0])
        tension = self.calculate_tension(action[0])
        tension_des = self.calculate_desried_tension(load_pos_des)
        reward = self.get_reward(tension_des, tension)
        done = self.terminate(time_out)

        info = {
            'time': self.clock.get(),
            'action': action,
            'reward': reward,
            'tension': tension,
            'tension_des': tension_des,
        }
        return obs, reward, done, info

    def observe(self, f):
        R_tilde = rot.angle2dcm(
            0,
            self.quad.theta_tilde.state.item(),
            self.quad.phi_tilde.state.item()
        )
        thrust = f * R_tilde.dot(self.e3)
        q_dot = hat(self.link.omega.state).dot(self.link.uvec.state)
        q_hat = hat(self.link.uvec.state)
        omega_square = self.link.omega.state[0]**2 + \
            self.link.omega.state[1]**2 + \
            self.link.omega.state[2]**2
        q_ddot = -(q_hat.dot(q_hat)).dot(thrust) \
            / (self.quad.mass*self.link.len**2) \
            - omega_square*self.link.uvec.state
        quad_acc = self.load.vel.dot + self.link.len*q_ddot
        obs = np.hstack([
            self.link.uvec.state.reshape(-1,),
            q_dot.reshape(-1,),
            thrust.reshape(-1,),
            quad_acc.reshape(-1,)
        ])
        return obs

    def terminate(self, done):
        done = 1. if (self.load.pos.state[2] < 0 or done) else 0.
        return done

    def logger_callback(self, t, action):
        quad_pos = self.load.pos.state + self.link.len*self.link.uvec.state
        quad_vel = self.load.vel.state + self.link.len*self.link.uvec.dot
        distance_btw_quad2load = np.sqrt(
            (quad_pos[0] - self.load.pos.state[0])**2 \
            + (quad_pos[1] - self.load.pos.state[1])**2 \
            + (quad_pos[2] - self.load.pos.state[2])**2
        )
        return dict(
            time=t, distance_btw_quad2load=distance_btw_quad2load, 
            **self.observe_dict(), quad_pos=quad_pos, quad_vel=quad_vel
        )

    def get_reward(self, tension_des, tension):
        r = -(tension - tension_des).T.dot(tension - tension_des).reshape(-1,)
        return r

    def calculate_desried_tension(self, load_pos_des):
        tmp = load_pos_des

    def calculate_tension(self, f):
        R_tilde = rot.angle2dcm(
            0,
            self.quad.theta_tilde.state.item(),
            self.quad.phi_tilde.state.item()
        )
        thrust = f * R_tilde.dot(self.e3)
        tension = self.quad.mass*(self.quad.vel.dot - 9.81*self.e3) - thrust
        return tension


        



