import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.art3d as art3d
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib.patches import Circle

import fym.logging as logging
from fym.utils import rot


def hat(v):
    v1, v2, v3 = v.squeeze()
    return np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0]
    ])

def wrap(angle):
    return (angle+np.pi) % (2*np.pi) - np.pi

def softupdate(target, behavior, softupdate_const):
    for targetParam, behaviorParam in zip(
            target.parameters(),
            behavior.parameters()
    ):
        targetParam.data.copy_(
            targetParam.data*(1.0-softupdate_const) \
            + behaviorParam.data*softupdate_const
        )

def hardupdate(target, behavior):
    for targetParam, behaviorParam in zip(
            target.parameters(),
            behavior.parameters()
    ):
        targetParam.data.copy_(behaviorParam.data)

def draw_plot(dir_env_data, dir_agent_data, dir_save):
    env_data, info = logging.load(dir_env_data, with_info=True)
    agent_data = logging.load(dir_agent_data)
    cfg = info['cfg']

    time = env_data['time']
    load_pos = env_data['load']['pos']
    load_vel = env_data['load']['vel']
    link_uvec = env_data['link']['uvec']
    link_omega = env_data['link']['omega']
    quad_phi_tilde = env_data['quad']['phi_tilde']
    quad_theta_tilde = env_data['quad']['theta_tilde']
    thrust = env_data['thrust']
    quad_pos = env_data['quad_pos']
    quad_vel = env_data['quad_vel']
    distance_btw_quad2load = env_data['distance_btw_quad2load']

    time_agent = agent_data['time']
    action = agent_data['action']

    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(time, load_pos[:,0])
    ax[1].plot(time, load_pos[:,1])
    ax[2].plot(time, load_pos[:,2])
    ax[0].set_title("Position of Load")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].axes.xaxis.set_ticklabels([])
    ax[0].set_ylabel("X [m]")
    ax[1].set_ylabel("Y [m]")
    ax[2].set_ylabel("Z [m]")
    ax[2].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(3)]
    fig.align_ylabels(ax)
    fig.savefig(
        Path(dir_save, "load_pos.png"),
        bbox_inches='tight'
    )
    plt.close('all')

    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(time, load_vel[:,0])
    ax[1].plot(time, load_vel[:,1])
    ax[2].plot(time, load_vel[:,2])
    ax[0].set_title("Velocity of Load")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].axes.xaxis.set_ticklabels([])
    ax[0].set_ylabel("$V_x$ [m/s]")
    ax[1].set_ylabel("$V_y$ [m/s]")
    ax[2].set_ylabel("$V_z$ [m/s]")
    ax[2].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(3)]
    fig.align_ylabels(ax)
    fig.savefig(
        Path(dir_save, "load_vel.png"),
        bbox_inches='tight'
    )
    plt.close('all')
    
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(time, quad_phi_tilde*np.pi/180)
    ax[1].plot(time, quad_theta_tilde*np.pi/180)
    ax[0].set_title("Attitude of Quadrotor")
    ax[0].set_ylabel("$\tilde{\phi}$ [deg]")
    ax[1].set_ylabel("$\tilde{\theta}$ [deg]")
    ax[1].set_xlabel("time [s]")
    ax[0].grid(True)
    ax[1].grid(True)
    fig.align_ylabels(ax)
    fig.savefig(
        Path(dir_save, "quad_att.png"),
        bbox_inches='tight'
    )
    plt.close('all')

    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(time, thrust[:,0])
    ax[1].plot(time, thrust[:,1])
    ax[2].plot(time, thrust[:,2])
    ax[0].set_title("Thrust of quadrotor")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].axes.xaxis.set_ticklabels([])
    ax[0].set_ylabel("$T_x$ [N]")
    ax[1].set_ylabel("$T_y$ [N]")
    ax[2].set_ylabel("$T_z$ [N]")
    ax[2].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(3)]
    fig.align_ylabels(ax)
    fig.savefig(
        Path(dir_save, "thrust.png"),
        bbox_inches='tight'
    )
    plt.close('all')

    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(time, quad_pos[:,0])
    ax[1].plot(time, quad_pos[:,1])
    ax[2].plot(time, quad_pos[:,2])
    ax[0].set_title("Position of Quadrotor")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].axes.xaxis.set_ticklabels([])
    ax[0].set_ylabel("X [m]")
    ax[1].set_ylabel("Y [m]")
    ax[2].set_ylabel("Z [m]")
    ax[2].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(3)]
    fig.align_ylabels(ax)
    fig.savefig(
        Path(dir_save, "quad_pos.png"),
        bbox_inches='tight'
    )
    plt.close('all')

    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(time, quad_vel[:,0])
    ax[1].plot(time, quad_vel[:,1])
    ax[2].plot(time, quad_vel[:,2])
    ax[0].set_title("Velocity of Quadrotor")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].axes.xaxis.set_ticklabels([])
    ax[0].set_ylabel("X [m/s]")
    ax[1].set_ylabel("Y [m/s]")
    ax[2].set_ylabel("Z [m/s]")
    ax[2].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(3)]
    fig.align_ylabels(ax)
    fig.savefig(
        Path(dir_save, "quad_vel.png"),
        bbox_inches='tight'
    )
    plt.close('all')

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(time, distance_btw_quad2load)
    ax.set_title("Distance between Qaudrotor and Load")
    ax.set_ylabel("Distance [m]")
    ax.set_xlabel("time [s]")
    ax.set_ylim(cfg.link.len-0.1, cfg.link.len+0.1)
    ax.grid(True)
    fig.savefig(
        Path(dir_save, "distance_bw_quad_load.png"),
        bbox_inches='tight'
    )
    plt.close('all')

class Quad_ani:
    def __init__(self, ax, cfg):
        d = cfg.animation.quad_size
        r = cfg.animation.rotor_size

        body_segs = np.array([
            [[d, 0, 0], [0, 0, 0]],
            [[-d, 0, 0], [0, 0, 0]],
            [[0, d, 0], [0, 0, 0]],
            [[0, -d, 0], [0, 0, 0]]
        ])
        colors = (
            (1, 0, 0, 1),
            (0, 0, 1, 1),
            (0, 0, 1, 1),
            (0, 0, 1, 1),
        )
        self.body = art3d.Line3DCollection(
            body_segs,
            colors=colors,
            linewidths=2
        )

        kwargs = dict(radius=r, ec="k", fc="k", alpha=0.3)
        self.rotors = [
            Circle((d, 0), **kwargs),
            Circle((0, d), **kwargs),
            Circle((-d, 0), **kwargs),
            Circle((0, -d), **kwargs),
        ]

        ax.add_collection3d(self.body)
        for rotor in self.rotors:
            ax.add_patch(rotor)
            art3d.pathpatch_2d_to_3d(rotor, z=0)

        self.body._base = self.body._segments3d
        for rotor in self.rotors:
            rotor._segment3d = np.array(rotor._segment3d)
            rotor._center = np.array(rotor._center + (0,))
            rotor._base = rotor._segment3d


    def set(self, pos, dcm=np.eye(3)):
        self.body._segments3d = np.array([
            dcm @ point for point in self.body._base.reshape(-1, 3)
        ]).reshape(self.body._base.shape)

        for rotor in self.rotors:
            rotor._segment3d = np.array([
                dcm @ point for point in rotor._base
            ])

        self.body._segments3d = self.body._segments3d + pos
        for rotor in self.rotors:
            rotor._segment3d += pos


class Link_ani:
    def __init__(self, ax):
        self.link = art3d.Line3D(
            [ ],
            [ ],
            [ ],
            color="k",
            linewidth=1
        )
        ax.add_line(self.link)

    def set(self, quad_pos, anchor_pos):
        self.link.set_data_3d(
            [anchor_pos[0], quad_pos[0]],
            [anchor_pos[1], quad_pos[1]],
            [anchor_pos[2], quad_pos[2]]
        )


class Payload_ani:
    def __init__(self, ax, edge_num):
        kwargs = dict(radius=r, ec="k", fc="k", alpha=0.3)
        self.rotor = Circle((0, 0), **kwargs)
        ax.add_patch(self.rotor)
        self.rotor._center = np.array(self.rotor._center + (0,))

    def set(self, pos):
        self.rotor._center += pos


class Animator:
    def __init__(self, fig, data_list, cfg, simple=False):
        self.offsets = ['collections', 'patches', 'lines', 'texts',
                        'artists', 'images']
        self.fig = fig
        self.axes = fig.axes
        # self.axes = axes
        self.data_list = data_list
        self.cfg = cfg
        self.len = len(data_list)
        self.simple = simple

    def init(self):
        self.frame_artists = []
        max_x = np.array(
            [data['load']['pos'][:,0,:].max() for data in self.data_list]
        ).max()
        min_x = np.array(
            [data['load']['pos'][:,0,:].min() for data in self.data_list]
        ).min()
        max_y = np.array(
            [data['load']['pos'][:,1,:].max() for data in self.data_list]
        ).max()
        min_y = np.array(
            [data['load']['pos'][:,1,:].min() for data in self.data_list]
        ).min()
        max_z = np.array(
            [data['load']['pos'][:,2,:].max() for data in self.data_list]
        ).max()
        min_z = np.array(
            [data['load']['pos'][:,2,:].min() for data in self.data_list]
        ).min()

        for i, ax in enumerate(self.axes):
            ax.quad = [Quad_ani(ax, self.cfg)
                       for _ in range(self.cfg.quad.num)]
            ax.link = [Link_ani(ax) for _ in range(self.cfg.quad.num)]
            ax.load = Payload_ani(ax, self.cfg.quad.num)
            ax.set_xlim3d([
                min_x - self.cfg.load.size - self.cfg.animation.quad_size,
                max_x + self.cfg.load.size + self.cfg.animation.quad_size
            ])
            ax.set_ylim3d([
                min_y - self.cfg.load.size - self.cfg.animation.quad_size,
                max_y + self.cfg.load.size + self.cfg.animation.quad_size
            ])
            ax.set_zlim3d([
                max(min_z - self.cfg.link.len[0], 0.),
                max_z + self.cfg.link.len[0] - self.cfg.load.cg[2]\
                + self.cfg.animation.quad_size
            ])
            ax.view_init(
                self.cfg.animation.view_angle[0],
                self.cfg.animation.view_angle[1]
            )
            if i >= self.len:
                ax.set_title("empty", fontsize='small', fontweight='bold')
            else:
                ax.set_title(
                    f"{(i+1)*self.cfg.epi_eval:05d}_epi",
                    fontsize='small',
                    fontweight='bold'
                )
            if self.simple:
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                ax.axes.zaxis.set_ticklabels([])
            else:
                ax.set_xlabel('x [m]', fontsize='small')
                ax.set_ylabel('y [m]', fontsize='small')
                ax.set_zlabel('z [m]', fontsize='small')
                ax.tick_params(axis='both', which='major', labelsize=8)

            for name in self.offsets:
                self.frame_artists += getattr(ax, name)
        self.fig.tight_layout()
        if not self.simple:
            self.fig.subplots_adjust(
                left=0,
                bottom=0.1,
                right=1,
                top=0.95,
                hspace=0.5,
                wspace=0
            )


        return self.frame_artists

    def get_sample(self, frame):
        self.init()
        self.update(frame)
        self.fig.show()

    def update(self, frame):
        for data, ax in zip(self.data_list, self.axes):
            load_verts = data['anchor_pos'][frame].squeeze().tolist()
            load_cg = data['load']['pos'][frame].squeeze().tolist()
            ax.load.set(load_verts, load_cg)

            for i in range(self.cfg.quad.num):
                ax.quad[i].set(
                    data["quad_pos"][frame,i,:,:].squeeze(),
                    data["quads"][f"quad{i:02d}"]["dcm"][frame,:,:].squeeze()
                )
                ax.link[i].set(
                    data["quad_pos"][frame,i,:,:].squeeze(),
                    data["anchor_pos"][frame,i,:,:].squeeze()
                )
        return self.frame_artists

    def animate(self, *args, **kwargs):
        data_len = [len(self.data_list[i]['time'])
                    for i in range(self.len)]
        frames = range(0, min(data_len), 10)
        self.anim = FuncAnimation(
            self.fig, self.update, init_func=self.init,
            frames=frames, interval=200, blit=True,
            *args, **kwargs
        )

    def save(self, path, *args, **kwargs):
        self.anim.save(path, writer="ffmpeg", fps=30, *args, **kwargs)


def compare_episode(past, ani=True):
    dir_save = list(Path('log').glob("*"))[past]
    epi_list = [x for x in dir_save.glob("*")]
    env_data_list = [
        logging.load(Path(epi_dir, "env_data.h5")) for epi_dir in epi_list
    ]
    agent_data_list = [
        logging.load(Path(epi_dir, "agent_data.h5")) for epi_dir in epi_list
    ]
    _, info = logging.load(Path(epi_list[0], "env_data.h5"), with_info=True)
    cfg = info['cfg']
    if ani == True:
        data_num = len(env_data_list)
        fig_shape = split_int(data_num)
        simple = False
        if fig_shape[0] >= 3:
            simple=True

        fig, _ = plt.subplots(
            fig_shape[0],
            fig_shape[1],
            subplot_kw=dict(projection="3d"),
        )

        ani = Animator(fig, env_data_list, cfg, simple=simple)
        ani.animate()
        ani.save(Path(dir_save, "compare-animation.mp4"))
        plt.close('all')

    return_list = []
    for i, data in enumerate(agent_data_list):
        G = [0]*cfg.quad.num
        for r in data['reward'][::-1]:
            for j in range(cfg.quad.num):
                G[j] = r[j].item() + cfg.ddpg.discount*G[j]
            # G = r.item() + cfg.ddpg.discount*G
        return_list_tmp = [(i+1)*cfg.epi_eval]
        for j in range(cfg.quad.num):
            return_list_tmp.append(G[j])
        return_list.append(return_list_tmp)
    return_list = np.array(return_list)

    for i in range(cfg.quad.num):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(return_list[:,0], return_list[:,i+1], "*")
        ax.set_title(f"Return for {i}th quadrotor")
        ax.set_ylabel("Return")
        ax.set_xlabel("Episode")
        ax.grid(True)
        fig.savefig(
            Path(dir_save, f"return_quad_{i}.png"),
            bbox_inches='tight'
        )
        plt.close('all')
