import akro
import math
import numpy as np
import os
from .utils.mygym import convert_to_gym
import gym
import opensim
import random
from .osim import OsimEnv
from gym import utils

class Arm2DEnv(OsimEnv):
    model_path = os.path.join(os.path.dirname(__file__), '../models/arm2dof6musc.osim')    
    time_limit = 100
    target_x = 0
    target_y = 0
    targCond = 1
    def get_observation(self):
        state_desc = self.get_state_desc()

        res = [self.target_x, self.target_y]

        # for body_part in ["r_humerus", "r_ulna_radius_hand"]:
        #     res += state_desc["body_pos"][body_part][0:2]
        #     res += state_desc["body_vel"][body_part][0:2]
        #     res += state_desc["body_acc"][body_part][0:2]
        #     res += state_desc["body_pos_rot"][body_part][2:]
        #     res += state_desc["body_vel_rot"][body_part][2:]
        #     res += state_desc["body_acc_rot"][body_part][2:]

        for joint in ["r_shoulder","r_elbow",]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]

        for muscle in sorted(state_desc["muscles"].keys()):
            res += [state_desc["muscles"][muscle]["activation"]]
            # res += [state_desc["muscles"][muscle]["fiber_length"]]
            # res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        res += state_desc["markers"]["r_radius_styloid"]["pos"][:2]

        return res
    
    # @property
    # def observation_space(self):
    #     return akro.Box(low = -math.pi*100, high = math.pi*100, shape = (self.osim_model.get_observation_space_size(), ) )

    # @property
    # def action_space(self):
    #     return akro.Box(low = 0, high = 1, shape = (self.osim_model.get_action_space_size(), ) )

    
    def get_observation_space_size(self):
        return 16 #46


    def generate_new_target(self):
        if self.targCond == 1:
            theta = math.pi/2
            radius = .6
        elif self.targCond ==2:
            thetas = (math.pi/2, math.pi/4)
            radii = (.4, .2)
            rand1 = random.randint(0, 1)
            theta = thetas[rand1]
            radius = radii[rand1]
        elif self.targCond == 3:
            thetas = (math.pi/2, math.pi/4, 0)
            radii = (.4, .2, .3)
            rand1 = random.randint(0, 2)
            theta = thetas[rand1]
            radius = radii[rand1]
        else:
            theta = random.uniform(math.pi*0, math.pi*2/3)
            radius = random.uniform(0.3, 0.65)

        self.target_x = math.cos(theta) * radius 
        self.target_y = -math.sin(theta) * radius + 0.8
        print('\ntarget: [{} {}]'.format(self.target_x, self.target_y))

        state = self.osim_model.get_state()

#        self.target_joint.getCoordinate(0).setValue(state, self.target_x, False)
        self.target_joint.getCoordinate(1).setValue(state, self.target_x, False)

        self.target_joint.getCoordinate(2).setLocked(state, False)
        self.target_joint.getCoordinate(2).setValue(state, self.target_y, False)
        self.target_joint.getCoordinate(2).setLocked(state, True)
        self.osim_model.set_state(state)
        
    def reset(self, random_target=True, obs_as_dict=True):
        obs = super(Arm2DEnv, self).reset(obs_as_dict=obs_as_dict)
        if random_target:
            self.generate_new_target()
        self.osim_model.reset_manager()
        return obs

    def __init__(self, *args, **kwargs):
        super(Arm2DEnv, self).__init__(*args, **kwargs)
        blockos = opensim.Body('target', 0.0001 , opensim.Vec3(0), opensim.Inertia(1,1,.0001,0,0,0) );
        self.target_joint = opensim.PlanarJoint('target-joint',
                                  self.osim_model.model.getGround(), # PhysicalFrame
                                  opensim.Vec3(0, 0, 0),
                                  opensim.Vec3(0, 0, 0),
                                  blockos, # PhysicalFrame
                                  opensim.Vec3(0, 0, -0.25),
                                  opensim.Vec3(0, 0, 0))

        self.noutput = self.osim_model.noutput

        geometry = opensim.Ellipsoid(0.02, 0.02, 0.02);
        geometry.setColor(opensim.Green);
        blockos.attachGeometry(geometry)

        self.osim_model.model.addJoint(self.target_joint)
        self.osim_model.model.addBody(blockos)
        
        self.osim_model.model.initSystem()
    
    def reward(self):
        state_desc = self.get_state_desc()
        penalty = (state_desc["markers"]["r_radius_styloid"]["pos"][0] - self.target_x)**2 + (state_desc["markers"]["r_radius_styloid"]["pos"][1] - self.target_y)**2
        # print(state_desc["markers"]["r_radius_styloid"]["pos"])
        # print((self.target_x, self.target_y))
        if np.isnan(penalty):
            penalty = 1
        # if penalty < 0.01:
        #     return 1
        # else:
        #     return 0
        return 1.-penalty

    def get_reward(self):
        return self.reward()


class Arm2DVecEnv(Arm2DEnv):
        
    def reset(self, obs_as_dict=False):
        obs = super(Arm2DVecEnv, self).reset(obs_as_dict=obs_as_dict)
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
        return obs
    def step(self, action, obs_as_dict=False):
        if np.isnan(action).any():
            action = np.nan_to_num(action)
        obs, reward, done, info = super(Arm2DVecEnv, self).step(action, obs_as_dict=obs_as_dict)
        # print(reward)
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
            done = True
            reward -10
        return obs, reward, done, info
    
    def compute_reward(self, achieved_goal, goal, info):
        """Function to compute new reward.
        Args:
            achieved_goal (numpy.ndarray): Achieved goal.
            goal (numpy.ndarray): Original desired goal.
            info (dict): Extra information.
        Returns:
            float: New computed reward.
        """
        del info
        penalty = (achieved_goal[-2] - goal[-2])**2 + (achieved_goal[-1] - goal[-1])**2
        # if penalty < 0.01:
        #     return 1
        # else:
        #     return 0
        return np.sum((achieved_goal[-2:] - goal[-2:])**2)
    
    def __getstate__(self):
        """See `Object.__getstate__.
        Returns:
            dict: The instance’s dictionary to be pickled.
        """
        return dict()

    def __setstate__(self, state):
        """See `Object.__setstate__.
        Args:
            state (dict): Unpickled state of this object.
        """
        self.__init__()