"""
Class containing the main ratSLAM class
"""

# -----------------------------------------------------------------------

import numpy as np

from ratSLAM.experience_map import ExperienceMap
from ratSLAM.odometry import Odometry
from ratSLAM.pose_cells import PoseCells
from ratSLAM.view_cells import ViewCells
from ratSLAM.input import Input
from ratSLAM.utilities import timethis
from utils.logger import root_logger
from utils.misc import getspeed, rotate

# -----------------------------------------------------------------------

X_DIM = 8
Y_DIM = 8
TH_DIM = 36

# -----------------------------------------------------------------------

class RatSLAM(object):
    """
    RatSLAM module.

    Divided into 4 submodules: odometry, view cells, pose
    cells, and experience map.
    """
    def __init__(self, absolute_rot=False):
        """
        Initializes the ratslam modules.
        """
        self.odometry = Odometry()
        self.view_cells = ViewCells()
        self.pose_cells = PoseCells(X_DIM, Y_DIM, TH_DIM)
        self.experience_map = ExperienceMap(X_DIM, Y_DIM, TH_DIM)

        self.absolute_rot = absolute_rot
        self.last_pose = None
        # TRACKING -------------------------------
        #self.odometry = self.odometry.odometry
        #self.active_pc = self.pose_cells.active_cell
        self.prev_trans = []
        self.ma_trans = 1
        self.prev_rot = []
        self.ma_rot = 1

        self.t_s_h = []
        self.t_d_h = []
        self.t_l_h = []
        self.e_s_h = []
        self.e_d_h = []
        self.e_l_h = []
        self.slam_l_h = []

    ###########################################################
    # Public Methods
    ###########################################################

    @timethis
    def step(self, input):
        """
        Performs a step of the RatSLAM algorithm by analysing given input data.
        """
        if not isinstance(input, Input):
            print("ERROR: input is not instance of Input class")
            exit(0)
        x_pc, y_pc, th_pc = self.pose_cells.active_cell
        #print(f"Current pose index is {x_pc}, {y_pc}, {th_pc}")
        # Get activated view cell
        view_cell = self.view_cells.observe_data(input, x_pc, y_pc, th_pc)
        # Get odometry readings
        vtrans, vrot = self.odometry.observe_data(input, absolute_rot=self.absolute_rot)
        # if vtrans < 1:
        #     print(vtrans)
        #     vtrans = 0
        #     if len(self.prev_rot) == 0:
        #         vrot = 0
        #     else:
        #         vrot = self.prev_rot[-1]

        # Perform moving average smoothing
        self.prev_trans = [vtrans] + self.prev_trans
        if len(self.prev_trans) > self.ma_trans:
            self.prev_trans = self.prev_trans[:self.ma_trans]
        self.prev_rot = [vrot] + self.prev_rot
        if len(self.prev_rot) > self.ma_rot:
            self.prev_rot = self.prev_rot[:self.ma_rot]
        vtrans = np.mean(self.prev_trans)
        vrot = np.mean(self.prev_rot)

        #print(f"Translation is {vtrans}, Rotation is {vrot}")
        #if self.last_pose is not None:
            #print(f"Actual Trans is {getspeed(self.last_pose[0], input.raw_data[1][0])}")
        # Update pose cell network, get index of most activated pose cell
        x_pc, y_pc, th_pc = self.pose_cells.step(view_cell, vtrans, vrot)
        # Execute iteration of experience map
        self.experience_map.step(view_cell, vtrans, vrot, x_pc, y_pc, th_pc,
                                 true_pose=(input.raw_data[1][0],input.raw_data[1][2]),
                                 true_odometry=(input.raw_data[1][3], input.raw_data[1][2]))

        self.last_pose = (input.raw_data[1][0],input.raw_data[1][2])

        self.t_s_h.append(input.raw_data[1][3])
        self.t_d_h.append(input.raw_data[1][2])
        self.t_l_h.append(input.raw_data[1][0])
        self.e_s_h.append(vtrans)
        self.e_d_h.append(vrot)
        self.e_l_h.append(input.template)
        self.slam_l_h.append((self.experience_map.current_exp.x_em ,
                             self.experience_map.current_exp.y_em))


    def showError(self):
        """
        Prints out error reading for all SLAM components
        """
        self.angle_hist = self.experience_map.angle_hist

        abs_ang_errors = [min(abs(self.angle_hist[i][0]-self.angle_hist[i][1]), 2*np.pi - abs(self.angle_hist[i][0]-self.angle_hist[i][1])) for i in range(1, len(self.t_d_h))]
        MAE_ang = np.mean(abs_ang_errors)
        abs_spd_errors = [abs(self.t_s_h[i] - self.e_s_h[i]) for i in range(1, len(self.t_s_h))]
        MAE_spd = np.mean(abs_spd_errors)
        abs_loc_errors = [ ((self.e_l_h[i][0]-self.t_l_h[i][0])**2 + (self.e_l_h[i][1]-self.t_l_h[i][1])**2)**0.5 for i in range(1, len(self.t_l_h)) ]
        MAE_loc = np.mean(abs_loc_errors)

        #rotate(self.true_pose[0] - self.initial_pose[0], degrees=self.initial_pose[1])
        roted = [rotate(self.t_l_h[i] - self.experience_map.initial_pose[0], degrees=self.experience_map.initial_pose[1]) for i in range(len(self.t_l_h))]
        abs_slam_loc_error = [ ((self.slam_l_h[i][0]-roted[i][0])**2 + (self.slam_l_h[i][1]-roted[i][1])**2)**0.5 for i in range(1, len(self.t_l_h)) ]
        MAE_slam_loc_error = np.mean(abs_slam_loc_error)

        print(f"MAE Direction:  {MAE_ang}")
        print(f"MAE Speed:  {MAE_spd}")
        print(f"MAE Location:  {MAE_loc}")
        print(f"MAE SLAM Location:  {MAE_slam_loc_error}")