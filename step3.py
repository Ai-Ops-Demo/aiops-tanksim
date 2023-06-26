''' ----------- Ai-Ops, Inc ----------------'''
# ---------------------------------------------------------------------------------------
#
#                A                 ii       OOO
#               A:A               i::i    OO:::OO
#              A:::A               ii   OO:::::::OO
#             A:::::A                  O::::OOO::::O
#            A:::::::A           iiiii O:::O   O:::Oppp   pppp       ssssss
#           A::::A::::A          i:::i O::O     O::Op::pp::::::p    sS:::::s
#          A::::A A::::A          i::i O::O     O::Op:::::::::::p ss::::::::s
#         A::::A   A::::A         i::i O::O     O::Opp:::ppppp:::ps:::ssss:::s
#        A::::A     A::::A        i::i O::O     O::O p::p     p::p s::s  ssss
#       A::::AAAAAAAAA::::A       i::i O::O     O::O p::p     p::p   s:::s
#      A:::::::::::::::::::A      i::i O::O     O::O p::p     p::p      s:::s
#     A::::AAAAAAAAAAAAA::::A     i::i O:::O   O:::O p::p    p:::pssss   s:::s
#    A::::A             A::::A   i::::iO::::OOO::::O p::ppppp::::ps::ssss:::s
#   A::::A               A::::A  i::::i OO:::::::OO  p:::::::::p s:::::::::s
#  A::::A                 A::::A i::::i   OO:::OO    p:::::::pp   s:::::ss
# AAAAAA                   AAAAAAiiiiii     OOO      p:::ppp       sssss
#                                                    p::p
#                                                    p::p
#                                                    p::::p
#                                                    p::::p
#                                                    p::::p
#                                                    pppppp
#
# --------------------------------------------------------------------------------------
# Ai-Ops, Inc. Control Model Builder LI101.AiMV for Lesson 1
# www.ai-op.com
# Copyright (C) 2020-2023, all rights reserved.
# Licensed granted under Apache 2.0, See License File for details
# --------------------------------------------------------------------------------------
from lib_aiopssimmaker import Simulator
from lib_aiopsdqn import (
    ReplayBuffer,
    Agent
    )
from lib_aiopstrainagent import TrainAgentDQN
from lib_aiopsbuildconfig import BuildConfigFile
import os

# prevent tf displaying text everytime it loads a model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# get a list of the training files
data_dir = 'traindata.csv'
val_file = 'valdata.csv'

pvIndex = 0
svIndex = 1
mvIndex = 2

# list relative path to digital twin experiment files
dt = ['./LIC101.AiPV']

# Variables List
# 0 = Level Value		LIC101.PV
# 1 = Level Setpoint	LIC101.SP
# 2 = Level Output		LIC101.MV
# 3 = Flow Input 1		FIC-1.PV
# 4 = Flow Input 2		FIC-2.PV
# 5 = Flow Input 3		FIC-3.PV

# what variables do you want the agent to see?
# should see the PV,SV,and MV at a minimum.
agentIndex = [
    0,
    1,
    2,
    3,
    4,
    5
    ]

# how much history of data for the agent to see
agent_lookback = 3

# about 2-3x as long as the system needs to respond to SP change
episode_length = 500

# add some noise to the SV to help with simulating responses.
svNoise = 0

# ------------------- Reward Paramaters -----------------------
# general = proportional to error signal
# stability - for stability within the response tolerance
# stability_tolerance - for a control move smaller than this.
# resposnse - for reaching the setpoint quickly
# response_tolerance - within this tolerance (tiered scheme)
general = 1
stability = 1
stability_tolerance = 0.0001
response = 1
response_tolerance = 0.10

# --------------------Variables for Agent-----------------------
# controller_name - name of DT file/folder
# gamma - how far into the future should the agent care about
# epsilon_decay - make smaller or larger for more/less episodes
# max_step - how much can the agent move each timestep
# scanrate - scan rate that DT's were trained at.
# velocity - have the output be a velocity model
controller_name = 'LIC101.AiMV'

# make a directory to save stuff in
model_dir = controller_name + '/'
os.mkdir(model_dir)
print(model_dir)

gamma = 0.99
epsilon_decay = 0.99997
max_step = 0.05
scanrate = 1
velocity = False

# ---------------------config file--------------------------------------------
config_file = BuildConfigFile(model_dir,
                              agentIndex, velocity, agent_lookback, scanrate,
                              svIndex, pvIndex, mvIndex, svNoise,
                              epsilon_decay, gamma, episode_length, max_step,
                              general, stability, stability_tolerance, response, response_tolerance)

# #####################################################################
# ------------Initalize Simulator from trained Environment-------------
# #####################################################################
sim = Simulator(
    data_dir=data_dir,
    val_file=val_file,
    agentIndex=agentIndex,
    mvIndex=mvIndex,
    svIndex=svIndex,
    pvIndex=pvIndex,
    agent_lookback=agent_lookback,
    episode_length=episode_length,
    model_dir=model_dir,
    dt=dt,
    svNoise=svNoise,
    stability=stability,
    stability_tolerance=stability_tolerance,
    response=response,
    response_tolerance=response_tolerance,
    general=general,
    scanrate=scanrate
    )

# ######################################################################
# --------------Initalize DQN Agent/ReplayBuffer-----------------------
# ######################################################################
buff = ReplayBuffer(
    agentIndex,
    agent_lookback=sim.agent_lookback,
    capacity=1000000,
    batch_size=64
    )

# Initialize Agent
agent = Agent(
    agentIndex=agentIndex,
    mvIndex=mvIndex,
    agent_lookback=agent_lookback,
    default_agent=False,
    gamma=gamma,
    epsilon_decay=epsilon_decay,
    min_epsilon=0.20,
    max_step=max_step,
    scanrate=scanrate,
    velocity=velocity
    )

# you have the option to explore with a PID controller,
# or comment out for random exploration
agent.PID(100,10,0,pvIndex,svIndex,scanrate=scanrate)

agent.buildQ(
    fc1_dims=128,
    fc2_dims=256,
    lr=.002
    )
agent.buildPolicy(
    fc1_dims=64,
    fc2_dims=64,
    lr=.002
    )

# #############################___DQN___#######################################
# ---------------------For Eposode 1, M do------------------------------------
# #############################################################################
trained_agent = TrainAgentDQN(agent, buff, sim, controller_name)
