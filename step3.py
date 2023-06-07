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
# Copyright (C) 2020-2022, all rights reserved.
# Licensed granted under Apache 2.0, See License File for details
# --------------------------------------------------------------------------------------
from lib_aiopssimmaker import Simulator
from lib_aiopsdqn import (
    ReplayBuffer,
    Agent
    )
import numpy as np
import os
import json

# prevent tf displaying text everytime it loads a model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# get a list of the training files
data_dir = 'traindata.csv'
val_file = 'valdata.csv'

pvIndex = 0
svIndex = 1
mvIndex = 2

# list relative path to digital twin experiment files
dt = ['./LIC101.AiPV8890']

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
episode_length = 20

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
stability_tolerance = 0.003
response = 1
response_tolerance = 0.05

# --------------------Variables for Agent-----------------------
# controller_name - name of DT file/folder
# gamma - how far into the future should the agent care about
# epsilon_decay - make smaller or larger for more/less episodes
# max_step - how much can the agent move each timestep
# scanrate - scan rate that DT's were trained at.
# velocity - have the output be a velocity model
controller_name = 'LIC0101.AiMV'
gamma = 0.95
epsilon_decay = 0.99997
max_step = 0.1
scanrate = 5
velocity = False

# ---------------------config file--------------------------------------------
# make a directory to save stuff in
model_dir = controller_name + str(int(round(np.random.rand()*10000, 0))) + '/'
os.mkdir(model_dir)
print(model_dir)

# create a config file
config = {}
config['variables'] = {
    'dependantVar': mvIndex,
    'independantVars': agentIndex,
    'velocity': velocity,
    'targetmean': None,
    'targetstd': None,
    'agent_lookback': agent_lookback,
    'scanrate': scanrate,
    'svIndex': svIndex,
    'pvIndex': pvIndex,
 }

config['epsilon_decay'] = epsilon_decay
config['gamma'] = gamma
config['svNoise'] = svNoise
config['episode_length'] = episode_length
config['max_step'] = max_step
config['rewards'] = {
    'general': general,
    'stability': stability,
    'stability_tolerance': stability_tolerance,
    'response': response,
    'response_tolerance': response_tolerance
    }

with open(model_dir + 'config.json', 'w') as outfile:
    json.dump(config, outfile, indent=4)

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
    min_epsilon=.980,
    max_step=max_step,
    scanrate=scanrate,
    velocity=velocity
    )

# you have the option to explore with a PID controller,
# or comment out for random exploration
# agent.PID(100,10,0,pvIndex,svIndex,scanrate=scanrate)

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
scores = []
moving_average = []
episode = 0

while agent.epsilon > agent.min_epsilon:
    # reset score
    score = 0

    # reset simulator
    state, done = sim.reset()

    # initalize the first control position to be the
    # same as the start of the episode data
    control = state[
     sim.agent_lookback-1,
     sim.mvpos
     ]

    # execute the episode
    while not done:

        # select action
        action = agent.selectAction(state)

        # change controller with max action
        control += agent.action_dict[action]

        # keep controller from going past what the env has seen in training
        control = np.clip(
            control,
            sim.mv_min,
            sim.mv_max
            )

        # advance Environment with max action and get state_
        state_, reward, done = sim.step(control)

        # Store Transition
        buff.storeTransition(
            state,
            action,
            reward,state_
            )

        # Sample Mini Batch of Transitions
        sample_s, sample_a, sample_r, sample_s_ = buff.sampleMiniBatch()

        # fit Q
        agent.qlearn(
            sample_s,
            sample_a,
            sample_r,
            sample_s_
            )

        # fit policy
        agent.policyLearn(sample_s)

        # advance state
        state = state_

        # get the score for the episode
        score += reward

    # save a history of episode scores
    scores = np.append(scores, score)

    if episode > 20:
        avg = np.mean(scores[episode-20:episode])
        if avg > max(moving_average):
            print('saved best model')
            agent.saveQ(model_dir+controller_name)

            if agent.epsilon < 0.5:
                agent.savePolicy(model_dir+controller_name)
                agent.tflitePolicy(model_dir+controller_name)

        moving_average.append(avg)

    else:
        moving_average.append(0)

    # save replay buffer every 10 episodes
    if episode % 10 == 0:
        buff.saveEpisodes(model_dir+'replayBuffer.csv')

    # print training episodes
    print('episode_',
          episode,
          'score_',
          round(score, 0),
          ' average_',
          round(moving_average[episode], 1),
          ' epsilon_', round(agent.epsilon, 3))

    episode += 1

# Update config with max and min
with open(model_dir + 'config.json',
          'r') as config_file:
    config = json.load(config_file)

buff_min, buff_max = buff.get_min_max()

min_training_range = {}
for i in buff_min.index:
    min_training_range[i] = str(buff_min[i])
config['min_training_range'] = min_training_range

max_training_range = {}
for i in buff_max.index:
    max_training_range[i] = str(buff_max[i])
config['max_training_range'] = max_training_range

with open(model_dir + 'config.json',
          'w') as outfile:
    json.dump(
        config,
        outfile,
        indent=4
        )
