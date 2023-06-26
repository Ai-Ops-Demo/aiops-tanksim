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
# Ai-Ops, Inc. Digital Twin Builder LI101.AiPV for Lesson 1
# www.ai-op.com
# Copyright (C) 2020-2023, all rights reserved.
# Licensed granted under Apache 2.0, See License File for details
# --------------------------------------------------------------------------------------
from lib_aiopsdigitaltwin import DigitalTwin
from lib_aiopsbumpdt import BumpTest


data_dir = 'traindata.csv'
val_file = 'valdata.csv'
dt_modelname = 'LIC101.AiPV'

# Variables List
# 0 = Level Value		LIC101.PV
# 1 = Level Setpoint	LIC101.SP
# 2 = Level Output		LIC101.MV
# 3 = Flow Input 1		FIC-1.PV
# 4 = Flow Input 2		FIC-2.PV
# 5 = Flow Input 3		FIC-3.PV
dependantVar = 0
independantVars = [2, 3, 4, 5]
dt_lookback = 3
scanrate = 1

dt = DigitalTwin(
    data_dir=data_dir,
    val_file=val_file,
    dt_modelname=dt_modelname,
    dependantVar=dependantVar,
    independantVars=independantVars,
    dt_lookback=dt_lookback,
    scanrate=scanrate,
    velocity=True,
    isGRU=True
    )
dt.trainDt(
    gru1_dims=32,
    gru2_dims=32,
    lr=.001,
    ep=10,
    batch_size=10000)
dt.validate(
    max_len=500,
    num_val=6
    )

bump_dt = BumpTest(dt_dir=dt_modelname, max_len=1000)