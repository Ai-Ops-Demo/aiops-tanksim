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
# Ai-Ops, Inc. Control Model Validation LI101.AiMV for Lesson 1
# www.ai-op.com
# Copyright (C) 2020-2023, all rights reserved.
# Licensed granted under Apache 2.0, See License File for details
# --------------------------------------------------------------------------------------
from lib_aiopsdqnvalidate import ValidateDQN

data_dir = 'traindata.csv'
val_file = 'valdata.csv'

dt = ['./LIC101.AiPV']

model_dir = './LIC0101.AiMV' + '/'
modelname = 'LIC0101.AiMV_policy.tflite'


# #############################___DQN___#######################################
# ----------------------------- Validate -------------------------------------
# #############################################################################
validate_agent = ValidateDQN(data_dir=data_dir, val_file=val_file,
                             dt=dt, model_dir=model_dir, modelname=modelname,
                             val_length=2000)