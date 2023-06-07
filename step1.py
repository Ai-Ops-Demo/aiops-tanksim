''' ----------- Ai-Ops, Inc ----------------'''
# ---------------------------------------------------------------------------------------
#
#                A                 ii       OOO
#               A:A               i::i    OO:::OO
#              A:::A               ii   OO:::::::OO
#             A:::::A                  O::::OOO::::O
#            A:::::::A           iiiii O:::O   O:::Oppp   pppp       ssssss
#           A::::A::::A          i:::i O::O     O::Op::pp::::::p    sS:::::s
#          A::::A A::::A          i::i O::O     O::Op:::::::::::p ss::::::::spython
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
# Ai-Ops, Inc. Data Generator for Lesson 1 - Tank Simulator
# www.ai-op.com
# Copyright (C) 2020-2022, all rights reserved.
# Licensed granted under Apache 2.0, See License File for details
# --------------------------------------------------------------------------------------
from lib_aiopsdatagen import (
    Generate_Training_Data,
    Generate_Validation_Data
    )

if __name__ == '__main__':
    td_result = Generate_Training_Data()
    print(f'Training Data Generation Results: {td_result}')
    vd_result = Generate_Validation_Data()
    print(f'Validation DAta Generation Results: {vd_result}')
