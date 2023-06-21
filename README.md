# Lesson 1 | aiops-tanksim
Ai-Ops, Curriculum Lesson 1, Tank Simulator Code Base
- [Go to Ai-Ops Lesson 1 Users Manual](https://ai-ops-inc.gitbook.io/ai-ops-tank-sim-users-guide/)

## Equipment Needed
- Windows OS (Windows 10)
- Linux OS
- Functional CUDA drivers for Nvidia (if using GPU(s))
  
## Project Objectives
- Acquire training data and validation data
  - This will be completed using the data generation script in **Step 1**, and will produce for you two (2) files.
    - traindata.csv
    - valdata.csv
  - It is important to understand, that your site and/or process will likely already possess all the data you need. In practice, manufacturing your process data will not be required, instead you will use your site's exisitng historical data.
- Build a Digital Twin **(AiPV)** of Level Process Value **(PV)** for Tank T-1 using Training Data, and then Validate with Validation Data
- Build a DQN Control Model **(AiMV)** for controlling LIC-101

## Tag and Extension Numbering Schemes
Each Control System controller has three (3) extensions, which are listed below.  Each tag, and its extension are referenced with an ID that corresponds to its data's column number in the traindata.csv file.
- PV: Process Variable (Input)
- SV: Set Point Variable
- MV: Manipulated Variable (Output)

![image](https://user-images.githubusercontent.com/84361913/196509969-769c6c33-8a7b-48b0-a4b8-87d3fbdaea4a.png)

## You Host Machine Requirements for Lesson #1
- Ensure that the python version is 3.9
- All examples shown in this ReadMe are using VSCode as the IDE
  - It is recommended that you use VSCode to make this exercise flow easier for you.
  - You can learn how to setup VSCode for Python at the following link: https://code.visualstudio.com/docs/languages/python
- Fresh build of a virtual venv
  - PIP Install requirements.txt
  - Based on Linux/Windows OS, choose the proper Ai-Ops Wheel to PIP install into venv.
    - Linux x86_64
    - Windows AMD64

## The User's Manual - Lesson 1
- [Go to Ai-Ops Lesson 1 Users Manual](https://ai-ops-inc.gitbook.io/ai-ops-tank-sim-users-guide/)
