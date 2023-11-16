# STMRL
A spatial-temporal multi-agent reinforcement learning framework (STMRL) to perform distributed decision-making in multi-edge empowered computation offloading systems
Code and Data for model STMRL 

Requirements:  
python 3.7  
tensorflow 2.4  
stellargraph 1.2.1  

We simulate four scenarios with "Simulation of Urban MObility" (SUMO) (https://sumo.dlr.de/docs/index.html) including GridNet3x3, Multilanes, Bologna-Pasubio, and Bologna-Acosta.  
The simulated datasets are available at https://drive.google.com/file/d/1RSx0zZnG8KestQ3EHL5fv9aPoSadD9dx/view?usp=sharing  
You should download them yourself and put them to build the directory: \sumo\data\xxxx (xxxx is the scenario name)

# File Structure
The simulation data and analysis code are provided under the directory: \sumo\
The spatial-temporal load prediction module and pre-trained models are provided under the directory: \spatiotemporal_prediction\
The code for running both STMRL and other baselines is under the directory: \MARL_vehicle\  

# Model Training
You can train each model in \MARL_vehicle\  with the command: python run_xxxx.py

# Model Test
For static strategies, you can test them in \MARL_vehicle\ with python run_xxxx.py  
For reinforcement learning methods, you can test them with python test_xxxx.py
