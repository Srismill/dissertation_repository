
#This file can be used for both high collaboration and low collaboration exp 2 simulations
#nominal subteam simulations are run in a separate file.
import Comms_framework as Comms_framework
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
import time


params = {}
params['update_interval'] = 20
params['compiler_iterations'] = 1
params['compiler_interval'] = 10
params['max_iterations'] = 1000
params['meeting_length'] = 50
params['reward_scale'] = 1


params['subteam_size'] = 3

params['joint_iteration'] = 1 #1 for high collaboration, 0 for others
params['partial_accept'] = 1 #1 for high collaboration, 0 for others
params['use_subsets'] = 0 #1 for nominal subteams, 0 for others
params['sync_merges'] = 1 #always 1
params['subset_size'] = 0 #set by max_designs for nominal files
params['nominal_PB'] = 0 #set by max_designs for nominal files

exploration_phase_fraction = 0.01 
max_designs = [1,3,6,9,12] #changes the max designs for the simulator

num_tries = 20
num_iterations = params['max_iterations']
#iteration_record = np.zeros((len(exploration_phase_fraction),len(num_iterations),num_tries))
obj_record2 = np.zeros((num_iterations,len(max_designs),num_tries))

for i in range(len(max_designs)):
    exploration_phase_iterations = math.ceil(num_iterations*exploration_phase_fraction)

    params['max_iterations'] = 10000
    max_iterations = num_iterations
    
    params['max_designs'] = max_designs[i]
    params['compiler_iterations'] = 1

    params['compiler_starting_samples'] = max_designs[i]*3
            
    for k in range(num_tries):

            
        test_framework = Comms_framework.comms_framework(params)
        test_framework.problem.weights[-1] *= 100 #multiplier for pitch moment weight; 1 for uncoupled, 100 for mixed, 1000 for coupled
        test_framework.best_solution_value = 10000;
        
        action = np.zeros(len(test_framework.action_space.high))
        action[0] = 1
        
        print(params['max_designs'])
        if params['max_designs'] == 1:
            
            test_framework.switch_to_integration()
        t_start = time.time()
        
        for j in range(max_iterations):
            
            if j == exploration_phase_iterations:
                test_framework.switch_to_integration()
                
            old_obj = test_framework.best_solution_value

            test_framework.step(action)

            new_obj = test_framework.best_solution_value
            obj_record2[j,i,k] = new_obj
        print(time.time()-t_start)

r_2 = obj_record2
np.save('high_com_PA_c100_short',obj_record2)

