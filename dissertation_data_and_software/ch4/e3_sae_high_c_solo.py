
#Example run file for the experiment 3 simulations
import Comms_framework as Comms_framework
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
import time

#mixed run weights; zero the last 4 entries for uncoupled
# run_weights = np.array([2.20e-01, 7.71e+01, 1.21e+00, 1.33e+00,
#  1.42e+00, 5.46e-06, 3.61e+02, 5.44e-01,
#  1.79e+00, 4.38e-01,1.62e-01])

#coupled run weights
run_weights = np.array([2.20e-01, 7.71e+01, 1.21e+00, 1.33e+00,
 1.42e+00, 5.46e-06, 3.61e+02, 5.44e-00,
 1.79e+01, 4.38e-00,1.62e-00])


params = {}
params['update_interval'] = 20
params['compiler_iterations'] = 1
params['compiler_interval'] = 10
params['max_iterations'] = 1000 #2000 for medium, 6000 for low decomposition when normalizing via iterations
params['meeting_length'] = 50
params['reward_scale'] = 1


params['subteam_size'] = 1 #2 for medium, 6 for low decomposition when normalizing via subteams

#none of these params are changed for the exp3 study
params['joint_iteration'] = 0
params['partial_accept'] = 0
params['use_subsets'] = 0
params['sync_merges'] = 1
params['subset_size'] = 0
params['nominal_PB'] = 0
params['VC_agents'] = 6


exploration_phase_fraction = 0.01 
max_designs = [1,2,3,4,5,7,9,11] #changes the max designs for the simulator

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
            

    params['nominal_PB'] = 0
    params['compiler_starting_samples'] = max_designs[i]*3*params['subteam_size']
        
    params['subset_size'] = max_designs[i]
            
    for k in range(num_tries):

            
        test_framework = Comms_framework.comms_framework(params,'SAE_high_decomp') #'SAE_low_decomp' for low decomposition, 'SAE' for medium decomposition
        test_framework.problem.weights = run_weights
        test_framework.best_solution_value = 100000
        
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
        
        np.save('e3_sae_high_c_solo',obj_record2) #rename as needed




