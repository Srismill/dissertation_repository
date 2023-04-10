#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import needed packages

import tensorflow as tf
import numpy as np 
import design_specialist
import comms_problem
# from tf_agents.environments import py_environment
# from tf_agents.environments import tf_environment
# from tf_agents.environments import tf_py_environment
# from tf_agents.environments import utils
# from tf_agents.specs import array_spec
# from tf_agents.environments import wrappers
# from tf_agents.environments import suite_gym
# from tf_agents.trajectories import time_step as ts
import random
import gym
import copy
import math
import time
#GOOD RESOURCE: https://github.com/tensorflow/agents/issues/27

#TODO: on refactor - find out a way to handle 'lower is better/higher is better' 
#properties receiving target constraints (or at least lower AND upper bounded constraints) (make EVERYTHING use bounds?)

#TODO: Initialize histories with initial design as well, so we don't try to merge with an empty history (that makes... issues)
#TODO: EXTREMELY fancy move - use shared references to link goals and needs together, so that as one changes the goals can update and adapt.
#There might be some use to it, some day.

#TODO: Figure out how to handle the PBCE nominal teams (do we want to collapse onto the merged design every merge?)
#also requires handling a set size of 1, with subsets. (maybe just make this a special case?)
class comms_framework(gym.Env):
    
    def __init__(self,params,problem = 'SAE'):
        super(comms_framework,self).__init__()
        
        #initial setup of framework
        self.num_comms_actions = 0;
        self.update_interval = params['update_interval'];
        self.max_designs = params['max_designs'];
        self.current_max_designs = self.max_designs;
        self.max_compiler_iterations = params['compiler_iterations'];
        self.compiler_starting_samples = params['compiler_starting_samples'];
        self.spec_compiler_interval = params['compiler_interval'];
        self.max_iterations = params['max_iterations'];
        self.meeting_length = params['meeting_length']
        self.reward_scale = params['reward_scale'];
        
        
        #Experiment 2 inputs
        self.subteam_size = params['subteam_size']
        self.use_joint_iteration = params['joint_iteration']
        self.partial_accept = params['partial_accept']
        self.use_subsets = params['use_subsets']
        self.subset_size = params['subset_size']
        self.sync_merges = params['sync_merges']
        self.current_subset_size = self.subset_size
        self.nominal_PB = params['nominal_PB']
        
        if self.use_subsets == True:
            self.max_designs = self.subset_size*self.subteam_size
            self.current_max_designs = self.max_designs
        if self.nominal_PB == True:
            self.max_designs = 1
            self.current_max_designs = 1
        if self.use_joint_iteration == True:
            self.compiler_starting_samples *= self.subteam_size

        
        self.target_tol = 0.05
        
        #pareto optimization terms
        self.dominance_coef = 30
        self.proximity_coef = 2
        self.proximity_eps = 0.5
        self.proximity_coef_lin = 2
        self.proximity_coef_target = 2
        
        self.goal_penalty = 500
        self.goal_scale = 1.5
        #self.goal_threshold = 30
        self.constraint_penalty = 1e6
        self.incompatibility_penalty = 1e6
        self.incompatibility_scale = 1e4
        self.PENALTY_SCORE = 1e20
        
        self.activity_penalty = 1
        self.failed_comms_penalty = 10
        
        
        self.integration_mode = 0
        
        #read in problem
        if problem == 'SAE':
            self.problem = comms_problem.sae_problem()
        elif problem == 'SAE_low_decomp':
            self.problem = comms_problem.sae_problem_low_decomp()
        elif problem == 'SAE_high_decomp':
            self.problem = comms_problem.sae_problem_high_decomp()
        elif problem == 'Convex':
            self.problem = comms_problem.convex_problem(num_agents)
        elif problem == 'S-P':
            self.problem = comms_problem.sine_parabola_problem()
        elif problem == 'variable_coord':
            num_agents = params['VC_agents']
            self.problem = comms_problem.variable_coord_problem(num_agents)
        #self.problem = sae_problem()
        
        self.num_subproblems = self.problem.number_of_subproblems
        learn_rates = self.problem.learn_rates
        actions_per_subproblem = self.problem.actions_per_subproblem + self.num_comms_actions
        self.temps = self.problem.temps
        self.pareto_weights = self.problem.pareto_weights
        self.target_weights = self.problem.target_weights
        

        
        if "learn_rate" in params:
            learn_rates = np.ones_like(learn_rates)*params["learn_rate"]
    
        self.timestep = 0;
        #initialize design properties
        self.active_teams = np.ones(self.num_subproblems) #set based init
        #self.active_teams = np.zeros(self.num_subproblems) #point based init
        
        #indexed [subproblem, design/team]
        self.committed = ([[copy.deepcopy(self.problem.subproblems[i])] for i in range(self.num_subproblems)]) #pack in an extra list for appending new problems
        self.wip = ([[copy.deepcopy(self.problem.subproblems[i])] for i in range(self.num_subproblems)])

        for i in range(self.num_subproblems):
            for j in range(self.max_designs-1): #Keeping it as this format because we might make max designs vary per designer
                self.committed[i].append(copy.deepcopy(self.problem.subproblems[i]))
            for k in range(self.subteam_size-1):
                self.wip[i].append(copy.deepcopy(self.problem.subproblems[i]))
                

        #indexed [subproblem,design]
        self.active_designs = ([np.zeros(self.max_designs) for i in range(self.num_subproblems)])
        self.design_props = ([np.tile(self.problem.design_props[i],(self.max_designs,1)) for i in range(self.num_subproblems)])
        self.design_targets = ([np.tile(self.problem.design_targets[i],(self.max_designs,1)) for i in range(self.num_subproblems)])
        
        for i in range(self.num_subproblems):
            self.active_designs[i][0] = 1
            self.design_props[i][1:] = np.zeros_like(self.design_props[i][1:])
            self.design_targets[i][1:] = np.zeros_like(self.design_targets[i][1:])
            
            if self.use_subsets == True and self.nominal_PB == False:
                for j in range(self.subteam_size):
                    self.active_designs[i][0+j*self.subset_size] = 1
                    self.design_props[i][0+j*self.subset_size] = copy.deepcopy(self.problem.design_props[i])
                    self.design_targets[i][0+j*self.subset_size] = copy.deepcopy(self.problem.design_targets[i])

                
        #indexed [subproblem,agent]
        self.wip_props = ([np.tile(self.problem.design_props[i],(self.subteam_size,1)) for i in range(self.num_subproblems)])
        self.wip_targets = ([np.tile(self.problem.design_targets[i],(self.subteam_size,1)) for i in range(self.num_subproblems)])

        self.is_requested = [np.zeros(self.subteam_size) for i in range(self.num_subproblems)]
        
        self.design_needs = ([[copy.deepcopy(self.problem.subproblem_needs[i])] for i in range(self.num_subproblems)])
        self.design_goals = ([[copy.deepcopy(self.problem.subproblem_goals[i])] for i in range(self.num_subproblems)])
        self.target_needs = ([[copy.deepcopy(self.problem.target_needs[i])] for i in range(self.num_subproblems)])

        self.target_goals = ([[copy.deepcopy(self.problem.target_goals[i])] for i in range(self.num_subproblems)])
        
        #indexes for wip needs/goals: [acting subproblem, acting agent, needed subproblem's design, property of needed subproblem's design]
        self.wip_needs = ([[copy.deepcopy(self.problem.subproblem_needs[i])] for i in range(self.num_subproblems)])
        self.wip_goals = ([[copy.deepcopy(self.problem.subproblem_goals[i])] for i in range(self.num_subproblems)])
        self.wip_target_needs = ([[copy.deepcopy(self.problem.target_needs[i])] for i in range(self.num_subproblems)])
        self.wip_target_goals = ([[copy.deepcopy(self.problem.target_goals[i])] for i in range(self.num_subproblems)])

                            
        for i in range(self.num_subproblems):
            for j in range(self.max_designs-1):
                #indexes for needs/goals: acting agent, acting design, needed agent's design, property of needed agent's design
                self.design_needs[i].append(copy.deepcopy(self.problem.subproblem_needs[i]))
                self.design_goals[i].append(copy.deepcopy(self.problem.subproblem_goals[i]))
                self.target_needs[i].append(copy.deepcopy(self.problem.target_needs[i]))
                self.target_goals[i].append(copy.deepcopy(self.problem.target_goals[i]))
            for j in range(self.subteam_size-1):
                self.wip_needs[i].append(copy.deepcopy(self.problem.subproblem_needs[i]))
                self.wip_goals[i].append(copy.deepcopy(self.problem.subproblem_goals[i]))
                self.wip_target_needs[i].append(copy.deepcopy(self.problem.target_needs[i]))
                self.wip_target_goals[i].append(copy.deepcopy(self.problem.target_goals[i]))

                            
        self.previous_solutions = ([np.zeros((self.subteam_size,self.num_subproblems)).astype(int) for i in range(self.num_subproblems)])
        self.old_obj_funcs = ([np.tile(self.local_objective(i,wip_props = self.wip_props[i][0],wip_targets = self.design_targets[i][0],wip_needs=[],wip_target_needs=[]) + self.check_constraints(i,self.wip[i][0]),(self.subteam_size,1)) for i in range(self.num_subproblems)])

        #best solution saving
        

        self.best_solution = np.zeros(self.num_subproblems).astype(int)
        self.best_designs = ([copy.deepcopy(self.committed[i][0]) for i in range(self.num_subproblems)])
        self.best_props = ([copy.deepcopy(self.design_props[i][0]) for i in range(self.num_subproblems)])
        self.best_needs = ([copy.deepcopy(self.design_needs[i][0]) for i in range(self.num_subproblems)])
        self.best_targets = ([copy.deepcopy(self.design_targets[i][0]) for i in range(self.num_subproblems)])
        self.best_target_needs = ([copy.deepcopy(self.target_needs[i][0]) for i in range(self.num_subproblems)])
        solution_props,solution_needs,solution_targets,solution_target_needs = self.get_solution_properties(self.best_solution)
        self.best_solution_value = self.evaluate_solution(solution_props,solution_needs,solution_targets,solution_target_needs)
        if self.best_solution_value >= self.PENALTY_SCORE:
            self.best_solution_validity = 0
        else:
            self.best_solution_validity = 1
        
        self.update_intervals = np.ones(self.num_subproblems)*self.update_interval

        
        #deprecated stuff, rework for subteams (self.moved may be repurposed in joint iteration)
        self.meeting_with = ([[] for i in range(self.num_subproblems)])
        self.moved = np.zeros(self.num_subproblems)
        self.meeting_timer = np.zeros(self.num_subproblems)
        self.is_communicating = np.zeros(self.num_subproblems)
        
        #initialize agents
        
        #indexes are [team,agent]
        
        self.team = ([[design_specialist.design_specialist(actions_per_subproblem[j],learn_rates[j],self.temps[j],self.max_designs) for i in range(self.subteam_size)] for j in range(self.num_subproblems)])
        #precompute active need indices for referencing during observation construction
        self.active_needs = []
        self.ATN_lower = []
        self.ATN_upper = []

        for i in range(self.num_subproblems):
            self.active_needs.append([])
            self.ATN_lower.append([])
            self.ATN_upper.append([])
            for j in range(self.num_subproblems):
                if i != j:
                    if len(self.design_needs[i][0][j]) == 0:
                        self.active_needs[i].append([])
                    else:
                        self.active_needs[i].append(np.nonzero(1-self.design_needs[i][0][j].mask)[0])
                        
                    if len(self.target_needs[i][0][j]) == 0:
                        self.ATN_lower[i].append([])
                        self.ATN_upper[i].append([])
                    else:
                        idx1 = np.nonzero(1-self.target_needs[i][0][j].mask)[0]
                        idx2 = np.nonzero(1-self.target_needs[i][0][j].mask)[1]
                        self.ATN_lower[i].append([])
                        self.ATN_upper[i].append([])
                        for k in range(len(idx2)):
                            if idx2[k] == 0:
                                self.ATN_lower[i][j].append(idx1[k])
                            else:
                                self.ATN_upper[i][j].append(idx1[k])
                else:
                    self.active_needs[i].append([])
                    self.ATN_lower[i].append([])
                    self.ATN_upper[i].append([])
                    
        #initialize action and observation spaces
        highs = np.ones(9+self.num_subproblems)*1e4
        #highs[9] = self.num_subproblems - 1e-2
        #highs[10+3*self.num_subproblems] = self.num_subproblems - 1e-2
        
        lows = -highs
        #lows = np.zeros(9+self.num_subproblems)
        #lows[0:9] = -1e6
        #lows[9:9+self.num_subproblems] = -1
        
        self.action_space = gym.spaces.Box(lows,highs,dtype=np.float16)
        
        num_props = 0
        num_needs = 0
        for i in range(self.num_subproblems):
            num_props += len(self.wip_props[i][0])
            num_props += len(self.wip_targets[i][0])
            for j in range(self.num_subproblems):
                if i != j:
                    if len(self.wip_needs[i][0][j]) != 0:
                        num_needs += np.sum(1-self.wip_needs[i][0][j].mask)
                    if len(self.wip_target_needs[i][0][j]) != 0:
                        num_needs += np.sum(1-self.wip_target_needs[i][0][j].mask)
                        
         #communicating and active agents  design props/needs  active designs        best value + time to deadline + best value being invalid
        high = np.ones(self.num_subproblems*2+(num_props+num_needs+self.num_subproblems)*self.max_designs+4)*np.finfo(np.float32).max
        
        self.observation_space = gym.spaces.Box(-high,high,dtype=np.float16) #States include capabilities of all subdesigns and the global objective function
        
    


    def reset(self):
        
        self.problem = comms_problem.sae_problem()
        self.current_max_designs = self.max_designs;
        
        self.integration_mode = 0
        
        self.num_subproblems = self.problem.number_of_subproblems
        learn_rates = self.problem.learn_rates
        actions_per_subproblem = self.problem.actions_per_subproblem + self.num_comms_actions
        self.temps = self.problem.temps
        self.pareto_weights = self.problem.pareto_weights
        self.target_weights = self.problem.target_weights
    
        self.timestep = 0;
        #initialize design properties
        self.active_teams = np.ones(self.num_subproblems) #set based init
        #self.active_teams = np.zeros(self.num_subproblems) #point based init
        
        self.committed = ([[copy.deepcopy(self.problem.subproblems[i])] for i in range(self.num_subproblems)]) #pack in an extra list for appending new problems
        self.wip = copy.deepcopy(self.problem.subproblems)

        for i in range(self.num_subproblems):
            for j in range(self.max_designs-1): #Keeping it as this format because we might make max designs vary per designer
                self.committed[i].append(copy.deepcopy(self.problem.subproblems[i]))
    

        self.active_designs = ([np.zeros(self.max_designs) for i in range(self.num_subproblems)])
        self.design_props = ([np.tile(self.problem.design_props[i],(self.max_designs,1)) for i in range(self.num_subproblems)])
        self.design_targets = ([np.tile(self.problem.design_targets[i],(self.max_designs,1)) for i in range(self.num_subproblems)])
        
        for i in range(self.num_subproblems):
            self.active_designs[i][0] = 1
            self.design_props[i][1:] = np.zeros_like(self.design_props[i][1:])
            self.design_targets[i][1:] = np.zeros_like(self.design_targets[i][1:])
        self.wip_props = copy.deepcopy(self.problem.design_props)
        self.wip_targets = copy.deepcopy(self.problem.design_targets)
            
        self.is_requested = [np.zeros(self.subteam_size) for i in range(self.num_subproblems)]
        
        self.design_needs = ([[copy.deepcopy(self.problem.subproblem_needs[i])] for i in range(self.num_subproblems)])
        self.design_goals = ([[copy.deepcopy(self.problem.subproblem_goals[i])] for i in range(self.num_subproblems)])
        self.target_needs = ([[copy.deepcopy(self.problem.target_needs[i])] for i in range(self.num_subproblems)])

        self.target_goals = ([[copy.deepcopy(self.problem.target_goals[i])] for i in range(self.num_subproblems)])
        #indexes for wip needs/goals: acting agent, needed agent's design, property of needed agent's design
        self.wip_needs = copy.deepcopy(self.problem.subproblem_needs)
        self.wip_goals = copy.deepcopy(self.problem.subproblem_goals)
        self.wip_target_needs = copy.deepcopy(self.problem.target_needs)
        self.wip_target_goals = copy.deepcopy(self.problem.target_goals)
        for i in range(self.num_subproblems):
            for j in range(self.max_designs-1):
                #indexes for needs/goals: acting agent, acting design, needed agent's design, property of needed agent's design
                self.design_needs[i].append(copy.deepcopy(self.problem.subproblem_needs[i]))
                self.design_goals[i].append(copy.deepcopy(self.problem.subproblem_goals[i]))
                self.target_needs[i].append(copy.deepcopy(self.problem.target_needs[i]))
                self.target_goals[i].append(copy.deepcopy(self.problem.target_goals[i]))

                            
        self.previous_solutions = ([np.zeros(self.num_subproblems,self.subteam_size).astype(int) for i in range(self.num_subproblems)])
        self.old_obj_funcs = ([np.tile(self.local_objective(i,wip_props = self.wip_props[i][0],wip_targets = self.design_targets[i][0],wip_needs=[],wip_target_needs=[]) + self.check_constraints(i,self.wip[i][0]),(self.subteam_size,1)) for i in range(self.num_subproblems)])
        
        #best solution saving
        

        self.best_solution = np.zeros(self.num_subproblems).astype(int)
        self.best_designs = ([copy.deepcopy(self.committed[i][0]) for i in range(self.num_subproblems)])
        self.best_props = ([copy.deepcopy(self.design_props[i][0]) for i in range(self.num_subproblems)])
        self.best_needs = ([copy.deepcopy(self.design_needs[i][0]) for i in range(self.num_subproblems)])
        self.best_targets = ([copy.deepcopy(self.design_targets[i][0]) for i in range(self.num_subproblems)])
        self.best_target_needs = ([copy.deepcopy(self.target_needs[i][0]) for i in range(self.num_subproblems)])
        solution_props,solution_needs,solution_targets,solution_target_needs = self.get_solution_properties(self.best_solution)
        self.best_solution_value = self.evaluate_solution(solution_props,solution_needs,solution_targets,solution_target_needs)
        if self.best_solution_value >= self.PENALTY_SCORE:
            self.best_solution_validity = 0
        else:
            self.best_solution_validity = 1
        
        self.update_intervals = np.ones(self.num_subproblems)*self.update_interval
        self.meeting_with = ([[] for i in range(self.num_subproblems)])
        self.moved = np.zeros(self.num_subproblems)
        self.meeting_timer = np.zeros(self.num_subproblems)
        self.is_communicating = np.zeros(self.num_subproblems)
        
        #initialize agents

        self.team = ([[design_specialist.design_specialist(actions_per_subproblem[i],learn_rates[i],self.temps[i],self.max_designs) for i in range(self.subteam_size)] for j in range(self.num_subproblems)])
          
        return self._next_observation()

    def _next_observation(self):
        #serialize all the designs and their needs
        obs = np.asarray([])
        for i in range(self.num_subproblems):
            obs = np.concatenate([obs,self.active_designs[i]])
            for j in range(self.max_designs):
                needs = []
                for k in range(self.num_subproblems):

                    for l in self.active_needs[i][k]:
                        needs.append(self.design_needs[i][j][k][l])
                    for l in self.ATN_lower[i][k]:
                        needs.append(self.target_needs[i][j][k][l,0])
                    for l in self.ATN_upper[i][k]:
                        needs.append(self.target_needs[i][j][k][l,1])
                        
                design_info = np.concatenate(([self.design_props[i][j],self.design_targets[i][j],needs]))
                obs = np.concatenate([obs,design_info])
        
        obs = np.concatenate([obs,
                              self.is_communicating,
                              self.active_teams,
                              [self.best_solution_value*self.reward_scale],
                              [self.max_iterations-self.timestep],
                              [self.integration_mode],
                              [self.best_solution_validity]])
        return obs
    
    def step(self, action_in):
        
        #"""Apply action and return new time_step."""
        #also return the new situation
        #action = copy.deepcopy(action_in)
        #action.flags.writeable=True
        
        action = [np.argmax(action_in[0:9])]
        agent_choices = copy.deepcopy(action_in[9:9+self.num_subproblems])
        agent_choices.flags.writeable=True
        #Figure out the action parser later once we're working on our RL agent. 
        #action #0 = 'do nothing'
        old_best_solution = copy.deepcopy(self.best_solution_value)
        success = 1
        self.state_dict = {}
        
        if action[0] < 2 and action[0] >= 1: #shrink design space
            self.shrink_design_space()
            
        elif action[0] < 3 and action[0] >= 2: #force an update of an agent's design
            #agent = int(np.floor(action[1]))
            agent = np.argmax(agent_choices)
            success = self.merge_design(agent)
            if success != None:    
                self.new_design(agent)
                
        elif action[0] < 4 and action[0] >= 3: #make a request to have many agents fulfill one
            #a1 = np.argmax(action[2:2+self.num_subproblems])
            a1 = np.argmax(agent_choices)
            agent_choices[a1] = 1e9

            #a2 = np.nonzero(np.where(action[2:2+self.num_subproblems]<-0.9,1,0))[0]
            a2 = np.nonzero(np.where(agent_choices < 0, 1, 0))[0]
            if a2.size == 0:
                #a2 = [np.argmin(action[2:2+self.num_subproblems])]
                a2 = [np.argmin(agent_choices)]
                if a1 == a2:
                    success = None
                else:
                    success = self.demanding_request(a1,a2)
            else:
                success = self.demanding_request(a1,a2)
            
        elif action[0] < 5 and action[0] >= 4: #make a request for one agent to fulfill many
            a1 = np.argmax(agent_choices)
            agent_choices[a1] = 1e9
            a2 = np.nonzero(np.where(agent_choices < 0, 1, 0))[0]
            if a2.size == 0:
                a2 = [np.argmin(agent_choices)]
                if a1 == a2:
                    success = None
                else:
                    success = self.fulfilling_request(a1,a2)
            else:
                success = self.fulfilling_request(a1,a2)

            
        elif action[0] < 6 and action[0] >= 5: #make a meeting
            #meeting_agents = np.nonzero(np.where(action[2+2*self.num_subproblems:2+3*self.num_subproblems]>0.9,1,0))[0]
            meeting_agents = np.nonzero(np.where(agent_choices>0,1,0))[0]
            success = self.create_meeting(meeting_agents)
            
        elif action[0] < 7 and action[0] >= 6: #start or stop an agent
            #agent = int(np.floor(action[2+3*self.num_subproblems]))
            agent = np.argmax(agent_choices)
            self.start_stop(agent)
            
        elif action[0] < 8 and action[0] >= 7:
            self.expand_design_space()
            
        elif action[0] < 9 and action[0] >= 8:
            self.return_to_exploration()
        
        self.state_dict["good_actions"] = np.zeros((self.num_subproblems,self.subteam_size))
        self.state_dict["accepted"] = np.zeros((self.num_subproblems,self.subteam_size))
        self.state_dict["meeting_obj_func"] = self.best_solution_value
        self.state_dict["replaced_designs"] = np.ones((self.num_subproblems,self.subteam_size))*-1
        self.state_dict["obj_func_on_merge"] = np.ones((self.num_subproblems,self.subteam_size))*-1
        self.state_dict["obj_func_on_merge2"] = np.ones((self.num_subproblems,self.subteam_size))*-1
        self.state_dict["obj_funcs"] = np.ones((self.num_subproblems,self.subteam_size))*-1
        self.state_dict["forked_designs"] = np.zeros((self.num_subproblems,self.subteam_size))
        self.state_dict["attempted_actions"] = np.zeros((self.num_subproblems,self.subteam_size))
        self.state_dict["temps"] = np.zeros((self.num_subproblems,self.subteam_size))
        self.state_dict["merged"] = 0
        self.state_dict["stored_temps"] = np.zeros((self.num_subproblems,self.subteam_size))
        
        
        self.moved = np.zeros(self.num_subproblems)
        
        if self.timestep % self.update_intervals[0] == 0 and self.sync_merges == 1:    
            self.merge_all_designs()
            self.state_dict["merged"] = 1
            
        if self.use_joint_iteration == True:
            for i in range(self.num_subproblems):
                self.iterate_team(i)
        else:
            for i in range(self.num_subproblems):
                for j in range(self.subteam_size):
                    if self.active_teams[i] > 0.5:
                        if self.meeting_with[i] == []:

                            self.iterate_agent(i,j)

                            #TODO: update meetings with joint iteration stuff
                        else:
                            self.iterate_meeting(i)

                        


        self.timestep += 1
        
        obs = self._next_observation()
        done = self.timestep == self.max_iterations

        reward = -(self.best_solution_value - old_best_solution) - np.sum(self.active_teams)*self.activity_penalty
            
        if success == None:
            reward = reward - self.failed_comms_penalty
        if reward > self.PENALTY_SCORE/10:
            reward = 0
        
        
        #TODO: Map the subproblems into cleanly viewable variables for the manager, no redundant information!
        
        return obs,reward*self.reward_scale,done, {}
    
    def render(self):
        
        self.state_dict["active_designs"] = self.active_designs
        self.state_dict["published_independent_vars"] = self.committed
        self.state_dict["WIP_independent_vars"] = self.wip
        self.state_dict["Published_dep_vars"] = self.design_props
        self.state_dict["Published_dep_targets"] = self.design_targets
        self.state_dict["Published_needs"] = self.design_needs 
        self.state_dict["Published_goals"] = self.design_goals
        self.state_dict["Published_target_needs"] = self.target_needs 
        self.state_dict["Published_target_goals"] = self.target_goals
        self.state_dict["wip_dep_vars"] = self.wip_props
        self.state_dict["wip_dep_targets"] = self.wip_targets
        self.state_dict["wip_needs"] = self.wip_needs 
        self.state_dict["wip_goals"] = self.wip_goals
        self.state_dict["wip_target_needs"] = self.wip_target_needs 
        self.state_dict["wip_target_goals"] = self.wip_target_goals
        self.state_dict["wip_solutions"] = self.previous_solutions
        self.state_dict["old_obj_funcs"] = self.old_obj_funcs
        self.state_dict["agents"] = self.team
        
        

        return self.timestep,self.best_solution_value,self.integration_mode,self.active_teams,self.state_dict
    
        
    def iterate_agent(self,team_id,agent_id):
        

        if self.timestep % self.update_intervals[team_id] == 0 and self.is_requested[team_id][agent_id] == 0 and self.sync_merges == 0:    
            merged_idx = self.merge_design(team_id,agent_id)
            self.new_design(team_id,agent_id)
            self.state_dict["merged"] = 1

        
        #Iterates on the problem 

        #Agent iterates on the problem, using others' committed subproblems, and its own wip subproblems
        move_id = self.team[team_id][agent_id].select_move()
        
        old_props,old_targets = self.problem.get_props(team_id,self.wip[team_id][agent_id])
        old_obj_func = self.old_obj_funcs[team_id][agent_id]

        new_vars, new_needs, new_target_needs  = self.problem.apply_move(team_id,move_id,self.wip[team_id][agent_id], self.wip_needs[team_id][agent_id],self.wip_target_needs[team_id][agent_id]); 
        
        new_props,new_targets = self.problem.get_props(team_id,new_vars)
        
        used_solution = self.previous_solutions[team_id][agent_id]
        if self.integration_mode == 1:
            penalty = self.check_constraints(team_id,new_vars)
            if penalty > 0:
                new_obj_func = penalty
            else:
                props,needs,targets,target_needs = self.get_solution_properties(used_solution,[team_id],new_props,new_needs,new_targets,new_target_needs)
                goal_penalties = self.check_goals(team_id,new_props,new_needs,self.wip_goals[team_id][agent_id])+self.check_targets(team_id,new_targets,new_target_needs,self.wip_target_goals[team_id][agent_id])
                if props != -1:
                    #evaluate how the design performs with the last solution used as well; useful for stability
                    new_obj_func = self.evaluate_solution(props,needs,targets,target_needs) + goal_penalties
                else:
                    new_obj_func = self.PENALTY_SCORE
                
                #periodically search for a new solution, or if the current one became invalid
                if self.timestep % self.spec_compiler_interval == 0 or new_obj_func == self.PENALTY_SCORE:  
                    new_obj_ns, new_solution = self.design_compiler([team_id],new_props,new_needs,new_targets,new_target_needs)
                    new_obj_ns += goal_penalties

                    if new_obj_ns < new_obj_func:
                        new_obj_func = new_obj_ns
                        used_solution = new_solution
                    
            self.team[team_id][agent_id].active_hist_solution.append(used_solution)
                
        else:
            new_obj_func = self.local_objective(team_id,new_props,new_needs,new_targets,new_target_needs) + self.check_constraints(team_id,new_vars)
        
        accept = self.team[team_id][agent_id].update_learn(new_obj_func - old_obj_func)
        
        self.state_dict["good_actions"][team_id][agent_id] = int(new_obj_func < old_obj_func)
        self.state_dict["attempted_actions"][team_id][agent_id] = move_id
        self.state_dict["temps"][team_id][agent_id] = self.team[team_id][agent_id].active_temp
       # self.state_dict["stored_temps"][team_id][agent_id] = self.team[team_id][agent_id].temp[agent_id]
        self.state_dict["obj_funcs"][team_id][agent_id] = old_obj_func
        
        #update history for future temperature updates 
        self.team[team_id][agent_id].active_hist.append(new_obj_func)
        self.team[team_id][agent_id].active_hist_needs.append(new_needs)
        self.team[team_id][agent_id].active_hist_target_needs.append(new_target_needs)
        self.team[team_id][agent_id].active_hist_design.append(new_vars)

        
        if (len(self.team[team_id][agent_id].active_hist) > self.team[team_id][agent_id].hist_length):
            self.team[team_id][agent_id].active_hist.pop(0)
            self.team[team_id][agent_id].active_hist_needs.pop(0)
            self.team[team_id][agent_id].active_hist_target_needs.pop(0)
            self.team[team_id][agent_id].active_hist_design.pop(0)
            if self.integration_mode == 1:
                self.team[team_id][agent_id].active_hist_solution.pop(0)
        

        self.team[team_id][agent_id].update_temp()
        self.state_dict["accepted"][team_id][agent_id] = accept

        if (accept == 1):              
            self.wip[team_id][agent_id] = new_vars
            self.wip_needs[team_id][agent_id] = new_needs
            self.wip_target_needs[team_id][agent_id] = new_target_needs
            self.team[team_id][agent_id].active_last_move = move_id
            self.wip_props[team_id][agent_id] = new_props
            self.wip_targets[team_id][agent_id] = new_targets
            self.old_obj_funcs[team_id][agent_id] = new_obj_func
            self.previous_solutions[team_id][agent_id] = used_solution
            
            if((self.is_requested[team_id][agent_id] == 1 and self.check_goals(team_id,new_props,self.wip_needs[team_id][agent_id],self.wip_goals[team_id][agent_id]) <= 0) or self.meeting_timer[team_id] >= self.meeting_length):
                self.close_request(team_id)
                self.new_design(team_id)
    
                
                
        return
    
    
    def iterate_team(self,team_id):
        
        #everyone works on the 0th wip design
        agent_id = 0
        
        
        move_id =  np.zeros(self.subteam_size)
        accept =  np.zeros(self.subteam_size)
        
        if self.timestep % self.update_intervals[team_id] == 0 and self.is_requested[team_id][agent_id] == 0 and self.sync_merges == 0:    
            merged_idx = self.merge_design(team_id,agent_id)
            self.new_design(team_id,agent_id)
            self.state_dict["merged"] = 1

        old_obj_func = self.old_obj_funcs[team_id][agent_id]
        used_solution = self.previous_solutions[team_id][agent_id]
        
        if self.integration_mode == 1 and self.timestep % self.spec_compiler_interval == 0:
            #periodically search for a new solution
            goal_penalties = self.check_goals(team_id,self.wip_props[team_id][agent_id],self.wip_needs[team_id][agent_id],self.wip_goals[team_id][agent_id])+self.check_targets(team_id,self.wip_targets[team_id][agent_id],self.wip_target_needs[team_id][agent_id],self.wip_target_goals[team_id][agent_id])
            new_obj_ns, new_solution = self.design_compiler([team_id],self.wip_props[team_id][agent_id],self.wip_needs[team_id][agent_id],self.wip_targets[team_id][agent_id],self.wip_target_needs[team_id][agent_id])
            new_obj_ns += goal_penalties

            if new_obj_ns < old_obj_func:
                self.old_obj_funcs[team_id][agent_id] = new_obj_ns
                self.previous_solutions[team_id][agent_id] = new_solution
                

        #Iterates on the problem 

        #Agent iterates on the problem, using others' committed subproblems, and its own wip subproblems
        
        for i in range(self.subteam_size):
            move_id[i] = self.team[team_id][i].select_move()
 
        
        if self.partial_accept == 0:
            new_vars = copy.deepcopy(self.wip[team_id][agent_id])
            new_needs = copy.deepcopy(self.wip_needs[team_id][agent_id])
            new_target_needs = copy.deepcopy(self.wip_target_needs[team_id][agent_id])
            for i in range(self.subteam_size):
                new_vars,new_needs,new_target_needs = self.problem.apply_move(team_id,move_id[i],new_vars,new_needs,new_target_needs)
            
            new_props,new_targets = self.problem.get_props(team_id,new_vars)
            
            if self.integration_mode == 1:
                props,needs,targets,target_needs = self.get_solution_properties(used_solution,[team_id],new_props,new_needs,new_targets,new_target_needs)
                goal_penalties = self.check_goals(team_id,new_props,new_needs,self.wip_goals[team_id][agent_id])+self.check_targets(team_id,new_targets,new_target_needs,self.wip_target_goals[team_id][agent_id])
                if props != -1:
                    #evaluate how the design performs with the last solution used as well; useful for stability
                    new_obj_func = self.evaluate_solution(props,needs,targets,target_needs) + goal_penalties
                else:
                    raise Exception("team iteration hit an invalid solution when it shouldnt!")

                for i in range(self.subteam_size):
                    self.team[team_id][i].active_hist_solution.append(used_solution)
            else:
                new_obj_func = self.local_objective(team_id,new_props,new_needs,new_targets,new_target_needs) + self.check_constraints(team_id,new_vars)

            for i in range(self.subteam_size):
                accept = self.team[team_id][i].update_learn(new_obj_func - old_obj_func)
                        #update history for future temperature updates 
                self.team[team_id][i].active_hist.append(new_obj_func)
                self.team[team_id][i].active_hist_needs.append(new_needs)
                self.team[team_id][i].active_hist_target_needs.append(new_target_needs)
                self.team[team_id][i].active_hist_design.append(new_vars)

                self.team[team_id][i].update_temp()


                if (len(self.team[team_id][i].active_hist) > self.team[team_id][i].hist_length):
                    self.team[team_id][i].active_hist.pop(0)
                    self.team[team_id][i].active_hist_needs.pop(0)
                    self.team[team_id][i].active_hist_target_needs.pop(0)
                    self.team[team_id][i].active_hist_design.pop(0)
                    if self.integration_mode == 1:
                        self.team[team_id][i].active_hist_solution.pop(0)
                    
            if (accept == 1):              
                self.wip[team_id][agent_id] = new_vars
                self.wip_needs[team_id][agent_id] = new_needs
                self.wip_target_needs[team_id][agent_id] = new_target_needs
                self.wip_props[team_id][agent_id] = new_props
                self.wip_targets[team_id][agent_id] = new_targets
                self.old_obj_funcs[team_id][agent_id] = new_obj_func
                self.previous_solutions[team_id][agent_id] = used_solution
                for i in range(self.subteam_size):

                    self.team[team_id][i].active_last_move = int(move_id[i])
        
        else:
            new_vars = [copy.deepcopy(self.wip[team_id][agent_id]) for i in range(2**self.subteam_size)]
            new_props = [copy.deepcopy(self.wip_props[team_id][agent_id]) for i in range(2**self.subteam_size)]
            new_targets = [copy.deepcopy(self.wip_targets[team_id][agent_id]) for i in range(2**self.subteam_size)]
            new_obj_func = np.asarray([copy.deepcopy(self.old_obj_funcs[team_id][agent_id]) for i in range(2**self.subteam_size)])
            new_needs = [copy.deepcopy(self.wip_needs[team_id][agent_id])for i in range(2**self.subteam_size)]
            new_target_needs = [copy.deepcopy(self.wip_target_needs[team_id][agent_id])for i in range(2**self.subteam_size)]
            executed_moves = np.arange(2**self.subteam_size)[:,np.newaxis] >> np.arange(self.subteam_size)[::-1] & 1 #thanks stackoverflow for this one...
            
            

            #search over all action combinations
            for i in range(1,2**self.subteam_size):
                #execute actions
                for j in range(self.subteam_size):
                    if executed_moves[i,j] == 1:
                        new_vars[i],new_needs[i],new_target_needs[i] = self.problem.apply_move(team_id,move_id[j],new_vars[i],new_needs[i],new_target_needs[i])
                    new_props[i],new_targets[i] = self.problem.get_props(team_id,new_vars[i])
                    
                
                if self.integration_mode == 1:
                    props,needs,targets,target_needs = self.get_solution_properties(used_solution,[team_id],new_props[i],new_needs[i],new_targets[i],new_target_needs[i])
                    goal_penalties = self.check_goals(team_id,new_props[i],new_needs[i],self.wip_goals[team_id][agent_id])+self.check_targets(team_id,new_targets[i],new_target_needs[i],self.wip_target_goals[team_id][agent_id])
                    if props != -1:
                        #evaluate how the design performs with the last solution used as well; useful for stability
                        new_obj_func[i] = self.evaluate_solution(props,needs,targets,target_needs) + goal_penalties
                    else:
                        raise Exception("team iteration hit an invalid solution when it shouldnt!")
                    for j in range(self.subteam_size):
                        self.team[team_id][j].active_hist_solution.append(used_solution)
                else:
                    new_obj_func[i] = self.local_objective(team_id,new_props[i],new_needs[i],new_targets[i],new_target_needs[i]) + self.check_constraints(team_id,new_vars[i])

                    
                for j in range(self.subteam_size):
                            #update history for future temperature updates 
                    self.team[team_id][j].active_hist.append(new_obj_func[i])
                    self.team[team_id][j].active_hist_needs.append(new_needs[i])
                    self.team[team_id][j].active_hist_target_needs.append(new_target_needs[i])
                    self.team[team_id][j].active_hist_design.append(new_vars[i])


                    if (len(self.team[team_id][j].active_hist) > self.team[team_id][j].hist_length):
                        self.team[team_id][j].active_hist.pop(0)
                        self.team[team_id][j].active_hist_needs.pop(0)
                        self.team[team_id][j].active_hist_target_needs.pop(0)
                        self.team[team_id][j].active_hist_design.pop(0)
                        if self.integration_mode == 1:
                            self.team[team_id][j].active_hist_solution.pop(0)
                            
            #create differences from base quality
            obj_func_diffs = new_obj_func - new_obj_func[0]
            min_obj = np.amin(obj_func_diffs)
            min_idx = np.where(obj_func_diffs == min_obj)[0][0]

            #greedily take the best solution
            self.wip[team_id][agent_id] = new_vars[min_idx]
            self.wip_needs[team_id][agent_id] = new_needs[min_idx]
            self.wip_target_needs[team_id][agent_id] = new_target_needs[min_idx]
            self.wip_props[team_id][agent_id] = new_props[min_idx]
            self.wip_targets[team_id][agent_id] = new_targets[min_idx]
            self.old_obj_funcs[team_id][agent_id] = new_obj_func[min_idx]
            self.previous_solutions[team_id][agent_id] = used_solution

            #update agents based on whose moves were accepted/rejected
            for i in range(self.subteam_size):
                accepted_move = executed_moves[min_idx,i]
                #update each agent's markov learning based on if their moves were accepted or rejected
                self.team[team_id][i].update_learn(2*accepted_move-1)
                self.team[team_id][i].update_temp()
                if accepted_move == 1:
                    self.team[team_id][i].active_last_move = int(move_id[i])
            accept = executed_moves[min_idx]
     
        #self.state_dict["good_actions"][team_id] = int(new_obj_func < old_obj_func)
        #self.state_dict["attempted_actions"][team_id] = move_id
                    
        return move_id, accept
               
    #Deprecated for now - not used in study 2/3
    def shrink_design_space(self):
        self.state_dict["culled_designs"] = np.zeros(self.num_subproblems)
        
        if self.integration_mode == 0:
            for i in range(self.num_subproblems):
                for k in range(self.subteam_size):
                    self.team[i][k].reset_temps()
                    self.team[i][k].reset_histories()
                #We also wipe all saved goals in integration mode;
                #preserving integrative capability is done by integrating evaluation
                for j in range(self.max_designs):
                    self.design_goals[i][j] = copy.deepcopy(self.problem.needs_bases)
                    self.target_goals[i][j] = copy.deepcopy(self.problem.target_need_bases)

        self.integration_mode = 1 #'throw the switch' and begin integrating
        
        temp_max_designs = int(self.current_max_designs-1)
        if temp_max_designs == 0:
            temp_max_designs = 1
            
        worst_objs = np.zeros(self.num_subproblems)
        design_culled = 0
        for i in range(self.num_subproblems): #cull down to the new max if needed
            while np.sum(self.active_designs[i]) > temp_max_designs:
                max_idx, max_obj = self.cull_design(i)
                design_culled = 1
            if design_culled == 1:
                worst_objs[i] = max_obj
                
        if design_culled == 1:
            designs_removed = 1
            while (designs_removed > 0 ):
                designs_removed = 0
                for i in range(self.num_subproblems):
                    designs_removed += self.remove_killed_designs(i,worst_objs[i])

                
        self.current_max_designs = temp_max_designs
                
        #also update wip designs with the new design space
        for i in range(self.num_subproblems):
            for k in range(self.subteam_size):
                obj, solution = self.design_compiler([i],self.wip_props[i][k],self.wip_needs[i][k],self.wip_targets[i][k],self.wip_target_needs[i][k])
                self.old_obj_funcs[i][k] = obj + self.check_constraints(i,self.wip[i][k])
                self.previous_solutions[i][k] = solution

    def switch_to_integration(self):
        
        if self.integration_mode == 0:
            for i in range(self.num_subproblems):
                for k in range(self.subteam_size):
                    self.team[i][k].reset_temps()
                    self.team[i][k].reset_histories()
                #We also wipe all saved goals in integration mode;
                #preserving integrative capability is done by global objective now
                for j in range(self.max_designs):
                    self.design_goals[i][j] = copy.deepcopy(self.problem.needs_bases)
                    self.target_goals[i][j] = copy.deepcopy(self.problem.target_need_bases)

        self.integration_mode = 1 #'throw the switch' and begin integrating
                
        #also update wip designs with the new design space
        for i in range(self.num_subproblems):
            for k in range(self.subteam_size):
                obj, solution = self.design_compiler([i],self.wip_props[i][k],self.wip_needs[i][k],self.wip_targets[i][k],self.wip_target_needs[i][k])
                self.old_obj_funcs[i][k] = obj + self.check_constraints(i,self.wip[i][k])
                self.previous_solutions[i][k] = solution
        
        
    #Deprecated for now - not used in study 2/3
    def expand_design_space(self):
        self.current_max_designs = math.ceil(self.current_max_designs*1.2)
        if self.current_max_designs >= self.max_designs:
            self.current_max_designs = self.max_designs
    #Deprecated for now - not used in study 2/3
    def return_to_exploration(self):
        if self.integration_mode == 1:
            for i in range(self.num_subproblems):
                for j in range(self.subteam_size):
                    self.team[i][k].reset_temps()
                    self.team[i][k].reset_histories()
        self.integration_mode = 0
        self.current_max_designs = self.max_designs
        for i in range(self.num_subproblems):
            for j in range(self.subteam_size):
                obj = self.local_objective(i,self.wip_props[i][j],self.wip_needs[i][j],self.wip_targets[i][j],self.wip_target_needs[i][j])+self.check_constraints(i,self.wip[i][j])
                self.old_obj_funcs[i][j] = obj
            
        
    #DEPRECATED
    #TODO: update this to handle subteam members.
    def create_meeting(self,team_ids):
        
        if len(team_ids) <= 1:
            return
        
        for i in team_ids: #make sure no ones already communicating
            if self.is_communicating[i]:
                return
        
        for i in team_ids:
            #set statuses to show agents as in a meeting
            self.is_communicating[i] = 1
            self.meeting_with[i] = team_ids
            
            merged_idx = self.merge_design(i)
            

        obj,solution = self.design_compiler()
        if type(solution) is int:
            return
        for i in team_ids:
            self.fork_design(i,solution[i])
            #reset objective functions & histories
            self.old_obj_funcs[i] = obj
            self.previous_solutions[i] = solution
            self.team[i].reset_active_histories()
            self.team[i].reset_active_temps()
              
        return 1
    
    #Deprecated
    #TODO: Update this to handle subteam members (may scrounge code here for joint iteration)
    def iterate_meeting(self,team_id):
        
        
        if self.moved[team_id] == 1:
            return
        
        team_ids = self.meeting_with[team_id]
         
        #Iterates on the problem 

        #Agent iterates on the problem, using others' committed subproblems, and its own wip subproblems
        meeting_size = len(team_ids)
        old_props = ([[] for i in range(meeting_size)])
        old_targets = ([[] for i in range(meeting_size)])
        old_obj_func = ([[] for i in range(meeting_size)])
        move_id =  ([[] for i in range(meeting_size)])
        new_vars = ([[] for i in range(meeting_size)])
        new_props = ([[] for i in range(meeting_size)])
        new_targets = ([[] for i in range(meeting_size)])
        new_needs = ([[] for i in range(meeting_size)])
        new_target_needs = ([[] for i in range(meeting_size)])
        
        
        old_obj_func = self.old_obj_funcs[team_ids[0]]
        self.state_dict["meeting_obj_func"] = old_obj_func
        for i in range(meeting_size):
            team_id = team_ids[i]
            
            self.meeting_timer[team_id] = self.meeting_timer[team_id] + 1
        
            move_id[i] = self.team[team_id][agent_id].select_move()
        
            old_props[i],old_targets[i] = self.problem.get_props(team_id,self.wip[team_id][agent_id])

            new_vars[i], new_needs[i], new_target_needs[i]  = self.problem.apply_move(team_id,move_id[i],self.wip[team_id][agent_id],self.wip_needs[team_id][agent_id],self.wip_target_needs[team_id][agent_id]); #Check this later
            new_props[i],new_targets[i] = self.problem.get_props(team_id,new_vars[i])
        
        used_solution = self.previous_solutions[team_id][agent_id]
        penalty = 0
        for i in range(meeting_size):
            team_id = team_ids[i]
            penalty += self.check_constraints(team_id,new_vars[i])
            
        if penalty > 0:
            new_obj_func = penalty
        else:
            
            props,needs,targets,target_needs = self.get_solution_properties(used_solution,team_ids,new_props,new_needs,new_targets,new_target_needs)

            if props != -1:
                #evaluate how the design performs with the last solution used as well; useful for stability
                new_obj_func = self.evaluate_solution(props,needs,targets,target_needs)
            else:
                new_obj_func = self.PENALTY_SCORE

            if self.timestep % self.spec_compiler_interval == 0 or new_obj_func == self.PENALTY_SCORE:  
                new_obj_ns, new_solution = self.design_compiler(team_ids,new_props,new_needs,new_targets,new_target_needs)

                if new_obj_ns < new_obj_func:
                    new_obj_func = new_obj_ns
                    used_solution = new_solution
                    
        for i in range(meeting_size):
            team_id = team_ids[i]
            self.team[team_id][agent_id].active_hist_solution.append(used_solution)
            accept = self.team[team_id][agent_id].update_learn(new_obj_func - old_obj_func)
            
                        #update history for future temperature updates 
            self.team[team_id][agent_id].active_hist.append(new_obj_func)
            self.team[team_id][agent_id].active_hist_needs.append(new_needs[i])
            self.team[team_id][agent_id].active_hist_target_needs.append(new_target_needs[i])
            self.team[team_id][agent_id].active_hist_design.append(new_vars[i])
        
            if (len(self.team[team_id][agent_id].hist) > self.team[team_id][agent_id].hist_length):
                self.team[team_id][agent_id].active_hist.pop(0)
                self.team[team_id][agent_id].active_hist_needs.pop(0)
                self.team[team_id][agent_id].active_hist_target_needs.pop(0)
                self.team[team_id][agent_id].active_hist_design.pop(0)
                if self.integration_mode == 1:
                    self.team[team_id][agent_id].active_hist_solution.pop(0)
            
            self.team[team_id][agent_id].update_temp()


        if (accept == 1):  
            for i in range(meeting_size):
                team_id = team_ids[i]
                
                self.wip[team_id][agent_id] = new_vars[i]
                self.wip_needs[team_id][agent_id] = new_needs[i]
                self.wip_target_needs[team_id][agent_id] = new_target_needs[i]
                self.team[team_id][agent_id].active_last_move = move_id[i]
                self.wip_props[team_id][agent_id] = new_props[i]
                self.wip_targets[team_id][agent_id] = new_targets[i]
                self.old_obj_funcs[team_id][agent_id] = new_obj_func
                self.previous_solutions[team_id][agent_id] = used_solution
        
        team_id = team_ids[0]
        if self.meeting_timer[team_id] >= self.meeting_length:
            self.end_meeting(team_id)
        
        return
        
    #Deprecated
    #TODO: Update this to handle subteam members        
    def end_meeting(self,team_id):
        team_ids = self.meeting_with[team_id]
        
        for i in team_ids:
            #set statuses to show agents as in a meeting
            self.is_communicating[i] = 0
            self.meeting_with[i] = []
            self.meeting_timer[i] = 0
            
            merged_idx = self.merge_design(i, must_merge = True)
            self.new_design(i)

        return
    
    #Deprecated
    #TODO: Update this to handle subteam members 
#agent 1 requires that agent 2(s) fulfill the needs for one of its designs
    def demanding_request(self,agent_1,agent_2s):
        #ensure all requested agents are free
        for i in agent_2s:
            if self.is_communicating[i]:
                return 
        
        active_idxs = np.nonzero(self.active_designs[agent_1])[0]
        comm_obj_funcs = np.asarray([self.local_objective(agent_1,self.design_props[agent_1][i],self.design_needs[agent_1][i],self.design_targets[agent_1][i],self.target_needs[agent_1][i],use_distances = False) for i in active_idxs])
        idx = random.choices(range(len(active_idxs)),np.exp(-(comm_obj_funcs-np.amin(comm_obj_funcs))))[0]
        design_id = active_idxs[idx]
        
        requested_goals = copy.deepcopy(self.problem.needs_bases)
        requested_targets = copy.deepcopy(self.problem.target_need_bases)
        
        working_agents = np.asarray([])
        
        requested_goals[agent_1] = np.ma.asarray(copy.deepcopy(self.design_props[agent_1][design_id]))
        requested_targets[agent_1] = np.ma.stack((np.ma.asarray(copy.deepcopy(self.design_targets[agent_1][design_id]))*(1-self.target_tol),np.ma.asarray(copy.deepcopy(self.design_targets[agent_1][design_id]))*(1+self.target_tol)),axis=-1)
        
        for agent_2 in agent_2s:
            requested_goals[agent_2] = np.ma.asarray(copy.deepcopy(self.design_needs[agent_1][design_id][agent_2]))
            requested_targets[agent_2] = np.ma.asarray(copy.deepcopy(self.target_needs[agent_1][design_id][agent_2]))
        
        #Have all participants take into account each others' needs when developing - 
        #this is needed to converge to the intersection of sets of all participants
        goals_table = [copy.deepcopy(self.problem.needs_bases) for i in range(len(agent_2s))]
        targets_table = [copy.deepcopy(self.problem.target_need_bases) for i in range(len(agent_2s))]

        
        useable_designs = []
        design_obj_funcs = []
        used_idxs = np.zeros(len(agent_2s)).astype(int)
        #find closest designs to the requester and propose them
        temp_idx = list(np.zeros(len(agent_2s)))
        #construct goals tables and check against the proposed agent 1 design
        for i in range(len(agent_2s)):
            
            agent_2 = agent_2s[i]
        
            temp_idx[i] = self.merge_design(agent_2)
            active_idxs = np.nonzero(self.active_designs[agent_2])[0]
            
            
            goal_rank = np.asarray([self.check_goals(agent_2,self.design_props[agent_2][j],self.design_needs[agent_2][j],self.merge_goals(agent_2,requested_goals,self.design_goals[agent_2][j]),apply_penalty = False)+self.check_targets(agent_2,self.design_targets[agent_2][j],self.target_needs[agent_2][j],self.merge_targets(agent_2,requested_targets,self.target_goals[agent_2][j]),apply_penalty = False) for j in active_idxs])
            
            #record which designs were used for later
            useable_designs.append(active_idxs)
            design_obj_funcs.append(comm_obj_funcs)
            proposed_design = active_idxs[np.argmin(goal_rank)]
            used_idxs[i] = proposed_design

            #resulting design must satisfy needs of designs requesting it 
            prop_goal = copy.deepcopy(self.design_needs[agent_2][proposed_design])
            target_goal = copy.deepcopy(self.target_needs[agent_2][proposed_design])
        
            #resulting design cannot have needs beyond the properties of designs requesting it
            need_goal = copy.deepcopy(self.design_props[agent_2][proposed_design])
            target_need_goal = np.ma.stack((self.design_targets[agent_2][proposed_design]*(1-self.target_tol),self.design_targets[agent_2][proposed_design]*(1+self.target_tol)),axis=-1)
            

            goals_table[i] = prop_goal
            goals_table[i][agent_2] = np.ma.asarray(need_goal)
            targets_table[i] = target_goal
            targets_table[i][agent_2] = np.ma.asarray(target_need_goal)

        #find if the proposed designs all work together 
        solution_found = 1
        best_solution_value = 0
        solution_value = 0
        for i in range(len(agent_2s)):
            agent_2 = agent_2s[i]
            proposed_design = used_idxs[i]
            
                    
            #merge the table with the requesting design to get proposed goals
            temp_goals = copy.deepcopy(requested_goals)
            temp_targets = copy.deepcopy(requested_targets)

            for j in range(len(agent_2s)):
                temp_goals = self.merge_goals(agent_2,temp_goals,goals_table[j])
                temp_targets = self.merge_targets(agent_2,temp_targets,targets_table[j], use_error = False)

            #check if anyone can't meet the updated goals; invalid target merging will also result in violations
            violation = self.check_goals(agent_2,self.design_props[agent_2][proposed_design],self.design_needs[agent_2][proposed_design],self.merge_goals(agent_2,temp_goals,self.design_goals[agent_2][proposed_design]),apply_penalty = False)+self.check_targets(agent_2,self.design_targets[agent_2][proposed_design],self.target_needs[agent_2][proposed_design],self.merge_targets(agent_2,temp_targets,self.target_goals[agent_2][proposed_design]),apply_penalty = False)
            
            #if a design doesn't work, we go to problem solving
            if violation > 0:
                solution_found = 0
                solution_value += violation
        
        if solution_found == 0 and self.current_max_designs > 1:
            #make initial population
            solutions = ([self.random_solution(useable_designs) for i in range(self.compiler_starting_samples)])
            solution_found = 0
            best_solution_value = solution_value
            requested_target_table = copy.deepcopy(targets_table)
            requested_goal_table = copy.deepcopy(goals_table)
            for i in range(self.max_compiler_iterations):
                #check if any member of the population has compatible goals
                
                for j in range(len(solutions)):
                    #make table...
                    for k in range(len(agent_2s)):

                        agent_2 = agent_2s[k]
                        proposed_design = solutions[j][k]

                        goals_table[k][agent_1] = np.ma.asarray(self.design_needs[agent_2][proposed_design][agent_1])
                        goals_table[k][agent_2] = np.ma.asarray(self.design_props[agent_2][proposed_design])
                        targets_table[k][agent_1] = np.ma.asarray(self.target_needs[agent_2][proposed_design][agent_1])
                        targets_table[k][agent_2] = np.ma.asarray(np.ma.stack((self.design_targets[agent_2][proposed_design]*(1-self.target_tol),self.design_targets[agent_2][proposed_design]*(1+self.target_tol)),axis=-1))

                    #score the combination
                    violation = 0
                    for k in range(len(agent_2s)):
                        agent_2 = agent_2s[k]

                        #merge and check goals...
                        temp_goals = copy.deepcopy(requested_goals)
                        temp_targets = copy.deepcopy(requested_targets)
                        for l in range(len(agent_2s)):
                            temp_goals = self.merge_goals(agent_2,temp_goals,goals_table[l])
                            temp_targets = self.merge_targets(agent_2,temp_targets,targets_table[l], use_error = False)

                        violation = violation + self.check_goals(agent_2,self.design_props[agent_2][proposed_design],self.design_needs[agent_2][proposed_design],self.merge_goals(agent_2,temp_goals,self.design_goals[agent_2][proposed_design]),apply_penalty = False)+self.check_targets(agent_2,self.design_targets[agent_2][proposed_design],self.target_needs[agent_2][proposed_design],self.merge_targets(agent_2,temp_targets,self.target_goals[agent_2][proposed_design]),apply_penalty = False)
                    #if no violators, we're done!

                    if violation == 0:
                        solution_found = 1
                        used_idxs = solutions[j]
                        requested_target_table = copy.deepcopy(targets_table)
                        requested_goal_table = copy.deepcopy(goals_table)
            
                        break
                    #else check if it beats the best found so far
                    elif violation < best_solution_value:
                        best_solution_value = violation
                        used_idxs = solutions[j]
                        requested_target_table = copy.deepcopy(targets_table)
                        requested_goal_table = copy.deepcopy(goals_table)


                if solution_found == 1:
                    break                     
                #change some subsystems around and see what we get
                for j in range(len(solutions)):
                    options = useable_designs
                    solutions[j],changed_idx = self.change_random_subsystem(solutions[j],options)


        #Everything already satisfies the goals!
        else:
            requested_target_table = copy.deepcopy(targets_table)
            requested_goal_table = copy.deepcopy(goals_table)

        for i in range(len(agent_2s)):
            agent_2 = agent_2s[i]
            proposed_design = used_idxs[i]

            #merge the table with the requesting design to get proposed goals
            final_goals = copy.deepcopy(requested_goals)
            final_targets = copy.deepcopy(requested_targets)

            for j in range(len(agent_2s)):
                final_goals = self.merge_goals(agent_2,final_goals,requested_goal_table[j])
                final_targets = self.merge_targets(agent_2,final_targets,requested_target_table[j])
                #If all processes have failed, call the request a failure and return to work
                #TODO - need better resolution in the problem solving as well
                if final_targets == -1:
                    for j in range(len(agent_2s)):
                        if temp_idx[j] == -1 or temp_idx[j] == None:
                            self.new_design(agent_2)
                        else:
                            self.fork_design(agent_2,temp_idx[i]) #pick back up on original work (from best point)
                    return

                
            #check if the agent already meets the combined goals
            violation = self.check_goals(agent_2,self.design_props[agent_2][proposed_design],self.design_needs[agent_2][proposed_design],self.merge_goals(agent_2,final_goals,self.design_goals[agent_2][proposed_design])) + self.check_targets(agent_2,self.design_targets[agent_2][proposed_design],self.target_needs[agent_2][proposed_design],self.merge_targets(agent_2,final_targets,self.target_goals[agent_2][proposed_design]))
            
            #if the agent already satisfies the requirements, just update their goals!
            if violation == 0:
                test_targets = self.merge_targets(agent_2,final_targets,self.target_goals[agent_2][proposed_design])
                if test_targets == -1:
                    raise Exception("Test targets = -1 when it shouldnt!")
                if self.integration_mode == 0:
                    self.target_goals[agent_2][proposed_design] = test_targets
                    self.design_goals[agent_2][proposed_design] = self.merge_goals(agent_2,final_goals,self.design_goals[agent_2][proposed_design])

                if temp_idx[i] == -1 or temp_idx[i] == None:
                    self.new_design(agent_2)
                else:
                    self.fork_design(agent_2,temp_idx[i]) #pick back up on original work (from best point)
            #if not, send them OFF TO WORK!
            else:
                if violation >= self.PENALTY_SCORE:
                    self.fork_design(agent_2,proposed_design)
                    self.wip_goals[agent_2] = final_goals
                    if final_targets == -1:
                        raise Exception("Final targets = -1 when it shouldnt!")
                    self.wip_target_goals[agent_2] = final_targets

                else:
                    self.fork_design(agent_2,proposed_design,inherit_goals = True)
                    self.wip_goals[agent_2] = self.merge_goals(agent_2,final_goals,self.design_goals[agent_2][proposed_design])
                    temp_targets = self.merge_targets(agent_2,final_targets,self.target_goals[agent_2][proposed_design])
                    if temp_targets == -1:
                        raise Exception("Temp targets = -1 when it shouldnt!")
                    self.wip_target_goals[agent_2] = temp_targets
                    self.is_requested[agent_2] = 1
                    self.is_communicating[agent_2] = 1
                    self.team[agent_2].requested_by = [[agent_1,design_id]]
                    for j in range(len(agent_2s)):
                        agent_22 = agent_2s[j]
                        if agent_2 != agent_22:
                            other_design = used_idxs[j]
                            self.team[agent_2].requested_by.append([agent_22,other_design])
            
        return 1

    #Deprecated
    #TODO: Update this to handle subteam members
    def fulfilling_request(self,agent_1,agent_2s):

        if self.is_communicating[agent_1]:
            return 
        
        requested_goals = copy.deepcopy(self.problem.needs_bases)
        requested_targets = copy.deepcopy(self.problem.target_need_bases)
        goals_table = [copy.deepcopy(self.problem.needs_bases) for i in range(len(agent_2s))]
        targets_table = [copy.deepcopy(self.problem.target_need_bases) for i in range(len(agent_2s))]
        
        useable_designs = []
        design_obj_funcs = []
        used_idxs = np.zeros(len(agent_2s)).astype(int)

        
        #fill the goals tables
        for i in range(len(agent_2s)):
            
            agent_2 = agent_2s[i]
            active_idxs = np.nonzero(self.active_designs[agent_2])[0]
            comm_obj_funcs = np.asarray([self.local_objective(agent_2,self.design_props[agent_2][i],self.design_needs[agent_2][i],self.design_targets[agent_2][i],self.target_needs[agent_2][i],use_distances = False) for i in active_idxs])
             
            idx = random.choices(range(len(active_idxs)),np.exp(-(comm_obj_funcs-np.amin(comm_obj_funcs))))[0]
            design_id = active_idxs[idx]
            
            #record which designs were used for later
            useable_designs.append(active_idxs)
            design_obj_funcs.append(comm_obj_funcs)
            used_idxs[i] = idx

            #resulting design must satisfy needs of designs requesting it 
            prop_goal = self.design_needs[agent_2][design_id][agent_1]
            target_goal = self.target_needs[agent_2][design_id][agent_1]
        
            #resulting design cannot have needs beyond the properties of designs requesting it
            need_goal = self.design_props[agent_2][design_id]
            target_need_goal = np.ma.stack((self.design_targets[agent_2][design_id]*(1-self.target_tol),self.design_targets[agent_2][design_id]*(1+self.target_tol)),axis=-1)

            goals_table[i][agent_1] = np.ma.asarray(prop_goal)
            goals_table[i][agent_2] = np.ma.asarray(need_goal)
            targets_table[i][agent_1] = np.ma.asarray(target_goal)
            targets_table[i][agent_2] = np.ma.asarray(target_need_goal)
            
        #correct the goals tables and turn them into a valid request

        temp_goals = copy.deepcopy(self.problem.needs_bases)
        temp_targets = copy.deepcopy(self.problem.target_need_bases)

        for i in range(len(agent_2s)):
            temp_goals = self.merge_goals(agent_1,temp_goals,goals_table[i])
            temp_targets = self.merge_targets(agent_1,temp_targets,targets_table[i], use_error = False)


        #find problematic bounds clashes, if any
        violating_idxs = []
        for i in range(len(agent_2s)):
            agent_2 = agent_2s[i]
            violation = np.sum(np.where(temp_targets[agent_1][:,0] > targets_table[i][agent_1][:,1],1,0)) #check for merged lb > agent ub
            violation = violation + np.sum(np.where(temp_targets[agent_1][:,1] < targets_table[i][agent_1][:,0],1,0)) #check for merged ub < agent lb
            if violation > 0:
                violating_idxs.append(i)

        #agents part of the problem attempt to unilaterally solve it 
        if len(violating_idxs) > 0:

            valid_solutions = []
            if self.current_max_designs > 1: #skip to bouncing back the request if we're in point based mode
                for i in range(len(violating_idxs)):
                    table_idx = violating_idxs[i]
                    agent_2 = agent_2s[table_idx]

                    #an agent analyzes its options
                    for j in range(len(useable_designs[table_idx])):
                        design_id = useable_designs[table_idx][j]
                        #replace table entry

                        goals_table[table_idx][agent_1] = np.ma.asarray(self.design_needs[agent_2][design_id][agent_1])
                        goals_table[table_idx][agent_2] = np.ma.asarray(self.design_props[agent_2][design_id])
                        targets_table[table_idx][agent_1] = np.ma.asarray(self.target_needs[agent_2][design_id][agent_1])
                        targets_table[table_idx][agent_2] = np.ma.asarray(np.ma.stack((self.design_targets[agent_2][design_id]*(1-self.target_tol),self.design_targets[agent_2][design_id]*(1+self.target_tol)),axis=-1))


                        temp_goals = copy.deepcopy(self.problem.needs_bases)
                        temp_targets = copy.deepcopy(self.problem.target_need_bases)

                        #create updated merged requirements
                        for k in range(len(agent_2s)):
                            temp_goals = self.merge_goals(agent_1,temp_goals,goals_table[k])
                            temp_targets = self.merge_targets(agent_1,temp_targets,targets_table[k], use_error = False)
                        #check updated merged requirements
                        valid_request = 1
                        for k in range(len(agent_2s)):
                            agent_2 = agent_2s[k]
                            violation = np.sum(np.where(temp_targets[agent_1][:,0] > targets_table[k][agent_1][:,1],1,0)) #check for merged lb > agent ub
                            violation = violation + np.sum(np.where(temp_targets[agent_1][:,1] < targets_table[k][agent_1][:,0],1,0)) #check for merged ub < agent lb
                            if violation > 0:
                                valid_request = 0
                                break
                        if valid_request == 1:
                            valid_solution = used_idxs
                            valid_solution[table_idx] = design_id
                            valid_solutions.append(valid_solution)

                    #return to original table entry for next agents
                    design_id = used_idxs[table_idx]
                    goals_table[table_idx][agent_1] = np.ma.asarray(self.design_needs[agent_2][design_id][agent_1])
                    goals_table[table_idx][agent_2] = np.ma.asarray(self.design_props[agent_2][design_id])
                    targets_table[table_idx][agent_1] = np.ma.asarray(self.target_needs[agent_2][design_id][agent_1])
                    targets_table[table_idx][agent_2] = np.ma.asarray(np.ma.stack((self.design_targets[agent_2][design_id]*(1-self.target_tol),self.design_targets[agent_2][design_id]*(1+self.target_tol)),axis=-1))

                #rank and implement the best solution found by unilateral moves
            if len(valid_solutions) > 0:
                solution_qualities = np.zeros(len(valid_solutions))
                for i in range(len(valid_solutions)):
                    solution_quality = 0
                    solution = valid_solutions[i]
                    for j in range(len(solution)):
                        agent_2 = agent_2s[j]
                        design = solution[j]
                        solution_quality += self.local_objective(agent_2,self.design_props[agent_2][design],self.design_needs[agent_2][design],self.design_targets[agent_2][design],self.target_needs[agent_2][design], use_distances = False)
                        
                    solution_qualities[i] = solution_quality
                        
                idx = random.choices(range(len(valid_solutions)),np.exp(-(solution_qualities-np.amin(solution_qualities))))[0]
                chosen_solution = valid_solutions[idx]
                
                #build the final table and targets/goals with the chosen solution
                
                for i in range(len(agent_2s)):
                    agent_2 = agent_2s[i]
                    design_id = chosen_solution[i]

                    goals_table[table_idx][agent_1] = np.ma.asarray(self.design_needs[agent_2][design_id][agent_1])
                    goals_table[table_idx][agent_2] = np.ma.asarray(self.design_props[agent_2][design_id])
                    targets_table[table_idx][agent_1] = np.ma.asarray(self.target_needs[agent_2][design_id][agent_1])
                    targets_table[table_idx][agent_2] = np.ma.asarray(np.ma.stack((self.design_targets[agent_2][design_id]*(1-self.target_tol),self.design_targets[agent_2][design_id]*(1+self.target_tol)),axis=-1))

                temp_goals = copy.deepcopy(self.problem.needs_bases)
                temp_targets = copy.deepcopy(self.problem.target_need_bases)

                for i in range(len(agent_2s)):
                    temp_goals = self.merge_goals(agent_1,temp_goals,goals_table[i])
                    temp_targets = self.merge_targets(agent_1,temp_targets,targets_table[i])
                    if temp_targets == -1:
                        raise Exception("Unilateral solving made a solution but it was wrong!")
                        
                else:
                    requested_targets = temp_targets
                    requested_goals = temp_goals
                    used_idxs = chosen_solution


            else: #Need joint moves to resolve conflict

                #call upon a process similar to the project solution searcher to represent a team's problem solving
                #make initial population
                solutions = ([self.random_solution(useable_designs) for i in range(self.compiler_starting_samples)])
                
                solution_found = 0
                if self.current_max_designs > 1: #Continue skipping right to bouncing back the request for point based models
                    for i in range(self.max_compiler_iterations):
                        #check if any member of the population has compatible goals
                        solution_non_violators = []

                        for j in range(len(solutions)):
                            #make table...
                            for k in range(len(agent_2s)):

                                agent_2 = agent_2s[k]
                                design_id = solutions[j][k]

                                goals_table[k][agent_1] = np.ma.asarray(self.design_needs[agent_2][design_id][agent_1])
                                goals_table[k][agent_2] = np.ma.asarray(self.design_props[agent_2][design_id])
                                targets_table[k][agent_1] = np.ma.asarray(self.target_needs[agent_2][design_id][agent_1])
                                targets_table[k][agent_2] = np.ma.asarray(np.ma.stack((self.design_targets[agent_2][design_id]*(1-self.target_tol),self.design_targets[agent_2][design_id]*(1+self.target_tol)),axis=-1))

                            #merge and check goals...
                            temp_goals = copy.deepcopy(self.problem.needs_bases)
                            temp_targets = copy.deepcopy(self.problem.target_need_bases)

                            for k in range(len(agent_2s)):
                                temp_goals = self.merge_goals(agent_1,temp_goals,goals_table[k])
                                temp_targets = self.merge_targets(agent_1,temp_targets,targets_table[k], use_error = False)

                            #find problematic bounds clashes, if any
                            safe_idxs = []
                            for i in range(len(agent_2s)):
                                agent_2 = agent_2s[i]
                                violation = np.sum(np.where(temp_targets[agent_1][:,0] > targets_table[i][agent_1][:,1],1,0)) #check for merged lb > agent ub
                                violation = violation + np.sum(np.where(temp_targets[agent_1][:,1] < targets_table[i][agent_1][:,0],1,0)) #check for merged ub < agent lb
                                #adding the GOOD agents to the list instead of the bad ones this time, for ease of use later
                                if violation == 0:
                                    safe_idxs.append(i)
                            #add non-violators to the list for resolution later
                            if len(safe_idxs) < len(agent_2s):
                                solution_non_violators.append(np.asarray(safe_idxs))
                            #if no violators, we're done!
                            else:
                                solution_found = 1
                                solution = solutions[j]
                                used_idxs = solution
                                requested_targets = temp_targets
                                requested_goals = temp_goals
                                break
                        if solution_found == 1:
                            break

                        #if no members have compatible goals, we change the problem-agents to see if we can make it so!
                        for j in range(len(solutions)):
                            options = useable_designs
                            for k in range(len(solution_non_violators[j])):
                                safe_idx = solution_non_violators[j][k]
                                options[safe_idx] = [solutions[j][safe_idx]]
                            solutions[j],changed_idx = self.change_random_subsystem(solutions[j],options)
                
                #If our problem solving fails, then we have to bounce back the request to the problem agents...
                if solution_found == 0:

                    problem_targets = []
                    
                    #gather the lower/upper bounds of the problem agents
                    for i in range(len(violating_idxs)):
                        
                        table_idx = violating_idxs[i]
                        agent_2 = agent_2s[table_idx]
                        design_id = used_idxs[table_idx]
                        
                        problem_targets.append(np.ma.asarray(self.target_needs[agent_2][design_id][agent_1]))
                    
                    #consolidate them to means to not make the resulting bounce-back too ridiculous 
                    problem_targets = np.ma.mean(np.ma.asarray(problem_targets),axis=0)
                    requested_targets[agent_1] = problem_targets

                    
                    for i in range(len(violating_idxs)):
                        
                        table_idx = violating_idxs[i]
                        agent_2 = agent_2s[table_idx]
                        design_id = used_idxs[table_idx]
                        
                        updated_targets = self.merge_targets(agent_2,requested_targets,self.target_goals[agent_2][design_id])
                        if requested_targets == -1:
                            raise Exception("requested targets = -1 when it shouldnt!")
                        if updated_targets == -1:
                            self.merge_design(agent_2,preserve_design = [design_id])
                    
                            self.fork_design(agent_2,design_id)
                            self.wip_target_goals[agent_2] = requested_targets
                            self.is_requested[agent_1] = 1
                            self.is_communicating[agent_1] = 1
                        
                        else:
                            self.merge_design(agent_2,preserve_design = [design_id])
                        
                            self.fork_design(agent_2,design_id,inherit_goals = True)
                            self.wip_target_goals[agent_2] = updated_targets
                            self.is_requested[agent_1] = 1
                            self.is_communicating[agent_1] = 1

                    return 1
                     
        #Initial request is good!
        else:
            requested_targets = temp_targets
            requested_goals = temp_goals
        
        temp_idx = self.merge_design(agent_1)
        active_designs = np.nonzero(self.active_designs[agent_1])[0]
        
        goal_rank = np.asarray([self.check_goals(agent_1,self.design_props[agent_1][i],self.design_needs[agent_1][i],self.merge_goals(agent_1,requested_goals,self.design_goals[agent_1][i]),apply_penalty = False)+self.check_targets(agent_1,self.design_targets[agent_1][i],self.target_needs[agent_1][i],self.merge_targets(agent_1,requested_targets,self.target_goals[agent_1][i]),apply_penalty = False) for i in active_designs])

        closest_distance = np.amin(goal_rank)
        closest_design = active_designs[np.argmin(goal_rank)]
        
        if (closest_distance <= 0): #goal is already achieved; distances are expected to be high with this form of request so there's no rejection
            if self.integration_mode == 0:
                #use the props of the best design as-is
                requested_goals[agent_1] = np.ma.asarray(self.design_props[agent_1][closest_design])
                requested_targets[agent_1] = np.ma.stack((np.ma.asarray(self.design_targets[agent_1][closest_design])*(1-self.target_tol),np.ma.asarray(self.design_targets[agent_1][closest_design])*(1+self.target_tol)),axis=-1)

                #requested agents set the current design's properties as a goal, it must at least maintain these
                updated_goals = self.merge_goals(agent_1,requested_goals,self.design_goals[agent_1][closest_design])
                updated_targets = self.merge_targets(agent_1,requested_targets,self.target_goals[agent_1][closest_design])
                if updated_targets == -1:
                    print(requested_targets)
                    print(self.target_goals[agent_1][closest_design])
                    print(closest_distance)
                    print(closest_design)
                    raise Exception("Supposed working design doesn't actually work!")
                self.design_goals[agent_1][closest_design] = updated_goals
                self.target_goals[agent_1][closest_design] = updated_targets
                
                
            if temp_idx == -1 or temp_idx == None:
                self.new_design(agent_1)
            else:
                self.fork_design(agent_1,temp_idx) #pick back up on original work (from best point)
            
        else:
            if closest_distance >= self.PENALTY_SCORE:
                self.fork_design(agent_1,closest_design)
            else:
                self.fork_design(agent_1,closest_design,inherit_goals = True)

            updated_goals = self.merge_goals(agent_1,requested_goals,self.wip_goals[agent_1])
            updated_targets = self.merge_targets(agent_1,requested_targets,self.wip_target_goals[agent_1])
            if updated_targets == -1:
                print(self.wip_target_goals[agent_1])
                print(requested_targets)
                raise Exception("Target clash never resolved!")
            self.wip_goals[agent_1] = updated_goals
            self.wip_target_goals[agent_1] = updated_targets
            self.is_requested[agent_1] = 1
            self.is_communicating[agent_1] = 1
            
            for i in range(len(agent_2s)):
                self.team[agent_1].requested_by.append([agent_2s[i],used_idxs[i]])
    
        return 1
                       
    
    #Deprecated
    #TODO: Update this to handle subteam members 
        #TODO: mirror the goals onto the requesting agent if it failed, if succeed allow the requesting agent to 'betray'
    def close_request(self,team_id):
        self.is_requested[team_id] = 0
        self.is_communicating[team_id] = 0
        
        if len(self.team[team_id][agent_id].requested_by) > 0:
            idx = self.merge_design(team_id, must_merge = True)
            if idx == -1 or None:
                self.team[team_id][agent_id].requested_by = []
                return
        else:
            self.team[team_id][agent_id].requested_by = []
            self.merge_design(team_id)
            return
        
        for i in self.team[team_id][agent_id].requested_by:
        
            requesting_agent = i[0]
            requesting_design = i[1]
            
            design_goals = copy.deepcopy(self.problem.needs_bases)
            target_goals = copy.deepcopy(self.problem.target_need_bases)
            design_goals[requesting_agent] = self.design_goals[team_id][idx][requesting_agent]
            target_goals[requesting_agent] = self.target_goals[team_id][idx][requesting_agent]
            
            #Bounce the request back if the agent couldn't satisfy its properties
            if (self.check_goals(team_id,self.design_props[team_id][idx],self.design_needs[team_id][idx],design_goals)+self.check_targets(team_id,self.design_targets[team_id][idx],self.target_needs[team_id][idx],target_goals)) > 0:
                final_goals = copy.deepcopy(self.problem.needs_bases)
                final_goals[team_id] = np.ma.asarray(copy.deepcopy(self.design_props[team_id][idx]))
                final_targets = copy.deepcopy(self.problem.target_need_bases)
                final_targets[team_id] = np.ma.stack((np.ma.asarray(copy.deepcopy(self.design_targets[team_id][idx]))*(1-self.target_tol),np.ma.asarray(copy.deepcopy(self.design_targets[team_id][idx]))*(1+self.target_tol)),axis=-1)
                if final_targets == -1:
                    raise Exception("Final targets = -1 when it shouldnt!")
                if self.merge_targets(requesting_agent,self.target_goals[requesting_agent][requesting_design],final_targets) == -1:
                    if not self.is_communicating[requesting_agent]:
                        self.merge_design(requesting_agent,preserve_design = [requesting_design])
                        self.fork_design(requesting_agent,requesting_design)
                        self.is_requested[requesting_agent] = 1
                        self.is_communicating[requesting_agent] = 1
                        
                        self.wip_goals[requesting_agent] = final_goals
                        self.wip_target_goals[requesting_agent] = final_targets
                else:
                    if not self.is_communicating[requesting_agent]:
                        
                        test_targets = self.merge_targets(requesting_agent,self.target_goals[requesting_agent][requesting_design],final_targets)
                        if test_targets == -1:
                            raise Exception("Final targets = -1 when it shouldnt!")

                        self.merge_design(requesting_agent,preserve_design = [requesting_design])
                        self.fork_design(requesting_agent,requesting_design,inherit_goals = True)
                        self.wip_goals[requesting_agent] = self.merge_goals(requesting_agent,self.design_goals[requesting_agent][requesting_design],final_goals)
                        self.wip_target_goals[requesting_agent] = test_targets
                        self.is_requested[requesting_agent] = 1
                        self.is_communicating[requesting_agent] = 1
                
        self.team[team_id][agent_id].requested_by = []
        
    def start_stop(self,team_id):
        if self.active_teams[team_id] > 0.5:
            self.active_teams[team_id] = 0
        else:
            self.active_teams[team_id] = 1
        return
    
    #start a new work-in-progress design either from an existing design, or from the base template
    def new_design(self, team_id,agent_id):
        
        #skip the whole new design thing if we're doing the nominal PB study. 
        if self.nominal_PB == True:
            return
        
        template_chance = np.random.uniform()
        if self.integration_mode == 1 or self.current_max_designs == 1:
            template_chance = 2
        if self.use_subsets == True:
            max_designs = self.current_subset_size
        else:
            max_designs = self.current_max_designs
        if template_chance > 1/(max_designs+1):
            
            if self.use_subsets == True:
                subset_offset = self.subset_size*agent_id
                active_idxs = np.nonzero(self.active_designs[team_id][subset_offset:subset_offset+self.subset_size])[0]
                choice = np.random.choice(active_idxs)+subset_offset
                self.fork_design(team_id,agent_id,choice)
            else:
                active_idxs = np.nonzero(self.active_designs[team_id])[0]
                choice = np.random.choice(active_idxs)
                self.fork_design(team_id,agent_id,choice)
            
        else:
            self.fork_base_design(team_id,agent_id) 
            choice = -1

        return choice
    
    def fork_base_design(self,team_id,agent_id):
        self.state_dict["forked_designs"][team_id][agent_id] = -1
        self.wip[team_id][agent_id]= copy.deepcopy(self.problem.subproblems[team_id])
        self.wip_props[team_id][agent_id] = copy.deepcopy(self.problem.design_props[team_id])
        self.wip_targets[team_id][agent_id] = copy.deepcopy(self.problem.design_targets[team_id])
        self.wip_needs[team_id][agent_id] = copy.deepcopy(self.problem.subproblem_needs[team_id])
        self.wip_target_needs[team_id][agent_id] = copy.deepcopy(self.problem.target_needs[team_id])
        self.wip_goals[team_id][agent_id] = copy.deepcopy(self.problem.subproblem_goals[team_id])
        self.wip_target_goals[team_id][agent_id] = copy.deepcopy(self.problem.target_goals[team_id])
        if self.integration_mode == 0:
            self.old_obj_funcs[team_id][agent_id] = self.local_objective(team_id,self.wip_props[team_id][agent_id],self.wip_needs[team_id][agent_id],self.wip_targets[team_id][agent_id],self.wip_target_needs[team_id][agent_id]) + self.check_constraints(team_id,self.wip[team_id][agent_id])
        else:
            self.old_obj_funcs[team_id][agent_id], self.previous_solutions[team_id][agent_id] = self.design_compiler([team_id],self.wip_props[team_id][agent_id],self.wip_needs[team_id][agent_id],self.wip_targets[team_id][agent_id],self.wip_target_needs[team_id][agent_id])
            #self.old_obj_funcs[team_id][agent_id] += self.check_constraints(team_id,self.wip[team_id][agent_id]) #implied to be satisfied

        self.team[team_id][agent_id].active_temp = self.team[team_id][agent_id].t_init
        self.team[team_id][agent_id].active_delt = self.team[team_id][agent_id].delt_init
        self.team[team_id][agent_id].active_hist = copy.deepcopy(self.team[team_id][agent_id].hist_init)
        self.team[team_id][agent_id].active_hist_needs = copy.deepcopy(self.team[team_id][agent_id].hist_init)
        self.team[team_id][agent_id].active_hist_target_needs = copy.deepcopy(self.team[team_id][agent_id].hist_init)
        self.team[team_id][agent_id].active_hist_design = copy.deepcopy(self.team[team_id][agent_id].hist_init)
        self.team[team_id][agent_id].active_hist_solution = copy.deepcopy(self.team[team_id][agent_id].hist_init)
        self.team[team_id][agent_id].active_last_move = np.random.choice(self.team[team_id][agent_id].move_ids)
        return

    def cull_design(self,team_id):
        
        active_idxs = np.nonzero(self.active_designs[team_id])[0]
        comm_obj_funcs = np.zeros(len(active_idxs))
        for i in range(len(active_idxs)):
            if active_idxs[i] == self.best_solution[team_id]:
                comm_obj_funcs[i] = -self.PENALTY_SCORE
            else:
                obj,solution = self.design_compiler([team_id],self.design_props[team_id][active_idxs[i]],self.design_needs[team_id][active_idxs[i]],self.design_targets[team_id][active_idxs[i]],self.target_needs[team_id][active_idxs[i]])
                comm_obj_funcs[i] = obj

                
        max_obj = np.amax(comm_obj_funcs)
        max_index = np.nonzero(comm_obj_funcs == max_obj)[0][0]

        self.active_designs[team_id][active_idxs[max_index]] = 0
        self.design_props[team_id][active_idxs[max_index]] = np.zeros_like(self.design_props[team_id][max_index])
        self.design_targets[team_id][active_idxs[max_index]] = np.zeros_like(self.design_targets[team_id][max_index])
        self.state_dict["culled_designs"][team_id] = max_index
        
        return max_index, max_obj

    def remove_killed_designs(self,team_id,max_obj):
        
        active_idxs = np.nonzero(self.active_designs[team_id])[0]
        comm_obj_funcs = np.zeros(len(active_idxs))
        for i in range(len(active_idxs)):
            if active_idxs[i] == self.best_solution[team_id]:
                comm_obj_funcs[i] = -self.PENALTY_SCORE
            else:
                obj,solution = self.design_compiler([team_id],self.design_props[team_id][active_idxs[i]],self.design_needs[team_id][active_idxs[i]],self.design_targets[team_id][active_idxs[i]],self.target_needs[team_id][active_idxs[i]])
                comm_obj_funcs[i] = obj
                
        designs_removed = 0
                
        for i in range(len(active_idxs)):
            if comm_obj_funcs[i] > max_obj:
                designs_killed = 1
                self.active_designs[team_id][active_idxs[i]] = 0
                self.design_props[team_id][active_idxs[i]] = np.zeros_like(self.design_props[team_id][i])
                self.design_targets[team_id][active_idxs[i]] = np.zeros_like(self.design_targets[team_id][i])

        
        return designs_removed
    
    def merge_all_designs(self,must_merge = [], preserve_design = []):
        for i in range(self.num_subproblems):
            if self.use_joint_iteration == 0:
                for j in range(self.subteam_size):
                    self.merge_design(i,j,update_other_wip = False)
                for j in range(self.subteam_size):
                    self.new_design(i,j)
            else:
            
                self.merge_design(i,0,update_other_wip = False)
                choice = self.new_design(i,0)
                for j in range(1,self.subteam_size):
                    if choice == -1:
                        self.fork_base_design(i,j)
                    else:
                        self.fork_design(i,j,choice)
                
        
        if self.integration_mode == 1:
            for i in range(self.num_subproblems):
                if self.use_joint_iteration == 0:
                    for j in range(self.subteam_size):
                        penalty = self.check_constraints(i,self.wip[i][j])
                        if penalty > 0:
                            self.old_obj_funcs[i][j] = penalty
                        else:
                            self.old_obj_funcs[i][j], self.previous_solutions[i][j] = self.design_compiler([i],self.wip_props[i][j],self.wip_needs[i][j],self.wip_targets[i][j],self.wip_target_needs[i][j]) 
                else: 
                    j=0
                    penalty = self.check_constraints(i,self.wip[i][j])
                    if penalty > 0:
                        self.old_obj_funcs[i][j] = penalty
                    else:
                        self.old_obj_funcs[i][j], self.previous_solutions[i][j] = self.design_compiler([i],self.wip_props[i][j],self.wip_needs[i][j],self.wip_targets[i][j],self.wip_target_needs[i][j]) 


        return
    

    def merge_design(self,team_id,agent_id, must_merge = False,preserve_design = None,update_other_wip = True): 

        #take the best design from the wip history
        if len(self.team[team_id][agent_id].active_hist) == 0:
            return
        
        #get the stats of the CURRENT wip design in case it's fallen off of the history
        used_solution = copy.deepcopy(self.previous_solutions[team_id][agent_id])
        if self.integration_mode == 1:
            penalty = self.check_constraints(team_id,self.wip[team_id][agent_id])
            if penalty > 0:
                wip_obj = penalty
            else:
                props,needs,targets,target_needs = self.get_solution_properties(used_solution,[team_id],self.wip_props[team_id][agent_id],self.wip_needs[team_id][agent_id],self.wip_targets[team_id][agent_id],self.wip_target_needs[team_id][agent_id])
                goal_penalties = self.check_goals(team_id,self.wip_props[team_id][agent_id],self.wip_needs[team_id][agent_id],self.wip_goals[team_id][agent_id])+self.check_targets(team_id,self.wip_targets[team_id][agent_id],self.wip_target_needs[team_id][agent_id],self.wip_target_goals[team_id][agent_id])
                if props != -1:
                    #evaluate how the design performs with the last solution used as well; useful for stability
                    wip_obj = self.evaluate_solution(props,needs,targets,target_needs) + goal_penalties
                else:
                    wip_obj = self.PENALTY_SCORE
          
        else:
            wip_obj = self.local_objective(team_id,self.wip_props[team_id][agent_id],self.wip_needs[team_id][agent_id],self.wip_targets[team_id][agent_id],self.wip_target_needs[team_id][agent_id]) + self.check_constraints(team_id,self.wip[team_id][agent_id])

        
        wip_funcs = np.asarray(self.team[team_id][agent_id].active_hist)
        min_wip_obj = np.amin(wip_funcs)
        
        if wip_obj < min_wip_obj:
            min_wip_idx = None
            min_wip_design = copy.deepcopy(self.wip[team_id][agent_id])
            min_wip_needs = copy.deepcopy(self.wip_needs[team_id][agent_id])
            min_wip_target_needs = copy.deepcopy(self.wip_target_needs[team_id][agent_id])
            min_wip_props = copy.deepcopy(self.wip_props[team_id][agent_id])
            min_wip_targets = copy.deepcopy(self.wip_targets[team_id][agent_id])
            min_wip_obj = wip_obj
            min_wip_solution = used_solution

        else:     
            min_wip_idx = np.nonzero(wip_funcs == min_wip_obj)[0][0]
            min_wip_design = copy.deepcopy(self.team[team_id][agent_id].active_hist_design[min_wip_idx])
            min_wip_needs = copy.deepcopy(self.team[team_id][agent_id].active_hist_needs[min_wip_idx])
            min_wip_target_needs = copy.deepcopy(self.team[team_id][agent_id].active_hist_target_needs[min_wip_idx])
            min_wip_props,min_wip_targets = self.problem.get_props(team_id,min_wip_design)
            if self.integration_mode == 1:
                min_wip_solution = copy.deepcopy(self.team[team_id][agent_id].active_hist_solution[min_wip_idx])
        
        
        
        #check to see if the solution the old design was used with is the same
        if self.integration_mode == 1:
            #if it's none, we're already using the present. 
            if min_wip_idx != None:
                min_wip_solution = copy.deepcopy(self.team[team_id][agent_id].active_hist_solution[min_wip_idx])
            
                props,needs,targets,target_needs = self.get_solution_properties(min_wip_solution,[team_id],min_wip_props,min_wip_needs,min_wip_targets,min_wip_target_needs)
                if props == -1:
                    min_wip_obj = self.PENALTY_SCORE
                else:
                    min_wip_obj = self.evaluate_solution(props,needs,targets,target_needs)

                #if it changed and is now worse than the present... use the present.
                if min_wip_obj > wip_obj:
#                     min_wip_obj = self.old_obj_funcs[team_id][agent_id]
#                     min_wip_idx = len(self.team[team_id][agent_id].active_hist)
#                     min_wip_design = copy.deepcopy(self.team[team_id][agent_id].active_hist_design[-1])
#                     min_wip_needs = copy.deepcopy(self.team[team_id][agent_id].active_hist_needs[-1])
#                     min_wip_target_needs = copy.deepcopy(self.team[team_id][agent_id].active_hist_target_needs[-1])
#                     min_wip_props,min_wip_targets = self.problem.get_props(team_id,min_wip_design)
#                     min_wip_solution = copy.deepcopy(self.team[team_id][agent_id].active_hist_solution[-1])

                    min_wip_obj = wip_obj
                    min_wip_idx = None
                    min_wip_design = copy.deepcopy(self.wip[team_id][agent_id])
                    min_wip_needs = copy.deepcopy(self.wip_needs[team_id][agent_id])
                    min_wip_target_needs = copy.deepcopy(self.wip_target_needs[team_id][agent_id])
                    min_wip_props = copy.deepcopy(self.wip_props[team_id][agent_id])
                    min_wip_targets = copy.deepcopy(self.wip_targets[team_id][agent_id])
                    min_wip_obj = wip_obj
                    min_wip_solution = used_solution

        
        
        #see if there's an open slot
        if self.use_subsets == True and self.nominal_PB == False:
            subset_offset = self.subset_size*agent_id
            num_active_designs = np.sum(self.active_designs[team_id][subset_offset:subset_offset+self.subset_size])
            current_max_designs = self.current_subset_size
        else:
            num_active_designs = np.sum(self.active_designs[team_id])
            current_max_designs = self.current_max_designs
        
        if( num_active_designs < current_max_designs):
            
            if self.use_subsets == True and self.nominal_PB == False:
                subset_offset = self.subset_size*agent_id
                idx = np.nonzero(1-self.active_designs[team_id][subset_offset:subset_offset+self.subset_size])[0][0]+subset_offset
            else:
                idx = np.nonzero(1-self.active_designs[team_id])[0][0]
            
            self.committed[team_id][idx] = copy.deepcopy(min_wip_design)
            self.design_props[team_id][idx] = copy.deepcopy(min_wip_props)
            self.design_targets[team_id][idx] = copy.deepcopy(min_wip_targets)
            self.design_needs[team_id][idx] = copy.deepcopy(min_wip_needs)
            self.target_needs[team_id][idx] = copy.deepcopy(min_wip_target_needs)
            if self.integration_mode == 0:
                self.design_goals[team_id][idx] = copy.deepcopy(self.wip_goals[team_id][agent_id])
                self.target_goals[team_id][idx] = copy.deepcopy(self.wip_target_goals[team_id][agent_id])

            
            self.active_designs[team_id][idx] = 1
            self.team[team_id][agent_id].temp[idx] = self.team[team_id][agent_id].active_temp
            self.team[team_id][agent_id].delt[idx] = self.team[team_id][agent_id].active_delt
            self.team[team_id][agent_id].hist[idx] = copy.deepcopy(self.team[team_id][agent_id].active_hist[:min_wip_idx])
            self.team[team_id][agent_id].hist_designs[idx] = copy.deepcopy(self.team[team_id][agent_id].active_hist_design[:min_wip_idx])
            self.team[team_id][agent_id].hist_solutions[idx] = copy.deepcopy(self.team[team_id][agent_id].active_hist_solution[:min_wip_idx])
            self.team[team_id][agent_id].hist_needs[idx] = copy.deepcopy(self.team[team_id][agent_id].active_hist_needs[:min_wip_idx])
            self.team[team_id][agent_id].hist_target_needs[idx] = copy.deepcopy(self.team[team_id][agent_id].active_hist_target_needs[:min_wip_idx])
            self.team[team_id][agent_id].last_moves[idx] = self.team[team_id][agent_id].active_last_move
            
            
            #update everyone's wip objective functions
            if self.integration_mode == 1 and update_other_wip == True:
                for i in range(self.num_subproblems):
                    for j in range(self.subteam_size):
                        if i != team_id:
                            penalty = self.check_constraints(i,self.wip[i][j])
                            if penalty > 0:
                                self.old_obj_funcs[i][j] = penalty
                            else:
                                self.old_obj_funcs[i][j], self.previous_solutions[i][j] = self.design_compiler([i],self.wip_props[i][j],self.wip_needs[i][j],self.wip_targets[i][j],self.wip_target_needs[i][j]) 
                       
            penalty = self.check_constraints(team_id,min_wip_design)
            if penalty <= 0:
                global_obj, global_sol = self.design_compiler([team_id],min_wip_props,min_wip_needs,min_wip_targets,min_wip_target_needs)
                global_sol[team_id] = idx
                self.update_global_objective(global_obj,global_sol)
            self.state_dict["replaced_designs"][team_id][agent_id] = idx
            return idx
        
        else:          
            #if not, find out what needs to be replaced
            #active designs
            if self.use_subsets == True and self.nominal_PB == False:
                subset_offset = self.subset_size*agent_id
                active_idxs = np.nonzero(self.active_designs[team_id][subset_offset:subset_offset+self.subset_size])[0]+subset_offset
            else:
                active_idxs = np.nonzero(self.active_designs[team_id])[0]
            #this line returns a list of obj funcs for each active design, considering the space INCLUDING the best wip design
            #a goal-bearing design should always be merged UNLESS the project has entered integration mode
            if self.integration_mode == 0:
                comm_obj_funcs = np.asarray([self.local_objective(team_id,self.design_props[team_id][i],self.design_needs[team_id][i],self.design_targets[team_id][i],self.target_needs[team_id][i], swap=[i,min_wip_props,min_wip_targets],check_goals = False) for i in active_idxs])
            else:
                best_sol = np.zeros(self.num_subproblems).astype(int)
                best_obj = self.PENALTY_SCORE
                comm_obj_funcs = np.zeros(len(active_idxs))
                comm_solutions = np.zeros((len(active_idxs),self.num_subproblems)).astype(int)
                for i in range(len(active_idxs)):
                    penalty = self.check_constraints(team_id,self.committed[team_id][active_idxs[i]])
                    if penalty > 0:
                        obj = penalty*self.PENALTY_SCORE/1000+self.PENALTY_SCORE
                    else:
                        obj,solution = self.design_compiler([team_id],self.design_props[team_id][active_idxs[i]],self.design_needs[team_id][active_idxs[i]],self.design_targets[team_id][active_idxs[i]],self.target_needs[team_id][active_idxs[i]])
                        solution[team_id] = active_idxs[i]
                        comm_solutions[i] = solution
                    comm_obj_funcs[i] = obj
                    
                    #make sure we're checking against the best solution; if we improved on it we can update it

                    
                    if  obj < self.best_solution_value:
                        self.update_global_objective(obj,solution)
                    else:
                        if self.best_solution[team_id] == active_idxs[i]:
                            comm_obj_funcs[i] = copy.deepcopy(self.best_solution_value)
                            comm_solutions[i] = copy.deepcopy(self.best_solution)
                            obj = copy.deepcopy(self.best_solution_value)
                            solution = copy.deepcopy(self.best_solution)
                    
                    if obj < best_obj:
                        best_obj = obj
                        best_sol = solution
                        
                if min_wip_obj < self.best_solution_value:
                    must_merge = True
                    
            self.state_dict["obj_func_on_merge"][team_id][agent_id] = comm_obj_funcs[0]
            self.state_dict["obj_func_on_merge2"][team_id][agent_id] = min_wip_obj
            #add the wip design to solutions that can be overwrote (making a 'no merge') - this is overridden for request fulfilling cases
            if must_merge == False:
                comm_obj_funcs = np.append(comm_obj_funcs, min_wip_obj)
                active_idxs = np.append(active_idxs,-1)
                
            if preserve_design is not None:
                for i in range(len(preserve_design)):
                    rem_idx = np.nonzero(active_idxs == preserve_design[i])[0][0]
                    comm_obj_funcs = np.delete(comm_obj_funcs,rem_idx)
                    active_idxs = np.delete(active_idxs, rem_idx)

            
            #pick a design to replace probabalistically
            #TODO: set up a non-probabalistic version (or keep it probabalistic) for PB nominal
            reject = 1
            while(reject == 1):
                reject = 0

                rem_idx = random.choices(range(len(active_idxs)),np.exp(comm_obj_funcs-np.amax(comm_obj_funcs)))[0]
                rem_design = active_idxs[rem_idx]

                #for a 'failed merge' - the wip solution is rejected
                if rem_design == -1:
                    #self.search_best_solution(team_id,best_sol,best_obj)
                    return -1
                #go through rejection conditions
                if self.integration_mode == 1:
                    #if the design to replace is used in THE BEST solution, the replacing design must further improve it
                    if self.best_solution[team_id] == rem_design:
                        if min_wip_obj > self.best_solution_value:
                            active_idxs = np.delete(active_idxs, rem_idx)
                            comm_obj_funcs = np.delete(comm_obj_funcs,rem_idx)
                            reject = 1
                #check that the new design meets all goals of the old design
                else:
                    if self.check_goals(team_id, min_wip_props, min_wip_needs, self.merge_goals(team_id,self.wip_goals[team_id][agent_id],self.design_goals[team_id][rem_design])) > 0 or self.check_targets(team_id, min_wip_targets, min_wip_target_needs, self.merge_targets(team_id,self.wip_target_goals[team_id][agent_id],self.target_goals[team_id][rem_design])) > 0:
                        active_idxs = np.delete(active_idxs, rem_idx)
                        comm_obj_funcs = np.delete(comm_obj_funcs,rem_idx)
                        reject = 1
                #this case should ONLY trip if must_merge == true
                if len(active_idxs) == 0:
                    #self.search_best_solution(team_id,best_sol,best_obj)
                    return -1


            #going through with the merge!

            
            self.design_props[team_id][rem_design] = copy.deepcopy(min_wip_props)
            self.design_targets[team_id][rem_design] = copy.deepcopy(min_wip_targets)
            self.committed[team_id][rem_design] = copy.deepcopy(min_wip_design)
            self.design_needs[team_id][rem_design] = copy.deepcopy(min_wip_needs)  
            self.target_needs[team_id][rem_design] = copy.deepcopy(min_wip_target_needs)
            if self.integration_mode == 0:
                
                self.design_goals[team_id][rem_design] = self.merge_goals(team_id,self.design_goals[team_id][rem_design],self.wip_goals[team_id][agent_id])
                test_targets = self.merge_targets(team_id,self.target_goals[team_id][rem_design],self.wip_target_goals[team_id][agent_id])
                if test_targets == -1:
                    raise Exception("Merging design targets is wrong!")
                self.target_goals[team_id][rem_design] = test_targets


            self.team[team_id][agent_id].temp[rem_design] = self.team[team_id][agent_id].active_temp
            self.team[team_id][agent_id].delt[rem_design] = self.team[team_id][agent_id].active_delt
            self.team[team_id][agent_id].hist[rem_design] = copy.deepcopy(self.team[team_id][agent_id].active_hist[:min_wip_idx])
            self.team[team_id][agent_id].hist_needs[rem_design] = copy.deepcopy(self.team[team_id][agent_id].active_hist_needs[:min_wip_idx])
            self.team[team_id][agent_id].hist_target_needs[rem_design] = copy.deepcopy(self.team[team_id][agent_id].active_hist_target_needs[:min_wip_idx])
            self.team[team_id][agent_id].hist_designs[rem_design] = copy.deepcopy(self.team[team_id][agent_id].active_hist_design[:min_wip_idx])
            self.team[team_id][agent_id].hist_solutions[rem_design] = copy.deepcopy(self.team[team_id][agent_id].active_hist_solution[:min_wip_idx])
            self.team[team_id][agent_id].last_moves[rem_design] = self.team[team_id][agent_id].active_last_move
            
            #update everyone's wip objective functions
            
            if self.integration_mode == 1 and update_other_wip == True:
                for i in range(self.num_subproblems):
                    for j in range(self.subteam_size):
                        if i != team_id:
                            penalty = self.check_constraints(i,self.wip[i][j])
                            if penalty > 0:
                                self.old_obj_funcs[i][j] = penalty
                            else:
                                self.old_obj_funcs[i][j], self.previous_solutions[i][j] = self.design_compiler([i],self.wip_props[i][j],self.wip_needs[i][j],self.wip_targets[i][j],self.wip_target_needs[i][j]) 

                if min_wip_obj < best_obj:
                    best_obj = min_wip_obj
                    best_sol = min_wip_solution
                    best_sol[team_id] = rem_design
                
                self.search_best_solution(team_id,best_sol,best_obj)
            else:
                penalty = self.check_constraints(team_id,min_wip_design)
                if penalty <= 0:
                    global_obj, global_sol = self.design_compiler([team_id],min_wip_props,min_wip_needs,min_wip_targets,min_wip_target_needs)
                    global_sol[team_id] = rem_design
                    self.update_global_objective(global_obj,global_sol)
                    
            
            self.state_dict["replaced_designs"][team_id][agent_id] = rem_design

            return rem_design



    #Forks an agent's existing design into a new copy to work on
    def fork_design(self,team_id,agent_id,design_id, inherit_goals = False):
        self.state_dict["forked_designs"][team_id][agent_id] = design_id
        if self.active_designs[team_id][design_id] == 0:
            return
        
        #same behavior question as in new design
        self.wip[team_id][agent_id] = copy.deepcopy(self.committed[team_id][design_id])
        self.wip_props[team_id][agent_id] = copy.deepcopy(self.design_props[team_id][design_id])
        self.wip_targets[team_id][agent_id] = copy.deepcopy(self.design_targets[team_id][design_id])
        self.wip_needs[team_id][agent_id] = copy.deepcopy(self.design_needs[team_id][design_id])
        self.wip_target_needs[team_id][agent_id] = copy.deepcopy(self.target_needs[team_id][design_id])
        if inherit_goals == True:
            self.wip_goals[team_id][agent_id] = copy.deepcopy(self.design_goals[team_id][design_id])
            self.wip_target_goals[team_id][agent_id] = copy.deepcopy(self.target_goals[team_id][design_id])
        else:
            self.wip_goals[team_id][agent_id] = copy.deepcopy(self.problem.subproblem_goals[team_id])
            self.wip_target_goals[team_id][agent_id] = copy.deepcopy(self.problem.target_goals[team_id])
        
        if self.integration_mode == 0:
            self.old_obj_funcs[team_id][agent_id] = self.local_objective(team_id,self.wip_props[team_id][agent_id],self.wip_needs[team_id][agent_id],self.wip_targets[team_id][agent_id],self.wip_target_needs[team_id][agent_id]) + self.check_constraints(team_id,self.wip[team_id][agent_id])
        else:

            penalty = self.check_constraints(team_id, self.wip[team_id][agent_id])
            if penalty > 0:
                self.old_obj_funcs[team_id][agent_id] = penalty
            else:
                self.old_obj_funcs[team_id][agent_id], self.previous_solutions[team_id][agent_id] = self.design_compiler([team_id],self.wip_props[team_id][agent_id],self.wip_needs[team_id][agent_id],self.wip_targets[team_id][agent_id],self.wip_target_needs[team_id][agent_id])

        #port in agent properties for the design
        self.team[team_id][agent_id].active_temp = self.team[team_id][agent_id].temp[design_id]
        self.team[team_id][agent_id].active_delt = self.team[team_id][agent_id].delt[design_id]
        self.team[team_id][agent_id].active_hist = copy.deepcopy(self.team[team_id][agent_id].hist[design_id])
        self.team[team_id][agent_id].active_hist_target_needs = copy.deepcopy(self.team[team_id][agent_id].hist_target_needs[design_id])
        self.team[team_id][agent_id].active_hist_needs = copy.deepcopy(self.team[team_id][agent_id].hist_needs[design_id])
        self.team[team_id][agent_id].active_hist_design = copy.deepcopy(self.team[team_id][agent_id].hist_designs[design_id])
        self.team[team_id][agent_id].active_hist_solution = copy.deepcopy(self.team[team_id][agent_id].hist_solutions[design_id])
        self.team[team_id][agent_id].active_last_move = self.team[team_id][agent_id].last_moves[design_id]

        return
    
    #Finds an open design slot for an agent 
    def get_empty_id(self,team_id):
        for i in range(self.max_designs):
            if self.active_designs[team_id][i] == 0:
                return i
        return -1

    def search_best_solution(self,start_subproblem,start_sol, start_obj):
        obj = start_obj
        sol = copy.deepcopy(start_sol)
        last_subproblem = start_subproblem
        for i in range(self.current_max_designs):
            chosen_subproblem = np.random.choice(self.num_subproblems)
            while chosen_subproblem == last_subproblem:
                chosen_subproblem = np.random.choice(self.num_subproblems)
            chosen_design = sol[chosen_subproblem]
            new_obj,new_sol = self.design_compiler([chosen_subproblem], self.design_props[chosen_subproblem][chosen_design],self.design_needs[chosen_subproblem][chosen_design],self.design_targets[chosen_subproblem][chosen_design],self.target_needs[chosen_subproblem][chosen_design])
            new_sol[chosen_subproblem] = chosen_design
            if new_obj < obj:
                obj = new_obj
                sol = copy.deepcopy(new_sol)
            else:
                self.update_global_objective(obj,sol)
                return 
        self.update_global_objective(obj,sol)
        return 
                
    
    def update_global_objective(self,obj,solution):
        if obj < self.best_solution_value:
            self.best_solution = copy.deepcopy(solution)
            self.best_solution_value = copy.deepcopy(obj)
            self.best_designs = ([copy.deepcopy(self.committed[i][solution[i]])for i in range(self.num_subproblems)])
            self.best_props = ([copy.deepcopy(self.design_props[i][solution[i]]) for i in range(self.num_subproblems)])
            self.best_needs = ([copy.deepcopy(self.design_needs[i][solution[i]]) for i in range(self.num_subproblems)])
            self.best_targets = ([copy.deepcopy(self.design_targets[i][solution[i]]) for i in range(self.num_subproblems)])
            self.best_target_needs = ([copy.deepcopy(self.target_needs[i][solution[i]]) for i in range(self.num_subproblems)])

            
        if self.best_solution_value < self.PENALTY_SCORE:
            self.best_solution_validity = 1

        
    #just a multiobjective simulated annnealing objective function!... of a sort.
    #Note for testing: being dominated by a design should ALWAYS be worse than any proximity penalties involved
    #TODO: rework for subsets
    def local_objective(self,team_id,wip_props,wip_needs,wip_targets,wip_target_needs, swap = [-1,[]], check_goals = True, use_distances = True, agent_id = -1):
        
        value = 0
            
        if check_goals == True:
            value += self.check_goals(team_id,wip_props,wip_needs,self.wip_goals[team_id][agent_id])
            value += self.check_targets(team_id,wip_targets,wip_target_needs,self.wip_target_goals[team_id][agent_id])

        if (value > 0): 
            return value + self.goal_penalty

        else:
            design_props = copy.deepcopy(self.design_props[team_id])
            design_targets = copy.deepcopy(self.design_targets[team_id])

            #add up the basic values, negative weight for higher is better, positive weight for lower is better
            value = np.sum(wip_props * self.pareto_weights[team_id])

            #swapping functionality for evaluating already-committed designs (needed for evluations when deciding what to cull/keep)
            #the already-committed design comes in the wip_props slot, and the wip design to be merged comes in the swap[1] field
            if swap[0] > -1:
                design_props[swap[0]] = swap[1]
                design_targets[swap[0]] = swap[2]
                
            #mask out inactive designs
            mask = 1- np.transpose(np.tile(self.active_designs[team_id],(len(design_props[0]),1)))
            design_props = np.ma.masked_array(design_props,mask)

            #negative values indicate other design is superior to the selected design
            distances = (design_props - wip_props) * self.pareto_weights[team_id]

            #kludge to handle '0 weight' properties used to share information; makes them not influence distance penalties
            if np.sum(self.active_designs[team_id]) > 1:
                for i in range(len(self.pareto_weights[team_id])): 
                    if self.pareto_weights[team_id][i] <= 1e-6:
                        distances[:,i] = -self.PENALTY_SCORE
                        
            if len(wip_props) != 0:
                #find dominating designs by finding ones with a negative 'maximum distance'
                dom_distances = np.amax(distances,axis=1)

                #NOTE: Masked_greater corresponds to min() andst maked_less to max() functions from the paper
                #they accomplish the same things, they just cleanly work with arrays and constant additions and the design masking
                #functions in the rest of the framework.
                
                #mask out non-dominating solutions (those with a positive max distance)
                dom_distance = np.ma.sum(np.ma.masked_greater_equal(dom_distances,0)-self.proximity_coef/self.proximity_eps)
                #a minimum of the proximity coefficient is set to ensure it connects smoothly with the proximity penalties 
                if dom_distance is not np.ma.masked:
                #Penalty for being dominated
                    value -= dom_distance*self.dominance_coef * use_distances
                #Proximity penalty - sum over positive distances
                distances = np.sum(np.ma.masked_less(distances,0),axis = 1)
                #can do a more complicated distance norm here; using L1 for now
                #penalty for being close to designs; want this to be sub-linear with distance too (derivative is smaller at greater distance)
                #Linear term is to help a design understand where it is relative to others; makes it favor the characteristics it's 'best' in
                if np.sum(distances) is not np.ma.masked:
                    value += (np.sum(self.proximity_coef/(distances+self.proximity_eps)) - np.sum(distances * self.proximity_coef_lin)) * use_distances
                #target proximity penalty

            target_distances = np.abs((design_targets-wip_targets) * self.target_weights[team_id])

            value += np.sum(self.proximity_coef_target/(target_distances+self.proximity_eps)) * use_distances  
        return value
    
    def design_compiler(self, team_ids = [-1], given_props = 0,given_needs = 0,given_targets = 0,given_target_needs = 0):
        
        
        props = copy.deepcopy(self.design_props)
        targets = copy.deepcopy(self.design_targets)
        needs = copy.deepcopy(self.design_needs)
        target_needs = copy.deepcopy(self.target_needs)
        active_designs = copy.deepcopy(self.active_designs)
        
        starting_samples = self.compiler_starting_samples
        
        if len(team_ids) == self.num_subproblems:
            return self.evaluate_solution(given_props,given_needs,given_targets,given_target_needs), np.zeros(self.num_subproblems)
    
        else:
            if self.current_max_designs == 1:
                only_solution = ([np.nonzero(active_designs[i])[0][0] for i in range(self.num_subproblems)])

                if len(team_ids) == 1:
                    if team_ids == [-1]:
                        props,needs,targets,target_needs = self.get_solution_properties(only_solution)
                    else:
                        props,needs,targets,target_needs = self.get_solution_properties(only_solution,team_ids,given_props,given_needs,given_targets,given_target_needs)
                        only_solution[team_ids[0]] = 0
                else:
                    props,needs,targets,target_needs = self.get_solution_properties(only_solution,team_ids,given_props,given_needs,given_targets,given_target_needs)
                    for i in range(len(team_ids)):
                        only_solution[team_ids[i]] = 0
                return self.evaluate_solution(props,needs,targets,target_needs), only_solution
                        
          
        
            if len(team_ids) == 1:
                if team_ids == [-1]:
                    starting_samples *= self.current_max_designs
                else:
                    given_props = [given_props]
                    given_needs = [given_needs]
                    given_targets = [given_targets]
                    given_target_needs = [given_target_needs]
                    
                    for i in range(len(team_ids)): #this one should always eval to 0
                        if self.is_design_compatible(team_ids[i],given_props[i],given_needs[i],given_targets[i],given_target_needs[i]):
                            props[team_ids[i]] = [given_props[i]]
                            needs[team_ids[i]] = [given_needs[i]]
                            targets[team_ids[i]] = [given_targets[i]]
                            target_needs[team_ids[i]] = [given_target_needs[i]]
                            active_designs[team_ids[i]] = 1
                        else: 
                            return self.PENALTY_SCORE,-1
                
            else:
                 #restrict testing to the designs in question to be used... circumvents active designs to allow use of wip designs
                for i in range(len(team_ids)):
                    if self.is_design_compatible(team_ids[i],given_props[i],given_needs[i],given_targets[i],given_target_needs[i]):
                        props[team_ids[i]] = [given_props[i]]
                        needs[team_ids[i]] = [given_needs[i]]
                        targets[team_ids[i]] = [given_targets[i]]
                        target_needs[team_ids[i]] = [given_target_needs[i]]
                        active_designs[team_ids[i]] = 1
                    else: 
                        return self.PENALTY_SCORE,-1
   
            
        active_ids = ([np.nonzero(active_designs[i])[0] for i in range(self.num_subproblems)])
        
        #find the best combination!
        
        solution_props = []
        solution_needs = []
        solution_targets = []
        solution_target_needs = []
        solutions = ([self.random_solution(active_ids) for i in range(starting_samples)])

        for i in range(starting_samples):
            solution_props.append([copy.deepcopy(props[j][solutions[i][j]]) for j in range(len(solutions[i]))])
            solution_needs.append([copy.deepcopy(needs[j][solutions[i][j]]) for j in range(len(solutions[i]))])
            solution_targets.append([copy.deepcopy(targets[j][solutions[i][j]]) for j in range(len(solutions[i]))])
            solution_target_needs.append([copy.deepcopy(target_needs[j][solutions[i][j]]) for j in range(len(solutions[i]))])
            
        solution_values = ([self.evaluate_solution(solution_props[i],solution_needs[i],solution_targets[i],solution_target_needs[i]) for i in range(len(solutions))])
        
        min_idx = np.argmin(solution_values)
        best_obj = solution_values[min_idx]
        best_solution = solutions[min_idx]
        
        countdown = len(solutions[0])*2
        for i in range(self.max_compiler_iterations):
            
            for i in range(len(solutions)):

                new_solution, changed_subsystem = self.change_random_subsystem(solutions[i], active_ids)
                
                t_solution = solutions[i]
                t_solution_props = solution_props[i][changed_subsystem]
                t_solution_needs = solution_needs[i][changed_subsystem]
                t_solution_targets = solution_targets[i][changed_subsystem]
                t_solution_target_needs = solution_target_needs[i][changed_subsystem]
                t_solution_value = solution_values[i]
                
                
                solutions[i] = new_solution
                solution_props[i][changed_subsystem] = props[changed_subsystem][new_solution[changed_subsystem]]
                solution_needs[i][changed_subsystem] = needs[changed_subsystem][new_solution[changed_subsystem]]
                solution_targets[i][changed_subsystem] = targets[changed_subsystem][new_solution[changed_subsystem]]
                solution_target_needs[i][changed_subsystem] = target_needs[changed_subsystem][new_solution[changed_subsystem]]
                solution_values[i] = self.evaluate_solution(solution_props[i],solution_needs[i],solution_targets[i],solution_target_needs[i])
                
                if t_solution_value < solution_values[i]:
                    solutions[i] = t_solution
                    solution_props[i][changed_subsystem] = t_solution_props
                    solution_needs[i][changed_subsystem] = t_solution_needs
                    solution_targets[i][changed_subsystem] = t_solution_targets
                    solution_target_needs[i][changed_subsystem] = t_solution_target_needs
                    solution_values[i] = t_solution_value

                min_idx = np.argmin(solution_values)
                
                if solution_values[min_idx] < best_obj:  
                    best_obj = solution_values[min_idx]
                    best_solution = solutions[min_idx]
                    countdown = len(solutions[0])*2+1
               
            countdown -= 1
            if countdown <= 0:
                break
                


        return best_obj,best_solution
    
    def get_solution_properties(self,solution, team_ids = [-1], given_props = 0,given_needs = 0, given_targets = 0,given_target_needs = 0):
        
        props = []
        needs = []
        targets = []
        target_needs = []
        idx = 0
        
        
        if len(team_ids) == 1:
            given_props = [given_props]
            given_needs = [given_needs]
            given_targets = [given_targets]
            given_target_needs = [given_target_needs]
        for i in range(self.num_subproblems):
            if self.active_designs[i][solution[i]] == 0:
                return -1,-1,-1,-1
            if idx == len(team_ids):
                props.append(self.design_props[i][solution[i]])
                needs.append(self.design_needs[i][solution[i]])
                targets.append(self.design_targets[i][solution[i]])
                target_needs.append(self.target_needs[i][solution[i]])
            elif i != team_ids[idx]:
                props.append(self.design_props[i][solution[i]])
                needs.append(self.design_needs[i][solution[i]])
                targets.append(self.design_targets[i][solution[i]])
                target_needs.append(self.target_needs[i][solution[i]])
            else:
                props.append(given_props[idx])
                needs.append(given_needs[idx])
                targets.append(given_targets[idx])
                target_needs.append(given_target_needs[idx])
                idx += 1
                
        return props, needs, targets, target_needs
    
    

    #Wrapper for global objective evaluations in the problem
    def evaluate_solution(self,props,needs,targets,target_needs):
        
        #compatible, penalty = self.is_solution_compatible(props,needs,targets,target_needs)
        compatible = True
        penalty = 0
        if compatible:

            obj = self.problem.global_objective(props,targets)

            if type(obj) is np.ndarray:
                return obj[0]
            else:
                return obj
        else:
            return penalty
    
        #function to find if the subdesigns in a given project solution are compatible with each other
    def is_solution_compatible(self,props,needs, targets, target_needs):
        penalty = 0
        for i in range(len(props)): #design that needs - requesting agent id
            for j in range(len(props)): #design that is needed - receiving agent id                
                if np.sum(needs[i][j]) == np.ma.masked or i == j:
                    continue
                else:
                    d1 = (needs[i][j] - props[j]) * self.pareto_weights[j]
                    
                    for k in range(len(self.pareto_weights[j])): 
                        if self.pareto_weights[j][k] <= 1e-6:
                            d1[k] = d1[k]*1e8
                    
                    d2 = targets[j] - target_needs[i][j][:,0]
                    d3 = target_needs[i][j][:,1] - targets[j]
                    violation = np.sum(np.ma.array([np.sum(np.minimum(d1,np.zeros_like(d1))), np.sum(np.minimum(d2,np.zeros_like(d2))),np.sum(np.minimum(d3,np.zeros_like(d3)))]))
                    if violation < 0:
                        penalty -= violation
        if penalty > 0:
             return False, penalty*self.incompatibility_scale + self.incompatibility_penalty
        else:
            return True, 0

    
    def find_incompatible_designs(self):
        #exclude incompatible designs - needs may be asymmetric
        incompatibles = []
        for i in range(self.num_subproblems):
            active_ids = np.nonzero(self.active_designs[i])[0]
            for j in active_ids:
                compatible, penalty = self.is_design_compatible(i,self.design_props[i][j],self.design_needs[i][j],self.design_targets[i][j], self.target_needs[i][j], calc_penalties = False)
                if not compatible:
                    incompatibles.append([i,j])
        return incompatibles
    
    #function to find if a subdesign is compatible with at least one subdesign in each other set. IE does it 
    #'intersect' with other specialist's sets
    def is_design_compatible(self,team_id, props, needs,targets, target_needs, calc_penalties = True):
        penalty = 0

        for i in range(self.num_subproblems):
            active_ids = np.nonzero(self.active_designs[i])[0]
            if i == team_id or np.sum(needs[i]) == np.ma.masked:
                violated = 0
                continue
            smallest_violation = -self.PENALTY_SCORE
            violated = 1
            for j in active_ids:
                #example - for lower is better, needs must be greater than corresponding props, and weight is positive.
                #therefore distances must be positive to be 'fulfilled' - negative distance is a problem!
                #first direction - this design's needs must be satisfied by others
                d1 = (needs[i] - self.design_props[i][j]) * self.pareto_weights[i]
                d2 = self.design_targets[i][j] - target_needs[i][:,0] 
                d3 = target_needs[i][:,1] - self.design_targets[i][j]
                
                #reverse direction - this design must satisfy the needs of others too

                
                d4 = (self.design_needs[i][j][team_id]-props)*self.pareto_weights[team_id]
                d5 = targets - self.target_needs[i][j][team_id][:,0]
                d6 = self.target_needs[i][j][team_id][:,1]- targets
                violation = np.sum(np.ma.array([np.sum(np.minimum(d1,np.zeros_like(d1))),
                                                np.sum(np.minimum(d2,np.zeros_like(d2))),
                                                np.sum(np.minimum(d3,np.zeros_like(d3))),
                                                np.sum(np.minimum(d4,np.zeros_like(d4))),
                                                np.sum(np.minimum(d5,np.zeros_like(d5))),
                                                np.sum(np.minimum(d6,np.zeros_like(d6)))]))
                if violation >= 0 or np.ma.is_masked(violation):
                    smallest_violation = 0
                    violated = 0
                    break
                else:
                    violated = 1
                    if violation > smallest_violation:
                        smallest_violation = violation
                        
            if calc_penalties == False and smallest_violation > 0:
                return False, 0
            
            penalty -= smallest_violation
        if penalty > 0:
            return False, penalty*self.incompatibility_scale + self.incompatibility_penalty
        else:
            return True, 0
        

    def check_constraints(self,team_id,design): 
    
         penalty = (self.timestep*100)/self.max_iterations*self.problem.get_constraints(team_id,design)
        
        
         if penalty > 0:
            return penalty + self.constraint_penalty*self.integration_mode
         else: 
            return 0
    
#Goals and targets                       
    def check_goals(self,team_id,props,needs,goals, apply_penalty = True):
        
        active_goals = np.zeros(self.num_subproblems)
        for i in range(self.num_subproblems):
            if len(goals[i]) > 0:
                active_goals[i] = np.sum(~goals[i].mask)
        
        
        if (np.sum(active_goals) == 0):
            return 0
        active_goals = np.nonzero(active_goals > 0 )[0]
        goal_distances = 0
        for i in active_goals:
            if i != team_id: #for handling need-goals, their priorities are reversed (lower is better for requiring a higher-is-better property)
                scaled_distances = (needs[i] - goals[i]) * -self.pareto_weights[i]
            else:
                scaled_distances = (props - goals[i]) * self.pareto_weights[i]
            
            for j in range(len(self.pareto_weights[i])): 
                if self.pareto_weights[i][j] <= 1e-6:
                    scaled_distances[j] = scaled_distances[j]*1e8
                   
            penalty = np.ma.sum(np.ma.maximum(scaled_distances,np.zeros_like(scaled_distances))**2)
            if not np.ma.is_masked(penalty):
                goal_distances = np.ma.sum((goal_distances,penalty))
                
        
        if goal_distances <= 0.001 or np.ma.is_masked(goal_distances):
            return 0
        else:
            return goal_distances*self.goal_scale + self.goal_penalty*apply_penalty
        
    def check_targets(self,team_id,targets,target_needs,target_goals,apply_penalty = True):
        #handle the error thrown by merge_targets if it pops into this function (It is intended for this to happen sometimes)
        if target_goals == -1:
            return self.PENALTY_SCORE**2
        
        active_targets_upper = np.zeros(self.num_subproblems)
        active_targets_lower = np.zeros(self.num_subproblems)
        for i in range(self.num_subproblems):
            if len(target_goals[i]) > 0:
                active_targets_upper[i] = np.sum(~target_goals[i][:,1].mask)
                active_targets_lower[i] = np.sum(~target_goals[i][:,0].mask)

        active_targets_upper = np.nonzero(active_targets_upper > 0 )[0]
        target_distances = 0
        if (np.sum(active_targets_upper) != 0):
            for i in active_targets_upper:
                if i != team_id:
                    scaled_distances_upper = (target_needs[i][:,1] - target_goals[i][:,1]) * -self.target_weights[i]
                else:
                    scaled_distances_upper = (targets - target_goals[i][:,1]) * self.target_weights[i]

                penalty = np.ma.sum(np.maximum(scaled_distances_upper,np.zeros_like(scaled_distances_upper))**2)
                target_distances = np.ma.sum((target_distances,penalty))

        active_targets_lower = np.nonzero(active_targets_lower > 0 )[0]
        if (np.sum(active_targets_lower) != 0):
            for i in active_targets_lower:
                if i != team_id:    
                    scaled_distances_lower = (target_needs[i][:,0] - target_goals[i][:,0]) * self.target_weights[i]
                else:
                    scaled_distances_lower = (targets - target_goals[i][:,0]) * -self.target_weights[i]

                penalty = np.ma.sum(np.ma.maximum(scaled_distances_lower,np.zeros_like(scaled_distances_lower))**2)
                target_distances = np.ma.sum((target_distances,penalty))
            
        if target_distances <= 0.001 or np.ma.is_masked(target_distances):
            return 0
        else:
            return target_distances*self.goal_scale + self.goal_penalty*apply_penalty
      
    #merges requested goals with any present goals, taking the more extreme requirements
    def merge_goals(self,team_id,goals_1t,goals_2t):
        goals_1 = copy.deepcopy(goals_1t)
        goals_2 = copy.deepcopy(goals_2t)
        for i in range(self.num_subproblems):
            signs = np.sign(self.pareto_weights[i])
            if i != team_id:
                signs *= -1
            goals_1[i] = np.ma.masked_equal(np.minimum((goals_1[i]*signs).filled(self.PENALTY_SCORE),(goals_2[i]*signs).filled(self.PENALTY_SCORE)),self.PENALTY_SCORE)*signs
        return goals_1
    
    def merge_targets(self,team_id,targets_1t,targets_2t, use_error = True):
        targets_1 = copy.deepcopy(targets_1t)
        targets_2 = copy.deepcopy(targets_2t)
        for i in range(self.num_subproblems):
            if i != team_id:
                targets_1[i][:,0] = np.ma.masked_equal(np.minimum(targets_1[i][:,0].filled(self.PENALTY_SCORE),targets_2[i][:,0].filled(self.PENALTY_SCORE)),self.PENALTY_SCORE)
                targets_1[i][:,1] = np.ma.masked_equal(np.maximum(targets_1[i][:,1].filled(-self.PENALTY_SCORE),targets_2[i][:,1].filled(-self.PENALTY_SCORE)),-self.PENALTY_SCORE)
            else:
                targets_1[i][:,0] = np.ma.masked_equal(np.maximum(targets_1[i][:,0].filled(-self.PENALTY_SCORE),targets_2[i][:,0].filled(-self.PENALTY_SCORE)),-self.PENALTY_SCORE)
                targets_1[i][:,1] = np.ma.masked_equal(np.minimum(targets_1[i][:,1].filled(self.PENALTY_SCORE),targets_2[i][:,1].filled(self.PENALTY_SCORE)),self.PENALTY_SCORE)
                #check to make sure lower bounds arent greater than upper bounds, return an error code if they are
                if np.sum(targets_1[i][:,0]>targets_1[i][:,1]) > 0.5 and use_error == True:
                    return -1
        return targets_1


#utilities for design compiler and problem solving prcoess
    def change_random_subsystem(self,ct, options):
        
        c = copy.deepcopy(ct)
        
        options_lengths = np.asarray([len(options[i]) for i in range(len(options))])
        valid_switches = np.nonzero(np.where(options_lengths > 1,1,0))[0]
        if len(valid_switches) == 0:
            return c.astype(int),0
                
        idx1 = np.random.randint(0,len(valid_switches))
        idx1 = valid_switches[idx1]
        idx2 = np.random.randint(0,len(options[idx1]))
        if len(options[idx1]) > 1:
            while options[idx1][idx2] == c[idx1]:
                idx2 = np.random.randint(0,len(options[idx1]))

        c[idx1] = options[idx1][idx2]
        return c.astype(int), idx1

    def random_solution(self,options):
        
        c = np.zeros(len(options))
        for i in range(len(options)):
            c[i] = options[i][np.random.randint(0,len(options[i]))]
        return c.astype(int)
    
    def change_weights(self):
        self.problem.change_weights()
        self.best_solution_value = self.evaluate_solution(self.best_props,self.best_needs,self.best_targets,self.best_target_needs)
        
        
        

