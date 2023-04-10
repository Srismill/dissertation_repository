#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import copy

class design_specialist:
    
    #hidden markov init
    #def __init__(self,num_states,num_ops,learn_rate): 
    
    def __init__(self,num_ops,learn_rate,temp,num_designs):
        
        
        self.learn_rate = learn_rate;
        self.selected_design = 0
        
        self.requested_by = []
        
 
        #objective function:
        
        #actions:
        self.move_ids = np.arange(0,num_ops)
        
        self.num_designs = num_designs
        
        #Learning mechanics: Markov
        self.move_prob = np.ones((num_ops,num_ops))/num_ops
        self.move = np.random.choice(self.move_ids)
        
        #Learning mechanics: hidden markov
        #self.state_ids = np.arange(0,num_states)
        #self.trans_prob = np.zeros(num_states,num_states)
        #self.op_prefs = np.zeros(num_states,num_ops)
        #self.hidden_state = random_choice(self.state_ids)

        
        #Triki temp stuff
        self.t_init = temp
        self.delt_init = 1e6
        self.hist_init = []
        self.temp = np.ones(num_designs)*self.t_init
        self.delt = np.ones(num_designs)*self.delt_init
        self.hist_length = 10;
        self.active_temp = self.t_init
        self.active_delt = self.delt_init
        self.active_last_move = random.randint(0,num_ops-1)
        self.move = random.randint(0,num_ops-1)

        self.last_moves = np.random.randint(0,num_ops,num_designs)
        
        self.hist = [copy.deepcopy(self.hist_init) for i in range(num_designs)]
        self.hist_needs = [copy.deepcopy(self.hist_init) for i in range(num_designs)]
        self.hist_target_needs = [copy.deepcopy(self.hist_init) for i in range(num_designs)]
        self.hist_designs = [copy.deepcopy(self.hist_init) for i in range(num_designs)]
        self.hist_solutions = [copy.deepcopy(self.hist_init) for i in range(num_designs)]
        self.active_hist = copy.deepcopy(self.hist_init)
        self.active_hist_needs = copy.deepcopy(self.hist_init)
        self.active_hist_target_needs = copy.deepcopy(self.hist_init)
        self.active_hist_design = copy.deepcopy(self.hist_init)
        self.active_hist_solution = copy.deepcopy(self.hist_init)

    def iterate(self):
        pass
        
    def select_move(self):
        
        #Markov learning
        #pick a new move # save it
        self.active_last_move = self.move

        
        self.move = random.choices(self.move_ids,self.move_prob[self.active_last_move])[0]
        
        
        #HIDDEN MARKOV 
        #store current statistics
        #self.old_state = self.hidden_state
        
        #pick new state
        #self.hidden_state = random.choices(self.move_ids,self.trans_prob[self.hidden_state])
        #pick a new move
        #self.move = random.choices(self.move_ids,self.op_prefs[self.hidden_state])
        
        #Move
        return self.move
        
    
    def update_learn(self, d_qual):
        
        if d_qual <= 0:
            #markov
            self.move_prob[self.active_last_move][self.move] *= (1+self.learn_rate);
            accept = 1
            
            #hidden markov
            #self.op_prefs[self.hidden_state,self.move] *= (1+self.learn_rate);
            #self.trans_prob[self.old_state,self.hidden_state] *= (1+self.learn_rate);
            
        elif d_qual > 0:
            #markov
            self.move_prob[self.active_last_move][self.move] *= (1-self.learn_rate);
            
            if random.uniform(0,1) < np.exp(-d_qual/self.active_temp):
                accept = 1
            else:
                accept = 0
            
            #hidden markov
            #self.op_prefs[self.hidden_state,self.move] *= (1-self.learn_rate);
            #self.trans_prob[self.hidden_state,self.move] *= (1-self.learn_rate);
            
        #Normalize arrays
        #markov
        self.move_prob[self.active_last_move] /= np.sum(self.move_prob[self.active_last_move])
        
        
        return accept
        
        #hidden markov
        #self.op_prefs /= np.sum(self.op_prefs)
        #self.trans_prob /= np.sum(self.trans_prob)
        
    def update_temp(self):
        if len(self.active_hist) <= 1:
            return
        else:
            var = np.var(np.asarray(self.active_hist))
            if var > 0:
                update_factor = self.active_delt * self.active_temp / var
                if update_factor > 1:
                    self.active_delt /= 2
                    update_factor /= 2
                if update_factor < 1:
                    self.active_temp = self.active_temp*(1-update_factor)
                    
    def reset_temps(self):
        self.temp = np.ones(self.num_designs)*self.t_init
        self.delt = np.ones(self.num_designs)*self.delt_init
        self.active_temp = self.t_init
        self.active_delt = self.delt_init
                
    def reset_active_temps(self):
        self.active_temp = self.t_init
        self.active_delt = self.delt_init
        
    def reset_histories(self):
        self.hist = [copy.deepcopy(self.hist_init) for i in range(self.num_designs)]
        self.hist_needs = [copy.deepcopy(self.hist_init) for i in range(self.num_designs)]
        self.hist_target_needs = [copy.deepcopy(self.hist_init) for i in range(self.num_designs)]
        self.hist_designs = [copy.deepcopy(self.hist_init) for i in range(self.num_designs)]
        self.hist_solutions = [copy.deepcopy(self.hist_init) for i in range(self.num_designs)]
        self.active_hist = copy.deepcopy(self.hist_init)
        self.active_hist_needs = copy.deepcopy(self.hist_init)
        self.active_hist_target_needs = copy.deepcopy(self.hist_init)
        self.active_hist_design = copy.deepcopy(self.hist_init)
        self.active_hist_solution = copy.deepcopy(self.hist_init)
        
    def reset_active_histories(self):
        self.active_hist = copy.deepcopy(self.hist_init)
        self.active_hist_needs = copy.deepcopy(self.hist_init)
        self.active_hist_target_needs = copy.deepcopy(self.hist_init)
        self.active_hist_design = copy.deepcopy(self.hist_init)
        self.active_hist_solution = copy.deepcopy(self.hist_init)


# In[ ]:




