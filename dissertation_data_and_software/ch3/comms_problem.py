import numpy as np
from numpy import array, ones, pi, cos, sqrt, sin
from pandas import read_csv, read_excel
from random import uniform, randint
from os.path import dirname
import copy

class sae_problem():

    #4 or more specialists for SAE problem - 11 total roles with some that may be chunked together
    def __init__(self):
        
        SAEdir = "SAE-fmincon-master\\SAE"
        
        # read data into dataframes
        params=read_excel(SAEdir+"\\resources\\params.xlsx", engine='openpyxl')
        materials=read_csv(SAEdir+"\\resources\\materials.csv")
        tires=read_csv(SAEdir+"\\resources\\tires.csv")
        motors=read_csv(SAEdir+"\\resources\\motors.csv")
        brakes=read_csv(SAEdir+"\\resources\\brakes.csv")
        suspension=read_csv(SAEdir+"\\resources\\suspension.csv")

        
            
        # constants
        self.v_car = 26.8 # m/s
        self.w_e = 3600*2*pi/60 # radians/sec
        self.rho_air = 1.225 # kg/m3
        self.r_track = 9 # m
        self.P_brk = 10**7 # Pascals
        self.C_dc = 0.04 # drag coefficient of cabin
        self.gravity = 9.81 # m/s^2
        self.y_suspension = 0.05 # m
        self.dydt_suspension = 0.025 # m/s
        self.num_materials = 12
        self.num_tires = 6
        self.num_engines = 20
        self.num_brakes = 33
        self.num_suspensions = 4
        
        self.constraint_penalty = 100
        self.weights = array([45,11000,45,3,1000,1/600,1,5,10,2,1/30])/100
        #self.weights = array([45,11000,45,3,1000,1/600,100,100,50,20,1/3])/100
        
        
        # car vector with continuous and integer variables
        self.vector = []
        
        # continuous parameters with fixed bounds
        for i in range(19):
            temp = uniform(params.at[i, 'min']+(params.at[i, 'max']-params.at[i, 'min'])*0.05, params.at[i, 'max']-(params.at[i, 'max']-params.at[i, 'min'])*0.25) #random; shaving off the edges to reduce disastrous starters
            #temp = (params.at[i, 'min']+ params.at[i, 'max'])/2 #same
            setattr(self, params.at[i, 'variable'], temp)
            self.vector.append(temp)
            
        # integer parameters
        # materials  
        for i in range(5):
            temp = randint(0,self.num_materials) #random
            #temp = 6 #same
            setattr(self, params.at[19+i, 'variable'], materials.at[temp,'q'])
            self.vector.append(temp)
        setattr(self, 'Eia', materials.at[temp,'E']) 
        
                # rear tires
        setattr(self, 'rear_tire', randint(0,self.num_tires)) #random
        #setattr(self, 'rear_tire', 3) #same
        setattr(self, params.at[25, 'variable'], tires.at[self.rear_tire,'radius'])
        setattr(self, params.at[26, 'variable'], tires.at[self.rear_tire,'mass'])
        self.vector.append(self.rear_tire)
        
                # front tires
        setattr(self, 'front_tire', randint(0,self.num_tires)) #random
        #setattr(self, 'front_tire', 3) #same
        setattr(self, params.at[27, 'variable'], tires.at[self.front_tire,'radius'])
        setattr(self, params.at[28, 'variable'], tires.at[self.front_tire,'mass'])
        self.vector.append(self.front_tire)
        
                # engine
        setattr(self, 'engine', randint(0,self.num_engines)) #random
        #setattr(self, 'engine', 10) #same
        setattr(self, params.at[29, 'variable'], motors.at[self.engine,'Power'])
        setattr(self, params.at[30, 'variable'], motors.at[self.engine,'Length'])
        setattr(self, params.at[31, 'variable'], motors.at[self.engine,'Height'])
        setattr(self, params.at[32, 'variable'], motors.at[self.engine,'Torque'])
        setattr(self, params.at[33, 'variable'], motors.at[self.engine,'Mass'])
        self.vector.append(self.engine)
        
                # brakes
        setattr(self, 'brakes', randint(0,self.num_brakes)) #random
        setattr(self, params.at[34, 'variable'], brakes.at[self.brakes,'rbrk'])
        setattr(self, params.at[35, 'variable'], brakes.at[self.brakes,'qbrk'])
        setattr(self, params.at[36, 'variable'], brakes.at[self.brakes,'lbrk'])
        setattr(self, params.at[37, 'variable'], brakes.at[self.brakes,'hbrk'])
        setattr(self, params.at[38, 'variable'], brakes.at[self.brakes,'wbrk'])
        setattr(self, params.at[39, 'variable'], brakes.at[self.brakes,'tbrk'])
        self.vector.append(self.brakes)
        
                # suspension
        setattr(self, 'rsp', randint(0,self.num_suspensions))
        setattr(self, params.at[40, 'variable'], suspension.at[self.rsp,'krsp'])
        setattr(self, params.at[41, 'variable'], suspension.at[self.rsp,'crsp'])
        setattr(self, params.at[42, 'variable'], suspension.at[self.rsp,'mrsp'])
        setattr(self, 'fsp', randint(0,self.num_suspensions))
        setattr(self, params.at[43, 'variable'], suspension.at[self.fsp,'kfsp'])
        setattr(self, params.at[44, 'variable'], suspension.at[self.fsp,'cfsp'])
        setattr(self, params.at[45, 'variable'], suspension.at[self.fsp,'mfsp'])
        self.vector.append(self.rsp)
        self.vector.append(self.fsp)
        
                # continuous parameters with variable bounds
        setattr(self, 'wrw', uniform(0.3 + (self.r_track - 2 * self.rrt - 0.3) * 0.05, self.r_track - 2 * self.rrt - (self.r_track - 2 * self.rrt-0.3)*0.25))
        setattr(self, 'yrw', uniform(.5 + self.hrw / 2, 1.2 - self.hrw / 2))
        setattr(self, 'yfw', uniform(0.03 + self.hfw / 2, .25 - self.hfw/2))
        setattr(self, 'ysw', uniform(0.03 + self.hsw/2, .250 - self.hsw/2))
        setattr(self, 'ye', uniform(0.03 + self.he / 2, .5 - self.he / 2))
        setattr(self, 'yc', uniform(0.03 + self.hc / 2, 1.200 - self.hc / 2))
        setattr(self, 'lia', uniform(0.2, .7  - self.lfw))
        setattr(self, 'yia', uniform(0.03 + self.hia / 2, 1.200 - self.hia / 2))
        setattr(self, 'yrsp', uniform(self.rrt, self.rrt * 2))
        setattr(self, 'yfsp', uniform(self.rft, self.rft * 2))
        
        for i in range(10):
            temp = getattr(self, params.at[46+i, 'variable'])
            self.vector.append(temp)
        
        self.params=params
        self.materials=materials
        self.tires=tires
        self.motors=motors
        self.brakes=brakes
        self.suspension=suspension

        
        
        self.number_of_subproblems = 6
        #self.num_agent_states = [? ? ?]
        self.learn_rates = np.asarray([0.01,0.01,0.01,0.01,0.01,0.01])
        self.actions_per_subproblem = np.asarray([12,12,12,10,16,18])
        self.temps = np.asarray([10000,10000,10000,10000,10000,10000])
        #WEIGHTS EXCEPTIONS: end of all wings (1*1e-8,length), end of cabin (1*1e-8,length)
        self.pareto_weights = [np.array([1/40,-1/40,1/4,1/0.6,1*1e-8,1*1e-8]),
                               np.array([1/15,-1/5,1/1.5,1/0.1,1*1e-8]),
                               np.array([1/4,-1/0.8,1/0.5,1/0.1,1*1e-8]),
                               np.array([1/4.5,1/4.5,1/10,-1/2e-5]),
                               np.array([1/20,1/10,1/.5,1/10,-1/30,1/.05,1/10000,1*1e-8,1*1e-8]),
                               np.array([1/35,1/.025,1/250000,1/0.6,1*1e-8,1/.25,1*1e-8,1/.25,1*1e-8])]

        
        self.target_weights = [np.array([]),
                               np.array([]),
                               np.array([]),
                               np.array([1/.1,1/.25,1/.1,1/.25]),
                               np.array([]),
                               np.array([1/1200,1/1200])]

        
        #Subproblems are given as a bunch of classes 
        self.subproblems = [self.vector[0:3]+self.vector[19:20]+self.vector[30:32], #rear wing
                            self.vector[3:7]+self.vector[20:21]+self.vector[32:33], #front wing
                            self.vector[7:11]+self.vector[21:22]+self.vector[33:34], #side wings
                            self.vector[11:12]+self.vector[24:25]+self.vector[12:13]+self.vector[25:26]+self.vector[27:28], #rear/front tires and brakes
                            self.vector[13:17]+self.vector[22:23]+self.vector[35:36]+self.vector[26:27]+self.vector[34:35], #cabin+engine
                            self.vector[17:19]+self.vector[23:24]+self.vector[36:38]+self.vector[28:29]+self.vector[38:39]+self.vector[29:30]+self.vector[39:40] #impact attenuator+suspensions
                           ]
        
        self.design_props = []
        self.design_targets = []
        for i in range(self.number_of_subproblems):
            props, targets = self.get_props(i,self.subproblems[i])
            self.design_props.append(props)
            self.design_targets.append(targets)

        
        #bases give shape of possible needs for each subproblem, organized by agent type index
        self.needs_bases = ([np.ma.masked_equal(np.zeros_like(self.design_props[i]),0) for i in range(self.number_of_subproblems)])

        self.target_need_bases = ([np.ma.masked_equal(np.stack((np.zeros_like(self.design_targets[i]),np.zeros_like(self.design_targets[i])),axis=-1),0) for i in range(self.number_of_subproblems)]) #create target needs as upper and lower bounds
        self.subproblem_needs = ([copy.deepcopy(self.needs_bases) for i in range(self.number_of_subproblems)]) #create table...
        self.target_needs = ([copy.deepcopy(self.target_need_bases) for i in range(self.number_of_subproblems)]) #create table...
        
        

        #self.target_needs = np.stack((self.target_needs,self.target_needs),axis=-1) #split it into upper/lower bounds
        #zero out self-needs
        self.subproblem_goals = copy.deepcopy(self.subproblem_needs)
        self.target_goals = copy.deepcopy(self.target_needs)
        for i in range(self.number_of_subproblems):
            self.subproblem_needs[i][i] = np.ma.asarray([])
            self.target_needs[i][i] = np.ma.asarray([])
        
        self.target_needs[0][3][1,1] = (self.r_track-self.subproblems[0][4])/2
        self.subproblem_needs[5][1][4] = 0.7-self.subproblems[5][3]
        self.target_needs[5][3][1,1] = self.subproblems[5][6]
        self.target_needs[5][3][1,0] = self.subproblems[5][6]/2
        self.target_needs[5][3][3,1] = self.subproblems[5][8]
        self.target_needs[5][3][3,0] = self.subproblems[5][8]/2

       
        

    def get_props(self, subproblem_id, subproblem):
        #calculation of subproblem results; last entry should be the global objective? First entry?
        
        props = np.asarray([])
        targets = np.asarray([])
        
        if (subproblem_id == 0): #rear wing
            #rear wing variable order: height, length, angle of attack, material, width, y position
            density = self.materials.at[subproblem[3],'q']
            mass = subproblem[0]*subproblem[1]*subproblem[4]*density
            downforce = self.F_down_wing(subproblem[4],subproblem[0],subproblem[1],subproblem[2],self.rho_air,self.v_car)
            drag = self.F_drag_wing(subproblem[4],subproblem[0],subproblem[1],subproblem[2],self.rho_air,self.v_car)
            CG_y = subproblem[5]
            props = np.asarray([mass,downforce,drag,CG_y,subproblem[1],subproblem[4]])
        elif (subproblem_id == 1): #front wing
            #variable order: height, length, width, angle of attack, material, y position
            density = self.materials.at[subproblem[4],'q']
            mass = subproblem[0]*subproblem[1]*subproblem[2]*density
            downforce = self.F_down_wing(subproblem[2],subproblem[0],subproblem[1],subproblem[3],self.rho_air,self.v_car)
            drag = self.F_drag_wing(subproblem[2],subproblem[0],subproblem[1],subproblem[3],self.rho_air,self.v_car)
            CG_y = subproblem[5]
            props = np.asarray([mass,downforce,drag,CG_y,subproblem[1]])
        elif (subproblem_id == 2): #side wings
            #variable order: height, length, width, angle of attack, material, y position
            density = self.materials.at[subproblem[4],'q']
            mass = 2*subproblem[0]*subproblem[1]*subproblem[2]*density
            downforce = 2*self.F_down_wing(subproblem[2],subproblem[0],subproblem[1],subproblem[3],self.rho_air,self.v_car)
            drag = 2*self.F_drag_wing(subproblem[2],subproblem[0],subproblem[1],subproblem[3],self.rho_air,self.v_car)
            CG_y = subproblem[5]
            props = np.asarray([mass,downforce,drag,CG_y,subproblem[1]])
        elif (subproblem_id == 3): #tires and brakes
            #order: rear tire pressure, rear tire selection, front tire pressure, front tire selection, brakes selection
            rear_pressure = subproblem[0]
            r_radius, r_mass = self.tire_props(subproblem[1])
            front_pressure = subproblem[2]
            f_radius,f_mass = self.tire_props(subproblem[3])
            b_radius, b_density, b_length,b_height,b_width,b_thickness = self.brake_props(subproblem[4])
            b_area = b_radius*b_width*b_height
            b_mass = b_length*b_width*b_height*b_density
            props = np.asarray([2*r_mass, 2*f_mass, 4*b_mass, b_area])
            targets = np.asarray([rear_pressure, r_radius, front_pressure,f_radius])
        elif (subproblem_id == 4): #cabin and engine
            #order: cabin height, cabin length, cabin width, cabin thickness,cabin material, cabin y pos, engine type, engine y pos
            c_density = self.materials.at[subproblem[4],'q']
            t1 = subproblem[0]*subproblem[1]*subproblem[3] 
            t2 = subproblem[0]*subproblem[2]*subproblem[3]
            t3 = subproblem[1]*subproblem[2]*subproblem[3]
            c_mass = 2*(t1+t2+t3)*c_density
            c_drag = self.F_drag(subproblem[2],subproblem[0],self.rho_air,self.v_car,self.C_dc)
            cabin_cgy = subproblem[5]
            m_pow, m_l,m_w, m_h, m_torque, m_mass = self.motor_props(subproblem[6])
            m_cgy = subproblem[7]
            
            props = np.asarray([c_mass,c_drag, cabin_cgy, m_mass,m_torque,m_cgy,m_pow,subproblem[1],subproblem[2]])
        elif (subproblem_id == 5): #impact attenuator+suspensions
            #order: IA height, IA width, IA material, IA length, IA y pos, rsp type, rsp y pos, fsp type, fsp y pos
            ia_density = self.materials.at[subproblem[2],'q']
            ia_stiffness = self.materials.at[subproblem[2],'E']
            ia_volume = subproblem[0]*subproblem[1]*subproblem[3]
            ia_mass = ia_volume*ia_density
            norm_crashforce =  sqrt( self.v_car**2 * subproblem[1] * subproblem[0] * ia_stiffness / (2*subproblem[3]))
            ia_cgy = subproblem[4]
            
            fsp_k, fsp_c, fsp_m = self.suspension_props(subproblem[7])
            fsp_cgy = subproblem[8]
            
            fsp_f = self.suspensionForce(fsp_k,fsp_c)
            
            rsp_k, rsp_c, rsp_m = self.suspension_props(subproblem[5])
            rsp_cgy = subproblem[6]
            
            rsp_f = self.suspensionForce(rsp_k,rsp_c)
            
            props = np.asarray([ia_mass, ia_volume, norm_crashforce, ia_cgy,rsp_m, rsp_cgy,fsp_m,fsp_cgy,subproblem[3]])
            targets = np.asarray([rsp_f,fsp_f])

        
        return props, targets
    
    def get_constraints(self,subproblem_id,subproblem):
        
        
        if (subproblem_id == 0): #rear wing
            #rear wing variable order: height, length, angle of attack, density, width, y position
            results = np.asarray([np.maximum(subproblem[5]-(1.2-subproblem[0]/2),0),np.minimum(subproblem[5]-(0.5+subproblem[0]/2),0)])
            results = np.sum(np.abs(results))
        elif (subproblem_id == 1): #front wing
            #variable order: height, length, width, angle of attack, density, y position
            results = np.asarray([np.maximum(subproblem[5]-(0.25-subproblem[0]/2),0),np.minimum(subproblem[5]-(0.03+subproblem[0]/2),0)])
            results = np.sum(np.abs(results))
        elif (subproblem_id == 2): #side wing
            #variable order: height, length, width, angle of attack, density, y position
            results = np.asarray([np.maximum(subproblem[5]-(0.25-subproblem[0]/2),0),np.minimum(subproblem[5]-(0.03+subproblem[0]/2),0)])
            results = np.sum(np.abs(results))
        elif (subproblem_id == 3): #tires and brakes
            #order: rear tire pressure, rear tire selection, front tire pressure, front tire selection, brakes selection
            results = 0
        elif (subproblem_id == 4): #cabin and engine
            #order: cabin height, cabin length, cabin width, cabin thickness,cabin material, cabin y pos, engine type, engine y pos
            results = np.asarray([np.maximum(subproblem[5]-(1.2-subproblem[0]/2),0),np.minimum(subproblem[5]-(0.03+subproblem[0]/2),0)])
            results = np.sum(np.abs(results))
            height_engine = self.motors.at[subproblem[6],'Height']
            results += np.sum(np.abs(np.asarray([np.maximum(subproblem[7]-(0.5-height_engine/2),0),np.minimum(subproblem[7]-(0.03+height_engine/2),0)])))
        elif (subproblem_id == 5): #impact attenuator+suspensions
            #order: IA height, IA width, IA material, IA length, IA y pos, rsp type, rsp y pos, fsp type, fsp y pos
            results = np.asarray([np.maximum(subproblem[4]-(1.2-subproblem[0]/2),0),np.minimum(subproblem[4]-(0.03+subproblem[0]/2),0)])
            results = np.sum(np.abs(results))
            
        return results

            
        # continuous parameters with variable bounds
        #setattr(self, 'wrw', uniform(0.3, self.r_track - 2 * self.rrt)) # relies on tires - implement as need
        #setattr(self, 'lia', uniform(0.2, .7  - self.lfw)) #relies on front wing
        #setattr(self, 'yrsp', uniform(self.rrt, self.rrt * 2)) #relies on tires
        #setattr(self, 'yfsp', uniform(self.rft, self.rft * 2))
    
    
    def apply_move(self,subproblem_id,move_id,subproblem_t,needs_t,target_needs_t):
        subproblem = copy.deepcopy(subproblem_t)
        needs = copy.deepcopy(needs_t)
        target_needs = copy.deepcopy(target_needs_t)
        
        #get a function back from the agent id/move id, and use it to make changes to subproblems and then returns them
        
        if subproblem_id == 0: #rear wing
            #rear wing variable order: height, length, angle of attack, material, width, y position
            if move_id == 0:
                subproblem[0] = self.continuous_increase(subproblem[0],0.7)
            elif move_id == 1:
                subproblem[0] = self.continuous_decrease(subproblem[0],0.025)
            elif move_id == 2:
                subproblem[1] = self.continuous_increase(subproblem[1],0.25)
            elif move_id == 3:
                subproblem[1] = self.continuous_decrease(subproblem[1],0.05)
            elif move_id == 4:
                subproblem[2] = self.continuous_increase(subproblem[2],0.785398)
            elif move_id == 5:
                subproblem[2] = self.continuous_decrease(subproblem[2],0)
            elif move_id == 6:
                subproblem[3] = self.discrete_increase(subproblem[3],self.num_materials)
            elif move_id == 7:
                subproblem[3] = self.discrete_decrease(subproblem[3],0)
            elif move_id == 8:
                subproblem[4] = self.continuous_increase(subproblem[4],self.r_track-2*self.tires.at[0,'radius']) #width of rear wing, relies on tires!, using upper bound
                target_needs[3][1,1] = (self.r_track-subproblem[4])/2
            elif move_id == 9:
                subproblem[4] = self.continuous_decrease(subproblem[4],0.3)
                target_needs[3][1,1] = (self.r_track-subproblem[4])/2
            elif move_id == 10:
                subproblem[5] = self.continuous_increase(subproblem[5],1.2-subproblem[0]/2)
            elif move_id == 11:
                subproblem[5] = self.continuous_decrease(subproblem[5],0.5+subproblem[0]/2)
        elif subproblem_id == 1: #front wing
            #variable order: height, length, width, angle of attack, material, y position
            if move_id == 0:
                subproblem[0] = self.continuous_increase(subproblem[0],0.2)
            elif move_id == 1:
                subproblem[0] = self.continuous_decrease(subproblem[0],0.025)
            elif move_id == 2:
                subproblem[1] = self.continuous_increase(subproblem[1],0.5)
            elif move_id == 3:
                subproblem[1] = self.continuous_decrease(subproblem[1],0.05)
            elif move_id == 4:
                subproblem[2] = self.continuous_increase(subproblem[2],3)
            elif move_id == 5:
                subproblem[2] = self.continuous_decrease(subproblem[2],0.3)
            elif move_id == 6:
                subproblem[4] = self.discrete_increase(subproblem[4],self.num_materials)
            elif move_id == 7:
                subproblem[4] = self.discrete_decrease(subproblem[4],0)
            elif move_id == 8:
                subproblem[3] = self.continuous_increase(subproblem[3],0.785398) 
            elif move_id == 9:
                subproblem[3] = self.continuous_decrease(subproblem[3],0)
            elif move_id == 10:
                subproblem[5] = self.continuous_increase(subproblem[5],0.25-subproblem[0]/2)
            elif move_id == 11:
                subproblem[5] = self.continuous_decrease(subproblem[5],0.03+subproblem[0]/2)
        elif subproblem_id == 2: #side wings
            #variable order: height, length, width, angle of attack, material, y position
            if move_id == 0:
                subproblem[0] = self.continuous_increase(subproblem[0],0.2)
            elif move_id == 1:
                subproblem[0] = self.continuous_decrease(subproblem[0],0.025)
            elif move_id == 2:
                subproblem[1] = self.continuous_increase(subproblem[1],0.55)
            elif move_id == 3:
                subproblem[1] = self.continuous_decrease(subproblem[1],0.05)
            elif move_id == 4:
                subproblem[2] = self.continuous_increase(subproblem[2],0.4)
            elif move_id == 5:
                subproblem[2] = self.continuous_decrease(subproblem[2],0.05)
            elif move_id == 6:
                subproblem[4] = self.discrete_increase(subproblem[4],self.num_materials)
            elif move_id == 7:
                subproblem[4] = self.discrete_decrease(subproblem[4],0)
            elif move_id == 8:
                subproblem[3] = self.continuous_increase(subproblem[3],0.785398) 
            elif move_id == 9:
                subproblem[3] = self.continuous_decrease(subproblem[3],0)
            elif move_id == 10:
                subproblem[5] = self.continuous_increase(subproblem[5],0.25-subproblem[0]/2)
            elif move_id == 11:
                subproblem[5] = self.continuous_decrease(subproblem[5],0.03+subproblem[0]/2)
        elif subproblem_id == 3: #tires and brakes
            #order: rear tire pressure, rear tire selection, front tire pressure, front tire selection, brakes selection
            if move_id == 0:
                subproblem[0] = self.continuous_increase(subproblem[0],1.03)
            elif move_id == 1:
                subproblem[0] = self.continuous_decrease(subproblem[0],0.758)
            elif move_id == 2:
                subproblem[1] = self.discrete_increase(subproblem[1],self.num_tires)
            elif move_id == 3:
                subproblem[1] = self.discrete_decrease(subproblem[1],0)
            elif move_id == 4:
                subproblem[2] = self.continuous_increase(subproblem[2],1.03)
            elif move_id == 5:
                subproblem[2] = self.continuous_decrease(subproblem[2],0.758)
            elif move_id == 6:
                subproblem[3] = self.discrete_increase(subproblem[3],self.num_tires)
            elif move_id == 7:
                subproblem[3] = self.discrete_decrease(subproblem[3],0)
            elif move_id == 8:
                subproblem[4] = self.discrete_increase(subproblem[4],self.num_brakes) 
            elif move_id == 9:
                subproblem[4] = self.discrete_decrease(subproblem[4],0)
        elif subproblem_id == 4: #cabin and engine
            #order: cabin height, cabin length, cabin width, cabin thickness,cabin material, cabin y pos, engine type, engine y pos
            if move_id == 0:
                subproblem[0] = self.continuous_increase(subproblem[0],1.17)
            elif move_id == 1:
                subproblem[0] = self.continuous_decrease(subproblem[0],0.5)
            elif move_id == 2:
                subproblem[1] = self.continuous_increase(subproblem[1],3)
            elif move_id == 3:
                subproblem[1] = self.continuous_decrease(subproblem[1],1.525)
            elif move_id == 4:
                subproblem[2] = self.continuous_increase(subproblem[2],2)
            elif move_id == 5:
                subproblem[2] = self.continuous_decrease(subproblem[2],0.5)
            elif move_id == 6:
                subproblem[4] = self.discrete_increase(subproblem[4],self.num_materials)
            elif move_id == 7:
                subproblem[4] = self.discrete_decrease(subproblem[4],0)
            elif move_id == 8:
                subproblem[3] = self.continuous_increase(subproblem[3],0.01) 
            elif move_id == 9:
                subproblem[3] = self.continuous_decrease(subproblem[3],0.0001)
            elif move_id == 10:
                subproblem[5] = self.continuous_increase(subproblem[5],1.2-subproblem[0]/2)
            elif move_id == 11:
                subproblem[5] = self.continuous_decrease(subproblem[5],0.03+subproblem[0]/2)
            elif move_id == 12:
                subproblem[6] = self.discrete_increase(subproblem[6],self.num_engines)
            elif move_id == 13:
                subproblem[6] = self.discrete_decrease(subproblem[6],0)
            elif move_id == 14:
                height_engine = self.motors.at[subproblem[6],'Height']
                subproblem[7] = self.continuous_increase(subproblem[7],0.5-height_engine/2)
            elif move_id == 15:
                height_engine = self.motors.at[subproblem[6],'Height']
                subproblem[7] = self.continuous_decrease(subproblem[7],0.03+height_engine/2)
        elif subproblem_id == 5: #impact attenuator+suspensions
            #order: IA height, IA width, IA material, IA length, IA y pos, rsp type, rsp y pos, fsp type, fsp y pos
            if move_id == 0:
                subproblem[0] = self.continuous_increase(subproblem[0],0.5)
            elif move_id == 1:
                subproblem[0] = self.continuous_decrease(subproblem[0],0.1)
            elif move_id == 2:
                subproblem[1] = self.continuous_increase(subproblem[1],0.5)
            elif move_id == 3:
                subproblem[1] = self.continuous_decrease(subproblem[1],0.2)
            elif move_id == 4:
                subproblem[3] = self.continuous_increase(subproblem[3],0.7 - 0.05) #depends on front wing, using upper bound
                needs[1][4] = 0.7-subproblem[3]
            elif move_id == 5:
                subproblem[3] = self.continuous_decrease(subproblem[3],0.2)
                needs[1][4] = 0.7-subproblem[3]
            elif move_id == 6:
                subproblem[2] = self.discrete_increase(subproblem[2],self.num_materials)
            elif move_id == 7:
                subproblem[2] = self.discrete_decrease(subproblem[2],0)
            elif move_id == 8:
                subproblem[4] = self.continuous_increase(subproblem[4],1.2 - subproblem[0]/2)
            elif move_id == 9:
                subproblem[4] = self.continuous_decrease(subproblem[4],0.03+ subproblem[0]/2)
            elif move_id == 10:
                subproblem[5] = self.discrete_increase(subproblem[5],self.num_suspensions)
            elif move_id == 11:
                subproblem[5] = self.discrete_decrease(subproblem[5],0)
            elif move_id == 12:
                subproblem[6] = self.continuous_increase(subproblem[6],2*self.tires.at[self.num_tires,'radius'])
                target_needs[3][1,1] = subproblem[6]
                target_needs[3][1,0] = subproblem[6]/2
            elif move_id == 13:
                subproblem[6] = self.continuous_decrease(subproblem[6],self.tires.at[0,'radius'])
                target_needs[3][1,1] = subproblem[6]
                target_needs[3][1,0] = subproblem[6]/2
            elif move_id == 14:
                subproblem[7] = self.discrete_increase(subproblem[7],self.num_suspensions)
            elif move_id == 15:
                subproblem[7] = self.discrete_decrease(subproblem[7],0)
            elif move_id == 16:
                subproblem[8] = self.continuous_increase(subproblem[8],2*self.tires.at[self.num_tires,'radius'])
                target_needs[3][3,1] = subproblem[8]
                target_needs[3][3,0] = subproblem[8]/2
            elif move_id == 17:
                subproblem[8] = self.continuous_decrease(subproblem[8],self.tires.at[0,'radius'])
                target_needs[3][3,1] = subproblem[8]
                target_needs[3][3,0] = subproblem[8]/2

        
        return subproblem,needs,target_needs
    
    #possible todo - include a weak objective to give continuous changes some directed optimization
    #find some way to shrink the search space based on variance?
    def continuous_increase(self, current_value,max_value = np.nan):
        if max_value == np.nan:
            return current_value*(1+0.2*np.random.random())
        else: 
            return current_value + (max_value-current_value)*np.random.random()
    
    def continuous_decrease(self, current_value,min_value = np.nan):
        if min_value == np.nan:
            return current_value*(1-0.2*np.random.random())
        else:
            return current_value + (min_value - current_value)*np.random.random()
        
    def discrete_increase(self,current_value, max_value):
        if current_value >= max_value-1:
            return max_value
        else:
            return current_value+np.random.randint(1,max_value-current_value)
    
    def discrete_decrease(self,current_value, min_value):
        if current_value <= min_value+1:
            return min_value
        else:
            return current_value-np.random.randint(1,current_value-min_value)
        
    def change_weights(self):
        self.weights = array([90,22000,45,3,1000,1/400,1,5,20,4,1/15])/100
        
        
    def global_objective(self, props, targets):
        #convert props inputs into meaningful variables to make decomposing in different ways easier
        rw_mass = props[0][0]
        rw_downforce = props[0][1]
        rw_drag = props[0][2]
        rw_cgy = props[0][3]
        rw_length = props[0][4]
        rw_width = props[0][5]
        
        fw_mass = props[1][0]
        fw_downforce = props[1][1]
        fw_drag = props[1][2]
        fw_cgy = props[1][3]
        fw_length = props[1][4]
        
        sw_mass = props[2][0]
        sw_downforce = props[2][1]
        sw_drag = props[2][2]
        sw_cgy = props[2][3]
        sw_length = props[2][4]
        
        rt_mass = props[3][0]
        ft_mass = props[3][1]
        b_mass = props[3][2]
        b_area = props[3][3]
        rt_pressure = targets[3][0]
        rt_radius = targets[3][1]
        ft_pressure = targets[3][2]
        ft_radius = targets[3][3]
        
        c_mass = props[4][0]
        c_drag = props[4][1]
        c_cgy = props[4][2]
        m_mass = props[4][3]
        m_torque = props[4][4]
        m_cgy = props[4][5]
        m_pow = props[4][6]
        c_length = props[4][7]
        c_width = props[4][8]
        
        ia_mass = props[5][0]
        ia_volume = props[5][1]
        norm_crashforce = props[5][2]
        ia_cgy = props[5][3]
        rsp_mass = props[5][4]
        rsp_cgy = props[5][5]
        fsp_mass = props[5][6]
        fsp_cgy = props[5][7]
        ia_length = props[5][8]
        rsp_f = targets[5][0]
        fsp_f = targets[5][1]
        
        
        if rw_width > (self.r_track-2*rt_radius):
            return 10000+rw_width*100
        elif ia_length > 0.7-fw_length:
            return 10000+ia_length*100
        elif rsp_cgy > 2*rt_radius:
            return 10000+rsp_cgy*100
        elif rsp_cgy < rt_radius:
            return 10000-rsp_cgy*100
        elif fsp_cgy > 2*ft_radius:
            return 10000+fsp_cgy*100
        elif fsp_cgy < ft_radius:
            return 10000-fsp_cgy*100
        

        weights = self.weights
        
        #mass (lower = better)
        t_1 = rw_mass + fw_mass + sw_mass + rt_mass + ft_mass 
        t_2 = b_mass + c_mass + m_mass + ia_mass + fsp_mass + rsp_mass
        subobj1= t_1+t_2
        
        #center of gravity (lower = better)
        t_1 = rw_mass*rw_cgy + fw_mass*fw_cgy + sw_mass*sw_cgy + rt_mass*rt_radius 
        t_2 = ft_mass*ft_radius + b_mass*ft_radius+b_mass*rt_radius + c_mass*c_cgy
        t_3 = m_mass*m_cgy + ia_mass*ia_cgy + fsp_mass*fsp_cgy + rsp_mass*rsp_cgy
        subobj2= t_1+t_2+t_3
        subobj2 = subobj2/subobj1
        
        #drag
        subobj3 = rw_drag+fw_drag+sw_drag+c_drag
        
        #downforce
        subobj4= rw_downforce+fw_downforce+sw_downforce
        
        #Precomputing total y forces on the car; if theyre less than 0 the car is infeasible
        Fy = subobj4 + subobj1*self.gravity-fsp_f-rsp_f 
        if Fy < 0:
            return 1e6 + (Fy**2)*100
        
        #acceleration
        total_res = subobj3 + self.rollingResistance(rt_pressure,self.v_car,subobj1)
        w_wheels = self.v_car/rt_radius
        efficiency = total_res*self.v_car/m_pow
        #efficiency = 1
        if m_pow == 0 or rt_radius == 0:
            raise Exception('aw shit here we go again')
            return 1e9
        F_wheels = m_torque*efficiency*self.w_e/(rt_radius*w_wheels)
        subobj5=(F_wheels-total_res)/subobj1
        
        #crash force
        subobj6=norm_crashforce*sqrt(subobj1)
        
        #IA volume
        subobj7=ia_volume
        
        #corner velocity

        
        Clat = 1.6
        #tire lateral grip limit
        subobj8=sqrt(Fy*Clat*self.r_track/subobj1)
        #rolling moment limit
        #subobj8 = min(subobj8, sqrt(c_width*self.r_track*(subobj1*self.gravity+subobj4)/(2*subobj1*subobj2)))
        
        #braking distance
        C = .005 + 1/ft_pressure * (.01 + .0095 * ((self.v_car*3.6/100)**2))
        Tbrk = 2*0.37*self.P_brk*b_area
        a_brk = Fy * C / subobj1 + 4*Tbrk/(rt_radius*subobj1)
        subobj9 = self.v_car**2/(2*a_brk)

        #suspension acceleration
        subobj10= -(rsp_f - fsp_f - subobj1*self.gravity - subobj2)/subobj1
        
        #pitch moment
        
        subobj11=abs(rsp_f/2+fsp_f/2+rw_downforce*(c_length-rw_length)-fw_downforce*(c_length-fw_length)-sw_downforce*(c_length-sw_length))
        
        #penalties = self.global_obj_constraints(props, subobj1, subobj4)
        

        return(np.array([subobj1*weights[0]+subobj2*weights[1]+subobj3*weights[2]-(subobj4)*weights[3]-
                      subobj5*weights[4]+subobj6*weights[5]+subobj7*weights[6]-subobj8*weights[7]+
                      subobj9*weights[8]+subobj10*weights[9]+subobj11*weights[10], 
                      subobj1, subobj2, subobj3, subobj4, subobj5, subobj6, subobj7, subobj8, subobj9, subobj10, subobj11]))


    # aspect ratio of wing
    def AR(self,w,alpha,l):
        return w* cos(alpha) / l

    # lift co-effecient
    def C_lift(self,AR,alpha):
        return 2*pi* (AR / (AR + 2)) * alpha
    
    # drag co-efficient
    def C_drag(self,C_lift, AR):
        return C_lift**2 / (pi * AR)
    
    # wing downforce
    def F_down_wing(self,w,h,l,alpha,rho_air,v_car):
        wingAR = self.AR(w,alpha,l)
        C_l = self.C_lift(wingAR, alpha)
        return 0.5 * alpha * h * w * rho_air * (v_car**2) * C_l
    
    # wing drag
    def F_drag_wing(self,w,h,l,alpha,rho_air,v_car):
        wingAR = self.AR(w,alpha,l)
        C_l = self.C_lift(wingAR, alpha)
        C_d = self.C_drag(C_l,wingAR)
        return self.F_drag(w,h,rho_air,v_car,C_d)
    
    # drag
    def F_drag(self,w,h,rho_air,v_car,C_d):
        return 0.5*w*h*rho_air*v_car**2*C_d
    
    def suspensionForce(self,k,c):
        return k*self.y_suspension + c*self.dydt_suspension
    
    # rolling resistance
    def rollingResistance(self,tirePressure,v_car,mass):
        C = .005 + 1/tirePressure * (.01 + .0095 * ((v_car*3.6/100)**2))
        return C * mass * self.gravity

    def tire_props(self, val):
        return [self.tires.at[val,'radius'],self.tires.at[val,'mass']]
    def motor_props(self,val):
        return [self.motors.at[val,'Power'], self.motors.at[val,'Length'], self.motors.at[val,'Width'],self.motors.at[val,'Height'],self.motors.at[val,'Torque'],self.motors.at[val,'Mass']]
    def brake_props(self,val):
        return [self.brakes.at[val,'rbrk'],self.brakes.at[val,'qbrk'],self.brakes.at[val,'lbrk'],self.brakes.at[val,'hbrk'],self.brakes.at[val,'wbrk'],self.brakes.at[val,'tbrk']]
    def suspension_props(self,val):
        return [self.suspension.at[val,'krsp'],self.suspension.at[val,'crsp'],self.suspension.at[val,'mrsp']]
    
    
    def bounds_constraint(self, lb, ub, value):
        if value < lb:
            return lb-value
        elif value > ub:
            return value-ub
        else:
            return 0
        
        
class sae_problem_low_decomp():

    #4 or more specialists for SAE problem - 11 total roles with some that may be chunked together
    def __init__(self):
        
        SAEdir = "SAE-fmincon-master\\SAE"
        
        # read data into dataframes
        params=read_excel(SAEdir+"\\resources\\params.xlsx", engine='openpyxl')
        materials=read_csv(SAEdir+"\\resources\\materials.csv")
        tires=read_csv(SAEdir+"\\resources\\tires.csv")
        motors=read_csv(SAEdir+"\\resources\\motors.csv")
        brakes=read_csv(SAEdir+"\\resources\\brakes.csv")
        suspension=read_csv(SAEdir+"\\resources\\suspension.csv")

        
            
        # constants
        self.v_car = 26.8 # m/s
        self.w_e = 3600*2*pi/60 # radians/sec
        self.rho_air = 1.225 # kg/m3
        self.r_track = 9 # m
        self.P_brk = 10**7 # Pascals
        self.C_dc = 0.04 # drag coefficient of cabin
        self.gravity = 9.81 # m/s^2
        self.y_suspension = 0.05 # m
        self.dydt_suspension = 0.025 # m/s
        self.num_materials = 12
        self.num_tires = 6
        self.num_engines = 20
        self.num_brakes = 33
        self.num_suspensions = 4
        
        self.constraint_penalty = 100
        self.weights = array([45,11000,45,3,1000,1/600,1,5,10,2,1/30])/100
        #self.weights = array([45,11000,45,3,1000,1/600,100,100,50,20,1/3])/100
        
        
        # car vector with continuous and integer variables
        self.vector = []
        
        # continuous parameters with fixed bounds
        for i in range(19):
            temp = uniform(params.at[i, 'min']+(params.at[i, 'max']-params.at[i, 'min'])*0.05, params.at[i, 'max']-(params.at[i, 'max']-params.at[i, 'min'])*0.25) #random; shaving off the edges to reduce disastrous starters
            #temp = (params.at[i, 'min']+ params.at[i, 'max'])/2 #same
            setattr(self, params.at[i, 'variable'], temp)
            self.vector.append(temp)
            
        # integer parameters
        # materials  
        for i in range(5):
            temp = randint(0,self.num_materials) #random
            #temp = 6 #same
            setattr(self, params.at[19+i, 'variable'], materials.at[temp,'q'])
            self.vector.append(temp)
        setattr(self, 'Eia', materials.at[temp,'E']) 
        
                # rear tires
        setattr(self, 'rear_tire', randint(0,self.num_tires)) #random
        #setattr(self, 'rear_tire', 3) #same
        setattr(self, params.at[25, 'variable'], tires.at[self.rear_tire,'radius'])
        setattr(self, params.at[26, 'variable'], tires.at[self.rear_tire,'mass'])
        self.vector.append(self.rear_tire)
        
                # front tires
        setattr(self, 'front_tire', randint(0,self.num_tires)) #random
        #setattr(self, 'front_tire', 3) #same
        setattr(self, params.at[27, 'variable'], tires.at[self.front_tire,'radius'])
        setattr(self, params.at[28, 'variable'], tires.at[self.front_tire,'mass'])
        self.vector.append(self.front_tire)
        
                # engine
        setattr(self, 'engine', randint(0,self.num_engines)) #random
        #setattr(self, 'engine', 10) #same
        setattr(self, params.at[29, 'variable'], motors.at[self.engine,'Power'])
        setattr(self, params.at[30, 'variable'], motors.at[self.engine,'Length'])
        setattr(self, params.at[31, 'variable'], motors.at[self.engine,'Height'])
        setattr(self, params.at[32, 'variable'], motors.at[self.engine,'Torque'])
        setattr(self, params.at[33, 'variable'], motors.at[self.engine,'Mass'])
        self.vector.append(self.engine)
        
                # brakes
        setattr(self, 'brakes', randint(0,self.num_brakes)) #random
        setattr(self, params.at[34, 'variable'], brakes.at[self.brakes,'rbrk'])
        setattr(self, params.at[35, 'variable'], brakes.at[self.brakes,'qbrk'])
        setattr(self, params.at[36, 'variable'], brakes.at[self.brakes,'lbrk'])
        setattr(self, params.at[37, 'variable'], brakes.at[self.brakes,'hbrk'])
        setattr(self, params.at[38, 'variable'], brakes.at[self.brakes,'wbrk'])
        setattr(self, params.at[39, 'variable'], brakes.at[self.brakes,'tbrk'])
        self.vector.append(self.brakes)
        
                # suspension
        setattr(self, 'rsp', randint(0,self.num_suspensions))
        setattr(self, params.at[40, 'variable'], suspension.at[self.rsp,'krsp'])
        setattr(self, params.at[41, 'variable'], suspension.at[self.rsp,'crsp'])
        setattr(self, params.at[42, 'variable'], suspension.at[self.rsp,'mrsp'])
        setattr(self, 'fsp', randint(0,self.num_suspensions))
        setattr(self, params.at[43, 'variable'], suspension.at[self.fsp,'kfsp'])
        setattr(self, params.at[44, 'variable'], suspension.at[self.fsp,'cfsp'])
        setattr(self, params.at[45, 'variable'], suspension.at[self.fsp,'mfsp'])
        self.vector.append(self.rsp)
        self.vector.append(self.fsp)
        
                # continuous parameters with variable bounds
        setattr(self, 'wrw', uniform(0.3 + (self.r_track - 2 * self.rrt - 0.3) * 0.05, self.r_track - 2 * self.rrt - (self.r_track - 2 * self.rrt-0.3)*0.25))
        setattr(self, 'yrw', uniform(.5 + self.hrw / 2, 1.2 - self.hrw / 2))
        setattr(self, 'yfw', uniform(0.03 + self.hfw / 2, .25 - self.hfw/2))
        setattr(self, 'ysw', uniform(0.03 + self.hsw/2, .250 - self.hsw/2))
        setattr(self, 'ye', uniform(0.03 + self.he / 2, .5 - self.he / 2))
        setattr(self, 'yc', uniform(0.03 + self.hc / 2, 1.200 - self.hc / 2))
        setattr(self, 'lia', uniform(0.2, .7  - self.lfw))
        setattr(self, 'yia', uniform(0.03 + self.hia / 2, 1.200 - self.hia / 2))
        setattr(self, 'yrsp', uniform(self.rrt, self.rrt * 2))
        setattr(self, 'yfsp', uniform(self.rft, self.rft * 2))
        
        for i in range(10):
            temp = getattr(self, params.at[46+i, 'variable'])
            self.vector.append(temp)
        
        self.params=params
        self.materials=materials
        self.tires=tires
        self.motors=motors
        self.brakes=brakes
        self.suspension=suspension

        
        
        self.number_of_subproblems = 2
        #self.num_agent_states = [? ? ?]
        self.learn_rates = np.asarray([0.01,0.01])
        self.actions_per_subproblem = np.asarray([36,44])
        self.temps = np.asarray([10000,10000])
        #WEIGHTS EXCEPTIONS: end of all wings (1*1e-8,length), end of cabin (1*1e-8,length)
        self.pareto_weights = [np.array([1/40,-1/40,1/4,1/0.6,1*1e-8,1/15,-1/5,1/1.5,1/0.1,1*1e-8,1/4,-1/0.8,1/0.5,1/0.1,1*1e-8,1*1e-8]),
                               np.array([1/4.5,1/4.5,1/10,-1/2e-5,1/20,1/10,1/.5,1/10,-1/30,1/.05,1/10000,1*1e-8,1*1e-8,1/35,1/.025,1/250000,1/0.6,1*1e-8,1/.25,1*1e-8,1/.25,1*1e-8])]

        
        self.target_weights = [np.array([]),
                               np.array([1/.1,1/.25,1/.1,1/.25,1/1200,1/1200])]

        
        #Subproblems are given as a bunch of classes 
        self.subproblems = [self.vector[0:3]+self.vector[19:20]+self.vector[30:32]+self.vector[3:7]+self.vector[20:21]+self.vector[32:33]+self.vector[7:11]+self.vector[21:22]+self.vector[33:34], #wings
                            self.vector[11:12]+self.vector[24:25]+self.vector[12:13]+self.vector[25:26]+self.vector[27:28]+self.vector[13:17]+self.vector[22:23]+self.vector[35:36]+self.vector[26:27]+self.vector[34:35]+self.vector[17:19]+self.vector[23:24]+self.vector[36:38]+self.vector[28:29]+self.vector[38:39]+self.vector[29:30]+self.vector[39:40]] #rest of the car
        
        self.design_props = []
        self.design_targets = []
        for i in range(self.number_of_subproblems):
            props, targets = self.get_props(i,self.subproblems[i])
            self.design_props.append(props)
            self.design_targets.append(targets)

        
        #bases give shape of possible needs for each subproblem, organized by agent type index
        self.needs_bases = ([np.ma.masked_equal(np.zeros_like(self.design_props[i]),0) for i in range(self.number_of_subproblems)])

        self.target_need_bases = ([np.ma.masked_equal(np.stack((np.zeros_like(self.design_targets[i]),np.zeros_like(self.design_targets[i])),axis=-1),0) for i in range(self.number_of_subproblems)]) #create target needs as upper and lower bounds
        self.subproblem_needs = ([copy.deepcopy(self.needs_bases) for i in range(self.number_of_subproblems)]) #create table...
        self.target_needs = ([copy.deepcopy(self.target_need_bases) for i in range(self.number_of_subproblems)]) #create table...
        
        

        #self.target_needs = np.stack((self.target_needs,self.target_needs),axis=-1) #split it into upper/lower bounds
        #zero out self-needs
        self.subproblem_goals = copy.deepcopy(self.subproblem_needs)
        self.target_goals = copy.deepcopy(self.target_needs)
        for i in range(self.number_of_subproblems):
            self.subproblem_needs[i][i] = np.ma.asarray([])
            self.target_needs[i][i] = np.ma.asarray([])
        
        self.target_needs[0][1][1,1] = (self.r_track-self.subproblems[0][4])/2
        self.subproblem_needs[1][0][9] = 0.7-self.subproblems[1][16]
        

    def get_props(self, subproblem_id, subproblem):
        #calculation of subproblem results; last entry should be the global objective? First entry?
        
        props = np.asarray([])
        targets = np.asarray([])
        
        if (subproblem_id == 0): #wings (rear, front,side)
            #rear wing variable order: height, length, angle of attack, material, width, y position
            r_density = self.materials.at[subproblem[3],'q']
            r_mass = subproblem[0]*subproblem[1]*subproblem[4]*r_density
            r_downforce = self.F_down_wing(subproblem[4],subproblem[0],subproblem[1],subproblem[2],self.rho_air,self.v_car)
            r_drag = self.F_drag_wing(subproblem[4],subproblem[0],subproblem[1],subproblem[2],self.rho_air,self.v_car)
            r_CG_y = subproblem[5]
            
            #variable order: height, length, width, angle of attack, material, y position
            f_density = self.materials.at[subproblem[10],'q']
            f_mass = subproblem[6]*subproblem[7]*subproblem[8]*f_density
            f_downforce = self.F_down_wing(subproblem[8],subproblem[6],subproblem[7],subproblem[9],self.rho_air,self.v_car)
            f_drag = self.F_drag_wing(subproblem[8],subproblem[6],subproblem[7],subproblem[9],self.rho_air,self.v_car)
            f_CG_y = subproblem[11]
            
            #variable order: height, length, width, angle of attack, material, y position
            s_density = self.materials.at[subproblem[16],'q']
            s_mass = 2*subproblem[12]*subproblem[13]*subproblem[14]*s_density
            s_downforce = 2*self.F_down_wing(subproblem[14],subproblem[12],subproblem[13],subproblem[15],self.rho_air,self.v_car)
            s_drag = 2*self.F_drag_wing(subproblem[14],subproblem[12],subproblem[13],subproblem[15],self.rho_air,self.v_car)
            s_CG_y = subproblem[17]
            
            props = np.asarray([r_mass,r_downforce,r_drag,r_CG_y,subproblem[1],f_mass,f_downforce,f_drag,f_CG_y,subproblem[7],s_mass,s_downforce,s_drag,s_CG_y,subproblem[13],subproblem[4]])

        elif (subproblem_id == 1): #rest of the car!
            #tires+brakes
            #order: rear tire pressure, rear tire selection, front tire pressure, front tire selection, brakes selection
            rear_pressure = subproblem[0]
            r_radius, r_mass = self.tire_props(subproblem[1])
            front_pressure = subproblem[2]
            f_radius,f_mass = self.tire_props(subproblem[3])
            b_radius, b_density, b_length,b_height,b_width,b_thickness = self.brake_props(subproblem[4])
            b_area = b_radius*b_width*b_height
            b_mass = b_length*b_width*b_height*b_density
            
            #cabin/motor
            #order: cabin height, cabin length, cabin width, cabin thickness,cabin material, cabin y pos, engine type, engine y pos
            c_density = self.materials.at[subproblem[9],'q']
            t1 = subproblem[5]*subproblem[6]*subproblem[8] 
            t2 = subproblem[5]*subproblem[7]*subproblem[8]
            t3 = subproblem[6]*subproblem[7]*subproblem[8]
            c_mass = 2*(t1+t2+t3)*c_density
            c_drag = self.F_drag(subproblem[7],subproblem[5],self.rho_air,self.v_car,self.C_dc)
            cabin_cgy = subproblem[10]
            m_pow, m_l,m_w, m_h, m_torque, m_mass = self.motor_props(subproblem[11])
            m_cgy = subproblem[12]
            
            #IA/suspensions
            #order: IA height, IA width, IA material, IA length, IA y pos, rsp type, rsp y pos, fsp type, fsp y pos
            ia_density = self.materials.at[subproblem[15],'q']
            ia_stiffness = self.materials.at[subproblem[15],'E']
            ia_volume = subproblem[13]*subproblem[14]*subproblem[16]
            ia_mass = ia_volume*ia_density
            norm_crashforce =  sqrt( self.v_car**2 * subproblem[14] * subproblem[13] * ia_stiffness / (2*subproblem[16]))
            ia_cgy = subproblem[17]
            
            fsp_k, fsp_c, fsp_m = self.suspension_props(subproblem[20])
            fsp_cgy = subproblem[21]
            fsp_f = self.suspensionForce(fsp_k,fsp_c)
            rsp_k, rsp_c, rsp_m = self.suspension_props(subproblem[18])
            rsp_cgy = subproblem[19]
            rsp_f = self.suspensionForce(rsp_k,rsp_c)
            

            props = np.asarray([2*r_mass, 2*f_mass, 4*b_mass, b_area, c_mass,c_drag, cabin_cgy, m_mass,m_torque,m_cgy,m_pow,subproblem[6],subproblem[7],ia_mass, ia_volume, norm_crashforce, ia_cgy,rsp_m, rsp_cgy,fsp_m,fsp_cgy,subproblem[16]])
            targets = np.asarray([rear_pressure, r_radius, front_pressure,f_radius,rsp_f,fsp_f])
            
        else:
            raise Exception('invalid subproblem!')

        return props, targets
    
    def get_constraints(self,subproblem_id,subproblem):
        
        
        if (subproblem_id == 0): #wings
            #rear wing variable order: height, length, angle of attack, density, width, y position
            results = np.asarray([np.maximum(subproblem[5]-(1.2-subproblem[0]/2),0),
                                  np.minimum(subproblem[5]-(0.5+subproblem[0]/2),0),
                                  np.maximum(subproblem[11]-(0.25-subproblem[6]/2),0),
                                  np.minimum(subproblem[11]-(0.03+subproblem[6]/2),0),
                                  np.maximum(subproblem[17]-(0.25-subproblem[12]/2),0),
                                  np.minimum(subproblem[17]-(0.03+subproblem[12]/2),0)])
            
            results = np.sum(np.abs(results))

        elif (subproblem_id == 1): #rest of the car
            #order: rear tire pressure, rear tire selection, front tire pressure, front tire selection, brakes selection
            
            #order: cabin height, cabin length, cabin width, cabin thickness,cabin material, cabin y pos, engine type, engine y pos

            height_engine = self.motors.at[subproblem[11],'Height']
            r_radius, r_mass = self.tire_props(subproblem[1])
            f_radius,f_mass = self.tire_props(subproblem[3])
            
            results = np.asarray([np.maximum(subproblem[10]-(1.2-subproblem[5]/2),0),
                                  np.minimum(subproblem[10]-(0.03+subproblem[5]/2),0),
                                  np.maximum(subproblem[12]-(0.5-height_engine/2),0),
                                  np.minimum(subproblem[12]-(0.03+height_engine/2),0),
                                  np.maximum(subproblem[17]-(1.2-subproblem[13]/2),0),
                                  np.minimum(subproblem[17]-(0.03+subproblem[13]/2),0),
                                  np.maximum(subproblem[19]-r_radius*2,0), #rsp
                                  np.minimum(subproblem[19]-r_radius,0),
                                  np.maximum(subproblem[21]-f_radius*2,0),#fsp
                                  np.minimum(subproblem[21]-f_radius,0)]) 
            

            results = np.sum(np.abs(results))

            
        return results

            
        # continuous parameters with variable bounds
        #setattr(self, 'wrw', uniform(0.3, self.r_track - 2 * self.rrt)) # relies on tires - implement as need
        #setattr(self, 'lia', uniform(0.2, .7  - self.lfw)) #relies on front wing
        #setattr(self, 'yrsp', uniform(self.rrt, self.rrt * 2)) #relies on tires
        #setattr(self, 'yfsp', uniform(self.rft, self.rft * 2))
    
    
    def apply_move(self,subproblem_id,move_id,subproblem_t,needs_t,target_needs_t):
        subproblem = copy.deepcopy(subproblem_t)
        needs = copy.deepcopy(needs_t)
        target_needs = copy.deepcopy(target_needs_t)
        
        #get a function back from the agent id/move id, and use it to make changes to subproblems and then returns them
        
        if subproblem_id == 0: #rear wing
            #rear wing variable order: height, length, angle of attack, material, width, y position
            if move_id == 0:
                subproblem[0] = self.continuous_increase(subproblem[0],0.7)
            elif move_id == 1:
                subproblem[0] = self.continuous_decrease(subproblem[0],0.025)
            elif move_id == 2:
                subproblem[1] = self.continuous_increase(subproblem[1],0.25)
            elif move_id == 3:
                subproblem[1] = self.continuous_decrease(subproblem[1],0.05)
            elif move_id == 4:
                subproblem[2] = self.continuous_increase(subproblem[2],0.785398)
            elif move_id == 5:
                subproblem[2] = self.continuous_decrease(subproblem[2],0)
            elif move_id == 6:
                subproblem[3] = self.discrete_increase(subproblem[3],self.num_materials)
            elif move_id == 7:
                subproblem[3] = self.discrete_decrease(subproblem[3],0)
            elif move_id == 8:
                subproblem[4] = self.continuous_increase(subproblem[4],self.r_track-2*self.tires.at[0,'radius']) #width of rear wing, relies on tires!, using upper bound
                target_needs[1][1,1] = (self.r_track-subproblem[4])/2
            elif move_id == 9:
                subproblem[4] = self.continuous_decrease(subproblem[4],0.3)
                target_needs[1][1,1] = (self.r_track-subproblem[4])/2
            elif move_id == 10:
                subproblem[5] = self.continuous_increase(subproblem[5],1.2-subproblem[0]/2)
            elif move_id == 11:
                subproblem[5] = self.continuous_decrease(subproblem[5],0.5+subproblem[0]/2)
                
            elif move_id == 12:
                subproblem[6] = self.continuous_increase(subproblem[6],0.2)
            elif move_id == 13:
                subproblem[6] = self.continuous_decrease(subproblem[6],0.025)
            elif move_id == 14:
                subproblem[7] = self.continuous_increase(subproblem[7],0.5)
            elif move_id == 15:
                subproblem[7] = self.continuous_decrease(subproblem[7],0.05)
            elif move_id == 16:
                subproblem[8] = self.continuous_increase(subproblem[8],3)
            elif move_id == 17:
                subproblem[8] = self.continuous_decrease(subproblem[8],0.3)
            elif move_id == 18:
                subproblem[10] = self.discrete_increase(subproblem[10],self.num_materials)
            elif move_id == 19:
                subproblem[10] = self.discrete_decrease(subproblem[10],0)
            elif move_id == 20:
                subproblem[9] = self.continuous_increase(subproblem[9],0.785398) 
            elif move_id == 21:
                subproblem[9] = self.continuous_decrease(subproblem[9],0)
            elif move_id == 22:
                subproblem[11] = self.continuous_increase(subproblem[11],0.25-subproblem[6]/2)
            elif move_id == 23:
                subproblem[11] = self.continuous_decrease(subproblem[11],0.03+subproblem[6]/2)
                
            elif move_id == 24:
                subproblem[12] = self.continuous_increase(subproblem[12],0.2)
            elif move_id == 25:
                subproblem[12] = self.continuous_decrease(subproblem[12],0.025)
            elif move_id == 26:
                subproblem[13] = self.continuous_increase(subproblem[13],0.55)
            elif move_id == 27:
                subproblem[13] = self.continuous_decrease(subproblem[13],0.05)
            elif move_id == 28:
                subproblem[14] = self.continuous_increase(subproblem[14],0.4)
            elif move_id == 29:
                subproblem[14] = self.continuous_decrease(subproblem[14],0.05)
            elif move_id == 30:
                subproblem[16] = self.discrete_increase(subproblem[16],self.num_materials)
            elif move_id == 31:
                subproblem[16] = self.discrete_decrease(subproblem[16],0)
            elif move_id == 32:
                subproblem[15] = self.continuous_increase(subproblem[15],0.785398) 
            elif move_id == 33:
                subproblem[15] = self.continuous_decrease(subproblem[15],0)
            elif move_id == 34:
                subproblem[17] = self.continuous_increase(subproblem[17],0.25-subproblem[12]/2)
            elif move_id == 35:
                subproblem[17] = self.continuous_decrease(subproblem[17],0.03+subproblem[12]/2)
           
        elif subproblem_id == 1: # rest of the car
            #order: rear tire pressure, rear tire selection, front tire pressure, front tire selection, brakes selection
            if move_id == 0:
                subproblem[0] = self.continuous_increase(subproblem[0],1.03)
            elif move_id == 1:
                subproblem[0] = self.continuous_decrease(subproblem[0],0.758)
            elif move_id == 2:
                subproblem[1] = self.discrete_increase(subproblem[1],self.num_tires)
            elif move_id == 3:
                subproblem[1] = self.discrete_decrease(subproblem[1],0)
            elif move_id == 4:
                subproblem[2] = self.continuous_increase(subproblem[2],1.03)
            elif move_id == 5:
                subproblem[2] = self.continuous_decrease(subproblem[2],0.758)
            elif move_id == 6:
                subproblem[3] = self.discrete_increase(subproblem[3],self.num_tires)
            elif move_id == 7:
                subproblem[3] = self.discrete_decrease(subproblem[3],0)
            elif move_id == 8:
                subproblem[4] = self.discrete_increase(subproblem[4],self.num_brakes) 
            elif move_id == 9:
                subproblem[4] = self.discrete_decrease(subproblem[4],0)
                
             #order: cabin height, cabin length, cabin width, cabin thickness,cabin material, cabin y pos, engine type, engine y pos
            elif move_id == 10:
                subproblem[5] = self.continuous_increase(subproblem[5],1.17)
            elif move_id == 11:
                subproblem[5] = self.continuous_decrease(subproblem[5],0.5)
            elif move_id == 12:
                subproblem[6] = self.continuous_increase(subproblem[6],3)
            elif move_id == 13:
                subproblem[6] = self.continuous_decrease(subproblem[6],1.525)
            elif move_id == 14:
                subproblem[7] = self.continuous_increase(subproblem[7],2)
            elif move_id == 15:
                subproblem[7] = self.continuous_decrease(subproblem[7],0.5)
            elif move_id == 16:
                subproblem[9] = self.discrete_increase(subproblem[9],self.num_materials)
            elif move_id == 17:
                subproblem[9] = self.discrete_decrease(subproblem[9],0)
            elif move_id == 18:
                subproblem[8] = self.continuous_increase(subproblem[8],0.01) 
            elif move_id == 19:
                subproblem[8] = self.continuous_decrease(subproblem[8],0.0001)
            elif move_id == 20:
                subproblem[10] = self.continuous_increase(subproblem[10],1.2-subproblem[5]/2)
            elif move_id == 21:
                subproblem[10] = self.continuous_decrease(subproblem[10],0.03+subproblem[5]/2)
            elif move_id == 22:
                subproblem[11] = self.discrete_increase(subproblem[11],self.num_engines)
            elif move_id == 23:
                subproblem[11] = self.discrete_decrease(subproblem[11],0)
            elif move_id == 24:
                height_engine = self.motors.at[subproblem[11],'Height']
                subproblem[12] = self.continuous_increase(subproblem[12],0.5-height_engine/2)
            elif move_id == 25:
                height_engine = self.motors.at[subproblem[11],'Height']
                subproblem[12] = self.continuous_decrease(subproblem[12],0.03+height_engine/2)
                
                #order: IA height, IA width, IA material, IA length, IA y pos, rsp type, rsp y pos, fsp type, fsp y pos
            elif move_id == 26:
                subproblem[13] = self.continuous_increase(subproblem[13],0.5)
            elif move_id == 27:
                subproblem[13] = self.continuous_decrease(subproblem[13],0.1)
            elif move_id == 28:
                subproblem[14] = self.continuous_increase(subproblem[14],0.5)
            elif move_id == 29:
                subproblem[14] = self.continuous_decrease(subproblem[14],0.2)
            elif move_id == 30:
                subproblem[16] = self.continuous_increase(subproblem[16],0.7 - 0.05) #depends on front wing, using upper bound
                needs[0][9] = 0.7-subproblem[16]
            elif move_id == 31:
                subproblem[16] = self.continuous_decrease(subproblem[16],0.2)
                needs[0][9] = 0.7-subproblem[16]
            elif move_id == 32:
                subproblem[15] = self.discrete_increase(subproblem[15],self.num_materials)
            elif move_id == 33:
                subproblem[15] = self.discrete_decrease(subproblem[15],0)
            elif move_id == 34:
                subproblem[17] = self.continuous_increase(subproblem[17],1.2 - subproblem[0]/2)
            elif move_id == 35:
                subproblem[17] = self.continuous_decrease(subproblem[17],0.03+ subproblem[0]/2)
            elif move_id == 36:
                subproblem[18] = self.discrete_increase(subproblem[18],self.num_suspensions)
            elif move_id == 37:
                subproblem[18] = self.discrete_decrease(subproblem[18],0)
            elif move_id == 38:
                subproblem[19] = self.continuous_increase(subproblem[19],2*self.tires.at[subproblem[1],'radius'])
            elif move_id == 39:
                subproblem[19] = self.continuous_decrease(subproblem[19],self.tires.at[subproblem[1],'radius'])
            elif move_id == 40:
                subproblem[20] = self.discrete_increase(subproblem[20],self.num_suspensions)
            elif move_id == 41:
                subproblem[20] = self.discrete_decrease(subproblem[20],0)
            elif move_id == 42:
                subproblem[21] = self.continuous_increase(subproblem[21],2*self.tires.at[subproblem[3],'radius'])
            elif move_id == 43:
                subproblem[21] = self.continuous_decrease(subproblem[21],self.tires.at[subproblem[3],'radius'])

        return subproblem,needs,target_needs
    
    #possible todo - include a weak objective to give continuous changes some directed optimization
    #find some way to shrink the search space based on variance?
    def continuous_increase(self, current_value,max_value = np.nan):
        if max_value == np.nan:
            return current_value*(1+0.2*np.random.random())
        else: 
            return current_value + (max_value-current_value)*np.random.random()
    
    def continuous_decrease(self, current_value,min_value = np.nan):
        if min_value == np.nan:
            return current_value*(1-0.2*np.random.random())
        else:
            return current_value + (min_value - current_value)*np.random.random()
        
    def discrete_increase(self,current_value, max_value):
        if current_value >= max_value-1:
            return max_value
        else:
            return current_value+np.random.randint(1,max_value-current_value)
    
    def discrete_decrease(self,current_value, min_value):
        if current_value <= min_value+1:
            return min_value
        else:
            return current_value-np.random.randint(1,current_value-min_value)
        
    def change_weights(self):
        self.weights = array([90,22000,45,3,1000,1/400,1,5,20,4,1/15])/100
        
        
    def global_objective(self, props, targets):
        #convert props inputs into meaningful variables to make decomposing in different ways easier
        rw_mass = props[0][0]
        rw_downforce = props[0][1]
        rw_drag = props[0][2]
        rw_cgy = props[0][3]
        rw_length = props[0][4]
        
        fw_mass = props[0][5]
        fw_downforce = props[0][6]
        fw_drag = props[0][7]
        fw_cgy = props[0][8]
        fw_length = props[0][9]
        
        sw_mass = props[0][10]
        sw_downforce = props[0][11]
        sw_drag = props[0][12]
        sw_cgy = props[0][13]
        sw_length = props[0][14]
        
        rw_width = props[0][15]
        
        rt_mass = props[1][0]
        ft_mass = props[1][1]
        b_mass = props[1][2]
        b_area = props[1][3]
        rt_pressure = targets[1][0]
        rt_radius = targets[1][1]
        ft_pressure = targets[1][2]
        ft_radius = targets[1][3]
        
        c_mass = props[1][4]
        c_drag = props[1][5]
        c_cgy = props[1][6]
        m_mass = props[1][7]
        m_torque = props[1][8]
        m_cgy = props[1][9]
        m_pow = props[1][10]
        c_length = props[1][11]
        c_width = props[1][12]
        
        ia_mass = props[1][13]
        ia_volume = props[1][14]
        norm_crashforce = props[1][15]
        ia_cgy = props[1][16]
        rsp_mass = props[1][17]
        rsp_cgy = props[1][18]
        fsp_mass = props[1][19]
        fsp_cgy = props[1][20]
        rsp_f = targets[1][4]
        fsp_f = targets[1][5]
        
        ia_length = props[1][21]
        
        
        if rw_width > (self.r_track-2*rt_radius):
            return 10000+rw_width*100
        elif ia_length > 0.7-fw_length:
            return 10000+ia_length*100

        

        weights = self.weights
        
        #mass (lower = better)
        t_1 = rw_mass + fw_mass + sw_mass + rt_mass + ft_mass 
        t_2 = b_mass + c_mass + m_mass + ia_mass + fsp_mass + rsp_mass
        subobj1= t_1+t_2
        
        #center of gravity (lower = better)
        t_1 = rw_mass*rw_cgy + fw_mass*fw_cgy + sw_mass*sw_cgy + rt_mass*rt_radius 
        t_2 = ft_mass*ft_radius + b_mass*ft_radius+b_mass*rt_radius + c_mass*c_cgy
        t_3 = m_mass*m_cgy + ia_mass*ia_cgy + fsp_mass*fsp_cgy + rsp_mass*rsp_cgy
        subobj2= t_1+t_2+t_3
        subobj2 = subobj2/subobj1
        
        #drag
        subobj3 = rw_drag+fw_drag+sw_drag+c_drag
        
        #downforce
        subobj4= rw_downforce+fw_downforce+sw_downforce
        
        #Precomputing total y forces on the car; if theyre less than 0 the car is infeasible
        Fy = subobj4 + subobj1*self.gravity-fsp_f-rsp_f 
        if Fy < 0:
            return 1e6 + (Fy**2)*100
        
        #acceleration
        total_res = subobj3 + self.rollingResistance(rt_pressure,self.v_car,subobj1)
        w_wheels = self.v_car/rt_radius
        #efficiency = total_res*self.v_car/m_pow
        efficiency = 1
        if m_pow == 0 or rt_radius == 0:
            raise Exception('aw shit here we go again')
            return 1e9
        F_wheels = m_torque*efficiency*self.w_e/(rt_radius*w_wheels)
        subobj5=(F_wheels-total_res)/subobj1
        
        #crash force
        subobj6=norm_crashforce*sqrt(subobj1)
        
        #IA volume
        subobj7=ia_volume
        
        #corner velocity

        
        Clat = 1.6
        #tire lateral grip limit
        subobj8=sqrt(Fy*Clat*self.r_track/subobj1)
        #rolling moment limit
        subobj8 = min(subobj8, sqrt(c_width*self.r_track*(subobj1*self.gravity+subobj4)/(2*subobj1*subobj2)))
        
        #braking distance
        C = .005 + 1/ft_pressure * (.01 + .0095 * ((self.v_car*3.6/100)**2))
        Tbrk = 2*0.37*self.P_brk*b_area
        a_brk = Fy * C / subobj1 + 4*Tbrk/(rt_radius*subobj1)
        subobj9 = self.v_car**2/(2*a_brk)

        #suspension acceleration
        subobj10= -(rsp_f - fsp_f - subobj1*self.gravity - subobj4)/subobj1
        
        #pitch moment
        
        subobj11=abs(rsp_f/2+fsp_f/2+rw_downforce*(c_length-rw_length)-fw_downforce*(c_length-fw_length)-sw_downforce*(c_length-sw_length))
        
        #penalties = self.global_obj_constraints(props, subobj1, subobj4)
        

        return(np.array([subobj1*weights[0]+subobj2*weights[1]+subobj3*weights[2]-(subobj4**0.5)*weights[3]-
                      subobj5*weights[4]+subobj6*weights[5]+subobj7*weights[6]-subobj8*weights[7]+
                      subobj9*weights[8]+subobj10*weights[9]+subobj11*weights[10], 
                      subobj1, subobj2, subobj3, subobj4, subobj5, subobj6, subobj7, subobj8, subobj9, subobj10, subobj11]))


    # aspect ratio of wing
    def AR(self,w,alpha,l):
        return w* cos(alpha) / l

    # lift co-effecient
    def C_lift(self,AR,alpha):
        return 2*pi* (AR / (AR + 2)) * alpha
    
    # drag co-efficient
    def C_drag(self,C_lift, AR):
        return C_lift**2 / (pi * AR)
    
    # wing downforce
    def F_down_wing(self,w,h,l,alpha,rho_air,v_car):
        wingAR = self.AR(w,alpha,l)
        C_l = self.C_lift(wingAR, alpha)
        return 0.5 * alpha * h * w * rho_air * (v_car**2) * C_l
    
    # wing drag
    def F_drag_wing(self,w,h,l,alpha,rho_air,v_car):
        wingAR = self.AR(w,alpha,l)
        C_l = self.C_lift(wingAR, alpha)
        C_d = self.C_drag(C_l,wingAR)
        return self.F_drag(w,h,rho_air,v_car,C_d)
    
    # drag
    def F_drag(self,w,h,rho_air,v_car,C_d):
        return 0.5*w*h*rho_air*v_car**2*C_d
    
    def suspensionForce(self,k,c):
        return k*self.y_suspension + c*self.dydt_suspension
    
    # rolling resistance
    def rollingResistance(self,tirePressure,v_car,mass):
        C = .005 + 1/tirePressure * (.01 + .0095 * ((v_car*3.6/100)**2))
        return C * mass * self.gravity

    def tire_props(self, val):
        return [self.tires.at[val,'radius'],self.tires.at[val,'mass']]
    def motor_props(self,val):
        return [self.motors.at[val,'Power'], self.motors.at[val,'Length'], self.motors.at[val,'Width'],self.motors.at[val,'Height'],self.motors.at[val,'Torque'],self.motors.at[val,'Mass']]
    def brake_props(self,val):
        return [self.brakes.at[val,'rbrk'],self.brakes.at[val,'qbrk'],self.brakes.at[val,'lbrk'],self.brakes.at[val,'hbrk'],self.brakes.at[val,'wbrk'],self.brakes.at[val,'tbrk']]
    def suspension_props(self,val):
        return [self.suspension.at[val,'krsp'],self.suspension.at[val,'crsp'],self.suspension.at[val,'mrsp']]
    
    
    def bounds_constraint(self, lb, ub, value):
        if value < lb:
            return lb-value
        elif value > ub:
            return value-ub
        else:
            return 0
        
        
class sae_problem_high_decomp():

    #4 or more specialists for SAE problem - 11 total roles with some that may be chunked together
    def __init__(self):
        
        SAEdir = "SAE-fmincon-master\\SAE"
        
        # read data into dataframes
        params=read_excel(SAEdir+"\\resources\\params.xlsx", engine='openpyxl')
        materials=read_csv(SAEdir+"\\resources\\materials.csv")
        tires=read_csv(SAEdir+"\\resources\\tires.csv")
        motors=read_csv(SAEdir+"\\resources\\motors.csv")
        brakes=read_csv(SAEdir+"\\resources\\brakes.csv")
        suspension=read_csv(SAEdir+"\\resources\\suspension.csv")

        
            
        # constants
        self.v_car = 26.8 # m/s
        self.w_e = 3600*2*pi/60 # radians/sec
        self.rho_air = 1.225 # kg/m3
        self.r_track = 9 # m
        self.P_brk = 10**7 # Pascals
        self.C_dc = 0.04 # drag coefficient of cabin
        self.gravity = 9.81 # m/s^2
        self.y_suspension = 0.05 # m
        self.dydt_suspension = 0.025 # m/s
        self.num_materials = 12
        self.num_tires = 6
        self.num_engines = 20
        self.num_brakes = 33
        self.num_suspensions = 4
        
        self.constraint_penalty = 100
        self.weights = array([45,11000,45,3,1000,1/600,1,5,10,2,1/30])/100
        #self.weights = array([45,11000,45,3,1000,1/600,100,100,50,20,1/3])/100
        
        
        # car vector with continuous and integer variables
        self.vector = []
        
        # continuous parameters with fixed bounds
        for i in range(19):
            temp = uniform(params.at[i, 'min']+(params.at[i, 'max']-params.at[i, 'min'])*0.05, params.at[i, 'max']-(params.at[i, 'max']-params.at[i, 'min'])*0.25) #random; shaving off the edges to reduce disastrous starters
            #temp = (params.at[i, 'min']+ params.at[i, 'max'])/2 #same
            setattr(self, params.at[i, 'variable'], temp)
            self.vector.append(temp)
            
        # integer parameters
        # materials  
        for i in range(5):
            temp = randint(0,self.num_materials) #random
            #temp = 6 #same
            setattr(self, params.at[19+i, 'variable'], materials.at[temp,'q'])
            self.vector.append(temp)
        setattr(self, 'Eia', materials.at[temp,'E']) 
        
                # rear tires
        setattr(self, 'rear_tire', randint(0,self.num_tires)) #random
        #setattr(self, 'rear_tire', 3) #same
        setattr(self, params.at[25, 'variable'], tires.at[self.rear_tire,'radius'])
        setattr(self, params.at[26, 'variable'], tires.at[self.rear_tire,'mass'])
        self.vector.append(self.rear_tire)
        
                # front tires
        setattr(self, 'front_tire', randint(0,self.num_tires)) #random
        #setattr(self, 'front_tire', 3) #same
        setattr(self, params.at[27, 'variable'], tires.at[self.front_tire,'radius'])
        setattr(self, params.at[28, 'variable'], tires.at[self.front_tire,'mass'])
        self.vector.append(self.front_tire)
        
                # engine
        setattr(self, 'engine', randint(0,self.num_engines)) #random
        #setattr(self, 'engine', 10) #same
        setattr(self, params.at[29, 'variable'], motors.at[self.engine,'Power'])
        setattr(self, params.at[30, 'variable'], motors.at[self.engine,'Length'])
        setattr(self, params.at[31, 'variable'], motors.at[self.engine,'Height'])
        setattr(self, params.at[32, 'variable'], motors.at[self.engine,'Torque'])
        setattr(self, params.at[33, 'variable'], motors.at[self.engine,'Mass'])
        self.vector.append(self.engine)
        
                # brakes
        setattr(self, 'brakes', randint(0,self.num_brakes)) #random
        setattr(self, params.at[34, 'variable'], brakes.at[self.brakes,'rbrk'])
        setattr(self, params.at[35, 'variable'], brakes.at[self.brakes,'qbrk'])
        setattr(self, params.at[36, 'variable'], brakes.at[self.brakes,'lbrk'])
        setattr(self, params.at[37, 'variable'], brakes.at[self.brakes,'hbrk'])
        setattr(self, params.at[38, 'variable'], brakes.at[self.brakes,'wbrk'])
        setattr(self, params.at[39, 'variable'], brakes.at[self.brakes,'tbrk'])
        self.vector.append(self.brakes)
        
                # suspension
        setattr(self, 'rsp', randint(0,self.num_suspensions))
        setattr(self, params.at[40, 'variable'], suspension.at[self.rsp,'krsp'])
        setattr(self, params.at[41, 'variable'], suspension.at[self.rsp,'crsp'])
        setattr(self, params.at[42, 'variable'], suspension.at[self.rsp,'mrsp'])
        setattr(self, 'fsp', randint(0,self.num_suspensions))
        setattr(self, params.at[43, 'variable'], suspension.at[self.fsp,'kfsp'])
        setattr(self, params.at[44, 'variable'], suspension.at[self.fsp,'cfsp'])
        setattr(self, params.at[45, 'variable'], suspension.at[self.fsp,'mfsp'])
        self.vector.append(self.rsp)
        self.vector.append(self.fsp)
        
                # continuous parameters with variable bounds
        setattr(self, 'wrw', uniform(0.3 + (self.r_track - 2 * self.rrt - 0.3) * 0.05, self.r_track - 2 * self.rrt - (self.r_track - 2 * self.rrt-0.3)*0.25))
        setattr(self, 'yrw', uniform(.5 + self.hrw / 2, 1.2 - self.hrw / 2))
        setattr(self, 'yfw', uniform(0.03 + self.hfw / 2, .25 - self.hfw/2))
        setattr(self, 'ysw', uniform(0.03 + self.hsw/2, .250 - self.hsw/2))
        setattr(self, 'ye', uniform(0.03 + self.he / 2, .5 - self.he / 2))
        setattr(self, 'yc', uniform(0.03 + self.hc / 2, 1.200 - self.hc / 2))
        setattr(self, 'lia', uniform(0.2, .7  - self.lfw))
        setattr(self, 'yia', uniform(0.03 + self.hia / 2, 1.200 - self.hia / 2))
        setattr(self, 'yrsp', uniform(self.rrt, self.rrt * 2))
        setattr(self, 'yfsp', uniform(self.rft, self.rft * 2))
        
        for i in range(10):
            temp = getattr(self, params.at[46+i, 'variable'])
            self.vector.append(temp)
        
        self.params=params
        self.materials=materials
        self.tires=tires
        self.motors=motors
        self.brakes=brakes
        self.suspension=suspension

        
        
        self.number_of_subproblems = 11
        #self.num_agent_states = [? ? ?]
        self.learn_rates = np.asarray([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])
        self.actions_per_subproblem = np.asarray([12,12,12,4,4,2,12,4,10,4,4])
        self.temps = np.asarray([10000,10000,10000,10000,10000,10000,10000,10000,10000,10000,10000])
        #WEIGHTS EXCEPTIONS: end of all wings (1*1e-8,length), end of cabin (1*1e-8,length)
        self.pareto_weights = [np.array([1/40,-1/40,1/4,1/0.6,1*1e-8,1*1e-8]),
                               np.array([1/15,-1/5,1/1.5,1/0.1,1*1e-8]),
                               np.array([1/4,-1/0.8,1/0.5,1/0.1,1*1e-8]),
                               np.array([1/4.5]),
                               np.array([1/4.5]),
                               np.array([1/10,-1/2e-5]),
                               np.array([1/20,1/10,1/.5,1*1e-8,1*1e-8]),
                               np.array([1/10,-1/30,1/.05,1/10000]),
                               np.array([1/35,1/.025,1/250000,1/0.6,1*1e-8]),
                               np.array([1*1e-8,1/.25]),
                               np.array([1*1e-8,1/.25])]

        
        self.target_weights = [np.array([]),
                               np.array([]),
                               np.array([]),
                               np.array([1/.1,1/.25]),
                               np.array([1/.1,1/.25]),
                               np.array([]),
                               np.array([]),
                               np.array([]),
                               np.array([]),
                               np.array([1/1200]),
                               np.array([1/1200])]

        
        #Subproblems are given as a bunch of classes 
        self.subproblems = [self.vector[0:3]+self.vector[19:20]+self.vector[30:32], #rear wing
                            self.vector[3:7]+self.vector[20:21]+self.vector[32:33], #front wing
                            self.vector[7:11]+self.vector[21:22]+self.vector[33:34], #side wings
                            self.vector[11:12]+self.vector[24:25],#rear tires
                            self.vector[12:13]+self.vector[25:26],#front tires
                            self.vector[27:28], #brakes
                            self.vector[13:17]+self.vector[22:23]+self.vector[35:36],#cabin
                            self.vector[26:27]+self.vector[34:35], #engine
                            self.vector[17:19]+self.vector[23:24]+self.vector[36:38], #impact attenuator
                            self.vector[28:29]+self.vector[38:39], #rear suspension
                            self.vector[29:30]+self.vector[39:40]] #front suspension
        
        self.design_props = []
        self.design_targets = []
        for i in range(self.number_of_subproblems):
            props, targets = self.get_props(i,self.subproblems[i])
            self.design_props.append(props)
            self.design_targets.append(targets)

        
        #bases give shape of possible needs for each subproblem, organized by agent type index
        self.needs_bases = ([np.ma.masked_equal(np.zeros_like(self.design_props[i]),0) for i in range(self.number_of_subproblems)])

        self.target_need_bases = ([np.ma.masked_equal(np.stack((np.zeros_like(self.design_targets[i]),np.zeros_like(self.design_targets[i])),axis=-1),0) for i in range(self.number_of_subproblems)]) #create target needs as upper and lower bounds
        self.subproblem_needs = ([copy.deepcopy(self.needs_bases) for i in range(self.number_of_subproblems)]) #create table...
        self.target_needs = ([copy.deepcopy(self.target_need_bases) for i in range(self.number_of_subproblems)]) #create table...
        
        

        #self.target_needs = np.stack((self.target_needs,self.target_needs),axis=-1) #split it into upper/lower bounds
        #zero out self-needs
        self.subproblem_goals = copy.deepcopy(self.subproblem_needs)
        self.target_goals = copy.deepcopy(self.target_needs)
        for i in range(self.number_of_subproblems):
            self.subproblem_needs[i][i] = np.ma.asarray([])
            self.target_needs[i][i] = np.ma.asarray([])
        
        self.target_needs[0][3][1,1] = (self.r_track-self.subproblems[0][4])/2
        self.subproblem_needs[8][1][4] = 0.7-self.subproblems[8][3]
        self.target_needs[9][3][1,1] = self.subproblems[9][1]
        self.target_needs[9][3][1,0] = self.subproblems[9][1]/2
        self.target_needs[10][4][1,1] = self.subproblems[10][1]
        self.target_needs[10][4][1,0] = self.subproblems[10][1]/2

       
        

    def get_props(self, subproblem_id, subproblem):
        #calculation of subproblem results; last entry should be the global objective? First entry?
        
        props = np.asarray([])
        targets = np.asarray([])
        
        if (subproblem_id == 0): #rear wing
            #rear wing variable order: height, length, angle of attack, material, width, y position
            density = self.materials.at[subproblem[3],'q']
            mass = subproblem[0]*subproblem[1]*subproblem[4]*density
            downforce = self.F_down_wing(subproblem[4],subproblem[0],subproblem[1],subproblem[2],self.rho_air,self.v_car)
            drag = self.F_drag_wing(subproblem[4],subproblem[0],subproblem[1],subproblem[2],self.rho_air,self.v_car)
            CG_y = subproblem[5]
            props = np.asarray([mass,downforce,drag,CG_y,subproblem[1],subproblem[4]])
        elif (subproblem_id == 1): #front wing
            #variable order: height, length, width, angle of attack, material, y position
            density = self.materials.at[subproblem[4],'q']
            mass = subproblem[0]*subproblem[1]*subproblem[2]*density
            downforce = self.F_down_wing(subproblem[2],subproblem[0],subproblem[1],subproblem[3],self.rho_air,self.v_car)
            drag = self.F_drag_wing(subproblem[2],subproblem[0],subproblem[1],subproblem[3],self.rho_air,self.v_car)
            CG_y = subproblem[5]
            props = np.asarray([mass,downforce,drag,CG_y,subproblem[1]])
        elif (subproblem_id == 2): #side wings
            #variable order: height, length, width, angle of attack, material, y position
            density = self.materials.at[subproblem[4],'q']
            mass = 2*subproblem[0]*subproblem[1]*subproblem[2]*density
            downforce = 2*self.F_down_wing(subproblem[2],subproblem[0],subproblem[1],subproblem[3],self.rho_air,self.v_car)
            drag = 2*self.F_drag_wing(subproblem[2],subproblem[0],subproblem[1],subproblem[3],self.rho_air,self.v_car)
            CG_y = subproblem[5]
            props = np.asarray([mass,downforce,drag,CG_y,subproblem[1]])
        elif (subproblem_id == 3): #rear tires
            #order: rear tire pressure, rear tire selection, front tire pressure, front tire selection, brakes selection
            rear_pressure = subproblem[0]
            r_radius, r_mass = self.tire_props(subproblem[1])
            props = np.asarray([2*r_mass])
            targets = np.asarray([rear_pressure, r_radius])
        elif (subproblem_id == 4): #front tires
            front_pressure = subproblem[0]
            f_radius,f_mass = self.tire_props(subproblem[1])
            props = np.asarray([2*f_mass])
            targets = np.asarray([front_pressure,f_radius])
        elif (subproblem_id == 5): #brakes
            b_radius, b_density, b_length,b_height,b_width,b_thickness = self.brake_props(subproblem[0])
            b_area = b_radius*b_width*b_height
            b_mass = b_length*b_width*b_height*b_density
            props = np.asarray([4*b_mass, b_area])
        elif (subproblem_id == 6): #cabin
            #order: cabin height, cabin length, cabin width, cabin thickness,cabin material, cabin y pos, engine type, engine y pos
            c_density = self.materials.at[subproblem[4],'q']
            t1 = subproblem[0]*subproblem[1]*subproblem[3] 
            t2 = subproblem[0]*subproblem[2]*subproblem[3]
            t3 = subproblem[1]*subproblem[2]*subproblem[3]
            c_mass = 2*(t1+t2+t3)*c_density
            c_drag = self.F_drag(subproblem[2],subproblem[0],self.rho_air,self.v_car,self.C_dc)
            cabin_cgy = subproblem[5]
            props = np.asarray([c_mass,c_drag, cabin_cgy,subproblem[1],subproblem[2]])
        elif (subproblem_id == 7): #engine
            m_pow, m_l,m_w, m_h, m_torque, m_mass = self.motor_props(subproblem[0])
            m_cgy = subproblem[1]
            props = np.asarray([m_mass,m_torque,m_cgy,m_pow])
        elif (subproblem_id == 8): #impact attenuator
            #order: IA height, IA width, IA material, IA length, IA y pos
            ia_density = self.materials.at[subproblem[2],'q']
            ia_stiffness = self.materials.at[subproblem[2],'E']
            ia_volume = subproblem[0]*subproblem[1]*subproblem[3]
            ia_mass = ia_volume*ia_density
            norm_crashforce =  sqrt( self.v_car**2 * subproblem[1] * subproblem[0] * ia_stiffness / (2*subproblem[3]))
            ia_cgy = subproblem[4]
            props = np.asarray([ia_mass, ia_volume, norm_crashforce, ia_cgy,subproblem[3]])
        elif (subproblem_id == 9): #rear suspension
            #order:rsp type, rsp y pos
            rsp_k, rsp_c, rsp_m = self.suspension_props(subproblem[0])
            rsp_cgy = subproblem[1] 
            rsp_f = self.suspensionForce(rsp_k,rsp_c)
            props = np.asarray([rsp_m,rsp_cgy])
            targets = np.asarray([rsp_f])
        elif (subproblem_id == 10): #front suspension
            #order: fsp type, fsp y pos
            fsp_k, fsp_c, fsp_m = self.suspension_props(subproblem[0])
            fsp_cgy = subproblem[1]
            fsp_f = self.suspensionForce(fsp_k,fsp_c)
            props = np.asarray([fsp_m, fsp_cgy])
            targets = np.asarray([fsp_f])
            
        else:
            raise Exception('invalid subproblem!')
            

        
        return props, targets
    
    def get_constraints(self,subproblem_id,subproblem):
        
        
        if (subproblem_id == 0): #rear wing
            #rear wing variable order: height, length, angle of attack, density, width, y position
            results = np.asarray([np.maximum(subproblem[5]-(1.2-subproblem[0]/2),0),np.minimum(subproblem[5]-(0.5+subproblem[0]/2),0)])
            results = np.sum(np.abs(results))
        elif (subproblem_id == 1): #front wing
            #variable order: height, length, width, angle of attack, density, y position
            results = np.asarray([np.maximum(subproblem[5]-(0.25-subproblem[0]/2),0),np.minimum(subproblem[5]-(0.03+subproblem[0]/2),0)])
            results = np.sum(np.abs(results))
        elif (subproblem_id == 2): #side wing
            #variable order: height, length, width, angle of attack, density, y position
            results = np.asarray([np.maximum(subproblem[5]-(0.25-subproblem[0]/2),0),np.minimum(subproblem[5]-(0.03+subproblem[0]/2),0)])
            results = np.sum(np.abs(results))
        elif (subproblem_id == 3): #rear tires
            #order: rear tire pressure, rear tire selection
            results = 0
        elif (subproblem_id == 4): #front tires
            #order: front tire pressure, front tire selection
            results = 0
        elif (subproblem_id == 5): #brakes
            #order:brakes selection
            results = 0
        elif (subproblem_id == 6): #cabin 
            #order: cabin height, cabin length, cabin width, cabin thickness,cabin material, cabin y pos
            results = np.asarray([np.maximum(subproblem[5]-(1.2-subproblem[0]/2),0),np.minimum(subproblem[5]-(0.03+subproblem[0]/2),0)])
            results = np.sum(np.abs(results))
        elif (subproblem_id == 7): #engine
            #order:engine type, engine y pos
            height_engine = self.motors.at[subproblem[0],'Height']
            results = np.sum(np.abs(np.asarray([np.maximum(subproblem[1]-(0.5-height_engine/2),0),np.minimum(subproblem[1]-(0.03+height_engine/2),0)])))

        elif (subproblem_id == 8): #impact attenuator
            #order: IA height, IA width, IA material, IA length, IA y pos
            results = np.asarray([np.maximum(subproblem[4]-(1.2-subproblem[0]/2),0),np.minimum(subproblem[4]-(0.03+subproblem[0]/2),0)])
            results = np.sum(np.abs(results))
        elif (subproblem_id == 9): #rsp
            results = 0
        elif (subproblem_id == 10): #fsp
            results = 0
            
        else:
            raise Exception('invalid subproblem!')
            
        return results

            
        # continuous parameters with variable bounds
        #setattr(self, 'wrw', uniform(0.3, self.r_track - 2 * self.rrt)) # relies on tires - implement as need
        #setattr(self, 'lia', uniform(0.2, .7  - self.lfw)) #relies on front wing
        #setattr(self, 'yrsp', uniform(self.rrt, self.rrt * 2)) #relies on tires
        #setattr(self, 'yfsp', uniform(self.rft, self.rft * 2))
    
    
    def apply_move(self,subproblem_id,move_id,subproblem_t,needs_t,target_needs_t):
        subproblem = copy.deepcopy(subproblem_t)
        needs = copy.deepcopy(needs_t)
        target_needs = copy.deepcopy(target_needs_t)
        
        #get a function back from the agent id/move id, and use it to make changes to subproblems and then returns them
        
        if subproblem_id == 0: #rear wing
            #rear wing variable order: height, length, angle of attack, material, width, y position
            if move_id == 0:
                subproblem[0] = self.continuous_increase(subproblem[0],0.7)
            elif move_id == 1:
                subproblem[0] = self.continuous_decrease(subproblem[0],0.025)
            elif move_id == 2:
                subproblem[1] = self.continuous_increase(subproblem[1],0.25)
            elif move_id == 3:
                subproblem[1] = self.continuous_decrease(subproblem[1],0.05)
            elif move_id == 4:
                subproblem[2] = self.continuous_increase(subproblem[2],0.785398)
            elif move_id == 5:
                subproblem[2] = self.continuous_decrease(subproblem[2],0)
            elif move_id == 6:
                subproblem[3] = self.discrete_increase(subproblem[3],self.num_materials)
            elif move_id == 7:
                subproblem[3] = self.discrete_decrease(subproblem[3],0)
            elif move_id == 8:
                subproblem[4] = self.continuous_increase(subproblem[4],self.r_track-2*self.tires.at[0,'radius']) #width of rear wing, relies on tires!, using upper bound
                target_needs[3][1,1] = (self.r_track-subproblem[4])/2
            elif move_id == 9:
                subproblem[4] = self.continuous_decrease(subproblem[4],0.3)
                target_needs[3][1,1] = (self.r_track-subproblem[4])/2
            elif move_id == 10:
                subproblem[5] = self.continuous_increase(subproblem[5],1.2-subproblem[0]/2)
            elif move_id == 11:
                subproblem[5] = self.continuous_decrease(subproblem[5],0.5+subproblem[0]/2)
        elif subproblem_id == 1: #front wing
            #variable order: height, length, width, angle of attack, material, y position
            if move_id == 0:
                subproblem[0] = self.continuous_increase(subproblem[0],0.2)
            elif move_id == 1:
                subproblem[0] = self.continuous_decrease(subproblem[0],0.025)
            elif move_id == 2:
                subproblem[1] = self.continuous_increase(subproblem[1],0.5)
            elif move_id == 3:
                subproblem[1] = self.continuous_decrease(subproblem[1],0.05)
            elif move_id == 4:
                subproblem[2] = self.continuous_increase(subproblem[2],3)
            elif move_id == 5:
                subproblem[2] = self.continuous_decrease(subproblem[2],0.3)
            elif move_id == 6:
                subproblem[4] = self.discrete_increase(subproblem[4],self.num_materials)
            elif move_id == 7:
                subproblem[4] = self.discrete_decrease(subproblem[4],0)
            elif move_id == 8:
                subproblem[3] = self.continuous_increase(subproblem[3],0.785398) 
            elif move_id == 9:
                subproblem[3] = self.continuous_decrease(subproblem[3],0)
            elif move_id == 10:
                subproblem[5] = self.continuous_increase(subproblem[5],0.25-subproblem[0]/2)
            elif move_id == 11:
                subproblem[5] = self.continuous_decrease(subproblem[5],0.03+subproblem[0]/2)
        elif subproblem_id == 2: #side wings
            #variable order: height, length, width, angle of attack, material, y position
            if move_id == 0:
                subproblem[0] = self.continuous_increase(subproblem[0],0.2)
            elif move_id == 1:
                subproblem[0] = self.continuous_decrease(subproblem[0],0.025)
            elif move_id == 2:
                subproblem[1] = self.continuous_increase(subproblem[1],0.55)
            elif move_id == 3:
                subproblem[1] = self.continuous_decrease(subproblem[1],0.05)
            elif move_id == 4:
                subproblem[2] = self.continuous_increase(subproblem[2],0.4)
            elif move_id == 5:
                subproblem[2] = self.continuous_decrease(subproblem[2],0.05)
            elif move_id == 6:
                subproblem[4] = self.discrete_increase(subproblem[4],self.num_materials)
            elif move_id == 7:
                subproblem[4] = self.discrete_decrease(subproblem[4],0)
            elif move_id == 8:
                subproblem[3] = self.continuous_increase(subproblem[3],0.785398) 
            elif move_id == 9:
                subproblem[3] = self.continuous_decrease(subproblem[3],0)
            elif move_id == 10:
                subproblem[5] = self.continuous_increase(subproblem[5],0.25-subproblem[0]/2)
            elif move_id == 11:
                subproblem[5] = self.continuous_decrease(subproblem[5],0.03+subproblem[0]/2)
        elif subproblem_id == 3: #rear tires
            #order: rear tire pressure, rear tire selection
            if move_id == 0:
                subproblem[0] = self.continuous_increase(subproblem[0],1.03)
            elif move_id == 1:
                subproblem[0] = self.continuous_decrease(subproblem[0],0.758)
            elif move_id == 2:
                subproblem[1] = self.discrete_increase(subproblem[1],self.num_tires)
            elif move_id == 3:
                subproblem[1] = self.discrete_decrease(subproblem[1],0)
        elif subproblem_id == 4: #front tires
            #order: front tire pressure, front tire selection
            if move_id == 0:
                subproblem[0] = self.continuous_increase(subproblem[0],1.03)
            elif move_id == 1:
                subproblem[0] = self.continuous_decrease(subproblem[0],0.758)
            elif move_id == 2:
                subproblem[1] = self.discrete_increase(subproblem[1],self.num_tires)
            elif move_id == 3:
                subproblem[1] = self.discrete_decrease(subproblem[1],0)
        elif subproblem_id == 5: #brakes
            #order: just brakes!
            if move_id == 0:
                subproblem[0] = self.discrete_increase(subproblem[0],self.num_brakes) 
            elif move_id == 1:
                subproblem[0] = self.discrete_decrease(subproblem[0],0)
        elif subproblem_id == 6: #cabin
            #order: cabin height, cabin length, cabin width, cabin thickness,cabin material, cabin y pos
            if move_id == 0:
                subproblem[0] = self.continuous_increase(subproblem[0],1.17)
            elif move_id == 1:
                subproblem[0] = self.continuous_decrease(subproblem[0],0.5)
            elif move_id == 2:
                subproblem[1] = self.continuous_increase(subproblem[1],3)
            elif move_id == 3:
                subproblem[1] = self.continuous_decrease(subproblem[1],1.525)
            elif move_id == 4:
                subproblem[2] = self.continuous_increase(subproblem[2],2)
            elif move_id == 5:
                subproblem[2] = self.continuous_decrease(subproblem[2],0.5)
            elif move_id == 6:
                subproblem[4] = self.discrete_increase(subproblem[4],self.num_materials)
            elif move_id == 7:
                subproblem[4] = self.discrete_decrease(subproblem[4],0)
            elif move_id == 8:
                subproblem[3] = self.continuous_increase(subproblem[3],0.01) 
            elif move_id == 9:
                subproblem[3] = self.continuous_decrease(subproblem[3],0.0001)
            elif move_id == 10:
                subproblem[5] = self.continuous_increase(subproblem[5],1.2-subproblem[0]/2)
            elif move_id == 11:
                subproblem[5] = self.continuous_decrease(subproblem[5],0.03+subproblem[0]/2)
        elif subproblem_id == 7: #engine
            #order: engine type, engine y pos
            if move_id == 0:
                subproblem[0] = self.discrete_increase(subproblem[0],self.num_engines)
            elif move_id == 1:
                subproblem[0] = self.discrete_decrease(subproblem[0],0)
            elif move_id == 2:
                height_engine = self.motors.at[subproblem[0],'Height']
                subproblem[1] = self.continuous_increase(subproblem[1],0.5-height_engine/2)
            elif move_id == 3:
                height_engine = self.motors.at[subproblem[0],'Height']
                subproblem[1] = self.continuous_decrease(subproblem[1],0.03+height_engine/2)
                
        elif subproblem_id == 8: #impact attenuator+suspensions
            #order: IA height, IA width, IA material, IA length, IA y pos, rsp type, rsp y pos, fsp type, fsp y pos
            if move_id == 0:
                subproblem[0] = self.continuous_increase(subproblem[0],0.5)
            elif move_id == 1:
                subproblem[0] = self.continuous_decrease(subproblem[0],0.1)
            elif move_id == 2:
                subproblem[1] = self.continuous_increase(subproblem[1],0.5)
            elif move_id == 3:
                subproblem[1] = self.continuous_decrease(subproblem[1],0.2)
            elif move_id == 4:
                subproblem[3] = self.continuous_increase(subproblem[3],0.7 - 0.05) #depends on front wing, using upper bound
                needs[1][4] = 0.7-subproblem[3]
            elif move_id == 5:
                subproblem[3] = self.continuous_decrease(subproblem[3],0.2)
                needs[1][4] = 0.7-subproblem[3]
            elif move_id == 6:
                subproblem[2] = self.discrete_increase(subproblem[2],self.num_materials)
            elif move_id == 7:
                subproblem[2] = self.discrete_decrease(subproblem[2],0)
            elif move_id == 8:
                subproblem[4] = self.continuous_increase(subproblem[4],1.2 - subproblem[0]/2)
            elif move_id == 9:
                subproblem[4] = self.continuous_decrease(subproblem[4],0.03+ subproblem[0]/2)
                
        elif subproblem_id == 9: #rear suspension
            if move_id == 0:
                subproblem[0] = self.discrete_increase(subproblem[0],self.num_suspensions)
            elif move_id == 1:
                subproblem[0] = self.discrete_decrease(subproblem[0],0)
            elif move_id == 2:
                subproblem[1] = self.continuous_increase(subproblem[1],2*self.tires.at[self.num_tires,'radius'])
                target_needs[3][1,1] = subproblem[1]
                target_needs[3][1,0] = subproblem[1]/2
            elif move_id == 3:
                subproblem[1] = self.continuous_decrease(subproblem[1],self.tires.at[0,'radius'])
                target_needs[3][1,1] = subproblem[1]
                target_needs[3][1,0] = subproblem[1]/2
        elif subproblem_id == 10: #front suspension
            if move_id == 0:
                subproblem[0] = self.discrete_increase(subproblem[0],self.num_suspensions)
            elif move_id == 1:
                subproblem[0] = self.discrete_decrease(subproblem[0],0)
            elif move_id == 2:
                subproblem[1] = self.continuous_increase(subproblem[1],2*self.tires.at[self.num_tires,'radius'])
                target_needs[4][1,1] = subproblem[1]
                target_needs[4][1,0] = subproblem[1]/2
            elif move_id == 3:
                subproblem[1] = self.continuous_decrease(subproblem[1],self.tires.at[0,'radius'])
                target_needs[4][1,1] = subproblem[1]
                target_needs[4][1,0] = subproblem[1]/2
        else:
            raise Exception('invalid subproblem!')            

        return subproblem,needs,target_needs
    
    #possible todo - include a weak objective to give continuous changes some directed optimization
    #find some way to shrink the search space based on variance?
    def continuous_increase(self, current_value,max_value = np.nan):
        if max_value == np.nan:
            return current_value*(1+0.2*np.random.random())
        else: 
            return current_value + (max_value-current_value)*np.random.random()
    
    def continuous_decrease(self, current_value,min_value = np.nan):
        if min_value == np.nan:
            return current_value*(1-0.2*np.random.random())
        else:
            return current_value + (min_value - current_value)*np.random.random()
        
    def discrete_increase(self,current_value, max_value):
        if current_value >= max_value-1:
            return max_value
        else:
            return current_value+np.random.randint(1,max_value-current_value)
    
    def discrete_decrease(self,current_value, min_value):
        if current_value <= min_value+1:
            return min_value
        else:
            return current_value-np.random.randint(1,current_value-min_value)
        
    def change_weights(self):
        self.weights = array([90,22000,45,3,1000,1/400,1,5,20,4,1/15])/100
        
        
    def global_objective(self, props, targets):
        #convert props inputs into meaningful variables to make decomposing in different ways easier
        rw_mass = props[0][0]
        rw_downforce = props[0][1]
        rw_drag = props[0][2]
        rw_cgy = props[0][3]
        rw_length = props[0][4]
        rw_width = props[0][5]
        
        fw_mass = props[1][0]
        fw_downforce = props[1][1]
        fw_drag = props[1][2]
        fw_cgy = props[1][3]
        fw_length = props[1][4]
        
        sw_mass = props[2][0]
        sw_downforce = props[2][1]
        sw_drag = props[2][2]
        sw_cgy = props[2][3]
        sw_length = props[2][4]
        
        rt_mass = props[3][0]
        ft_mass = props[4][0]
        b_mass = props[5][0]
        b_area = props[5][1]
        rt_pressure = targets[3][0]
        rt_radius = targets[3][1]
        ft_pressure = targets[4][0]
        ft_radius = targets[4][1]
        
        c_mass = props[6][0]
        c_drag = props[6][1]
        c_cgy = props[6][2]
        m_mass = props[7][0]
        m_torque = props[7][1]
        m_cgy = props[7][2]
        m_pow = props[7][3]
        c_length = props[6][3]
        c_width = props[6][4]
        
        ia_mass = props[8][0]
        ia_volume = props[8][1]
        norm_crashforce = props[8][2]
        ia_cgy = props[8][3]
        ia_length = props[8][4]
        rsp_mass = props[9][0]
        rsp_cgy = props[9][1]
        fsp_mass = props[10][0]
        fsp_cgy = props[10][1]
        rsp_f = targets[9][0]
        fsp_f = targets[10][0]
        
        
        if rw_width > (self.r_track-2*rt_radius):
            return 10000+rw_width*100
        elif ia_length > 0.7-fw_length:
            return 10000+ia_length*100
        elif rsp_cgy > 2*rt_radius:
            return 10000+rsp_cgy*100
        elif rsp_cgy < rt_radius:
            return 10000-rsp_cgy*100
        elif fsp_cgy > 2*ft_radius:
            return 10000+fsp_cgy*100
        elif fsp_cgy < ft_radius:
            return 10000-fsp_cgy*100
        

        weights = self.weights
        
        #mass (lower = better)
        t_1 = rw_mass + fw_mass + sw_mass + rt_mass + ft_mass 
        t_2 = b_mass + c_mass + m_mass + ia_mass + fsp_mass + rsp_mass
        subobj1= t_1+t_2
        
        #center of gravity (lower = better)
        t_1 = rw_mass*rw_cgy + fw_mass*fw_cgy + sw_mass*sw_cgy + rt_mass*rt_radius 
        t_2 = ft_mass*ft_radius + b_mass*ft_radius+b_mass*rt_radius + c_mass*c_cgy
        t_3 = m_mass*m_cgy + ia_mass*ia_cgy + fsp_mass*fsp_cgy + rsp_mass*rsp_cgy
        subobj2= t_1+t_2+t_3
        subobj2 = subobj2/subobj1
        
        #drag
        subobj3 = rw_drag+fw_drag+sw_drag+c_drag
        
        #downforce
        subobj4= rw_downforce+fw_downforce+sw_downforce
        
        #Precomputing total y forces on the car; if theyre less than 0 the car is infeasible
        Fy = subobj4 + subobj1*self.gravity-fsp_f-rsp_f 
        if Fy < 0:
            return 1e6 + (Fy**2)*100
        
        #acceleration
        total_res = subobj3 + self.rollingResistance(rt_pressure,self.v_car,subobj1)
        w_wheels = self.v_car/rt_radius
        #efficiency = total_res*self.v_car/m_pow
        efficiency = 1
        if m_pow == 0 or rt_radius == 0:
            raise Exception('aw shit here we go again')
            return 1e9
        F_wheels = m_torque*efficiency*self.w_e/(rt_radius*w_wheels)
        subobj5=(F_wheels-total_res)/subobj1
        
        #crash force
        subobj6=norm_crashforce*sqrt(subobj1)
        
        #IA volume
        subobj7=ia_volume
        
        #corner velocity

        
        Clat = 1.6
        #tire lateral grip limit
        subobj8=sqrt(Fy*Clat*self.r_track/subobj1)
        #rolling moment limit
        subobj8 = min(subobj8, sqrt(c_width*self.r_track*(subobj1*self.gravity+subobj4)/(2*subobj1*subobj2)))
        
        #braking distance
        C = .005 + 1/ft_pressure * (.01 + .0095 * ((self.v_car*3.6/100)**2))
        Tbrk = 2*0.37*self.P_brk*b_area
        a_brk = Fy * C / subobj1 + 4*Tbrk/(rt_radius*subobj1)
        subobj9 = self.v_car**2/(2*a_brk)

        #suspension acceleration
        subobj10= -(rsp_f - fsp_f - subobj1*self.gravity - subobj4)/subobj1
        
        #pitch moment
        
        subobj11=abs(rsp_f/2+fsp_f/2+rw_downforce*(c_length-rw_length)-fw_downforce*(c_length-fw_length)-sw_downforce*(c_length-sw_length))
        
        #penalties = self.global_obj_constraints(props, subobj1, subobj4)
        

        return(np.array([subobj1*weights[0]+subobj2*weights[1]+subobj3*weights[2]-(subobj4**0.5)*weights[3]-
                      subobj5*weights[4]+subobj6*weights[5]+subobj7*weights[6]-subobj8*weights[7]+
                      subobj9*weights[8]+subobj10*weights[9]+subobj11*weights[10], 
                      subobj1, subobj2, subobj3, subobj4, subobj5, subobj6, subobj7, subobj8, subobj9, subobj10, subobj11]))


    # aspect ratio of wing
    def AR(self,w,alpha,l):
        return w* cos(alpha) / l

    # lift co-effecient
    def C_lift(self,AR,alpha):
        return 2*pi* (AR / (AR + 2)) * alpha
    
    # drag co-efficient
    def C_drag(self,C_lift, AR):
        return C_lift**2 / (pi * AR)
    
    # wing downforce
    def F_down_wing(self,w,h,l,alpha,rho_air,v_car):
        wingAR = self.AR(w,alpha,l)
        C_l = self.C_lift(wingAR, alpha)
        return 0.5 * alpha * h * w * rho_air * (v_car**2) * C_l
    
    # wing drag
    def F_drag_wing(self,w,h,l,alpha,rho_air,v_car):
        wingAR = self.AR(w,alpha,l)
        C_l = self.C_lift(wingAR, alpha)
        C_d = self.C_drag(C_l,wingAR)
        return self.F_drag(w,h,rho_air,v_car,C_d)
    
    # drag
    def F_drag(self,w,h,rho_air,v_car,C_d):
        return 0.5*w*h*rho_air*v_car**2*C_d
    
    def suspensionForce(self,k,c):
        return k*self.y_suspension + c*self.dydt_suspension
    
    # rolling resistance
    def rollingResistance(self,tirePressure,v_car,mass):
        C = .005 + 1/tirePressure * (.01 + .0095 * ((v_car*3.6/100)**2))
        return C * mass * self.gravity

    def tire_props(self, val):
        return [self.tires.at[val,'radius'],self.tires.at[val,'mass']]
    def motor_props(self,val):
        return [self.motors.at[val,'Power'], self.motors.at[val,'Length'], self.motors.at[val,'Width'],self.motors.at[val,'Height'],self.motors.at[val,'Torque'],self.motors.at[val,'Mass']]
    def brake_props(self,val):
        return [self.brakes.at[val,'rbrk'],self.brakes.at[val,'qbrk'],self.brakes.at[val,'lbrk'],self.brakes.at[val,'hbrk'],self.brakes.at[val,'wbrk'],self.brakes.at[val,'tbrk']]
    def suspension_props(self,val):
        return [self.suspension.at[val,'krsp'],self.suspension.at[val,'crsp'],self.suspension.at[val,'mrsp']]
    
    
    def bounds_constraint(self, lb, ub, value):
        if value < lb:
            return lb-value
        elif value > ub:
            return value-ub
        else:
            return 0
        
       

    


class convex_problem():
    def __init__(self):
        self.number_of_subproblems = 4
        #self.num_agent_states = [? ? ?]
        self.learn_rates = np.asarray([0.01,0.01,0.01,0.01])
        self.actions_per_subproblem = np.asarray([8,8,8,8])
        self.temps = np.asarray([10000,10000,10000,10000])
        #WEIGHTS EXCEPTIONS: end of all wings (1*1e-8,length), end of cabin (1*1e-8,length)
        self.pareto_weights = [np.array([]),
                               np.array([]),
                               np.array([]),
                               np.array([])]

        
        self.target_weights = [np.array([1,1]),
                               np.array([1,1]),
                               np.array([1,1]),
                               np.array([1,1])]

        
        #Subproblems are given as a bunch of classes 
        self.subproblems = [[uniform(-100,100),uniform(-100,100),uniform(-100,100),uniform(-100,100)],
                            [uniform(-100,100),uniform(-100,100),uniform(-100,100),uniform(-100,100)], 
                            [uniform(-100,100),uniform(-100,100),uniform(-100,100),uniform(-100,100)],
                            [uniform(-100,100),uniform(-100,100),uniform(-100,100),uniform(-100,100)]]
        
        self.design_props = []
        self.design_targets = []
        for i in range(self.number_of_subproblems):
            props, targets = self.get_props(i,self.subproblems[i])
            self.design_props.append(props)
            self.design_targets.append(targets)

        
        #bases give shape of possible needs for each subproblem, organized by agent type index
        self.needs_bases = ([np.ma.masked_equal(np.zeros_like(self.design_props[i]),0) for i in range(self.number_of_subproblems)])

        self.target_need_bases = ([np.ma.masked_equal(np.stack((np.zeros_like(self.design_targets[i]),np.zeros_like(self.design_targets[i])),axis=-1),0) for i in range(self.number_of_subproblems)]) #create target needs as upper and lower bounds
        self.subproblem_needs = ([copy.deepcopy(self.needs_bases) for i in range(self.number_of_subproblems)]) #create table...
        self.target_needs = ([copy.deepcopy(self.target_need_bases) for i in range(self.number_of_subproblems)]) #create table...
        
        

        #self.target_needs = np.stack((self.target_needs,self.target_needs),axis=-1) #split it into upper/lower bounds
        #zero out self-needs
        self.subproblem_goals = copy.deepcopy(self.subproblem_needs)
        self.target_goals = copy.deepcopy(self.target_needs)
        for i in range(self.number_of_subproblems):
            self.subproblem_needs[i][i] = np.ma.asarray([])
            self.target_needs[i][i] = np.ma.asarray([])
            
    def get_props(self,subproblem_id,subproblem):
        return np.asarray([]),np.asarray([subproblem[0]*subproblem[1],subproblem[2]*subproblem[3]])

    def get_constraints(self,subproblem_id,subproblem):
        return 0
    
    def apply_move(self,subproblem_id,move_id,subproblem_t,needs_t,target_needs_t):
        subproblem = copy.deepcopy(subproblem_t)
        needs = copy.deepcopy(needs_t)
        target_needs = copy.deepcopy(target_needs_t)
        if move_id == 0:
            subproblem[0] = self.continuous_increase(subproblem[0],100)
        elif move_id == 1:
            subproblem[0] = self.continuous_decrease(subproblem[0],-100)
        elif move_id == 2:
            subproblem[1] = self.continuous_increase(subproblem[1],100)
        elif move_id == 3:
            subproblem[1] = self.continuous_decrease(subproblem[1],-100)
        elif move_id == 4:
            subproblem[2] = self.continuous_increase(subproblem[2],100)
        elif move_id == 5:
            subproblem[2] = self.continuous_decrease(subproblem[2],-100)
        elif move_id == 6:
            subproblem[3] = self.continuous_decrease(subproblem[3],100)
        elif move_id == 7:
            subproblem[3] = self.continuous_decrease(subproblem[3],-100)
            
        return subproblem, needs, target_needs

    def continuous_increase(self, current_value,max_value = np.nan):
        if max_value == np.nan:
            return current_value*(1+0.2*np.random.random())
        else: 
            return current_value + (max_value-current_value)*np.random.random()

    def continuous_decrease(self, current_value,min_value = np.nan):
        if min_value == np.nan:
            return current_value*(1-0.2*np.random.random())
        else:
            return current_value + (min_value - current_value)*np.random.random()

    def discrete_increase(self,current_value, max_value):
        if current_value >= max_value-1:
            return max_value
        else:
            return current_value+np.random.randint(1,max_value-current_value)

    def discrete_decrease(self,current_value, min_value):
        if current_value <= min_value+1:
            return min_value
        else:
            return current_value-np.random.randint(1,current_value-min_value)

    def global_objective(self, props, targets):
        t1 = np.abs(targets[0][0]+targets[0][1]+targets[1][0])
        t2 = np.abs(targets[1][0]+targets[1][1]+targets[2][0])
        t3 = np.abs(targets[2][0]+targets[2][1]+targets[3][0])
        t4 = np.abs(targets[3][0]+targets[3][1]+targets[0][0])
        return t1+t2+t3+t4

            

class sine_parabola_problem():
    def __init__(self):
        self.number_of_subproblems = 4
        #self.num_agent_states = [? ? ?]
        self.learn_rates = np.asarray([0.01,0.01,0.01,0.01])
        self.actions_per_subproblem = np.asarray([8,8,8,8])
        self.temps = np.asarray([10000,10000,10000,10000])
        #WEIGHTS EXCEPTIONS: end of all wings (1*1e-8,length), end of cabin (1*1e-8,length)
        self.pareto_weights = [np.array([]),
                               np.array([]),
                               np.array([]),
                               np.array([])]

        
        self.target_weights = [np.array([1,1]),
                               np.array([1,1]),
                               np.array([1,1]),
                               np.array([1,1])]

        
        #Subproblems are given as a bunch of classes 
        self.subproblems = [[uniform(-100,100),uniform(-100,100),uniform(-100,100),uniform(-100,100)],
                            [uniform(-100,100),uniform(-100,100),uniform(-100,100),uniform(-100,100)], 
                            [uniform(-100,100),uniform(-100,100),uniform(-100,100),uniform(-100,100)],
                            [uniform(-100,100),uniform(-100,100),uniform(-100,100),uniform(-100,100)]]
        
        self.design_props = []
        self.design_targets = []
        for i in range(self.number_of_subproblems):
            props, targets = self.get_props(i,self.subproblems[i])
            self.design_props.append(props)
            self.design_targets.append(targets)

        
        #bases give shape of possible needs for each subproblem, organized by agent type index
        self.needs_bases = ([np.ma.masked_equal(np.zeros_like(self.design_props[i]),0) for i in range(self.number_of_subproblems)])

        self.target_need_bases = ([np.ma.masked_equal(np.stack((np.zeros_like(self.design_targets[i]),np.zeros_like(self.design_targets[i])),axis=-1),0) for i in range(self.number_of_subproblems)]) #create target needs as upper and lower bounds
        self.subproblem_needs = ([copy.deepcopy(self.needs_bases) for i in range(self.number_of_subproblems)]) #create table...
        self.target_needs = ([copy.deepcopy(self.target_need_bases) for i in range(self.number_of_subproblems)]) #create table...
        
        

        #self.target_needs = np.stack((self.target_needs,self.target_needs),axis=-1) #split it into upper/lower bounds
        #zero out self-needs
        self.subproblem_goals = copy.deepcopy(self.subproblem_needs)
        self.target_goals = copy.deepcopy(self.target_needs)
        for i in range(self.number_of_subproblems):
            self.subproblem_needs[i][i] = np.ma.asarray([])
            self.target_needs[i][i] = np.ma.asarray([])
            
    def get_props(self,subproblem_id,subproblem):
        return np.asarray([]),np.asarray([subproblem[0]*subproblem[1],subproblem[2]*subproblem[3]])

    def get_constraints(self,subproblem_id,subproblem):
        return 0
    
    def apply_move(self,subproblem_id,move_id,subproblem_t,needs_t,target_needs_t):
        subproblem = copy.deepcopy(subproblem_t)
        needs = copy.deepcopy(needs_t)
        target_needs = copy.deepcopy(target_needs_t)
        if move_id == 0:
            subproblem[0] = self.continuous_increase(subproblem[0],100)
        elif move_id == 1:
            subproblem[0] = self.continuous_decrease(subproblem[0],-100)
        elif move_id == 2:
            subproblem[1] = self.continuous_increase(subproblem[1],100)
        elif move_id == 3:
            subproblem[1] = self.continuous_decrease(subproblem[1],-100)
        elif move_id == 4:
            subproblem[2] = self.continuous_increase(subproblem[2],100)
        elif move_id == 5:
            subproblem[2] = self.continuous_decrease(subproblem[2],-100)
        elif move_id == 6:
            subproblem[3] = self.continuous_decrease(subproblem[3],100)
        elif move_id == 7:
            subproblem[3] = self.continuous_decrease(subproblem[3],-100)
            
        return subproblem, needs, target_needs

    def continuous_increase(self, current_value,max_value = np.nan):
        if max_value == np.nan:
            return current_value*(1+0.2*np.random.random())
        else: 
            return current_value + (max_value-current_value)*np.random.random()

    def continuous_decrease(self, current_value,min_value = np.nan):
        if min_value == np.nan:
            return current_value*(1-0.2*np.random.random())
        else:
            return current_value + (min_value - current_value)*np.random.random()

    def discrete_increase(self,current_value, max_value):
        if current_value >= max_value-1:
            return max_value
        else:
            return current_value+np.random.randint(1,max_value-current_value)

    def discrete_decrease(self,current_value, min_value):
        if current_value <= min_value+1:
            return min_value
        else:
            return current_value-np.random.randint(1,current_value-min_value)

    def global_objective(self, props, targets):
        t1 = np.abs(targets[0][0]+targets[0][1]+targets[1][0]+targets[3][1])
        t2 = np.abs(targets[1][0]+targets[1][1]+targets[2][0]+targets[0][1])
        t3 = np.abs(targets[2][0]+targets[2][1]+targets[3][0]+targets[1][1])
        t4 = np.abs(targets[3][0]+targets[3][1]+targets[0][0]+targets[2][1])
        return t1**2+t2**2+t3**2+t4**2+10*(sin(t1)+sin(t2)+sin(t3)+sin(t4))

class variable_coord_problem():
    def __init__(self, num_agents = 4):
        self.number_of_subproblems = num_agents
        #self.num_agent_states = [? ? ?]
        
        self.learn_rates = np.ones(self.number_of_subproblems)*0.01
        self.actions_per_subproblem = np.ones(self.number_of_subproblems).astype(int)*8
        self.temps = np.ones(self.number_of_subproblems)*100
        #WEIGHTS EXCEPTIONS: end of all wings (1*1e-8,length), end of cabin (1*1e-8,length)

        self.pareto_weights = [np.array([-1]) for i in range(self.number_of_subproblems)]

        self.target_weights = [np.array([1]) for i in range(self.number_of_subproblems)]
        
        self.weights = [1,1]

        
        #Subproblems are given as a bunch of classes 
        self.subproblems = [[uniform(0,10),uniform(0,10),uniform(0,0.5),uniform(0,0.5)] for i in range(self.number_of_subproblems)]
        
        self.design_props = []
        self.design_targets = []
        for i in range(self.number_of_subproblems):
            props, targets = self.get_props(i,self.subproblems[i])
            self.design_props.append(props)
            self.design_targets.append(targets)

        
        #bases give shape of possible needs for each subproblem, organized by agent type index
        self.needs_bases = ([np.ma.masked_equal(np.zeros_like(self.design_props[i]),0) for i in range(self.number_of_subproblems)])

        self.target_need_bases = ([np.ma.masked_equal(np.stack((np.zeros_like(self.design_targets[i]),np.zeros_like(self.design_targets[i])),axis=-1),0) for i in range(self.number_of_subproblems)]) #create target needs as upper and lower bounds
        self.subproblem_needs = ([copy.deepcopy(self.needs_bases) for i in range(self.number_of_subproblems)]) #create table...
        self.target_needs = ([copy.deepcopy(self.target_need_bases) for i in range(self.number_of_subproblems)]) #create table...
        
        

        #self.target_needs = np.stack((self.target_needs,self.target_needs),axis=-1) #split it into upper/lower bounds
        #zero out self-needs
        self.subproblem_goals = copy.deepcopy(self.subproblem_needs)
        self.target_goals = copy.deepcopy(self.target_needs)
        for i in range(self.number_of_subproblems):
            self.subproblem_needs[i][i] = np.ma.asarray([])
            self.target_needs[i][i] = np.ma.asarray([])
            
            
        self.sign = np.asarray([-1+2*(i%2) for i in range(self.number_of_subproblems-1)])
            
    def get_props(self,subproblem_id,subproblem):
        return np.asarray([subproblem[0]*subproblem[1]]),np.asarray([subproblem[2]*subproblem[3]])

    def get_constraints(self,subproblem_id,subproblem):
        return 0
    
    def apply_move(self,subproblem_id,move_id,subproblem_t,needs_t,target_needs_t):
        subproblem = copy.deepcopy(subproblem_t)
        needs = copy.deepcopy(needs_t)
        target_needs = copy.deepcopy(target_needs_t)
        if move_id == 0:
            subproblem[0] = self.continuous_increase(subproblem[0],10)
        elif move_id == 1:
            subproblem[0] = self.continuous_decrease(subproblem[0],0)
        elif move_id == 2:
            subproblem[1] = self.continuous_increase(subproblem[1],10)
        elif move_id == 3:
            subproblem[1] = self.continuous_decrease(subproblem[1],0)
        elif move_id == 4:
            subproblem[2] = self.continuous_increase(subproblem[2],0.5)
        elif move_id == 5:
            subproblem[2] = self.continuous_decrease(subproblem[2],0)
        elif move_id == 6:
            subproblem[3] = self.continuous_decrease(subproblem[3],0.5)
        elif move_id == 7:
            subproblem[3] = self.continuous_decrease(subproblem[3],0)
            
        return subproblem, needs, target_needs

    def continuous_increase(self, current_value,max_value = np.nan):
        if max_value == np.nan:
            return current_value*(1+0.2*np.random.random())
        else: 
            return current_value + (max_value-current_value)*np.random.random()

    def continuous_decrease(self, current_value,min_value = np.nan):
        if min_value == np.nan:
            return current_value*(1-0.2*np.random.random())
        else:
            return current_value + (min_value - current_value)*np.random.random()

    def discrete_increase(self,current_value, max_value):
        if current_value >= max_value-1:
            return max_value
        else:
            return current_value+np.random.randint(1,max_value-current_value)

    def discrete_decrease(self,current_value, min_value):
        if current_value <= min_value+1:
            return min_value
        else:
            return current_value-np.random.randint(1,current_value-min_value)

    def global_objective(self, props, targets):
        #set up arrays
        #arr_1 = np.zeros(self.number_of_subproblems)
        arr_2 = np.zeros(self.number_of_subproblems)
        #unconnected part
        props = np.asarray(props)
        targets = np.asarray(targets)

        arr_1 = props[:,0]
        
        obj_1 = np.sum(arr_1)

        #connected part
        arr_2[0] = arr_1[0]
        
        tarray = targets[1:,0]

        arr_2[1:] = arr_1[1:]*self.sign*(targets[0][0]-tarray+1.125)
        obj_2 = np.abs(np.sum(arr_2))

        return np.array([-obj_1*self.weights[0] + obj_2*self.weights[1],obj_1,obj_2])