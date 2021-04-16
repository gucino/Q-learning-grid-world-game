%matplotlib inline

#import library
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
##############################################################################
##############################################################################

def create_environment(gold_position,bomb_position,normal_reward,gold_reward,bomb_reward):
    #env position
    env_position = np.arange(0,25).reshape(5,5)
    
    #coordinate
    env_coordinate_dict = {}
    for row in range(env_position.shape[0]):
        for col in range(env_position.shape[1]):
            env_coordinate_dict[env_position[row,col]] = [row,col]
    
    #reward
    env_reward = np.zeros(env_position.shape) + normal_reward
    env_reward[env_coordinate_dict[gold_position][0],\
               env_coordinate_dict[gold_position][1]] = gold_reward
    env_reward[env_coordinate_dict[bomb_position][0],\
               env_coordinate_dict[bomb_position][1]] = bomb_reward    
    
    return env_position,env_coordinate_dict,env_reward

def availabel_action():
    return np.arange(0,4),["up","left","right","down"]

def current_state_initialiser():
    return np.random.randint(20,25)

def move(action,current_state):
    if all_action_name[action] == "up":
        if env_coordinate_dict[current_state][0] == 0:
            new_state = current_state
        else:
            new_state = current_state - 5

    elif all_action_name[action] == "down":
        if env_coordinate_dict[current_state][0] == 4:
            new_state = current_state
        else:
            new_state = current_state + 5      

    elif all_action_name[action] == "left":
        if env_coordinate_dict[current_state][1] == 0:
            new_state = current_state
        else:
            new_state = current_state - 1   

    elif all_action_name[action] == "right":
        if env_coordinate_dict[current_state][1] == 4:
            new_state = current_state
        else:
            new_state = current_state + 1
    return new_state

def get_reward(new_state):
    row = env_coordinate_dict[new_state][0]
    col = env_coordinate_dict[new_state][1]
    return env_reward[row,col]

def Q_table_initialiser():
    return np.zeros((25,4))

def update_Q_table(Q_table,current_state,new_state,action,reward):
    
    current_Q_value = Q_table[current_state,action]
    term = reward + np.amax(Q_table[new_state,:]) - current_Q_value
    new_Q_value = current_Q_value + (alpha * term)
    
    #update to Q table
    Q_table[current_state,action] = new_Q_value
    
    
##############################################################################
##############################################################################
##############################################################################
'''
#test
env_position,env_coordinate_dict,env_reward = \
    create_environment(gold_position,bomb_position,normal_reward,gold_reward,bomb_reward)
    
all_action_index,all_action_name = availabel_action()
current_state = current_state_initialiser()
action = 0
new_state = move(action,current_state)
reward = get_reward(new_state)
'''
##############################################################################
##############################################################################
##############################################################################
#Q learning agent
#input params
gold_position = 3
bomb_position = 8
normal_reward = -1
gold_reward = 10
bomb_reward = -10
alpha = 0.1
num_episode = 1000

#initialise Q table
Q_table = Q_table_initialiser()

#initliase env, state, reward, action
env_position,env_coordinate_dict,env_reward = \
    create_environment(gold_position,bomb_position,normal_reward,gold_reward,bomb_reward)
all_action_index,all_action_name = availabel_action()    

total_reward = [] #per eps
for i in range(num_episode):
    
    #initlise current state
    current_state = current_state_initialiser()
    
    #play
    reward_this_eps = 0
    while True:
        #take action
        rand = np.random.uniform()
        if rand < 0.8:
            #exploit: take best action
            action = np.argmax(Q_table[current_state,:])
        else:
            #explore: random action
            action = np.random.randint(0,4)
        
        #observe new state and reward
        new_state = move(action,current_state)
        reward = get_reward(new_state)
        reward_this_eps += reward
        
        #update Q table
        update_Q_table(Q_table,current_state,new_state,action,reward)
        
        #update state
        current_state = new_state
        
        #check termination condition
        if current_state == gold_position or current_state == bomb_position:
            break
            
    total_reward.append(reward_this_eps)



#visualisation
plt.figure()
plt.plot(total_reward)
plt.show()

##############################################################################
##############################################################################
##############################################################################
#random agent
#input params
gold_position = 3
bomb_position = 8
normal_reward = -1
gold_reward = 10
bomb_reward = -10
alpha = 0.1
num_episode = 1000

#initialise Q table
Q_table = Q_table_initialiser()

#initliase env, state, reward, action
env_position,env_coordinate_dict,env_reward = \
    create_environment(gold_position,bomb_position,normal_reward,gold_reward,bomb_reward)
all_action_index,all_action_name = availabel_action()    

total_reward_random = [] #per eps
for i in range(num_episode):
    
    #initlise current state
    current_state = current_state_initialiser()
    
    #play
    reward_this_eps = 0
    while True:
        #take action

        #explore: random action
        action = np.random.randint(0,4)
        
        #observe new state and reward
        new_state = move(action,current_state)
        reward = get_reward(new_state)
        reward_this_eps += reward
        
        #update Q table
        update_Q_table(Q_table,current_state,new_state,action,reward)
        
        #update state
        current_state = new_state
        
        #check termination condition
        if current_state == gold_position or current_state == bomb_position:
            break
            
    total_reward_random.append(reward_this_eps)



#visualisation
plt.figure()
plt.plot(total_reward,label = "Q leanring agent")
plt.plot(total_reward_random,label = "Random agent")
plt.legend()
plt.show()


##############################################################################
##############################################################################
##############################################################################





