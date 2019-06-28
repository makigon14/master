

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 00:22:57 2018

@author: makigondesk
"""
import gym
import gym.spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math
import cmath
import random
import csv

num=500
num_2 = 314

#w=[0.4 for x in range(6)]
w = np.zeros((num,num))
for i in range(num):
  for j in range(num):
    #rand = random.random()
    rand=(0.9 - 0) * np.random.rand() 
    w[i][j] = rand
    w[j][i] = rand

omega=[random.uniform(2.5, 3.4) for i in range(num) ]
#omega=[-0.3,-0.1,0.1,0.3]
ti=0
R_0=0.7 #目標の蔵本秩序パラメータ
number_of_step=0
#r_li=[0 for i in range(num)]
r_li=[]
#roop=130000
roop = 150000
phi_3 = []
phi_real = []
phi_first = []
phi_save = []

for i in range(num_2):
    phi_real.append((2 * math.pi / num_2) * i)


    
"""
def Kuramoto(phi_2,acc):
    delta_t = 0.1/2
    phi_next = [0 for x in range(10)]
    acc = acc*0.1
    for i in range(num):
        sum_sin = 0
        for j in range(num):
            sum_sin +=math.sin(phi_2[j]-phi_2[i])
        phi_next[i] = phi_2[i]+(omega[i]+K*sum_sin/num+math.sin(phi_2[i])*acc)*delta_t
        phi_next[i] = phi_next[i] % (2*math.pi)
    for i in range(num):
        phi_2[i] = phi_next[i]
        phi_2[i] = phi_2[i] % (2*math.pi)
        
    return [phi_2[0],phi_2[1],phi_2[2],phi_2[3]]
    
"""
def Kuramoto(phi_first,acc):
    delta_t = 0.1/4
    phi_next = [0 for x in range(num)]
    acc = (acc-10)*0.3
    # = []

    """ 
    hi_2_number = list(map(int, phi_2_number))
    for k, phi_num in enumerate(phi_2_number):
        phi_2.extend([phi_real[k] for i in range(phi_num)])  
    """
    for i in range(num):
        sum_sin = 0
        for j in range(num):
            sum_sin +=w[i][j]*math.sin(phi_first[j]-phi_first[i])
        if i!=0:
             phi_next[i] = phi_first[i]+(omega[i]+sum_sin/num+math.sin(phi_first[i])*acc)*delta_t
             phi_next[i] = phi_next[i] % (2*math.pi)
        else: 
             phi_next[i] = phi_first[i]+(omega[i]+sum_sin/num+math.sin(phi_first[i])*0)*delta_t
             phi_next[i] = phi_next[i] % (2*math.pi)
    for i in range(num):
        phi_first[i] = phi_next[i]
        phi_first[i] = phi_first[i] % (2*math.pi)   
 
        
    return [phi_first[i] for i in range(num)]
        
  

    
    
class PointOnLine(gym.core.Env):
    def __init__(self):
        self.action_space=gym.spaces.Discrete(20)#行動空間wは上下
        
        high=np.array([100 for i in range(num_2)])
        self.observation_space=gym.spaces.Box(low=np.array([0 for i in range(num_2)]),high=high)
        
   

    def step(self,action):
        #actionを受け取り、次のstateを決める
        global number_of_step,r_li,phi_3,phi_first,phi_first_p,phi_save
        number_of_step+=1
        dt=0.01
        #acc=action-1 #もしw上なら+になる
        
        
        phi_3 = Kuramoto(phi_3,action)
        phi_save.append(phi_3)
        """
        self.state = list(map(lambda x:x%(2*math.pi),self.state))
        bins = np.linspace(0,2*math.pi,628)
        bins_radix = np.digitize(self.state,bins,right=False)
        self.state = bins[bins_radix]
        """

        
        a=0
        for i in range(num):
            a = a + cmath.exp(phi_3[i]*complex(0,1))
            
        r_li.append(abs(a)/num)
        
            
        if number_of_step>1200:
            done=True
        else:
            done=False
        #終了条件
        #done=abs(R-R_0)<0.05 or flag==True
        """     
        if abs(r_li[number_of_step]-R_0)<=0.03:#0.02
            reward = 1.1
        elif abs(r_li[number_of_step]-R_0)<=0.09:#0.07
            #reward=1.5-abs(R-R_0)*20
            reward=0.75
        else:
            reward=-0.4*abs(r_li[number_of_step]-R_0)
        """
        """
        if abs(r_li[number_of_step]-R_0)<=0.02:#0.02
            reward = 1.1
        elif abs(r_li[number_of_step]-R_0)<=0.09:#0.07
            #reward=1.5-abs(R-R_0)*20
            reward=0.75
        else:
            reward=-0.6*abs(r_li[number_of_step]-R_0)
        """
        if abs(r_li[number_of_step-1]-R_0)<=0.09:
            reward = -5*abs(r_li[number_of_step-1]-R_0)+1.0
        else:
            reward = -3.75*abs(r_li[number_of_step-1]-R_0)+0.3375
        
        
        flag = 1    
        if number_of_step>=6:
            for i in range(6):
                if abs(r_li[number_of_step-i-1]-R_0)>=0.08:
                    flag=0
        else :
            flag=0
                    
            
        if  flag==1:
            reward=1.5
            
        b = np.histogram(phi_3, bins=num_2,range=(0,2*math.pi))
        self.state = b[0]
        
        return np.array(self.state),reward,done,{}
    
    def reset(self):
        global number_of_step,r_li,w,phi_3,phi_re,phi_first,phi_first_p,phi_save
        #w=[0.3 for x in range(6)]
        #w=[random.uniform(0,1.0) for x in range(6)]
        #r_li=[0 for i in range(num)]
        r_li=[]
        phi_save = []
        number_of_step=0
        phi_3.clear()
        #phi_first = []
        phi_first = np.array([random.randint(1,314) for k in range(num)]) #phi_first = [3,205,1..(num個)]
        phi_3 = phi_first*(2 * math.pi / num_2) 
        a = np.histogram(phi_first, bins=num_2)
        self.state = a[0]
        return self.state
        
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory      
        
env = PointOnLine()
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

 # experience replay用のmemory
memory = SequentialMemory(limit=roop, window_length=1)
# 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
policy = EpsGreedyQPolicy(eps=0.1)
#policy = BoltzmannQPolicy() 
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
#dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
#               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
       
history = dqn.fit(env, nb_steps=roop, visualize=False, verbose=2, nb_max_episode_steps=1000)
#history = dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)
#学習の様子を描画したいときは、Envに_render()を実装して、visualize=True にします,     

import rl.callbacks
class EpisodeLogger(rl.callbacks.Callback):
    def __init__(self):
        self.observations = {}
        self.rewards = {}
        self.actions = {}

    def on_episode_begin(self, episode, logs):
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])

cb_ep = EpisodeLogger()
dqn.test(env, nb_episodes=1, visualize=False, callbacks=[cb_ep])


#以下グラフの作成


res_r=[]
res_R=[]
res_1=[]
phi_g=[]

"""
#位相の拡大図（５００から６００）
for obs in cb_ep.observations.values():
    plt.xlim([500,600])
    for i in range(num):
         plt.plot([o[i] for o in obs])
    plt.xlabel("step",fontsize=14)
    plt.ylabel("Phase",fontsize=14)
    plt.tick_params(labelsize=14)
    plt.savefig('kuramoto2_position_large.eps',bbox_inches="tight")
    plt.pause(3)
plt.close()  

#位相全ステップ
for obs in cb_ep.observations.values():
    for i in range(num):
        plt.plot([o[i] for o in obs])
    plt.xlabel("step",fontsize=14)
    plt.ylabel("Phase",fontsize=14)
    plt.tick_params(labelsize=14)
    plt.savefig('kuramoto2_position.eps',bbox_inches="tight")
    plt.pause(3)
plt.close()  
"""
 
#行動の拡大図
for act in cb_ep.actions.values():
    plt.plot([(a-100)*0.3 for a in act])
    #plt.xlim([500,600])
    plt.xlabel("step",fontsize=14)
    plt.ylabel("Action",fontsize=14)
    plt.tick_params(labelsize=14)
plt.savefig('kuramoto2_action.eps',bbox_inches="tight")
plt.pause(3)
plt.close()  
    
obs2=[]

"""
for obs2 in cb_ep.observations.values():
    for i in range(300):
        if i<len(obs2):
            res_r.append(abs(cmath.exp(obs2[i][0]*complex(0,1))+cmath.exp(obs2[i][1]*complex(0,1))
               +cmath.exp(obs2[i][2]*complex(0,1))+cmath.exp(obs2[i][3]*complex(0,1)))/4)
            res_R.append(sum([res_r[cnt]/5 for cnt in range(i,i+5)]))
"""
#蔵本秩序パラメータ

ave=150
print(ave)


    
for obs2 in cb_ep.observations.values():
    for i,obsline in enumerate(obs2) :
        for k, phi_num in enumerate(obsline):
            phi_g.extend([phi_real[k] for i in range(phi_num)])
        p=0
        q=0
        for j in range(num):
             p = p + cmath.exp(phi_g[j]*complex(0,1))
             q = q + cmath.exp(phi_g[j]*complex(0,1))
        res_r.append(abs(p)/num)
        res_1.append(abs(q)/num)
        if i<=ave:
            res_R.append(sum([res_r[cnt]/(i+1) for cnt in range(0,i)]))
        else:
            res_R.append(sum([res_r[cnt]/ave for cnt in range(i-ave,i)]))
        phi_g=[]
            
    t = np.linspace(0,len(obs2),len(obs2))
    plt.plot(t,res_R,label="average")
    #plt.plot(t, res_1,"r-",lw=2,color='gray', alpha=0.22,label="$r(t)$")
    plt.plot(t, res_1,"r-",lw=2,color='gray', alpha=0.41,label="$r(t)$") 
    plt.legend(loc='lower right',fontsize=14)
    plt.tick_params(labelsize=14)
    res_r=[]          
    res_R=[]
    res_1=[]

  
plt.xlabel("step",fontsize=14)
plt.ylabel("Kuramoto order parameter",fontsize=14)
plt.ylim([0,1])
plt.axhline(y=R_0, xmin=0, xmax=1, color='gray',linewidth=1,linestyle='dashed')
plt.savefig('kuramoto2_Res.pdf',bbox_inches="tight")
plt.pause(3)
plt.close()

time = np.linspace(0, len(phi_save),len(phi_save))
plt.plot(time,phi_save)
plt.xlim([500,800])
plt.savefig('kuramoto2_phidiff.pdf',bbox_inches="tight")
plt.pause(3)
plt.close()

"""
num=4
delta_t=0.1/4
time=1500
theta=obs[0]
theta_next=[0 for x in range(4)]
theta_np=np.array([theta])
for t in range(time):
    
  for i in range(num):
      sum_sin=0
      for j in range(num):
          sum_sin += w[i][j]*math.sin(theta[j]-theta[i])
      theta_next[i] =theta[i]+(omega[i]+sum_sin/num)*delta_t
      theta_next[i] = theta_next[i]%(2*math.pi)
    
    
  for i in range(num):
    theta[i] = theta_next[i]
    theta[i] = theta[i]%(2*math.pi)
    
  theta_gyou = np.array([theta[0],theta[1],theta[2],theta[3]])
  
  theta_np=np.append(theta_np,[theta_gyou],axis=0)

x=np.linspace(0, time+1, time+1)
for i in range(num):
    plt.plot(x,theta_np[:,i])
    plt.xlim([500,600])

fig = plt.figure()
ims = []

ax = fig.add_subplot(1,1,1, projection='polar')
theta_d = np.arange(0,2*np.pi,0.01)
r_d = np.full(629,3)
d = ax.scatter(theta_d,r_d,s=1)

for obs3 in cb_ep.observations.values():
    for obsline in obs3[:230]:
        r = [3,3,3,3]
        theta = obsline
        area = 120
        colors =['red','blue','green','grey']
        c = ax.scatter(theta,r,c=colors,s=area,cmap='hsv',alpha=0.30)
        ims.append([c])

ani = animation.ArtistAnimation(fig, ims, interval=100)
ani.save('kuramoto.gif', writer='pillow')
plt.show()
"""
"""
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
artists = []
for obs4 in cb_ep.observations.values():
    for i,obsline in enumerate(obs4[:460]):
        artist = ax.bar(np.linspace(0,num_2,num_2), obsline, width=1.0)
        artists.append(artist)
        print(i)
ani = animation.ArtistAnimation(fig, artists, interval=20)
#ani.save('animebar.gif', writer='pillow')
ani.save('animebarrrr.mp4')
plt.show()
"""