import numpy as np
import tensorflow as tf
import threading

class DummyEnv:
    LOCK=threading.Lock()
    
    def __init__(self):
        #problems is a generator function
        self.step_limit=25
        self.board_size=5
        self.steps=0
        self.pos_x=2
        self.pos_y=2
        self.done=False

    @property
    def action_space_size(self)->int:
        return 4
    
    def step(self,action):
        self.steps+=1

        #TODO output obs,reward,done, info
        #TODO BUG#001
        if(action.shape!=(1)):
            action=np.argmax(action)
        
        reward=0.0

        if(action==0):
            self.pos_x+=1
        elif(action==1):
            self.pos_x-=1
            reward=0.1
        elif(action==2):
            self.pos_y+=1
        else:
            self.pos_y-=1
            reward=0.1

        if(self.pos_x==0 and self.pos_y==0):
            reward=100.0
            self.done=True
        elif(self.pos_x<0 or self.pos_y<0 or self.pos_x>= self.board_size or self.pos_y>= self.board_size):
            reward=-1.0
            self.done=True
        else:
            reward=0.1*(10-self.pos_x-self.pos_y)**2
        if(not self.done and self.steps>=self.step_limit):
            reward=-1.0
            self.done=True

        
        image=np.zeros(self.board_size**2)
        dist=lambda x,y: abs(self.pos_x-x)+abs(self.pos_y-y)
        _lambda=np.log(0.7)
        if(not self.done):
            #Trivial
            #image[self.pos_x+self.pos_y*self.board_size]=1.0
            for x in range(self.board_size):
                for y in range(self.board_size):
                    image[x+y*self.board_size]=dist(x,y)
            image=np.exp(_lambda*image)

        return (
            {'image':image},
            np.float64(reward), 
            self.done,
            {}) 
    
    def reset(self):
        #problems is a generator function
        self.step_limit=25
        self.board_size=5
        self.steps=0
        self.pos_x=2
        self.pos_y=2
        self.done=False
        return {'image':np.zeros(self.board_size**2)}

    @property
    def output_sign(self):
        return {'image': tf.TensorSpec(shape=(None, self.board_size**2), dtype=tf.float32)}
