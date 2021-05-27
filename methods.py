from functools import wraps
from tools import Every
import tensorflow as tf

class FunTracker:
  def __init__(self,input_keys, frequency):
    self._input_keys=input_keys
    self._frequency=Every(frequency)
    self.step=1
    self.buffer={key:[] for key in self._input_keys}
    self.buffer['output']=[]
    self.name=""

  def __call__(self,f):
    self.name=f.__name__
    
    @wraps(f)
    def tracked_f(*args,**kwargs):
      if(Every(self.step)):
        for key in self._input_keys:
          self.buffer[key].append(kwargs[key].copy())

      output=f(*args,**kwargs)
      self.buffer['output'].append(output.copy())

      return output
    return tracked_f

class Tracker:
  def __init__(self):
    self.tracked_funs=[]

  def __call__(self, input_keys=None, frequency=1):
    funTracker=FunTracker(input_keys, frequency)
    self.tracked_funs.append(funTracker)
    return funTracker
  
  def summary(self):
    for fun in self.tracked_funs:
      print(fun.buffer) 


class Method:
  def __init__(self):
    self.useless=True


class Reconstructor(Method):
  def __init__(self, worldModel, track_frequency=0):
    super().__init__()
    self._wm=worldModel
    if(track_frequency):
      self._freq=track_frequency
      self.tracker=Tracker()
      self.encode=self.tracker(input_keys=["input"],frequency=self._freq)(self.encode)
      self.decode=self.tracker(input_keys=["input"],frequency=self._freq)(self.decode)

    
  
  def encode(self, input):
    return self._wm.encoder(input)

  def decode(self, input):
    return self._wm.heads['image'](input)

  def __call__(self, input):
    state=self.encode(input=input)
    image=self.decode(input=state)
    return image.mode()

  @tf.function
  def train(self, data):
    data=self._wm.preproecess(data)
    with tf.GradientTape() as model_tape:
      state=self._wm.encoder(data)
      pred=self._wm.heads['image'](state)
      like=pred.log_prob(tf.cast(data['image'],tf.float32))
      model_loss=-like
    model_parts=[self._wm.encoder, self._wm.heads['image']]
    metrics=self._wm._model_opt(model_tape,model_loss, model_parts)

    return metrics



    

        
      

