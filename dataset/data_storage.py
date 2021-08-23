import numpy as np
from numpy.lib.function_base import select

class DataStorage:
  def __init__(self):
    #container for episodes
    self._episodes={}
    #problem_name-> length -> reward
    self._storage={}
    #statistics
    self._stats={}

    self._lengthToTag = lambda x: 'small' if x<6 else 'medium' if x<18 else 'large'
    self._tagToLength = lambda x: 2 if x=='small' else 4 if x=='medium' else 8
    seed=69
    self._random = np.random.RandomState(seed)

  def sample_problem(self, treshold, balance=False):
    if treshold:
      options=[opt for opt, stat in self._stats.items() if isinstance(stat, dict) and stat['stats_count']>=treshold]
    else:
      options=self._stats.keys()
    if self._random.randint(2):
      new_options=[opt for opt in options if self._stats[opt]['stats_done']>0]
      if len(new_options):
        options=new_options
    return self._random.choice(options)

  def sample_lengthIndex(self, problem, treshold, weigth=lambda x:x**0.7):
    x=self._stats[problem]
    options={}
    for k in x.keys():
      if k[:6]=='stats_': continue
      options[k]=x[k]['stats_count']
    #Emit small ones and weigth
    options={k: v for k,v in options.items() if v>=treshold}
    probs=np.array(list(options.values()))
    probs=weigth(probs)
    probs/=np.sum(probs)

    lengthIndex=self._random.choice(list(options.keys()), p=probs)
    return lengthIndex
  
  def sample_episode(self, problem, length, positive=True):
    episodes=self._storage[problem][length]
    if positive:
      ep_names=[]
      for ep in reversed(episodes):
        #ep=(stat, name), stat=(reward, end_reward, length)
        if ep[0][0]>=1: ep_names.append(ep)
        else: break
      if len(ep_names)>0:
        selected_ep=ep_names[self._random.randint(len(ep_names))]
      else:
        selected_ep=episodes[self._random.randint(len(episodes))]
    else:
      selected_ep=episodes[self._random.randint(len(episodes))]

    episode, stat=self._episodes[selected_ep[1]], selected_ep[0]
    length=self._tagToLength(length)
    ep_length=stat[2]

    if positive and self._random.randint(2):
      index=ep_length-length
    else:
      index=self._random.randint(ep_length-length+1)
    episode = {k: v[index: index + length] for k, v in episode.items() if k!='action_space'}
    episode['gnn']={k: [episode['gnn'][i][k] for i in range(length)] for k in episode['gnn'][0].keys()}
    return episode

  def store(self, episode, ep_name):
    self._episodes[ep_name]=episode
    problem_name, _, _, length=ep_name[:-4].split("-")
    #problem_name=problem_name.split('\\')[-1]
    lengthTag=self._lengthToTag(int(length))
    self.add_episode(ep_name, [problem_name, lengthTag], (np.sum(episode['reward']), episode['reward'][-1], int(length)))

  def get_stat(self, keys, name):
    x=self._stats
    for g in keys:
      if g not in x: x[g]={}
      x=x[g] 
    return x['stats_'+name]
    
  def upd_stat(self, keys, name, value, update_fun):
    x=self._stats
    for g in keys:
      if g not in x: x[g]={}
      x=x[g] 
    name='stats_'+name 
    if name not in x:
      x[name]=value
    else:
      x[name]=update_fun(x[name], value)

  def add_episode(self, name, group_by, stats):
    #group_by: [], stats: reward
    data=(stats, name)
    
    x=self._storage
    for g in group_by[:-1]:
      if g not in x: x[g]={}
      x=x[g]
    g=group_by[-1]
    if g not in x: x[g]=[data]
    else:
      x=x[g]
      #inserting the element, stats should be comparable
      l,r=0, len(x)
      while( l!= r):
        p=(l+r)//2
        if data < x[p]: r=p
        else: l=p+1
      x.insert(l, data)
    self.update_statistic(group_by, stats)

  def update_statistic(self, keys, stats):
    #count
    self.upd_stat(keys[:1], 'count', 1, lambda l, n: l+n)
    self.upd_stat(keys[:2], 'count', 1, lambda l, n: l+n)

    #done
    done=stats[1]==1.0
    self.upd_stat(keys[:1], 'done', int(done), lambda l, n: l+n)
    if done and self.get_stat(keys[:1], 'done')==1:
      self.upd_stat([], 'solved', 1, lambda l, n: l+n)

    #reward_sum
    reward=stats[0]
    self.upd_stat(keys[:1], 'reward_sum', reward, lambda l, n: l+n)

    #weigthed_reward
    mu=0.95
    self.upd_stat(keys[:1], 'weighted_reward', reward, lambda l, n: mu*l+(1-mu)*n)

  @staticmethod
  def get_statistic(nested_stats):
    stats={}
    def export(x, pretag=''):
      for k,v in x.items():
        if k[:6]=='stats_':
          stats[pretag+k[6:]]=v
        else:
          export(v, pretag+k+'_')
    export(nested_stats)
    return stats

  def __len__(self):
    return len(self._episodes)

