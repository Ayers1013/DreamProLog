import time
import multiprocessing as mp

from .__init__ import Environment

start = time.time()
print("hello")
end = time.time()
print(end - start)

def test_init_Environment_0(logger, **kwargs):
    return 'Disabled.'
    num_envs = 1
    env = Environment([], task='prolog', num_envs=num_envs)
    steps = 1000
    start = time.time()
    for i in range(steps):
        env.step([0]*num_envs)
    end = time.time()

    print(f'Run: {num_envs} envs done {steps} in {end-start}')

    num_envs = 8
    env = Environment([], task='prolog', num_envs=num_envs)
    #steps = 100
    start = time.time()
    for i in range(steps):
        env.step([0]*num_envs)
    end = time.time()

    print(f'Run: {num_envs} envs done {steps} in {end-start}')

    ''' num_envs = 4
    env = Environment([], task='prolog', num_envs=num_envs, parallel_execute=True)
    #steps = 100
    start = time.time()
    for i in range(steps):
        env.step([0]*num_envs)
    end = time.time()

    num_envs = 4
    env = Environment([], task='prolog', num_envs=num_envs, parallel_execute=True)
    #steps = 100
    start = time.time()
    for i in range(steps):
        env.step([0]*num_envs)
    end = time.time()

    print(f'Parallel run: {num_envs} envs done {steps} in {end-start}')'''

    return 'env.Environment test 0 passed.'

def step(x):
    env, act = x
    return env.step(act)

def reset(x):
    env, done = x
    if done: return env.reset()
    else: None

def test_init_Environment(logger, **kwargs):
    return 'Disabled.'
    from .ProLog import ProLog
    envs = [ProLog() for k in range(12)]

    pool = mp.Pool(mp.cpu_count())

    actions = [0]*12

    start = time.time()
    for i in range(100):
        results = map(step, zip(envs, actions))
        obs, _, done = zip(*[p[:3] for p in results])
        obs = list(obs)
        results = map(reset, zip(envs, done))
        
        indices = [index for index, d in enumerate(done) if d]
        for index, result in zip(indices, results):
            obs[index] = result
    end = time.time()

    print(f'Single core time: {end - start}')

    start = time.time()
    for i in range(100):
        results = pool.map(step, zip(envs, actions))
        obs, _, done = zip(*[p[:3] for p in results])
        obs = list(obs)
        results = pool.map(reset, zip(envs, done))

        indices = [index for index, d in enumerate(done) if d]
        for index, result in zip(indices, results):
            obs[index] = result
    end = time.time()

    print(f'Multi core time: {end - start}')

    return 'env.Environment test 1 (multiprocessing) passed.'