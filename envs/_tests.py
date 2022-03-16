import time

from .__init__ import Environment

start = time.time()
print("hello")
end = time.time()
print(end - start)

def test_init_Environment_0(logger, **kwargs):
    num_envs = 4
    env = Environment([], task='prolog', num_envs=num_envs)
    steps = 40
    start = time.time()
    for i in range(steps):
        env.step([0]*num_envs)
    end = time.time()

    print(f'Run: {num_envs} envs done {steps} in {end-start}')

    num_envs = 32
    env = Environment([], task='prolog', num_envs=num_envs)
    #steps = 100
    start = time.time()
    for i in range(steps):
        env.step([0]*num_envs)
    end = time.time()

    print(f'Run: {num_envs} envs done {steps} in {end-start}')

    num_envs = 4
    env = Environment([], task='prolog', num_envs=num_envs, parallel_execute=True)
    #steps = 100
    start = time.time()
    for i in range(steps):
        env.step([0]*num_envs)
    end = time.time()

    print(f'Parallel run: {num_envs} envs done {steps} in {end-start}')

    return 'env.Environment test 0 passed.'