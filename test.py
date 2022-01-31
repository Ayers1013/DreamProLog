from misc.tests import *
from transformer.tests import *

import tensorflow as tf

VERBOSITY = True

logger = lambda *args, **kwargs: None
#logger = print

if __name__ == '__main__':
    #allow tensorflow to be initialized before tests start
    tf.zeros(4)+tf.ones(4)*0.1

    tests = []
    for k, fun in list(globals().items()):
        if k[:5] == 'test_': tests.append(fun)

    print('Test starts\n\n')

    for i, fun in enumerate(tests): 
        out = fun(logger)
        if VERBOSITY: print(i, out, sep = '\t')

    print(f'\n\nAll the {len(tests)} tests were run succesfully.\n')