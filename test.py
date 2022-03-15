from misc._tests import *
from transformer._tests import *

#import tensorflow as tf
import sys

VERBOSITY = True
USE_PRINT = False

def run_tests(logger, health):
    tests = []
    for k, fun in list(globals().items()):
        if k[:5] == 'test_': tests.append(fun)

    print('Test starts\n\n')

    for i, fun in enumerate(tests): 
        out = fun(logger, health=health)
        if VERBOSITY: print(i, out, sep = '\t')

    print(f'\n\nAll the {len(tests)} tests were run succesfully.\n')

if __name__ == '__main__':
    if len(sys.argv)>1:
        health = sys.argv[1]
    else:
        health = 'full'
    logger = print if USE_PRINT else lambda *args, **kwargs: None
    run_tests(logger, health)