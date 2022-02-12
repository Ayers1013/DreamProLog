from .module import *

def test_modules_0(logger):
    class A:
        def __init__(self, a, b):
            self._a = a
            self._b = b

    ccore = ConfigurationCore()

    obj = ccore(4, b = 5) - A
    assert obj._a == 4
    assert obj._b == 5
    assert len(ccore._children) == 1
    
    return 'autoconfig.module test 0 passed.'

def _test_modules_1(logger):
    pass