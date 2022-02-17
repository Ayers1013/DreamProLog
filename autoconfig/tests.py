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

def test_modules_1(logger):
    class A:
        def __init__(self, a, b, *args, **kwargs):
            self._a = a
            self._b = b

    ccore = ConfigurationCore()

    obj = ccore(4, b = 5) - A
    logger(ccore._children[0]._params)
    assert obj._a == 4
    assert obj._b == 5
    assert len(ccore._children) == 1
    
    return 'autoconfig.module test 1 passed.'
    
def test_modules_2(logger):
    class A:
        def __init__(self, a, b):
            self._a = a
            self._b = b

    class B(ConfiguredModule):
        def __init__(self, a, b):
            super().__init__(a = a, b = b)
            conf = self._config
            self.m1 = conf(a + 2) - A
            self.m2 = conf(b = 33) - A
            self.m3 = conf('end', name = 'end') - A
        def test(self):
            return (self.m1._a, self.m1._b, self.m2._a, self.m2._b, self.m3._a, self.m3._b)

    b = B(42, 666)

    logger(b._config._params)
    logger(*b.test())
    logger(b._config._name, *[c._name for c in b._config._children])
    return 'autoconfig.module test 2 passed.'
