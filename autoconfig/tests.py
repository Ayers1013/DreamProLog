from module import *

class A(AutoConfigure):
    def __init__(self, a = 5, b = 7):
        config = {'B' : { 'a' : 7}}
        super().__init__('A', config)
        print(f'a:{a}, b:{b}')
        self.create('sub_B', B)

class B(Module):
    def __init__(self, a):
        print(f'B == a:{a}')

if __name__ == '__main__':
    a = A()
    print(a)