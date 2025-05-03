import numpy as np
from base_enviroment import base_station

class M:
    def __init__(self):
        self.a=1

m = M()

class L:
    def __init__(self, m):
        self.a = m.a

class n:
    def __init__(self):
        self.m = m
        self.l =L(m)
    def changem(self):
        self.m.a = 2

test = n()
print(test.l.a)
test.changem()
print(test.l.a)