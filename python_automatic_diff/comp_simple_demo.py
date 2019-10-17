#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:45:36 2019

BSD 3-Clause License

Copyright (c) 2019, Adam Celarek | Research Unit of Computer Graphics | TU Wien
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import math

class OperatorAdd:
    def evalForward(a, b):
        return a + b
    
    def differentiateWrtA(a, b):
        return 1
    
    def differentiateWrtB(a, b):
        return 1
    
class OperatorSubtract:
    def evalForward(a, b):
        return a - b
    
    def differentiateWrtA(a, b):
        return 1
    
    def differentiateWrtB(a, b):
        return -1
    
class OperatorMultiply:
    def evalForward(a, b):
        return a * b
    
    def differentiateWrtA(a, b):
        return b.evalForward()
    
    def differentiateWrtB(a, b):
        return a.evalForward()

class OperatorDivide:
    def evalForward(a, b):
        return a / b
    
    def differentiateWrtA(a, b):
        return 1 / b.evalForward()
    
    def differentiateWrtB(a, b):
        return -a.evalForward() / (b.evalForward() * b.evalForward())
    
class OperatorPow:
    def evalForward(a, b):
        return a ** b
    
    def differentiateWrtA(a, b):
        return b.evalForward() * (a.evalForward() ** (b.evalForward() - 1))
    
    def differentiateWrtB(a, b):
        return (a.evalForward() ** b.evalForward()) * math.log(a.evalForward())

class OperatorLog:
    def evalForward(a, b):
        return math.log(a);
    
    def differentiateWrtA(a, b):
        return 1 / a.evalForward()
    
    def differentiateWrtB(a, b):
        return 1


class Expression:
    def __init__(self, a, b, op):
        self.a = a
        self.b = b
        self.op = op
        self.aOpB = None
    
    def __add__(self, other):
        return Expression(self, other, OperatorAdd)
    
    def __sub__(self, other):
        return Expression(self, other, OperatorSubtract)
    
    def __mul__(self, other):
        return Expression(self, other, OperatorMultiply)
    
    def __truediv__(self, other):
        return Expression(self, other, OperatorDivide)
    
    def __pow__(self, other):
        return Expression(self, other, OperatorPow)
    
    def reset(self):
        self.a.reset()
        self.b.reset()
        
    def differentiateBackward(self, factors=1):
        self.a.differentiateBackward(factors * self.op.differentiateWrtA(self.a, self.b))
        self.b.differentiateBackward(factors * self.op.differentiateWrtB(self.a, self.b))
    
    def evalForward(self):
        if (self.aOpB != None):
            return self.aOpB
        
        a = self.a.evalForward()
        b = self.b.evalForward()
        self.aOpB = self.op.evalForward(a, b)
        return self.aOpB

class Variable(Expression):
    def __init__(self, value):
        self.value = value
        self.derivative  = 0;
        
    def asssign(self, value):
        self.value = value
        
    def reset(self):
        self.derivative  = 0;
        pass
        
    def evalForward(self):
        return self.value
    
    def derivative(self):
        return self.derivative
    
    def differentiateBackward(self, factors):
        self.derivative += factors

    
def log(a):
    return Expression(a, Variable(1), OperatorLog)


x = Variable(1)
y = Variable(2)
two = Variable(2)
three = Variable(3)
e = Variable(math.e)

# https://www.wolframalpha.com/input/?i=d+%28x%5E3+*+%28x+%2B+3+*+y%29+%2F+%28y%5E2*x%29%29+%2F+dx++with+x%3D1%2C+y%3D2
#f = x ** three * (x + three * y) / (y**two * x)

# https://www.wolframalpha.com/input/?i=d+%28log%28x*x+%2B+x*y+%2B+y*y%29+*+x+*+y%29+%2F+dx++with+x%3D1%2C+y%3D2
#f = log(x*x + x*y + y*y)*x*y

f = (x + y) * (y + three)

print(f"f = {f.evalForward()}")
f.differentiateBackward()
print(f"df/dx = {x.derivative}")
print(f"df/dy = {y.derivative}")























































#x = Variable(800)
#f = e ** x / (e ** x + e ** y)







