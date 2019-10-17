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
    
#    def derrive(a, b):
#        return a.derrived() * b.evalForward() + a.evalForward() * b.derrived()
    
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
#        print(f"a={a.evalForward()}")
#        print(f"b={b.evalForward()}")
#        print(f"(a.evalForward() ** b.evalForward())={(a.evalForward() ** b.evalForward())}")
#        print(f"math.log(b.evalForward())={math.log(-2)}")
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
        
class Constant(Expression):
    def __init__(self, value):
        self.value = value
    
    def reset(self):
        pass
    
    def evalForward(self):
        return self.value
    
    def differentiateBackward(self, factors):
        pass
    
def log(a):
    return Expression(a, Constant(1), OperatorLog)

def cwiseOp(A, B, op):
    assert(len(A) == len(B))
    C = []
    for r in range(0, len(A)):
        C.append([])
        assert(len(A[r]) == len(B[r]))
        for c in range(0, len(A[r])):
            C[r].append([])
            C[r][c] = op(A[r][c], B[r][c])
    return C

def cwisemul(A, B):
    return cwiseOp(A, B, lambda a, b: a * b)

def cwiseadd(A, B):
    return cwiseOp(A, B, lambda a, b: a + b)

def cwisesub(A, B):
    return cwiseOp(A, B, lambda a, b: a - b)

def matmul(A, B):
    Arows = len(A)
    assert(Arows > 0)
    Acols = len(A[0])
    
    Brows = len(B)
    assert(Brows > 0)
    Bcols = len(B[0])
    assert(Acols == Brows)
    
    C = []
    
    for r in range(0, Arows):
        C.append([])
        for c in range(0, Bcols):
            C[r].append(Constant(0))

    for r in range(0, Arows):
        for c in range(0, Bcols):
            for i in range(0, Acols):
                T = C[r][c] + A[r][i] * B[i][c]
                C[r][c] = T
    return C

def reduce_sum(A):
    C = Constant(0)
    for r in range(0, len(A)):
        for c in range(0, len(A[r])):
            C = C + A[r][c]
    return C

def reduce_prod(A):
    C = Constant(1)
    for r in range(0, len(A)):
        for c in range(0, len(A[r])):
            C = C * A[r][c]
    return C

def tostring(A, fun):
    out = "[\n"
    for r in range(0, len(A)):
        out += "["
        for c in range(0, len(A[r])):
            out += f"{fun(A[r][c])}, "
        out += "]\n"
    out += "]\n"
    return out

W = [[Variable(1.1), Variable(1.2)], [Variable(1.3), Variable(1.4)]]

x = [[Variable(1)], [Variable(2)]]
y = [[Variable(3), Variable(4)]]

#x = Variable(2)
#y = Variable(3)
#e = Constant(math.e)
#no = Constant(-1)

#f = x * x * x * (x + Constant(3) * y) / (y*x*y)
#f = Constant(2)**(no * x)
#f = (e ** x - e**(no * x)) / (e**x + e**(no * x))
#f = log(x*x + x*y + y*y)*x*y
#f = x - y
#f = reduce_prod(cwisemul(W, matmul(matmul(W, x), y)))
#f = reduce_sum(matmul(W, x))
f = reduce_sum(cwisesub(matmul(x, y), W))

print(f"f = {f.evalForward()}")
print(f"x = {tostring(x, Variable.evalForward)}")
print(f"y = {tostring(y, Variable.evalForward)}")
print(f"W = {tostring(W, Variable.evalForward)}")
f.differentiateBackward()
print(f"df/dx = {tostring(x, Variable.derivative)}")
print(f"df/dy = {tostring(y, Variable.derivative)}")
print(f"df/dW = {tostring(W, Variable.derivative)}")


#print(f"df/dx = {x.derivative}")
#print(f"df/dy = {y.derivative}")
