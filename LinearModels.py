import numpy as np

class Linear:
    
    def __init__(self,x,y):
        self.x = None
        self.y = None
    
    def fit(self,x,y):
        return x*y