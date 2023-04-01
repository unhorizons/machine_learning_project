from sympy import re
from torch import classes


class ImageData:
    def __init__(self,path,pixel):
        self.path = path
        self.byte = pixel
        
        self.index = 0

    def byte(self):
        return self.byte
    def index(self):
        return self.index
    def set_index(self, index):
        self.index = index
    def size (self):
        return self.byte.shape()
