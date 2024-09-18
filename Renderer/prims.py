# File containing primitive shapes needed to render
# ----------------------------------------------------------------------------------- TO DO: drop cupy for numpy arrays here, and code a way to check for GPU / cuda to even use cupy!
import numpy as np
import ctypes
from OpenGL.GL import *

# Add the default shader into an abstract shape class-------------------------------- TO DO:
class Triangle:
    def __init__(self):
        self.vertices = (
            -0.5, -0.5, 0.0, 1.0, 0.0, 0.0, # pos: (x,y,z) col:red (1.0, 0.0, 0.0)
            0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.5, 0.0, 0.0, 0.0, 1.0
        )
    
        # np.float32 too big for color vals and pos vals?
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.vertex_count = 3
        
        # Vertex Array Object
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Vertex Buffer Object
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    
    # Delete the object when called (great example of abstract shape class)
    def delete(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))