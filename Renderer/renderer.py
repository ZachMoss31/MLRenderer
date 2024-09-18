from OpenGL.GL import *
from prims import Triangle

import sys, os
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from assets.shaders.shader import *
import pygame as pg


class Renderer:
    
    def __init__(self):
        
        # Initialize PyGame
        pg.init()
        pg.display.set_mode((1920, 1080), pg.OPENGL | pg.DOUBLEBUF)
        pg.display.set_caption('ML Renderer')
        self.clock = pg.time.Clock()
        glClearColor(0.1, 0.1, 0.1, 1)
        print(f'Using OpenGL Version:: {glGetString(GL_VERSION)}')
        self.render_queue = []
        initial_tri = Triangle()
        self.initial_shader = Shader()
        self.render_queue.append(initial_tri)
        self.main_loop()

    # Main rendering loop
    def main_loop(self):
        
        running = True
        while(running):
            
            for event in pg.event.get():
                if(event.type ==pg.QUIT):
                    running = False

            glClear(GL_COLOR_BUFFER_BIT)
            # How to pass objects to the renderer in a list or graph instead?
            
            # This should be a scene that holds a tree of objects. ------------------------------------------- TO DO
            self.render_scene()
            pg.display.flip() # Flip as OpenGL texture render is upsdide down probably?
            
            self.clock.tick(60)
        
        self.quit()

    # Render the object and it's data passed in (hopefully a LIST of objects later, then a CUDA list of them)
        # (?) Can I type hint a baseclass here? i.e., renderable: Shape, vs renderable: Triangle, etc?
    def render_scene(self):
        # Use the shader program
        # Bind the vertex array (i.e., renderable.vao)
        
        # (?) Could this be strips if its more complex and saved in the renderable object?
        # (?) Could this be all stored in the object itself, or even in a list of them sorted by prox to the camera, 
            # and rendered in batch?
        for object in self.render_queue:
            glUseProgram(self.initial_shader.shader_program)
            glBindVertexArray(object.vao)
            glDrawArrays(GL_TRIANGLES, 0, object.vertex_count)
    
    # Exit Call
    def quit(self):
        for object in self.render_queue:
            object.delete()
        pg.quit()

if __name__ == "__main__":
    renderer = Renderer()