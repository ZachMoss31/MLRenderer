from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


class Shader:
    # Initialize shaders to basic shader if there is not one passed, i.e., hot development pink
    def __init__(self, vertex_shader='assets/shaders/vertex.vert', fragment_shader='assets/shaders/fragment.frag'):
        self.vertex_shader = vertex_shader
        self.fragment_shader = fragment_shader
        self.shader_program = self.create_shader(self.vertex_shader, self.fragment_shader)

    # Compile shaders and create a shader program
    def create_shader(self, vert, frag):
        # Open and read the shader files (How can I error check this better? Try: Catch: style?)
            # Likely compile them into the base shaders hardcoded in if the attempted open fails.
        with open(vert, 'r') as vf:
            vert_src = vf.readlines()
        
        with open(frag, 'r') as ff:
            frag_src = ff.readlines()
        
        # Here we would open any geometry shaders etc as well
        
        shader = compileProgram(compileShader(vert_src, GL_VERTEX_SHADER),
                                compileShader(frag_src, GL_FRAGMENT_SHADER))
        
        # Do we need to delete our source shaders now kind of?
        return shader

    # Use the shader program
    def use(self):
        glUseProgram(self.shader_program)

    # Delete the running shader program
    def delete_shader(self):
        glDeleteProgram(self.shader_program)

    # Methods for changing uniforms go here