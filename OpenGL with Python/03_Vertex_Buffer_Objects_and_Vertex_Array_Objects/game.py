import pygame as pg
from OpenGL.GL import *
import numpy as np
import ctypes
from OpenGL.GL.shaders import compileProgram, compileShader


class App:
    def __init__(self):
        """ Initializamos el programa """
        # Inicializa pygame
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                    pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode((640, 480), pg.OPENGL | pg.DOUBLEBUF)
        self.clock = pg.time.Clock()
        # initializamos OpenGl
        glClearColor(0.1, 0.2, 0.2, 1)

        # carga vs y fs, los compila y devuelve el programa shader
        self.shader = self.createShader("shaders/shader.vertex", "shaders/shader.fragment")

        # Usar este programa shader
        glUseProgram(self.shader)

        # Creamos una instancia de Triangle
        self.triangle = Triangle()

        self.mainLoop()

    def createShader(self, vertexFilepath, fragmentFilepath):

        # Abrir vs y fs y leerlos
        with open(vertexFilepath, 'r') as f:
            vertex_src = f.readlines()

        with open(fragmentFilepath, 'r') as f:
            fragment_src = f.readlines()

        # compilar vs y fs, crear programa y devolverlo
        shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )

        return shader

    def mainLoop(self):
        """ Ejecutamos la aplicación """

        running = True
        while (running):
            # prueba eventos
            for event in pg.event.get():
                if (event.type == pg.QUIT):
                    running = False
            # refresca pantalla
            glClear(GL_COLOR_BUFFER_BIT)

            # Usar el programa shader en self
            glUseProgram(self.shader)

            # Vinculamos el VAO
            glBindVertexArray(self.triangle.vao)

            # ---- Dibujamos el array ----
            # GL_TRIANGLES -- Tipo de primitiva
            # 0 -- Indice inicial en los arreglos habilitados.
            # self.triangle.vertex_count -- Número de vertices a renderizar
            glDrawArrays(GL_TRIANGLES, 0, self.triangle.vertex_count)

            pg.display.flip()

            # tiempo
            self.clock.tick(60)
        self.quit()

    def quit(self):
        """ limpia la aplicación, código de salida """
        pg.quit()


class Triangle:
    def __init__(self):
        # x, y,z, r, g, b normalizadas
        self.vertices = (
            -0.5, -0.5, 0.0, 1.0, 0.0, 0.0,  # Vertice 1
            0.5, -0.5, 0.0, 0.0, 1.0, 0.0,  # Vertice 2
            0.0, 0.5, 0.0, 0.0, 0.0, 1.0,  # Vertice 3
        )

        # Convertimos los datos a numpy, ojo 32 bits, sino OpenGl no los leerá
        self.vertices = np.array(self.vertices, dtype=np.float32)

        # Haremos un triángulo
        self.vertex_count = 3

        # Creamos un Vertex Array Object (VAO) y lo vinculamos
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Creamos un Vertex Buffer Object (VBO)
        self.vbo = glGenBuffers(1)

        # Vinculamos VBO a VAO
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # Ponemos los datos
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # Activamos el el atributo 0 del VAO (coordenadas x,y,z)
        glEnableVertexAttribArray(0)

        # ---- Le decimos a OpenGL que significan los datos ----
        # 0 -- el índice del vértice genérico para enlazar,
        # 3 -- número de elementos básicos por registro, 1,2,3 o 4 (cogemos x,y,z)
        # GL_FLOAT -- tipo de datos
        # GL_FALSE -- si los datos están normalizados
        # 24 -- bytes entre registros consecutivos
        # ctypes.c_void_p(0) -- puntero de datos

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

        # Activamos el atributo 1 del VAO (color r,g,b)
        glEnableVertexAttribArray(1)

        # Le decimos a OpenGL que significan los datos
        # 1 -- el índice del vértice genérico para enlazar,
        # 3 -- número de elementos básicos por registro, 1,2,3 o 4 (cogemos x,y,z)
        # GL_FLOAT -- tipo de datos
        # GL_FALSE -- si los datos están normalizados
        # 24 -- bytes entre registros consecutivos
        # ctypes.c_void_p(12) -- puntero de datos (12 bytes desde el inicio del vértice)

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    # Despues de enviar los datos a la tj gráfica podemos borrar lo que tenemos en memoria
    def destroy(self):
        # la función pide una lista por lo que debemos poner esa coma
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))


if __name__ == "__main__":
    myApp = App()
