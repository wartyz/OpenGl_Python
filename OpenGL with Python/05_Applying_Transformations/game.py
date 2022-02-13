import pygame as pg
from OpenGL.GL import *
import numpy as np
import ctypes
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr


class Cube:
    def __init__(self, position, eulers):
        self.position = np.array(position, dtype = np.float32)
        self.eulers = np.array(eulers, dtype = np.float32)


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

        # Enviamos el valor 0 a uniform imageTexture en FS (es textura 0)
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"), 0)

        # # Habilitamos la transparencia alfa
        # glEnable(GL_BLEND)

        # Habilitamos la pruwba de profundidad
        glEnable(GL_DEPTH_TEST)

        # Cargamos textura 0
        self.wood_texture = Material("gfx/wood.jpeg")

        # # Para hacer aritmética con pixels
        # # GL_SRC_ALPHA -- Cómo es el origen
        # # GL_ONE_MINUS_SRC_ALPHA Cómo es el destino (resultado de la operación)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Creamos una instancia del cubo_mesh
        self.cube_mesh = CubeMesh()

        # Creamos una instancia del cubo
        self.cube = Cube(
            position = [0, 0, -3],
            eulers = [0, 0, 0]
        )

        # Creamos una matrix Projection
        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy = 45, aspect = 640 / 480,
            near = 0.1, far = 10, dtype = np.float32
        )

        # Enviamos la matriz de proyección al shader
        # glGetUniformLocation(self.shader, "projection") -- Localización de matriz
        # 1 -- Número de matrices
        # GL_FALSE -- Si está transpuesta
        # projection_transform -- Puntero a la matriz
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"),
            1, GL_FALSE, projection_transform
        )

        # Pedimos a OpenGL la localización de la matriz model
        self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")

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

            # Actualiza cubo
            # [0] -- pitch: rotación alrededor del eje x
            # [1] -- roll: rotación alrededor del eje z
            # [2] -- yaw: rotación alrededor del eje y
            self.cube.eulers[1] += 0.25
            if self.cube.eulers[1] > 360:
                self.cube.eulers[1] -= 360

            # refresca pantalla
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Usar el programa shader en self
            glUseProgram(self.shader)

            # Matriz model_transform = matriz identity * matriz creada con eulers
            model_transform = pyrr.matrix44.create_identity(dtype = np.float32)

            model_transform = pyrr.matrix44.multiply(
                m1 = model_transform,
                m2 = pyrr.matrix44.create_from_eulers(
                    eulers = np.radians(self.cube.eulers),
                    dtype = np.float32
                )
            )

            # Matriz model_transform = matriz model_transform * matriz creada con translation
            model_transform = pyrr.matrix44.multiply(
                m1 = model_transform,
                m2 = pyrr.matrix44.create_from_translation(
                    vec = self.cube.position,
                    dtype = np.float32
                )
            )

            # Enviamos al shader la matriz modelo
            glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, model_transform)

            # Usar la textura wood_texture
            self.wood_texture.use()

            # Vinculamos el VAO
            glBindVertexArray(self.cube_mesh.vao)

            # ---- Dibujamos el array ----
            # GL_TRIANGLES -- Tipo de primitiva
            # 0 -- Indice inicial en los arreglos habilitados.
            # self.triangle.vertex_count -- Número de vertices a renderizar
            glDrawArrays(GL_TRIANGLES, 0, self.cube_mesh.vertex_count)

            pg.display.flip()

            # tiempo
            self.clock.tick(60)
        self.quit()

    def quit(self):
        """ limpia la aplicación, código de salida """
        self.cube_mesh.destroy()
        self.wood_texture.destroy()
        glDeleteProgram(self.shader)
        pg.quit()


class CubeMesh:
    def __init__(self):
        # x, y, z, s, t normalizadas
        self.vertices = (
            -0.5, -0.5, -0.5, 0, 0,
            0.5, -0.5, -0.5, 1, 0,
            0.5, 0.5, -0.5, 1, 1,

            0.5, 0.5, -0.5, 1, 1,
            -0.5, 0.5, -0.5, 0, 1,
            -0.5, -0.5, -0.5, 0, 0,

            -0.5, -0.5, 0.5, 0, 0,
            0.5, -0.5, 0.5, 1, 0,
            0.5, 0.5, 0.5, 1, 1,

            0.5, 0.5, 0.5, 1, 1,
            -0.5, 0.5, 0.5, 0, 1,
            -0.5, -0.5, 0.5, 0, 0,

            -0.5, 0.5, 0.5, 1, 0,
            -0.5, 0.5, -0.5, 1, 1,
            -0.5, -0.5, -0.5, 0, 1,

            -0.5, -0.5, -0.5, 0, 1,
            -0.5, -0.5, 0.5, 0, 0,
            -0.5, 0.5, 0.5, 1, 0,

            0.5, 0.5, 0.5, 1, 0,
            0.5, 0.5, -0.5, 1, 1,
            0.5, -0.5, -0.5, 0, 1,

            0.5, -0.5, -0.5, 0, 1,
            0.5, -0.5, 0.5, 0, 0,
            0.5, 0.5, 0.5, 1, 0,

            -0.5, -0.5, -0.5, 0, 1,
            0.5, -0.5, -0.5, 1, 1,
            0.5, -0.5, 0.5, 1, 0,

            0.5, -0.5, 0.5, 1, 0,
            -0.5, -0.5, 0.5, 0, 0,
            -0.5, -0.5, -0.5, 0, 1,

            -0.5, 0.5, -0.5, 0, 1,
            0.5, 0.5, -0.5, 1, 1,
            0.5, 0.5, 0.5, 1, 0,

            0.5, 0.5, 0.5, 1, 0,
            -0.5, 0.5, 0.5, 0, 0,
            -0.5, 0.5, -0.5, 0, 1
        )

        # Número de vértices
        self.vertex_count = len(self.vertices) // 5

        # Convertimos los datos a numpy, ojo 32 bits, sino OpenGl no los leerá
        self.vertices = np.array(self.vertices, dtype = np.float32)

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
        # 20 -- bytes entre registros consecutivos
        # ctypes.c_void_p(0) -- puntero de datos

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))

        # Activamos el atributo 1 del VAO (color r,g,b)
        glEnableVertexAttribArray(1)

        # 1 -- el índice del vértice genérico para enlazar,
        # 2 -- número de elementos básicos por registro, 1,2,3 o 4 (cogemos s,t)
        # GL_FLOAT -- tipo de datos
        # GL_FALSE -- si los datos están normalizados
        # 20 -- bytes entre registros consecutivos
        # ctypes.c_void_p(12) -- puntero de datos (12 bytes desde el inicio del vértice)

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))

    # Despues de enviar los datos a la tj gráfica podemos borrar lo que tenemos en memoria
    def destroy(self):
        # la función pide una lista por lo que debemos poner esa coma
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))


class Material:
    def __init__(self, filepath):
        # Creamos y enlazamos una textura
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        # S = 0 es el lado izquierdo de una textura S = 1 el lado derecho
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        # T = 0 es el lado superior de una textura S = 1 el lado inferior
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        # Filtro de reducción y aumento
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Cargamos imagen y la convertimos a pixels compatibles con el dispositivo
        image = pg.image.load(filepath).convert()
        # Obtenemos las medidas de la imagen
        image_width, image_height = image.get_rect().size
        # Transfiere la imagen al buffer de string
        image_data = pg.image.tostring(image, 'RGBA')

        # Especifica una imagen de textura bidimensional
        # GL_TEXTURE_2D -- Tipo de textura de destino
        # 0 -- Nivel de detalle
        # GL_RGBA -- Formato interno
        # image_width, image_height -- ancho y alto de la textura
        # 0 -- siempre debe ser 0
        # GL_RGBA -- formato
        # GL_UNSIGNED_BYTE -- Tipo de dato
        # image_data -- Puntero a los datos

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)

        # Creamos los Mipmaps
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        # Activamos la textura 0
        glActiveTexture(GL_TEXTURE0)
        # La enlazamos al punto GL_TEXTURE_2D
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def destroy(self):
        glDeleteTextures(1, (self.texture,))


if __name__ == "__main__":
    myApp = App()
