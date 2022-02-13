import pygame as pg
from OpenGL.GL import *
import numpy as np
import ctypes
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr


# class Cube:
#     def __init__(self, position, eulers):
#         self.position = np.array(position, dtype = np.float32)
#         self.eulers = np.array(eulers, dtype = np.float32)


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

        # Inicializamos el ratón en el centro de la pantalla
        pg.mouse.set_pos(320, 240)
        pg.mouse.set_visible(False)  # Que no se vea

        self.lastTime = 0
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0

        # self.clock = pg.time.Clock()
        # initializamos OpenGL
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

        # Creamos una instancia del cubo
        self.cube = Cube(self.shader, self.wood_texture, [1, 1, 0.5])

        # Creamos el Player
        self.player = Player([0, 0, 1.2])
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

        # # Cámara
        # pos = np.array([0, 0, 2], dtype = np.float32)  # Posición de la cámara
        # forward = np.array([1, 1, -2], dtype = np.float32)  # Vector hacia delante de la cámara
        # global_up = np.array([0, 0, 1], dtype = np.float32)  # Vector hacia arriba global
        # right = pyrr.vector3.cross(global_up, forward)  # Vector hacia la derecha de la cámara
        # up = pyrr.vector3.cross(forward, right)  # Vector hacia arriba de la cámara
        # # Creamos matriz View y la enviamos al shader
        # lookat_matrix = pyrr.matrix44.create_look_at(pos, np.array([1, 1, 0], dtype = np.float32), up, dtype = np.float32)
        # glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, lookat_matrix)

        # Pedimos a OpenGL la localización de la matriz model
        # self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")

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
            # Prueba eventos
            for event in pg.event.get():
                if (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    running = False

            # Controla ratón y teclado
            self.handleMouse()
            self.handleKeys()

            # Actualiza objetos
            # [0] -- pitch: rotación alrededor del eje x
            # [1] -- roll: rotación alrededor del eje z
            # [2] -- yaw: rotación alrededor del eje y
            self.cube.update()
            self.player.update([self.shader, ])

            # refresca pantalla
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Dibuja objetos
            self.cube.draw()

            # # Usar el programa shader en self
            # glUseProgram(self.shader)
            #
            # # Matriz model_transform = matriz identity * matriz creada con eulers
            # model_transform = pyrr.matrix44.create_identity(dtype = np.float32)
            #
            # model_transform = pyrr.matrix44.multiply(
            #     m1 = model_transform,
            #     m2 = pyrr.matrix44.create_from_eulers(
            #         eulers = np.radians(self.cube.eulers),
            #         dtype = np.float32
            #     )
            # )
            #
            # # Matriz model_transform = matriz model_transform * matriz creada con translation
            # model_transform = pyrr.matrix44.multiply(
            #     m1 = model_transform,
            #     m2 = pyrr.matrix44.create_from_translation(
            #         vec = self.cube.position,
            #         dtype = np.float32
            #     )
            # )
            #
            # # Enviamos al shader la matriz modelo
            # glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, model_transform)
            #
            # # Usar la textura wood_texture
            # self.wood_texture.use()
            #
            # # Vinculamos el VAO
            # glBindVertexArray(self.cube_mesh.vao)
            #
            # # ---- Dibujamos el array ----
            # # GL_TRIANGLES -- Tipo de primitiva
            # # 0 -- Indice inicial en los arreglos habilitados.
            # # self.triangle.vertex_count -- Número de vertices a renderizar
            # glDrawArrays(GL_TRIANGLES, 0, self.cube_mesh.vertex_count)

            pg.display.flip()

            # tiempo
            self.showFrameRate()
        self.quit()

    def handleKeys(self):
        keys = pg.key.get_pressed()
        if keys[pg.K_w]:
            self.player.move(0, 0.01 * self.frameTime)
            return
        if keys[pg.K_a]:
            self.player.move(90, 0.01 * self.frameTime)
            return
        if keys[pg.K_s]:
            self.player.move(180, 0.01 * self.frameTime)
            return
        if keys[pg.K_d]:
            self.player.move(-90, 0.01 * self.frameTime)
            return

    def handleMouse(self):
        (x, y) = pg.mouse.get_pos()
        theta_increment = self.frameTime * 0.05 * (320 - x)
        phi_increment = self.frameTime * 0.05 * (240 - y)
        self.player.increment_direction(theta_increment, phi_increment)
        pg.mouse.set_pos((320, 240))

    def showFrameRate(self):
        self.currentTime = pg.time.get_ticks()
        delta = self.currentTime - self.lastTime
        if (delta >= 1000):
            framerate = int(1000.0 * self.numFrames / delta)
            pg.display.set_caption(f"Running at {framerate} fps.")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0 / framerate)
        self.numFrames += 1

    def quit(self):
        """ limpia la aplicación, código de salida """
        self.cube.destroy()
        self.wood_texture.destroy()
        glDeleteProgram(self.shader)
        pg.quit()


class _Mesh:
    def __init__(self, filepath):
        # x, y, z, s, t, nx, ny, nz normalizadas
        self.vertices = self.loadMesh(filepath)

        # Número de vértices
        self.vertex_count = len(self.vertices) // 8

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

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

        # Activamos el atributo 1 del VAO (color r,g,b)
        glEnableVertexAttribArray(1)

        # 1 -- el índice del vértice genérico para enlazar,
        # 2 -- número de elementos básicos por registro, 1,2,3 o 4 (cogemos s,t)
        # GL_FLOAT -- tipo de datos
        # GL_FALSE -- si los datos están normalizados
        # 20 -- bytes entre registros consecutivos
        # ctypes.c_void_p(12) -- puntero de datos (12 bytes desde el inicio del vértice)

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

    def loadMesh(self, filepath):

        vertices = []
        # raw, datos no ensamblados
        v = []
        vt = []
        vn = []

        with open(filepath, "r") as f:
            line = f.readline()
            while line:
                firstSpace = line.find(" ")
                flag = line[0:firstSpace]

                if flag == "v":
                    line = line.replace("v ", "")
                    # [x,y,z]
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    # Añadimos el vértice
                    v.append(l)
                elif flag == "vt":
                    line = line.replace("vt ", "")
                    # [s,t]
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    # Añadimos el vértice
                    vt.append(l)
                elif flag == "vn":
                    line = line.replace("vn ", "")
                    # [nx,ny,nz]
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    # Añadimos el vértice
                    vn.append(l)
                elif flag == "f":
                    # cara, tres o más vertices en la forma v/vt/vn
                    line = line.replace("f ", "")
                    line = line.replace('\n', "")
                    # [../../.., ../../.., ../../..]
                    line = line.split(" ")
                    faceVertices = []
                    faceTextures = []
                    faceNormals = []
                    for vertex in line:
                        # vertex = v/vt/vt
                        # [v,vt,vn]
                        l = vertex.split("/")
                        position = int(l[0]) - 1
                        faceVertices.append(v[position])
                        texture = int(l[1]) - 1
                        faceTextures.append(vt[texture])
                        normal = int(l[2]) - 1
                        faceNormals.append(vn[normal])
                    # [0,1,2,3] -> [0,1,2,0,2,3]
                    triangles_in_face = len(line) - 2
                    vertex_order = []
                    for i in range(triangles_in_face):
                        vertex_order.append(0)
                        vertex_order.append(i + 1)
                        vertex_order.append(i + 2)
                    for i in vertex_order:
                        for x in faceVertices[i]:
                            vertices.append(x)
                        for x in faceTextures[i]:
                            vertices.append(x)
                        for x in faceNormals[i]:
                            vertices.append(x)
                line = f.readline()
        return vertices

    # Despues de enviar los datos a la tj gráfica podemos borrar lo que tenemos en memoria
    def destroy(self):
        # la función pide una lista por lo que debemos poner esa coma
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))


class Cube:
    def __init__(self, shader, material, position):
        self.material = material
        self.shader = shader
        self.position = position
        glUseProgram(shader)
        # x, y, z, s, t
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

    def update(self):
        angle = np.radians((20 * (pg.time.get_ticks() / 1000)) % 360)
        model_transform = pyrr.matrix44.create_identity(dtype = np.float32)
        model_transform = pyrr.matrix44.multiply(model_transform, pyrr.matrix44.create_from_z_rotation(theta = angle, dtype = np.float32))
        model_transform = pyrr.matrix44.multiply(model_transform,
                                                 pyrr.matrix44.create_from_translation(vec = np.array(self.position), dtype = np.float32))
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, model_transform)

    def draw(self):
        glUseProgram(self.shader)
        self.material.use()
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

    def destroy(self):
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


class Player:
    def __init__(self, position):
        self.position = np.array(position, dtype = np.float32)  # Posición de la cámara
        self.forward = np.array([0, 0, 0], dtype = np.float32)  # Vector hacia delante de la cámara
        self.theta = 0
        self.phi = 0
        self.moveSpeed = 1
        self.global_up = np.array([0, 0, 1], dtype = np.float32)  # Vector hacia arriba global

    def move(self, direction, amount):
        walkDirection = (direction + self.theta) % 360
        self.position[0] += amount * self.moveSpeed * np.cos(np.radians(walkDirection), dtype = np.float32)
        self.position[1] += amount * self.moveSpeed * np.sin(np.radians(walkDirection), dtype = np.float32)

    def increment_direction(self, theta_increase, phi_increase):
        self.theta = (self.theta + theta_increase) % 360
        self.phi = min(max(self.phi + phi_increase, -89), 89)

    def update(self, shaders):
        camera_cos = np.cos(np.radians(self.theta), dtype = np.float32)
        camera_sin = np.sin(np.radians(self.theta), dtype = np.float32)
        camera_cos2 = np.cos(np.radians(self.phi), dtype = np.float32)
        camera_sin2 = np.sin(np.radians(self.phi), dtype = np.float32)
        self.forward[0] = camera_cos * camera_cos2
        self.forward[1] = camera_sin * camera_cos2
        self.forward[2] = camera_sin2

        # Vector hacia la derecha de la cámara
        right = pyrr.vector3.cross(self.global_up, self.forward)
        # Vector hacia arriba de la cámara
        up = pyrr.vector3.cross(self.forward, right)

        # # Creamos matriz View y la enviamos a los shaders
        lookat_matrix = pyrr.matrix44.create_look_at(self.position, self.position + self.forward, up, dtype = np.float32)
        for shader in shaders:
            glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, lookat_matrix)


if __name__ == "__main__":
    myApp = App()
