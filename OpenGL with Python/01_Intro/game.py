import pygame as pg
from OpenGL.GL import *


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
        self.mainLoop()

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
            pg.display.flip()

            # tiempo
            self.clock.tick(60)
        self.quit()

    def quit(self):
        """ limpia la aplicación, código de salida """
        pg.quit()


if __name__ == "__main__":
    myApp = App()
