from panda3d.core import loadPrcFile

loadPrcFile("config/conf.prc")

from panda3d.core import ConfigVariableManager

# confVars = """
# win-size 1280 720
# window-title Mi Juego
# show-frame-rate-meter True
# """
#
# loadPrcFileData("", confVars)
#
from direct.showbase.ShowBase import ShowBase


class MyGame(ShowBase):
    def __init__(self):
        super().__init__()
        # print(ConfigVariableManager.getGlobalPtr().listVariables())

        # Quitamos el movimiento con el ratón de la cámara
        # self.disableMouse()

        # objetos 3d que vienen com panda3D
        # box = self.loader.loadModel("models/box")
        # box.setPos(0, 10, 0)
        # box.reparentTo(self.render)
        #
        # panda = self.loader.loadModel("models/panda")
        # panda.setPos(-2, 10, 0)
        # panda.setScale(0.2, 0.2, 0.2)
        # panda.reparentTo(self.render)

        env = self.loader.loadModel("models/environment")
        env.reparentTo(self.render)


if __name__ == "__main__":
    game = MyGame()
    game.run()
