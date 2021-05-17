import numpy as np
import copy
import random
import pygame
pygame.font.init()
random.seed()


class gui:
    def __init__(self, sizeX, sizeY):
        self.dimension = 28
        self.sizeX = sizeX
        self.sizeY = sizeY
        
        self.field = [ [0 for i in range(self.dimension)] for j in range(self.dimension)]
        self.gridcolor = (0,0,0)
        self.windowcolor = (0,0,0)
        self.lastpainted = None
        
        self.window = pygame.display.set_mode( (sizeX,sizeY) )
        pygame.display.set_caption("Neural Network Number Recognition")
        self.draw()

    def resetGrid(self):
        self.field = [ [0 for i in range(self.dimension)] for j in range(self.dimension)]
        self.draw()


    def draw(self):
        self.window.fill( self.windowcolor )
        
        for i in range(self.dimension):
            for j in range(self.dimension):
                if self.field[i][j] > 0:
                    color = (255*self.field[i][j], 255*self.field[i][j], 255*self.field[i][j])
                    pygame.draw.rect(self.window, color,
                              (i*(self.sizeX - 4)/self.dimension + 3,  
                               j*(self.sizeY - 4)/self.dimension + 3, 
                               (self.sizeX-8)/self.dimension, 
                               (self.sizeY-8)/self.dimension ),
                              0                                                     # linewidth
                             )
        
        linewidth = 1
        for i in range(self.dimension):
            pygame.draw.line( self.window, self.gridcolor,                       
                             (0         , i*(self.sizeY - 4)/self.dimension + 2),    # startpoint
                             (self.sizeX, i*(self.sizeY - 4)/self.dimension + 2),    # endpoint
                             linewidth)
            pygame.draw.line( self.window, self.gridcolor,                       
                             (i*(self.sizeX - 4)/self.dimension + 2, 0),             # startpoint
                             (i*(self.sizeX - 4)/self.dimension + 2, self.sizeY),    # endpoint
                             linewidth)
                    
        pygame.display.update()






    def paint(self, position):
        if self.sizeX < position[0] or self.sizeY < position[1]:
            return
        pos = (int(position[0]/self.sizeX * self.dimension), \
               int(position[1]/self.sizeY * self.dimension)  )
        if pos == self.lastpainted:
            return
        GaussMatrix = [[0.2, 0.5, 0.2],
                       [0.5,   1, 0.5],
                       [0.2, 0.5, 0.2]]
        for i in range(-1,2,1):
            if pos[0] + i < 0 or pos[0] + i >= self.dimension:
                continue
            for j in range(-1,2,1):
                if pos[1] + j < 0 or pos[1] + j >= self.dimension:
                    continue
                self.field[pos[0]+i][pos[1]+j] += GaussMatrix[i+1][j+1]
                if self.field[pos[0]+i][pos[1]+j] > 1:
                    self.field[pos[0]+i][pos[1]+j] = 1
        self.lastpainted = pos
        self.draw()
        return
        
        
    def onedimList(self):
        result = []
        for i in range(self.dimension):
            for j in range(self.dimension):
                result.append( [self.field[j][i]] )
        return result
        
        