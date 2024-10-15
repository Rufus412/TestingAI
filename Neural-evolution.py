import random
import math
import numpy as np
from numpy import zeros, argmax
import time
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from tkinter import *
from PIL import Image, ImageTk
import threading
import multiprocessing


#import pandas as pd





Crash = 0
#image = Image.new("RGB", worldsize2)

#random.seed(10)

#win.mainloop()

#image.show()

#region class
class Cell:
    def __init__(self, brain, colour):
        self.brain = brain
        self.colour = colour
        self.canKill = False
        while True:
            x =  random.randint(0,worldsize[0])
            y =  random.randint(0,worldsize[1])
            if worldMap[x][y] == 0:
                self.pos = [x,y]
                break
        self.lastTraveled = (0,0)
        worldMap[self.pos[0]][self.pos[1]] = 1
        self.bias = []
        temp = []
        for i in range(1, len(Layers)):
            for p in range(Layers[i]):
                temp.append(random.randint(-10,10))
            self.bias.append(temp.copy())
            temp.clear()
                

    def moveL(self):
        global Crash
        if toL(self) == 0:
            worldMap[self.pos[0]][self.pos[1]] = 0 
            self.pos[0] -= 1
            self.lastTraveled =  (-1, 0)
            worldMap[self.pos[0]][self.pos[1]] = 1
        else:
            return
            Crash +=1
            worldMap[self.pos[0]][self.pos[1]] = 0
            Cells.remove(self)
            del(self)
        #for i in range (worldsize[0] + 1):
         #   print(arr2[i]) 
        
    def moveR(self) :
        global Crash
        if toR(self) == 0:
            worldMap[self.pos[0]][self.pos[1]] = 0
            self.pos[0] += 1
            self.lastTraveled = (1, 0)
            worldMap[self.pos[0]][self.pos[1]] = 1
        else:
            return
            Crash += 1
            worldMap[self.pos[0]][self.pos[1]] = 0
            Cells.remove(self)
            del(self)
        #for i in range (worldsize[0] + 1):
         #   print(arr2[i]) 

    def moveU(self):
        global Crash
        if toU(self) == 0:
            worldMap[self.pos[0]][self.pos[1]] = 0
            self.pos[1] -= 1
            self.lastTraveled = (0, 1)
            worldMap[self.pos[0]][self.pos[1]] = 1
        else:
            return
            Crash += 1
            worldMap[self.pos[0]][self.pos[1]] = 0
            Cells.remove(self)
            del(self)  
        #for i in range (worldsize[0] + 1):
         #   print(arr2[i])  
    def moveD(self):
        global Crash
        if toD(self) == 0:
            worldMap[self.pos[0]][self.pos[1]] = 0
            self.pos[1] += 1
            self.lastTraveled = (0, -1)
            worldMap[self.pos[0]][self.pos[1]] = 1
        else:
            return
            Crash +=1
            worldMap[self.pos[0]][self.pos[1]] = 0
            Cells.remove(self)
            del(self)
        #for i in range (worldsize[0] + 1):
            #print(arr2[i]) 
    def kill(self):
        if canKill(self):
            if worldMap[self.pos[0] + self.lastTraveled[0]][self.pos[1] + self.lastTraveled[1]] == 1:
                worldMap[self.pos[0] + self.lastTraveled[0]][self.pos[1] + self.lastTraveled[1]] = 5
    def doN(self):
        pass

#endregion

def checkY( a , x):
    count = 0
    q = 0 if x<0 else worldsize2[1]

    for i in range(a[1], q , x) :
        if worldMap[a[0]][i] == 1:
            count += 1
    return count#/worldsize[1]

def checkX( a , x):
    count = 0
    q = 0 if x < 0 else worldsize2[0]

    for i in range(a[0], q , x):
        if worldMap[i][a[1]] == 1:
            count += 1
    return count#/worldsize[0]

#region in

def inFront(x):
    count = 0
    Tx = x.lastTraveled[0]
    Ty = x.lastTraveled[1]

    if Ty != 0:
        return checkY(x.pos, Ty)
    elif Tx != 0:
        return checkX(x.pos, Tx)
    else: return 0
    
def aroundG(x):
    count = -1
    for i in range(-4 , 4):
        for b in range(-4 , 4):
            if worldMap[i][b] == 1:
                count += 1
    return count

def WSx(x):
    return (worldsize[0] - x.pos[0])#/worldsize[0]

def WSy(x):
    return (worldsize[1] - x.pos[1])#/worldsize[1]

def posX(x):
    return x.pos[0]#/worldsize[0]

def posY(x):
    return x.pos[1]#/worldsize[1]

def drX(x):
    return x.lastTraveled[0]

def drY(x):
    return x.lastTraveled[1]

def canKill(x):
    
    return worldMap[x.pos[0] + x.lastTraveled[0]][x.pos[1] + x.lastTraveled[1]] if (worldsize[0] >= (x.pos[0] + x.lastTraveled[0]) >= 0) and  (worldsize[1] >= (x.pos[1] + x.lastTraveled[1]) >= 0) else 0
    
    
    if x.lastTraveled[0]:
        x.canKill = True
        return worldMap[x.pos[0] + x.lastTraveled[0]][x.pos[1]] if x.pos[0] + x.lastTraveled[0] <= worldsize[0] and x.pos[0] + x.lastTraveled[0] >= 0 else 0
    elif x.lastTraveled[1]:
        x.canKill = True
        return worldMap[x.pos[0]][x.pos[1] + x.lastTraveled[1]] if x.pos[1] + x.lastTraveled[1] < worldsize[1] and x.pos[1] + x.lastTraveled[1] >= 0 else 0
    else:
        x.canKill = False
        return 0

def toL(x):
    if (x.pos[0] - 1) >= 0:
        return worldMap[(x.pos[0] - 1)][x.pos[1]]# == 0:
        #    return 1
        #elif worldMap[(x.pos[0] - 1)][x.pos[1]] == 2:
        #    return 2
        #else:
        #    return 0
    return 10
    
def toR(x):
    if (x.pos[0] + 1) <= worldsize[0]:
        return worldMap[x.pos[0] + 1][x.pos[1]]# == 0:
        #    return 1
        #elif worldMap[x.pos[0] + 1][x.pos[1]] == 2:
        #    return 2
        #else:
        #    return 0
    return 10
    
def toU(x):
    if (x.pos[1] - 1) >= 0 :
        return worldMap[x.pos[0]][x.pos[1] - 1]# == 0:
        #    return 1
        #elif worldMap[x.pos[0]][x.pos[1] - 1] == 2:
        #    return 2     
        #else:
        #    return 0
    return 10
    
def toD(x):
    if (x.pos[1] + 1) <= worldsize[1]:
        return worldMap[x.pos[0]][x.pos[1] + 1]# == 0:
        #    return 1
        #elif worldMap[x.pos[0]][x.pos[1] + 1] == 2:
        #    return 2
        #else:
        #    return 0
    return 10

def toL2(x):
    if (x.pos[0] - 2) >= 0:
        return worldMap[(x.pos[0] - 2)][x.pos[1]]# == 0:
        #    return 1
        #elif worldMap[(x.pos[0] - 2)][x.pos[1]] == 2:
        #    return 2
        #else:
        #    return 0
    return 10 
    
def toR2(x):
    if (x.pos[0] + 2) <= worldsize[0]:
        return worldMap[x.pos[0] + 2][x.pos[1]]# == 0:
        #    return 1
        #elif worldMap[x.pos[0] + 2][x.pos[1]] == 2:
        #    return 2
        #else:
        #    return 0
    return 10
    
def toU2(x):
    if (x.pos[1] - 2) >= 0 :
        return worldMap[x.pos[0]][x.pos[1] - 2]# == 0:
        #    return 1
        #elif  worldMap[x.pos[0]][x.pos[1] - 2] == 2:
        #    return 2
        #else:
        #    return 0
    return 10
    
def toD2(x):
    if (x.pos[1] + 2) <= worldsize[1]:
        return worldMap[x.pos[0]][x.pos[1] + 2]# == 0:
        #   return 1
        #elif  worldMap[x.pos[0]][x.pos[1] + 2] == 2:
        #    return 2
        #else:
        #    return 0
    return 10

def toUL(x):
    if (x.pos[0] - 1) >= 0 and (x.pos[1] - 1) >= 0:
        return worldMap[(x.pos[0] - 1)][x.pos[1] - 1]# == 0:
        #    return 1
        #elif worldMap[(x.pos[0] - 1)][x.pos[1] - 1] == 2:
        #    return 2
        #else:
        #    return 0
    return 10
    
def toUR(x):
    if (x.pos[0] + 1) <= worldsize[0] and (x.pos[1] - 1) >= 0:
        return worldMap[x.pos[0] + 1][x.pos[1] - 1]# == 0:
        #    return 1
        #elif worldMap[x.pos[0] + 1][x.pos[1] - 1] == 2:
        #    return 2
        #else:
        #    return 0
    return 10
    
def toDL(x):
    if (x.pos[0] - 1) >= 0 and (x.pos[1] + 1) <= worldsize[1]:
        return worldMap[x.pos[0] - 1][x.pos[1] + 1]
        #    return 1
        #elif worldMap[x.pos[0] - 1][x.pos[1] + 1] == 2:
        #    return 2     
        #else:
        #    return 0
    return 10
    
def toDR(x):
    if (x.pos[1] + 1) <= worldsize[1] and (x.pos[0] + 1) <= worldsize[0]:
        return worldMap[x.pos[0] + 1][x.pos[1] + 1]
        #return 1
            #elif worldMap[x.pos[0] + 1][x.pos[1] + 1] == 2:
        #return 2
        #else:
        return 0
    return 10






#endregion

def sigmoid(x):
    if x < 0:
        return 0
    #else:
    #    return x
    elif 0<= x <= 1:
        return x
    else:
        return 1


 
def draw():
    im = Image.new("RGB", worldsize2)
    for x in range(len(worldMap[0]) ):
        for y in range(len(worldMap) ):

            im.putpixel((x,y), (255,255,255))
            if x == 16 or x == worldsize[0] - 17:
                im.putpixel((x,y), (255,0,0))
            #if worldMap[x][y] == 1:
            #    im.putpixel((x,y), (0,0,0))
    for element in Cells:
        im.putpixel((element.pos), element.colour)

    
    return im.resize((700,700))
    


def mutation(x):
    if random.randint(0, mutationChance) == 0:
        if random.randint(0, 1) == 0:
            for i in range(mutationSize):
                whatMut = random.randint(0, len(Layers)-2)
                x.brain[whatMut][random.randint(0, Layers[whatMut + 1]-1)][random.randint(0, Layers[whatMut]-1)] += (random.uniform(-2, 2))
        else:
            for i in range(mutationSize):
                whatMut = random.randint(0, len(Layers)-2)
                x.bias[whatMut][random.randint(0, len(x.bias[whatMut])-1)] += (random.uniform(-2, 2))

def HiddenLayer(x, Count):
    a = HiddenLayer(x, Count-1) if Count > 0 else FirstLayer(x)
    prevLayer = np.array(a)
    returnV = np.dot(np.array(x.brain[Count]), prevLayer)
    return np.add(returnV, x.bias[Count])


def FirstLayer(x):
    val = []
    for i in range((len(listOfFunc))):
        a = listOfFunc[i](x)
        val.append(a)
    return val

        #for p in range(layer1):
        #        out[p] += x.brain[0][p][i] * a
    
    #a = (sigmoid(out[0])
    #    ,sigmoid(out[1])
    #    ,sigmoid(out[2])
    #    ,sigmoid(out[3])
    #    ,sigmoid(out[4]))
    #print(a)
    #return a

brain = []
Cells = []
innerB = []


def step(x):
        if worldMap[x.pos[0]][x.pos[1]] == 5:
            worldMap[x.pos[0]][x.pos[1]] = 0
            Cells.remove(x)
            del(x)
            return
        count = len(Layers)-2
        store = HiddenLayer(x, count)
        neuronFire = np.argmax(store)
        match neuronFire:
            case 0:
                x.moveL()
            case 1:
                x.moveR()
            case 2:
                x.moveU()
            case 3:
                x.moveD()
            #case 4:
            #    x.kill()
            case 4:
                x.doN()  

def colour():
    return (random.randint(0,255),random.randint(0,255),random.randint(0,255))

def reColour(x):
    b = []
    for i in range(3):
        a = np.ones(len(x.brain[i if i == 0 else i - 1]), dtype = int)
        b.append(list(np.dot(x.brain[i], a)))
    colMulti = []
    for i in range(3):
        colMulti.append(int(sum(b[i]) + 254/2))
    
    x.colour = tuple(colMulti)

def makeChild(selfIn):    
        Cells.append(Cell(selfIn.brain.copy(), selfIn.colour))

def makeCells():
    temp = []
    actBrains = []
    for F in range(startPopulation):
        for U in range(len(Layers) - 1):
            for C in range(Layers[U+1]):
                for K in range(Layers[U]):
                    temp.append(random.randint(-4, 4))
                brain.append(temp.copy())
                temp.clear()
            actBrains.append(brain.copy())
            brain.clear()

        Cells.append(Cell(actBrains.copy(), colour()))
        actBrains.clear()
    return

def Criteria():
    storage = Cells.copy()
    for blobi in storage:
        if  blobi.pos[0] > (worldsize2[0]/2): # > 23 and 8 > blobi.pos[1] > 23 :
            worldMap[blobi.pos] = 0
            Cells.remove(blobi)
            del(blobi)
            


class main:
    def run(self):
        makeCells()
        global Crash
        Gen = 1
        totalDeaths = 0
        survivors =[]
        generations = []
        survivalCriteria = 0
        images = []
        ima = Image.new("RGB", worldsize)
        images.append(ima)
        #start = time.time()
        for p in range(itterations):
            images.clear()
            if Gen % 1 == 0:
                canvas.delete("all")
            #if Gen % 10 == 0:
            #    for elem in Cells:
            #        reColour(elem)
            for i in range(stepCount):
                qwe = Cells.copy()
                #start = time.time()
                for blobItem in qwe:
                    step(blobItem)
                #end = time.time()
                #print("Time elapsed = {}".format(end-start))
                #print("Cells alive = {}".format(len(Cells)))     
                if Gen % drawInterval == 0:
                    im = draw() 
                    showcanvas = ImageTk.PhotoImage (im)
                    canvas.create_image(0,0, anchor = NW, image = showcanvas)
                    images.append(im)
                    canvas.pack()

            storage = Cells.copy()
            temp = []
            for blobi in storage:
                if  blobi.pos[0] > (15) and blobi.pos[0] < worldsize[0] - 16:
                    worldMap[blobi.pos] = 0
                    Cells.remove(blobi)
                    #del(blobi)
                    temp.append(blobi)
            for element in temp:
                del(element)
            if Gen % 5000 == 0:
                images[0].save('pillow_imagedraw.gif', save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)
                
            #arr2.fill(0)
            totalDeaths -=  (len(Cells) - startPopulation)
            Gen+=1
            
            
            if Gen % 10 == 0:
                
                #print(len(blobs)) 
                #print("pop = {}".format(len(blobs)))
                print("Generation = {}".format(Gen))
                #print("B = {}".format(b))
                #print("A = {}".format(a))
                print("Brave voluenters = {}".format(totalDeaths))
                print("Crashes or out of bounds deaths = {}".format(Crash))
                print("Deaths from survial criteria = {}".format(survivalCriteria))
                #print("Total misfits = {}".format(totMutations))
                totMutations = 0
                
                Crash = 0
                survivalCriteria = 0
                
                totalDeaths = 0
                #end = time.time()
                
                #elapsed = end - start 
                #print("Time elapsed = {}".format(elapsed))
                #start = time.time()
            
            
            
            generations.append(Gen)
            survivors.append(len(Cells)-1)
            if Gen % 200 == 0:

                
                #plt.plot(generations, survivors, label = "line 2")
                return
                x = np.array(generations)
                y = np.array(survivors)

                polyline = np.linspace(x.min(), x.max(), Gen)
                plt.scatter(x, y, 0.5)   
                model5 = np.poly1d(np.polyfit(x, y, 8))

                plt.plot(model5(polyline), color='orange')
                plt.show()
                #X_Y_Spline = make_interp_spline(x, y)
                #X_ = np.linspace(x.min(), x.max(), 500)
                #Y_ = X_Y_Spline(X_)
                # Plotting the Graph
                #plt.plot(X_, Y_)
                #plt.xlabel('x - axis')
                #plt.ylabel('y - axis')
                #plt.title('Generational fittness')
            worldMap.fill(0)
            remaining = len(Cells)
            if remaining == 0:
                file = open("myfile.txt", "r")
                a = file.read()
                b = a.split("'")
                
                for i in range(1, len(b), 2):
                    rem = list(b[i].replace("'", "")) 
                    print(b[i])
                    Cells.append(Cell(rem.copy()))

                Gen = Gen - (Gen%100)
                print("EXTINCTION Returning to previous save")
                print(len(b))
                
            else:
                storage = Cells.copy()
                for i in range(startPopulation):
                    makeChild(storage[(i % remaining)])
                for i in range(remaining):
                    Cells.remove(Cells[0])

            for element in Cells:
                mutation(element)
            
            if Gen % 5 == 0:
                file = open("myfile.txt","w")
                temp = []
                for obj in Cells:
                    temp.append(obj.brain)
                file.write(str(temp))
                file.close()

win = Tk()
win.geometry("1200x1200")
canvas= Canvas(win, width= 1200, height= 1200)





def runTest():
    import cProfile 
    import pstats
    
    r = main()
    
    
    with cProfile.Profile() as pr:
        r.run()
        
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()


if __name__ == '__main__':

    #listOfFunc = [inFront, toL, toR, toU, toD, WSx, WSy, posX, posY, drX, drY, canKill, toL2, toR2, toU2, toD2, toUL, toUR, toDL, toDR]
    listOfFunc = [inFront, toL, toR, toU, toD, WSx, WSy, posX, posY, drX, drY]
    worldsize = (199, 199)
    worldsize2 = (worldsize[0] + 1, worldsize[1] + 1)
    WSdiv = worldsize[0]
    startPopulation = 3000
    worldMap = np.zeros(worldsize2, dtype=int)
    mutationChance = 500
    stepCount = 240
    itterations = 10000000
    drawInterval = 1
    Layers = [len(listOfFunc),13,13,5] # dont touch the last number
    mutationSize = 1
    
    r = main()
    
    #runTest()

    threading.Thread(target=r.run).start()
    #multiprocessing.Process(target=r.run).start()
    win.mainloop()