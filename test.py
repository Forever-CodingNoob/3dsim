import colorama
import numpy as np
from numpy.linalg import inv
import math
import typing

colorama.init()
(r, g, b) = 200, 255, 255
print(f"\x1b[38;2;{r};{g};{b}mTRUECOLOR\x1b[0m\n")

Numeric = typing.Union[int, float]


def rotate(theta, axis: str) -> np.ndarray:
    if axis.upper() == 'X':
        mx = np.array([[1, 0, 0],
                       [0, math.cos(theta), -math.sin(theta)],
                       [0, math.sin(theta), math.cos(theta)]])
    elif axis.upper() == 'Y':
        mx = np.array([[math.cos(theta), 0, math.sin(theta)],
                       [0, 1, 0],
                       [-math.sin(theta), 0, math.cos(theta)]])
    elif axis.upper() == 'Z':
        mx = np.array([[math.cos(theta), -math.sin(theta), 0],
                       [math.sin(theta), math.cos(theta), 0],
                       [0, 0, 1]])
    else:
        raise ValueError()
    return mx




# def crange(start,stop,step=1):
#     return range(start,stop+step,step)
def crange(start, end, step=1.0):
    return np.arange(start, end + step / 2, step)

def betterColor(a:np.ndarray,b:np.ndarray):
    if not b.any():
        return a
    return b

class Vector(np.ndarray):
    @staticmethod
    def arrToVector(arr: typing.Union[typing.List, typing.Tuple, np.ndarray]):
        if type(arr) != np.ndarray:
            arr = np.array(arr)
        return arr.reshape((3, 1))
    def __new__(cls, x,y,z):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = cls.arrToVector((x,y,z)).view(cls)
        # Finally, we must return the newly created object:
        return obj
    def __array_finalize__(self,obj):
        pass


    @property
    def x(self):
        return self[0,0]
    @property
    def y(self):
        return self[1,0]
    @property
    def z(self):
        return self[2,0]

    def norm(self):
        return math.sqrt(np.sum(self ** 2))

class Sphere:
    def __init__(self, center: np.ndarray, rad: Numeric, rgba: np.ndarray):
        self.center = center
        self.rad = rad
        self.color = rgba

    def __call__(self, coor: np.ndarray) -> typing.Union[bool, typing.Tuple, np.ndarray, None]:
        return self.color if np.sum((coor - self.center) ** 2) <= self.rad ** 2 else np.zeros(4,dtype=float)


elevation = math.pi * (1 / 6)
rotationY = 0
rotationZ = -math.pi * (1 / 6)

camPos = Vector(10, 20, 30)
spec2Screen = Vector(0,0,-10)
screen_width=100
screen=np.zeros((screen_width,screen_width,4),dtype=float)

rotateMat = rotate(rotationZ, 'Z') @ rotate(elevation, 'X') @ rotate(rotationY, 'Y') @ rotate(math.radians(90), 'X')
inv_rotateMat = inv(rotateMat)

camDirec = rotateMat @ Vector(0, 0, -1)
print(camDirec)

objs=[Sphere(camPos+camDirec*20, 10, np.array((255,255,255,1),dtype=float)),
     Sphere(camPos+camDirec*15+Vector(30,0,0), 15, np.array((255,255,255,1),dtype=float))]
print(camPos+camDirec*15+Vector(30,0,0))
print([obj(camPos) for obj in objs])

'''trash'''
# coor=crange(0,100,1)
# space=np.zeros((100,100,100,4),dtype=float) #x,y,z,color(rgba)
# with np.nditer(space, flags=['multi_index','refs_ok'], op_flags=['writeonly']) as it:
#     for i in it:
#         # print(it.multi_index)
#         # i= b(arrToVector(it.multi_index))
#         i[...]=(1,1,1,1)
#         print(space[it.multi_index])
# for i in np.ndindex(*space.shape[:3]):
#     print(rotateMat@arrToVector(i))
#     space[i]=b(rotateMat@arrToVector(i))
# print(np.sum(space))
''''''

for x,y,z in np.ndindex(100,100,100):
    # print(x,y,z)
    vec=Vector(x,y,z)
    color=np.zeros(4,dtype=float)
    for obj in objs:
        color=betterColor(color,obj(Vector(x,y,z)))
    if not color.any(): continue

    viewVec=inv_rotateMat@(vec-camPos)
    # print(viewVec)

    if viewVec.z==0: continue
    projectVec=Vector(viewVec.x-spec2Screen.x,viewVec.y-spec2Screen.y,viewVec.z)*(abs(spec2Screen.z)/abs(viewVec.z))
    print(projectVec)
    
    screenx=math.floor(projectVec.x-(-screen_width/2))
    screeny=math.floor(-(projectVec.y-(screen_width/2)))
    if screenx>=screen_width or screenx<0 or screeny>=screen_width or screeny<0: #restrict projection vectors
        continue
    if not screen[screeny,screenx].any(): #all zero
        screen[screeny,screenx]=color
    print(screenx,screeny)
#ppp
print(screen)

with open('result.txt',mode='w+') as f:
    f.write('\n'.join([''.join(['@@' if color.any() else '..' for color in row]) for row in screen]))
    # for i in screen:
    #     for color in i:
    #         f.write('@@' if color.any() else '..')
    # f.write('\n')







