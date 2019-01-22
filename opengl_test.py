# Basic OBJ file viewer. needs objloader from:
#  http://www.pygame.org/wiki/OBJFileLoader
# LMB + move: rotate
# RMB + move: pan
# Scroll wheel: zoom in/out
import sys, pygame
from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *

# IMPORT OBJECT LOADER
from objloader import *

pygame.init()
viewport = (800, 600)
hx = viewport[0] / 2
hy = viewport[1] / 2
srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)

# glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
# glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
# glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
# glEnable(GL_LIGHT0)
# glEnable(GL_LIGHTING)
# glEnable(GL_COLOR_MATERIAL)
# glEnable(GL_DEPTH_TEST)
# glShadeModel(GL_SMOOTH)           # most obj files expect to be smooth-shaded

draw_2side = False
mat_specular = [0.18, 0.18, 0.18, 0.18]
mat_shininess = [64]
global_ambient = [0.3, 0.3, 0.3, 0.05]
light0_ambient = [0, 0, 0, 0]
light0_diffuse = [0.85, 0.85, 0.8, 0.85]

light1_diffuse = [-0.01, -0.01, -0.03, -0.03]
light0_specular = [0.85, 0.85, 0.85, 0.85]
glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular)
glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess)
glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient)
glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse)
glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular)
glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse)
glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient)
glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_FALSE)
glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, draw_2side)
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glEnable(GL_LIGHT1)
glEnable(GL_COLOR_MATERIAL)
glEnable(GL_NORMALIZE)

# LOAD OBJECT AFTER PYGAME INIT
obj = OBJ('./yht_triangle4/Neutral.obj', swapyz=False)

clock = pygame.time.Clock()

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
width, height = viewport
gluPerspective(60.0, width / float(height), 1, 1000.0)
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_MODELVIEW)

rx, ry = (0, 0)
tx, ty = (0, 0)
zpos = 500
rotate = move = False

vertex_index = []
new_texcoords = obj.texcoords[0:len(obj.vertices)]

for face in obj.faces:
    vertices, normals, texture_coords, material = face
    for i in range(len(vertices)):
        vertex_index.append(vertices[i] - 1)
        new_texcoords[vertices[i] - 1] = obj.texcoords[texture_coords[i] - 1]

while 1:
    clock.tick(30)
    for e in pygame.event.get():
        if e.type == QUIT:
            sys.exit()
        elif e.type == KEYDOWN and e.key == K_ESCAPE:
            sys.exit()
        elif e.type == MOUSEBUTTONDOWN:
            if e.button == 4:
                zpos = max(1, zpos - 1)
            elif e.button == 5:
                zpos += 1
            elif e.button == 1:
                rotate = True
            elif e.button == 3:
                move = True
        elif e.type == MOUSEBUTTONUP:
            if e.button == 1:
                rotate = False
            elif e.button == 3:
                move = False
        elif e.type == MOUSEMOTION:
            i, j = e.rel
            if rotate:
                rx += i
                ry += j
            if move:
                tx += i
                ty -= j

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    # RENDER OBJECT
    glTranslate(0, 0, - zpos)
    glRotate(ry, 1, 0, 0)
    glRotate(rx, 0, 1, 0)

    glEnable(GL_TEXTURE_2D)
    glFrontFace(GL_CCW)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_TEXTURE_COORD_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)

    glVertexPointer(3, GL_FLOAT, 0, obj.vertices)
    glTexCoordPointer(2, GL_FLOAT, 0, new_texcoords)
    glNormalPointer(GL_FLOAT, 0, obj.normals)

    #    glDrawArrays(GL_TRIANGLES, 0, len(obj.vertices))
    glDrawElements(GL_TRIANGLES, len(vertex_index), GL_UNSIGNED_INT, vertex_index)

    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_TEXTURE_COORD_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)
    glDisable(GL_TEXTURE_2D)

    #    glCallList(obj.gl_list)

    pygame.display.flip()
