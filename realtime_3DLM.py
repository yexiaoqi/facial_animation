import cv2
from predictor_3D import *
from pygame.constants import *
from OpenGL.GLU import *
import sys
import time
from PRNet_api import PRN

num_bs = 52
num_landmark = 68

cap = cv2.VideoCapture(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU number, -1 for CPU
prn = PRN(is_dlib=True)
predictor = Predictor_3D('./yht_triangle4', './yht_triangle4/Index68-yht.txt', num_bs, num_landmark)
landmark_rigid_pos = np.zeros((predictor.num_rigid_index, 3))
landmark_pos = np.zeros((num_landmark, 3))
ret, img = cap.read()

pygame.init()
viewport = img.shape[:2][::-1]
hx = viewport[0] / 2
hy = viewport[1] / 2
srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)
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

result = OBJ('./yht_triangle4' + '/' + 'Neutral.obj', swapyz=False, display=True)

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
width, height = viewport
gluPerspective(60.0, width / float(height), 1, 1500.0)
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_MODELVIEW)

rx, ry = (0, 0)
tx, ty = (-138, -145)
zpos = 500
rotate = move = False

vertex_index = []
new_texcoords = predictor.bs[0].texcoords[0:len(predictor.bs[0].vertices)]

for face in predictor.bs[0].faces:
    vertices, normals, texture_coords, material = face
    for i in range(len(vertices)):
        vertex_index.append(vertices[i] - 1)
        new_texcoords[vertices[i] - 1] = predictor.bs[0].texcoords[texture_coords[i] - 1]

while (1):
    start = time.clock()
    ret, img = cap.read()
    pos = prn.process(img)
    preds = prn.get_landmarks(pos)
    for i in range(num_landmark):
        # cv2.circle(img, (preds[0][i, 0], preds[0][i, 1]), 1, (0, 0, 255), -1)
        landmark_pos[i, :] = (preds[i, 0], 255 - preds[i, 1], preds[i, 2])
    # landmark_rigid_pos[0:5, :] = landmark_pos[0:5, :]
    # landmark_rigid_pos[5:10, :] = landmark_pos[12:17, :]
    # landmark_rigid_pos[10:19, :] = landmark_pos[27:36, :]
    landmark_rigid_pos = landmark_pos
    print('landmark detection:%f\n', time.clock() - start)

    with open('./yht.obj', "w") as f:
        for i in range(pos.shape[0]):
            for j in range(pos.shape[1]):
                f.write("v %.6f %.6f %.6f\n" % (pos[i, j, 0], pos[i, j, 1], pos[i, j, 2]))

    predictor.optimize_Rt(landmark_rigid_pos)
    print('optimize_Rt:%f\n', time.clock() - start)
    predictor.optimize_bs_weight(landmark_pos)
    print('optimize_bs:%f\n', time.clock() - start)

    for e in pygame.event.get():
        if e.type == QUIT:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
        elif e.type == KEYDOWN and e.key == K_ESCAPE:
            cap.release()
            cv2.destroyAllWindows()
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
    glTranslate(predictor.Rt[3], predictor.Rt[4], predictor.Rt[5])
    glTranslate(tx, ty, -zpos)
    glRotate(ry, 1, 0, 0)
    glRotate(rx, 0, 1, 0)
    rot_vec = predictor.Rt[0:3]
    theta = np.linalg.norm(rot_vec)
    with np.errstate(invalid='ignore'):
        v = rot_vec / theta
        v = np.nan_to_num(v)
    glRotate(theta / np.pi * 180, v[0], v[1], v[2])
    result.vertices = 0
    result.normals = 0

    # glCallList(result.gl_list)

    for i in range(num_bs):
        result.vertices = predictor.bs_vertex[i, :, :] * predictor.bs_weight[i] + result.vertices
        result.normals = predictor.bs_normal[i, :, :] * predictor.bs_weight[i] + result.normals
    print('construct_mesh:%f\n', time.clock() - start)

    glEnable(GL_TEXTURE_2D)
    glFrontFace(GL_CCW)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_TEXTURE_COORD_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)

    glVertexPointer(3, GL_FLOAT, 0, result.vertices)
    glTexCoordPointer(2, GL_FLOAT, 0, new_texcoords)
    glNormalPointer(GL_FLOAT, 0, result.normals)

    #    glDrawArrays(GL_TRIANGLES, 0, len(obj.vertices))
    glDrawElements(GL_TRIANGLES, len(vertex_index), GL_UNSIGNED_INT, vertex_index)

    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_TEXTURE_COORD_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)

    glDisable(GL_TEXTURE_2D)
    pygame.display.flip()

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    print('render:%f\n', time.clock() - start)

cap.release()
cv2.destroyAllWindows()
