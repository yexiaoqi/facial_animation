import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize
from objloader import *


class Predictor_3D:
    def __init__(self, obj_path, index_path, num_bs, num_landmarks):
        self.num_bs = num_bs
        self.num_landmarks = num_landmarks
        self.num_rigid_index = 68
        self.bs = []
        self.vertex_index = np.zeros(num_landmarks, dtype=np.int)
        self.bs_vertex_landmark = np.zeros((num_bs, num_landmarks, 3))
        self.bs_vertex_landmark_flatten = np.zeros((num_landmarks * 3, num_bs))
        self.bs_vertex_rigid = np.zeros((self.num_rigid_index, 3))
        self.Rt = np.array([0, 1, 0, 0, 0, -500], dtype=np.double)
        self.bs_weight = np.zeros(num_bs)
        self.bs_weight[0] = 1
        self.inner_matrix = np.array([(511.57, 0, 300.50), (0, 514.05, 249.94), (0, 0, 1)])
        self.bounds = []
        self.points_rotate = np.zeros((self.num_rigid_index, 3))

        self.bs.append(OBJ(obj_path + '/' + 'Neutral.obj', swapyz=False))
        for i in range(num_bs - 1):
            self.bs.append(OBJ(obj_path + '/' + 'mesh_' + str(i) + '.obj', swapyz=False))
            print('loading mesh ' + str(i))

        # self.result = OBJ(obj_path + '/' + 'Neutral.obj', swapyz=False, display=True)

        with open(index_path) as f:
            for i in range(num_landmarks):
                index = f.readline()
                self.vertex_index[i] = int(index)

        for i in range(num_bs):
            for j in range(num_landmarks):
                self.bs_vertex_landmark[i, j, :] = self.bs[i].vertices[self.vertex_index[j]]
                self.bs_vertex_landmark_flatten[j:j + 3, i] = self.bs[i].vertices[self.vertex_index[j]]

        self.bs_jac_precompute = self.bs_vertex_landmark_flatten.T.dot(self.bs_vertex_landmark_flatten)

        # self.bs_generalized_inverse = np.linalg.inv(
        #    (self.bs_vertex_landmark_flatten.T.dot(self.bs_vertex_landmark_flatten))).dot(
        #    self.bs_vertex_landmark_flatten.T)
        # self.bs_vertex_rigid[0:5, :] = self.bs_vertex_landmark[0, 0:5, :]
        # self.bs_vertex_rigid[5:10, :] = self.bs_vertex_landmark[0, 12:17, :]
        # self.bs_vertex_rigid[10:19, :] = self.bs_vertex_landmark[0, 27:36, :]
        self.bs_vertex_rigid[:, :] = self.bs_vertex_landmark[0, :, :]

        for i in range(num_bs):
            self.bounds.append((0, 1))

        self.bs_vertex = np.zeros([num_bs, len(self.bs[0].vertices), 3])
        self.bs_normal = np.zeros([num_bs, len(self.bs[0].normals), 3])
        for i in range(num_bs):
            self.bs_vertex[i, :, :] = np.array(self.bs[i].vertices)
            self.bs_normal[i, :, :] = np.array(self.bs[i].normals)

    def rotate(self, points, rot_vec):
        """Rotate points by given rotation vectors.
        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vec)
        with np.errstate(invalid='ignore'):
            v = rot_vec / theta
            v = np.nan_to_num(v)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + (points.dot(v.T) * (1 - cos_theta)).dot(v)

    def project(self, Rt, points):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = self.rotate(points, Rt[0:3].reshape(1, 3))
        self.points_rotate[0:self.num_rigid_index, :] = points_proj[0:self.num_rigid_index, :]
        points_proj += Rt[3:6]
        points_proj = (self.inner_matrix.dot(points_proj.T)).T
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]

        return points_proj

    def compute_res_Rt(self, Rt, landmark_pos):
        """Compute residuals."""
        #        landmark_pos = landmark_pos.reshape(self.num_rigid_index, 2)
        #        points_proj = self.project(self.bs_vertex_rigid)
        landmark_pos = landmark_pos.reshape(self.num_rigid_index, 2)
        points_proj = self.project(Rt, self.bs_vertex_rigid)
        return (points_proj - landmark_pos).ravel()

    #        return np.linalg.norm(points_proj - landmark_pos, ord='fro')

    def Jacobian_Rt(self, Rt, landmark_pos):
        Jacobian = np.zeros((self.num_rigid_index * 2, 6))
        for i in range(self.num_rigid_index):
            Jacobian_rotate = np.array([(0, self.points_rotate[i, 2], -self.points_rotate[i, 1]),
                                        (-self.points_rotate[i, 2], 0, self.points_rotate[i, 0]),
                                        (self.points_rotate[i, 1], -self.points_rotate[i, 0], 0)])
            Jacobian[i * 2, 3] = self.inner_matrix[0, 0] / (Rt[5] + self.points_rotate[i, 2])
            Jacobian[i * 2 + 1, 3] = 0
            Jacobian[i * 2, 4] = 0
            Jacobian[i * 2 + 1, 4] = self.inner_matrix[1, 1] / (Rt[5] + self.points_rotate[i, 2])
            Jacobian[i * 2, 5] = - self.inner_matrix[0, 0] * (Rt[3] + self.points_rotate[i, 0]) / np.square(
                Rt[5] + self.points_rotate[i, 2])
            Jacobian[i * 2 + 1, 5] = - self.inner_matrix[1, 1] * (Rt[4] + self.points_rotate[i, 1]) / np.square(
                Rt[5] + self.points_rotate[i, 2])
            Jacobian[i * 2:i * 2 + 2, 0:3] = Jacobian[i * 2:i * 2 + 2, 3:6].dot(Jacobian_rotate)

        return Jacobian

    def optimize_Rt(self, landmark_3D):
        bs_center = np.mean(self.bs_vertex_rigid, axis=0)
        lm_center = np.mean(landmark_3D, axis=0)
        bs_vertex_rigid_centered = self.bs_vertex_rigid - bs_center
        landmark_3D_centered = landmark_3D - lm_center
        correlation_matrix = np.zeros((3, 3))
        for i in range(self.num_rigid_index):
            correlation_matrix = correlation_matrix + (np.transpose([landmark_3D_centered[i, :]])).dot(
                [bs_vertex_rigid_centered[i, :]])
        U, sigma, VT = np.linalg.svd(correlation_matrix)
        rotation = U.dot(VT)
        translate = lm_center - rotation.dot(bs_center)
        theta = np.arccos((rotation[0, 0] + rotation[1, 1] + rotation[2, 2] - 1) / 2)
        if theta == 0:
            axis = np.array([0, 0, 0])
        elif theta == np.pi or theta == -np.pi:
            axis = np.array([np.sqrt((rotation[0, 0] + 1) / 2), np.sqrt((rotation[1, 1] + 1) / 2),
                             np.sqrt((rotation[2, 2] + 1) / 2)])
        else:
            axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                             rotation[1, 0] - rotation[0, 1]]) / 2 / np.sin(theta)
        axis = axis / np.linalg.norm(axis) * theta
        self.Rt[0:3] = axis
        self.Rt[3:6] = translate
        print(self.Rt)

    def compute_res_bs(self, bs_weight, landmark_pos):
        return np.power(np.linalg.norm(self.bs_vertex_landmark_flatten.dot(bs_weight) - landmark_pos), 2) / 2

    def Jacobian_bs(self, bs_weight, landmark_pos):
        return self.bs_jac_precompute.dot(bs_weight) - self.bs_vertex_landmark_flatten.T.dot(landmark_pos.T).flatten()

    def optimize_bs_weight(self, landmark_pos):
        landmark_pos = landmark_pos - self.Rt[3:6]
        landmark_pos = self.rotate(landmark_pos, -self.Rt[0:3].reshape(1, 3))
        # with open('./yht.obj', "w") as f:
        #    for i in range(landmark_pos.shape[0]):
        #        f.write("v %.6f %.6f %.6f\n" % (landmark_pos[i, 0], landmark_pos[i, 1], landmark_pos[i, 2]))
        landmark_pos = landmark_pos.reshape(1, -1)
        # self.bs_weight=self.bs_generalized_inverse.dot(landmark_pos.reshape(-1, 1))
        res = minimize(self.compute_res_bs, self.bs_weight, method='L-BFGS-B', jac=self.Jacobian_bs, bounds=self.bounds,
                       args=(landmark_pos), options={'ftol': 1e-5, 'maxiter': 20})
        self.bs_weight = res.x
        print self.bs_weight
