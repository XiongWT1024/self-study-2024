import numpy as np
import torch
import pickle
import time
import argparse
import os


class MeshCondition(torch.nn.Module):
    def __init__(self, in_features, out_features_list, dropouts=[0, 0]):
        super(MeshCondition, self).__init__()
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features_list[0], bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=dropouts[0])
        )
        self.dense2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=out_features_list[0], out_features=out_features_list[1], bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=dropouts[1])
        )

    def forward(self, mesh_info):
        '''
        :param mesh_info: shape: (batch, seq_len, 72 + 10 + 3 = 85)
        :return:
        '''
        batch, seq_len, _ = mesh_info.shape
        mesh_in = mesh_info.view(batch * seq_len, -1)
        out = self.dense1(mesh_in)
        out = self.dense2(out)
        out = out.view(batch, seq_len, -1)
        return out


class SMPL(torch.nn.Module):
    def __init__(self, gender, mesh_location='./data/smpl_models/', gpu_id=0):
        super(SMPL, self).__init__()
        if gender not in ['m', 'f', 'n']:
            raise ValueError('unconfirmed gender')
        smpl_path = {}
        smpl_path['m'] = mesh_location + 'smpl_m.pkl'
        smpl_path['f'] = mesh_location + 'smpl_f.pkl'
        smpl_path['n'] = mesh_location + 'smpl_n.pkl'
        with open(smpl_path[gender], 'rb') as f:
            params = pickle.load(f)
        self.J_regressor = torch.from_numpy(
            np.array(params['J_regressor'].todense())
        ).type(torch.float32)
        if 'joint_regressor' in params.keys():
            self.joint_regressor = torch.from_numpy(
                np.array(params['joint_regressor'].T.todense())
            ).type(torch.float32)
        else:
            self.joint_regressor = torch.from_numpy(
                np.array(params['J_regressor'].todense())
            ).type(torch.float32)
        self.weights = torch.from_numpy(params['weights']).type(torch.float32)
        self.posedirs = torch.from_numpy(params['posedirs']).type(torch.float32)
        self.v_template = torch.from_numpy(params['v_template']).type(torch.float32)
        self.shapedirs = torch.from_numpy(params['shapedirs']).type(torch.float32)
        self.kintree_table = params['kintree_table']
        self.faces = params['f']

        self.device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
        for name in ['J_regressor', 'joint_regressor', 'weights', 'posedirs', 'v_template', 'shapedirs']:
            _tensor = getattr(self, name)
            # print(' Tensor {} shape: '.format(name), _tensor.shape)
            setattr(self, name, _tensor.to(self.device))

    @staticmethod
    def rodrigues(rvec):
        '''
        Rodrigues's rotation formula that turns axis-angle tenser into roration matrix
        :param rvec: [batch, num_vec, 3]
        :return: [batch, num_vec, 3, 3]
        '''

        batch, num_vec, _ = rvec.shape
        r = rvec.reshape(-1, 1, 3)
        eps = r.clone().normal_(std=1e-8)
        theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)
        theta_dim = theta.shape[0]
        r_hat = r / theta  # (_, 1, 3)
        cos = torch.cos(theta)
        z_stick = torch.zeros(theta_dim, dtype=torch.float32).to(r.device)
        m = torch.stack((z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
                         r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
                         -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1).reshape(-1, 3, 3)
        i_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) +
                  torch.zeros((theta_dim, 3, 3), dtype=torch.float32)).to(r.device)
        A = r_hat.permute(0, 2, 1)  # (_, 3, 1)
        dot = torch.matmul(A, r_hat)  # (_, 3, 3)
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R
    
    def rigid_transformation(self, pose, joints):
        '''
        Compute rigid transformations for each joint
        :param pose: [batch, 24, 3] - axis-angle representations for each joint
        :param joints: [batch, 24, 3] - joint positions in T-pose
        :return: [batch, 24, 4, 4] - transformation matrices for each joint
        '''
        batch_size = pose.shape[0]
        Ts = torch.zeros((batch_size, 24, 4, 4), dtype=torch.float32, device=self.device)
        
        # Root joint (index 0)
        root_rotmat = self.rodrigues(pose[:, 0].unsqueeze(1)).squeeze(1)  # [batch, 3, 3]
        Ts[:, 0, :3, :3] = root_rotmat
        Ts[:, 0, :3, 3] = joints[:, 0]
        Ts[:, 0, 3, 3] = 1

        # Child joints
        for i in range(1, 24):
            parent = self.kintree_table[0][i]
            T = torch.zeros((batch_size, 4, 4), dtype=torch.float32, device=self.device)
            T[:, :3, :3] = self.rodrigues(pose[:, i].unsqueeze(1)).squeeze(1)
            T[:, :3, 3] = joints[:, i] - joints[:, parent]
            T[:, 3, 3] = 1
            Ts[:, i] = torch.matmul(Ts[:, parent], T)
        
        return Ts

    def blend_skin(self, pose, v_posed, joints, weights):
        '''
        Apply Linear Blend Skinning (LBS)
        :param pose: [batch, 24, 3] - axis-angle representations for each joint
        :param v_posed: [batch, num_verts, 3] - vertex positions in posed shape
        :param joints: [batch, 24, 3] - joint positions
        :param weights: [num_verts, 24] - skinning weights
        :return: [batch, num_verts, 3] - transformed vertices
        '''
        batch_size = pose.shape[0]
        num_verts = v_posed.shape[1]

        # Compute transformation matrices for each joint
        Ts = self.rigid_transformation(pose, joints)  # [batch, 24, 4, 4]

        # Prepare vertices in homogeneous coordinates
        v_posed_homo = torch.cat([v_posed, torch.ones((batch_size, num_verts, 1), device=self.device)], dim=2)

        # Apply skinning transformations
        Tsw = torch.einsum('bvk,kji->bvji', weights, Ts.view(batch_size, 24, 16)).view(batch_size, num_verts, 4, 4)
        v_homo = torch.matmul(Tsw, v_posed_homo.unsqueeze(-1)).squeeze(-1)
        vertices = v_homo[:, :, :3]  # Extract 3D coordinates

        return vertices
    
    def transform_tpose_to_pose(self, betas, pose, trans, tpose_points, gs_rotations):
        device = self.device
        betas = betas.to(device)
        pose = pose.to(device)          # shape: [batch_size, 24, 3]
        trans = trans.to(device)
        tpose_points = tpose_points.to(device)
        gs_rotations = gs_rotations.to(device)

        batch_size = betas.shape[0]

        # Step 1: Compute shape blend shapes
        shapedirs = self.shapedirs.reshape(-1, self.shapedirs.shape[-1]).T  # [10, 6890*3]
        v_shaped = self.v_template + torch.einsum('bl,lv->bv', betas, shapedirs).reshape(batch_size, -1, 3)  # [batch_size, 6890, 3]

        # Step 2: Compute joint locations
        J = torch.einsum('bik,ij->bjk', v_shaped, self.J_regressor.T)  # [batch_size, 24, 3]

        # Step 3: Compute pose rotation matrices
        Rs = self.rodrigues(pose).view(batch_size, 24, 3, 3)

        # Step 4: Compute pose blend shapes
        I_cube = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        pose_feature = (Rs[:, 1:, :, :] - I_cube)  # [batch_size, 23, 3, 3]
        pose_feature = pose_feature.reshape(batch_size, -1)  # [batch_size, 23*3*3=207]

        posedirs = self.posedirs.reshape(-1, self.posedirs.shape[-1]).T  # [207, 6890*3]
        v_poseblend = torch.einsum('bl,lv->bv', pose_feature, posedirs).reshape(batch_size, -1, 3)  # [batch_size, 6890, 3]

        # Step 5: Apply pose blend shapes to shape blend shapes
        v_posed = v_shaped + v_poseblend  # [batch_size, 6890, 3]
    
        # Step 6: Compute global joint transformation matrices
        parents = self.kintree_table[0].astype(np.int32)
        G = torch.zeros((batch_size, 24, 4, 4), dtype=torch.float32, device=device)
        for i in range(24):
            parent = parents[i]
            if parent == -1:
                # Root joint
                T = torch.cat([Rs[:, i], J[:, i:i+1].transpose(1, 2)], dim=2)  # [batch_size, 3, 4]
                hom_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
                hom_row = hom_row.expand(batch_size, -1, -1)  # [batch_size, 1, 4]
                T = torch.cat([T, hom_row], dim=1)  # [batch_size, 4, 4]
                G[:, i] = T
            else:
                # Child joints
                rel_j = J[:, i] - J[:, parent]  # [batch_size, 3]
                T = torch.cat([Rs[:, i], rel_j.unsqueeze(-1)], dim=2)  # [batch_size, 3, 4]
                hom_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
                hom_row = hom_row.expand(batch_size, -1, -1)  # [batch_size, 1, 4]
                T = torch.cat([T, hom_row], dim=1)  # [batch_size, 4, 4]
                G[:, i] = torch.matmul(G[:, parent], T)

        # Step 7: Correct the transformation matrices
        rest_J = torch.cat([J, torch.zeros([batch_size, 24, 1], device=device)], dim=2)  # [batch_size, 24, 4]
        rest_J = rest_J.unsqueeze(-1)  # [batch_size, 24, 4, 1]
        G[:, :, :3, 3] -= torch.matmul(G[:, :, :3, :3], rest_J[:, :, :3, :1]).squeeze(-1)

        # Step 8: Apply Linear Blend Skinning (LBS)
        weights = self.weights.unsqueeze(0).expand(batch_size, -1, -1).to(device)  # [batch_size, 6890, 24]
        T = torch.einsum('bvn,bnjk->bvjk', weights, G)  # [batch_size, 6890, 4, 4]

        # Convert tpose_points to homogeneous coordinates
        tpose_points = tpose_points.expand(batch_size, -1, -1)
        num_verts = tpose_points.shape[1]
        tpose_homo = torch.cat([tpose_points, torch.ones([batch_size, num_verts, 1], device=device)], dim=2)  # [batch_size, num_verts, 4]

        # Apply transformations
        v_homo = torch.matmul(T, tpose_homo.unsqueeze(-1)).squeeze(-1)  # [batch_size, num_verts, 4]
        v_posed = v_homo[:, :, :3] + trans.unsqueeze(1)  # [batch_size, num_verts, 3]

        # Step 9: Process rotational features
        weighted_rotations = torch.einsum('bvn,bnjk->bvjk', weights, G[:, :, :3, :3])  # [batch_size, 6890, 3, 3]
        rotated_features = torch.matmul(weighted_rotations, gs_rotations)  # [batch_size, 6890, 3, 3]

        return v_posed, rotated_features

    @staticmethod
    def with_zero(x):
        '''
         Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.
        :param x: (batch, 3, 4)
        :return: (batch, 4, 4)
        '''
        # -1 means not changing the size of that dimension
        ones = torch.tensor([0, 0, 0, 1], dtype=torch.float32).view(1, 4).expand(x.shape[0], -1, -1).to(x.device)
        return torch.cat([x, ones], dim=1)

    @staticmethod
    def pack(x):
        '''
        Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.
        :param x: (batch, _, 4, 1)
        :return: (batch, _, 4, 4)
        '''
        zeros = torch.zeros((x.shape[0], x.shape[1], 4, 3), dtype=torch.float32).to(x.device)
        return torch.cat([zeros, x], dim=-1)

    def write_obj(self, verts, file_name):
        with open(file_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def forward(self, betas, pose, trans, simplify=False):
        '''

        :param betas: (batch, 10)
        :param pose: (batch, 24, 3) or (batch, 24, 4) or (batch, 24, 3, 3)
        :param trans: (batch, 3)
        :param simplify: if true, pose is not considered
        :return: vertices matrix (batch, 6890, 3) and joint positions (batch, 19, 3) ?
        '''
        extend_flag=False
        if len(betas.shape)==3 and len(trans.shape)==3:
            extend_flag=True
            extend_batch=pose.shape[0]
            extend_length=pose.shape[1]
            betas=betas.view((extend_batch*extend_length, 10))
            trans=trans.view((extend_batch*extend_length, 3))
            if len(pose.shape)==4:
                pose=pose.view((extend_batch*extend_length, 24, pose.shape[-1]))
            elif len(pose.shape)==5:
                pose=pose.view((extend_batch*extend_length, 24, 3, 3))
            else:
                print('SMPL Pose Error!')
                exit()
        else:
            assert len(betas.shape)==2 and len(trans.shape)==2

        batch = betas.shape[0]
        id_to_col = {self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])}
        parent = {i: id_to_col[self.kintree_table[0, i]] for i in range(1, self.kintree_table.shape[1])}
        v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [2])) + self.v_template  # (batch, 6890, 3)
        # print('v_shaped: ', v_shaped.shape)
        J = torch.matmul(self.J_regressor, v_shaped)  # (batch, 24, 3)
        # print('J: ', J.shape)
        if len(pose.shape)==4:
            R_cube_big=pose
        elif pose.shape[-1] == 3:
            R_cube_big = self.rodrigues(pose.reshape(-1, 1, 3)).reshape(batch, -1, 3, 3)  # (batch, 24, 3, 3)
        else:
            #print('use quater')
            R_cube_big = self.quaternion2rotmat(pose)
        # print('R_cube_big: ', R_cube_big.shape)
        if simplify:
            v_posed = v_shaped
        else:
            R_cube = R_cube_big[:, 1:, :, :]  # remove first rot matrix (global rotation)
            I_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) +
                      torch.zeros((batch, R_cube.shape[1], 3, 3), dtype=torch.float32)).to(self.device)
            lrotmin = (R_cube - I_cube).reshape(batch, -1, 1).squeeze(dim=2)  # (batch, 23 * 9)
            # print('lrotmin: ', lrotmin.shape)
            v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=([1], [2]))  # (batch, 6890, 3)
        # print('v_posed: ', v_posed.shape)
        results = []
        results.append(
            self.with_zero(torch.cat([R_cube_big[:, 0], J[:, 0, :].reshape(-1, 3, 1)], dim=2)))  # (batch, 4, 4)
        for i in range(1, self.kintree_table.shape[1]):
            # parent to children
            res = torch.matmul(results[parent[i]],  # (batch, 4, 4)
                               self.with_zero(torch.cat([R_cube_big[:, i],  # (batch, 3, 3), below (batch, 3, 1)
                                                         (J[:, i, :] - J[:, parent[i], :]).reshape(-1, 3, 1)], dim=2)))
            results.append(res)
        stacked = torch.stack(results, dim=1)  # (batch, 24, 4, 4)
        # print('stacked: ', stacked.shape)
        zeros = torch.zeros((batch, 24, 1), dtype=torch.float32).to(self.device)
        results = stacked - self.pack(torch.matmul(stacked,
                                                   torch.cat([J, zeros], dim=2).reshape(batch, 24, 4, 1)))
        # print('results: ', results.shape)
        T = torch.tensordot(results, self.weights, dims=([1], [1])).permute(0, 3, 1, 2)  # (batch, 6890, 4, 4)
        # print('T: ', T.shape)
        ones = torch.ones((batch, v_posed.shape[1], 1), dtype=torch.float32).to(self.device)
        rest_shape_h = torch.cat([v_posed, ones], dim=2)  # (batch, 6890, 3+1)
        # print('rest_shape_h; ', rest_shape_h.shape)
        v = torch.matmul(T, rest_shape_h.reshape(batch, -1, 4, 1))  # (batch, 6890, 4, 1)
        v = v.reshape(batch, -1, 4)[..., :3]
        mesh_v = v + trans.reshape(batch, 1, 3)
        # print('mesh_v: ', v.shape)

        joints = torch.tensordot(mesh_v, self.joint_regressor, dims=([1], [1])).transpose(1, 2)  # (batch, 24, 3)
        # print('joints: ', joints.shape)

        if extend_flag:
            mesh_v=mesh_v.view(extend_batch, extend_length, 6890, 3)
            joints=joints.view(extend_batch, extend_length, 24, 3)
        return mesh_v, joints

    @staticmethod
    def quaternion2vec(batch_q):
        theta = 2 * torch.acos(batch_q[:, :, -1:])
        vecs = (theta / torch.sin(theta / 2)) * batch_q[:, :, :-1]
        return vecs

    @staticmethod
    def quaternion2rotmat(batch_q):
        '''
        quaternion to rotation matrix
        :param batch_q:  (batch, 24, 4)
        :return:
        '''
        qw, qx, qy, qz = batch_q[:, :, 3], batch_q[:, :, 0], batch_q[:, :, 1], batch_q[:, :, 2]

        x2 = qx + qx
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        dim0 = torch.stack((1.0 - (yy + zz), xy - wz, xz + wy), dim=-1)
        dim1 = torch.stack((xy + wz, 1.0 - (xx + zz), yz - wx), dim=-1)
        dim2 = torch.stack((xz - wy, yz + wx, 1.0 - (xx + yy)), dim=-1)
        m = torch.stack((dim0, dim1, dim2), dim=-2)
        return m


def test_smpl():
    pose_size = 72
    beta_size = 10

    np.random.seed(9608)
    model = SMPL('m')

    #pose_q = torch.from_numpy((np.random.rand(32, 16, 24, 4) - 0.5) * 0.4).type(torch.float32)
    pose_q = torch.from_numpy((np.random.rand(32, 16, 24, 3,3) - 0.5) * 0.4).type(torch.float32)
    betas = torch.from_numpy((np.random.rand(32, 16, beta_size) - 0.5) * 0.06) \
        .type(torch.float32)
    s = time.time()
    trans = torch.from_numpy(np.zeros((32, 16, 3))).type(torch.float32)
    result, joints = model(betas, pose_q, trans)
    print(time.time() - s)
    return result, joints

'''
def parse_args():
    parser = argparse.ArgumentParser(description='SMPL Generator')
    parser.add_argument('directory', type=str, help='Directory where pose and beta stored in')
    parser.add_argument('-p', '--pose_prefix', type=str, default='pose_', help='prefix of pose data')
    parser.add_argument('-b', '--beta_prefix', type=str, default='beta_', help='prefix of beta data')
    parser.add_argument('-r', '--reduce', action='store_true', help='adapt 9 mesh version')
    parser.add_argument('-g', '--gpu_id', type=int, nargs='*',
                        help='GPU to be used, range: (0~3), total number must be less than 4')
    parser.add_argument('-sd', '--save_dir', type=str, default='./res', help='Directory to save mesh results')
    return parser.parse_args()


if __name__ == '__main__':
    a, b =test_smpl()
    print(a.shape)
    print(b.shape)
'''