
# conf.debug
# params.keys()
# params['shs'].sum(dim=0)
# params['scales'] > 1
# params['rotations'].sum(dim=0)
# params['cov3D_precomp']


# vm = conf.viewmatrix.to("cpu")
# [*x, xhat] = vm[:, :3].T[0]
# [*y, yhat] = vm[:, :3].T[1]
# [*z, zhat] = vm[:, :3].T[2]

# cp = conf.campos.to("cpu")
# xhat, cp.dot(torch.tensor(x))
# yhat, cp.dot(torch.tensor(y))
# zhat, cp.dot(torch.tensor(z))

# import torch
# import numpy as np
# # construct view matrix by hand


# eye = np.array([1., 0., 0.])
# at = np.array([0., 0., 0.])
# up = np.array([0., 1., 0.])

# # make projection matrix
# pm = conf.projmatrix.to("cpu")
# pm @ torch.tensor([1,0,0,1], dtype=torch.float)
# xline = np.linspace(0, 1, 1000)
# line =torch.stack([torch.tensor([x, 0, 0, 1], dtype=torch.float) for x in xline])
# proj_line = pm @ line.T
# proj_line.T

# znear = 0.5
# zfar = 1.5
# tanfovx = 0.5
# tanfovy = 0.5
# top = znear * tanfovy
# bottom = -top
# right = znear * tanfovx
# left = - right
# P = torch.tensor([
#     [2 * znear / (right - left),    0,                          (right+left)/(right-left),      0                       ],
#     [0,                             2 * znear / (top - bottom), (top + bottom)/(top - bottom),  0                       ],
#     [0,                             0,                          zfar/(zfar - znear),            -zfar*znear/(zfar-znear)],
#     [0,                             0,                          1,                              0                       ]
# ])




# mod = dg.GaussianRasterizer(conf)

# ans = mod.forward(**params)

# t, _ = ans
# t.shape

# n = t.cpu().detach().numpy()
# n.shape
# n.transpose([1, 2, 0]).shape
# im = plt.imshow(n.transpose([1, 2, 0]))

# plt.savefig("ooo.png")

# vm = torch.tensor(
#     [[ 0.5326, -0.4825,  0.6954,  0.0000],
#     [ 0.4736,  0.8508,  0.2276,  0.0000],
#     [-0.7015,  0.2081,  0.6816,  0.0000],
#     [-0.2072,  0.0363, -3.4346,  1.0000]], 
#     dtype=torch.float, device="cuda"
# ).t()
# vm = vm.cpu()



# pm = torch.tensor([[ 0.8271, -0.9987,  0.6955,  0.6954],
#         [ 0.7355,  1.7611,  0.2277,  0.2276],
#         [-1.0894,  0.4307,  0.6817,  0.6816],
#         [-0.3218,  0.0752, -3.4450, -3.4346]], dtype=torch.float, device="cuda").cpu()
# pm = pm.numpy()

# # the truuuue projection matrix
# p = pm.T @ np.linalg.inv(vm)

# # p = np.linalg.inv(vm) @ pm.cpu().numpy() 
# p
# p[np.abs(p)<1e-4] = 0
# p
# looks like a real projection matrix!!!!
# those fucking bitches and their fd up names

# creating my own shit
import pickle
import matplotlib.pyplot as plt
import diff_gaussian_rasterization as dg 
import numpy as np
import torch
from torch import tensor

with open("raster_settings.pkl", 'rb') as f:
    l = f.read()
    conf = pickle.loads(l)

with open("params.pkl", 'rb') as f:
    l = f.read()
    params = pickle.loads(l)

c2 = dg.GaussianRasterizationSettings(
    image_height=1023, 
    image_width=1361, 
    tanfovx=0.643915208536692, 
    tanfovy=0.483127747830031, 
    bg=tensor([0., 0., 0.], device='cuda:0'), 
    scale_modifier=1.0, 
    viewmatrix=tensor([[ 0.5326, -0.4825,  0.6954,  0.0000],
        [ 0.4736,  0.8508,  0.2276,  0.0000],
        [-0.7015,  0.2081,  0.6816,  0.0000],
        [-0.2072,  0.0363, -3.4346,  1.0000]], device='cuda:0'), 
    projmatrix=tensor([[ 0.8271, -0.9987,  0.6955,  0.6954],
        [ 0.7355,  1.7611,  0.2277,  0.2276],
        [-1.0894,  0.4307,  0.6817,  0.6816],
        [-0.3218,  0.0752, -3.4450, -3.4346]], device='cuda:0'), 
    sh_degree=0,
    campos=tensor([0, 0, 0], dtype=torch.float, device='cuda:0'), 
    prefiltered=False, 
    debug=False
)

print(params)
params.keys()
params["means3D"].shape
params["means2D"].shape
params["shs"].shape
params["opacities"].shape
params["rotations"].shape
params["scales"].shape

type(params["means2D"])
type(params["shs"])
type(params["opacities"])
type(params["rotations"])
type(params['means3D'])

params['means3D'].data

rast = dg.GaussianRasterizer(c2)
rast
im, _ = rast(**params)
params.keys()

l = torch.abs(im).mean()
l.backward()
l

def make_proj_matrix(znear, zfar, fovx, fovy):
    tanfovx = np.tan(fovx/2)
    tanfovy = np.tan(fovy/2)
    top = znear * tanfovy
    bottom = -top
    right = znear * tanfovx
    left = - right
    return torch.tensor([
        [2 * znear / (right - left),    0,                          (right+left)/(right-left),      0                       ],
        [0,                             2 * znear / (top - bottom), (top + bottom)/(top - bottom),  0                       ],
        [0,                             0,                          (znear + zfar)/(zfar - znear),  2*zfar*znear/(zfar-znear)],
        [0,                             0,                          1,                              0                       ]
    ], dtype=torch.float, device="cuda")

normalize = lambda v: v if (n:=np.linalg.norm(v)) == 0 else v/n
def make_view_matrix(eye, at, up):
    z  = normalize(at-eye)
    x = normalize(np.cross(up, z))
    y = np.cross(z, x)
    xhat = x.T @ eye
    yhat = y.T @ eye
    zhat = z.T @ eye
    return torch.tensor(np.array([
        [*x, -xhat],
        [*y, -yhat],
        [*z, -zhat],
        [0, 0, 0, 1]
    ]),dtype=torch.float, device="cuda")

def filter_gaussians(parameters, start=0, end=10):
    return dict(
        means3D=parameters["means3D"][start:end],
        means2D=parameters["means2D"][start:end],
        shs=parameters["shs"][start:end],
        opacities=parameters["opacities"][start:end],
        scales=parameters["scales"][start:end],
        rotations=parameters["rotations"][start:end],
    )

def make_model(eye, at, up, znear, zfar, fovx, fovy):
    vm = make_view_matrix(eye, at, up).to("cuda")
    pm = make_proj_matrix(znear, zfar, fovx, fovy).to("cuda")
    viewmatrix = vm.t()
    projmatrix = vm.t() @ pm.t()
    configuration = dg.GaussianRasterizationSettings(
        image_height=1024,
        image_width=1024,
        tanfovx=np.tan(fovx/2),
        tanfovy=np.tan(fovy/2),
        bg=torch.tensor([0, 0, 0.1], dtype=torch.float, device="cuda"),
        scale_modifier=1.,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=torch.tensor(eye, dtype=torch.float, device="cuda"),
        prefiltered=False,
        debug=False
    )
    return dg.GaussianRasterizer(configuration)

# testing with our image gaussians
gaussian_model = make_model(
    eye = np.array([2, 1, 5]),
    at = np.array([5, 1, 4]),
    up = np.array([0., 1., 0.]),
    znear = 1,
    zfar = 100,
    fovx = np.pi/4,
    fovy = np.pi/4
)
img, _ = gaussian_model.forward(**filter_gaussians(params, 0, 4000))
npimg = img.cpu().detach().numpy().transpose([1, 2, 0])
im = plt.imshow(npimg)
plt.savefig("gaussian.png")


# making one gaussian at center
gaussian_model = make_model(
    eye = np.array([10, 0, 0]),
    at = np.array([0, 0, 0]),
    up = np.array([0., 1., 0.]),
    znear = 1,
    zfar = 100,
    fovx = np.pi/4,
    fovy = np.pi/4
)
n_gaussians=1
means3D = torch.tensor([[0, 0, 0]]*n_gaussians, dtype=torch.float, device="cuda")
means2D = torch.zeros_like(means3D, dtype=torch.float, device="cuda")
shs = torch.tensor([[[1, 1, 1], * [[0, 0, 0]]*15]] * n_gaussians, dtype=torch.float, device="cuda")
opacities = torch.ones([n_gaussians, 1], dtype=torch.float, device="cuda") * 1
scales = torch.tensor([[.4, .1, .1]* n_gaussians], dtype=torch.float, device="cuda")
rotations = torch.tensor([[1, 0, 0, 0]* n_gaussians], dtype=torch.float, device="cuda")

img, _ = gaussian_model.forward(means3D, means2D, opacities, shs=shs, scales=scales, rotations=rotations)
npimg = img.cpu().detach().numpy().transpose([1, 2, 0])
im = plt.imshow(npimg)
plt.savefig("ball.png")




# matrices
import numpy as np 
# this definition makes sense , the other does not fucking make sense
normalize = lambda v: v if (n:=np.linalg.norm(v)) == 0 else v/n
def make_view_matrix(eye, at, up):
    z  = normalize(at-eye)
    x = normalize(np.cross(up, z))
    y = np.cross(z, x)
    xhat = x.T @ eye
    yhat = y.T @ eye
    zhat = z.T @ eye
    return np.array([
        [*x, -xhat],
        [*y, -yhat],
        [*z, -zhat],
        [0, 0, 0, 1]
    ])

eye = np.array([1, 0, 0])
at = np.array([0,0,0])
up = np.array([0,1, 0])
vm = make_view_matrix(eye, at, up)
vm @ np.array([.5, 0, 0, 1])