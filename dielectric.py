from math import gamma
from dolfin import *
#from dolfin_adjoint import *
import fenics_optimize as fo
import numpy as np
import stack as st
from stack.kinematics import nagdi_strains

# Meta parameters
EPS = 3e-16
parameters["form_compiler"]["quadrature_degree"] = 5

# Stack configulation
hs_grobal = [0.20*1e-3, 0.1*1e-3]
hs_active_index = [0]
hs_passive_index = [1]
thetas = [0, np.deg2rad(0)]

# mesh definition
import mshr
radius = 50.0*1e-3
n_division = 12.0
center = Point(0, 0)
geom = mshr.Circle(center, radius)
mesh = mshr.generate_mesh(geom, n_division)

# mark of boundary conditions
def center_point(x, on_boundary):
    r = sqrt(x[0]**2 + x[1]**2)
    return r < EPS

# Function spaces definition
elements = st.nagdi_elements()
Q = FunctionSpace(mesh, elements)
Phi = FunctionSpace(mesh, VectorElement("Lagrange", triangle, degree = 1, dim = 3))
Beta = FunctionSpace(mesh, VectorElement("CG", triangle, degree = 2, dim = 2))

# Functions
q_, q, q_t = Function(Q), TrialFunction(Q), TestFunction(Q)
u_, beta_, rg_, p_ = split(q_)

phi0_expression = Expression(['x[0]', 'x[1]', 'phi0_z'], phi0_z=0, degree=4)
beta0_expression = Expression(('beta0_x', 'beta0_y'), beta0_x=0, beta0_y=0, degree=4)
phi0 = project(phi0_expression, Phi)
beta0 = project(beta0_expression, Beta)

def director(beta):
    return as_vector([sin(beta[1])*cos(beta[0]), -sin(beta[0]), cos(beta[1])*cos(beta[0])])
d0 = director(beta0)

# Boundary conditions
bc_fix_u = DirichletBC(Q.sub(0), Constant((0.0,0.0,0.0)), center_point, method='pointwise')
bc_fix_R = DirichletBC(Q.sub(1), Constant((0.0,0.0)), center_point, method='pointwise')
bcs = [bc_fix_u, bc_fix_R]

# kinematics
F = grad(u_) + grad(phi0)
d = director(beta_ + beta0)
e, k, gamma = nagdi_strains(phi0, d0)

# Active layers
# Mechanical energies
nu = 0.49
a1 = 0.7
a2 = 3.25
a3 = -3.7
mu1 = 54.88*1e3
mu2 = 910
mu3 = -6.3
mu = 0.5*(a1*mu1+a2*mu2+a3*mu3)
A_neo, D_neo = st.AD_neo(mu, nu, hs_grobal, hs_active_index)
F_neo = st.F_neo(mu, hs_grobal, hs_active_index)
def Wm(F):
    C = F.T*F
    return 0.5*A_neo/4*(tr(C) + 1/det(C) - 3)
def Wb(F, d):
    C = F.T*F
    K = inv(C)*k(F, d)
    return 0.5/det(C)*D_neo*((tr(K)**2-det(K)))
def Ws(rg_):
    return 0.5*F_neo*inner(rg_, rg_)

# Electrical energies
V = Expression('Voltage', Voltage=0, degree=0)
dV = 1.0
epsr = 4.7
A_ele, D_ele = st.AD_ele(epsr, V, hs_grobal, hs_active_index)
def Wm_el(F, V):
    C = F.T*F
    return -A_ele*V*det(C)*dV
def Wb_el(F, d, V):
    C = F.T*F
    K = inv(C)*k(F, d)
    return -2*D_ele*A_ele*V*det(C)*tr(K)*dV

#Passive layers
E1 = 6.0*1e6 #Polyflex
E2 = 1e-3
G12 = 1e-3
nu12 = 0.3
A, B, D = st.ABD(E1, E2, G12, nu12, hs_grobal, thetas, hs_passive_index)
Fs = st.Fs(G12, G12, hs_grobal, thetas, hs_passive_index)
ev = st.strain_to_voigt(e(F))
Ai = project(A, TensorFunctionSpace(mesh, 'CG', 1, shape=(3,3)))
psi_N = .5*dot(Ai*ev, ev)
Fi = project(Fs, TensorFunctionSpace(mesh, 'CG', 1, shape=(2,2)))
psi_T = .5*dot(Fi*rg_, rg_)
kv = st.strain_to_voigt(k(F, d))
Di = project(D, TensorFunctionSpace(mesh, 'CG', 1, shape=(3,3)))
psi_M = .5*dot(Di*kv, kv) 
Bi = project(B, TensorFunctionSpace(mesh, 'CG', 1, shape=(3,3)))
psi_MN = dot(Bi*kv, ev)

# Helmholtz free energy
energy_active = Wb(F, d) + Wb_el(F, d, V) + Wm(F) + Wm_el(F, V) + Ws(rg_)
energy_passive = psi_M + psi_MN + psi_N + psi_T
L_R = st.inner_e(gamma(F, d) - rg_, p_)
Pi = (energy_active + energy_passive)*dx + L_R
dPi = derivative(Pi, q_, q_t)
J = derivative(dPi, q_, q)

# Problem definition
problem = st.Problem(J, dPi, bcs=bcs)
solver = NewtonSolver()
solver.parameters['error_on_nonconvergence'] = False
solver.parameters['maximum_iterations'] = 20
solver.parameters['linear_solver'] = "mumps"
solver.parameters['absolute_tolerance'] = 1E-10
solver.parameters['relative_tolerance'] = 1E-6

V_cur = 0
V_final = 1000
disp0_expression = Expression(['0', '0', '0'], degree=4)
disp0 = Function(VectorFunctionSpace(mesh, 'CG', 1, dim=3))
disp0 = project(disp0_expression, FunctionSpace(mesh, VectorElement("Lagrange", triangle, degree = 1, dim = 3)))
while V_cur < V_final:
    V.Voltage = V_cur
    solver.solve(problem, q_.vector())
    u_h, beta_h, _, _ = q_.split(deepcopy=True)
    disp = project(disp0+u_h, VectorFunctionSpace(mesh, 'CG', 1, dim=3))
    assign(phi0, project(phi0+u_h, VectorFunctionSpace(mesh, 'CG', 1, dim=3)))
    assign(beta0, project(beta0+beta_h, VectorFunctionSpace(mesh, 'CG', 2, dim=2)))
    assign(disp0, project(disp0+u_h, VectorFunctionSpace(mesh, 'CG', 1, dim=3)))
    disp.rename('disp', 'disp')
    V_cur += dV
    if V_cur % 100 == 0:
        file = XDMFFile('output/frame/step_{}.xdmf'.format(int(V_cur)))
        file.write(disp)