from dolfin import *
from dolfin_adjoint import *
import numpy as np
import stack as st
import fenics_optimize as op
from ufl import tanh 

parameters["form_compiler"]["quadrature_degree"] = 5

w = 0.5
a = (np.pi-w)/2
ya = - np.sin(a)*(-(1/np.cosh(np.cos(a)))**2)

class VectorField(UserExpression):
    def eval(self, val, x):
        val[0] = 1
        val[1] = 0
    def value_shape(self):
        return (2,)

def boundary(x, on_boundary):
    return 0

L = 10
W = 10
size = 1.
mesh = RectangleMesh(Point((0, 0)), Point(L, W), L*10,W*10)

P = np.pi/size

def stripe_projection(phi, P):
    return cos(P*phi)

Phi = FunctionSpace(mesh, 'CG', 1)
V = VectorFunctionSpace(mesh, 'CG', 1)
U = FunctionSpace(mesh, 'CG', 1)

phi_, phi, phi_t = Function(Phi), TrialFunction(Phi), TestFunction(Phi)
vec = Function(V)
vec.interpolate(VectorField())

# Energy
residual = (grad(phi_) - vec)[0]**2 + (grad(phi_) - vec)[1]**2
L = 0.5*residual*dx
dL = derivative(L, phi_, phi_t)
J = derivative(dL, phi_, phi)

bc = DirichletBC(Phi, 0, boundary)
problem = st.Problem(J, dL, bcs=[bc])
solver = NewtonSolver()
solver.parameters['error_on_nonconvergence'] = False
solver.parameters['maximum_iterations'] = 20
solver.parameters['linear_solver'] = "mumps"
solver.parameters['absolute_tolerance'] = 1E-10
solver.parameters['relative_tolerance'] = 1E-10

solver.solve(problem, phi_.vector())

stripe = op.hevisideFilter(abs(tanh(stripe_projection(phi_, P)).dx(0)), U, beta=10, eta=ya)
pattern = project(stripe, U)
file = XDMFFile('pattern.xdmf')
file.write(pattern)