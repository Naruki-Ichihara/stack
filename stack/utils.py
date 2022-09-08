from dolfin import *
from dolfin_adjoint import *
import ufl

def strain_to_voigt(e):
    r"""Returns the pseudo-vector in the Voigt notation associate to a 2x2
    symmetric strain tensor, according to the following rule (see e.g.
    https://en.wikipedia.org/wiki/Voigt_notation),
        .. math::
         e  = \begin{bmatrix} e_{00} & e_{01}\\ e_{01} & e_{11} \end{bmatrix}\quad\to\quad
         e_\mathrm{voigt}= \begin{bmatrix} e_{00} & e_{11}& 2e_{01} \end{bmatrix}
    Args:
        e: a symmetric 2x2 strain tensor, typically UFL form with shape (2,2)
    Returns:
        a UFL form with shape (3,1) corresponding to the input tensor in Voigt
        notation.
    """
    return as_vector((e[0, 0], e[1, 1], 2*e[0, 1]))

def stress_to_voigt(sigma):
    r"""Returns the pseudo-vector in the Voigt notation associate to a 2x2
    symmetric stress tensor, according to the following rule (see e.g.
    https://en.wikipedia.org/wiki/Voigt_notation),
        .. math::
         \sigma  = \begin{bmatrix} \sigma_{00} & \sigma_{01}\\ \sigma_{01} & \sigma_{11} \end{bmatrix}\quad\to\quad
         \sigma_\mathrm{voigt}= \begin{bmatrix} \sigma_{00} & \sigma_{11}& \sigma_{01} \end{bmatrix}
    Args:
        sigma: a symmetric 2x2 stress tensor, typically UFL form with shape
        (2,2).
    Returns:
        a UFL form with shape (3,1) corresponding to the input tensor in Voigt notation.
    """
    return as_vector((sigma[0, 0], sigma[1, 1], sigma[0, 1]))

def strain_from_voigt(e_voigt):
    r"""Inverse operation of strain_to_voigt.
    Args:
        sigma_voigt: UFL form with shape (3,1) corresponding to the strain
        pseudo-vector in Voigt format
    Returns:
        a symmetric stress tensor, typically UFL form with shape (2,2)
    """
    return as_matrix(((e_voigt[0], e_voigt[2]/2.), (e_voigt[2]/2., e_voigt[1])))


def stress_from_voigt(sigma_voigt):
    r"""Inverse operation of stress_to_voigt.
    Args:
        sigma_voigt: UFL form with shape (3,1) corresponding to the stress
        pseudo-vector in Voigt format.
    Returns:
        a symmetric stress tensor, typically UFL form with shape (2,2)
    """
    return as_matrix(((sigma_voigt[0], sigma_voigt[2]), (sigma_voigt[2], sigma_voigt[1])))

class Problem(NonlinearProblem):
    r"""Probrem
    """
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)

def inner_e(x, y, restrict_to_one_side=False, quadrature_degree=1):
    r"""The inner product of the tangential component of a vector field on all
    of the facets of the mesh (Measure objects dS and ds).
    By default, restrict_to_one_side is False. In this case, the function will
    return an integral that is restricted to both sides ('+') and ('-') of a
    shared facet between elements. You should use this in the case that you
    want to use the 'projected' version of DuranLibermanSpace.
    If restrict_to_one_side is True, then this will return an integral that is
    restricted ('+') to one side of a shared facet between elements. You should
    use this in the case that you want to use the `multipliers` version of
    DuranLibermanSpace.
    Args:
        x: DOLFIN or UFL Function of rank (2,) (vector).
        y: DOLFIN or UFL Function of rank (2,) (vector).
        restrict_to_one_side (Optional[bool]: Default is False.
        quadrature_degree (Optional[int]): Default is 1.
    Returns:
        UFL Form.
    """
    dSp = Measure('dS', metadata={'quadrature_degree': quadrature_degree})
    dsp = Measure('ds', metadata={'quadrature_degree': quadrature_degree})
    n = ufl.geometry.FacetNormal(x.ufl_domain())
    t = as_vector((-n[1], n[0]))
    a = (inner(x, t)*inner(y, t))('+')*dSp + \
        (inner(x, t)*inner(y, t))*dsp
    if not restrict_to_one_side:
        a += (inner(x, t)*inner(y, t))('-')*dSp
    return a