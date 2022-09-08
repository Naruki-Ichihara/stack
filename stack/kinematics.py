from dolfin import *
from dolfin_adjoint import *
from ufl import RestrictedElement

def nagdi_elements():
    r"""nagdi_elements
    Return finite elements for the non linear nagdi shell.
    Args:
    Returns:
        elements: Mixed elements for nagdi shell.
                    
    """
    return MixedElement([VectorElement("Lagrange", triangle, 1, dim=3),
                        VectorElement("Lagrange", triangle, 2),
                        FiniteElement("N1curl", triangle, 1),
                        RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge")])

def nagdi_strains(phi0, d0):
    r"""nagdi_strains
    Return strain measures for the nagdi-shell model.
    Args:
        phi0: Reference configuration.
        d0: Reference director.
    Returns:
        e(F): Membrane strain measure.
        k(F, d): Bending strain measure.
        gamma(F, d): Shear strain measure.
    """
    a0 = grad(phi0).T*grad(phi0)
    b0 = -0.5*(grad(phi0).T*grad(d0) + grad(d0).T*grad(phi0))
    e = lambda F: 0.5*(F.T*F - a0)
    k = lambda F, d: -0.5*(F.T*grad(d)+grad(d).T*F) - b0
    gamma = lambda F, d: F.T*d - grad(phi0).T*d0
    return e, k, gamma