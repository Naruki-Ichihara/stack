import numpy as np
from dolfin import *

def z_coordinates(hs):
    r"""Return a list with the thickness coordinate of the top surface of each layer
    taking the midplane as z = 0.
    Args:
        hs: a list giving the thinckesses of each layer
            ordered from bottom (layer - 0) to top (layer n-1).
    Returns:
        z: a list of coordinate of the top surface of each layer
           ordered from bottom (layer - 0) to top (layer n-1)
    """

    z0 = sum(hs)/2.
    z = [(-sum(hs)/2. + sum(hs for hs in hs[0:i])) for i in range(len(hs)+1)]
    return z

def AD_neo(mu, nu, hs_grobal, index):
    r"""Return the stiffness matrix of an isotropic neo-hooken mechanical stiffness.
    Args:
        mu: shear stiffness for neo-hooken.
        nu: poisson's ratio.
        hs_grobal: a list with length n with the thicknesses of the layers (from top to bottom).
        index: index of active layer.
    Returns:
        A: a symmetric 3x3 ufl matrix giving the membrane stiffness in Voigt notation.
        D: a symmetric 3x3 ufl matrix giving the bending stiffness in Voigt notation.
    """

    z = z_coordinates(hs_grobal)
    A = 0.
    D = 0.

    Y = 3*mu
    for i in range(len(hs_grobal)):
        if i in index:
            A += Y/(1-nu**2)*(z[i+1]-z[i])
            D += Y/3/(1-nu**2)*(z[i+1]**3-z[i]**3)
        else:
            A += 0.
            D += 0.      

    return (A, D)

def F_neo(mu, hs_grobal, index):
    r"""Return the shear stiffness matrix of a Reissner-Midlin model of a
    laminate obtained by stacking n isotropic lamina.
    It assumes a plane-stress state.
    Args:
        G13: The transverse shear modulus between the material directions 1-3.
        G23: The transverse shear modulus between the material directions 2-3.
        hs: a list with length n with the thicknesses of the layers (from top to bottom).
        theta: a list with the n orientations (in radians) of the layers (from top to bottom).
    Returns:
        F: a symmetric 2x2 ufl matrix giving the shear stiffness in Voigt notation.
    """

    z = z_coordinates(hs_grobal)
    F = 0.

    for i in range(len(hs_grobal)):
        if i in index:
            F += mu*(z[i+1]-z[i])

    return F

def AD_ele(epsr, V, hs_grobal, index, eps0=8.85*1e-12):
    r"""Return the general stiffness matrix of an isotropic dielectric mechanics.
        epsr: Relative permittivity.
        V: Reference potential.
        hs_grobal: a list with length n with the thicknesses of the layers (from top to bottom).
        h: total thickness.
        index: index of active layer.
        eps0: permittivity.
    Returns:
        A: a symmetric 3x3 ufl matrix giving the membrane stiffness in Voigt notation.
        D: a symmetric 3x3 ufl matrix giving the bending stiffness in Voigt notation.
    """
    z = z_coordinates(hs_grobal)
    A = 0.
    D = 0.
    for i in range(len(hs_grobal)):
        if i in index:
            A += eps0*epsr*V/hs_grobal[i]**2*(z[i+1]-z[i])
            D += eps0*epsr*V/hs_grobal[i]**2*(z[i+1]**2-z[i]**2)
        else:
            A += 0.
            D += 0.
    return (A, D)
