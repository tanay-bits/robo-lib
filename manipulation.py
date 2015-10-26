import math
from numpy import *
from random import uniform


def randomUnitAxisAngle():
    # Random longitude
    u = uniform(0, 1)
    longitude = 2*math.pi*u

    # Random latitude
    v = uniform(0, 1)
    latitude = 2*math.pi*v

    # Random rotation axis
    axis = zeros((3,1))
    axis[0] = math.cos(latitude)*math.cos(longitude)
    axis[1] = math.cos(latitude)*math.sin(longitude)
    axis[2] = math.sin(latitude)

    # Random angle
    theta = 2*math.pi*uniform(0, 1)

    return axis, theta


def normalize(v):
    norm = linalg.norm(v)
    if norm == 0: 
       return v
    return v/norm


def is_identity_matrix(M):
    if len(M) != len(M[0]):
        return False

    c = list()
    for i, row in enumerate(M):
        for j, val in enumerate(row):
            if i==j:
                if val < 1.001 and val > 0.999:
                    c.append(True)
                else:
                    c.append(False)
            else:
                if abs(val) < 0.001:
                    c.append(True)
                else:
                    c.append(False)

    return all(c)


def is_rot_matrix(R):
    return is_identity_matrix(dot(R.T,R)) and (abs(linalg.det(R)-1) < 0.001)


def RotInv(R):
    R = asarray(R)
    if not is_rot_matrix(R):
        raise RuntimeError('Not a valid rotation matrix')

    return R.T


def VecToso3(w):
    w = asarray(w)
    if len(w) != 3:
        raise RuntimeError('Not a 3-vector')

    w = w.flatten()
    w_so3mat = array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return w_so3mat


def so3ToVec(w_so3mat):
    w_so3mat = asarray(w_so3mat)
    if w_so3mat.shape != (3,3):
        raise RuntimeError('Not a 3x3 matrix')

    w = array([[w_so3mat[2,1]], [w_so3mat[0,2]], [w_so3mat[1,0]]])
    return w


def AxisAng3(r):
    r = asarray(r)
    if len(r) != 3:
        raise RuntimeError('Not a 3-vector')

    theta = linalg.norm(r)
    w_unit = normalize(r)

    return w_unit, theta


def MatrixExp3(r):
    r = asarray(r)
    if len(r) != 3:
        raise RuntimeError('Not a 3-vector')

    w_unit, theta = AxisAng3(r)
    w_so3mat = VecToso3(w_unit)

    R = identity(3) + math.sin(theta)*w_so3mat + (1-math.cos(theta))*dot(w_so3mat, w_so3mat)
    return R


def MatrixLog3(R):
    R = asarray(R)
    if not is_rot_matrix(R):
        raise RuntimeError('Not a valid rotation matrix')

    if is_identity_matrix(R):
        return zeros((3,1))

    if trace(R) > -1.001 and trace(R) < -0.999:
        theta = math.pi
        c = max(diag(R))
        if c == R[2,2]:
            w_unit = array([[R[0,2]],[R[1,2]],[1+c]])*1/((2*(1+c))**0.5)
            return w_unit*theta
        if c == R[1,1]:
            w_unit = array([[R[0,1]],[1+c],[R[2,1]]])*1/((2*(1+c))**0.5)
            return w_unit*theta
        if c == R[0,0]:
            w_unit = array([[1+c],[R[1,0]],[R[2,0]]])*1/((2*(1+c))**0.5)
            return w_unit*theta

    theta = math.acos((trace(R)-1)/2)
    w_so3mat = (R - R.T)/(2*math.sin(theta))
    w_unit = normalize(so3ToVec(w_so3mat))
    return w_unit*theta


def RpToTrans(R,p):
    p = asarray(p)
    R = asarray(R)
    assert len(p) == 3, "Point not a 3-vector"
    assert is_rot_matrix(R), "R not a valid rotation matrix"

    p.shape = (3,1)
    T = vstack((hstack((R,p)), array([0,0,0,1])))
    return T


def TransToRp(T):
    T = asarray(T)
    assert T.shape == (4,4), "Input not a 4x4 matrix"

    R = T[:3,:3]
    assert is_rot_matrix(R), "Input not a valid rotation matrix"

    p = T[:3,-1]
    p.shape = (3,1)

    return R, p


def TransInv(T):
    T = asarray(T)
    R, p = TransToRp(T)

    T_inv = RpToTrans(RotInv(R), dot(-RotInv(R),p))
    return T_inv


def VecTose3(V):
    V = asarray(V)
    assert len(V) == 6, "Input not a 6-vector"
    
    V.shape = (6,1)
    w = V[:3]
    w_so3mat = VecToso3(w)
    v = V[3:]
    V_se3mat = vstack((hstack((w_so3mat,v)), zeros(4)))    
    return V_se3mat


def se3ToVec(V_se3mat):
    V_se3mat = asarray(V_se3mat)
    assert V_se3mat.shape == (4,4), "Matrix is not 4x4"

    w_so3mat = V_se3mat[:3,:3]
    w = so3ToVec(w_so3mat)

    v = V_se3mat[:3,-1]
    v.shape = (3,1)

    V = vstack((w,v))
    return V


def Adjoint(T):
    T = asarray(T)
    assert T.shape == (4,4), "Input not a 4x4 matrix"

    R, p = TransToRp(T)
    p = p.flatten()
    p_skew = array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])

    ad1 = vstack((R, dot(p_skew,R)))
    ad2 = vstack((zeros((3,3)),R))
    adT = hstack((ad1,ad2))
    return adT


def ScrewToAxis(q,shat,h):
    q = asarray(q)
    shat = asarray(shat)
    assert len(q) == len(shat) == 3, "q or shat not a 3-vector"
    assert abs(linalg.norm(shat) - 1) < 0.001, "shat not a valid unit vector"
    assert isscalar(h), "h not a scalar"

    q = q.flatten()
    shat = shat.flatten()

    v_wnorm = -cross(shat, q) + h*shat
    v_wnorm.shape = (3,1)
    w_unit = shat
    w_unit.shape = (3,1)
    S = vstack((w_unit,v_wnorm))
    return S


def AxisAng6(STheta):
    STheta = asarray(STheta)
    assert len(STheta) == 6, 'Input not a 6-vector'

    w = STheta[:3]
    v = STheta[3:]

    if linalg.norm(w) == 0:
        theta = linalg.norm(v)
        v_unit = normalize(v)
        v_unit.shape = (3,1)
        S = vstack((zeros((3,1)), v_unit))
        return S, theta

    theta = linalg.norm(w)
    w_unit = normalize(w)
    w_unit.shape = (3,1)
    v_unit = v/theta
    v_unit.shape = (3,1)
    S = vstack((w_unit,v_unit))
    return S, theta


def MatrixExp6(STheta):
    STheta = asarray(STheta)
    assert len(STheta) == 6, 'Input not a 6-vector'

    S, theta = AxisAng6(STheta)

    w_unit = S[:3]
    v_unit = S[3:]

    vTheta = STheta[3:]
    vTheta.shape = (3,1)
    
    if linalg.norm(w_unit) == 0:
        R = identity(3)
        p = vTheta
        T = RpToTrans(R,p)
        return T

    r = w_unit*theta
    R = MatrixExp3(r)
    w_so3mat = VecToso3(w_unit)
    p = dot((identity(3)*theta+(1-math.cos(theta))*w_so3mat+(theta-math.sin(theta))*dot(w_so3mat,w_so3mat)), v_unit) 
    T = RpToTrans(R,p)
    return T









