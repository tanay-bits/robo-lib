import math
from numpy import *

### HELPER FUNCTIONS ###
def randomVec(x):
    '''
    Generates a vector of length x, each component being a random float b/w -10 and 10
    '''
    return random.uniform(-10,10,(x,1))


def randomUnitAxisAngle():
    '''
    Generates a random unit axis and an angle
    '''
    # Random longitude
    u = random.uniform(0, 1)
    longitude = 2*math.pi*u

    # Random latitude
    v = random.uniform(0, 1)
    latitude = 2*math.pi*v

    # Random unit rotation axis
    axis = zeros((3,1))
    axis[0] = math.cos(latitude)*math.cos(longitude)
    axis[1] = math.cos(latitude)*math.sin(longitude)
    axis[2] = math.sin(latitude)

    # Random angle b/w 0 and 2pi
    theta = 2*math.pi*random.uniform(0, 1)

    return axis, theta


def normalize(v):
    '''
    Returns the normalized version of the vector v
    '''
    norm = linalg.norm(v)
    if norm == 0: 
       return v
    return v/norm


def is_identity_matrix(M):
    '''
    Returns True if input M is close to an identity matrix
    '''
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
    '''
    Returns True if input R is a valid rotation matrix.
    '''
    R = asarray(R)
    return is_identity_matrix(dot(R.T,R)) and (abs(linalg.det(R)-1) < 0.001)


### MAIN FUNCTIONS ###
def RotInv(R):
    '''
    Takes a rotation matrix belonging to SO(3) and returns its inverse.
    Example:

    R = [[.707,-.707,0],[.707,.707,0],[0,0,1]]
    RotInv(R)
    >> array([[ 0.707,  0.707,  0.   ],
              [-0.707,  0.707,  0.   ],
              [ 0.   ,  0.   ,  1.   ]])
    '''
    R = asarray(R)
    assert is_rot_matrix(R), 'Not a valid rotation matrix'

    return R.T


def VecToso3(w):
    '''
    Takes a 3-vector representing angular velocity and returns the 3x3 skew-symmetric matrix version, an element
    of so(3).
    Example:

    w = [2, 1, -4]
    VecToso3(w)
    >> array([[ 0,  4,  1],
              [-4,  0, -2],
              [-1,  2,  0]])
    '''
    w = asarray(w)
    assert len(w) == 3, 'Not a 3-vector'

    w = w.flatten()
    w_so3mat = array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return w_so3mat


def so3ToVec(w_so3mat):
    '''
    Takes a 3x3 skew-symmetric matrix (an element of so(3)) and returns the corresponding 3-vector.
    Example:

    w_so3mat = [[ 0,  4,  1],[-4,  0, -2],[-1,  2,  0]]  
    so3ToVec(w_so3mat)
    >> array([[ 2],
              [ 1],
              [-4]]) 
    '''
    w_so3mat = asarray(w_so3mat)
    assert w_so3mat.shape == (3,3), 'Not a 3x3 matrix'

    w = array([[w_so3mat[2,1]], [w_so3mat[0,2]], [w_so3mat[1,0]]])
    return w


def AxisAng3(r):
    '''
    Takes a 3-vector of exp coords r = w_unit*theta and returns w_unit and theta.
    Example:
    
    r = [2, 1, -4]
    w_unit, theta = AxisAng3(r)
    w_unit
    >> array([ 0.43643578,  0.21821789, -0.87287156])
    theta
    >> 4.5825756949558398
    '''
    r = asarray(r)
    assert len(r) == 3, 'Not a 3-vector'

    theta = linalg.norm(r)
    w_unit = normalize(r)

    return w_unit, theta


def MatrixExp3(r):
    '''
    Takes a 3-vector of exp coords r = w_unit*theta and returns the corresponding
    rotation matrix R (an element of SO(3)).
    Example:

    r = [2, 1, -4]
    MatrixExp3(r)
    >> array([[ 0.08568414, -0.75796072, -0.64664811],
              [ 0.97309386, -0.07566572,  0.2176305 ],
              [-0.21388446, -0.64789679,  0.73108357]])
    '''
    r = asarray(r)
    assert len(r) == 3, 'Not a 3-vector'

    w_unit, theta = AxisAng3(r)
    w_so3mat = VecToso3(w_unit)

    R = identity(3) + math.sin(theta)*w_so3mat + (1-math.cos(theta))*dot(w_so3mat, w_so3mat)
    assert is_rot_matrix(R), 'Did not produce a valid rotation matrix'
    return R


def MatrixLog3(R):
    '''
    Takes a rotation matrix R and returns the corresponding 3-vector of exp coords r = w_unit*theta.
    Example:

    R = [[.707,-.707,0],[.707,.707,0],[0,0,1]]
    MatrixLog3(R)
    >> array([[ 0.        ],
              [ 0.        ],
              [ 0.78554916]])
    '''
    R = asarray(R)
    assert is_rot_matrix(R), 'Not a valid rotation matrix'

    if is_identity_matrix(R):
        return zeros((3,1))

    if trace(R) > -1.001 and trace(R) < -0.999:
        theta = math.pi
        c = max(diag(R))
        if c == R[2,2]:
            w_unit = array([[R[0,2]],[R[1,2]],[1+c]])*1/((2*(1+c))**0.5)
            return w_unit*theta
        elif c == R[1,1]:
            w_unit = array([[R[0,1]],[1+c],[R[2,1]]])*1/((2*(1+c))**0.5)
            return w_unit*theta
        elif c == R[0,0]:
            w_unit = array([[1+c],[R[1,0]],[R[2,0]]])*1/((2*(1+c))**0.5)
            return w_unit*theta

    theta = math.acos((trace(R)-1)/2)
    w_so3mat = (R - R.T)/(2*math.sin(theta))
    w_unit = normalize(so3ToVec(w_so3mat))
    return w_unit*theta


def RpToTrans(R,p):
    '''
    Takes a rotation matrix R and a point (3-vector) p, and returns the corresponding
    4x4 transformation matrix T, an element of SE(3).
    Example:

    R = [[.707,-.707,0],[.707,.707,0],[0,0,1]]
    p = [5,-4,9]
    RpToTrans(R,p)
    >> array([[ 0.707, -0.707,  0.   ,  5.   ],
              [ 0.707,  0.707,  0.   , -4.   ],
              [ 0.   ,  0.   ,  1.   ,  9.   ],
              [ 0.   ,  0.   ,  0.   ,  1.   ]])    
    '''
    p = asarray(p)
    R = asarray(R)
    assert len(p) == 3, "Point not a 3-vector"
    assert is_rot_matrix(R), "R not a valid rotation matrix"

    p.shape = (3,1)
    T = vstack((hstack((R,p)), array([0,0,0,1])))
    return T


def TransToRp(T):
    '''
    Takes a transformation matrix T and returns the corresponding R and p.
    Example:

    T = [[0.707,-0.707,0,5],[0.707,0.707,0,-4],[0,0,1,9],[0,0,0,1]]
    R, p = TransToRp(T)
    R
    >> array([[ 0.707, -0.707,  0.   ],
              [ 0.707,  0.707,  0.   ],
              [ 0.   ,  0.   ,  1.   ]])
    p
    >> array([[ 5.],
              [-4.],
              [ 9.]])
    '''
    T = asarray(T)
    assert T.shape == (4,4), "Input not a 4x4 matrix"

    R = T[:3,:3]
    assert is_rot_matrix(R), "Input not a valid transformation matrix"

    p = T[:3,-1]
    p.shape = (3,1)

    return R, p


def TransInv(T):
    '''
    Returns inverse of transformation matrix T.
    Example:

    T = [[0.707,-0.707,0,5],[0.707,0.707,0,-4],[0,0,1,9],[0,0,0,1]]
    TransInv(T)
    >> array([[ 0.707,  0.707,  0.   , -0.707],
              [-0.707,  0.707,  0.   ,  6.363],
              [ 0.   ,  0.   ,  1.   , -9.   ],
              [ 0.   ,  0.   ,  0.   ,  1.   ]])
    '''
    T = asarray(T)
    R, p = TransToRp(T)

    T_inv = RpToTrans(RotInv(R), dot(-RotInv(R),p))
    return T_inv


def VecTose3(V):
    '''
    Takes a 6-vector (representing spatial velocity) and returns the corresponding 4x4 matrix,
    an element of se(3).
    Example:

    V = [3,2,5,-3,7,0]
    VecTose3(V)
    >> array([[ 0., -5.,  2., -3.],
              [ 5.,  0., -3.,  7.],
              [-2.,  3.,  0.,  0.],
              [ 0.,  0.,  0.,  0.]])
    '''
    V = asarray(V)
    assert len(V) == 6, "Input not a 6-vector"
    
    V.shape = (6,1)
    w = V[:3]
    w_so3mat = VecToso3(w)
    v = V[3:]
    V_se3mat = vstack((hstack((w_so3mat,v)), zeros(4)))    
    return V_se3mat


def se3ToVec(V_se3mat):
    '''
    Takes an element of se(3) and returns the corresponding (6-vector) spatial velocity.
    Example:

    V_se3mat = [[ 0., -5.,  2., -3.],
                [ 5.,  0., -3.,  7.],
                [-2.,  3.,  0.,  0.],
                [ 0.,  0.,  0.,  0.]]
    se3ToVec(V_se3mat)
    >> array([[ 3.],
              [ 2.],
              [ 5.],
              [-3.],
              [ 7.],
              [ 0.]])
    '''
    V_se3mat = asarray(V_se3mat)
    assert V_se3mat.shape == (4,4), "Matrix is not 4x4"

    w_so3mat = V_se3mat[:3,:3]
    w = so3ToVec(w_so3mat)

    v = V_se3mat[:3,-1]
    v.shape = (3,1)

    V = vstack((w,v))
    return V


def Adjoint(T):
    '''
    Takes a transformation matrix T and returns the 6x6 adjoint representation [Ad_T]
    Example:

    T = [[0.707,-0.707,0,5],[0.707,0.707,0,-4],[0,0,1,9],[0,0,0,1]]
    Adjoint(T)
    >> array([[ 0.707, -0.707,  0.   ,  0.   ,  0.   ,  0.   ],
              [ 0.707,  0.707,  0.   ,  0.   ,  0.   ,  0.   ],
              [ 0.   ,  0.   ,  1.   ,  0.   ,  0.   ,  0.   ],
              [-6.363, -6.363, -4.   ,  0.707, -0.707,  0.   ],
              [ 6.363, -6.363, -5.   ,  0.707,  0.707,  0.   ],
              [ 6.363,  0.707,  0.   ,  0.   ,  0.   ,  1.   ]])
    '''
    T = asarray(T)
    assert T.shape == (4,4), "Input not a 4x4 matrix"

    R, p = TransToRp(T)
    p = p.flatten()
    p_skew = array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])

    ad1 = vstack((R, dot(p_skew,R)))
    ad2 = vstack((zeros((3,3)),R))
    adT = hstack((ad1,ad2))
    return adT


def ScrewToAxis(q,s_hat,h):
    '''
    Takes a point q (3-vector) on the screw, a unit axis s_hat (3-vector) in the direction of the screw,
    and a screw pitch h (scalar), and returns the corresponding 6-vector screw axis S (a normalized 
    spatial velocity).
    Example:

    q = [3,0,0]
    s_hat = [0,0,1]
    h = 2
    ScrewToAxis(q,s_hat,h)
    >> array([[ 0],
              [ 0],
              [ 1],
              [ 0],
              [-3],
              [ 2]])
    '''
    q = asarray(q)
    s_hat = asarray(s_hat)
    assert len(q) == len(s_hat) == 3, "q or s_hat not a 3-vector"
    assert abs(linalg.norm(s_hat) - 1) < 0.001, "s_hat not a valid unit vector"
    assert isscalar(h), "h not a scalar"

    q = q.flatten()
    s_hat = s_hat.flatten()

    v_wnorm = -cross(s_hat, q) + h*s_hat
    v_wnorm.shape = (3,1)
    w_unit = s_hat
    w_unit.shape = (3,1)
    S = vstack((w_unit,v_wnorm))
    return S


def AxisAng6(STheta):
    '''
    Takes a 6-vector of exp coords STheta and returns the screw axis S and the distance traveled along/
    about the screw axis theta.
    Example:

    STheta = [0,0,1,0,-3,2]
    S, theta = AxisAng6(STheta)
    S
    >> array([[ 0.],
              [ 0.],
              [ 1.],
              [ 0.],
              [-3.],
              [ 2.]])
    theta
    >> 1.0
    '''
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
    '''
    Takes a 6-vector of exp coords STheta and returns the corresponding 4x4 transformation matrix T.
    Example:

    STheta = [0,0,1,0,-3,2]
    MatrixExp6(STheta)
    >> array([[ 0.54030231, -0.84147098,  0.        ,  1.37909308],
              [ 0.84147098,  0.54030231,  0.        , -2.52441295],
              [ 0.        ,  0.        ,  1.        ,  2.        ],
              [ 0.        ,  0.        ,  0.        ,  1.        ]])
    '''
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


def MatrixLog6(T):
    '''
    Takes a transformation matrix T and returns the corresponding 6-vector of exp coords STheta.
    Example:

    T = [[ 0.54030231, -0.84147098,  0.        ,  1.37909308],
         [ 0.84147098,  0.54030231,  0.        , -2.52441295],
         [ 0.        ,  0.        ,  1.        ,  2.        ],
         [ 0.        ,  0.        ,  0.        ,  1.        ]]
    MatrixLog6(T):
    >> array([[  0.00000000e+00],
              [  0.00000000e+00],
              [  9.99999995e-01],
              [  1.12156694e-08],
              [ -2.99999999e+00],
              [  2.00000000e+00]])
    '''
    T = asarray(T)
    assert T.shape == (4,4), "Input not a 4x4 matrix"

    R, p = TransToRp(T)

    if is_identity_matrix(R):
        w_unit = zeros((3,1))
        vTheta = p
        STheta = vstack((w_unit, p))
        return STheta

    if trace(R) > -1.001 and trace(R) < -0.999:
        theta = math.pi
        wTheta = MatrixLog3(R)
        w_unit = wTheta/theta
        w_so3mat = VecToso3(w_unit)
        Ginv = identity(3)/theta - w_so3mat/2 + (1/theta - 1/(math.tan(theta/2)*2))*dot(w_so3mat,w_so3mat)
        v_unit = dot(Ginv, p)
        vTheta = v_unit*theta
        STheta = vstack((wTheta, vTheta))
        return STheta

    theta = math.acos((trace(R)-1)/2)
    w_so3mat = (R - R.T)/(2*math.sin(theta))
    Ginv = identity(3)/theta - w_so3mat/2 + (1/theta - 1/(math.tan(theta/2)*2))*dot(w_so3mat,w_so3mat)
    wTheta = MatrixLog3(R)
    v_unit = dot(Ginv, p)
    vTheta = v_unit*theta
    STheta = vstack((wTheta, vTheta)) 
    return STheta


def FKinFixed(M,Slist,thetalist):
    '''
    Takes
    - an element of SE(3): M representing the configuration of the end-effector frame when
      the manipulator is at its home position (all joint thetas = 0),
    - a list of screw axes Slist for the joints w.r.t fixed world frame,
    - a list of joint coords thetalist,
    and returns the T of end-effector frame w.r.t. fixed world frame when the joints are at
    the thetas specified.
    Example:

    S1 = [0,0,1,4,0,0]
    S2 = [0,0,0,0,1,0]
    S3 = [0,0,-1,-6,0,-0.1]
    Slist = [S1, S2, S3]
    thetalist = [math.pi/2, 3, math.pi]
    M = [[-1,  0,  0,  0],
         [ 0,  1,  0,  6],
         [ 0,  0, -1,  2],
         [ 0,  0,  0,  1]]
    FKinFixed(M,Slist,thetalist)
    >> array([[ -1.14423775e-17,   1.00000000e+00,   0.00000000e+00, -5.00000000e+00],
              [  1.00000000e+00,   1.14423775e-17,   0.00000000e+00, 4.00000000e+00],
              [  0.00000000e+00,   0.00000000e+00,  -1.00000000e+00, 1.68584073e+00],
              [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]])
    '''
    M = asarray(M)
    R_M = TransToRp(M)[0]
    assert M.shape == (4,4), "M not a 4x4 matrix"
    assert len(Slist[0]) == 6, "Incorrect Screw Axis length"
    Slist = asarray(Slist).T

    c = MatrixExp6(Slist[:,0]*thetalist[0])
    for i in range(len(thetalist)-1):        
        nex = MatrixExp6(Slist[:,i+1]*thetalist[i+1])
        c = dot(c, nex)

    T_se = dot(c, M)
    return T_se


def FKinBody(M,Slist,thetalist):
    '''
    Same as FKinFixed, except here the screw axes are expressed in the end-effector frame.
    Example:

    B1 = [0,0,-1,2,0,0]
    B2 = [0,0,0,0,1,0]
    B3 = [0,0,1,0,0,0.1]
    Blist = [S1b, S2b, S3b]
    thetalist = [math.pi/2, 3, math.pi]
    M = [[-1,  0,  0,  0],
         [ 0,  1,  0,  6],
         [ 0,  0, -1,  2],
         [ 0,  0,  0,  1]]
    FKinBody(M,Blist,thetalist)
    >> array([[ -1.14423775e-17,   1.00000000e+00,   0.00000000e+00, -5.00000000e+00],
              [  1.00000000e+00,   1.14423775e-17,   0.00000000e+00, 4.00000000e+00],
              [  0.00000000e+00,   0.00000000e+00,  -1.00000000e+00, 1.68584073e+00],
              [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]])
    '''
    M = asarray(M)
    R_M = TransToRp(M)[0]
    assert M.shape == (4,4), "M not a 4x4 matrix"
    assert len(Slist[0]) == 6, "Incorrect Screw Axis length"
    Slist = asarray(Slist).T

    c = dot(M, MatrixExp6(Slist[:,0]*thetalist[0]))
    for i in range(len(thetalist)-1):        
        nex = MatrixExp6(Slist[:,i+1]*thetalist[i+1])
        c = dot(c, nex)

    T_se = c
    return T_se


### end of HW2 functions #############################
### start of HW1 functions ###########################

def FixedJacobian(Slist,thetalist):
    '''
    Takes a list of joint angles (thetalist) and a list of screw axes (Slist) expressed in
    fixed space frame, and returns the space Jacobian (a 6xN matrix, where N is the # joints).
    Example:

    S1 = [0,0,1,4,0,0]
    S2 = [0,0,0,0,1,0]
    S3 = [0,0,-1,-6,0,-0.1]
    Slist = [S1, S2, S3]
    thetalist = [math.pi/2, 3, math.pi]
    FixedJacobian(Slist,thetalist)
    >> array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
              [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
              [  1.00000000e+00,   0.00000000e+00,  -1.00000000e+00],
              [  4.00000000e+00,  -1.00000000e+00,  -4.00000000e+00],
              [  0.00000000e+00,   1.11022302e-16,  -5.00000000e+00],
              [  0.00000000e+00,   0.00000000e+00,  -1.00000000e-01]])
        '''
    N = len(thetalist)
    J = zeros((6,N))
    Slist = asarray(Slist).T

    J[:,0] = Slist[:,0]
    for k in range(1,N):
        c = MatrixExp6(Slist[:,0]*thetalist[0])
        for i in range(k-1):        
            nex = MatrixExp6(Slist[:,i+1]*thetalist[i+1])
            c = dot(c, nex)
        J[:,k] = dot(Adjoint(c), Slist[:,k])

    return J


def BodyJacobian(Slist,thetalist):
    '''
    Takes a list of joint angles (thetalist) and a list of screw axes (Slist) expressed in
    end-effector body frame, and returns the body Jacobian (a 6xN matrix, where N is the # joints).
    Example:
    
    B1 = [0,0,-1,2,0,0]
    B2 = [0,0,0,0,1,0]
    B3 = [0,0,1,0,0,0.1]
    Blist = [B1, B2, B3]
    thetalist = [math.pi/2, 3, math.pi]
    BodyJacobian(Blist,thetalist)
    >> array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
              [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
              [ -1.00000000e+00,   0.00000000e+00,   1.00000000e+00],
              [ -5.00000000e+00,   1.22464680e-16,   0.00000000e+00],
              [ -6.12323400e-16,  -1.00000000e+00,   0.00000000e+00],
              [  0.00000000e+00,   0.00000000e+00,   1.00000000e-01]])
    '''
    N = len(thetalist)
    J = zeros((6,N))
    Slist = asarray(Slist).T

    J[:,N-1] = Slist[:,N-1]
    for k in range(N-1):
        c = MatrixExp6(-Slist[:,k+1]*thetalist[k+1])
        for i in range(k+2, len(thetalist)):        
            nex = MatrixExp6(-Slist[:,i]*thetalist[i])
            c = dot(nex, c)
        J[:,k] = dot(Adjoint(c), Slist[:,k])

    return J


def IKinBody(Slist, M, T_sd, thetalist_init, wthresh, vthresh):
    T_sd = asarray(T_sd)
    assert T_sd.shape == (4,4), "T_sd not a 4x4 matrix"
    
    maxiterates = 100
    
    N = len(thetalist_init)

    jointAngles = asarray(thetalist_init).reshape(1,N)

    T_sb = FKinBody(M, Slist, thetalist_init)
    Vb = MatrixLog6(dot(TransInv(T_sb), T_sd))
    wb = Vb[:3, 0]
    vb = Vb[3:, 0]

    thetalist_i = asarray(thetalist_init)

    i = 0

    while i<maxiterates and (linalg.norm(wb)>wthresh or linalg.norm(vb)>vthresh):
        thetalist_next = thetalist_i.reshape(N,1) + dot(linalg.pinv(BodyJacobian(Slist,thetalist_i)), Vb)
        jointAngles = vstack((jointAngles, thetalist_next.reshape(1,N)))
        T_sb = FKinBody(M, Slist, thetalist_next.flatten())
        Vb = MatrixLog6(dot(TransInv(T_sb), T_sd))
        thetalist_i = thetalist_next.reshape(N,)
        wb = Vb[:3, 0]
        vb = Vb[3:, 0]
        i += 1

    return jointAngles


def IKinFixed(Slist, M, T_sd, thetalist_init, wthresh, vthresh):
    T_sd = asarray(T_sd)
    assert T_sd.shape == (4,4), "T_sd not a 4x4 matrix"
    
    maxiterates = 100
    
    N = len(thetalist_init)

    jointAngles = asarray(thetalist_init).reshape(1,N)

    T_sb = FKinFixed(M, Slist, thetalist_init)
    Vb = MatrixLog6(dot(TransInv(T_sb), T_sd))
    wb = Vb[:3, 0]
    vb = Vb[3:, 0]

    thetalist_i = asarray(thetalist_init)

    i = 0

    while i<maxiterates and (linalg.norm(wb)>wthresh or linalg.norm(vb)>vthresh):
        Jb = dot(Adjoint(TransInv(T_sb)), FixedJacobian(Slist,thetalist_i))
        thetalist_next = thetalist_i.reshape(N,1) + dot(linalg.pinv(Jb), Vb)
        jointAngles = vstack((jointAngles, thetalist_next.reshape(1,N)))
        
        T_sb = FKinFixed(M, Slist, thetalist_next.flatten())
        Vb = MatrixLog6(dot(TransInv(T_sb), T_sd))
        thetalist_i = thetalist_next.reshape(N,)
        wb = Vb[:3, 0]
        vb = Vb[3:, 0]
        i += 1

    return jointAngles
