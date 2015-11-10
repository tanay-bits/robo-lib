from manipulation import *

### Problem 2 (i) ###
q = [3,0,0]
s_hat = [0,0,1]
h = 2
print 'Output of problem 2 (i) is: '
print ScrewToAxis(q,s_hat,h)
print ' '


### Problem 2 (ii) ###
S = [0,1/math.sqrt(2),1/math.sqrt(2),1,2,3]
theta = 1
print 'Output of problem 2 (ii) is: '
print MatrixExp6(S*theta)
print ' '



### Problem 2 (iii) ###
T = [[1,0,0,0],[0,0,-1,0],[0,1,0,3],[0,0,0,1]]
print 'Output of problem 2 (iii) is: '
print MatrixLog6(T)
print ' '



### Problem 2 (iv) ###
S1 = array([[0],[0],[1],[4],[0],[0]])
S2 = array([[0],[0],[0],[0],[1],[0]])
S3 = array([[0],[0],[-1],[-6],[0],[-0.1]])
Slist = [S1, S2, S3]
thetalist = [math.pi/2, 3, math.pi]
M = array([[-1,  0,  0,  0],
           [ 0,  1,  0,  6],
           [ 0,  0, -1,  2],
           [ 0,  0,  0,  1]])
print 'Output of problem 2 (iv) is: '
print FKinFixed(M,Slist,thetalist)
print ' '



### Problem 2 (v) ###
S1b = array([[0],[0],[-1],[2],[0],[0]])
S2b = array([[0],[0],[0],[0],[1],[0]])
S3b = array([[0],[0],[1],[0],[0],[0.1]])
Sblist = [S1b, S2b, S3b]
thetalist = [math.pi/2, 3, math.pi]
M = array([[-1,  0,  0,  0],
           [ 0,  1,  0,  6],
           [ 0,  0, -1,  2],
           [ 0,  0,  0,  1]])
print 'Output of problem 2 (v) is: '
print FKinBody(M,Sblist,thetalist)
print ' '