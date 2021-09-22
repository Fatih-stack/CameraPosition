import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import ransac

def rigid_transform_3d(xs,ys):
    """
    expects 2 arrays of shape (3, N)

    rigid transform algorithm from
    http://nghiaho.com/?page_id=671
    """
    assert xs.shape == ys.shape
    assert xs.shape[0] == 3, 'The points must be of dimmensionality 3'

    # find centroids and H
    x_centroid = np.mean(xs, axis=1)[:, np.newaxis]
    y_centroid = np.mean(ys, axis=1)[:, np.newaxis]
    
    H = (xs - x_centroid)@(ys - y_centroid).T

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    rotation = Vt.T@U.T

    # handling reflection
    if np.linalg.det(rotation) < 0:
        Vt[2, :] *= -1
        rotation = np.dot(Vt.T, U.T)
    
    # find translation
    translation = y_centroid - rotation@x_centroid
    
    return translation, rotation

class Translation:
    def __init__(self):
        self.R = np.eye(3)
        self.t = np.zeros(3)

    def estimate(self, src, dst):
        self.t, self.R = rigid_transform_3d(src.T, dst.T)

    def residuals(self, src, dst):
        residuals = []
        for p1, p2 in zip(src, dst):
            diff = np.dot(self.R, p1) + self.t - p2
            residuals.append(np.linalg.norm(diff))

        return np.array(residuals)

IntrinsicC = [[100.0, 0.0, 960.0], [0.0, 100.0, 540.0], [0.0, 0.0, 1.0]]

def to_real(x,y,z):
    A = np.asarray(IntrinsicC)
    return (z*np.linalg.inv(A)@np.asarray([x,y,1]))

def matchgen(n):
    imageA = cv2.imread('img'+str(n)+'.png',cv2.IMREAD_UNCHANGED)
    imageB = cv2.imread('img'+str(n+1)+'.png',cv2.IMREAD_UNCHANGED)
    
    # ORB Detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(imageA, None)
    kp2, des2 = orb.detectAndCompute(imageB, None)

    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    
    setA,setB = [],[]
    for match in matches[:200]:
        
        xa,ya = kp1[match.queryIdx].pt
        xb,yb = kp2[match.trainIdx].pt
        za = 1
        zb = 1
    
        if za!=0 and zb!=0:
            setA.append(list(map(int,to_real(xa,ya,za))))
            setB.append(list(map(int,to_real(xb,yb,zb))))

    return np.asarray(setA),np.asarray(setB)

points2d = np.load('vr2d.npy')
points3d = np.load('vr3d.npy')
#f=100 cx=960 cy=540

Rotation = []
Translat = []

SetA,SetB = matchgen(1)

R,t = ransac((SetA,SetB),
             Translation,
             min_samples=10,
             residual_threshold=100
            )

if np.linalg.norm(R.t) < 50:
    Rotation.append(R.R)
    Translat.append(R.t)

SetA,SetB = matchgen(2)

R,t = ransac((SetA,SetB),
             Translation,
             min_samples=10,
             residual_threshold=100
            )

if np.linalg.norm(R.t) < 50:
    Rotation.append(R.R)
    Translat.append(R.t)
        
Rotation = np.array(Rotation)
Translat = np.array(Translat)

norms = [np.linalg.norm(i) for i in Translat]
plt.plot(range(len(norms)), norms)

Fullrotation= np.eye(3)
TransInCrdnt = []

for i in range(len(Rotation)):
    
    TransInCrdnt.append( Fullrotation@Translat[i].copy() )
    Fullrotation = Fullrotation@np.linalg.inv(Rotation[i].copy())
    
TransInCrdnt = np.squeeze( np.array(TransInCrdnt) )

traj = []
summ = np.array([0.,0.,0.])

for i in range(TransInCrdnt.shape[0]):
    traj.append(summ)
    summ = summ + TransInCrdnt[i]
    
traj = np.array(traj)
plt.plot(traj[:,0], traj[:,2])

Camera = np.asarray(IntrinsicC)
(result, Fullrotation, TransInCrdnt, inliers) = cv2.solvePnPRansac(points3d, points2d, Camera, None,
                                                    Fullrotation, TransInCrdnt, useExtrinsicGuess=False)
if result:
    Rned2cam, jac = cv2.Rodrigues(Fullrotation)
    pos = -np.matrix(Rned2cam[:3,:3]).T * np.matrix(TransInCrdnt)
    newned = pos.T[0].tolist()[0]
    file = open("position.txt","w") 
    file.write("Position : ")
    file.write(str(newned))
    print("Position : ",newned)
    file.close()


plt.imshow( cv2.imread('img1.png',cv2.IMREAD_UNCHANGED), extent=[300, -300, 300, -300])
plt.plot(traj[:250,0], -1*traj[:250,1],linewidth=8,c='red')

A = np.asarray(IntrinsicC)
camtraj = (A@traj.T).T
plt.plot(-1*camtraj[:100,0],camtraj[:100,1])
plt.savefig('output.png')
plt.show()
