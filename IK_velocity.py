import numpy as np
from lib.calcJacobian import calcJacobian
from scipy.linalg import lstsq


def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """
    ## STUDENT CODE GOES HERE

    dq = np.zeros((1, 7))
    J = calcJacobian(q_in)
    #J_inv = np.linalg.pinv(J)
    v_in = v_in.reshape((3,1))
    omega_in = omega_in.reshape((3,1))
    
    #zeta = np.vstack((v_in,omega_in))
    zeta = np.zeros((1,6))
    zeta = zeta.reshape((6,1))
    zeta[0:3 , 0:1]  = v_in
    zeta[3:6 , 0:1]  = omega_in
    nan_= ~np.isnan(zeta)
    J = calcJacobian(q_in)
    J_nan_=J[nan_.flatten(),:]
    J_nan_inv = np.linalg.pinv(J_nan_)
    zeta_=zeta[nan_]
    zeta_=zeta.reshape((zeta_.shape[0],1))
    #dq = np.linalg.lstsq(J_nan_,zeta_, rcond=None)[0]
    dq = np.dot(J_nan_inv,zeta_)
    return dq
