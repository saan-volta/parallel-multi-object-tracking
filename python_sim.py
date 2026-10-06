
import numpy as np

import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
import json
from tqdm import tqdm

class KalmanSim2dConstVel:
    def __init__(self, dt, rvar, qvar):
        self.dt = dt
        
        # x_t = [p_t, v_t]
        self.x_dim = 4
        self.z_dim = 2
        
        self.A = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])

        self.qvar = qvar
        self.Q = np.array([
            [1/4*dt**4,  1/2*dt**3, 0, 0],
            [1/2*dt**3,  1*dt**2, 0, 0  ],
            [0, 0, 1/4*dt**4, 1/2*dt**3],
            [0, 0, 1/2*dt**3, 1*dt**2]
        ]) * self.qvar

        self.rvar = rvar
        self.R = np.identity(self.z_dim) * self.rvar


        # start
        self.x = np.array([0, 1, 0, 1])
        self.x_est = np.array([0, 1, 0, 1])
        self.P_est = np.identity(self.x_dim)
        
    def start(self):
        self.x = np.array([0, 1, 0, 1])
        self.x_est = np.array([0, 1, 0, 1])
        self.P_est = np.identity(self.x_dim)
        
    def measure(self):
        W = np.random.multivariate_normal( np.zeros((self.x_dim,)), cov=self.Q)
        V = np.random.multivariate_normal( np.zeros((self.z_dim,)), cov=self.R)
        self.x = self.A @ self.x + W
        z = self.H @ self.x + V
        return z
        
        
    def step(self, z):
        A = self.A
        H = self.H
        x_est = self.x_est
        P_est = self.P_est

        x_pred = A @ x_est
        P_pred = A @ P_est @ A.T + self.Q

        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x_est = x_pred + K @ (z - H @ x_pred)
        P_est = (np.identity(self.x_dim) - K @ H) @ P_pred

        self.x_est = x_est
        self.P_est = P_est
        
        return self.x_est
    
    def run(self, n):        
        x_est_list = []
        x_list = []
        z_list = []
        
        for i in range(n):
            z = self.measure()
            x_est = self.step(z)
            x_est_list.append(x_est)
            x_list.append(self.x)
            z_list.append(z)
            
            
        return np.array(x_est_list), np.array(x_list), np.array(z_list)



def sim(N_tracks, T, dt, rvar, qvar, seed=1337):
    tracks = []
    for _ in range(N_tracks): tracks.append( KalmanSim2dConstVel(dt,  rvar, qvar) )

    X_initial = np.stack([t.x for t in tracks])
    
    X_data = np.zeros((T, N_tracks, 4))
    X_est_data = np.zeros((T, N_tracks, 4))
    Z_data = np.zeros((T, N_tracks, 2))

    for t in tqdm(range(T)):
        x_est_list = []
        for i in range(N_tracks):
            z = tracks[i].measure()
            x_est = tracks[i].step(z)
        
            X_data[t, i] = tracks[i].x
            X_est_data[t, i] = x_est
            Z_data[t, i] = z   

    return X_data, X_est_data, Z_data, X_initial


def main():

    N = 128
    T = 100
    dt = 0.1
    rvar = 1
    qvar = 1

    
    _, _, Z, X0 = sim(N, T, dt, rvar, qvar)

    np.ascontiguousarray(Z, dtype=np.float32).tofile("zdata.bin")
    np.ascontiguousarray(X0, dtype=np.float32).tofile("x0data.bin")
    with open("data_in.json", "w") as f:
        json.dump({
                      "N" : N,
                      "T" : T,
                      "dt" : dt,
                      "Q_var" : qvar,
                      "R_var" : rvar,
                      "x0fname" : "x0data.bin",
                      "zfname" : "zdata.bin"
                  }, f)

    
