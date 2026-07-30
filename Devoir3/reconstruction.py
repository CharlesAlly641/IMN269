import numpy as np

def triangulation(pg, pd, R, T, Mg, Md):
    ones = np.ones((pg.shape[0], 1))
    pg_h = np.hstack((pg, ones))
    pd_h = np.hstack((pd, ones))

    # pg et pd exprimés dans leur repère
    pg = np.dot(np.linalg.inv(Mg), pg_h)
    pd = np.dot(np.linalg.inv(Md), pd_h)

    A = np.array([np.dot(-R.T, pd), pg, np.cross(pg_h, R.T * pd)])
    a, b, c = np.linalg.solve(A, T)

    Pg = b * pg
    Pd = a * np.dot(-R.T, pd)
    Ps = c * np.cross(pg, np.dot(R.T, pd))


