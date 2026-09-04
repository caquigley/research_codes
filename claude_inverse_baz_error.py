#%%
import numpy as np

a0 = -5.00919269
a1 = -16.57537395; b1 = -25.89907705
a2 = -17.23837615; b2 = -14.54117385
a3 = -3.74992031;  b3 = -7.58511478
a4 = 0.7111211;    b4 = -7.3153489
a5 = -0.41993486;  b5 = -4.77020138

def f(theta_deg):
    theta = np.deg2rad(theta_deg)
    return (a0 + a1*np.cos(1*theta) + b1*np.sin(1*theta)
               + a2*np.cos(2*theta) + b2*np.sin(2*theta)
               + a3*np.cos(3*theta) + b3*np.sin(3*theta)
               + a4*np.cos(4*theta) + b4*np.sin(4*theta)
               + a5*np.cos(5*theta) + b5*np.sin(5*theta))

def f_prime(theta_deg):
    # derivative w.r.t. theta_deg, chain rule through deg2rad
    theta = np.deg2rad(theta_deg)
    dtheta_ddeg = np.pi / 180
    dfdtheta = (-a1*1*np.sin(1*theta) + b1*1*np.cos(1*theta)
                -a2*2*np.sin(2*theta) + b2*2*np.cos(2*theta)
                -a3*3*np.sin(3*theta) + b3*3*np.cos(3*theta)
                -a4*4*np.sin(4*theta) + b4*4*np.cos(4*theta)
                -a5*5*np.sin(5*theta) + b5*5*np.cos(5*theta))
    return dfdtheta * dtheta_ddeg

def estimate_real_baz(baz_array_new, tol=1e-8, max_iter=50):
    baz_array_new = np.atleast_1d(baz_array_new).astype(float)
    x = baz_array_new.copy()  # initial guess: assume error is small, start at baz_array

    for _ in range(max_iter):
        g = x - f(x) - baz_array_new
        g_prime = 1 - f_prime(x)
        step = g / g_prime
        x = x - step
        if np.max(np.abs(step)) < tol:
            break

    return x % 360
xvals = np.linspace(180,270, 40)
array_new = estimate_real_baz(208)
print('New value:', array_new)
# %%
