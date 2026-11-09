import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

A = 2.0e-11
K = 4.0e4
Ms = 1.7e6
mu0 = 4*np.pi*1e-7
D = np.sqrt(A/K)
d0 = np.pi*D
HK = 2*K/(mu0*Ms)
s0 = 4*np.sqrt(A*K)


def rk4(t, du, h):
    k1 = np.sin(t)-h
    k2 = np.sin(t+du*k1/2)-h
    k3 = np.sin(t+du*k2/2)-h
    k4 = np.sin(t+du*k3)-h
    return t+du*(k1+2*k2+2*k3+k4)/6


def solve(h, n=4001):
    u = np.linspace(-5*np.pi, 5*np.pi, n)
    t = np.zeros(n)
    m = n//2
    t[m] = np.pi/2
    du = u[1]-u[0]
    for i in range(m+1, n):
        t[i] = rk4(t[i-1], du, h)
    for i in range(m-1, -1, -1):
        t[i] = rk4(t[i+1], -du, h)
    return D*u, t


def exact(x):
    return 2*np.arctan(np.exp(x/D))


def width(x, t):
    x1 = np.interp(0.1*np.pi, t, x)
    x2 = np.interp(0.9*np.pi, t, x)
    xc = np.interp(np.pi/2, t, x)
    return x2-x1, xc


def press(h, perp):
    if perp:
        w1 = -K*h*h
        w2 = w1
    else:
        w1 = -mu0*Ms*h*HK
        w2 = mu0*Ms*h*HK
    return w2-w1


def main():
    data = Path(__file__).resolve().parent/'output'/'data'
    data.mkdir(parents=True, exist_ok=True)
    hs = [0, 0.05, 0.1, 0.2]
    ds = []
    rows = []
    curve = []
    ps = []
    small = []
    x, t = solve(0)
    db, xc = width(x, t)
    dt = np.pi/np.gradient(t, x)[len(x)//2]
    print(f'Ширина по формуле: {d0*1e9:.2f} нм')
    print(f'Ширина по касательной: {dt*1e9:.2f} нм')
    print(f'Энергия границы: {s0*1000:.3f} мДж/м²')

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    ax[0].plot(x*1e9, t, label='Численное решение')
    ax[0].plot(x*1e9, exact(x), '--', label='Аналитическое решение')
    xt = np.array([-dt/2, dt/2])*1e9
    ax[0].plot(xt, [0, np.pi], 'o--', label='Касательная')
    ax[0].set_ylabel(r'$\theta$, рад')
    ax[0].legend(loc='lower right')
    ax[1].plot(x*1e9, np.cos(t), label=r'$M_z/M_s$')
    ax[1].plot(x*1e9, np.sin(t), label=r'$M_y/M_s$')
    ax[1].set_ylabel('Намагниченность / Ms')
    ax[1].legend()
    for a in ax:
        a.set(xlabel='x, нм', xlim=(x[0]*1e9, x[-1]*1e9))
        a.grid()

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), layout='constrained')
    for h in hs:
        x, t = solve(h)
        d, xc = width(x, t)
        err = np.max(np.abs(t+t[::-1]-np.pi))
        ds.append([h, d*1e9, d/d0, xc*1e9, err])
        pt = press(h, True)
        pl = press(h, False)
        ps.append([h, pt, pl])
        for j in range(len(x)):
            rows.append([h, x[j]*1e9, t[j], np.sin(t[j]), np.cos(t[j])])
        ax[0].plot(x*1e9, t, label=f'h = {h:g}')
        ax[1].plot(x*1e9, np.sin(t))
        ax[2].plot(x*1e9, np.cos(t))
        print(f'h = {h:g}, ширина = {d*1e9:.2f} нм, центр = {xc*1e9:.2f} нм')
        print(f'Давление, Па: поперечное {pt:.0f}, продольное {pl:.0f}')
    ax[0].set_ylabel(r'$\theta$, рад')
    ax[0].legend(ncol=4, loc='upper left')
    ax[1].set_ylabel(r'$M_y/M_s$')
    ax[2].set_ylabel(r'$M_z/M_s$')
    for a in ax:
        a.set(xlabel='x, нм', xlim=(x[0]*1e9, x[-1]*1e9))
        a.grid()

    for h in np.linspace(0, 0.2, 81):
        x, t = solve(h)
        d, xc = width(x, t)
        curve.append([h, d/d0])
    curve = np.array(curve)
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.plot(curve[:, 0], curve[:, 1])
    ax.plot(hs, np.array(ds)[:, 2], 'o')
    ax.set(xlabel=r'$h=H_\perp/H_K$', ylabel=r'$\delta(H_\perp)/\delta_0$')
    ax.grid()

    for h in [0.001, 0.002]:
        x, t = solve(h)
        d, xc = width(x, t)
        dd = d/db-1
        small.append([h, dd])
        print(f'h = {h:g}: изменение ширины = {dd*100:.3f}%')

    files = [('profiles.csv', ['h', 'x_nm', 'theta_rad', 'My_Ms', 'Mz_Ms'], rows),
              ('widths.csv', ['h', 'width_nm', 'width_delta0', 'center_nm',
               'symmetry_error_rad'], ds),
              ('width_curve.csv', ['h', 'width_delta0'], curve),
              ('pressure.csv', ['h', 'p_perp_Pa', 'p_parallel_Pa'], ps),
              ('small_fields.csv', ['h', 'relative_width_change'], small)]
    for name, head, vals in files:
        with (data/name).open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(vals)
    plt.show()


if __name__ == '__main__':
    main()
