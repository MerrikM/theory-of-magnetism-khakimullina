"""Визуализация магнитной анизотропии однодоменной частицы"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def energy_uniaxial(x, y, z, s=1, h=0.):
    w = 1-z**2
    return s*w-2*h*x


def energy_cubic(x, y, z, s=1, h=0.):
    w = x**2*y**2+y**2*z**2+x**2*z**2
    return s*w-2*h*x


def minima(kind='uniaxial', s=1, h=0.):
    if kind == 'uniaxial':
        if s == -1:
            if h == 0:
                return None
            return np.array([[1., 0., 0.]])
        if h >= 1:
            return np.array([[1., 0., 0.]])
        z = np.sqrt(1-h**2)
        return np.array([[h, 0., z], [h, 0., -z]])

    if h >= 1:
        return np.array([[1., 0., 0.]])

    if s == 1:
        points = [[1., 0., 0.], [-1., 0., 0.]]
        if h < 2/(3*np.sqrt(6)):
            values = np.roots([2., 0., -1., h])
            for r in values:
                if abs(r.imag) < 1e-8 and 0 <= r.real < 1/np.sqrt(6):
                    u = r.real
                    v = np.sqrt(1-u**2)
                    points.append([u, v, 0.])
                    points.append([u, -v, 0.])
                    points.append([u, 0., v])
                    points.append([u, 0., -v])
        return np.array(points)

    values = np.roots([3., 0., -1., -2*h])
    real = []
    for r in values:
        if abs(r.imag) < 1e-8:
            real.append(r.real)
    branches = [max(real)]
    if h < 1/9:
        branches.append(min(real))
    points = []
    for u in branches:
        v = np.sqrt((1-u**2)/2)
        for y in [-v, v]:
            for z in [-v, v]:
                points.append([u, y, z])
    return np.array(points)


def surface(ax, kind, s, h, title, fig):
    t, p = np.meshgrid(np.linspace(0, np.pi, 91),
                       np.linspace(0, 2*np.pi, 181), indexing='ij')
    x = np.sin(t)*np.cos(p)
    y = np.sin(t)*np.sin(p)
    z = np.cos(t)
    if kind == 'uniaxial':
        w = energy_uniaxial(x, y, z, s, h)
    else:
        w = energy_cubic(x, y, z, s, h)
    m = minima(kind, s, h)
    lo = w.min()
    if m is not None:
        if kind == 'uniaxial':
            wm = energy_uniaxial(m[:, 0], m[:, 1], m[:, 2], s, h)
        else:
            wm = energy_cubic(m[:, 0], m[:, 1], m[:, 2], s, h)
        lo = min(lo, wm.min())
    c = .18-lo
    r = w+c
    X = r*x
    Y = r*y
    Z = r*z
    norm = matplotlib.colors.Normalize(lo, w.max())
    ax.plot_surface(X, Y, Z, rstride=1, cstride=2,
                    facecolors=plt.cm.viridis(norm(w)), shade=False)
    lim = max(abs(X).max(), abs(Y).max(), abs(Z).max())*1.06
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim),
           xlabel='X', ylabel='Y', zlabel='Z')
    ax.set_box_aspect((1, 1, 1))
    ax.locator_params(nbins=4)
    if h > 0:
        ax.quiver(0, 0, 0, lim*.9, 0, 0, color='royalblue')
    if m is None:
        n = '∞'
    else:
        n = len(m)
    ax.set_title(f'{title}\nN = {n}; C = {c:.3f}')
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'),
                       ax=ax, shrink=.6)
    bar.set_label(r'$\widetilde w$')


def main():
    out = Path(__file__).resolve().parent/'output'
    plt.rcParams['font.size'] = 11
    data = out/'data'
    data.mkdir(parents=True, exist_ok=True)
    hs = [0., .3, .7, 4.]
    cases = [('uniaxial', 1, r'Одноосная $K_1 > 0$'),
             ('uniaxial', -1, r'Одноосная $K_1 < 0$'),
             ('cubic', 1, r'Кубическая $K_{1c} > 0$'),
             ('cubic', -1, r'Кубическая $K_{1c} < 0$')]

    # Четыре случая без поля
    fig = plt.figure(figsize=(12, 9), layout='constrained')
    for i in range(4):
        kind, s, title = cases[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        surface(ax, kind, s, 0, title, fig)

    # Три серии во внешнем поле
    field_cases = [cases[0], cases[2], cases[3]]
    for i in range(3):
        kind, s, title = field_cases[i]
        fig = plt.figure(figsize=(12, 9), layout='constrained')
        for j in range(4):
            h = hs[j]
            ax = fig.add_subplot(2, 2, j+1, projection='3d')
            surface(ax, kind, s, h, f'{title}; h = {h:g}', fig)

    # Направления минимумов
    with (data/'minima.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['kind', 'sign', 'h', 'N',
                         'alpha_x', 'alpha_y', 'alpha_z',
                         'theta_deg', 'phi_deg', 'w_tilde'])
        for kind, s, title in cases:
            if kind == 'uniaxial' and s == -1:
                fields = [0.]
            else:
                fields = hs
            for h in fields:
                m = minima(kind, s, h)
                if m is None:
                    n = '∞'
                else:
                    n = len(m)
                print(f'{title}; h = {h:g}: N = {n}')
                if m is None:
                    writer.writerow([kind, s, h, 'inf', '', '', 0, 90, 'any', -1])
                else:
                    for a in m:
                        theta = np.degrees(np.arccos(np.clip(a[2], -1, 1)))
                        if np.hypot(a[0], a[1]) > 1e-10:
                            phi = np.degrees(np.arctan2(a[1], a[0])) % 360
                        else:
                            phi = 'any'
                        if kind == 'uniaxial':
                            w = energy_uniaxial(a[0], a[1], a[2], s, h)
                        else:
                            w = energy_cubic(a[0], a[1], a[2], s, h)
                        writer.writerow([kind, s, h, len(m), a[0], a[1], a[2],
                                         theta, phi, w])

    # Число локальных минимумов
    hp, hn = 2/(3*np.sqrt(6)), 1/9
    hs_count = np.linspace(0, 4, 801)
    hs_count = np.append(hs_count, [hp, hn, 1.])
    hs_count = np.unique(hs_count)
    counts = []
    for h in hs_count:
        row = []
        for i in range(3):
            kind, s, title = field_cases[i]
            m = minima(kind, s, h)
            row.append(len(m))
        counts.append(row)
    counts = np.array(counts)
    np.savetxt(data/'counts.csv', np.column_stack((hs_count, counts)),
               delimiter=',',
               header='h,N_uniaxial_positive,N_cubic_positive,N_cubic_negative',
               comments='')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for ax in axes:
        for j in range(3):
            kind, s, title = field_cases[j]
            if j == 1:
                style = '--'
            else:
                style = '-'
            ax.step(hs_count, counts[:, j], where='post', label=title,
                    linestyle=style)
        ax.set(xlabel='h = H / Hₖ', ylabel='Число минимумов N', yticks=range(1, 9))
        ax.grid()
    axes[0].set(xlim=(0, 1.15), title='Область переходов')
    axes[1].set(xlim=(0, 4), title='Полный диапазон поля')
    axes[1].legend()
    print('Данные:', data.resolve())
    plt.show()


if __name__ == '__main__':
    main()
