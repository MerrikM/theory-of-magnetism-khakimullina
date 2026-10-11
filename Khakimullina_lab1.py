import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'font.size': 14})

h = np.linspace(4, -4, 8001)
eps = 1e-6
out = Path(__file__).parent / 'output'
(out / 'figures').mkdir(parents=True, exist_ok=True)
(out / 'data').mkdir(parents=True, exist_ok=True)


def mag(a, h):
    # Частные случаи
    if a == 0:
        if h[0] > h[-1]:
            return np.where(h > -1, 1.0, -1.0)
        return np.where(h < 1, -1.0, 1.0)
    if a == 90:
        return np.clip(h, -1, 1)

    t = np.radians(a)
    p = t if h[0] > 0 else t - np.pi
    m = []
    for x in h:
        # Начальное приближение
        for k in range(1000):
            f = 0.5 * np.sin(2 * p) - x * np.sin(t - p)
            df = np.cos(2 * p) + x * np.cos(t - p)
            if df <= 0:  # Потеря устойчивости
                p -= 0.05
                continue
            q = p - f / df
            if abs(q - p) < eps:
                p = q
                break
            p = q
        m.append(np.cos(t - p))
    return np.array(m)


def hc(h, m):
    i = np.where(
        (m[:-1] >= 0) & (m[1:] <= 0))[0][0]
    return abs(h[i] - m[i] * (h[i + 1] - h[i])
               / (m[i + 1] - m[i]))


# Петли гистерезиса
fig, ax = plt.subplots(2, 2, figsize=(8, 6.6))
for i, a in enumerate([0, 30, 60, 90]):
    m = mag(a, h)
    n = mag(a, h[::-1])
    ax.flat[i].plot(h, m, label='Убывание h')
    ax.flat[i].plot(h[::-1], n, '--',
                    label='Возрастание h')
    ax.flat[i].set(title=f'θ = {a}°',
                   xlabel='h = H / Hₖ',
                   ylabel='m = M∥ / Mₛ',
                   xlim=(-2, 2), ylim=(-1.1, 1.1))
    ax.flat[i].grid(alpha=0.2)
fig.legend(['Убывание h', 'Возрастание h'],
           loc='lower center', ncol=2)
fig.tight_layout(rect=[0, 0.09, 1, 1])
fig.savefig(out / 'figures/hysteresis.png', dpi=220)
plt.close(fig)

# Астроида
p = np.linspace(-np.pi, np.pi, 2001)
hx = -np.sin(p)**3
hz = np.cos(p)**3
fig, ax = plt.subplots(figsize=(5.8, 4.7))
ax.plot(hx, hz)
ax.set(xlabel=r'$H_x$', ylabel=r'$H_z$',
       xlim=(-1.12, 1.12), ylim=(-1.12, 1.12),
       aspect='equal',
       xticks=[-1, -0.5, 0, 0.5, 1])
ax.axhline(0, color='grey', linewidth=0.7)
ax.axvline(0, color='grey', linewidth=0.7)
ax.grid(alpha=0.2)
fig.subplots_adjust(left=0.16, right=0.97,
                    bottom=0.15, top=0.97)
fig.savefig(out / 'figures/astroid.png', dpi=220)
plt.close(fig)

# Коэрцитивная сила
a = np.arange(91)
c = np.array([hc(h, mag(t, h)) for t in a])
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(a, c)
ax.set(xlabel=r'Угол $\theta$, градусы',
       ylabel=r'Коэрцитивная сила $h_c$',
       xlim=(0, 90), ylim=(-0.03, 1.05),
       xticks=np.arange(0, 91, 15))
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(out / 'figures/angular_fields.png',
            dpi=220)
plt.close(fig)

np.savetxt(out / 'data/hc.csv',
           np.column_stack((a, c)), delimiter=',',
           header='theta_deg,hc', comments='')
for t in [0, 30, 45, 60, 90]:
    print(f'θ = {t}°: hc = {round(c[t], 6):.3f}')
