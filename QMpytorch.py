# Python program to do some QM calculations in PyTorch
# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchquad import VEGAS, set_up_backend, set_log_level
from torch import vmap
import time
from noisyopt import minimizeCompass, minimizeSPSA
from itertools import permutations
from sympy.combinatorics.permutations import Permutation

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# this may tune the performance and memory usage
# smaller values on cpu can imporove cache usage
# larger values on gpu can improve parallelization
vmap_chunk_size = 100000  # 2**13
print("vmap_chunk_size", vmap_chunk_size)

N_Int_Plot = 5000
N_Int_Points = 3000000

set_up_backend("torch", data_type="float64")
# set_log_level("INFO")

# if j is not None then complex numbers are used
j = None
# j = torch.complex(torch.tensor(0, dtype=torch.float64), torch.tensor(1, dtype=torch.float64))

# start values of the parameters of the wave function
# ppp[0] = distance between nuclei
# ppp[1] = is used for the integration range of the nuclei
# ppp[2] = is used for the integration range of the electrons
# ppp = np.array([3.25, 0.342, 5.695, 11.8, 0.5, 0.5, 1.0])

ppp = np.array([3.0, 0.5, 6, 12, 0.5, 0.5, 1.0])
H_precision_expected = 0.05  # This is the c parameter of the SPSA optimizer, it is approximatly the standard deviation of the energy

do_plot_every = None
plot_factor = 1.0
plot_npoints = 30
replot = False

V_strength = 1.0

# the following functions are used to calculate the integration ranges
def calc_int_electron(ppp):
    return (ppp[2] + 0.5) * 1.1


def calc_int_nuclei(ppp):
    return 1.3 * ppp[1]


dist_nuclei = ppp[0]
nNuclei = 3
nElectrons = 3

m_Nuclei = 1836
m_Electron = 1

Integrator = VEGAS()
Integrator_plot = VEGAS()

nParticles = nElectrons + nNuclei
m = torch.tensor([m_Electron]*nElectrons + [m_Nuclei]*nNuclei)

offsets = torch.zeros(nParticles)  # at 0 there must be high spatial probability density for VEGAS integration to work
for i in range(nNuclei):
    offsets[i+nElectrons] = dist_nuclei * (i - nNuclei//2)


def CutRange(x, r, f):
    return torch.sigmoid((x+r)*f)*(1-torch.sigmoid((x-r)*f))


q = torch.tensor([-1]*nElectrons + [1]*nNuclei)


def V(dx):
    return torch.exp(-dx**2)


def Vpot(xinp):
    """Potential energy"""
    x = xinp + offsets
    x1 = x.reshape(-1, 1)
    x2 = x.reshape(1, - 1)
    dx = x1 - x2
    Vdx = q.reshape(-1, 1) * V(dx) * q.reshape(1, -1)
    Vdx = Vdx.triu(diagonal=1)
    return Vdx.sum() * V_strength


def Epot(wf, x):
    return (torch.conj(wf(x)) * Vpot(x) * wf(x)).real


def Vpot_plot(xinp, plot_pos):
    """Potential energy for plotting only"""
    x = xinp + offsets
    x1 = x.reshape(-1, 1)
    x2 = x.reshape(1, - 1)
    dx = x1 - x2
    Vdx = q.reshape(-1, 1) * V(dx) * q.reshape(1, -1)
    Vdx *= 1-torch.eye(nParticles)
    return Vdx[plot_pos].sum()


def Epot_plot(wf, x, plot_pos):
    """Energy for plotting"""
    return (torch.conj(wf(x)) * Vpot_plot(x, plot_pos) * wf(x)).real


def H_single(wf, x):
    # <see cref="file://./Docs/PartIntMulti.jpg"/>
    if j is None:
        gg = torch.func.grad(lambda x: wf(x).real)(x)
    else:
        gg = torch.complex(torch.func.grad(lambda x: wf(x).real)(x), torch.func.grad(lambda x: wf(x).imag)(x))
    v = 1/(2*m)  # from partial integration the minus sign already present
    gg = torch.sqrt(v) * gg
    return ((torch.dot(torch.conj(gg), gg) + Epot(wf, x)).real)


def H(wf, x):
    gg = vmap(lambda x: H_single(wf, x), chunk_size=vmap_chunk_size)(x)
    return gg


def CorrelateWF(xinp, ppp_part):
    """increase the spatial probability density if different charged particles are close together, decrease it if they are of the same charge"""
    x = xinp + offsets
    x1 = x.reshape(-1, 1)
    x2 = x.reshape(1, - 1)
    mask = (q.reshape(-1, 1) * q.reshape(1, -1))
    dx = x1 - x2
    res = torch.where(mask > 0, -ppp_part[0]*torch.exp(-ppp_part[2]*dx**2), ppp_part[1]*torch.exp(-ppp_part[2]*dx**2))
    res = res.triu(diagonal=1)
    return (torch.ones((nParticles, nParticles)) + res).prod()


def testwf(ppp, xx):
    res = 0
    for i in range(perms.shape[0]):
        x = xx[tuple(perms[i]), ]
        res += perms_p[i] * torch.exp(-(x[nElectrons:] / ppp[1])**2).prod(-1) * CutRange(x[:nElectrons], ppp[2], 5).prod(-1) * (
                                                                                                                                torch.cos(x[0] * torch.pi / ppp[3]) *
                                                                                                                                torch.sin(x[1] * torch.pi / ppp[3] * 2) *
                                                                                                                                torch.cos(x[2] * torch.pi / ppp[3] * 3) *
                                                                                                                                CorrelateWF(x, ppp[[4, 5, 6]])
                                                                                                                                )
    return res


def Norm(wf, x):
    return (torch.conj(wf(x)) * wf(x)).real

# create offsets for the chain, one particle is in the middle (should be uneven)
for i in range(nNuclei):
    offsets[i+nElectrons] = ppp[0] * (i - nNuclei//2)

# create permutations
perms = []
perms_p = []
for i in permutations(list(range(nElectrons))):
    a = Permutation(list(i))
    p = a.parity()
    if p == 0:
        p = -1
    # print(i, p)
    perms.append(list(i) + list(range(nElectrons, nParticles)))
    perms_p.append(p)

perms = np.array(perms)
# print(perms)
# print(perms.shape)


def plotwf(ppp, plot_pos, where):
    global offsets
    for i in range(nNuclei):
        offsets[i+nElectrons] = ppp[0] * (i - nNuclei//2)
    IntElectron = [[-calc_int_electron(ppp), calc_int_electron(ppp)]]
    IntNuclei = [[-calc_int_nuclei(ppp), calc_int_nuclei(ppp)]]
    if plot_pos < nElectrons:
        pl_x = np.linspace(-calc_int_electron(ppp.cpu()) * plot_factor, calc_int_electron(ppp.cpu()) * plot_factor, plot_npoints)
    else:
        pl_x = np.linspace(-calc_int_nuclei(ppp.cpu()) * plot_factor, calc_int_nuclei(ppp.cpu()) * plot_factor, plot_npoints)
    pl_y = []
    pl_y2 = []

    for x in pl_x:
        def wf(x):
            return testwf(ppp, x)
        xinp = [0]*plot_pos + [x] + [0]*(nParticles-1-plot_pos)
        xinp = torch.from_numpy(np.array(xinp))
        if plot_pos < nElectrons:
            int_domain = [IntElectron[0]]*plot_pos + [[x, x+0.01]] + [IntElectron[0]]*(nElectrons-1-plot_pos) + IntNuclei*nNuclei
        else:
            int_domain = [IntElectron[0]]*nElectrons + IntNuclei*(plot_pos-nElectrons) + [[x, x+0.01]] + IntNuclei*(nParticles - 1 - plot_pos)
        set_log_level('ERROR')
        integral_value_epot = Integrator_plot.integrate(lambda y: vmap(lambda y: Epot_plot(lambda x: testwf(ppp, x), y, plot_pos), chunk_size=vmap_chunk_size)(y), dim=nParticles, N=N_Int_Plot,  integration_domain=int_domain)
        integral_value = Integrator_plot.integrate(lambda y: vmap(lambda y: Norm(lambda x: testwf(ppp, x), y), chunk_size=vmap_chunk_size)(y), dim=nParticles, N=N_Int_Plot,  integration_domain=int_domain)
        set_log_level('WARNING')
        pl_y.append(integral_value.cpu())
        pl_y2.append(integral_value_epot.cpu())  # / (integral_value + 0.000001).cpu())
    pl_y = np.array(pl_y)
    pl_y2 = np.array(pl_y2)
    pl_y = pl_y / abs(pl_y).max() * 100
    pl_y2 = pl_y2 / abs(pl_y2).max() * 100
    where.plot(pl_x + offsets[plot_pos].cpu().numpy(), pl_y, pl_x + offsets[plot_pos].cpu().numpy(), pl_y2)
    # where.show()


if replot:
    fig, axs = plt.subplots(nrows=4, sharex=True)


def doplot(pinp):
    global fig, axs
    ppp = torch.tensor(pinp)
    if not replot:
        fig, axs = plt.subplots(nrows=4, sharex=True)
    fig.suptitle('Parameters ' + str(ppp.cpu().numpy()) + "V_strength=" + str(V_strength))
    axs[0].clear()
    axs[1].clear()
    axs[2].clear()
    # axs[3].clear()
    plotwf(ppp, plot_pos=1, where=axs[0])
    plotwf(ppp, plot_pos=3, where=axs[1])
    plotwf(ppp, plot_pos=4, where=axs[2])
    # plotwf(ppp, plot_pos=5, where=axs[3])
    if replot:
        plt.draw()
        plt.pause(0.1)
    else:
        plt.show()


if do_plot_every is not None:
    doplot(ppp)
    if replot:
        plt.show(block=False)

plotcounter = 0

map_Norm = None
map_H = None


def doIntegration(pinp, seed=None, N=None):
    global plotcounter, map_Norm, map_H
    if N is not None:
        N_Int_Points_loc = N
    else:
        N_Int_Points_loc = N_Int_Points
    start = time.time()
    global offsets
    ppp = torch.tensor(pinp)
    for i in range(nNuclei):
        offsets[i+nElectrons] = ppp[0] * (i - nNuclei//2)
    if do_plot_every is not None:
        plt.pause(0.1)
        plotcounter += 1
        if plotcounter >= do_plot_every:
            plotcounter = 0
            doplot(pinp)
    plottime = time.time() - start
    IntElectron = [[-calc_int_electron(ppp), calc_int_electron(ppp)]]
    IntNuclei = [[-calc_int_nuclei(ppp), calc_int_nuclei(ppp)]]
    Normvalue = integral_value = Integrator.integrate(lambda y: vmap(lambda y: Norm(lambda x: testwf(ppp, x), y), chunk_size=vmap_chunk_size)(y),
                                                      dim=nParticles, N=N_Int_Points_loc,  integration_domain=IntElectron*nElectrons+IntNuclei*nNuclei, vegasmap=map_Norm, use_warmup=(map_Norm is None), seed=seed)
    if seed is None:
        map_Norm = Integrator.map
    integral_value = Integrator.integrate(lambda y: H(lambda x: testwf(ppp, x), y),
                                          dim=nParticles, N=N_Int_Points_loc,  integration_domain=IntElectron*nElectrons+IntNuclei*nNuclei, vegasmap=map_H, use_warmup=(map_H is None), seed=seed)
    if seed is None:
        map_H = Integrator.map
    retH = integral_value/Normvalue
    print("              H", "{:.5f}".format(float(retH.cpu())), ppp.cpu().numpy(), "{:.2f}".format(time.time() - start), "(" + "{:.2f}".format(plottime) + ")", seed)
    return retH.cpu().numpy()


doIntegration(ppp, seed=0)
doIntegration(ppp, seed=0)
calc_std = []
for s in range(5):
    calc_std.append(doIntegration(ppp, seed=s))
calc_std = float(np.array(calc_std).std())
print("calc_std", calc_std, 'H_precision_expected', H_precision_expected, ' both should be similar in the thumb rule of Spall, IEEE, 1998, 34, 817-823')

starttime = time.time()

# This very robust optimizer is not very fast in my tests, at least two orders of magnitude slower than SPSA
# ret = minimizeCompass(doIntegration, x0=ppp, deltainit=0.6, deltatol=0.1, bounds=[[0.01, 20.0]] * (ppp.shape[0]), errorcontrol=do_errorcontrol, funcNinit=30, feps=0.003, disp=True, paired=True, alpha=0.2)
ret = minimizeSPSA(doIntegration, x0=ppp, bounds=[[0.01, 20.0]] * (ppp.shape[0]), disp=True, niter=100, c=H_precision_expected)  # , gamma=0.2, a=0.2)

print(ret)
print("time", time.time() - starttime)

doIntegration(ret.x, seed=None, N=10 * N_Int_Points)
doIntegration(ret.x, seed=None, N=100 * N_Int_Points)

if do_plot_every is not None:
    doplot(torch.tensor(ret.x))
    plt.show(block=True)
