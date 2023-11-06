# Python program to do some QM calculations in PyTorch
# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchquad import VEGAS, set_up_backend, set_log_level
from torch import vmap
import time
from noisyopt import minimizeSPSA
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
N_Int_Points = 1000000
do_paired = False

set_up_backend("torch", data_type="float64")
# set_log_level("INFO")

# if j is not None then complex numbers are used
j = None
# j = torch.complex(torch.tensor(0, dtype=torch.float64), torch.tensor(1, dtype=torch.float64))

# start values of the parameters of the wave function
# ppp[0] = distance between nuclei
# ppp[1] = is used for the integration range of the nuclei
# ppp[2] = is used for the integration range of the electrons

ppp = np.array([1.5,  0.2,  2.5,  5,  4])
H_precision_expected = 0.005   # This is the c parameter of the SPSA optimizer, it is approximatly the standard deviation of the energy
start_step = 0.3

do_plot_every = None
plot_factor = 1.0
plot_npoints = 30
replot = False

V_strength = 1.0


# the following functions are used to calculate the integration ranges
def calc_int_electron(ppp):
    """
    ppp: tensor of parameters

    result is the + - range of the electron integration
    """
    return (ppp[0] * 1.5 + ppp[1]) * 1.2


def calc_int_nuclei(ppp):
    """
    ppp: tensor of parameters

    result is the + - range of the nuclei integration
    """
    return 1.3 * ppp[1]


dist_nuclei = ppp[0]
nNuclei = 3
nElectrons = 3
q_nuclei = 1

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
    """
    helper to be used in wave functions, cuts an range -r to +r with some parameter f for the hardness of the cut
    x: position
    r: + - range
    f: hardness of the cut
    """
    return torch.sigmoid((x+r)*f)*(1-torch.sigmoid((x-r)*f))


q = torch.tensor([-1]*nElectrons + [q_nuclei]*nNuclei)


def damp(x, w):
    """
    typical gaussian form to be used in wave functions
    x: position
    w: width parameter
    """
    return torch.exp(-(x/w)**2)
    # ret = torch.sigmoid(x/w - w) * torch.sigmoid(-x/w - w)
    # return ret / torch.sigmoid(-w)**2


def wf_form(x, w):
    """
    typical gaussian form to be used in wave functions
    x: position
    w: width parameter
    """
    return torch.exp(-(x/w)**2)
    # ret = torch.sigmoid(x/w * 2.0) * torch.sigmoid(-x/w * 2.0)
    # return ret * 4.0


# A potential of a one dimensional chain simelar to the 3D Coulumb, checked with Mathematica
# V[x_, y_, z_] := 1/Sqrt[x^2 + y^2 + z^2]
# Plot[{NIntegrate[V[x, y, z], {y, -0.5, 0.5}, {z, -0.5, 0.5}], 1/(Abs[x/1.2] + 0.28)}, {x, -3, 3}]
def V(dx):
    """
    dx: distance of two particles
    retuns the potential from a toy function, not exactly coulomb ...
    """
    # return 1.0 / (torch.abs(dx / 1.2) + 0.28)    # Coulomb potential part integrated from 3D
    return torch.exp(-dx**2)            # Easy to integrate potential


def Vpot(xinp):
    """
    xinp: positions
    returns the Potential
    """
    x = xinp + offsets
    x1 = x.reshape(-1, 1)
    x2 = x.reshape(1, - 1)
    dx = x1 - x2
    Vdx = q.reshape(-1, 1) * V(dx) * q.reshape(1, -1)
    Vdx = Vdx.triu(diagonal=1)
    return Vdx.sum() * V_strength


def Epot(wf, x):
    """
    wf: wave function takes only x
    x: position
    returns the value of the potential energy integrand
    """
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
    """
    wf: wave function takes only x
    x: position
    returns the value of the H integrand
    """
    # <see cref="file://./Docs/PartIntMulti.jpg"/>
    if j is None:
        gg = torch.func.grad(lambda x: wf(x).real)(x)
    else:
        gg = torch.complex(torch.func.grad(lambda x: wf(x).real)(x), torch.func.grad(lambda x: wf(x).imag)(x))
    v = 1/(2*m)  # from partial integration the minus sign already present
    gg = torch.sqrt(v) * gg
    return ((torch.dot(torch.conj(gg), gg) + Epot(wf, x)).real)


def H(wf, x):
    """
    wf: wave function takes only x
    x: position

    vectorized function of H_single
    returns the value of the H integrand
    """
    gg = vmap(lambda x: H_single(wf, x), chunk_size=vmap_chunk_size)(x)
    return gg


def CorrelateWF(x, ppp_part):
    """increase the spatial probability density if different charged particles are close together, decrease it if they are of the same charge"""
    # x = xinp + offsets
    x1 = x.reshape(-1, 1)
    x2 = x.reshape(1, - 1)
    mask = (q.reshape(-1, 1) * q.reshape(1, -1))
    dx = x1 - x2
    # res = torch.where(mask > 0, -ppp_part[0]*torch.exp(-torch.abs(dx / ppp_part[2])), ppp_part[1]*torch.exp(-torch.abs(dx / ppp_part[2])))
    res = torch.where(mask > 0, -ppp_part[0] * damp(dx, ppp_part[2]), ppp_part[1] * damp(dx, ppp_part[2]))
    res = res.triu(diagonal=1)
    return (torch.ones((nParticles, nParticles)) + res).prod()


# spin wave function
sf_array = torch.zeros([2] * nElectrons)
sf_array[0, 1, 0] = 1
sf_array[0, 0, 1] = -1
sf_array[1, 0, 0] = -1


def testwf(ppp, xx):
    """
    ppp: tensor of the parameters for the wave function
    xx: position
    """
    def sf(x):
        x = x[:nElectrons]
        res = sf_array[tuple(x)]
        if res == 0:
            raise Exception("unknown spin configuration", x)
        return res
    res = 0
    for i in range(perms.shape[0]):
        tt = torch.tensor([0, 1, 0, 0, 0, 0])
        x = xx[tuple(perms[i]), ]
        t = tt[tuple(perms[i]), ]
        xo = x + offsets
        res += perms_p[i] * sf(t) * (
                            # CorrelateWF(xo, ppp[[3, 4, 5]]) *
                            (1 + ppp[4] * wf_form(xo[4]-xo[0], ppp[3])) *
                            (1 + ppp[4] * wf_form(xo[5]-xo[0], ppp[3])) *
                            (1 + ppp[4] * wf_form(xo[3]-xo[1], ppp[3])) *
                            (1 + ppp[4] * wf_form(xo[5]-xo[1], ppp[3])) *
                            (1 + ppp[4] * wf_form(xo[3]-xo[2], ppp[3])) *
                            (1 + ppp[4] * wf_form(xo[4]-xo[2], ppp[3])) *
                            wf_form(xo[[0, 1, 2]] - xo[[3, 4, 5]], ppp[2]).prod()
                            )
    return res * torch.exp(-(x[nElectrons:] / ppp[1])**2).prod(-1)  # nuclei not permuted here


def Norm(wf, x):
    """
    wf: wave function, takes only a x tensor as input
    x: x tensor to calculate
    returns the spacial density
    """
    return (torch.conj(wf(x)) * wf(x)).real


# create offsets for the chain, one particle is in the middle (should be uneven)
for i in range(nNuclei):
    offsets[i+nElectrons] = ppp[0] * (i - nNuclei//2)

# create permutations of identical particles, only the electrons are permuted, the nuclei not at the moment.
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
print('perms.shape', perms.shape)


def plotwf(ppp, plot_pos, where):
    """
    plot function for wave function
    ppp: parameters
    plot_pos: which position to plot
    where: on which subplot to plot
    """
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
    """
    pinp: parameters
    plot some wave functions and potential energy
    """
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
    """
    pinp: list (or tensor) of the parameters
    seed: random seed
    N: Number of integration points
    """
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
    else:
        map_Norm = None
    integral_value = Integrator.integrate(lambda y: H(lambda x: testwf(ppp, x), y),
                                          dim=nParticles, N=N_Int_Points_loc,  integration_domain=IntElectron*nElectrons+IntNuclei*nNuclei, vegasmap=map_H, use_warmup=(map_H is None), seed=seed)
    if seed is None:
        map_H = Integrator.map
    else:
        map_H = None
    if Normvalue < 0.0001:
        print("Normvalue too small", Normvalue)
        raise Exception("Normvalue too small, probable zero wave function (e.g. due to antisymmetry)")
    retH = integral_value / Normvalue
    print("              H", "{:.5f}".format(float(retH.cpu())), ppp.cpu().numpy(), "{:.2f}".format(time.time() - start), "(" + "{:.2f}".format(plottime) + ")", 'raw Norm integral value', "{:.5f}".format(Normvalue), 'seed', seed)
    return retH.cpu().numpy()


def get_ratio(wf, x, kk):
    """
    wf: wavefunction
    xx: The variable, nParticles-1 nParticles-1 and one variable at pos kk
    kk: position of the extra variable
    """
    xx1 = torch.cat((x[:kk], x[-2:-1], x[kk:nParticles-1]))
    xx2 = torch.cat((x[:kk], x[-1:], x[kk:nParticles-1]))
    t1 = Norm(wf, xx1)
    t2 = Norm(wf, xx2)
    return t1, t2


def do_entangled_std(pinp, kk, seed=None, N=None):
    """
    calculate entanglement of one position against all others
    pinp: parameters
    kk: which position
    N: number of random points to use
    """
    ppp = torch.tensor(pinp)
    if N is not None:
        N_Int_Points_loc = N
    else:
        N_Int_Points_loc = N_Int_Points
    N_Int_Points_loc = int(np.sqrt(N_Int_Points_loc))
    IntElectron = [[-calc_int_electron(ppp), calc_int_electron(ppp)]]
    IntNuclei = [[-calc_int_nuclei(ppp), calc_int_nuclei(ppp)]]
    intdomain = IntElectron*nElectrons+IntNuclei*nNuclei
    intdomain_d = intdomain[:kk] + intdomain[kk+1:]
    stds = []
    means = []
    densities = []
    for _ in range(N_Int_Points_loc):
        rand_tensor = torch.rand(N_Int_Points_loc, len(intdomain_d)) * (torch.tensor(intdomain_d)[:, 1]-torch.tensor(intdomain_d)[:, 0]) + torch.tensor(intdomain_d)[:, 0]
        rand_same = torch.rand(1, 2) * (torch.tensor(intdomain[kk])[1] - torch.tensor(intdomain[kk])[0]) + torch.tensor(intdomain[kk])[0]
        rand_tensor = torch.cat((rand_tensor, rand_same.repeat((N_Int_Points_loc, 1))), dim=1)
        t1, t2 = vmap(lambda y: get_ratio(lambda x: testwf(ppp, x), y, kk), chunk_size=vmap_chunk_size)(rand_tensor)
        # res = torch.log(t1 / (t2+1E-10))
        # stds.append(res.std())
        # means.append(res.mean())
        # print('mean', res.mean(), 'std', res.std())
        res = torch.log(t1 / (t2+1E-10))
        dens_mean = (t1+t2).sum()
        res_mean = (res * (t1+t2)).sum() / dens_mean
        res_std = (res**2 * (t1+t2)).sum() / dens_mean - res_mean**2
        stds.append(res_std)
        means.append(res_mean)
        densities.append(dens_mean)
        # print('mean', res_mean, 'std', res_std)

    # res_mean = np.array(means).mean()
    # res = np.array(stds).mean()
    res_mean = (np.array(means)*np.array(densities)).sum() / (np.array(densities)).sum()
    res = (np.array(stds)*np.array(densities)).sum() / (np.array(densities)).sum()
    print('entanglement measure of', kk, 'full mean', res_mean, 'std', res)
    return res, res_mean


if do_paired:
    # this checks for the integration error
    # doIntegration(ppp)  # activate this to check, if integration with the same seed gives the same result
    calc_std = []
    for s in range(5):
        calc_std.append(doIntegration(ppp, seed=s))
    calc_std = float(np.array(calc_std).std())
    print("calc_std", calc_std, 'H_precision_expected', H_precision_expected, ' both should be similar in the thumb rule of Spall, IEEE, 1998, 34, 817-823')
else:
    doIntegration(ppp)

starttime = time.time()


def show_entanglement(ppp):
    for k in range(nParticles):
        do_entangled_std(ppp, k)


show_entanglement(ppp)

# This very robust optimizer is not very fast in my tests, at least two orders of magnitude slower than SPSA
# ret = minimizeCompass(doIntegration, x0=ppp, deltainit=0.6, deltatol=0.1, bounds=[[0.01, 20.0]] * (ppp.shape[0]), errorcontrol=do_errorcontrol, funcNinit=30, feps=0.003, disp=True, paired=True, alpha=0.2)
ret = minimizeSPSA(doIntegration, x0=ppp, bounds=[[0.01, 20.0]] * (ppp.shape[0]), disp=True, niter=100, c=H_precision_expected, paired=do_paired, a=start_step)  # , gamma=0.2, a=0.2)

print(ret)
print("time", time.time() - starttime)

print("some checks on the minimum")
center = doIntegration(ret.x, seed=0, N=10 * N_Int_Points)
for k in range(len(ret.x)):
    addx = np.zeros(len(ret.x))
    addx[k] = 0.5
    newx = ret.x + addx
    ny = doIntegration(newx, seed=0, N=10 * N_Int_Points)
    print(k, (ny - center))

print("integration with higher precision")
doIntegration(ret.x, seed=None, N=10 * N_Int_Points)
doIntegration(ret.x, seed=None, N=100 * N_Int_Points)

show_entanglement(ret.x)

if do_plot_every is not None:
    doplot(torch.tensor(ret.x))
    plt.show(block=True)
