# Python program to do some QM calculations in PyTorch
__docformat__ = "google"
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

vmap_chunk_size = 100000
"""
this may tune the performance and memory usage
smaller values on cpu can imporove cache usage
larger values on gpu can improve parallelization
"""
print("vmap_chunk_size", vmap_chunk_size)

N_Int_Plot = 5000
"""number of integration points for plotting"""
N_Int_Points = 10000000
"""number of integration points for calculation"""
do_paired = True
"""use same seed for pairs of integrations for the optimization of the parameters"""

set_up_backend("torch", data_type="float64")
# set_log_level("INFO")

j = None
# j = torch.complex(torch.tensor(0, dtype=torch.float64), torch.tensor(1, dtype=torch.float64))
"""if j is not None but complex j, then complex numbers are used"""

doexcited = False
"""if True, then the excited state is calculated, if False, then the ground state"""
doSchmidtGraham = False
"""if True, then the Schmidt Graham orthogonalization is done"""
H_precision_expected = 0.05
"""This is the c parameter of the SPSA optimizer, it is approximatly the standard deviation of the energy"""
start_step = 0.1
"""This is the a parameter of the SPSA optimizer"""

do_plot_every = None
"""plot during calculation after do_plot_every integrations """

check_pair_entanglement = False
"""check entanglement of pairs of particles"""


class wfunc:
    """This class defines the wave function and the parameters"""
    ppp_ground = np.array([1.2445, 0.1535, 1.5])
    """parameters for the ground state"""
    ppp_excited = np.array([1.6617, 0.3013, 2.0675])  # , 5.5342])
    """parameters for the excited state"""

    # start values of the parameters of the wave function
    # ppp[0] = distance between nuclei
    # ppp[1] = is used for the integration range of the nuclei
    # ppp[2] = is used for the integration range of the electrons

    nNuclei = 4
    """number of nuclei"""
    nElectrons = 4
    """number of electrons"""

    nParticles = nElectrons + nNuclei
    """total number of particles"""
    def set_ppp():
        """set the parameters for the wave function"""
        if doexcited:
            wfunc.ppp = torch.tensor(wfunc.ppp_excited)
        else:
            wfunc.ppp = torch.tensor(wfunc.ppp_ground)
    
    spin_state = torch.tensor([0, 1, 1, 0, 0, 0, 0, 0])
    """corresponds to the wave function"""
    # spin wave function
    sf_array = torch.zeros([2] * nElectrons)
    """spin wave function, implemented as array"""
    sf_array[0, 0, 1, 1] = 1
    sf_array[0, 1, 0, 1] = -1
    sf_array[0, 1, 1, 0] = 1
    sf_array[1, 0, 0, 1] = 1
    sf_array[1, 0, 1, 0] = -1
    sf_array[1, 1, 0, 0] = 1

    offsets = torch.zeros(nParticles)
    """at 0 there must be high spatial probability density for VEGAS integration to work
    therefore offsets are used to shift"""
    def setoffset(ppp):
        """create offset from parameters

        Args:
            ppp (tensor): parameters of the wave function
        """
        for i in range(wfunc.nNuclei):
            """now also even number of nuclei should work"""
            wfunc.offsets[i+wfunc.nElectrons] = wfunc.calc_dist_nuclei(ppp) * (2*i + 1 - wfunc.nNuclei)/2

    def xo_from_x(x):
        """calculate the x with offset from the x without

        Args:
            x (tensor): x without offset

        Returns:
            tensor: x with offsets
        """
        return x + wfunc.offsets

    # the following functions are used to calculate the integration ranges
    def calc_int_electron(ppp):
        """
        Args:
            ppp: tensor of parameters
        Returns:
            result is the + - range of the electron integration
        """
        return (ppp[0] * wfunc.nNuclei / 2 + ppp[2]) * 1.2

    def calc_int_nuclei(ppp):
        """
        Args:
            ppp: tensor of parameters
        Returns:
            result is the + - range of the nuclei integration
        """
        return 2.0 * ppp[1]

    def calc_dist_nuclei(ppp):
        """get the nuclei distance from parameters for the offsets

        Args:
            ppp (tensor): parameters
        """
        return ppp[0]

    def wf_form(x, w):
        """typical gaussian form to be used in wave functions
        Args:
            x: position
            w: width parameter
        """
        # return torch.exp(-(x/w)**2)
        ret = torch.sigmoid(x/w * 2.0) * torch.sigmoid(-x/w * 2.0)
        return ret * 4.0

    def ground(ppp, xx):
        """This defines the wave function

        Args:
            ppp (tensor): tensor of the parameters for the wave function
            xx (tensor): position
        Returns:
            float: value of the wave function
        """
        def sf(x):
            x = x[:wfunc.nElectrons]
            res = wfunc.sf_array[tuple(x)]
            if res == 0:
                raise Exception("unknown spin configuration", x)
            return res
        res = 0
        for i in range(perms.shape[0]):
            tt = wfunc.spin_state  # torch.tensor([0, 1, 0, 0, 0, 0])
            x = xx[tuple(perms[i]), ]
            t = tt[tuple(perms[i]), ]
            xo = wfunc.xo_from_x(x)
            res += perms_p[i] * sf(t) * (
                                # CorrelateWF(xo, ppp[[3, 4, 5]]) *
                                # (1 + ppp[4] * wf_form(xo[4]-xo[0], ppp[3])) *
                                # (1 + ppp[4] * wf_form(xo[5]-xo[0], ppp[3])) *
                                # (1 + ppp[4] * wf_form(xo[3]-xo[1], ppp[3])) *
                                # (1 + ppp[4] * wf_form(xo[5]-xo[1], ppp[3])) *
                                # (1 + ppp[4] * wf_form(xo[3]-xo[2], ppp[3])) *
                                # (1 + ppp[4] * wf_form(xo[4]-xo[2], ppp[3])) *
                                wfunc.wf_form(xo[:wfunc.nElectrons] - xo[wfunc.nElectrons:], ppp[2]).prod()
                                )
        return res * torch.exp(-(x[4] / ppp[1])**2) * torch.exp(-((x[5] - x[4]) / ppp[1])**2) * torch.exp(-((x[3] - x[4]) / ppp[1])**2)
        return res * torch.exp(-(x[wfunc.nElectrons:] / ppp[1])**2).prod(-1)  # nuclei not permuted here

    def excited(ppp, xx):
        """This defines the wave function of the excited state

        This will be orthogonalized against the ground state during the calculation

        Args:
            ppp (tensor): tensor of the parameters for the wave function
            xx (tensor): position
        Returns:
            float: value of the wave function
        """
        def sf(x):
            x = x[:wfunc.nElectrons]
            res = wfunc.sf_array[tuple(x)]
            if res == 0:
                raise Exception("unknown spin configuration", x)
            return res
        res = 0
        for i in range(perms.shape[0]):
            tt = wfunc.spin_state  # torch.tensor([0, 1, 0, 0, 0, 0])
            x = xx[tuple(perms[i]), ]
            t = tt[tuple(perms[i]), ]
            xo = wfunc.xo_from_x(x)
            res += perms_p[i] * sf(t) * (
                                # CorrelateWF(xo, ppp[[3, 4, 5]]) *
                                # (1 + ppp[4] * wf_form(xo[4]-xo[0], ppp[3])) *
                                # (1 + ppp[4] * wf_form(xo[5]-xo[0], ppp[3])) *
                                # (1 + ppp[4] * wf_form(xo[3]-xo[1], ppp[3])) *
                                # (1 + ppp[4] * wf_form(xo[5]-xo[1], ppp[3])) *
                                # (1 + ppp[4] * wf_form(xo[3]-xo[2], ppp[3])) *
                                # (1 + ppp[4] * wf_form(xo[4]-xo[2], ppp[3])) *
                                # torch.sin((xo[1]-xo[4])*2*3.1415927 / 2 / ppp[3]) *
                                (xo[1]-xo[4]) *
                                wfunc.wf_form(xo[[0, 1, 2]] - xo[[3, 4, 5]], ppp[2]).prod()
                                )
        return res * torch.exp(-(x[4] / ppp[1])**2) * torch.exp(-((x[5] - x[4]) / ppp[1])**2) * torch.exp(-((x[3] - x[4]) / ppp[1])**2)
        return res * torch.exp(-(x[wfunc.nElectrons:] / ppp[1])**2).prod(-1)  # nuclei not permuted here

    factor_for_Schmidt_Graham = None  # if we need orthogonalization this is different from 0

    def used(ppp, xx):
        """This defines the wave function used from defined wave functions to allow Schmidt Graham orthogonalization

        Args:
            ppp (tensor): parameters of the wave function
            xx (tensor): the spatial positions

        Returns:
            float: value of the wave function at xx
        """
        if doexcited:
            if doSchmidtGraham:
                return wfunc.excited(ppp, xx) - factor_for_Schmidt_Graham * wfunc.ground(wfunc.ppp_ground, xx)  # only if not orthogonal
            else:
                return wfunc.excited(ppp, xx)
        else:
            return wfunc.ground(ppp, xx)


class Hamiltonian:
    q_nuclei = 1
    """charge of the nuclei"""
    m_Nuclei = 1836
    """mass of the nuclei"""
    m_Electron = 1
    """mass of the electrons"""
    m = torch.tensor([m_Electron]*wfunc.nElectrons + [m_Nuclei]*wfunc.nNuclei)
    """tensor of the masses of all particles, taken from previous definitions"""
    q = torch.tensor([-1]*wfunc.nElectrons + [q_nuclei]*wfunc.nNuclei)
    """the charges of all particles"""
    V_strength = 1.0
    """Factor for the potential to quickly change strength"""

    # A potential of a one dimensional chain simelar to the 3D Coulumb, checked with Mathematica
    # V[x_, y_, z_] := 1/Sqrt[x^2 + y^2 + z^2]
    # Plot[{NIntegrate[V[x, y, z], {y, -0.5, 0.5}, {z, -0.5, 0.5}], 1/(Abs[x/1.2] + 0.28)}, {x, -3, 3}]
    def V(dx):
        """
        Args:
            dx: distance of two particles
        Returns:
            the potential from a toy function, not exactly coulomb ...
        """
        # return 1.0 / (torch.abs(dx / 1.2) + 0.28)    # Coulomb potential part integrated from 3D
        return torch.exp(-dx**2)            # Easy to integrate potential

    def Vpot(xinp):
        """
        Args:
            xinp: positions
        Returns:
            the Potential
        """
        x = wfunc.xo_from_x(xinp)
        x1 = x.reshape(-1, 1)
        x2 = x.reshape(1, - 1)
        dx = x1 - x2
        Vdx = Hamiltonian.q.reshape(-1, 1) * Hamiltonian.V(dx) * Hamiltonian.q.reshape(1, -1)
        Vdx = Vdx.triu(diagonal=1)
        return Vdx.sum() * Hamiltonian.V_strength

    def Epot(wf, x):
        """
        Args:
            wf: wave function takes only x
            x: position
        Returns:
            returns the value of the potential energy integrand
        """
        return (torch.conj(wf(x)) * Hamiltonian.Vpot(x) * wf(x)).real

    def Vpot_plot(xinp, plot_pos):
        """Potential energy for plotting only"""
        x = wfunc.xo_from_x(xinp)
        x1 = x.reshape(-1, 1)
        x2 = x.reshape(1, - 1)
        dx = x1 - x2
        Vdx = Hamiltonian.q.reshape(-1, 1) * Hamiltonian.V(dx) * Hamiltonian.q.reshape(1, -1)
        Vdx *= 1-torch.eye(wfunc.nParticles)
        return Vdx[plot_pos].sum()

    def Epot_plot(wf, x, plot_pos):
        """Potential Energy of a particle for plotting"""
        xx = torch.cat((x[:plot_pos], torch.tensor([0]), x[plot_pos+1:]))  # plotting the potential energy of one particle not taking into account the own spatial probability density
        return (torch.conj(wf(xx)) * Hamiltonian.Vpot_plot(x, plot_pos) * wf(xx)).real

    def H_single(wf, x):
        """calculates the value of the H integrand

        Args:
            wf (function): wave function takes only x
            x (tensor): position

        Returns:
            float: the value of the H integrand
        """
        # <see cref="file://./Docs/PartIntMulti.jpg"/>
        if j is None:
            gg = torch.func.grad(lambda x: wf(x).real)(x)
        else:
            gg = torch.complex(torch.func.grad(lambda x: wf(x).real)(x), torch.func.grad(lambda x: wf(x).imag)(x))
        v = 1/(2*Hamiltonian.m)  # from partial integration the minus sign already present
        gg = torch.sqrt(v) * gg
        return ((torch.dot(torch.conj(gg), gg) + Hamiltonian.Epot(wf, x)).real)

    def H(wf, x):
        """vectorized function of H_single"""
        gg = vmap(lambda x: Hamiltonian.H_single(wf, x), chunk_size=vmap_chunk_size)(x)
        return gg


class plot:
    plot_factor = 1.0
    """scaling of the plot"""
    plot_npoints = 60
    """number of x points for the plots"""
    replot = False
    """reuse open pyplot window"""

    def plotwf(ppp, plot_pos, where):
        """plot function for wave function

        Args:
            ppp (tensor): parameters
            plot_pos (int): which position to plot
            where (int): on which subplot to plot

        """
        wfunc.setoffset(ppp)
        IntElectron = [[-wfunc.calc_int_electron(ppp), wfunc.calc_int_electron(ppp)]]
        IntNuclei = [[-wfunc.calc_int_nuclei(ppp), wfunc.calc_int_nuclei(ppp)]]
        if plot_pos < wfunc.nElectrons:
            pl_x = np.linspace(-wfunc.calc_int_electron(ppp.cpu()) * plot.plot_factor, wfunc.calc_int_electron(ppp.cpu()) * plot.plot_factor, plot.plot_npoints)
        else:
            pl_x = np.linspace(-wfunc.calc_int_nuclei(ppp.cpu()) * plot.plot_factor, wfunc.calc_int_nuclei(ppp.cpu()) * plot.plot_factor, plot.plot_npoints)
        pl_y = []
        pl_y2 = []

        for x in pl_x:
            def wf(x):
                return wfunc.wf_used(ppp, x)
            xinp = [0]*plot_pos + [x] + [0]*(wfunc.nParticles-1-plot_pos)
            xinp = torch.from_numpy(np.array(xinp))
            if plot_pos < wfunc.nElectrons:
                int_domain = [IntElectron[0]]*plot_pos + [[x, x+0.01]] + [IntElectron[0]]*(wfunc.nElectrons-1-plot_pos) + IntNuclei*wfunc.nNuclei
            else:
                int_domain = [IntElectron[0]]*wfunc.nElectrons + IntNuclei*(plot_pos-wfunc.nElectrons) + [[x, x+0.01]] + IntNuclei*(wfunc.nParticles - 1 - plot_pos)
            set_log_level('ERROR')
            Integrator_plot = VEGAS()
            integral_value_epot = Integrator_plot.integrate(lambda y: vmap(lambda y: Hamiltonian.Epot_plot(lambda x: wfunc.used(ppp, x), y, plot_pos), chunk_size=vmap_chunk_size)(y), dim=wfunc.nParticles, N=N_Int_Plot,  integration_domain=int_domain)
            Integrator_plot = VEGAS()
            integral_value = Integrator_plot.integrate(lambda y: vmap(lambda y: Integration.Norm(lambda x: wfunc.used(ppp, x), y), chunk_size=vmap_chunk_size)(y), dim=wfunc.nParticles, N=N_Int_Plot,  integration_domain=int_domain)
            set_log_level('WARNING')
            pl_y.append(integral_value.cpu())
            pl_y2.append(integral_value_epot.cpu())  # / (integral_value + 0.000001).cpu())
        pl_y = np.array(pl_y)
        pl_y2 = np.array(pl_y2)
        pl_y = pl_y / abs(pl_y).max() * 100
        pl_y2 = pl_y2 / abs(pl_y2).max() * 100
        where.plot(pl_x + wfunc.offsets[plot_pos].cpu().numpy(), pl_y, pl_x + wfunc.offsets[plot_pos].cpu().numpy(), pl_y2)
        # where.show()

    if replot:
        fig, axs = plt.subplots(nrows=4, sharex=True)

    def doplot(pinp):
        """ plot some wave functions and potential energy

        Args:
            pinp (tensor or list): parameters
        """
        global fig, axs
        ppp = torch.tensor(pinp)
        if not plot.replot:
            fig, axs = plt.subplots(nrows=4, sharex=True)
        fig.suptitle('Parameters ' + str(ppp.cpu().numpy()) + "V_strength=" + str(Hamiltonian.V_strength))
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        # axs[3].clear()
        plot.plotwf(ppp, plot_pos=1, where=axs[0])
        plot.plotwf(ppp, plot_pos=3, where=axs[1])
        plot.plotwf(ppp, plot_pos=4, where=axs[2])
        # plotwf(ppp, plot_pos=5, where=axs[3])
        if plot.replot:
            plt.draw()
            plt.pause(0.1)
        else:
            plt.show()


class Integration:
    def Overlap(wf1, wf2, x):
        """Calculates the spatial density of the wave function
        Args:
            wf (function): wave function, takes only a x tensor as input
            x (tensor): x tensor to calculate

        Returns:
            float: the spacial density
        """
        return (torch.conj(wf1(x)) * wf2(x)).real

    def Norm(wf, x):
        """Calculates the spatial density of the wave function
        Args:
            wf (function): wave function, takes only a x tensor as input
            x (tensor): x tensor to calculate

        Returns:
            float: the spacial density
        """
        return (torch.conj(wf(x)) * wf(x)).real

    def doIntegration(pinp, seed=None, N=None):
        """Does the multi dimensional integration to calculate the energy of the Schrödinger equation
        Args:
            pinp (tensor or list): wave function parameters
            seed (int, optional): random seed for integration. Defaults to None.
            N (int, optional): Number of integration points. Defaults to None, taking a global variable.

        Returns:
            float: value of the integral
        """
        global plotcounter, map_Norm, map_H
        if N is not None:
            N_Int_Points_loc = N
        else:
            N_Int_Points_loc = N_Int_Points
        start = time.time()
        ppp = torch.tensor(pinp)
        wfunc.setoffset(ppp)
        # print('offsets', wfunc.offsets.cpu().numpy())
        if do_plot_every is not None:
            plt.pause(0.1)
            plotcounter += 1
            if plotcounter >= do_plot_every:
                plotcounter = 0
                plot.doplot(pinp)
        plottime = time.time() - start
        IntElectron = [[-wfunc.calc_int_electron(ppp), wfunc.calc_int_electron(ppp)]]
        IntNuclei = [[-wfunc.calc_int_nuclei(ppp), wfunc.calc_int_nuclei(ppp)]]
        Normvalue = integral_value = Integrator.integrate(lambda y: vmap(lambda y: Integration.Norm(lambda x: wfunc.used(ppp, x), y), chunk_size=vmap_chunk_size)(y),
                                                          dim=wfunc.nParticles, N=N_Int_Points_loc,  integration_domain=IntElectron*wfunc.nElectrons+IntNuclei*wfunc.nNuclei, seed=seed, vegasmap=map_Norm, use_warmup=(map_Norm is None))
        if seed is None:
            map_Norm = Integrator.map
        else:
            map_Norm = None
        integral_value = Integrator.integrate(lambda y: Hamiltonian.H(lambda x: wfunc.used(ppp, x), y),
                                              dim=wfunc.nParticles, N=N_Int_Points_loc,  integration_domain=IntElectron*wfunc.nElectrons+IntNuclei*wfunc.nNuclei, seed=seed, vegasmap=map_H, use_warmup=(map_H is None))
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

    def doNormOverlap(pinp, seed=None, N=None):
        """Does the multi dimensional integration to calculate calculate the Schmidt Graham orthogonalization of the wave function
        Args:
            pinp (tensor or list): wave function parameters
            seed (int, optional): random seed for integration. Defaults to None.
            N (int, optional): Number of integration points. Defaults to None, taking a global variable.

        Returns:
            float: value of the integral
        """
        global plotcounter, map_Norm, map_H
        if N is not None:
            N_Int_Points_loc = N
        else:
            N_Int_Points_loc = N_Int_Points
        start = time.time()
        ppp = torch.tensor(pinp)
        wfunc.setoffset(ppp)
        if do_plot_every is not None:
            plt.pause(0.1)
            plotcounter += 1
            if plotcounter >= do_plot_every:
                plotcounter = 0
                plot.doplot(pinp)
        plottime = time.time() - start
        IntElectron = [[-wfunc.calc_int_electron(ppp), wfunc.calc_int_electron(ppp)]]
        IntNuclei = [[-wfunc.calc_int_nuclei(ppp), wfunc.calc_int_nuclei(ppp)]]
        Integrator = VEGAS()
        Normvalue = Integrator.integrate(lambda y: vmap(lambda y: Integration.Norm(lambda x: wfunc.ground(wfunc.ppp_ground, x), y), chunk_size=vmap_chunk_size)(y),
                                         dim=wfunc.nParticles, N=N_Int_Points_loc,  integration_domain=IntElectron*wfunc.nElectrons+IntNuclei*wfunc.nNuclei, seed=seed, vegasmap=map_Norm, use_warmup=(map_Norm is None))
        if seed is None:
            map_Norm = Integrator.map
        else:
            map_Norm = None
        Integrator = VEGAS()
        Overlapvalue = Integrator.integrate(lambda y: vmap(lambda y: Integration.Overlap(lambda x: wfunc.ground(wfunc.ppp_ground, x), lambda x: wfunc.excited(ppp, x), y), chunk_size=vmap_chunk_size)(y),
                                            dim=wfunc.nParticles, N=N_Int_Points_loc,  integration_domain=IntElectron*wfunc.nElectrons+IntNuclei*wfunc.nNuclei, seed=seed, vegasmap=map_Norm, use_warmup=(map_Norm is None))
        if seed is None:
            map_H = Integrator.map
        else:
            map_H = None
        if Normvalue < 0.0001:
            print("Normvalue too small", Normvalue)
            raise Exception("Normvalue too small, probable zero wave function (e.g. due to antisymmetry)")
        retH = Overlapvalue / Normvalue
        print("              Overlap", "{:.5f}".format(float(retH.cpu())), ppp.cpu().numpy(), "{:.2f}".format(time.time() - start), "(" + "{:.2f}".format(plottime) + ")", 'raw Norm integral value',
              "{:.5f}".format(Normvalue), "{:.5f}".format(Overlapvalue), 'seed', seed)
        return retH.cpu().numpy()

    def get_ratio(wf, x, kk):
        """ratio of norm with different values at one position

        Args:
            wf: wavefunction
            x: The variable, (nParticles-1) and one variable at pos kk from the last two values in the input
                The input has therefore nParticles+1 values
            kk: position of the extra variable

        Returns:
            tuple of float: norm of the two cases
        """
        xx1 = torch.cat((x[:kk], x[-2:-1], x[kk:wfunc.nParticles-1]))
        xx2 = torch.cat((x[:kk], x[-1:], x[kk:wfunc.nParticles-1]))
        t1 = Integration.Norm(wf, xx1)
        t2 = Integration.Norm(wf, xx2)
        return t1, t2

    def do_entangled_std(pinp, kk, seed=None, N=None):
        """calculate entanglement of one position against all others

        Args:
            pinp (tensor or list): parameters
            kk (int): which position
            seed (int, optional): seed for the random points. Defaults to None.
            N (int, optional): number of random points to use. Defaults to None and takes a global variable.

        Returns:
            tuple of float: standard deviation and mean of the entanglement measure
        """
        ppp = torch.tensor(pinp)
        if N is not None:
            N_Int_Points_loc = N
        else:
            N_Int_Points_loc = N_Int_Points
        N_Int_Points_loc = int(np.sqrt(N_Int_Points_loc))
        IntElectron = [[-wfunc.calc_int_electron(ppp), wfunc.calc_int_electron(ppp)]]
        IntNuclei = [[-wfunc.calc_int_nuclei(ppp), wfunc.calc_int_nuclei(ppp)]]
        intdomain = IntElectron*wfunc.nElectrons+IntNuclei*wfunc.nNuclei
        intdomain_d = intdomain[:kk] + intdomain[kk+1:]
        stds = []
        means = []
        densities = []
        for _ in range(N_Int_Points_loc):
            rand_tensor = torch.rand(N_Int_Points_loc, len(intdomain_d)) * (torch.tensor(intdomain_d)[:, 1]-torch.tensor(intdomain_d)[:, 0]) + torch.tensor(intdomain_d)[:, 0]
            rand_same = torch.rand(1, 2) * (torch.tensor(intdomain[kk])[1] - torch.tensor(intdomain[kk])[0]) + torch.tensor(intdomain[kk])[0]
            rand_tensor = torch.cat((rand_tensor, rand_same.repeat((N_Int_Points_loc, 1))), dim=1)
            t1, t2 = vmap(lambda y: Integration.get_ratio(lambda x: wfunc.used(ppp, x), y, kk), chunk_size=vmap_chunk_size)(rand_tensor)
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

    def get_ratio_pair(wf, x, kk1, kk2):
        """ratio of norm with different values at one position

        Args:
            wf: wavefunction
            x: The variable, (nParticles-1) and one variable at pos kk from the last two values in the input
                The input has therefore nParticles+1 values
            kk: position of the extra variable

        Returns:
            tuple of float: norm of the two cases
        """
        if kk2 > kk1:
            xx1 = torch.cat((x[:kk1], x[-3:-2], x[kk1:kk2-1], x[-2:-1], x[kk2-1:wfunc.nParticles-2]))
            xx2 = torch.cat((x[:kk1], x[-3:-2], x[kk1:kk2-1], x[-1:], x[kk2-1:wfunc.nParticles-2]))
        else:
            xx1 = torch.cat((x[:kk2], x[-2:-1], x[kk2:kk1-1], x[-3:-2], x[kk1-1:wfunc.nParticles-2]))
            xx2 = torch.cat((x[:kk2], x[-1:], x[kk2:kk1-1], x[-3:-2], x[kk1-1:wfunc.nParticles-2]))
        t1 = Integration.Norm(wf, xx1)
        t2 = Integration.Norm(wf, xx2)
        return t1, t2

    def do_entangled_std_pair(pinp, kk1, kk2, seed=None, N=None):
        """calculate entanglement of one position against all others

        Args:
            pinp (tensor or list): parameters
            kk (int): which position
            seed (int, optional): seed for the random points. Defaults to None.
            N (int, optional): number of random points to use. Defaults to None and takes a global variable.

        Returns:
            tuple of float: standard deviation and mean of the entanglement measure
        """
        if kk1 == kk2:
            raise Exception('k1 and k2 must be different')
        ppp = torch.tensor(pinp)
        if N is not None:
            N_Int_Points_loc = N
        else:
            N_Int_Points_loc = N_Int_Points
        N_Int_Points_loc = int(np.sqrt(N_Int_Points_loc))
        IntElectron = [[-wfunc.calc_int_electron(ppp), wfunc.calc_int_electron(ppp)]]
        IntNuclei = [[-wfunc.calc_int_nuclei(ppp), wfunc.calc_int_nuclei(ppp)]]
        intdomain = IntElectron*wfunc.nElectrons+IntNuclei*wfunc.nNuclei
        if kk2 > kk1:
            intdomain_d = intdomain[:kk1] + intdomain[kk1+1:kk2] + intdomain[kk2+1:]
        else:
            intdomain_d = intdomain[:kk2] + intdomain[kk2+1:kk1] + intdomain[kk1+1:]
        stds = []
        means = []
        densities = []
        for _ in range(N_Int_Points_loc):
            rand_tensor = torch.rand(1, len(intdomain_d)) * (torch.tensor(intdomain_d)[:, 1]-torch.tensor(intdomain_d)[:, 0]) + torch.tensor(intdomain_d)[:, 0]
            rand_same1 = torch.rand(N_Int_Points_loc, 1) * (torch.tensor(intdomain[kk1])[1] - torch.tensor(intdomain[kk1])[0]) + torch.tensor(intdomain[kk1])[0]
            rand_same2 = torch.rand(1, 2) * (torch.tensor(intdomain[kk2])[1] - torch.tensor(intdomain[kk2])[0]) + torch.tensor(intdomain[kk2])[0]
            rand_tensor = torch.cat((rand_tensor.repeat((N_Int_Points_loc, 1)), rand_same1, rand_same2.repeat((N_Int_Points_loc, 1))), dim=1)
            t1, t2 = vmap(lambda y: Integration.get_ratio_pair(lambda x: wfunc.used(ppp, x), y, kk1, kk2), chunk_size=vmap_chunk_size)(rand_tensor)
            # res = torch.log(t1 / (t2+1E-10))
            # stds.append(res.std())
            # means.append(res.mean())
            # print('mean', res.mean(), 'std', res.std())
            res = torch.log((t1+1E-10) / (t2+1E-10))
            dens_mean = (t1+t2).sum() + 1E-5
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
        print('entanglement measure of pair', kk1, kk2, 'full mean', res_mean, 'std', res)
        return res, res_mean

    def show_entanglement(ppp, N=None):
        """calculate for every particle the entanglement ratio"""
        for k in range(wfunc.nParticles):
            Integration.do_entangled_std(ppp, k, N)


if __name__ == "__main__":
    # create permutations of identical particles, only the electrons are permuted, the nuclei not at the moment.
    perms = []
    perms_p = []
    for i in permutations(list(range(wfunc.nElectrons))):
        a = Permutation(list(i))
        p = a.parity()
        if p == 0:
            p = -1
        # print(i, p)
        perms.append(list(i) + list(range(wfunc.nElectrons, wfunc.nParticles)))
        perms_p.append(p)

    perms = np.array(perms)
    # print(perms)

    wfunc.set_ppp()

    print('perms.shape', perms.shape)
    if do_plot_every is not None:
        Integrator_plot = VEGAS()
        plot.doplot(wfunc.ppp)
        if plot.replot:
            plt.show(block=False)

    plotcounter = 0

    Integrator = VEGAS()
    map_Norm = None
    map_H = None

    if check_pair_entanglement:
        Integration.do_entangled_std_pair(wfunc.ppp, 2, 0)
        Integration.do_entangled_std_pair(wfunc.ppp, 0, 2)
        Integration.do_entangled_std_pair(wfunc.ppp, 0, 1)
        Integration.do_entangled_std_pair(wfunc.ppp, 0, 3)
        Integration.do_entangled_std_pair(wfunc.ppp, 1, 3)
        Integration.do_entangled_std_pair(wfunc.ppp, 0, 4)
        Integration.do_entangled_std_pair(wfunc.ppp, 1, 4)
        Integration.do_entangled_std_pair(wfunc.ppp, 3, 4)
        Integration.do_entangled_std_pair(wfunc.ppp, 3, 5)
        Integration.do_entangled_std_pair(wfunc.ppp, 2, 5)
        Integration.do_entangled_std_pair(wfunc.ppp, 5, 2)
        Integration.do_entangled_std_pair(wfunc.ppp, 3, 0)

    if doexcited:
        print("calculation of excited state, check if already orthogonal")
        if not doSchmidtGraham:
            print("otherwize set doSchmidtGraham=True")
        else:
            print("Schmidt Graham orthogonalization is active")
        factor_for_Schmidt_Graham = Integration.doNormOverlap(wfunc.ppp)
        print("factor_for_Schmidt_Graham", factor_for_Schmidt_Graham, "if close to 0, then the wave function is already orthogonalized")

    # this checks for the integration error
    # doIntegration(ppp)  # activate this to check, if integration with the same seed gives the same result
    calc_std = []
    for s in range(5):
        calc_std.append(Integration.doIntegration(wfunc.ppp, seed=s))
    calc_std = float(np.array(calc_std).std())
    print("calc_std", calc_std, 'H_precision_expected', H_precision_expected, ' both should be similar in the thumb rule of Spall, IEEE, 1998, 34, 817-823')

    starttime = time.time()
    Integration.show_entanglement(wfunc.ppp, 10 * N_Int_Points)

    # This very robust optimizer is not very fast in my tests, at least two orders of magnitude slower than SPSA
    # ret = minimizeCompass(doIntegration, x0=ppp, deltainit=0.6, deltatol=0.1, bounds=[[0.01, 20.0]] * (ppp.shape[0]), errorcontrol=do_errorcontrol, funcNinit=30, feps=0.003, disp=True, paired=True, alpha=0.2)
    ret = minimizeSPSA(Integration.doIntegration, x0=wfunc.ppp, bounds=[[0.01, 20.0]] * (wfunc.ppp.shape[0]), disp=True, niter=100, c=H_precision_expected, paired=do_paired, a=start_step)  # , gamma=0.2, a=0.2)

    print(ret)
    print("time", time.time() - starttime)

    print("some checks on the minimum")
    center = Integration.doIntegration(ret.x, seed=0, N=10 * N_Int_Points)
    for k in range(len(ret.x)):
        addx = np.zeros(len(ret.x))
        addx[k] = 0.5
        newx = ret.x + addx
        ny = Integration.doIntegration(newx, seed=0, N=10 * N_Int_Points)
        print(k, (ny - center))
        newx = ret.x - addx
        newx = np.clip(newx, 0.1, 20.0)
        ny = Integration.doIntegration(newx, seed=0, N=10 * N_Int_Points)
        print(k, (ny - center))

    print("integration with higher precision")
    Integration.doIntegration(ret.x, seed=None, N=10 * N_Int_Points)
    Integration.doIntegration(ret.x, seed=None, N=100 * N_Int_Points)

    Integration.show_entanglement(ret.x, 10 * N_Int_Points)

    if do_plot_every is not None:
        plot.doplot(torch.tensor(ret.x))
        plt.show(block=True)
