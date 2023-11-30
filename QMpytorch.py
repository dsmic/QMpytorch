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

N_Int_Plot = 50000
"""number of integration points for plotting"""
N_Int_Points = 2000000
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
doSchmidtGraham = True
"""if True, then the Schmidt Graham orthogonalization is done"""
H_precision_expected = 0.05
"""This is the c parameter of the SPSA optimizer, it is approximatly the standard deviation of the energy"""
start_step = 0.3
"""This is the a parameter of the SPSA optimizer"""
num_iterations = 100
"""Number of iterations for optimization"""
do_plot_every = 111
"""plot during calculation after do_plot_every integrations """

check_pair_entanglement = True
"""check entanglement of pairs of particles"""
check_pair_symmety = False
"""check symmetry of pairs of particles"""
check_integration_error = True
"""check integration error"""
check_entanglement = True
"""check entanglement of one particle against all others"""
check_long_integration = False
"""check long integration"""
check_at_minimum = True
"""check at minimum"""


class wfunc_class:
    def __init__(self):
        """This class defines the wave function and the parameters"""
        self.ppp_ground = np.array([1.9578, 1.3500, 2.0136])
        """parameters for the ground state"""
        self.ppp_excited = np.array([1.7312, 0.2464, 2.6218])  # , 5.5342])
        """parameters for the excited state"""

        # start values of the parameters of the wave function
        # ppp[0] = distance between nuclei
        # ppp[1] = is used for the integration range of the nuclei
        # ppp[2] = is used for the integration range of the electrons

        self.nNuclei = 4
        """number of nuclei"""
        self.nElectrons = 4
        """number of electrons"""

        self.nParticles = self.nElectrons + self.nNuclei
        """total number of particles"""
        self.spin_state = [
            torch.tensor([0, 1, 0, 0, 0, 0, 0, 0]),
            # torch.tensor([0, 0, 1, 0, 0, 0, 0, 0]),
            ]
        """corresponds to the wave function"""
        # spin wave function
        self.sf_array = torch.zeros([2] * self.nElectrons)
        """spin wave function, implemented as array"""
        self.sf_array[0, 1, 0, 0] = -1
        self.sf_array[0, 0, 1, 0] = 1
        self.sf_array[1, 0, 0, 0] = 1
        self.sf_array[0, 0, 0, 1] = -1

        self.factor_for_Schmidt_Graham = None  # if we need orthogonalization this is different from 0

    def set_ppp(self, ):
        """set the parameters for the wave function"""
        if doexcited:
            self.ppp = torch.tensor(self.ppp_excited)
        else:
            self.ppp = torch.tensor(self.ppp_ground)

    def setoffset(self, ppp):
        """create offset from parameters
        at 0 there must be high spatial probability density for VEGAS integration to work therefore offsets are used to shift
        Args:
            ppp (tensor): parameters of the wave function
        """
        self.offsets = torch.zeros(self.nParticles)
        for i in range(self.nNuclei):
            """now also even number of nuclei should work"""
            self.offsets[i+self.nElectrons] = self.calc_dist_nuclei(ppp) * (2*i + 1 - self.nNuclei)/2

    def set_perms(self):
        """create all permutations of the electrons only"""
        perms = []
        self.perms_p = []
        for i in permutations(list(range(self.nElectrons))):
            a = Permutation(list(i))
            p = a.parity()
            if p == 0:
                p = -1
            # print(i, -p)
            perms.append(list(i) + list(range(self.nElectrons, self.nParticles)))
            self.perms_p.append(p)

        self.perms = np.array(perms)
        print('perms.shape', self.perms.shape)

    def xo_from_x(self, x):
        """calculate the x with offset from the x without

        Args:
            x (tensor): x without offset

        Returns:
            tensor: x with offsets
        """
        return x + self.offsets

    # the following functions are used to calculate the integration ranges
    def calc_int_electron(self, ppp):
        """
        Args:
            ppp: tensor of parameters
        Returns:
            result is the + - range of the electron integration
        """
        return (ppp[0] * self.nNuclei / 2 + ppp[2]) * 1.2

    def calc_int_nuclei(self, ppp):
        """
        Args:
            ppp: tensor of parameters
        Returns:
            result is the + - range of the nuclei integration
        """
        return 1.3 * ppp[1]

    def calc_dist_nuclei(self, ppp):
        """get the nuclei distance from parameters for the offsets

        Args:
            ppp (tensor): parameters
        """
        return ppp[0]

    def wf_form(self, x, w):
        """typical gaussian form to be used in wave functions
        Args:
            x: position
            w: width parameter
        """
        return torch.exp(-(x/w)**2)
        # ret = torch.sigmoid(x/w * 2.0) * torch.sigmoid(-x/w * 2.0)
        # return ret * 4.0

    def ground(self, ppp, xx):
        """This defines the wave function

        Args:
            ppp (tensor): tensor of the parameters for the wave function
            xx (tensor): position
        Returns:
            float: value of the wave function
        """
        def sf(x):
            x = x[:self.nElectrons]
            res = self.sf_array[tuple(x)]
            # if res == 0:
            #     raise Exception("unknown spin configuration", x)
            return res
        res = 0
        for tt in self.spin_state:  # torch.tensor([0, 1, 0, 0, 0, 0])
            for i in range(self.perms.shape[0]):
                t = tt[tuple(self.perms[i]), ]
                if sf(t) != 0:
                    x = xx[tuple(self.perms[i]), ]
                    xo = self.xo_from_x(x)
                    res += self.perms_p[i] * sf(t) * (
                                                self.wf_form(xo[:self.nElectrons] - xo[self.nElectrons:], ppp[2]).prod()
                                                # self.wf_form(xo[:self.nElectrons], ppp[3] * self.nElectrons).prod()
                                                )
        # return res * torch.exp(-(x[4] / ppp[1])**2) * torch.exp(-((x[5] - x[4]) / ppp[1])**2) * torch.exp(-((x[3] - x[4]) / ppp[1])**2)
        # return res * torch.exp(-(x[self.nElectrons:] / ppp[1])**2).prod(-1)  # nuclei not permuted here
        return res * torch.exp(-((x[self.nElectrons+1:] - x[self.nElectrons:-1]) / ppp[1])**2).prod(-1) * torch.exp(-(x[self.nElectrons:].sum() / self.nNuclei / ppp[1])**2).prod(-1)  # nuclei not permuted here

    def excited(self, ppp, xx):
        """This defines the wave function of the excited state

        This will be orthogonalized against the ground state during the calculation

        Args:
            ppp (tensor): tensor of the parameters for the wave function
            xx (tensor): position
        Returns:
            float: value of the wave function
        """
        def sf(x):
            x = x[:self.nElectrons]
            res = self.sf_array[tuple(x)]
            # if res == 0:
            #     raise Exception("unknown spin configuration", x)
            return res
        res = 0
        for i in range(self.perms.shape[0]):
            for tt in self.spin_state:  # torch.tensor([0, 1, 0, 0, 0, 0])
                x = xx[tuple(self.perms[i]), ]
                t = tt[tuple(self.perms[i]), ]
                xo = self.xo_from_x(x)
                res += self.perms_p[i] * sf(t) * (
                                    # (xo[0]-xo[0 + self.nElectrons]) *
                                    # (xo[1]-xo[1 + self.nElectrons]) *
                                    # (xo[2]-xo[2 + self.nElectrons]) *
                                    (xo[3]-xo[3 + self.nElectrons]) *
                                    self.wf_form(xo[:self.nElectrons] - xo[self.nElectrons:], ppp[2]).prod()
                                    )
        # return res * torch.exp(-(x[4] / ppp[1])**2) * torch.exp(-((x[5] - x[4]) / ppp[1])**2) * torch.exp(-((x[3] - x[4]) / ppp[1])**2)
        return res * torch.exp(-(x[self.nElectrons:] / ppp[1])**2).prod(-1)  # nuclei not permuted here

    def used(self, ppp, xx):
        """This defines the wave function used from defined wave functions to allow Schmidt Graham orthogonalization

        Args:
            ppp (tensor): parameters of the wave function
            xx (tensor): the spatial positions

        Returns:
            float: value of the wave function at xx
        """
        if doexcited:
            if doSchmidtGraham:
                return self.excited(ppp, xx) - self.factor_for_Schmidt_Graham * self.ground(self.ppp_ground, xx)  # only if not orthogonal
            else:
                return self.excited(ppp, xx)
        else:
            return self.ground(ppp, xx)


class Hamiltonian_class:
    def __init__(self, wfunc):
        self.wfunc = wfunc  # prepared for instances
        self.q_nuclei = 1
        """charge of the nuclei"""
        self.m_Nuclei = 18.36
        """mass of the nuclei"""
        self.m_Electron = 1
        """mass of the electrons"""
        self.m = torch.tensor([self.m_Electron]*wfunc.nElectrons + [self.m_Nuclei]*wfunc.nNuclei)
        """tensor of the masses of all particles, taken from previous definitions"""
        self.q = torch.tensor([-1]*wfunc.nElectrons + [self.q_nuclei]*wfunc.nNuclei)
        """the charges of all particles"""
        self.V_strength = 1.0
        """Factor for the potential to quickly change strength"""

    # A potential of a one dimensional chain simelar to the 3D Coulumb, checked with Mathematica
    # V[x_, y_, z_] := 1/Sqrt[x^2 + y^2 + z^2]
    # Plot[{NIntegrate[V[x, y, z], {y, -0.5, 0.5}, {z, -0.5, 0.5}], 1/(Abs[x/1.2] + 0.28)}, {x, -3, 3}]
    def V(self, dx):
        """
        Args:
            dx: distance of two particles
        Returns:
            the potential from a toy function, not exactly coulomb ...
        """
        # return 1.0 / (torch.abs(dx / 1.2) + 0.28)    # Coulomb potential part integrated from 3D
        return torch.exp(-dx**2)            # Easy to integrate potential

    def Vpot(self, xinp):
        """
        Args:
            xinp: positions
        Returns:
            the Potential
        """
        x = self.wfunc.xo_from_x(xinp)
        x1 = x.reshape(-1, 1)
        x2 = x.reshape(1, - 1)
        dx = x1 - x2
        Vdx = self.q.reshape(-1, 1) * self.V(dx) * self.q.reshape(1, -1)
        Vdx = Vdx.triu(diagonal=1)
        return Vdx.sum() * self.V_strength

    def Epot(self, wf, x):
        """
        Args:
            wf: wave function takes only x
            x: position
        Returns:
            returns the value of the potential energy integrand
        """
        return (torch.conj(wf(x)) * self.Vpot(x) * wf(x)).real

    def Vpot_plot(self, xinp, plot_pos):
        """Potential energy for plotting only"""
        x = self.wfunc.xo_from_x(xinp)
        x1 = x.reshape(-1, 1)
        x2 = x.reshape(1, - 1)
        dx = x1 - x2
        Vdx = self.q.reshape(-1, 1) * self.V(dx) * self.q.reshape(1, -1)
        Vdx *= 1-torch.eye(self.wfunc.nParticles)
        return Vdx[plot_pos].sum()

    def Epot_plot(self, wf, x, plot_pos):
        """Potential Energy of a particle for plotting"""
        xx = torch.cat((x[:plot_pos], torch.tensor([0]), x[plot_pos+1:]))  # plotting the potential energy of one particle not taking into account the own spatial probability density
        return (torch.conj(wf(xx)) * self.Vpot_plot(x, plot_pos) * wf(xx)).real

    def H_single(self, wf, x):
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
        v = 1/(2*self.m)  # from partial integration the minus sign already present
        gg = torch.sqrt(v) * gg
        return ((torch.dot(torch.conj(gg), gg) + self.Epot(wf, x)).real)

    def H(self, wf, x):
        """vectorized function of H_single"""
        gg = vmap(lambda x: self.H_single(wf, x), chunk_size=vmap_chunk_size)(x)
        return gg


class plot_class:
    def __init__(self, wfunc, Hamiltonian, Integration):
        self.plot_factor = 1.0
        """scaling of the plot with respect to the integration range"""
        self.plot_npoints = 30
        """number of x points for the plots"""
        self.plot_positions = [0, 4, 5, 6]
        self.wfunc = wfunc
        self.Hamiltonian = Hamiltonian
        self.Integration = Integration

    def plotwf(self, ppp, plot_pos, where):
        """plot function for wave function

        Args:
            ppp (tensor): parameters
            plot_pos (int): which position to plot
            where (int): on which subplot to plot

        """
        if plot_pos is None:
            return
        self.wfunc.setoffset(ppp)
        IntElectron = [[-self.wfunc.calc_int_electron(ppp), self.wfunc.calc_int_electron(ppp)]]
        IntNuclei = [[-self.wfunc.calc_int_nuclei(ppp), self.wfunc.calc_int_nuclei(ppp)]]
        if plot_pos < self.wfunc.nElectrons:
            pl_x = np.linspace(-self.wfunc.calc_int_electron(ppp.cpu()) * self.plot_factor, self.wfunc.calc_int_electron(ppp.cpu()) * self.plot_factor, self.plot_npoints)
        else:
            pl_x = np.linspace(-self.wfunc.calc_int_nuclei(ppp.cpu()) * self.plot_factor, self.wfunc.calc_int_nuclei(ppp.cpu()) * self.plot_factor, self.plot_npoints)
        pl_y = []
        pl_y2 = []

        for x in pl_x:
            def wf(x):
                return self.wfunc.wf_used(ppp, x)
            xinp = [0]*plot_pos + [x] + [0]*(self.wfunc.nParticles-1-plot_pos)
            xinp = torch.from_numpy(np.array(xinp))
            if plot_pos < self.wfunc.nElectrons:
                int_domain = [IntElectron[0]]*plot_pos + [[x, x+0.01]] + [IntElectron[0]]*(self.wfunc.nElectrons-1-plot_pos) + IntNuclei*self.wfunc.nNuclei
            else:
                int_domain = [IntElectron[0]]*self.wfunc.nElectrons + IntNuclei*(plot_pos-self.wfunc.nElectrons) + [[x, x+0.01]] + IntNuclei*(self.wfunc.nParticles - 1 - plot_pos)
            set_log_level('ERROR')
            Integrator_plot = VEGAS()
            integral_value_epot = Integrator_plot.integrate(lambda y: vmap(lambda y: self.Hamiltonian.Epot_plot(lambda x: self.wfunc.used(ppp, x), y, plot_pos),
                                                                           chunk_size=vmap_chunk_size)(y), dim=self.wfunc.nParticles, N=N_Int_Plot,  integration_domain=int_domain)
            Integrator_plot = VEGAS()
            integral_value = Integrator_plot.integrate(lambda y: vmap(lambda y: self.Integration.Norm(lambda x: self.wfunc.used(ppp, x), y), chunk_size=vmap_chunk_size)(y), dim=self.wfunc.nParticles, N=N_Int_Plot,  integration_domain=int_domain)
            set_log_level('WARNING')
            pl_y.append(integral_value.cpu())
            pl_y2.append(integral_value_epot.cpu())  # / (integral_value + 0.000001).cpu())
        pl_y = np.array(pl_y)
        pl_y2 = np.array(pl_y2)
        pl_y = pl_y / abs(pl_y).max() * 100
        pl_y2 = pl_y2 / abs(pl_y2).max() * 100
        where.plot(pl_x + self.wfunc.offsets[plot_pos].cpu().numpy(), pl_y, pl_x + self.wfunc.offsets[plot_pos].cpu().numpy(), pl_y2)

    def doplot(self, pinp):
        """ plot some wave functions and potential energy

        Args:
            pinp (tensor or list): parameters
        """
        ppp = torch.tensor(pinp)
        fig, axs = plt.subplots(nrows=4, sharex=True)
        fig.suptitle('Parameters ' + str(ppp.cpu().numpy()) + "V_strength=" + str(self.Hamiltonian.V_strength))
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        axs[3].clear()
        self.plotwf(ppp, plot_pos=self.plot_positions[0], where=axs[0])
        self.plotwf(ppp, plot_pos=self.plot_positions[1], where=axs[1])
        self.plotwf(ppp, plot_pos=self.plot_positions[2], where=axs[2])
        self.plotwf(ppp, plot_pos=self.plot_positions[3], where=axs[3])
        plt.show()


class Integration_class:
    def __init__(self, wfunc, Hamiltonian):
        self.wfunc = wfunc
        self.Integrator = VEGAS()
        self.map_Norm = None
        self.map_H = None
        self.plotcounter = 0
        self.Hamiltonian = Hamiltonian

    def setplot(self, plot):
        """set the plot class

        Args:
            plot (plot_class): plot class
        """
        self.plot = plot

    def Overlap(self, wf1, wf2, x):
        """Calculates the spatial density of the wave function
        Args:
            wf (function): wave function, takes only a x tensor as input
            x (tensor): x tensor to calculate

        Returns:
            float: the spacial density
        """
        return (torch.conj(wf1(x)) * wf2(x)).real

    def Norm(self, wf, x):
        """Calculates the spatial density of the wave function
        Args:
            wf (function): wave function, takes only a x tensor as input
            x (tensor): x tensor to calculate

        Returns:
            float: the spacial density
        """
        return (torch.conj(wf(x)) * wf(x)).real

    def doIntegration(self, pinp, seed=None, N=None):
        """Does the multi dimensional integration to calculate the energy of the Schrödinger equation
        Args:
            pinp (tensor or list): wave function parameters
            seed (int, optional): random seed for integration. Defaults to None.
            N (int, optional): Number of integration points. Defaults to None, taking a global variable.

        Returns:
            float: value of the integral
        """
        if N is not None:
            N_Int_Points_loc = N
        else:
            N_Int_Points_loc = N_Int_Points
        start = time.time()
        ppp = torch.tensor(pinp)
        self.wfunc.setoffset(ppp)
        # print('offsets', self.wfunc.offsets.cpu().numpy())
        if do_plot_every is not None:
            plt.pause(0.1)
            self.plotcounter += 1
            if self.plotcounter >= do_plot_every:
                self.plotcounter = 0
                self.plot.doplot(pinp)
        plottime = time.time() - start
        IntElectron = [[-self.wfunc.calc_int_electron(ppp), self.wfunc.calc_int_electron(ppp)]]
        IntNuclei = [[-self.wfunc.calc_int_nuclei(ppp), self.wfunc.calc_int_nuclei(ppp)]]
        Normvalue = self.Integrator.integrate(lambda y: vmap(lambda y: self.Norm(lambda x: self.wfunc.used(ppp, x), y), chunk_size=vmap_chunk_size)(y),
                                              dim=self.wfunc.nParticles, N=N_Int_Points_loc,  integration_domain=IntElectron*self.wfunc.nElectrons+IntNuclei*self.wfunc.nNuclei, seed=seed, vegasmap=self.map_Norm, use_warmup=(self.map_Norm is None))
        if seed is None:
            self.map_Norm = self.Integrator.map
        else:
            self.map_Norm = None
        integral_value = self.Integrator.integrate(lambda y: self.Hamiltonian.H(lambda x: self.wfunc.used(ppp, x), y),
                                                   dim=self.wfunc.nParticles, N=N_Int_Points_loc,  integration_domain=IntElectron*self.wfunc.nElectrons+IntNuclei*self.wfunc.nNuclei, seed=seed, vegasmap=self.map_H, use_warmup=(self.map_H is None))
        if seed is None:
            self.map_H = self.Integrator.map
        else:
            self.map_H = None
        if Normvalue < 0.000001:
            print("Normvalue too small", Normvalue, ppp)
            raise Exception("Normvalue too small, probable zero wave function (e.g. due to antisymmetry)")
        retH = integral_value / Normvalue
        print("              H", "{:.5f}".format(float(retH.cpu())), ppp.cpu().numpy(), "{:.2f}".format(time.time() - start), "(" + "{:.2f}".format(plottime) + ")", 'raw Norm integral value', "{:.5f}".format(Normvalue), 'seed', seed)
        return retH.cpu().numpy()

    def doNormOverlap(self, pinp, seed=None, N=None, which_fun=None):
        """Does the multi dimensional integration to calculate calculate the Schmidt Graham orthogonalization of the wave function
        Args:
            pinp (tensor or list): wave function parameters
            seed (int, optional): random seed for integration. Defaults to None.
            N (int, optional): Number of integration points. Defaults to None, taking a global variable.

        Returns:
            float: value of the integral
        """
        if which_fun is None:
            which_fun = self.wfunc.excited
        if N is not None:
            N_Int_Points_loc = N
        else:
            N_Int_Points_loc = N_Int_Points
        start = time.time()
        ppp = torch.tensor(pinp)
        self.wfunc.setoffset(ppp)
        if do_plot_every is not None:
            plt.pause(0.1)
            self.plotcounter += 1
            if self.plotcounter >= do_plot_every:
                self.plotcounter = 0
                self.plot.doplot(pinp)
        plottime = time.time() - start
        IntElectron = [[-self.wfunc.calc_int_electron(ppp), self.wfunc.calc_int_electron(ppp)]]
        IntNuclei = [[-self.wfunc.calc_int_nuclei(ppp), self.wfunc.calc_int_nuclei(ppp)]]
        Normvalue = self.Integrator.integrate(lambda y: vmap(lambda y: self.Norm(lambda x: self.wfunc.ground(self.wfunc.ppp_ground, x), y), chunk_size=vmap_chunk_size)(y),
                                              dim=self.wfunc.nParticles, N=N_Int_Points_loc,  integration_domain=IntElectron*self.wfunc.nElectrons+IntNuclei*self.wfunc.nNuclei, seed=seed, vegasmap=self.map_Norm, use_warmup=(self.map_Norm is None))
        if seed is None:
            self.map_Norm = self.Integrator.map
        else:
            self.map_Norm = None
        Overlapvalue = self.Integrator.integrate(lambda y: vmap(lambda y: self.Overlap(lambda x: self.wfunc.ground(self.wfunc.ppp_ground, x), lambda x: which_fun(ppp, x), y), chunk_size=vmap_chunk_size)(y),
                                                 dim=self.wfunc.nParticles, N=N_Int_Points_loc,  integration_domain=IntElectron*self.wfunc.nElectrons+IntNuclei*self.wfunc.nNuclei, seed=seed, vegasmap=self.map_Norm, use_warmup=(self.map_Norm is None))
        if seed is None:
            self.map_H = self.Integrator.map
        else:
            self.map_H = None
        if Normvalue < 0.000001:
            print("Normvalue too small", Normvalue, ppp)
            raise Exception("Normvalue too small, probable zero wave function (e.g. due to antisymmetry)")
        retH = Overlapvalue / Normvalue
        print("              Overlap", "{:.5f}".format(float(retH.cpu())), ppp.cpu().numpy(), "{:.2f}".format(time.time() - start), "(" + "{:.2f}".format(plottime) + ")", 'raw Norm integral value',
              "{:.5f}".format(Normvalue), "{:.5f}".format(Overlapvalue), 'seed', seed)
        return retH.cpu().numpy()

    def get_ratio(self, wf, x, kk):
        """ratio of norm with different values at one position

        Args:
            wf: wavefunction
            x: The variable, (nParticles-1) and one variable at pos kk from the last two values in the input
                The input has therefore nParticles+1 values
            kk: position of the extra variable

        Returns:
            tuple of float: norm of the two cases
        """
        xx1 = torch.cat((x[:kk], x[-2:-1], x[kk:self.wfunc.nParticles-1]))
        xx2 = torch.cat((x[:kk], x[-1:], x[kk:self.wfunc.nParticles-1]))
        t1 = self.Norm(wf, xx1)
        t2 = self.Norm(wf, xx2)
        return t1, t2

    def do_entangled_std(self, pinp, kk, seed=None, N=None):
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
        self.wfunc.setoffset(ppp)
        if N is not None:
            N_Int_Points_loc = N
        else:
            N_Int_Points_loc = N_Int_Points
        N_Int_Points_loc = int(np.sqrt(N_Int_Points_loc))
        IntElectron = [[-self.wfunc.calc_int_electron(ppp), self.wfunc.calc_int_electron(ppp)]]
        IntNuclei = [[-self.wfunc.calc_int_nuclei(ppp), self.wfunc.calc_int_nuclei(ppp)]]
        intdomain = IntElectron*self.wfunc.nElectrons+IntNuclei*self.wfunc.nNuclei
        intdomain_d = intdomain[:kk] + intdomain[kk+1:]
        stds = []
        means = []
        densities = []
        for _ in range(N_Int_Points_loc):
            rand_tensor = torch.rand(N_Int_Points_loc, len(intdomain_d)) * (torch.tensor(intdomain_d)[:, 1]-torch.tensor(intdomain_d)[:, 0]) + torch.tensor(intdomain_d)[:, 0]
            rand_same = torch.rand(1, 2) * (torch.tensor(intdomain[kk])[1] - torch.tensor(intdomain[kk])[0]) + torch.tensor(intdomain[kk])[0]
            rand_tensor = torch.cat((rand_tensor, rand_same.repeat((N_Int_Points_loc, 1))), dim=1)
            t1, t2 = vmap(lambda y: self.get_ratio(lambda x: self.wfunc.used(ppp, x), y, kk), chunk_size=vmap_chunk_size)(rand_tensor)
            res = torch.log((t1+1E-5) / (t2+1E-5))
            dens_mean = (t1+t2).sum() + 1E-5
            res_mean = (res * (t1+t2)).sum() / dens_mean
            res_std = (res**2 * (t1+t2)).sum() / dens_mean - res_mean**2
            stds.append(res_std)
            means.append(res_mean)
            densities.append(dens_mean)
        res_mean = (np.array(means)*np.array(densities)).sum() / (np.array(densities)).sum()
        res = (np.array(stds)*np.array(densities)).sum() / (np.array(densities)).sum()
        print('entanglement measure of', kk, 'full mean', res_mean, 'std', res)
        return res, res_mean

    def get_ratio_pair(self, wf, x, kk1, kk2):
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
            xx1 = torch.cat((x[:kk1], x[-3:-2], x[kk1:kk2-1], x[-2:-1], x[kk2-1:self.wfunc.nParticles-2]))
            xx2 = torch.cat((x[:kk1], x[-3:-2], x[kk1:kk2-1], x[-1:], x[kk2-1:self.wfunc.nParticles-2]))
        else:
            xx1 = torch.cat((x[:kk2], x[-2:-1], x[kk2:kk1-1], x[-3:-2], x[kk1-1:self.wfunc.nParticles-2]))
            xx2 = torch.cat((x[:kk2], x[-1:], x[kk2:kk1-1], x[-3:-2], x[kk1-1:self.wfunc.nParticles-2]))
        t1 = self.Norm(wf, xx1)
        t2 = self.Norm(wf, xx2)
        return t1, t2

    def do_entangled_std_pair(self, pinp, kk1, kk2, seed=None, N=None):
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
        self.wfunc.setoffset(ppp)
        if N is not None:
            N_Int_Points_loc = N
        else:
            N_Int_Points_loc = N_Int_Points
        N_Int_Points_loc = int(np.sqrt(N_Int_Points_loc))
        IntElectron = [[-self.wfunc.calc_int_electron(ppp), self.wfunc.calc_int_electron(ppp)]]
        IntNuclei = [[-self.wfunc.calc_int_nuclei(ppp), self.wfunc.calc_int_nuclei(ppp)]]
        intdomain = IntElectron*self.wfunc.nElectrons+IntNuclei*self.wfunc.nNuclei
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
            t1, t2 = vmap(lambda y: self.get_ratio_pair(lambda x: self.wfunc.used(ppp, x), y, kk1, kk2), chunk_size=vmap_chunk_size)(rand_tensor)
            res = torch.log((t1+1E-5) / (t2+1E-5))
            dens_mean = (t1+t2).sum() + 1E-5
            res_mean = (res * (t1+t2)).sum() / dens_mean
            res_std = (res**2 * (t1+t2)).sum() / dens_mean - res_mean**2
            stds.append(res_std)
            means.append(res_mean)
            densities.append(dens_mean)
        res_mean = (np.array(means)*np.array(densities)).sum() / (np.array(densities)).sum()
        res = (np.array(stds)*np.array(densities)).sum() / (np.array(densities)).sum()
        print('entanglement measure of pair', kk1, kk2, 'full mean', res_mean, 'std', res)
        return res, res_mean

    def show_entanglement(self, ppp, N=None):
        """calculate for every particle the entanglement ratio"""
        print("entanglement measure at", ppp, "N", N)
        for k in range(self.wfunc.nParticles):
            self.do_entangled_std(ppp, k, N)


def run_all(wfunc=None, Hamiltonian=None, plot=None, Integration=None):
    if wfunc is None:
        wfunc = wfunc_class()
    if Hamiltonian is None:
        Hamiltonian = Hamiltonian_class(wfunc)
    if Integration is None:
        Integration = Integration_class(wfunc, Hamiltonian)
    if plot is None:
        plot = plot_class(wfunc, Hamiltonian, Integration)
    Integration.setplot(plot)
    wfunc.set_perms()
    wfunc.set_ppp()

    if do_plot_every is not None:
        # Integrator_plot = VEGAS()
        plot.doplot(wfunc.ppp)

    if doexcited:
        print("calculation of excited state, check if already orthogonal")
        if not doSchmidtGraham:
            print("otherwize set doSchmidtGraham=True")
        else:
            print("Schmidt Graham orthogonalization is active")
        factor_for_Schmidt_Graham = torch.tensor(Integration.doNormOverlap(wfunc.ppp))
        print("factor_for_Schmidt_Graham", factor_for_Schmidt_Graham, "if close to 0, then the wave function is already orthogonalized")
        print("check overlap with used function after Schmidt Graham orthogonalization")
        Integration.doNormOverlap(wfunc.ppp, which_fun=wfunc.used)

    if check_pair_entanglement:
        for k1 in range(wfunc.nParticles):
            if check_pair_symmety:
                s = 0
            else:
                s = k1+1
            for k2 in range(s, wfunc.nParticles):
                if k1 != k2:
                    Integration.do_entangled_std_pair(wfunc.ppp, k1, k2)

    # this checks for the integration error
    # doIntegration(ppp)  # activate this to check, if integration with the same seed gives the same result
    if check_integration_error:
        calc_std = []
        for s in range(5):
            calc_std.append(Integration.doIntegration(wfunc.ppp, seed=s))
        calc_std = float(np.array(calc_std).std())
        print("calc_std", calc_std, 'H_precision_expected', H_precision_expected, ' both should be similar in the thumb rule of Spall, IEEE, 1998, 34, 817-823')

    starttime = time.time()
    if check_entanglement:
        Integration.doIntegration(wfunc.ppp, seed=None, N=N_Int_Points)
        Integration.show_entanglement(wfunc.ppp, N_Int_Points)

    # This very robust optimizer is not very fast in my tests, at least two orders of magnitude slower than SPSA
    # ret = minimizeCompass(doIntegration, x0=ppp, deltainit=0.6, deltatol=0.1, bounds=[[0.01, 20.0]] * (ppp.shape[0]), errorcontrol=do_errorcontrol, funcNinit=30, feps=0.003, disp=True, paired=True, alpha=0.2)
    ret = minimizeSPSA(Integration.doIntegration, x0=wfunc.ppp, bounds=[[0.01, 20.0]] * (wfunc.ppp.shape[0]), disp=True, niter=num_iterations, c=H_precision_expected, paired=do_paired, a=start_step)  # , gamma=0.2, a=0.2)

    print(ret)
    print("time", time.time() - starttime)

    if check_at_minimum:
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

    if check_long_integration:
        print("integration with higher precision")
        Integration.doIntegration(ret.x, seed=None, N=10 * N_Int_Points)
        Integration.doIntegration(ret.x, seed=None, N=100 * N_Int_Points)

    if check_entanglement:
        Integration.show_entanglement(ret.x, 10 * N_Int_Points)

    if do_plot_every is not None:
        plot.doplot(torch.tensor(ret.x))
        plt.show(block=True)


if __name__ == "__main__":
    run_all()
