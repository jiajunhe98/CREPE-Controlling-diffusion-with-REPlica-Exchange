import torch
from torch import nn
import numpy as np

from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools import testsystems

import energy.openmm_interface as omi
import multiprocessing as mp


class AldpBoltzmann(nn.Module):
    def __init__(self, temperature=300, env='implicit', n_threads=8, device='cpu'):
        super(AldpBoltzmann, self).__init__()

        ndim = 66
        self.device = torch.device(device)  # Set device
        
        # System setup
        if env == 'vacuum':
            self.system = testsystems.AlanineDipeptideVacuum(constraints=None)
        elif env == 'implicit':
            self.system = testsystems.AlanineDipeptideImplicit(constraints=None)
        else:
            raise NotImplementedError('This environment is not implemented.')
    
        
        # Enable CUDA in OpenMM
        self.platform = mm.Platform.getPlatformByName('CUDA') if device == 'cuda' else mm.Platform.getPlatformByName('CPU')
        
        self.openmm_energy = omi.OpenMMEnergyInterfaceParallel.apply
        self.regularize_energy = omi.regularize_energy

        energy_cut = torch.tensor(1.e+8, device=self.device)
        energy_max = torch.tensor(1.e+20, device=self.device)

        # Multiprocessing is CPU-based, consider alternative approaches for GPU parallelism
        self.pool = mp.Pool(n_threads, omi.OpenMMEnergyInterfaceParallel.var_init,
                            (self.system, temperature))
        
        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos.to(self.device), self.pool)[:, 0],  # Move input tensor to GPU
            energy_cut, energy_max
        )


    def log_prob(self, x: torch.tensor):
        return -self.norm_energy(x.to(self.device))  # Ensure x is on GPU
