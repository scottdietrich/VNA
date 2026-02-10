"""Instrument controllers for VNA FMR measurement system."""

from .vna import VNAController
from .magnet import MagnetController
from .cryomagnetics import CryomagneticsController
from .nhmfl import NHMFLMagnetController
from .keithley import KeithleyController
from .gate_safety import GateSafetyWrapper
from .lakeshore import Lakeshore370Controller


def create_magnet_controller(model='SCM1', resource_manager=None):
    """Create magnet controller based on model selection.

    Args:
        model: 'SCM1' or 'Cryomagnetics 4G'
        resource_manager: Optional pyvisa ResourceManager (not used for SCM1)

    Returns:
        NHMFLMagnetController or CryomagneticsController instance
    """
    if model == 'Cryomagnetics 4G':
        return CryomagneticsController(resource_manager)
    else:
        return NHMFLMagnetController()  # SCM1 (default)


__all__ = [
    'VNAController',
    'MagnetController',
    'CryomagneticsController',
    'NHMFLMagnetController',
    'KeithleyController',
    'GateSafetyWrapper',
    'Lakeshore370Controller',
    'create_magnet_controller',
]
