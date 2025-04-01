"""TaxTracker package.

Provides modules for tracking and calculating the tax treatment of trades and distributions.
"""
from .tax_tracker import CapGainsTracker, Config, Lot, DistributionTracker, PNLtracker, TaxLotSelection

__all__ = ('Config', 'Lot', 'TaxLotSelection','CapGainsTracker', 'DistributionTracker', 'PNLtracker')
__version__ = '0.1.2'
__author__ = 'David Bookstaber'