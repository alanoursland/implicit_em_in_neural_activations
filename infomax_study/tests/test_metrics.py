import torch
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import weight_redundancy, effective_rank, dead_units

def test_effective_rank_identity():
    # Identity matrix should have full rank
    W = torch.eye(10)
    rank = effective_rank(W)
    assert abs(rank - 10) < 0.1

def test_effective_rank_rank_one():
    # Rank-1 matrix
    W = torch.ones(10, 5)
    rank = effective_rank(W)
    assert abs(rank - 1) < 0.1

def test_weight_redundancy_orthogonal():
    # Orthogonal rows should have low redundancy
    W = torch.eye(5, 10)
    redundancy = weight_redundancy(W)
    assert redundancy < 0.1

def test_weight_redundancy_identical():
    # Identical rows should have high redundancy
    W = torch.ones(5, 10)
    redundancy = weight_redundancy(W)
    assert redundancy > 1.0

def test_dead_units_none():
    a = torch.randn(100, 16)
    dead = dead_units(a)
    assert dead == 0

def test_dead_units_all():
    a = torch.zeros(100, 16)
    dead = dead_units(a)
    assert dead == 16
