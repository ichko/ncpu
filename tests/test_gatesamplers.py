
import uuid
import time
import random
import shutil
import unittest
import tempfile
from pathlib import Path
from typing import Optional, List

from ncpu.dataset import NCPUDataset
from ncpu.config import TINY_NAND_TRAINING_CONFIG, TINY_AND_TRAINING_CONFIG, TINY_NOR_TRAINING_CONFIG, TINY_OR_TRAINING_CONFIG, TINY_XOR_TRAINING_CONFIG

from src.ncpu.probe import probe

class TestGates(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass
        
    def tearDown(self):
        pass

    def test_XOR_gate_generation(self):
        config = TINY_XOR_TRAINING_CONFIG
        dataset = NCPUDataset(
            W=config.W,
            H=config.H,
            r=config.r,
            spacing=config.spacing,
            sampler=config.sampler,
            balanced=config.balanced,
            apply_gaussian_noise=config.gaussian_noise,
        ) 
        
        for _ in range(64):
            ing, outg = dataset.get_sample()
            ret_in = probe(ing, config, 2, 0)/255.0
            ret_out = probe(outg, config, 0, 1)/255.0

            a = (int)(ret_in[0].item())
            b = (int)(ret_in[1].item())
            self.assertTrue(((a ^ b) == ret_out[0].item()))

    def test_NOR_gate_generation(self):
        config = TINY_NOR_TRAINING_CONFIG
        dataset = NCPUDataset(
            W=config.W,
            H=config.H,
            r=config.r,
            spacing=config.spacing,
            sampler=config.sampler,
            balanced=config.balanced,
            apply_gaussian_noise=config.gaussian_noise,
        ) 
        
        for _ in range(64):
            ing, outg = dataset.get_sample()
            ret_in = probe(ing, config, 2, 0)/255.0
            ret_out = probe(outg, config, 0, 1)/255.0

            a = (int)(ret_in[0].item())
            b = (int)(ret_in[1].item())
            self.assertTrue((not (a | b) == ret_out[0].item()))

    def test_OR_gate_generation(self):
        config = TINY_OR_TRAINING_CONFIG
        dataset = NCPUDataset(
            W=config.W,
            H=config.H,
            r=config.r,
            spacing=config.spacing,
            sampler=config.sampler,
            balanced=config.balanced,
            apply_gaussian_noise=config.gaussian_noise,
        ) 
        
        for _ in range(64):
            ing, outg = dataset.get_sample()
            ret_in = probe(ing, config, 2, 0)/255.0
            ret_out = probe(outg, config, 0, 1)/255.0

            a = (int)(ret_in[0].item())
            b = (int)(ret_in[1].item())
            self.assertTrue(((a | b) == ret_out[0].item()))

    def test_AND_gate_generation(self):
        config = TINY_AND_TRAINING_CONFIG
        dataset = NCPUDataset(
            W=config.W,
            H=config.H,
            r=config.r,
            spacing=config.spacing,
            sampler=config.sampler,
            balanced=config.balanced,
            apply_gaussian_noise=config.gaussian_noise,
        ) 
        
        for _ in range(64):
            ing, outg = dataset.get_sample()
            ret_in = probe(ing, config, 2, 0)/255.0
            ret_out = probe(outg, config, 0, 1)/255.0

            a = (int)(ret_in[0].item())
            b = (int)(ret_in[1].item())
            self.assertTrue(((a & b) == ret_out[0].item()))
    
    def test_NAND_gate_generation(self):
        config = TINY_NAND_TRAINING_CONFIG
        dataset = NCPUDataset(
            W=config.W,
            H=config.H,
            r=config.r,
            spacing=config.spacing,
            sampler=config.sampler,
            balanced=config.balanced,
            apply_gaussian_noise=config.gaussian_noise,
        ) 
        
        for _ in range(64):
            ing, outg = dataset.get_sample()
            ret_in = probe(ing, config, 2, 0)/255.0
            ret_out = probe(outg, config, 0, 1)/255.0

            a = (int)(ret_in[0].item())
            b = (int)(ret_in[1].item())
            self.assertTrue((not (a & b) == ret_out[0].item()))