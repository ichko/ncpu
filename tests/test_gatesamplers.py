
import uuid
import time
import random
import shutil
import unittest
import tempfile
from pathlib import Path
from typing import Optional, List

from ncpu.dataset import NCPUDataset, ScheduledDataset
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
        dataset = NCPUDataset(TINY_XOR_TRAINING_CONFIG) 
        
        for _ in range(64):
            ing, outg = dataset.get_sample()
            ret_in = probe(ing, TINY_XOR_TRAINING_CONFIG, 2, 0)/255.0
            ret_out = probe(outg, TINY_XOR_TRAINING_CONFIG, 0, 1)/255.0

            a = (int)(ret_in[0].item())
            b = (int)(ret_in[1].item())
            self.assertTrue(((a ^ b) == ret_out[0].item()))

    def test_NOR_gate_generation(self):
        dataset = NCPUDataset(TINY_NOR_TRAINING_CONFIG) 
        
        for _ in range(64):
            ing, outg = dataset.get_sample()
            ret_in = probe(ing, TINY_NOR_TRAINING_CONFIG, 2, 0)/255.0
            ret_out = probe(outg, TINY_NOR_TRAINING_CONFIG, 0, 1)/255.0

            a = (int)(ret_in[0].item())
            b = (int)(ret_in[1].item())
            self.assertTrue((not (a | b) == ret_out[0].item()))

    def test_OR_gate_generation(self):
        dataset = NCPUDataset(TINY_OR_TRAINING_CONFIG) 
        
        for _ in range(64):
            ing, outg = dataset.get_sample()
            ret_in = probe(ing, TINY_OR_TRAINING_CONFIG, 2, 0)/255.0
            ret_out = probe(outg, TINY_OR_TRAINING_CONFIG, 0, 1)/255.0

            a = (int)(ret_in[0].item())
            b = (int)(ret_in[1].item())
            self.assertTrue(((a | b) == ret_out[0].item()))

    def test_AND_gate_generation(self):
        dataset = NCPUDataset(TINY_AND_TRAINING_CONFIG) 
        
        for _ in range(64):
            ing, outg = dataset.get_sample()
            ret_in = probe(ing, TINY_AND_TRAINING_CONFIG, 2, 0)/255.0
            ret_out = probe(outg, TINY_AND_TRAINING_CONFIG, 0, 1)/255.0

            a = (int)(ret_in[0].item())
            b = (int)(ret_in[1].item())
            self.assertTrue(((a & b) == ret_out[0].item()))
    
    def test_NAND_gate_generation(self):
        dataset = NCPUDataset(TINY_NAND_TRAINING_CONFIG) 
        
        for _ in range(64):
            ing, outg = dataset.get_sample()
            ret_in = probe(ing, TINY_NAND_TRAINING_CONFIG, 2, 0)/255.0
            ret_out = probe(outg, TINY_NAND_TRAINING_CONFIG, 0, 1)/255.0

            a = (int)(ret_in[0].item())
            b = (int)(ret_in[1].item())
            self.assertTrue((not (a & b) == ret_out[0].item()))

class TestDatasets(unittest.TestCase):

    def test_Scheduled_Dataset_generation(self):
        dataset_nand = NCPUDataset(TINY_NAND_TRAINING_CONFIG) 
        dataset_or = NCPUDataset(TINY_OR_TRAINING_CONFIG) 
        dataset_nor = NCPUDataset(TINY_NOR_TRAINING_CONFIG) 

        steps = 64

        datasets = [dataset_nand, dataset_or, dataset_nor]
        print(datasets)
        scheduledDS = ScheduledDataset(
            datasets = datasets,
            steps = steps
        )

        for _ in range(steps):
            ing, outg = scheduledDS.get_sample()
            ret_in = probe(ing, TINY_NAND_TRAINING_CONFIG, 2, 0)/255.0
            ret_out = probe(outg, TINY_NAND_TRAINING_CONFIG, 0, 1)/255.0

            a = (int)(ret_in[0].item())
            b = (int)(ret_in[1].item())
            self.assertTrue((not (a & b) == ret_out[0].item()))

        for _ in range(steps):
            ing, outg = scheduledDS.get_sample()
            ret_in = probe(ing, TINY_OR_TRAINING_CONFIG, 2, 0)/255.0
            ret_out = probe(outg, TINY_OR_TRAINING_CONFIG, 0, 1)/255.0

            a = (int)(ret_in[0].item())
            b = (int)(ret_in[1].item())
            self.assertTrue(((a | b) == ret_out[0].item()))

        for _ in range(steps):
            ing, outg = scheduledDS.get_sample()
            ret_in = probe(ing, TINY_NOR_TRAINING_CONFIG, 2, 0)/255.0
            ret_out = probe(outg, TINY_NOR_TRAINING_CONFIG, 0, 1)/255.0

            a = (int)(ret_in[0].item())
            b = (int)(ret_in[1].item())
            self.assertTrue((not (a | b) == ret_out[0].item()))