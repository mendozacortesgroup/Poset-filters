import torch
from poset_utils import filter_n
import unittest

class TestFilterN(unittest.TestCase):
    def test_oddInput(self):
        """we first verify that it accepts odd inputs"""
        input_tensor = torch.zeros(1, 4,5, 5)
        input_tensor[0,0,0::2,0::2]=1 
        input_tensor[0,1:,:,:]=1 
        input_tensor[0,2,1::2,:]=-1
        input_tensor[0,3,:,0::2]=-1     
        result = filter_n.forward( input_tensor)
        expected_tensor = torch.tensor([[[[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],

         [[4., 4., 2.],
          [4., 4., 2.],
          [2., 2., 1.]],

         [[2., 2., 1.],
          [2., 2., 1.],
          [2., 2., 1.]],

         [[2., 2., 0.],
          [2., 2., 0.],
          [1., 1., 0.]]]])
        self.assertTrue(torch.equal(result, expected_tensor), "The tensors do not match.")

    def test_evenInput(self):
        """Even inputs test"""
        input_tensor = torch.ones(1, 3,8, 8)
        input_tensor[0,0,0::2,0::2]=-1 
        input_tensor[0,1,0::2,:]=-1 
        input_tensor[0,2,1::2,:]=-1
        input_tensor[0,2,:,0::2]=-1 
        result = filter_n.forward( input_tensor)
        expected_tensor = torch.tensor([[[[3., 3., 3., 3.],
          [3., 3., 3., 3.],
          [3., 3., 3., 3.],
          [3., 3., 3., 3.]],

         [[1., 1., 1., 1.],
          [1., 1., 1., 1.],
          [1., 1., 1., 1.],
          [1., 1., 1., 1.]],

         [[1., 1., 1., 1.],
          [1., 1., 1., 1.],
          [1., 1., 1., 1.],
          [1., 1., 1., 1.]]]])
        self.assertTrue(torch.equal(result, expected_tensor), "The tensors do not match.")
    
if __name__ == "__main__":
    unittest.main()