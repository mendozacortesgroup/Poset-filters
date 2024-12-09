import torch
from poset_utils import filter_n, dict_posets
import unittest
from debugging._poset_utils import _Poset_Activations

class TestFilterN(unittest.TestCase):
    def test_oddInput(self):
        """we first verify that it accepts odd inputs"""
        input_tensor = torch.zeros(1, 4,5, 5)
        input_tensor[0,0,0::2,0::2]=1 
        input_tensor[0,1:,:,:]=1 
        input_tensor[0,2,1::2,:]=-1
        input_tensor[0,3,:,0::2]=-1     
        result = filter_n.forward( input_tensor)
        expected_tensor = torch.tensor([[[[1., 1., 1.],[1., 1., 1.],[1., 1., 1.]],[[4., 4., 2.],[4., 4., 2.],[2., 2., 1.]],[[2., 2., 1.],[2., 2., 1.],[2., 2., 1.]],[[2., 2., 0.],[2., 2., 0.],[1., 1., 0.]]]])
        self.assertTrue(torch.equal(result, expected_tensor), "The tensors do not match.")

    def test_evenInput(self):
        """Even inputs test"""
        input_tensor = torch.ones(1, 3,8, 8)
        input_tensor[0,0,0::2,0::2]=-1 
        input_tensor[0,1,0::2,:]=-1 
        input_tensor[0,2,1::2,:]=-1
        input_tensor[0,2,:,0::2]=-1 
        result = filter_n.forward( input_tensor)
        expected_tensor = torch.tensor([[[[3., 3., 3., 3.],[3., 3., 3., 3.],[3., 3., 3., 3.],[3., 3., 3., 3.]],
         [[1., 1., 1., 1.],[1., 1., 1., 1.],[1., 1., 1., 1.],[1., 1., 1., 1.]],
         [[1., 1., 1., 1.],[1., 1., 1., 1.],[1., 1., 1., 1.],[1., 1., 1., 1.]]]]
          )
        self.assertTrue(torch.equal(result, expected_tensor), "The tensors do not match.")
    
    def test_backpropagation_shape(self):
        """We compute the back propagation expected output shape"""
        input_tensor = torch.zeros(1, 4,5, 5)
        input_tensor[0,0,0::2,0::2]=1
        input_tensor[0,1:,:,:]=1
        input_tensor[0,2,1::2,:]=-1
        input_tensor[0,3,:,0::2]=-1
        ctx={}
        preresult, ctx = _Poset_Activations.forward(ctx, input_tensor,dict_posets[12])
        result,_ = _Poset_Activations.backward(ctx, torch.ones(1,4,3,3))
        expected_shape_tensor = input_tensor
        self.assertEqual(result.shape, expected_shape_tensor.shape, "The tensors shape do not match.")



    def test_backpropagation(self):
        """We compute the back propagation expected output"""

        input_tensor = torch.ones(1, 3,8, 8)
        input_tensor[0,0,0::2,0::2]=-1
        input_tensor[0,1,0::2,:]=-1
        input_tensor[0,2,1::2,:]=-1
        input_tensor[0,2,:,0::2]=-1

        ctx = {}
        preresult, ctx = _Poset_Activations.forward(ctx, input_tensor, dict_posets[12])
        result,_ = _Poset_Activations.backward(ctx, torch.ones(1,3,4,4))
        expected_tensor = torch.tensor([[[[0., 1., 0., 1., 0., 1., 0., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1.],
          [0., 1., 0., 1., 0., 1., 0., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1.],
          [0., 1., 0., 1., 0., 1., 0., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1.],
          [0., 1., 0., 1., 0., 1., 0., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1.]],
         [[0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 1., 0., 1., 0., 1., 0., 1.],
          [0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 1., 0., 1., 0., 1., 0., 1.],
          [0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 1., 0., 1., 0., 1., 0., 1.],
          [0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 1., 0., 1., 0., 1., 0., 1.]],
         [[0., 1., 0., 1., 0., 1., 0., 1.],
          [0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 1., 0., 1., 0., 1., 0., 1.],
          [0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 1., 0., 1., 0., 1., 0., 1.],
          [0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 1., 0., 1., 0., 1., 0., 1.],
          [0., 0., 0., 0., 0., 0., 0., 0.]]]])
        self.assertTrue(torch.equal(result, expected_tensor), "The tensors do not match.")



if __name__ == "__main__":
    unittest.main()
    
