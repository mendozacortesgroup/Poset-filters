import unittest
import numpy as np

# Try to import PyTorch and TensorFlow, set the framework variable
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class TestFilterN(unittest.TestCase):
    def setUp(self):
        if TORCH_AVAILABLE:
            self.framework = "torch"
        elif TF_AVAILABLE:
            self.framework = "tf"
        else:
            raise ImportError("Neither PyTorch nor TensorFlow is installed. Please install at least one.")

    def numpy_to_tensor(self, data):
        if self.framework == "torch":
            return torch.tensor(data, dtype=torch.float32)
        elif self.framework == "tf":
            return tf.convert_to_tensor(data, dtype=tf.float32)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

    def tensor_to_numpy(self, tensor):
        if self.framework == "torch":
            return tensor.detach().cpu().numpy()
        elif self.framework == "tf":
            return tensor.numpy()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

    def test_odd_input(self):
        # Example inputs and outputs in NumPy
        input_data = np.zeros((1, 4, 5, 5), dtype=np.float32)
        input_data[0, 0, 0::2, 0::2] = 1
        input_data[0, 1:, :, :] = 1
        input_data[0, 2, 1::2, :] = -1
        input_data[0, 3, :, 0::2] = -1

        expected_output = np.array(
            [[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
              [[4., 4., 2.], [4., 4., 2.], [2., 2., 1.]],
              [[2., 2., 1.], [2., 2., 1.], [2., 2., 1.]],
              [[2., 2., 0.], [2., 2., 0.], [1., 1., 0.]]]]
        )

        # Convert to framework-specific tensor
        input_tensor = self.numpy_to_tensor(input_data)

        # Call the filter_n function (assume a framework-agnostic interface is implemented)
        if self.framework == "torch":
            from poset_utils import filter_n
            result_tensor = filter_n.forward(input_tensor)
        elif self.framework == "tf":
            try:
                from poset_utils_tf import filter_n_tf
            except ImportError as e:
                raise NotImplementedError("The function 'filter_n_tf' is not implemented yet or the module 'poset_utils_tf' is missing.\n Please, help us creating a TF version of the posets-filters.") from e

            result_tensor = filter_n_tf(input_tensor)

        # Convert result back to NumPy for validation
        result = self.tensor_to_numpy(result_tensor)
        np.testing.assert_array_almost_equal(result, expected_output, decimal=5)

    def test_even_input(self):
        # Example inputs and outputs in NumPy
        input_data = np.ones((1, 3,8, 8), dtype=np.float32)
        input_data[0,0,0::2,0::2]=-1
        input_data[0,1,0::2,:]=-1
        input_data[0,2,1::2,:]=-1
        input_data[0,2,:,0::2]=-1
        expected_output = np.array(
           [[[[3., 3., 3., 3.],[3., 3., 3., 3.],[3., 3., 3., 3.],[3., 3., 3., 3.]],
           [[1., 1., 1., 1.],[1., 1., 1., 1.],[1., 1., 1., 1.],[1., 1., 1., 1.]],
           [[1., 1., 1., 1.],[1., 1., 1., 1.],[1., 1., 1., 1.],[1., 1., 1., 1.]]]]
        )

        # Convert to framework-specific tensor
        input_tensor = self.numpy_to_tensor(input_data)

        # Call the filter_n function (assume a framework-agnostic interface is implemented)
        if self.framework == "torch":
            from poset_utils import filter_n
            result_tensor = filter_n.forward(input_tensor)
        elif self.framework == "tf":
            try:
                from poset_utils_tf import filter_n_tf
            except ImportError as e:
                raise NotImplementedError("The function 'filter_n_tf' is not implemented yet or the module 'poset_utils_tf' is missing. \n Please, help us creating a TF version of the posets-filters.") from e

            result_tensor = filter_n_tf(input_tensor)

        # Convert result back to NumPy for validation
        result = self.tensor_to_numpy(result_tensor)
        np.testing.assert_array_almost_equal(result, expected_output, decimal=5)

    def test_backpropagation_shape(self):
        """This is an implementation specific test"""
        """We compute the back propagation expected output shape"""
        if self.framework == "torch":
            from debugging._poset_utils import _Poset_Activations
            from poset_utils import dict_posets
        elif self.framework == "tf":
            self.skipTest("No need to test back propagation on tf.")
    
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
        """This is an implementation specific test"""
        """We compute the back propagation expected output"""
        if self.framework == "torch":
            from debugging._poset_utils import _Poset_Activations
            from poset_utils import dict_posets
        elif self.framework == "tf":
            self.skipTest("No need to test back propagation on tf.") 
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
