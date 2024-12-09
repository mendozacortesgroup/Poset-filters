#to test back propagation
# Dolores Cuenca Eric
import torch
import torch.nn as nn
import torch.autograd
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
# -------------------------------------------------------------------------
# DEFINITION OF NEW POOLING
# -------------------------------------------------------------------------

class _Poset_Activations(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, coeffs):
        device = input.device  # Ensure all operations are on the same device
        batch_size, channels, height, width = input.shape
        pad_h = (1 if height % 2 != 0 else 0)# Dynamically add padding for odd dimensions
        pad_w = (1 if width % 2 != 0 else 0)
        padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
        #assert width % 2 == 0 and height % 2 == 0, f"Width and height must be even for halving {input.shape}"

        if pad_h > 0 or pad_w > 0:
            input = F.pad(input, padding) #When using the CUDA backend, this operation may induce nondeterministic behaviour in its backward pass that is not easily switched off.
            
        ctx['save_for_backward']=input
        ctx['coeffs'] = coeffs
        ctx['padding'] = padding  # Save padding for backward pass
        # Reshape and stack input blocks for polynomial computation
        blocks = input.unfold(2, 2, 2).unfold(3, 2, 2)  # shape (batch_size, channels, height//2, width//2, 2, 2)
        blocks = blocks.reshape(batch_size, channels, (height+pad_h)//2, (width+pad_w)//2, 4)  # shape (batch_size, channels, height//2, width//2, 4)
        # Send data that is outside the network to GPU
        blocks = blocks.to(device)
        coeffs = [[torch.tensor(c, device=device) for c in coeff] for coeff in coeffs]
        # Prepare polynomials
        polys = []
        for coeff in coeffs:
            a, b, c, d = coeff
            poly = a * blocks[..., 0] + b * blocks[..., 1] + c * blocks[..., 2] + d * blocks[..., 3]
            polys.append(poly)
        poly_stack = torch.stack(polys, dim=-1)  # shape (batch_size, channels, height//2, width//2, num_polys)
        # Find max polynomial value for each block
        max_poly, max_idx = torch.max(poly_stack, dim=-1)  # shape (batch_size, channels, height//2, width//2)
        #print(f"\n this are the values of indices of the functions that maximize the input \n {max_idx}\n{[-]^45}")
        ctx['max_idx'] = max_idx
        return max_poly, ctx

    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device  # Ensure all operations are on the same device
        input = ctx['save_for_backward']
        coeffs = ctx['coeffs']
        max_idx = ctx['max_idx']
        padding = ctx['padding']
        batch_size, channels, padded_height, padded_width = input.shape #batch_size, channels, height, width = input.shape
        grad_input = torch.zeros_like(input).to(device)
        # Reshape and stack input blocks for polynomial computation
        blocks = input.unfold(2, 2, 2).unfold(3, 2, 2)  # shape (batch_size, channels, height//2, width//2, 2, 2)
        blocks = blocks.reshape(batch_size, channels, padded_height//2, padded_width//2, 4)  # shape (batch_size, channels, height//2, width//2, 4)
        blocks = blocks.to(device)
        coeffs = [[torch.tensor(c, device=device) for c in coeff] for coeff in coeffs]
        # Repeat grad_output to match block size
        grad_output_expanded = grad_output.unsqueeze(-1).expand(-1, -1, -1, -1, 4)  # shape (batch_size, channels, height//2, width//2, 4)
        grad_output_expanded = grad_output_expanded.to(device)
        for idx, coeff in enumerate(coeffs):
            a, b, c, d = coeff
            # Create gradient block
            grad_block = torch.zeros_like(blocks).to(device)
            mask = (max_idx == idx).unsqueeze(-1).to(device)
            grad_block[..., 0] = grad_output_expanded[..., 0] * a
            grad_block[..., 1] = grad_output_expanded[..., 1] * b
            grad_block[..., 2] = grad_output_expanded[..., 2] * c
            grad_block[..., 3] = grad_output_expanded[..., 3] * d
            # Apply mask
            grad_input_blocks = grad_block * mask
            # Accumulate gradients, if you uses AI, the AI tends to confuse the next line, it is important to verify it
            grad_input += grad_input_blocks.view(batch_size, channels, padded_height//2, padded_width//2, 2, 2).permute(0, 1, 2, 4, 3, 5).contiguous().view(batch_size, channels, padded_height, padded_width)
        # Remove padding from grad_input if padding was applied
        if padding != (0, 0, 0, 0):
            left, right, top, bottom = padding
            grad_input = grad_input[:, :, top:padded_height - bottom, left:padded_width - right]
    
        return grad_input, None

class _PosetFilter(nn.Module):
    def __init__(self, coeffs):
        super(_PosetFilter, self).__init__()
        self.coeffs = coeffs
    def forward(self, x):
        #print(f"input vector with shape {x.shape}")
        return _Poset_Activations.apply(x, self.coeffs)    

#Example use:
'''#on the init section of the NN
self.pool = PosetActivation(coeffs=[
            (0., 0.0, 0.0, 0.),  # Coefficients for f_1
            (0.0, 0., 0., 1.0),  # Coefficients for f_2
            (0., 1.0, 0.0, 0.),  # Coefficients for f_3
            (0., 1.0, 0.0, 1.),  # Coefficients for f_4
            (1., 1.0, 0.0, 0.),  # Coefficients for f_5
            (0., 1.0, 1.0, 1.),  # Coefficients for f_6
            (1., 1.0, 0.0, 1.),   # Coefficients for f_7
            (1., 1.0, 1.0, 1.)   # Coefficients for f_0
        ])
#on the forward section of the NN
out= self.N_pool(out)
'''

_filter_n = _PosetFilter(coeffs=[
            (0., 0.0, 0.0, 0.),  
            (0.0, 0., 0., 1.0),  
            (0., 1.0, 0.0, 0.),  
            (0., 1.0, 0.0, 1.),  
            (1., 1.0, 0.0, 0.),  
            (0., 1.0, 1.0, 1.),  
            (1., 1.0, 0.0, 1.),  
            (1., 1.0, 1.0, 1.)   
        ])


