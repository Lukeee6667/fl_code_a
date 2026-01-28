import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from agg_ims_fast import MaskContext

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.fc = nn.Linear(10 * 26 * 26, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def test_mask_context():
    model = SimpleModel()
    prunable_layers = [
        {'name': 'conv1', 'shape': (10, 1, 1, 1), 'size': 10},
        {'name': 'fc', 'shape': (10, 1), 'size': 10}
    ]
    
    # Random masks
    mask_list = [torch.ones(10), torch.ones(10)]
    
    # Original weights
    orig_conv_weight = model.conv1.weight.clone()
    
    print("Original weight norm:", torch.norm(orig_conv_weight).item())

    # Use context
    print("Entering context with ones mask...")
    with MaskContext(model, mask_list, prunable_layers):
        # Modify weights in place check
        # With ones mask, weights should be same
        assert torch.allclose(model.conv1.weight, orig_conv_weight)
        print("  Inside context (ones): weights match original.")
        
        # Try zero mask
        print("  Entering nested context with zeros mask...")
        mask_list_zero = [torch.zeros(10), torch.zeros(10)]
        with MaskContext(model, mask_list_zero, prunable_layers):
            assert torch.allclose(model.conv1.weight, torch.zeros_like(orig_conv_weight))
            print("    Inside nested context (zeros): weights are zero.")
            
        # Back to ones (outer context)
        assert torch.allclose(model.conv1.weight, orig_conv_weight)
        print("  Back to outer context (ones): weights match original.")

    # Back to original
    assert torch.allclose(model.conv1.weight, orig_conv_weight)
    print("Exited context: weights match original.")
    print("MaskContext Test Passed!")

if __name__ == "__main__":
    test_mask_context()
