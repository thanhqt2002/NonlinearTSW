import torch
import random
import pytest
from tsw import TSW

tw = TSW(device='cuda')

@pytest.mark.parametrize('n_trees', [random.randint(1, 10) for _ in range(5)])
@pytest.mark.parametrize('n_lines', [random.randint(1, 20) for _ in range(5)])
@pytest.mark.parametrize('N', [random.randint(1, 20) for _ in range(5)])
def test_edge_length(n_trees, n_lines, N):
    # Create combined data as before
    mass_XY = torch.rand(n_trees, n_lines, 2 * N)
    combined_axis_coordinate = torch.randn(n_trees, n_lines, 2 * N)
    
    # Split into X and Y components
    mass_X = mass_XY[:, :, :N]
    mass_Y = mass_XY[:, :, N:]
    axis_coordinate_X = combined_axis_coordinate[:, :, :N]
    axis_coordinate_Y = combined_axis_coordinate[:, :, N:]
    
    mass_X = mass_X.to("cuda")
    mass_Y = mass_Y.to("cuda")
    axis_coordinate_X = axis_coordinate_X.to("cuda")
    axis_coordinate_Y = axis_coordinate_Y.to("cuda")
    combined_axis_coordinate = combined_axis_coordinate.to("cuda")
    
    _, _, edge_length = tw.tw_concurrent_lines(mass_X, axis_coordinate_X, mass_Y, axis_coordinate_Y)
    
    coord = torch.cat((torch.zeros(n_trees, n_lines, 1, device="cuda"), combined_axis_coordinate), dim=2)
    coord = torch.sort(coord, dim=2)[0]
    actual_edge = coord[:, :, 1:] - coord[:, :, :-1]
    actual_edge = actual_edge.to("cuda")
    assert torch.allclose(edge_length, actual_edge) 
