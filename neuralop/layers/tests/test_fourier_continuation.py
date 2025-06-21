import torch
from ..fourier_continuation import FCLegendre, FCGram

def test_Legendre_fourier_continuation_1d():
    
    batch_size = 3
    length_signal = 101
    add_pts = 40

    x = torch.linspace(0, 1, length_signal).repeat(batch_size,1)
    f = torch.sin(16 * x) - torch.cos(8 * x)

    Extension = FCLegendre(d=2, n_additional_pts=add_pts)
    f_extend_one_side = Extension(f, dim=1, one_sided=True)
    f_extend_both_sides = Extension(f, dim=1, one_sided=False)

    # Check shape
    assert f_extend_both_sides.shape[-1] == f_extend_one_side.shape[-1] == f.shape[-1] + add_pts
    # Check values of original signal
    torch.testing.assert_close(f, f_extend_one_side[...,:length_signal])
    torch.testing.assert_close(f, f_extend_both_sides[...,add_pts//2:-add_pts//2])

    # # Plot to verify
    # import matplotlib.pyplot as plt
    # x_extended_one_side = torch.linspace(0, 1.4, 141) 
    # x_extended_both_sides = torch.linspace(-0.2, 1.2, 141) 
    # # Add 0.5 and -0.5 for plotting purposes
    # f_extend_one_side = f_extend_one_side + 0.6
    # f_extend_both_sides = f_extend_both_sides - 0.6
    # plt.figure(figsize=(14, 5))
    # plt.plot(x[0], f[0], 'k', label='Original Function', lw=2.2)
    # plt.plot(x_extended_one_side, f_extend_one_side[0] , 'b',label='One-sided Extension', lw=2.2)
    # plt.plot(x_extended_both_sides, f_extend_both_sides[0] , 'g', label='Two-sided Extension', lw=2.2)
    # plt.plot([0, 0], [-2.7, 1.8], '-', color='gray', lw=1.5)  
    # plt.plot([1, 1], [-2.7, 1.8], '-', color='gray', lw=1.5)  
    # plt.plot([0, 1.4], [f_extend_one_side[0,0],f_extend_one_side[0,0]], '--', color='b', lw=1.4)  
    # plt.plot([-0.2, 1.2], [f_extend_both_sides[0,0],f_extend_both_sides[0,0]], '--', color='g', lw=1.4) 
    # # Create legend with FC-Legendre at the top
    # from matplotlib.lines import Line2D
    # legend_elements = [
    #     Line2D([0], [0], color='none', label='FC-Legendre'),
    #     Line2D([0], [0], color='k', lw=2.2, label='Original Function'),
    #     Line2D([0], [0], color='b', lw=2.2, label='One-sided Extension'),
    #     Line2D([0], [0], color='g', lw=2.2, label='Two-sided Extension')
    # ]
    # legend = plt.legend(handles=legend_elements, fontsize=19)
    # # Make the first legend entry bold
    # legend.get_texts()[0].set_fontweight('bold')
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.tick_params(axis='x', which='major', labelsize=19)
    # ax.tick_params(axis='y', which='major', labelsize=19)
    # plt.xticks([-0.2,0,1,1.2, 1.4])
    # plt.yticks([-2,2])
    # plt.xlim([-0.24, 1.52])
    # plt.ylim([-2.7, 2.5])
    # plt.tight_layout()
    # plt.savefig('1d_FCLegendre_examples.pdf', dpi=500, bbox_inches='tight')


def test_Legendre_fourier_continuation_2d():
    
    batch_size = 3
    length_signal = 101
    add_pts = 40

    x = torch.linspace(0, 1, length_signal).view(1, length_signal, 1).repeat(batch_size, 1, length_signal)
    y = torch.linspace(0, 1, length_signal).view(1, 1, length_signal).repeat(batch_size, length_signal, 1)
    f = torch.sin(12 * x)  - torch.cos(14 * y) + 3*x*y

    Extension = FCLegendre(d=3, n_additional_pts=add_pts)
    f_extend_one_side = Extension(f, dim=2, one_sided=True)
    f_extend_both_sides = Extension(f, dim=2, one_sided=False)

    # Check shape
    assert f_extend_one_side.shape[-1] == f.shape[-1] + add_pts
    assert f_extend_one_side.shape[-2] == f.shape[-2] + add_pts
    assert f_extend_both_sides.shape[-1] == f.shape[-1] + add_pts
    assert f_extend_both_sides.shape[-2] == f.shape[-2] + add_pts
    # Check values of original signal
    torch.testing.assert_close(f, f_extend_both_sides[...,add_pts//2:-add_pts//2, add_pts//2:-add_pts//2])
    torch.testing.assert_close(f, f_extend_one_side[...,:length_signal, :length_signal])

    # # Plot to verify
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(figsize=(14,6), nrows=1, ncols=3)
    # axs[0].imshow(f[0])
    # axs[0].set_title(r"Original Function", fontsize=15.5)
    # axs[1].imshow(f_extend_one_side[0])
    # axs[1].plot([length_signal, length_signal], [0, length_signal], '-', color='k', lw=3)
    # axs[1].plot([0, length_signal], [length_signal, length_signal], '-', color='k', lw=3)
    # axs[1].set_title(r"FC-Legendre One-sided Extension", fontsize=15.5)
    # axs[2].imshow(f_extend_both_sides[0])
    # axs[2].set_title(r"FC-Legendre Two-sided Extension", fontsize=15.5)
    # axs[2].plot([add_pts//2, length_signal + add_pts//2], [add_pts//2, add_pts//2], '-', color='k', lw=3)
    # axs[2].plot([add_pts//2, add_pts//2], [add_pts//2, length_signal + add_pts//2], '-', color='k', lw=3)
    # axs[2].plot([add_pts//2, length_signal + add_pts//2], [length_signal + add_pts//2, length_signal + add_pts//2], '-', color='k', lw=3)
    # axs[2].plot([length_signal + add_pts//2, length_signal + add_pts//2], [add_pts//2, length_signal + add_pts//2], '-', color='k', lw=3)
    # for ax in axs.flat:
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    # plt.savefig('2d_FCLegendre_examples.pdf', dpi=500, bbox_inches='tight')


def test_Legendre_fourier_continuation_3d():
    
    batch_size = 2
    length_signal = 51
    add_pts = 20

    # Create 3D grid
    x = torch.linspace(0, 1, length_signal).view(1, length_signal, 1, 1).repeat(batch_size, 1, length_signal, length_signal)
    y = torch.linspace(0, 1, length_signal).view(1, 1, length_signal, 1).repeat(batch_size, length_signal, 1, length_signal)
    z = torch.linspace(0, 1, length_signal).view(1, 1, 1, length_signal).repeat(batch_size, length_signal, length_signal, 1)

    # Create 3D function
    f = torch.exp(-2*z) + 2*z*x + torch.sin(12*x*y) + y*torch.sin(10*y*z) 

    Extension = FCLegendre(d=3, n_additional_pts=add_pts)
    f_extend_one_side = Extension(f, dim=3, one_sided=True)
    f_extend_both_sides = Extension(f, dim=3, one_sided=False)

    # Check shape
    assert f_extend_one_side.shape[-1] == f.shape[-1] + add_pts
    assert f_extend_one_side.shape[-2] == f.shape[-2] + add_pts
    assert f_extend_one_side.shape[-3] == f.shape[-3] + add_pts
    assert f_extend_both_sides.shape[-1] == f.shape[-1] + add_pts
    assert f_extend_both_sides.shape[-2] == f.shape[-2] + add_pts
    assert f_extend_both_sides.shape[-3] == f.shape[-3] + add_pts

    # # Create 3D visualizations
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # import numpy as np

    # fig = plt.figure(figsize=(20, 15))

    # f_min = f.min().item()
    # f_max = f.max().item()
    # f_ext1_min = f_extend_one_side.min().item()
    # f_ext1_max = f_extend_one_side.max().item()
    # f_ext2_min = f_extend_both_sides.min().item()
    # f_ext2_max = f_extend_both_sides.max().item()
    # global_min = min(f_min, f_ext1_min, f_ext2_min)
    # global_max = max(f_max, f_ext1_max, f_ext2_max)

    # # Figure for X slices
    # slice_indices = [length_signal//4, length_signal//2, 3*length_signal//4]
    # slice_names = ['First Quarter', 'Middle', 'Third Quarter']

    # for i, (idx, name) in enumerate(zip(slice_indices, slice_names)):
    #     ax = fig.add_subplot(3, 3, i+1)
    #     im = ax.imshow(f[0, idx, :, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'Original: X-slice {name}', fontsize=12)
    #     ax.set_xlabel('Y')
    #     ax.set_ylabel('Z')
    #     plt.colorbar(im, ax=ax)
        
    #     # One-sided extension - X-slice
    #     ax = fig.add_subplot(3, 3, i+4)
    #     im = ax.imshow(f_extend_one_side[0, idx, :, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'One-sided: X-slice {name}', fontsize=12)
    #     ax.set_xlabel('Y')
    #     ax.set_ylabel('Z')
    #     # Draw boundary lines
    #     ax.axhline(y=length_signal, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=length_signal, color='white', linewidth=2, linestyle='-')
    #     plt.colorbar(im, ax=ax)
        
    #     # Two-sided extension - X-slice
    #     ax = fig.add_subplot(3, 3, i+7)
    #     ext_idx = idx + add_pts//2
    #     im = ax.imshow(f_extend_both_sides[0, ext_idx, :, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'Two-sided: X-slice {name}', fontsize=12)
    #     ax.set_xlabel('Y')
    #     ax.set_ylabel('Z')
    #     # Draw boundary lines
    #     ax.axhline(y=add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axhline(y=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    #     plt.colorbar(im, ax=ax)

    # plt.tight_layout()
    # plt.savefig('3d_FCLegendre_x_slices.pdf', dpi=500, bbox_inches='tight')

    # # Figure for Y-slices
    # fig2 = plt.figure(figsize=(20, 15))

    # for i, (idx, name) in enumerate(zip(slice_indices, slice_names)):
    #     # Original function - Y-slice
    #     ax = fig2.add_subplot(3, 3, i+1)
    #     im = ax.imshow(f[0, :, idx, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'Original: Y-slice {name}', fontsize=12)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Z')
    #     plt.colorbar(im, ax=ax)
        
    #     # One-sided extension - Y-slice
    #     ax = fig2.add_subplot(3, 3, i+4)
    #     im = ax.imshow(f_extend_one_side[0, :, idx, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'One-sided: Y-slice {name}', fontsize=12)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Z')
    #     # Draw boundary lines
    #     ax.axhline(y=length_signal, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=length_signal, color='white', linewidth=2, linestyle='-')
    #     plt.colorbar(im, ax=ax)
        
    #     # Two-sided extension - Y-slice
    #     ax = fig2.add_subplot(3, 3, i+7)
    #     ext_idx = idx + add_pts//2
    #     im = ax.imshow(f_extend_both_sides[0, :, ext_idx, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'Two-sided: Y-slice {name}', fontsize=12)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Z')
    #     # Draw boundary lines
    #     ax.axhline(y=add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axhline(y=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    #     plt.colorbar(im, ax=ax)

    # plt.tight_layout()
    # plt.savefig('3d_FCLegendre_y_slices.pdf', dpi=500, bbox_inches='tight')

    # # Figure for Z-slices
    # fig3 = plt.figure(figsize=(20, 15))

    # for i, (idx, name) in enumerate(zip(slice_indices, slice_names)):
    #     # Original function - Z-slice
    #     ax = fig3.add_subplot(3, 3, i+1)
    #     im = ax.imshow(f[0, :, :, idx].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'Original: Z-slice {name}', fontsize=12)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     plt.colorbar(im, ax=ax)
        
    #     # One-sided extension - Z-slice
    #     ax = fig3.add_subplot(3, 3, i+4)
    #     im = ax.imshow(f_extend_one_side[0, :, :, idx].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'One-sided: Z-slice {name}', fontsize=12)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     # Draw boundary lines
    #     ax.axhline(y=length_signal, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=length_signal, color='white', linewidth=2, linestyle='-')
    #     plt.colorbar(im, ax=ax)
        
    #     # Two-sided extension - Z-slice
    #     ax = fig3.add_subplot(3, 3, i+7)
    #     ext_idx = idx + add_pts//2
    #     im = ax.imshow(f_extend_both_sides[0, :, :, ext_idx].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'Two-sided: Z-slice {name}', fontsize=12)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     # Draw boundary lines
    #     ax.axhline(y=add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axhline(y=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    #     plt.colorbar(im, ax=ax)

    # plt.tight_layout()
    # plt.savefig('3d_FCLegendre_z_slices.pdf', dpi=500, bbox_inches='tight')
    
    
def test_FCGram_fourier_continuation_1d():
    
    batch_size = 3
    length_signal = 101
    add_pts = 40

    x = torch.linspace(0, 1, length_signal, dtype=torch.float64).repeat(batch_size,1)
    f = torch.sin(16 * x) - torch.cos(8 * x)

    Extension = FCGram(d=3, n_additional_pts=add_pts, matrices_path='./FCGram_matrices')
    f_extend_one_side = Extension(f, dim=1, one_sided=True)
    f_extend_both_sides = Extension(f, dim=1, one_sided=False)

    # Check shape
    assert f_extend_both_sides.shape[-1] == f_extend_one_side.shape[-1] == f.shape[-1] + add_pts
    # Check values of original signal
    torch.testing.assert_close(f, f_extend_one_side[...,:length_signal])
    torch.testing.assert_close(f, f_extend_both_sides[...,add_pts//2:-add_pts//2])

    # # Plot to verify
    # import matplotlib.pyplot as plt
    # x_extended_one_side = torch.linspace(0, 1.4, 141, dtype=torch.float64) 
    # x_extended_both_sides = torch.linspace(-0.2, 1.2, 141, dtype=torch.float64) 
    # # Add 0.5 and -0.5 for plotting purposes
    # f_extend_one_side = f_extend_one_side + 0.6
    # f_extend_both_sides = f_extend_both_sides - 0.6
    # plt.figure(figsize=(14, 5))
    # plt.plot(x[0], f[0], 'k', label='Original Function', lw=2.2)
    # plt.plot(x_extended_one_side, f_extend_one_side[0] , 'b',label='One-sided Extension', lw=2.2)
    # plt.plot(x_extended_both_sides, f_extend_both_sides[0] , 'g', label='Two-sided Extension', lw=2.2)
    # plt.plot([0, 0], [-2.5, 1.9], '-', color='gray', lw=1.5)  
    # plt.plot([1, 1], [-2.5, 1.9], '-', color='gray', lw=1.5)  
    # plt.plot([0, 1.4], [f_extend_one_side[0,0],f_extend_one_side[0,0]], '--', color='b', lw=1.4)  
    # plt.plot([-0.2, 1.2], [f_extend_both_sides[0,0],f_extend_both_sides[0,0]], '--', color='g', lw=1.4) 
    # # Create legend with FC-Gram at the top
    # from matplotlib.lines import Line2D
    # legend_elements = [
    #     Line2D([0], [0], color='none', label='FC-Gram'),
    #     Line2D([0], [0], color='k', lw=2.2, label='Original Function'),
    #     Line2D([0], [0], color='b', lw=2.2, label='One-sided Extension'),
    #     Line2D([0], [0], color='g', lw=2.2, label='Two-sided Extension')
    # ]
    # legend = plt.legend(handles=legend_elements, fontsize=19)
    # # Make the first legend entry bold
    # legend.get_texts()[0].set_fontweight('bold')
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.tick_params(axis='x', which='major', labelsize=19)
    # ax.tick_params(axis='y', which='major', labelsize=19)
    # plt.xticks([-0.2,0,1,1.2, 1.4])
    # plt.yticks([-2,2])
    # plt.xlim([-0.24, 1.52])
    # plt.ylim([-2.5, 2.9])
    # plt.tight_layout()
    # plt.savefig('1d_FCGram_examples.pdf', dpi=500, bbox_inches='tight')
    
    
def test_FCGram_fourier_continuation_2d():
    
    batch_size = 3
    length_signal = 101
    add_pts = 40

    x = torch.linspace(0, 1, length_signal, dtype=torch.float64).view(1, length_signal, 1).repeat(batch_size, 1, length_signal)
    y = torch.linspace(0, 1, length_signal, dtype=torch.float64).view(1, 1, length_signal).repeat(batch_size, length_signal, 1)
    f = torch.sin(12 * x)  - torch.cos(14 * y) + 3*x*y

    Extension = FCGram(d=3, n_additional_pts=add_pts, matrices_path='./FCGram_matrices')
    f_extend_one_side = Extension(f, dim=2, one_sided=True)
    f_extend_both_sides = Extension(f, dim=2, one_sided=False)

    # Check shape
    assert f_extend_one_side.shape[-1] == f.shape[-1] + add_pts
    assert f_extend_one_side.shape[-2] == f.shape[-2] + add_pts
    assert f_extend_both_sides.shape[-1] == f.shape[-1] + add_pts
    assert f_extend_both_sides.shape[-2] == f.shape[-2] + add_pts
    # Check values of original signal
    torch.testing.assert_close(f, f_extend_both_sides[...,add_pts//2:-add_pts//2, add_pts//2:-add_pts//2])
    torch.testing.assert_close(f, f_extend_one_side[...,:length_signal, :length_signal])

    # # Plot to verify
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(figsize=(14,6), nrows=1, ncols=3)
    # axs[0].imshow(f[0])
    # axs[0].set_title(r"Original Function", fontsize=15.5)
    # axs[1].imshow(f_extend_one_side[0])
    # axs[1].plot([length_signal, length_signal], [0, length_signal], '-', color='k', lw=3)
    # axs[1].plot([0, length_signal], [length_signal, length_signal], '-', color='k', lw=3)
    # axs[1].set_title(r"FC-Gram One-sided Extension", fontsize=15.5)
    # axs[2].imshow(f_extend_both_sides[0])
    # axs[2].set_title(r"FC-Gram Two-sided Extension", fontsize=15.5)
    # axs[2].plot([add_pts, length_signal + add_pts], [add_pts, add_pts], '-', color='k', lw=3)
    # axs[2].plot([add_pts, add_pts], [add_pts, length_signal + add_pts], '-', color='k', lw=3)
    # axs[2].plot([add_pts, length_signal + add_pts], [length_signal + add_pts, length_signal + add_pts], '-', color='k', lw=3)
    # axs[2].plot([length_signal + add_pts, length_signal + add_pts], [add_pts, length_signal + add_pts], '-', color='k', lw=3)
    # for ax in axs.flat:
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    # plt.savefig('2d_FCGram_examples.pdf', dpi=500, bbox_inches='tight')
    
    
def test_FCGram_fourier_continuation_3d():
    
    batch_size = 2
    length_signal = 51 
    add_pts = 20

    # Create 3D grid
    x = torch.linspace(0, 1, length_signal, dtype=torch.float64).view(1, length_signal, 1, 1).repeat(batch_size, 1, length_signal, length_signal)
    y = torch.linspace(0, 1, length_signal, dtype=torch.float64).view(1, 1, length_signal, 1).repeat(batch_size, length_signal, 1, length_signal)
    z = torch.linspace(0, 1, length_signal, dtype=torch.float64).view(1, 1, 1, length_signal).repeat(batch_size, length_signal, length_signal, 1)

    # Create 3D function
    f = torch.exp(-2*z) + 2*z*x + torch.sin(12*x*y) + y*torch.sin(10*y*z)

    Extension = FCGram(d=3, n_additional_pts=add_pts, matrices_path='./FCGram_matrices')
    f_extend_one_side = Extension(f, dim=3, one_sided=True)
    f_extend_both_sides = Extension(f, dim=3, one_sided=False)

    # Check shape
    assert f_extend_one_side.shape[-1] == f.shape[-1] + add_pts
    assert f_extend_one_side.shape[-2] == f.shape[-2] + add_pts
    assert f_extend_one_side.shape[-3] == f.shape[-3] + add_pts
    assert f_extend_both_sides.shape[-1] == f.shape[-1] + add_pts
    assert f_extend_both_sides.shape[-2] == f.shape[-2] + add_pts
    assert f_extend_both_sides.shape[-3] == f.shape[-3] + add_pts

    # # Create 3D visualizations
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # import numpy as np

    # fig = plt.figure(figsize=(20, 15))

    # f_min = f.min().item()
    # f_max = f.max().item()
    # f_ext1_min = f_extend_one_side.min().item()
    # f_ext1_max = f_extend_one_side.max().item()
    # f_ext2_min = f_extend_both_sides.min().item()
    # f_ext2_max = f_extend_both_sides.max().item()

    # global_min = min(f_min, f_ext1_min, f_ext2_min)
    # global_max = max(f_max, f_ext1_max, f_ext2_max)

    # # Figure for X slices
    # slice_indices = [length_signal//4, length_signal//2, 3*length_signal//4]
    # slice_names = ['First Quarter', 'Middle', 'Third Quarter']

    # for i, (idx, name) in enumerate(zip(slice_indices, slice_names)):
    #     ax = fig.add_subplot(3, 3, i+1)
    #     im = ax.imshow(f[0, idx, :, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'Original: X-slice {name}', fontsize=12)
    #     ax.set_xlabel('Y')
    #     ax.set_ylabel('Z')
    #     plt.colorbar(im, ax=ax)
        
    #     ax = fig.add_subplot(3, 3, i+4)
    #     im = ax.imshow(f_extend_one_side[0, idx, :, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'One-sided: X-slice {name}', fontsize=12)
    #     ax.set_xlabel('Y')
    #     ax.set_ylabel('Z')
    #     # Draw boundary lines
    #     ax.axhline(y=length_signal, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=length_signal, color='white', linewidth=2, linestyle='-')
    #     plt.colorbar(im, ax=ax)
        
    #     ax = fig.add_subplot(3, 3, i+7)
    #     ext_idx = idx + add_pts//2
    #     im = ax.imshow(f_extend_both_sides[0, ext_idx, :, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'Two-sided: X-slice {name}', fontsize=12)
    #     ax.set_xlabel('Y')
    #     ax.set_ylabel('Z')
    #     # Draw boundary lines
    #     ax.axhline(y=add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axhline(y=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    #     plt.colorbar(im, ax=ax)

    # plt.tight_layout()
    # plt.savefig('3d_FCGram_x_slices.pdf', dpi=500, bbox_inches='tight')

    # # Figure for Y-slices
    # fig2 = plt.figure(figsize=(20, 15))

    # for i, (idx, name) in enumerate(zip(slice_indices, slice_names)):
    #     ax = fig2.add_subplot(3, 3, i+1)
    #     im = ax.imshow(f[0, :, idx, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'Original: Y-slice {name}', fontsize=12)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Z')
    #     plt.colorbar(im, ax=ax)
        
    #     ax = fig2.add_subplot(3, 3, i+4)
    #     im = ax.imshow(f_extend_one_side[0, :, idx, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'One-sided: Y-slice {name}', fontsize=12)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Z')
    #     # Draw boundary lines
    #     ax.axhline(y=length_signal, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=length_signal, color='white', linewidth=2, linestyle='-')
    #     plt.colorbar(im, ax=ax)
        
    #     ax = fig2.add_subplot(3, 3, i+7)
    #     ext_idx = idx + add_pts//2
    #     im = ax.imshow(f_extend_both_sides[0, :, ext_idx, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'Two-sided: Y-slice {name}', fontsize=12)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Z')
    #     # Draw boundary lines
    #     ax.axhline(y=add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axhline(y=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    #     plt.colorbar(im, ax=ax)

    # plt.tight_layout()
    # plt.savefig('3d_FCGram_y_slices.pdf', dpi=500, bbox_inches='tight')

    # # Figure for Z-slices
    # fig3 = plt.figure(figsize=(20, 15))

    # for i, (idx, name) in enumerate(zip(slice_indices, slice_names)):
    #     ax = fig3.add_subplot(3, 3, i+1)
    #     im = ax.imshow(f[0, :, :, idx].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'Original: Z-slice {name}', fontsize=12)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     plt.colorbar(im, ax=ax)
        
    #     ax = fig3.add_subplot(3, 3, i+4)
    #     im = ax.imshow(f_extend_one_side[0, :, :, idx].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'One-sided: Z-slice {name}', fontsize=12)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     # Draw boundary lines
    #     ax.axhline(y=length_signal, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=length_signal, color='white', linewidth=2, linestyle='-')
    #     plt.colorbar(im, ax=ax)
        
    #     ax = fig3.add_subplot(3, 3, i+7)
    #     ext_idx = idx + add_pts//2
    #     im = ax.imshow(f_extend_both_sides[0, :, :, ext_idx].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    #     ax.set_title(f'Two-sided: Z-slice {name}', fontsize=12)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     # Draw boundary lines
    #     ax.axhline(y=add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axhline(y=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=add_pts//2, color='white', linewidth=2, linestyle='-')
    #     ax.axvline(x=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    #     plt.colorbar(im, ax=ax)

    # plt.tight_layout()
    # plt.savefig('3d_FCGram_z_slices.pdf', dpi=500, bbox_inches='tight')