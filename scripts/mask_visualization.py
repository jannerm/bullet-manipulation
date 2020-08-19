import numpy as np
from scipy.stats import multivariate_normal
import matplotlib
import matplotlib.pyplot as plt

def plot_Gaussian(
        mu,
        sigma=None,
        sigma_inv=None,
        bounds=None,
        list_of_dims=[[0, 1], [2, 3], [0, 2], [1, 3]],
        pt1=None,
        pt2=None,
        add_title=True,
):
    num_subplots = len(list_of_dims)
    if num_subplots == 1:
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig, axs = plt.subplots(2, num_subplots // 2, figsize=(10, 10))
    lb, ub = bounds
    gran = (ub - lb) / 50
    x, y = np.mgrid[lb:ub:gran, lb:ub:gran]
    pos = np.dstack((x, y))

    assert (sigma is not None) ^ (sigma_inv is not None)
    if sigma is None:
        sigma = np.linalg.inv(sigma_inv + np.eye(len(mu)) * 1e-6)


    for i in range(len(list_of_dims)):
        dims = list_of_dims[i]
        rv = multivariate_normal(mu[dims], sigma[dims][:,dims], allow_singular=True)

        if num_subplots == 1:
            axs_obj = axs
        else:
            plt_idx1 = i // 2
            plt_idx2 = i % 2
            axs_obj = axs[plt_idx1, plt_idx2]

        axs_obj.contourf(x, y, rv.pdf(pos), cmap="Blues")
        if add_title:
            axs_obj.set_title(str(dims))

        if pt1 is not None:
            axs_obj.scatter([pt1[dims][0]], [pt1[dims][1]])

        if pt2 is not None:
            axs_obj.scatter([pt2[dims][0]], [pt2[dims][1]])

    return fig, axs


def save_fig(fig, axs, filename):
    plt.axis('off')
    axs.get_xaxis().set_visible(False)
    axs.get_yaxis().set_visible(False)
    axs.axhline(y=0, color='k')
    axs.axvline(x=0, color='k')
    axs.spines['left'].set_position('zero')
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_position('zero')
    axs.spines['top'].set_visible(False)
    axs.xaxis.set_ticks_position('bottom')
    axs.yaxis.set_ticks_position('left')
    axs.plot((1), (0), ls="", marker=">", ms=10, color="k",
             transform=axs.get_yaxis_transform(), clip_on=False)
    axs.plot((0), (1), ls="", marker="^", ms=10, color="k",
             transform=axs.get_xaxis_transform(), clip_on=False)
    plt.savefig(
        filename,
        bbox_inches='tight', pad_inches=0
    )


#filename = '/Users/sasha/Desktop/masks/masks.npy'
filename = '/Users/sasha/Desktop/masks/masks_x_equals_y.npy'
masks = np.load(filename, allow_pickle=True).item()

#mu, inv_sig = masks['mask_mu_w'], masks['mask_sigma_inv']
mu, inv_sig = masks['mask_mu'], masks['mask_sigma_inv']


# mu = np.zeros(2)
# sig = np.array([[1e9, 1e9-1], [1e9-1, 1e9]])

# mu = np.random.randint(-6,0)
# mu = np.array([mu, mu]) + np.random.normal(scale=1, size=(2,))

# sig = np.eye(2) + 0.1 * np.random.normal(size=(2,2))

# fig, axs = plot_Gaussian(
# 	mu=mu,
# 	sigma=sig,
# 	bounds=[-10, 10],
# 	list_of_dims=[[0, 1]],
# 	add_title=False,
# )
# save_fig(
# 	fig, axs,
# 	"/Users/sasha/Desktop/masks/true_dist.pdf",
# )

for i in range(mu.shape[0]):
	fig, axs = plot_Gaussian(
    	mu=mu[i],
    	sigma_inv=inv_sig[i],
    	bounds=[-3,3],
    	list_of_dims=[[2, 10]],
    	add_title=False,
	)
	save_fig(
    	fig, axs,
    	"/Users/sasha/Desktop/masks/stage_{0}.pdf".format(i),
	)
