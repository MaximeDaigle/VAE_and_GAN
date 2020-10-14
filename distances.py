import torch
import sampler
from torch import nn


class Critic(nn.Module):
    def __init__(self, in_size=2):
        super(Critic, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, input):
        output = self.main(input)
        return output.squeeze()

def lp_reg(x, y, critic):
    """
    The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. The norm used is the L2 norm.

    :param x: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    : (FloatTensor) - shape: (1,) - Lipschitz penalty
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = torch.empty(x.shape).uniform_(0, 1).to(device) # [0,1] shape=(batchsize, 1)
    x_hat = (t * x + (1-t)*y)
    # if not training:
    x_hat.requires_grad = True
    critic = critic(x_hat)
    grad = torch.autograd.grad(critic, x_hat, torch.ones(critic.shape).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]
    norm = torch.sqrt(torch.sum(grad**2, dim=1))
    return torch.mean(torch.nn.functional.relu(norm-1)**2)

def vf_wasserstein_distance(x, y, critic):
    """
    The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper.

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """
    distance = critic(x).mean() - critic(y).mean()
    return distance


def vf_squared_hellinger(x, y, critic):
    """
    The notation used for the parameters follow the one from Nowazin et al: https://arxiv.org/pdf/1606.00709.pdf
    In other word, x are samples from the distribution P and y are samples from the distribution Q. Please note that the Critic is unbounded.

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Squared Hellinger.
    :return: (FloatTensor) - shape: (1,) - Estimate of the Squared Hellinger
    """
    gf_p = torch.tensor([1.]) - torch.exp(-critic(x))
    gf_q = torch.tensor([1.]) - torch.exp(-critic(y))
    return torch.mean(gf_p) + torch.mean(-(gf_q/(1-gf_q)))


if __name__ == '__main__':
    model = Critic(2)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    sampler1 = iter(sampler.distribution1(0, 512))
    theta = 0

    sampler2 = iter(sampler.distribution1(theta, 512))
    lambda_reg_lp = 50  # Recommended hyper parameters for the lipschitz regularizer.
