import torch
import torchvision
from model import Critic, Generator
from torch import optim
import distances
import numpy as np
from itertools import repeat
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler

def svhn_sampler(root, train_batch_size, test_batch_size, valid_split=0):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    transform = transforms.Compose((
        transforms.ToTensor(),
        normalize))
    train = datasets.SVHN(root, split='train', download=True, transform=transform)
    valid = datasets.SVHN(root, split='train', download=True, transform=transform)
    test = datasets.SVHN(root, split='test', download=True, transform=transform)

    idxes = np.arange(len(train))
    split = int(valid_split * len(idxes))
    train_sampler = SubsetRandomSampler(idxes[split:])
    valid_sampler = SubsetRandomSampler(idxes[:split])
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               sampler=train_sampler, num_workers=4, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=train_batch_size,
                                               sampler=valid_sampler, num_workers=4, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, num_workers=4)

    return train_loader, valid_loader, test_loader,

def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

def validation():
    generator.eval()
    critic.eval()
    with torch.no_grad():
        # critic
        x = next(test_iter)[0].to(device)
        noise = torch.randn(train_batch_size, z_dim).to(device)
        y = generator(noise)
        score = (-distances.vf_wasserstein_distance(x, y, critic)) #+ (lp_coeff * distances.lp_reg(x, y, critic))
        log['valid_critic_loss'] += score

        if i % n_critic_updates == 0:
            #generator
            noise = torch.randn(train_batch_size, z_dim).to(device)
            x_generated = generator(noise)
            out = critic(x_generated)
            score = -torch.mean(out)
            log['valid_generator_loss'] += score


if __name__ == '__main__':
    data_root = './'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_iter = 50000 # N training iterations
    n_critic_updates = 5 # N critic updates per generator update
    lp_coeff = 10 # Lipschitz penalty coefficient
    train_batch_size = 64
    test_batch_size = 64
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 100
    log_interval = 1000
    save_interval = 10000
    log = {
        'critic_loss': 0.0,
        'generator_loss': 0.0,
        'penalty': 0.0,
        'valid_critic_loss': 0.0,
        'valid_generator_loss': 0.0
    }

    train_loader, valid_loader, test_loader = svhn_sampler(data_root, train_batch_size, test_batch_size)
    train_loader, valid_loader, test_loader = repeater(train_loader), repeater(valid_loader), repeater(test_loader)
    train_iter, valid_iter, test_iter = iter(train_loader), iter(valid_loader), iter(test_loader)

    generator = Generator(z_dim=z_dim).to(device)
    critic = Critic().to(device)

    optim_critic = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

    checkpoint = torch.load('save.tar')
    critic.load_state_dict(checkpoint['critic'])
    generator.load_state_dict(checkpoint['generator'])
    optim_critic.load_state_dict(checkpoint['optim_critic'])
    optim_generator.load_state_dict(checkpoint['optim_generator'])

    critic.train()
    generator.train()

    for i in range(n_iter*n_critic_updates):
        generator.train()
        critic.train()

        # update critic
        x = next(train_iter)[0].to(device)
        noise = torch.randn(train_batch_size, z_dim).to(device)
        y = generator(noise).detach()
        optim_critic.zero_grad()
        score = (-distances.vf_wasserstein_distance(x, y, critic))
        lp = (lp_coeff * distances.lp_reg(x, y, critic))
        log['critic_loss'] += score
        log['penalty'] += lp
        score += lp
        score.backward()
        optim_critic.step()

        if i % n_critic_updates == 0:
            # update generator
            optim_generator.zero_grad()
            noise = torch.randn(train_batch_size, z_dim).to(device)
            x_generated = generator(noise)
            out = critic(x_generated)
            score = -torch.mean(out)
            log['generator_loss'] += score
            score.backward()
            optim_generator.step()

        # Validation
        validation()
        iteration = i/n_critic_updates
        if iteration % log_interval == 0:
            nb_samples = log_interval * train_batch_size
            print(f'\nIteration {int(iteration)}\n            Avg Critic loss: {log["critic_loss"]/nb_samples}  Avg generator loss: {log["generator_loss"]/(nb_samples*n_critic_updates)}  Penalty: {log["penalty"]/nb_samples}')
            log['generator_loss'], log['critic_loss'], log['penalty'] = 0.0, 0.0, 0.0
            print(f'Validation: Avg Critic loss: {log["valid_critic_loss"] / nb_samples}  Avg generator loss: {log["valid_generator_loss"] / (nb_samples*n_critic_updates)}')
            log['valid_generator_loss'], log['valid_critic_loss'] = 0.0, 0.0
            
            # save image generated
            generator.eval()
            noise = torch.randn(train_batch_size, z_dim).to(device)
            y = generator(noise) 
            torchvision.utils.save_image(y, f"images-{int(iteration)}.png", normalize=True)
        
        if iteration % save_interval == 0:
                torch.save({
                    'critic': critic.state_dict(),
                    'generator': generator.state_dict(),
                    'optim_critic': optim_critic.state_dict(),
                    'optim_generator': optim_generator.state_dict()
                }, f'chkpt-{int(iteration)}.tar')

    # SAVE model
    torch.save({
        'critic': critic.state_dict(),
        'generator': generator.state_dict(),
        'optim_critic': optim_critic.state_dict(),
        'optim_generator': optim_generator.state_dict()
    }, 'save.tar')