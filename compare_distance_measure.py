import torch
import numpy as np
import sampler
import distances
import time


thetas = np.linspace(0, 2, num=21)

for mode in ['w', 's']:
    print('\n\n\n', mode)
    for theta in thetas:
        #print('\n\n\n\n\n start ', mode, theta)
        model = distances.Critic(2)
        optim = torch.optim.SGD(model.parameters(), lr=1e-3)
        sampler_p = iter(sampler.distribution1(0, 512))
        sampler_q = iter(sampler.distribution1(theta, 512))
        lambda_reg_lp = 50  # Recommended hyper parameters for the lipschitz regularizer.
        start = time.time()
        for i in range(600):  # train
            x = torch.from_numpy(next(sampler_p)).float()
            y = torch.from_numpy(next(sampler_q)).float()
            optim.zero_grad()
            if mode == 'w':
                score = - distances.vf_wasserstein_distance(x, y, model) + lambda_reg_lp * distances.lp_reg(x, y, model)
            else:
                score = distances.vf_squared_hellinger(x, y, model)
            if mode == 's':
                score = -score #maximize
            score.backward()
            optim.step()
            #print(i, score.item())

        #print(time.time() - start)
        if mode == 'w':
            print(distances.vf_wasserstein_distance(x, y, model).item())
        else:
            print(distances.vf_squared_hellinger(x, y, model).item())
