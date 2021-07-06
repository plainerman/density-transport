from sklearn.preprocessing import StandardScaler
import pyro.distributions as dist
import torch
import numpy as np


def train(transform, base_dist, data, steps, lr, normalize):
    mean = np.mean(data)
    s = np.sqrt(np.var(data))

    if normalize:
        # normalize with (x-u)/s
        #data = (data - mean) / s
        flow_dist_bimodal = dist.TransformedDistribution(base_dist, [transform, dist.transforms.AffineTransform(mean, s)])
    else:
        flow_dist_bimodal = dist.TransformedDistribution(base_dist, [transform])

    dataset = torch.tensor(data, dtype=torch.float)
    optimizer = torch.optim.Adam(transform.parameters(), lr=lr)
    for step in range(steps):
        optimizer.zero_grad()
        loss = -flow_dist_bimodal.log_prob(dataset).mean()
        loss.backward()
        optimizer.step()
        flow_dist_bimodal.clear_cache()

        if step % 200 == 0:
            print('step: {}, loss: {}'.format(step, loss.item()))
    if normalize:
        return flow_dist_bimodal, dist.transforms.ComposeTransform([transform, dist.transforms.AffineTransform(mean, s)])
    return flow_dist_bimodal, transform


def normal_to_samples(A, normalize=False, count_bins=32, steps=1001, lr=1e-2):
    #A_normalized = StandardScaler().fit_transform(A) if normalize else A
    base_dist = dist.Normal(torch.zeros(1), torch.ones(1))

    return train(dist.transforms.Spline(1, count_bins=count_bins), base_dist, A, steps, lr, normalize)


def samples_to_samples(A, B, count_bins=32, steps=1001, lr=1e-2):
    '''
    This function takes two list of samples and maps them with normalizing flows via a normal distribution
    :return:
    '''

    normalize = False

    #A_normalized = StandardScaler().fit_transform(A) if normalize else A
    #B_normalized = StandardScaler().fit_transform(B) if normalize else B

    base_dist = dist.Normal(torch.zeros(1), torch.ones(1))

    _, transform_normal_A = train(dist.transforms.Spline(1, count_bins=count_bins), base_dist, A, steps, lr, normalize)
    _, transform_normal_B = train(dist.transforms.Spline(1, count_bins=count_bins), base_dist, B, steps, lr, normalize)

    return dist.transforms.ComposeTransform([transform_normal_A.inv, transform_normal_B])

def main():
    dist_x1 = dist.Normal(torch.tensor([-3.0]), torch.tensor([2.0]))
    dist_x2 = dist.Normal(torch.tensor([3.0]), torch.tensor([1.0]))

    X_bimodal = np.concatenate([dist_x1.sample([10000]).numpy(), dist_x2.sample([10000]).numpy()])

    dist_x1 = dist.Normal(torch.tensor([-4.0]), torch.tensor([2.0]))
    dist_x2 = dist.Normal(torch.tensor([4.0]), torch.tensor([1.0]))
    dist_x3 = dist.Normal(torch.tensor([0.5]), torch.tensor([0.5]))

    X_trimodal = np.concatenate(
        [dist_x1.sample([10000]).numpy(), dist_x2.sample([10000]).numpy(), dist_x3.sample([10000]).numpy()])

    transform = samples_to_samples(X_bimodal, X_trimodal)

    print(transform)


if __name__ == "__main__":
    main()
