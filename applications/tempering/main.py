'''
Inference time tempering with CREPE
'''

import torch
from energy.a4 import A4, A6, AldpBoltzmann
from network.egnn import EGNN_dynamics_AD2, EGNN_dynamics_AD4, EGNN_dynamics_AD6, remove_mean
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mdtraj
import argparse

def log_norm_prob(x, mu, std):
    # expand std to match x shape
    std = std.expand(x.shape)
    return -0.5 * ((x - mu) / std).pow(2).sum(-1) - (std.log() + np.log(2*np.pi)/2).sum(-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=int, default=1)                                                # whether to use reference process
    parser.add_argument('--fwd', type=float, default=0.0)                                            # forward process coef (0 in CREPE paper)
    parser.add_argument('--bwd', type=float, default=1.0)                                            # backward process coef (1 in CREPE paper)
    parser.add_argument('--device', type=str, default='cuda')                                        # device
    parser.add_argument('--init_temp', type=float, default=800)                                      # starting temperature
    parser.add_argument('--final_temp', type=float, default=500)                                     # final temperature
    parser.add_argument('--n_steps', type=int, default=201)                                          # number of steps
    parser.add_argument('--n_samples', type=int, default=50000)                                      # number of samples
    parser.add_argument('--target', type=str, default='a4', choices=['a4', 'a6', 'aldp'])            # target 
    parser.add_argument('--high_data_path', type=str, default='trajectory_1.0_800.0.h5')             # high temperature data path
    parser.add_argument('--low_data_path', type=str, default='trajectory_1.0_500.0.h5')              # low temperature data path, used only for visualisation
    parser.add_argument('--net_path', type=str, default='net_800k.pt')                       # ema net path

    args = parser.parse_args()
    args.ref = bool(args.ref)

    beta = args.init_temp / args.final_temp
    device = torch.device(args.device)

    if args.target == 'a4':
        n_particles = 43
        target = A4(500, 'cuda', scaling=1.0)
    elif args.target == 'a6':
        n_particles = 63
        target = A6(600, 'cuda', scaling=1.0)
    elif args.target == 'aldp':
        n_particles = 22
        target = AldpBoltzmann(300, 'cuda')
    else:
        raise ValueError(f"Unknown target: {args.target}")

    # load data (we also *5 to make training easier)
    data = torch.from_numpy(remove_mean(mdtraj.load(args.high_data_path).xyz, n_particles, 3)).to(device).reshape(-1, n_particles * 3) * 5
    data_low = torch.from_numpy(remove_mean(mdtraj.load(args.low_data_path).xyz, n_particles, 3)).to(device).reshape(-1, n_particles * 3) * 5

    if args.target == 'a4':
        ema_net = EGNN_dynamics_AD4(
            n_particles=n_particles, n_dimension=3, hidden_nf=256, device='cuda',
            act_fn=torch.nn.SiLU(), n_layers=5, recurrent=True, attention=True,
            condition_time=True, tanh=True, mode='egnn_dynamics', agg='sum', data_sigma=data.std().item()
        ).to(device).requires_grad_(False)
    elif args.target == 'a6':
        ema_net = EGNN_dynamics_AD6(
            n_particles=n_particles, n_dimension=3, hidden_nf=512, device='cuda',
            act_fn=torch.nn.SiLU(), n_layers=5, recurrent=True, attention=True,
            condition_time=True, tanh=True, mode='egnn_dynamics', agg='sum', data_sigma=data.std().item()
        ).to(device).requires_grad_(False)
    elif args.target == 'aldp':
        ema_net = EGNN_dynamics_AD2(
            n_particles=n_particles, n_dimension=3, hidden_nf=256, device='cuda',
            act_fn=torch.nn.SiLU(), n_layers=5, recurrent=True, attention=True,
            condition_time=True, tanh=True, mode='egnn_dynamics', agg='sum', data_sigma=data.std().item()
        ).to(device).requires_grad_(False)
    else:
        raise ValueError(f"Unknown target: {args.target}")

    ema_net.load_state_dict(torch.load(args.net_path, map_location=device))

    tmax = 10
    tmin = 1e-3
    rho = 7
    steps = args.n_steps
    ts = tmin ** (1 / rho) + np.arange(steps) / (steps - 1) * (tmax ** (1 / rho) - tmin ** (1 / rho))
    ts = ts ** rho

    gap = 4
    assert (steps-1) % gap == 0, "Steps must be divisible by gap + 1"
    all_levels = np.linspace(0, steps-1, (steps-1)//gap + 1)
    print('Levels:', all_levels)


        
    def EM_solve_temper(model, start_samples, beta, fwd, bwd, all_levels):
        with torch.no_grad():
            samples = start_samples
            Samples = [(ts[-1], len(ts)-1, start_samples.detach().cpu())]
            for i in tqdm(range(ts.shape[0]-1, 0, -1)):
                t = torch.ones(samples.shape[0], 1).to(samples.device) * ts[i]
                t_1 = torch.ones(samples.shape[0], 1).to(samples.device) * ts[i-1]
                Delta_t = (t - t_1).abs()
                x_hat = model(samples, t.squeeze(-1)) 
                std = torch.sqrt(2*Delta_t*t)
                score = - (samples - x_hat) / t ** 2 
                dx = score * 2 * t * Delta_t * beta  * bwd + std * remove_mean(torch.randn_like(samples), n_particles, 3)
                samples_new = samples + dx
                if i-1 in all_levels:
                    Samples.append((ts[i-1], i-1, samples_new.detach().cpu()))
                samples = samples_new
            return Samples

    # run initial samples
    start_samples = remove_mean(torch.randn(1, n_particles * 3, device=device) * ts[-1] / beta ** 0.5, n_particles, 3)
    Samples = EM_solve_temper(ema_net, start_samples, 1, 1, 1, all_levels)

    # calculate local move step size
    idx = np.arange(steps)[::-1]
    Delta_t = (ts[idx] - ts[np.clip(idx-1, 0, len(ts)-1)])
    Delta_t[-1] = Delta_t[-2] 
    step_size = Delta_t*ts[::-1]

    def APT_control(model, Samples, beta, step_size, fwd, bwd, gap, with_reference=True):
        # we run two steps of APT
        # in the first step, we only exchange the samples at even indices
        # in the second step, we exchange the samples at odd indices
        
        # in the current implementation, we call the network twice for each step, this can be optimised, but it is not a big deal
        # TODO: optimise this
        time_steps = torch.from_numpy(np.array([s[0] for s in Samples])).to(device).float()
        samples = torch.concat([s[2] for s in Samples], dim=0).to(device)
        idx = np.array([s[1] for s in Samples])

        ALL_SAMPLES = []
        start_idx = 0
        MASKS1 = []
        MASKS2 = []
        NEW_SAMPLES = []
        for start_idx in [0, 1]:
            with torch.no_grad():
                # local move 
                # for local move, fwd / bwd are not used
                x_hat = model(samples, time_steps.squeeze(-1)) 
                score = - (samples - x_hat) / time_steps.reshape(-1, 1) ** 2 * beta
                if score.isnan().any():
                    print('NaN detected in score, skipping step')
                    continue
                dx = score * step_size.reshape(-1, 1)  + (step_size*2)**0.5 * remove_mean(torch.randn_like(samples), n_particles, 3)
                samples = samples + dx 

                # replace the first sample with the start sample, do not forget anneal!!
                samples[0] = remove_mean(torch.randn(samples.shape[-1], device=device) * ts[-1], n_particles, 3) / beta ** 0.5

                # swap
                x1 = samples[start_idx:-1:2]
                x0 = samples[start_idx+1::2]
                idx_1 = idx[start_idx:-1:2]
                idx_0 = idx[start_idx+1::2]

                x1_original = x1.clone()
                x0_original = x0.clone()

                w = 0

    
                # x1 to x0
                for step in range(gap):
                    t_cur = torch.from_numpy(ts[idx_1 - step].reshape(-1, 1)).float().to(x1.device)
                    t_next = torch.from_numpy(ts[idx_1 - 1 - step].reshape(-1, 1)).float().to(x1.device)
                    Delta_t = (t_cur - t_next).abs()

                    x_hat = model(x1, t_cur.squeeze(-1)) 
                    std = torch.sqrt(2*Delta_t*t_cur)
                    score = - (x1 - x_hat) / t_cur ** 2 

                    dx = score * 2 * t_cur * Delta_t * beta * bwd + std * remove_mean(torch.randn_like(x1), n_particles, 3)
                    x0_candidate = x1 + dx

                    # target forward process with score * fwd
                    # note that enen though I implement here with additional call, this can be optimised, as the calling will be overlapped a lot with the next local move
                    x_hat = model(x0_candidate, t_next.squeeze(-1)) 
                    std = torch.sqrt(2*Delta_t*t_next)
                    score_x0_candidate = - (x0_candidate - x_hat) / t_next ** 2 

                    # diffusion forward process
                    fwd_mean = x0_candidate
                    fwd_std = torch.sqrt(2*Delta_t*t_next)
                    dm_fwd = log_norm_prob(x1, fwd_mean, fwd_std) 

                    # diffusion backward process
                    bwd_mean = x1 + score * 2 * t_cur * Delta_t
                    bwd_std = torch.sqrt(2*Delta_t*t_cur)
                    dm_bwd = log_norm_prob(x0_candidate, bwd_mean, bwd_std)


                    # sampling forward process
                    fwd_mean = x0_candidate + score_x0_candidate * 2 * t_next * Delta_t * beta * fwd
                    fwd_std = torch.sqrt(2*Delta_t*t_next)
                    sample_fwd = log_norm_prob(x1, fwd_mean, fwd_std) 

                    # sampling backward process
                    bwd_mean = x1 + score * 2 * t_cur * Delta_t * beta * bwd
                    bwd_std = torch.sqrt(2*Delta_t*t_cur)
                    sample_bwd = log_norm_prob(x0_candidate, bwd_mean, bwd_std)
                

                    if not with_reference:

                        # log weight
                        w += (sample_fwd - sample_bwd) + (dm_bwd - dm_fwd) * beta
                    
                    else:
                        # first define reference distribution
                        ref_std = lambda time: (1**2 + time ** 2)**0.5
                        ref_score = lambda x, time: - x / ref_std(time) ** 2
                        # ref forward process
                        fwd_mean = x0_candidate
                        fwd_std = torch.sqrt(2*Delta_t*t_next)
                        ref_fwd = log_norm_prob(x0_candidate, 0, ref_std(t_next)) + log_norm_prob(x1, fwd_mean, fwd_std) 

                        # ref backward process
                        bwd_mean = x1 + ref_score(x1, t_cur) * 2 * t_cur * Delta_t
                        bwd_std = torch.sqrt(2*Delta_t*t_cur)
                        ref_bwd = log_norm_prob(x1, 0, ref_std(t_cur)) + log_norm_prob(x0_candidate, bwd_mean, bwd_std)

                        # log weight
                        w += (sample_fwd - ref_fwd - sample_bwd + ref_bwd) + (dm_bwd - ref_bwd - dm_fwd + ref_fwd) * beta
                    x1 = x0_candidate.clone()
                        

                

                # x0 to x1
                for step in range(gap):
                    t_cur = torch.from_numpy(ts[idx_0 + step].reshape(-1, 1)).float().to(x0.device)
                    t_next = torch.from_numpy(ts[idx_0 + 1 + step].reshape(-1, 1)).float().to(x0.device)
                    Delta_t = (t_cur - t_next).abs()

                    # proposal forward process with score * fwd
                    # note that enen though I implement here with additional call, this can be optimised, as the calling will be overlapped a lot with the next local move
                    x_hat = model(x0, t_cur.squeeze(-1)) 
                    std = torch.sqrt(2*Delta_t*t_cur)
                    score_x0 = - (x0 - x_hat) / t_cur ** 2 
                    
                    std = torch.sqrt(2*Delta_t*t_cur)
                    dx = std * remove_mean(torch.randn_like(x0), n_particles, 3) + score_x0 * 2 * t_cur * Delta_t * beta * fwd
                    x1_candidate = x0 + dx



                    # diffusion forward process
                    fwd_mean = x0
                    fwd_std = torch.sqrt(2*Delta_t*t_cur)
                    dm_fwd = log_norm_prob(x1_candidate, fwd_mean, fwd_std) 

                    # diffusion backward process
                    x_hat = model(x1_candidate, t_next.squeeze(-1)) 
                    score = - (x1_candidate - x_hat) / t_next ** 2 
                    bwd_mean = x1_candidate + score * 2 * t_next * Delta_t
                    bwd_std = torch.sqrt(2*Delta_t*t_next)
                    dm_bwd = log_norm_prob(x0, bwd_mean, bwd_std)


                    # sampling forward process
                    fwd_mean = x0 + score_x0 * 2 * t_cur * Delta_t * beta * fwd
                    fwd_std = torch.sqrt(2*Delta_t*t_cur)
                    sample_fwd = log_norm_prob(x1_candidate, fwd_mean, fwd_std) 

                    # sampling backward process
                    bwd_mean = x1_candidate + score * 2 * t_next * Delta_t * beta  * bwd
                    bwd_std = torch.sqrt(2*Delta_t*t_next)
                    sample_bwd = log_norm_prob(x0, bwd_mean, bwd_std)
                

                    if not with_reference:

                        # log weight
                        w += (sample_bwd - sample_fwd) + (dm_fwd - dm_bwd) * beta
                    
                    else:
                        # first define reference distribution
                        ref_std = lambda time: (1**2 + time ** 2)**0.5
                        ref_score = lambda x, time: - x / ref_std(time) ** 2
                        # ref forward process
                        fwd_mean = x0
                        fwd_std = torch.sqrt(2*Delta_t*t_cur)
                        ref_fwd = log_norm_prob(x0, 0, ref_std(t_cur)) + log_norm_prob(x1_candidate, fwd_mean, fwd_std) 

                        # ref backward process
                        bwd_mean = x1_candidate + ref_score(x1_candidate, t_next) * 2 * t_next * Delta_t
                        bwd_std = torch.sqrt(2*Delta_t*t_next)
                        ref_bwd = log_norm_prob(x1_candidate, 0, ref_std(t_next)) + log_norm_prob(x0, bwd_mean, bwd_std)

                        # log weight
                        w += -(sample_fwd - ref_fwd - sample_bwd + ref_bwd) - (dm_bwd - ref_bwd - dm_fwd + ref_fwd) * beta
                    x0 = x1_candidate.clone()

                u = torch.rand_like(w).log()
                mask = (u < w).reshape(-1, 1)
                if start_idx == 0:
                    MASKS2.append(mask.reshape(-1))
                else:
                    MASKS1.append(mask.reshape(-1))
                samples[start_idx:-1:2] = torch.where(mask, x1_candidate, x1_original)
                samples[start_idx+1::2] = torch.where(mask, x0_candidate, x0_original)
                ALL_SAMPLES.append(samples.detach().cpu())


        for i in range(len(ALL_SAMPLES[-1])):
            NEW_SAMPLES.append(
                (time_steps[i].item(), idx[i].item(), ALL_SAMPLES[-1][i][None])
            )    
        return ALL_SAMPLES, MASKS1, MASKS2, NEW_SAMPLES

    # thin step sizes by taking the step sizes at the levels that we want to use
    used_step_sizes = step_size[::-1][all_levels.astype(int)][::-1].copy()
    used_step_sizes = torch.from_numpy(used_step_sizes).to(device).reshape(-1, 1).float()

    # run PT
    ALL_SAMPLES = []
    # MASK1 = []
    # MASK2 = [] # one can also check the swap rate (sanity check it is non 0 anywhere)

    for total_idx in tqdm(range(args.n_samples // 2)):
        all_samples, MASKS1, MASKS2, NEW_SAMPLES = APT_control(ema_net, 
                                                            Samples, 
                                                            beta, 
                                                            used_step_sizes, 
                                                            args.fwd, 
                                                            args.bwd, 
                                                            gap,
                                                            with_reference=True)
        ALL_SAMPLES += all_samples
        # MASK1 += MASKS1
        # MASK2 += MASKS2
        Samples = NEW_SAMPLES

        if (total_idx % 1000 == 0 and len(ALL_SAMPLES) > 0) or total_idx == args.n_samples // 2 - 1:
            samples = torch.stack(ALL_SAMPLES, 0)[:, -1]
            # thin
            samples = samples[np.random.choice(samples.shape[0], 1000)]
            log_s = target.log_prob(samples / 5).detach().cpu().numpy()
            log_d = target.log_prob(data_low[::100] / 5).detach().cpu().numpy()
            plt.figure(figsize=(3, 3))
            plt.hist(log_s, bins=100, density=True, alpha=0.5, label='samples')
            plt.hist(log_d, bins=100, density=True, alpha=0.5, label='data')
            plt.legend()
            plt.savefig(f'hist.png')
            plt.close()

            samples = torch.stack(ALL_SAMPLES, 0)[:, -1]
            np.save(f'crepe_fwd{args.fwd}_bwd{args.bwd}_target{args.target}_final_temp{args.final_temp}.npy', samples.cpu().numpy() / 5)

if __name__ == "__main__":
    main()