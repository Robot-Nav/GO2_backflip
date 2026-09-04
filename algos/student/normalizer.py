import torch

class ObsRMS(object):
    def __init__(self, name:str, obs_dim:int, history_len:int, device:torch.device, max_cnt=None):
        self.name = name
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.max_cnt = max_cnt

        self.raw_obs_dim = int(self.obs_dim/self.history_len)
        self.mean = torch.zeros(self.raw_obs_dim, dtype=torch.float32, requires_grad=False, device=device)
        self.var = torch.ones(self.raw_obs_dim, dtype=torch.float32, requires_grad=False, device=device)
        self.count = 0

        self.cur_mean = torch.zeros_like(self.mean)
        self.cur_var = torch.ones_like(self.var)

    @torch.no_grad()
    def update(self, raw_data):
        if self.max_cnt is not None and self.count >= self.max_cnt:
            return

        data = raw_data[:, -self.raw_obs_dim:]
        count = data.shape[0]
        mean = data.mean(dim=0) # (raw_obs_dim,)
        var = data.var(dim=0, unbiased=False) # (raw_obs_dim,)
        delta_mean = mean - self.mean
        
        total_count = self.count + count
        m_a = self.var * self.count
        m_b = var * count
        M2 = m_a + m_b + delta_mean**2 * (self.count * count / total_count)
        
        self.mean += delta_mean * count / total_count
        self.var = M2 / total_count
        self.count += count

    @torch.no_grad()
    def normalize(self, observations, shifted_mean=0.0, shifted_std=1.0):
        reshaped_obs = observations.view(-1, self.obs_dim)
        reshaped_mean = self.cur_mean.view(1, -1).tile(1, self.history_len)
        reshaped_var = self.cur_var.view(1, -1).tile(1, self.history_len)
        norm_obs = (reshaped_obs - reshaped_mean)/torch.sqrt(reshaped_var + 1e-8)
        return norm_obs.view_as(observations)*shifted_std + shifted_mean
    
    def upgrade(self):
        self.cur_mean[:] = self.mean
        self.cur_var[:] = self.var
        return
    
    def state_dict(self):
        return {'mean': self.mean.cpu().numpy(), 'var': self.var.cpu().numpy(), 'count': self.count}

    def load_state_dict(self, d):
        self.mean[:] = torch.tensor(d['mean'], dtype=torch.float32, device=self.mean.device)
        self.var[:] = torch.tensor(d['var'], dtype=torch.float32, device=self.var.device)
        self.count = d['count']
        self.upgrade()
