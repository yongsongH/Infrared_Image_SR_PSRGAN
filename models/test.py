import torch
t = torch.ones((2,3))
t_0 = torch.cat([t,t], dim=0)
t_1 = torch.cat([t,t], dim=1)
t = torch.cat()
print('t_0:{} shape:{}\nt_1:{} shape:{}'.format(t_0,t_0.shape,t_1,t_1.shape))