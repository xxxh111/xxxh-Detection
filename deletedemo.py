import os
import torch
from statistics import  mean
#[[tensor(2., device='cuda:0'), tensor(112., device='cuda:0'), tensor(94., device='cuda:0'), tensor(200., device='cuda:0')], [tensor(202., device='cuda:0'), tensor(101., device='cuda:0'), tensor(277., device='cuda:0'), tensor(180., device='cuda:0')], [tensor(355., device='cuda:0'), tensor(115., device='cuda:0'), tensor(433., device='cuda:0'), tensor(179., device='cuda:0')]]230727103736c001070012.jpg")
tensors=[]
tensr1=[]
tensr1.append(torch.tensor(2))
tensr1.append(torch.tensor(112))
tensr1.append(torch.tensor(94))
tensr1.append(torch.tensor(200))
tensors.append(tensr1)
tensr2=[]
tensr2.append(torch.tensor(202))
tensr2.append(torch.tensor(101))
tensr2.append(torch.tensor(277))
tensr2.append(torch.tensor(180))
tensors.append(tensr2)
tensr3=[]
tensr3.append(torch.tensor(355))
tensr3.append(torch.tensor(115))
tensr3.append(torch.tensor(433))
tensr3.append(torch.tensor(179))
tensors.append(tensr2)
tensors.append(tensr3)
aa=[]
a=torch.zeros([1])
c=a.float()  
a=torch.add(a,torch.tensor(3))
aa.extend((torch.tensor(5),torch.tensor(3),torch.tensor(1)))
c=mean(aa)
dd=[32,3,4]
sorrde=sorted(dd,key=lambda  x: aa[dd.index(x)])
print(sorrde)