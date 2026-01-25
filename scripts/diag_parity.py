import random
import sys
from pathlib import Path
import numpy as np
import torch
# Ensure backend package importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backend.src.differentiable_integrator as di
import runtime_core as rc
res=32
xs=np.linspace(-1.5,1.5,res)
ys=np.linspace(-1.5,1.5,res)
X,Y=np.meshgrid(xs,ys)
R=np.sqrt(X**2+Y**2)
field=np.clip(1.0 - R / R.max(), 0.0, 1.0).astype(np.float32)
flat=list(field.ravel())
rc_df=rc.DistanceField(flat,res,(-1.5,1.5),(-1.5,1.5),1.0,0.05)
td_field=torch.from_numpy(field)
tdf=di.TorchDistanceField(td_field,real_min=-1.5,real_max=1.5,imag_min=-1.5,imag_max=1.5,max_distance=1.0,slowdown_threshold=0.05,use_runtime_sampler=True)
# pick failing sample from test seed sequence
rng=random.Random(123)
for _ in range(5):
    real=rng.uniform(-1.5,1.5); imag=rng.uniform(-1.5,1.5); u_real=rng.uniform(-0.05,0.05); u_imag=rng.uniform(-0.05,0.05); h=rng.choice([0.0,1.0])
# run one
print('sample:',real,imag,u_real,u_imag,h)
out_py=rc.contour_biased_step(real,imag,u_real,u_imag,h,0.5,0.05,rc_df)
print('rc out:', out_py.real, out_py.imag)
# now compute using torch path and print intermediates
c_r=torch.tensor([real],dtype=torch.float32); c_i=torch.tensor([imag],dtype=torch.float32); u_r=torch.tensor([u_real],dtype=torch.float32); u_i=torch.tensor([u_imag],dtype=torch.float32); ht=torch.tensor([h],dtype=torch.float32)
# d and grad via runtime
print('rc sample d, grad at c:')
print('d:', rc_df.sample_bilinear(rc.Complex(real,imag)))
print('grad:', rc_df.gradient(real,imag))
# torch path
d_t = tdf.sample_bilinear(c_r,c_i)
gx,gy = tdf.gradient(c_r,c_i)
print('torch d:', float(d_t.item()))
print('torch grad:', float(gx.item()), float(gy.item()))
nr,ni=di.contour_biased_step_torch(c_r,c_i,u_r,u_i,ht,0.5,0.05,tdf)
print('torch out:', float(nr.item()), float(ni.item()))
