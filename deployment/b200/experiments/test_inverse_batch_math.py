"""Discriminative CPU checks of real released per-item E/J batch math."""
import sys
import unittest
sys.path.insert(0,'/opt/fmrg/source')
import torch
from fluxfm_sampler_ip import FluxFlowMapSampler


class BatchMath(unittest.TestCase):
    def sampler(self):
        s=object.__new__(FluxFlowMapSampler)
        s.dtype=torch.float32;s.decode=lambda z:z
        s.predict_vector=lambda z,*a,**k:.2*z+.1
        return s

    def test_e_loss_is_sum_without_batch_averaging(self):
        s=self.sampler();z=torch.arange(16,dtype=torch.float32).reshape(4,1,2,2)/20
        op=type('Op',(),{'A':lambda _,x:2*x})()
        y=torch.linspace(-.2,.3,16).reshape_as(z)
        batch=s.fmrg_gradient_step_euclidean(z,op,y,'sr4',.05,5,'pixel','sum',False)[0]
        singles=torch.cat([s.fmrg_gradient_step_euclidean(z[i:i+1],op,y[i:i+1],'sr4',.05,5,'pixel','sum',False)[0] for i in range(4)])
        torch.testing.assert_close(batch,singles,atol=2e-7,rtol=2e-6)

    def test_j_norm_is_per_item_and_velocity_rescale_independent(self):
        s=self.sampler();z=(torch.arange(16,dtype=torch.float32).reshape(4,1,2,2)/20).requires_grad_(True)
        op=type('Op',(),{'A':lambda _,x:2*x})()
        y=torch.linspace(-.2,.3,16).reshape_as(z)
        def run(x,target):
            x=x.clone().detach().requires_grad_(True);v=s.predict_vector(x)
            norm=torch.linalg.vector_norm(v,dim=(1,2,3),keepdim=True).detach()
            return s.fmrg_gradient_step_jacobian(x,.7,op,target,'sr4',stepsize=5,num_iters=1,
                loss_mode='pixel',loss_func='sum',velocity=v,normalize_grad=True,velocity_norm=norm,t_next=0)[0]
        batch=run(z,y);single=torch.cat([run(z[i:i+1],y[i:i+1]) for i in range(4)])
        torch.testing.assert_close(batch,single,atol=2e-7,rtol=2e-6)


if __name__=='__main__':unittest.main()
