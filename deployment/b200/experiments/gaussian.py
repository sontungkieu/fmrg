"""Appendix C.5 continuous oracle and low-NFE Euler diagnostics."""
import json,math,sys
from pathlib import Path
import numpy as np
from scipy.integrate import quad
def run(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);s=.5;a=1.
 def c(t):return (1-t)**2+(s*t)**2
 def theta(t):return math.atan2(s*t,1-t)
 def gain(t,l,m):
  if m=="greedy":return 2*l*s*s/c(t)
  if m=="exact_lqr":return l/(c(t)*(1/(2*s*s)+l*(math.pi/2-theta(t))/s))
  return 0.
 checks=[]
 for l in [0.,.1,1.,10.]:
  for m in ["greedy","exact_lqr"]:
   numerical=s*math.exp(-quad(lambda t:gain(t,l,m),0,1,epsabs=1e-11)[0])
   analytic=s*(math.exp(-math.pi*l*s) if m=="greedy" else 1/(1+math.pi*l*s))
   assert math.isclose(numerical,analytic,rel_tol=1e-9,abs_tol=1e-13)
   checks.append(dict(lam=l,method=m,numeric=numerical,analytic=analytic))
 rows=[]
 for seed in [0,1,2]:
  x=np.random.default_rng(seed).normal(size=10000)
  for l in [0.,.1,1.,10.]:
   for n in [3,6,12,30]:
    for m in ["unguided","exact_lqr","greedy"]:
     for cutoff in ([.25,.5,.75,1.] if m=="greedy" else [1.]):
      coef,b,energy=1.,0.,0.;effective=min(1.,math.ceil(cutoff*n)/n)
      for i in range(n):
       t=i/n;v=(-2*(1-t)+2*t*s*s)/(2*c(t));k=gain(t,l,m) if t<cutoff else 0.;xm=a*math.sqrt(c(t))/s
       energy+=k*k*(coef*coef+(b-xm)**2)/n
       coef,b=coef+(v-k)*coef/n,b+((v-k)*b+k*xm)/n
      end=coef*x+b
      shrink=math.exp(-2*l*s*theta(effective)) if m=="greedy" else 1/(1+math.pi*l*s) if m=="exact_lqr" else 1.
      mean=a*(1-shrink) if m!="unguided" else 0.;std=s*shrink
      row=dict(seed=seed,lam=l,nfe=n,method=m,requested_cutoff=cutoff,effective_cutoff=effective,samples=10000,
       mean=float(end.mean()),std=float(end.std()),reward=float(-np.mean((end-a)**2)),
       oracle_mean=mean,oracle_std=std,coefficient_mean=b,coefficient_std=abs(coef),
       w2_to_oracle=math.hypot(b-mean,abs(coef)-std),discrete_control_energy=energy)
      assert all(math.isfinite(v) for v in row.values() if isinstance(v,float));rows.append(row)
 (out/"rows.json").write_text(json.dumps(rows,indent=2));(out/"oracle-checks.json").write_text(json.dumps(checks,indent=2))
 summary=dict(status="COMPLETE",rows=len(rows),oracle_checks=len(checks),scope="Appendix C.5 formula/quadrature validation and Euler discretization; not image efficacy")
 (out/"summary.json").write_text(json.dumps(summary,indent=2));print("FMRG_GAUSSIAN_JSON="+json.dumps(summary),flush=True);return summary
if __name__=="__main__":run(sys.argv[1])
