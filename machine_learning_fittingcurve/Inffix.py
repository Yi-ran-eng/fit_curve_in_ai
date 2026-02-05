import torch
import numpy as np
import pandas as pd
from abc import abstractmethod
from torch import nn
from typing import Type,Callable,Any
import matplotlib.pyplot as plt
device='cuda' if torch.cuda.is_available() else 'cpu'

def add_func(method:list[Callable],newcls:str=None):
    #firstly obtain parentclass' name ,func is the target method should be added to new
    #class.
    def _deco(cls:Type):
        #use type() to create a new class,type(name,(parentclass,),{parameters})
        newname=f'{cls.__name__}-son' if newcls is None else newcls
        new_cls=type(newname,(cls,),{})#空字典说明还没有传实例方法
        # add the new method,use setattr
        for me in method:
            setattr(new_cls,me.__name__,me)
        #return new class
        return new_cls
    #return deco
    return _deco
class Wedradown(nn.Module):
    def __init__(self,x,obs):
        super().__init__()
        assert isinstance(x,torch.Tensor),'input must be a tensor'
        x=x.view(-1,1) if x.ndim == 1 else x
        self.samples,self.features=x.shape[0],x.shape[1]
        self.x,self.vars=x.to(device),obs.to(device)
        #updated parameters
        self.alphaNL=nn.Parameter(torch.tensor(-100.).to(device))
        self.flatten=nn.Flatten()
        self.stack_relu=nn.Sequential(
            nn.Linear(self.features,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.Sigmoid(),
            nn.Linear(128,self.vars.shape[0])
        )
    @abstractmethod
    def _mapping_extend(self,*args,**kw):
        '''
        this function is to make customize method to add some transitions to ur target func
        '''
        pass
    def forward(self,x=None,**kw):
        if x is not None:
            self.x=x
        locs=locals()
        locs.pop('self')
        T_hat=self._mapping_extend(**locs,**kw)
        if T_hat is not None:
            for p in self.stack_relu.parameters():
                p.requires_grad_(False)
            return T_hat
        else:
            x=self.flatten(self.x)
            forx=self.stack_relu(x)
            sor=kw.get('use_softmax')
            out=nn.Softmax(dim=0)(forx) if sor is not None and sor else forx
            return out
class Trinall:
    @abstractmethod
    def Jloss(self,*args,**kw):
        pass
    def runall(self,cls:Type,feats,targets,**kw):
        methods,opt=kw.get('methods'),kw.get('optway')
        maxinter=kw.get('maxinter')
        kw={k:kw[k] for k in kw if k not in ['methods','optway','maxinter']}
        enhanc=add_func(methods,'_gradown')(cls) if methods is not None else cls
        targets=targets.view(-1,1) if targets.ndim == 1 else targets
        g=enhanc(feats,targets).to(device)
        losses=[]
        g.forward(**kw)
        if opt == 'adam':
            optimizer=torch.optim.Adam([g.alphaNL],lr=7e-6)
        else:
            optimizer=torch.optim.SGD([g.alphaNL],lr=1e-4,momentum=0.9,weight_decay=0)
        for epoch in range(maxinter):
            optimizer.zero_grad()
            out=g(**kw)
            if self.Jloss(out,torch.abs(g.vars)) is not None:
                loss=nn.MSELoss()(out,torch.abs(g.vars))
            else:
                loss=self.Jloss(out,torch.abs(g.vars))
            losses.append(loss.item())
            loss.backward()
            # optimizer.step()
            _costmGradown(model=g,lr=50.)
        loss_np=np.array(losses)
        plt.plot(loss_np)
        plt.show()
        return g
    def _predict(self,x:torch.Tensor,model,**kw):
        x0=x.view(-1,1) if x.ndim == 1 else x
        with torch.no_grad():
            out=model(x0,**kw)
        out_onnp=out.flatten().cpu().numpy()
        x_np=x.flatten().cpu().numpy()
        plt.plot(x_np,out_onnp,c='blue')
        plt.plot(x_np,model.vars.flatten().cpu().numpy(),c='red')
        plt.show()
        return out
def _mapping_extend(self:Wedradown,**kw):
    M,U=kw.get('M'),kw.get('U')
    I0,L_eff=kw.get('I0'),kw.get('L_eff')
    self.z0=kw.get('z0')
    if self.x.ndim == 1:
        self.x=self.x.view(-1,1)
    du=1*U/M
    us=np.arange(0,U,du,dtype=np.float32)
    dev=self.x.device
    Us=torch.from_numpy(us).to(dev)
    ts=np.exp(0.5*np.pi*np.sinh(us))
    Ts=torch.from_numpy(ts).view(1,-1).to(dev)
    self.Ts=Ts
    denom=torch.exp(Ts**2)*(1.0+self.x**2/self.z0**2).to(dev)
    # alphaNL=self.alphaNL
    self.scalar=torch.as_tensor(I0*L_eff,device=device)
    # incoding=torch.log(1+(self.alphaNL*scalar)/denom)
    C= self.alphaNL*I0*L_eff / (1.0 + self.x**2 /self.z0**2)
    self.c=C
    integrand = torch.log(1.0 + C * torch.exp(-Ts**2))
    # print(integrand.dtype,'dtype')
    self.cofactor=(torch.pi/2)*torch.cosh(Us)*Ts*du
   
    up=integrand*self.cofactor
    self.up=up
    if hasattr(self,'_contini'):
        A=self._contini(up,self.scalar,self.z0)
        self.A=A
    outpre=torch.sum(up,dim=1).view(-1,1)
    out=A*outpre+0.7
    self.hat=torch.abs(out)
    return self.hat
def _contini(self, sumup: torch.Tensor, scalar, z0):
    """
    A = 1 / (sqrt(pi) * C)
    C = alphaNL * scalar / (1 + x**2/z0**2)
    """
    C = self.alphaNL * scalar / (1.0 + self.x**2 / z0**2)          # (N,1)
    A = 2.0 / (torch.sqrt(torch.tensor(np.pi, device=self.x.device)) * C)
    return A          # (N,1)
def _costmGradown(model,lr):
    '''
    c_sum means up in _mapp function
    '''
    N=model.x.size(0)
    if hasattr(model,'up'):
        out_pre=torch.sum(model.up,dim=1).view(-1,1)
        dAdc=-1/(torch.sqrt(torch.tensor(np.pi,device=device))*model.c**2)
        dCdalpha=model.scalar/(1+model.x**2/model.z0**2)
        dAdalpha=dAdc*dCdalpha
        dupdalpha=model.cofactor*(torch.exp(-model.Ts**2)/(1.+model.c*torch.exp(-model.Ts**2)))
        dupdalpha*=dCdalpha
        dout=torch.sum(dupdalpha,dim=1).view(-1,1)
        dYdalpha=dAdalpha*out_pre+model.A*dout
        dLdy=(model.hat-model.vars)/(N)
        dLdalpha=(dYdalpha*dLdy).sum()
        with torch.no_grad():
            model.alphaNL.data.sub_(lr*dLdalpha.real)
df=pd.read_excel("D:/5.xls")
features=df.iloc[:,0].to_numpy()
targets=df.iloc[:,1].to_numpy()
features=torch.from_numpy(features)
targets=torch.from_numpy(targets)
features=torch.complex(features,torch.zeros_like(features))
targets=torch.complex(targets,torch.zeros_like(targets))
# enhanc=add_func([_mapping_extend,_contini],'_gradown')(Wedradown)
# g=enhanc(features,0)
# g.forward(M=10,U=4.,z0=9.7e-3,I0=2.,L_eff=2.1)

trainer=Trinall()
model=trainer.runall(
    cls=Wedradown,
    feats=features,targets=targets,
    methods=[_mapping_extend,_contini],
    optway='adam',maxinter=20000,
    M=10,U=4.0,z0=9.7e-3,L_eff=1.,I0=1.
)
xew=features.to(device)
ypre=trainer._predict(xew,model,M=10,U=4.0,z0=9.7e-3,L_eff=1.,I0=1.)

print(model.alphaNL)