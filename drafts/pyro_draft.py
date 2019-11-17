from ipywidgets import FloatProgress
from IPython.display import display
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as pdist
import torch.distributions.constraints as constraints
import pyro.infer
from pyro.infer import SVI, Trace_ELBO
import pyro.optim
from matplotlib import pyplot as plt
import numpy as np

torch.manual_seed(101)

softplus = torch.nn.Softplus()

# ADD: change figure size
plt.rc('figure', figsize=(12.0, 7.0))


# Function that Load the tdoc file into a Matrix
def load_data(filetdoc, document_length, nw):
    matrix = [[0 for x in range(document_length)] for x in range(nw)]
    # np.zeros((documentLength, W))  # Initialize Matrix of length 300x25 with Zeros   ,dtype=np.uint8

    Dict = {}  # Define Dictionary , Index of line position : List (that contain the Values)
    # Read the file tdoc
    infile = open(filetdoc, 'r')  # Read the tdoc file
    i = 0
    for data in infile:
        if len(data.strip()) != 0:  # take not empty line in tdoc and make some processing on Data
            Dict[i] = [data]  # Example Dict[113] = ['12:2 13:3 14:4 15:2 16:1 \n']
        i += 1
    infile.close()
    for j in Dict.keys():
        x = Dict[j]  # x is the Value in the Dict at Index j
        y = x[0].split(' ')  # Split at the space to get something like this ['17:1', '\n']
        for k in range(len(y)):
            z = y[k].split(':')  # split at : so we get ['17:1']['\n']
            for m in range(len(z) - 1):
                b = int(z[m])  # b is the position of Word W and j is the time , For Example b=17
                q = float(z[m + 1])  # q is the value should be filled in the matrix , For example q=1 (at position j,b)
                matrix[b][j] = q  # Fill the Matrix with v
#                 matrix= np.transpose(matrix)         # To put the Matrix in the correct form

    return np.array(matrix)  # Return the Matrix



# ADD: number of documents
nd = 1
ntr=20
Tdo=3602
Td=Tdo-ntr+1
nw=100

# ADD: introduce some variables
nz = 5
# prior0 = 0.1*N/nd / nz / Td
# prior1 = 0.1*N/nz / nw / ntr

#randinit = 0


data = torch.tensor(load_data('Junction1-b-s-m-plsa.tdoc',Tdo,nw), dtype=torch.float32).view(-1)


data_vis = torch.tensor(load_data('Junction1-b-s-m-plsa.tdoc',Tdo,nw), dtype=torch.float32)


def p_w_ta_d(z,motifs):
    t = F.conv_transpose2d(z,motifs)
    # CHANGE: use shape (-1) to auto-infer
    return t.view(-1)



pyro.clear_param_store()


def model(data):
    # ADD: factor out the shapes
    # NB: this is just the initialization
    s0 = (nd,nz,1,Td)
    s1 = (nz,1,nw,ntr)
    alpha0 = torch.ones(*s0)
    alpha1 = torch.ones(*s1)
    # CHANGE: use the fact that dirichlet can draw independant dirichlets
    # TODO: essayer "get_param"
    z = pyro.sample("latent0", pdist.Dirichlet(concentration=alpha0.view(nd, -1)))
    motifs = pyro.sample("latent1", pdist.Dirichlet(concentration=alpha1.view(nz, -1)))
    # ADD: resize z and motifs
    z = z.reshape(*s0)
    motifs = motifs.reshape(*s1)
    with pyro.iarange("data", len(data)):
        # CHANGE: make explicit the fact that the number of observation is unused here
        pyro.sample("observe", pdist.Multinomial(-999, probs=p_w_ta_d(z, motifs)), obs=data)



def guide(data):
    qalpha0 = pyro.param("qalpha0", torch.ones(nd, nz, 1, Td), constraint=constraints.positive)
    qalpha1 = pyro.param("qalpha1", torch.ones(nz, 1, nw, ntr), constraint=constraints.positive)
    # CHANGE: use the fact that dirichlet can draw independant dirichlets
    pyro.sample("latent0", pdist.Dirichlet(concentration=qalpha0.view(nd, -1)))
    pyro.sample("latent1", pdist.Dirichlet(concentration=qalpha1.view(nz, -1)))


# CHANGE: change adam params
adam_params = {"lr": 0.1}
#adam_params = {"lr": 0.005, "betas": (0.9, 0.999)}
optimizer = pyro.optim.Adam(adam_params)
#optimizer = pyro.optim.SGD(adam_params)

svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# CHANGE: use a nice progress bar
n_steps = 400
pro = FloatProgress(min=0, max=n_steps - 1)
display(pro)

for step in range(n_steps):
    svi.step(data)
    pro.value += 1
    pro.description = str(step)

# CHANGE: change only at the end
np.save(file="qalpha0", arr=pyro.param("qalpha0").detach().numpy())
np.save(file="qalpha1", arr=pyro.param("qalpha1").detach().numpy())


plt.imshow(data_vis[:,80:180])


# ADD: quick plot before exhaustive plot
loaded = np.load("qalpha1.npy")
plt.imshow( - loaded.reshape(-1, ntr), cmap="gray")
plt.show()

for i in range(nz):
    #plt.imshow(loaded[i].squeeze())
    print(loaded[i].sum())
    #plt.show()


loaded.sum(axis=3).shape

loaded.sum(axis=3)/loaded.sum(axis=3).sum(axis=2,keepdims=True)

plt.imshow(loaded[0].squeeze())

np.load("qalpha1.npy").sum(axis=3).sum()


loaded = np.load("qalpha1.npy")
pwz=loaded.sum(axis=3)
pwz/=pwz.sum(axis=2,keepdims=True)
np.savetxt('results.pwz',pwz.squeeze().transpose())
loaded/=loaded.sum(axis=3, keepdims=True)
np.savetxt('results.ptrwz',np.stack([loaded[i,0].transpose() for i in range(nz)]).reshape(-1,nw))
loaded.shape

loaded.sum(axis=3)

loaded = np.load("qalpha0.npy")
pzd=loaded.sum(axis=3)
pzd/=pzd.sum(axis=1,keepdims=True)
np.savetxt('results.pzd',pzd.squeeze().transpose())
loaded/=loaded.sum(axis=3, keepdims=True)
np.savetxt('results.ptszd',np.stack([loaded[i,:,0].transpose() for i in range(nd)]).reshape(-1,nz))


np.load("qalpha0.npy").shape



