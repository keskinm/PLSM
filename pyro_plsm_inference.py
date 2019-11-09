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
from tqdm import tqdm
import os
import argparse

torch.manual_seed(101)

softplus = torch.nn.Softplus()

# ADD: change figure size
plt.rc('figure', figsize=(12.0, 7.0))


class PyroPLSMInference:
    def __init__(self, documents_number, relative_time_length, words_number, documents_length, latent_motifs_number,
                 n_steps, lr, observations_file_path):
        self.documents_number = documents_number
        self.relative_time_length = relative_time_length
        self.words_number = words_number
        self.documents_length = documents_length
        self.latent_motifs_number = latent_motifs_number
        self.n_steps = n_steps
        self.lr = lr
        self.observations_file_path = observations_file_path
        self.adjusted_documents_length = documents_length - relative_time_length + 1

        # prior0 = 0.1*N/nd / nz / Td
        # prior1 = 0.1*N/nz / nw / ntr
        # randinit = 0

    @staticmethod
    def parse_tdoc_file(filetdoc, document_length, nw):
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

    @staticmethod
    def p_w_ta_d(z, motifs):
        t = F.conv_transpose2d(z, motifs)
        # CHANGE: use shape (-1) to auto-infer
        return t.view(-1)

    def model(self, data):
        # ADD: factor out the shapes
        # NB: this is just the initialization
        s0 = (self.documents_number, self.latent_motifs_number, 1, self.adjusted_documents_length)
        s1 = (self.latent_motifs_number, 1, self.words_number, self.relative_time_length)
        alpha0 = torch.ones(*s0)
        alpha1 = torch.ones(*s1)
        # CHANGE: use the fact that dirichlet can draw independant dirichlets
        # TODO: essayer "get_param"
        z = pyro.sample("latent0", pdist.Dirichlet(concentration=alpha0.view(self.documents_number, -1)))
        motifs = pyro.sample("latent1", pdist.Dirichlet(concentration=alpha1.view(self.documents_number, -1)))
        # ADD: resize z and motifs
        z = z.reshape(*s0)
        motifs = motifs.reshape(*s1)
        with pyro.plate("data", 360200, subsample_size=100):
            # CHANGE: make explicit the fact that the number of observation is unused here
            pyro.sample("observe", pdist.Multinomial(-999, probs=self.p_w_ta_d(z, motifs)), obs=data)

    def guide(self, data):
        qalpha0 = pyro.param("qalpha0", torch.ones(self.documents_number, self.latent_motifs_number, 1,
                                                   self.adjusted_documents_length), constraint=constraints.positive)

        qalpha1 = pyro.param("qalpha1", torch.ones(self.latent_motifs_number, 1, self.words_number,
                                                   self.relative_time_length), constraint=constraints.positive)

        # CHANGE: use the fact that dirichlet can draw independant dirichlets
        pyro.sample("latent0", pdist.Dirichlet(concentration=qalpha0.view(self.documents_number, -1)))
        pyro.sample("latent1", pdist.Dirichlet(concentration=qalpha1.view(self.latent_motifs_number, -1)))

    def run_inference(self):
        data = torch.tensor(self.load_data('./data/Junction1-b-s-m-plsa.tdoc', self.documents_length, self.words_number),
                            dtype=torch.float32).view(-1)
        data_vis = torch.tensor(self.load_data('./data/Junction1-b-s-m-plsa.tdoc', Tdo, nw), dtype=torch.float32)

        pyro.clear_param_store()

        # CHANGE: change adam params
        adam_params = {"lr": 0.1}
        #adam_params = {"lr": 0.005, "betas": (0.9, 0.999)}
        optimizer = pyro.optim.Adam(adam_params)
        #optimizer = pyro.optim.SGD(adam_params)

        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        # CHANGE: use a nice progress bar
        n_steps = 400

        # for step in tqdm(range(n_steps)):
        #     svi.step(data)
        #
        # # CHANGE: change only at the end
        # np.save(file="./data/qalpha0", arr=pyro.param("qalpha0").detach().numpy())
        # np.save(file="./data/qalpha1", arr=pyro.param("qalpha1").detach().numpy())



def plots():
    # ADD: quick plot before exhaustive plot
    file0 = "./data/qalpha0.npy"
    file1 = "./data/qalpha1.npy"

    loaded = np.load(file1)
    plt.imshow(-loaded.reshape(-1, ntr), cmap="gray")
    plt.show()
    for i in range(nz):
        # plt.imshow(loaded[i].squeeze())
        print(loaded[i].sum())
        # plt.show()
    loaded.sum(axis=3).shape
    loaded.sum(axis=3) / loaded.sum(axis=3).sum(axis=2, keepdims=True)
    plt.imshow(loaded[0].squeeze())

    np.load(file1).sum(axis=3).sum()
    loaded = np.load(file1)
    pwz = loaded.sum(axis=3)
    pwz /= pwz.sum(axis=2, keepdims=True)
    np.savetxt('./data/results.pwz', pwz.squeeze().transpose())
    loaded /= loaded.sum(axis=3, keepdims=True)
    np.savetxt('./data/results.ptrwz', np.stack([loaded[i, 0].transpose() for i in range(nz)]).reshape(-1, nw))
    loaded.shape
    loaded.sum(axis=3)

    loaded = np.load(file0)
    pzd = loaded.sum(axis=3)
    pzd /= pzd.sum(axis=1, keepdims=True)
    np.savetxt('./data/results.pzd', pzd.squeeze().transpose())
    loaded /= loaded.sum(axis=3, keepdims=True)
    np.savetxt('./data/results.ptszd', np.stack([loaded[i, :, 0].transpose() for i in range(nd)]).reshape(-1, nz))
    loaded.shape


plots()


def main():
    parser = argparse.ArgumentParser(prog='pyro_inference')

    parser.add_argument(
        '-nd',
        '--documents-number',
        type=int,
        default=1,
        help='Number of documents')

    parser.add_argument(
        '-ntr',
        '--relative-time-length',
        type=int,
        default=20,
        help='Length of the motifs'
    )

    parser.add_argument(
        '-nz'
        '--words-number',
        type=int,
        default=100,
        help='number of words')

    parser.add_argument(
        '-Td'
        '--documents-length',
        action='store_true',
        help='Length of documents')

    parser.add_argument(
        '-nz',
        '--latent-motifs-number',
        type=int,
        default=5,
        help='Assumed number of latent motifs')

    parser.add_argument(
        '--n-steps',
        type=int,
        default=400,
        help=
        'number of steps for inference'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help='learning rate of the optimizer')

    parser.add_argument(
        '--observations-file-path',
        type=str,
        default='./data/real_data/vocabulary-set-b/Junction1/Junction1-b-s-m-plsa.tdoc',
        help=
        'path to the .tdoc observations file'
    )

    parser.add_argument(
        '--seed', type=int, default=42, help='seed of random generator')

    return parser


if __name__ == "__main__":
    main()

