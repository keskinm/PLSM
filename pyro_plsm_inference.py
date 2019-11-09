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

softplus = torch.nn.Softplus()

# ADD: change figure size
plt.rc('figure', figsize=(12.0, 7.0))


class PyroPLSMInference:
    def __init__(self, documents_number, relative_time_length, words_number, documents_length, latent_motifs_number,
                 n_steps, lr, observations_file_path, seed):
        self.documents_number = documents_number
        self.relative_time_length = relative_time_length
        self.words_number = words_number
        self.documents_length = documents_length
        self.latent_motifs_number = latent_motifs_number
        self.n_steps = n_steps
        self.lr = lr
        self.observations_file_path = observations_file_path
        self.adjusted_documents_length = documents_length - relative_time_length + 1
        torch.manual_seed(seed)

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
        motifs_starting_times = torch.ones(*s0)
        motifs = torch.ones(*s1)
        # CHANGE: use the fact that dirichlet can draw independant dirichlets
        # TODO: essayer "get_param"
        z = pyro.sample("motifs_starting_times", pdist.Dirichlet(concentration=
                                                                 motifs_starting_times.view(self.documents_number, -1)))
        motifs = pyro.sample("motifs", pdist.Dirichlet(concentration=motifs.view(self.documents_number, -1)))
        # ADD: resize z and motifs
        z = z.reshape(*s0)
        motifs = motifs.reshape(*s1)
        with pyro.plate("data", 360200, subsample_size=100):
            # CHANGE: make explicit the fact that the number of observation is unused here
            pyro.sample("observe", pdist.Multinomial(-999, probs=self.p_w_ta_d(z, motifs)), obs=data)

    def guide(self, data):
        q_motifs_starting_times = pyro.param("q_motifs_starting_times", torch.ones(self.documents_number,
                                                                                   self.latent_motifs_number, 1,
                                                                                   self.adjusted_documents_length),
                                             constraint=constraints.positive)

        q_motifs = pyro.param("q_motifs", torch.ones(self.latent_motifs_number, 1, self.words_number,
                                                     self.relative_time_length), constraint=constraints.positive)

        # CHANGE: use the fact that dirichlet can draw independant dirichlets
        pyro.sample("motifs_starting_times", pdist.Dirichlet(
            concentration=q_motifs_starting_times.view(self.documents_number, -1)))

        pyro.sample("motifs", pdist.Dirichlet(concentration=q_motifs.view(self.latent_motifs_number, -1)))

    def run_inference(self):
        data = torch.tensor(self.load_data('./data/Junction1-b-s-m-plsa.tdoc', self.documents_length,
                                           self.words_number), dtype=torch.float32).view(-1)

        pyro.clear_param_store()

        adam_params = {"lr": 0.1, "betas": (0.9, 0.999)}
        optimizer = pyro.optim.Adam(adam_params)

        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

        for _ in tqdm(range(self.n_steps)):
            svi.step(data)

        # CHANGE: change only at the end
        np.save(file="./data/motifs_starting_times", arr=pyro.param("q_motifs_starting_times").detach().numpy())
        np.save(file="./data/motifs", arr=pyro.param("q_motifs").detach().numpy())

        self.plots()

    def plots(self):
        # ADD: quick plot before exhaustive plot
        motifs_starting_times_file_path = "./data/motifs_starting_times.npy"
        motifs_file_path = "./data/motifs.npy"

        loaded = np.load(motifs_file_path)
        plt.imshow(-loaded.reshape(-1, self.relative_time_length), cmap="gray")
        plt.show()

        for i in range(self.latent_motifs_number):
            plt.imshow(loaded[i].squeeze())
            print(loaded[i].sum())
            plt.show()

        plt.imshow(loaded[0].squeeze())
        plt.show()

        loaded = np.load(motifs_starting_times_file_path)
        pzd = loaded.sum(axis=3)
        pzd /= pzd.sum(axis=1, keepdims=True)
        np.savetxt('./data/results.pzd', pzd.squeeze().transpose())
        loaded /= loaded.sum(axis=3, keepdims=True)
        np.savetxt('./data/results.ptszd', np.stack([loaded[i, :, 0].transpose() for i in range(self.documents_number)
                                                     ]).reshape(-1, self.latent_motifs_number))

        np.load(motifs_file_path).sum(axis=3).sum()
        loaded = np.load(motifs_file_path)
        pwz = loaded.sum(axis=3)
        pwz /= pwz.sum(axis=2, keepdims=True)
        np.savetxt('./data/results.pwz', pwz.squeeze().transpose())
        loaded /= loaded.sum(axis=3, keepdims=True)
        np.savetxt('./data/results.ptrwz', np.stack([loaded[i, 0].transpose() for i in range(self.latent_motifs_number)
                                                     ]).reshape(-1, self.words_number))
        loaded.sum(axis=3)


def main():
    parser = argparse.ArgumentParser(prog='pyro_inference')

    parser.add_argument(
        '-nd',
        '--documents-number',
        type=int,
        default=1,
        help='Number of documents')

    parser.add_argument(
        '--relative-time-length',
        type=int,
        default=20,
        help='Length of the motifs'
    )

    parser.add_argument(
        '--words-number',
        type=int,
        default=100,
        help='number of words')

    parser.add_argument(
        '--documents-length',
        action='store_true',
        help='Length of documents')

    parser.add_argument(
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
        '--seed', type=int, default=101, help='seed of random generator')

    args = parser.parse_args()
    args = vars(args)

    pyro_plsm_inference = PyroPLSMInference(**args)
    pyro_plsm_inference.run_inference()

    return parser


if __name__ == "__main__":
    main()

