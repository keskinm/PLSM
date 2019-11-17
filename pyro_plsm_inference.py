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


# ADD: change figure size
plt.rc('figure', figsize=(12.0, 7.0))


class PyroPLSMInference:
    def __init__(self, documents_number, relative_time_length, words_number, documents_length, latent_motifs_number,
                 n_steps, lr, observations_file_path, work_dir, seed, plot_results, n_samples, ism=True):
        self.documents_number = documents_number
        self.relative_time_length = relative_time_length
        self.words_number = words_number
        self.documents_length = documents_length
        self.latent_motifs_number = latent_motifs_number
        self.n_steps = n_steps
        self.lr = lr
        self.observations_file_path = observations_file_path
        self.adjusted_documents_length = documents_length - relative_time_length + 1
        self.work_dir = work_dir
        self.plot_results = plot_results if plot_results is not None else None
        self.n_samples = n_samples
        os.makedirs(work_dir, exist_ok=True)
        torch.manual_seed(seed)

        self.initalized_motifs = None
        self.ism = ism if ism is not None else None
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
    def p_w_ta_d(motifs_starting_times, motifs):
        t = F.conv_transpose2d(motifs_starting_times, motifs)
        # CHANGE: use shape (-1) to auto-infer
        return t.view(-1)

    def model(self, data):
        # ADD: factor out the shapes
        # NB: this is just the initialization
        motifs_starting_times_shape = (self.documents_number, self.latent_motifs_number, 1,
                                       self.adjusted_documents_length)

        motifs_shape = (self.latent_motifs_number, 1, self.words_number, self.relative_time_length)
        motifs_starting_times_concentration = torch.ones(*motifs_starting_times_shape)
        motifs_concentration = torch.ones(*motifs_shape)

        # CHANGE: use the fact that dirichlet can draw independant dirichlets
        # TODO: essayer "get_param"
        motifs_starting_times = pyro.sample("motifs_starting_times", pdist.Dirichlet(
            concentration=motifs_starting_times_concentration.view(self.documents_number, -1)))
        motifs = pyro.sample("motifs", pdist.Dirichlet(
            concentration=motifs_concentration.view(self.latent_motifs_number, -1)))

        # ADD: resize motifs_starting_times and motifs
        motifs_starting_times = motifs_starting_times.reshape(*motifs_starting_times_shape)
        motifs = motifs.reshape(*motifs_shape)

        with pyro.plate("data", len(data), subsample_size=100):
            # CHANGE: make explicit the fact that the number of observation is unused here
            pyro.sample("observe", pdist.Multinomial(-999, probs=self.p_w_ta_d(motifs_starting_times, motifs)),
                        obs=data)

    def guide(self, data):
        q_motifs_starting_times = pyro.param("q_motifs_starting_times", torch.ones(self.documents_number,
                                                                                   self.latent_motifs_number, 1,
                                                                                   self.adjusted_documents_length),
                                             constraint=constraints.positive)

        q_motifs = pyro.param("q_motifs", self.initalized_motifs, constraint=constraints.positive)

        # CHANGE: use the fact that dirichlet can draw independant dirichlets
        pyro.sample("motifs_starting_times", pdist.Dirichlet(
            concentration=q_motifs_starting_times.view(self.documents_number, -1)))

        pyro.sample("motifs", pdist.Dirichlet(concentration=q_motifs.view(self.latent_motifs_number, -1)))

    def run_inference(self):
        data = torch.tensor(self.parse_tdoc_file(self.observations_file_path, self.documents_length,
                                                 self.words_number), dtype=torch.float32).view(-1)

        self.initalized_motifs = self.initialize_motifs(data) if self.ism else torch.ones(self.latent_motifs_number, 1, self.words_number,
                                                     self.relative_time_length)

        pyro.clear_param_store()

        adam_params = {"lr": 0.1, "betas": (0.9, 0.999)}
        optimizer = pyro.optim.Adam(adam_params)

        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO(), num_samples=self.n_samples)

        for _ in tqdm(range(self.n_steps)):
            svi.step(data)

        motifs_starting_times_file_path = os.path.join(self.work_dir, 'motifs_starting_times.npy')
        motifs_file_path = os.path.join(self.work_dir, 'motifs.npy')

        motifs_starting_times = pyro.param("q_motifs_starting_times").detach().numpy()
        motifs = pyro.param("q_motifs").detach().numpy()

        np.save(file=motifs_starting_times_file_path, arr=motifs_starting_times)
        np.save(file=motifs_file_path, arr=motifs)

        self.dump_motifs_and_starting_times(motifs_starting_times, motifs)

        if self.plot_results:
            self.plot_motifs_and_starting_times(motifs)

    def plot_motifs_and_starting_times(self, motifs):
        plt.imshow(-motifs.reshape(-1, self.relative_time_length), cmap="gray")
        plt.show()

        for motif in motifs:
            plt.imshow(motif.squeeze())
            print(motif.sum())
            plt.show()

    def return_col_raw(self, num):
        ntr = self.relative_time_length
        cor = []
        col_index = num % ntr
        if col_index == 0:
            col_index = ntr - 1
            raw_index = int(num / ntr) - 1
        else:
            col_index = col_index - 1
            raw_index = int(num / ntr)
        cor.append(raw_index)
        cor.append(col_index)
        return cor

    def compute_motif_initialization(self, data):
        nz = self.latent_motifs_number
        nw = self.words_number
        ntr = self.relative_time_length
        nd = self.documents_number
        Td = self.adjusted_documents_length
        non_num_data = data.reshape(-1, Td+ntr-1).cpu().numpy()
        seq = [[247, 272, 295, 297, 342], [366], [275, 300]]

        init_motif = np.ones((nz, 1, nw, ntr))
        step = 100
        for i in range(nd):
            cur_raw = i * nw
            cur_col = 0
            while cur_col <= (non_num_data.shape[1] - ntr + 1):
                tem_data = data[cur_raw:cur_raw + nw, cur_col:cur_col + ntr]
                for i in range(len(seq)):
                    tem_seq = seq[i]
                    cur_motif = init_motif[i, 0, :, :]
                    for sub_seq in tem_seq:
                        tem_cor = self.return_col_raw(sub_seq)
                        cur_motif[tem_cor[0], tem_cor[1]] += tem_data[tem_cor[0], tem_cor[1]]
                    init_motif[i, 0, :, :] = cur_motif
                cur_col += step
        return init_motif

    def initialize_motifs(self, data):
        nz = self.latent_motifs_number
        nw = self.words_number
        ntr = self.relative_time_length
        nd = self.documents_number
        Td = self.adjusted_documents_length

        init_motif = self.compute_motif_initialization(data)
        init_motif = torch.from_numpy(init_motif).cpu()
        init_motif = init_motif.type_as(torch.ones(nd, nz, 1, Td).cpu())
        return init_motif

    def dump_motifs_and_starting_times(self, motifs_starting_times, motifs):
        pzd = motifs_starting_times.sum(axis=3)
        pzd /= pzd.sum(axis=1, keepdims=True)
        pzd = pzd.squeeze().transpose()

        ptszd = motifs_starting_times
        ptszd /= ptszd.sum(axis=3, keepdims=True)
        ptszd = np.stack([ptszd[i, :, 0].transpose() for i in range(self.documents_number)]).reshape(
            -1, self.latent_motifs_number)

        pwz = motifs.sum(axis=3)
        pwz /= pwz.sum(axis=2, keepdims=True)
        pwz = pwz.squeeze().transpose()

        ptrwz = motifs
        ptrwz /= ptrwz.sum(axis=3, keepdims=True)
        ptrwz = np.stack([ptrwz[i, 0].transpose() for i in range(self.latent_motifs_number)]).reshape(
            -1, self.words_number)

        pzd_file_path = os.path.join(self.work_dir, 'results.pzd')
        ptszd_file_path = os.path.join(self.work_dir, 'results.ptszd')
        pwz_file_path = os.path.join(self.work_dir, 'results.pwz')
        ptrwz_file_path = os.path.join(self.work_dir, 'results.ptrwz')

        np.savetxt(pzd_file_path, pzd)
        np.savetxt(ptszd_file_path, ptszd)
        np.savetxt(pwz_file_path, pwz)
        np.savetxt(ptrwz_file_path, ptrwz)


def main():
    parser = argparse.ArgumentParser(prog='pyro_inference')

    parser.add_argument(
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
        type=int,
        default=3602,
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
        '--n-samples',
        type=int,
        default=10,
        help=
        'number of samples to use for inference'
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
        '--work-dir',
        type=str,
        default='./data/inference',
        help=
        'path to the working dir where results are stored'
    )

    parser.add_argument(
        '--seed', type=int, default=101, help='seed of random generator')

    parser.add_argument(
        '--plot-results', action='store_true', help='plot motifs and their starting times')

    args = parser.parse_args()
    args = vars(args)

    pyro_plsm_inference = PyroPLSMInference(**args)
    pyro_plsm_inference.run_inference()

    return parser


if __name__ == "__main__":
    main()

