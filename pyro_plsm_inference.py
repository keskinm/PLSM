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
                 n_steps, lr, observations_file_path, work_dir, seed, plot_results, n_samples, use_ism):
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
        self.use_ism = use_ism if use_ism is not None else None

        self.step_motif_count = 0
        self.step_motif_count_divisor = 5
        self.motifs_list_for_metrics = []

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

        if self.step_motif_count % self.step_motif_count_divisor == 0:
            self.motifs_list_for_metrics.append(q_motifs)

        # CHANGE: use the fact that dirichlet can draw independant dirichlets
        pyro.sample("motifs_starting_times", pdist.Dirichlet(
            concentration=q_motifs_starting_times.view(self.documents_number, -1)))

        pyro.sample("motifs", pdist.Dirichlet(concentration=q_motifs.view(self.latent_motifs_number, -1)))

    def run_inference(self):
        data = torch.tensor(self.parse_tdoc_file(self.observations_file_path, self.documents_length,
                                                 self.words_number), dtype=torch.float32).view(-1)

        seq = self.format_seq('./mutu_data/seq.txt')

        self.initalized_motifs = self.initialize_motifs(data, seq) if self.use_ism else torch.ones(self.latent_motifs_number, 1, self.words_number,
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

    def format_seq(self, seq_file_path):
        seq_file = open(seq_file_path, "r")
        contents = seq_file.read()
        seq_file.close()

        contents = contents.split(',')

        new_contents = []

        for content in contents:
            stop_idx = content.find(']')
            new_contents.append(content[:stop_idx+1])

        filtered_contents = list(filter(lambda x: len(x) != 0, new_contents))

        filtered_contents = [int(content[1:-1]) for content in filtered_contents]

        filtered_contents = [filtered_contents]

        return filtered_contents

    def compute_motif_initialization(self, data, seq):
        nz = self.latent_motifs_number
        nw = self.words_number
        ntr = self.relative_time_length
        nd = self.documents_number
        Td = self.adjusted_documents_length
        ism_data = data.reshape(-1, Td + ntr - 1).cpu().numpy()
        original_data = data.reshape(nw*nd,ntr+Td-1).cpu().numpy()

        non_num_data = ism_data
        for i in range(ism_data.shape[0]):
            non_num_data[i, :] = np.where(ism_data[i, :] > 0, i + 1 if i < 20 else i - 19, 0)

        init_motif = np.ones((nz, 1, nw, ntr))
        step = 100
        for i in range(nd):
            cur_raw = i * nw
            cur_col = 0
            while cur_col <= (non_num_data.shape[1] - ntr + 1):
                tem_data = original_data[cur_raw:cur_raw + nw, cur_col:cur_col + ntr]
                for i in range(len(seq)):
                    tem_seq = seq[i]
                    cur_motif = init_motif[i, 0, :, :]
                    for sub_seq in tem_seq:
                        tem_cor = self.return_col_raw(sub_seq)
                        cur_motif[tem_cor[0], tem_cor[1]] += tem_data[tem_cor[0], tem_cor[1]]
                    init_motif[i, 0, :, :] = cur_motif
                cur_col += step
        return init_motif

#java -jar sequence-mining/target/sequence-mining-1.0.jar -i 100 -f /home/keskin/PycharmProjects/PLSM/mutu_data/ism_data.dat -v
    def save_ism_data(self, data):
        nz = self.latent_motifs_number
        nw = self.words_number
        ntr = self.relative_time_length
        nd = self.documents_number
        Td = self.adjusted_documents_length

        # Consider about the time. Assign each square in the motif a number ranging from 1 to nw*ntr
        ism_data = data.reshape(-1, Td + ntr - 1).cpu().numpy()
        tem_data = np.zeros((ism_data.shape[0], 3))
        ism_data = np.concatenate((ism_data, tem_data), axis=1)
        ism_seq = []
        step = 20
        for i in range(nd):
            cur_col = 0
            start_raw = i * nw
            while cur_col < (ism_data.shape[1] - ntr + 1):
                # During the process, using matrix to simplify the calculation
                # Choose all sqaures in the cur_window(Moving the window to generate the sequence)
                # pick its corresponding index in motif(1~nw*ntr)
                tem_index_pos = []
                tem_seq = []
                cur_window = ism_data[start_raw:start_raw + nw, cur_col:cur_col + ntr]
                cmp_mat = np.ones((nw, ntr))
                pos_mat = np.arange(nw * ntr).reshape(nw, ntr)
                tem_seq = pos_mat * (cur_window > cmp_mat)
                tem_seq = tem_seq.reshape(-1, nw * ntr)[0]
                tem_index_pos = np.array(np.where(tem_seq > 0))
                tem_index_pos = tem_index_pos + 1
                ism_seq.append(tem_index_pos)

                cur_col += step
        with open("./mutu_data/ism_data.dat", "w") as file:
            file.truncate()
            for i in range(len(ism_seq)):
                if len(ism_seq[i][0]) != 0:
                    tem_str = ism_seq[i][0].astype(int).astype(str).tolist()
                    file.write(" -1 ".join(tem_str))
                    file.write(' -2\n')
        file.close()

    def initialize_motifs(self, data, seq):
        nz = self.latent_motifs_number
        nw = self.words_number
        ntr = self.relative_time_length
        nd = self.documents_number
        Td = self.adjusted_documents_length

        init_motif = self.compute_motif_initialization(data, seq)
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

    def cal_median_KL(self, infered_motifs, labeled_motifs):
        nz = self.latent_motifs_number
        nw = self.words_number
        ntr = self.relative_time_length

        KL = []
        infered_motif0 = infered_motifs[0, 0, :, :]
        infered_motif1 = infered_motifs[1, 0, :, :]
        infered_motif2 = infered_motifs[2, 0, :, :]
        norm_infer_motif0 = infered_motif1 / infered_motif1.sum()
        norm_infer_motif1 = infered_motif2 / infered_motif2.sum()
        norm_infer_motif2 = infered_motif0 / infered_motif0.sum()
        normalizer = 0
        for n in range(nz):
            temKL = 0
            real_motif = labeled_motifs[n, 0, :, :].cpu()
            norm_real_motif = real_motif / real_motif.sum()
            for i in range(nw):
                for j in range(ntr):
                    if norm_real_motif[i, j] == 0:
                        temKL += 0
                    else:
                        normalizer += norm_real_motif[i, j] + locals()['norm_infer_motif' + str(n)][i, j]
                        temKL += norm_real_motif[i, j] * (
                            np.log(norm_real_motif[i, j] / locals()['norm_infer_motif' + str(n)][i, j]))
            temKL = temKL / normalizer
            KL.append(temKL)
        mean_KL = np.sum(KL) / nz
        print(mean_KL)
        return mean_KL

    def compute_metrics(self, labeled_motifs):
        metrics = []
        for i in range(len(self.motifs_list_for_metrics)):
            motif_at_given_step = self.motifs_list_for_metrics[i]
            motif_at_given_step = motif_at_given_step.cpu().detach().numpy()
            median_kl = self.cal_median_KL(motif_at_given_step, labeled_motifs)
            metrics.append(median_kl.item())
        return metrics


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

    parser.add_argument(
        '--use-ism', action='store_true', help='Use ism for initializing motifs')

    args = parser.parse_args()
    args = vars(args)

    pyro_plsm_inference = PyroPLSMInference(**args)
    pyro_plsm_inference.run_inference()

    return parser


if __name__ == "__main__":
    main()

