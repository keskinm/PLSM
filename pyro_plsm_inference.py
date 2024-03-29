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
from utils.parser import parse_tdoc_file
from ism.utils import format_seq_file
from ism.ism import IsmHandler

# ADD: change figure size
plt.rc('figure', figsize=(12.0, 7.0))


class PyroPLSMInference:
    def __init__(self, documents_number, relative_time_length, words_number, documents_length, latent_motifs_number,
                 n_steps, lr, observations_file_path, work_dir, seed, plot_results, n_samples, use_ism, create_ism_data_file):
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

        if self.use_ism:
            self.ism = IsmHandler(documents_number=documents_number, relative_time_length=relative_time_length,
                                  words_number=words_number,
                                  documents_length=documents_length,
                                  latent_motifs_number=latent_motifs_number,
                                  adjusted_documents_length=self.adjusted_documents_length)

        self.step_motif_count = 0
        self.step_motif_count_divisor = 5
        self.motifs_list_for_metrics = []

        self.create_ism_data_file = create_ism_data_file

        # prior0 = 0.1*N/nd / nz / Td
        # prior1 = 0.1*N/nz / nw / ntr
        # randinit = 0

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
        data = torch.tensor(parse_tdoc_file(self.observations_file_path, self.documents_length,
                                                 self.words_number), dtype=torch.float32).view(-1)

        if self.create_ism_data_file:
            self.ism.save_ism_data(data)
            return 0

        if self.use_ism:
            seq = format_seq_file('./mutu_data/seq.txt')
            self.initalized_motifs = self.ism.initialize_motifs(data, seq)
        else:
            self.initalized_motifs = torch.ones(self.latent_motifs_number, 1, self.words_number, self.relative_time_length)

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

    parser.add_argument(
        '--create-ism-data-file', action='store_true', help='Create ism data file to be used for mining')

    args = parser.parse_args()
    args = vars(args)

    pyro_plsm_inference = PyroPLSMInference(**args)
    pyro_plsm_inference.run_inference()

    return parser


if __name__ == "__main__":
    main()

