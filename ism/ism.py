import numpy as np
import torch


class IsmHandler:
    def __init__(self, documents_number, relative_time_length, words_number, documents_length, latent_motifs_number,
                 adjusted_documents_length):
        self.nz = latent_motifs_number
        self.nw = words_number
        self.ntr = relative_time_length
        self.nd = documents_number
        self.Td = adjusted_documents_length

    #java -jar sequence-mining/target/sequence-mining-1.0.jar -i 100 -f /home/keskin/PycharmProjects/PLSM/mutu_data/ism_data.dat -v
    def save_ism_data(self, data):
        # Consider about the time. Assign each square in the motif a number ranging from 1 to nw*ntr
        ism_data = data.reshape(-1, self.Td + self.ntr - 1).cpu().numpy()
        tem_data = np.zeros((ism_data.shape[0], 3))
        ism_data = np.concatenate((ism_data, tem_data), axis=1)
        ism_seq = []
        step = 20
        for i in range(self.nd):
            cur_col = 0
            start_raw = i * self.nw
            while cur_col < (ism_data.shape[1] - self.ntr + 1):
                # During the process, using matrix to simplify the calculation
                # Choose all sqaures in the cur_window(Moving the window to generate the sequence)
                # pick its corresponding index in motif(1~nw*ntr)
                tem_index_pos = []
                tem_seq = []
                cur_window = ism_data[start_raw:start_raw + self.nw, cur_col:cur_col + self.ntr]
                cmp_mat = np.ones((self.nw, self.ntr))
                pos_mat = np.arange(self.nw * self.ntr).reshape(self.nw, self.ntr)
                tem_seq = pos_mat * (cur_window > cmp_mat)
                tem_seq = tem_seq.reshape(-1, self.nw * self.ntr)[0]
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

    def return_col_raw(self, num):
        ntr = self.ntr
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

    def compute_motif_initialization(self, data, seq):
        ism_data = data.reshape(-1, self.Td + self.ntr - 1).cpu().numpy()
        original_data = data.reshape(self.nw*self.nd, self.ntr+self.Td-1).cpu().numpy()

        non_num_data = ism_data
        for i in range(ism_data.shape[0]):
            non_num_data[i, :] = np.where(ism_data[i, :] > 0, i + 1 if i < 20 else i - 19, 0)

        init_motif = np.ones((self.nz, 1, self.nw, self.ntr))
        step = 100
        for i in range(self.nd):
            cur_raw = i * self.nw
            cur_col = 0
            while cur_col <= (non_num_data.shape[1] - self.ntr + 1):
                tem_data = original_data[cur_raw:cur_raw + self.nw, cur_col:cur_col + self.ntr]
                for i in range(len(seq)):
                    tem_seq = seq[i]
                    cur_motif = init_motif[i, 0, :, :]
                    for sub_seq in tem_seq:
                        tem_cor = self.return_col_raw(sub_seq)
                        cur_motif[tem_cor[0], tem_cor[1]] += tem_data[tem_cor[0], tem_cor[1]]
                    init_motif[i, 0, :, :] = cur_motif
                cur_col += step
        return init_motif

    def initialize_motifs(self, data, seq):
        init_motif = self.compute_motif_initialization(data, seq)
        init_motif = torch.from_numpy(init_motif).cpu()
        init_motif = init_motif.type_as(torch.ones(self.nd, self.nz, 1, self.Td).cpu())

        return init_motif
