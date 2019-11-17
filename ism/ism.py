import numpy as np


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