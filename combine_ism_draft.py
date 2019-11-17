from __future__ import print_function
import math
import matplotlib.cm as cm
from matplotlib import pyplot as plt


import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pyro
from pyro.optim import Adam
from pyro.infer import SVI
import pyro.distributions as pdist
import torch.distributions as tdist
import torch.distributions.constraints as constraints
import pyro.infer
from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO
import pyro.optim


torch.manual_seed(101)

flag_ISM = 0
overlap_flag = 3
nw = 20  # number of words 25
ntr = 25  # number of relative times in a motif 70
nd = 2  # number of documents
Td = 150  # number of time period
nz = 3

# -----------------------------------------------------------------
plt.rc('figure', figsize=(12.0, 7.0))


def get_size(txt, font):
    test_img = Image.new('L', (1, 1))
    test_draw = ImageDraw.Draw(test_img)
    return test_draw.textsize(txt, font)


def string_to_matrix(s, fontname, fontsize, nw, ntr):
    # Define the Text Color and the Background
    color_text = "White"
    color_background = "Black"
    # Define the image font and resize the nword in a rectangle that suit it
    text = s
    font = ImageFont.truetype(fontname, fontsize)
    width, height = get_size(text, font)
    img = Image.new('L', (ntr, nw), color_background)
    d = ImageDraw.Draw(img)
    d.text((3, height / 10), text, fill=color_text, font=font)
    # d.rectangle((0, 0, width, height))
    path = './mutu_data/' + 'Image_' + text + '.png'
    img.save(path)
    im = Image.open(path).convert('L')
    motif = np.asarray(im, np.float32)  # Motif Matrix
    return motif


# motifs_as_string = ["eggplop", "eggnog", "eggplant", "banana", "apple"]
motifs_as_string = ["WYX", "Crab", "HJQ"]

nz0 = len(motifs_as_string)

fontname = 'JennaSue.ttf'
fontsize = 18

motifs_as_matrix = [string_to_matrix(st, fontname, fontsize, nw, ntr) for st in motifs_as_string]

# CHANGE: simplified using
motifs = torch.stack([torch.tensor(m[np.newaxis, :, :]) for m in motifs_as_matrix], 0).cpu()

# ADD: number of documents

z = torch.zeros(nd, nz0, 1, Td).cpu()

# the following tries to generate the clean data(patterns are clear, no overlao)
# clear version
if overlap_flag == 0:
    z[0, 0, 0, 1] = 1
    z[0, 1, 0, 99] = 1
    z[0, 2, 0, 30] = 1
    z[0, 2, 0, 70] = 1
    z[0, 0, 0, 149] = 1

    z[1, 2, 0, 90] = 1
    z[1, 2, 0, 10] = 1
    z[1, 1, 0, 40] = 1
    z[1, 0, 0, 120] = 1
    z[1, 1, 0, 140] = 1
elif overlap_flag == 1:
    # overlap version
    z[0, 0, 0, 20] = 1
    z[0, 2, 0, 99] = 1
    z[0, 1, 0, 30] = 1
    z[0, 2, 0, 70] = 1
    z[0, 1, 0, 110] = 1
    z[0, 0, 0, 149] = 1

    z[1, 2, 0, 90] = 1
    z[1, 2, 0, 10] = 1
    z[1, 1, 0, 20] = 1
    z[1, 0, 0, 120] = 1
    z[1, 1, 0, 130] = 1
elif overlap_flag == 2:
    # overlap version
    z[0, 0, 0, 20] = 1
    z[0, 2, 0, 99] = 1
    z[0, 1, 0, 25] = 1
    z[0, 2, 0, 70] = 1
    z[0, 1, 0, 60] = 1
    z[0, 1, 0, 110] = 1
    z[0, 2, 0, 130] = 1
    z[0, 0, 0, 149] = 1

    z[1, 2, 0, 90] = 1
    z[1, 1, 0, 85] = 1
    z[1, 2, 0, 10] = 1
    z[1, 1, 0, 20] = 1
    z[1, 0, 0, 50] = 1
    z[1, 1, 0, 60] = 1
    z[1, 0, 0, 120] = 1
    z[1, 1, 0, 130] = 1
elif overlap_flag == 3:
    # overlap version
    z[0, 0, 0, 10] = 1
    z[0, 1, 0, 20] = 1
    z[0, 2, 0, 45] = 1
    z[0, 1, 0, 69] = 1
    z[0, 0, 0, 80] = 1
    z[0, 2, 0, 110] = 1
    z[0, 1, 0, 140] = 1

    z[1, 2, 0, 10] = 1
    z[1, 1, 0, 20] = 1
    z[1, 0, 0, 50] = 1
    z[1, 1, 0, 60] = 1
    z[1, 2, 0, 90] = 1
    z[1, 0, 0, 130] = 1
    z[1, 1, 0, 110] = 1

# CHANGE: rename to avoid conflict with a defined variable later
p_w_ta_d0 = F.conv_transpose2d(z, motifs).cpu()
# CHANGE: use (-1) as a shape to let it infer the size
print(p_w_ta_d0.shape)
p_w_ta_d0 = p_w_ta_d0.view(-1)

# CHANGE: don't sample but rather "get the average"
data = 1 * p_w_ta_d0
N = data.sum()

plt.imshow(data.reshape(-1, Td + ntr - 1).cpu())


# java -jar sequence-mining/target/sequence-mining-1.0.jar -i 100 -f ism_data.dat -v -l INFO
def save_ism_data(data):
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


save_ism_data(data)

#Add 3 columns of zero to avoid exceeding data bound
ism_data = data.reshape(-1,Td+ntr-1).cpu().numpy()
non_num_data = ism_data
for i in range(ism_data.shape[0]):
    non_num_data[i, :] = np.where(ism_data[i, :]> 0, i+1 if i < 20 else i-19, 0)
#     [275, 300]
# seq = [[366],[322],[97]]
# seq = [[366],[322],[97]]
seq = [[247, 272, 295, 297, 342],[366],[275, 300]]

original_data = data.reshape(nw*nd,ntr+Td-1).cpu().numpy()
tem_zero = np.zeros((non_num_data.shape[0], 3))
non_num_data = np.concatenate((non_num_data, tem_zero), axis=1)


def return_col_raw(num):
    cor = []
    col_index = num%ntr
    if col_index == 0:
        col_index = ntr - 1
        raw_index = int(num/ntr) - 1
    else:
        col_index = col_index - 1
        raw_index = int(num/ntr)
    cor.append(raw_index)
    cor.append(col_index)
    return cor


def initialize_motif():
    init_motif = np.ones((nz,1,nw,ntr))
    step = 100
    for i in range(nd):
        cur_raw = i*nw
        cur_col = 0
        while cur_col <= (non_num_data.shape[1] - ntr + 1):
            tem_data = original_data[cur_raw:cur_raw+nw,cur_col:cur_col+ntr]
            for i in range(len(seq)):
                tem_seq = seq[i]
                cur_motif = init_motif[i,0,:,:]
                for sub_seq in tem_seq:
                    tem_cor = return_col_raw(sub_seq)
                    cur_motif[tem_cor[0],tem_cor[1]] += tem_data[tem_cor[0],tem_cor[1]]
                init_motif[i,0,:,:] = cur_motif
            cur_col += step
    return init_motif

init_motif = np.ones((nz,1,nw,ntr))
init_motif = initialize_motif()
init_motif = torch.from_numpy(init_motif).cpu()
init_motif = init_motif.type_as(torch.ones(nd,nz,1,Td).cpu())


def p_w_ta_d(z, motifs):
    t = F.conv_transpose2d(z, motifs)
    return t.view(-1)


# ADD: introduce some variables
prior0 = 0.1 * N / nd / nz / Td
prior1 = 0.1 * N / nz / nw / ntr


# randinit = 0

def model(data):
    s0 = (nd, nz, 1, Td)
    s1 = (nz, 1, nw, ntr)
    alpha0 = torch.ones(*s0).cpu()
    alpha1 = torch.ones(*s1).cpu()
    z = pyro.sample("latent0", pdist.Dirichlet(concentration=alpha0.view(nd, -1)))
    motifs = pyro.sample("latent1", pdist.Dirichlet(concentration=alpha1.view(nz, -1)))

    z = z.reshape(*s0)
    motifs = motifs.reshape(*s1)
    p = p_w_ta_d(z, motifs)
    with pyro.iarange("data", len(data)):
        zts = pyro.sample("zts", pdist.Categorical(probs=z))
        pyro.sample("observe", pdist.Multinomial(probs=p), obs=data)


# Give the initialization

step_motif_count = 0
tem_motif = []


def guide(data):
    qalpha0 = pyro.param("qalpha0", torch.ones(nd, nz, 1, Td).cpu(), constraint=constraints.positive)  # z_ts table
    global step_motif_count
    if flag_ISM:
        qalpha1 = pyro.param("qalpha1", init_motif, constraint=constraints.positive)  # motif
        if step_motif_count % 5 == 0:
            tem_motif.append(qalpha1)
    else:
        qalpha1 = pyro.param("qalpha1", torch.ones(nz, 1, nw, ntr).cpu(), constraint=constraints.positive)  # motif
        if step_motif_count % 5 == 0:
            tem_motif.append(qalpha1)

    #     CHANGE: use the fact that dirichlet can draw independant dirichlets
    pyro.sample("latent0", pdist.Dirichlet(concentration=qalpha0.view(nd, -1)))
    pyro.sample("latent1", pdist.Dirichlet(concentration=qalpha1.view(nz, -1)))


def show_motifs():
    infer_motifs = np.load("./mutu_data/qalpha1.npy")
    for i in range(nz):
        plt.figure(i)
        locals()['infer_motif' + str(i)] = infer_motifs[i, 0, :, :]
        plt.imshow(-locals()['infer_motif' + str(i)], cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.show()


def show_real_motifs():
    for i in range(nz):
        plt.figure(i)
        locals()['real_motif' + str(i)] = motifs[0, 0, :, :].cpu().numpy()
        plt.imshow(-locals()['real_motif' + str(i)], cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.show()


# CHANGE: change adam params
adam_params = {"lr": 0.005}
optimizer = pyro.optim.Adam(adam_params)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

n_steps = 100
# data_cuda = data.cpu()

for step in range(n_steps):
    loss = svi.step(data)
    print(loss)

# CHANGE: change only at the end
np.save(file="./mutu_data/qalpha0.npy", arr=pyro.param("qalpha0").detach().cpu().numpy())
np.save(file="./mutu_data/qalpha1.npy", arr=pyro.param("qalpha1").detach().cpu().numpy())

# ADD: quick plot before exhaustive plot
loaded = np.load("./mutu_data/qalpha1.npy")
plt.imshow(-loaded.reshape(-1, ntr), cmap="gray")
plt.show()

for i in range(nz):
    # plt.imshow(loaded[i].squeeze())
    print(loaded[i].sum())
    # plt.show()














def cal_KL():
    KL = []
    infer_motif0 = loaded[0,0,:,:]
    infer_motif1 = loaded[1,0,:,:]
    infer_motif2 = loaded[2,0,:,:]
    norm_infer_motif0 = infer_motif1/infer_motif1.sum()
    norm_infer_motif1 = infer_motif2/infer_motif2.sum()
    norm_infer_motif2 = infer_motif0/infer_motif0.sum()
    for n in range(nz):
        temKL = 0
        real_motif = motifs[n,0,:,:].cpu()
        norm_real_motif = real_motif/real_motif.sum()
        for i in range(nw):
            for j in  range(ntr):
                if norm_real_motif[i,j] == 0:
                    temKL += 0
                else:
                    temKL += norm_real_motif[i,j] * np.log(norm_real_motif[i,j]/ locals()['norm_infer_motif'+str(n)][i,j])
        KL.append(temKL)
    mean_KL = np.sum(KL)/nz
    print(mean_KL)
    return mean_KL


def cal_median_KL(infer_motif):
    KL = []
    infer_motif0 = infer_motif[0,0,:,:]
    infer_motif1 = infer_motif[1,0,:,:]
    infer_motif2 = infer_motif[2,0,:,:]
    norm_infer_motif0 = infer_motif1/infer_motif1.sum()
    norm_infer_motif1 = infer_motif2/infer_motif2.sum()
    norm_infer_motif2 = infer_motif0/infer_motif0.sum()
    normalizer = 0
    for n in range(nz):
        temKL = 0
        real_motif = motifs[n,0,:,:].cpu()
        norm_real_motif = real_motif/real_motif.sum()
        for i in range(nw):
            for j in  range(ntr):
                if norm_real_motif[i,j] == 0:
                    temKL += 0
                else:
                    normalizer += norm_real_motif[i,j] +  locals()['norm_infer_motif'+str(n)][i,j]
                    temKL += norm_real_motif[i,j] * (np.log(norm_real_motif[i,j]/ locals()['norm_infer_motif'+str(n)][i,j]))
        temKL = temKL/normalizer
        KL.append(temKL)
    mean_KL = np.sum(KL)/nz
    print(mean_KL)
    return mean_KL










final_data = []
for i in range(len(tem_motif)):
    tem = tem_motif[i]
    tem = tem.cpu().detach().numpy()
    tem_rec = cal_median_KL(tem)
    final_data.append(tem_rec.item())







print(final_data)


show_motifs()


for i in range(len(tem_motif)):
    tem = tem_motif[i]
    tem = tem.cpu().detach().numpy()
    cal_median_KL(tem)


show_motifs()


