"""This file is made to process the raw AASP training and tesing files
The label is provided by annotation2, e.g. script02_sid.txt
All the training and testing data will be cut in to 5-sec mono time series
training data will have 1-2 events overlaping
"""

from utils import *

route = '/mnt/d/Downloads/AASP_train/annotation2/'
with open(route + 'script01_sid.txt', 'r') as d:
    # print(type(d.read()))
    d1 = d.read()
with open(route + 'script01_sid.txt', 'r') as d:
    d2 = d.read()
with open(route + 'script01_sid.txt', 'r') as d:
    d3 = d.read()
# make the string into list of events
l1, l2, l3 = d1.split('\n'), d2.split('\n'), d3.split('\n')
l1 = [l.split('\t') for l in l1[:-1]]  # [:-1] to remove the '' in the end
l2 = [l.split('\t') for l in l2[:-1]]
l3 = [l.split('\t') for l in l3[:-1]]

# get the data from wave files
soundtrack = []
route = '/mnt/d/Downloads/AASP_train/bformat/'
track_name = [route + 'script0'+str(i1)+'-0'+str(i2)+'.wav' for i1 in range(1, 4) for i2 in range(1, 5)]
for i in track_name:
    samp_freq, _, d = readwav(i)  # samp_freq is float, d is a numpy array
    d = (d / np.linalg.norm(d)).squeeze()  # d is 1-d time series
    soundtrack.append(d)

# get the indices for cut and label the data
sec = 10  # 10 seconds long data
# for ii in [l1, l2, l3]:
mark = np.array([i[0:2] for i in l1]).astype('double')*samp_freq
ind_mark = mark.astype('int')
trunc_size = sec*samp_freq
for i in range(36):
    ind1 = ind_mark[i-1, 1] if i >0 else 0
    ind2 = ind_mark[i, 0]
    # start should between ind1 and ind2, ends should between ind1_end and ind2_end
    start = int((ind1 + ind2)/2)
    ends = start + trunc_size
    sect_ind = bisect.bisect(ind_mark.reshape(-1), ends)
    if sect_ind % 2 == 0:  # This is OK to choose
        samp1 = soundtrack[0][start:ends]
    else:  # ends in the range that will cut the event
        if ends - ind_mark[sect_ind][0] > ends- ind_mark[sect_ind][1]:  # ends closer to the right(bigger number)
            ends = ind_mark[sect_ind][1] +1  # include the tail event
            start = ends - trunc_size
            if ind1 < start <ind2:  # check the updated start available
                samp1 = soundtrack[0][start:ends]

        else: # ends closer to the right(bigger number)


