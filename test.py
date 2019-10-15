"""This file is made to process the raw AASP training and tesing files
The label is provided by annotation2, e.g. script02_sid.txt
All the training and testing data will be cut in to 5-sec mono time series
training data will have 1-2 events overlaping
"""

from utils import *
file ='''20120921room104script1take2
20120921roomDILscript2take1
20120921room104script2take3
20120921roomitltopscript1take2
20120921room104script3take1
20120921roomitltopscript2take2
20120921room203script1take2
20120921roomitltopscript3take2
20120921room203script2take1
20120924room112script1take3
20120921roomDILscript1take1'''
files = file.split('\n')
files.sort()

route = '/mnt/d/Downloads/AASP_test/'
# route = '/home/chenhao1/Hpython/AASP_train/annotation2/'
script = []
for i in files:
    with open(route + i +'_sid.txt', 'r') as d:
        # print(type(d.read()))
        script.append(d.read())

tables = []
for l in script:
    ll = l.split('\n')[:-1]
    tables.append([i.split('\t') for i in ll])

# get the data from wave files
soundtrack = []
route = '/mnt/d/Downloads/AASP_test/'
# route = '/home/chenhao1/Hpython/AASP_test/'
track_name = [route + i +'.wav' for i in files]
for i in track_name:
    samp_freq, _, d = readwav(i)  # samp_freq is float, d is a numpy array
    d = (d / np.linalg.norm(d)).squeeze()  # d is 1-d time series with 2 channels shape of [N, 2]
    soundtrack.append(d)

# get the indices for cut and label the data
sec = 10  # 10 seconds long data
trunc_size = sec*samp_freq
ntracks = len(track_name)
samp_pool = np.random.rand(trunc_size)
label_pool = []
for ii in range(ntracks):
    mark = np.array([i[0:2] for i in tables[ii]]).astype('double')*samp_freq
    ind_mark = mark.astype('int')
    all_labels =  [i[2] for i in tables[ii]]
    st_len, st = soundtrack[ii].shape[0], soundtrack[ii].T  # each soundtrack is shape of [N, 2]

    for i in range(len(tables[ii])):
        ind1 = ind_mark[i-1, 1] if i >0 else 0
        ind2 = ind_mark[i, 0]
        if ind1 + trunc_size > st_len : break  # out of boundry
        # start should between ind1 and ind2, ends should between ind1_end and ind2_end
        start = int((ind1 + ind2)/2)
        ends = start + trunc_size
        sect_ends = bisect.bisect(ind_mark.reshape(-1), ends)
        if sect_ends % 2 == 0:  # This is OK to choose
            samp_pool = np.vstack((samp_pool, st[:, start:ends]))
            sect_start = bisect.bisect(ind_mark.reshape(-1), start)
            for m in range(2):  # because st contains 2 samples
                label_pool.append(all_labels[sect_start // 2: sect_ends // 2])
        else:  # ends in the range that will cut the event
            indx = int((sect_ends - 1)/2)
            if abs(ends - ind_mark[indx][0]) > abs(ends- ind_mark[indx][1]):  # ends closer to the right(bigger number)
                ends = ind_mark[indx][1] +1  # include the tail event
                start = ends - trunc_size
                if ind1 < start <ind2:  # check the updated start available
                    samp_pool = np.vstack((samp_pool, st[:, start:ends]))
                    sect_start = bisect.bisect(ind_mark.reshape(-1), start)
                    sect_ends = bisect.bisect(ind_mark.reshape(-1), ends)
                    for m in range(2):  # because st contains 2 samples
                        label_pool.append(all_labels[sect_start // 2: sect_ends // 2])
            else: # ends closer to the right(bigger number)
                ends = ind_mark[indx][0] - 1  # exclude the tail event
                start = ends - trunc_size
                if ind1 < start <ind2:  # check the updated start available
                    samp_pool = np.vstack((samp_pool, st[:, start:ends]))
                    sect_start = bisect.bisect(ind_mark.reshape(-1), start)
                    sect_ends = bisect.bisect(ind_mark.reshape(-1), ends)
                    for m in range(2):  # because st contains 2 samples
                        label_pool.append(all_labels[sect_start // 2: sect_ends // 2])

X = downsample(samp_pool[1:, :], t_len=150, f_len=80)
Y = label_str2num(l1, l2, l3, label_pool)
# torch.save([X,Y], 'aasp_test.pt')

