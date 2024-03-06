
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import timeit

# paths to files
noise_csv_path = 'data/chunk1/chunk1.csv'
noise_sig_path = 'data/chunk1/chunk1.hdf5'
eq1_csv_path = 'data/chunk2/chunk2.csv'
eq1_sig_path = 'data/chunk2/chunk2.hdf5'


# csv file to dataframe

noise = pd.read_csv(noise_csv_path)
earthquakes_1 = pd.read_csv(eq1_csv_path)

chunk_name = earthquakes_1 # select chunk of data
data_start = 0 # select start of data rows you want to pull from that chunk
data_end = 12000 # select end of data rows you want to pull from that chunk
data_interval = 2000 # select interval you'd like to pull (smaller interval with more loops may run faster)
eqpath = eq1_sig_path # select path to data chunk

img_save_path = 'images/big_data_random/waves_long/'


## Make images

eqlist = chunk_name['trace_name'].to_list()

# randomly choose with 1200 interval
eqlist = np.random.choice(eqlist,12000,replace=False) 

starts = list(np.linspace(data_start, data_end - data_interval, int((data_end - data_start)/data_interval)))
ends = list(np.linspace(data_interval, data_end, int((data_end - data_start)/data_interval)))
set = str(chunk_name)

count = 0
for n in range(0,len(starts)):
    traces = eqlist[int(starts[n]):int(ends[n])]
    path = eqpath
    count += 1
    
    def make_images(i):
        # retrieving selected waveforms from the hdf5 file:
        try:
            dtfl = h5py.File(path, 'r')            
            dataset = dtfl.get('data/'+str(traces[i]))
            # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
            data = np.array(dataset)
            print('working on ' + set + ' waveform ' +str(traces[i]) + ' chunk '+str(count) + ' number ' +str(i))
            
            fig, ax = plt.subplots(figsize=(3,2))
            ax.plot(np.linspace(0,60,6000),data[:,2],color='k',linewidth=1)
            ax.set_xlim([0,60])
            ax.axis('off')
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.savefig(img_save_path+traces[i]+'.png',bbox_inches='tight',dpi=50)
            plt.close()
            
            fig, ax = plt.subplots(figsize=(3,2))
            ax.specgram(data[:,2],Fs=100,NFFT=256,cmap='gray',vmin=-10,vmax=25);
            ax.set_xlim([0,60])
            ax.axis('off')
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.savefig(img_save_path+traces[i]+'.png',bbox_inches='tight',transparent = True,pad_inches=0,dpi=50)
            plt.close()
            
        except:
            print('String index out of range')
    
    # create images for selected data (runs in parallel using joblib)
    start = time.time()
    print(start)
    Parallel(n_jobs=-2)(delayed(make_images)(i) for i in range(0,len(traces))) # run make_images loop in parallel on all but 2 cores for each value of i
    end = time.time()
    print(f'Took {end-start} s')
