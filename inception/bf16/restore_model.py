import time
import os
import numpy as np
import sys
# Silence caffe network loading output. Must be set before importing caffe
os.environ["GLOG_minloglevel"] = '3'
import caffe

path = 'fifopipe'
def attachFMA_BF16_FP32_WU_BN():
    fifo = open(path, 'w')
    fifo.write('E\n')
    fifo.close()
    time.sleep(3)

# To measure elapsed time
start = time.time()
caffe.set_mode_cpu()

solver_path = sys.argv[1]
solver = None
solver = caffe.SGDSolver(solver_path)

solver.restore(sys.argv[2]) # sys.argv[3] contains the path to .solverstate file

epochs = 16
niter = (400 * epochs) + 1  # One ResNet epoch is equal to 4000 steps because
                            # the batch size is 64
take_snapshot = 100     # Take snapshot each quarter epoch.

lastBatch = 4800

training_loss = np.zeros((niter - lastBatch)+1) # lastBatch is the last generated snapshot to use at the restore
batches = np.zeros((niter - lastBatch)+1)
n_steps = 10 # Number of steps to advance each iteration

attachFMA_BF16_FP32_WU_BN()

for i in range(lastBatch+1, niter+1):
    solver.step(n_steps)
    training_loss[i-(lastBatch+1)] = solver.net.blobs['loss3/loss'].data
    batches[i-(lastBatch+1)] = i * n_steps
    print batches[i-(lastBatch+1)],',',training_loss[i-(lastBatch+1)]
    if(i%take_snapshot==0):
        np.savetxt('training_v4.csv', np.c_[batches,training_loss], delimiter=',')
        solver.snapshot()
np.savetxt('training_v4.csv', np.c_[batches, training_loss], delimiter=',')
solver.snapshot()

# To measure elapsed time
end = time.time()
print('Elapsed time during execution in minutes = ' + str((end - start)/60))
print 'end ...'
