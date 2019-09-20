import time
import os
import numpy as np
import sys
# Silence caffe network loading output. Must be set before importing caffe
os.environ["GLOG_minloglevel"] = '3'
import caffe

path = "fifopipe"

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

# This training process is made with the PinTool
epochs = int(sys.argv[2])
niter = (100 * epochs) + 1 # One AlexNet epoch is equal to 1000 steps because the batch size is 256
take_snapshot = 25

training_loss = np.zeros(niter)
batches = np.zeros(niter)

attachFMA_BF16_FP32_WU_BN()

solver.net.forward()
batches[0] = 0;
training_loss[0] = solver.net.blobs['loss'].data
print batches[0],',',training_loss[0]
j = 0
for i in range(1,niter):
    solver.step(10)
    training_loss[i] = solver.net.blobs['loss'].data
    batches[i] = i * 10
    print batches[i],',',training_loss[i]
    if(i%take_snapshot==0):
        solver.snapshot()
        np.savetxt('training.csv', np.c_[batches,training_loss], delimiter=',')
np.savetxt('training.csv', np.c_[batches,training_loss], delimiter=',')
solver.snapshot()
# To measure elapsed time
end = time.time()
print('Elapsed time during execution in minutes = ' + str((end - start)/60))
print 'end ...'
