import time
import os
import numpy as np
import sys
# Silence caffe network loading output. Must be set before importing caffe
os.environ["GLOG_minloglevel"] = '3'
import caffe


path = 'fifopipe'
def attachBF16():
    fifo = open(path, 'w')
    fifo.write('A\n')
    fifo.close()
    time.sleep(3)


def attachFMA_BF16_FP32_WU():
    fifo = open(path, 'w')
    fifo.write('F\n')
    fifo.close()
    time.sleep(3)


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

solver.restore(sys.argv[2])

# This training process is made with the PinTool
epochs = 30
niter = (400 * epochs) + 1 # One resnet epoch is equal to 4000 steps because the batch size is 64
take_snapshot = 100

training_loss = np.zeros(niter - 10800) # 1200 is the last generated snapshot to use at the restore
batches = np.zeros(niter - 10800)


attachFMA_BF16_FP32_WU_BN()


for i in range(10801, niter):
    solver.step(10)
    training_loss[i-10801] = solver.net.blobs['loss'].data
    batches[i-10801] = i * 10
    if(i%take_snapshot==0):
        solver.snapshot()
        np.savetxt('training_v9.csv', np.c_[batches,training_loss], delimiter=',')
    print batches[i-10801],',',training_loss[i-10801]
np.savetxt('training_v9.csv', np.c_[batches,training_loss], delimiter=',')
solver.snapshot()
# To measure elapsed time
end = time.time()
print('Elapsed time during execution in minutes = ' + str((end - start)/60))
print 'end ...'
