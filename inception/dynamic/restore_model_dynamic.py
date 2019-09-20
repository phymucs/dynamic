import time
import os
import numpy as np
import sys
# Silence caffe network loading output. Must be set before importing caffe
os.environ["GLOG_minloglevel"] = '3'
import caffe

path = 'fifopipe'


def attachFMA_BF16():
    fifo = open(path, 'w')
    fifo.write('A\n')
    fifo.close()
    time.sleep(3)


def attachFMA_MP_FP32_WU_BN():
    fifo = open(path, 'w')
    fifo.write('B\n')
    fifo.close()
    time.sleep(3)


def attachFMA_MP_FP32_WU():
    fifo = open(path, 'w')
    fifo.write('C\n')
    fifo.close()
    time.sleep(3)


def attachFMA_MP():
    fifo = open(path, 'w')
    fifo.write('D\n')
    fifo.close()
    time.sleep(3)


def attachFMA_BF16_FP32_WU_BN():
    fifo = open(path, 'w')
    fifo.write('E\n')
    fifo.close()
    time.sleep(3)


def attachFMA_BF16_FP32_WU():
    fifo = open(path, 'w')
    fifo.write('F\n')
    fifo.close()
    time.sleep(3)


def detach():
    fifo = open(path, 'w')
    fifo.write('G\n')
    fifo.close()
    time.sleep(3)


# Exponential Moving Average Calculation
# The first EMA value is calculated as the AVERAGE of the first n values
# n = 5
# alpha = 2/(n+1)
# EMAt = alpha x actualvalue + (1-alpha)xEMAt-1
def ema(n, actualValue, EMAprev):
    alpha = 2.0 / (float(n) + 1.0)
    EMAt = alpha * actualValue + (1-alpha)*EMAprev
    return EMAt


# To measure elapsed time
start = time.time()
caffe.set_mode_cpu()

solver_path = sys.argv[1]
solver = None
solver = caffe.SGDSolver(solver_path)

solver.restore(sys.argv[3]) # sys.argv[3] contains the path to .solverstate file

epochs = 16
niter = (400 * epochs) + 1  # One ResNet epoch is equal to 4000 steps because
                            # the batch size is 64
take_snapshot = 100     # Take snapshot each quarter epoch.

niterToTestEMABF16 = int(sys.argv[2])     # Defines the number of iterations on BF16 before
                            # do a test to see if EMA is improving
                            # Remember that 1 iteration is equal to n_steps
counterAttachBF16 = 0


lastBatch = 4800

training_loss = np.zeros((niter - lastBatch)+1) # lastBatch is the last generated snapshot to use at the restore
batches = np.zeros((niter - lastBatch)+1)


n = 5 # Parameter to calculate the EMA
n_steps = 10 # Number of steps to advance each iteration
flagAttachBF16 = False  # This flag is used to change
                        # between BF16 or BF16WUFP32
ema_change = 0.04   # Value to define when to change the previous flag

j = 0
for i in range(lastBatch+1, niter+1):
    solver.step(n_steps)
    training_loss[i-(lastBatch+1)] = solver.net.blobs['loss3/loss'].data
    if (i - (lastBatch+1)) == n:
        EMA = np.average(training_loss[0:n-1])
    if (i - (lastBatch+1)) > n:
        EMAprev = EMA
        EMA = ema(n, training_loss[i-(lastBatch+1)], EMAprev)
        print 'flagAttachBF16 ', flagAttachBF16, 'EMAVal: ', EMAprev, EMA, ema_change
        if(flagAttachBF16 != True):
            if (EMAprev-EMA) > ema_change:
                print 'Changing to FMA_BF16_FP32_WU_BN ...'
                flagAttachBF16 = True
                attachFMA_BF16_FP32_WU_BN()
        else:
            counterAttachBF16 = counterAttachBF16 + 1
            if (counterAttachBF16 == niterToTestEMABF16):
                if(EMAprev-EMA) > ema_change:
                    counterAttachBF16 = 0
                else:
                    print 'Changing to FMA_MP-FP32_WU_BN ...'
                    flagAttachBF16 = False
                    attachFMA_MP_FP32_WU_BN()
                    counterAttachBF16 = 0
    batches[i-(lastBatch+1)] = i * n_steps
    print batches[i-(lastBatch+1)],',',training_loss[i-(lastBatch+1)]
    if(i%take_snapshot==0):
        np.savetxt('training_v3.csv', np.c_[batches,training_loss], delimiter=',')
        solver.snapshot()
np.savetxt('training_v3.csv', np.c_[batches, training_loss], delimiter=',')
solver.snapshot()

# To measure elapsed time
end = time.time()
print('Elapsed time during execution in minutes = ' + str((end - start)/60))
print 'end ...'
