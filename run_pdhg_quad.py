import scipy.io
from pdhg_quadratic import pd_nonlocal_HSI
import timeit
mat_file_path = "data_rand.mat"
mat = scipy.io.loadmat(mat_file_path)
f = mat['H']
W = mat['W_binary_10']
mu = 1e-5
lamda = 1e6
tao = 10.0
sigma = 1.0/tao/4.0/10.0
endmem_rand = mat['endmem_rand']
sigma = sigma/3
theta = 1
tol = -1e-2
iter_stop = 1.1
innerloop = 10
outerloop = 10
output_path = "data/quad_hybrid_output_"
# endmem = endmem_rand
# image = f
start_time = timeit.default_timer()
error = pd_nonlocal_HSI(f,W,mu,endmem_rand,lamda,tao,sigma,theta,tol,iter_stop,innerloop,outerloop,output_path)
print(timeit.default_timer() - start_time),"total execution time"
