import numpy
import igl
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
import time

from jaxtyping import Shaped, jaxtyped
from scipy.sparse import diags,coo_matrix
from scipy.sparse import csc_matrix as sp_csc
from typeguard import typechecked


USE_TORCH_SPARSE = True ## This uses TORCH_SPARSE instead of TORCH.SPARSE

# This four are mutually exclusive
USE_CUPY = False  ## This uses CUPY LU decomposition on GPU
USE_CHOLESPY_GPU = True  ## This uses cholesky decomposition on GPU
USE_CHOLESPY_CPU = False  ## This uses cholesky decomposition on CPU
USE_SCIPY = False ## This uses CUPY LU decomposition on CPU



# If USE_SCIPY = True, wether or not to use enhanced backend
USE_SCIKITS_UMFPACK = False  ## This uses UMFPACK backend for scipy instead of naive scipy.

if USE_CHOLESPY_GPU or USE_CHOLESPY_CPU:
    from cholespy import CholeskySolverD, MatrixType

if USE_CUPY and torch.cuda.is_available():
    from cupyx.scipy.sparse.linalg import spsolve_triangular
    from cupyx.scipy.sparse import csr_matrix
    import cupy
    from torch.utils.dlpack import to_dlpack, from_dlpack

if USE_SCIPY:
    if  USE_SCIKITS_UMFPACK:
        # This is a bit slower in practice
        # https://stackoverflow.com/questions/64401503/is-there-a-way-to-further-improve-sparse-solution-times-using-python
        from scikits.umfpack import splu as scipy_splu
    else:
        import scipy.sparse.linalg as lg
        lg.use_solver(useUmfpack=False)
        # Slight performance gain with True
        # conda install -c conda-forge scikit-umfpack
        # forward pass goes from 0.038 to 0.036
        # assumeSortedIndices=True Does not bring any boost
        from scipy.sparse.linalg import splu as scipy_splu
        from scipy.sparse.linalg import spsolve_triangular, spsolve


print("torch sparse")

if USE_TORCH_SPARSE:
    import torch_sparse

print("imported torch sparse")


USE_UGLY_PATCH_FOR_CUPY_ERROR = False


class SparseMat:
    '''
    Sparse matrix object represented in the COO format
    Refacto : consider killing this object, byproduct of torch_sparse instead of torch.sparse (new feature)
    '''

    @staticmethod
    def from_M(M,ttype):
        return SparseMat(M[0],M[1],M[2],M[3],ttype)

    @staticmethod
    def from_coo(coo,ttype):
        inds = numpy.vstack((coo.row,coo.col))
        return SparseMat(inds,coo.data,coo.shape[0],coo.shape[1],ttype)

    def __init__(self,inds,vals,n,m,ttype):
        self.n = n
        self.m = m
        self.vals = vals
        self.inds = inds
        assert(inds.shape[0] == 2)
        assert(inds.shape[1] == vals.shape[0])
        assert(np.max(inds[0,:]) <= n)
        assert(np.max(inds[1,:] <= m))
        #TODO figure out how to extract the I,J,V,m,n from this, then load a COO mat directly from npz
        #self.coo_mat = coo_matrix((cupy.array(self.vals), (cupy.array(self.inds[0,:]), cupy.array(self.inds[1,:]))))
        self.vals = torch.from_numpy(self.vals).type(ttype).contiguous()
        self.inds = torch.from_numpy(self.inds).type(torch.int64).contiguous()

    def transpose(self):
        return SparseMat(
            self.inds[[1,0],:].cpu().numpy(),
            self.vals.cpu().numpy(),
            self.m,
            self.n,
            self.vals.dtype,
        ).to(self.vals.device)

    def to_coo(self):
        return coo_matrix((self.vals, (self.inds[0,:], self.inds[1,:])), shape = (self.n, self.m))

    def to_csc(self):
        return sp_csc((self.vals, (self.inds[0,:], self.inds[1,:])), shape = (self.n, self.m))

    def to_cholesky(self):
        return CholeskySolverD(self.n, self.inds[0,:], self.inds[1,:], self.vals, MatrixType.COO)

    def to(self,device):
        new_matrix = SparseMat(
            self.inds.cpu().numpy(),
            self.vals.cpu().numpy(),
            self.n,
            self.m,
            self.vals.dtype,
        )
        new_matrix.inds = new_matrix.inds.to(device)
        new_matrix.vals = new_matrix.vals.to(device)

        return new_matrix

    def pin_memory(self):
        return
        # self.vals.pin_memory()
        # self.inds.pin_memory()

    def multiply_with_dense(self, dense):
        if USE_TORCH_SPARSE:
            res = torch_sparse.spmm(self.inds,self.vals, self.n, self.m, dense)
            # 1000 for loop on the above line takes 0.13 sec. Fast but annoying to have this dependency
        else:
            # Somehow this is not implemented for now?
            # res = torch.smm(torch.sparse_coo_tensor(self.inds,self.vals) , (dense.float())).to_dense().to(dense.device)
            # 1000 for loop on the above line takes 10 sec on the CPU. It is not implemented on gpu yet Slower but no dependency
            if self.vals.device.type == 'cpu':
                tensor_zero_hack  = torch.FloatTensor([0]).double() # This line was somehow responsible for a nasty NAN bug
            else:
                tensor_zero_hack  =  torch.cuda.FloatTensor([0]).to(dense.get_device()).double()
            # beware with addmm, it is experimental and gave me a NaN bug!
            res = torch.sparse.addmm(tensor_zero_hack, torch.sparse_coo_tensor(self.inds.double(),self.vals.double()) , (dense.double())).type_as(self.vals)
            # 1000 for loop on the above line takes 0.77 sec. Slower but no dependency
        return res.contiguous()

class PoissonSystemMatrices:
    '''
    Holds the matrices needed to perform gradient and Poisson computations for 2D vertices.
    Logic: This class holds everything needed to compute the Poisson Solver.
    Refacto: Merge with Poisson Solver. Only accept SparseMat representation.
    '''

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        V: Shaped[ndarray, "num_vertex 2"],
        F: Shaped[ndarray, "num_face 3"],
        grad: SparseMat,
        rhs: SparseMat,
        w: Shaped[ndarray, "num_face 2 2"],  # Updated for 2D local basis
        ttype,
        is_sparse: bool = True,
        lap=None,
        cpuonly: bool = False,
    ) -> None:
        """
        Constructs an instance holding matrices used to formulate and solve Poisson's equation.

        Args:
            V: Vertex coordinates (2D)
            F: Vertex indices constituting each triangle
            grad: Gradient operator
            rhs: Right-hand side of the Poisson equation
            w: Local basis (2D)
            ttype: Type of tensor (e.g., float, double)
            is_sparse: Indicates if the matrices are sparse
            lap: Laplacian operator
            cpuonly: If true, the solver will be computed on the CPU
        """
        self.dim = 2  # Dimension is now 2
        self.is_sparse = is_sparse
        self.w = w
        self.rhs = rhs
        self.igl_grad = grad
        self.ttype = ttype
        self.__splu_L = None
        self.__splu_U = None
        self.__splu_perm_c = None
        self.__splu_perm_r = None
        self.lap = lap
        self.__V = V
        self.__F = F
        self.cpuonly = cpuonly
        self.cpu_splu = None

    def create_poisson_solver(self):
        return PoissonSolver(self.igl_grad, self.w, self.rhs, None, self.lap)

    def create_poisson_solver_from_splu_old(self, lap_L, lap_U, lap_perm_c, lap_perm_r):
        w = torch.from_numpy(self.w).type(self.ttype)
        lap = None
        my_splu = None
        if not self.cpuonly:
            if USE_CUPY:
                my_splu = MyCuSPLU(lap_L, lap_U, lap_perm_c, lap_perm_r)
            else:
                if self.lap is not None:
                    lap = self.lap
                else:
                    my_splu = MyCuSPLU_CPU(lap_L, lap_U, lap_perm_c, lap_perm_r)
        else:
            if self.lap is not None:
                my_splu = scipy_splu(self.lap)
            else:
                raise ValueError("Laplacian operator is required.")

        return PoissonSolver(self.igl_grad, w, self.rhs, my_splu, lap)

    def compute_poisson_solver_from_laplacian(self, compute_splu=True):
        self.compute_laplacian()
        if compute_splu:
            self.compute_splu()
        return self.create_poisson_solver_from_splu(self.__splu_L, self.__splu_U, self.__splu_perm_c, self.__splu_perm_r)

    def compute_laplacian(self):
        if self.lap is None:
            self.lap = igl.cotmatrix(self.__V, self.__F)
            self.lap = self.lap[1:, 1:]  # Exclude boundary conditions if applicable
            self.lap = SparseMat.from_coo(self.lap.tocoo(), torch.float64)

        # Additional check specific to 2D
        if isinstance(self.lap, PoissonSystemMatrices) and self.lap.vals.shape[0] == self.__V.shape[0]:
            raise ValueError("This should not happen, the fix is to remove a column and row of the Laplacian.")
            self.lap = self.lap[1:, 1:]

        return self.lap

    def compute_splu(self):
        print("Computing SPLU...")
        if self.cpu_splu is None:
            s = scipy_splu(self.lap)
            self.cpu_splu = s
            self.__splu_L = s.L
            self.__splu_U = s.U
            self.__splu_perm_c = s.perm_c
            self.__splu_perm_r = s.perm_r
        return self.__splu_L, self.__splu_U, self.__splu_perm_c, self.__splu_perm_r

    def get_new_grad(self):
        grad = self.igl_grad.to_coo()
        self.igl_grad = SparseMat.from_M(_convert_sparse_igl_grad_to_our_convention(grad.tocsc()), torch.float64)
        return self.igl_grad

def _convert_sparse_igl_grad_to_our_convention(input):
    '''
    The grad operator computed from igl.grad() results in a matrix of shape (2*#tri x #verts) for 2D.
    It is packed such that all the x-coordinates are placed first, followed by y.
    '''

    assert isinstance(input, sp.csc_matrix), 'Input should be a scipy csc sparse matrix'
    T = input.tocoo()

    r_c_data = np.hstack((T.row[..., np.newaxis], T.col[..., np.newaxis],
                          T.data[..., np.newaxis]))  # Horizontally stack row, col and data arrays
    r_c_data = r_c_data[r_c_data[:, 0].argsort()]  # Sort along the row column

    # Separate out x and y blocks
    L = T.shape[0]
    Tx = r_c_data[:L, :]
    Ty = r_c_data[L:2 * L, :]

    # Align the y rows with x so that they too start from 0
    Ty[:, 0] -= Ty[0, 0]

    # 'Stretch' the x, y rows so that they can be interleaved.
    Tx[:, 0] *= 2
    Ty[:, 0] *= 2

    # Interleave the y into x
    Ty[:, 0] += 1

    Tc = np.zeros((input.shape[0] * 2, 3))
    Tc[::2] = Tx
    Tc[1::2] = Ty

    indices = Tc[:, :-1].astype(int)
    data = Tc[:, -1]

    return (indices.T, data, input.shape[0], input.shape[1])
    

class PoissonSolver:
    '''
    An object to compute gradients and solve Poisson equations for 2D vertices.
    '''

    def __init__(
        self,
        grad,
        W,
        rhs,
        my_splu,
        lap=None,
    ):
        """
        Initializes the PoissonSolver with the necessary parameters.
        """
        self.W = torch.from_numpy(W).double()  # Ensure W is suitable for 2D
        self.grad = grad
        self.rhs = rhs
        self.my_splu = my_splu
        self.lap = lap
        self.sparse_grad = grad
        self.sparse_rhs = rhs

    def to(self, device):
        self.W = self.W.to(device)
        self.sparse_grad = self.sparse_grad.to(device)
        self.sparse_rhs = self.sparse_rhs.to(device)
        if USE_CUPY or USE_CHOLESPY_GPU:
            self.lap = self.lap.to(device)
        return self

    def jacobians_from_vertices(
        self,
        V: Shaped[Tensor, "batch_size num_vertex 2"],  # Change 3 to 2 for 2D
    ) -> Shaped[Tensor, "batch_size num_face 2 2"]:  # Change 3 to 2 for 2D
        res = _multiply_sparse_2d_by_dense_3d(self.sparse_grad, V).type_as(V)
        res = res.unsqueeze(2)
        return res.view(V.shape[0], -1, 2, 2).transpose(2, 3)  # Change 3 to 2 for 2D

    def restrict_jacobians(self, D):
        assert isinstance(D, torch.Tensor) and len(D.shape) in [3, 4]
        assert D.shape[-1] == 2 and D.shape[-2] == 2  # Change 3 to 2 for 2D
        assert isinstance(self.W, torch.Tensor) and len(self.W.shape) == 3
        assert self.W.shape[-1] == 2 and self.W.shape[-2] == 2  # Ensure 2D

        if len(D.shape) == 4:
            DW = torch.einsum("abcd,bde->abce", (D, self.W.type_as(D)))
        else:
            DW = torch.einsum("abcd,bde->abce", (D.unsqueeze(0), self.W)).squeeze(0)

        if len(DW.shape) > 4:
            DW = DW.squeeze(0)
        return DW

    def restricted_jacobians_from_vertices(self, V):
        return self.restrict_jacobians(self.jacobians_from_vertices(V))

    def solve_poisson(self, jacobians): 
        assert(len(jacobians.shape) == 4)
        assert(jacobians.shape[2] == 2 and jacobians.shape[3] == 2)  # Change 3 to 2 for 2D

        if self.my_splu is None:
            if isinstance(self.lap, SparseMat):
                if USE_CHOLESPY_CPU or USE_CHOLESPY_GPU:
                    self.my_splu = self.lap.to_cholesky()
                else:
                    self.my_splu = scipy_splu(self.lap.to_coo())
            else:
                self.my_splu = scipy_splu(self.lap)

        sol = _predicted_jacobians_to_vertices_via_poisson_solve(
            self.my_splu,
            self.sparse_rhs,
            jacobians.transpose(2, 3).reshape(jacobians.shape[0], -1, 2, 1).squeeze(3).contiguous(),
        )
        c = torch.mean(sol, axis=1).unsqueeze(1)  # Beware the predicted mesh is centered here.
        return sol - c

    def pin_memory(self):
        return

@jaxtyped(typechecker=typechecked)
def poisson_system_matrices_from_mesh(
    V: Shaped[ndarray, "num_vertex 2"],  # Change 3 to 2 for 2D
    F: Shaped[ndarray, "num_face 3"],
    dim: int=2,  # Set dimension to 2 for 2D
    ttype=torch.float64,
    is_sparse: bool=True,
    cpuonly: bool=False,
) -> PoissonSystemMatrices:
    '''
    Computes matrices involving in Poisson's equation from the given 2D mesh.
    
    Args:
        V: Vertex coordinates (2D)
        F: Vertex indices constituting each triangle
        dim: Set to 2 for 2D
        ttype: Type of tensor (e.g., float, double)
        is_sparse: for now always true
    
    Returns:
        An instance of PoissonMatricese holding the computed matrices
    '''

    assert type(dim) == int and dim == 2, f'Only two dimensional meshes are supported'
    assert type(is_sparse) == bool
    vertices = V
    faces = F

    # Compute gradients
    grad = igl.grad(vertices, faces)  # (3F, V) remains as is since grad may still represent 3D gradients

    # Mass matrix
    mass = _get_mass_matrix(vertices, faces, is_sparse)  # (3F, 3F)
    grad = grad[:-grad.shape[0]//3, :]  # Remove z components
    mass = mass[:-mass.shape[0]//3, :-mass.shape[0]//3]  # Adjust for 2D

    # Compute Laplacian and RHS
    laplace = grad.T @ mass @ grad  # (V, 3F) x (3F, 3F) x (3F, V) = (V, V)
    laplace = laplace[1:, 1:]  # Exclude first row and column

    rhs = grad.T @ mass
    b1, b2, _ = igl.local_basis(V, F)
    w = np.stack((b1, b2), axis=-1)

    rhs = rhs[1:, :]

    if is_sparse:
        laplace = laplace.tocoo()
        rhs = rhs.tocoo()
        grad = grad.tocsc()
    else:
        laplace = laplace.toarray()
        rhs = rhs.toarray()
        grad = grad.toarray()

    # Convention conversion + build sparse matrix
    grad = SparseMat.from_M(
        _convert_sparse_igl_grad_to_our_convention(grad),
        torch.float64,
    )

    poissonbuilder = PoissonSystemMatrices(
        V=V,
        F=F,
        grad=grad,
        rhs=SparseMat.from_coo(rhs, torch.float64),
        w=w,
        ttype=ttype,
        is_sparse=is_sparse,
        lap=SparseMat.from_coo(laplace, torch.float64),
        cpuonly=cpuonly,
    )
    
    return poissonbuilder

@jaxtyped(typechecker=typechecked)
def _get_mass_matrix(
    vertices: Shaped[ndarray, "num_vertex 2"],  # Change 3 to 2 for 2D
    faces: Shaped[ndarray, "num_face 3"],
    is_sparse: bool,
):
    d_area = igl.doublearea(vertices, faces)
    d_area = np.hstack((d_area, d_area))  # Adjust for 2D (only x and y)
    if is_sparse:
        return sp_csc(diags(d_area))
    return diags(d_area)

class SPLUSolveLayer(torch.autograd.Function):
    '''
    Implements the SPLU solve as a differentiable layer, with a forward and backward function
    '''

    @staticmethod
    def forward(ctx, solver, b):
        '''
        override forward function
        :param ctx: context object (to keep the lu object for the backward pass)
        :param lu: splu object
        :param b: right hand side, could be a vector or matrix
        :return: the vector or matrix x which holds lu.solve(b) = x
        '''
        assert isinstance(b, torch.Tensor)
        assert b.shape[-1] >= 1 and b.shape[-1] <= 3, f'got shape {b.shape} expected last dim to be in range 1-3'
        b = b.contiguous()
        ctx.solver = solver

        # st = time.time()
        vertices = SPLUSolveLayer.solve(solver, b).type_as(b)
        # print(f"FORWARD SOLVE {time.time() - st}")

        assert not torch.isnan(vertices).any(), "Nan in the forward pass of the POISSON SOLVE"
        return vertices

    def backward(ctx, grad_output):
        '''
        overrides backward function
        :param grad_output: the gradient to be back-propped
        :return: the outgoing gradient to be back-propped
        '''

        assert isinstance(grad_output, torch.Tensor)
        assert grad_output.shape[-1] >= 1 and grad_output.shape[
            -1] <= 3, f'got shape {grad_output.shape} expected last dim to be in range 1-3'
        # when backpropping, if a layer is linear with matrix M, x ---> Mx, then the backprop of gradient g is M^Tg
        # in our case M = A^{-1}, so the backprop is to solve x = A^-T g.
        # Because A is symmetric we simply solve A^{-1}g without transposing, but this will break if A is not symmetric.
        # st = time.time()
        grad_output = grad_output.contiguous()
        grad = SPLUSolveLayer.solve(ctx.solver,
                                          grad_output)
        # print(f"BACKWARD SOLVE {time.time() - st}")
        # At this point we perform a NAN check because the backsolve sometimes returns NaNs.
        assert not torch.isnan(grad).any(),  "Nan in the backward pass of the POISSON SOLVE"

        if USE_CUPY:
            mempool = cupy.get_default_memory_pool()
            pinned_mempool = cupy.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            del ctx.lu
        
        return None, grad

    @staticmethod
    def solve(solver, b):
        '''
        solve the linear system defined by an SPLU object for a given right hand side. if the RHS is a matrix, solution will also be a matrix.
        :param solver: the splu object (LU decomposition) or cholesky object
        :param b: the right hand side to solve for
        :return: solution x which satisfies Ax = b where A is the poisson system lu describes
        '''

        if  USE_CUPY:
            b_cupy = cupy.fromDlpack(to_dlpack(b))
            with cupy.cuda.Device(solver.device()):
                # this will hold the solution
                sol = cupy.ndarray(b_cupy.shape)
                for i in range(b_cupy.shape[2]):  # b may have multiple columns, solve for each one
                    b2d = b_cupy[..., i]  # cupy.expand_dims(b_cpu[...,i],2)
                    s = solver.solve(b2d.T).T
                    sol[:, :, i] = s
            # # # convert back to torch
            res = from_dlpack(sol.toDlpack())
            # np.save("res_gpu.npy", res.cpu().numpy())
            # res = torch.zeros((1, 6889, 3), device=b.device)+  torch.mean(b)

            return res.type_as(b.type())

        elif USE_SCIPY:
            #only CPU
            # st = time.time()
            assert(b.shape[0]==1), "Need to code parrallel implem on the first dim"
            sol = solver.solve(b[0].double().cpu().numpy())
            res = torch.from_numpy(sol).to(b.device).reshape(b.shape)
            # print(time.time() - st)
            return res.type_as(b).contiguous()

            # Legacy code, I don't understand what is the reason for having a for loop
            # sol = np.ndarray(b.shape)
            # for i in range(b.shape[2]):  # b may have multiple columns, solve for each one
            #     b2d = b[..., i]  # cupy.expand_dims(b_cpu[...,i],2)
            #     s = lu.solve(b2d.double().cpu().float().numpy().T).T
            #     sol[:, :, i] = s
            # res = torch.from_numpy(sol).to(b.device)
            # # np.save("res_cpu.npy", sol)
            # print(f"time {time.time() - st}" )
        elif USE_CHOLESPY_GPU:
            # torch.cuda.synchronize()
            # # st = time.time()
            # assert(b.shape[0]==1), "Need to code parrallel implem on the first dim"
            # b = b.squeeze().double()
            # x = torch.zeros_like(b)
            # solver.solve(b, x)
            # # torch.cuda.synchronize()
            # # print(f"time cholescky GPU {time.time() - st}" )
            # return x.contiguous().unsqueeze(0)
            # st = time.time()
            # print(b.get_device(), b.shape)
            b = b.double().contiguous()
            c = b.permute(1,2,0).contiguous()
            c = c.view(c.shape[0], -1)
            x = torch.zeros_like(c)
            solver.solve(c, x)
            x = x.view(b.shape[1], b.shape[2], b.shape[0])
            x = x.permute(2,0,1).contiguous()
            # torch.cuda.synchronize()
            # print(f"time cholescky GPU {time.time() - st}" )
            return x.contiguous()

        elif USE_CHOLESPY_CPU:
            # st = time.time()
            assert(b.shape[0]==1), "Need to code parrallel implem on the first dim"
            b = b.squeeze()
            b_cpu = b.cpu()
            x = torch.zeros_like(b_cpu)
            solver.solve(b_cpu, x)
            # print(f"time cholescky CPU {time.time() - st}" )
            return x.contiguous().to(b.device).unsqueeze(0)


        return res.type_as(b)

def _predicted_jacobians_to_vertices_via_poisson_solve(Lap, rhs, jacobians):
    '''
    Convert the predictions to the correct convention and feed it to the Poisson solve.
    '''
    def _batch_rearrange_input(input):
        assert isinstance(input, torch.Tensor) and len(input.shape) in [2, 3]
        P = torch.zeros(input.shape).type_as(input)
        if len(input.shape) == 3:
            k = input.shape[1] // 3  # Adjust for 2D input
            P[:, :k, :] = input[:, ::3]
            P[:, k:2 * k, :] = input[:, 1::3]
            # Exclude z component for 2D
        else:
            k = input.shape[0] // 3
            P[:k, :] = input[::3]
            P[k:2 * k, :] = input[1::3]

        return P

    if isinstance(jacobians, list):
        P = _list_rearrange_input(jacobians)
    else:
        P = _batch_rearrange_input(jacobians)

    assert isinstance(P, torch.Tensor) and len(P.shape) in [2, 3]
    assert len(P.shape) == 3

    P = P.double()
    input_to_solve = _multiply_sparse_2d_by_dense_3d(rhs, P)

    return Lap.solve(input_to_solve.reshape(-1)).view(P.shape[0], -1, 2)


@jaxtyped(typechecker=typechecked)
def _multiply_sparse_2d_by_dense_3d(
    mat: SparseMat,
    B: Shaped[Tensor, "num_dense ..."],
) -> Shaped[Tensor, "num_dense ..."]:
    """
    Computes multiplication of a sparse matrix with a batch of dense matrices.
    """
    ret = []
    for i in range(B.shape[0]):
        C = mat.multiply_with_dense(B[i, ...])
        ret.append(C)
    ret = torch.stack(tuple(ret))
    return ret

def _predicted_jacobians_to_vertices_via_poisson_solve(Lap, rhs, jacobians):
    '''
    convert the predictions to the correct convention and feed it to the poisson solve
    '''

    def _batch_rearrange_input(input):
        assert isinstance(input, torch.Tensor) and len(input.shape) in [2, 3]
        P = torch.zeros(input.shape).type_as(input)
        if len(input.shape) == 3:
            # Batched input
            k = input.shape[1] // 3
            P[:, :k, :] = input[:, ::3]
            P[:, k:2 * k, :] = input[:, 1::3]
            P[:, 2 * k:, :] = input[:, 2::3]

        else:
            k = input.shape[0] // 3
            P[:k, :] = input[::3]
            P[k:2 * k, :] = input[1::3]
            P[2 * k:, :] = input[2::3]

        return P

    def _list_rearrange_input(input):
        assert isinstance(input, list) and all([isinstance(x, torch.Tensor) and len(x.shape) in [2, 3] for x in input])
        P = []
        for p in input:
            P.append(_batch_rearrange_input(p))
        return P

    if isinstance(jacobians, list):
        P = _list_rearrange_input(jacobians)
    else:
        P = _batch_rearrange_input(jacobians)

    # return solve_poisson(Lap, rhs, P)
    assert isinstance(P, torch.Tensor) and len(P.shape) in [2, 3]
    assert len(P.shape) == 3

    # torch.cuda.synchronize()
    # st = time.time()
    P = P.double()
    input_to_solve = _multiply_sparse_2d_by_dense_3d(rhs, P)

    out = SPLUSolveLayer.apply(Lap, input_to_solve)

    out = torch.cat([torch.zeros(out.shape[0], 1, out.shape[2]).type_as(out), out], dim=1)  ## Why?? Because!
    out = out - torch.mean(out, axis=1, keepdim=True)

    return out.type_as(jacobians)

def _multiply_sparse_2d_by_dense_3d(mat, B):
    ret = []
    for i in range(B.shape[0]):
        C = mat.multiply_with_dense(B[i, ...])
        ret.append(C)
    ret = torch.stack(tuple(ret))
    return ret

class MyCuSPLU:
    '''
    Implementation of SPLU on the GPU via CuPy for 2D vertices.
    '''
    def __init__(self, L, U, perm_c=None, perm_r=None):
        self.__orgL = L
        self.__orgU = U
        self.L = None
        self.U = None
        self.perm_c = perm_c
        self.perm_r = perm_r
        self.__device = None

    def to(self, device):
        self.__device = device.index
        with cupy.cuda.Device(self.__device):
            self.L = csr_matrix(self.__orgL)
            self.U = csr_matrix(self.__orgU)
        return self

    def device(self):
        return self.__device

    def solve(self, b):
        """ Solve Ax = b for 2D vertices using SuperLU factorization. """
        assert self.__device is not None, "need to explicitly call to() before solving"

        with cupy.cuda.Device(self.__device):
            b = cupy.array(b)
            if self.perm_r is not None:
                b_old = b.copy()
                b[self.perm_r] = b_old

        assert b.device.id == self.__device, "got device " + str(b.device.id) + " instead of " + str(self.__device)

        try:
            c = spsolve_triangular(self.L, b, lower=True, overwrite_b=True)
        except TypeError:
            c = spsolve_triangular(self.L, b, lower=True, overwrite_b=True)

        px = spsolve_triangular(self.U, c, lower=False, overwrite_b=True)

        if self.perm_c is None:
            return px
        px = px[self.perm_c]

        return px


class MyCuSPLU_CPU:
    '''
    Implementation of SPLU on the CPU for 2D vertices.
    '''
    def __init__(self, L, U, perm_c=None, perm_r=None):
        self.__orgL = L
        self.__orgU = U
        self.L = L
        self.U = U
        self.perm_c = perm_c
        self.perm_r = perm_r
        self.__device = 'cpu'

    def to(self, device):
        return self

    def device(self):
        return self.__device

    def solve(self, b):
        """ Solve Ax = b for 2D vertices using SuperLU factorization on CPU. """
        if self.perm_r is not None:
            b_old = b.copy()
            b[self.perm_r] = b_old

        st = time.time()
        try:
            c = spsolve(self.L, b, permc_spec="NATURAL")
        except TypeError:
            c = spsolve(self.L, b, permc_spec="NATURAL")

        px = spsolve(self.U, c, permc_spec="NATURAL")
        print(f"time for spsolve_triangular CPU: {time.time() - st}")

        if self.perm_c is None:
            return px
        px = px[self.perm_c]

        return px
        
