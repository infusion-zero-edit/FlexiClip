"""
poisson_system.py

An implementation of Poisson system solver.
"""

from typing import Any, Optional, Tuple

import cholespy
import igl
from jaxtyping import Int, Shaped, jaxtyped
import numpy as np
from numpy import ndarray
from scipy.sparse import (
    csc_matrix,
    coo_matrix,
    diags,
)
import torch
from torch import Tensor
from typeguard import typechecked

####
print("reached 1")
####


from .PoissonSystem import SPLUSolveLayer, SparseMat

####
print("reached 2")
####



class PoissonSystem:
    
    v_src: Shaped[Tensor, "num_vertex 2"]
    """The vertices of the source mesh in 2D"""
    f_src: Shaped[Tensor, "num_face 3"]
    """The faces of the source mesh (triangles)"""
    anchor_inds: Optional[Int[Tensor, "num_handle"]]
    """The indices of the handle vertices used as the constraints"""
    constraint_lambda: float
    """The weight of the constraint term in the system"""
    is_constrained: bool
    """The Boolean flag indicating the presence of the constraint term in the system"""
    device: torch.device
    """The device where the computation is performed"""
    torch_dtype: Any
    """The data type of PyTorch tensors used in Poisson system"""
    is_sparse: bool
    """The flag for enabling computation of sparse tensors"""
    cpu_only: bool
    """The flag for forcing the computation on CPU"""

    # differential operators for unconstrained system
    grad: SparseMat
    """The gradient operator computed using the source mesh"""
    rhs: SparseMat
    """The right-hand side of the Poisson system"""
    w: Shaped[ndarray, "num_face 3 2"]
    """The local basis of the source mesh"""
    L: SparseMat
    """The Laplacian operator computed using the source mesh"""
    L_fac: cholespy.CholeskySolverD
    """The factorization of the Laplacian operator"""
    J: Shaped[Tensor, "num_face 2 2"]
    """The per-triangle Jacobians computed using the source mesh"""

    # differential operators for constrained system
    indicator_matrix: Optional[SparseMat] = None
    """The indicator matrix for the constrained system"""
    indicator_product: Optional[SparseMat] = None
    """The product of the indicator matrix and its transpose"""

    # matrix operators for computing ARAP energy
    L_fac_arap: cholespy.CholeskySolverD
    """The factorization of the Laplacian operator used to compute ARAP energy"""
    CSM: Optional[SparseMat] = None
    """The covariance scatter matrix used to compute per-vertex covariance matrices"""
    arap_energy_type: int = igl.ARAP_ENERGY_TYPE_SPOKES
    """The type of ARAP energy used in the system"""

    deltas: Shaped[Tensor, "num_vertex 2"]
    """The differential coordinates of the source mesh"""
    w_arap: Shaped[Tensor, "num_vertex"]
    """The learnable per-vertex weights for ARAP energy"""
    w_scale: Shaped[Tensor, "num_vertex"]
    """The learnable per-vertex weights for scaling matrices"""

    # additional flags
    no_factorization: bool = False
    """The flag for disabling matrix factorization. Enabled when debugging the module"""

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        v: Shaped[Tensor, "num_vertex 2"],
        f: Shaped[Tensor, "num_face 3"],
        device: torch.device,
        anchor_inds: Optional[Int[Tensor, "num_handle"]] = None,
        constraint_lambda: float = 1.0,
        train_J: bool = True,
        is_sparse: bool = True,
        cpu_only: bool = False,
        no_factorization: bool = False,
    ) -> None:
        """
        Constructor of the Poisson system solver for 2D.
        """
        self.v_src: Shaped[Tensor, "num_vertex 2"] = v
        self.f_src: Shaped[Tensor, "num_face 3"] = f
        self.anchor_inds: Optional[Int[Tensor, "num_handle"]] = anchor_inds
        self.constraint_lambda: float = constraint_lambda
        self.train_J: bool = train_J
        self.device: torch.device = device
        self.torch_dtype = torch.float64  # NOTE: Use double precision for numerical stability
        self.is_sparse: bool = is_sparse
        self.cpu_only: bool = cpu_only
        self.no_factorization = no_factorization

        self._build_poisson_system()

    @jaxtyped(typechecker=typechecked)
    def _build_poisson_system(self) -> None:
        # Check whether the system is constrained
        self.is_constrained = self.anchor_inds is not None
        if self.is_constrained:
            assert self.constraint_lambda > 0.0, (
                f"Invalid weight value: {self.constraint_lambda}"
            )

        # Compute gradient, Laplacian, and right-hand side of the system
        self._compute_differential_operators_constrained()

        # Build the indicator matrix if the system is constrained
        if self.is_constrained:
            self._build_indicator_matrix()

        # Pre-factorize the matrices used in the system
        if not self.no_factorization:
            self._factorize_matrices()
        else:
            print("[PoissonSystem2D] Coefficient matrix is not factorized")
            self.L_fac = None

        # Initialize per-triangle Jacobians
        self._compute_per_triangle_jacobian()

        # Set the learnable parameters (Jacobians)
        self.J.requires_grad_(self.train_J)

    @jaxtyped(typechecker=typechecked)
    def get_current_mesh(
        self,
        constraints: Optional[Shaped[Tensor, "num_handle 2"]] = None,
        return_float32: bool = True,
    ) -> Tuple[Shaped[Tensor, "num_vertex 2"], Shaped[Tensor, "num_face 3"]]:
        """
        Returns the mesh whose geometry is determined by the current values of Jacobians.
        """
        new_v = self.solve(constraints)
        new_f = self.f_src
        if return_float32:
            new_v = new_v.type(torch.float32)
        return new_v, new_f
    
    @jaxtyped(typechecker=typechecked)
    def solve(
        self,
        constraints: Optional[Shaped[Tensor, "num_handle 2"]] = None
    ) -> Shaped[Tensor, "num_vertex 2"]:
        """
        Computes vertex positions according to the current per-triangle Jacobians.
        """
        return self._solve_constrained(constraints)
    
    @jaxtyped(typechecker=typechecked)
    def _solve_constrained(
        self,
        constraint: Shaped[Tensor, "num_handle 2"]
    ) -> Shaped[Tensor, "num_vertex 2"]:
        # Compute the RHS of the system
        J_ = self.J
        J_ = J_.transpose(1, 2).reshape(-1, 2, 1).squeeze(2).contiguous()
        J_ = rearrange_jacobian_elements(J_)
        rhs = self.rhs.multiply_with_dense(J_)
        rhs = self.L.transpose().multiply_with_dense(rhs)
        rhs = rhs + self.constraint_lambda * (
            self.indicator_matrix.transpose().multiply_with_dense(constraint)
        )

        # solve the constrained least squares
        v_sol: Shaped[Tensor, "num_vertex 2"] = SPLUSolveLayer.apply(
            self.L_fac,
            rhs[None, ...],
        )[0, ...]
        v_sol = v_sol.type_as(self.J)
        
        return v_sol

    @jaxtyped(typechecker=typechecked)
    def vertices_from_jacobians(
        self,
        jacobians: Shaped[Tensor, "num_face 2 2"],
        constraints: Optional[Shaped[Tensor, "num_handle 2"]] = None,
        return_float32: bool = True,
    ) -> Shaped[Tensor, "num_vertex 2"]:
        """
        Computes the vertex positions from given per-triangle Jacobians.
        """
        # Use provided Jacobians to form the right-hand side
        J_flat = jacobians.transpose(1, 2).reshape(-1, 2, 1).squeeze(2).contiguous()
        J_flat = rearrange_jacobian_elements(J_flat)
        rhs = self.rhs.multiply_with_dense(J_flat)
        rhs = self.L.transpose().multiply_with_dense(rhs)
    
        if self.is_constrained and constraints is not None:
            # If constraints are applied, add them to the right-hand side
            rhs = rhs + self.constraint_lambda * (
                self.indicator_matrix.transpose().multiply_with_dense(constraints)
            )
    
        # Solve the linear system to obtain the vertices
        vertices: Shaped[Tensor, "num_vertex 2"] = SPLUSolveLayer.apply(
            self.L_fac,
            rhs[None, ...],
        )[0, ...]
        
        # Convert to float32 if needed
        if return_float32:
            vertices = vertices.type(torch.float32)
        
        return vertices


    def _predicted_jacobians_to_vertices_via_poisson_solve(self, Lap, rhs, jacobians):
        '''
        convert the predictions to the correct convention and feed it to the poisson solve
        '''
    
        def _batch_rearrange_input(input):
            assert isinstance(input, torch.Tensor) and len(input.shape) in [2, 3]
            P = torch.zeros(input.shape).type_as(input)
            if len(input.shape) == 3:
                # Batched input
                k = input.shape[1] // 2
                P[:, :k, :] = input[:, ::2]
                P[:, k:2 * k, :] = input[:, 1::2]
            else:
                k = input.shape[0] // 2
                P[:k, :] = input[::2]
                P[k:2 * k, :] = input[1::2]
    
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
        return out.type_as(jacobians)



    @jaxtyped(typechecker=typechecked)
    def _compute_differential_operators_constrained(self) -> None:
        """
        Computes operators involving in Poisson's equation from the given mesh.
        """
        # Prepare Numpy arrays
        v = self.v_src.clone().cpu().numpy()
        f = self.f_src.clone().cpu().numpy()

        # Promote 2D vertices to 3D by adding a zero z-coordinate
        vertices_3d = np.hstack((v, np.zeros((v.shape[0], 1))))
        
        # Compute gradient
        grad: Shaped[ndarray, "2F V"] = igl.grad(vertices_3d, f)
        
    
        # Compute mass matrix
        mass: Shaped[ndarray, "F F"] = compute_mass_matrix(v, f, self.is_sparse)
        
        grad = grad[:mass.shape[0], :]
    
        # Compute Laplacian
        laplace: Shaped[ndarray, "V V"] = grad.T @ mass @ grad
        self.laplace = laplace.copy()
        self.LTL = self.laplace.transpose() @ self.laplace
    
        # Compute right-hand side of the system
        rhs = grad.T @ mass
        b1, b2, _ = igl.local_basis(vertices_3d, f)
        b1 = b1[:, :2]
        b2 = b2[:, :2]
        w = np.stack((b1, b2), axis=-1)
    
        # Handle sparse matrix storage
        if self.is_sparse:
            laplace = laplace.tocoo()
            rhs = rhs.tocoo()
            grad = grad.tocsc()
        else:
            laplace = laplace.toarray()
            rhs = rhs.toarray()
            grad = grad.toarray()
    
        # Rearrange the elements of the gradient matrix
        grad = rearrange_igl_grad_elements(grad)
    
        # Construct 'back-prop'able sparse matrices
        grad = SparseMat.from_M(grad, self.torch_dtype).to(self.device)
        rhs = SparseMat.from_coo(rhs, self.torch_dtype).to(self.device)
        laplacian = SparseMat.from_coo(laplace, self.torch_dtype).to(self.device)
    
        # Register the computed operators
        self.grad = grad
        self.w = w
        self.rhs = rhs
        self.L = laplacian
    
    @jaxtyped(typechecker=typechecked)
    def _build_indicator_matrix(self) -> None:
        """
        Builds a matrix that indicates the vertices being constrained in the system.
        """
        num_handle: int = int(len(self.anchor_inds))
        num_vertex: int = int(self.v_src.shape[0])
        anchor_inds: Int[ndarray, "num_handle"] = self.anchor_inds.cpu().numpy()
    
        # Check whether handle indices are unique
        assert len(np.unique(anchor_inds)) == num_handle, (
            "Handle indices must be unique."
        )
    
        # Build a list of non-zero indices
        indicator_indices: Int[ndarray, "2 num_handle"] = (
            np.zeros((2, num_handle), dtype=np.int64)
        )
        indicator_indices[0, :] = np.arange(num_handle, dtype=np.int64)
        indicator_indices[1, :] = anchor_inds
    
        # Create a sparse matrix
        self.indicator_matrix = SparseMat(
            indicator_indices,
            np.ones_like(anchor_inds),
            n=num_handle,
            m=num_vertex,
            ttype=self.torch_dtype,
        ).to(self.device)
    
        # Create the product of the indicator transposed and itself
        product_indices: Int[ndarray, "2 num_handle"] = np.concatenate(
            (anchor_inds[None, :], anchor_inds[None, :]), axis=0
        )
        self.indicator_product = SparseMat(
            product_indices,
            np.ones_like(anchor_inds),
            n=num_vertex,
            m=num_vertex,
            ttype=self.torch_dtype,
        ).to(self.device)
    
    @jaxtyped(typechecker=typechecked)
    def _factorize_matrices(self) -> None:
        """
        Factorizes the large, sparse matrices used in the system.
        """
        # Compute the matrix to factorize
        mat_to_fac: SparseMat = self._compute_matrix_to_factorize()
        self.lap_factored = mat_to_fac
    
        # Cholesky factorization
        self.L_fac: cholespy.CholeskySolverD = mat_to_fac.to_cholesky()
    
    @jaxtyped(typechecker=typechecked)
    def _compute_matrix_to_factorize(self):  
        """Computes the matrix to be factorized"""
        # Retrieve the Laplacian matrix
        mat_to_fac: SparseMat = self.L
    
        # Add constraint term if necessary
        if self.is_constrained:
            mat_to_fac: coo_matrix = mat_to_fac.to("cpu").to_coo()
    
            # Compute L^{T} @ L
            mat_to_fac = mat_to_fac.transpose() @ mat_to_fac
    
            # Compute L^{T} @ L + lambda * K
            indicator_product: coo_matrix = self.indicator_product.to("cpu").to_coo()
            mat_to_fac: coo_matrix = (
                mat_to_fac + self.constraint_lambda * indicator_product
            )
    
            # Convert the matrix back to the sparse matrix format
            mat_to_fac = SparseMat.from_coo(
                mat_to_fac.tocoo(),  # Ensure COO-format to be passed
                self.torch_dtype,
            ).to(self.device)
    
        return mat_to_fac
    
    @jaxtyped(typechecker=typechecked)
    def _compute_per_triangle_jacobian(self) -> None:
        """
        Computes per-triangle Jacobians from the given vertices.
        """
        # Retrieve operands
        grad: SparseMat = self.grad
        v: Shaped[Tensor, "num_vertex 2"] = self.v_src
    
        # Compute Jacobians
        J = grad.multiply_with_dense(v)
        J = J[:, None]
        J = J.reshape(-1, 2, 2)  # Adjust for 2D
        J = J.transpose(1, 2)
    
        # Register the computed Jacobians
        self.J = J
    
    @jaxtyped(typechecker=typechecked)
    def compute_per_triangle_jacobian(
        self,
        vertices: Shaped[Tensor, "num_vertex 2"],
    ) -> Shaped[Tensor, "num_face 2 2"]:
        """
        Computes per-triangle Jacobians from the given vertices.
        """
        J = self.grad.multiply_with_dense(vertices)
        J = J[:, None]
        J = J.reshape(-1, 2, 2)  # Adjust for 2D
        J = J.transpose(1, 2)
        return J


@jaxtyped(typechecker=typechecked)
def compute_mass_matrix(
    vertices: Shaped[ndarray, "num_vertex 2"],
    faces: Shaped[ndarray, "num_face 3"],
    is_sparse: bool,
):
    """
    Computes the mass matrix of a 2D mesh.
    """
    double_area = igl.doublearea(vertices, faces)
    double_area = np.hstack(
        (
            double_area,
            double_area,
        )
    )  # Only need two copies for x and y

    if is_sparse:
        return csc_matrix(diags(double_area))
    else:
        return diags(double_area)

@jaxtyped
def rearrange_igl_grad_elements(input):
    """
    Rearranges the grad operator computed from igl.grad() for a 2D mesh.
    The grad operator results in a matrix of shape (2 * #tri x #verts).
    The rearrangement aligns the y-coordinates after the x-coordinates.
    """
    assert type(input) == csc_matrix, 'Input should be a scipy csc sparse matrix'
    T = input.tocoo()

    r_c_data = np.hstack((T.row[..., np.newaxis], T.col[..., np.newaxis],
                          T.data[..., np.newaxis]))  # Horizontally stack row, col, and data arrays
    r_c_data = r_c_data[r_c_data[:, 0].argsort()]  # Sort along the row column

    # Separate out x and y blocks
    L = T.shape[0]
    Tx = r_c_data[:L, :]  # x-coordinates
    Ty = r_c_data[L:2 * L, :]  # y-coordinates

    # Align the y rows with x so that they too start from 0
    Ty[:, 0] -= Ty[0, 0]

    # 'Stretch' the x and y rows so that they can be interleaved
    Tx[:, 0] *= 2
    Ty[:, 0] *= 2

    # Interleave y into x
    Ty[:, 0] += 1

    Tc = np.zeros((input.shape[0] * 2, 3))
    Tc[::2] = Tx
    Tc[1::2] = Ty

    indices = Tc[:, :-1].astype(int)
    data = Tc[:, -1]

    return (indices.T, data, input.shape[0], input.shape[1])

@jaxtyped
def rearrange_jacobian_elements(jacobians):
    """
    Rearranges the elements of the given per-triangle Jacobian matrix for 2D.
    
    Args:
        jacobians: Per-triangle Jacobian matrices.
    """
    jacobians_rearranged = torch.zeros_like(jacobians)
    num_face = jacobians.shape[0] // 2  # Since we only have x and y components

    # Rearrange the elements
    jacobians_rearranged[:num_face, :] = jacobians[::2, ...]
    jacobians_rearranged[num_face : 2 * num_face, :] = jacobians[1::2, ...]

    return jacobians_rearranged

def _multiply_sparse_2d_by_dense_3d(mat, B):
    ret = []
    for i in range(B.shape[0]):
        C = mat.multiply_with_dense(B[i, ...])
        ret.append(C)
    ret = torch.stack(tuple(ret))
    return ret
