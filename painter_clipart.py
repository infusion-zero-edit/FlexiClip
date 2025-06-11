import warnings
warnings.filterwarnings("ignore")
from typing import Union
from torch.nn.functional import conv1d
from jaxtyping import Shaped, jaxtyped
import numpy as np
import json
from typeguard import typechecked
import math
import os
import os.path as osp
import random
import numpy as np
from math import comb
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.checkpoint import checkpoint
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList
from easydict import EasyDict as edict
from torch.optim.lr_scheduler import LambdaLR
from torchdiffeq import odeint_adjoint as odeint
import pydiffvg
from torch.nn import Linear, Module
from events import EventDispatcher, QKVEvent, AttentionEvent, IntermediateOutput
from masking import FullMask, LengthMask
import utils.util as util
from math import sqrt
from torch.nn import Dropout, Module
from utils.mesh_util import (
    silhouette,
    find_contour_point,
    triangulate,
    dilate,
    all_inside_contour,
    prepare_barycentric_coord,
    add_mesh_to_svg,
)
from svglib.svg import SVG
from svglib.geom import Bbox
import matplotlib.pyplot as plt

from pwarp.core.arap_torch import StepOne_torch, StepTwo_torch
from pwarp.core.poisson_system import PoissonSystem
from pwarp.core import ops_torch
from utils.arap_util import warp_svg


@jaxtyped(typechecker=typechecked)
def normalize_vs(
    vs: Union[Shaped[np.ndarray, "* D"], Shaped[torch.Tensor, "* D"]]
) -> Union[Shaped[np.ndarray, "* D"], Shaped[torch.Tensor, "* D"]]:
    """
    Normalize a batch of vectors.
    """
    eps = 1e-8

    # PyTorch
    if isinstance(vs, torch.Tensor):
        is_batch = False
        if vs.ndim == 2:
            is_batch = True
        else:
            assert vs.ndim == 1, f"Expected 1D or 2D array, got {vs.ndim}D"
            vs = vs[None, ...]
        norm = torch.norm(vs, dim=1, keepdim=True)
        vs = vs / (norm + eps)
        if not is_batch:
            vs = vs.squeeze(0)
    
    # Numpy
    else:
        is_batch = False
        if vs.ndim == 2:
            is_batch = True
        else:
            assert vs.ndim == 1, f"Expected 1D or 2D array, got {vs.ndim}D"
            vs = vs[np.newaxis, ...]
        norm = np.linalg.norm(vs, axis=1, keepdims=True)
        vs = vs / (norm + eps)
        if not is_batch:
            vs = vs.squeeze(0)
    
    return vs


@jaxtyped(typechecker=typechecked)
def quat_to_mat_torch(
    quats: Shaped[torch.Tensor, "* 4"]
) -> Shaped[torch.Tensor, "* 2 2"]:
    """
    Convert a batch of quaternions to a batch of 2D rotation matrices.
    The rotation is done around the Z-axis (which is out of the plane for 2D).
    """
    is_batch = False
    if quats.ndim == 2:  # received a batch of quaternions
        is_batch = True
    else:  # received a single quaternion
        assert quats.ndim == 1, f"Expected 1D or 2D array, got {quats.ndim}D"
        quats = quats[None, ...]  # Add batch dimension

    norm = torch.sqrt(quats[:, 0] ** 2 + quats[:, 1] ** 2 + quats[:, 2] ** 2 + quats[:, 3] ** 2)
    assert torch.allclose(norm, torch.ones_like(norm)), f"Expected unit quaternions"

    B = quats.shape[0]
    mats = torch.zeros((B, 2, 2), dtype=quats.dtype, device=quats.device)

    W, X, Y, Z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    # Calculate the 2D rotation matrix elements
    mats[:, 0, 0] = W**2 + X**2 - Y**2 - Z**2  # cos(theta)
    mats[:, 0, 1] = 2 * (X * Y - W * Z)  # sin(theta)
    mats[:, 1, 0] = 2 * (X * Y + W * Z)  # -sin(theta)
    mats[:, 1, 1] = W**2 - X**2 + Y**2 - Z**2  # cos(theta)

    if not is_batch:
        mats = mats.squeeze(0)

    return mats

class TransformerEncoderLayer(Module):
    """Self attention and feed forward network with skip connections.

    This transformer encoder layer implements the same encoder layer as
    PyTorch but is a bit more open for extension by receiving the attention
    implementation as a constructor argument.

    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(TransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = getattr(F, activation)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, attn_mask=None, length_mask=None):
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, x, x,
            attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask
        ))

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)


class TransformerEncoder(Module):
    """TransformerEncoder is little more than a sequence of transformer encoder
    layers.

    It contains an optional final normalization layer as well as the ability to
    create the masks once and save some computation.

    Arguments
    ---------
        layers: list, TransformerEncoderLayer instances or instances that
                implement the same interface.
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, layers, norm_layer=None, event_dispatcher=""):
        super(TransformerEncoder, self).__init__()
        self.layers = ModuleList(layers)
        self.norm = norm_layer
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, attn_mask=None, length_mask=None):
        """Apply all transformer encoder layers to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor of each transformer encoder layer.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Apply all the transformers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, length_mask=length_mask)
            self.event_dispatcher.dispatch(IntermediateOutput(self, x))

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x

class FullAttention(Module):
    """Implement the scaled dot product attention with softmax.

    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, softmax_temp=None, attention_dropout=0.1,
                 event_dispatcher=""):
        super(FullAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """Implements the multihead softmax attention.

        Arguments
        ---------
            queries: (N, L, H, E) The tensor containing the queries
            keys: (N, S, H, E) The tensor containing the keys
            values: (N, S, H, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        # Extract some shapes and compute the temperature
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        # Scale the queries instead of applying the softmax temperature to the
        # dot products
        queries = queries * softmax_temp

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhe,nshe->nhls", queries, keys)
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix
        if not key_lengths.all_ones:
            QK = QK + key_lengths.additive_matrix[:, None, None]

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(QK, dim=-1))
        V = torch.einsum("nhls,nshd->nlhd", A, values)

        # Let the world know of the attention matrix
        self.event_dispatcher.dispatch(AttentionEvent(self, A))

        # Make sure that what we return is contiguous
        return V.contiguous()

class AttentionLayer(Module):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.

    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality for the queries source
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        d_model_keys: The input feature dimensionality for keys source
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, d_model_keys=None, event_dispatcher=""):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        d_model_keys = d_model_keys or d_model

        self.inner_attention = attention
        self.query_projection = Linear(d_model, d_keys * n_heads)
        self.key_projection = Linear(d_model_keys, d_keys * n_heads)
        self.value_projection = Linear(d_model_keys, d_values * n_heads)
        self.out_projection = Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """Apply attention to the passed in queries/keys/values after
        projecting them to multiple heads.

        In the argument description we make use of the following sizes

            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of

        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)

class SeqTransformerJacobians(nn.Module):
    def __init__(self,code_size):
        super(SeqTransformerJacobians,self).__init__()
        model_dimensions = 4
        d_keys = 32 
        d_values = 16 
        d_ff = 32 
        n_heads = 2
        self.transformer_encoder = TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(FullAttention(),model_dimensions, n_heads,d_keys=d_keys,d_values=d_keys),
                    model_dimensions,
                    activation="relu",
                    dropout=0.0,
                    d_ff=d_ff
                )
            ]
        )

    def forward(self, embeddings, sample=True):
        trans_out = self.transformer_encoder(embeddings)
        attention_jacobians = trans_out[:,0,:].unsqueeze(1)

        return attention_jacobians

class TriangleODE(nn.Module):
    def __init__(self,jac_first_order_func):
        super(TriangleODE, self).__init__()
        self.zeros_num_faces = torch.zeros(1,1).float().cuda()
        self.jac_first_order_func = jac_first_order_func 

    def forward(self, t, state):
        current_state = state[0][0]
        start_index = 0
        end_index = start_index+1
        num_faces = current_state[start_index:end_index].int()

        start_index = end_index
        end_index = start_index + num_faces*2*2
        j0 = current_state[start_index:end_index].view(num_faces,4)

        start_index = end_index
        end_index = start_index + num_faces*2*2
        previous_attention = current_state[start_index:end_index].view(num_faces,4)

        start_index = end_index
        end_index = start_index + num_faces*2*2
        target_j = current_state[start_index:end_index].view(num_faces,4)

        djacobians_dt = self.jac_first_order_func(j0, previous_attention, target_j, t)
        zeros_triangle = torch.zeros(1,num_faces*2*2).float().cuda()

        dstate_dt = tuple([torch.cat([self.zeros_num_faces, djacobians_dt.view(1,-1), zeros_triangle, zeros_triangle, zeros_triangle],1)])
        return dstate_dt

class JacFC(nn.Module):
    def __init__(self,n_dims,n_shapes):
        super(JacFC,self).__init__()
        self.fc1 = nn.Linear(n_shapes*n_dims,64)
        self.fc2 = nn.Linear(64,32)

    def forward(self,triangles):
        xt = F.relu(self.fc1(triangles))
        feature = self.fc2(xt)
        return feature

class JacobianNetworkD4D(nn.Module):
    def __init__(self):
        super(JacobianNetworkD4D,self).__init__()
        jac_size = 32 
        previous_attention_size = 32 
        time_size = 1

        self.fc1 = nn.Linear(2*jac_size + time_size, 512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256, 4)

        self.jac_fc = JacFC(4,1)


    def forward(self, j0, previous_attention, target_j, t):
        flat_j0 = j0.view(j0.size()[0], 4).float()
        flat_previous_attention = previous_attention.view(j0.size()[0], 4).float()
        # flat_target_j = target_j.view(target_j.size()[0], 4).float()
        
        j0_feat = self.jac_fc(flat_j0)
        # target_j_feat = self.jac_fc(flat_target_j)
        previous_attention_feat = self.jac_fc(flat_previous_attention)
        
        expanded_time = t.unsqueeze(0).expand(j0.size()[0],1).float()
        
        xt = torch.cat([previous_attention_feat, j0_feat, expanded_time],-1).float()
        xt = F.relu(self.fc1(xt))
        xt = F.relu(self.fc2(xt))
        dj_dt = self.fc3(xt)

        return dj_dt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear')!=-1:
        nn.init.xavier_normal_(m.weight)
        #m.bias.data.fill_(0.0)
    if classname.find('Conv1D')!=-1:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

class Painter(torch.nn.Module):
    def __init__(self,
                 args,
                 svg_path: str,
                 num_frames: int,
                 device,
                 path_to_trained_mlp=None,
                 inference=False):
        super(Painter, self).__init__()
        self.svg_path = svg_path
        self.num_frames = num_frames
        self.device = device
        self.optim_bezier_points = args.optim_bezier_points
        self.opt_bezier_points_with_mlp = args.opt_bezier_points_with_mlp
        self.fix_start_points = args.fix_start_points
        self.render = pydiffvg.RenderFunction.apply
        self.normalize_input = args.normalize_input
        self.arap_weight = args.arap_weight
        self.opt_with_layered_arap = args.opt_with_layered_arap and osp.exists(f'{self.svg_path}_layer.json')
        self.loop_num = args.loop_num
        trans_network = SeqTransformerJacobians(16).cuda()
        trans_network.apply(weights_init)
        self.attention = trans_network
        jacobian_network = JacobianNetworkD4D().cuda()
        jacobian_network.apply(weights_init)
        self.jacobian_func = jacobian_network
        self.ode_evaluator = TriangleODE(self.jacobian_func)
        self.method = 'euler'
        if self.optim_bezier_points:
            if self.opt_with_layered_arap:
                self.init_layered_mesh(cfg=args)
            else:
                self.init_mesh(cfg=args)
        
        if self.opt_bezier_points_with_mlp:
            self.points_bezier_mlp_input_ = self.point_bezier_mlp_input.float().unsqueeze(0).to(device)

            self.points_per_frame = 1  # FIXME
            self.mlp_points = PointMLP(input_dim=torch.numel(self.points_bezier_mlp_input_),
                                        inter_dim=args.inter_dim,
                                        num_frames=num_frames,
                                        device=device,
                                        inference=inference).to(device)
            
            if path_to_trained_mlp:
                print(f"Loading MLP from {path_to_trained_mlp}")
                self.mlp_points.load_state_dict(torch.load(path_to_trained_mlp))
                self.mlp_points.eval()

            # Init the weights of LayerNorm for global translation MLP if needed.
            if args.translation_layer_norm_weight:
                self.init_translation_norm(args.translation_layer_norm_weight)

    def init_mesh(self, cfg):
        """
        Loads the svg file from svg_path and set grads to the parameters we want to optimize
        In this case, we optimize the control points of bezier paths
        """
        parameters = edict()
        parameters.point_bezier = []

        svg_path = f'{self.svg_path}_scaled.svg'
        svg_keypts_path = f'{self.svg_path}_keypoint_scaled.svg'
        src = SVG.load_svg(svg_path)
        control_pts = SVG.load_svg(svg_keypts_path)
        control_pts = np.array([c.center.pos for c in control_pts.svg_path_groups])
        self.control_pts = control_pts
        self.num_control_pts = len(control_pts)

        # init the canvas_width, canvas_height
        width = int(src.viewbox.wh.x)
        height = int(src.viewbox.wh.y)
        self.canvas_width = cfg.width = width
        self.canvas_height = cfg.height = height

        # find contour points, num is controlled by cfg.boundary_simplify_level
        contour_pts = get_contour(src, cfg)

        # dilate contour to include all pts
        contour_pts = dilate_contour(contour_pts, src)

        # prepare segments for CDT
        segments = np.array([(i, (i + 1) % len(contour_pts)) for i in range(len(contour_pts))])
        control_pts_index = np.arange(len(contour_pts), len(contour_pts) + len(control_pts))
        all_pts = np.concatenate([contour_pts, control_pts], axis=0)

        mesh = triangulate(cfg, all_pts, segments)
        plt.savefig(osp.join(cfg.mesh_dir, 'mesh.png'))
        vertices, triangles = mesh['vertices'], mesh['triangles']

        # add mesh to source for visualization
        src_mesh = add_mesh_to_svg(src.copy(), mesh)
        src_mesh.save_svg(osp.join(cfg.mesh_dir, 'source_mesh.svg'))

        src.drop_z()
        src.filter_consecutives()
        src.save_svg(osp.join(cfg.svg_dir, 'init.svg'))
        if cfg.need_subdivide:
            print('start subdivide...')
            edges = ops_torch.get_edges(len(triangles), triangles)
            src = src.subdivide(edges, vertices)
            svg_path = osp.join(cfg.svg_dir, 'init_subdiv.svg')  # update svg_path
            src.save_svg(svg_path)
            print('end subdivide...')

        # barycentric coordinate
        _, _, src_shapes, src_shape_groups = pydiffvg.svg_to_scene(svg_path)  # preprocessing done, so just load the same svg
        face_index, bary_coord = prepare_barycentric_coord(src_shapes, vertices, triangles)
        cum_sizes =  np.cumsum([shape.points.shape[0] for shape in src_shapes])
        cum_sizes = np.concatenate([[0], cum_sizes])  # used in warp_svg
        print('bary coord computed')

        # prepare ARAP
        vertices = torch.from_numpy(vertices.astype(np.float32))
        faces = torch.from_numpy(triangles.astype(np.int64))
        edges = ops_torch.get_edges(len(faces), faces)
        # Initialize Poisson System
        self.poisson = PoissonSystem(vertices.to(self.device), faces.to(self.device), self.device, train_J=True, anchor_inds=torch.from_numpy(control_pts_index.astype(np.int32)).to(self.device))
        print("Initialized Poisson system")
        self.poisson.J = self.poisson.compute_per_triangle_jacobian(vertices.to(self.device))
        self.poisson.J.requires_grad_(True)
        # Stage 1
        optim_vars = []
        optim_vars.append(self.poisson.J)

        optim = torch.optim.Adam(optim_vars, lr=1e-3)
        
        for _ in range(100000):

            loss = (((self.poisson.J) - torch.eye(2, 2, device=self.device)) ** 2).mean()
            print("Loss Poisson", loss.item())
            
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        self.vertices = vertices.to(self.device)
        self.faces = faces.to(self.device)
        self.edges = edges.to(self.device)

        self.bary_coord = bary_coord.to(self.device)
        # print('ARAP prepared')

        self.control_pts_index = control_pts_index
        self.face_index = face_index
        self.src_shapes = src_shapes
        self.src_shape_groups = src_shape_groups
        self.cum_sizes = cum_sizes

        # init bezier path
        bezier_shapes, bezier_shape_groups = init_bezier_with_start_point(control_pts, width, height, radius=cfg.bezier_radius, device=self.device)
        pydiffvg.save_svg(osp.join(cfg.bezier_dir, 'init_bezier.svg'), width, height, bezier_shapes, bezier_shape_groups)

        for path in bezier_shapes:
            if self.optim_bezier_points and not self.opt_bezier_points_with_mlp:
                path.points.requires_grad = True
            parameters.point_bezier.append(path.points)

        self.bezier_shapes = bezier_shapes
        self.bezier_shape_groups = bezier_shape_groups

        tensor_point_bezier_init = [torch.cat([path.points]) for path in bezier_shapes]
        self.point_bezier_mlp_input = torch.cat(tensor_point_bezier_init)  # [4*num_control_pts, 2]
        
        self.parameters_ = parameters

    def init_layered_mesh(self, cfg):
        """
        Loads the svg file from svg_path and set grads to the parameters we want to optimize
        In this case, we optimize the control points of bezier paths
        """
        parameters = edict()
        parameters.point_bezier = []

        svg_path = f'{self.svg_path}_scaled.svg'
        svg_keypts_path = f'{self.svg_path}_keypoint_scaled.svg'
        src = SVG.load_svg(svg_path)
        control_pts = SVG.load_svg(svg_keypts_path)
        control_pts = np.array([c.center.pos for c in control_pts.svg_path_groups])
        self.control_pts = control_pts
        self.num_control_pts = len(control_pts)

        # init the canvas_width, canvas_height
        width = int(src.viewbox.wh.x)
        height = int(src.viewbox.wh.y)
        self.canvas_width = cfg.width = width
        self.canvas_height = cfg.height = height

        # layered-ARAP
        with open(f'{self.svg_path}_layer.json', 'r') as f:
            layers = json.load(f)
            self.num_of_layers = len(layers)

        # for arap
        self.vertices = []
        self.faces = []
        self.edges = []
        self.bary_coord = []
        self.control_pts_index = []
        self.face_index = []
        self.src_shapes_layer = []
        self.cum_sizes = []
        self.poissons = []

        # for layered-arap
        self.control_index_layer = []
        self.path_index_layer = []

        # in order to restore the entire svg
        entire_svg = src.copy()

        for i, layer in enumerate(layers):
            path_index_layer = [index - 1 for index in layer['path_index']]
            control_index_layer = [index - 1 for index in layer['control_index']]
            self.path_index_layer.extend(path_index_layer)
            self.control_index_layer.append(np.array(control_index_layer))
            print('========== layer', i, '==========')
            print('path idx', path_index_layer)
            print('ctrl idx', control_index_layer)

            # create individual layer svg
            control_pts_layer = control_pts[control_index_layer]
            src_layer = SVG([src.svg_path_groups[index].copy() for index in path_index_layer], src.viewbox)

            # find contour points, num is controlled by cfg.boundary_simplify_level
            contour_pts = get_contour(src_layer, cfg)

            # dilate contour to include all pts
            contour_pts = dilate_contour(contour_pts, src_layer)

            # prepare segments for CDT
            segments = np.array([(i, (i + 1) % len(contour_pts)) for i in range(len(contour_pts))])
            control_pts_index = np.arange(len(contour_pts), len(contour_pts) + len(control_pts_layer))
            all_pts = np.concatenate([contour_pts, control_pts_layer], axis=0)

            mesh = triangulate(cfg, all_pts, segments)
            plt.savefig(osp.join(cfg.mesh_dir, f'layer{i}_mesh.png'))
            vertices, triangles = mesh['vertices'], mesh['triangles']

            # add mesh to source for visualization
            src_mesh = add_mesh_to_svg(src_layer.copy(), mesh)
            src_mesh.save_svg(osp.join(cfg.mesh_dir, f'layer{i}_mesh.svg'))

            src_layer.drop_z()
            src_layer.filter_consecutives()
            src_layer.save_svg(osp.join(cfg.svg_dir, f'layer{i}_init.svg'))
            if cfg.need_subdivide:
                print('start subdivide...')
                edges = ops_torch.get_edges(len(triangles), triangles)
                src_layer = src_layer.subdivide(edges, vertices)
                svg_path = osp.join(cfg.svg_dir, f'layer{i}_init_subdiv.svg')  # update svg_path
                src_layer.save_svg(svg_path)
                print('end subdivide...')

            for j, path_index in enumerate(path_index_layer):
                entire_svg.svg_path_groups[path_index] = src_layer.svg_path_groups[j].copy()

            # barycentric coordinate
            _, _, src_shapes, _ = pydiffvg.svg_to_scene(svg_path)  # preprocessing done, so just load the same svg
            face_index, bary_coord = prepare_barycentric_coord(src_shapes, vertices, triangles)
            cum_sizes = np.cumsum([shape.points.shape[0] for shape in src_shapes])
            cum_sizes = np.concatenate([[0], cum_sizes])  # used in warp_svg
            print('bary coord computed')

            # prepare ARAP
            vertices = torch.from_numpy(vertices.astype(np.float32))
            faces = torch.from_numpy(triangles.astype(np.int64))
            edges = ops_torch.get_edges(len(faces), faces)

            poisson = PoissonSystem(vertices.to(self.device), faces.to(self.device), self.device, train_J=True, anchor_inds=torch.from_numpy(control_pts_index.astype(np.int32)).to(self.device))
            print("Initialized Poisson system")
            poisson.J = poisson.compute_per_triangle_jacobian(vertices.to(self.device))
            poisson.J.requires_grad_(True)
            
            # Stage 1
            optim_vars = []
            optim_vars.append(poisson.J)

            optim = torch.optim.Adam(optim_vars, lr=1e-3)
            
            for _ in range(100000):
    
                loss = (((poisson.J) - torch.eye(2, 2, device=self.device)) ** 2).mean()
                print("Loss Poisson", loss.item())
                
                optim.zero_grad()
                loss.backward()
                optim.step()

            self.poissons.append(poisson)
            self.vertices.append(vertices.to(self.device))
            self.faces.append(faces.to(self.device))
            self.edges.append(edges.to(self.device))
            # self.poisson.append(poisson)

            self.bary_coord.append(bary_coord.to(self.device))
            self.control_pts_index.append(control_pts_index)
            self.face_index.append(face_index)
            self.src_shapes_layer.append(src_shapes)
            self.cum_sizes.append(cum_sizes)

        self.path_index_layer = np.array(self.path_index_layer).flatten()
        self.path_index_layer_sorted = self.path_index_layer.argsort()


        # save entire svg
        svg_path = osp.join(cfg.svg_dir, f'entire_init_subdiv.svg')  # update svg_path
        entire_svg.save_svg(svg_path)
        _, _, _, self.src_shape_groups = pydiffvg.svg_to_scene(svg_path)

        # init bezier path
        bezier_shapes, bezier_shape_groups = init_bezier_with_start_point(control_pts, width, height, cfg.bezier_radius, self.device)
        pydiffvg.save_svg(osp.join(cfg.bezier_dir, 'init_bezier.svg'), width, height, bezier_shapes, bezier_shape_groups)

        for path in bezier_shapes:
            if self.optim_bezier_points and not self.opt_bezier_points_with_mlp:
                path.points.requires_grad = True
            parameters.point_bezier.append(path.points)

        self.bezier_shapes = bezier_shapes
        self.bezier_shape_groups = bezier_shape_groups

        tensor_point_bezier_init = [torch.cat([path.points]) for path in bezier_shapes]
        self.point_bezier_mlp_input = torch.cat(tensor_point_bezier_init)  # [4*num_control_pts, 2]
        self.parameters_ = parameters

    def get_ode_solution_first_order(self, initial_delta, j0, prev_pred_attention, target_j, times):
        num_faces = j0.size()[0]
        prev_pred_attention = prev_pred_attention.clone().detach()
        num_faces_tensor = torch.ones(1,1).int().cuda()*num_faces
        initial_tensor = torch.cat([num_faces_tensor, initial_delta.contiguous().view(1,-1), j0.contiguous().view(1,-1), prev_pred_attention.contiguous().view(1,-1), target_j.contiguous().view(1,-1)], 1)
        initial_state = tuple([initial_tensor])
        solution = odeint(self.ode_evaluator, initial_state, times, method=self.method)[0].squeeze(1)

        start_ix = 1 ; end_ix = start_ix +(num_faces*2*2)
        jac_first_solution_fetch_indices = torch.from_numpy(np.array(range(start_ix,end_ix))).type(torch.int64).unsqueeze(0).cuda()
        fetch_indices = jac_first_solution_fetch_indices.repeat(len(times),1)
        jac_first_order = torch.gather(solution,1,fetch_indices).view(-1,num_faces,2,2).contiguous()

        return jac_first_order

    def compute_jacobians_from_constraints(self, v_src, f_src, v_target):
        """
        Computes per-triangle Jacobians that map source vertices to target (deformed) vertices.
        
        Args:
            v_src (Tensor): Source vertices, shape (num_vertices, 2).
            f_src (Tensor): Triangle face indices, shape (num_faces, 3).
            v_target (Tensor): Deformed target vertices, shape (num_vertices, 2).
            
        Returns:
            jacobians (Tensor): Per-triangle Jacobians, shape (num_faces, 2, 2).
        """
        num_faces = f_src.shape[0]
        jacobians = torch.zeros((num_faces, 2, 2), dtype=v_src.dtype, device=v_src.device)
        
        for i in range(num_faces):
            # Indices of the vertices of the i-th triangle
            idx = f_src[i]
            v1, v2, v3 = v_src[idx[0]], v_src[idx[1]], v_src[idx[2]]
            v1_target, v2_target, v3_target = v_target[idx[0]], v_target[idx[1]], v_target[idx[2]]
            
            # Compute edge vectors for source and target triangles
            V_src = torch.stack([v2 - v1, v3 - v1], dim=1)  # Shape (2, 2)
            V_target = torch.stack([v2_target - v1_target, v3_target - v1_target], dim=1)  # Shape (2, 2)
            
            # Compute the Jacobian for the current triangle
            try:
                J = V_target @ torch.linalg.inv(V_src)  # Shape (2, 2)
            except RuntimeError as e:
                print(f"Inversion failed for face {i}. Adjusting source vertices might help.")
                continue
            
            jacobians[i] = J
        
        return jacobians    

    def neutralize_global_transform(self, jacobian):
        U, S, Vt = torch.linalg.svd(jacobian)  # Decompose using SVD
        neutralized_jacobian = torch.matmul(U, Vt)  # Retain only rotation and exclude scaling
        return neutralized_jacobian
    
    def render_frames_to_tensor_direct_optim_bezier(self, point_bezier):
        # point_bezier: List[Tensor], each tensor is [4, 2]
        frames_init, frames_svg, points_init_frame = [], [], []
        frames_init_p, frames_svg_p = [], []
        
        shifted_locations = []
        new_vertices = []
        new_vertices_pose = []
        times = []
        pose_jacobians = []
        prev_pred_last_def = None
        temp_jacobians = []
        for t in self.sample_on_bezier_path(self.loop_num):
            loc = torch.stack([cubic_bezier(p, t) for p in point_bezier])
            curr_v, _ = self.poisson.get_current_mesh(loc)
            new_vertices_pose.append(curr_v.unsqueeze(0))
            deformedJ = self.compute_jacobians_from_constraints(self.vertices, self.faces, curr_v)
            pose_jacobians.append(deformedJ.unsqueeze(0))
            times.append(t)
            if t == 0:
                previous_attention = self.poisson.J
                previous_attention = previous_attention / previous_attention.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                first_def = torch.eye(2, device='cuda').expand_as(self.poisson.J).clone().detach()
            else:
                prev_pred_jacobians = torch.cat(pose_jacobians, dim=0)
                prev_jacobians_for_attention = prev_pred_jacobians.view(len(pose_jacobians), pose_jacobians[0].size()[1], 4).permute(1,0,2)
                previous_attention = self.attention(prev_jacobians_for_attention.float())
                previous_attention = previous_attention / previous_attention.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                first_def = prev_pred_last_def.clone().detach()
            deformation_jacobians = self.get_ode_solution_first_order(first_def.float(), self.poisson.J, previous_attention.float(), pose_jacobians[-1].squeeze(0).float(), torch.Tensor(times).cuda())
            final_jacobians = torch.cat(pose_jacobians, dim=0) + deformation_jacobians/torch.norm(deformation_jacobians)
            curr_v = self.poisson.vertices_from_jacobians(final_jacobians[-1], loc)
            temp_jacobians.append(final_jacobians[-1].unsqueeze(0))
            new_vertices.append(curr_v.unsqueeze(0))
            last_def = deformation_jacobians[-1]
            shifted_locations.append(loc.unsqueeze(0))
            prev_pred_last_def = last_def
            
        shifted_locations = torch.cat(shifted_locations, dim=0)
        new_vertices = torch.cat(new_vertices, dim=0)
        final_jacobians = torch.cat(temp_jacobians, dim=0)
        new_vertices_pose = torch.cat(new_vertices_pose, dim=0)
        pose_jacobians = torch.cat(pose_jacobians, dim=0)
        
        # warp svg based on the updated mesh
        for vs, vs_p in zip(new_vertices, new_vertices_pose):
            new_shapes = warp_svg(self.src_shapes, self.faces, self.face_index, self.bary_coord, vs, self.cum_sizes)
            new_shapes_p = warp_svg(self.src_shapes, self.faces, self.face_index, self.bary_coord, vs_p, self.cum_sizes)
            
            scene_args = pydiffvg.RenderFunction.serialize_scene(self.canvas_width, self.canvas_height, new_shapes, self.src_shape_groups)
            scene_args_p = pydiffvg.RenderFunction.serialize_scene(self.canvas_width, self.canvas_height, new_shapes_p, self.src_shape_groups)

            cur_im = self.render(self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args)
            cur_im_p = self.render(self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args_p)
            
            cur_im = cur_im[:, :, 3:4] * cur_im[:, :, :3] + \
                     torch.ones(cur_im.shape[0], cur_im.shape[1], 3, device=self.device) * (1 - cur_im[:, :, 3:4])
            cur_im = cur_im[:, :, :3]
            
            frames_init.append(cur_im)
            frames_svg.append((new_shapes, self.src_shape_groups))

            cur_im_p = cur_im_p[:, :, 3:4] * cur_im_p[:, :, :3] + torch.ones(cur_im_p.shape[0], cur_im_p.shape[1], 3, device=self.device) * (1 - cur_im_p[:, :, 3:4])
            cur_im_p = cur_im_p[:, :, :3]

            frames_init_p.append(cur_im_p)
            
            frames_svg_p.append((new_shapes_p, self.src_shape_groups))
        
        # motion repeat
        if self.loop_num > 0:
            # one loop
            frames_init = frames_init + frames_init[::-1]
            frames_svg = frames_svg + frames_svg[::-1]
            if self.loop_num > 1:
                # two loops
                frames_init = frames_init + frames_init
                frames_svg = frames_svg + frames_svg

        # motion repeat
        if self.loop_num > 0:
            # one loop
            frames_init_p = frames_init_p + frames_init_p[::-1]
            frames_svg_p = frames_svg_p + frames_svg_p[::-1]
            if self.loop_num > 1:
                # two loops
                frames_init_p = frames_init_p + frames_init_p
                frames_svg_p = frames_svg_p + frames_svg_p

        return torch.stack(frames_init), frames_svg, torch.stack(frames_init_p), frames_svg_p, points_init_frame, shifted_locations, point_bezier, final_jacobians, pose_jacobians
    
    def render_frames_to_tensor_direct_optim_bezier_layered(self, point_bezier):
        # point_bezier: List[Tensor], each tensor is [4, 2]
        frames_init, frames_svg, points_init_frame = [], [], []
        frames_init_p, frames_svg_p = [], []

        shifted_locations = []  # compute points on bezier curves
        ts = self.sample_on_bezier_path(self.loop_num)
        for t in ts:
            loc = torch.stack([cubic_bezier(p, t) for p in point_bezier])
            shifted_locations.append(loc.unsqueeze(0))
        
        shifted_locations = torch.cat(shifted_locations, dim=0)  # [frame_num, num_bezier, 2]

        new_vertices_layer = [[] for _ in range(len(ts))]
        new_vertices_pose_layer = [[] for _ in range(len(ts))]

        final_jacobians_all = []
        pose_jacobians_all = []
        
        for i in range(self.num_of_layers):
            # arap
            vertices = self.vertices[i]
            edges = self.edges[i]
            faces = self.faces[i]

            poisson = self.poissons[i]
            bary_coord = self.bary_coord[i]
            control_pts_index = self.control_pts_index[i]
            face_index = self.face_index[i]
            src_shapes = self.src_shapes_layer[i]
            cum_sizes = self.cum_sizes[i]

            # layered-arap
            control_index_layer = self.control_index_layer[i]
            shifted_locations_layer = shifted_locations[:, control_index_layer, :]

            new_vertices, new_vertices_pose, times, pose_jacobians, temp_jacobians = [], [], [], [], []
            prev_pred_last_def = None
            for f_i in range(shifted_locations_layer.shape[0]):
                curr_v, _ = poisson.get_current_mesh(shifted_locations_layer[f_i, :, :])
                new_vertices_pose.append(curr_v.unsqueeze(0))
                deformedJ = self.compute_jacobians_from_constraints(vertices, faces, curr_v)
                pose_jacobians.append(deformedJ.unsqueeze(0))
                times.append(f_i)
                if f_i == 0:
                    previous_attention = poisson.J
                    previous_attention = previous_attention / previous_attention.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    first_def = torch.eye(2, device='cuda').expand_as(poisson.J).clone().detach()
                else:
                    prev_pred_jacobians = torch.cat(pose_jacobians, dim=0)
                    prev_jacobians_for_attention = prev_pred_jacobians.view(len(pose_jacobians), pose_jacobians[0].size()[1], 4).permute(1,0,2)
                    previous_attention = self.attention(prev_jacobians_for_attention.float())
                    previous_attention = previous_attention / previous_attention.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    first_def = prev_pred_last_def.clone().detach()

                deformation_jacobians = self.get_ode_solution_first_order(first_def.float(), poisson.J, previous_attention.float(), pose_jacobians[-1].squeeze(0).float(), torch.Tensor(times).cuda())

                final_jacobians = torch.cat(pose_jacobians, dim=0) + deformation_jacobians/torch.norm(deformation_jacobians)
                curr_v = poisson.vertices_from_jacobians(final_jacobians[-1], shifted_locations_layer[f_i, :, :])
                temp_jacobians.append(final_jacobians[-1].unsqueeze(0))
                new_vertices.append(curr_v.unsqueeze(0))
                last_def = deformation_jacobians[-1]
                prev_pred_last_def = last_def

            new_vertices = torch.cat(new_vertices, dim=0)
            final_jacobians = torch.cat(temp_jacobians, dim=0)
            new_vertices_pose = torch.cat(new_vertices_pose, dim=0)
            pose_jacobians = torch.cat(pose_jacobians, dim=0)
            
            final_jacobians_all.append(final_jacobians)
            pose_jacobians_all.append(pose_jacobians)
            
            # warp svg based on the updated mesh
            for j, (vs, vs_p) in enumerate(zip(new_vertices, new_vertices_pose)):
                new_vertices_layer[j].extend(warp_svg(src_shapes, faces, face_index, bary_coord, vs, cum_sizes))
                new_vertices_pose_layer[j].extend(warp_svg(src_shapes, faces, face_index, bary_coord, vs_p, cum_sizes))
        
        # construct entire svg
        for new_vertices, new_vertices_p in zip(new_vertices_layer, new_vertices_pose_layer):  # num of ts
            # sort new_shapes according to path_index_layer
            new_shapes = [new_vertices[i] for i in self.path_index_layer_sorted]
            new_shapes_p = [new_vertices_p[i] for i in self.path_index_layer_sorted]

            scene_args = pydiffvg.RenderFunction.serialize_scene(self.canvas_width, self.canvas_height, new_shapes, self.src_shape_groups)
            scene_args_p = pydiffvg.RenderFunction.serialize_scene(self.canvas_width, self.canvas_height, new_shapes_p, self.src_shape_groups)
            
            cur_im = self.render(self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args)
            cur_im_p = self.render(self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args_p)
            
            cur_im = cur_im[:, :, 3:4] * cur_im[:, :, :3] + \
                     torch.ones(cur_im.shape[0], cur_im.shape[1], 3, device=self.device) * (1 - cur_im[:, :, 3:4])
            cur_im = cur_im[:, :, :3]

            cur_im_p = cur_im_p[:, :, 3:4] * cur_im_p[:, :, :3] + \
                     torch.ones(cur_im_p.shape[0], cur_im_p.shape[1], 3, device=self.device) * (1 - cur_im_p[:, :, 3:4])
            cur_im_p = cur_im_p[:, :, :3]

            frames_init.append(cur_im)
            frames_svg.append((new_shapes, self.src_shape_groups))

            frames_init_p.append(cur_im_p)
            frames_svg_p.append((new_shapes_p, self.src_shape_groups))
        
        # motion repeat
        if self.loop_num > 0:
            # one loop
            frames_init = frames_init + frames_init[::-1]
            frames_svg = frames_svg + frames_svg[::-1]
            if self.loop_num > 1:
                # two loops
                frames_init = frames_init + frames_init
                frames_svg = frames_svg + frames_svg

        # motion repeat
        if self.loop_num > 0:
            # one loop
            frames_init_p = frames_init_p + frames_init_p[::-1]
            frames_svg_p = frames_svg_p + frames_svg_p[::-1]
            if self.loop_num > 1:
                # two loops
                frames_init_p = frames_init_p + frames_init_p
                frames_svg_p = frames_svg_p + frames_svg_p
        
        final_jacobians = torch.cat(final_jacobians_all, dim=1)
        pose_jacobians = torch.cat(pose_jacobians_all, dim=1)
        return torch.stack(frames_init), frames_svg,  torch.stack(frames_init_p), frames_svg_p, points_init_frame, shifted_locations, point_bezier, final_jacobians, pose_jacobians
    
    def render_frames_to_tensor_mlp_bezier(self):
        frame_input = self.points_bezier_mlp_input_
        if self.normalize_input:
            frame_input = util.normalize_tensor(frame_input)  # [0, 1]
        # predict the delta of control points of all bezier paths
        delta_prediction = self.mlp_points(frame_input)  # [4 * num_control_pts, 2]

        # add predicted delta to the original bezier shapes
        point_bezier = []
        for i in range(self.num_control_pts):
            updated_points = self.point_bezier_mlp_input[i * 4 : (i + 1) * 4] + delta_prediction[i * 4 : (i + 1) * 4]
            if self.fix_start_points:
                updated_points[0] = self.point_bezier_mlp_input[i * 4]
            point_bezier.append(updated_points)
            # update shapes for visualization
            self.bezier_shapes[i].points = updated_points.detach()

        if self.opt_with_layered_arap:
            return self.render_frames_to_tensor_direct_optim_bezier_layered(point_bezier)
        return self.render_frames_to_tensor_direct_optim_bezier(point_bezier)

    def render_frames_to_tensor_with_bezier(self, mlp=True):
        if self.opt_bezier_points_with_mlp and mlp:
            return self.render_frames_to_tensor_mlp_bezier()
        else:
            if self.opt_with_layered_arap:
                return self.render_frames_to_tensor_direct_optim_bezier_layered(self.parameters_["point_bezier"])
            return self.render_frames_to_tensor_direct_optim_bezier(self.parameters_["point_bezier"])
    
    def get_bezier_params(self):
        param_list = []
        for m in [self.attention,self.jacobian_func]:
            param_list.extend(m.parameters())
        if self.opt_bezier_points_with_mlp:
            return self.mlp_points.get_points_params() + param_list
        return self.parameters_["point_bezier"]

    def log_state(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if self.opt_bezier_points_with_mlp:
            torch.save(self.mlp_points.state_dict(), f"{output_path}/model.pt")
            print(f"Model saved to {output_path}/model.pt")

    def init_translation_norm(self, translation_layer_norm_weight):
        print(f"Initializing translation layerNorm to {translation_layer_norm_weight}")
        for child in self.mlp_points.frames_rigid_translation.children():
            if isinstance(child, nn.LayerNorm):
                with torch.no_grad():
                    child.weight *= translation_layer_norm_weight

    def sample_on_bezier_path(self, loop_num):
        segment_len = self.num_frames if loop_num == 0 else self.num_frames // (loop_num * 2)
        ts = torch.linspace(0, 1, segment_len)
        return ts


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=16):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        # x = x + 
        return self.dropout(self.pe[:x.size(0), :])


class PointModel(nn.Module):

    def __init__(self, input_dim, inter_dim, num_frames, device, inference=False):

        super().__init__()
        self.num_frames = num_frames
        self.inter_dim = inter_dim
        self.input_dim = input_dim
        self.embed_dim = inter_dim
        self.inference = inference

        self.project_points = nn.Sequential(nn.Linear(2, inter_dim),
                                            nn.LayerNorm(inter_dim),
                                            nn.LeakyReLU(),
                                            nn.Linear(inter_dim, inter_dim))

        self.embedding = nn.Embedding(input_dim, self.embed_dim)
        self.pos_encoder = PositionalEncoding(d_model=self.embed_dim, max_len=input_dim)
        self.inds = torch.tensor(range(int(input_dim / 2))).to(device)


    def get_position_encoding_representation(self, init_points):
        # input dim: init_points [num_frames * points_per_frame, 2], for ballerina [832,2] = [16*52, 2]
        # the input are the points of the given initial frame (user's drawing)
        # note that we calculate the point's distance from the object's center, and operate on this distance
        emb_xy = self.project_points(init_points)  # output shape: [1,num_frames * points_per_frame,128] -> [1,832,128]
        embed = self.embedding(self.inds) * math.sqrt(self.embed_dim)  # inds dim is N*K, embed dim is [N*K, 128]
        pos = self.pos_encoder(embed.unsqueeze(1)).permute(1, 0, 2)  # [1, N*K, 128]
        init_points_pos_enc = emb_xy + pos  # [1, N*K, 128]
        return init_points_pos_enc

    def forward(self, init_points):
        raise NotImplementedError("PointModel is an abstract class. Please inherit from it and implement a forward function.")

    def get_shared_params(self):
        project_points_p = list(self.project_points.parameters())
        embedding_p = list(self.embedding.parameters())
        pos_encoder_p = list(self.pos_encoder.parameters())

        return project_points_p + embedding_p + pos_encoder_p
        
    def get_points_params(self):
        shared_params = self.get_shared_params()
        project_xy_p = list(self.project_xy.parameters())
        model_p = list(self.model.parameters())
        last_lin = list(self.last_linear_layer.parameters())
        return shared_params + project_xy_p + model_p + last_lin
        

class PointMLP(PointModel):
    def __init__(self, input_dim, inter_dim, num_frames, device, inference):

        super().__init__(input_dim, inter_dim, num_frames, device, inference)

        self.project_xy = nn.Sequential(nn.Flatten(),
                                        nn.Linear(int(input_dim * inter_dim / 2), inter_dim),
                                        nn.LayerNorm(inter_dim),
                                        nn.LeakyReLU())

        self.model = nn.Sequential(
            nn.Linear(inter_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.LeakyReLU(),
            nn.Linear(inter_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.LeakyReLU(),
        )

        self.last_linear_layer = nn.Linear(inter_dim, input_dim)

    def forward(self, init_points):
        init_points_pos_enc = self.get_position_encoding_representation(init_points)

        project_xy = self.project_xy(init_points_pos_enc)  # Flatten, output is [1, 128]
        delta = self.model(project_xy)  # [1,128]
        delta_xy = self.last_linear_layer(delta).reshape(init_points.shape)  # [1,128] -> [1, N*K, 2]

        return delta_xy.squeeze(0)

class PainterOptimizer:
    def __init__(self, args, painter):
        self.painter = painter
        self.lr_init = args.lr_init
        self.lr_final = args.lr_final
        self.lr_delay_mult = args.lr_delay_mult
        self.lr_delay_steps = args.lr_delay_steps
        self.lr_bezier = args.lr_bezier
        self.max_steps = args.num_iter
        self.lr_lambda = lambda step: self.learning_rate_decay(step) / self.lr_init
        self.optim_bezier_points = args.optim_bezier_points
        self.init_optimizers()

    def learning_rate_decay(self, step):
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return delay_rate * log_lerp

    def init_optimizers(self):
        if self.optim_bezier_points:
            bezier_delta_params = self.painter.get_bezier_params()
            self.bezier_delta_optimizer = torch.optim.Adam(bezier_delta_params, lr=self.lr_bezier,
                                                           betas=(0.9, 0.9), eps=1e-6)
            self.scheduler_bezier = LambdaLR(self.bezier_delta_optimizer, lr_lambda=self.lr_lambda, last_epoch=-1)

    def update_lr(self):
        if self.optim_bezier_points:
            self.scheduler_bezier.step()

    def zero_grad_(self):
        if self.optim_bezier_points:
            self.bezier_delta_optimizer.zero_grad()
            self.painter.attention.zero_grad()
            self.painter.jacobian_func.zero_grad()

    def step_(self):
        if self.optim_bezier_points:
            self.bezier_delta_optimizer.step()
        if self.painter.fix_start_points and not self.painter.opt_bezier_points_with_mlp:
            with torch.no_grad():
                for i in range(self.painter.num_control_pts):
                    self.painter.parameters_["point_bezier"][i][0] = self.painter.point_bezier_mlp_input[i * 4]

    def get_lr(self, optim="points"):
        if optim == "bezier_points" and self.optim_bezier_points:
            return self.bezier_delta_optimizer.param_groups[0]['lr']
        else:
            return None


def get_center_of_mass(shapes):
    all_points = []
    for shape in shapes:
        all_points.append(shape.points)
    points_vars = torch.vstack(all_points)
    center = points_vars.mean(dim=0)
    return center, all_points


def get_deltas(all_points, center, device):
    deltas_from_center = []
    for points in all_points:
        deltas = (points - center).to(device)
        deltas_from_center.append(deltas)
    return deltas_from_center


def get_contour(svg, cfg, render_size=512):
    svg = svg.copy().normalize(Bbox(render_size, render_size))  # high resolution can produce accurate contour
    src_png = svg.draw(return_png=True, do_display=False)
    s = silhouette(src_png)  # black & white
    s_cont, contour_pts = find_contour_point(s, epsilon=cfg.boundary_simplify_level)
    s.save(osp.join(cfg.mesh_dir, 'silhouette.png'))
    s_cont.save(osp.join(cfg.mesh_dir, 'silhouette_contour.png'))
    contour_pts = contour_pts.astype(np.float32) / (render_size / cfg.render_size_h)
    return contour_pts


def dilate_contour(contour_pts, svg):
    # dilate contour to include all pts of svg
    step = 0.1
    total_step = step
    svg_pts = svg.to_points()
    while not all_inside_contour(svg_pts, contour_pts):
        contour_pts = dilate(contour_pts, step)
        total_step += step
    print('contour expansion:', total_step)
    return contour_pts

# def init_bezier_with_start_point(start_points, W, H, radius=1, device='cpu'):
#     def perturb_point(p, radius=1):
#         return [p[0] + radius * (random.random() - 0.5),
#                 p[1] + radius * (random.random() - 0.5)]
    
#     shapes = []
#     shape_groups = []
    
#     # Loop over start points and generate Bézier paths
#     for j, p0 in enumerate(start_points):
#         # Generate 16 control points with perturbation
#         control_points = [torch.tensor(perturb_point(p0, radius*(1+j/4)*100)).to(device)]
#         for i in range(1, 6):
#             control_points.append(torch.tensor(perturb_point(control_points[i-1], radius)).to(device))
        
#         # Convert control points to tensor for use in path
#         points = torch.stack(control_points)
#         points[:, 0] = points[:, 0].clip(min=5, max=W-5)
#         points[:, 1] = points[:, 1].clip(min=5, max=H-5)
        
#         # Create Bézier path with 16 control points
#         path = pydiffvg.Path(num_control_points=torch.tensor([2]), 
#                              points=points,
#                              stroke_width=torch.tensor(0.5),
#                              is_closed=False)
#         shapes.append(path)
        
#         path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
#                                          fill_color=None,
#                                          stroke_color=torch.tensor([1.0, 0.0, 0.0, 1]))
#         shape_groups.append(path_group)

#     return shapes, shape_groups


# def binomial_coeff(n, i, device):
#     """
#     Efficiently compute the binomial coefficient C(n, i) using a product approach.
#     Ensures that the coefficient is stored as a float tensor.
#     """
#     if i > n - i:  # Use symmetry to minimize the number of terms
#         i = n - i
    
#     # Create a tensor for the result, starting at 1 (as a float)
#     coeff = torch.tensor(1.0, dtype=torch.float32, device=device)
    
#     # Calculate binomial coefficient using a product
#     for j in range(i):
#         coeff *= (n - j) / (j + 1)
    
#     return coeff

# def cubic_bezier(P, t):
#     """
#     Compute the complex Bézier curve for 16 control points at parameter t.
#     P: List of 16 control points, each is a 2D point.
#     t: Parameter t ranging from 0 to 1.
#     """
#     device = P[0].device  # Get the device from the first point
#     n = len(P) - 1  # Degree of Bézier curve (15 for 16 points)
#     B_t = torch.zeros(2, dtype=torch.float32, device=device)  # Initialize result as zero vector
    
#     # Vectorized computation of Bernstein weights and Bézier curve
#     for i in range(n + 1):
#         binomial_coeff_value = binomial_coeff(n, i, device)  # Compute binomial coefficient
#         bernstein_weight = binomial_coeff_value * (1 - t)**(n - i) * t**i
#         B_t += bernstein_weight * P[i]
    
#     return B_t
    
def init_bezier_with_start_point(start_points, W, H, radius=1, device='cpu'):
    def perturb_point(p, radius=1):
        return [p[0] + radius * (random.random() - 0.5),
                p[1] + radius * (random.random() - 0.5)]

    shapes = []
    shape_groups = []
    for j, p0 in enumerate(start_points):
        p1 = perturb_point(p0, radius)
        p2 = perturb_point(p1, radius)
        p3 = perturb_point(p2, radius)
        
        points = torch.tensor(np.array([p0, p1, p2, p3])).to(device)
        points[:, 0] = points[:, 0].clip(min=5, max=W-5)
        points[:, 1] = points[:, 1].clip(min=5, max=H-5)

        path = pydiffvg.Path(num_control_points=torch.tensor([2]),
                             points=points,
                             stroke_width=torch.tensor(0.5),
                             is_closed=False)
        shapes.append(path)  # must `append` before creating path_group
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                         fill_color=None,
                                         stroke_color=torch.tensor([1.0, 0.0, 0.0, 1]))
        shape_groups.append(path_group)

    return shapes, shape_groups

def cubic_bezier(P, t):
    return (1.0-t)**3*P[0] + 3*(1.0-t)**2*t*P[1] + 3*(1.0-t)*t**2*P[2] + t**3*P[3]