import os
import sys

from tbe import tik

sys.path.append(os.path.dirname(__file__))

from tiling_strategy.strategy import TilingStrategy
from attention import FlashAttention

from constants import FP16
from constants import FP32
from constants import INT32
from constants import GM
from constants import L1
from constants import UB


class FlashAttentionBwd(FlashAttention):
    """The implementation of FlashAttention backward
    This function contains the flash attention backward implementation used in flash attention (see paper)
    `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/pdf/2205.14135.pdf>`
    """

    def __init__(self, q, k, v, O, dO, l, m, attn_mask, dropout_mask, alibi_mask,
                 prev_block_num,
                 next_block_num,
                 high_precision,
                 kernel_name,
                 tiling_stgy: TilingStrategy,
                 disable_debug):
        super().__init__(q, k, v, attn_mask, dropout_mask, alibi_mask, kernel_name,
                         tiling_stgy, prev_block_num, next_block_num, high_precision, disable_debug)

        if isinstance(q, dict):
            self.dO_shape = dO["shape"]  # [B, Nq, d]
        else:
            self.dO_shape = dO.shape

        self.dV_shape = self.v_shape
        self.dQ_shape = self.q_shape
        self.dK_shape = self.k_shape

    def define_outputs(self):
        """define output gm tensors"""
        self.dQ_gm = self.tik_instance.Tensor(FP32, self.dQ_shape, name="dQ_gm", scope=GM, is_atomic_add=True)
        self.dK_gm = self.tik_instance.Tensor(FP32, self.dK_shape, name="dK_gm", scope=GM, is_atomic_add=True)
        self.dV_gm = self.tik_instance.Tensor(FP32, self.dV_shape, name="dV_gm", scope=GM, is_atomic_add=True)

    def define_custom_inputs(self):
        """define input gm tensors"""
        self.O_gm = self.tik_instance.Tensor(FP16, self.O_shape, name="O_gm", scope=GM)
        self.dO_gm = self.tik_instance.Tensor(FP16, self.dO_shape, name="dO_gm", scope=GM)
        self.l_gm = self.tik_instance.Tensor(self.precision_type, self.l_shape, name="l_gm", scope=GM)
        self.m_gm = self.tik_instance.Tensor(FP16, self.m_shape, name="m_gm", scope=GM)

    def collect_inputs(self):
        """collect all input gm tensors into input_gm_list,
        the input list should keep order with the para order in Primitive and init
        """
        input_gm_list = [
            self.Q_gm, self.K_gm, self.V_gm, self.O_gm, self.dO_gm, self.l_gm,
            self.m_gm
        ]
        if self.has_attn_mask:
            input_gm_list.append(self.att_mask_gm)
        if self.has_drop_mask:
            input_gm_list.append(self.drop_mask_gm)
        if self.has_alibi_mask:
            input_gm_list.append(self.alibi_mask_gm)
        return input_gm_list

    def compute_Pij(self, Qi_l1_K1MK0_ed, KjT_l1_K1NK0_ed, m, k, n, lm_gm_offset, attn_mask_gm_offset,
                    dropout_mask_gm_offset, alibi_mask_gm_offset):
        """Refer to Algorithm 4 line11-14 in FlashAttention implement Pij computation"""
        m_aligned = self.tik_ops_utils.up_align_to_K0(m)
        n_aligned = self.tik_ops_utils.up_align_to_K0(n)
        # Sij <- Qi * KjT
        Sij_ub = self.tik_ops_utils.matmul_compute(Qi_l1_K1MK0_ed, KjT_l1_K1NK0_ed, m, k, n, N1MN0_to_MN=False)
        if self.has_drop_mask:
            Pij_drop_ed_ub = self.tik_instance.Tensor(FP16, (n_aligned // self.N0, m_aligned, self.N0),
                                                      name="Pij_drop_ed_ub", scope=UB)

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # Sij <- Sij / sqrt(d) TODO NZ 适配 alibi
            if self.has_alibi_mask:
                self.do_alibi_mask(Sij_ub, alibi_mask_gm_offset, m_aligned, n_aligned)
            # att_mask
            if self.has_attn_mask:
                self.do_att_mask(Sij_ub, attn_mask_gm_offset, m, n, m_aligned, n_aligned)

            # move li (ith block of l_gm) and mi (ith block of m_gm) from gm to ub
            li_ub = self.tik_instance.Tensor(self.precision_type, (m_aligned,), name="li_ub", scope=UB)
            mi_ub = self.tik_instance.Tensor(FP16, (m_aligned,), name="mi_ub", scope=UB)
            self.tik_ops_utils.move_vector_from_gm_to_ub(li_ub, self.l_gm, lm_gm_offset, m)
            self.tik_ops_utils.move_vector_from_gm_to_ub(mi_ub, self.m_gm, lm_gm_offset, m)
            # Sij <- Sij - mi (Br, Bc) (Br)
            n1 = n_aligned // self.N0
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                broadcast_mi_ub = self.tik_ops_utils.broadcast(mi_ub, (m, self.N0))
                broadcast_mi_ub = broadcast_mi_ub.reshape((1, m, self.N0))
                for idx in range(n1):
                    self.tik_instance.h_sub(Sij_ub[idx, :, :], Sij_ub[idx, :, :], broadcast_mi_ub)
            # Sij <- diag(li)^-1 * Sij
            li_rec_ub = self.tik_ops_utils.calc_vec_rec(li_ub, m)
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                if self.high_precision:
                    # fp16 -> fp32
                    Sij_ub_fp32 = self.tik_instance.Tensor(FP32, (n_aligned // self.N0, m_aligned, self.N0),
                                                           name="Sij_ub_fp32", scope=UB)
                    self.tik_instance.h_cast(Sij_ub_fp32, Sij_ub, "none")
                    # Sij <- exp(Sij)
                    self.tik_instance.h_exp(Sij_ub_fp32, Sij_ub_fp32)
                    # Sij <- diag(li)^-1 * Sij
                    # TODO: fp32广播优化
                    cur_row_sum_rec = self.tik_instance.Tensor(FP32, (m_aligned, self.N0), name="cur_row_sum_rec",
                                                               scope=UB)
                    for i in range(m_aligned):
                        src_scalar = self.tik_instance.Scalar(init_value=li_rec_ub[i], dtype=FP32)
                        self.tik_instance.h_duplicate(cur_row_sum_rec[i, :], src_scalar)
                    cur_row_sum_rec = cur_row_sum_rec.reshape((1, m_aligned, self.N0))
                    with self.tik_instance.for_range(0, n_aligned // self.N0) as idx:
                        self.tik_instance.h_mul(Sij_ub_fp32[idx, :, :], Sij_ub_fp32[idx, :, :], cur_row_sum_rec)
                    # fp32 -> fp16
                    self.tik_instance.h_cast(Sij_ub, Sij_ub_fp32, "none")
                else:
                    # Sij <- exp(Sij)
                    self.tik_instance.h_exp(Sij_ub, Sij_ub)
                    broadcast_li_rec_ub = self.tik_ops_utils.broadcast(li_rec_ub, (m_aligned, self.N0))
                    broadcast_li_rec_ub = broadcast_li_rec_ub.reshape((1, m_aligned, self.N0))
                    for idx in range(n1):
                        self.tik_instance.h_mul(Sij_ub[idx, :, :], Sij_ub[idx, :, :], broadcast_li_rec_ub)

            # dropout_mask
            if self.has_drop_mask:
                self.do_dropout_mask(Sij_ub, dropout_mask_gm_offset, n_aligned, n, m_aligned, m, workspace=Pij_drop_ed_ub)
            else:
                Pij_drop_ed_ub = Sij_ub

        return Sij_ub, Pij_drop_ed_ub

    def compute_Di(self, Di_ub, dOi_ub, qo_gm_offset, q_blk_height):
        """Refer to Algorithm 4 line19 in FlashAttention implement Di computation"""
        q_blk_height_aligned = self.tik_ops_utils.up_align_to_K0(q_blk_height)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            Oi_ub = self.tik_instance.Tensor(FP16, (self.d // self.N0, q_blk_height_aligned, self.N0),
                                             scope=UB, name="Oi_ub")
            self.tik_instance.data_move(dst=Oi_ub, src=self.O_gm[qo_gm_offset],
                                        sid=0, nburst=self.N1, burst=q_blk_height * self.N0 // 16,
                                        src_stride=(self.Nq - q_blk_height) * self.N0 // 16, dst_stride=0)
            # dOi*Oi (elementwise)
            self.tik_instance.h_mul(Oi_ub, dOi_ub, Oi_ub)
            # rowsum (dOi*Oi)
            dOi_Oi_l1 = self.tik_instance.Tensor(FP16, (self.d // self.N0, q_blk_height_aligned, self.N0),
                                                 name="dOi_Oi_l1", scope=L1)

            self.cont_data_mv_1_bust(dst=dOi_Oi_l1, src=Oi_ub, burst=q_blk_height_aligned * self.d // 16)
            self.tik_ops_utils.row_sum_cube_impl(dOi_Oi_l1, Di_ub, q_blk_height,
                                                 self.actual_d, precision_type=FP16)

    def compute_dSij(self, Pij_ub, dOi_l1_K1MK0_ed, VjT_K1NK0_ed, Di_ub, kv_blk_height, q_blk_height,
                     dropout_mask_gm_offset):
        """Refer to Algorithm 4 line20 in FlashAttention implement dSij computation"""
        with self.tik_instance.new_stmt_scope(disable_sync=False):  # 为了释放dPij_ub
            dPij_ub = self.tik_ops_utils.matmul_compute(dOi_l1_K1MK0_ed, VjT_K1NK0_ed,
                                                        q_blk_height, self.actual_d, kv_blk_height, N1MN0_to_MN=False)
            q_blk_height_aligned = self.tik_ops_utils.up_align_to_K0(q_blk_height)
            kv_blk_height_aligned = self.tik_ops_utils.up_align_to_K0(kv_blk_height)
            # dropout_mask
            if self.has_drop_mask:
                self.do_dropout_mask(dPij_ub, dropout_mask_gm_offset, kv_blk_height_aligned, kv_blk_height,
                                     q_blk_height_aligned, q_blk_height)
            # dPij - Di
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                broadcast_Di_ub = self.tik_ops_utils.broadcast(Di_ub, (q_blk_height_aligned, self.N0))
                broadcast_Di_ub = broadcast_Di_ub.reshape((1, q_blk_height_aligned, self.N0))
                n1 = kv_blk_height_aligned // self.N0
                for idx in range(n1):
                    self.tik_instance.h_sub(dPij_ub[idx, :, :], dPij_ub[idx, :, :], broadcast_Di_ub)
            # Pij * (dPij - Di) (elementwise)
            self.tik_instance.h_mul(Pij_ub, Pij_ub, dPij_ub)  # 复用Pij_ub内存
        return Pij_ub

    def update_dVj(self,
                   PijT_l1_K1MK0_ed,
                   dOi_l1_K1NK0_ed,
                   kv_gm_offset,
                   kv_blk_height,
                   q_blk_height):
        """Refer to Algorithm 4 line16 in FlashAttention implement dVj update"""
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            PijT_Oi_ub = self.tik_ops_utils.matmul_compute(PijT_l1_K1MK0_ed, dOi_l1_K1NK0_ed,
                                                           kv_blk_height, q_blk_height,
                                                           self.actual_d, N1MN0_to_MN=False,
                                                           precision_type=FP32)
            self.tik_instance.set_atomic_add(1)  # 实测 2 (fp16) 不支持
            self.tik_instance.data_move(dst=self.dV_gm[kv_gm_offset], src=PijT_Oi_ub, sid=0,
                                        nburst=self.N1, burst=kv_blk_height * self.N0 // 8,
                                        src_stride=0, dst_stride=(self.Nq - kv_blk_height) * self.N0 // 8)
            self.tik_instance.set_atomic_add(0)

    def update_dQi(self,
                   dSij_l1_K1MK0_ed,
                   Kj_l1_K1NK0_ed,
                   qo_gm_offset,
                   q_blk_height,
                   kv_blk_height):
        """Refer to Algorithm 4 line21 in FlashAttention implement dQi update"""
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            dSij_Kj_ub = self.tik_ops_utils.matmul_compute(dSij_l1_K1MK0_ed, Kj_l1_K1NK0_ed,
                                                           q_blk_height, kv_blk_height,
                                                           self.actual_d, N1MN0_to_MN=False, precision_type=FP32)
            self.tik_instance.set_atomic_add(1)
            self.tik_instance.data_move(dst=self.dQ_gm[qo_gm_offset], src=dSij_Kj_ub, sid=0,
                                        nburst=self.d // self.N0, burst=q_blk_height * self.N0 // 8,
                                        src_stride=0, dst_stride=(self.Nq - q_blk_height) * self.N0 // 8)
            self.tik_instance.set_atomic_add(0)

    def update_dKj(self,
                   dSijT_l1_K1MK0_ed,
                   Qi_l1_K1NK0_ed,
                   kv_gm_offset,
                   kv_blk_height,
                   q_blk_height):
        """Refer to Algorithm 4 line22 in FlashAttention implement dKi update"""
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            dSijT_Qi_ub = self.tik_ops_utils.matmul_compute(dSijT_l1_K1MK0_ed, Qi_l1_K1NK0_ed,
                                                            kv_blk_height, q_blk_height,
                                                            self.actual_d, N1MN0_to_MN=False, precision_type=FP32)
            self.tik_instance.set_atomic_add(1)
            self.tik_instance.data_move(dst=self.dK_gm[kv_gm_offset], src=dSijT_Qi_ub, sid=0,
                                        nburst=self.d // self.N0, burst=kv_blk_height * self.N0 // 8,
                                        src_stride=0, dst_stride=(self.Nq - kv_blk_height) * self.N0 // 8)
            self.tik_instance.set_atomic_add(0)

    def compute_in_each_kv_block(self, batch_start, batch_idx, kv_blk_idx, kv_blk_height,
                                 core_idx_to_tr_info, core_idx):
        """The backward computation in each outer loop"""
        kv_blk_height_aligned = self.tik_ops_utils.up_align_to_K0(kv_blk_height)
        kv_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.N, self.d,
                                          self.Bc, kv_blk_idx)
        # load KjT
        Kj_l1_1_K1MK0 = self.tik_instance.Tensor(FP16, (self.d // self.N0, kv_blk_height_aligned, self.N0),
                                                 name="Kj_l1_1_K1MK0",
                                                 scope=L1)  # for Qi*KjT (算法11行)
        self.tik_instance.data_move(dst=Kj_l1_1_K1MK0, src=self.K_gm[kv_gm_offset],
                                    sid=0, nburst=self.N1, burst=kv_blk_height_aligned * self.N0 // 16,
                                    src_stride=(self.N - kv_blk_height_aligned) * self.N0 // 16, dst_stride=0)

        # load Kj
        Kj_l1_2 = self.tik_instance.Tensor(FP16, (kv_blk_height_aligned, self.d), name="Kj_l1_2",
                                           scope=L1)  # for dSij*Kj （算法21行）
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            Kj_ub = self.tik_instance.Tensor(FP16, (self.d // self.N0, kv_blk_height_aligned, self.N0),
                                             name="Kj_ub", scope=UB)
            self.tik_instance.data_move(dst=Kj_ub, src=self.K_gm[kv_gm_offset],
                                        sid=0, nburst=self.N1, burst=kv_blk_height_aligned * self.N0 // 16,
                                        src_stride=(self.N - kv_blk_height_aligned) * self.N0 // 16, dst_stride=0)
            # (N1, K, N0) -> (K, N)
            Kj_ub = self.tik_ops_utils.N1MN0_TO_MN(Kj_ub)
            # (K, N) -> (K1, N, K0)
            Kj_l1_2_K1NK0_ed = self.tik_ops_utils.KN_TO_K1NK0(Kj_ub, workspace_tensor=Kj_l1_2)

        # load VjT
        Vj_l1 = self.tik_instance.Tensor(FP16, (self.d // self.N0, kv_blk_height_aligned, self.N0), name="Vj_l1",
                                         scope=L1)
        self.tik_instance.data_move(dst=Vj_l1, src=self.V_gm[kv_gm_offset],
                                    sid=0, nburst=self.N1, burst=kv_blk_height_aligned * self.N0 // 16,
                                    src_stride=(self.N - kv_blk_height_aligned) * self.N0 // 16, dst_stride=0)

        tr_start_s = self.tik_instance.Scalar("int32", name="tr_start_s")
        tr_end_s = self.tik_instance.Scalar("int32", name="tr_end_s")
        tr_start_s.set_as(core_idx_to_tr_info[core_idx, batch_start + batch_idx, 0])
        tr_end_s.set_as(core_idx_to_tr_info[core_idx, batch_start + batch_idx, 1])
        with self.tik_instance.for_range(tr_start_s, tr_end_s, name="q_blk_idx") as q_blk_idx:
            # 根据 atten_mask倒三角特性，过滤无效计算
            with self.tik_instance.if_scope(tik.all(kv_blk_idx - self.next_block_num <= q_blk_idx,
                                                    q_blk_idx <= kv_blk_idx + self.prev_block_num)):
                with self.tik_instance.if_scope(q_blk_idx != self.Tr - 1):
                    self.compute_in_each_q_block(Kj_l1_1_K1MK0,
                                                 Kj_l1_2_K1NK0_ed,
                                                 Vj_l1,
                                                 batch_idx,
                                                 batch_start,
                                                 kv_gm_offset,
                                                 kv_blk_height,
                                                 self.Br,
                                                 kv_blk_idx,
                                                 q_blk_idx)
                with self.tik_instance.else_scope():
                    self.compute_in_each_q_block(Kj_l1_1_K1MK0,
                                                 Kj_l1_2_K1NK0_ed,
                                                 Vj_l1,
                                                 batch_idx,
                                                 batch_start,
                                                 kv_gm_offset,
                                                 kv_blk_height,
                                                 self.last_Br,
                                                 kv_blk_idx,
                                                 q_blk_idx)

    def compute_in_each_q_block(self, KjT_l1_K1NK0_ed, Kj_l1_K1NK0_ed, VjT_l1_K1NK0_ed,
                                batch_idx, batch_start, kv_gm_offset, kv_blk_height,
                                q_blk_height, kv_blk_idx, q_blk_idx):
        """The backward computation in each inner loop"""
        kv_blk_height_alig = self.tik_ops_utils.up_align_to_K0(kv_blk_height)
        q_blk_height_alig = self.tik_ops_utils.up_align_to_K0(q_blk_height)

        # load Qi并做分形 (L1: 256K)
        qo_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.Nq, self.d, self.Br, q_blk_idx)
        # load dOi并做分形，提前计算Di，否则dOi_ub无法释放，或者后面计算Di时需要再次load dOi (L1: 320K)
        dOi_l1_right = self.tik_instance.Tensor(FP16, (q_blk_height_alig, self.d), name="dOi_l1_right",
                                                scope=L1)  # for PijT*dOi (算法16行)
        Di_ub = self.tik_instance.Tensor(FP16, (q_blk_height_alig,), name="Di_ub", scope=UB)
        # 为了释放dOi_ub，提前计算Di
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            dOi_ub = self.tik_instance.Tensor(FP16, (self.d // self.N0, q_blk_height_alig, self.N0),
                                              name="dOi_ub", scope=UB)
            self.tik_instance.data_move(dst=dOi_ub, src=self.dO_gm[qo_gm_offset],
                                        sid=0, nburst=self.N1, burst=q_blk_height_alig * self.N0 // 16,
                                        src_stride=(self.Nq - q_blk_height_alig) * self.N0 // 16, dst_stride=0)

            self.compute_Di(Di_ub, dOi_ub, qo_gm_offset, q_blk_height)
            # (N1, K, N0) -> (K, N)
            dOi_ub = self.tik_ops_utils.N1MN0_TO_MN(dOi_ub)
            # (K, N) -> (K1, N, K0)
            dOi_l1_K1NK0_ed = self.tik_ops_utils.KN_TO_K1NK0(dOi_ub, workspace_tensor=dOi_l1_right)

        Qi_l1_K1MK0 = self.tik_instance.Tensor(FP16, (self.d // self.N0, q_blk_height_alig, self.N0),
                                               name="Qi_l1_K1MK0",
                                               scope=L1)  # for Qi*KjT (算法11行)
        self.tik_instance.data_move(dst=Qi_l1_K1MK0, src=self.Q_gm[qo_gm_offset],
                                    sid=0, nburst=self.N1, burst=q_blk_height_alig * self.N0 // 16,
                                    src_stride=(self.Nq - q_blk_height_alig) * self.N0 // 16, dst_stride=0)

        # load Qi_right
        Qi_l1_right = self.tik_instance.Tensor(FP16, (q_blk_height_alig, self.d), name="Qi_l1_right",
                                               scope=L1)  # for dSijT*Qi (算法22行)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            Qi_ub = self.tik_instance.Tensor(FP16, (self.d // self.N0, q_blk_height_alig, self.N0),
                                             name="Qi_ub", scope=UB)  # ub是不是可以重复利用，上面的datamove就直接用这个move给l1
            self.tik_instance.data_move(dst=Qi_ub, src=self.Q_gm[qo_gm_offset],
                                        sid=0, nburst=self.N1, burst=q_blk_height_alig * self.N0 // 16,
                                        src_stride=(self.N - q_blk_height_alig) * self.N0 // 16, dst_stride=0)
            # (N1, K, N0) -> (K, N)
            Qi_ub = self.tik_ops_utils.N1MN0_TO_MN(Qi_ub)
            # (K, N) -> (K1, N, K0)
            Qi_l1_K1NK0_ed = self.tik_ops_utils.KN_TO_K1NK0(Qi_ub, workspace_tensor=Qi_l1_right)

        lm_gm_offset = self.get_l_m_gm_offset(batch_start, batch_idx, self.Nq, self.Br, q_blk_idx)
        attn_mask_gm_offset, dropout_mask_gm_offset, alibi_mask_gm_offset = None, None, None
        if self.has_attn_mask:
            attn_mask_gm_offset = self.get_attn_mask_gm_offset(batch_start, batch_idx, self.Nq, self.N,
                                                               self.Br, q_blk_idx, self.Bc, kv_blk_idx)
        if self.has_drop_mask:
            dropout_mask_gm_offset = self.get_drop_mask_gm_offset(batch_start, batch_idx, self.Nq, self.N,
                                                                  self.Br, q_blk_idx, self.Bc, kv_blk_idx)
        if self.has_alibi_mask:
            alibi_mask_gm_offset = self.get_alibi_gm_offset(batch_start, batch_idx, self.N, self.Bc, kv_blk_idx)
        # 算法11~15行
        Pij_ub, Pij_drop_ed_ub = self.compute_Pij(Qi_l1_K1MK0, KjT_l1_K1NK0_ed,
                                                  q_blk_height, self.actual_d, kv_blk_height,
                                                  lm_gm_offset, attn_mask_gm_offset,
                                                  dropout_mask_gm_offset, alibi_mask_gm_offset)

        dOi_l1_K1MK0 = self.tik_instance.Tensor(FP16, (self.d // self.N0, q_blk_height_alig, self.N0),
                                                name="dOi_l1_K1MK0",
                                                scope=L1)  # for dOi * VjT (算法17行) okk

        self.tik_instance.data_move(dst=dOi_l1_K1MK0, src=self.dO_gm[qo_gm_offset],
                                    sid=0, nburst=self.N1, burst=q_blk_height_alig * self.N0 // 16,
                                    src_stride=(self.Nq - q_blk_height_alig) * self.N0 // 16, dst_stride=0)
        # 分形，为PijT*dOi(算法16行)做准备
        # 左矩阵转置后做MK->K1MK0分形等价于不做转置的KN->K1NK0分形 (L1: 384K)
        Pij_l1 = self.tik_instance.Tensor(FP16, (q_blk_height_alig, kv_blk_height_alig), name="Pij_l1", scope=L1)
        Pij_drop_ed_ub = self.tik_ops_utils.N1MN0_TO_MN(Pij_drop_ed_ub)
        PijT_l1_K1MK0_ed = self.tik_ops_utils.KN_TO_K1NK0(Pij_drop_ed_ub, workspace_tensor=Pij_l1)
        # 算法24行：write dKj dVj (放入内循环=>增大tiling的blocksize)
        self.update_dVj(PijT_l1_K1MK0_ed, dOi_l1_K1NK0_ed,
                        kv_gm_offset, kv_blk_height, q_blk_height)
        # (L1: 512K)
        dSij_l1_K1MK0_ed = self.tik_instance.Tensor(FP16, (kv_blk_height_alig // self.N0, q_blk_height_alig, self.N0),
                                                    name="dSij_l1_1", scope=L1)  # for dSij*Kj (算法21行)
        dSij_l1_2 = self.tik_instance.Tensor(FP16, (q_blk_height_alig, kv_blk_height_alig),
                                             name="dSij_l1_2", scope=L1)  # for dSijT*Qi (算法22行)
        with self.tik_instance.new_stmt_scope(disable_sync=False):  # 为了释放dSij_ub
            dSij_ub = self.compute_dSij(Pij_ub,
                                        dOi_l1_K1MK0,
                                        VjT_l1_K1NK0_ed,
                                        Di_ub,
                                        kv_blk_height,
                                        q_blk_height,
                                        dropout_mask_gm_offset)
            # for dSij*Kj (算法21行)
            self.cont_data_mv_1_bust(dst=dSij_l1_K1MK0_ed, src=dSij_ub,
                                     burst=kv_blk_height_alig * q_blk_height_alig // 16)
            dSij_ub = self.tik_ops_utils.N1MN0_TO_MN(dSij_ub)
            # for dSijT*Qi (算法22行)
            dSijT_l1_K1MK0_ed = self.tik_ops_utils.KN_TO_K1NK0(dSij_ub, workspace_tensor=dSij_l1_2)
        # 算法21行
        self.update_dQi(dSij_l1_K1MK0_ed, Kj_l1_K1NK0_ed,
                        qo_gm_offset, q_blk_height, kv_blk_height)
        # 算法22行
        self.update_dKj(dSijT_l1_K1MK0_ed, Qi_l1_K1NK0_ed,
                        kv_gm_offset, kv_blk_height, q_blk_height)

    def compute_one_core(self, batch_start_sc, batch_num_sc, core_idx_to_tr_info, core_idx):
        """The computation of FlashAttention backward on each core"""
        with self.tik_instance.for_range(0, batch_num_sc, name="batch_index") as batch_idx:
            with self.tik_instance.for_range(0, self.Tc, name="kv_blk_idx") as kv_blk_idx:
                with self.tik_instance.if_scope(kv_blk_idx != self.Tc - 1):
                    self.compute_in_each_kv_block(batch_start_sc, batch_idx, kv_blk_idx, self.Bc,
                                                  core_idx_to_tr_info, core_idx)
                with self.tik_instance.else_scope():
                    self.compute_in_each_kv_block(batch_start_sc, batch_idx, kv_blk_idx, self.last_Bc,
                                                  core_idx_to_tr_info, core_idx)

    def compute_process(self):
        """The compute process of FlashAttention backward"""
        self.init()

        core_idx_to_batch_info, core_idx_to_tr_info = self.get_core_bath_info()
        with self.tik_instance.for_range(begint=0, endt=self.core_num, name="core_index",
                                         block_num=self.core_num) as core_idx:
            batch_start_s = self.tik_instance.Scalar("int32", name="batch_start_s")
            batch_num_s = self.tik_instance.Scalar("int32", name="batch_num_s")

            batch_start_s.set_as(core_idx_to_batch_info[core_idx, 0])
            batch_num_s.set_as(core_idx_to_batch_info[core_idx, 1])

            self.compute_one_core(batch_start_s, batch_num_s, core_idx_to_tr_info, core_idx)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=self.collect_inputs(),
            outputs=self.collect_outputs(),
            config={"dump_cce_code": False, "save_temp_cce_file": True, "enable_const_fold": True},
            enable_l2=True
        )

    def collect_outputs(self):
        """collect all output gm tensors into output_gm_list,
        the output list should keep order with the para order in Primitive and init
        """
        output_gm_list = [self.dQ_gm, self.dK_gm, self.dV_gm]
        return output_gm_list


def flash_attention_grad(Q, K, V, O, dO, l, m, attn_mask, dropout_mask, alibi_mask, dq, dk, dv,
                         prev_block_num=65536,
                         next_block_num=65536,
                         high_precision=False,
                         tiling_stgy_name='xunfei',
                         kernel_name="flash_attention_grad",
                         disable_debug=True):
    """
    algorithm: flash_attention_backward

    Parameters
    ----------
    Q : dict. shape and dtype of input, only support float16
    K : dict. shape and dtype of input, only support float16
    V: dict. shape and dtype of input, only support float16
    O: dict. shape and dtype of input, only support float16
    dO: dict. shape and dtype of input, only support float16
    l: dict. shape and dtype of input, only support float16
    m: dict. shape and dtype of input, only support float16
    dropout_mask: dict. shape and dtype of input, only support float16
    dropout_mask: dict. shape and dtype of input, only support float16
    alibi_mask: dict. shape and dtype of input, only support float16
    dq: dict. shape and dtype of output, only support float16
    dk: dict. shape and dtype of output, only support float16
    dv: dict. shape and dtype of output, only support float16
    prev_block_num: int. an attribute used to define sparse attention
    next_block_num: int. an attribute used to define sparse attention
    tiling_stgy_name: str. an attribute used to choose the tiling strategy
    kernel_name: str. cce kernel name, default value is real_div
    disable_debug: bool. whether disable debug

    Returns
    -------
    tik_instance
    """
    fa_grad = FlashAttentionBwd(Q, K, V, O, dO, l, m, attn_mask, dropout_mask,
                                alibi_mask, prev_block_num=prev_block_num,
                                next_block_num=next_block_num,
                                high_precision=high_precision,
                                kernel_name=kernel_name,
                                tiling_stgy=TilingStrategy.from_strategy_name(tiling_stgy_name),
                                disable_debug=disable_debug)
    fa_grad.compute_process()
    return fa_grad.tik_instance
