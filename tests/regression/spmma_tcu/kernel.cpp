#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::spmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);
  auto pM = reinterpret_cast<ctx::meta_t *>(arg->M_addr); 

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  ctx::fragment_a        fragA;
  ctx::fragment_b        fragB;
  ctx::fragment_acc      fragC;
  ctx::fragment_metadata fragM;

  // calculate tile row & column based on block index
  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  // Initialize accumulator tile to zero
  ctx::fill_fragment(fragC, 0);

  for (int i = 0; i < K; i += ctx::tileK) {
    uint32_t compressed_K = K / 2; 
    auto pTileASparse = pA + (tile_row * compressed_K) + (i / 2); // sparse tiles are half in size, and thus increment half in length (i is halved)
    auto pTileMeta = pMeta + tile_row;

    // Load A tile
    ctx::load_sparse_matrix_sync(fragA, pTileASparse, compressed_K);
    ctx::load_sparse_matrix_sync(fragM, pTileMeta, M);

    // Load B tile
    if constexpr (vt::ITYPE::bits < 8) {
      // For sub-byte matrix B must be in col-major format
      auto pTileB = pB + tile_col * K + i;
      ctx::load_sparse_matrix_sync<vt::col_major>(fragB, pTileB, K);
    } else {
      auto pTileB = pB + i * N + tile_col;
      ctx::load_sparse_matrix_sync(fragB, pTileB, N);
    }

    // Matrix multiply-accumulate: c += a * b
    ctx::mma_sync(fragC, fragA, fragM, fragB, fragC);
  }

  // Store the computed C tile
  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}

int main() {
  auto arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
