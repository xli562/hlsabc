//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;
void conv2D_lb_wb_schedule(
  int32_t v0[10][10],
  int32_t v1[8][2][4]
) {	// L4
  for (int v2 = 0; v2 < 8; v2++) {	// L7
    for (int v3 = 0; v3 < 2; v3++) {	// L7
      for (int v4 = 0; v4 < 4; v4++) {	// L7
        v1[v2][v3][v4] = 0;	// L7
      }
    }
  }
  int32_t S_y_x_0_reuse_1[3][10];	// L8
  int32_t S_y_x_0_reuse_2[3][3];	// L9
  l_S_y_x_0_x_outer: for (int x_outer = 0; x_outer < 2; x_outer++) {	// L10
    l_y: for (int y = 0; y < 10; y++) {	// L11
      for (int v9 = 0; v9 < 10; v9++) {	// L12
        int32_t v10 = S_y_x_0_reuse_1[1][v9];	// L13
        S_y_x_0_reuse_1[0][v9] = v10;	// L14
        int32_t v11 = S_y_x_0_reuse_1[2][v9];	// L15
        S_y_x_0_reuse_1[1][v9] = v11;	// L16
        int32_t v12 = v0[y][v9];	// L17
        S_y_x_0_reuse_1[2][v9] = v12;	// L18
      }
      if ((y - 2) >= 0) {	// L20
        l_x_inner: for (int x_inner = 0; x_inner < 6; x_inner++) {	// L21
          for (int v14 = 0; v14 < 3; v14++) {	// L22
            int32_t v15 = S_y_x_0_reuse_2[v14][1];	// L23
            S_y_x_0_reuse_2[v14][0] = v15;	// L24
            int32_t v16 = S_y_x_0_reuse_2[v14][2];	// L25
            S_y_x_0_reuse_2[v14][1] = v16;	// L26
            int v17 = (x_inner + (x_outer * 4));	// L27
            int32_t v18 = S_y_x_0_reuse_1[v14][v17];	// L28
            S_y_x_0_reuse_2[v14][2] = v18;	// L29
          }
          if ((x_inner - 2) >= 0) {	// L31
            int v19 = (x_inner + (x_outer * 4));	// L32
            int32_t v;	// L33
            v = 0;	// L34
            l_S_r_c_0_r: for (int r = 0; r < 3; r++) {	// L35
              l_c: for (int c = 0; c < 3; c++) {	// L36
                int32_t v23 = S_y_x_0_reuse_2[r][c];	// L37
                int32_t v24 = v;	// L38
                int32_t v25 = v24 + v23;	// L39
                v = v25;	// L40
              }
            }
            int32_t v26 = v;	// L43
            v1[((y + (((v19 - 2) / 4) / 2)) - 2)][(((v19 - 2) / 4) % 2)][((v19 + (((v19 - 2) / 4) * -4)) - 2)] = v26;	// L44
          }
        }
      }
    }
  }
}
