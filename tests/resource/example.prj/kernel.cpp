
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
) {	// L2
  for (int v2 = 0; v2 < 8; v2++) {	// L5
    for (int v3 = 0; v3 < 2; v3++) {	// L5
      for (int v4 = 0; v4 < 4; v4++) {	// L5
        v1[v2][v3][v4] = 0;	// L5
      }
    }
  }
  l_S_y_x_0_y: for (int y = 0; y < 8; y++) {	// L6
    l_x: for (int x = 0; x < 8; x++) {	// L7
      int32_t v;	// L8
      v = 0;	// L9
      l_S_r_c_0_r: for (int r = 0; r < 3; r++) {	// L10
        l_c: for (int c = 0; c < 3; c++) {	// L11
          int32_t v10 = v0[(y + r)][(x + c)];	// L12
          int32_t v11 = v;	// L13
          int32_t v12 = v11 + v10;	// L14
          v = v12;	// L15
        }
      }
      int32_t v13 = v;	// L18
      v1[(y + ((x / 4) / 2))][((x / 4) % 2)][(x % 4)] = v13;	// L19
    }
  }
}

