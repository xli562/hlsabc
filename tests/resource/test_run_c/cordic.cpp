//=========================================================================
// cordic.cpp
//=========================================================================
// @brief : A CORDIC implementation of sine and cosine functions.

#include "cordic.h"
#include <math.h>

#include <iostream>

//-----------------------------------
// cordic function
//-----------------------------------
// @param[in]  : theta - input angle
// @param[out] : s - sine output
// @param[out] : c - cosine output
void cordic(theta_type theta, cos_sin_type &s, cos_sin_type &c, int num_iter) {

  // Set initial vector that we will rotate
  cos_sin_type curr_cos = init_cos[num_iter];
  cos_sin_type curr_sin = 0.0;

  // The 2^(-L) value
  cos_sin_type factor = 1.0;

  #ifdef FIXED_TYPE // fixed-point design
  FIXED_STEP_LOOP:
  for (int step = 0; step < 20; step++) {
      // Determine if we are rotating by a positive or negative angle
      #pragma HLS unroll
      // HLS_PIPE(InitIntv)
      // HLS_ARRAY_PARTITION(PART_MODE, PART_DIM, PART_FACT)
      // #pragma HLS pipeline II=1
      // #pragma HLS array_partition block variable=cordic_ctab dim=0 factor=128
      if (theta > 0) {
        // save curr_cos for the later sine calculation
        cos_sin_type temp_cos = curr_cos;

        // do rotation
        curr_cos = curr_cos - (curr_sin >> step);
        curr_sin = curr_sin + (temp_cos >> step);

        // update theta
        theta = theta - cordic_ctab[step];

        // calculate the next 2^(-L) value
        factor = factor / 2;
      } else {
        // save curr_cos for the later sine calculation
        cos_sin_type temp_cos = curr_cos;

        // do rotation
        curr_cos = curr_cos + (curr_sin >> step);
        curr_sin = curr_sin - (temp_cos >> step);

        // update theta
        theta = theta + cordic_ctab[step];
      }
    }

  #else // floating point design

  FLOAT_STEP_LOOP:
    for (int step = 0; step < NUM_ITER; step++) {
      // Determine if we are rotating by a positive or negative angle
      int sigma = (theta > 0) ? 1 : -1;

      // save curr_cos for the later sine calculation
      cos_sin_type temp_cos = curr_cos;

      // do rotation
      curr_cos = curr_cos - sigma * curr_sin * factor;
      curr_sin = curr_sin + sigma * temp_cos * factor;

      // update theta
      theta = theta - sigma * cordic_ctab[step];

      // calculate the next 2^(-L) calue
      factor = factor / 2;
    }

  #endif

    // store the result
    s = curr_sin;
    c = curr_cos;
}
