module {
  func.func @conv2D_lb_wb_schedule(%arg0: memref<10x10xi32>) -> memref<8x8xi32> attributes {itypes = "s", otypes = "s"} {
    %alloc = memref.alloc() {name = "B"} : memref<8x8xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<8x8xi32>)
    affine.for %arg1 = 0 to 8 {
      affine.for %arg2 = 0 to 8 {
        %alloc_0 = memref.alloc() {name = "v"} : memref<i32>
        %c0_i32_1 = arith.constant 0 : i32
        affine.store %c0_i32_1, %alloc_0[] {to = "v"} : memref<i32>
        affine.for %arg3 = 0 to 3 {
          affine.for %arg4 = 0 to 3 {
            %1 = affine.load %arg0[%arg1 + %arg3, %arg2 + %arg4] {from = "A"} : memref<10x10xi32>
            %2 = affine.load %alloc_0[] {from = "v"} : memref<i32>
            %3 = arith.addi %2, %1 : i32
            affine.store %3, %alloc_0[] {to = "v"} : memref<i32>
          } {loop_name = "c", reduction}
        } {loop_name = "r", op_name = "S_r_c_0", reduction}
        %0 = affine.load %alloc_0[] {from = "v"} : memref<i32>
        affine.store %0, %alloc[%arg1, %arg2] {to = "B"} : memref<8x8xi32>
      } {loop_name = "x"}
    } {loop_name = "y", op_name = "S_y_x_0"}
    return %alloc : memref<8x8xi32>
  }
}
