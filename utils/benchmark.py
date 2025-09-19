from multiprocessing import Process
from benchmarks.gemm import gemm


REPEAT_COUNT = 1

if __name__ == '__main__':
    procs:list[Process] = []
    for _ in range(1):
        for __ in range(1):
            for variant in range(16):
                # p = Process(target=gemm.test_gemm, args=[variant])
                # p.start()
                # procs.append(p)
                gemm.test_gemm(variant)

        for p in procs:
            p.join()

