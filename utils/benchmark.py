import time
from multiprocessing import Process, get_context

from tqdm import tqdm

from benchmarks.abcbench import gemm, two_mm, nussinov
from utils.xlogging import logger


REPEAT_COUNT = 1

# '0' supresses the corresponding optimization.
variants = [0b00, 0b01, 0b10, 0b11]

if __name__ == '__main__':
    for variant in tqdm(variants, position=0):
        for _ in range(5):
            start_time = time.time()
            exit_state = nussinov.test_nussinov(variant)
            if exit_state != 0:
                break
            end_time = time.time()
            time.sleep(end_time - start_time)



    # for i in range(2**6):
    #     procs:list[Process] = []
    #     for j in range(2**5):
    #         variant = i * (2**5) + j
    #         procs.append(variant)
    #         p = Process(target=two_mm.test_two_mm, args=[variant])
    #         p.start()
    #         procs.append(p)
    #         two_mm.test_two_mm(variant)
    #         gemm.test_gemm(variant)
    #     for p in procs:
    #         p.join()
    #         p.close()
    #     del procs

    # assert procs == list(range(2048))