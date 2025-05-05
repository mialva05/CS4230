from mpi4py import MPI
import numpy as np
import math

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def fw_parallel(start_row, end_row, matrix):
    for k in range(len(matrix)):
        # todo: broadcast implementation

            for i in range(start_row, end_row):
                for j in range(0, len(matrix)):
                    if(matrix[i][k] != 100000000 and matrix[k][j] != 100000000):
                        matrix[i][j] = min(matrix[i][j], matrix[i][k] + matrix[k][j])


def main():
    # adj_matrix = np.zeros((4039, 4039), dtype=int)   # facebook dataset has 4039 nodes

    # matrix for testing
    INF = 100000000
    dist = [
        [0, 4, INF, 5, INF],
        [INF, 0, 1, INF, 6],
        [2, INF, 0, 3, INF],
        [INF, INF, 1, 0, 2],
        [1, INF, INF, 4, 0]
    ]

    fw_parallel(0, 5, dist)

    # read in facebook dataset

    # with open("facebook_combined.txt", "r") as file:
    #     for line in file:
    #         idx = tuple(line.split())
    #         adj_matrix[int(idx[0])][int(idx[1])] = 1
    #         adj_matrix[int(idx[1])][int(idx[0])] = 1

    # for i in range(len(adj_matrix)):
    #     for j in range(len(adj_matrix[i])):
    #         if adj_matrix[i][j] == 0 and i != j:
    #             adj_matrix[i][j] = 10000000

    # with open("output.txt", "w") as output:
    #     print(adj_matrix, file = output)

    print("done")
    print(dist)

main()


   

