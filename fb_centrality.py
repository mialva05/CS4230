from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def fw_parallel(start_row, end_row, matrix):
    n = len(matrix)

    for k in range(n):
        # todo: broadcast implementation
        owner = k * size // n

        if rank == owner:
            row_k = matrix[k].copy()
        else: 
            row_k = np.empty(n, dtype=int)

        comm.Bcast(row_k, root=owner)

        for i in range(start_row, end_row):
            for j in range(n):
                if matrix[i][j] > matrix[i][k] + row_k[j]:
                    matrix[i][j] = matrix[i][k] + row_k[j]

    return matrix[start_row:end_row]

def get_closeness_centrality(matrix):
    cc_list = []
    for row in matrix:
        cc = 1 / sum(row)
        cc_list.append(cc)
    return cc_list

def main():
    adj_matrix = np.zeros((4039, 4039), dtype=int)   # facebook dataset has 4039 nodes

    # matrix for testing
    # INF = 100000000
    # dist = [
    #     [0, 4, INF, 5, INF],
    #     [INF, 0, 1, INF, 6],
    #     [2, INF, 0, 3, INF],
    #     [INF, INF, 1, 0, 2],
    #     [1, INF, INF, 4, 0]
    # ]
    # dist = np.array(dist, dtype=int)
    # n = len(dist)

    # read in facebook dataset

    with open("facebook_combined.txt", "r") as file:
        for line in file:
            idx = tuple(line.split())
            adj_matrix[int(idx[0])][int(idx[1])] = 1
            adj_matrix[int(idx[1])][int(idx[0])] = 1

    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] == 0 and i != j:
                adj_matrix[i][j] = 10000000

    n = len(adj_matrix)

    #divide the work among processes
    rows_per_proc = n // size
    extra_rows = n % size

    if rank < extra_rows:
        start_row = rank * (rows_per_proc + 1)
        end_row = start_row + rows_per_proc + 1
    else:
        start_row = rank * rows_per_proc + extra_rows
        end_row = start_row + rows_per_proc

    end_row = min(end_row, n)  

    start_time = time.perf_counter()
    fw_parallel(start_row, end_row, adj_matrix)

    comm.Barrier()  # Synchronize all processes

    updated_chunk = fw_parallel(start_row, end_row, adj_matrix)

    all_chunks = comm.gather(updated_chunk, root=0)
    end_time = time.perf_counter()

    if rank == 0:
        adj_matrix = np.vstack(all_chunks)
        cc = get_closeness_centrality(adj_matrix)
        with open("output.txt", "w") as output:
            # print closeness centrality of each node
            for i in range(0, len(cc)):
                print(f"{cc[i]:.7f}", end = "", file = output)
                if (i+1) % 10 == 0 and i > 0:
                    print(file = output)
                else:
                    print(", ", end = "", file = output)
            print(file = output)
            # get top 5 nodes in terms of centrality value
            print("Nodes with top 5 centrality values: ", file = output)
            for i in range(0, 5):
                max_index = cc.index(max(cc))
                print(max_index, file = output)
                cc[max_index] = 0
            average = sum(cc) / len(cc)
            print(f"Average closeness centrality: {average:.4f}", file = output)
            print()
        print("done")
        print(f"Execution time: {end_time - start_time}")

main()


   

