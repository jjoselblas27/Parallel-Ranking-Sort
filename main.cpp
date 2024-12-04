#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <string>

void generate_random_vector(std::vector<int>& vec, int M) {
    for (int i = 0; i < M; i++) {
        vec.push_back(i);
    }

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(vec.begin(), vec.end(), g);
}

void print_vector_pretty(const std::vector<int>& vec) {
    std::cout << "<";
    for (size_t i = 0; i < vec.size(); i++) {
        if (i == vec.size() - 1) {
            std::cout << vec[i];
        } else {
            std::cout << vec[i] << ", ";
        }
    }
    std::cout << ">" << std::endl;
}

void print_vector_pretty(const std::vector<char>& vec) {
    std::cout << "<";
    for (size_t i = 0; i < vec.size(); i++) {
        if (i == vec.size() - 1) {
            std::cout << vec[i];
        } else {
            std::cout << vec[i] << ", ";
        }
    }
    std::cout << ">" << std::endl;
}

void validate_mpi_properties(int rank, int size, int M) {
    if (rank == 0) {
        int sqrt_size = static_cast<int>(std::sqrt(size));
        if (sqrt_size * sqrt_size != size) {
            std::cerr << "Error: The number of processes must be a perfect square!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (M % sqrt_size != 0) {
            std::cerr << "Error: The number of elements must be divisible by the square root of the number of processes!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

std::vector<int> gossip_phase(const std::vector<int>& local_data, int local_size, int rank, int grid_size) {
    std::vector<int> column_data(local_size * grid_size);
    int column = rank % grid_size;
    MPI_Comm column_comm;
    MPI_Comm_split(MPI_COMM_WORLD, column, rank, &column_comm);

    MPI_Allgather(local_data.data(), local_size, MPI_INT, 
                  column_data.data(), local_size, MPI_INT, 
                  column_comm);

    MPI_Comm_free(&column_comm);
    return column_data;
}

std::vector<int> broadcast_phase(const std::vector<int>& local_data, int local_size, int rank, int grid_size) {
    int row = rank / grid_size;
    int diag_process = row * grid_size + row;

    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, rank, &row_comm);

    std::vector<int> diagonal_data(local_size * grid_size);
    if (rank == diag_process) {
        diagonal_data = local_data;
    }

    MPI_Bcast(diagonal_data.data(), local_size * grid_size, MPI_INT, row, row_comm);

    MPI_Comm_free(&row_comm);
    return diagonal_data;
}

std::vector<int> sort_phase(const std::vector<int>& column_data, int local_size, int rank, int grid_size) {
    std::vector<int> sorted_data(column_data);
    std::sort(sorted_data.begin(), sorted_data.end());
    return sorted_data;
}
/*
std::vector<int> local_rank_phase(const std::vector<int>& diagonal_data, const std::vector<int>& local_data, int local_size, int grid_size) {
    std::vector<int> local_ranks(local_size * grid_size);
    for (int i = 0; i < local_size * grid_size; i++) {
        for (int j = 0; j < local_size * grid_size; j++) {
            if (diagonal_data[i] >= local_data[j]) {
                local_ranks[i]++;
            }
        }
    }

    return local_ranks;
}
*/
/*
std::vector<int> local_rank_phase(const std::vector<int>& diagonal_data, const std::vector<int>& local_data, int local_size, int grid_size) {
    std::vector<int> local_ranks(local_size * grid_size, 0);
    int i = 0;  // Índice para recorrer local_data

    // Recorrer los elementos de diagonal_data
    for (int j = 0; j < local_size * grid_size; ++j) {
        // Mover el índice 'i' en local_data hasta encontrar los elementos menores o iguales a diagonal_data[j]
        while (i < local_size * grid_size && local_data[i] <= diagonal_data[j]) {
            i++;
        }
        // El índice 'i' en este punto nos dice cuántos elementos en local_data son menores o iguales a diagonal_data[j]
        local_ranks[j] = i;
    }

    return local_ranks;
}
*/


std::vector<int> local_rank_phase(const std::vector<int>& diagonal_data, const std::vector<int>& local_data, int local_size, int grid_size) {
    std::vector<int> local_ranks(local_size * grid_size, 0);

    // Recorrer cada elemento de diagonal_data
    for (int j = 0; j < local_size * grid_size; ++j) {
        // Usamos lower_bound para encontrar la posición donde el valor diagonal_data[j] debería insertarse
        // en local_data, manteniendo el orden
        auto it = std::lower_bound(local_data.begin(), local_data.end(), diagonal_data[j]);
        
        // El número de elementos menores o iguales a diagonal_data[j] es la posición de 'it'
        // que indica cuántos elementos en local_data son <= diagonal_data[j]
        local_ranks[j] = std::distance(local_data.begin(), it);
    }

    return local_ranks;
}

std::vector<int> reduce_phase(const std::vector<int>& local_rank, int local_size, int rank, int grid_size) {
    std::vector<int> global_ranks(local_size * grid_size, 0);
    int row = rank / grid_size;

    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, rank, &row_comm);

    MPI_Reduce(local_rank.data(), global_ranks.data(), local_size * grid_size, MPI_INT, MPI_SUM, row, row_comm);

    MPI_Comm_free(&row_comm);
    return global_ranks;
}

std::vector<int> gather_global_ranks(const std::vector<int>& global_rank, int local_size, int rank, int grid_size) {
    std::vector<int> all_global_ranks(local_size* grid_size * grid_size * grid_size, 0);

    MPI_Gather(global_rank.data(), local_size * grid_size, MPI_INT,
               all_global_ranks.data(), local_size * grid_size, MPI_INT,
               0, MPI_COMM_WORLD);

    return all_global_ranks;
}

std::vector<int> gather_diagonal_data(const std::vector<int>& diagonal_data, int local_size, int rank, int grid_size) {
    std::vector<int> all_diagonal_data(local_size * grid_size * grid_size * grid_size, 0);

    MPI_Gather(diagonal_data.data(), local_size * grid_size, MPI_INT,
               all_diagonal_data.data(), local_size * grid_size, MPI_INT,
               0, MPI_COMM_WORLD);

    return all_diagonal_data;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> data;
    int M = 160000000;
    
    generate_random_vector(data, M);

    /* validar las propiedades */
    validate_mpi_properties(rank, size, M);

    double start_time = MPI_Wtime();

    // scatter data (1)
    int grid_size = static_cast<int>(std::sqrt(size));
    int local_size = M / size;
    std::vector<int> local_data(local_size);

    MPI_Scatter(data.data(), local_size, MPI_INT, 
                local_data.data(), local_size, MPI_INT, 
                0, MPI_COMM_WORLD);

    // gossip phase (2)
    local_data = gossip_phase(local_data, local_size, rank, grid_size);
    MPI_Barrier(MPI_COMM_WORLD);

    // broadcast phase (3)
    std::vector<int> diagonal_data = broadcast_phase(local_data, local_size, rank, grid_size);
    MPI_Barrier(MPI_COMM_WORLD);

    // sort phase (4)
    local_data = sort_phase(local_data, local_size, rank, grid_size);
    MPI_Barrier(MPI_COMM_WORLD);

    // local rank phase (5)
    std::vector<int> local_rank = local_rank_phase(diagonal_data, local_data, local_size, grid_size);
    MPI_Barrier(MPI_COMM_WORLD);

    // Fase de reducción (6): sumar los local_ranks por filas
    std::vector<int> global_rank = reduce_phase(local_rank, local_size, rank, grid_size);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // Recolectar todos los global_ranks en el proceso 0
    std::vector<int> all_global_ranks = gather_global_ranks(global_rank, local_size, rank, grid_size);
    std::vector<int> all_diagonal_data = gather_diagonal_data(diagonal_data, local_size, rank, grid_size);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<int> ordered_data(M);

        for (int i = 0; i < all_global_ranks.size(); i++) {
            int global_rank = all_global_ranks[i];
            if (global_rank != 0) {
                ordered_data[global_rank] = all_diagonal_data[i];
            }
        }

        double end_time = MPI_Wtime();
        if (rank == 0) {
            std::cout << "Total time: " << (end_time - start_time) << " seconds." << std::endl;
        }

        //print_vector_pretty(ordered_data);
    }

    MPI_Finalize();
    return 0;
}
