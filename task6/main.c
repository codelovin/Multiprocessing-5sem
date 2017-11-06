//
//  main.c
//  mpi-5
//
//  Created by Vlad on 04/11/2017.
//  Copyright © 2017 Codelovin. All rights reserved.
//

#include <stdio.h>
#include <mpi.h>
#include <pthread.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <string.h>
#include <time.h>
#include <omp.h>

int reallocation_factor = 2;
int messaging_interval = 10;

omp_lock_t lock;

typedef int bool;
#define FALSE 0
#define TRUE 1

typedef enum {
    DEFAULT,
    LEFT,
    RIGHT,
    UP,
    DOWN
} dir_t;

typedef struct {
    int l;
    int a;
    int b;
    int n;
    int N;
    float pl;
    float pr;
    float pu;
    float pd;
    int rank;
    int size;
} ctx_t;

typedef struct {
    int x;
    int y;
} coords_t;

typedef struct {
    coords_t coords;
    int n;
    int seed;
    int initial_rank;
} particle_t;

coords_t get_coordinates(int rank, int a, int b) {
    return (coords_t) { rank % a, rank / a};
}

int get_rank(coords_t coords, int a, int b) {
    return coords.y * a + coords.x;
}

int get_adjacent_rank(coords_t coords, int a, int b, dir_t dir) {
    if (dir == LEFT) {
        coords.x -= 1;
        if (coords.x < 0) {
            coords.x = a - 1;
        }
    }
    if (dir == RIGHT) {
        coords.x += 1;
        if (coords.x >= a) {
            coords.x = 0;
        }
    }
    if (dir == UP) {
        coords.y -= 1;
        if (coords.y < 0) {
            coords.y = b - 1;
        }
    }
    if (dir == DOWN) {
        coords.y += 1;
        if (coords.y >= b) {
            coords.y = 0;
        }
    }
    return get_rank(coords, a, b);
}

void push_back(particle_t** array, int* n, int* capacity, particle_t* element) {
    if (*n < *capacity) {
        (*array)[*n] = *element;
        (*n)++;
    } else {
        *array = realloc(*array, *capacity * reallocation_factor * sizeof(particle_t));
        *capacity *= reallocation_factor;
        (*array)[*n] = *element;
        (*n)++;
    }
}

void pop(particle_t** array, int* n, int index) {
    (*array)[index] = (*array)[(*n) - 1];
    (*n)--;
}

dir_t decide(float dir_pr_1, float dir_pr_2, float dir_pr_3, float dir_pr_4) {
    if (dir_pr_1 >= dir_pr_2 && dir_pr_1 >= dir_pr_3 && dir_pr_1 >= dir_pr_4) {
        return LEFT;
    } else if (dir_pr_2 >= dir_pr_1 && dir_pr_2 >= dir_pr_3 && dir_pr_2 >= dir_pr_4) {
        return RIGHT;
    } else if (dir_pr_3 >= dir_pr_1 && dir_pr_3 >= dir_pr_2 && dir_pr_3 >= dir_pr_4) {
        return UP;
    } else {
        return DOWN;
    }
}

void* process_region(void* ctx_void) {
    
    ctx_t *ctx = (ctx_t*) ctx_void;
    int l = ctx->l;
    int a = ctx->a;
    int b = ctx->b;
    int n = ctx->n;
    int N = ctx->N;
    int rank = ctx->rank;
    int size = ctx->size;
    float pl = ctx->pl;
    float pr = ctx->pr;
    float pu = ctx->pu;
    float pd = ctx->pd;
    coords_t coords = get_coordinates(rank, a, b);
    
    int left_rank = get_adjacent_rank(coords, a, b, LEFT);
    int right_rank = get_adjacent_rank(coords, a, b, RIGHT);
    int up_rank = get_adjacent_rank(coords, a, b, UP);
    int down_rank = get_adjacent_rank(coords, a, b, DOWN);
    
    int particles_size = N;
    int particles_capacity = N;
    particle_t* particles = (particle_t*) malloc(particles_capacity * sizeof(particle_t));
    
    int send_left_size = 0;
    int send_left_capacity = N;
    particle_t* send_left = (particle_t*) malloc(send_left_capacity * sizeof(particle_t));
    
    int send_right_size = 0;
    int send_right_capacity = N;
    particle_t* send_right = (particle_t*) malloc(send_right_capacity * sizeof(particle_t));
    
    int send_up_size = 0;
    int send_up_capacity = N;
    particle_t* send_up = (particle_t*) malloc(send_up_capacity * sizeof(particle_t));
    
    int send_down_size = 0;
    int send_down_capacity = N;
    particle_t* send_down = (particle_t*) malloc(send_down_capacity * sizeof(particle_t));
    
    int receive_left_capacity = 0;
    int receive_right_capacity = 0;
    int receive_up_capacity = 0;
    int receive_down_capacity = 0;
    
    int completed_size = 0;
    int completed_capacity = N;
    particle_t* completed = (particle_t*) malloc(completed_capacity * sizeof(particle_t));
    int* total = (int*) malloc(sizeof(int) * size);
    
    struct timeval t0, t1;
    assert(gettimeofday(&t0, NULL) == 0);
    
    bool is_done = FALSE;
    
    omp_set_num_threads(2);
    omp_init_lock(&lock);
    omp_set_lock(&lock);
    
#pragma omp parallel
{
#pragma omp single nowait
    {
#pragma omp task
    {
    // The master process sends the seeds to the other processes
    int* seeds = malloc(sizeof(int) * size);
    int seed;
    if (rank == 0) {
        srand(time(NULL));
        for (int i = 0; i < size; i++)
            seeds[i] = (int) rand();
    }
    MPI_Scatter(seeds, 1, MPI_INT, &seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    srand(seed);
    for (int i = 0; i < N; i++) {
        particles[i].coords = (coords_t) {rand() % l, rand() % l};
        particles[i].n = n;
        particles[i].seed = rand();
        particles[i].initial_rank = rank;
    }
    
    free(seeds);
    omp_unset_lock(&lock);
        
    while (!is_done) {
//        if (rank == 0)
//            printf(" Starting exchange\n");
        // Sending and receiving basic info
        
        omp_set_lock(&lock);
        
        MPI_Request* metadata = (MPI_Request*) malloc(sizeof(MPI_Request) * 8);
        MPI_Isend(&send_left_size, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, metadata+0);
        MPI_Isend(&send_right_size, 1, MPI_INT, right_rank, 1, MPI_COMM_WORLD, metadata+1);
        MPI_Isend(&send_up_size, 1, MPI_INT, up_rank, 2, MPI_COMM_WORLD, metadata+2);
        MPI_Isend(&send_down_size, 1, MPI_INT, down_rank, 3, MPI_COMM_WORLD, metadata+3);
        
        MPI_Irecv(&receive_left_capacity, 1, MPI_INT, left_rank, 1, MPI_COMM_WORLD, metadata+4);
        MPI_Irecv(&receive_right_capacity, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, metadata+5);
        MPI_Irecv(&receive_up_capacity, 1, MPI_INT, up_rank, 3, MPI_COMM_WORLD ,metadata+6);
        MPI_Irecv(&receive_down_capacity, 1, MPI_INT, down_rank, 2, MPI_COMM_WORLD, metadata+7);
        
        // Waiting till all the processes reach this line
        MPI_Waitall(8, metadata, MPI_STATUS_IGNORE);
        
        particle_t* receive_left = (particle_t*) malloc(receive_left_capacity * sizeof(particle_t));
        particle_t* receive_right = (particle_t*) malloc(receive_right_capacity * sizeof(particle_t));
        particle_t* receive_up = (particle_t*) malloc(receive_up_capacity * sizeof(particle_t));
        particle_t* receive_down = (particle_t*) malloc(receive_down_capacity * sizeof(particle_t));
        
        MPI_Request* data = (MPI_Request*) malloc(sizeof(MPI_Request) * 8);
        MPI_Issend(send_left, sizeof(particle_t) * send_left_size, MPI_BYTE, left_rank, 0, MPI_COMM_WORLD, data+0);
        MPI_Issend(send_right, sizeof(particle_t) * send_right_size, MPI_BYTE, right_rank, 1, MPI_COMM_WORLD, data+1);
        MPI_Issend(send_up, sizeof(particle_t) * send_up_size, MPI_BYTE, up_rank, 2, MPI_COMM_WORLD, data+2);
        MPI_Issend(send_down, sizeof(particle_t) * send_down_size, MPI_BYTE, down_rank, 3, MPI_COMM_WORLD, data+3);
        
        MPI_Irecv(receive_left, sizeof(particle_t) * receive_left_capacity, MPI_BYTE, left_rank, 1, MPI_COMM_WORLD, data+4);
        MPI_Irecv(receive_right, sizeof(particle_t) * receive_right_capacity, MPI_BYTE, right_rank, 0, MPI_COMM_WORLD, data+5);
        MPI_Irecv(receive_up, sizeof(particle_t) * receive_up_capacity, MPI_BYTE, up_rank, 3, MPI_COMM_WORLD, data+6);
        MPI_Irecv(receive_down, sizeof(particle_t) * receive_down_capacity, MPI_BYTE, down_rank, 2, MPI_COMM_WORLD, data+7);
        
        // Waiting till all the processes reach this line
        MPI_Waitall(8, data, MPI_STATUS_IGNORE);
        
        free(metadata);
        free(data);
        
        // Add received particles to current particles
        for (int j = 0; j < receive_left_capacity; j++) {
            push_back(&particles, &particles_size, &particles_capacity, receive_left + j);
        }
        for (int j = 0; j < receive_right_capacity; j++) {
            push_back(&particles, &particles_size, &particles_capacity, receive_right + j);
        }
        for (int j = 0; j < receive_up_capacity; j++) {
            push_back(&particles, &particles_size, &particles_capacity, receive_up + j);
        }
        for (int j = 0; j < receive_down_capacity; j++) {
            push_back(&particles, &particles_size, &particles_capacity, receive_down + j);
        }
        
        free(receive_left);
        free(receive_right);
        free(receive_up);
        free(receive_down);
        
        // All previous particles have been sent
        send_left_size = 0;
        send_right_size = 0;
        send_up_size = 0;
        send_down_size = 0;
        
        // Sending info to the master process
        int* recv_buff = (int*) malloc(sizeof(int));
        //        MPI_Gather(&completed_size, 1, MPI_INT, recv_buff, size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Reduce(&completed_size, recv_buff, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        // Synchronizing
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (*recv_buff == size * N) {
                is_done = TRUE;
            }
        }
        MPI_Bcast(&is_done, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        free(recv_buff);
        
        if (is_done) {
            MPI_Gather(&completed_size, 1, MPI_INT, total, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
        omp_unset_lock(&lock);
        
    }
        
        MPI_Barrier(MPI_COMM_WORLD);
//        printf("end");
        
        if (rank == 0) {
            assert(gettimeofday(&t1, NULL) == 0);
            double delta = ((t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec) / 1000000.0;

            FILE *file;
            file = fopen("stats.txt", "w");
            fprintf(file, "%d %d %d %d %d %f %f %f %f %fs\n", l, a, b, n, N, pl, pr, pu, pd, delta);
            for (int rk = 0; rk < size; rk++) {
                fprintf(file, "%d: %d\n", rk, total[rk]);
            }
            fclose(file);
        }

        // Writing to file data.bin
        MPI_File f;
        MPI_File_delete("data.bin", MPI_INFO_NULL);
        MPI_File_open(MPI_COMM_WORLD, "data.bin", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f);

        int oy_size = l;
        int ox_size = l * size;

        int** result = calloc(oy_size, sizeof(int*));
        for(int init_idx = 0; init_idx < oy_size; init_idx++) {
            result[init_idx] = calloc(ox_size, sizeof(int));
        }

        for (int pt_idx = 0; pt_idx < completed_size; pt_idx++) {
            int y = completed[pt_idx].coords.y;
            int x = completed[pt_idx].coords.x;
            int initial_rank = completed[pt_idx].initial_rank;
            result[y][x*size+initial_rank]++;
        }

        int a_global = rank % a;
        int b_global = rank / a;

        // Iterating over each point in a square
        for (int y_c = 0; y_c < oy_size; y_c++) {
            for (int x_c = 0; x_c < l; x_c++) {
                int num_bytes_by_line = l * size * a * sizeof(int);
                int bytes_till_square_line = num_bytes_by_line * b_global * l;
                int bytes_before_y_c = bytes_till_square_line + num_bytes_by_line * y_c;
                int bytes_per_first_line_elem = bytes_before_y_c + a_global * l * size * sizeof(int);
                int bytes_before_x_c = bytes_per_first_line_elem + x_c * size * sizeof(int);
                MPI_File_set_view(f, bytes_before_x_c, MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
                MPI_File_write(f, &result[y_c][x_c], size, MPI_INT, MPI_STATUS_IGNORE);
            }
        }

        MPI_File_close(&f);

        for (int p = 0; p < oy_size; p++) {
            free(result[p]);
        }
        free(result);
    }
    
#pragma omp task
    {
    while (!is_done) {
        omp_set_lock(&lock);
        int particle_index = 0;
        while (particle_index < particles_size) {
            particle_t* particle = particles + particle_index;
            bool inc_part_idx = TRUE;
            for (int t = 0; t < messaging_interval; t++) {
                if (particle->n == 0) {
                    push_back(&completed, &completed_size, &completed_capacity, particle);
                    pop(&particles, &particles_size, particle_index);
                    inc_part_idx = FALSE;
                    break;
                }
                float left_pr = rand_r((unsigned int*) &particle->seed) * pl;
                float right_pr = rand_r((unsigned int*) &particle->seed) * pr;
                float up_pr = rand_r((unsigned int*) &particle->seed) * pu;
                float down_pr = rand_r((unsigned int*) &particle->seed) * pd;
                dir_t dir = decide(left_pr, right_pr, up_pr, down_pr);
                if (dir == LEFT) {
                    particle->coords.x -= 1;
                } else if (dir == RIGHT) {
                    particle->coords.x += 1;
                } else if (dir == UP) {
                    particle->coords.y -= 1;
                } else {
                    particle->coords.y += 1;
                }
                particle->n -= 1;
                if (particle->coords.x < 0) {
                    particle->coords.x = l - 1;
                    push_back(&send_left, &send_left_size, &send_left_capacity, particle);
                    pop(&particles, &particles_size, particle_index);
                    inc_part_idx = FALSE;
                    break;
                }
                if (particle->coords.x >= l) {
                    particle->coords.x = 0;
                    push_back(&send_right, &send_right_size, &send_right_capacity, particle);
                    pop(&particles, &particles_size, particle_index);
                    inc_part_idx = FALSE;
                    break;
                }
                if (particle->coords.y < 0) {
                    particle->coords.y = l - 1;
                    push_back(&send_up, &send_up_size, &send_up_capacity, particle);
                    pop(&particles, &particles_size, particle_index);
                    inc_part_idx = FALSE;
                    break;
                }
                if (particle->coords.y >= l) {
                    particle->coords.y = 0;
                    push_back(&send_down, &send_down_size, &send_down_capacity, particle);
                    pop(&particles, &particles_size, particle_index);
                    inc_part_idx = FALSE;
                    break;
                }
            }
            if (inc_part_idx) {
                particle_index += 1;
            }
        }
        omp_unset_lock(&lock);
    }
    }
}
}
#pragma omp taskwait
    omp_destroy_lock(&lock);
    
    free(total);
    
    free(send_left);
    free(send_right);
    free(send_up);
    free(send_down);
    
    free(particles);
    free(completed);
    
    return NULL;
}

int main(int argc, char * argv[]) {
    
    MPI_Init(&argc, &argv);
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    ctx_t ctx = (ctx_t) {
        atoi(argv[1]),
        atoi(argv[2]),
        atoi(argv[3]),
        atoi(argv[4]),
        atoi(argv[5]),
        atof(argv[6]),
        atof(argv[7]),
        atof(argv[8]),
        atof(argv[9]),
        world_rank,
        world_size
    };
    
    // Setting a separate thread is required by the task
    process_region(&ctx);
    
    MPI_Finalize();
    return 0;
}

