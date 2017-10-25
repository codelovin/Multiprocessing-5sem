#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef int bool;
static int count = 0;

pthread_mutex_t mutex;

void* mergesort_parallel_helper(void *args_);
void* merge_parallel_helper(void *args_);

typedef struct {
    int *A; // Denotes SOURCE
    int *B; // Denotes destination

    // Denote borders
    int x1;
    int x2;
    int x3;
    int x4;

    int x5; // Denotes left destination border
    int m;
} args_t;


void* get_args_t(int *A, int *B, int x1, int x2, int x3, int x4, int x5, int m) {
    args_t* args = (args_t *) malloc(sizeof(args_t));
    args->A = A;
    args->B = B;
    args->x1 = x1;
    args->x2 = x2;
    args->x3 = x3;
    args->x4 = x4;
    args->x5 = x5;
    args->m = m;
    return (void*) args;
}

// Binary Search
int bs(int x, int *arr, int low, int high) {
    assert(low <= high);
    if (low == high) {
        return arr[low] < x ? low + 1 : low;
    }
    int mid = (low + high) / 2;
    if (arr[mid] == x)
        return mid;
    else if (arr[mid] < x)
        return bs(x, arr, mid + 1, high);
    else
        return bs(x, arr, low, mid);
}

// Comparator for built-in qsort
int cmp(const void *a, const void *b) {
    int a_val = *(int*)(a);
    int b_val = *(int*)(b);
    // Cannot use subtraction here for the reason of overflowing
    return a_val == b_val ? 0 : a_val < b_val ? -1 : 1;
}

// Sequentially merges two subarrays (one thread)
void merge_sequential(int *A, int *CP, int left1, int right1, int left2, int right2, int leftCP) {
    while (left1 <= right1 && left2 <= right2) {
        if (A[left1] <= A[left2]) {
            CP[leftCP++] = A[left1++];
        } else {
            CP[leftCP++] = A[left2++];
        }
    }
    while (left1 <= right1) {
        CP[leftCP++] = A[left1++];
    }
    while (left2 <= right2) {
        CP[leftCP++] = A[left2++];
    }
}

// Merging two subarrays in parallel
void merge_parallel(int *A, int *CP, int leftA1, int rightA1, int leftA2,
                    int rightA2, int leftCP, int m) {
    int n1 = rightA1 - leftA1 + 1;
    int n2 = rightA2 - leftA2 + 1;
    if (n1 < n2) {
        merge_parallel(A, CP, leftA2, rightA2, leftA1, rightA1, leftCP, m);
        return;
    }
    if (n2 == 0) {
        memcpy(CP + leftCP, A + leftA1, n1 * sizeof(int));
        return;
    }
    if (n1 <= m) {
        merge_sequential(A, CP, leftA1, rightA1, leftA2, rightA2, leftCP);
        return;
    }
    int midA1 = (leftA1 + rightA1) / 2;
    int element = A[midA1];
    int midA2 = bs(A[midA1], A, leftA2, rightA2);
    int mid_insert = leftCP + (midA1 - leftA1) + (midA2 - leftA2);
    CP[mid_insert] = A[midA1];

    /*
     * Idea:
     * For each half, attempt to create a thread and assign the task to this thread
     * In case of failure (count = 0), complete the task using the current thread
     */
    pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t) * 2);
    bool created[2];
    created[0] = 0;
    created[1] = 0;
    pthread_mutex_lock(&mutex);
    if (count > 0){
        count--;
        pthread_mutex_unlock(&mutex);
        created[0]++;
        pthread_create(threads, NULL, merge_parallel_helper, get_args_t(A, CP, leftA1, midA1 - 1, leftA2, midA2 - 1, leftCP, m));
    } else {
        pthread_mutex_unlock(&mutex);
        merge_parallel(A, CP, leftA1, midA1 - 1, leftA2, midA2 - 1, leftCP, m);
    }

    pthread_mutex_lock(&mutex);
    if (count > 0){
        count--;
        pthread_mutex_unlock(&mutex);
        created[1]++;
        pthread_create(threads + 1, NULL, merge_parallel_helper, get_args_t(A, CP, midA1 + 1, rightA1, midA2,rightA2, mid_insert + 1, m));
    } else {
        pthread_mutex_unlock(&mutex);
        merge_parallel(A, CP, midA1 + 1, rightA1, midA2, rightA2, mid_insert + 1, m);
    }
    for (int i = 0; i < 2; i++) {
        if (created[i]) {
            pthread_join(threads[i], NULL);
        }
    }
    free(threads);
}

void* merge_parallel_helper(void *args_) {
    args_t *args = (args_t *) args_;
    merge_parallel(args->A, args->B, args->x1, args->x2, args->x3, args->x4, args->x5, args->m);
    return NULL;
}

// Main sorting function
void mergesort_parallel(int *A, int *CP, int lA, int hA, int lCP, int m) {
    int len = hA - lA + 1;
    if (len <= m) {
        qsort(A + lA, len, sizeof(int), cmp);
        memcpy(CP + lCP, A + lA, len * sizeof(int));
        return;
    }
    int *T = (int*)calloc(len, sizeof(int));
    int mid = (lA + hA) / 2;
    int t_mid = mid - lA + 1;

    /*
     * Idea:
     * For each half, attempt to create a thread and assign the task to this thread
     * In case of failure (count = 0), complete the task using the current thread
     */
    pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t) * 2);
    bool created[2];
    created[0] = 0;
    created[1] = 0;
    pthread_mutex_lock(&mutex);
    if (count > 0){
        count--;
        pthread_mutex_unlock(&mutex);
        created[0] = 1;
        pthread_create(threads, NULL, mergesort_parallel_helper, get_args_t(A, T, lA, mid, 0, 0, 0, m));
    } else {
        pthread_mutex_unlock(&mutex);
        mergesort_parallel(A, T, lA, mid, 0, m);
    }

    pthread_mutex_lock(&mutex);
    if (count > 0){
        count--;
        pthread_mutex_unlock(&mutex);
        created[1] = 1;
        pthread_create(threads + 1, NULL, mergesort_parallel_helper, get_args_t(A, T, mid + 1, hA, 0, 0, t_mid, m));
    } else {
        pthread_mutex_unlock(&mutex);
        mergesort_parallel(A, T, mid + 1, hA, t_mid, m);
    }
    for (int i = 0; i < 2; i++) {
        if (created[i]) {
            pthread_join(threads[i], NULL);
        }
    }
    merge_parallel(T, CP, 0, t_mid - 1, t_mid, len - 1, lCP, m);
    free(T);
    free(threads);
}

void* mergesort_parallel_helper(void *args_) {
    args_t *args = (args_t *) args_;
    mergesort_parallel(args->A, args->B, args->x1, args->x2, args->x5, args->m);
    return NULL;
}

int main(int argc, char **argv) {

   int n = atoi(argv[1]);
   int m = atoi(argv[2]);
   int P = atoi(argv[3]);
    count = P - 1;
    FILE *file;
    // file = fopen("stats.txt", "a");
    file = fopen("qsort_stats.txt", "a");
    if (file == NULL) {
        printf("Could not create 'stats.txt'.\n");
        exit(1);
    }
    srand((int)time(NULL));
    int *A = (int*)calloc(n, sizeof(int));
    for(int i = 0; i < n; i++){
        A[i] = rand();
    }
    int*CP = (int*)calloc(n, sizeof(int));
    memcpy(CP, A, sizeof(int) * n);

    struct timeval t0, t1;
    assert(gettimeofday(&t0, NULL) == 0);
    mergesort_parallel_helper(get_args_t(A, CP, 0, n - 1, 0, 0, 0, m));
    // qsort(CP, n, sizeof(int), cmp);
    assert(gettimeofday(&t1, NULL) == 0);

    double delta = ((t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec) / 1000000.0;

    fprintf(file, "%fs %d %d %d\n", delta, n, m, P);
    fclose(file);

    file = fopen("data.txt", "w");
    if (file == NULL) {
        printf("Could not create 'data.txt'.\n");
        exit(1);
    }
    for (int i = 0; i < n; i++) {
      fprintf(file, "%d ", A[i]);
    }
    fprintf(file, "\n");
    for (int i = 0; i < n; i++) {
      fprintf(file, "%d ", CP[i]);
    }
    fclose(file);

    free(A);
    free(CP);
    return 0;
}
