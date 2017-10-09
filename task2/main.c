#include <stdio.h>
#include <omp.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>

#undef T1_TESTING

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
      merge_sequential(A, CP, leftA1, leftA2, rightA1, rightA2, leftCP);
      return;
    }
    int midA1 = (leftA1 + rightA1) / 2;
    int element = A[midA1];
    int midA2 = bs(A[midA1], A, leftA2, rightA2);
    int mid_insert = leftCP + (midA1 - leftA1) + (midA2 - leftA2);
    CP[mid_insert] = A[midA1];
#ifndef T1_TESTING
    #pragma omp parallel
    {
      #pragma omp single nowait
      {
        #pragma omp task
        {
          merge_parallel(A, CP, leftA1, midA1 - 1, leftA2, midA2 - 1, leftCP, m);
        }
        #pragma omp task
        {
          merge_parallel(A, CP, midA1 + 1, rightA1, midA2, rightA2, mid_insert + 1, m);
        }
      }
    }
#else
  merge_parallel(A, CP, leftA1, midA1 - 1, leftA2, midA2 - 1, leftCP, m);
  merge_parallel(A, CP, midA1 + 1, rightA1, midA2, rightA2, mid_insert + 1, m);
#endif
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
#ifndef T1_TESTING
    #pragma omp parallel
    {
      #pragma omp single nowait
      {
        #pragma omp task
        {
          mergesort_parallel(A, T, lA, mid, 0, m);
        }
        #pragma omp task
        {
          mergesort_parallel(A, T, mid + 1, hA, t_mid, m);
        }
      }
    }
#else
    mergesort_parallel(A, T, lA, mid, 0, m);
    mergesort_parallel(A, T, mid + 1, hA, t_mid, m);
#endif
    merge_parallel(T, CP, 0, t_mid - 1, t_mid, len - 1, lCP, m);
    free(T);
}

int main(int argc, char **argv) {

    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int P = atoi(argv[3]);

    omp_set_num_threads(P);

    FILE *file;
    file = fopen("stats.txt", "a");
    // file = fopen("qsort_stats.txt", "a");
    if (file == NULL) {
      printf("Could not create 'stats.txt'.\n");
      exit(1);
    }
    srand(time(NULL));
    int *A = (int*)calloc(n, sizeof(int));
    for(int i = 0;i<n;i++){
        A[i]= rand();
    }
    int*CP = (int*)calloc(n, sizeof(int));
    memcpy(CP, A, sizeof(int) * n);

    struct timeval t0, t1;
    assert(gettimeofday(&t0, NULL) == 0);
    mergesort_parallel(A, CP, 0, n - 1, 0, m);
    // qsort(CP, n, sizeof(int), cmp);
    assert(gettimeofday(&t1, NULL) == 0);

    double delta = ((t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec) / 1000000.0;

    fprintf(file, "%fs %d %d %d\n", delta, n, m, P);
    fclose(file);
    free(A);
    free(CP);
    return 0;
}
