#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#define VEC_WIDTH 8

__m256 compute_error(__m256 guess, __m256 x){
    // error = fabs(guess * guess * x - 1.f)
    __m256 const_ones = _mm256_set1_ps(1.f);
    __m256 const_zeros = _mm256_set1_ps(0);

    __m256 guess_squared_times_x = _mm256_mul_ps(_mm256_mul_ps (guess, guess), x);

    __m256 diff = _mm256_sub_ps(guess_squared_times_x, const_ones);

    __m256 gt_zero = _mm256_cmp_ps(diff, const_zeros, _CMP_GT_OQ);

    // Zero out negative elements by anding with the comp_zero mask
    __m256 pos = _mm256_and_ps(gt_zero, diff);
    __m256 neg = _mm256_andnot_ps(gt_zero, diff);

    __m256 err = _mm256_sub_ps(pos, neg);
    return err;
}

__m256 update_guess(__m256 guess, __m256 x){
    // (3.f * guess - x * guess * guess * guess) * 0.5f
    __m256 const_threes = _mm256_set1_ps(3.f);
    __m256 const_half = _mm256_set1_ps(.5f);

    __m256 guess_squared_times_x = _mm256_mul_ps(_mm256_mul_ps (guess, guess), x);
    __m256 diff = _mm256_sub_ps(const_threes, guess_squared_times_x);

    __m256 half_guess = _mm256_mul_ps(const_half, guess);
    
    __m256 new_guess = _mm256_mul_ps(half_guess, diff);
    return new_guess;
}

void sqrtVec(int N,
                float initialGuess,
                float values[],
                float output[])
{

    __m256 kThreshold = _mm256_set1_ps(0.00001f);
    __m256 init_guess_vec = _mm256_set1_ps(initialGuess);

    for (int i=0; i<N; i+=VEC_WIDTH) {

        __m256 x = _mm256_loadu_ps(values + i);
        __m256 guess = init_guess_vec;

        __m256 error = compute_error(guess, x);

        __m256 cmp_err_thresh = _mm256_cmp_ps(error, kThreshold, _CMP_GT_OQ); 

        int first_zeros = _mm256_movemask_ps(cmp_err_thresh); 
        while (first_zeros > 0) {
            // printf("Sum errors: %d \n", first_zeros); 
            guess = update_guess(guess, x);
            error = compute_error(guess, x);

            cmp_err_thresh = _mm256_cmp_ps(error, kThreshold, _CMP_GT_OQ); 
            first_zeros = _mm256_movemask_ps(cmp_err_thresh); 
        }
        
        _mm256_storeu_ps(output+i, _mm256_mul_ps(x, guess));
    }
}

