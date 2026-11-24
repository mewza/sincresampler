/**
 ** sincresamp.h - 3 different HiFi SIMD vectorized sinc resamplers
 ** SURREAL® DSP 2.0
 ** (C) 2025 Dmitry Boldyrev, All rights reserved.
 */

#pragma once

// C specific includes
#include <assert.h>
#include <mss/const1.h>

// C++ specific includes
#include <thread>
#include <vector>
#include <atomic>
#include <future>
#include <mutex>
#include <bitset>
#include <array>
#include <cmath>
#include <cstdint>

// Choices: OriginalSincResampler SincResamplerBeta
#define SincResampler OriginalSincResampler

// OriginalSincResampler's SAMP_DOUBLE2FLOAT dithers with caout's SSRCDither (Apple builds, where mss/plugins/caout is on the path)
#if defined(__APPLE__) && __has_include("dither.h")
#include <memory>
#include "dither.h"
#define SINC_HAS_DITHER 1
#else
#define SINC_HAS_DITHER 0
#endif

// Enable NEON optimizations

#if USE_NEON
#define USE_LANE
#define SINC_USE_NEON
#endif

#include <functional>
#include <vector>
#include <memory>
#include <chrono>
#define RS_OPTIMIZE
#define RS_BLOCKSIZE 4

// KERNEL TABLE CACHE (user 2026-10-28 "go", "preserve sinc resampler how it was"): every IR (re)load made a fresh resampler = a fresh
// 512-tap × 10,000-phase table (41 MB, 10 M sin, 58 ms on the Mac) with settings that never change. OriginalSincResampler instances
// with the same order / precision / cutoff / β now share one read-only table — the bits a fresh build gives; the resampler itself is
// untouched. A table no instance holds is freed after SR_SINC_CACHE_IDLE s. 0 = every instance builds its own (the old way).
#ifndef SR_SINC_CACHE
#if defined(__APPLE__) || defined(__ANDROID__)
#define SR_SINC_CACHE 1
#else
#define SR_SINC_CACHE 0   // wasm: no thread for the reaper
#endif
#endif
#ifndef SR_SINC_CACHE_IDLE
#define SR_SINC_CACHE_IDLE 30.0
#endif

enum SAMP_MODE {
    SAMP_DOUBLE2FLOAT = 0,
    SAMP_FLOAT2DOUBLE,
    SAMP_UNCHANGED
};

#define OVERSHOOT_HEADROOM 0.891f

// SIMD helper functions
template<typename T>
constexpr bool is_simd_v = SimdSize<T> > 1;

template <typename T, WindowType wt = WINTYPE_KAISER>
class OriginalSincResampler_O {
public:   // the type aliases are part of the interface (a hook lambda names SincResampler<T>::TF, 2026-09-26)
    using TD = SimdSame<T,double>;
    using TF = SimdSame<T,float>;
    using T1 = SimdBase<T>;
    // DOUBLE→FLOAT HOOK (2026-09-26, user: "hack dither / sharp curve right into sinc resampler, it has the
    // DOUBLE_TO_FLOAT flag"): when set, SAMP_DOUBLE2FLOAT resamples at full T precision into a scratch buffer and
    // hands the reduction to the hook (out, in, count) — the mp3 plugin plugs its noise-shaped SSRCDither in here,
    // so the ONLY 64→32 rounding in the chain is dithered. Unset = the plain convert, as before.
    using D2FHook = std::function<void(TF *out, const T *in, int count)>;
    void setDoubleToFloatHook(D2FHook h) { mD2FHook = std::move(h); }
private:
    D2FHook        mD2FHook;
    std::vector<T> mD2FScratch;
    using T8 = Simd<SimdBase<T>,8>;
    using T4 = Simd<SimdBase<T>,4>;
    using T2 = Simd<SimdBase<T>,2>;
    using TX = typename std::conditional<SimdSize<T> == 2,T2,T1>::type;
    static constexpr float DEFAULT_CUTOFF = 0.46f;
    static const int DEFAULT_ORDER = 128;
    static const int DEFAULT_PRECISION = 10000;
protected:
    
    std::atomic<bool> initialized{false};
    
    typedef struct sinc_table {
        int order;
        int precision;
        float cutoff;
        T1 *kernel;
        int sz;
    } sinc_table;
    
    float rt, leftover;
    sinc_table tbl;
    T* p;
    int order = DEFAULT_ORDER;
    int precision = DEFAULT_PRECISION;
    T1* _window_table = nullptr;
    double _kaiser_beta  = 8.6f;
    double _stopband_att = 100.0f;
public:
    OriginalSincResampler_O() {
        p = NULL;
        tbl.kernel = NULL;
        rt = -1;
    }
    
    ~OriginalSincResampler_O() {
        cleanup();
    }
    void setFilterLength(int length, int precision = DEFAULT_PRECISION)  { order = length; }
   
    // Thread-safe predict function
    int estimateOutputLength(int n_input) const {
        if (rt <= 0 || n_input <= 0 || !initialized.load()) {
            return 0;
        }
        
        int ord = tbl.order;
        if (n_input < ord * 2) {
            return static_cast<int>(n_input / rt);
        }
        
        float x = leftover - ord;
        int available_samples = n_input - ord;

        return static_cast<int>(available_samples * (1.0f / rt) - x * (1.0f / rt));
    }
    
    void init(double in_rate, double out_rate, int nch) {
        initialized.store(false);
        
        rt = in_rate / out_rate;
        leftover = 0.0f;
        
        float c = 0.95f / std::max(1.0f, (float)rt);
        bool success = sr_create(order, precision, c, rt);
        
        if (success) {
            initialized.store(true);
        }
        reset();
    }
    
    void setKaiserBeta(double beta) {
        _kaiser_beta = (float)beta;
        // window will rebuild on next init() call
    }
    
    void setStopbandAttenuation(double att) {
        _stopband_att = (float)att;
        // could derive beta from att: beta ≈ 0.1102*(att-8.7) for att>50dB
        if (att > 50.0) {
            _kaiser_beta = (float)(0.1102 * (att - 8.7));
        } else if (att >= 21.0) {
            _kaiser_beta = (float)(0.5842 * pow(att - 21.0, 0.4) + 0.07886 * (att - 21.0));
        } else {
            _kaiser_beta = 0.0f;
        }
    }
    
    void precompute_window(int window_size) {
        if (_window_table) callocT_free(_window_table);
        _window_table = callocT<T1>(window_size);
        if (!_window_table) return;
        
        for (int n = 0; n < window_size; n++) {
            if constexpr (wt == WINTYPE_KAISER) {
                _window_table[n] = (T1)KaiserWindow::kaiser(n, window_size, _kaiser_beta);
            } else if constexpr (wt == WINTYPE_HANNING) {
                _window_table[n] = (T1)HanningWindow::hanning(n, window_size);
            } else if constexpr (wt == WINTYPE_SINE) {
                _window_table[n] = (T1)SineWindow::sine(n, window_size);
            } else if constexpr (wt == WINTYPE_BLACKMAN) {
                // inline since you don't have a class for it yet
                double x = (double)n / (window_size - 1);
                _window_table[n] = (T1)(0.3635819
                                        - 0.4891775 * F_COS(2.0 * M_PI * x)
                                        + 0.1365995 * F_COS(4.0 * M_PI * x)
                                        - 0.0106411 * F_COS(6.0 * M_PI * x));
            } else {
                // WINTYPE_NONE — rectangular
                _window_table[n] = (T1)1.0;
            }
        }
    }
    
    void reset() {
        leftover = 0.0f;
        if (p) {
            memset(p, 0, tbl.order * 2 * sizeof(T));
        }
    }
    
    void cleanup() {
        initialized.store(false);
        if (tbl.kernel) {
            callocT_free(tbl.kernel);
            tbl.kernel = nullptr;
        }
        if (p) {
            callocT_free(p);
            p = nullptr;
        }
    }
    
    inline T1 kernel_func(T1 x, T1 cutoff, int ord) {
        if (x <= -(T1)ord || x >= (T1)ord) return (T1)0.0;
        
        // Map x back to window table index
        // x ranges (-ord, +ord), table is [0, ord*2)
        int n = (int)(x + ord);  // shift to [0, ord*2)
        n = F_CLAMP(n, 0, ord * 2 - 1);
        
        return sinc(cutoff * x) * _window_table[n];
    }
    
    static inline T1 sinc(T1 x) {
        return (x == 0.0) ? 1.0 : F_SIN(M_PI * x) / (M_PI * x);
    }
    
    static inline T1 sine2(T1 t, int N) {
        T1 m = F_SIN(t * M_PI / (T1)(N - 1));
        return m * m;
    }
    
    static inline T1 hanning(T1 t, int N) {
        return 0.5 * (1.0 - F_COS(2.0 * M_PI * t / (T1)(N - 1)));
    }
    
    static inline T1 blackman_nuttall(T1 x, int N) {
        return 0.3635819 - 0.4891775 * F_COS((2.0 * M_PI * x) / (T1)(N - 1))
        + 0.1365995 * F_COS((4.0 * M_PI * x) / (T1)(N - 1))
        - 0.0106411 * F_COS((6.0 * M_PI * x) / (T1)(N - 1));
    }
    
    // Main multicore-capable process function
    int process(const T*__restrict in, int n_input, void *__restrict out, int &n_output, SAMP_MODE mode = SAMP_UNCHANGED, double scale = 1.0)
    {
        if (rt < 0 || !initialized.load() || !tbl.kernel || !p || !out) {
            n_output = 0;
            return -10;
        }
        const int ord2 = tbl.order * 2;
        if (n_input < ord2) {
            n_output = 0;
            LOG("SincResampler: %d < %d err = -11", n_input, ord2);
            return -11;
        }
        
        n_output = estimateOutputLength(n_input);
        if (n_output <= 0) return n_output;
        
        n_output = std::min(n_output, (int)(n_input / rt) + 2);
        
        process_internal(in, n_input, out, n_output, mode, scale);
        return n_output;
    }
    
private:
    
    void process_internal(const T *__restrict in, int n_input, void *__restrict out, int &n_output, SAMP_MODE mode, T1 scale)
    {
        int ord = tbl.order, ord2 = ord * 2;
        float x = leftover - ord;
        int estimated_outputs = estimateOutputLength(n_input);
        int wr_pos = 0;
        
        switch (mode) {
            case SAMP_UNCHANGED: {
                T *output = (T*)out;
                if constexpr(std::is_same<T, float>::value) {
                    process_float_internal(x, in, output, estimated_outputs, n_input, scale, wr_pos);
                } else if constexpr(std::is_same<T, mssFloat2>::value) {
                    process_float2_internal(x, in, output, estimated_outputs, n_input, scale, wr_pos);
                    
                } else {
                    process_generic_internal(x, in, output, estimated_outputs, n_input, scale, wr_pos);
                }
                break;
            }
            case SAMP_DOUBLE2FLOAT: {
                TF *output = (TF*)out;
                if (mD2FHook) {   // full-precision resample, then the hooked (dithered) reduction — see D2FHook
                    if ((int)mD2FScratch.size() < estimated_outputs + 4) mD2FScratch.resize(estimated_outputs + 4);
                    T *tmp = mD2FScratch.data();
                    if constexpr(std::is_same<T, float>::value) {
                        process_float_internal(x, in, tmp, estimated_outputs, n_input, scale, wr_pos);
                    } else if constexpr(std::is_same<T, mssFloat2>::value) {
                        process_float2_internal(x, in, tmp, estimated_outputs, n_input, scale, wr_pos);
                    } else {
                        process_generic_internal(x, in, tmp, estimated_outputs, n_input, scale, wr_pos);
                    }
                    mD2FHook(output, tmp, wr_pos);
                } else {
                    process_conversion_internal<TF>(x, in, output, estimated_outputs, n_input, scale, wr_pos, mode);
                }
                break;
            }
            case SAMP_FLOAT2DOUBLE: {
                TD *output = (TD*)out;
                process_conversion_internal<TD>(x, in, output, estimated_outputs, n_input, scale, wr_pos, mode);
                break;
            }
        }
        
        leftover = x - (n_input - ord);
        
        int copy_size = std::min(ord2, n_input);
        memcpy(p, &in[n_input - copy_size], copy_size * sizeof(T));
        n_output = wr_pos;
    }
    
    // Thread-safe interpolation function (reads shared state safely)
    __inline T sr_intp(float x, const T *__restrict input, T1 gain)
    {
        if (!p || !input) return T{0};
        
        const int fx = (int)floor(x);
        float fraction = F_ABS(x - fx);
        T sample, sum{0};
        
        int ord = tbl.order, ord2 = ord << 1;
        int dx = (int)(fraction * tbl.precision + 0.50) * ord2;
        
        if (dx < 0 || dx >= tbl.sz) return T{0};
        
        T1* coeffs = &tbl.kernel[dx];
        for (int i = fx - ord + 1; i <= fx + ord; i += RS_BLOCKSIZE)
        {
            int idx = i + ord2 - 1;
            if (idx + RS_BLOCKSIZE < ord2) {
                sum += mulsum(&p[idx], coeffs);
            } else if (idx >= ord2) {
                sum += mulsum(&input[idx - ord2], coeffs);
            } else {
                for (int j=0, k=idx; j < RS_BLOCKSIZE; j++, k++)
                {
                    sample = (k >= ord2) ? input[k - ord2] : p[k];
                    sum += sample * coeffs[j];
                }
            }
            coeffs += RS_BLOCKSIZE;
        }
        return sum * gain;
    }
    
#if defined(SINC_USE_NEON) && defined(USE_LANE)
    
    void process_float_internal(float& x, const float *__restrict input, float *__restrict output,
                                int estimated_outputs, int n_input, float scale, int& wr_pos)
    {
        const int unroll = 4;
        const int ord = tbl.order;
        const float x_limit = n_input - ord;    // ← add this
        int i = 0;
        
        for (; i + unroll <= estimated_outputs && (x + rt * 3) < x_limit; i += unroll) {
            output[wr_pos]   = neon_sr_intp_float(x,            input, n_input, scale);
            output[wr_pos+1] = neon_sr_intp_float(x + rt,       input, n_input, scale);
            output[wr_pos+2] = neon_sr_intp_float(x + rt*2,     input, n_input, scale);
            output[wr_pos+3] = neon_sr_intp_float(x + rt*3,     input, n_input, scale);
            
            wr_pos += unroll;
            x += rt * unroll;
        }
        
        for (; i < estimated_outputs && x < x_limit; i++) {
            output[wr_pos++] = neon_sr_intp_float(x, input, n_input, scale);
            x += rt;
        }
    }
    
    void process_float2_internal(float& x, const mssFloat2 *__restrict input, mssFloat2 *__restrict output, int estimated_outputs, int n_input, float scale, int& wr_pos)
    {
        const int unroll = 4;
        int i = 0;
        
        for (; i + unroll <= estimated_outputs; i += unroll) {
            output[wr_pos] = neon_sr_intp_float2(x, input, n_input, scale);
            output[wr_pos+1] = neon_sr_intp_float2(x + rt, input, n_input, scale);
            output[wr_pos+2] = neon_sr_intp_float2(x + rt*2, input, n_input, scale);
            output[wr_pos+3] = neon_sr_intp_float2(x + rt*3, input, n_input, scale);
            
            wr_pos += unroll;
            x += rt * unroll;
        }
        
        for (; i < estimated_outputs; i++) {
            output[wr_pos++] = neon_sr_intp_float2(x, input, n_input, scale);
            x += rt;
        }
    }
    
    void process_generic_internal(float& x, const T *__restrict input, T *__restrict output, int estimated_outputs, int n_input, T1 scale, int& wr_pos)
    {
        int ord = tbl.order;
        while (x < n_input - ord) {
            output[wr_pos++] = sr_intp(x, input, scale);
            x += rt;
        }
    }
    
    template<typename OutputType>
    void process_conversion_internal(float& x, const T *__restrict input, OutputType *__restrict output, int estimated_outputs, int n_input, T1 scale, int& wr_pos, SAMP_MODE mode)
    {
        if constexpr(std::is_same<OutputType, TF>::value && std::is_same<T, double>::value) {
            // Double to float conversion
            const int unroll = 4;
            int i = 0;
            
            for (; i + unroll <= estimated_outputs; i += unroll) {
                output[wr_pos] = neon_sr_intp_double_to_float(x, input, n_input, scale);
                output[wr_pos+1] = neon_sr_intp_double_to_float(x + rt, input, n_input, scale);
                output[wr_pos+2] = neon_sr_intp_double_to_float(x + rt*2, input, n_input, scale);
                output[wr_pos+3] = neon_sr_intp_double_to_float(x + rt*3, input, n_input, scale);
                
                wr_pos += unroll;
                x += rt * unroll;
            }
            
            for (; i < estimated_outputs; i++) {
                output[wr_pos++] = neon_sr_intp_double_to_float(x, input, n_input, scale);
                x += rt;
            }
        } else if constexpr(std::is_same<OutputType, TD>::value && std::is_same<T, float>::value) {
            // Float to double conversion
            const int unroll = 4;
            int i = 0;
            
            for (; i + unroll <= estimated_outputs; i += unroll) {
                output[wr_pos] = neon_sr_intp_float_to_double(x, input, n_input, scale);
                output[wr_pos+1] = neon_sr_intp_float_to_double(x + rt, input, n_input, scale);
                output[wr_pos+2] = neon_sr_intp_float_to_double(x + rt*2, input, n_input, scale);
                output[wr_pos+3] = neon_sr_intp_float_to_double(x + rt*3, input, n_input, scale);
                
                wr_pos += unroll;
                x += rt * unroll;
            }
            
            for (; i < estimated_outputs; i++) {
                output[wr_pos++] = neon_sr_intp_float_to_double(x, input, n_input, scale);
                x += rt;
            }
        } else {
            // Generic conversion
            int ord = tbl.order;
            while (x < n_input - ord) {
                output[wr_pos++] = convertvector_safe<OutputType>(sr_intp(x, input, scale));
                x += rt;
            }
        }
    }
    
protected:
    
    __inline float neon_sr_intp_float(float x, const float *__restrict input, int n_input, float scale)
    {
        const int fx = (int)floor(x);
        float fraction = F_ABS(x - fx);
        
        int ord = tbl.order, ord2 = ord << 1;
        int dx = (int)(fraction * tbl.precision + 0.50) * ord2;
        float* coeffs = (float*)&tbl.kernel[dx];
        
        float32x4_t sum_vec = vdupq_n_f32(0.0f);  // Single accumulator
        
        // Single simple loop: process RS_BLOCKSIZE (4) samples at a time
        for (int i = fx - ord + 1; i <= fx + ord; i += RS_BLOCKSIZE) {
            const int idx = i + ord2 - 1;
            
            alignas(16) simd_float4 data_wide = {0};  // Exactly RS_BLOCKSIZE floats
            float* buffer = (float*)&data_wide;       // Alias for element access
            
            int srcIdx_start = idx;
            int srcIdx_end = idx + RS_BLOCKSIZE - 1;  // idx + 3
            
            // Bulk load when data is contiguous from single source
            if (srcIdx_start >= 0 && srcIdx_end < ord2) {
                // All 4 samples from p array - single simd_float4 load!
                data_wide = *((simd_float4*)&p[srcIdx_start]);
            } else if (srcIdx_start >= ord2 && (srcIdx_end - ord2) < n_input) {
                // All samples potentially from input array
                int inputIdx_start = srcIdx_start - ord2;
                int inputIdx_end = srcIdx_end - ord2;
                
                if (inputIdx_start >= 0 && inputIdx_end < n_input) {
                    // All 4 samples from input - single simd_float4 load!
                    data_wide = *((simd_float4*)&input[inputIdx_start]);
                } else {
                    // Partial input range - use individual loads
                    goto individual_loads;
                }
            } else {
                // Mixed sources or boundary conditions
                individual_loads:
                for (int j = 0; j < RS_BLOCKSIZE; j++) {
                    int srcIdx = idx + j;
                    if (srcIdx < 0) {
                        buffer[j] = 0.0f;
                    } else if (srcIdx < ord2) {
                        buffer[j] = p[srcIdx];
                    } else {
                        int inputIdx = srcIdx - ord2;
                        if (inputIdx >= 0 && inputIdx < n_input) {
                            buffer[j] = input[inputIdx];
                        } else {
                            buffer[j] = 0.0f;
                        }
                    }
                }
            }
            
            // Load coefficients and data directly as NEON vectors
            float32x4_t coef_vec = vld1q_f32(coeffs);
            float32x4_t data_vec = vld1q_f32(buffer);
            
            // Single multiply-accumulate operation
            sum_vec = vmlaq_f32(sum_vec, data_vec, coef_vec);
            
            coeffs += RS_BLOCKSIZE;
        }
        
        // Simple horizontal sum
        float32x2_t sum_pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
        sum_pair = vpadd_f32(sum_pair, sum_pair);
        
        return vget_lane_f32(sum_pair, 0) * scale;
    }
    
    __inline float neon_sr_intp_double(float x, const double *__restrict input, int n_input, float scale)
    {
        const int fx = (int)floor(x);
        float fraction = F_ABS(x - fx);
        
        int ord = tbl.order, ord2 = ord << 1;
        int dx = (int)(fraction * tbl.precision + 0.50) * ord2;
        float* coeffs = (float*)&tbl.kernel[dx];
        
        float32x4_t sum_vec = vdupq_n_f32(0.0f);  // Single accumulator
        
        // Simple single loop: process RS_BLOCKSIZE (4) samples at a time
        for (int i = fx - ord + 1; i <= fx + ord; i += RS_BLOCKSIZE) {
            const int idx = i + ord2 - 1;
            
            alignas(16) float buffer[RS_BLOCKSIZE]{0};
            
            int srcIdx_start = idx;
            int srcIdx_end = idx + RS_BLOCKSIZE - 1;
            
            // Fast path: bulk processing when data is contiguous
            if (srcIdx_start >= 0 && srcIdx_end < ord2) {
                // All from p array (float) - direct copy
                simd_float4 data_wide = *((simd_float4*)&p[srcIdx_start]);
                *((simd_float4*)buffer) = data_wide;
                
            } else if (srcIdx_start >= ord2 && (srcIdx_end - ord2) < n_input) {
                // All from input array (double->float conversion)
                int inputIdx_start = srcIdx_start - ord2;
                
                // Load 4 doubles and convert to 4 floats efficiently
                float64x2_t doubles1 = vld1q_f64(&input[inputIdx_start]);
                float64x2_t doubles2 = vld1q_f64(&input[inputIdx_start + 2]);
                
                float32x2_t floats1 = vcvt_f32_f64(doubles1);
                float32x2_t floats2 = vcvt_f32_f64(doubles2);
                
                float32x4_t result = vcombine_f32(floats1, floats2);
                vst1q_f32(buffer, result);
                
            } else {
                // Mixed sources - individual loads with conversion
                for (int j = 0; j < RS_BLOCKSIZE; j++) {
                    int srcIdx = idx + j;
                    if (srcIdx < 0) {
                        // buffer[j] already 0
                    } else if (srcIdx < ord2) {
                        buffer[j] = p[srcIdx];  // float from p
                    } else {
                        int inputIdx = srcIdx - ord2;
                        if (inputIdx >= 0 && inputIdx < n_input) {
                            buffer[j] = (float)input[inputIdx];  // Convert double to float
                        }
                        // else buffer[j] stays 0
                    }
                }
            }
            
            // Process with NEON
            float32x4_t data_vec = vld1q_f32(buffer);
            float32x4_t coef_vec = vld1q_f32(coeffs);
            
            sum_vec = vmlaq_f32(sum_vec, data_vec, coef_vec);
            
            coeffs += RS_BLOCKSIZE;
        }
        
        // Simple horizontal sum
        float32x2_t sum_pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
        sum_pair = vpadd_f32(sum_pair, sum_pair);
        
        return vget_lane_f32(sum_pair, 0) * scale;
    }
    
    
    __inline mssFloat2 neon_sr_intp_float2(float x, const mssFloat2 *__restrict input, int n_input, float scale)
    {
        const int fx = (int)floor(x);
        float fraction = F_ABS(x - fx);
        
        int ord = tbl.order, ord2 = ord * 2;
        int dx = (int)(fraction * tbl.precision + 0.50) * ord2;
        float* coeffs = (float*)&tbl.kernel[dx];
        
        float32x4_t sum_left = vdupq_n_f32(0.0f);
        float32x4_t sum_right = vdupq_n_f32(0.0f);
        
        // Simple single loop: process RS_BLOCKSIZE samples at a time
        for (int i = fx - ord + 1; i <= fx + ord; i += RS_BLOCKSIZE) {
            const int idx = i + ord2 - 1;
            
            alignas(16) simd_float2 stereo_buffer[RS_BLOCKSIZE] = {0};
            
            int srcIdx_start = idx;
            int srcIdx_end = idx + RS_BLOCKSIZE - 1;
            
            // Fast path: bulk loading
            if (srcIdx_start >= 0 && srcIdx_end < ord2) {
                // All from p array
                memcpy(stereo_buffer, &p[srcIdx_start], RS_BLOCKSIZE * sizeof(simd_float2));
            } else if (srcIdx_start >= ord2 && (srcIdx_end - ord2) < n_input) {
                // All from input array
                memcpy(stereo_buffer, &input[srcIdx_start - ord2], RS_BLOCKSIZE * sizeof(simd_float2));
            } else {
                // Mixed sources - individual loads
                for (int j = 0; j < RS_BLOCKSIZE; j++) {
                    int srcIdx = idx + j;
                    if (srcIdx < 0) {
                        // stereo_buffer[j] already {0,0}
                    } else if (srcIdx < ord2) {
                        stereo_buffer[j] = p[srcIdx];
                    } else {
                        int inputIdx = srcIdx - ord2;
                        if (inputIdx >= 0 && inputIdx < n_input) {
                            stereo_buffer[j] = input[inputIdx];
                        }
                    }
                }
            }
            
            // Process with NEON
            float32x4_t coeffs_vec = vld1q_f32(coeffs);
            float32x4x2_t stereo = vld2q_f32((const float*)stereo_buffer);
            
            sum_left = vmlaq_f32(sum_left, stereo.val[0], coeffs_vec);
            sum_right = vmlaq_f32(sum_right, stereo.val[1], coeffs_vec);
            
            coeffs += RS_BLOCKSIZE;
        }
        
        // Horizontal sum
        float32x2_t sum_left_pair = vpadd_f32(vget_low_f32(sum_left), vget_high_f32(sum_left));
        float32x2_t sum_right_pair = vpadd_f32(vget_low_f32(sum_right), vget_high_f32(sum_right));
        
        sum_left_pair = vpadd_f32(sum_left_pair, sum_left_pair);
        sum_right_pair = vpadd_f32(sum_right_pair, sum_right_pair);
        
        return mssFloat2{
            vget_lane_f32(sum_left_pair, 0) * scale,
            vget_lane_f32(sum_right_pair, 0) * scale
        };
    }
    
    __inline float neon_sr_intp_double_to_float(float x, const double *__restrict input, int n_input, float scale) {
        return neon_sr_intp_double(x, input, n_input, scale);   // was missing n_input (never instantiated before the double mp3 path)
    }
    
    __inline double neon_sr_intp_float_to_double(float x, const float *__restrict input, int n_input, float scale) {
        return neon_sr_intp_float(x, input, n_input, scale);
    }
    
#elif !defined(SINC_USE_NEON)
    
    // Non-NEON single-core implementations remain the same as original
    void process_float_internal(float& x, const float *__restrict input, float *__restrict output, int estimated_outputs, int n_input, float scale, int& wr_pos)
    {
        int ord = tbl.order;
        while (x < n_input - ord && wr_pos < estimated_outputs) {
            output[wr_pos++] = sr_intp(x, input, scale);
            x += rt;
        }
    }
    
    void process_float2_internal(float& x, const mssFloat2 *__restrict input, mssFloat2 *__restrict output, int estimated_outputs, int n_input, float scale, int& wr_pos)
    {
        int ord = tbl.order;
        while (x < n_input - ord) {
            output[wr_pos++] = sr_intp(x, input, scale);
            x += rt;
        }
    }
    
    void process_generic_internal(float& x, const T *__restrict input, T *__restrict output, int estimated_outputs, int n_input, T1 scale, int& wr_pos)
    {
        int ord = tbl.order;
        while (x < n_input - ord) {
            output[wr_pos++] = sr_intp(x, input, scale);
            x += rt;
        }
    }
    
    template<typename OutputType>
    void process_conversion_internal(float& x, const T *__restrict input, OutputType *__restrict output, int estimated_outputs, int n_input, T1 scale, int& wr_pos, SAMP_MODE mode)
    {
        int ord = tbl.order;
        while (x < n_input - ord) {
            if constexpr(IS_VECTOR(T)) {
                output[wr_pos++] = convertvector(sr_intp(x, input, scale), OutputType);
            } else {
                output[wr_pos++] = convertvector_safe<OutputType>(sr_intp(x, input, scale));
            }
            x += rt;
        }
    }
    
#endif
    
    // Generic mulsum implementation (thread-safe, read-only access to coefficients)
#if defined(RS_OPTIMIZE) && RS_BLOCKSIZE == 4
    __inline TX mulsum(const T  *__restrict samples, const T1 *__restrict coeffs)
    {
        if constexpr( SimdSize<T> == 2 ) {
            const T4 c = *(T4*) coeffs;
            const T8 s = shufflevector(*(T8*) samples, int8v{0,2,4,6,1,3,5,7}) * __builtin_shufflevector(c, c, 0,1,2,3,0,1,2,3);   // (c,c) for float AND double T (was simd_make_float8)
            return T2{ simd_reduce_add(s.lo), simd_reduce_add(s.hi) };
        } else if constexpr( SimdSize<T> == 1 )
        {
            const T4 c = *(T4*) coeffs;
            const T4 s = (*(T4*) samples) * c;
            
            return simd_reduce_add(s);
        } else {
            assert(NULL);
        }
        return 0.0;
    }
#else
    __inline T mulsum(const T*__restrict samples, const T1 *__restrict coeffs)
    {
        T sum = 0.0;
        for (int i = 0; i < RS_BLOCKSIZE; i++)
            sum += samples[i] * coeffs[i];
        return sum;
    }
#endif
    
    // Thread-safe kernel creation (protected by mutex in setup())
    bool sr_create(int order, int precision, float cutoff, float rate)
    {
        bool same = false;
        int idx = 0;
        
        if (tbl.order == order && tbl.precision == precision && tbl.cutoff == cutoff && tbl.kernel != NULL) {
            same = true;
        }
        
        leftover = 0.0;
        if (p) memset(p, 0, tbl.order * 2 * sizeof(T));
        
        if (!same || !tbl.kernel) {
            if (tbl.kernel) {
                callocT_free(tbl.kernel);
                tbl.kernel = NULL;
            }
            if (p) {
                callocT_free(p);
                p = NULL;
            }
            
            memset(&tbl, 0, sizeof(sinc_table));
            
            tbl.order = order;
            tbl.precision = precision;
            tbl.cutoff = cutoff;
            precompute_window(order * 2);
            
            p = callocT<T>(order * 2);
            if (!p) {
                LOG("SincResampler: Failed to allocate buffer");
                return false;
            }
            memset(p, 0, order * 2 * sizeof(T));
            
            tbl.sz = order * 2 * (precision + 1);
            
            T1 *kernel = callocT<T1>(tbl.sz);
            if (!kernel) {
                LOG("SincResampler: Failed to allocate kernel table of size %d", tbl.sz);
                callocT_free(p);
                p = NULL;
                return false;
            }
            
            tbl.kernel = kernel;
            
            for (int offset = 0; offset <= precision; offset++)
            {
                T1 scale = 0.0;
                int startidx = idx;
                for (int i = -order + 1; i <= order; i++) {
                    T1 xk = kernel[idx] = kernel_func((T1) offset / (T1)precision - i, cutoff, order);
                    scale += xk;
                    idx++;
                }
                idx = startidx;
                T1 one_scale = 1./scale;
                for (int i = -order + 1; i <= order; i++)
                    kernel[idx++] *= one_scale;
            }
        }
        return true;
    }
};

template <typename T, WindowType wt = WINTYPE_KAISER>
class OriginalSincResampler {
public:   // the type aliases are part of the interface
    using TD = SimdSame<T,double>;
    using TF = SimdSame<T,float>;
    using T1 = SimdBase<T>;
private:
    // DOUBLE→FLOAT = DITHERED (2026-10-25, user: "twist the SincResampler to include dither and sharp curve"): a double instance owns
    // caout's SSRCDither (dither.h: DITHER_TYPE = 15kSharpEx, the OUTPUT rate's curve) and SAMP_DOUBLE2FLOAT resamples at full precision,
    // then reduces with it at kD2FQuantScale = 2^23 (float32's mantissa), as caout's ConvertToFloat does. Float instances: plain convert.
    static constexpr bool   kD2FDither     = SINC_HAS_DITHER && std::is_same_v<SimdBase<T>, double>;
    static constexpr double kD2FQuantScale = 8388608.0;
#if SINC_HAS_DITHER
    std::unique_ptr<SSRCDither<T, 1>> mDither;    // one core: 1 MB of noise per stereo instance, not 6
#endif
    std::vector<T> mD2FScratch;
    using T8 = Simd<SimdBase<T>,8>;
    using T4 = Simd<SimdBase<T>,4>;
    using T2 = Simd<SimdBase<T>,2>;
    static constexpr int TN = SimdSize<T>;         // scalars per sample: 1 mono, 2 stereo
    static const int DEFAULT_ORDER = 128;
    static const int DEFAULT_PRECISION = 10000;
protected:

    std::atomic<bool> initialized{false};

    typedef struct sinc_table {
        int order;
        int precision;
        float cutoff;
        T1 *kernel;
        int sz;
    } sinc_table;

    double rt, leftover;                           // double: positions are x0 + k*rt, no accumulated rounding
    sinc_table tbl;
    T* p;                                          // history: the ord2 samples preceding the current block
    int order = DEFAULT_ORDER;
    int precision = DEFAULT_PRECISION;
    T1* _window_table = nullptr;
    double _kaiser_beta  = 8.6;
    double _stopband_att = 100.0;
#if SR_SINC_CACHE
    std::shared_ptr<T1> m_kref;                    // owns tbl.kernel (shared with every instance of the same key)
    struct KEntry { int order, precision; float cutoff; double beta; std::shared_ptr<T1> k; double used; };
    struct KCache { std::mutex mx; std::vector<KEntry> e; bool reaping = false; };
    static KCache &kcache() { static KCache *c = new KCache; return *c; }   // never destroyed: the reaper may outlive exit
    static double kNow() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    static std::shared_ptr<T1> kFind(KCache &kc, int o, int pr, float c, double b) {   // under kc.mx
        for (KEntry &e : kc.e) if (e.order == o && e.precision == pr && e.cutoff == c && e.beta == b) { e.used = kNow(); return e.k; }
        return nullptr;
    }
    static std::shared_ptr<T1> kGet(int o, int pr, float c, double b) {
        KCache &kc = kcache(); std::lock_guard<std::mutex> l(kc.mx); return kFind(kc, o, pr, c, b);
    }
    static std::shared_ptr<T1> kPut(int o, int pr, float c, double b, std::shared_ptr<T1> k) {   // a racing build of the key: the first wins
        KCache &kc = kcache(); std::lock_guard<std::mutex> l(kc.mx);
        if (std::shared_ptr<T1> had = kFind(kc, o, pr, c, b)) return had;
        kc.e.push_back({o, pr, c, b, k, kNow()});
        if (!kc.reaping) { kc.reaping = true; std::thread(kReap).detach(); }
        return k;
    }
    static void kReap() {                          // every 5 s: frees tables no instance holds, idle SR_SINC_CACHE_IDLE s; ends when none is left
        KCache &kc = kcache();
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            std::vector<std::shared_ptr<T1>> drop;     // destroyed after the unlock
            std::lock_guard<std::mutex> l(kc.mx);
            const double now = kNow();
            for (size_t i = 0; i < kc.e.size();)
                if (kc.e[i].k.use_count() == 1 && now - kc.e[i].used > SR_SINC_CACHE_IDLE) { drop.push_back(std::move(kc.e[i].k)); kc.e.erase(kc.e.begin() + i); }
                else i++;
            if (kc.e.empty()) { kc.reaping = false; return; }
        }
    }
#endif
public:
    OriginalSincResampler() {
        p = NULL;
        tbl.kernel = NULL;
        rt = -1;
        leftover = 0;
        memset(&tbl, 0, sizeof(sinc_table));
    }

    ~OriginalSincResampler() {
        cleanup();
    }

    void setFilterLength(int length, int precision = DEFAULT_PRECISION) { order = length; this->precision = precision; }

    // Upper bound on the outputs the next process(n_input) will produce — size output buffers with it.
    // The exact count is ceil((n_input - leftover) / rt); +1 absorbs floating-point rounding at the limit.
    // Reads leftover, so call it from the thread that calls process().
    int estimateOutputLength(int n_input) const {
        if (rt <= 0 || n_input <= 0 || !initialized.load()) {
            return 0;
        }
        if (n_input < tbl.order * 2) {
            return (int)(n_input / rt) + 1;          // process() refuses such blocks anyway (-11)
        }
        return (int)ceil((n_input - leftover) / rt) + 1;
    }

    void init(double in_rate, double out_rate, int nch) {
        initialized.store(false);

        rt = in_rate / out_rate;
        leftover = 0.0;

        float c = 0.95f / std::max(1.0f, (float)rt);
        bool success = sr_create(order, precision, c, (float)rt);
#if SINC_HAS_DITHER
        if constexpr (kD2FDither) {   // SAMP_DOUBLE2FLOAT's ditherer, at the curve of the rate it writes
            if (!mDither) mDither = std::make_unique<SSRCDither<T, 1>>();
            if (mDither->Init((int)out_rate, DITHER_ONLY_NATIVE_SRATE, -kD2FQuantScale * 2, kD2FQuantScale * 2, DITHER_SHAPER[DITHER_TYPE],
                              DITHER_FILLER[DITHER_TYPE], DITHER_PEAKS[DITHER_TYPE], (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)this) != 0)
                mDither.reset();                      // no curve for that rate: the plain convert
            if (mD2FScratch.size() < 16384) mD2FScratch.resize(16384);   // 2048 frames at 8 kHz → 48k, sized off the audio thread
        }
#endif

        if (success) {
            initialized.store(true);
        }
        reset();
    }

    void setKaiserBeta(double beta) {
        _kaiser_beta = beta;
        // window will rebuild on next init() call
    }

    void setStopbandAttenuation(double att) {
        _stopband_att = att;
        // could derive beta from att: beta ≈ 0.1102*(att-8.7) for att>50dB
        if (att > 50.0) {
            _kaiser_beta = 0.1102 * (att - 8.7);
        } else if (att >= 21.0) {
            _kaiser_beta = 0.5842 * pow(att - 21.0, 0.4) + 0.07886 * (att - 21.0);
        } else {
            _kaiser_beta = 0.0;
        }
    }

    void precompute_window(int window_size) {
        if (_window_table) callocT_free(_window_table);
        _window_table = callocT<T1>(window_size);
        if (!_window_table) return;

        for (int n = 0; n < window_size; n++) {
            if constexpr (wt == WINTYPE_KAISER) {
                _window_table[n] = (T1)KaiserWindow::kaiser(n, window_size, _kaiser_beta);
            } else if constexpr (wt == WINTYPE_HANNING) {
                _window_table[n] = (T1)HanningWindow::hanning(n, window_size);
            } else if constexpr (wt == WINTYPE_SINE) {
                _window_table[n] = (T1)SineWindow::sine(n, window_size);
            } else if constexpr (wt == WINTYPE_BLACKMAN) {
                // inline since you don't have a class for it yet
                double x = (double)n / (window_size - 1);
                _window_table[n] = (T1)(0.3635819
                                        - 0.4891775 * F_COS(2.0 * M_PI * x)
                                        + 0.1365995 * F_COS(4.0 * M_PI * x)
                                        - 0.0106411 * F_COS(6.0 * M_PI * x));
            } else {
                // WINTYPE_NONE — rectangular
                _window_table[n] = (T1)1.0;
            }
        }
    }

    void reset() {
        leftover = 0.0;
        if (p) {
            memset(p, 0, tbl.order * 2 * sizeof(T));
        }
    }

    void cleanup() {
        initialized.store(false);
        if (tbl.kernel) {
#if SR_SINC_CACHE
            m_kref.reset();
#else
            callocT_free(tbl.kernel);
#endif
            tbl.kernel = nullptr;
        }
        if (p) {
            callocT_free(p);
            p = nullptr;
        }
        if (_window_table) {
            callocT_free(_window_table);
            _window_table = nullptr;
        }
    }

    inline T1 kernel_func(T1 x, T1 cutoff, int ord) {
        if (x <= -(T1)ord || x >= (T1)ord) return (T1)0.0;

        // Map x back to window table index
        // x ranges (-ord, +ord), table is [0, ord*2)
        int n = (int)(x + ord);  // shift to [0, ord*2)
        n = F_CLAMP(n, 0, ord * 2 - 1);

        return sinc(cutoff * x) * _window_table[n];
    }

    static inline T1 sinc(T1 x) {
        return (x == 0.0) ? 1.0 : F_SIN(M_PI * x) / (M_PI * x);
    }

    static inline T1 sine2(T1 t, int N) {
        T1 m = F_SIN(t * M_PI / (T1)(N - 1));
        return m * m;
    }

    static inline T1 hanning(T1 t, int N) {
        return 0.5 * (1.0 - F_COS(2.0 * M_PI * t / (T1)(N - 1)));
    }

    static inline T1 blackman_nuttall(T1 x, int N) {
        return 0.3635819 - 0.4891775 * F_COS((2.0 * M_PI * x) / (T1)(N - 1))
        + 0.1365995 * F_COS((4.0 * M_PI * x) / (T1)(N - 1))
        - 0.0106411 * F_COS((6.0 * M_PI * x) / (T1)(N - 1));
    }

    // Main process function
    int process(const T*__restrict in, int n_input, void *__restrict out, int &n_output, SAMP_MODE mode = SAMP_UNCHANGED, double scale = 1.0)
    {
        if (rt <= 0 || !initialized.load() || !tbl.kernel || !p || !in || !out) {
            n_output = 0;
            return -10;
        }
        const int ord2 = tbl.order * 2;
        if (n_input < ord2) {
            n_output = 0;
            LOG("SincResampler: %d < %d err = -11", n_input, ord2);
            return -11;
        }

        n_output = estimateOutputLength(n_input);
        if (n_output <= 0) return n_output;

        n_output = std::min(n_output, (int)(n_input / rt) + 2);

        process_internal(in, n_input, out, n_output, mode, (T1)scale);
        return n_output;
    }

private:

    // n_output in: the caller's buffer size (= estimateOutputLength); out: the number actually written.
    void process_internal(const T *__restrict in, int n_input, void *__restrict out, int &n_output, SAMP_MODE mode, T1 gain)
    {
        const int    ord = tbl.order, ord2 = ord * 2;
        const double x0      = leftover - ord;         // position of the first output
        const double limit   = n_input - ord;          // first position whose tap window is not complete yet
        const int    max_out = n_output;
        int k = 0;

        switch (mode) {
            case SAMP_UNCHANGED: {
                k = run(x0, limit, max_out, in, n_input, (T *)out, gain, [](T v) { return v; });
                break;
            }
            case SAMP_DOUBLE2FLOAT: {
#if SINC_HAS_DITHER
                if constexpr (kD2FDither) {
                    if (mDither) {   // full-precision resample, then dither.h's noise-shaped reduction (see kD2FDither)
                        if ((int)mD2FScratch.size() < max_out) mD2FScratch.resize(max_out);
                        k = run(x0, limit, max_out, in, n_input, mD2FScratch.data(), gain, [](T v) { return v; });
                        mDither->process((typename SSRCDither<T, 1>::TF *)out, mD2FScratch.data(), k, 1.0 / kD2FQuantScale, 0);
                        break;
                    }
                }
#endif
                k = run(x0, limit, max_out, in, n_input, (TF *)out, gain, [](T v) { return convertvector_safe<TF>(v); });
                break;
            }
            case SAMP_FLOAT2DOUBLE: {
                k = run(x0, limit, max_out, in, n_input, (TD *)out, gain, [](T v) { return convertvector_safe<TD>(v); });
                break;
            }
        }

        leftover = (x0 + k * rt) - limit;              // in [0, rt) when the loop stopped at the limit

        int copy_size = std::min(ord2, n_input);
        memcpy(p, &in[n_input - copy_size], copy_size * sizeof(T));
        n_output = k;
    }

    // The driver ("lane") loop: outputs k = 0.. while x0 + k*rt < limit, 4 independent interpolations per
    // iteration so their dot products overlap in the pipeline. Returns the number of outputs written.
    template <typename Out, typename Conv>
    __inline int run(double x0, double limit, int max_out, const T *__restrict in, int n_input, Out *__restrict out, T1 gain, Conv conv) const
    {
        int k = 0;
        for (; k + 4 <= max_out && x0 + (k + 3) * rt < limit; k += 4) {
            out[k]     = conv(sr_intp(x0 +  k      * rt, in, n_input, gain));
            out[k + 1] = conv(sr_intp(x0 + (k + 1) * rt, in, n_input, gain));
            out[k + 2] = conv(sr_intp(x0 + (k + 2) * rt, in, n_input, gain));
            out[k + 3] = conv(sr_intp(x0 + (k + 3) * rt, in, n_input, gain));
        }
        for (; k < max_out && x0 + k * rt < limit; k++) {
            out[k] = conv(sr_intp(x0 + k * rt, in, n_input, gain));
        }
        return k;
    }

    // ---- vector helpers for sr_intp -------------------------------------------------------------------------

    // Unaligned loads: the window starts wherever x lands, so T8/T4/T2 (32/16/8-byte aligned types) can't be
    // dereferenced directly. memcpy compiles to plain vector loads.
    template <typename V> static __inline V ld(const T1 *s) { V v; memcpy(&v, s, sizeof(V)); return v; }

    // Coefficients for one 8/4/2-lane step: one per lane (mono) or one per lane pair (stereo: c0 c0 c1 c1 ...)
    static __inline T8 cf8(const T1 *c) {
        if constexpr (TN == 1) return ld<T8>(c);
        else { const T4 v = ld<T4>(c); return __builtin_shufflevector(v, v, 0, 0, 1, 1, 2, 2, 3, 3); }
    }
    static __inline T4 cf4(const T1 *c) {
        if constexpr (TN == 1) return ld<T4>(c);
        else { const T2 v = ld<T2>(c); return __builtin_shufflevector(v, v, 0, 0, 1, 1); }
    }
    static __inline T2 cf2(const T1 *c) {
        if constexpr (TN == 1) return ld<T2>(c);
        else { T2 v = c[0]; return v; }
    }

    // acc += samples[0..n) * coeffs[0..n) over one contiguous run of n taps (d = the first tap's scalars).
    // 16/8/4-lane steps, the tail in 2s, one scalar for an odd run (block edges only).
    static __inline void accumulate(T8 &acc0, T8 &acc1, const T1 *__restrict d, const T1 *__restrict c, int n)
    {
        constexpr int S8 = 8 / TN, S4 = 4 / TN, S2 = 2 / TN;   // taps per 8/4/2-lane step
        int k = 0;
        for (; k + 2 * S8 <= n; k += 2 * S8) {
            acc0 = F_FMA(ld<T8>(d + TN * k),        cf8(c + k),      acc0);
            acc1 = F_FMA(ld<T8>(d + TN * (k + S8)), cf8(c + k + S8), acc1);
        }
        if (k + S8 <= n) { acc0       = F_FMA(ld<T8>(d + TN * k), cf8(c + k), acc0);       k += S8; }
        if (k + S4 <= n) { acc1.lo    = F_FMA(ld<T4>(d + TN * k), cf4(c + k), acc1.lo);    k += S4; }
        if (k + S2 <= n) { acc0.lo.lo = F_FMA(ld<T2>(d + TN * k), cf2(c + k), acc0.lo.lo); k += S2; }
        if constexpr (TN == 1) { if (k < n) acc1[0] = F_FMA(d[k], c[k], acc1[0]); }
    }

    static __inline T fold(T8 a) {                               // lanes -> sample
        if constexpr (TN == 1) return simd_reduce_add(a);
        else { const T4 b = a.lo + a.hi; return b.lo + b.hi; }   // L R L R L R L R -> L R L R -> {L, R}
    }

    // ⭐ 8-LANE FRAMES (TN == 8, 2026-11-08, user: "one simd_float8 resampler over all 8 lanes"): the AAC plugin's surround lanes in ONE
    // resampler instead of four float2 ones (half a NEON register each). One tap = one 8-wide FMA. The eight partial sums are the
    // float2 path's exactly — lane pair q of acc0 / acc1 there = partial q / q+4 here, the same 8-, 4-, 2- and 1-tap steps, the same
    // final tree — so every lane comes out BIT-IDENTICAL to the pair-wise resampler (tools/resampler_test/lanes8_test).
    static __inline T8 splat8(T1 v) { T8 r = v; return r; }
    static __inline void accumulate8(T8 *__restrict acc, const T1 *__restrict d, const T1 *__restrict c, int n)
    {
        int k = 0;
        for (; k + 8 <= n; k += 8) {                                  // the double 8-lane step: taps k..k+3 -> p0..p3, k+4..k+7 -> p4..p7
            acc[0] = F_FMA(ld<T8>(d + 8 * k),       splat8(c[k]),     acc[0]);
            acc[1] = F_FMA(ld<T8>(d + 8 * (k + 1)), splat8(c[k + 1]), acc[1]);
            acc[2] = F_FMA(ld<T8>(d + 8 * (k + 2)), splat8(c[k + 2]), acc[2]);
            acc[3] = F_FMA(ld<T8>(d + 8 * (k + 3)), splat8(c[k + 3]), acc[3]);
            acc[4] = F_FMA(ld<T8>(d + 8 * (k + 4)), splat8(c[k + 4]), acc[4]);
            acc[5] = F_FMA(ld<T8>(d + 8 * (k + 5)), splat8(c[k + 5]), acc[5]);
            acc[6] = F_FMA(ld<T8>(d + 8 * (k + 6)), splat8(c[k + 6]), acc[6]);
            acc[7] = F_FMA(ld<T8>(d + 8 * (k + 7)), splat8(c[k + 7]), acc[7]);
        }
        if (k + 4 <= n) { for (int q = 0; q < 4; q++) acc[q] = F_FMA(ld<T8>(d + 8 * (k + q)), splat8(c[k + q]), acc[q]); k += 4; }   // the 8-lane step: -> p0..p3
        if (k + 2 <= n) { acc[4] = F_FMA(ld<T8>(d + 8 * k), splat8(c[k]), acc[4]); acc[5] = F_FMA(ld<T8>(d + 8 * (k + 1)), splat8(c[k + 1]), acc[5]); k += 2; }   // the 4-lane step: acc1.lo -> p4, p5
        if (k < n)      { acc[0] = F_FMA(ld<T8>(d + 8 * k), splat8(c[k]), acc[0]); }   // the 2-lane step: acc0.lo.lo -> p0
    }

    // Interpolated sample at position x. Tap k reads input[fx-ord+k]; negative indices come from the history
    // (p[j+ord2] for j in [-ord2,-1]); anything outside both contributes zero.
    __inline T sr_intp(double x, const T *__restrict input, int n_input, T1 gain) const
    {
        const int    fx       = (int)floor(x);
        const double fraction = x - fx;                                      // [0,1)
        const int    ord = tbl.order, ord2 = ord << 1;
        const int    dx = (int)(fraction * tbl.precision + 0.5) * ord2;
        if (dx < 0 || dx >= tbl.sz) return T{0};
        const T1    *coeffs = &tbl.kernel[dx];
        const int    first  = fx - ord;                                      // input index of tap 0

        if constexpr (TN == 8) {                                             // 8-lane frames: accumulate8 (the float2 path's partials and tree)
            T8 acc[8];
            for (int q = 0; q < 8; q++) acc[q] = 0;
            if (first >= 0 && first + ord2 <= n_input) {
                accumulate8(acc, (const T1 *)(input + first), coeffs, ord2);
            } else {
                const int h0 = std::max(0, -first - ord2), h1 = std::min(ord2, -first);
                if (h1 > h0) accumulate8(acc, (const T1 *)(p + (first + ord2 + h0)), coeffs + h0, h1 - h0);
                const int i0 = std::max(0, -first), i1 = std::min(ord2, n_input - first);
                if (i1 > i0) accumulate8(acc, (const T1 *)(input + (first + i0)), coeffs + i0, i1 - i0);
            }
            return (((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]))) * gain;   // = fold(acc0 + acc1)
        } else {
        T8 acc0 = 0, acc1 = 0;
        if (first >= 0 && first + ord2 <= n_input) {                         // steady state: one contiguous run
            accumulate(acc0, acc1, (const T1 *)(input + first), coeffs, ord2);
        } else {                                                             // block edge: history run, then input run
            const int h0 = std::max(0, -first - ord2), h1 = std::min(ord2, -first);
            if (h1 > h0) accumulate(acc0, acc1, (const T1 *)(p + (first + ord2 + h0)), coeffs + h0, h1 - h0);
            const int i0 = std::max(0, -first), i1 = std::min(ord2, n_input - first);
            if (i1 > i0) accumulate(acc0, acc1, (const T1 *)(input + (first + i0)), coeffs + i0, i1 - i0);
        }
        return fold(acc0 + acc1) * gain;
        }
    }

    // Kernel table creation
    bool sr_create(int order, int precision, float cutoff, float rate)
    {
        bool same = false;
        int idx = 0;

        if (tbl.order == order && tbl.precision == precision && tbl.cutoff == cutoff && tbl.kernel != NULL) {
            same = true;
        }

        leftover = 0.0;
        if (p) memset(p, 0, tbl.order * 2 * sizeof(T));

        if (!same || !tbl.kernel) {
            if (tbl.kernel) {
#if SR_SINC_CACHE
                m_kref.reset();
#else
                callocT_free(tbl.kernel);
#endif
                tbl.kernel = NULL;
            }
            if (p) {
                callocT_free(p);
                p = NULL;
            }

            memset(&tbl, 0, sizeof(sinc_table));

            tbl.order = order;
            tbl.precision = precision;
            tbl.cutoff = cutoff;
            precompute_window(order * 2);

            p = callocT<T>(order * 2);
            if (!p) {
                LOG("SincResampler: Failed to allocate buffer");
                return false;
            }
            memset(p, 0, order * 2 * sizeof(T));

            tbl.sz = order * 2 * (precision + 1);
#if SR_SINC_CACHE
            if ((m_kref = kGet(order, precision, cutoff, _kaiser_beta))) {   // built before with these settings: the same table
                tbl.kernel = m_kref.get();
                return true;
            }
#endif

            T1 *kernel = callocT<T1>(tbl.sz);
            if (!kernel) {
                LOG("SincResampler: Failed to allocate kernel table of size %d", tbl.sz);
                callocT_free(p);
                p = NULL;
                return false;
            }

            tbl.kernel = kernel;

            for (int offset = 0; offset <= precision; offset++)
            {
                T1 scale = 0.0;
                int startidx = idx;
                for (int i = -order + 1; i <= order; i++) {
                    T1 xk = kernel[idx] = kernel_func((T1) offset / (T1)precision - i, cutoff, order);
                    scale += xk;
                    idx++;
                }
                idx = startidx;
                T1 one_scale = 1./scale;
                for (int i = -order + 1; i <= order; i++)
                    kernel[idx++] *= one_scale;
            }
#if SR_SINC_CACHE
            m_kref = kPut(order, precision, cutoff, _kaiser_beta, std::shared_ptr<T1>(kernel, [](T1 *q) { callocT_free(q); }));
            tbl.kernel = m_kref.get();             // a racing build of the same key won: its identical table, ours is freed
#endif
        }
        return true;
    }
};

