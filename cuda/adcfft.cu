// CUDA spectrum + envelope engine for the AD9643 capture path.
//  - u16 (14-bit, right-aligned) -> float, DC removed, Hann windowed
//  - batched cuFFT R2C, power averaged across frames (Welch-style)
//  - min/max envelope decimation for the time-domain trace
//  - per-stage timing via CUDA events
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>

#define FS_CODES 8192.0f          // 14-bit full-scale amplitude

// The converter delivers 14-bit TWO'S COMPLEMENT, right-aligned in a 16-bit
// word -- confirmed by the vendor's own client, which sign-extends with
// ((int16_t)(x<<2))>>2 and scales by 1.75 V / 8192. Reading it as unsigned
// makes a signal sitting near zero jump between ~0 and ~16383 whenever it
// crosses zero, which looks like violent spikes and wrecks the spectrum.
__device__ __forceinline__ float s14(unsigned short v){
    int i = (int)v;
    return (float)((i ^ 0x2000) - 0x2000);
}
#define MAXB 256

struct FftCtx {
    int nfft, maxs, tw, nbins;
    unsigned short *h_in;         // pinned
    unsigned short *d_raw;
    float *d_f, *d_win, *d_pow, *d_spec;
    float *d_part;                // partial reductions
    float *d_tmin, *d_tmax, *d_tmean;
    cufftComplex *d_c;
    int max_frames;
    int win_type;
    float win_sum, win_enbw;    // coherent gain and equiv. noise bandwidth (bins)
    cufftHandle plan;
    int plan_frames;
    cudaStream_t s;
    cudaEvent_t e0,e1,e2,e3,e4;
};

// Cosine-sum windows: w[n] = sum_k (-1)^k a_k cos(2*pi*k*n/(N-1)).
// Hann's -31.5 dB sidelobes let a strong tone's leakage skirt masquerade as
// spurs, which is fatal for peak classification; Blackman-Harris trades
// resolution for -92 dB sidelobes, flat-top trades both for +-0.01 dB
// amplitude accuracy. Coefficients per harris, Proc. IEEE 66(1), 1978.
#define WIN_HANN 0
#define WIN_BH4  1
#define WIN_FLAT 2
#define WIN_RECT 3

__constant__ float c_wcoef[5];
__constant__ int   c_wterms;

__global__ void k_window(float*w,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float x=2.0f*M_PI*i/(n-1), v=0.0f, sgn=1.0f;
    for(int k=0;k<c_wterms;k++){ v+=sgn*c_wcoef[k]*cosf(k*x); sgn=-sgn; }
    w[i]=v;
}

// host-side coefficient table + analytic sums (exact to O(1) in N)
static int win_coeffs(int type,float*a){
    switch(type){
      case WIN_BH4:  a[0]=0.35875f;a[1]=0.48829f;a[2]=0.14128f;a[3]=0.01168f; return 4;
      case WIN_FLAT: a[0]=0.21557895f;a[1]=0.41663158f;a[2]=0.277263158f;
                     a[3]=0.083578947f;a[4]=0.006947368f; return 5;
      case WIN_RECT: a[0]=1.0f; return 1;
      case WIN_HANN:
      default:       a[0]=0.5f;a[1]=0.5f; return 2;
    }
}

// pass 1: partial sum / sumsq / min / max
__global__ void k_stats(const unsigned short*x,int n,float*part){
    __shared__ float ss[MAXB],sq[MAXB],mn[MAXB],mx[MAXB];
    int t=threadIdx.x, i=blockIdx.x*blockDim.x+t, st=gridDim.x*blockDim.x;
    float s=0,q=0,a=1e30f,b=-1e30f;
    for(int j=i;j<n;j+=st){ float v=s14(x[j]); s+=v; q+=v*v; a=fminf(a,v); b=fmaxf(b,v); }
    ss[t]=s; sq[t]=q; mn[t]=a; mx[t]=b; __syncthreads();
    for(int d=blockDim.x/2; d>0; d>>=1){
        if(t<d){ ss[t]+=ss[t+d]; sq[t]+=sq[t+d];
                 mn[t]=fminf(mn[t],mn[t+d]); mx[t]=fmaxf(mx[t],mx[t+d]); }
        __syncthreads();
    }
    if(t==0){ part[blockIdx.x]=ss[0]; part[gridDim.x+blockIdx.x]=sq[0];
              part[2*gridDim.x+blockIdx.x]=mn[0]; part[3*gridDim.x+blockIdx.x]=mx[0]; }
}
__global__ void k_stats_fin(float*part,int nb,float*out,int n){
    float s=0,q=0,a=1e30f,b=-1e30f;
    for(int i=0;i<nb;i++){ s+=part[i]; q+=part[nb+i];
        a=fminf(a,part[2*nb+i]); b=fmaxf(b,part[3*nb+i]); }
    float m=s/n; float var=q/n-m*m; if(var<0) var=0;
    out[0]=a; out[1]=b; out[2]=m; out[3]=sqrtf(var); out[4]=sqrtf(var);
}

// convert + DC-remove + window, framed for the batched FFT
__global__ void k_win(const unsigned short*x,float*y,const float*w,
                      int nfft,int frames,const float*stats){
    int i=blockIdx.x*blockDim.x+threadIdx.x, tot=nfft*frames;
    if(i<tot) y[i]=(s14(x[i])-stats[2])*w[i%nfft];   // stats[2] = mean
}

// average |X|^2 across frames
__global__ void k_pow(const cufftComplex*c,float*p,int nbins,int frames){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nbins) return;
    float acc=0;
    for(int f=0;f<frames;f++){ cufftComplex v=c[(size_t)f*nbins+i]; acc+=v.x*v.x+v.y*v.y; }
    p[i]=acc/frames;
}
__global__ void k_db(const float*p,float*db,int nbins,float norm){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<nbins){ float a=sqrtf(p[i])*norm; db[i]=20.0f*log10f(fmaxf(a,1e-9f)/FS_CODES); }
}

// min/max envelope decimation: one block per output column
// min/max/mean envelope decimation: one block per output column. The mean
// matters: the drawn centre line used to be (min+max)/2, the midpoint of the
// envelope, which for asymmetric noise or an offset signal sits somewhere no
// sample actually is.
__global__ void k_env(const unsigned short*x,int n,float*mn,float*mx,float*mv,int tw){
    int col=blockIdx.x; if(col>=tw) return;
    long a=(long)n*col/tw, b=(long)n*(col+1)/tw; if(b<=a) b=a+1;
    __shared__ float sm[MAXB],sx[MAXB],ss[MAXB];
    float lo=1e30f,hi=-1e30f,sum=0.0f;
    for(long i=a+threadIdx.x;i<b;i+=blockDim.x){ float v=s14(x[i]);
        lo=fminf(lo,v); hi=fmaxf(hi,v); sum+=v; }
    sm[threadIdx.x]=lo; sx[threadIdx.x]=hi; ss[threadIdx.x]=sum; __syncthreads();
    for(int d=blockDim.x/2;d>0;d>>=1){
        if(threadIdx.x<d){ sm[threadIdx.x]=fminf(sm[threadIdx.x],sm[threadIdx.x+d]);
                           sx[threadIdx.x]=fmaxf(sx[threadIdx.x],sx[threadIdx.x+d]);
                           ss[threadIdx.x]+=ss[threadIdx.x+d]; }
        __syncthreads();
    }
    if(threadIdx.x==0){ mn[col]=sm[0]; mx[col]=sx[0]; mv[col]=ss[0]/(float)(b-a); }
}

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return NULL; } }while(0)

extern "C" {

static int apply_window(FftCtx*c,int type){
    float a[5]={0,0,0,0,0};
    int nt=win_coeffs(type,a);
    if(cudaMemcpyToSymbol(c_wcoef,a,sizeof(a))!=cudaSuccess) return -1;
    if(cudaMemcpyToSymbol(c_wterms,&nt,sizeof(int))!=cudaSuccess) return -1;
    k_window<<<(c->nfft+255)/256,256,0,c->s>>>(c->d_win,c->nfft);
    if(cudaStreamSynchronize(c->s)!=cudaSuccess) return -1;
    double s1=a[0], s2=a[0]*(double)a[0];
    for(int k=1;k<nt;k++) s2+=0.5*a[k]*(double)a[k];
    c->win_type=type;
    c->win_sum=(float)(s1*c->nfft);                 // sum(w)
    c->win_enbw=(float)(s2/(s1*s1));                // ENBW in bins
    return 0;
}

extern "C" int adc_set_window(FftCtx*c,int type){ return apply_window(c,type); }
extern "C" float adc_win_enbw(FftCtx*c){ return c->win_enbw; }

FftCtx* adc_create(int nfft,int maxs,int tw){
    FftCtx*c=(FftCtx*)calloc(1,sizeof(FftCtx));
    c->nfft=nfft; c->maxs=maxs; c->tw=tw; c->nbins=nfft/2+1; c->plan_frames=0;
    c->max_frames=maxs/nfft; if(c->max_frames>512)c->max_frames=512; if(c->max_frames<1)c->max_frames=1;
    CK(cudaHostAlloc((void**)&c->h_in,(size_t)maxs*2,cudaHostAllocPortable));
    CK(cudaMalloc(&c->d_raw,(size_t)maxs*2));
    CK(cudaMalloc(&c->d_f,(size_t)maxs*sizeof(float)));
    CK(cudaMalloc(&c->d_win,(size_t)nfft*sizeof(float)));
    CK(cudaMalloc(&c->d_pow,(size_t)c->nbins*sizeof(float)));
    CK(cudaMalloc(&c->d_spec,(size_t)c->nbins*sizeof(float)));
    CK(cudaMalloc(&c->d_part,(size_t)4*MAXB*sizeof(float)+8*sizeof(float)));
    CK(cudaMalloc(&c->d_tmin,(size_t)tw*sizeof(float)));
    CK(cudaMalloc(&c->d_tmax,(size_t)tw*sizeof(float)));
    CK(cudaMalloc(&c->d_tmean,(size_t)tw*sizeof(float)));
    CK(cudaMalloc(&c->d_c,(size_t)c->max_frames*c->nbins*sizeof(cufftComplex)));
    CK(cudaStreamCreate(&c->s));
    CK(cudaEventCreate(&c->e0)); CK(cudaEventCreate(&c->e1)); CK(cudaEventCreate(&c->e2));
    CK(cudaEventCreate(&c->e3)); CK(cudaEventCreate(&c->e4));
    if(apply_window(c,WIN_HANN)!=0) return NULL;
    return c;
}

unsigned short* adc_hostbuf(FftCtx*c){ return c->h_in; }
int adc_nbins(FftCtx*c){ return c->nbins; }
int adc_maxframes(FftCtx*c){ return c->max_frames; }

int adc_process(FftCtx*c,int nsamples,int max_frames,int trace_n,
                float*spec,float*tmin,float*tmax,float*tmean,
                float*stats,float*times,int*nframes_out){
    int nfft=c->nfft, frames=nsamples/nfft;
    if(trace_n<=0||trace_n>nsamples) trace_n=nsamples;   // envelope span, decoupled from FFT
    if(frames<1) return -1;
    if(frames>max_frames) frames=max_frames;
    if(frames>c->max_frames) frames=c->max_frames;
    *nframes_out=frames;

    if(c->plan_frames!=frames){
        if(c->plan_frames) cufftDestroy(c->plan);
        if(cufftPlan1d(&c->plan,nfft,CUFFT_R2C,frames)!=CUFFT_SUCCESS) return -2;
        cufftSetStream(c->plan,c->s);
        c->plan_frames=frames;
    }
    cufftComplex* d_c=c->d_c;   // preallocated; no malloc in the hot path

    cudaEventRecord(c->e0,c->s);
    cudaMemcpyAsync(c->d_raw,c->h_in,(size_t)nsamples*2,cudaMemcpyHostToDevice,c->s);
    cudaEventRecord(c->e1,c->s);

    int nb=64;
    k_stats<<<nb,MAXB,0,c->s>>>(c->d_raw,nsamples,c->d_part);
    k_stats_fin<<<1,1,0,c->s>>>(c->d_part,nb,c->d_part+4*MAXB,nsamples);

    int tot=nfft*frames;
    // mean is read on-device (stats[2]) - no host round-trip, no mid-pipeline
    // cudaStreamSynchronize stalling the stream every frame
    k_win<<<(tot+255)/256,256,0,c->s>>>(c->d_raw,c->d_f,c->d_win,nfft,frames,
                                        c->d_part+4*MAXB);
    k_env<<<c->tw,MAXB,0,c->s>>>(c->d_raw,trace_n,c->d_tmin,c->d_tmax,
                                 c->d_tmean,c->tw);
    cudaEventRecord(c->e2,c->s);

    cufftExecR2C(c->plan,c->d_f,d_c);
    cudaEventRecord(c->e3,c->s);

    k_pow<<<(c->nbins+255)/256,256,0,c->s>>>(d_c,c->d_pow,c->nbins,frames);
    // amplitude = 2*|X|/sum(w) -- coherent gain depends on the window
    k_db<<<(c->nbins+255)/256,256,0,c->s>>>(c->d_pow,c->d_spec,c->nbins,
                                            2.0f/c->win_sum);
    cudaMemcpyAsync(spec,c->d_spec,(size_t)c->nbins*sizeof(float),cudaMemcpyDeviceToHost,c->s);
    cudaMemcpyAsync(tmin,c->d_tmin,(size_t)c->tw*sizeof(float),cudaMemcpyDeviceToHost,c->s);
    cudaMemcpyAsync(tmax,c->d_tmax,(size_t)c->tw*sizeof(float),cudaMemcpyDeviceToHost,c->s);
    cudaMemcpyAsync(tmean,c->d_tmean,(size_t)c->tw*sizeof(float),cudaMemcpyDeviceToHost,c->s);
    float h_stats[5];
    cudaMemcpyAsync(h_stats,c->d_part+4*MAXB,5*sizeof(float),cudaMemcpyDeviceToHost,c->s);
    cudaEventRecord(c->e4,c->s);
    cudaStreamSynchronize(c->s);

    for(int i=0;i<5;i++) stats[i]=h_stats[i];
    float t;
    cudaEventElapsedTime(&t,c->e0,c->e1); times[0]=t;   // H2D
    cudaEventElapsedTime(&t,c->e1,c->e2); times[1]=t;   // window+env kernels
    cudaEventElapsedTime(&t,c->e2,c->e3); times[2]=t;   // cuFFT
    cudaEventElapsedTime(&t,c->e3,c->e4); times[3]=t;   // power+dB+D2H
    cudaEventElapsedTime(&t,c->e0,c->e4); times[4]=t;   // total GPU
    return 0;
}

void adc_destroy(FftCtx*c){
    if(!c) return;
    if(c->plan_frames) cufftDestroy(c->plan);
    cudaFreeHost(c->h_in); cudaFree(c->d_raw); cudaFree(c->d_f); cudaFree(c->d_win);
    cudaFree(c->d_pow); cudaFree(c->d_spec); cudaFree(c->d_part);
    cudaFree(c->d_tmin); cudaFree(c->d_tmax); cudaFree(c->d_tmean); cudaFree(c->d_c);
    cudaStreamDestroy(c->s); free(c);
}

int adc_meminfo(size_t*freeb,size_t*totb){ return (int)cudaMemGetInfo(freeb,totb); }
}
