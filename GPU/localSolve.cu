// localSolve.cu (super simple Tet4 -> 12x12 Ke per element)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__
void build_D_3D(double E, double nu, double D[36]) {
    // isotropic 3D elasticity in Voigt order [xx,yy,zz,yz,zx,xy]
    double c = E / ((1.0 + nu) * (1.0 - 2.0*nu));
    double a = c * (1.0 - nu);
    double b = c * nu;
    double g = c * 0.5 * (1.0 - 2.0*nu);
    #pragma unroll
    for (int i=0;i<36;++i) D[i]=0.0;
    D[0]=a; D[1]=b; D[2]=b;
    D[6]=b; D[7]=a; D[8]=b;
    D[12]=b; D[13]=b; D[14]=a;
    D[21]=g; D[28]=g; D[35]=g;
}

__global__
void kernelKe_Tet4_3D(
    const double* __restrict__ X,
    const double* __restrict__ Y,
    const double* __restrict__ Z,
    const int*    __restrict__ conn,   // 4*nElem
    double E, double nu,
    double* __restrict__ elemKe,       // nElem*144
    int nElem)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nElem) return;

    // --- 1) connectivity & coords
    int n0 = conn[4*e+0], n1 = conn[4*e+1], n2 = conn[4*e+2], n3 = conn[4*e+3];
    double x0=X[n0], y0=Y[n0], z0=Z[n0];
    double x1=X[n1], y1=Y[n1], z1=Z[n1];
    double x2=X[n2], y2=Y[n2], z2=Z[n2];
    double x3=X[n3], y3=Y[n3], z3=Z[n3];

    // --- 2) Jacobian J = [x1-x0, x2-x0, x3-x0; ...]
    double j00=x1-x0, j01=x2-x0, j02=x3-x0;
    double j10=y1-y0, j11=y2-y0, j12=y3-y0;
    double j20=z1-z0, j21=z2-z0, j22=z3-z0;

    double detJ = j00*(j11*j22 - j12*j21) - j01*(j10*j22 - j12*j20) + j02*(j10*j21 - j11*j20);
    // Degenerate / inverted? write zeros and return.
    double* Ke = elemKe + e*144;
    if (detJ <= 0.0) { for (int i=0;i<144;++i) Ke[i]=0.0; return; }

    double invDet = 1.0 / detJ;
    double volume = detJ / 6.0;

    // --- 3) J^{-1} and J^{-T}
    double i00 =  (j11*j22 - j12*j21)*invDet;
    double i01 = -(j01*j22 - j02*j21)*invDet;
    double i02 =  (j01*j12 - j02*j11)*invDet;
    double i10 = -(j10*j22 - j12*j20)*invDet;
    double i11 =  (j00*j22 - j02*j20)*invDet;
    double i12 = -(j00*j12 - j02*j10)*invDet;
    double i20 =  (j10*j21 - j11*j20)*invDet;
    double i21 = -(j00*j21 - j01*j20)*invDet;
    double i22 =  (j00*j11 - j01*j10)*invDet;

    // J^{-T}
    double it00=i00, it01=i10, it02=i20;
    double it10=i01, it11=i11, it12=i21;
    double it20=i02, it21=i12, it22=i22;

    // --- 4) grad N in physical coords (constant for linear tet)
    // reference grads: N1[-1,-1,-1], N2[1,0,0], N3[0,1,0], N4[0,0,1]
    double dNr[4][3] = {{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
    double gx[4], gy[4], gz[4];
    #pragma unroll
    for (int a=0;a<4;++a) {
        double a0=dNr[a][0], a1=dNr[a][1], a2=dNr[a][2];
        gx[a] = it00*a0 + it01*a1 + it02*a2;
        gy[a] = it10*a0 + it11*a1 + it12*a2;
        gz[a] = it20*a0 + it21*a1 + it22*a2;
    }

    // --- 5) Build B (6x12)
    double B[72] = {0.0};
    #pragma unroll
    for (int a=0;a<4;++a) {
        int c = 3*a;
        double dNx=gx[a], dNy=gy[a], dNz=gz[a];
        B[0*12 + c+0] = dNx;            // exx
        B[1*12 + c+1] = dNy;            // eyy
        B[2*12 + c+2] = dNz;            // ezz
        B[3*12 + c+1] = dNz; B[3*12 + c+2] = dNy; // gamma_yz
        B[4*12 + c+0] = dNz; B[4*12 + c+2] = dNx; // gamma_zx
        B[5*12 + c+0] = dNy; B[5*12 + c+1] = dNx; // gamma_xy
    }

    // --- 6) D, DB, Ke = B^T D B * volume
    double D[36]; build_D_3D(E, nu, D);
    double DB[72];
    // DB = D(6x6) * B(6x12)
    #pragma unroll
    for (int i=0;i<6;++i){
        for (int j=0;j<12;++j){
            double acc=0.0;
            #pragma unroll
            for (int k=0;k<6;++k) acc += D[i*6+k]*B[k*12+j];
            DB[i*12+j]=acc;
        }
    }
    // Ke = B^T(12x6) * DB(6x12)
    #pragma unroll
    for (int i=0;i<144;++i) Ke[i]=0.0;
    #pragma unroll
    for (int i=0;i<12;++i){
        for (int j=0;j<12;++j){
            double acc=0.0;
            #pragma unroll
            for (int k=0;k<6;++k) acc += B[k*12+i]*DB[k*12+j];
            Ke[i*12+j] = acc * volume;
        }
    }
}

// -------- Host launcher (simple) --------
// Inputs are generated from mesh parsing
void launchLocalKe_Tet4_3D(
    const double* d_nodes_x,
    const double* d_nodes_y,
    const double* d_nodes_z,
    const int*    d_elem_conn,
    double E_uniform, double nu_uniform,
    double* d_elemKe,
    int nElem, cudaStream_t stream = 0)
{
    int block = 128;
    int grid  = (nElem + block - 1) / block;
    kernelKe_Tet4_3D<<<grid, block, 0, stream>>>(
        d_nodes_x, d_nodes_y, d_nodes_z,
        d_elem_conn,
        E_uniform, nu_uniform,
        d_elemKe,
        nElem);
}