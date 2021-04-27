/*------------------------------------------------------------------------------------------------------------------------------------
|This code is written for matrix multiplication and upp triangular matrix manupulation using shared memory parallelisation(Openmp)    |
|The matrix multilication C=A*B, in which the A matrix is divided among cores and row wise matrix decomposition is done for matrix A *|
|The upper triangular matrix is parallized only for the last loop and it is decomposed in column wise form.                           |
|In this version all cores are doing there core operations including master thread(i.e master thread does not only do send recieve but| 
| also do matrix multiplication and triangulation)                                                                                    |
|//written by: Aayush, Date: 5-04-2021                                                                                                |
-------------------------------------------------------------------------------------------------------------------------------------*/                                                                                   

#include <cmath>
#include <random>
#include<iomanip>
#include <omp.h>
#include <iostream>
#include <random>
#include <chrono>

using namespace std::chrono;
using namespace std;
double** intialize_matrix(int ,int );
void  print_matrix(double**,int,int);
double** multiply_matrix(double** ,double**,double**,int,int );
double** upptriangular_matrix(double**,int,int);

int main(){
    int N=1000;
    int N_row= N;
    int N_col=N;
    double** A;
    double** B;
    double** C;
    omp_set_num_threads(1);
    //cout.precision(10);        //set precision
    //cout.setf(ios::fixed);

    //initialization
    A=intialize_matrix(N_row,N_col);
    B=intialize_matrix(N_row,N_col);
    C=intialize_matrix(N_row,N_col);
    //print_matrix(C,N_row,N_col);

    
    auto start1=high_resolution_clock::now();
    for(int rep=0;rep<5;rep++){
        C=multiply_matrix(A,B,C,N_row,N_col);
    }
    auto stop1=high_resolution_clock::now();
    auto duration1=duration_cast<milliseconds>(stop1-start1);
    auto start2=high_resolution_clock::now();
    for(int rep=0;rep<5;rep++){
        C=upptriangular_matrix(C,N_row,N_col);
    }
    
    //print_matrix(C,N_row,N_col);
    auto stop2=high_resolution_clock::now();
    auto duration2=duration_cast<milliseconds>(stop2-start2);
    cout<<"average time to compute multiplication for 5 cylcles: "<<duration1.count()/5<<endl;
    cout<<"average time for upper triangular for 5 cylces: "<<duration2.count()/5<<endl;


}
double** intialize_matrix(int N_row ,int N_col){
        double** matrix= new double*[N_row];
        for (int i=0;i<N_row;i++){
            matrix[i]=new double[N_col];
        }
        for (int i=0;i<N_row;i++){
            for (int j=0;j<N_col;j++){
                matrix[i][j]=rand()/1000000000+1;;
            }
        }
        return matrix;
}
 void print_matrix(double** matrix,int N_row,int N_col){
      for (int i=0;i<N_row;i++){            //print C matrix {
        for (int j=0;j<N_col;j++){
            cout<<matrix[i][j]<<setw(8)<<" ";  
        }
        cout<<"\n";
      }
    }   
double** multiply_matrix(double** A,double** B,double** C,int N_row,int N_col){
    int i,j,k;
    #pragma omp parallel for private(i,j,k) 
    for (int i=0;i<N_row;i++){
        for(int j=0;j<N_col;j++){
            for (int k=0;k<N_row;k++){
                C[i][j]+=A[i][k]*B[k][j];
             }
        }
    }  
    return C;
}

double** upptriangular_matrix(double** M,int N_row,int N_col){
    for (int k=0;k<N_row-1;k++){
    #pragma omp parallel for collapse(2)
        for (int i=k+1;i<N_row;i++){
            double M_old=M[i][k];
            int j;
            for (int j=0;j<N_col;j++){
                M[i][j]=M[i][j]-(M_old*M[k][j]/M[k][k]);
            }

        }
    }
    return M;
}

//g++ -Xpreprocessor -fopenmp main2.cpp -o main2 -lomp
