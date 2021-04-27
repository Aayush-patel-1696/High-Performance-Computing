/*----------------------------------------------------------------------------------------------------------------------------------
-This code is written for matrix multiplication and upp triangular matrix manupulation using disrtibuted memory parallelisation (MPI)
-The matrix multilication C=A*B, in which the A matrix is divided among cores and row wise matrix decomposition is done for matrix A *
-The upper triangular matrix is parallized only for the last loop and it is decomposed in column wise form.
- In this version all cores are doing there core operations including master thread(i.e master thread does not only do send recieve but 
  also do matrix multiplication and triangulation)
-Remendior loop extension is not written but can be extended 
  //written by: Aayush, Date: 5-04-2021
-------------------------------------------------------------------------------------------------------------------------------------*/                                                                                   


#include <cmath>
#include <random>
#include<iomanip>
#include <omp.h>
#include <iostream>
#include <random>
#include "mpi.h" 

int pid;
double* intialize_matrix(int ,int );
double* intialize_matrix_zero(int ,int);
void  print_matrix(double*,int,int);
double* multiply_matrix(double* ,double*,double*,int,int,int);
double* multiply_matrix_rem(double* ,double* ,double* ,int ,int ,int ,int);
double* upptriangular_matrix(double*,int,int,int);
void func1(double ,int );
int main(int argc,char** argv){
    int N=1500;
    int N_row= N;
    int N_col=N;
    double temp;
    double* A;
    double* B;
    double* C;
    double* D;
    double* M;

    //intialization
    A=intialize_matrix(N_row,N_col);
    B=intialize_matrix(N_row,N_col);
    C=intialize_matrix(N_row,N_col);
    D=intialize_matrix_zero(N_row,N_col);
    M=intialize_matrix_zero(N_row,N_col);
    int my_PE_num;          //process id
    int N_cpu=2;            //number of CPUs
    int N_ps=N_row/int(N_cpu);
    int rem=(N_row)%(N_cpu);
    MPI_Status status;
    //MPI_Comm comm;

    //starting mpi process
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_PE_num);
    double start1=MPI_Wtime();

    if (my_PE_num!=0){
        if (rem!=0){
            if(my_PE_num==(N_cpu-1)){
                multiply_matrix_rem(C,A,B,N_ps,N_col,my_PE_num,rem);
            //print_matrix(C,N_row,N_col);
                MPI_Send( &C[(my_PE_num)*N_ps*N_col],(N_ps+rem)*N_col,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
            //MPI_Send( &C[0],N_col,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
            }
            else{
                multiply_matrix(C,A,B,N_ps,N_col,my_PE_num);
                MPI_Send( &C[(my_PE_num)*N_ps*N_col],(N_ps)*N_col,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
            }
        }
        else{
            multiply_matrix(C,A,B,N_ps,N_col,my_PE_num);
            MPI_Send( &C[(my_PE_num)*N_ps*N_col],(N_ps)*N_col,MPI_DOUBLE,0,0,MPI_COMM_WORLD);

        }
    
    }
    else{
       
        multiply_matrix(C,A,B,N_ps,N_col,my_PE_num);
        //print_matrix(C,N_row,N_col);
        for(int j=1;j<N_cpu;j++){
            if (j==(N_cpu-1) & rem!=0){
                MPI_Recv( &C[(j)*N_ps*N_col],(N_ps+rem)*N_col,MPI_DOUBLE,j,0,MPI_COMM_WORLD,&status);
        
            }
            else{
                MPI_Recv( &C[(j)*N_ps*N_col],(N_ps)*N_col,MPI_DOUBLE,j,0,MPI_COMM_WORLD,&status);
            }
        }
       }
    int N_colps=N_ps;


    
    MPI_Bcast(&C[0],N_row*N_col,MPI_DOUBLE,0,MPI_COMM_WORLD);
        
    /*if (my_PE_num==1){
       print_matrix(C,N_row,N_col);
    }*/
    
    double end1=MPI_Wtime();
    std::cout<<"Time taken for multiplication by the process id "<<my_PE_num <<"is"<<(end1-start1)<<std::endl;
    double start2=MPI_Wtime();
   /* if (my_PE_num==(N_cpu-1) & rem!=0){
        int id=my_PE_num;
        
        for(int k=0;k<N_row-1;k++){
            MPI_Barrier(MPI_COMM_WORLD);
            if(k>=(id*N_colps)& k<(id+1)*N_colps+rem){
                    MPI_Bcast(&C[k*N_col+k],1,MPI_DOUBLE,id,MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            for(int i=k+1;i<N_row;i++){
                if(k>=(id*N_colps)& k<(id+1)*N_colps+rem){
                    double C_old=C[i*N_col+k];
                    MPI_Bcast(&C_old,1,MPI_DOUBLE,id,MPI_COMM_WORLD);
                }
                for (int j=(id*N_colps);j<((id+1)*N_colps+rem);j++){
                    C[i*N_col+j]=C[i*N_col+j]-(C_old*C[k*N_col+j]/C[k*N_col+k]);
                    D[i*N_col+j]=C[i*N_col+j];
                }
                
            }

        }
    }*/
   
    int id=my_PE_num;
    double thread;
    for(int k=0;k<N_row-1;k++){
        if(k>=(id*N_colps)& k<(id+1)*N_colps){
                
                for(int p=0;p<N_cpu;p++){
                    if(p!=id){
                        //std::cout<<"send id is "<<id<<std::endl;
                        //MPI_Send(&C_old,1,MPI_DOUBLE,p,10,MPI_COMM_WORLD);
                        MPI_Send(&C[k*N_col+k],1,MPI_DOUBLE,p,20,MPI_COMM_WORLD);
                    }
                }
            }
            else{
                //std::cout<<"recieve id is "<<id<<std::endl;
               // MPI_Recv(&C_old,1,MPI_DOUBLE,MPI_ANY_SOURCE,10,MPI_COMM_WORLD,&status);
                MPI_Recv(&C[k*N_col+k],1,MPI_DOUBLE,MPI_ANY_SOURCE,20,MPI_COMM_WORLD,&status);
            }
        for(int i=k+1;i<N_row;i++){
            double C_old;
            if(k>=(id*N_colps)& k<(id+1)*N_colps){
                C_old=C[i*N_col+k];
                
                for(int p=0;p<N_cpu;p++){
                    if(p!=id){
                        //std::cout<<"send id is "<<id<<std::endl;
                        MPI_Send(&C_old,1,MPI_DOUBLE,p,10,MPI_COMM_WORLD);
                        //MPI_Send(&C[k*N_col+k],1,MPI_DOUBLE,p,20,MPI_COMM_WORLD);
                    }
                }
            }
            else{
                //std::cout<<"recieve id is "<<id<<std::endl;
                MPI_Recv(&C_old,1,MPI_DOUBLE,MPI_ANY_SOURCE,10,MPI_COMM_WORLD,&status);
               // MPI_Recv(&C[k*N_col+k],1,MPI_DOUBLE,MPI_ANY_SOURCE,20,MPI_COMM_WORLD,&status);
            }
            
            for (int j=(id*N_colps);j<((id+1)*N_colps);j++){
                    C[i*N_col+j]=C[i*N_col+j]-(C_old*C[k*N_col+j]/C[k*N_col+k]);
                    D[i*N_col+j]=C[i*N_col+j];
            }
                
        }

    }


   MPI_Reduce(D,M,N_row*N_col,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    if(my_PE_num==0){
        
        for (int i=0;i<N_col;i++){
            M[i]=C[i];
        }
       //print_matrix(M,N_row,N_col);

    }
    
    double end2=MPI_Wtime();
    
   std::cout<<"Time taken for upptirangular the process id "<<my_PE_num <<"is"<<(end2-start2)<<std::endl;
   MPI_Finalize();
    





}
double* multiply_matrix(double* C,double* A,double* B,int N_ps,int N_col,int id){
     for ( int i=(id)*N_ps;i<(id+1)*N_ps;i++){
        for(int j=0;j<N_col;j++){
            for(int k=0;k<N_col;k++){
                C[i*N_col+j]+=A[i*N_col+k]*B[k*N_col+j];
            }

        }
     }
    return C;
}

double* intialize_matrix(int N_row ,int N_col){
        double* matrix =new double[N_row*N_col];
        for (int i=0;i<N_row*N_col;i++){
            matrix[i]=rand()/1000000000+1;
        }
        return matrix;
} 
double* intialize_matrix_zero(int N_row ,int N_col){
        double* matrix =new double[N_row*N_col];
        for (int i=0;i<N_row*N_col;i++){
            matrix[i]=0;
        }
        return matrix;
} 

void print_matrix(double* matrix,int N_row,int N_col){
      for (int i=0;i<N_row;i++){            //print C matrix {
        for (int j=0;j<N_col;j++){
            std::cout<<matrix[i*N_col+j]<<std::setw(4)<<" ";  
        }
        std::cout<<"\n";
      }
    }   



    double* multiply_matrix_rem(double* C,double* A,double* B,int N_ps,int N_col,int id,int rem){
     for ( int i=(id)*N_ps;i<(id+1)*N_ps+rem;i++){
        for(int j=0;j<N_col;j++){
            for(int k=0;k<N_col;k++){
                C[i*N_col+j]+=A[i*N_col+k]*B[k*N_col+j];
            }

        }
    }
    return C;
}
   // mpic++ -o mpiupp6 main_mpi_upp_v6.cpp
   //mpirun -np 2 ./mpiupp6
