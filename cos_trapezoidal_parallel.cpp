#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <ctime>
#include <iomanip>
using namespace std;

int main(){
    
    double upp_limit=(M_PI/2),low_limit=-(M_PI/2);
    long long N=10000000;
    
    
    int threads;
    double h= M_PI/(N-1);
    double func_sum=0;
    //cout << M_PI << endl;
    
    //#pragma omp threadprivate(val)
    clock_t time_req;
    time_req = clock();
    for (int j=0;j<=5;j++){
        
        func_sum=0;
        
        long long i=0;
        omp_set_num_threads(8);
        #pragma omp parallel for reduction(+:func_sum)
        for (i=0;i<=N-1;i++){
        
            if (i==0 || i==N-1){
                func_sum += cos(low_limit+((upp_limit-low_limit)*i)/(N-1));
            }
            else{
        
                func_sum+= 2*cos(low_limit+((upp_limit-low_limit)*i)/(N-1));
            }
            
        
        }
        
    }
    threads=omp_get_num_threads();
    double integral_val=0;
    integral_val=(func_sum*h)/2;
    time_req =clock()- time_req;
    cout << "integral value is " <<setprecision(10)<<integral_val<<endl;
    cout<< "relative error "<<((integral_val-2)/2)*100<<endl;
    cout<< ((float)time_req/(CLOCKS_PER_SEC))/(5*threads) <<endl;

}
//g++ -Xpreprocessor -fopenmp cos_trapezoidal_2.cpp -o cos2 -lomp

