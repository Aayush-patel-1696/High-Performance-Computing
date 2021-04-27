#include <iostream>
#include <cmath>
#include <random>
#include <omp.h>
#include <iomanip>
#include <ctime>
using namespace std;
std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(0,1);

int main(){
    long long N=1;
    double a;
    long long count;
    clock_t time_req;
    int threads;
    time_req = clock();
    for (int j=0;j<=5;j++){
        omp_set_num_threads(2);
        //#pragma omp threadprivate(val)
        count=0;
        time_t start1;
        #pragma omp parallel for reduction(+:count)
        for (int i=0;i<=N;i++){
            double m = distribution(generator);
            double n= distribution(generator);
            //cout<< m << endl <<n<<endl;
            if (n<cos(m*(M_PI)-(M_PI/2))){
            count=count+1;
            }
            threads=omp_get_num_threads();
    
        }
        //cout << threads<< endl;
        //time_t total_time;
        
        //a = a+total_time;
        
        if (j==7) break;
    }
    time_req =clock()- time_req;
    cout<<count<<endl;
    //cout<<N<<endl;
    cout<< ((float)time_req/(CLOCKS_PER_SEC))/(5) <<endl;
    double fraction= double(count)/long(N);
    cout<<threads<<endl;
    cout << "value of integral of cosx using montecarlo is " << setprecision(10)<<fraction*M_PI<<endl;
    cout<< "relative error "<<((fraction*M_PI-2)/2)*100<<endl;

}


//g++ -Xpreprocessor -fopenmp monte_carlo.cpp -o monte -lomp
