#include<iostream>
#include"complex.h"

using namespace std;

int main(){
    complex n1(1.0, 2.3);
    complex n2(1.0, 2.3);
    n1 += n2;  
    std::cout << n1.real() << ' ' << n1.imag() << std::endl;
}
