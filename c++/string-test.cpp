#include<iostream>
#include"string.h"

using namespace std;

int main(){
    String s1;
    String s2("Hello world");
    s1 = s2;
    String s3(s2);

    cout << s1 << endl;

    return 0;
}
