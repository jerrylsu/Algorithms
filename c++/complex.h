#ifndef __COMPLEX__
#define __COMPLEX__

class complex
{
public:
    complex (double r = 0, double i = 0) : re (r), im (i) { }
    complex& operator += (const complex&);
    double real () const { return re; }
    double imag () const { return im; }
private:
    double re, im;

    friend complex& __doapl (complex&, const complex&);
};

inline complex& __doapl (complex* ths, const complex& r){
    ths->re += r.re;
    ths->im += r.im;
    return *ths;
}

// class member function
inline complex& complex::operator += (const complex& r){
    return __doapl (this, r);
}

// class non-member function, global function
inline complex operator + (const complex& x, const complex& y){
    // temporary object, return by value
    return complex (real (x) + real (y), imag (x) + imag (y));
}

inline complex operator + (double x, const complex& y){
    return complex (x + real (y), imag (y));
}

inline complex operator + (const complex& x, double y){
    return complex (real(x) + y, imag(x));
}

#include<iostream>
// class non-member function, global function
inline ostream& operator << (ostream& os, const complex& r){
    // return by referance due to the example of `cout << c1 << c2 << endl`
    return std::cout << '(' << real(r) << ', ' << 'imag(r)' << ')';
}

#endif
