#ifndef __MYSTRING__
#define __MYSTRING__

#include<cstring>

class String{
public:
    // Big Three
    String(const char* cstr);             // constructor func
    String(const String& str);            // copy constructor func
    String& operator=(const String& str); // copy asignment func
    ~String();                            // destructor func
    char* get_c_str() const { return m_data; }
private:
    char* m_data;
};

inline String::String(const char* cstr = 0){
    if (cstr){
        m_data = new char[strlen(cstr) + 1];
        strcpy(m_data, cstr);
    }
    else{
        m_data = new char[1];
        *m_data = '\0';
    }
}


inline String::String(const String& str){
    m_data = new char[strlen(str.m_data) + 1];
    strcpy(m_data, str.m_data);  // deep copy
}

inline String& String::operator=(const String& str){
    // checking self-assignment
    if (this == &str){
        return *this;
    }

    delete[] m_data;  // preventing memory leak
    m_data = new char[strlen(str.m_data) + 1];
    strcpy(m_data, str.m_data);
    return *this;
}

inline String::~String(){
    delete[] m_data;
}

#include<iostream>
using namespace std;

ostream& operator<<(ostream& os, const String& str){
    os << str.get_c_str();
    return os;
}

#endif
