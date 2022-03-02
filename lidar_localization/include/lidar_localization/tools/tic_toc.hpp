#ifndef TIC_TOC_HPP_
#define TIC_TOC_HPP_

#include <iostream>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <string>

using namespace std;

class TicToc
{
private:
    std::chrono::time_point<std::chrono::system_clock> start,end;
public:
    TicToc()
    {
        tic();
    }
    void tic()
    {
        start =  std::chrono::system_clock::now();
    }
    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> dur = end- start; // s
        return dur.count()*1000; // ms
    }
    double toc(std::string thing)
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> dur = end- start; // s
        printf("%s consume %f milli_seconds\n",thing.c_str(),dur.count()*1000);
        return dur.count()*1000; // ms
    }
};
#endif