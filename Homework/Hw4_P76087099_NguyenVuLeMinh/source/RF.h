#ifndef RFILE
#define RFILE
#include <bits/stdc++.h>
#include "DS.h"
using namespace std;

class ReadFile{
public:
    ReadFile();
    ReadFile(Cells *cells, Nets *nets){
        this->cells = cells;
        this->nets = nets;
    };
    ~ReadFile();
    void read_file(string node_file, string net_file);
    string parse(string &);
    Cells *cells;
    Nets  *nets;
    string net_name; 
};
#endif
