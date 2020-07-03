#ifndef DS
#define DS
#include <bits/stdc++.h>
using namespace std;
class Cell;
class Net;

class Cell {
public:
    string node_name;
    string group;
    unsigned int area;
    int gain;
	bool locked;
    vector<Net*> NetList;
};

class Cells {
public:
    Cells(): NumNodes(0), NumTerminals(0), bound(0.0), Pmax(0), Max_Size(0), Total_Size(0){};
    ~Cells();
    int NumNodes;
    int NumTerminals;
    double bound;
	unsigned Pmax;
    unsigned int Max_Size;
    unsigned int Total_Size;
    map<string, Cell* > CellArray;
};

class Net {
public:
    int net_degree;
    string net_name;
    vector<Cell *> CellList;
    unsigned int nG1, nG2;
};

class Nets {
public:
    int NumNets; 
    int NumPins; 
    map<string, Net* > NetArray;
};
#endif
