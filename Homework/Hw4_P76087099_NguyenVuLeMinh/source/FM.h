#ifndef FM_H
#define FM_H
#include <bits/stdc++.h>
#include "DS.h"
using namespace std;

class NetDis{
public:
  map<string, Cell*> Group1;
  map<string, Cell*> Group2;
  unsigned int G1_size, G2_size, cut_size;
  vector<Net*> CutNetList;
};

class BP{
public:
  BP();
  BP(string group, int max_cell_gain){
    this->group = group;
    this-> max_cell_gain = max_cell_gain;
  };
  string group;
  int max_cell_gain;
};

class Position{
public:
  Position();
  Position(unsigned int position, unsigned int Gk){
    this->position = position;
    this->Gk = Gk;
  };
  unsigned int position;
  int Gk;
};

class FM{
public:
  FM();
  FM(NetDis *netDis, Cells *cells, Nets *nets) {
    this->netDis = netDis;
    this->cells = cells;
    this->nets = nets;
    BuckList1.resize( 2*(this->cells->Pmax) + 10);
    BuckList2.resize( 2*(this->cells->Pmax) + 10);
  }
  ~FM();
  void Output(double, string);
  void Compute_Init_Gains();
  void Partition();
  void MvCell(Cell *, list<BP *>::reverse_iterator);
  void UpdateBaseCell(Cell *, list<BP *>::reverse_iterator);
  void UpdateNet(Net *, string, string, unsigned int &, unsigned int &, vector<map<string, Cell *> > &, vector<map<string, Cell *> > &, list<BP *> &);
  void UpdateGain(Cell *);
  vector<map<string, Cell *> >BuckList1;
  vector<map<string, Cell *> >BuckList2;
  list<BP *> Bucket_Pointer;
  vector<Cell *> tmp_CellList;
  vector<int> GkList;
  list<Position *> Gk_List;
  vector<unsigned int> pos_List;
  double lower_bound;
  double upper_bound;
  unsigned int G1_size;
  unsigned int G2_size;
  unsigned int num_lock;

  Cells *cells;
  Nets  *nets;
  NetDis *netDis;  
};
#endif
