import os
import re
from datetime import datetime
import numpy as np
from functools import reduce 
Path_node_1 = "./Benchmark/adaptec1.nodes"
Path_net_1 = "./Benchmark/adaptec1.nets"
file_node = open(Path_node_1,"r")
file_net = open(Path_net_1,"r")

def Read_benchmark():

    print("Start Read Benchmark File")
    # list_note = file_node.read().split("\n")
    # DF_note = pd.DataFrame(list_note)
    # print(DF_note)
    List_Total = []
    List_Infor = []
    List_G1 = []
    List_G2 = []
    bool_checknote = 0
    total_group  = 0

    # Take value in net file
    print("****Take net value****")
    List_net = []
    Dict_net = {}
    lst_filenet = file_net.read().split("NetDegree")
    for index, inet in enumerate(lst_filenet):
        if index == 0:
            # Total net in this file
            numnets = re.findall(r'NumNets : \d+',inet)[0]
            numnets = numnets.split(":")[1].strip() 
            # Total Pin in this file
            numpins = re.findall(r'NumPins : \d+',inet)[0]
            numpins = numpins.split(":")[1].strip()
            print("Total of Nets: %s" %numnets)
            print("Total of Pins: %s" %numpins)
        else:
            lst_cell = re.findall(r'[a-z]\d+',inet)
            # List_net.append(lst_cell)
            for idict in lst_cell[1:]:
                check_dict = idict in Dict_net
                if check_dict:
                    value = Dict_net.get(idict)
                    Dict_net.update(idict = value.append(lst_cell[0]))
                else:
                    Dict_net[idict]  = [lst_cell[0]]
    # print(Dict_net)
    # numpyy_Listnet = np.array(List_net)
    # maxlen = len(max(numpyy_Listnet,key=len)) - 1
    # print(numpyy_Listnet.shape)
    # print(numpyy_Listnet)
    
    # print(maxlen)
    # # a = np.where(np.isin(numpyy_Listnet[:,5],np.array(["o2"])))
    # # a = numpyy_Listnet[:,0[0]]
    # print(numpyy_Listnet)
    # print(a)
    # Take value in node file
    print("****Take node value****")
    for inote in file_node:
        inote = inote.replace("\n","")

        if "NumNodes" in inote: # Total nodes in this file
            numnodes = list(filter(lambda x: x.isdigit(), inote.split("\t")))[0]
            List_Total.append(numnodes)
            print("Total of Nodes: %s" %numnodes)
        elif "NumTerminals" in inote: # Total terminals in this file
            numterminal = list(filter(lambda x: x.isdigit(), inote.split("\t")))[0]
            bool_checknote = 1
            List_Total.append(numterminal)
            print("Total of Terminal: %s" %numterminal)
        elif bool_checknote == 1: # Take information of nodes (area, total,name node)
            lst_data = []
            list_node = inote.split("\t")
            num_area = int(list_node[2]) * int(list_node[3])
            # seprate_group = 0.4 * float(numnodes)
            
            if len(list_node) == 4:
                check_terminal = 0
                total_group += num_area # Total area don't need to consider Terminal
                total_term = total_group
            elif len(list_node) == 5:
                check_terminal = 1
                total_term = 0
            # filter_cell = list(filter(lambda j,x: list_node[1] in j, List_net))
            # lst_filter_cell = list(list(zip(*filter_cell))[0])
            # lst_filter_cell = filter_cell
            lst_filter_net = Dict_net.get(list_node[1])

            # print(filter_cell)
            lst_data.append(list_node[1])
            lst_data.append(num_area)
            lst_data.append(check_terminal)
            lst_data.append(total_term)
            lst_data.append(lst_filter_net)

            List_Infor.append(lst_data)

    # Seprate group and calculate groip size
    print("****Separate Group****")
    denote_value = 0.4 * float(total_group)
    List_G1 = list(filter(lambda j: j[3] < denote_value and j[3] != 0, List_Infor))
    List_G2 = list(filter(lambda j: j[3] >= denote_value and j[3] != 0, List_Infor))
    size_G1 = sum(map(lambda x: int(x[1]), List_G1))
    size_G2 = sum(map(lambda x: int(x[1]), List_G2))
    min_value_G1 = min(map(lambda x: int(x[1]), List_G1))
    max_value_G1 = max(map(lambda x: int(x[1]), List_G1))
    List_net_G1 = "-".join(list(map(lambda x: "-".join(x[4]) , List_G1))).split("-")
    List_net_G2 = "-".join(list(map(lambda x: "-".join(x[4]) , List_G2))).split("-")
    Cut_size = set(List_net_G1) & set(List_net_G2)
  
    print("Size of Total: %s"%total_group)
    print("Size of G1: %s"%size_G1)
    print("Size of G2: %s"%size_G2)
    print("Min of G1: %s"%min_value_G1)
    print("Max of G1: %s"%max_value_G1)
    print("Cut size: %s" %len(Cut_size))
    # print(Dict_net)
    # print(List_net_G1)
    # a = np.where(np.array(List_net_G1) == "n452")[0]
    # b = np.where(np.array(List_net_G2) == "n452")[0]
    # print(a)
    # print(b)
    print(len(List_Infor))
    # Find minimum cutsize by FM Algorithm
    print("FM algorithm")
    List_Fixed = []
    List_net_g = []
    for index, icell in enumerate(List_Infor):
        if icell[0] in List_Fixed:
            pass
        else:
            net_g = 0
            for inet in icell[4]:
                net_a = len(np.where(np.array(List_net_G1) == inet)[0])
                net_b = len(np.where(np.array(List_net_G2) == inet)[0])
                net_g = net_g + net_b - net_a 
                print(np.where(np.array(List_net_G1) == inet)[0])
                print(np.where(np.array(List_net_G2) == inet)[0])
               
            List_net_g.append(net_g)
            input()
if __name__ == "__main__":
    start = datetime.now()
    Read_benchmark()
    file_node.close()
    file_net.close()

    finished = datetime.now() - start
    print("Finished: %s" %finished)





# Mapping Cell and net
# a = map(lambda x,y: x if x[0] in y  ,List_Infor,List_net)
# a = filter(lambda x: x[0] in List_net, List_Infor)
# print(list(a))
# print("Mapping Cell and Net")
# for data in List_Infor:
#     filter_cell = list(filter(lambda j: data[0] in j, List_net))
#     lst_filter_cell = list(list(zip(*filter_cell))[0])
#     data.append(lst_filter_cell)

    # print(filter_cell)
    # print(lst_filter_cell)
    # print(data)
    # input()



# for inet in file_net: 
#     inet = inet.replace("\n","")
#     if "NumNets" in inet: # Total net in this file
#         numnets = list(filter(lambda x: x.isdigit(), inet.split(" ")))[0]
#         List_Total.append(numnets)
#         print(numnets)
#     elif "NumPins" in inet: # Total Pin in this file
#         numpins = list(filter(lambda x: x.isdigit(), inet.split(" ")))[0]
#         List_Total.append(numpins)
#         print(numpins)
#     elif "NetDegree" in inet:
#         numcells = re.findall(r'\d+|\w+\d+',inet)
#         print(inet)
#         print(numcells)