
import sys,getopt
import datetime
import time
import os
def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"t/bf:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('Error - Please run again: P76087099_HW3.py -t/-b -f <inputfile> -o <outputfile>')
        sys.exit(2)
    # print(opts)
    if len(opts) != 3:
        print ('Error - Please run again: P76087099_HW3.py -t/-b -f <inputfile> -o <outputfile>')
        sys.exit(2)
    filetype = opts[0][0]
    filename_input = opts[1][1]
    filename_output = opts[2][1]
    return filetype,filename_input,filename_output

def Process_input(filename_input):
    dirname = filename_input.replace("\\","@").replace("/","@")
    Check_dir = "@" in dirname
    Check_file = ".txt" in filename_input
    if Check_dir == True:
        Dir_input = filename_input
    else:
        Dir_input = "./benchmark/" + filename_input
    if Check_file == True:
        Dir_input = Dir_input
    else:
        Dir_input = Dir_input + ".txt"
    # print(Dir_input)
    datafile = open(Dir_input,"r")
    Data_input = datafile.read().replace("A=<","").replace("B=<","").replace(",>","")
    Data_input_A = Data_input.split("\n")[0].replace(",","")
    Data_input_B = Data_input.split("\n")[1].replace(",","")
    return Data_input_A, Data_input_B
    
def Bottom_Up():
    for i in range(len(Data_input_A)):
        Listcheck = []
        for j in range(len(Data_input_B)):
            Check_A_in_B = Data_input_A[i:j] in Data_input_B
            if Check_A_in_B is True:
                Listcheck.append(Data_input_A[i:j])
            else:
                break
        if len(Listcheck) > 0:
            List_sum.append(Listcheck[-1])
    Len_Elements_List = list(map(lambda x: len(x), List_sum))
    Data_Max_List = [x for x in List_sum if len(x) == max(Len_Elements_List)]
    return Data_Max_List
   
def Top_Down(s, t):
    if (len(s) < 0):
        if not s or not t:
            return ''
        if s[0] == t[0]:
            result0 = s[0] + Top_Down(s[1:], t[1:])
            # print("0: %s"%result0)
            List_sum.append(result0)
            return result0
        else:
            print(len(s),len(t))
            return max(Top_Down(s[1:], t),Top_Down(s, t[1:]))
        
        if len(s) == Length_A and len(t) == Length_B:
            print(4444)
            Len_Elements_List = list(map(lambda x: len(x), List_sum))
            Data_Max_List = [x for x in List_sum if len(x) == max(Len_Elements_List)]
            Data_Max_List = list(set(Data_Max_List))
            return Data_Max_List
    else:
        Data_Max_List = Bottom_Up()
        return Data_Max_List

def Process_Output(Data_Max_List,Fisnishtime,filename_ouput,filetype):
    dirname = filename_ouput.replace("\\","@").replace("/","@")
    Check_dir = "@" in dirname
    Check_file = ".txt" in filename_ouput
    if Check_dir == True:
        Dir_output = filename_ouput
    else:
        Dir_output = "./benchmark/" + filename_ouput
    if Check_file == True:
        Dir_output = Dir_output
    else:
        Dir_output = Dir_output + ".txt"
    if filetype == "-b":
        Dir_output = Dir_output.replace(".txt","_bo.txt")
    else:
        Dir_output = Dir_output.replace(".txt","_to.txt")
    # print(Dir_input)
    datafile = open(Dir_output,"w")
    text = "Runtime: %s \nCheck up times: %s \nCommon sequence: %s" %(str(Fisnishtime),str(Fisnishtime),str(Data_Max_List))
    datafile.write(text)
    datafile.close()
    return Dir_output
if __name__ == "__main__":
    # Check parameter
    filetype,filename_input,filename_ouput = main(sys.argv[1:])
    List_sum =[]
    Starttime = datetime.datetime.now()
    # Processing data 
    Data_input_A,Data_input_B = Process_input(filename_input)
    print("Please, Waiting a moment")
    if filetype == "-b": #Bottom_Up
        Data_Max_List = Bottom_Up() 
    elif filetype == "-t": #Top_Down
        time.sleep(10)
        Length_A = len(Data_input_A)
        Length_B = len(Data_input_B)
        Data_Max_List = Top_Down(Data_input_A,Data_input_B) 
    Fisnishtime = datetime.datetime.now() - Starttime
    # Processing Output
    Dir_output = Process_Output(Data_Max_List,Fisnishtime,filename_ouput,filetype)
    print("Run time: %s" %str(Fisnishtime))
    print("Common sequence: %s"%str(Data_Max_List))
    print("Output file: %s"%Dir_output)
    print("------ DONE ------")

    