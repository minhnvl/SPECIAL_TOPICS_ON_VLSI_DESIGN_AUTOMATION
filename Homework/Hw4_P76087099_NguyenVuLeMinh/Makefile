CC = g++
CFLAGS = -Wall -O3
OUT_FILE=FM_76087099

DIR_SRC = ./source
DIR_OBJ = ./file_make

SRC = $(wildcard ${DIR_SRC}/*.cpp)  
OBJ = $(patsubst %.cpp,${DIR_OBJ}/%.o,$(notdir ${SRC})) 


main:${OBJ}
	$(CC) $(OBJ) -o ${OUT_FILE}
${DIR_OBJ}/%.o:${DIR_SRC}/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@
.PHONY:clean
clean:
	find ${DIR_OBJ} -name *.o -exec rm -rf {} \;
	rm -f ${OUT_FILE};

benchmark1:
	@./${OUT_FILE} benchmark/adaptec1.nodes benchmark/adaptec1.nets output_data/adaptec1.out
benchmark2:
	@./${OUT_FILE} benchmark/adaptec2.nodes benchmark/adaptec2.nets output_data/adaptec2.out
benchmark3:
	@./${OUT_FILE} benchmark/adaptec3.nodes benchmark/adaptec3.nets output_data/adaptec3.out
benchmark4:
	@./${OUT_FILE} benchmark/superblue1.nodes benchmark/superblue1.nets output_data/superblue1.out
benchmark5:
	@./${OUT_FILE} benchmark/superblue2.nodes benchmark/superblue2.nets output_data/superblue2.out

benchmark_all:
	@./${OUT_FILE} benchmark/adaptec1.nodes benchmark/adaptec1.nets output_data/adaptec1.out 
	@./${OUT_FILE} benchmark/adaptec2.nodes benchmark/adaptec2.nets output_data/adaptec2.out
	@./${OUT_FILE} benchmark/adaptec3.nodes benchmark/adaptec3.nets output_data/adaptec3.out
	@./${OUT_FILE} benchmark/superblue1.nodes benchmark/superblue1.nets output_data/superblue1.out
	@./${OUT_FILE} benchmark/superblue2.nodes benchmark/superblue2.nets output_data/superblue2.out

