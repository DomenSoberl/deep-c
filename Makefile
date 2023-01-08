CC := gcc
AR := ar
STD := c17
CFLAGS := -Wall -O3

INCLUDE_DIR := ./include

MLPC_SRCS := \
	mlp.c \
	matrix.c \
	activation.c \
	loss.c \
	adam.c \
	random.c

DDPGC_SRCS := \
	ddpg.c

MLPC_OBJS := $(MLPC_SRCS:%.c=./build/mlpc/%.o)
DDPGC_OBJS := $(DDPGC_SRCS:%.c=./build/ddpgc/%.o)

.PHONY: all clean

all: ./lib/mlpc.a ./lib/ddpgc.a ./bin/saddle ./bin/pendulum

./lib/mlpc.a: $(MLPC_OBJS)
	@echo "Linking $@"
	@mkdir -p $(dir $@)
	@$(AR) rcs $@ $(MLPC_OBJS)

./build/mlpc/%.o: ./src/mlpc/%.c
	@echo "Compiling $<"
	@mkdir -p $(dir $@)
	@$(CC) -std=$(STD) $(CFLAGS) -c $< -o $@

./lib/ddpgc.a: $(DDPGC_OBJS)
	@echo "Linking $@"
	@mkdir -p $(dir $@)
	@$(AR) rcs $@ $(DDPGC_OBJS)

./build/ddpgc/%.o: ./src/ddpgc/%.c
	@echo "Compiling $<"
	@mkdir -p $(dir $@)
	@$(CC) -std=$(STD) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

./bin/saddle: ./examples/saddle.c
	@echo "Compiling $@"
	@mkdir -p $(dir $@)
	@$(CC) -std=$(STD) $(CFLAGS) $< -I$(INCLUDE_DIR) ./lib/mlpc.a -lm -o $@

./bin/pendulum: ./examples/pendulum.c
	@echo "Compiling $@"
	@mkdir -p $(dir $@)
	@$(CC) -std=$(STD) $(CFLAGS) $< -I$(INCLUDE_DIR) ./lib/ddpgc.a ./lib/mlpc.a -lm -o $@

clean:
	@rm -rf ./build
	@rm -rf ./lib
	@rm -rf ./bin