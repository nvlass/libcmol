CC      = cc
AR      = ar
CFLAGS  = -std=c99 -Wall -Wextra -Wpedantic -I include -I src
DFLAGS  = -g -O0 -DCMOL_DEBUG -fsanitize=address,undefined
RFLAGS  = -O3 -DNDEBUG

# Runtime SIMD dispatch: compile release with native CPU features enabled.
# Each hot kernel function carries its own target attribute, so the compiler
# emits the right ISA per function rather than assuming it globally.
RFLAGS += -march=native

# Pthreads — Linux needs -lpthread; macOS has it in libc.
UNAME   := $(shell uname -s)
ifeq ($(UNAME), Linux)
    LDFLAGS += -lpthread
endif

LIB_SRC     = src/cmol.c
BUILD       = build
BUILD_TESTS = $(BUILD)/tests
BUILD_EX    = $(BUILD)/examples

TEST_SRCS   = $(wildcard tests/test_*.c)
TESTS       = $(patsubst tests/%.c,$(BUILD_TESTS)/%,$(TEST_SRCS))

EXAMPLE_SRCS = $(wildcard examples/*.c)
EXAMPLES     = $(patsubst examples/%.c,$(BUILD_EX)/%,$(EXAMPLE_SRCS))

.PHONY: all debug release test examples amalgamate clean

all: debug

# ── debug (ASAN + UBSAN, no optimisation) ────────────────────────────────
debug: $(BUILD)/cmol_d.o

$(BUILD)/cmol_d.o: $(LIB_SRC) | $(BUILD)
	$(CC) $(CFLAGS) $(DFLAGS) -c $< -o $@

# ── release (optimised static library) ───────────────────────────────────
release: $(BUILD)/libcmol.a

$(BUILD)/cmol.o: $(LIB_SRC) | $(BUILD)
	$(CC) $(CFLAGS) $(RFLAGS) -c $< -o $@

$(BUILD)/libcmol.a: $(BUILD)/cmol.o
	$(AR) rcs $@ $<
	@echo "  AR  $@"

# ── tests ─────────────────────────────────────────────────────────────────
test: $(BUILD)/cmol_d.o | $(BUILD_TESTS)
	@passed=0; failed=0; \
	for src in $(TEST_SRCS); do \
		name=$$(basename $$src .c); \
		bin=$(BUILD_TESTS)/$$name; \
		$(CC) $(CFLAGS) $(DFLAGS) $$src $(BUILD)/cmol_d.o \
		      -o $$bin $(LDFLAGS) 2>&1 \
		  && $$bin \
		  && { echo "  PASS  $$name"; passed=$$((passed+1)); } \
		  || { echo "  FAIL  $$name"; failed=$$((failed+1)); }; \
	done; \
	echo ""; \
	echo "  Results: $$passed passed, $$failed failed."

# ── examples ──────────────────────────────────────────────────────────────
examples: $(BUILD)/cmol.o | $(BUILD_EX)
	@for src in $(EXAMPLE_SRCS); do \
		name=$$(basename $$src .c); \
		$(CC) $(CFLAGS) $(RFLAGS) $$src $(BUILD)/cmol.o \
		      -o $(BUILD_EX)/$$name $(LDFLAGS) \
		  && echo "  BUILD $$name"; \
	done

# ── single-header amalgamation ────────────────────────────────────────────
amalgamate:
	python3 tools/amalgamate.py > cmol_amalgam.h
	@echo "  GEN   cmol_amalgam.h"

# ── directories ───────────────────────────────────────────────────────────
$(BUILD) $(BUILD_TESTS) $(BUILD_EX):
	mkdir -p $@

# ── clean ─────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD) cmol_amalgam.h
