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
    LDFLAGS += -lpthread -lm
endif

LIB_SRC     = src/cmol.c
BUILD       = build
BUILD_TESTS = $(BUILD)/tests
BUILD_EX    = $(BUILD)/examples

TEST_SRCS   = $(wildcard tests/test_*.c)
TESTS       = $(patsubst tests/%.c,$(BUILD_TESTS)/%,$(TEST_SRCS))

EXAMPLE_SRCS = $(wildcard examples/*.c)
EXAMPLES     = $(patsubst examples/%.c,$(BUILD_EX)/%,$(EXAMPLE_SRCS))

.PHONY: all debug release test model-tests examples amalgamate compdb clean

all: debug

# ── debug (ASAN + UBSAN, no optimisation) ────────────────────────────────
debug: $(BUILD)/cmol_d.o

UNITY_DEPS = $(wildcard src/*.c src/*.h include/*.h)

$(BUILD)/cmol_d.o: $(LIB_SRC) $(UNITY_DEPS) | $(BUILD)
	$(CC) $(CFLAGS) $(DFLAGS) -c $< -o $@

# ── release (optimised static library) ───────────────────────────────────
release: $(BUILD)/libcmol.a

$(BUILD)/cmol.o: $(LIB_SRC) $(UNITY_DEPS) | $(BUILD)
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

# ── model-tests (reference logit comparison, requires CMOL_TEST_GGUF) ────
# Usage:
#   CMOL_TEST_GGUF=models/SmolLM2-360M-Instruct-Q4_K_M.gguf make model-tests
#
# Runs tools/gen_ref.py (requires: pip install gguf) to generate reference
# logits via the Python gguf dequantizer, then compiles and runs
# tests/test_model_ref which compares the C forward pass against that
# reference at every position (last prefill + N generation steps).

REF_BIN   ?= /tmp/cmol_ref.bin
N_GEN     ?= 3

model-tests: $(BUILD)/cmol_d.o | $(BUILD_TESTS)
	@if [ -z "$(CMOL_TEST_GGUF)" ]; then \
		echo "  SKIP  model-tests: set CMOL_TEST_GGUF=<path-to.gguf>"; \
		exit 0; \
	fi
	@echo "  GEN   $(REF_BIN)  (Python reference, may take ~30s)"
	python3 tools/gen_ref.py "$(CMOL_TEST_GGUF)" "$(REF_BIN)" --n-gen $(N_GEN)
	$(CC) $(CFLAGS) $(DFLAGS) tests/test_model_ref.c $(BUILD)/cmol_d.o \
	      -o $(BUILD_TESTS)/test_model_ref $(LDFLAGS)
	@echo "  RUN   test_model_ref"
	CMOL_TEST_GGUF="$(CMOL_TEST_GGUF)" CMOL_REF_BIN="$(REF_BIN)" \
	  $(BUILD_TESTS)/test_model_ref

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

# ── compile_commands.json for clangd ──────────────────────────────────────
# Regenerate whenever sources change.  Runs the compiler in dry-run mode
# (no -c/-o) so clangd gets correct include paths for every translation unit.
COMPDB_FLAGS = -std=c99 -Wall -Wextra -Wpedantic -I$(CURDIR)/include -I$(CURDIR)/src
compdb:
	@python3 -c "\
import json, glob, os; \
root = '$(CURDIR)'; \
flags = '$(COMPDB_FLAGS)'; \
entries = []; \
srcs = ['src/cmol.c'] + glob.glob('tests/test_*.c') + glob.glob('examples/*.c'); \
[entries.append({'directory': root, 'command': 'cc ' + flags + ' ' + s, 'file': os.path.join(root, s)}) for s in srcs]; \
open('compile_commands.json','w').write(json.dumps(entries, indent=2)); \
print('  GEN   compile_commands.json (' + str(len(entries)) + ' entries)')"

# ── Model fetch targets ───────────────────────────────────────────────────
#
# Downloads SmolLM2 GGUF files from Hugging Face into ./models/.
# Uses `huggingface-cli` when available (pip install huggingface_hub);
# falls back to curl/wget.
#
# Usage:
#   make fetch-smol2-135m     # ~90 MB  Q4_K_M — fits RPi Zero W / any phone
#   make fetch-smol2-360m     # ~230 MB Q4_K_M
#   make fetch-smol2-1.7b     # ~1.1 GB Q4_K_M
#   make fetch-smol3-1.7b     # ~1.1 GB Q4_K_M (SmolLM3 — NoPE + QK-norm)
#
# To pick a different quant level set QUANT=Q8_0 (or Q5_K_M, etc.):
#   make fetch-smol2-1.7b QUANT=Q8_0

MODELS_DIR = models
QUANT      ?= Q4_K_M

# Repo / filename patterns on HF
HF_SMOL2_135M_REPO = bartowski/SmolLM2-135M-Instruct-GGUF
HF_SMOL2_360M_REPO = bartowski/SmolLM2-360M-Instruct-GGUF
HF_SMOL2_1B7_REPO  = bartowski/SmolLM2-1.7B-Instruct-GGUF
HF_SMOL3_1B7_REPO  = bartowski/SmolLM3-1.7B-Instruct-GGUF

HF_SMOL2_135M_FILE = SmolLM2-135M-Instruct-$(QUANT).gguf
HF_SMOL2_360M_FILE = SmolLM2-360M-Instruct-$(QUANT).gguf
HF_SMOL2_1B7_FILE  = SmolLM2-1.7B-Instruct-$(QUANT).gguf
HF_SMOL3_1B7_FILE  = SmolLM3-1.7B-Instruct-$(QUANT).gguf

# Base CDN URL (hf-mirror works without login for public models)
HF_BASE = https://huggingface.co

define fetch_gguf
	@mkdir -p $(MODELS_DIR)
	@dest=$(MODELS_DIR)/$(2); \
	url="$(HF_BASE)/$(1)/resolve/main/$(2)"; \
	if [ -f "$$dest" ]; then \
		echo "  SKIP  $$dest (already exists)"; \
	elif command -v huggingface-cli >/dev/null 2>&1; then \
		echo "  HF    $$dest"; \
		huggingface-cli download $(1) $(2) --local-dir $(MODELS_DIR) \
		  --local-dir-use-symlinks False; \
	elif command -v curl >/dev/null 2>&1; then \
		echo "  CURL  $$dest"; \
		curl -L --progress-bar -o "$$dest" "$$url"; \
	elif command -v wget >/dev/null 2>&1; then \
		echo "  WGET  $$dest"; \
		wget -q --show-progress -O "$$dest" "$$url"; \
	else \
		echo "ERROR: install huggingface-cli, curl, or wget"; exit 1; \
	fi
endef

.PHONY: fetch-smol2-135m fetch-smol2-360m fetch-smol2-1.7b fetch-smol3-1.7b

fetch-smol2-135m:
	$(call fetch_gguf,$(HF_SMOL2_135M_REPO),$(HF_SMOL2_135M_FILE))
	@echo "  OK    models/$(HF_SMOL2_135M_FILE)"
	@echo "        set CMOL_TEST_GGUF=models/$(HF_SMOL2_135M_FILE) to run live tests"

fetch-smol2-360m:
	$(call fetch_gguf,$(HF_SMOL2_360M_REPO),$(HF_SMOL2_360M_FILE))
	@echo "  OK    models/$(HF_SMOL2_360M_FILE)"

fetch-smol2-1.7b:
	$(call fetch_gguf,$(HF_SMOL2_1B7_REPO),$(HF_SMOL2_1B7_FILE))
	@echo "  OK    models/$(HF_SMOL2_1B7_FILE)"

fetch-smol3-1.7b:
	$(call fetch_gguf,$(HF_SMOL3_1B7_REPO),$(HF_SMOL3_1B7_FILE))
	@echo "  OK    models/$(HF_SMOL3_1B7_FILE)"

# ── directories ───────────────────────────────────────────────────────────
$(BUILD) $(BUILD_TESTS) $(BUILD_EX):
	mkdir -p $@

# ── clean ─────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD) cmol_amalgam.h

# models/ intentionally excluded from clean — re-downloading GGUFs is slow.
# Use: rm -rf models/  to purge them manually.
