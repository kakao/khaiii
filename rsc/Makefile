HOME_DIR = .
BIN_DIR = $(HOME_DIR)/bin
SRC_PYTHON = $(HOME_DIR)/../src/main/python
RSC_SRC = $(HOME_DIR)/src
PREFIX = /usr/local
_RSC_DIR = $(PREFIX)/share/
RSC_DIR = $(PREFIX)/share/khaiii
MODEL_SIZE = base

MODEL = \
    $(RSC_DIR)/config.json \
    $(RSC_DIR)/embed.bin \
    $(RSC_DIR)/conv.2.fil \
    $(RSC_DIR)/conv.3.fil \
    $(RSC_DIR)/conv.4.fil \
    $(RSC_DIR)/conv.5.fil \
    $(RSC_DIR)/cnv2hdn.lin \
    $(RSC_DIR)/hdn2tag.lin

RESTORE = \
    $(RSC_DIR)/restore.key \
    $(RSC_DIR)/restore.val \
    $(RSC_DIR)/restore.one

PREANAL = \
    $(RSC_DIR)/preanal.tri \
    $(RSC_DIR)/preanal.val

ERRPATCH = \
    $(RSC_DIR)/errpatch.tri \
    $(RSC_DIR)/errpatch.val \
    $(RSC_DIR)/errpatch.len

all: $(MODEL) $(PREANAL) $(RESTORE) $(ERRPATCH)

$(wordlist 2,100,$(MODEL)): $(firstword $(MODEL))
$(firstword $(MODEL)): $(RSC_SRC)/$(MODEL_SIZE).config.json $(RSC_SRC)/$(MODEL_SIZE).model.pickle
	cmake -E make_directory $(_RSC_DIR)
	cmake -E make_directory $(RSC_DIR)
	cmake -E env PYTHONPATH=$(SRC_PYTHON) python3 $(BIN_DIR)/compile_model.py --model-size $(MODEL_SIZE) --rsc-src $(RSC_SRC) --rsc-dir $(RSC_DIR)

$(wordlist 2,100,$(PREANAL)): $(firstword $(PREANAL))
$(firstword $(PREANAL)): $(RSC_SRC)/preanal.auto $(RSC_SRC)/preanal.manual
	cmake -E make_directory $(_RSC_DIR)
	cmake -E make_directory $(RSC_DIR)
	cmake -E env PYTHONPATH=$(SRC_PYTHON) python3 $(BIN_DIR)/compile_preanal.py --rsc-src $(RSC_SRC) --rsc-dir $(RSC_DIR)

$(wordlist 2,100,$(RESTORE)): $(firstword $(RESTORE))
$(firstword $(RESTORE)): $(RSC_SRC)/restore.dic $(RSC_SRC)/vocab.out $(RSC_SRC)/vocab.out.more
	cmake -E make_directory $(_RSC_DIR)
	cmake -E make_directory $(RSC_DIR)
	cmake -E env PYTHONPATH=$(SRC_PYTHON) python3 $(BIN_DIR)/compile_restore.py --rsc-src $(RSC_SRC) --rsc-dir $(RSC_DIR)

$(wordlist 2,100,$(ERRPATCH)): $(firstword $(ERRPATCH))
$(firstword $(ERRPATCH)): $(RSC_SRC)/$(MODEL_SIZE).errpatch.auto $(RSC_SRC)/$(MODEL_SIZE).errpatch.manual
	cmake -E make_directory $(_RSC_DIR)
	cmake -E make_directory $(RSC_DIR)
	cmake -E env PYTHONPATH=$(SRC_PYTHON) python3 $(BIN_DIR)/compile_errpatch.py --model-size $(MODEL_SIZE) --rsc-src $(RSC_SRC) --rsc-dir $(RSC_DIR)

clean:
	rm -rf $(RSC_DIR)
