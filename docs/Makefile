# Minimal makefile for Sphinx documentation.
# Treat warnings as errors (-W) to match the strict CI build in
# .github/workflows/docs.yml; --keep-going reports every warning, not just the
# first.

SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put "help" first so "make" without arguments prints the target list.
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all: route every other target (html, clean, ...) to sphinx-build.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
