MAIN = main
OUTDIR = manuscript

export BIBINPUTS := .:../:
export BSTINPUTS := .:../:

.PHONY: all clean watch wc

all:
	latexmk -f -pdf -outdir=$(OUTDIR) $(OUTDIR)/$(MAIN)

watch:
	latexmk -pdf -pvc -outdir=$(OUTDIR) $(OUTDIR)/$(MAIN)

clean:
	latexmk -C -outdir=$(OUTDIR) $(OUTDIR)/$(MAIN)

open: all
	open $(OUTDIR)/$(MAIN).pdf

wc:
	@detex $(OUTDIR)/$(MAIN).tex | wc -w
