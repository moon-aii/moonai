PUML_FILES := $(wildcard diagrams/*.puml)
PNG_FILES  := $(patsubst diagrams/%.puml,img/generated/%.png,$(PUML_FILES))

PLANTUML   ?= plantuml

all: img/generated $(PNG_FILES) main.pdf

img/generated:
	mkdir -p img/generated

img/generated/%.png: diagrams/%.puml
	$(PLANTUML) -o ../img/generated $<

main.pdf: main.tex sections/*.tex $(PNG_FILES)
	xelatex -interaction=nonstopmode main.tex
	xelatex -interaction=nonstopmode main.tex

clean:
	rm -f *.aux *.log *.toc *.out main.pdf
	rm -rf img/generated/

.PHONY: all clean
