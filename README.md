# Nuts and Bolts: Analyzing the Development of Technological Breakthroughs Through Clustering

## Description

This is a project created to analyze how technological breakthroughs happen. The impetus behind this project is a desire to discover how technology progresses and how that knowledge can serve to accelerate future developments. This project analyzes technological breakthroughs by using patent data. The forward citations of highly cited patents are analyzed in respect to their distribution. Certain features of this analysis are employed in a clustering algorithm to determine general groups that patents can fall into.

## Installation

This project can be installed from git using:

```
git clone https://github.com/JH4789/PatentAnalysis.git
```
Package dependencies can be installed using max_requirements.txt:
```
pip install -r max_requirements.txt
```
This includes all Jupyter Lab components and dependencies

Running

```
pip install -r min_requirements.txt
```

installs  all non-Jupyter packages 

## Usage

Navigate to the Analysis directory and use Jupyter Lab to open ```demo.ipynb```

Running all cells in this Jupyter notebook yields a demonstration of the clustering algorithm and histograms tracking a given patents forward citations over time

A copy of the slideshow can be compiled by running the following commands:
```
libreoffice --headless --convert-to pdf Huang_ICR_Presentation.odp
```
The paper can be compiled by running the following
```
lualatex main.tex
```
## Acknowledgements

I would like to thank Taylor Blair and Andrew J. Ouderkirk for their invaluable contributions to this project. This project would not be possible without their mentorship.

I would also like to acknowledge the Institute for Computing in Research for providing me with the opportunity to conduct this project and contribute to a rich history of scientific discovery. 

## License

This project is licensed under the GNU General Public License v3.0





