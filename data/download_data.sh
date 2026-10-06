#!/bin/bash
# Downloads the UCI protein tertiary structure data (CASP.csv) used in E8. California housing is fetched by scikit-learn.
cd "$(dirname "$0")"
curl -sS -o protein.zip "https://archive.ics.uci.edu/static/public/265/physicochemical+properties+of+protein+tertiary+structure.zip" && unzip -o -q protein.zip CASP.csv && rm -f protein.zip && ls -la CASP.csv
