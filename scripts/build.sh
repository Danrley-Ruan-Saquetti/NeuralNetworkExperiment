#!/bin/bash
mkdir -p build && cd build && cmake .. -G "MinGW Makefiles" && make
cd .. && ./scripts/copy-package.sh
