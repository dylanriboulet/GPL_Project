#!/bin/bash

curl -s https://tradingeconomics.com/mmm:us > webpage.html

stock_price=$(grep -oP '(?<=id="market_last">)[0-9]+\.[0-9]+' webpage.html)

echo "$(date +%Y-%m-%d\ %H:%M:%S), $stock_price" >> stock_price.txt
