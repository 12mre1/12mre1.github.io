---
layout: post
title: "Scraping Financial Data with Beautiful Soup"
date: 2021-05-12
---
_Prerequisite Math: None_

_Prerequisite Coding: Python (Functions), HTML (Tags, Classes, IDs)_

## Scraping Yahoo Finance Data with Python's Beautiful Soup

In this short post, I'm going to show you how to scrape stock price data using the _Beautiful Soup_ html parser. Since this parser is written in Python, so will be the code that follows (although there are other great parsers for web scraping in other programming languages). Why is it called 'Beautiful Soup'? Well back in the day (think early 2000s), most html parsers could only interpret well-formed XML or HTML. But much of the web contained more poorly-formed markup (I'm not suggesting the code was written poorly, just that what appears neat to computers does not always appear organized to humans). This was called 'tag soup', and the aforeementioned parser was designed to take that soup and make it beautiful.

So how does an html parser work? Basically, all code on the internet is written to be read by computers, which means that the information it contains is not easily extracted by humans. HTML parsers are designed to allow users/programmers to extract specific information from a website by specifying particular html tags. You can think of it like a filter designed to separate characters that are used to markup the information from the characters representing the information itself. Let me give you a concrete example. We'll be extracting stock prices from Yahoo Finance, and these specific prices can be found (buried quite deeply) within the HTML code of the YF website. To see the code itself, most browsers will let you right-click, and select _Inspect(Q)_. In Firefox, for example, you would see a result like the following:

<center><img src="/img/yahoo-finance.png" alt = "basic-nn"></center>

In the top of the image, we see the website as the browser displays it. What we're interested in here is the tabular data in the center. Below, you can see just how much extra markup language is needed to format the cells in the table. We want to automate the process of extracting the tabular data and removing the markup text. Good news! Beautiful Soup is awesome at this. But before we can extract the information from the markup text, we need a way to automatically download the code in its entirety. For this, we will use the `requests` library, which allows for simple retrieval via URL. Here is the URL for the image above, which is shows S&P 500 historical data:

<center>

https://ca.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC

</center>

Notice the format of the url. We have everything up to the base ('quote'), and after this, we specify the parameters. In this case, the only thing unique to the S&P 500 is the stock symbol provided at the end. When scraping from a website, it's common practice to experiment with URL construction by clicking through various pages to see what information is contained in the URLs for that specific site. I've already done this for you, so I know that only the stock symbol is needed. For example, to pull identical historical data for Apple (AAPL), I can just substitute the GSPC in the URL above for Apple's stock symbol. That said, here is the function that constructs the URL. Notice that the only argument is the stock symbol:

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

def get_url(stock_symbol):
    assert(isinstance(stock_symbol, str))
    base_url = "https://ca.finance.yahoo.com/quote/"
    url_params = "%5E" + stock_symbol + "/history?p=%5E" + stock_symbol
    return base_url + url_params
```
This function is designed to accept string-type symbols, so I also check for that.

## Some Caveats

## Conclusion

## Further Reading