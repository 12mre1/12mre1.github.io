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

## Conclusion

## Further Reading