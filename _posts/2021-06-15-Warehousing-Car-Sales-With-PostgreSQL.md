---
layout: post
title: "Warehousing Car Sale Data with PostGreSQL"
date: 2021-06-15
---

## Databasing the Car Sales Dataset Using PSQL and PostgreSQL

_Prerequisite Math: None_

_Prerequisite Coding: SQL (CRUD), Bash (Basic Linux)_

Most of the posts I write on this blog usually involve fitting a model to data. However, an equally important part of the data science product cycle is deciding __where and how to store your data__. With this in mind, I'm going to walk through creating and implementing a data model using the famous _Car Sales_ dataset, which you can find [here](https://www.kaggle.com/gagandeep16/car-sales). Obviously this dataset is quite small, and in fact I'm not going to enter the whole set (though you can if you want). The focus on this post will be how we structure the database itself for scalability.

Before we go into more detail, note that I'll be interacting with PostgreSQL through the _bash_ shell. I assume you're familiar with basic bash command, but if not (or if you need installation instructions) I suggest you read my slides on version control [here](https://12mre1.github.io/teach/). Make sure you have it installed before you continue reading.

## What's In the dataset?

So what variables do we have in our car dataset? Well, here's what we're working with:

- __Maufacturer__(TEXT). The company that makes the vehicle, for example Audi.
- __Model__(TEXT). The model of the car. For example, Regal.
- __Sales__(NUMERIC, in thousands). The number of cars sold, in thousands.
- __Year Resale Value__ (NUMERIC). The resale value of the car.
- __Vehicle Type__(TEXT). The type of vehicle. For example, passenger.
- __Price__(NUMERIC, in thousands). The selling price of the car.
- __Engine Size__(NUMERIC). The size (volume) of the engine, in Litres.
- __Horsepower__(NUMERIC). The Horsepower of the car's engine.
- __Fuel Efficiency__(NUMERIC). The fuel efficiency of the car, in MPG.

Note that the actual dataset has more variables, but I'm using a subset just to illustrate the database construction process. Now, here's what this dataset looks like when stored (and displayed) using a simple spreadsheet:

<center><img src="/img/car-spreadsheet.png" alt = "car-sales-spreadsheet"></center>

## The Data Model

## Initializing the PostgreSQL Database

## Adding Data

## Querying the Database

## Conclusions

## Further Reading