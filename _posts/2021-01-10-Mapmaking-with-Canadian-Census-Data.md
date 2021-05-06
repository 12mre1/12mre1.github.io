---
layout: post
title: "Map-making with Canadian Census Data"
date: 2021-01-10
---

## A Simple Choropleth Map with Mapbox JS API

_Prerequisite Math: None_

_Prerequisite Coding: HTML, CSS, JavaScript_

When I worked for the Government of Canada, one of the first big projects I was assigned was co-developing an open source GIS application. A __Geographic Information System__ is just a framework for analyzing spatial data (stuff you might see on a map), and being employed by the government meant that I had reasonably quick access to most Canadian census data. The application I'll show you how to build today is a bit more basic, however maps remain one of the most effective tools in communicating data insights to audiences without much data experience. Everyone sees visually, and while a picture is worth 1000 words, a map can have even more impact. Before we get started on the code, in order to use Mapbox like I do today, you'll need to create a free account. You can do so [here](https://account.mapbox.com/auth/signup/). After you sign up (and verify your email), you'll be able to log into the Mapbox Dashboard, which should present you with a public token. This token is what we will use to query the base layers of the map we create, using the JavaScript API. If you have no idea what I just said, don't worry - I'll explain more later. Today I'm going to show you how to create a map of income.


## Canadian Census Regions

Before we get started on what data we'll actually be mapping, let's discuss boundaries (I don't mean limitations, but rather physical borders on the map). The Government of Canada collects data according to officially defined boundaries, that vary in size/area depending usually on population. Let me be clear; we usually call such definitions __census polygons__, because the areas they define cover two dimensions. It is possible to collect data about roads, point coordinates, and other information, but for the purposes of our choropleth (a map that shades areas according to some value) we only want polygon boundaries. At the largest level, we aggregate data by province, but there are many other lesser known levels of granularity you might want to account for. Roughly, the order of boundary size is (from largest to smallest area):

- __Province__ 
- __Census Divisions__ (Subsets of Each Province, often corresponding to larger counties)
- __Census Subdivisions__ (Subsets of highly populated divisions, often corresponding to municipalities)
- __Census Dissemination Areas__ (Some CSDs are further split), usually 400-700 people

For more rural areas, there are so few people that the lowest level of polygon is not needed - thus if we were to obtain what is called the __boundary file__ for Canadian Subdivisions, many of the areas would simply be divisions, with further granularity only included where needed. Note that this standard of boundaries is fairly recent (10 or so years old), so older data may not be available in aggregations corresponding to these boundaries. When that happens, you may either try a larger level of aggregation, or compute such statistics yourself. At the subnational level, the government does use other boundaries in certain situations. For example:

- __Census Metropolitan Areas__, which are urban clusters of CSDs corresponding to 100,000 people or more
- __Census Agglomerations__, corresponding to CSD groups of greater than 10,000, but not CMAs

Just to give you an idea of what this defines, the Greater Golden Horseshoe Metropolitan Area (Hamilton, Niagara, and other territory in Southern Ontario), is Canada's geographically largest CMA. Note that other countries will have different boundary definitions, like the Counties in the US (called parishes in Louisiana), or Local Authorities in the UK. Additionally, some data may be available in boundaries that are context-specific (for example polling or electoral districts). Unfortunately the choice of boundary and thus the level of granularity you can put on a map is often constrained by data availability.

My personal favorite for Canadian Census data is the CSD (I believe it to be the Goldilocks size - not too big, not too small), which has been around long enough for there to be an abundance of associated data. Now you may be thinking it might be quite difficult to obtain a dataset containing a specific measure like income for each of these boundaries. Normally you'd be correct, but thanks to a wonderful website called __CensusMapper__, which you can go to [here](https://censusmapper.ca/), extracting such information is not very difficult. At this point, you should visit this site, and sign up for an account if you do not have one already. Then go over to the API section, which let's us extract the dataset we need. Under the __Variable Selection__ tab, select the variable __income__ (you may have to search, in which case search the word income, and select v_CA16_2213). Under the tab __Region Selection__ ,  go ahead and choose __Census District__ (another name for division, which I use here for expediency), then select all CSDs in the Greater Toronto Area. Although that definition is somewhat ambiguous, you can refer to the image below. It's not important that your selected region looks exactly like mine, but the area you choose should be continuous.

<center><img src="/img/censusmapper.png" alt = "CensusMapper"></center>

Stretching out from the Financial district of Toronto, I go as far north as Barrie, as far east as Oshawa/Whitby, and as far West as Orangeville. After you've chosen those districts, you'll need to specify __Selected Regions__ for the Geographic Aggregation Level. Lastly, you should see the download options at the bottom of the page. Download both the variable data, and the geographic data. Once you have these two files (one should be a geojson type, and the other a csv), we can continue our work in R. I prefer RStudio, but regular old R will work fine for our purposes.

Now in R, we can start by loading in the boundary file (_geos.geojson_), which can be read using the `geojsonio` package. Make sure to install this package and load it into namespace if you have not already done so. At the same time, I will use the common tidyverse function `read_csv()` to import our income data.
```R
require(tidyverse)
require(geojsonio)
districts <- geojson_read('./data/geos.geojson', what = "sp")
income <- read_csv('./data/data.csv')
names(districts)
 [1] "a"    "t"    "dw"   "hh"   "id"   "pop"  "name" "pop2" "rgid"
[10] "rpid" "ruid"
```

Note that R stores our boundary file as a `SpatialPolygonDataFrame`, which is a special data structure designed to handle a very large number of reference points (usually latitude and longitude). It essentially operates as a nested list, with 36 individual districts/divisions each containing their own information. We can see that the dataframe itself has an attribute called `data`, which contains several other pieces of information, including household number and id, as well as population (in 2 different forms). Essentially what we're trying to do is add our income data as another variable in this set. Once we do that, creating the map is fairly straightforward.

Before we do this, let's take a look at our income data. I'm going to start by renaming the default label to `median income`. Then I'll take a look at the distribution of income in our area.
```R
income <- income %>% rename(med_income = `v_CA16_2213: Median after-tax income in 2015 among recipients ($)`)

income %>% ggplot() + geom_histogram(aes(x = med_income), 
                                     binwidth = 500, fill = 'red') +
  ggtitle('Histogram of Median After-tax Income in the GTA') +
  labs(x = 'Median Income') + theme_classic()
```
This code generates the following plot:

<center><img src="/img/med-inc-hist.png" alt = "IncomeHistogram"></center>