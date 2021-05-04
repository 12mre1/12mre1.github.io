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

My personal favorite for Canadian Census data is the CSD (I believe it to be the Goldilocks size - not too big, not too small), which has been around long enough for there to be an abundance of associated data. Now you may be thinking it might be quite difficult to obtain a dataset containing a specific measure like income for each of these boundaries. Normally you'd be correct, but thanks to a wonderful website called __CensusMapper__, which you can go to [here](https://censusmapper.ca/), extracting such information is not very difficult. At this point, you should visit this site, and sign up for an account if you do not have one already. Then go over to the API section, which let's us extract the dataset we need. Under the __Variable Selection__ tab, select the variable __income__ (you may have to search, in which case search the word income, and select v_CA16_2201). Under the tab __Region Selection__ ,  go ahead and choose __Census District__ (another name for division, which I use here for expediency), then select all CDs in the province of British Columbia (this may take a minute, since there are several).

<center><img src="/img/censusmapper.png" alt = "CensusMapper"></center>
