#### Simple Choropleth of Income ####
require(tidyverse)
require(geojsonio)
districts <- geojson_read('./data/map.geojson')
income <- read_csv('./data/data.csv')

names(districts)

income <- income %>% rename(med_income = `v_CA16_2213: Median after-tax income in 2015 among recipients ($)`)

income %>% ggplot() + geom_histogram(aes(x = med_income), 
                                     binwidth = 500, fill = 'red') +
  ggtitle('Histogram of Median After-tax Income in the GTA') +
  labs(x = 'Median Income') + theme_classic()

districts$med_inc <- income$med_income
names(districts)

m <- leaflet(districts) %>% 
  addTiles()  %>% 
  setView( lat=43.8369994, lng=,-79.7060246 , zoom=8) 

bins <- c(25000, 27500, 30000, 32500, 35000, 37500, 40000, Inf)
pal <- colorBin("YlOrRd", domain = districts$med_inc, bins = bins)


# Add Labels
labels <- sprintf(
  "<strong>%s</strong><br/>$ %g",
  districts$name, districts$med_inc
) %>% lapply(htmltools::HTML)

m <- m %>% addPolygons(
  fillColor = ~pal(med_inc),
  weight = 2,
  opacity = 1,
  color = "white",
  dashArray = "3",
  fillOpacity = 0.7,
  highlight = highlightOptions(
    weight = 5,
    color = "#666",
    dashArray = "",
    fillOpacity = 0.7,
    bringToFront = TRUE),
  label = labels,
  labelOptions = labelOptions(
    style = list("font-weight" = "normal", padding = "3px 8px"),
    textsize = "15px",
    direction = "auto")) %>%
  addLegend(pal = pal, values = districts$med_inc, opacity = 0.7, title = NULL,
             position = "bottomright")



library(htmlwidgets)
saveWidget(m, file=paste0( getwd(), "/data/income-choropleth.html"))