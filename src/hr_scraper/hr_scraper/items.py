# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

class GacetaItem(scrapy.Item):
    title = scrapy.Field()
    file_urls = scrapy.Field()
    files = scrapy.Field()


class GacetaOficialItem(scrapy.Item):
    tipo_edicion = scrapy.Field()
    fecha = scrapy.Field()
    numero = scrapy.Field()
    normas = scrapy.Field()
    pdf_url = scrapy.Field()
    file_urls = scrapy.Field()
    files = scrapy.Field()